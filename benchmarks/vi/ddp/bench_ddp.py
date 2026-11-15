"""First-hardware verification of ov.vi's multi-GPU DDP training
(``omicverse/vi/_distributed.py``, merged in #1074): ``model.train(devices=[...])``
and the ``torchrun`` entry path. Each check is independent; run with
``--check``. Results: ../results_ddp.md.

    python bench_ddp.py --check inmemory   # NCCL DDP, in-memory AnnData, 1 vs 2 GPU quality
    python bench_ddp.py --check oom_lock   # AnnDataOOM DDP while a separate process holds
                                            # the file open (writer, spawn, no HDF5 locking)
    python bench_ddp.py --check dask       # Dask(cupy)-backed AnnData DDP
    python bench_ddp.py --check disjoint   # each rank sees disjoint rows every epoch
    python bench_ddp.py --check scaling --data small   # 1 vs 2 GPU epoch time, SEA-AD 300k
    python bench_ddp.py --check scaling --data full    # ... SEA-AD full (1,378,211 cells)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

BMMC_DONORS = [
    "/scratch/users/steorra/tmp/vi_tut/run/data/neurips2021_s1d3.h5ad",
    "/scratch/users/steorra/tmp/vi_tut/run/data/neurips2021_s2d1.h5ad",
    "/scratch/users/steorra/tmp/vi_tut/run/data/neurips2021_s3d7.h5ad",
]
SEAAD_300K = os.path.join(common.CACHE, "seaad300k_hvg2000.h5ad")


def _bmmc_merged():
    """Real NeurIPS 2021 BMMC donors (s1d3/s2d1/s3d7), concatenated on their
    common genes -- a small dataset with a real multi-donor batch key."""
    import anndata as ad
    adatas = [ad.read_h5ad(p) for p in BMMC_DONORS]
    genes = adatas[0].var_names
    for a in adatas[1:]:
        genes = genes.intersection(a.var_names)
    adatas = [a[:, genes].copy() for a in adatas]
    return ad.concat(adatas, label="donor_src", keys=["s1d3", "s2d1", "s3d7"], index_unique="-")


def _scib_quick(z, labels):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    k = len(np.unique(labels))
    pred = KMeans(k, n_init=5, random_state=0).fit_predict(z)
    return (normalized_mutual_info_score(labels, pred), adjusted_rand_score(labels, pred))


def check_inmemory():
    """NCCL DDP on 2 GPUs, in-memory AnnData: 1-GPU vs 2-GPU (devices=[0,1])
    training quality on real multi-donor BMMC data."""
    from omicverse.vi import SCVI
    a = _bmmc_merged()
    rows = []
    for tag, kw in (("1gpu", dict(device="cuda:0", batch_size=256)),
                    ("2gpu", dict(devices=[0, 1], batch_size=128, latent_key="X_ddp"))):
        m = SCVI(a, layer="counts", batch_key="batch", n_hidden=128, n_latent=10, seed=0)
        t0 = time.perf_counter()
        m.train(max_epochs=60, progress=False, seed=0, **kw)
        wall = time.perf_counter() - t0
        z = m.get_latent_representation()
        elbo = m.get_elbo(indices=m.validation_indices)
        nmi, ari = _scib_quick(z, a.obs["cell_type"].values)
        rows.append(dict(check="inmemory", tag=tag, world=getattr(m, "_world_size", 1),
                         wall_s=round(wall, 2), elbo_val=round(elbo, 2), nmi=round(nmi, 4),
                         ari=round(ari, 4)))
        print(rows[-1], flush=True)
    common.write_table(rows, "ddp_inmemory")


def _writer_proc(path, ready, stop):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # same convention as ov.vi's DDP workers
    import h5py
    f = h5py.File(path, "a")
    ready.set()
    while not stop.is_set():
        _ = f["X"]
        time.sleep(0.2)
    f.close()


def check_oom_lock():
    """AnnDataOOM DDP (devices=[0,1]) while a separate (spawned) process holds
    the same h5ad open in h5py mode 'a' for the whole run."""
    import omicverse as ov
    from omicverse.vi import SCVI
    path = os.path.join(common.CACHE, "ddp_bmmc_merged.h5ad")
    if not os.path.exists(path):
        _bmmc_merged().write_h5ad(path)
    ov.settings.cpu_init()
    a = ov.read(path, storage="disk")
    m = SCVI(a, layer="counts", batch_key="batch", n_hidden=64, n_latent=8, seed=0)

    ctx = mp.get_context("spawn")  # fork would inherit this process's already-opened libhdf5
    ready, stop = ctx.Event(), ctx.Event()
    w = ctx.Process(target=_writer_proc, args=(path, ready, stop))
    w.start()
    ok = ready.wait(timeout=30)
    row = dict(check="oom_lock", writer_opened=ok)
    try:
        m.train(max_epochs=10, devices=[0, 1], batch_size=256, progress=False, seed=0,
                block_size=2000, buffer_blocks=4)
        z = m.get_latent_representation()
        row.update(world=m._world_size, finite=bool(np.isfinite(z).all()), error="")
    except Exception as e:
        row.update(error=repr(e))
    finally:
        stop.set()
        w.join(timeout=10)
        if w.is_alive():
            w.terminate()
    print(row, flush=True)
    common.write_table([row], "ddp_oom_lock")


def check_dask():
    """Dask(cupy)-backed AnnData DDP on 2 GPUs."""
    import omicverse as ov
    from omicverse.vi import SCVI
    from omicverse.pp._dask import read_h5ad
    path = os.path.join(common.CACHE, "ddp_bmmc_merged.h5ad")
    if not os.path.exists(path):
        _bmmc_merged().write_h5ad(path)
    ov.settings.mode = "gpu"
    da_ad = read_h5ad(path, chunk_size=4096, device="gpu", layer="counts")
    m = SCVI(da_ad, batch_key="batch", n_hidden=64, n_latent=8, seed=0)
    m.train(max_epochs=10, devices=[0, 1], batch_size=256, progress=False, seed=0)
    z = m.get_latent_representation()
    row = dict(check="dask", world=m._world_size, source=m._source().kind,
              finite=bool(np.isfinite(z).all()))
    print(row, flush=True)
    common.write_table([row], "ddp_dask")


def check_disjoint():
    """Every rank sees disjoint rows every epoch (in-memory block/random shuffle
    and AnnDataOOM block shuffle), the property ov.vi's DDP fit() loop relies on."""
    import anndata as ad
    import omicverse as ov
    from omicverse.vi._data import build_source
    path = os.path.join(common.CACHE, "ddp_bmmc_merged.h5ad")
    a = ad.read_h5ad(path) if os.path.exists(path) else _bmmc_merged()
    if not os.path.exists(path):
        a.write_h5ad(path)
    codes = (np.asarray(a.obs["batch"].values) == a.obs["batch"].values[0]).astype(np.int64)
    rows = np.arange(a.n_obs)
    rows_checked = []

    def _check(name, src, shuffle, world=2):
        for epoch in range(3):
            parts = []
            for r in range(world):
                idx = [b["idx"].cpu().numpy() for b in
                       src.iterate(rows, batch_size=128, shuffle=shuffle, seed=0, epoch=epoch,
                                  rank=r, world=world)]
                parts.append(np.sort(np.concatenate(idx)) if idx else np.array([], np.int64))
            allp = np.concatenate(parts)
            ok_disjoint = len(allp) == len(np.unique(allp))
            ok_cover = len(allp) == len(rows)
            rows_checked.append(dict(check="disjoint", source=name, shuffle=shuffle,
                                     epoch=epoch, disjoint=ok_disjoint, full_coverage=ok_cover))
            assert ok_disjoint and ok_cover, rows_checked[-1]

    src_mem, _ = build_source(a, layer="counts", obs={"batch": codes}, device="cpu")
    _check("in-memory", src_mem, "block")
    _check("in-memory", src_mem, "random")
    ov.settings.cpu_init()
    oom = ov.read(path, storage="disk")
    src_oom, _ = build_source(oom, layer="counts", obs={"batch": codes}, device="cpu",
                              block_size=1000)
    _check("oom", src_oom, "block")
    for r in rows_checked:
        print(r, flush=True)
    common.write_table(rows_checked, "ddp_disjoint")


def check_scaling(data):
    """1-GPU vs 2-GPU epoch time, same architecture / epoch count, back-to-back
    under this node's (contended) load."""
    from omicverse.vi import SCVI
    if data == "small":
        import anndata as ad
        a = ad.read_h5ad(SEAAD_300K)
        kw_model = dict(layer="counts", batch_key="donor", n_hidden=128, n_latent=10)
        epochs = 8
        tag = "seaad300k"
    else:
        import omicverse as ov
        ov.settings.cpu_gpu_mixed_init()
        a = ov.read(common.SEAAD, storage="disk")
        a = ov.pp.qc(a, tresh={"mito_perc": 0.2, "nUMIs": 500, "detected_genes": 250},
                    doublets=False)
        ov.pp.preprocess(a, mode="shiftlog|pearson", n_HVGs=2000)
        kw_model = dict(layer="counts", batch_key="donor", genes="highly_variable_features",
                        n_hidden=128, n_latent=10)
        epochs = 4
        tag = "seaad_full"

    rows = []
    models = {}
    for gtag, kw in (("1gpu", dict(device="cuda:0")), ("2gpu", dict(devices=[0, 1]))):
        m = SCVI(a, seed=0, **kw_model)
        t0 = time.perf_counter()
        m.train(max_epochs=epochs, progress=False, seed=0, **kw)
        wall = time.perf_counter() - t0
        et = m.history_["time"]
        mean_after_first = float(np.mean(et[1:])) if len(et) > 1 else et[0]
        rows.append(dict(data=tag, impl=gtag, n_obs=a.n_obs, epochs=epochs,
                         wall_s=round(wall, 2), epoch_s=json.dumps([round(x, 2) for x in et]),
                         mean_epoch_s_after_first=round(mean_after_first, 2)))
        models[gtag] = m
        print(rows[-1], flush=True)
    speedup = rows[0]["mean_epoch_s_after_first"] / rows[1]["mean_epoch_s_after_first"]
    e1 = models["1gpu"].get_elbo(indices=models["1gpu"].validation_indices)
    e2 = models["2gpu"].get_elbo(indices=models["2gpu"].validation_indices)
    print(f"RESULT {tag} speedup_2gpu_vs_1gpu={speedup:.2f}x elbo1={e1:.2f} elbo2={e2:.2f}")
    common.write_table(rows, f"ddp_scaling_{tag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", required=True,
                   choices=["inmemory", "oom_lock", "dask", "disjoint", "scaling"])
    p.add_argument("--data", choices=["small", "full"], default="small")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    common.set_threads(args.threads)
    {"inmemory": check_inmemory, "oom_lock": check_oom_lock, "dask": check_dask,
     "disjoint": check_disjoint}.get(args.check, lambda: check_scaling(args.data))()


if __name__ == "__main__":
    main()
