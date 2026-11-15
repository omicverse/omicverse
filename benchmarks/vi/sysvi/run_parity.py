"""SysVI vs scvi.external.SysVI 1.5.2 on real cross-system data.

Baron et al. 2016 pancreas (GEO GSE84133): human (4 donors) vs mouse (2
strains) scRNA-seq, MGI-orthologue-matched genes, 2000 HVGs — the same kind
of cross-species "system" integration task as the SysVI paper (Hrovatin et
al. 2023). ``system`` = human/mouse (the covariate corrected via
conditioning + the cycle loss), ``batch`` = donor/strain (a weaker covariate,
passed as ``categorical_covariate_keys``), ``cell_type`` = the authors'
labels (bio-conservation ground truth).

    python prep_data.py                      # writes $SYSVI_DATA/baron_pancreas_hs_mm.h5ad
    python run_parity.py --device cuda        # ov vs scvi-tools, 3 seeds
    python run_parity.py --device cuda --impl ov --oom   # + one AnnDataOOM run
    python run_parity.py --device cuda --impl ov --dask  # + one Dask run
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Bound BLAS/OpenMP thread pools *before* numpy/torch first import: on a
# heavily shared node torch.set_num_threads() alone (common.set_threads(),
# called from main() below) leaves numpy's / anndataoom's own thread pools at
# their (huge) default, causing catastrophic oversubscription.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # noqa: E402

DATA = os.environ.get("SYSVI_DATA", "/scratch/users/steorra/tmp/vi_p2/data/sysvi")
H5AD = os.path.join(DATA, "baron_pancreas_hs_mm.h5ad")

HP = dict(n_hidden=256, n_latent=15, n_layers=2, n_prior_components=5)


def _load():
    import anndata as ad
    if not os.path.exists(H5AD):
        raise FileNotFoundError(f"{H5AD}: run prep_data.py first")
    return ad.read_h5ad(H5AD)


def run_sysvi(a, impl, seed, device, max_epochs=None):
    if impl == "ov":
        from omicverse.vi import SysVI
        m = SysVI(a, batch_key="system", categorical_covariate_keys=["batch"], seed=seed, **HP)
        t = time.perf_counter()
        m.train(max_epochs=max_epochs, device=device, seed=seed, progress=False)
        tt = time.perf_counter() - t
        return (m.get_latent_representation(), tt, -m.get_elbo(indices=m.train_indices),
                -m.get_elbo(indices=m.validation_indices), len(m.history_["epoch"]))
    import scvi
    scvi.settings.seed = seed
    b = a.copy()
    scvi.external.SysVI.setup_anndata(b, batch_key="system", categorical_covariate_keys=["batch"])
    m = scvi.external.SysVI(b, n_latent=HP["n_latent"], n_hidden=HP["n_hidden"],
                            n_layers=HP["n_layers"], n_prior_components=HP["n_prior_components"])
    t = time.perf_counter()
    m.train(max_epochs=max_epochs, accelerator="gpu" if device.startswith("cuda") else "cpu",
            devices=1, enable_progress_bar=False)
    tt = time.perf_counter() - t
    return (m.get_latent_representation(), tt, -float(m.get_elbo(indices=m.train_indices)),
            -float(m.get_elbo(indices=m.validation_indices)), len(m.history["elbo_train"]))


def _write_h5ad_oom(a, path):
    # anndata-rs (ov.read(..., storage='disk')) rejects a CSR matrix with
    # unsorted minor (column) indices, which the Baron prep's boolean
    # column-subset (prep_data.py's HVG selection) leaves behind.
    a = a.copy()
    a.X.sort_indices()
    a.write_h5ad(path)


def run_oom_or_dask(a, kind, device, seed=0, max_epochs=None):
    """One run on AnnDataOOM (kind='oom') or Dask-backed AnnData (kind='dask');
    compares its latent to the in-memory run with the same seed."""
    import tempfile
    from omicverse.vi import SysVI
    path = os.path.join(common.HERE, "sysvi", "_baron_tmp.h5ad")
    if not os.path.exists(path):
        _write_h5ad_oom(a, path)
    # same block layout on every source (mem / AnnDataOOM / dask) -> identical
    # minibatches -> bitwise-identical trained latent (tests/vi/test_sysvi.py's
    # test_sysvi_disk_and_dask_give_same_latent_as_in_memory checks this too)
    block_kw = dict(shuffle="block", block_size=512, buffer_blocks=4)
    m_mem = SysVI(a, batch_key="system", categorical_covariate_keys=["batch"], seed=seed, **HP)
    m_mem.train(max_epochs=max_epochs, device=device, seed=seed, progress=False, **block_kw)
    z_mem = m_mem.get_latent_representation()

    if kind == "oom":
        import omicverse as ov
        ov.settings.cpu_init()
        a2 = ov.read(path, storage="disk")
    else:
        from omicverse.pp._dask import read_h5ad
        a2 = read_h5ad(path, chunk_size=512, device="cpu")
    m2 = SysVI(a2, batch_key="system", categorical_covariate_keys=["batch"], seed=seed, **HP)
    m2.train(max_epochs=max_epochs, device=device, seed=seed, progress=False, **block_kw)
    z2 = m2.get_latent_representation()
    src_kind = m2._source().kind
    corr = float(np.corrcoef(z_mem.ravel(), z2.ravel())[0, 1])
    print(f"[{kind}] source.kind={src_kind} corr(latent_mem, latent_{kind})={corr:.4f} "
          f"elbo_mem={m_mem.get_elbo():.2f} elbo_{kind}={m2.get_elbo():.2f}")
    return src_kind, corr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", nargs="+", default=["ov", "scvi"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--oom", action="store_true")
    p.add_argument("--dask", action="store_true")
    args = p.parse_args()
    common.set_threads(args.threads)
    a = _load()
    for seed in args.seeds:
        for impl in args.impl:
            z, tt, et, ev, ep = run_sysvi(a, impl, seed, args.device, args.max_epochs)
            b = a.copy()
            b.obsm["Z"] = z
            met = common.scib(b, "Z", batch_key="system", label_key="cell_type",
                              n_jobs=args.threads)
            r = dict(model="SysVI", data="baron_pancreas_hs_mm", impl=impl, seed=seed,
                     epochs=ep, neg_elbo_train=et, neg_elbo_validation=ev, train_time_s=tt,
                     **met)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                  flush=True)
            common.write_table([r], "parity_sysvi")
    if args.oom:
        run_oom_or_dask(a, "oom", args.device, seed=args.seeds[0], max_epochs=args.max_epochs)
    if args.dask:
        run_oom_or_dask(a, "dask", args.device, seed=args.seeds[0], max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
