"""Disk training on the full SEA-AD MTG atlas (spec Validation 4).

    python disk_seaad.py --mode oom    # AnnDataOOM: cpu_gpu_mixed_init + storage='disk'
    python disk_seaad.py --mode dask   # Dask: GPU mode + storage='disk' (cupy row blocks)

ov.read(storage='disk') -> ov.pp.qc -> ov.pp.preprocess (shiftlog|pearson, 2000
HVGs; raw counts stay on disk as layers['counts']) -> ov.vi.SCVI(layer='counts',
genes='highly_variable_features', batch_key='donor') -> train on the GPU ->
streamed latent.  Records per-epoch time, host RSS (sampled every 0.2 s:
baseline before training, peak during training / inference), GPU memory
(device-wide used and torch peak), and scib-metrics on a 50k-cell subsample of
the latent.  Appends to disk_seaad.csv.

GPU mode here is ``ov.settings.mode = 'gpu'`` without ``gpu_init()``'s
rapids_singlecell / rmm imports (no wheels for this OS); the Dask path itself
only needs cupy.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


class Sampler:
    """Background RSS / GPU-memory sampler."""

    def __init__(self, period=0.2):
        import psutil
        import torch
        self.proc = psutil.Process()
        self.torch = torch
        self.period = period
        self.peak_rss = 0
        self.peak_gpu = 0
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
            if self.torch.cuda.is_available():
                free, total = self.torch.cuda.mem_get_info()
                self.peak_gpu = max(self.peak_gpu, total - free)
            time.sleep(self.period)

    def start(self):
        self.peak_rss = self.proc.memory_info().rss
        self.peak_gpu = 0
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join()
        return self.peak_rss / 2**30, self.peak_gpu / 2**30


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["oom", "dask"], default="oom")
    p.add_argument("--path", default=common.SEAAD)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--buffer-blocks", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=20000, help="dask rows per block")
    p.add_argument("--subsample", type=int, default=50000)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    common.set_threads(args.threads)
    import torch
    import omicverse as ov
    from omicverse.vi import SCVI
    res = dict(mode=args.mode, tag=args.tag)
    t = time.perf_counter()
    if args.mode == "oom":
        ov.settings.cpu_gpu_mixed_init()
        adata = ov.read(args.path, storage="disk")
    else:
        ov.settings.mode = "gpu"
        adata = ov.read(args.path, storage="disk", chunk_size=args.chunk_size)
    res["read_s"] = time.perf_counter() - t
    res["n_obs_raw"] = adata.n_obs
    t = time.perf_counter()
    adata = ov.pp.qc(adata, tresh={"mito_perc": 0.2, "nUMIs": 500, "detected_genes": 250},
                     doublets=False)
    res["qc_s"] = time.perf_counter() - t
    t = time.perf_counter()
    ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000)
    res["preprocess_s"] = time.perf_counter() - t
    res["n_obs"] = adata.n_obs
    res["n_hvg"] = int(np.asarray(adata.var["highly_variable_features"].values).sum())
    print({k: v for k, v in res.items()}, flush=True)

    smp = Sampler()
    x = torch.randn(512, 512, device="cuda")  # CUDA context + cuBLAS outside the baseline
    (x @ x).sum().item()
    rss0 = smp.proc.memory_info().rss / 2**30
    m = SCVI(adata, layer="counts", batch_key="donor", genes="highly_variable_features")
    torch.cuda.reset_peak_memory_stats()
    smp.start()
    t = time.perf_counter()
    m.train(max_epochs=args.epochs, device="cuda", block_size=args.block_size,
            buffer_blocks=args.buffer_blocks, progress=False)
    torch.cuda.synchronize()
    res["train_s"] = time.perf_counter() - t
    rss_peak_train, gpu_peak_train = smp.stop()
    res["source"] = m._source().kind
    res["epochs"] = len(m.history_["time"])
    res["epoch_s"] = json.dumps([round(x, 1) for x in m.history_["time"]])
    res["epoch_s_mean_after_first"] = float(np.mean(m.history_["time"][1:])) \
        if res["epochs"] > 1 else float(m.history_["time"][0])
    res["elbo_train_last"] = m.history_["elbo_train"][-1]
    res["rss_before_train_gb"] = rss0
    res["rss_peak_train_gb"] = rss_peak_train
    res["rss_growth_train_gb"] = rss_peak_train - rss0
    res["gpu_used_peak_train_gb"] = gpu_peak_train
    res["torch_peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
    print(res, flush=True)

    smp.start()
    t = time.perf_counter()
    z = m.get_latent_representation()
    res["latent_s"] = time.perf_counter() - t
    rss_peak_inf, gpu_peak_inf = smp.stop()
    res["rss_peak_latent_gb"] = rss_peak_inf
    res["maxrss_process_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    os.makedirs(os.path.join(common.CACHE, "latents"), exist_ok=True)
    np.save(os.path.join(common.CACHE, "latents", f"seaad_full_{args.mode}{args.tag}.npy"), z)

    import anndata as ad
    rs = np.random.default_rng(0)
    idx = np.sort(rs.choice(adata.n_obs, min(args.subsample, adata.n_obs), replace=False))
    obs = adata.obs.iloc[idx][["donor", "cell_type"]].copy()
    sub = ad.AnnData(obs=obs)
    sub.obsm["Z"] = z[idx]
    t = time.perf_counter()
    res.update(common.scib(sub, "Z", n_jobs=args.threads))
    res["metrics_s"] = time.perf_counter() - t
    print(res, flush=True)
    common.write_table([res], "disk_seaad")


if __name__ == "__main__":
    main()
