"""Small helpers around ``ov.vi`` models: training curves, integration scores and
host / GPU memory tracking (used by the ov.vi tutorials).

No torch import at module level: :func:`plot_history` and
:func:`integration_metrics` work on a trained model's history / an embedding.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd


def plot_history(model, keys=("elbo_train", "elbo_validation"), ax=None, skip_first: int = 0,
                 title=None):
    """Plot per-epoch training curves of a trained ``ov.vi`` model.

    Arguments:
        model: Trained :class:`SCVI` / :class:`SCANVI` / :class:`TOTALVI`.
        keys: History keys to draw (missing keys are skipped).  Validation
            metrics recorded every *n* epochs are drawn at those epochs.
        ax: Matplotlib axes (new figure if None).
        skip_first: Leave out the first epochs (the initial ELBO dwarfs the rest).
        title: Axes title (default: model class and epoch count).
    Returns:
        The matplotlib axes.
    """
    import matplotlib.pyplot as plt
    h = getattr(model, "history_", None) or {}
    if not h:
        raise ValueError("the model has no training history (train it first)")
    n = len(h.get("epoch", h.get("time", [])))
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.2))
    for k in keys:
        v = h.get(k)
        if not v:
            continue
        v = np.asarray(v, dtype=float)
        if len(v) == n:
            x = np.arange(n)
        else:  # checked every n // len(v) epochs (the last epoch of each period)
            step = max(n // len(v), 1)
            x = np.arange(1, len(v) + 1) * step - 1
        keep = x >= skip_first
        ax.plot(x[keep], v[keep], label=k, lw=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("negative ELBO" if all("elbo" in k for k in keys) else "value")
    ax.set_title(title or f"{type(model).__name__}: {n} epochs")
    ax.legend(frameon=False)
    return ax


def integration_metrics(adata, obsm_keys, batch_key: str = "batch",
                        label_key: str = "cell_type", n_jobs: int = 8,
                        subsample: int | None = None, seed: int = 0,
                        nmi_ari: bool = True) -> pd.DataFrame:
    """scib-style batch-removal and bio-conservation scores of embeddings.

    Computed with ``scib-metrics`` (``pip install scib-metrics``) on a shared
    90-nearest-neighbour graph per embedding, as in the ov.vi benchmarks:

    * batch removal: ``iLISI`` (scaled LISI over batches), ``kBET`` (per-label
      acceptance rate), ``ASW_batch`` (batch silhouette, BRAS);
    * bio conservation: ``cLISI``, ``ASW_label`` (cell-type silhouette),
      ``NMI`` / ``ARI`` (Leiden clusters vs labels, resolution optimised),
      ``graph_conn`` (graph connectivity per label).

    All are scaled to [0, 1], higher is better.

    Arguments:
        adata: AnnData / AnnDataOOM with the embeddings in ``obsm``.
        obsm_keys: One key or a list of keys of ``adata.obsm``.
        batch_key, label_key: ``obs`` columns with batch and cell-type labels.
        n_jobs: Threads for the nearest-neighbour search.
        subsample: Score a random subset of this many cells (None = all).
        seed: Seed of the subsample and the kNN search.
        nmi_ari: Also compute NMI / ARI (Leiden over a resolution grid; the
            slowest metric).
    Returns:
        DataFrame, one row per embedding.
    """
    try:
        import scib_metrics as sm
        from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("integration_metrics needs scib-metrics: pip install scib-metrics") from e
    keys = [obsm_keys] if isinstance(obsm_keys, str) else list(obsm_keys)
    n = adata.n_obs
    idx = np.arange(n)
    if subsample is not None and subsample < n:
        idx = np.sort(np.random.default_rng(seed).choice(n, subsample, replace=False))
    batch = np.asarray(adata.obs[batch_key].astype(str).values)[idx]
    labels = np.asarray(adata.obs[label_key].astype(str).values)[idx]
    rows = {}
    for key in keys:
        X = np.ascontiguousarray(np.asarray(adata.obsm[key], dtype=np.float32)[idx])
        nn90 = pynndescent(X, n_neighbors=90, random_state=seed, n_jobs=n_jobs)

        def sub(k, nn=nn90):  # contiguous views (umap's numba kernels reject strided ones)
            return NeighborsResults(indices=np.ascontiguousarray(nn.indices[:, :k]),
                                    distances=np.ascontiguousarray(nn.distances[:, :k]))

        nn15, nn50 = sub(15), sub(50)
        r = {
            "iLISI": float(sm.ilisi_knn(nn90, batch)),
            "kBET": float(sm.kbet_per_label(nn50, batch, labels)),
            "ASW_batch": float(sm.bras(X, labels, batch)),
            "cLISI": float(sm.clisi_knn(nn90, labels)),
            "ASW_label": float(sm.silhouette_label(X, labels)),
        }
        if nmi_ari:
            na = sm.nmi_ari_cluster_labels_leiden(nn15, labels, optimize_resolution=True)
            r["NMI"], r["ARI"] = float(na["nmi"]), float(na["ari"])
        r["graph_conn"] = float(sm.graph_connectivity(nn15, labels))
        rows[key] = r
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "embedding"
    return df


class MemoryMonitor:
    """Track wall time, host RSS and GPU memory of a block of code.

    A background thread samples the process RSS (``psutil``) and, when CUDA is
    available, the device-wide used memory every ``period`` seconds::

        with ov.vi.MemoryMonitor() as mon:
            model.train()
        mon                      # MemoryMonitor(wall 12.3 s, RSS 2.10 -> peak 2.45 GB, ...)
        mon.summary()            # the same numbers as a dict

    Attributes (GB / s): ``rss_start_gb``, ``rss_peak_gb``, ``rss_growth_gb``,
    ``gpu_peak_gb`` (None without CUDA), ``wall_s``.
    """

    def __init__(self, period: float = 0.1, gpu: bool = True):
        import psutil
        self._proc = psutil.Process()
        self.period = period
        self._gpu = gpu
        self.rss_start_gb = self.rss_peak_gb = self.gpu_peak_gb = self.wall_s = None
        self._stop = threading.Event()
        self._thread = None

    def _gpu_used(self):
        if not self._gpu:
            return None
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            free, total = torch.cuda.mem_get_info()
            return (total - free) / 2**30
        except Exception:
            return None

    def _sample(self):
        rss = self._proc.memory_info().rss / 2**30
        self.rss_peak_gb = max(self.rss_peak_gb, rss)
        g = self._gpu_used()
        if g is not None:
            self.gpu_peak_gb = g if self.gpu_peak_gb is None else max(self.gpu_peak_gb, g)

    def _run(self):
        while not self._stop.wait(self.period):
            self._sample()

    def __enter__(self):
        self.rss_start_gb = self.rss_peak_gb = self._proc.memory_info().rss / 2**30
        self.gpu_peak_gb = self._gpu_used()
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()
        self.wall_s = time.perf_counter() - self._t0
        return False

    @property
    def rss_growth_gb(self):
        if self.rss_start_gb is None:
            return None
        return self.rss_peak_gb - self.rss_start_gb

    def summary(self) -> dict:
        return {"wall_s": self.wall_s, "rss_start_gb": self.rss_start_gb,
                "rss_peak_gb": self.rss_peak_gb, "rss_growth_gb": self.rss_growth_gb,
                "gpu_peak_gb": self.gpu_peak_gb}

    def __repr__(self):
        if self.wall_s is None:
            return "MemoryMonitor(not run)"
        gpu = "" if self.gpu_peak_gb is None else f", GPU used peak {self.gpu_peak_gb:.2f} GB"
        return (f"MemoryMonitor(wall {self.wall_s:.1f} s, RSS {self.rss_start_gb:.2f} -> "
                f"peak {self.rss_peak_gb:.2f} GB (+{self.rss_growth_gb:.2f}){gpu})")


__all__ = ["plot_history", "integration_metrics", "MemoryMonitor"]
