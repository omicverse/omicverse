"""Shared helpers for the ov.ge.SIMBA vs. original SIMBA+PBG parity
benchmark (SPEC_ge_simba.md section 3).

Reference run (original simba 1.2 / simba_pbg, bioconda env
``/scratch/users/steorra/env/simbaref``): see
``/scratch/users/steorra/tmp/simba_parity/ref/README.md``.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

REF = "/scratch/users/steorra/tmp/simba_parity/ref"

# Canonical scanpy pbmc3k tutorial marker genes per cell type (the same
# labels used in pbmc3k_processed.h5ad's obs['louvain'] / our obs['celltype']).
PBMC3K_MARKERS = {
    "CD4 T cells": ["IL7R"],
    "CD14+ Monocytes": ["CD14", "LYZ"],
    "B cells": ["MS4A1"],
    "CD8 T cells": ["CD8A"],
    "NK cells": ["GNLY", "NKG7"],
    "FCGR3A+ Monocytes": ["FCGR3A", "MS4A7"],
    "Dendritic cells": ["FCER1A", "CST3"],
    "Megakaryocytes": ["PPBP"],
}


def set_threads(n: int = 4):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(k, str(n))
    import torch
    torch.set_num_threads(n)


def seed_everything(seed: int):
    import random
    import numpy as _np
    import torch
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)


def alias_layer(adata, want: str, have: str):
    """Make ``adata.layers[want]`` available by aliasing ``layers[have]`` if
    ``want`` is missing -- bioconda simba 1.2 renamed the discretized-matrix
    / mutual-kNN-connectivity layer from ``'disc'``/``'conn'`` (the GitHub
    dev clone this port started from) to a single ``'simba'`` key; the math
    is identical, only the key name differs (see README "Patches" #5)."""
    if want not in adata.layers and have in adata.layers:
        adata.layers[want] = adata.layers[have]
    return adata


def load_ref_training_stats(path: str) -> list[dict]:
    """Parse PBG's JSON-lines ``training_stats.json`` (one dict per bucket/
    chunk event; the last entry with an ``eval_stats_chunk_avg`` key per
    epoch is the post-epoch held-out eval)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def ref_final_eval(path: str) -> dict:
    stats = load_ref_training_stats(path)
    for d in reversed(stats):
        if "eval_stats_chunk_avg" in d:
            return d["eval_stats_chunk_avg"]["metrics"]
    raise ValueError(f"no eval_stats_chunk_avg found in {path}")


def ref_eval_trajectory(path: str) -> list[dict]:
    stats = load_ref_training_stats(path)
    return [d["eval_stats_chunk_avg"]["metrics"] for d in stats if "eval_stats_chunk_avg" in d]


def _neighbors(X: np.ndarray, k: int = 30):
    from sklearn.neighbors import NearestNeighbors

    from scib_metrics.nearest_neighbors import NeighborsResults

    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, ind = nn.kneighbors(X)
    return NeighborsResults(indices=ind, distances=dist)


def ari_nmi_leiden(X: np.ndarray, labels, seed: int = 0) -> dict:
    import scib_metrics

    nbrs = _neighbors(X, k=30)
    res = scib_metrics.nmi_ari_cluster_labels_leiden(
        nbrs, np.asarray(labels), optimize_resolution=True, seed=seed)
    return {"ari": float(res["ari"]), "nmi": float(res["nmi"])}


def knn_label_accuracy(X: np.ndarray, labels, k: int = 15, seed: int = 0) -> float:
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf, X, np.asarray(labels), cv=5)
    return float(scores.mean())


def ilisi(X: np.ndarray, batches, k: int = 60) -> float:
    import scib_metrics

    nbrs = _neighbors(X, k=k)
    return float(scib_metrics.ilisi_knn(nbrs, np.asarray(batches)))


def kbet_score(X: np.ndarray, batches, k: int = 60) -> float:
    """kBET acceptance rate (higher = more batch mixing); ``scib_metrics.kbet``
    returns ``(acceptance_rate, test_statistics, p_values)``, not a scalar --
    only the first element is the summary score."""
    import scib_metrics

    nbrs = _neighbors(X, k=k)
    try:
        acceptance_rate, _stat, _p = scib_metrics.kbet(nbrs, np.asarray(batches))
        return float(acceptance_rate)
    except Exception:  # kBET can fail to converge on tiny/degenerate batches
        return float("nan")


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def top_k_genes(cmp_adata, metric: str, k: int = 50) -> list[str]:
    s = pd.Series(np.asarray(cmp_adata.var[metric]), index=cmp_adata.var_names)
    return list(s.sort_values(ascending=False).index[:k])


def align_rows(adata, ids):
    """Reorder ``adata`` to the given id order (PBG writes embeddings in its
    own internal entity-alias order, not the original obs_names order)."""
    return adata[pd.Index(ids)].copy()
