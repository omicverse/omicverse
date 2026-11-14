"""Shared helpers for the ov.vi benchmarks (data preparation, scib metrics).

Benchmark scripts may import scanpy / scvi-tools; omicverse itself does not.
"""
from __future__ import annotations

import os
import time

import numpy as np

SMOKE = "/scratch/users/steorra/tmp/oomsmoke_sc.h5ad"
SEAAD = "/scratch/users/steorra/analysis/omicverse_dev/ovfast/data/seaad/SEAAD_raw.h5ad"
CACHE = os.environ.get("OV_VI_BENCH_CACHE", "/scratch/users/steorra/tmp/vi/data")
HERE = os.path.dirname(os.path.abspath(__file__))


def set_threads(n=16):
    import torch
    torch.set_num_threads(n)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))


def read_rows(path, rows, key="X"):
    """Rows (sorted) of an on-disk CSR via anndataoom-fast's pread reader."""
    import scipy.sparse as sp
    from omicverse.vi._data import CSRFile
    f = CSRFile.open(path, key)
    ip = f.indptr
    rows = np.sort(np.asarray(rows))
    lens = ip[rows + 1] - ip[rows]
    nnz = int(lens.sum())
    data = np.empty(nnz, np.float32)
    idx = np.empty(nnz, np.int32)
    o = 0
    # merge runs of consecutive rows into one read
    br = np.flatnonzero(np.diff(rows) != 1) + 1
    for run in np.split(rows, br):
        e0, e1 = int(ip[run[0]]), int(ip[run[-1] + 1])
        f.read_into(e0, e1, data[o:o + e1 - e0], idx[o:o + e1 - e0], 8)
        o += e1 - e0
    indptr = np.concatenate([[0], np.cumsum(lens)])
    return sp.csr_matrix((data, idx, indptr), shape=(len(rows), f.shape[1]))


def _obs(path):
    import h5py
    from anndata.io import read_elem
    with h5py.File(path, "r") as f:
        return read_elem(f["obs"]), read_elem(f["var"])


def hvg(adata, n_top=2000):
    import scanpy as sc
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top, layer="counts")
    return adata[:, adata.var["highly_variable"].values].copy()


def load(name: str, n_top=2000):
    """'smoke' (5k SEA-AD cells) or 'seaad<N>k' (random N thousand SEA-AD cells),
    HVG-subset, raw counts in X and layers['counts'] (cached as h5ad)."""
    import anndata as ad
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f"{name}_hvg{n_top}.h5ad")
    if os.path.exists(out):
        return ad.read_h5ad(out)
    t = time.time()
    if name == "smoke":
        a = ad.read_h5ad(SMOKE)
    elif name.startswith("seaad") and name.endswith("k"):
        n = int(name[5:-1]) * 1000
        obs, var = _obs(SEAAD)
        rows = np.sort(np.random.default_rng(0).choice(len(obs), n, replace=False))
        a = ad.AnnData(read_rows(SEAAD, rows), obs=obs.iloc[rows].copy(), var=var)
    else:
        raise ValueError(name)
    a.var_names_make_unique()
    a.layers["counts"] = a.X.copy()
    keep = np.asarray((a.X > 0).sum(0)).ravel() >= 3
    a = a[:, keep].copy()
    a = hvg(a, n_top)
    for k in ("donor", "sample", "cell_type", "cell_type_broad"):
        if k in a.obs:
            a.obs[k] = a.obs[k].astype(str).astype("category")
    a.write_h5ad(out)
    print(f"prepared {name}: {a.shape} in {time.time() - t:.1f}s -> {out}")
    return a


def scib(adata, key, batch_key="donor", label_key="cell_type", n_jobs=16):
    """scib-metrics on ``adata.obsm[key]``: iLISI, cLISI, kBET, ASW (batch: BRAS),
    ASW label, NMI / ARI of Leiden, graph connectivity, isolated labels."""
    import scib_metrics as sm
    from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent
    X = np.asarray(adata.obsm[key], dtype=np.float32)
    batch = np.asarray(adata.obs[batch_key].astype(str).values)
    labels = np.asarray(adata.obs[label_key].astype(str).values)
    nn90 = pynndescent(X, n_neighbors=90, random_state=0, n_jobs=n_jobs)

    def sub(k):  # C-contiguous (umap's numba kernels reject strided views)
        return NeighborsResults(indices=np.ascontiguousarray(nn90.indices[:, :k]),
                                distances=np.ascontiguousarray(nn90.distances[:, :k]))

    nn15, nn50 = sub(15), sub(50)
    res = {}
    res["iLISI"] = float(sm.ilisi_knn(nn90, batch))
    res["cLISI"] = float(sm.clisi_knn(nn90, labels))
    res["kBET"] = float(sm.kbet_per_label(nn50, batch, labels))
    res["ASW_batch"] = float(sm.bras(X, labels, batch))
    res["ASW_label"] = float(sm.silhouette_label(X, labels))
    nmi_ari = sm.nmi_ari_cluster_labels_leiden(nn15, labels, optimize_resolution=True)
    res["NMI"] = float(nmi_ari["nmi"])
    res["ARI"] = float(nmi_ari["ari"])
    res["graph_conn"] = float(sm.graph_connectivity(nn15, labels))
    res["isolated_labels"] = float(sm.isolated_labels(X, labels, batch))
    return res


def write_table(rows, name):
    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(HERE, f"{name}.csv")
    if os.path.exists(csv):
        df = pd.concat([pd.read_csv(csv), df], ignore_index=True)
    df.to_csv(csv, index=False)
    return df


def summarize(df, by, cols):
    """mean ± sd table (markdown) over seeds."""
    g = df.groupby(by)[cols]
    m, s = g.mean(), g.std()
    out = m.copy().astype(object)
    for c in cols:
        out[c] = [f"{a:.4g} ± {b:.2g}" for a, b in zip(m[c], s[c].fillna(0))]
    return out.reset_index()


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)
