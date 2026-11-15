"""Data for the DRVI benchmarks: NeurIPS 2021 BMMC CITE-seq GEX, three donors from
three sites (s1d3, s2d1, s3d7; Luecken et al. 2021), raw counts, 2000 HVGs
(seurat_v3 per batch).  Writes

``bmmc3_hvg2000.h5ad`` (counts in X and layers['counts'], CSR; the AnnDataOOM /
Dask runs read the same file from disk).

Usage: python prep.py
"""
from __future__ import annotations

import os

import numpy as np

SRC = "/scratch/users/steorra/tmp/vi_tut/run/data/neurips2021_{}.h5ad"
DONORS = ("s1d3", "s2d1", "s3d7")
OUT = os.environ.get("OV_DRVI_DATA", "/scratch/users/steorra/tmp/vi_p2/data/drvi")


def main(n_top=2000):
    import anndata as ad
    import scanpy as sc
    os.makedirs(OUT, exist_ok=True)
    parts = []
    for d in DONORS:
        a = ad.read_h5ad(SRC.format(d))
        a.X = a.layers["counts"].astype(np.float32)
        del a.layers["counts"]
        a.obs = a.obs[["cell_type", "batch", "Site", "DonorNumber", "GEX_phase"]].copy()
        a.obsm = {}
        parts.append(a)
    a = ad.concat(parts, join="inner", merge="same")
    a.obs_names_make_unique()
    for k in a.obs:
        a.obs[k] = a.obs[k].astype(str).astype("category")
    keep = np.asarray((a.X > 0).sum(0)).ravel() >= 3
    a = a[:, keep].copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n_top, batch_key="batch",
                                layer="counts")
    a = a[:, a.var["highly_variable"].values].copy()
    a.var = a.var[["gene_id", "highly_variable"]].copy() if "gene_id" in a.var else a.var
    a.uns = {}
    p = os.path.join(OUT, f"bmmc3_hvg{n_top}.h5ad")
    a.write_h5ad(p)
    print(a)
    print(a.obs["batch"].value_counts().to_dict())
    print(a.obs["cell_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
