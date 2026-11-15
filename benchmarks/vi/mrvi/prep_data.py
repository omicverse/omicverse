"""Prepare the Wilk et al. 2020 COVID-19 PBMC cohort (13 donors: 7 COVID, 6 healthy;
Seq-Well scRNA-seq) for the MrVI validation: raw counts, basic QC, top HVGs, a
moderate cell count so 3 seeds x {ours, scvi-tools} finish in reasonable GPU time.

Output: benchmarks/vi/mrvi/wilk2020_prepped.h5ad
  X / layers['counts']: raw UMI counts (HVGs only)
  obs['donor_id']: sample_key (13 categories)
  obs['disease']: sample-level covariate ('COVID-19' / 'normal')
  obs['cell_type']: cell_type (coarse, for groupby / scib bio labels)
"""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

SRC = "/scratch/users/steorra/tmp/vi_p2/data/mrvi/wilk2020.h5ad"
OUT = "/scratch/users/steorra/tmp/vi_p2/data/mrvi/wilk2020_prepped.h5ad"
N_HVG = 2000
N_CELLS_TARGET = 12000
SEED = 0


def main():
    a = ad.read_h5ad(SRC)
    counts = a.raw.X.tocsr().astype(np.float32)
    assert np.allclose(counts.data, np.round(counts.data)), "raw.X is not integer counts"

    # basic QC on the raw counts
    n_genes = (counts > 0).sum(1).A1
    total = counts.sum(1).A1
    keep_cells = (n_genes >= 200) & (total >= 500)
    a = a[keep_cells].copy()
    counts = counts[keep_cells]
    keep_genes = np.asarray((counts > 0).sum(0)).ravel() >= 10
    a = a[:, keep_genes].copy()
    counts = counts[:, keep_genes]
    print(f"after QC: {a.n_obs} cells x {a.n_vars} genes")

    # HVGs on the normalized log1p data already in a.X (seurat-flavor dispersion,
    # computed manually to avoid a scanpy dependency in this prep script)
    X = a.X.tocsr()
    mean = np.asarray(X.mean(0)).ravel()
    sq = X.copy(); sq.data **= 2
    var = np.asarray(sq.mean(0)).ravel() - mean ** 2
    disp = var / np.maximum(mean, 1e-12)
    bins = pd.qcut(mean, 20, duplicates="drop")
    z = (disp - pd.Series(disp).groupby(bins).transform("mean")) / \
        pd.Series(disp).groupby(bins).transform("std").replace(0, 1)
    hvg_idx = np.argsort(-z.values)[:N_HVG]
    a = a[:, hvg_idx].copy()
    counts = counts[:, hvg_idx]
    a.layers["counts"] = counts
    print(f"after HVG selection: {a.n_vars} genes")

    # stratified subsample by donor to a moderate total size
    rng = np.random.default_rng(SEED)
    donor_counts = a.obs["donor_id"].value_counts()
    frac = min(1.0, N_CELLS_TARGET / a.n_obs)
    keep = np.zeros(a.n_obs, dtype=bool)
    for donor in donor_counts.index:
        idx = np.flatnonzero(a.obs["donor_id"].values == donor)
        n_keep = max(50, int(round(len(idx) * frac)))
        sel = rng.choice(idx, size=min(n_keep, len(idx)), replace=False)
        keep[sel] = True
    a = a[keep].copy()
    print(f"after subsampling: {a.n_obs} cells, donors={a.obs['donor_id'].nunique()}")
    print(a.obs["donor_id"].value_counts())
    print(a.obs["disease"].value_counts())

    a.obs["cell_type"] = a.obs["cell.type.coarse"].astype(str).astype("category")
    keep_obs = ["donor_id", "disease", "cell_type", "DPS", "DTF", "Admission", "Ventilated"]
    a.obs = a.obs[keep_obs].copy()
    a.var = a.var[["feature_name"]].copy()
    del a.obsm["X_pca"], a.obsm["X_umap"]
    a.X = a.layers["counts"]  # keep X == counts too (some tooling reads X by default)
    a.X.sort_indices()        # anndata-rs / AnnDataOOM rejects unsorted minor indices
    a.layers["counts"].sort_indices()
    a.write_h5ad(OUT)
    print(f"wrote {OUT}: {a.shape}")


if __name__ == "__main__":
    main()
