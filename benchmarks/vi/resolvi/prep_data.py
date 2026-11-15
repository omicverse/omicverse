"""Build the validation AnnData: 10x Xenium FFPE human breast cancer Rep1
(Janesick et al. 2023), 313-gene panel, supervised cell-type labels from 10x
(Cell_Barcode_Type_Matrices.xlsx, sheet 'Xenium R1 Fig1-5 (supervised)').

Downloads (cf.10xgenomics.com / cdn.10xgenomics.com):
  Xenium_FFPE_Human_Breast_Cancer_Rep1_cell_feature_matrix.h5
  Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.csv.gz
  Cell_Barcode_Type_Matrices.xlsx
Writes xenium_breast_r1.h5ad (all cells with >= 10 counts) and
xenium_breast_r1_crop.h5ad (a contiguous tissue window used by the benchmark).
"""
import os
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

D = sys.argv[1] if len(sys.argv) > 1 else "/scratch/users/steorra/tmp/vi_p2/data/resolvi"

with h5py.File(os.path.join(D, "rep1_cell_feature_matrix.h5"), "r") as f:
    m = f["matrix"]
    shape = tuple(m["shape"][:])
    X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape).T.tocsr()
    barcodes = [b.decode() for b in m["barcodes"][:]]
    feat = m["features"]
    names = [b.decode() for b in feat["name"][:]]
    ftype = [b.decode() for b in feat["feature_type"][:]]
keep_g = np.array([t == "Gene Expression" for t in ftype])
X = X[:, keep_g].astype(np.float32)
var = pd.DataFrame(index=pd.Index(np.array(names)[keep_g], name="gene"))
obs = pd.DataFrame(index=pd.Index([str(b) for b in barcodes], name="cell_id"))
cells = pd.read_csv(os.path.join(D, "rep1_cells.csv.gz"))
cells.index = cells["cell_id"].astype(str)
obs = obs.join(cells.drop(columns="cell_id"))
lab = pd.read_excel(os.path.join(D, "Cell_Barcode_Type_Matrices.xlsx"),
                    sheet_name="Xenium R1 Fig1-5 (supervised)")
lab.index = lab["Barcode"].astype(str)
obs["cell_type"] = lab["Cluster"].reindex(obs.index).fillna("Unlabeled").astype("category")
a = ad.AnnData(X=X, obs=obs, var=var)
a.layers["counts"] = a.X.copy()
a.obsm["spatial"] = a.obs[["x_centroid", "y_centroid"]].to_numpy(np.float64)
a = a[np.asarray(a.X.sum(1)).ravel() >= 10].copy()
a.obs["slide"] = "rep1"
print(a)
print(a.obs["cell_type"].value_counts())
a.write_h5ad(os.path.join(D, "xenium_breast_r1.h5ad"))
# crop: left part of the slide (x below the 45th percentile), ~75k cells
x = a.obsm["spatial"][:, 0]
crop = a[x < np.quantile(x, 0.45)].copy()
print("crop", crop.shape)
print(crop.obs["cell_type"].value_counts())
crop.write_h5ad(os.path.join(D, "xenium_breast_r1_crop.h5ad"))
