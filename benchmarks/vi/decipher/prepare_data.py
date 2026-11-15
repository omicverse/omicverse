"""Build the Decipher validation dataset: the erythroid-differentiation lineage
(HSC -> MK/E prog -> Proerythroblast -> Erythroblast -> Normoblast -> Reticulocyte)
of the NeurIPS 2021 BMMC CITE-seq donors s1d3, s2d1, s3d7 (already local at
/scratch/users/steorra/tmp/vi_tut/run/data/).  These donors carry
``obs['GEX_pseudotime_order']``, a ground-truth pseudotime (Palantir, from the
NeurIPS 2021 competition task 3) defined for exactly this lineage -- the same
kind of normal haematopoietic differentiation trajectory the Decipher paper
validates on.

Output: /scratch/users/steorra/tmp/vi_p2/data/decipher/erythroid_neurips2021.h5ad
(raw counts in ``layers['counts']`` and ``.X``, top 2000 HVGs, ``obs['donor']``,
``obs['cell_type']`` and ``obs['GEX_pseudotime_order']`` (NaN off the lineage).
"""
import anndata as ad
import numpy as np
import scanpy as sc

LINEAGE = ["HSC", "MK/E prog", "Proerythroblast", "Erythroblast", "Normoblast",
           "Reticulocyte"]
DONORS = ["s1d3", "s2d1", "s3d7"]
SRC = "/scratch/users/steorra/tmp/vi_tut/run/data/neurips2021_{}.h5ad"
OUT = "/scratch/users/steorra/tmp/vi_p2/data/decipher/erythroid_neurips2021.h5ad"

parts = []
for d in DONORS:
    a = ad.read_h5ad(SRC.format(d))
    a = a[a.obs["cell_type"].isin(LINEAGE)].copy()
    a.obs["donor"] = d
    a.X = a.layers["counts"].copy()
    del a.layers["counts"]
    a.obsm.clear()
    print(d, a.n_obs, a.obs["cell_type"].value_counts().reindex(LINEAGE).to_dict())
    parts.append(a)

full = ad.concat(parts, join="outer", index_unique="-")
full.obs["cell_type"] = full.obs["cell_type"].astype("category").cat.set_categories(LINEAGE)
print("combined:", full.shape, "with ground-truth pseudotime:",
      int(full.obs["GEX_pseudotime_order"].notna().sum()))

# HVGs on a log-normalized copy (Decipher itself trains on the raw counts)
norm = full.copy()
sc.pp.normalize_total(norm, target_sum=1e4)
sc.pp.log1p(norm)
sc.pp.highly_variable_genes(norm, n_top_genes=2000, flavor="seurat")
full = full[:, norm.var["highly_variable"].values].copy()
full.layers["counts"] = full.X.copy()

full.write_h5ad(OUT)
print("wrote", OUT, full.shape)
