"""Baron et al. 2016 pancreas (GEO GSE84133), human (4 donors) vs mouse (2 strains).

Cross-species "system" integration as in the SysVI paper (Hrovatin et al.):
* per species: normalize_total(1e4) + log1p on all genes;
* mouse genes mapped to human by MGI one-to-one orthologs
  (HOM_MouseHumanSequence.rpt, informatics.jax.org);
* 2000 HVGs (scanpy 'seurat' flavour on the log data, batch_key='system');
* obs: system (human / mouse), batch (donor / strain), cell_type (authors' labels,
  human 't_cell' merged with mouse 'T_cell', otherwise as given).

    python prep_data.py   # writes $DATA/baron_pancreas_hs_mm.h5ad
"""
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

DATA = os.environ.get("SYSVI_DATA", "/scratch/users/steorra/tmp/vi_p2/data/sysvi")
OUT = os.path.join(DATA, "baron_pancreas_hs_mm.h5ad")


def _read(name):
    df = pd.read_csv(os.path.join(DATA, name), index_col=0)
    obs = df[["barcode", "assigned_cluster"]].copy()
    X = sp.csr_matrix(df.drop(columns=["barcode", "assigned_cluster"]).values.astype(np.float32))
    a = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=df.columns[2:]))
    a.obs["batch"] = name.split("_")[1]
    return a


def one2one():
    t = pd.read_csv(os.path.join(DATA, "HOM_MouseHumanSequence.rpt"), sep="\t")
    t = t[["DB Class Key", "Common Organism Name", "Symbol"]]
    mm = t[t["Common Organism Name"].str.startswith("mouse")]
    hs = t[t["Common Organism Name"] == "human"]
    mm = mm[~mm["DB Class Key"].duplicated(keep=False)]  # one gene per class and species
    hs = hs[~hs["DB Class Key"].duplicated(keep=False)]
    m = mm.merge(hs, on="DB Class Key", suffixes=("_mm", "_hs"))
    return dict(zip(m["Symbol_mm"], m["Symbol_hs"]))


def main():
    files = sorted(f for f in os.listdir(DATA) if f.endswith("_umifm_counts.csv.gz"))
    parts = {}
    for sp_ in ("human", "mouse"):
        a = ad.concat([_read(f) for f in files if f.split("_")[1].startswith(sp_)])
        a.obs_names_make_unique()
        a.layers["counts"] = a.X.copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        parts[sp_] = a
    o = one2one()
    mm = parts["mouse"]
    keep = [g in o for g in mm.var_names]
    mm = mm[:, keep].copy()
    mm.var_names = [o[g] for g in mm.var_names]
    mm = mm[:, ~mm.var_names.duplicated()].copy()
    hs = parts["human"]
    common = hs.var_names.intersection(mm.var_names)
    hs, mm = hs[:, common].copy(), mm[:, common].copy()
    hs.obs["system"], mm.obs["system"] = "human", "mouse"
    a = ad.concat([hs, mm], merge="same")
    a.obs["cell_type"] = a.obs["assigned_cluster"].astype(str).replace({"t_cell": "T_cell"})
    for k in ("system", "batch", "cell_type"):
        a.obs[k] = a.obs[k].astype("category")
    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat", batch_key="system")
    a = a[:, a.var["highly_variable"].values].copy()
    a.X = sp.csr_matrix(a.X, dtype=np.float32)
    a.X.sort_indices()  # boolean column-subsetting above can leave indices unsorted,
                        # which anndata-rs (ov.read(..., storage='disk')) rejects
    a.write_h5ad(OUT)
    print(a)
    print(pd.crosstab(a.obs["cell_type"], a.obs["system"]))


if __name__ == "__main__":
    main()
