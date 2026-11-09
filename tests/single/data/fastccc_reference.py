"""Build tests/single/data/fastccc_reference.npz from original FastCCC 1.0.1 on a PBMC3k subset."""
import sys, glob, warnings, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, zipfile
warnings.filterwarnings("ignore")
import fastccc
from loguru import logger; logger.remove()
a = ad.read_h5ad(sys.argv[1]); db = sys.argv[2]; out = sys.argv[3]
rng = np.random.default_rng(0)
cap = {"CD4 T cells": 70, "CD14+ Monocytes": 60, "B cells": 45, "CD8 T cells": 35, "NK cells": 32,
       "FCGR3A+ Monocytes": 30, "Dendritic cells": 20, "Megakaryocytes": 12}
idx = np.concatenate([rng.choice(np.where(a.obs.cell_type == k)[0], v, replace=False) for k, v in cap.items()])
a = a[np.sort(idx)].copy()
genes = set(pd.read_csv(f"{db}/gene_table.csv")["hgnc_symbol"])
a = a[:, [g in genes for g in a.var_names]].copy()
a.obs["cell_type"] = a.obs["cell_type"].astype(str)
X = sp.csr_matrix(a.X)
res = {}
modes = {"default": ("Mean", "Minimum", "Arithmetic", False), "q90_avg_geo": ("Quantile_0.9", "Average", "Geometric", False),
         "deg": ("Q3", "Minimum", "Arithmetic", True)}
for name, (s, c, l, d) in modes.items():
    st, pv, _ = fastccc.statistical_analysis_method(db, None, a, "hgnc_symbol", single_unit_summary=s, complex_aggregation=c,
                                                    LR_combination=l, use_DEG=d, meta_key="cell_type", save_path=out + "_tmp/")
    res[name] = (st, pv)
fastccc.Cauchy_combination_of_statistical_analysis_methods(db, None, a, "hgnc_symbol", meta_key="cell_type", save_path=out + "_tmpc/")
pc = pd.read_csv(glob.glob(out + "_tmpc/*_Cauchy_pvals.tsv")[0], index_col=0, sep="\t")
sc = pd.read_csv(glob.glob(out + "_tmpc/*_average_interactions_strength.tsv")[0], index_col=0, sep="\t")
res["cauchy"] = (sc, pc)
st0 = res["default"][0]
payload = dict(data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
               genes=np.array(a.var_names, dtype="U"), labels=a.obs["cell_type"].to_numpy(dtype="U"),
               pairs=np.array(st0.index, dtype="U"), interactions=np.array(st0.columns, dtype="U"))
for name, (st, pv) in res.items():
    assert list(st.index) == list(st0.index) and list(st.columns) == list(st0.columns)
    payload[f"{name}_strength"] = st.to_numpy()
    payload[f"{name}_pvals"] = pv.to_numpy()
np.savez_compressed(out, **payload)
print(a.shape, a.obs.cell_type.value_counts().to_dict(), st0.shape, {k: (v[1] < .05).to_numpy().sum() for k, v in res.items()})

# Usage (in an environment with fastccc==1.0.1):
#   python fastccc_reference.py pbmc3k_lognorm.h5ad <cellphonedb v5 folder> fastccc_reference.npz
# fastccc_cpdb_subset.zip is cellphonedb.zip (v5.0.0) pruned to the fixture's genes; FastCCC keeps an
# interaction only when every subunit is measured, so the pruning does not change any result.
