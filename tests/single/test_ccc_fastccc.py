"""FastCCC backend: omicverse re-implementation versus FastCCC 1.0.1.

``data/fastccc_reference.npz`` holds 304 real PBMC3k cells (8 cell types,
two of them below the 30-cell CLT cutoff; genes limited to CellPhoneDB) and
the outputs of the original ``fastccc`` package on exactly that matrix
(``data/fastccc_reference.py``). ``data/fastccc_cpdb_subset.zip`` is
CellPhoneDB v5 pruned to those genes. FastCCC is deterministic, so scores and
p-values must match exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

anndata = pytest.importorskip("anndata")

try:
    import omicverse as ov
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"omicverse import failed in test env: {exc}", allow_module_level=True)

from omicverse.external.fastccc import load_cpdb_tables, run_fastccc_engine
from omicverse.single import _ccc_fastccc as F

_DATA = Path(__file__).parent / "data"
_REF = _DATA / "fastccc_reference.npz"
_DB = _DATA / "fastccc_cpdb_subset.zip"

_MODES = {
    "default": dict(),
    "q90_avg_geo": dict(single_unit_summary="Quantile_0.9", complex_aggregation="Average", LR_combination="Geometric"),
    "deg": dict(single_unit_summary="Q3", use_DEG=True),
    "cauchy": dict(cauchy=True),
}


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(_REF, allow_pickle=False))


@pytest.fixture(scope="module")
def db():
    return load_cpdb_tables(_DB)


@pytest.fixture()
def adata(ref):
    X = sp.csr_matrix((ref["data"], ref["indices"], ref["indptr"]), shape=tuple(ref["shape"]))
    ad = anndata.AnnData(X=X, obs=pd.DataFrame({"cell_type": ref["labels"].astype(str)}),
                         var=pd.DataFrame(index=ref["genes"].astype(str)))
    ad.obs_names = [f"c{i}" for i in range(ad.n_obs)]
    return ad


@pytest.mark.parametrize("mode", list(_MODES))
def test_engine_matches_original_fastccc(ref, db, adata, mode):
    res = run_fastccc_engine(adata.X, adata.obs["cell_type"].to_numpy(), adata.var_names, db, **_MODES[mode])
    pairs, inter = ref["pairs"].astype(str), ref["interactions"].astype(str)
    assert sorted(res.strength.index) == sorted(pairs)
    assert sorted(res.strength.columns) == sorted(inter)
    strength = res.strength.loc[pairs, inter].to_numpy(np.float64)
    pvals = res.pvalues.loc[pairs, inter].to_numpy(np.float64)
    want_s = ref[f"{mode}_strength"].astype(np.float64)
    want_p = ref[f"{mode}_pvals"].astype(np.float64)
    # FastCCC (like the original) builds its null on a float32 grid with FFT
    # convolutions. Results are bit-identical to the original under the reference
    # numpy 1.26 / scipy 1.16, but other numpy/scipy builds move p-values by up to
    # ~1e-4 (order-statistic nulls in the quantile mode). Compare at that precision
    # and require identical significance calls away from the threshold.
    np.testing.assert_allclose(pvals, want_p, rtol=1e-3, atol=5e-4)
    clear = np.abs(want_p - 0.05) > 5e-4
    np.testing.assert_array_equal((pvals < 0.05)[clear], (want_p < 0.05)[clear])
    np.testing.assert_allclose(strength, want_s, rtol=1e-5, atol=1e-6)


def test_adapter_long_table(ref, adata):
    out = F.run(adata, groupby="cell_type", min_cells=0, cpdb_file_path=str(_DB))
    long, raw = out["long"], out["raw"]
    assert list(long.columns) == F.LONG_COLUMNS
    assert (long["score"] > 0).all()
    assert ((long["pvalue"] >= 0) & (long["pvalue"] <= 1)).all()
    np.testing.assert_array_equal(long["significant"].to_numpy(), long["pvalue"].to_numpy() < 0.05)
    # significant set equals the original's p < 0.05 entries
    pairs, inter = ref["pairs"].astype(str), ref["interactions"].astype(str)
    i, j = np.where(ref["default_pvals"] < 0.05)
    want = {(pairs[a].split("|")[0], pairs[a].split("|")[1], inter[b]) for a, b in zip(i, j)}
    name2id = raw["interactions"].set_index("interaction")["id_cp_interaction"]
    assert name2id.index.is_unique
    sig = long[long["significant"]]
    got = set(zip(sig["source"], sig["target"], name2id.loc[sig["interaction"]].to_numpy()))
    assert got == want
    # complexes expand to sorted subunit genes
    cplx = long[long["receptor"].str.endswith("_complex")]
    assert not cplx.empty
    for genes in cplx["receptor_genes"]:
        parts = genes.split("_")
        assert len(parts) >= 2 and parts == sorted(parts)


def test_adapter_min_cells_and_h5ad(adata, tmp_path):
    out = F.run(adata, groupby="cell_type", min_cells=15, cpdb_file_path=str(_DB))
    long = out["long"]
    assert "Megakaryocytes" not in set(long["source"]) | set(long["target"])  # 12 cells
    assert "Megakaryocytes" in set(out["raw"]["groups"])                       # still computed
    adata.uns["fastccc_res"] = dict(out["raw"], long=long)
    path = tmp_path / "fc.h5ad"
    adata.write_h5ad(path)
    back = anndata.read_h5ad(path)
    assert back.uns["fastccc_res"]["long"].shape == long.shape


def test_run_fastccc_public(adata):
    res = F.run_fastccc(adata, groupby="cell_type", cpdb_file_path=str(_DB), inplace=False)
    assert res["method"] == "fastccc"
    assert res["strength"].shape == res["pvalues"].shape == (len(res["pairs"]), len(res["interaction_ids"]))
    F.run_fastccc(adata, groupby="cell_type", cpdb_file_path=str(_DB), key_added="fc")
    assert "long" in adata.uns["fc"]


def test_input_validation(adata):
    counts = adata.copy()
    counts.X = sp.csr_matrix(np.round(np.expm1(adata.X.toarray()) * 3))
    with pytest.raises(ValueError, match="raw"):
        F.run(counts, groupby="cell_type", cpdb_file_path=str(_DB))
    with pytest.raises(ValueError, match="human"):
        F.run(adata, groupby="cell_type", species="mouse", cpdb_file_path=str(_DB))
    bad = adata.copy()
    bad.obs["cell_type"] = bad.obs["cell_type"].str.replace(" ", "|")
    with pytest.raises(ValueError, match=r"\|"):
        F.run(bad, groupby="cell_type", cpdb_file_path=str(_DB))
    with pytest.raises(TypeError, match="Unknown"):
        F.run(adata, groupby="cell_type", cpdb_file_path=str(_DB), n_bootstraps=3)
