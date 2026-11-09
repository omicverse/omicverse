"""SecAct backend: bit-level parity with R SecAct / SecActpy on real-data fixtures.

Fixtures (see data/secact_reference.R): a PBMC3k counts subset and a CosMx
window with signature subsets, plus R SecAct (8d9ce83) and SecActpy (13bb6d2)
outputs. No network access: signatures are passed as DataFrames.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from omicverse.external import secact as sec
from omicverse.single import _ccc_secact as backend

DATA = Path(__file__).parent / "data"


def _load(name):
    f = dict(np.load(DATA / name, allow_pickle=False))
    X = sp.csr_matrix((f["data"], f["indices"], f["indptr"]), shape=tuple(f["shape"])).astype(np.float64)
    sig = pd.DataFrame(f["sig"], index=f["sig_genes"], columns=f["sig_proteins"])
    return f, X, sig


@pytest.fixture(scope="module")
def pbmc():
    return _load("secact_pbmc_fixture.npz")


@pytest.fixture(scope="module")
def cosmx():
    return _load("secact_st_fixture.npz")


@pytest.fixture(autouse=True)
def _no_download(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("tests must not download SecAct resources")
    monkeypatch.setattr("omicverse.external.secact._signature.fetch_resource", fail)
    monkeypatch.setattr("omicverse.external.secact._core.fetch_resource", fail)


# --------------------------------------------------------------------- RNG / stats
def test_mt19937_table_matches_r():
    expected = np.array([[9, 2, 4, 0, 5, 7, 3, 6, 1, 8],
                         [6, 9, 2, 7, 1, 4, 3, 5, 0, 8],
                         [2, 6, 9, 4, 7, 8, 0, 1, 5, 3]])  # R .gsl_mt19937_perm_table(10L, 3L)
    np.testing.assert_array_equal(sec.permutation_table(10, 3, "mt19937"), expected)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="compares with glibc rand()")
def test_srand_table_matches_glibc():
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.srand(0)
    arr, rows = list(range(50)), []
    for _ in range(4):
        for i in range(49):
            j = i + libc.rand() // (2147483647 // (50 - i) + 1)
            arr[i], arr[j] = arr[j], arr[i]
        rows.append(list(arr))
    np.testing.assert_array_equal(sec.permutation_table(50, 4, "srand"), np.array(rows))


def test_r_sample_matches_r():
    a, b = sec.r_sample_replace([np.array([5, 7, 9]), np.arange(1, 100001)], 8, seed=123)
    np.testing.assert_array_equal(a, [9, 9, 9, 7, 9, 7, 7, 7])
    np.testing.assert_array_equal(b, [29925, 95246, 68293, 62555, 45404, 65161, 46435, 9642])


def test_bh_and_fisher_match_r():
    np.testing.assert_allclose(sec.p_adjust_bh([0.01, 0.04, 0.03, np.nan, 0.2, 0.04]),
                               [0.05, 0.05, 0.05, np.nan, 0.2, 0.05], rtol=1e-15)
    from omicverse.external.secact._core import fisher_combine
    np.testing.assert_allclose(fisher_combine(np.array([1e-3]), np.array([0.2])), [0.0019034386382832491],
                               rtol=1e-14)


# --------------------------------------------------------------------- scRNA-seq
def test_ccc_scrnaseq_matches_r(pbmc):
    f, X, sig = pbmc
    res = sec.ccc_scrnaseq(X, f["genes"], f["cell_type"], sig_matrix=sig, map_alias=False)
    z = res["activity"]["zscore"].loc[f["r_proteins"], f["r_celltypes"]]
    p = res["activity"]["pvalue"].loc[f["r_proteins"], f["r_celltypes"]]
    np.testing.assert_allclose(z.to_numpy(), f["r_zscore"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(p.to_numpy(), f["r_pvalue"], rtol=0, atol=1e-15)
    b = res["expression"]["B cells"].loc[f["r_exp_b_genes"]]
    np.testing.assert_allclose(b["exp_pv"], f["r_exp_b_pv"], rtol=1e-10)
    np.testing.assert_allclose(b["exp_logFC"], f["r_exp_b_logfc"], rtol=1e-10, atol=1e-14)
    ccc = res["ccc"]
    keys = ccc["sender"] + "|" + ccc["secretedProtein"] + "|" + ccc["receiver"]
    assert set(keys) == set(f["r_ccc_key"]) and len(keys) == len(f["r_ccc_key"]) > 0
    got = ccc.set_index(keys).loc[f["r_ccc_key"]]
    np.testing.assert_allclose(got["overall_strength"], f["r_ccc_strength"], rtol=1e-10)
    np.testing.assert_allclose(got["overall_pv.adj"], f["r_ccc_padj"], rtol=1e-10)


def test_ccc_scrnaseq_matches_secactpy(pbmc):
    f, X, sig = pbmc
    res = sec.ccc_scrnaseq(X, f["genes"], f["cell_type"], sig_matrix=sig, compat="secactpy")
    assert res["rng_method"] == "srand"
    z = res["activity"]["zscore"].loc[f["py_proteins"], f["py_celltypes"]]
    np.testing.assert_allclose(z.to_numpy(), f["py_zscore"], rtol=1e-10, atol=1e-12)
    ccc = res["ccc"]
    keys = ccc["sender"] + "|" + ccc["secretedProtein"] + "|" + ccc["receiver"]
    assert set(keys) == set(f["py_ccc_key"])
    got = ccc.set_index(keys).loc[f["py_ccc_key"]]
    np.testing.assert_allclose(got["overall_strength"], f["py_ccc_strength"], rtol=1e-10)
    np.testing.assert_allclose(got["overall_pv.adj"], f["py_ccc_padj"], rtol=1e-10)


# --------------------------------------------------------------------- spatial
def test_spatial_matches_r(cosmx):
    f, X, sig = cosmx
    act = sec.spot_activity(X, f["genes"], sig_matrix=sig, nrand=100, map_alias=False)
    z = act["zscore"].loc[f["r_proteins"]]
    np.testing.assert_allclose(z.to_numpy(), f["r_zscore"], rtol=1e-10, atol=1e-12)
    z.columns = f["cells"]
    res = sec.ccc_spatial(X, f["genes"], f["xy"], f["cell_type"], z, padj_cutoff=1.01, map_alias=False)
    corr = res["correlation"].loc[f["r_proteins"]]
    np.testing.assert_allclose(corr["r"], f["r_corr_r"], rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(corr["padj"], f["r_corr_padj"], rtol=1e-10, equal_nan=True)
    ccc = res["ccc"]
    keys = ccc["sender"] + "|" + ccc["secretedProtein"] + "|" + ccc["receiver"]
    assert set(keys) == set(f["r_ccc_key"]) and len(keys) > 0
    got = ccc.set_index(keys).loc[f["r_ccc_key"]]
    np.testing.assert_array_equal(got["neighboringCellPairs"], f["r_ccc_pairs"])
    np.testing.assert_allclose(got["ratio"], f["r_ccc_ratio"], rtol=1e-13)
    np.testing.assert_allclose(got["pv"], f["r_ccc_pv"], rtol=1e-13)
    np.testing.assert_allclose(got["pv.adj"], f["r_ccc_padj"], rtol=1e-13)


# --------------------------------------------------------------------- adapter
def _adata(f, X):
    return ad.AnnData(X.astype(np.float32), obs=pd.DataFrame({"ct": f["cell_type"]}, index=f["cells"]),
                      var=pd.DataFrame(index=f["genes"]))


def test_run_adapter_contract_and_h5ad(pbmc, tmp_path):
    f, X, sig = pbmc
    adata = _adata(f, X)
    out = backend.run(adata, groupby="ct", min_cells=30, n_perms=1000, pvalue_threshold=0.01, progress=False,
                      sig_matrix=sig, map_alias=False)
    long = out["long"]
    assert list(long.columns) == backend.LONG_COLUMNS
    assert backend.SCORE_SCALE == "activity" and backend.NEEDS_SPATIAL is False
    assert (long["receptor"] == "").all() and (long["receptor_genes"] == "").all()
    assert (long["significant"] == (long["pvalue"] < 0.01)).all()
    assert long["significant"].sum() == len(f["r_ccc_key"])  # same calls as R
    assert (long["score"] > 0).all() and long["pvalue"].between(0, 1).all()
    assert long["interaction"].str.endswith("→activity").all()
    adata.uns["secact"] = out["raw"]
    adata.write_h5ad(tmp_path / "secact.h5ad")
    back = ad.read_h5ad(tmp_path / "secact.h5ad")
    assert set(back.uns["secact"]) >= {"ccc", "candidates", "expression", "activity", "parameters"}


def test_run_secact_inplace(pbmc):
    f, X, sig = pbmc
    adata = _adata(f, X)
    ov_res = backend.run_secact(adata, groupby="ct", progress=False, sig_matrix=sig, map_alias=False)
    assert ov_res is adata
    assert len(adata.uns["secact_res"]["ccc"]) == len(f["r_ccc_key"])


def test_spatial_adapter(cosmx):
    f, X, sig = cosmx
    adata = _adata(f, X)
    adata.obsm["spatial"] = f["xy"]
    out = backend.run(adata, groupby="ct", spatial_key="spatial", n_perms=100, pvalue_threshold=1.01,
                      progress=False, sig_matrix=sig, map_alias=False)
    assert list(out["long"].columns) == backend.LONG_COLUMNS
    assert len(out["long"]) == len(f["r_ccc_key"])
    act = out["obsm"]["secact_activity"]
    assert act.shape == (adata.n_obs, sig.shape[1]) and list(act.index) == list(adata.obs_names)


def test_requires_counts(pbmc):
    f, X, sig = pbmc
    adata = _adata(f, X)
    adata.X = sp.csr_matrix(np.log1p(X.toarray()))
    with pytest.raises(ValueError, match="raw counts"):
        backend.run(adata, groupby="ct", sig_matrix=sig, progress=False)
