"""ov.single.run_cellchat: PyTorch CellChat versus R CellChat 2.2.0.

``data/cellchat_r_reference.npz`` holds 400 real PBMC3k cells (genes limited to
CellChatDB.human) and the outputs of ``data/cellchat_r_reference.R`` on exactly
that matrix. Deterministic steps must match R to floating-point precision;
permutation p-values use a different RNG than R, so only their agreement rate
is checked.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

anndata = pytest.importorskip("anndata")
torch = pytest.importorskip("torch")

try:
    import omicverse as ov
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"omicverse import failed in test env: {exc}", allow_module_level=True)

from omicverse.external.cellchat import _engine as eng
from omicverse.external.cellchat import _pipeline as rp
from omicverse.external.cellchat import load_cellchat_db

_REF = Path(__file__).parent / "data" / "cellchat_r_reference.npz"


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(_REF, allow_pickle=False))


@pytest.fixture(scope="module")
def adata(ref):
    X = sp.csr_matrix((ref["data"], ref["indices"], ref["indptr"]), shape=tuple(ref["shape"]))
    obs = pd.DataFrame({"cell_type": ref["labels"]}, index=ref["cells"])
    return anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=ref["genes"]))


@pytest.fixture(scope="module")
def result(adata):
    a = adata.copy()
    ov.single.run_cellchat(a, groupby="cell_type", n_perms=100, seed=1, progress=False)
    return a


def _r_tensor(ref, groups, lr_ids):
    G, L = list(ref["r_groups"]), list(ref["r_lr"])
    prob = np.zeros((len(G), len(G), len(L)))
    prob[tuple(ref["r_prob_idx"].astype(int))] = ref["r_prob_val"]
    pval = ref["r_pval_pct"].reshape(prob.shape) / 100.0
    gi = [G.index(g) for g in groups]
    li = [L.index(x) for x in lr_ids]
    return prob[np.ix_(gi, gi, li)], pval[np.ix_(gi, gi, li)]


# ---------------------------------------------------------------- R parity
def test_over_expressed_genes_match_r(result, ref):
    assert set(result.uns["cellchat_res"]["over_expressed_genes"]) == set(ref["r_features"])


def test_lrsig_matches_r_in_order(result, ref):
    assert list(result.uns["cellchat_res"]["LRsig"]["id"]) == list(ref["r_lrsig"])


def test_communication_probability_matches_r(result, ref):
    res = result.uns["cellchat_res"]
    groups = [str(g) for g in res["groups"]]
    r_prob, _ = _r_tensor(ref, groups, list(res["LRsig"]["id"]))
    np.testing.assert_array_equal(res["prob"] > 0, r_prob > 0)
    np.testing.assert_allclose(res["prob"], r_prob, rtol=1e-9, atol=1e-15)


def test_filter_communication_removes_small_group(result):
    res = result.uns["cellchat_res"]
    mk = [str(g) for g in res["groups"]].index("Megakaryocytes")
    assert not res["prob"][mk].any() and not res["prob"][:, mk].any()


def test_permutation_pvalues_agree_with_r(result, ref):
    res = result.uns["cellchat_res"]
    groups = [str(g) for g in res["groups"]]
    r_prob, r_pval = _r_tensor(ref, groups, list(res["LRsig"]["id"]))
    tested = r_prob > 0
    assert np.mean(np.abs(res["pval"][tested] - r_pval[tested]) <= 0.05) > 0.95
    sig_r = tested & (r_pval <= 0.05)
    sig_ov = (res["prob"] > 0) & (res["pval"] <= 0.05)
    assert (sig_r & sig_ov).sum() / (sig_r | sig_ov).sum() > 0.9


def test_pathways_match_r(result, ref):
    assert set(result.uns["cellchat_res"]["netP"]["pathways"]) == set(ref["r_pathways"])


def test_centrality_matches_r(ref):
    groups, pathways = list(ref["r_groups"]), list(ref["r_pathways"])
    ours = rp.net_analysis_compute_centrality(ref["r_netp"], pathways, groups)
    cols = list(ref["r_centr_cols"])
    for row, pw, g in zip(ref["r_centr"], ref["r_centr_pathway"], ref["r_centr_group"]):
        np.testing.assert_allclose(ours[pw].loc[g, cols].to_numpy(float), row, rtol=1e-8, atol=1e-12,
                                   err_msg=f"{pw}/{g}")


# ---------------------------------------------------------------- engine
def test_trimean_is_type7_quartile_mean():
    rng = np.random.default_rng(0)
    X = rng.gamma(1.0, 1.0, size=(57, 6)) * (rng.random((57, 6)) > 0.4)
    codes = rng.integers(0, 4, size=57)
    tm = eng._GroupTriMean(torch.as_tensor(X), 4, torch.device("cpu"), torch.float64, gene_chunk=4)(codes)
    for k in range(4):
        q = np.quantile(X[codes == k], [0.25, 0.5, 0.75], axis=0)  # numpy default = type 7
        np.testing.assert_allclose(tm[:, k].numpy(), (q[0] + 2 * q[1] + q[2]) / 4, rtol=1e-12)


def test_complex_uses_geometric_mean_and_cofactors():
    db = load_cellchat_db("human")
    lrsig = db.interaction.loc[["TGFB1_TGFBR1_TGFBR2"]]
    genes = rp.extract_genes(db)
    design = eng.build_design(lrsig, db, genes)
    rec_genes = [design.genes[i] for i in design.rec[0] if i >= 0]
    assert sorted(rec_genes) == ["TGFBR1", "TGFBR2"]
    assert (design.agonist[0] >= 0).any() and (design.antagonist[0] >= 0).any()


# ---------------------------------------------------------------- API
def test_format_cellchat_results_layout(result):
    res = result.uns["cellchat_res"]
    comm = ov.single.format_cellchat_results(result)
    n = len(res["groups"])
    assert comm.shape == (n * n, len(res["LRsig"]))
    s, r = str(res["groups"][1]), str(res["groups"][2])
    row = comm.obs_names.get_loc(f"{s}|{r}")
    np.testing.assert_array_equal(comm.layers["means"][row], res["prob"][1, 2])
    np.testing.assert_array_equal(comm.layers["pvalues"][row], res["pval"][1, 2])
    assert comm.var_names.is_unique
    assert ov.single.to_comm_adata(result).shape == comm.shape


def test_h5ad_roundtrip(result, tmp_path):
    path = tmp_path / "cellchat.h5ad"
    result.write_h5ad(path)
    back = anndata.read_h5ad(path)
    np.testing.assert_array_equal(back.uns["cellchat_res"]["prob"], result.uns["cellchat_res"]["prob"])
    assert ov.single.format_cellchat_results(back).shape == ov.single.format_cellchat_results(result).shape


def test_subset_communication_is_significant_only(result):
    df = ov.single.cellchat_subset_communication(result)
    assert len(df) and (df["prob"] > 0).all() and (df["pval"] <= 0.05).all()


def test_raw_counts_are_rejected(adata):
    counts = adata.copy()
    counts.X = sp.csr_matrix(np.round(np.expm1(counts.X.toarray()) * 3))
    with pytest.raises(ValueError, match="log-normalised"):
        ov.single.run_cellchat(counts, groupby="cell_type", n_perms=2, progress=False)
