"""ov.single.run_ccc: unified entry point, consensus and method comparison.

Uses the 400 real PBMC3k cells of ``data/cellchat_r_reference.npz``. CellChat
runs for real; a second method is registered from a transformed copy of the
CellChat long table, so consensus/comparison have known answers without
optional dependencies. LIANA and CellPhoneDB run only when installed
(CellPhoneDB also needs ``OV_CPDB_ZIP`` pointing at ``cellphonedb.zip``; the fixture keeps only
CellChatDB genes, so the 200-genes-per-cell filter is disabled there).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

anndata = pytest.importorskip("anndata")
pytest.importorskip("torch")

try:
    import omicverse as ov
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"omicverse import failed in test env: {exc}", allow_module_level=True)

from omicverse.single import _ccc as C

_REF = Path(__file__).parent / "data" / "cellchat_r_reference.npz"


@pytest.fixture(scope="module")
def adata():
    ref = dict(np.load(_REF, allow_pickle=False))
    X = sp.csr_matrix((ref["data"], ref["indices"], ref["indptr"]), shape=tuple(ref["shape"]))
    obs = pd.DataFrame({"cell_type": ref["labels"]}, index=ref["cells"])
    return anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=ref["genes"]))


@pytest.fixture(scope="module")
def cellchat_run(adata):
    a = adata.copy()
    comm = ov.single.run_ccc(a, "cell_type", method="cellchat", seed=1, n_perms=50,
                             progress=False)
    return a, comm


def _mirror_backend(adata, *, groupby, pvalue_threshold=0.05, **_):
    """A fake second method: CellChat calls with reversed orientation and reversed scores."""
    long = pd.DataFrame(adata.uns["ccc"]["long"]["cellchat"]).copy()
    out = long.rename(columns={"source": "target", "target": "source", "ligand": "receptor", "receptor": "ligand",
                               "ligand_genes": "receptor_genes", "receptor_genes": "ligand_genes"})
    out["score"] = 1.0 / (1.0 + long["score"].to_numpy())
    return {"raw": None, "long": out[C.LONG_COLUMNS], "params": {}}


def test_long_table_contract(cellchat_run):
    a, comm = cellchat_run
    long = a.uns["ccc"]["long"]["cellchat"]
    assert list(long.columns) == C.LONG_COLUMNS
    assert (long["score"] > 0).all() and long["pvalue"].between(0, 1).all()
    assert (long["significant"] == (long["pvalue"] < 0.05)).all()
    # Complex receptors are expanded to sorted subunit genes.
    row = long[long["receptor"] == "TGFbR1_R2"]
    if len(row):
        assert set(row["receptor_genes"]) == {"TGFBR1_TGFBR2"}
    assert "cellchat_res" in a.uns  # native result kept for cellchat_centrality etc.


def test_comm_matches_native_cellchat(cellchat_run):
    a, comm = cellchat_run
    native = ov.single.format_cellchat_results(a)
    assert comm.uns["comm_source"] == "cellchat" and comm.uns["comm_score_scale"] == "probability"
    # Groups with <= min_cells cells are dropped from the unified output (all-zero in CellChat).
    assert set(comm.obs_names) < set(native.obs_names)
    assert "Megakaryocytes" not in set(comm.obs["sender"])
    # Every non-zero native probability appears with the same value.
    nat = native.to_df(layer="means")
    lr_key = dict(zip(native.var["gene_a"] + "|" + native.var["gene_b"], native.var_names))
    new = comm.to_df(layer="means")
    for vk, lig, rec in zip(comm.var_names[:40], comm.var["gene_a"][:40], comm.var["gene_b"][:40]):
        col = lr_key[f"{lig}|{rec}"]
        np.testing.assert_allclose(new[vk].to_numpy(), nat.loc[new.index, col].to_numpy(), rtol=0, atol=1e-15)


def test_get_ccc_and_to_comm_adata(cellchat_run, tmp_path):
    a, comm = cellchat_run
    long = ov.single.get_ccc(a, method="cellchat", as_long=True, significant_only=True)
    assert long["significant"].all()
    path = tmp_path / "ccc.h5ad"
    a.write_h5ad(path)
    b = anndata.read_h5ad(path)
    back = ov.single.to_comm_adata(b)
    np.testing.assert_allclose(back.layers["means"], comm.layers["means"])


def test_consensus_and_compare(adata):
    a = adata.copy()
    ov.single.register_ccc_method("mirror", _mirror_backend, score_scale="expression")
    try:
        ov.single.run_ccc(a, "cell_type", method="cellchat", seed=1, n_perms=50, progress=False)
        comm = ov.single.run_ccc(a, "cell_type", method=["cellchat", "mirror"], seed=1, n_perms=50,
                                 progress=False, consensus="vote", min_methods=2)
        cons = ov.single.get_ccc(a, as_long=True)
        cc = pd.DataFrame(a.uns["ccc"]["long"]["cellchat"])
        # The mirrored copy describes the same co-expression events, so every interaction matches.
        assert (cons["n_methods_detected"] == 2).all()
        assert len(cons) == len(C._per_method_key_table(cc))
        assert cons["significant"].sum() == C._per_method_key_table(cc)["significant"].sum()
        assert {"cellchat_score", "mirror_score", "n_methods_significant"} <= set(comm.layers)
        res = ov.single.compare_ccc(a, shared_only=False)
        assert res["jaccard"].loc["cellchat", "mirror"] == pytest.approx(1.0)
        assert res["score_spearman"].loc["cellchat", "mirror"] == pytest.approx(-1.0)
        # Cell-pair counts follow each method's own orientation: the mirror is the transpose.
        np.testing.assert_array_equal(res["cellpair_counts"]["cellchat"].to_numpy(),
                                      res["cellpair_counts"]["mirror"].to_numpy().T)
        fig = ov.pl.ccc_compare(res, plot_type="upset")
        assert fig is not None
    finally:
        C._REGISTRY.pop("mirror", None)


def test_rra_rho_properties():
    ranks = np.array([[0.01, 0.02, 0.03], [0.9, 0.95, 1.0], [0.01, 1.0, 1.0]])
    rho = C._rra_rho(ranks)
    assert rho[0] < 1e-3 < rho[2] < rho[1] <= 1.0


def test_canonical_key_aligns_directions():
    rows = [("A", "B", "L1", "R1", 2.0), ("B", "A", "R1", "L1", 1.0)]
    long = pd.DataFrame(rows, columns=["source", "target", "ligand_genes", "receptor_genes", "score"])
    for col in ("ligand", "receptor", "interaction", "pathway", "annotation"):
        long[col] = ""
    long["pvalue"], long["significant"] = 0.01, True
    t = C._per_method_key_table(long)
    assert len(t) == 1 and t["score"].iat[0] == 2.0 and t["source"].iat[0] == "A"


def test_errors(adata):
    with pytest.raises(ValueError, match="Unknown CCC method"):
        ov.single.run_ccc(adata.copy(), "cell_type", method="nope")
    with pytest.raises(KeyError):
        ov.single.run_ccc(adata.copy(), "missing", method="cellchat")
    with pytest.raises(KeyError, match="needs coordinates"):
        ov.single.run_ccc(adata.copy(), "cell_type", method="commot")
    assert {"cellchat", "cellphonedb", "liana", "fastccc", "secact", "cytosignal", "commot"} <= set(
        ov.single.available_ccc_methods())


def test_liana_backend(adata):
    pytest.importorskip("liana")
    a = adata.copy()
    comm = ov.single.run_ccc(a, "cell_type", method="liana", n_perms=50, seed=0, progress=False)
    long = a.uns["ccc"]["long"]["liana"]
    assert len(long) > 0 and long["score"].between(0, 1).all()
    assert comm.uns["comm_score_scale"] == "rank"


@pytest.mark.skipif(not os.environ.get("OV_CPDB_ZIP"), reason="set OV_CPDB_ZIP to run CellPhoneDB")
def test_cellphonedb_backend_orientation(adata):
    pytest.importorskip("cellphonedb")
    a = adata.copy()
    ov.single.run_ccc(a, "cell_type", method="cellphonedb", n_perms=100, progress=False,
                      method_kwargs={"cpdb_file_path": os.environ["OV_CPDB_ZIP"], "threads": 2, "min_genes": 0})
    long = a.uns["ccc"]["long"]["cellphonedb"]
    raw = a.uns["cpdb_results"]["means"]
    rev = raw[raw["receptor_a"].astype(str).eq("True") & ~raw["receptor_b"].astype(str).eq("True")]
    for name in rev["interacting_pair"].head(5):
        sub = long[long["interaction"] == name]
        # Receptor-first records are flipped so the ligand side sends.
        assert not sub.empty
        assert (sub["ligand"] != rev.loc[rev["interacting_pair"] == name, "gene_a"].iat[0]).all()


def _ligand_only_backend(adata, *, groupby, pvalue_threshold=0.05, **_):
    """A SecAct-like method: CellChat calls collapsed to the ligand, receptor left empty."""
    long = pd.DataFrame(adata.uns["ccc"]["long"]["cellchat"]).copy()
    long["receptor"], long["receptor_genes"] = "", ""
    return {"raw": None, "long": long[C.LONG_COLUMNS], "params": {}}


def test_ligand_level_comparison(adata):
    a = adata.copy()
    ov.single.register_ccc_method("ligonly", _ligand_only_backend, score_scale="activity")
    try:
        ov.single.run_ccc(a, "cell_type", method="cellchat", seed=1, n_perms=50, progress=False)
        ov.single.run_ccc(a, "cell_type", method="ligonly", consensus=None, progress=False)
        res = ov.single.compare_ccc(a, shared_only=False)
        assert res["level"] == "ligand"
        assert res["jaccard"].loc["cellchat", "ligonly"] == pytest.approx(1.0)
        assert all(" -> " not in str(x) for x in res["top_lr"]["cellchat"].dropna())
        cons = ov.single.ccc_consensus_table({m: a.uns["ccc"]["long"][m] for m in ("cellchat", "ligonly")})
        assert (cons["receptor_genes"] == "").all() and (cons["n_methods_detected"] == 2).all()
    finally:
        C._REGISTRY.pop("ligonly", None)


def test_space_run_ccc_requires_coordinates(adata):
    with pytest.raises(KeyError, match="Spatial coordinates"):
        ov.space.run_ccc(adata.copy(), "cell_type", method="cytosignal")


def test_gene_case_is_ignored_when_matching():
    # Mouse CellChatDB writes Nrg3/Erbb4, CytoSignal and SecAct NRG3/ERBB4.
    rows = [("A", "B", "Nrg3", "Erbb4", 2.0), ("A", "B", "Cd74_Cxcr4", "Mif", 1.0)]
    a = pd.DataFrame(rows, columns=["source", "target", "ligand_genes", "receptor_genes", "score"])
    b = a.assign(ligand_genes=a["ligand_genes"].str.upper(), receptor_genes=a["receptor_genes"].str.upper())
    for frame in (a, b):
        for col in ("ligand", "receptor", "interaction", "pathway", "annotation"):
            frame[col] = ""
        frame["pvalue"], frame["significant"] = 0.01, True
    assert set(C._per_method_key_table(a).index) == set(C._per_method_key_table(b).index)
    assert C._norm_key("Cxcr4_Cd74") == "CD74_CXCR4"
