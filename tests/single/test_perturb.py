"""Unit tests for :mod:`omicverse.single._perturb`.

These tests stub out the optional backend libraries (``sctenifoldknk`` /
``celloracle`` / ``dynamo``) and verify:

* the public API surface (``ov.single.perturb`` + ``PerturbResult``)
* mode / backend validation
* the lazy-import error path when no backend is installed
* end-to-end dispatch through all three backends with stubs
* the GRN helper utilities (``_diff_grn`` / ``_apply_perturbation_to_graph``)

The tests intentionally **do not** import the real ``sctenifoldknk`` /
``celloracle`` / ``dynamo`` packages — they monkey-patch fake modules into
``sys.modules`` first, so CI doesn't need any of them installed. Optional
native dynamo tests skip when that dependency is unavailable; the
hematopoiesis integration also requires a local dataset. Set
``OMICVERSE_HEMATOPOIESIS_PATH`` to override ``data/hematopoiesis.h5ad``.
Biological ranking assertions apply to the reference fixture checksum;
structural mapping assertions also run on other dataset versions. The full
raw-data pipeline test uses ``data/hematopoiesis_raw.h5ad`` or the
``OMICVERSE_RAW_HEMATOPOIESIS_PATH`` override, and never downloads fixtures.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_adata():
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(40, 8)).astype(float)
    return AnnData(
        X=X,
        var=pd.DataFrame(index=[f"G{i}" for i in range(X.shape[1])]),
        obs=pd.DataFrame(index=[f"C{i}" for i in range(X.shape[0])]),
    )


@pytest.fixture
def fake_grn(tiny_adata):
    """Tiny 8×8 GRN with 'G0' regulating G1/G2/G3 strongly."""
    import networkx as nx
    G = nx.DiGraph()
    genes = list(tiny_adata.var_names)
    for g in genes:
        G.add_node(g)
    G.add_edge("G0", "G1", weight=1.0)
    G.add_edge("G0", "G2", weight=0.7)
    G.add_edge("G0", "G3", weight=0.4)
    G.add_edge("G4", "G5", weight=0.9)
    G.add_edge("G6", "G7", weight=0.8)
    return G


# ---------------------------------------------------------------------------
# Stubs for the optional backends
# ---------------------------------------------------------------------------


def _install_fake_sctenifoldknk(monkeypatch, base_graph):
    """Install a fake ``scTenifold`` module returning ``base_graph``.

    Mirrors the real ``scTenifold.scTenifoldKnk`` API:
        knk = scTenifoldKnk(data=df, ko_genes=[...])
        knk.build()
        knk.WT_data['network'] / knk.KO_data['network']
    """
    fake = types.ModuleType("scTenifold")

    class _FakeKnk:
        def __init__(self, data=None, ko_genes=None, **kwargs):
            import numpy as np
            self.data = data
            self.ko_genes = list(ko_genes or [])
            self.shared_gene_names = list(base_graph.nodes)
            self.tensor_dict: dict = {}
            self.d_regulation = None

        def build(self):
            import numpy as np
            import networkx as nx
            n = len(self.shared_gene_names)
            idx = {g: i for i, g in enumerate(self.shared_gene_names)}
            wt = np.zeros((n, n), dtype=float)
            for u, v, d in base_graph.edges(data=True):
                wt[idx[u], idx[v]] = float(d.get("weight", 1.0))
            ko = wt.copy()
            for g in self.ko_genes:
                if g in idx:
                    ko[idx[g], :] = 0.0
            self.tensor_dict = {
                key: pd.DataFrame(value, index=self.shared_gene_names,
                                  columns=self.shared_gene_names)
                for key, value in {"WT": wt, "KO": ko}.items()
            }

    fake.scTenifoldKnk = _FakeKnk
    monkeypatch.setitem(sys.modules, "scTenifold", fake)
    return fake


def _install_fake_celloracle(monkeypatch, base_graph, *, adata, hit_simulate=None):
    """Install a fake ``celloracle`` module + Oracle whose
    ``simulate_shift`` records the perturbation dict.
    """
    fake = types.ModuleType("celloracle")

    class _FakeOracle:
        def __init__(self):
            self.coef_matrix_baseline = base_graph
            self.coef_matrix = base_graph.copy()
            self.adata = adata
            self.transition_prob = np.eye(adata.n_obs)

        def import_anndata_as_normalized_count(self, *_, **__):
            return None

        def import_TF_data(self, *_, **__):
            return None

        def fit_GRN_for_simulation(self, *_, **__):
            return None

        def simulate_shift(self, *, perturb_condition, n_propagation=3):
            if hit_simulate is not None:
                hit_simulate["calls"].append(
                    {"cond": dict(perturb_condition), "n_prop": n_propagation}
                )
            # crude effect: reduce coef_matrix in-edges to KO'd nodes
            import networkx as nx
            self.coef_matrix = base_graph.copy()
            for g, v in perturb_condition.items():
                if g in self.coef_matrix:
                    for u, w, d in list(self.coef_matrix.in_edges(g, data=True)):
                        d["weight"] = float(d.get("weight", 1.0)) * (v / 1.0)
                    for u, w, d in list(self.coef_matrix.out_edges(g, data=True)):
                        d["weight"] = float(d.get("weight", 1.0)) * (v / 1.0)
            # crude delta layers
            n_cells, n_genes = adata.n_obs, adata.n_vars
            base = np.asarray(adata.X)
            sim = base.copy()
            for g, v in perturb_condition.items():
                if g in adata.var_names:
                    j = adata.var_names.get_loc(g)
                    sim[:, j] = v
            adata.layers["imputed_count"] = base
            adata.layers["simulated_count"] = sim
            return None

    fake.Oracle = _FakeOracle

    # data submodule with stub GRN loaders so we don't trip on missing files
    fake_data = types.ModuleType("celloracle.data")
    fake_data.load_human_promoter_base_GRN = lambda: base_graph
    fake_data.load_mouse_promoter_base_GRN = lambda: base_graph
    fake.data = fake_data

    monkeypatch.setitem(sys.modules, "celloracle", fake)
    monkeypatch.setitem(sys.modules, "celloracle.data", fake_data)
    return fake


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_module_imports():
    from omicverse.single import perturb, PerturbResult

    assert callable(perturb)
    assert PerturbResult.__dataclass_fields__  # is a dataclass


def test_mode_validation(tiny_adata):
    from omicverse.single import perturb

    with pytest.raises(ValueError, match="`mode` must be one of"):
        perturb(tiny_adata, target="G0", mode="invalid")


def test_backend_validation(tiny_adata):
    from omicverse.single import perturb

    with pytest.raises(ValueError, match="`backend` must be one of"):
        perturb(tiny_adata, target="G0", backend="not_a_real_backend")


def test_missing_gene(tiny_adata):
    from omicverse.single import perturb

    with pytest.raises(KeyError, match="not in adata.var_names"):
        perturb(tiny_adata, target="NOT_REAL")


def test_empty_target(tiny_adata):
    from omicverse.single import perturb

    with pytest.raises(ValueError, match="must name at least one gene"):
        perturb(tiny_adata, target=[])


# ---------------------------------------------------------------------------
# scTenifoldKnk backend (with stub)
# ---------------------------------------------------------------------------


def test_sctenifoldknk_backend_ko(monkeypatch, tiny_adata, fake_grn):
    from omicverse.single import perturb

    _install_fake_sctenifoldknk(monkeypatch, fake_grn)

    result = perturb(tiny_adata, target="G0", mode="ko",
                     backend="sctenifoldknk")
    assert result.backend == "sctenifoldknk"
    assert result.target == "G0"
    assert result.mode == "ko"

    # Δ-grn should record that G0's outgoing edges (G1/G2/G3) went to 0
    assert not result.delta_grn.empty
    out_edges = result.delta_grn[result.delta_grn["source"] == "G0"]
    assert (out_edges["weight_pert"] == 0.0).all()
    assert (out_edges["delta"] < 0).all()

    # baseline edges should be preserved
    assert result.grn_base is not None
    assert "G1" in result.grn_base.successors("G0")


@pytest.mark.parametrize("mode, factor", [("ko", 0.0), ("kd", 1 / 3), ("oe", 3.0)])
@pytest.mark.parametrize("grn_output", [True, False])
def test_sctenifoldknk_delta_matches_perturbation(
    monkeypatch, tiny_adata, fake_grn, mode, factor, grn_output,
):
    from omicverse.single import perturb

    tiny_adata = tiny_adata[:, ::-1].copy()
    tiny_adata.layers["counts"] = tiny_adata.X * 2
    fake_grn.add_edge("G4", "G6", weight=0.001)
    _install_fake_sctenifoldknk(monkeypatch, fake_grn)

    result = perturb(tiny_adata, target="G0", mode=mode, layer="counts",
                     fold_change=3.0, backend="sctenifoldknk",
                     grn_output=grn_output)
    assert result.mode == mode
    out_edges = result.delta_grn[result.delta_grn["source"] == "G0"]
    np.testing.assert_allclose(
        out_edges["weight_pert"].to_numpy(),
        out_edges["weight_base"].to_numpy() * factor,
    )
    expected = np.zeros((tiny_adata.n_obs, len(result.gene_names)))
    target_counts = tiny_adata[:, "G0"].layers["counts"][:, 0]
    for gene, weight in [("G1", 1.0), ("G2", 0.7), ("G3", 0.4)]:
        expected[:, result.gene_names.index(gene)] = target_counts * weight * (factor - 1)
    np.testing.assert_allclose(result.delta_X, expected)
    assert result.cell_names == list(tiny_adata.obs_names)
    assert (result.grn is not None) == grn_output


@pytest.mark.parametrize("mode, factor", [("kd", 1 / 3), ("oe", 3.0)])
@pytest.mark.parametrize("unrelated_weight", [1.0, 3.0])
def test_sctenifoldknk_numeric_effect_ignores_display_threshold(
    monkeypatch, tiny_adata, fake_grn, mode, factor, unrelated_weight,
):
    from omicverse.single import perturb

    fake_grn.remove_edges_from(list(fake_grn.edges))
    fake_grn.add_edge("G0", "G1", weight=0.01)
    fake_grn.add_edge("G4", "G5", weight=unrelated_weight)
    _install_fake_sctenifoldknk(monkeypatch, fake_grn)
    result = perturb(tiny_adata, target="G0", mode=mode, fold_change=3.0,
                     backend="sctenifoldknk")
    expected = np.zeros_like(tiny_adata.X)
    expected[:, 1] = tiny_adata.X[:, 0] * 0.01 * (factor - 1)
    np.testing.assert_allclose(result.delta_X, expected)


@pytest.mark.parametrize("mode, factor, fold_change", [
    ("ko", 0.0, 3.0), ("kd", 1 / 3, 3.0), ("oe", 3.0, 3.0),
    ("kd", 1.0, 1.0), ("oe", 1.0, 1.0),
])
def test_sctenifoldknk_scales_only_target_rows_once(
    monkeypatch, tiny_adata, fake_grn, mode, factor, fold_change,
):
    from omicverse.single import perturb
    import networkx as nx

    fake_grn.add_edge("G4", "G0", weight=-0.5)
    fake_grn.add_edge("G1", "G2", weight=-0.2)
    _install_fake_sctenifoldknk(monkeypatch, fake_grn)
    result = perturb(tiny_adata, target=["G0", "G1", "G0"], mode=mode,
                     fold_change=fold_change, backend="sctenifoldknk")
    wt = nx.to_numpy_array(fake_grn, nodelist=list(tiny_adata.var_names))
    delta_input = np.zeros_like(tiny_adata.X)
    delta_input[:, [0, 1]] = tiny_adata.X[:, [0, 1]] * (factor - 1)
    np.testing.assert_allclose(result.delta_X, delta_input @ wt)
    assert result.grn["G4"]["G0"]["weight"] == -0.5
    pert = nx.to_numpy_array(result.grn, nodelist=list(tiny_adata.var_names))
    assert pert[0, 1] == pytest.approx(factor)
    assert pert[1, 2] == pytest.approx(-0.2 * factor)


def test_sctenifoldknk_backend_missing_raises(monkeypatch, tiny_adata):
    """If neither the system package nor the vendored copy is importable,
    the backend should raise the optional-dependency error.
    """
    from omicverse.single import perturb

    monkeypatch.delitem(sys.modules, "scTenifold", raising=False)
    # Also clear the vendored copy so we test the fallback failure path.
    for mod in list(sys.modules):
        if mod.startswith("omicverse.external.scTenifold"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    original_import = __import__

    def blocked(name, *args, **kwargs):
        if (name == "scTenifold" or name.startswith("scTenifold.")
                or name == "omicverse.external.scTenifold"
                or name.startswith("omicverse.external.scTenifold.")):
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)

    with pytest.raises(ImportError, match="(?i)scTenifold"):
        perturb(tiny_adata, target="G0", backend="sctenifoldknk")


# ---------------------------------------------------------------------------
# CellOracle backend (with stub)
# ---------------------------------------------------------------------------


def test_cell_oracle_backend_ko(monkeypatch, tiny_adata, fake_grn):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_celloracle(monkeypatch, fake_grn,
                             adata=tiny_adata, hit_simulate=hits)

    result = perturb(
        tiny_adata, target="G0", mode="ko",
        backend="cell_oracle", grn_base=fake_grn,
    )
    assert result.backend == "cell_oracle"
    assert hits["calls"], "simulate_shift should have been called once"
    assert hits["calls"][0]["cond"] == {"G0": 0.0}

    assert result.grn is not None
    assert result.adata_perturbed is not None
    # trajectory_shift is set by the stub
    assert result.trajectory_shift is not None


def test_cell_oracle_backend_oe(monkeypatch, tiny_adata, fake_grn):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_celloracle(monkeypatch, fake_grn,
                             adata=tiny_adata, hit_simulate=hits)

    result = perturb(
        tiny_adata, target="G4", mode="oe", fold_change=4.0,
        backend="cell_oracle", grn_base=fake_grn,
    )
    cond = hits["calls"][0]["cond"]
    # mean(G4) of the Poisson(2) matrix, times 4
    g4_base = float(np.asarray(tiny_adata[:, "G4"].X).mean())
    assert cond["G4"] == pytest.approx(g4_base * 4.0)


def test_cell_oracle_requires_base_grn(monkeypatch, tiny_adata, fake_grn):
    """CellOracle backend without `grn_base` and no stash should raise."""
    from omicverse.single import perturb

    _install_fake_celloracle(monkeypatch, fake_grn, adata=tiny_adata)

    with pytest.raises(ValueError, match="(?i)needs a base GRN"):
        perturb(tiny_adata, target="G0", backend="cell_oracle")


@pytest.mark.parametrize("key", ["base_grn", "celloracle_base_grn"])
def test_cell_oracle_dataframe_grn_in_uns(monkeypatch, tiny_adata, fake_grn, key):
    from omicverse.single import perturb

    backend = _install_fake_celloracle(monkeypatch, fake_grn, adata=tiny_adata)
    grn = pd.DataFrame({"gene_short_name": ["G1"], "peak_id": ["peak1"], "G0": [1]})
    tiny_adata.uns[key] = grn
    if key == "base_grn":
        tiny_adata.uns["celloracle_base_grn"] = pd.DataFrame()
    received = []
    monkeypatch.setattr(backend.Oracle, "import_TF_data",
                        lambda self, TF_info_matrix: received.append(TF_info_matrix))

    result = perturb(tiny_adata, target="G0", backend="auto")
    assert result.backend == "cell_oracle"
    assert len(received) == 1 and received[0] is grn
    assert result.grn_base.has_edge("G0", "G1")


def test_cell_oracle_imputation_failure_stops_simulation(monkeypatch, tiny_adata, fake_grn):
    from omicverse.single import perturb

    hits = {"calls": []}
    backend = _install_fake_celloracle(monkeypatch, fake_grn,
                                       adata=tiny_adata, hit_simulate=hits)
    monkeypatch.setattr(backend.Oracle, "perform_PCA", lambda self: None, raising=False)

    def fail_imputation(self, **kwargs):
        raise ValueError("invalid imputation input")

    monkeypatch.setattr(backend.Oracle, "knn_imputation", fail_imputation, raising=False)
    with pytest.raises(ValueError, match="invalid imputation input"):
        perturb(tiny_adata, target="G0", backend="cell_oracle", grn_base=fake_grn)
    assert not hits["calls"]
    assert "imputed_count" not in tiny_adata.layers


def test_auto_backend_picks_sctenifoldknk_without_base_grn(monkeypatch, tiny_adata, fake_grn):
    """`backend='auto'` falls back to sctenifoldknk when no base GRN
    is provided."""
    from omicverse.single import perturb

    _install_fake_sctenifoldknk(monkeypatch, fake_grn)

    result = perturb(tiny_adata, target="G0", backend="auto")
    assert result.backend == "sctenifoldknk"


def test_auto_backend_picks_cell_oracle_with_base_grn(monkeypatch, tiny_adata, fake_grn):
    """`backend='auto'` picks cell_oracle when a base GRN is present."""
    from omicverse.single import perturb

    _install_fake_celloracle(monkeypatch, fake_grn, adata=tiny_adata)
    tiny_adata.uns["base_grn"] = fake_grn

    result = perturb(tiny_adata, target="G0", backend="auto")
    assert result.backend == "cell_oracle"


# ---------------------------------------------------------------------------
# PerturbResult helpers
# ---------------------------------------------------------------------------


def test_perturb_result_summary(monkeypatch, tiny_adata, fake_grn):
    from omicverse.single import perturb

    _install_fake_sctenifoldknk(monkeypatch, fake_grn)
    result = perturb(tiny_adata, target="G0", mode="ko",
                     backend="sctenifoldknk")
    top = result.summary(top_n=3)
    assert isinstance(top, pd.DataFrame)
    assert len(top) <= 3
    # the gene whose in-edges changed the most should be near the top
    assert "G1" in top["gene"].values or "G2" in top["gene"].values







def _install_fake_dynamo(monkeypatch, *, hits, emb_basis="umap"):
    """Install a fake ``dynamo`` module whose ``pd.perturbation`` mimics
    the key layout dynamo-release 1.5.3 writes:

      * ``adata.layers['<method>_perturbation']``        — per-cell Δ
      * ``adata.obsp['perturbation_transition_matrix']`` — cell transitions
      * ``adata.obsm['velocity_<emb>_perturbation']``    — embedding arrows
    """
    fake = types.ModuleType("dynamo")
    fake_pd = types.ModuleType("dynamo.pd")

    def _perturbation(adata, genes, expression, perturb_mode="raw",
                      emb_basis="umap", pertubation_method="j_delta_x",
                      add_delta_Y_key=None, add_transition_key=None,
                      add_velocity_key=None, add_embedding_key=None,
                      projection_method="fp", **kwargs):
        if f"X_{emb_basis}" not in adata.obsm:
            raise KeyError(f"X_{emb_basis}")
        if "distances" not in adata.obsp and "indices" not in adata.uns.get("neighbors", {}):
            raise RuntimeError("missing neighbors")
        pca_genes = adata.var_names[adata.var.use_for_pca]
        ordered = list(pca_genes.intersection(genes))
        reconstructed = adata.obsm["X_pca"] @ adata.uns["PCs"].T + adata.uns["pca_mean"]
        hits["levels"] = {}
        for i, g in enumerate(ordered):
            x = reconstructed[:, pca_genes.get_loc(g)]
            hits["levels"][g] = (
                expression[i] * x.std() + x.mean()
                if perturb_mode == "z_score" else expression[i]
            )
        hits["calls"].append(dict(genes=list(genes),
                                  expression=list(expression),
                                  perturb_mode=perturb_mode,
                                  method=pertubation_method))
        rng = np.random.default_rng(0)

        delta = rng.normal(0, 0.01, size=adata.shape)
        delta[:, 1] = -0.5
        pkey = add_delta_Y_key if add_delta_Y_key is not None else f"{pertubation_method}_perturbation"
        adata.layers[pkey] = delta
        adata.obsm[pkey] = delta[:, :3]

        n = adata.n_obs
        T = np.zeros((n, n))
        for i in range(n):
            js = [(i + k + 1) % n for k in range(6)]
            T[i, js] = 1.0
            T[i] /= T[i].sum()
        tkey = add_transition_key if add_transition_key is not None else "perturbation_transition_matrix"
        vkey = add_velocity_key if add_velocity_key is not None else f"velocity_{emb_basis}_perturbation"
        ekey = add_embedding_key if add_velocity_key is not None else f"X_{emb_basis}_perturbation"
        if projection_method != "transform":
            adata.obsp[tkey] = T
        adata.obsm[vkey] = rng.normal(size=(n, 2))
        adata.obsm[ekey] = adata.obsm[f"X_{emb_basis}"].copy()

    def _rank_perturbation_genes(adata, pkey="j_delta_x_perturbation",
                                 prefix_store="rank", **kwargs):
        hits["ranked"] = dict(pkey=pkey, kwargs=kwargs)
        adata.uns[f"{prefix_store}_{pkey}"] = {"groups": {}}

    def _ko(adata, KO_genes, emb_basis="umap", **kwargs):
        hits["ko"] = list(KO_genes)
        adata.obsm["velocity_pca_KO"] = np.zeros((adata.n_obs, 3))
        adata.obsm[f"velocity_{emb_basis}_KO"] = np.zeros((adata.n_obs, 2))

    fake_pd.perturbation = _perturbation
    fake_pd.rank_perturbation_genes = _rank_perturbation_genes
    fake_pd.KO = _ko
    fake.pd = fake_pd
    monkeypatch.setitem(sys.modules, "dynamo", fake)
    monkeypatch.setitem(sys.modules, "dynamo.pd", fake_pd)
    return fake


def _dynamo_ready_adata(tiny_adata):
    """Mark a tiny AnnData as carrying dynamo's prerequisites."""
    tiny_adata.obsm["X_pca"] = np.random.default_rng(0).normal(
        size=(tiny_adata.n_obs, 3))
    tiny_adata.uns["PCs"] = np.random.default_rng(0).normal(
        size=(tiny_adata.n_vars, 3))
    tiny_adata.uns["pca_mean"] = np.zeros(tiny_adata.n_vars)
    tiny_adata.var["use_for_pca"] = True
    tiny_adata.uns["VecFld_pca"] = {"model": None}
    tiny_adata.uns["dynamics"] = {}
    tiny_adata.obsm["X_umap"] = np.random.default_rng(1).normal(size=(tiny_adata.n_obs, 2))
    tiny_adata.uns["neighbors"] = {
        "indices": np.tile(np.arange(tiny_adata.n_obs), (tiny_adata.n_obs, 1))}
    return tiny_adata


def test_dynamo_backend_ko(monkeypatch, tiny_adata):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)

    result = perturb(tiny_adata, target="G0", mode="ko", backend="dynamo")
    assert result.backend == "dynamo"

    assert hits["calls"][0]["expression"] == [0.0]
    assert hits["calls"][0]["genes"] == ["G0"]

    assert result.delta_X is not None
    assert result.delta_X.shape == tiny_adata.shape
    np.testing.assert_allclose(result.delta_X[:, 1], -0.5)

    assert result.trajectory_shift is not None
    np.testing.assert_allclose(result.trajectory_shift.sum(axis=1), 1.0)

    assert result.delta_expr is not None and not result.delta_expr.empty
    top = result.delta_expr.reindex(
        result.delta_expr["delta"].abs().sort_values(ascending=False).index
    )
    assert top.iloc[0]["gene"] == "G1"
    assert result.meta["perturbation_key"] == "j_delta_x_perturbation"


def test_dynamo_backend_oe_expression(monkeypatch, tiny_adata):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)

    perturb(tiny_adata, target="G4", mode="oe", fold_change=5.0,
            backend="dynamo")
    g4_base = float(np.asarray(tiny_adata[:, "G4"].X).mean())
    assert hits["calls"][0]["expression"] == [pytest.approx(g4_base * 5.0)]


def test_dynamo_requires_processed_adata(monkeypatch, tiny_adata):
    """Raw AnnData without auto_pipeline should raise an instructive error."""
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)

    with pytest.raises(RuntimeError, match="(?i)auto_pipeline"):
        perturb(tiny_adata, target="G0", backend="dynamo")
    assert not hits["calls"]


def test_dynamo_backend_missing_raises(monkeypatch, tiny_adata):
    from omicverse.single import perturb

    monkeypatch.delitem(sys.modules, "dynamo", raising=False)
    original_import = __import__

    def blocked(name, *args, **kwargs):
        if name == "dynamo" or name.startswith("dynamo."):
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)

    with pytest.raises(ImportError, match="(?i)dynamo"):
        perturb(tiny_adata, target="G0", backend="dynamo")


def test_dynamo_vf_ko_and_ranking(monkeypatch, tiny_adata):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)
    tiny_adata.obs["cluster"] = ["a", "b"] * (tiny_adata.n_obs // 2)

    result = perturb(
        tiny_adata, target="G0", mode="ko", backend="dynamo",
        backend_kwargs={"vf_ko": True, "group": "cluster"},
    )
    assert hits["ko"] == ["G0"]
    assert result.meta["vf_ko_keys"] == {
        "pca": "velocity_pca_KO", "emb_basis": "velocity_umap_KO"}
    assert hits["ranked"]["kwargs"] == {"groups": "cluster"}
    assert result.meta["rank_key"] == "rank_j_delta_x_perturbation"
    assert "velocity_umap_KO" in tiny_adata.obsm


def test_dynamo_vf_ko_warns_for_oe(monkeypatch, tiny_adata):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)

    with pytest.warns(RuntimeWarning, match="(?i)vf_ko"):
        perturb(tiny_adata, target="G0", mode="oe", backend="dynamo",
                backend_kwargs={"vf_ko": True})
    assert "ko" not in hits


def test_auto_backend_picks_dynamo_with_vector_field(monkeypatch, tiny_adata):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)

    result = perturb(tiny_adata, target="G0", backend="auto")
    assert result.backend == "dynamo"
    assert hits["calls"]






_HEMATOPOIESIS = Path(__file__).resolve().parents[2] / "data" / "hematopoiesis.h5ad"
_HEMATOPOIESIS_SHA256 = "635898d1e5ced4faac6546f0930548ceca56d0662b2527b867ab8f8a3ce62c11"


def _hematopoiesis_path():
    return Path(os.environ.get("OMICVERSE_HEMATOPOIESIS_PATH", _HEMATOPOIESIS)).expanduser()


def _is_reference_hematopoiesis(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == _HEMATOPOIESIS_SHA256


def _real_dynamo_and_data():
    path = _hematopoiesis_path()
    if not path.is_file():
        return None
    try:
        import dynamo
    except ImportError:
        return None
    return path


def test_dynamo_real_hematopoiesis_smoke():
    """End-to-end GATA1 KO on dynamo's hematopoiesis tutorial object.

    Verifies the mapping dynamo → PerturbResult on a real learned vector
    field: per-cell Δ (delta_X), dynamo's transition matrix, and the
    erythroid biology (GATA1 KO hits HBB/HBD hardest).
    """
    path = _real_dynamo_and_data()
    if path is None:
        pytest.skip("dynamo-release or the hematopoiesis dataset is not available")
    import anndata as ad

    import omicverse as ov

    adata = ad.read_h5ad(path)
    result = ov.single.perturb(adata, target="GATA1", mode="ko",
                               backend="auto")
    assert result.backend == "dynamo"
    assert result.delta_X.shape == adata.shape
    assert result.trajectory_shift.shape == (adata.n_obs, adata.n_obs)
    np.testing.assert_allclose(
        result.trajectory_shift.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8)
    top = result.delta_expr.reindex(
        result.delta_expr["delta"].abs().sort_values(ascending=False).index
    )
    if _is_reference_hematopoiesis(path):
        assert {"HBB", "HBD"} & set(top.head(10)["gene"]), (
            "GATA1 KO should hit erythroid genes hardest on the reference fixture")
    else:
        import warnings
        warnings.warn("Hematopoiesis fixture checksum differs; only structural assertions apply.")

    de = result.delta_embedding(adata)
    assert de.shape == (adata.n_obs, 2)


@pytest.mark.parametrize("mode,factor", [("ko", 0.0), ("kd", 0.5), ("oe", 2.0)])
def test_dynamo_zscore_uses_pca_distribution(monkeypatch, tiny_adata, mode, factor):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)
    tiny_adata.uns["pca_mean"] = np.full(tiny_adata.n_vars, 4.0)
    base = tiny_adata.X[:, 0].mean()
    perturb(tiny_adata, target="G0", mode=mode, backend="dynamo",
            backend_kwargs={"perturb_mode": "z_score"})
    assert hits["levels"]["G0"] == pytest.approx(base * factor, abs=1e-12)


@pytest.mark.parametrize("mode,factor", [("kd", 0.5), ("oe", 2.0)])
@pytest.mark.parametrize("perturb_mode", ["raw", "z_score"])
def test_dynamo_multitarget_keeps_gene_values_aligned(
        monkeypatch, tiny_adata, mode, factor, perturb_mode):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)
    tiny_adata.X[:, 4] *= 10.0
    result = perturb(tiny_adata, target=["G4", "G0"], mode=mode, backend="dynamo",
                     backend_kwargs={"perturb_mode": perturb_mode})
    assert hits["levels"]["G4"] == pytest.approx(tiny_adata.X[:, 4].mean() * factor)
    assert hits["levels"]["G0"] == pytest.approx(tiny_adata.X[:, 0].mean() * factor)
    assert result.target == ["G4", "G0"]


@pytest.mark.parametrize("key,value", [
    ("add_delta_Y_key", "custom_response"),
    ("add_transition_key", "custom_transitions"),
    ("add_velocity_key", "custom_velocity"),
    ("add_embedding_key", "custom_embedding"),
])
def test_dynamo_custom_output_keys(monkeypatch, tiny_adata, key, value):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)
    tiny_adata.layers["j_delta_x_perturbation"] = np.zeros(tiny_adata.shape)
    tiny_adata.obsp["perturbation_transition_matrix"] = np.eye(tiny_adata.n_obs)
    kwargs = {key: value, "group": "cluster"}
    tiny_adata.obs["cluster"] = "all"
    result = perturb(tiny_adata, target="G0", backend="dynamo", backend_kwargs=kwargs)
    assert kwargs == {key: value, "group": "cluster"}
    pkey = value if key == "add_delta_Y_key" else "j_delta_x_perturbation"
    tkey = value if key == "add_transition_key" else "perturbation_transition_matrix"
    vkey = value if key == "add_velocity_key" else "velocity_umap_perturbation"
    ekey = value if key == "add_embedding_key" else "X_umap_perturbation"
    np.testing.assert_allclose(result.delta_X, tiny_adata.layers[pkey])
    np.testing.assert_allclose(result.trajectory_shift, tiny_adata.obsp[tkey])
    assert not np.allclose(result.trajectory_shift, np.eye(tiny_adata.n_obs))
    assert result.meta["velocity_embedding_key"] == vkey
    assert ekey in tiny_adata.obsm
    assert result.meta["rank_key"] == f"rank_{pkey}"


def _install_fake_dynamo_pipeline(fake, hits):
    hits["pipeline"] = []

    def recipe(adata, **kwargs):
        hits["pipeline"].append("recipe")
        hits["recipe_kwargs"] = kwargs
        _dynamo_ready_adata(adata)
        for key in ("VecFld_pca", "dynamics", "neighbors"):
            adata.uns.pop(key)
        del adata.obsm["X_umap"]

    def dynamics(adata, **kwargs):
        hits["pipeline"].append("dynamics")
        hits["dynamics_kwargs"] = kwargs
        adata.uns["dynamics"] = {"est_method": kwargs.get("est_method", "auto")}

    def neighbors(adata, basis="pca", **kwargs):
        hits["pipeline"].append("neighbors")
        adata.uns["neighbors"] = {"indices": np.zeros((adata.n_obs, 3), dtype=int)}

    def reduce_dimension(adata, reduction_method="umap", **kwargs):
        hits["pipeline"].append(reduction_method)
        adata.obsm[f"X_{reduction_method}"] = adata.obsm["X_pca"][:, :2].copy()

    def velocities(adata, basis, enforce=False):
        hits["velocity_enforce"] = enforce
        assert "neighbors" in adata.uns
        assert "X_umap" in adata.obsm
        hits["pipeline"].append("velocities")
        adata.obsm[f"velocity_{basis}"] = np.zeros_like(adata.obsm[f"X_{basis}"])

    def vector_field(adata, basis):
        assert f"velocity_{basis}" in adata.obsm
        hits["pipeline"].append("vector_field")
        adata.uns[f"VecFld_{basis}"] = {"model": None}

    class Preprocessor:
        def __init__(self, convert_gene_name_function, gene_append_list):
            self.convert_gene_name = convert_gene_name_function
            self.targets = gene_append_list
            hits["preprocessor_targets"] = list(gene_append_list)

        def preprocess_adata(self, adata, recipe="monocle"):
            assert recipe == "monocle"
            original = adata.var_names.copy()
            self.convert_gene_name(adata)
            assert adata.var_names.equals(original)
            prepare(adata, genes_to_append=self.targets)

    prepare = recipe
    fake.pp = types.SimpleNamespace(recipe_monocle=recipe, Preprocessor=Preprocessor)
    fake.tl = types.SimpleNamespace(dynamics=dynamics, neighbors=neighbors,
                                    reduceDimension=reduce_dimension,
                                    cell_velocities=velocities)
    fake.vf = types.SimpleNamespace(VectorField=vector_field)


def test_dynamo_auto_pipeline_prepares_neighbors_and_embedding(monkeypatch, tiny_adata):
    from omicverse.single import perturb

    hits = {"calls": []}
    fake = _install_fake_dynamo(monkeypatch, hits=hits)
    _install_fake_dynamo_pipeline(fake, hits)
    tiny_adata.layers["spliced"] = tiny_adata.X.copy()
    tiny_adata.layers["unspliced"] = tiny_adata.X.copy()
    tiny_adata.obsp["pearson_transition_matrix"] = np.eye(tiny_adata.n_obs)
    tiny_adata.uns["jacobian_pca"] = {"stale": True}
    result = perturb(tiny_adata, target="G0", backend="dynamo",
                     backend_kwargs={"auto_pipeline": True})
    assert hits["pipeline"] == ["recipe", "dynamics", "neighbors", "umap", "velocities", "vector_field"]
    assert result.embedding.shape == (tiny_adata.n_obs, 2)
    assert hits["velocity_enforce"] is True
    assert "jacobian_pca" not in tiny_adata.uns
    assert hits["preprocessor_targets"] == ["G0"]


def test_dynamo_auto_pipeline_preserves_learned_field(monkeypatch, tiny_adata):
    from omicverse.single import perturb

    hits = {"calls": []}
    fake = _install_fake_dynamo(monkeypatch, hits=hits)
    _install_fake_dynamo_pipeline(fake, hits)
    _dynamo_ready_adata(tiny_adata)
    field = tiny_adata.uns["VecFld_pca"]
    del tiny_adata.obsm["X_umap"]
    del tiny_adata.uns["neighbors"]
    perturb(tiny_adata, target="G0", backend="dynamo",
            backend_kwargs={"auto_pipeline": True})
    assert hits["pipeline"] == ["neighbors", "umap"]
    assert tiny_adata.uns["VecFld_pca"] is field


@pytest.mark.parametrize("missing", ["embedding", "neighbors", "dynamics"])
def test_dynamo_checks_projection_prerequisites(monkeypatch, tiny_adata, missing):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)
    if missing == "embedding":
        del tiny_adata.obsm["X_umap"]
    else:
        del tiny_adata.uns[missing]
    with pytest.raises(RuntimeError, match="auto_pipeline"):
        perturb(tiny_adata, target="G0", backend="dynamo")
    assert not hits["calls"]


def test_dynamo_fixture_path_is_portable(monkeypatch, tmp_path):
    monkeypatch.delenv("OMICVERSE_HEMATOPOIESIS_PATH", raising=False)
    assert _hematopoiesis_path() == Path(__file__).resolve().parents[2] / "data/hematopoiesis.h5ad"
    supplied = tmp_path / "hematopoiesis.h5ad"
    monkeypatch.setenv("OMICVERSE_HEMATOPOIESIS_PATH", str(supplied))
    assert _hematopoiesis_path() == supplied
    assert _real_dynamo_and_data() is None


def test_dynamo_reference_checksum(monkeypatch, tmp_path):
    supplied = tmp_path / "fixture.h5ad"
    supplied.write_bytes(b"reference data")
    monkeypatch.setattr(sys.modules[__name__], "_HEMATOPOIESIS_SHA256",
                        hashlib.sha256(b"reference data").hexdigest())
    assert _is_reference_hematopoiesis(supplied)
    supplied.write_bytes(b"updated data")
    assert not _is_reference_hematopoiesis(supplied)


def test_dynamo_registered_plot_example_binds():
    import ast
    import inspect
    import omicverse as ov
    from omicverse.single import perturb

    tree = ast.parse(Path(inspect.getsourcefile(perturb)).read_text())
    examples = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value.startswith("ov.pl.perturb_quiver(")]
    assert examples
    signature = inspect.signature(ov.pl.perturb_quiver)
    fake = types.SimpleNamespace(pl=types.SimpleNamespace(perturb_quiver=signature.bind))
    for example in examples:
        exec(example, {"ov": fake, "adata": object(), "result": object()})


@pytest.mark.parametrize("language", ["docs", "docs_zh"])
@pytest.mark.parametrize("workflow", [0, 1])
def test_dynamo_documented_raw_workflow(monkeypatch, tiny_adata, language, workflow):
    import json
    import re
    from omicverse.single import perturb

    notebook = (Path(__file__).resolve().parents[2] / "omicverse_guide" / language
                / "Tutorials-single/perturb/t_perturb_dynamo.ipynb")
    if not notebook.is_file():
        pytest.skip("tutorial submodule is not available")
    cells = json.loads(notebook.read_text())["cells"]
    section = next("".join(c["source"]) for c in cells
                   if "".join(c["source"]).startswith("## 7."))
    example = re.findall(r"```python\n(.*?)```", section, re.S)[workflow]
    hits = {"calls": []}
    fake = _install_fake_dynamo(monkeypatch, hits=hits)
    _install_fake_dynamo_pipeline(fake, hits)
    tiny_adata.var_names = ["GATA1", *tiny_adata.var_names[1:]]
    tiny_adata.layers["spliced"] = tiny_adata.X.copy()
    tiny_adata.layers["unspliced"] = tiny_adata.X.copy()
    scope = {"adata": tiny_adata, "ov": types.SimpleNamespace(single=types.SimpleNamespace(perturb=perturb))}
    exec(example, scope)
    assert scope["result"].backend == "dynamo"
    assert hits["pipeline"] == ["recipe", "dynamics", "neighbors", "umap", "velocities", "vector_field"]


@pytest.mark.parametrize("mode,factor", [("ko", 0.0), ("kd", 0.5), ("oe", 2.0)])
@pytest.mark.parametrize("perturb_mode", ["raw", "z_score"])
def test_dynamo_native_math_and_output_keys(monkeypatch, tiny_adata, mode, factor, perturb_mode):
    import importlib
    pytest.importorskip("dynamo")
    from omicverse.single import perturb

    native = importlib.import_module("dynamo.prediction.perturbation")
    _dynamo_ready_adata(tiny_adata)
    tiny_adata.X[:, 4] *= 10.0
    pcs = np.zeros((tiny_adata.n_vars, 3))
    pcs[[0, 4, 7], np.arange(3)] = 1.0
    tiny_adata.uns["PCs"] = pcs
    means = tiny_adata.X.mean(axis=0) + 1.0
    tiny_adata.uns["pca_mean"] = means
    tiny_adata.obsm["X_pca"] = 0.2 * (tiny_adata.X[:, [0, 4, 7]] - means[[0, 4, 7]])
    tiny_adata.uns["jacobian_pca"] = {
        "jacobian": np.repeat(np.eye(3)[:, :, None], tiny_adata.n_obs, axis=2)}
    reconstructed = tiny_adata.obsm["X_pca"] @ pcs.T + means

    def project(adata, *, V, add_transition_key, add_velocity_key, **kwargs):
        adata.obsp[add_transition_key] = np.eye(adata.n_obs)
        adata.obsm[add_velocity_key] = V[:, :2]

    monkeypatch.setattr(native, "cell_velocities", project)
    result = perturb(tiny_adata, target=["G4", "G0"], mode=mode, backend="dynamo",
                     backend_kwargs={"perturb_mode": perturb_mode,
                                     "add_delta_Y_key": "native_response",
                                     "add_transition_key": "native_transitions",
                                     "add_velocity_key": "native_velocity",
                                     "add_embedding_key": "native_embedding"})
    for col in (0, 4):
        np.testing.assert_allclose(
            result.delta_X[:, col], tiny_adata.X[:, col].mean() * factor - reconstructed[:, col],
            atol=1e-12)
    np.testing.assert_allclose(result.trajectory_shift, tiny_adata.obsp["native_transitions"])
    assert result.meta["velocity_embedding_key"] == "native_velocity"
    assert "native_embedding" in tiny_adata.obsm



def test_dynamo_zscore_constant_reconstruction_rejected(monkeypatch, tiny_adata):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)
    tiny_adata.uns["PCs"][0] = 0.0
    with pytest.raises(ValueError, match="perturb_mode='raw'"):
        perturb(tiny_adata, target="G0", backend="dynamo",
                backend_kwargs={"perturb_mode": "z_score"})
    assert not hits["calls"]


def test_dynamo_rejects_targets_outside_pca(monkeypatch, tiny_adata):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)
    tiny_adata.var.loc["G4", "use_for_pca"] = False
    with pytest.raises(ValueError, match="G4"):
        perturb(tiny_adata, target=["G0", "G4"], backend="dynamo")
    assert not hits["calls"]


def test_dynamo_auto_pipeline_requires_velocity_layers(monkeypatch, tiny_adata):
    from omicverse.single import perturb

    hits = {"calls": []}
    fake = _install_fake_dynamo(monkeypatch, hits=hits)
    _install_fake_dynamo_pipeline(fake, hits)
    with pytest.raises(RuntimeError, match="spliced/unspliced"):
        perturb(tiny_adata, target="G0", backend="dynamo",
                backend_kwargs={"auto_pipeline": True})
    assert not hits["pipeline"]


def test_dynamo_native_rejects_unknown_option(tiny_adata):
    pytest.importorskip("dynamo")
    from omicverse.single import perturb

    _dynamo_ready_adata(tiny_adata)
    with pytest.raises(TypeError, match="made_up_argument"):
        perturb(tiny_adata, target="G0", backend="dynamo",
                backend_kwargs={"made_up_argument": True})



def test_dynamo_transform_does_not_reuse_stale_transition(monkeypatch, tiny_adata):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)
    previous = perturb(tiny_adata, target="G0", backend="dynamo")
    assert previous.trajectory_shift is not None
    current = perturb(tiny_adata, target="G0", backend="dynamo",
                      backend_kwargs={"projection_method": "transform"})
    assert current.trajectory_shift is None
    np.testing.assert_allclose(
        tiny_adata.obsp["perturbation_transition_matrix"], previous.trajectory_shift)



@pytest.mark.parametrize("mode", ["ko", "kd", "oe"])
def test_dynamo_raw_hematopoiesis_full_pipeline(record_property, monkeypatch, mode):
    """Fit every stage on all raw cells and genes without cached learned state."""
    path = Path(os.environ.get(
        "OMICVERSE_RAW_HEMATOPOIESIS_PATH",
        Path(__file__).resolve().parents[2] / "data/hematopoiesis_raw.h5ad",
    )).expanduser()
    if not path.is_file():
        pytest.skip("raw hematopoiesis fixture is not available locally")
    pytest.importorskip("dynamo")
    import anndata as ad
    import omicverse as ov
    import time

    source = ad.read_h5ad(path)
    raw_keys = ("spliced", "unspliced", "new", "total", "uu", "ul", "su", "sl")
    raw = ad.AnnData(
        X=source.X.copy(), obs=source.obs.copy(), var=source.var.copy(),
        layers={key: source.layers[key].copy() for key in raw_keys if key in source.layers},
    )
    input_shape = raw.shape
    assert raw.shape == source.shape
    assert not raw.obsm and not raw.obsp and not raw.uns
    target = "ENSG00000102145"
    assert target in raw.var_names
    assert any(keys.issubset(raw.layers) for keys in (
        {"spliced", "unspliced"}, {"new", "total"}, {"uu", "ul", "su", "sl"}))
    del source

    def reject_annotation_conversion(*args, **kwargs):
        raise AssertionError("auto_pipeline must preserve input gene IDs without downloading annotations")

    monkeypatch.setattr("dynamo.preprocessing.utils.convert2gene_symbol", reject_annotation_conversion)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    record_property("raw_sha256", digest.hexdigest())
    record_property("raw_input_shape", str(input_shape))
    record_property("raw_target", target)
    print(f"Full raw input: {input_shape}, target={target}, layers={list(raw.layers)}")
    started = time.monotonic()
    result = ov.single.perturb(
        raw, target=target, mode=mode, backend="dynamo",
        backend_kwargs={"auto_pipeline": True, "group": "cell_type",
                        "vf_ko": mode == "ko", "add_delta_Y_key": f"raw_{mode}_response"},
    )
    seconds = time.monotonic() - started
    record_property("processed_shape", str(raw.shape))
    record_property("mode", mode)
    record_property("group", "cell_type")
    record_property("vf_ko", mode == "ko")
    record_property("pipeline_seconds", seconds)
    record_property("pca_genes", int(raw.var["use_for_pca"].sum()))
    record_property("pca_components", raw.obsm["X_pca"].shape[1])
    assert result.backend == "dynamo"
    assert result.adata_perturbed is raw
    assert raw.n_obs <= input_shape[0] and raw.n_vars <= input_shape[1]
    assert raw.var.loc[target, "use_for_pca"]
    assert result.meta["dynamics_est_method"] == "twostep"
    record_property("dynamics_est_method", result.meta["dynamics_est_method"])
    assert {"PCs", "pca_mean", "dynamics", "VecFld_pca", "jacobian_pca"}.issubset(raw.uns)
    assert {"X_pca", "X_umap", "velocity_pca"}.issubset(raw.obsm)
    assert "distances" in raw.obsp
    assert result.delta_X.shape == raw.shape
    assert np.isfinite(result.delta_X).all()
    assert np.any(np.abs(result.delta_X) > 1e-12)
    assert result.trajectory_shift.shape == (raw.n_obs, raw.n_obs)
    assert np.isfinite(result.trajectory_shift).all()
    assert np.all(result.trajectory_shift >= -1e-12)
    np.testing.assert_allclose(result.trajectory_shift.sum(axis=1), 1.0, atol=1e-8)
    shifts = result.delta_embedding(raw)
    assert shifts.shape == (raw.n_obs, 2)
    assert np.isfinite(shifts).all()
    _assert_dynamo_real_group_ranking(raw, result)
    record_property("groups", len(raw.uns[result.meta["rank_key"]].columns))
    if mode == "ko":
        _assert_dynamo_real_vf_ko(raw, result)
    factor = {"ko": 0.0, "kd": 0.5, "oe": 2.0}[mode]
    expected = float(raw.X[:, raw.var_names.get_loc(target)].mean()) * factor
    assert result.meta["expression"][target] == pytest.approx(expected)
    print(f"Full raw pipeline {mode}: {input_shape} -> {raw.shape}, {seconds:.2f}s")



def test_dynamo_auto_pipeline_preserves_ensembl_identifiers(monkeypatch, tiny_adata):
    from omicverse.single import perturb

    hits = {"calls": []}
    fake = _install_fake_dynamo(monkeypatch, hits=hits)
    _install_fake_dynamo_pipeline(fake, hits)
    tiny_adata.var_names = [f"ENSG{i:011d}" for i in range(tiny_adata.n_vars)]
    original_names = tiny_adata.var_names.copy()
    tiny_adata.layers["spliced"] = tiny_adata.X.copy()
    tiny_adata.layers["unspliced"] = tiny_adata.X.copy()

    def legacy_recipe(*args, **kwargs):
        raise AssertionError("legacy recipe would convert Ensembl IDs")

    monkeypatch.setattr(fake.pp, "recipe_monocle", legacy_recipe)
    result = perturb(tiny_adata, target=original_names[0], backend="dynamo",
                     backend_kwargs={"auto_pipeline": True})
    assert tiny_adata.var_names.equals(original_names)
    assert result.target == original_names[0]



@pytest.mark.parametrize("experiment,has_splicing,resolved,expected", [
    ("kin", True, False, {"est_method": "twostep"}),
    ("kin", False, False, {}),
    ("conventional", True, False, {}),
    ("kin", True, True, {}),
])
def test_dynamo_auto_pipeline_selects_supported_kinetics(
        monkeypatch, tiny_adata, experiment, has_splicing, resolved, expected):
    from omicverse.single import perturb

    hits = {"calls": []}
    fake = _install_fake_dynamo(monkeypatch, hits=hits)
    _install_fake_dynamo_pipeline(fake, hits)
    _dynamo_ready_adata(tiny_adata)
    tiny_adata.layers["spliced"] = tiny_adata.X.copy()
    tiny_adata.layers["unspliced"] = tiny_adata.X.copy()
    tiny_adata.uns["pp"] = {"experiment_type": experiment,
                             "has_splicing": has_splicing,
                             "splicing_labeling": resolved}
    del tiny_adata.uns["dynamics"]
    del tiny_adata.uns["VecFld_pca"]
    perturb(tiny_adata, target="G0", backend="dynamo",
            backend_kwargs={"auto_pipeline": True})
    assert hits["dynamics_kwargs"] == expected



def _assert_dynamo_real_group_ranking(adata, result):
    pkey = result.meta["perturbation_key"]
    labels = np.asarray(adata.obs["cell_type"])
    genes = adata.var_names[adata.var["use_for_dynamics"].to_numpy()]
    signed = adata.uns[result.meta["rank_key"]]
    absolute = adata.uns[f"rank_abs_{pkey}"]
    for frame, use_absolute in ((signed, False), (absolute, True)):
        assert isinstance(frame, pd.DataFrame)
        assert set(frame.columns) == set(np.unique(labels))
        for group in frame.columns:
            assert set(frame[group]) == set(genes)
            positions = adata.var_names.get_indexer(frame[group])
            values = result.delta_X[labels == group][:, positions]
            means = (np.abs(values) if use_absolute else values).mean(axis=0)
            tolerance = max(float(np.max(np.abs(means))), 1.0) * 1e-10
            assert np.all(np.diff(means) <= tolerance)


def _assert_dynamo_real_vf_ko(adata, result):
    assert result.meta["vf_ko_keys"] == {
        "pca": "velocity_pca_KO", "emb_basis": "velocity_umap_KO"}
    pca = np.asarray(adata.obsm["velocity_pca_KO"])
    embedding = np.asarray(adata.obsm["velocity_umap_KO"])
    assert pca.shape == adata.obsm["X_pca"].shape
    assert embedding.shape == (adata.n_obs, 2)
    assert np.isfinite(pca).all() and np.isfinite(embedding).all()
    assert np.any(np.abs(pca) > 1e-12)
    np.testing.assert_allclose(adata.obsm["X_pca_KO"], adata.obsm["X_pca"])
    np.testing.assert_allclose(adata.obsm["X_umap_KO"], adata.obsm["X_umap"])


@pytest.mark.parametrize("mode,factor", [("ko", 0.0), ("kd", 0.5), ("oe", 2.0)])
@pytest.mark.parametrize("perturb_mode", ["raw", "z_score"])
def test_dynamo_real_modes_group_and_output_keys(mode, factor, perturb_mode, record_property):
    path = _real_dynamo_and_data()
    if path is None:
        pytest.skip("dynamo-release or the hematopoiesis dataset is not available")
    import anndata as ad
    import dynamo as dyn
    from omicverse.single import perturb

    adata = ad.read_h5ad(path)
    reference = adata.copy()
    base = float(adata.X[:, adata.var_names.get_loc("GATA1")].mean())
    prefix = f"{mode}_{perturb_mode}"
    result = perturb(
        adata, target="GATA1", mode=mode, backend="dynamo",
        backend_kwargs={"perturb_mode": perturb_mode, "group": "cell_type",
                        "add_delta_Y_key": f"{prefix}_response",
                        "add_transition_key": f"{prefix}_transitions",
                        "add_velocity_key": f"{prefix}_velocity",
                        "add_embedding_key": f"{prefix}_embedding"},
    )
    dyn.pd.perturbation(reference, genes=["GATA1"], expression=[base * factor],
                        perturb_mode="raw", emb_basis="umap", pertubation_method="j_delta_x")
    expected = reference.layers["j_delta_x_perturbation"].toarray()
    np.testing.assert_array_equal(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        reference.X.toarray() if hasattr(reference.X, "toarray") else reference.X,
    )
    assert result.grn is None and result.grn_base is None and result.delta_grn.empty
    np.testing.assert_allclose(result.delta_X, expected, rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(result.trajectory_shift,
                               reference.obsp["perturbation_transition_matrix"].toarray(),
                               rtol=1e-4, atol=1e-6)
    assert np.isfinite(result.delta_X).all()
    assert np.any(np.abs(result.delta_X) > 1e-12)
    np.testing.assert_allclose(result.trajectory_shift.sum(axis=1), 1.0, atol=1e-8)
    shifts = result.delta_embedding(adata)
    assert shifts.shape == (adata.n_obs, 2) and np.isfinite(shifts).all()
    _assert_dynamo_real_group_ranking(adata, result)
    record_property("mode", mode)
    record_property("perturb_mode", perturb_mode)
    record_property("input_shape", str(adata.shape))
    record_property("groups", len(adata.uns[result.meta["rank_key"]].columns))


def test_dynamo_real_vf_ko_group_and_native_parity(record_property):
    path = _real_dynamo_and_data()
    if path is None:
        pytest.skip("dynamo-release or the hematopoiesis dataset is not available")
    import anndata as ad
    import dynamo as dyn
    from omicverse.single import perturb

    adata = ad.read_h5ad(path)
    reference = adata.copy()
    original_coordinates = adata.obsm["X_pca"].copy()
    original_field = adata.uns["VecFld_pca"]
    result = perturb(adata, target="GATA1", mode="ko", backend="dynamo",
                     backend_kwargs={"vf_ko": True, "group": "cell_type",
                                     "add_delta_Y_key": "vf_response",
                                     "add_transition_key": "vf_transitions"})
    dyn.pd.perturbation(reference, genes=["GATA1"], expression=[0.0],
                        perturb_mode="raw", emb_basis="umap", pertubation_method="j_delta_x")
    np.testing.assert_allclose(result.delta_X, reference.layers["j_delta_x_perturbation"].toarray())
    np.testing.assert_allclose(result.trajectory_shift,
                               reference.obsp["perturbation_transition_matrix"].toarray())
    dyn.pd.KO(reference, KO_genes=["GATA1"], emb_basis="umap")
    _assert_dynamo_real_vf_ko(adata, result)
    _assert_dynamo_real_group_ranking(adata, result)
    for key in ("velocity_pca_KO", "velocity_umap_KO"):
        np.testing.assert_allclose(adata.obsm[key], reference.obsm[key], rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(adata.obsm["X_pca"], original_coordinates)
    assert adata.uns["VecFld_pca"] is original_field
    np.testing.assert_allclose(result.trajectory_shift, adata.obsp["vf_transitions"].toarray())
    assert np.isfinite(result.delta_embedding(adata)).all()
    record_property("input_shape", str(adata.shape))
    record_property("groups", len(adata.uns[result.meta["rank_key"]].columns))



def test_dynamo_real_sequential_results_keep_custom_keys():
    path = _real_dynamo_and_data()
    if path is None:
        pytest.skip("dynamo-release or the hematopoiesis dataset is not available")
    import anndata as ad
    from omicverse.single import perturb

    adata = ad.read_h5ad(path)
    results = {}
    snapshots = {}
    for mode in ("kd", "oe"):
        result = perturb(
            adata, target="GATA1", mode=mode, backend="dynamo",
            backend_kwargs={"group": "cell_type", "add_delta_Y_key": f"sequence_{mode}",
                            "add_transition_key": f"sequence_{mode}_transitions",
                            "add_velocity_key": f"sequence_{mode}_velocity",
                            "add_embedding_key": f"sequence_{mode}_embedding"},
        )
        results[mode] = result
        snapshots[mode] = (result.delta_X.copy(), result.trajectory_shift.copy(),
                            adata.uns[result.meta["rank_key"]].copy())
    assert not np.allclose(results["kd"].delta_X, results["oe"].delta_X)
    for mode, result in results.items():
        dx, transition, ranking = snapshots[mode]
        np.testing.assert_array_equal(result.delta_X, dx)
        np.testing.assert_array_equal(result.trajectory_shift, transition)
        pd.testing.assert_frame_equal(adata.uns[result.meta["rank_key"]], ranking)
        np.testing.assert_allclose(adata.layers[f"sequence_{mode}"].toarray(), dx)
        _assert_dynamo_real_group_ranking(adata, result)


def test_dynamo_real_copy_keeps_input_clean_with_vf_ko():
    path = _real_dynamo_and_data()
    if path is None:
        pytest.skip("dynamo-release or the hematopoiesis dataset is not available")
    import anndata as ad
    from omicverse.single import perturb

    adata = ad.read_h5ad(path)
    keys = {name: set(getattr(adata, name)) for name in ("layers", "obsm", "obsp", "uns")}
    original_x = adata.X.copy()
    result = perturb(adata, target="GATA1", mode="ko", backend="dynamo", copy=True,
                     backend_kwargs={"group": "cell_type", "vf_ko": True})
    assert result.adata_perturbed is not adata
    for name, expected in keys.items():
        assert set(getattr(adata, name)) == expected
    observed = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    expected = original_x.toarray() if hasattr(original_x, "toarray") else np.asarray(original_x)
    np.testing.assert_array_equal(observed, expected)
    _assert_dynamo_real_group_ranking(result.adata_perturbed, result)
    _assert_dynamo_real_vf_ko(result.adata_perturbed, result)


@pytest.mark.parametrize("mode", ["kd", "oe"])
def test_dynamo_real_nonko_vf_ko_warns(mode):
    path = _real_dynamo_and_data()
    if path is None:
        pytest.skip("dynamo-release or the hematopoiesis dataset is not available")
    import anndata as ad
    from omicverse.single import perturb

    adata = ad.read_h5ad(path)
    ko_keys = {key for key in adata.obsm if key.endswith("_KO")}
    with pytest.warns(RuntimeWarning, match="vf_ko"):
        result = perturb(adata, target="GATA1", mode=mode, backend="dynamo",
                         backend_kwargs={"group": "cell_type", "vf_ko": True})
    assert result.meta["vf_ko_keys"] == {}
    assert {key for key in adata.obsm if key.endswith("_KO")} == ko_keys
    assert np.isfinite(result.delta_X).all()
    _assert_dynamo_real_group_ranking(adata, result)



@pytest.mark.parametrize("option,key", [
    ("add_delta_Y_key", "velocity_pca_KO"),
    ("add_transition_key", "pearson_transition_matrix"),
    ("add_velocity_key", "velocity_umap_KO"),
    ("add_embedding_key", "X_pca_KO"),
])
def test_dynamo_vf_ko_rejects_output_key_collisions(monkeypatch, tiny_adata, option, key):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)
    with pytest.raises(ValueError, match="collid"):
        perturb(tiny_adata, target="G0", backend="dynamo",
                backend_kwargs={"vf_ko": True, option: key})
    assert not hits["calls"]


@pytest.mark.parametrize("option,key", [
    ("add_transition_key", "pearson_transition_matrix"),
    ("add_velocity_key", "velocity_umap_KO"),
])
def test_dynamo_real_vf_ko_rejects_collisions_before_mutation(option, key):
    path = _real_dynamo_and_data()
    if path is None:
        pytest.skip("dynamo-release or the hematopoiesis dataset is not available")
    import anndata as ad
    from omicverse.single import perturb

    adata = ad.read_h5ad(path)
    before = {name: set(getattr(adata, name)) for name in ("layers", "obsm", "obsp", "uns")}
    with pytest.raises(ValueError, match="collid"):
        perturb(adata, target="GATA1", backend="dynamo",
                backend_kwargs={"vf_ko": True, "group": "cell_type", option: key})
    for name, keys in before.items():
        assert set(getattr(adata, name)) == keys


@pytest.mark.parametrize("baseline,response,expected_log2_fc", [
    (0.1, -0.5, np.nan),
    (-1.0, -0.5, np.nan),
    (0.0, 0.0, 0.0),
    (2.0, 2.0, 0.999999639326375),
])
def test_dynamo_response_summary_preserves_signed_effects(
    monkeypatch, tiny_adata, baseline, response, expected_log2_fc,
):
    from omicverse.single import perturb

    fake = _install_fake_dynamo(monkeypatch, hits={"calls": []})
    native_perturbation = fake.pd.perturbation

    def fixed_response(adata, **kwargs):
        native_perturbation(adata, **kwargs)
        adata.layers[kwargs["add_delta_Y_key"]][:, 1] = response

    fake.pd.perturbation = fixed_response
    _dynamo_ready_adata(tiny_adata)
    tiny_adata.X[:, 1] = baseline
    with np.errstate(invalid="raise", divide="raise"):
        result = perturb(tiny_adata, target="G0", backend="dynamo")
    summary = result.delta_expr.set_index("gene").loc["G1"]
    assert summary["mean_base"] == pytest.approx(baseline)
    assert summary["mean_pert"] == pytest.approx(baseline + response)
    assert summary["delta"] == pytest.approx(response)
    np.testing.assert_allclose(result.delta_X[:, 1], response)
    if np.isnan(expected_log2_fc):
        assert np.isnan(summary["log2_fc"])
    else:
        assert summary["log2_fc"] == pytest.approx(expected_log2_fc)


@pytest.mark.parametrize("copy", [False, True])
def test_dynamo_result_contract_keeps_baseline_without_grn(monkeypatch, tiny_adata, copy):
    from omicverse.single import perturb

    _install_fake_dynamo(monkeypatch, hits={"calls": []})
    _dynamo_ready_adata(tiny_adata)
    baseline = tiny_adata.X.copy()
    input_layer_keys = set(tiny_adata.layers)
    result = perturb(tiny_adata, target="G0", backend="dynamo", grn_output=True, copy=copy)
    np.testing.assert_array_equal(tiny_adata.X, baseline)
    np.testing.assert_array_equal(result.adata_perturbed.X, baseline)
    assert (result.adata_perturbed is tiny_adata) == (not copy)
    assert result.grn is None and result.grn_base is None and result.delta_grn.empty
    np.testing.assert_array_equal(
        result.delta_X, result.adata_perturbed.layers[result.meta["perturbation_key"]])
    assert np.any(result.delta_X != 0)
    if copy:
        assert set(tiny_adata.layers) == input_layer_keys


@pytest.mark.parametrize("mode,level", [("kd", 2.0), ("oe", 8.0)])
@pytest.mark.parametrize("perturb_mode", ["raw", "z_score"])
def test_dynamo_kd_oe_request_common_level_across_heterogeneous_cells(
    monkeypatch, tiny_adata, mode, level, perturb_mode,
):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)
    tiny_adata.X[:, 0] = np.tile([0.0, 2.0, 6.0, 8.0], 10)
    baseline = tiny_adata.X.copy()
    perturb(tiny_adata, target="G0", mode=mode, fold_change=2.0, backend="dynamo",
            backend_kwargs={"perturb_mode": perturb_mode})
    assert hits["levels"]["G0"] == pytest.approx(level)
    np.testing.assert_array_equal(tiny_adata.X, baseline)


def test_dynamo_bilingual_tutorial_examples_and_navigation():
    import ast
    import json
    import re

    guide = Path(__file__).resolve().parents[2] / "omicverse_guide"
    relative = Path("Tutorials-single/perturb/t_perturb_dynamo.ipynb")
    if not (guide / "docs" / relative).is_file():
        pytest.skip("tutorial submodule is not available")
    notebooks = [json.loads((guide / language / relative).read_text())
                 for language in ("docs", "docs_zh")]
    examples = []
    for notebook in notebooks:
        examples.append([
            ast.dump(ast.parse(source))
            for cell in notebook["cells"]
            for source in (["".join(cell["source"])] if cell["cell_type"] == "code"
                           else re.findall(r"```python\n(.*?)```", "".join(cell["source"]), re.S))
            if not source.startswith("%")
        ])
    assert examples[0] == examples[1]
    assert examples[0]
    for language in ("docs", "docs_zh"):
        index = guide / language / relative.parent / "index.md"
        assert "t_perturb_dynamo" in index.read_text()
        navigation = guide / language / "tutorials" / (
            "index_single_perturb.md" if language == "docs" else "index_single.md")
        assert "../Tutorials-single/perturb/index" in navigation.read_text()


@pytest.mark.parametrize("language", ["docs", "docs_zh"])
def test_dynamo_tutorial_tables_have_portable_html_outputs(language):
    from html.parser import HTMLParser
    import json

    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = 0
            self.headers = []
            self.in_header = False

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.rows += 1
            elif tag == "th":
                self.in_header = True

        def handle_endtag(self, tag):
            if tag == "th":
                self.in_header = False

        def handle_data(self, data):
            if self.in_header:
                self.headers.append(data)

    notebook = (Path(__file__).resolve().parents[2] / "omicverse_guide" / language
                / "Tutorials-single/perturb/t_perturb_dynamo.ipynb")
    if not (notebook.parents[3] / "docs").is_dir():
        pytest.skip("tutorial submodule is not available")
    cells = json.loads(notebook.read_text())["cells"]
    table_cells = [cell for cell in cells if cell["cell_type"] == "code" and any(
        fragment in "".join(cell["source"]) for fragment in (
            "result.summary(top_n=10)", "result_oe.summary(top_n=5)",
            "ct.round(2)", "summary.sort_values('mean')", "absolute_rank_df.head(8)",
        ))]
    assert table_cells
    for cell in table_cells:
        outputs = [output["data"] for output in cell["outputs"] if "data" in output]
        assert outputs
        for data in outputs:
            assert not any(key.startswith("application/vnd.omicverse.") for key in data)
            assert "text/html" in data and "text/plain" in data
            parser = TableParser()
            parser.feed("".join(data["text/html"]))
            assert parser.rows > 1 and parser.headers


@pytest.mark.parametrize("storage", ["keys", "arrays", "coordinates", "cached_jacobian"])
def test_dynamo_custom_pca_inputs_match_native(monkeypatch, tiny_adata, storage):
    import importlib
    pytest.importorskip("dynamo")
    from omicverse.single import perturb

    native = importlib.import_module("dynamo.prediction.perturbation")
    _dynamo_ready_adata(tiny_adata)
    tiny_adata.uns["jacobian_pca"] = {
        "jacobian": np.repeat(np.eye(3)[:, :, None], tiny_adata.n_obs, axis=2)}
    options = {"pca_key": "custom_pca"}
    tiny_adata.obsm["custom_pca"] = tiny_adata.obsm.pop("X_pca")
    if storage == "keys":
        for old, new, key in [("PCs", "custom_PCs", "PCs_key"),
                              ("pca_mean", "custom_mean", "pca_mean_key")]:
            tiny_adata.uns[new] = tiny_adata.uns.pop(old)
            options[key] = new
    elif storage == "arrays":
        options["PCs_key"] = tiny_adata.uns.pop("PCs")
        options["pca_mean_key"] = tiny_adata.uns.pop("pca_mean")
    else:
        options["X_pca"] = tiny_adata.obsm.pop("custom_pca")
        options.pop("pca_key")
    if storage == "cached_jacobian":
        tiny_adata.uns.pop("VecFld_pca")
    reference = tiny_adata.copy()

    def project(adata, *, V, add_transition_key, add_velocity_key, **kwargs):
        adata.obsp[add_transition_key] = np.eye(adata.n_obs)
        adata.obsm[add_velocity_key] = V[:, :2]

    monkeypatch.setattr(native, "cell_velocities", project)
    reference_options = dict(options)
    if storage in ("coordinates", "cached_jacobian"):
        reference_options.update(PCs_key="PCs", pca_mean_key="pca_mean")
    native.perturbation(reference, genes=["G0"], expression=[0.0], **reference_options)
    result = perturb(tiny_adata, target="G0", backend="dynamo", backend_kwargs=options)
    np.testing.assert_allclose(result.delta_X, reference.layers["j_delta_x_perturbation"].toarray())
    np.testing.assert_allclose(result.trajectory_shift, reference.obsp["perturbation_transition_matrix"])


@pytest.mark.parametrize("perturb_mode", ["raw", "z_score"])
def test_dynamo_explicit_levels_match_native(monkeypatch, tiny_adata, perturb_mode):
    import importlib
    pytest.importorskip("dynamo")
    from omicverse.single import perturb

    native = importlib.import_module("dynamo.prediction.perturbation")
    _dynamo_ready_adata(tiny_adata)
    tiny_adata.uns["jacobian_pca"] = {
        "jacobian": np.repeat(np.eye(3)[:, :, None], tiny_adata.n_obs, axis=2)}
    reference = tiny_adata.copy()

    def project(adata, *, V, add_transition_key, add_velocity_key, **kwargs):
        adata.obsp[add_transition_key] = np.eye(adata.n_obs)
        adata.obsm[add_velocity_key] = V[:, :2]

    monkeypatch.setattr(native, "cell_velocities", project)
    native.perturbation(reference, genes=["G0", "G4"], expression=[-0.5, 1.25],
                        perturb_mode=perturb_mode)
    options = {"expression": [1.25, -0.5], "perturb_mode": perturb_mode}
    result = perturb(tiny_adata, target=["G4", "G0"], backend="dynamo", backend_kwargs=options)
    np.testing.assert_allclose(result.delta_X, reference.layers["j_delta_x_perturbation"].toarray())
    assert result.meta["expression"] == {"G0": -0.5, "G4": 1.25}
    assert options["expression"] == [1.25, -0.5]


@pytest.mark.parametrize("option,error,match", [
    ("delta_Y", NotImplementedError, "delta_Y"),
    ("pca_key", TypeError, "pca_key must be a string"),
    ("zero_perturb_genes_vel", ValueError, "zero_perturb_genes_vel"),
])
def test_dynamo_rejects_native_faults_before_writing(monkeypatch, tiny_adata, option, error, match):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)
    value = True if option == "zero_perturb_genes_vel" else tiny_adata.obsm["X_pca"].copy()
    if option == "zero_perturb_genes_vel":
        mask = tiny_adata.var_names.isin(["G0", "G4", "G7"])
        tiny_adata.var["use_for_pca"] = mask
        tiny_adata.uns["PCs"] = tiny_adata.uns["PCs"][mask]
        tiny_adata.uns["pca_mean"] = tiny_adata.uns["pca_mean"][mask]
    keys = {name: set(getattr(tiny_adata, name)) for name in ("layers", "obsm", "obsp", "uns")}
    with pytest.raises(error, match=match):
        perturb(tiny_adata, target="G4", backend="dynamo", backend_kwargs={option: value})
    assert not hits["calls"]
    assert keys == {name: set(getattr(tiny_adata, name)) for name in keys}


@pytest.mark.parametrize("method", [
    "j_delta_x", "j_x_prime", "j_jv", "f_x_prime", "f_x_prime_minus_f_x_0",
])
@pytest.mark.parametrize("custom_model", [False, True])
def test_dynamo_real_native_methods_and_parameters(method, custom_model, record_property):
    path = _real_dynamo_and_data()
    if path is None:
        pytest.skip("dynamo-release or the hematopoiesis dataset is not available")
    import anndata as ad
    import dynamo as dyn
    from omicverse.single import perturb

    adata = ad.read_h5ad(path)
    base = float(adata.X[:, adata.var_names.get_loc("GATA1")].mean())
    adata.uns["custom_jacobian"] = adata.uns.pop("jacobian_pca")
    options = {"perturbation_method": method, "cells": np.arange(0, adata.n_obs, 3),
               "delta_t": 0.7, "J_jv_delta_t": 1.3, "jac_key": "custom_jacobian",
               "add_delta_Y_key": "method_response", "add_transition_key": "method_transitions",
               "add_velocity_key": "method_velocity", "add_embedding_key": "method_embedding"}
    if custom_model:
        adata.obsm["custom_pca"] = adata.obsm.pop("X_pca")
        options.update(pca_key="custom_pca", PCs_key=adata.uns.pop("PCs"),
                       pca_mean_key=adata.uns.pop("pca_mean"),
                       expression=0.25, perturb_mode="z_score", basis="pca")
        native_expression = [0.25]
    else:
        options["X_pca"] = adata.obsm["X_pca"].copy()
        native_expression = [base / 2.0]
        pca_genes = adata.var_names[adata.var.use_for_pca]
        options["zero_perturb_genes_vel"] = (
            pca_genes.get_loc("GATA1") == adata.var_names.get_loc("GATA1"))
    reference = adata.copy()
    native_options = dict(options)
    native_options.pop("perturbation_method")
    native_options.pop("expression", None)
    if not custom_model:
        native_options.update(PCs_key="PCs", pca_mean_key="pca_mean")
    dyn.pd.perturbation(reference, genes=["GATA1"], expression=native_expression,
                        pertubation_method=method, **native_options)
    result = perturb(adata, target="GATA1", mode="kd", backend="dynamo", backend_kwargs=options)
    expected = reference.layers["method_response"].toarray()
    np.testing.assert_allclose(result.delta_X, expected, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(adata.obsm["method_response"], reference.obsm["method_response"],
                               rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(result.trajectory_shift,
                               reference.obsp["method_transitions"].toarray(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(adata.obsm["method_velocity"], reference.obsm["method_velocity"],
                               rtol=1e-4, atol=1e-6)
    np.testing.assert_array_equal(adata.obsm["method_embedding"], reference.obsm["method_embedding"])
    record_property("method", method)
    record_property("custom_model", custom_model)
    record_property("cells", len(options["cells"]))
    record_property("max_abs_response_error", float(np.max(np.abs(result.delta_X - expected))))


@pytest.mark.parametrize("projection", ["fp", "pearson", "cosine", "kmc"])
def test_dynamo_real_native_projection_methods(projection, record_property):
    path = _real_dynamo_and_data()
    if path is None:
        pytest.skip("dynamo-release or the hematopoiesis dataset is not available")
    import anndata as ad
    import dynamo as dyn
    from omicverse.single import perturb

    adata = ad.read_h5ad(path)
    reference = adata.copy()
    dyn.pd.perturbation(reference, genes=["GATA1"], expression=[0.0], projection_method=projection)
    result = perturb(adata, target="GATA1", backend="dynamo",
                     backend_kwargs={"projection_method": projection})
    np.testing.assert_allclose(result.delta_X,
                               reference.layers["j_delta_x_perturbation"].toarray(), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(result.trajectory_shift,
                               reference.obsp["perturbation_transition_matrix"].toarray(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(adata.obsm["velocity_umap_perturbation"],
                               reference.obsm["velocity_umap_perturbation"], rtol=1e-4, atol=1e-6)
    record_property("projection_method", projection)


@pytest.mark.parametrize("levels", [[1.0], [np.nan, 1.0]])
def test_dynamo_rejects_invalid_explicit_levels(monkeypatch, tiny_adata, levels):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)
    with pytest.raises(ValueError, match="expression"):
        perturb(tiny_adata, target=["G0", "G4"], backend="dynamo",
                backend_kwargs={"expression": levels})
    assert not hits["calls"]


@pytest.mark.parametrize("model", ["cached_jacobian", "custom_pca"])
def test_dynamo_vf_ko_requires_default_model_before_response(monkeypatch, tiny_adata, model):
    from omicverse.single import perturb

    hits = {"calls": []}
    _install_fake_dynamo(monkeypatch, hits=hits)
    _dynamo_ready_adata(tiny_adata)
    tiny_adata.uns["jacobian_pca"] = {
        "jacobian": np.repeat(np.eye(3)[:, :, None], tiny_adata.n_obs, axis=2)}
    options = {"vf_ko": True}
    if model == "cached_jacobian":
        tiny_adata.uns.pop("VecFld_pca")
    else:
        tiny_adata.obsm["custom_pca"] = tiny_adata.obsm.pop("X_pca")
        tiny_adata.uns["custom_PCs"] = tiny_adata.uns.pop("PCs")
        tiny_adata.uns["custom_mean"] = tiny_adata.uns.pop("pca_mean")
        options.update(pca_key="custom_pca", PCs_key="custom_PCs", pca_mean_key="custom_mean")
    keys = {name: set(getattr(tiny_adata, name)) for name in ("layers", "obsm", "obsp", "uns")}
    with pytest.raises(RuntimeError, match="vf_ko"):
        perturb(tiny_adata, target="G0", backend="dynamo", backend_kwargs=options)
    assert not hits["calls"] and "ko" not in hits
    assert keys == {name: set(getattr(tiny_adata, name)) for name in keys}
