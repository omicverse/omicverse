"""Unit tests for :mod:`omicverse.single._perturb`.

These tests stub out the optional backend libraries (``sctenifoldknk`` /
``celloracle``) and verify:

* the public API surface (``ov.single.perturb`` + ``PerturbResult``)
* mode / backend validation
* the lazy-import error path when neither backend is installed
* end-to-end dispatch through both backends with stubs
* the GRN helper utilities (``_diff_grn`` / ``_apply_perturbation_to_graph``)

The tests intentionally **do not** import the real ``sctenifoldknk`` /
``celloracle`` packages — they monkey-patch fake modules into
``sys.modules`` first, so CI doesn't need either dependency installed.
"""

from __future__ import annotations

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
                    ko[:, idx[g]] = 0.0
            self.tensor_dict = {"WT": wt, "KO": ko}

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


def test_sctenifoldknk_backend_oe(monkeypatch, tiny_adata, fake_grn):
    from omicverse.single import perturb

    _install_fake_sctenifoldknk(monkeypatch, fake_grn)

    result = perturb(tiny_adata, target="G0", mode="oe",
                     fold_change=3.0, backend="sctenifoldknk")
    assert result.mode == "oe"
    out_edges = result.delta_grn[result.delta_grn["source"] == "G0"]
    # weights should be tripled
    np.testing.assert_allclose(
        out_edges["weight_pert"].to_numpy(),
        out_edges["weight_base"].to_numpy() * 3.0,
    )


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
