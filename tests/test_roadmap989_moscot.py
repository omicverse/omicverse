"""Boundary tests and separately selected real-backend integration checks."""
import importlib
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from omicverse import _backend_validation as validation


from omicverse import _moscot as ot

@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    a = ad.AnnData(rng.poisson(4, (64, 24)).astype(np.float32))
    a.layers["counts"] = a.X.copy()
    a.obs["sample"] = pd.Categorical(np.repeat(["a", "b", "c", "d"], 16))
    a.obs["time"] = np.repeat([0, 1], 32)
    a.obs["condition"] = pd.Categorical(np.repeat(["ctrl", "stim"], 32))
    a.obsm["spatial"] = rng.normal(size=(64, 2))
    a.obsm["X_pca"] = rng.normal(size=(64, 4))
    return a


def test_moscot_routes_kwargs_and_retains_problem(data, monkeypatch):
    class Problem:
        def __init__(self, a):
            self.adata = a
            self.solutions = {(0, 1): SimpleNamespace(converged=True)}
        def prepare(self, **kw):
            self.prepare_args = kw
        def solve(self, **kw):
            self.solve_args = kw
    monkeypatch.setattr(ot, "import_optional_module", lambda *a, **k: SimpleNamespace(TemporalProblem=Problem))
    solve = {"epsilon": .1}
    p = ot.moscot_temporal(data, time_key="time", joint_attr="X_pca",
                           prepare_kwargs={"policy": "sequential"}, solve_kwargs=solve)
    assert p.adata is data
    assert p.prepare_args == dict(time_key="time", joint_attr="X_pca", policy="sequential")
    assert p.solve_args == solve == {"epsilon": .1}


def test_moscot_nonconvergence_is_not_success():
    class Problem:
        solutions = {(0, 1): SimpleNamespace(converged=False)}
        def prepare(self, **kw): pass
        def solve(self, **kw): pass
    with pytest.raises(RuntimeError, match="did not converge"):
        ot._solve(Problem(), {}, {}, True)
    assert isinstance(ot._solve(Problem(), {}, {}, False), Problem)


def test_moscot_rejects_ambiguous_inputs_before_import(data, monkeypatch):
    monkeypatch.setattr(ot, "import_optional_module", lambda *a, **k: pytest.fail("Backend should not load"))
    with pytest.raises(ValueError, match="numeric"):
        ot.moscot_temporal(data, time_key="condition")
    with pytest.raises(ValueError, match="named arguments"):
        ot.moscot_temporal(data, time_key="time", prepare_kwargs={"time_key": "sample"})
    with pytest.raises(ValueError, match="sc_attr"):
        ot.moscot_mapping(data, data.copy())


@pytest.mark.integration
def test_real_moscot_temporal(data):
    pytest.importorskip("moscot", minversion="0.5.2")
    p = ot.moscot_temporal(data, time_key="time", joint_attr="X_pca",
                           solve_kwargs={"epsilon": .5, "threshold": 1e-5, "max_iterations": 500})
    matrix = np.asarray(p.solutions[(0, 1)].transport_matrix)
    assert matrix.shape == (32, 32)
    assert np.isfinite(matrix).all() and (matrix >= 0).all()
    np.testing.assert_allclose(matrix.sum(0), np.full(32, 1 / 32), atol=2e-4)
    np.testing.assert_allclose(matrix.sum(1), np.full(32, 1 / 32), atol=2e-4)


@pytest.mark.integration
def test_real_moscot_recovers_known_correspondence():
    pytest.importorskip("moscot", minversion="0.5.2")
    a = ad.AnnData(np.ones((10, 5)))
    a.obs["time"] = np.repeat([0, 1], 5)
    a.obsm["X_known"] = np.vstack([np.eye(5), np.eye(5)])
    p = ot.moscot_temporal(a, time_key="time", joint_attr="X_known",
                           solve_kwargs={"epsilon": .05})
    matrix = np.asarray(p.solutions[(0, 1)].transport_matrix)
    assert np.trace(matrix) > .99
    np.testing.assert_array_equal(matrix.argmax(axis=1), np.arange(5))


@pytest.mark.integration
@pytest.mark.parametrize("alpha,sc_attr", [(0, None), (.5, "X_pca")])
def test_real_moscot_mapping(data, alpha, sc_attr):
    pytest.importorskip("moscot", minversion="0.5.2")
    ref, sp = data[:20].copy(), data[20:32].copy()
    p = ot.moscot_mapping(ref, sp, sc_attr=sc_attr,
                          prepare_kwargs={"joint_attr": {"attr": "X"}},
                          solve_kwargs={"alpha": alpha, "epsilon": .5, "threshold": 1e-5})
    matrix = np.asarray(next(iter(p.solutions.values())).transport_matrix)
    assert matrix.shape == (12, 20)  # spatial rows, reference columns
    np.testing.assert_allclose(matrix.sum(0), np.full(20, 1 / 20), atol=2e-4)
    np.testing.assert_allclose(matrix.sum(1), np.full(12, 1 / 12), atol=2e-4)


@pytest.mark.integration
def test_real_moscot_alignment(data):
    pytest.importorskip("moscot", minversion="0.5.2")
    data.obs["slice"] = pd.Categorical(np.repeat(["left", "right"], 32))
    original = data.obsm["spatial"].copy()
    p = ot.moscot_alignment(data, batch_key="slice", joint_attr="X_pca",
                            solve_kwargs={"alpha": .5, "epsilon": .5, "max_iterations": 100})
    assert len(p.solutions) == 1
    np.testing.assert_array_equal(original, data.obsm["spatial"])
