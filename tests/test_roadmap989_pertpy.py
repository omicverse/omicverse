"""Boundary tests and separately selected real-backend integration checks."""
import importlib
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from omicverse import _backend_validation as validation


from importlib.metadata import version
from packaging.version import Version

def _require_pertpy():
    pytest.importorskip("pertpy")
    if Version(version("pertpy")) < Version("1.4.0"):
        pytest.skip("pertpy >= 1.4.0 is required")

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


@pytest.mark.integration
def test_real_pertpy_guide_assignment():
    _require_pertpy()
    from omicverse.single import assign_guides
    a = ad.AnnData(np.array([[0, 0], [8, 1], [1, 9], [2, 3]], dtype=float))
    a.var_names = ["guide_a", "guide_b"]
    result = assign_guides(a, assignment_threshold=5)
    assert result.tolist() == ["Negative", "guide_a", "guide_b", "Negative"]


@pytest.mark.integration
def test_real_pertpy_distance_matches_definition(data):
    _require_pertpy()
    from omicverse.single import perturbation_distance
    from scipy.spatial.distance import cdist
    x, y = data.obsm["X_pca"][:32], data.obsm["X_pca"][32:]
    expected = 2 * cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean()
    result = perturbation_distance(data, groupby="condition",
                                   pairwise_kwargs={"show_progressbar": False, "n_jobs": 1})
    assert result.loc["ctrl", "stim"] == pytest.approx(expected, abs=1e-6)
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(np.diag(result), 0, atol=1e-6)


@pytest.mark.integration
def test_real_pertpy_mixscape(data):
    _require_pertpy()
    from omicverse.single import mixscape
    data.X = np.log1p(data.X)
    data.X[32:, :10] += 3
    data.obs["replicate"] = "one"
    original = data.X.copy()
    result = mixscape(data, pert_key="condition", control="ctrl", split_by="replicate",
                       signature_kwargs={"ref_selection_mode": "split_by"},
                       mixscape_kwargs={"min_de_genes": 2, "iter_num": 2, "test_method": "t-test"})
    assert "mixscape_class_global" in result.obs
    assert result.layers["X_pert"].shape == data.shape
    np.testing.assert_array_equal(data.X, original)
    assert "X_pert" not in data.layers


def test_mixscape_rejects_splits_without_controls(data):
    from omicverse.single import mixscape
    with pytest.raises(ValueError, match="Every split"):
        mixscape(data, pert_key="condition", control="ctrl", split_by="sample")
