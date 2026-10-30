"""Boundary tests and separately selected real-backend integration checks."""
import importlib
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from omicverse import _backend_validation as validation


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


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1, .5])
@pytest.mark.parametrize("as_sparse", [False, True])
def test_reject_invalid_counts(data, bad, as_sparse):
    data.layers["counts"][0, 0] = bad
    if as_sparse:
        data.layers["counts"] = sparse.csr_matrix(data.layers["counts"])
    with pytest.raises(ValueError):
        validation.expression(data, "counts", counts=True)


def test_validation_preserves_sparse(data):
    data.layers["counts"] = sparse.csr_matrix(data.X)
    assert validation.expression(data, "counts", counts=True) is data.layers["counts"]


def test_reject_missing_and_duplicate_names(data):
    with pytest.raises(KeyError):
        validation.expression(data, "absent")
    data.obs_names = ["same"] * data.n_obs
    with pytest.raises(ValueError, match="unique"):
        validation.validate_adata(data)


def test_view_requires_explicit_copy(data):
    with pytest.raises(ValueError, match="view"):
        validation.validate_adata(data[:10])


def test_failed_embedding_does_not_overwrite(data):
    old = data.obsm["X_pca"].copy()
    with pytest.raises(ValueError):
        validation.write_embedding(data, "X_pca", np.full((64, 3), np.nan))
    np.testing.assert_array_equal(data.obsm["X_pca"], old)
    with pytest.raises(ValueError, match="already exists"):
        validation.output_key(data, "X_pca")
