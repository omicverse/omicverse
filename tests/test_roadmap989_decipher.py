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


@pytest.mark.integration
@pytest.mark.parametrize("method", ["decipher"])
def test_real_scvi_embeddings(data, method):
    pytest.importorskip("scvi", minversion="1.5.2")
    from omicverse.single import _decipher as _scvi_embeddings
    kwargs = {}
    if method == "sysvi":
        data.X = np.log1p(data.X)
        kwargs = {"batch_key": "sample", "model_kwargs": {"n_latent": 3}}
    elif method == "drvi":
        kwargs = {"batch_key": "sample", "model_kwargs": {"n_latent": 4}}
    before = data.layers["counts"].copy()
    model = getattr(_scvi_embeddings, method)(
        data, **kwargs, train_kwargs={"max_epochs": 1, "batch_size": 32,
                                    "accelerator": "cpu", "enable_progress_bar": False})
    assert model.is_trained
    embedding = data.obsm[f"X_{method}"]
    assert embedding.shape[0] == data.n_obs and np.isfinite(embedding).all()
    np.testing.assert_array_equal(before, data.layers["counts"])
