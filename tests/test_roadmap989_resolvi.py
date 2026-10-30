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
def test_real_resolvi_training():
    pytest.importorskip("scvi", minversion="1.5.2")
    from omicverse.space._resolvi import resolvi
    rng = np.random.default_rng(51)
    a = ad.AnnData(rng.poisson(4, (80, 60)).astype(np.float32))
    a.layers["counts"] = a.X.copy()
    a.obsm["spatial"] = rng.normal(size=(80, 2))
    counts = a.layers["counts"].copy()
    model = resolvi(a, model_kwargs={"n_latent": 3, "n_hidden": 16, "mixture_k": 4},
                    train_kwargs={"max_epochs": 1, "batch_size": 40, "accelerator": "cpu",
                                  "enable_progress_bar": False})
    assert model.is_trained
    assert a.obsm["X_resolvi"].shape == (80, 3)
    np.testing.assert_array_equal(counts, a.layers["counts"])


def test_resolvi_validates_before_import(data):
    from omicverse.space._resolvi import resolvi
    with pytest.raises(ValueError, match="50 genes"):
        resolvi(data)
    with pytest.raises(ValueError, match="n_neighbors"):
        resolvi(data, n_neighbors=2)
