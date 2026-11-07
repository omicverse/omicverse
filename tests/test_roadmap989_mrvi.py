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
def test_real_mrvi_training(data):
    pytest.importorskip("scvi", minversion="1.5.2")
    from omicverse.single._mrvi import mrvi
    model = mrvi(data, sample_key="sample", model_kwargs={"n_latent": 3},
                 train_kwargs={"max_epochs": 1, "batch_size": 32, "accelerator": "cpu",
                               "enable_progress_bar": False})
    assert model.is_trained
    assert data.obsm["X_mrvi"].shape[0] == data.n_obs
    assert np.isfinite(data.obsm["X_mrvi"]).all()


def test_mrvi_parameter_routing_and_output(data, monkeypatch):
    module = importlib.import_module("omicverse.single._mrvi")
    calls = {}
    class Model:
        @staticmethod
        def setup_anndata(a, **kwargs): calls["setup"] = kwargs
        def __init__(self, a, **kwargs): calls["model"] = kwargs
        def train(self, **kwargs): calls["train"] = kwargs
        def get_latent_representation(self, **kwargs):
            calls["latent"] = kwargs
            return np.arange(128).reshape(64, 2)
    monkeypatch.setattr(module, "import_optional_module", lambda *a, **k: SimpleNamespace(MRVI=Model))
    module.mrvi(data, sample_key="sample", model_kwargs={"n_latent": 2}, train_kwargs={"max_epochs": 2})
    assert calls["setup"] == dict(sample_key="sample", layer="counts", batch_key=None)
    assert calls["model"] == {"n_latent": 2}
    assert calls["train"] == {"max_epochs": 2}
    assert calls["latent"] == {"give_z": False}
    np.testing.assert_array_equal(data.obsm["X_mrvi"], np.arange(128).reshape(64, 2))


def test_mrvi_requires_real_samples(data):
    from omicverse.single._mrvi import mrvi
    data.obs["sample"] = "one"
    with pytest.raises(ValueError, match="at least 2"):
        mrvi(data, sample_key="sample")
