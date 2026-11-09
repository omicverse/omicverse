"""Boundary tests and separately selected real-backend integration checks."""
import importlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest


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
    from scvi.external import RESOLVI
    from omicverse.space import resolvi
    rng = np.random.default_rng(51)
    a = ad.AnnData(rng.poisson(4, (30, 60)).astype(np.float32))
    a.layers["counts"] = a.X.copy()
    a.obsm["spatial"] = rng.normal(size=(a.n_obs, 2))
    a.obs["slide"] = pd.Categorical(np.repeat(["a", "b"], 15))
    counts = a.layers["counts"].copy()
    fresh = a.copy()
    prepare = {"spatial_rep": "spatial", "n_neighbors": 10}
    RESOLVI.setup_anndata(a, layer="counts", batch_key="slide", prepare_data=True, prepare_data_kwargs=prepare)
    a.obsm["spatial"] = rng.normal(size=(a.n_obs, 2))
    fresh.obsm["spatial"] = a.obsm["spatial"].copy()
    RESOLVI.setup_anndata(fresh, layer="counts", batch_key="slide", prepare_data=True, prepare_data_kwargs=prepare)
    model = resolvi(a, batch_key="slide", model_kwargs={"n_latent": 3, "n_hidden": 16, "mixture_k": 4},
                    train_kwargs={"max_epochs": 1, "batch_size": 15, "accelerator": "cpu",
                                  "enable_progress_bar": False})
    assert model.is_trained
    assert a.obsm["X_resolvi"].shape == (30, 3)
    np.testing.assert_array_equal(counts, a.layers["counts"])
    np.testing.assert_array_equal(a.obsm["index_neighbor"], fresh.obsm["index_neighbor"])
    np.testing.assert_allclose(a.obsm["distance_neighbor"], fresh.obsm["distance_neighbor"])


def test_resolvi_validates_before_import(data):
    from omicverse.space._resolvi import resolvi
    with pytest.raises(ValueError, match="50 genes"):
        resolvi(data)
    with pytest.raises(ValueError, match="n_neighbors"):
        resolvi(data, n_neighbors=2)


@pytest.mark.parametrize("key", ["spatial", "X_spatial", "index_neighbor", "distance_neighbor"])
def test_resolvi_rejects_output_key_that_overwrites_spatial_data(key, monkeypatch):
    module = importlib.import_module("omicverse.space._resolvi")
    monkeypatch.setattr(module, "import_optional_module", lambda *a, **k: pytest.fail("Backend loaded"))
    a = ad.AnnData(np.ones((20, 60)))
    a.layers["counts"] = a.X.copy()
    a.obsm["spatial"] = np.zeros((20, 2))
    with pytest.raises(ValueError, match="spatial"):
        module.resolvi(a, key_added=key, overwrite=True)
