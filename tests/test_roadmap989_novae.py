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
def test_real_novae_training_and_local_checkpoint_inference(tmp_path, monkeypatch):
    pytest.importorskip("novae", minversion="1.1.2")
    import torch
    torch.manual_seed(42)
    from omicverse.space._novae import novae
    rng = np.random.default_rng(4)
    a = ad.AnnData(rng.poisson(5, (100, 60)).astype(np.float32))
    a.var_names = [f"GENE{i}" for i in range(60)]
    a.obsm["spatial"] = np.column_stack(np.unravel_index(np.arange(100), (10, 10))).astype(float)
    before = a.X.copy()
    model = novae(a, model_kwargs={"embedding_size": 8, "output_size": 8, "hidden_size": 8,
                                  "heads": 2, "num_layers": 2, "num_prototypes": 4,
                                  "batch_size": 16},
                   train_kwargs={"max_epochs": 1, "num_workers": 0, "accelerator": "cpu",
                                 "enable_progress_bar": False},
                   representation_kwargs={"num_workers": 0, "accelerator": "cpu"},
                   domains_kwargs={"level": 1})
    assert model is not None
    assert a.obsm["X_novae"].shape == (100, 8)
    assert a.obs["novae_domains"].notna().any()
    np.testing.assert_array_equal(a.X, before)

    # Exercise the actual pretrained branch without downloading a checkpoint.
    # Two overlapping coordinate grids must remain separate physical slides.
    checkpoint = tmp_path / "novae"
    model.save_pretrained(checkpoint)
    import novae as backend
    restored = backend.Novae.from_pretrained(checkpoint)
    query = ad.AnnData(rng.poisson(5, (200, 40)).astype(np.float32))
    query.var_names = [f"GENE{i}" for i in range(10, 50)]
    query.obs_names = [f"query_{i}" for i in range(200)]
    query.obs["slide"] = pd.Categorical(np.repeat(["left", "right"], 100))
    query.obsm["spatial"] = np.tile(a.obsm["spatial"], (2, 1))
    query.obsp["spatial_connectivities"] = sparse.eye(200, format="csr")
    query_before = query.copy()
    native_neighbors = backend.spatial_neighbors
    def check_neighbors(work, **kwargs):
        native_neighbors(work, **kwargs)
        rows, cols = work.obsp["spatial_connectivities"].nonzero()
        slides = work.obs["slide"].to_numpy()
        assert len(rows) > 0
        assert np.all(slides[rows] == slides[cols])
    monkeypatch.setattr(backend, "spatial_neighbors", check_neighbors)
    result = novae(query, model=restored, slide_key="slide",
                   representation_kwargs={"num_workers": 0, "accelerator": "cpu"},
                   domains_kwargs={"level": 1})
    assert result is restored
    assert query.obsm["X_novae"].shape == (200, 8)
    assert np.isfinite(query.obsm["X_novae"]).all()
    assert query.obs["novae_domains"].notna().all()
    assert query.obs_names.equals(query_before.obs_names)
    assert query.var_names.equals(query_before.var_names)
    np.testing.assert_array_equal(query.X, query_before.X)
    np.testing.assert_array_equal(query.obsm["spatial"], query_before.obsm["spatial"])
    assert (query.obsp["spatial_connectivities"] != query_before.obsp["spatial_connectivities"]).nnz == 0


def test_novae_does_not_silently_overwrite(data):
    from omicverse.space._novae import novae
    data.obs["novae_domains"] = "existing"
    with pytest.raises(ValueError, match="already exists"):
        novae(data)
