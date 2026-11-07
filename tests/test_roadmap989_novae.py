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
def test_real_novae_training():
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


def test_novae_does_not_silently_overwrite(data):
    from omicverse.space._novae import novae
    data.obs["novae_domains"] = "existing"
    with pytest.raises(ValueError, match="already exists"):
        novae(data)
