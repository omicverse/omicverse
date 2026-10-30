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
def test_real_lemur_predictions(data):
    pytest.importorskip("pylemur", minversion="0.3.1")
    from omicverse.single._lemur import lemur
    data.X = np.log1p(data.X)
    before = data.X.copy()
    model = lemur(data, design="~ condition", n_embedding=3, fit_kwargs={"verbose": False})
    assert data.obsm["X_lemur"].shape == (64, 3)
    pred = model.predict(new_condition=model.cond(condition="stim"))
    assert pred.shape == data.shape and np.isfinite(pred).all()
    np.testing.assert_array_equal(data.X, before)
    np.testing.assert_allclose(model.embedding, data.obsm["X_lemur"])


def test_lemur_cannot_override_selected_expression(data):
    from omicverse.single._lemur import lemur
    with pytest.raises(ValueError, match="named arguments"):
        lemur(data, design="~ condition", model_kwargs={"layer": "counts"})


@pytest.mark.integration
def test_lemur_recovers_known_condition_effect():
    pytest.importorskip("pylemur", minversion="0.3.1")
    from omicverse.single._lemur import lemur
    rng = np.random.default_rng(42)
    baseline = rng.normal(size=(40, 3)) @ rng.normal(size=(3, 20))
    effect = np.r_[np.full(5, 3.), np.zeros(15)]
    a = ad.AnnData(np.vstack([baseline, baseline + effect]))
    a.obs["condition"] = pd.Categorical(np.repeat(["ctrl", "stim"], 40))
    model = lemur(a, design="~ condition", n_embedding=3, fit_kwargs={"verbose": False})
    contrast = model.predict(new_condition=model.cond(condition="stim")) - model.predict(
        new_condition=model.cond(condition="ctrl"))
    np.testing.assert_allclose(contrast.mean(0), effect, atol=.05)
