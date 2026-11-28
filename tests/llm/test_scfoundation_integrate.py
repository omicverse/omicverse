"""ScFoundationModel.integrate: embeddings + post-hoc batch correction (no checkpoint needed)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from anndata import AnnData


@pytest.fixture
def model_and_data(monkeypatch):
    from omicverse.llm.scfoundation_model import ScFoundationModel
    model = ScFoundationModel(device="cpu")
    rng = np.random.default_rng(0)
    n = 60
    batch = np.repeat(["a", "b", "c"], n // 3)
    emb = rng.normal(size=(n, 8)) + (batch == "b")[:, None] * 5.0
    adata = AnnData(X=np.zeros((n, 3), np.float32), obs=pd.DataFrame({"batch": batch}))
    monkeypatch.setattr(model, "_predict_embedding",
                        lambda ad, **kw: {"embeddings": emb.copy()})
    monkeypatch.setattr(model, "preprocess", lambda ad, **kw: ad)  # gene selection needs the checkpoint
    model.is_loaded = True
    return model, adata, emb, batch


def test_integrate_is_a_method_of_the_model():
    from omicverse.llm.scfoundation_model import LinearProbingClassifier, ScFoundationModel
    assert hasattr(ScFoundationModel, "integrate")
    assert not hasattr(LinearProbingClassifier, "integrate")


def test_integrate_center_scale_removes_batch_shift(model_and_data):
    model, adata, emb, batch = model_and_data
    res = model.integrate(adata, batch_key="batch", correction_method="center_scale")
    z = res["embeddings"]
    assert z.shape == emb.shape and res["n_batches"] == 3
    np.testing.assert_allclose(res["original_embeddings"], emb)
    means = np.stack([z[batch == b].mean(0) for b in "abc"])
    assert np.abs(means - means.mean(0)).max() < 1e-6   # batch means aligned


def test_integrate_matches_predict_task(model_and_data):
    model, adata, _, _ = model_and_data
    a = model.integrate(adata, correction_method="none")
    b = model.predict(adata, task="integration", correction_method="none")
    np.testing.assert_array_equal(a["embeddings"], b["embeddings"])


def test_integrate_errors(model_and_data):
    model, adata, _, _ = model_and_data
    with pytest.raises(ValueError, match="Unknown correction method"):
        model.integrate(adata, correction_method="nope")
    with pytest.raises(ValueError, match="not found"):
        model.integrate(adata, batch_key="missing", correction_method="none")
    model.is_loaded = False
    with pytest.raises(ValueError, match="not loaded"):
        model.integrate(adata)
