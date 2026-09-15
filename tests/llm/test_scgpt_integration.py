"""Integration inference contracts using the real scGPT transformer."""
import copy
import importlib.util

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("datasets") is None:
    pytest.skip("scGPT requires datasets", allow_module_level=True)

pytest.importorskip("torch")
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix
from omicverse.llm.scgpt_model import ScGPTModel, tokenize_and_pad_batch
from omicverse.llm.scgpt.model import TransformerModel


@pytest.fixture
def model():
    wrapper = ScGPTModel(device="cpu", seed=42)
    wrapper.config = copy.deepcopy(wrapper.default_config)
    wrapper.config.max_seq_len = 16
    wrapper.vocab = {s: i for i, s in enumerate(
        ["<pad>", "<cls>", "<eoc>"] + [f"g{i}" for i in range(8)])}
    wrapper.model = TransformerModel(
        ntoken=len(wrapper.vocab), d_model=8, nhead=2, d_hid=16,
        nlayers=1, dropout=0, vocab=wrapper.vocab, pad_token="<pad>",
        pad_value=-2, use_batch_labels=True, num_batch_labels=2,
        domain_spec_batchnorm=True, use_fast_transformer=False,
    )
    # Deliberately distinct learned domain statistics expose mixed-domain calls.
    with torch.no_grad():
        wrapper.model.dsbn.bns[1].running_mean.copy_(torch.linspace(-3, 3, 8))
        wrapper.model.dsbn.bns[1].running_var.copy_(torch.arange(1, 9))
    wrapper.is_loaded = True
    wrapper._integration_trained = True
    return wrapper


@pytest.fixture
def data():
    adata = AnnData(np.array([
        [0, 2, 3, 4, 5, 6, 7, 8], [8, 7, 0, 5, 4, 3, 2, 1],
        [1, 3, 2, 6, 4, 5, 8, 7], [3, 2, 1, 4, 8, 7, 6, 5],
        [4, 1, 2, 3, 6, 5, 8, 7]], dtype=np.float32))
    adata.var_names = [f"g{i}" for i in range(8)]
    adata.obs["batch"] = pd.Categorical(["a", "b", "a", "b", "b"])
    return adata


@pytest.mark.parametrize("sparse", [False, True])
def test_public_entries_preprocess_without_mutating(model, data, sparse):
    if sparse:
        data.X = csr_matrix(data.X)
    original = data.copy()
    np.random.seed(7)
    actual = model.integrate(data)["embeddings"]
    np.random.seed(7)
    expected = model.predict(data, task="integration")["embeddings"]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert "X_binned" not in data.layers
    np.testing.assert_array_equal(data.X.toarray() if sparse else data.X,
                                  original.X.toarray() if sparse else original.X)


def test_unmasked_cls_reference_and_row_order(model, data):
    data.layers["X_binned"] = data.X.copy()
    tokens = tokenize_and_pad_batch(
        data.X, np.arange(3, 11), max_len=16, vocab=model.vocab,
        pad_token="<pad>", pad_value=-2, append_cls=True, include_zero_gene=True,
    )
    model.model.eval()
    # Single-cell evaluation is independent of grouping and minibatch boundaries.
    with torch.no_grad():
        expected = model.model.encode_batch(
            tokens["genes"], tokens["values"].float(),
            tokens["genes"].eq(model.vocab["<pad>"]), batch_size=1,
            batch_labels=torch.tensor(data.obs.batch.cat.codes.values.astype(int)),
            time_step=0, return_np=True,
        )
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    for batch_size in (1, 2, 32):
        result = model.integrate(data, batch_size=batch_size)
        np.testing.assert_allclose(result["embeddings"], expected, atol=1e-6)
        np.testing.assert_array_equal(result["batch_labels"], [0, 1, 0, 1, 1])
        assert result["integration_stats"]["batch_distribution"] == [2, 3]
        np.testing.assert_array_equal(result["embeddings"], result["integrated_embeddings"])
    order = [3, 0, 4, 1, 2]
    np.testing.assert_allclose(model.integrate(data[order].copy())["embeddings"],
                               expected[order], atol=1e-6)


def test_non_domain_model(model, data):
    model.model = TransformerModel(
        ntoken=len(model.vocab), d_model=8, nhead=2, d_hid=16, nlayers=1,
        dropout=0, vocab=model.vocab, pad_token="<pad>", pad_value=-2,
        use_fast_transformer=False,
    )
    data.layers["X_binned"] = data.X.copy()
    first = model.integrate(data)["embeddings"]
    np.testing.assert_allclose(model.integrate(data)["embeddings"], first, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1, atol=1e-6)


def test_missing_batch_label(model, data):
    data.layers["X_binned"] = data.X.copy()
    data.obs.loc[data.obs_names[0], "batch"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        model.integrate(data)


def test_subset_with_training_categories(model, data):
    data.layers["X_binned"] = data.X.copy()
    expected = model.integrate(data)["embeddings"][[1, 3, 4]]
    subset = data[[1, 3, 4]].copy()
    subset.obs["batch"] = subset.obs["batch"].cat.set_categories(data.obs.batch.cat.categories)
    np.testing.assert_allclose(model.integrate(subset)["embeddings"], expected, atol=1e-6)


def test_raw_subset_retains_categories_after_filtering(model, data):
    subset = data[[1, 3, 4]].copy()
    subset.obs["batch"] = subset.obs["batch"].cat.set_categories(data.obs.batch.cat.categories)
    result = model.integrate(subset, filter_gene_by_counts=1)
    np.testing.assert_array_equal(result["batch_labels"], [1, 1, 1])


@pytest.mark.parametrize("entry", ["integrate", "predict"])
@pytest.mark.parametrize("binned", [False, True])
def test_empty_integration_input(model, data, entry, binned):
    if binned:
        data.layers["X_binned"] = data.X.copy()
    empty = data[:0].copy()
    kwargs = {"task": "integration"} if entry == "predict" else {}
    with pytest.raises(ValueError, match="at least one cell"):
        getattr(model, entry)(empty, **kwargs)


@pytest.mark.parametrize("entry", ["integrate", "predict"])
@pytest.mark.parametrize("style", ["avg-pool", "w-pool"])
def test_integration_requires_cls(model, data, entry, style):
    model.model.cell_emb_style = style
    kwargs = {"task": "integration"} if entry == "predict" else {}
    with pytest.raises(ValueError, match="cell_emb_style='cls'"):
        getattr(model, entry)(data, **kwargs)
