"""Regression tests for the shared spatial pymclustR integration."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import pytest

from omicverse.external._pymclustr import fit_pymclustr


ROOT = Path(__file__).resolve().parents[2]


def _fake_fit(labels):
    return SimpleNamespace(
        classification=np.asarray(labels),
        model_name="EEE",
        G=len(np.unique(labels)),
        loglik=-12.5,
        bic=-31.0,
    )


def _load_source_module(monkeypatch, module_name, relative_path):
    if "torch_geometric.data" not in sys.modules:
        torch_geometric = types.ModuleType("torch_geometric")
        torch_geometric_data = types.ModuleType("torch_geometric.data")
        torch_geometric_data.Data = object
        torch_geometric.data = torch_geometric_data
        monkeypatch.setitem(sys.modules, "torch_geometric", torch_geometric)
        monkeypatch.setitem(
            sys.modules, "torch_geometric.data", torch_geometric_data
        )

    if "omicverse.external.STAligner.mnn_utils" not in sys.modules:
        mnn_utils = types.ModuleType("omicverse.external.STAligner.mnn_utils")
        mnn_utils.create_dictionary_mnn = lambda *args, **kwargs: {}
        monkeypatch.setitem(
            sys.modules, "omicverse.external.STAligner.mnn_utils", mnn_utils
        )

    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_adapter_matches_direct_pymclustr_and_preserves_one_based_labels():
    from mclust_py import Mclust

    data = np.random.default_rng(42).normal(size=(80, 4))
    labels, fit = fit_pymclustr(
        data, n_components=3, model_names="EEE", random_state=17
    )
    direct = Mclust(data, G=[3], model_names=["EEE"])

    np.testing.assert_array_equal(labels, direct.classification)
    assert labels.min() == 1
    assert labels.max() == 3
    assert fit.model_name == direct.model_name
    assert fit.G == direct.G
    assert fit.loglik == direct.loglik
    assert fit.bic == direct.bic


def test_adapter_restores_numpy_random_state(monkeypatch):
    class FakeMclust:
        def __init__(self, data, G, model_names, **kwargs):
            np.random.random(5)
            self.classification = np.ones(len(data), dtype=int)
            self.model_name = model_names[0]
            self.G = G[0]

    monkeypatch.setitem(
        sys.modules, "mclust_py", SimpleNamespace(Mclust=FakeMclust)
    )
    data = np.arange(12, dtype=float).reshape(6, 2)

    np.random.seed(123)
    expected = np.random.random(4)
    np.random.seed(123)
    fit_pymclustr(data, 1, random_state=999)
    observed = np.random.random(4)

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize(
    ("data", "n_components", "error", "message"),
    [
        (np.ones(5), 2, ValueError, "two-dimensional"),
        (np.ones((2, 2)), 3, ValueError, "cannot exceed"),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), 2, ValueError, "finite"),
        (np.ones((4, 2)), 0, ValueError, "positive integer"),
        (np.ones((4, 2)), 1.5, TypeError, "positive integer"),
    ],
)
def test_adapter_validates_inputs(data, n_components, error, message):
    with pytest.raises(error, match=message):
        fit_pymclustr(data, n_components)


def test_adapter_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "mclust_py", None)
    with pytest.raises(ImportError, match=r"pip install pymclustR>=0\.2\.1"):
        fit_pymclustr(np.ones((4, 2)), 2)


@pytest.mark.parametrize(
    ("module_name", "relative_path", "function_name", "embedding_key"),
    [
        (
            "omicverse.external.GraphST._utils_pymclustr_test",
            "omicverse/external/GraphST/utils.py",
            "mclust_R",
            "emb_pca",
        ),
        (
            "omicverse.external.STAGATE_pyG._utils_pymclustr_test",
            "omicverse/external/STAGATE_pyG/utils.py",
            "mclust_R",
            "STAGATE",
        ),
        (
            "omicverse.external.STAligner._utils_pymclustr_test",
            "omicverse/external/STAligner/ST_utils.py",
            "mclust_R",
            "STAGATE",
        ),
    ],
)
def test_anndata_wrappers_preserve_contract(
    monkeypatch, module_name, relative_path, function_name, embedding_key
):
    module = _load_source_module(monkeypatch, module_name, relative_path)
    labels = np.array([1, 1, 2, 2, 3, 3])
    calls = {}

    def fake_adapter(data, **kwargs):
        calls.update(kwargs)
        np.testing.assert_array_equal(data, adata.obsm[embedding_key])
        return labels, _fake_fit(labels)

    monkeypatch.setattr(module, "fit_pymclustr", fake_adapter)
    adata = anndata.AnnData(np.ones((6, 2)))
    adata.obsm[embedding_key] = np.arange(18, dtype=float).reshape(6, 3)

    result = getattr(module, function_name)(
        adata,
        num_cluster=3,
        modelNames="EEE",
        used_obsm=embedding_key,
        random_seed=27,
    )

    assert result is adata
    np.testing.assert_array_equal(adata.obs["mclust"].astype(int), labels)
    assert isinstance(adata.obs["mclust"].dtype, pd.CategoricalDtype)
    assert calls == {
        "n_components": 3,
        "model_names": "EEE",
        "random_state": 27,
    }


def test_prost_wrapper_returns_one_based_label_array(monkeypatch):
    module = _load_source_module(
        monkeypatch,
        "omicverse.external.PROST._utils_pymclustr_test",
        "omicverse/external/PROST/utils.py",
    )
    labels = np.array([1, 1, 2, 3])
    monkeypatch.setattr(
        module,
        "fit_pymclustr",
        lambda data, **kwargs: (labels, _fake_fit(labels)),
    )

    result = module.mclust(
        np.ones((4, 2)), num_cluster=3, modelNames="EEE", random_seed=818
    )

    np.testing.assert_array_equal(result, labels)


def test_binary_wrapper_respects_custom_add_key(monkeypatch):
    module = _load_source_module(
        monkeypatch,
        "omicverse.external.BINARY._utils_pymclustr_test",
        "omicverse/external/BINARY/utils.py",
    )
    labels = np.array([1, 1, 2, 2])
    monkeypatch.setattr(
        module,
        "fit_pymclustr",
        lambda data, **kwargs: (labels, _fake_fit(labels)),
    )
    adata = anndata.AnnData(np.ones((4, 2)))
    adata.obsm["BINARY"] = np.ones((4, 3))

    result = module.mclust_R(adata, 2, add_key="domain")

    assert result is adata
    np.testing.assert_array_equal(adata.obs["domain"].astype(int), labels)
    assert isinstance(adata.obs["domain"].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize(
    ("module_name", "relative_path", "function_name", "embedding_key"),
    [
        (
            "omicverse.external.GraphST._real_pymclustr_test",
            "omicverse/external/GraphST/utils.py",
            "mclust_R",
            "emb_pca",
        ),
        (
            "omicverse.external.STAGATE_pyG._real_pymclustr_test",
            "omicverse/external/STAGATE_pyG/utils.py",
            "mclust_R",
            "STAGATE",
        ),
        (
            "omicverse.external.STAligner._real_pymclustr_test",
            "omicverse/external/STAligner/ST_utils.py",
            "mclust_R",
            "STAGATE",
        ),
        (
            "omicverse.external.BINARY._real_pymclustr_test",
            "omicverse/external/BINARY/utils.py",
            "mclust_R",
            "BINARY",
        ),
    ],
)
def test_anndata_wrappers_match_real_adapter(
    monkeypatch, module_name, relative_path, function_name, embedding_key
):
    module = _load_source_module(monkeypatch, module_name, relative_path)
    embedding = np.random.default_rng(18).normal(size=(30, 3))
    expected, _ = fit_pymclustr(
        embedding, n_components=3, model_names="EEE", random_state=29
    )
    adata = anndata.AnnData(np.ones((30, 2)))
    adata.obsm[embedding_key] = embedding

    kwargs = {
        "num_cluster": 3,
        "modelNames": "EEE",
        "used_obsm": embedding_key,
        "random_seed": 29,
    }
    if "BINARY" in module_name:
        kwargs["add_key"] = "domain"
    result = getattr(module, function_name)(adata, **kwargs)
    output_key = "domain" if "BINARY" in module_name else "mclust"

    assert result is adata
    np.testing.assert_array_equal(
        adata.obs[output_key].astype(int).to_numpy(), expected
    )


def test_prost_wrapper_matches_real_adapter(monkeypatch):
    module = _load_source_module(
        monkeypatch,
        "omicverse.external.PROST._real_pymclustr_test",
        "omicverse/external/PROST/utils.py",
    )
    data = np.random.default_rng(21).normal(size=(30, 3))
    expected, _ = fit_pymclustr(
        data, n_components=3, model_names="EEE", random_state=818
    )

    observed = module.mclust(data, num_cluster=3)

    np.testing.assert_array_equal(observed, expected)


def test_utils_cluster_supports_output_key_aliases_without_leaking(monkeypatch):
    from omicverse.utils import _cluster

    calls = []
    labels = np.array([1, 1, 2, 2])
    fit = _fake_fit(labels)

    def fake_adapter(data, **kwargs):
        calls.append(kwargs)
        return labels, fit

    monkeypatch.setattr(_cluster, "fit_pymclustr", fake_adapter)
    adata = anndata.AnnData(np.ones((4, 2)))
    adata.obsm["X_pca"] = np.ones((4, 3))

    assert (
        _cluster.cluster(
            adata,
            method="pymclustR",
            n_components=2,
            key_added="domain",
            modelNames="EEE",
        )
        is None
    )
    np.testing.assert_array_equal(adata.obs["domain"].astype(int), labels)
    assert calls == [
        {
            "n_components": 2,
            "model_names": "EEE",
            "random_state": 1024,
        }
    ]
    assert adata.uns["pymclustR"]["X_pca"] == {
        "modelName": "EEE",
        "G": 2,
        "loglik": -12.5,
        "bic": -31.0,
    }

    second = anndata.AnnData(np.ones((4, 2)))
    second.obsm["X_pca"] = np.ones((4, 3))
    _cluster.cluster(
        second,
        method="pymclustR",
        n_components=2,
        add_key="legacy_domain",
    )
    assert "legacy_domain" in second.obs

    default = anndata.AnnData(np.ones((4, 2)))
    default.obsm["X_pca"] = np.ones((4, 3))
    _cluster.cluster(default, method="pymclustR", n_components=2)
    assert "pymclustR" in default.obs


def test_utils_cluster_rejects_conflicting_output_keys():
    from omicverse.utils import _cluster

    adata = anndata.AnnData(np.ones((4, 2)))
    adata.obsm["X_pca"] = np.ones((4, 3))
    with pytest.raises(ValueError, match="must match"):
        _cluster.cluster(
            adata,
            method="pymclustR",
            n_components=2,
            key_added="one",
            add_key="two",
        )

    with pytest.raises(ValueError, match="non-empty string"):
        _cluster.cluster(
            adata,
            method="pymclustR",
            n_components=2,
            key_added="",
        )


def test_target_wrappers_no_longer_execute_rpy2_mclust():
    paths = [
        ROOT / "omicverse/external/PROST/utils.py",
        ROOT / "omicverse/external/GraphST/utils.py",
        ROOT / "omicverse/external/STAGATE_pyG/utils.py",
        ROOT / "omicverse/external/STAligner/ST_utils.py",
        ROOT / "omicverse/external/BINARY/utils.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "import rpy2" not in source
        assert 'library("mclust")' not in source


def test_spatial_wrapper_signatures_remain_compatible(monkeypatch):
    expected = {
        "PROST": "(data, num_cluster, modelNames='EEE', random_seed=818)",
        "GraphST": (
            "(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', "
            "random_seed=2020)"
        ),
        "STAGATE": (
            "(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', "
            "random_seed=2020)"
        ),
        "STAligner": (
            "(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', "
            "random_seed=666)"
        ),
        "BINARY": (
            "(adata, num_cluster, add_key='mclust', modelNames='EEE', "
            "used_obsm='BINARY', random_seed=2020)"
        ),
    }
    modules = {
        "PROST": (
            "omicverse.external.PROST._signature_test",
            "omicverse/external/PROST/utils.py",
            "mclust",
        ),
        "GraphST": (
            "omicverse.external.GraphST._signature_test",
            "omicverse/external/GraphST/utils.py",
            "mclust_R",
        ),
        "STAGATE": (
            "omicverse.external.STAGATE_pyG._signature_test",
            "omicverse/external/STAGATE_pyG/utils.py",
            "mclust_R",
        ),
        "STAligner": (
            "omicverse.external.STAligner._signature_test",
            "omicverse/external/STAligner/ST_utils.py",
            "mclust_R",
        ),
        "BINARY": (
            "omicverse.external.BINARY._signature_test",
            "omicverse/external/BINARY/utils.py",
            "mclust_R",
        ),
    }
    for name, (module_name, path, function_name) in modules.items():
        module = _load_source_module(monkeypatch, module_name, path)
        assert str(inspect.signature(getattr(module, function_name))) == expected[name]
