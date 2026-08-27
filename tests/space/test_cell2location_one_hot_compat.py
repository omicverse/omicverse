from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional

pyro = pytest.importorskip("pyro")


CELL2LOCATION_ROOT = (
    Path(__file__).parents[2] / "omicverse" / "external" / "space" / "cell2location"
)


def _install_scvi_14_surface(monkeypatch):
    """Expose the one-hot API shape used by scvi-tools 1.4.x."""

    class RegistryKeys:
        X_KEY = "X"
        BATCH_KEY = "batch"

    scvi = types.ModuleType("scvi")
    scvi.REGISTRY_KEYS = RegistryKeys
    scvi_nn = types.ModuleType("scvi.nn")
    scvi_nn.__path__ = []
    scvi_nn.one_hot = functional.one_hot
    monkeypatch.setitem(sys.modules, "scvi", scvi)
    monkeypatch.setitem(sys.modules, "scvi.nn", scvi_nn)


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, CELL2LOCATION_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_spatial_model_handles_column_batch_indices_from_scvi_14(monkeypatch):
    _install_scvi_14_surface(monkeypatch)
    module = _load_module("cell2location_spatial_module", "models/_cell2location_module.py")
    model_class = (
        module.LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel
    )
    model = model_class(
        n_obs=4,
        n_vars=3,
        n_factors=2,
        n_batch=2,
        cell_state_mat=np.ones((3, 2), dtype=np.float32),
        n_groups=2,
    )

    x_data = torch.ones((4, 3), dtype=torch.float32)
    batch_index = torch.tensor([[0], [0], [1], [1]], dtype=torch.long)
    pyro.clear_param_store()
    pyro.enable_validation(True)

    trace = pyro.poutine.trace(model).get_trace(x_data, torch.arange(4), batch_index)
    trace.compute_log_prob()

    assert trace.nodes["data_target"]["log_prob"].shape == (4, 3)


def test_vendored_fc_layers_support_scvi_14_without_private_utils(monkeypatch):
    _install_scvi_14_surface(monkeypatch)
    module = _load_module("cell2location_fc_layers", "nn/fclayers.py")
    layers = module.FCLayers(
        n_in=2,
        n_out=3,
        n_cat_list=[2],
        dropout_rate=0,
        use_batch_norm=False,
        use_activation=False,
    )

    result = layers(
        torch.ones((4, 2), dtype=torch.float32),
        torch.tensor([[0], [0], [1], [1]], dtype=torch.long),
    )

    assert result.shape == (4, 3)
    assert result.dtype == torch.float32


def test_reference_model_handles_column_categorical_indices(monkeypatch):
    _install_scvi_14_surface(monkeypatch)
    module = _load_module(
        "cell2location_reference_module", "models/reference/_reference_module.py"
    )
    model = module.RegressionBackgroundDetectionTechPyroModel(
        n_obs=4,
        n_vars=3,
        n_factors=2,
        n_batch=2,
        n_extra_categoricals=[2],
    )

    x_data = torch.ones((4, 3), dtype=torch.float32)
    batch_index = torch.tensor([[0], [0], [1], [1]], dtype=torch.long)
    label_index = torch.tensor([[0], [1], [0], [1]], dtype=torch.long)
    extra_categoricals = torch.tensor([[0], [1], [1], [0]], dtype=torch.long)
    pyro.clear_param_store()

    trace = pyro.poutine.trace(model).get_trace(
        x_data,
        torch.arange(4),
        batch_index,
        label_index,
        extra_categoricals,
    )
    trace.compute_log_prob()

    assert trace.nodes["data_target"]["log_prob"].shape == (4, 3)
