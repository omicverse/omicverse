"""CAMEX (cross-species cell-type annotation & integration via multi-species
expression graphs) is vendored under ``omicverse.external.camex`` because its
upstream pins conflicting deps (numpy<=1.24.4 / scanpy<=1.9.3 / pandas==1.5.3 /
numba==0.56.4 / harmonypy==0.0.9) and is not on PyPI.

These tests check that the vendored package imports and exposes a callable
``run_camex`` entry point under the current environment (numpy 2.x / new
scanpy). The heavy end-to-end GNN training run is validated separately, not in
CI. Mirrors ``tests/single/test_cross_species_backends.py``.
"""
import importlib

import pytest


def test_camex_vendored_imports():
    # CAMEX builds a DGL heterograph and trains a torch GNN
    pytest.importorskip("torch")
    pytest.importorskip("dgl")

    # importing the package itself must stay light (lazy heavy submodules)
    m = importlib.import_module("omicverse.external.camex")
    run = importlib.import_module("omicverse.external.camex._run")
    assert callable(run.run_camex)
    # lazy re-export from the package namespace
    assert callable(m.run_camex)


def test_camex_core_submodules_import():
    """The compat risk: the vendored base/trainer (which import torch+dgl and
    rely on the DataLoader/pandas-2 patches) must actually import under the
    installed torch/dgl/pandas, not just the light package shell."""
    pytest.importorskip("torch")
    pytest.importorskip("dgl")

    base = importlib.import_module("omicverse.external.camex.base")
    trainer = importlib.import_module("omicverse.external.camex.trainer")
    assert isinstance(base.Dataset, type)
    assert isinstance(trainer.Trainer, type)
    # DGL>=1.0 NodeDataLoader compat shim is present
    assert callable(trainer._dgl_node_dataloader)
