"""DecontX must receive per-cell batch labels when ``batch_key`` is set."""
from __future__ import annotations

import types

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from omicverse.pp.ambient import _ambient


def _make_adata():
    X = np.array([
        [12, 0, 0],
        [9, 0, 0],
        [0, 11, 0],
        [0, 7, 0],
    ], dtype=float)
    return anndata.AnnData(
        X=X,
        obs=pd.DataFrame({
            "cluster": ["A", "A", "B", "B"],
            "batch": ["s1", "s1", "s2", "s2"],
        }),
    )


def _fake_backend(monkeypatch, fake_decontx):
    fake_mod = types.SimpleNamespace(decontx=fake_decontx)
    monkeypatch.setattr(
        _ambient, "_require",
        lambda modname, role, extra="ambient": fake_mod)
    return fake_mod


def _fake_result(adata):
    return types.SimpleNamespace(
        decontx_counts=sp.csr_matrix(
            np.ones((adata.n_vars, adata.n_obs))),
        contamination=np.zeros(adata.n_obs),
    )


def test_decontx_backend_passes_batch_labels(monkeypatch):
    adata = _make_adata()
    captured = {}

    def fake_decontx(x, z=None, batch=None, background=None, **kwargs):
        captured["z"] = z
        captured["batch"] = batch
        return _fake_result(adata)

    _fake_backend(monkeypatch, fake_decontx)

    corrected, rho, n_genes, meta = _ambient._run_decontx(
        adata, cluster_key="cluster", batch_key="batch")

    np.testing.assert_array_equal(captured["z"], ["A", "A", "B", "B"])
    np.testing.assert_array_equal(
        captured["batch"], ["s1", "s1", "s2", "s2"])
    assert meta["batch_key"] == "batch"
    assert corrected.shape == adata.shape
    assert rho.shape == (adata.n_obs,)
    assert n_genes == adata.n_vars


def test_decontx_backend_defaults_to_all_cells(monkeypatch):
    adata = _make_adata()
    captured = {}

    def fake_decontx(x, z=None, batch=None, background=None, **kwargs):
        captured["batch"] = batch
        return _fake_result(adata)

    _fake_backend(monkeypatch, fake_decontx)
    _ambient._run_decontx(adata, cluster_key="cluster")

    assert captured["batch"] is None


def test_decontx_backend_rejects_missing_batch_key(monkeypatch):
    adata = _make_adata()
    _fake_backend(monkeypatch, lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="batch_key='missing'"):
        _ambient._run_decontx(
            adata, cluster_key="cluster", batch_key="missing")


def test_remove_ambient_forwards_batch_key(monkeypatch):
    adata = _make_adata()
    captured = {}

    def fake_runner(adata, *, raw=None, layer=None, cluster_key=None,
                    batch_key=None, verbose=False, **kwargs):
        captured["batch_key"] = batch_key
        return (
            adata.X.copy(),
            np.zeros(adata.n_obs),
            adata.n_vars,
            {"cluster_key": cluster_key, "batch_key": batch_key},
        )

    monkeypatch.setattr(_ambient, "_DISPATCH", {"decontx": fake_runner})
    out = _ambient.remove_ambient(
        adata, method="decontx", cluster_key="cluster",
        batch_key="batch", check_integrity=False)

    assert captured["batch_key"] == "batch"
    assert out.uns["ambient"]["batch_key"] == "batch"


def test_estimate_contamination_forwards_batch_key(monkeypatch):
    adata = _make_adata()
    captured = {}

    def fake_runner(adata, *, raw=None, layer=None, cluster_key=None,
                    batch_key=None, verbose=False, **kwargs):
        captured["batch_key"] = batch_key
        return (
            None,
            np.zeros(adata.n_obs),
            0,
            {"cluster_key": cluster_key, "batch_key": batch_key},
        )

    monkeypatch.setattr(_ambient, "_DISPATCH", {"decontx": fake_runner})
    series = _ambient.estimate_contamination(
        adata, method="decontx", cluster_key="cluster", batch_key="batch")

    assert captured["batch_key"] == "batch"
    assert series.attrs["batch_key"] == "batch"
