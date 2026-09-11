from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from omicverse.space._cluster import pySTAGATE
from omicverse.space._spaceflow import _sampled_regularization_distances, pySpaceFlow


def test_spaceflow_regularization_samples_independent_spatial_pairs(monkeypatch):
    import torch

    draws = iter(
        [
            torch.tensor([0, 0], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        ]
    )
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: next(draws),
    )
    latent = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    coords = torch.tensor([[0.0, 0.0], [3.0, 0.0]])

    latent_dist, spatial_dist = _sampled_regularization_distances(
        latent,
        coords,
        edge_subset_sz=2,
    )

    assert latent_dist[0] < 1e-3
    assert spatial_dist[0] < 1e-3
    assert torch.isclose(latent_dist[1], torch.tensor(1.0))
    assert torch.isclose(spatial_dist[1], torch.tensor(1.0))


def _patch_scanpy_trajectory(monkeypatch):
    monkeypatch.setattr("scanpy.pp.neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr("scanpy.tl.umap", lambda *args, **kwargs: None)
    monkeypatch.setattr("scanpy.tl.leiden", lambda *args, **kwargs: None)
    monkeypatch.setattr("scanpy.tl.paga", lambda *args, **kwargs: None)
    monkeypatch.setattr("scanpy.tl.diffmap", lambda *args, **kwargs: None)

    def fake_dpt(adata, *args, **kwargs):
        adata.obs["dpt_pseudotime"] = np.linspace(0, 1, adata.n_obs)

    monkeypatch.setattr("scanpy.tl.dpt", fake_dpt)


def test_identical_sampled_points_have_zero_distance():
    import torch
    z = torch.zeros((3, 2))
    latent, spatial = _sampled_regularization_distances(z, z, 8)
    assert torch.equal(latent, torch.zeros(8))
    assert torch.equal(spatial, torch.zeros(8))


def test_spaceflow_zero_epochs_rejected_before_training():
    model = object.__new__(pySpaceFlow)
    with pytest.raises(ValueError, match='positive integer'):
        model.train(epochs=0)


def _adata_with_distant_subsample_root(rep_key):
    adata = AnnData(np.ones((6, 2), dtype=np.float32))
    embedding = np.zeros((6, 2), dtype=np.float64)
    embedding[5] = [100.0, 0.0]
    embedding[4] = [1.0, 0.0]
    adata.obsm[rep_key] = embedding
    return adata


def test_spaceflow_subsample_root_maps_back_to_global_index(monkeypatch):
    _patch_scanpy_trajectory(monkeypatch)
    monkeypatch.setattr(
        "numpy.random.choice",
        lambda *args, **kwargs: np.array([5, 2, 4]),
    )
    model = object.__new__(pySpaceFlow)
    model.adata = _adata_with_distant_subsample_root("spaceflow")

    model.cal_pSM(max_cell_for_subsampling=3)

    assert model.adata.uns["iroot"] == 5


def test_stagate_subsample_root_maps_back_to_global_index(monkeypatch):
    _patch_scanpy_trajectory(monkeypatch)
    monkeypatch.setattr(
        "numpy.random.choice",
        lambda *args, **kwargs: np.array([5, 2, 4]),
    )
    model = object.__new__(pySTAGATE)
    model.adata = _adata_with_distant_subsample_root("STAGATE")

    model.cal_pSM(max_cell_for_subsampling=3)

    assert model.adata.uns["iroot"] == 5
