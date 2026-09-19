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


def test_spaceflow_accelerated_regularization_tiny_runtime():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("gudhi")
    import omicverse as ov

    rng = np.random.default_rng(0)
    adata = AnnData(rng.poisson(3, size=(24, 12)).astype(np.float32))
    adata.obsm["spatial"] = rng.uniform(0, 10, size=(24, 2))

    model = ov.space.pySpaceFlow(adata)
    embedding = model.train(
        z_dim=4,
        epochs=2,
        max_patience=2,
        min_stop=0,
        regularization_acceleration=True,
        edge_subset_sz=64,
    )

    assert embedding.shape == (24, 4)
    assert adata.obsm["spaceflow"].shape == (24, 4)
    assert np.isfinite(embedding).all()


def test_stagate_edge_index_runtime_does_not_require_torch_sparse():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import matplotlib.pyplot as plt
    import omicverse as ov

    rng = np.random.default_rng(1)
    adata = AnnData(rng.poisson(3, size=(16, 8)).astype(np.float32))
    xx, yy = np.meshgrid(np.arange(4), np.arange(4))
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    adata.obsm["spatial"] = coords
    adata.obs["X"] = coords[:, 0]
    adata.obs["Y"] = coords[:, 1]

    model = ov.space.pySTAGATE(
        adata,
        num_batch_x=2,
        num_batch_y=2,
        rad_cutoff=1.5,
        num_epoch=1,
        hidden_dims=[4, 2],
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="train.*predicted"):
        model.predicted()
    model.train()
    model.predicted()
    plt.close("all")

    assert adata.obsm["STAGATE"].shape == (16, 2)
    assert adata.layers["STAGATE_ReX"].shape == adata.shape
    assert np.isfinite(adata.obsm["STAGATE"]).all()


def test_stagate_rejects_zero_epochs_before_returning_random_embeddings():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import omicverse as ov

    adata = AnnData(np.ones((4, 2), dtype=np.float32))
    adata.obsm["spatial"] = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float
    )

    with pytest.raises(ValueError, match="positive integer.*untrained random"):
        ov.space.pySTAGATE(
            adata,
            num_batch_x=1,
            num_batch_y=1,
            num_epoch=0,
            device="cpu",
        )


@pytest.mark.parametrize('backend', ['wrapper', 'vendored'])
def test_spaceflow_restores_best_epoch_weights(monkeypatch, tmp_path, backend):
    pytest.importorskip('gudhi')
    import torch
    from torch_geometric.nn import DeepGraphInfomax

    original_loss = DeepGraphInfomax.loss
    original_load = DeepGraphInfomax.load_state_dict
    calls = []
    restored = []

    def increasing_loss(model, *args, **kwargs):
        calls.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return original_loss(model, *args, **kwargs) + 100 * len(calls)

    def checked_load(model, state, *args, **kwargs):
        assert len(calls) == 3
        assert any(not torch.equal(calls[1][k], calls[2][k]) for k in calls[1])
        for name, expected in calls[1].items():
            torch.testing.assert_close(state[name], expected, rtol=0, atol=0)
        restored.append(True)
        return original_load(model, state, *args, **kwargs)

    monkeypatch.setattr(DeepGraphInfomax, 'loss', increasing_loss)
    monkeypatch.setattr(DeepGraphInfomax, 'load_state_dict', checked_load)
    rng = np.random.default_rng(9)
    adata = AnnData(rng.poisson(3, (24, 12)).astype(np.float32))
    adata.obsm['spatial'] = rng.uniform(0, 10, (24, 2))
    model = pySpaceFlow(adata)
    runner = model if backend == 'wrapper' else model.sf
    options = {} if backend == 'wrapper' else {'embedding_save_filepath': str(tmp_path / 'embedding.tsv')}
    result = runner.train(epochs=3, z_dim=4, edge_subset_sz=64, **options)
    assert restored == [True]
    assert result.shape == (24, 4)
    assert np.isfinite(result).all()


@pytest.mark.parametrize('spatial_key', ['spatial', ['X', 'Y'], None])
def test_stagate_uses_selected_coordinates_without_overwriting_input(spatial_key):
    import matplotlib.pyplot as plt

    xx, yy = np.meshgrid(np.arange(4), np.arange(4))
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    adata = AnnData(np.ones((16, 4), dtype=np.float32))
    adata.obsm['spatial'] = coords.copy()
    adata.obs['X'] = coords[:, 0] * 100
    adata.obs['Y'] = coords[:, 1] * 100
    original_obs = adata.obs.copy()
    model = pySTAGATE(adata, 2, 2, spatial_key=spatial_key, rad_cutoff=1.5,
                      num_epoch=1, hidden_dims=[4, 2], device='cpu')
    if spatial_key == 'spatial':
        assert model.data.edge_index.shape[1] > adata.n_obs
    else:
        assert model.data.edge_index.shape[1] == adata.n_obs
    np.testing.assert_array_equal(adata.obsm['spatial'], coords)
    assert adata.obs.equals(original_obs)
    plt.close('all')
