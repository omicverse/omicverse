from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from scipy import sparse

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from omicverse.space import _integrate
from omicverse.space._integrate import Cal_Spatial_Net, pySTAligner


def test_staligner_missing_batch_list_has_actionable_error():
    adata = AnnData(
        np.ones((4, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"batch": ["a", "a", "b", "b"]},
            index=[f"c{i}" for i in range(4)],
        ),
    )

    with pytest.raises(ValueError, match="Batch_list.*Cal_Spatial_Net"):
        pySTAligner(adata, batch_key="batch")


def _batch(prefix, offset=0.0):
    adata = AnnData(
        np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
        obs=pd.DataFrame(index=[f"{prefix}{i}" for i in range(3)]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    adata.obsm["spatial"] = np.array(
        [[offset, 0], [offset + 1, 0], [offset + 2, 0]], dtype=float
    )
    adata.uns["adj"] = sparse.eye(adata.n_obs, format="coo")
    adata.uns['_omicverse_staligner_obs_names'] = adata.obs_names.to_numpy(dtype=str)
    return adata


def _combined(*batches):
    from anndata import concat

    keys = [f"s{i}" for i in range(len(batches))]
    return concat(
        batches,
        label="batch",
        keys=keys,
        index_unique="-",
        merge="same",
    )


class _TinySTAligner(torch.nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.encoder = torch.nn.Linear(hidden_dims[0], hidden_dims[-1], bias=False)
        self.decoder = torch.nn.Linear(hidden_dims[-1], hidden_dims[0], bias=False)
        self.seen_x = []

    def forward(self, x, edge_index):
        del edge_index
        self.seen_x.append(x.detach().cpu().clone())
        z = self.encoder(x)
        return z, self.decoder(z)


def test_staligner_accepts_anndata_concat_suffixes_and_defaults_to_adjacent_pairs(
    monkeypatch,
):
    batches = [_batch("a"), _batch("b", 10), _batch("c", 20)]
    combined = _combined(*batches)
    monkeypatch.setattr(
        _integrate,
        "_get_staligner_backend",
        lambda: (_TinySTAligner, lambda *args, **kwargs: {}),
    )

    model = pySTAligner(
        combined,
        batch_key="batch",
        Batch_list=batches,
        hidden_dims=[4, 2],
        n_epochs=2,
        mnn_approx=False,
        device="cpu",
    )

    assert model.iter_comb == [(0, 1), (1, 2)]
    with pytest.raises(RuntimeError, match="train"):
        model.predicted()


def test_staligner_rejects_configuration_that_skips_alignment(monkeypatch):
    batches = [_batch("a"), _batch("b", 10)]
    combined = _combined(*batches)

    with pytest.raises(ValueError, match="at least 2"):
        pySTAligner(
            combined,
            batch_key="batch",
            Batch_list=batches,
            n_epochs=1,
            device="cpu",
        )


def test_staligner_two_epochs_enters_mnn_alignment(monkeypatch):
    batches = [_batch("a"), _batch("b", 10)]
    combined = _combined(*batches)
    calls = []

    def fake_mnn(batch_pair, **kwargs):
        calls.append(kwargs["k"])
        left = batch_pair.obs_names[batch_pair.obs["batch"] == "s0"][0]
        right = batch_pair.obs_names[batch_pair.obs["batch"] == "s1"][0]
        return {"s0_s1": {left: [right]}}

    monkeypatch.setattr(
        _integrate,
        "_get_staligner_backend",
        lambda: (_TinySTAligner, fake_mnn),
    )

    model = pySTAligner(
        combined,
        batch_key="batch",
        Batch_list=batches,
        hidden_dims=[4, 2],
        n_epochs=2,
        knn_neigh=100,
        mnn_approx=False,
        device="cpu",
    )
    model.train()
    result = model.predicted()

    assert calls == [3]
    assert result.obsm["STAligner"].shape == (6, 2)
    assert np.isfinite(result.obsm["STAligner"]).all()


def test_cal_spatial_net_handles_small_slices_and_caps_knn():
    adata = _batch("a")

    with pytest.warns(UserWarning, match="available"):
        Cal_Spatial_Net(
            adata,
            model="KNN",
            k_cutoff=10,
            max_neigh=50,
            verbose=False,
        )

    assert adata.uns["adj"].shape == (3, 3)
    assert adata.uns["Spatial_Net"].shape[0] == 6


def test_cal_spatial_net_removes_self_by_identity_with_duplicate_coordinates():
    adata = _batch("x")
    adata.obsm["spatial"] = np.array([[0, 0], [0, 0], [1, 0]], dtype=float)

    Cal_Spatial_Net(
        adata,
        model="KNN",
        k_cutoff=1,
        max_neigh=1,
        verbose=False,
    )

    edges = adata.uns["Spatial_Net"]
    assert not (edges["Cell1"] == edges["Cell2"]).any()
    assert ((edges["Cell1"] == "x0") & (edges["Cell2"] == "x1")).any()


def test_staligner_requires_explicit_identity_for_shared_visium_barcodes(monkeypatch):
    batch_a = _batch("shared")
    batch_b = _batch("shared", 10)
    batch_b.X = np.full(batch_b.shape, 9.0, dtype=np.float32)
    combined = _combined(batch_a, batch_b)
    monkeypatch.setattr(
        _integrate,
        "_get_staligner_backend",
        lambda: (_TinySTAligner, lambda *args, **kwargs: {}),
    )

    with pytest.raises(ValueError, match="order cannot be verified.*batch_ids"):
        pySTAligner(
            combined,
            batch_key="batch",
            Batch_list=[batch_a, batch_b],
            n_epochs=2,
            mnn_approx=False,
            device="cpu",
        )

    model = pySTAligner(
        combined,
        batch_key="batch",
        Batch_list={"s0": batch_a, "s1": batch_b},
        hidden_dims=[4, 2],
        n_epochs=2,
        mnn_approx=False,
        device="cpu",
    )
    np.testing.assert_array_equal(model.data_list[0].x.numpy(), np.asarray(batch_a.X))
    np.testing.assert_array_equal(model.data_list[1].x.numpy(), np.asarray(batch_b.X))


def test_staligner_rejects_disconnected_custom_pair_graph(monkeypatch):
    batches = [_batch("a"), _batch("b", 10), _batch("c", 20)]
    combined = _combined(*batches)

    with pytest.raises(ValueError, match="connected graph.*uncovered"):
        pySTAligner(
            combined,
            batch_key="batch",
            Batch_list=batches,
            iter_comb=[(0, 1)],
            n_epochs=2,
            mnn_approx=False,
            device="cpu",
        )


def test_staligner_seed_controls_model_initialization(monkeypatch):
    batches = [_batch("a"), _batch("b", 10)]
    combined = _combined(*batches)
    monkeypatch.setattr(
        _integrate,
        "_get_staligner_backend",
        lambda: (_TinySTAligner, lambda *args, **kwargs: {}),
    )
    kwargs = dict(
        batch_key="batch",
        Batch_list=batches,
        hidden_dims=[4, 2],
        n_epochs=2,
        random_seed=19,
        mnn_approx=False,
        device="cpu",
    )

    model_a = pySTAligner(combined, **kwargs)
    model_b = pySTAligner(combined, **kwargs)

    for left, right in zip(model_a.model.parameters(), model_b.model.parameters()):
        assert torch.equal(left, right)


def test_staligner_alignment_uses_same_expression_source_as_pretraining(monkeypatch):
    batches = [_batch("a"), _batch("b", 10)]
    combined = _combined(*batches)
    combined.X = np.asarray(combined.X) * np.float32(100.0)

    def fake_mnn(batch_pair, **kwargs):
        left = batch_pair.obs_names[batch_pair.obs["batch"] == "s0"][0]
        right = batch_pair.obs_names[batch_pair.obs["batch"] == "s1"][0]
        return {"s0_s1": {left: [right]}}

    monkeypatch.setattr(
        _integrate,
        "_get_staligner_backend",
        lambda: (_TinySTAligner, fake_mnn),
    )
    model = pySTAligner(
        combined,
        batch_key="batch",
        Batch_list=batches,
        hidden_dims=[4, 2],
        n_epochs=2,
        mnn_approx=False,
        device="cpu",
    )
    model.train()

    assert max(float(seen.max()) for seen in model.model.seen_x) < 10.0


def test_staligner_empty_mnn_cannot_be_reported_as_fitted(monkeypatch):
    batches = [_batch("a"), _batch("b", 10)]
    combined = _combined(*batches)
    monkeypatch.setattr(
        _integrate,
        "_get_staligner_backend",
        lambda: (_TinySTAligner, lambda *args, **kwargs: {}),
    )
    model = pySTAligner(
        combined,
        batch_key="batch",
        Batch_list=batches,
        hidden_dims=[4, 2],
        n_epochs=2,
        mnn_approx=False,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="no usable mutual-nearest-neighbor"):
        model.train()

    assert not model._is_fitted


def test_staligner_rebuilds_stale_positional_adjacency_from_named_edges():
    batch = _batch("a")
    batch.uns["Spatial_Net"] = pd.DataFrame(
        {"Cell1": ["a0", "a1"], "Cell2": ["a1", "a0"], "Distance": [1.0, 1.0]}
    )
    old = sparse.coo_matrix(
        (np.ones(2), ([0, 1], [1, 0])),
        shape=(3, 3),
    ).tocsr() + sparse.eye(3, format="csr")
    batch.uns["adj"] = old
    reordered = batch[[2, 0, 1]].copy()

    with pytest.warns(UserWarning, match="not aligned.*name-aligned"):
        adjacency = _integrate._validated_staligner_adjacency(reordered, 0)

    current = {name: i for i, name in enumerate(reordered.obs_names)}
    assert adjacency[current["a0"], current["a1"]] == 1
    assert adjacency[current["a1"], current["a0"]] == 1


def test_real_staligner_two_stage_cpu_smoke_without_optional_compiled_neighbors(
    monkeypatch,
):
    import builtins

    real_import = builtins.__import__

    def import_without_hnswlib(name, *args, **kwargs):
        if name == "hnswlib":
            raise ImportError("simulated missing optional hnswlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_hnswlib)
    batch_a = _batch("a")
    batch_b = _batch("b", 10)
    batch_b.X = np.asarray(batch_a.X) + np.float32(0.01)
    combined = _combined(batch_a, batch_b)

    with pytest.warns(UserWarning, match="exact scikit-learn"):
        model = pySTAligner(
            combined,
            batch_key="batch",
            Batch_list=[batch_a, batch_b],
            hidden_dims=[4, 2],
            n_epochs=2,
            knn_neigh=1,
            device="cpu",
            random_seed=7,
        )
    model.train()
    result = model.predicted()

    assert model._is_fitted
    assert result.obsm["STAligner"].shape == (6, 2)
    assert np.isfinite(result.obsm["STAligner"]).all()
