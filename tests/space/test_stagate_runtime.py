from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData


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
