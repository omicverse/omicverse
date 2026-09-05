from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData


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
