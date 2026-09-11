from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from omicverse.space._tangram import construct_obs_plot


def test_construct_obs_plot_keeps_constant_columns_finite():
    adata = AnnData(np.ones((2, 1), dtype=np.float32))
    adata.obs_names = ["a", "b"]
    values = pd.DataFrame(
        {
            "uniform": [0.5, 0.5],
            "varying": [0.2, 0.8],
        },
        index=adata.obs_names,
    )

    construct_obs_plot(values, adata)

    assert adata.obs["uniform"].tolist() == [0.5, 0.5]
    assert adata.obs["varying"].tolist() == [0.0, 1.0]
    assert np.isfinite(adata.obs[["uniform", "varying"]].to_numpy()).all()


def test_tangram_tiny_cpu_runtime():
    pytest.importorskip("tangram")
    from omicverse.space import Tangram

    rng = np.random.default_rng(3)
    adata_sc = AnnData(rng.poisson(3, size=(24, 8)).astype(np.float32))
    adata_sc.var_names = [f"g{i}" for i in range(8)]
    adata_sc.obs["cell_type"] = ["A"] * 12 + ["B"] * 12
    adata_sp = AnnData(rng.poisson(4, size=(10, 8)).astype(np.float32))
    adata_sp.var_names = adata_sc.var_names.copy()
    adata_sp.obsm["spatial"] = rng.uniform(0, 10, size=(10, 2))

    model = Tangram(
        adata_sc,
        adata_sp,
        clusters="cell_type",
        marker_size=4,
    )
    model.train(num_epochs=1, device="cpu")
    result = model.cell2location()

    assert result.obsm["tangram_ct_pred"].shape == (10, 2)
    assert np.isfinite(result.obs[["A", "B"]].to_numpy()).all()
