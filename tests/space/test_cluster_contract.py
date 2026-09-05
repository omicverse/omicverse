from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from omicverse.space import _cluster


def _adata():
    adata = AnnData(np.ones((6, 3), dtype=np.float32))
    adata.obsm["spatial"] = np.column_stack(
        [np.arange(6, dtype=float), np.zeros(6, dtype=float)]
    )
    return adata


def test_clusters_rejects_typos_instead_of_returning_false_success():
    with pytest.raises(ValueError, match="GrahpST.*GraphST"):
        _cluster.clusters(_adata(), methods=["GrahpST"], methods_kwargs={})


def test_clusters_does_not_mutate_user_method_configuration(monkeypatch):
    class FakeSTAGATE:
        def __init__(self, adata, **kwargs):
            self.adata = adata

        def train(self):
            return None

        def predicted(self):
            self.adata.obsm["STAGATE"] = np.ones((self.adata.n_obs, 2))
            self.adata.layers["STAGATE_ReX"] = np.asarray(self.adata.X).copy()

    def fake_hvg(adata, **kwargs):
        adata.var["highly_variable"] = [True, True, False]

    monkeypatch.setattr(_cluster, "pySTAGATE", FakeSTAGATE)
    monkeypatch.setattr(_cluster.sc.pp, "highly_variable_genes", fake_hvg)
    config = {
        "STAGATE": {
            "num_batch_x": 1,
            "num_batch_y": 1,
            "n_top_genes": 2,
        }
    }
    original = copy.deepcopy(config)

    _cluster.clusters(_adata(), methods=["stagate"], methods_kwargs=config)

    assert config == original


def test_merge_cluster_maps_real_string_categories():
    adata = _adata()
    adata.obs["domain"] = pd.Categorical(
        ["alpha", "alpha", "beta", "beta", "gamma", "gamma"],
        categories=["alpha", "beta", "gamma"],
    )
    adata.obsm["X_domain"] = np.array(
        [[0, 0], [0, 0.1], [5, 5], [5, 5.1], [10, 0], [10, 0.1]],
        dtype=float,
    )

    mapping = _cluster.merge_cluster(
        adata,
        groupby="domain",
        use_rep="X_domain",
        threshold=1,
        plot=False,
    )

    assert set(mapping) == {"alpha", "beta", "gamma"}
    assert adata.obs["domain_tree"].notna().all()
    assert list(adata.obs["domain"].cat.categories) == ["alpha", "beta", "gamma"]
