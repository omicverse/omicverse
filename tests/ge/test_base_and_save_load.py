"""``BaseGraphModel``/``SIMBA``: get_embedding, write_embeddings, save/load
round trip."""
from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
anndata = pytest.importorskip("anndata")

from omicverse.ge._base import BaseGraphModel
from omicverse.ge._graph import HeteroGraph
from omicverse.ge._simba import SIMBA


def _tiny_graph(seed=0):
    rng = np.random.default_rng(seed)
    g = HeteroGraph()
    src = rng.integers(0, 12, 60)
    dst = rng.integers(0, 9, 60)
    g.add_entity_type("C", [f"cell{i}" for i in range(12)])
    g.add_entity_type("G", [f"gene{i}" for i in range(9)])
    g.add_edges("C", "G", src, dst, name="r0", weight=1.0)
    g.build()
    return g


def test_get_embedding_shapes_and_ids():
    model = BaseGraphModel(dimension=5)
    model.graph = _tiny_graph()
    model.build()
    emb_c = model.get_embedding("C")
    emb_g = model.get_embedding("G")
    assert emb_c.shape == (12, 5)
    assert emb_g.shape == (9, 5)
    assert list(emb_c.obs_names) == [f"cell{i}" for i in range(12)]
    with pytest.raises(KeyError):
        model.get_embedding("nope")


def test_write_embeddings_aligns_to_adata_obs_var():
    model = BaseGraphModel(dimension=4)
    model.graph = _tiny_graph()
    model.build()
    adata = anndata.AnnData(X=np.zeros((12, 9), dtype=np.float32))
    adata.obs_names = [f"cell{i}" for i in reversed(range(12))]  # different order
    adata.var_names = [f"gene{i}" for i in range(9)]
    model.write_embeddings(adata, cell_type="C", gene_type="G")
    assert adata.obsm["X_simba"].shape == (12, 4)
    assert adata.varm["simba"].shape == (9, 4)
    emb = model.get_embedding("C")
    want = emb[emb.obs_names.get_indexer(adata.obs_names)].X
    np.testing.assert_allclose(adata.obsm["X_simba"], want)


def test_save_load_round_trip_base_graph_model(tmp_path):
    model = BaseGraphModel(dimension=4)
    model.graph = _tiny_graph()
    model.build()
    model.train(num_epochs=2, batch_size=30, progress=False, auto_wd=False, wd=0.0)
    before = model.get_embedding("C").X.copy()

    path = str(tmp_path / "model")
    model.save(path)
    loaded = BaseGraphModel.load(path)
    after = loaded.get_embedding("C").X
    np.testing.assert_allclose(before, after)
    assert loaded.is_trained_ is True
    assert loaded.history_["train_loss"] == model.history_["train_loss"]


def test_save_load_round_trip_simba(tmp_path):
    model = SIMBA(dimension=4)
    model.graph = _tiny_graph()
    model.build()
    model.train(num_epochs=2, batch_size=30, progress=False, auto_wd=False, wd=0.0)
    path = str(tmp_path / "simba_model")
    model.save(path)
    loaded = SIMBA.load(path)
    assert isinstance(loaded, SIMBA)
    np.testing.assert_allclose(model.get_embedding("G").X, loaded.get_embedding("G").X)


def test_save_refuses_overwrite_without_flag(tmp_path):
    model = BaseGraphModel(dimension=3)
    model.graph = _tiny_graph()
    model.build()
    path = str(tmp_path / "m")
    model.save(path)
    with pytest.raises(FileExistsError):
        model.save(path)
    model.save(path, overwrite=True)  # should not raise
