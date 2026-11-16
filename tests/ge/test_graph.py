"""``HeteroGraph`` / ``gen_graph``: entity ids, relation weights and edge
counts per relation against SIMBA's own ``gen_graph`` recipe, on a small
real scRNA matrix (``sample/rna_test.h5ad``, bundled with the repo); plus
an AnnDataOOM-backed build producing byte-identical edges to the in-memory
build.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

anndata = pytest.importorskip("anndata")

from omicverse.ge._graph import HeteroGraph

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SAMPLE = os.path.join(_HERE, "sample", "rna_test.h5ad")


@pytest.fixture(scope="module")
def small_real_adata():
    pytest.importorskip("sklearn")
    if not os.path.exists(_SAMPLE):
        pytest.skip(f"bundled sample not found: {_SAMPLE}")
    adata = anndata.read_h5ad(_SAMPLE)[:120, :150].copy()
    from omicverse.ge._simba import discretize, log_transform, normalize

    adata.var["highly_variable"] = True  # use all 150 genes of this subset
    normalize(adata)
    log_transform(adata)
    discretize(adata, n_bins=5)
    return adata


def _reference_cg_recipe(adata):
    """Re-derive SIMBA's ``gen_graph`` ``list_CG`` edge/weight recipe by
    hand (straight off ``simba/tools/_pbg.py``, not through our own code),
    to compare against ``gen_graph``'s output independently: one relation
    per discretization level, with ``linspace(1, 5)`` weights; the edge
    count of a level is simply how many non-zero entries carry that level
    (no CSR round-trip needed -- ``disc.data`` already holds exactly one
    level value per non-zero entry)."""
    disc = sp.csr_matrix(adata.layers["disc"])
    levels, counts_per_level = np.unique(disc.data, return_counts=True)
    weights = np.linspace(1, 5, num=len(levels))
    return {round(float(w), 2): int(c) for w, c in zip(weights, counts_per_level)}


def test_cg_edge_counts_and_weights_match_reference_recipe(small_real_adata):
    from omicverse.ge._simba import gen_graph

    adata = small_real_adata
    g = gen_graph(list_CG=[adata], use_highly_variable=False)
    g.build()

    ref_counts = _reference_cg_recipe(adata)
    got = {r["weight"]: g.edges_per_relation()[r["name"]] for r in g.relations}

    assert set(got.keys()) == set(ref_counts.keys())
    for w in ref_counts:
        assert got[w] == ref_counts[w]
    assert g.n_edges == sum(ref_counts.values())
    # SIMBA recipe: weights are linspace(1, 5) across expression levels.
    assert min(got) == pytest.approx(1.0)
    assert max(got) == pytest.approx(5.0)


def test_entity_ids_match_obs_var_names(small_real_adata):
    from omicverse.ge._simba import gen_graph

    adata = small_real_adata
    g = gen_graph(list_CG=[adata], use_highly_variable=False)
    assert list(g.entity_ids["C"]) == list(adata.obs_names)
    assert list(g.entity_ids["G"]) == list(adata.var_names)
    assert g.n_entities("C") == adata.n_obs
    assert g.n_entities("G") == adata.n_vars


def test_add_edges_from_sparse_matches_manual_nonzero():
    rng = np.random.default_rng(0)
    X = sp.random(10, 6, density=0.4, random_state=0,
                  data_rvs=lambda n: rng.integers(1, 5, n)).tocsr()
    row_ids = [f"c{i}" for i in range(10)]
    col_ids = [f"g{i}" for i in range(6)]

    g = HeteroGraph()
    rel_id = g.add_edges_from_sparse("C", "G", X, row_ids, col_ids, weight=2.5)
    g.build()

    rows, cols = X.nonzero()
    assert g.n_edges == len(rows)
    assert g.relations[rel_id]["weight"] == 2.5
    # every edge decodes back to a true non-zero entry of X
    c_ids = np.asarray(row_ids)[g.edges_src]
    g_ids = np.asarray(col_ids)[g.edges_dst]
    for c, gn in zip(c_ids, g_ids):
        ci, gi = int(c[1:]), int(gn[1:])
        assert X[ci, gi] != 0


class _FakeOOMReader:
    """Minimal stand-in for anndataoom-fast's row reader (the one primitive
    ``_iter_row_blocks`` needs: ``_read_rows(start, end)``)."""

    def __init__(self, X):
        self._X = sp.csr_matrix(X)
        self.shape = self._X.shape

    def _read_rows(self, start, end):
        return self._X[start:end]


def test_oom_backed_build_matches_in_memory_build():
    rng = np.random.default_rng(1)
    X = sp.random(37, 19, density=0.25, random_state=1,
                  data_rvs=lambda n: rng.integers(1, 5, n)).tocsr()
    row_ids = [f"c{i}" for i in range(37)]
    col_ids = [f"g{i}" for i in range(19)]

    g_mem = HeteroGraph()
    g_mem.add_edges_from_sparse("C", "G", X, row_ids, col_ids, block_size=4096)
    g_mem.build()

    g_oom = HeteroGraph()
    g_oom.add_edges_from_sparse("C", "G", _FakeOOMReader(X), row_ids, col_ids, block_size=7)
    g_oom.build()

    assert list(g_mem.entity_ids["C"]) == list(g_oom.entity_ids["C"])
    assert list(g_mem.entity_ids["G"]) == list(g_oom.entity_ids["G"])
    np.testing.assert_array_equal(g_mem.edges_src, g_oom.edges_src)
    np.testing.assert_array_equal(g_mem.edges_dst, g_oom.edges_dst)
    np.testing.assert_array_equal(g_mem.edges_rel, g_oom.edges_rel)


def test_oom_backed_discretized_relations_match_in_memory(small_real_adata):
    adata = small_real_adata
    disc = adata.layers["disc"]

    g_mem = HeteroGraph()
    g_mem.add_discretized_relations("C", "G", disc, adata.obs_names, adata.var_names,
                                    block_size=4096)
    g_mem.build()

    g_oom = HeteroGraph()
    g_oom.add_discretized_relations("C", "G", _FakeOOMReader(disc), adata.obs_names,
                                    adata.var_names, block_size=11)
    g_oom.build()

    np.testing.assert_array_equal(g_mem.edges_src, g_oom.edges_src)
    np.testing.assert_array_equal(g_mem.edges_dst, g_oom.edges_dst)
    np.testing.assert_array_equal(g_mem.edges_rel, g_oom.edges_rel)
    assert [r["weight"] for r in g_mem.relations] == [r["weight"] for r in g_oom.relations]


def test_auto_wd_formula():
    g = HeteroGraph()
    g.add_edges("C", "G", np.zeros(2725781, dtype=np.int64), np.zeros(2725781, dtype=np.int64))
    assert g.auto_wd() == pytest.approx(0.013, abs=1e-6)

    g2 = HeteroGraph()
    g2.add_edges("C", "G", np.zeros(59103481, dtype=np.int64), np.zeros(59103481, dtype=np.int64))
    assert g2.auto_wd() == pytest.approx(0.0004, abs=1e-6)


def test_save_load_round_trip(tmp_path):
    g = HeteroGraph()
    g.add_discretized_relations("C", "G",
                                sp.random(8, 5, density=0.5, random_state=0,
                                         data_rvs=lambda n: np.random.default_rng(0).integers(1, 4, n)),
                                [f"c{i}" for i in range(8)], [f"g{i}" for i in range(5)])
    g.build()
    path = str(tmp_path / "graph")
    g.save(path)
    g2 = HeteroGraph.load(path)
    np.testing.assert_array_equal(g.edges_src, g2.edges_src)
    np.testing.assert_array_equal(g.edges_dst, g2.edges_dst)
    np.testing.assert_array_equal(g.edges_rel, g2.edges_rel)
    assert g.relations == g2.relations
    assert list(g.entity_ids["C"]) == list(g2.entity_ids["C"])
