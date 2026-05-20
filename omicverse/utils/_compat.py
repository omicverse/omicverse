"""Vendored compatibility helpers.

This module reimplements small pieces of scanpy's private API so omicverse
does not break when scanpy changes its internals. Keep these copies faithful
to the upstream behaviour they replace.
"""

from __future__ import annotations

import numpy as np
import scipy as sp
from scipy.sparse.csgraph import minimum_spanning_tree

__all__ = ["PAGA"]

_AVAIL_MODELS = {"v1.0", "v1.2"}


def _get_igraph_from_adjacency(adjacency, *, directed: bool = False):
    """Build an igraph graph from a (sparse) adjacency matrix.

    Vendored from ``scanpy._utils.get_igraph_from_adjacency``.
    """
    import igraph as ig

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es["weight"] = weights
    except KeyError:
        pass
    if g.vcount() != adjacency.shape[0]:
        print(
            f"The constructed graph has only {g.vcount()} nodes. "
            "Your adjacency matrix contained redundant nodes."
        )
    return g


class PAGA:
    """Partition-based graph abstraction.

    Vendored from ``scanpy.tools._paga.PAGA`` so omicverse does not depend on
    scanpy's private ``scanpy.tools._paga`` module. Uses omicverse's own
    :class:`~omicverse.pp._neighbors.Neighbors` for the neighbor graph.
    """

    def __init__(self, adata, groups, model: str = "v1.2", neighbors_key=None):
        assert groups in adata.obs.columns
        from ..pp._neighbors import Neighbors

        self._adata = adata
        self._neighbors = Neighbors(adata, neighbors_key=neighbors_key)
        self._model = model
        self._groups_key = groups

    def compute_connectivities(self):
        if self._model == "v1.2":
            return self._compute_connectivities_v1_2()
        elif self._model == "v1.0":
            return self._compute_connectivities_v1_0()
        else:
            msg = f"`model` {self._model} needs to be one of {_AVAIL_MODELS}."
            raise ValueError(msg)

    def _compute_connectivities_v1_2(self):
        import igraph

        ones = self._neighbors.distances.copy()
        ones.data = np.ones(len(ones.data))
        # should be directed if we deal with distances
        g = _get_igraph_from_adjacency(ones, directed=True)
        vc = igraph.VertexClustering(
            g, membership=self._adata.obs[self._groups_key].cat.codes.values
        )
        ns = vc.sizes()
        n = sum(ns)
        es_inner_cluster = [vc.subgraph(i).ecount() for i in range(len(ns))]
        cg = vc.cluster_graph(combine_edges="sum")
        inter_es = cg.get_adjacency_sparse(attribute="weight")
        es = np.array(es_inner_cluster) + inter_es.sum(axis=1).A1
        inter_es = inter_es + inter_es.T  # \epsilon_i + \epsilon_j
        connectivities = inter_es.copy()
        expected_n_edges = inter_es.copy()
        inter_es = inter_es.tocoo()
        for i, j, v in zip(inter_es.row, inter_es.col, inter_es.data):
            expected_random_null = (es[i] * ns[j] + es[j] * ns[i]) / (n - 1)
            scaled_value = v / expected_random_null if expected_random_null != 0 else 1
            scaled_value = min(scaled_value, 1)
            connectivities[i, j] = scaled_value
            expected_n_edges[i, j] = expected_random_null
        # set attributes
        self.ns = ns
        self.expected_n_edges_random = expected_n_edges
        self.connectivities = connectivities
        self.connectivities_tree = self._get_connectivities_tree_v1_2()
        return inter_es.tocsr(), connectivities

    def _compute_connectivities_v1_0(self):
        import igraph

        ones = self._neighbors.connectivities.copy()
        ones.data = np.ones(len(ones.data))
        g = _get_igraph_from_adjacency(ones)
        vc = igraph.VertexClustering(
            g, membership=self._adata.obs[self._groups_key].cat.codes.values
        )
        ns = vc.sizes()
        cg = vc.cluster_graph(combine_edges="sum")
        inter_es = cg.get_adjacency_sparse(attribute="weight") / 2
        connectivities = inter_es.copy()
        inter_es = inter_es.tocoo()
        n_neighbors_sq = self._neighbors.n_neighbors**2
        for i, j, v in zip(inter_es.row, inter_es.col, inter_es.data):
            # have n_neighbors**2 inside sqrt for backwards compat
            geom_mean_approx_knn = np.sqrt(n_neighbors_sq * ns[i] * ns[j])
            scaled_value = v / geom_mean_approx_knn if geom_mean_approx_knn != 0 else 1
            connectivities[i, j] = scaled_value
        # set attributes
        self.ns = ns
        self.connectivities = connectivities
        self.connectivities_tree = self._get_connectivities_tree_v1_0(inter_es)
        return inter_es.tocsr(), connectivities

    def _get_connectivities_tree_v1_2(self):
        inverse_connectivities = self.connectivities.copy()
        inverse_connectivities.data = 1.0 / inverse_connectivities.data
        connectivities_tree = minimum_spanning_tree(inverse_connectivities)
        connectivities_tree_indices = [
            connectivities_tree[i].nonzero()[1]
            for i in range(connectivities_tree.shape[0])
        ]
        connectivities_tree = sp.sparse.lil_matrix(
            self.connectivities.shape, dtype=float
        )
        for i, neighbors in enumerate(connectivities_tree_indices):
            if len(neighbors) > 0:
                connectivities_tree[i, neighbors] = self.connectivities[i, neighbors]
        return connectivities_tree.tocsr()

    def _get_connectivities_tree_v1_0(self, inter_es):
        inverse_inter_es = inter_es.copy()
        inverse_inter_es.data = 1.0 / inverse_inter_es.data
        connectivities_tree = minimum_spanning_tree(inverse_inter_es)
        connectivities_tree_indices = [
            connectivities_tree[i].nonzero()[1]
            for i in range(connectivities_tree.shape[0])
        ]
        connectivities_tree = sp.sparse.lil_matrix(inter_es.shape, dtype=float)
        for i, neighbors in enumerate(connectivities_tree_indices):
            if len(neighbors) > 0:
                connectivities_tree[i, neighbors] = self.connectivities[i, neighbors]
        return connectivities_tree.tocsr()
