"""Louvain clustering without scanpy (native port of ``scanpy.tl.louvain``).

Ported from scanpy ``scanpy/tools/_louvain.py`` (BSD-3-Clause,
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab).

Backends for the default ``flavor='vtraag'``:

1. the ``louvain`` package (Traag 2017) if installed -- exactly what scanpy
   runs, so labels are identical to ``scanpy.tl.louvain``;
2. otherwise ``igraph.Graph.community_multilevel`` (igraph is a hard
   dependency; Blondel et al. 2008 multilevel modularity optimisation) on the
   undirected graph with the same ``resolution``, seeded from
   ``random_state``. On a 2,000-cell mouse-cortex scRNA-seq subset this agrees with the
   ``louvain`` package at ARI ~0.95, the same level as the ``louvain``
   package's own seed-to-seed agreement (ARI 0.946-0.947 for seeds 1-3 vs 0).
   (scanpy itself raises ImportError in this situation.)

``flavor='igraph'`` reproduces scanpy's ``flavor='igraph'`` exactly.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from natsort import natsorted

from ._leiden import (
    _choose_graph,
    get_igraph_from_adjacency,
    rename_groups,
    restrict_adjacency,
    set_igraph_random_state,
)


LAST_BACKEND = None


def _louvain_partition(g, *, resolution, random_state, use_weights, weights,
                       partition_type, partition_kwargs):
    partition_kwargs = dict(partition_kwargs)
    try:
        import louvain as _lv  # Traag's louvain package (scanpy's default)
        if partition_type is None:
            partition_type = _lv.RBConfigurationVertexPartition
        if resolution is not None:
            partition_kwargs["resolution_parameter"] = resolution
        if use_weights:
            partition_kwargs["weights"] = weights
        part = _lv.find_partition(g, partition_type, seed=random_state, **partition_kwargs)
        return np.array(part.membership), "louvain"
    except ImportError:
        pass
    g_u = g.as_undirected(combine_edges="first") if g.is_directed() else g
    w = np.array(g_u.es["weight"], dtype=np.float64) if use_weights else None
    with set_igraph_random_state(0 if random_state is None else random_state):
        part = g_u.community_multilevel(
            weights=w, resolution=1.0 if resolution is None else resolution)
    return np.array(part.membership), "igraph.community_multilevel"


def louvain(
    adata,
    resolution: float | None = None,
    *,
    random_state=0,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    key_added: str = "louvain",
    adjacency=None,
    flavor: str = "vtraag",
    directed: bool = True,
    use_weights: bool = False,
    partition_type=None,
    partition_kwargs: Mapping[str, Any] = MappingProxyType({}),
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False,
):
    """Louvain clustering of the neighbour graph (drop-in for ``scanpy.tl.louvain``).

    Writes ``adata.obs[key_added]`` (categorical, natsorted string labels) and
    ``adata.uns[key_added]['params']`` exactly like scanpy.
    """
    if flavor not in {"vtraag", "igraph"}:
        raise ValueError('`flavor` needs to be "vtraag" or "igraph".')
    if flavor != "vtraag" and partition_type is not None:
        raise ValueError('`partition_type` is only a valid argument when `flavour` is "vtraag"')
    adata = adata.copy() if copy else adata
    if adjacency is None:
        adjacency = _choose_graph(adata, obsp, neighbors_key)
    if restrict_to is not None:
        restrict_key, restrict_categories = restrict_to
        adjacency, restrict_indices = restrict_adjacency(
            adata, restrict_key, restrict_categories=restrict_categories,
            adjacency=adjacency)
    if flavor == "igraph":
        directed = False
    g = get_igraph_from_adjacency(adjacency, directed=directed)
    weights = np.array(g.es["weight"]).astype(np.float64) if use_weights else None
    if flavor == "vtraag":
        groups, backend = _louvain_partition(
            g, resolution=resolution, random_state=random_state,
            use_weights=use_weights, weights=weights,
            partition_type=partition_type, partition_kwargs=partition_kwargs)
    else:
        groups = np.array(g.community_multilevel(weights=weights).membership)
        backend = "igraph.community_multilevel"
    global LAST_BACKEND
    LAST_BACKEND = backend
    if restrict_to is not None:
        if key_added == "louvain":
            key_added += "_R"
        groups = rename_groups(adata, key_added=key_added, restrict_key=restrict_key,
                               restrict_categories=restrict_categories,
                               restrict_indices=restrict_indices, groups=groups)
    groups = np.asarray(groups)
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=natsorted(map(str, np.unique(groups))),
    )
    adata.uns[key_added] = {"params": dict(resolution=resolution, random_state=random_state)}
    return adata if copy else None
