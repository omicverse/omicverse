"""Force-directed graph layouts (native port of ``scanpy.tl.draw_graph``).

Ported from scanpy ``scanpy/tools/_draw_graph.py`` (BSD-3-Clause,
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab). Same layouts,
seeding and outputs (``obsm['X_draw_graph_<layout>']``, ``uns['draw_graph']``).
ForceAtlas2 (``layout='fa'``) uses the optional ``fa2_modified`` package and,
exactly like scanpy, falls back to igraph's Fruchterman-Reingold when it is
not installed.
"""

from __future__ import annotations

import random
import warnings
from importlib.util import find_spec

import numpy as np

from .._registry import register_function
from ._compat import get_init_pos_from_paga
from ._leiden import _choose_graph, get_igraph_from_adjacency

_LAYOUTS = ("fr", "drl", "kk", "grid_fr", "lgl", "rt", "rt_circular", "fa")


def fa2_positions(adjacency, init_coords: np.ndarray, **kwds):
    from fa2_modified import ForceAtlas2

    forceatlas2 = ForceAtlas2(
        outboundAttractionDistribution=False,
        linLogMode=False,
        adjustSizes=False,
        edgeWeightInfluence=1.0,
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,
        verbose=False,
    )
    if "maxiter" in kwds:
        iterations = kwds["maxiter"]
    elif "iterations" in kwds:
        iterations = kwds["iterations"]
    else:
        iterations = 500
    return forceatlas2.forceatlas2(adjacency, pos=init_coords, iterations=iterations)


def coerce_fa2_layout(layout: str) -> str:
    if layout != "fa":
        return layout
    if find_spec("fa2_modified") is None:
        warnings.warn(
            "Package 'fa2-modified' is not installed, falling back to layout 'fr'. "
            "To use the faster and better ForceAtlas2 layout, install package "
            "'fa2-modified' (`pip install fa2-modified`).",
            UserWarning, stacklevel=3)
        return "fr"
    return "fa"


@register_function(
    aliases=["draw_graph", "force_directed", "forceatlas2", "力导向图", "fa_layout"],
    category="preprocessing",
    description=(
        "Force-directed layout of the kNN graph (ForceAtlas2 via fa2-modified, else "
        "igraph Fruchterman-Reingold / other igraph layouts); writes "
        "obsm['X_draw_graph_<layout>']."
    ),
    prerequisites={"functions": ["neighbors"]},
    requires={"uns": ["neighbors"], "obsp": ["connectivities"]},
    produces={"obsm": ["X_draw_graph_fa"], "uns": ["draw_graph"]},
    auto_fix="none",
    examples=[
        "ov.pp.draw_graph(adata)",
        "ov.pp.draw_graph(adata, layout='fr')",
        "ov.pl.embedding(adata, basis='X_draw_graph_fa', color='leiden')",
    ],
    related=["pp.neighbors", "pp.umap"],
)
def draw_graph(
    adata,
    layout: str = "fa",
    *,
    init_pos: str | bool | None = None,
    root: int | None = None,
    random_state=0,
    n_jobs: int | None = None,
    adjacency=None,
    key_added_ext: str | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False,
    **kwds,
):
    """Force-directed graph drawing (drop-in for ``scanpy.tl.draw_graph``).

    Parameters
    ----------
    adata
        Annotated data matrix with a neighbour graph.
    layout
        ``'fa'`` (ForceAtlas2) or an igraph layout: ``'fr'``, ``'grid_fr'``,
        ``'kk'``, ``'lgl'``, ``'drl'``, ``'rt'``, ``'rt_circular'``.
    init_pos
        ``'paga'``/``True``, ``None``/``False`` or a 2-d ``.obsm`` key used to
        initialise coordinates (random by default).
    root
        Root for tree layouts.
    random_state
        Seed for the random initialisation.
    n_jobs
        Unused (API compatibility).
    adjacency
        Adjacency matrix (default: neighbour connectivities).
    key_added_ext
        Suffix of the ``.obsm`` key (default: ``layout``).
    neighbors_key, obsp
        Where to find the graph.
    copy
        Return a copy instead of writing to ``adata``.
    **kwds
        Passed to the igraph layout (e.g. ``maxiter``) or ``iterations`` for FA2.

    Returns
    -------
    ``None`` or the copy; sets ``adata.obsm['X_draw_graph_<layout>']`` and
    ``adata.uns['draw_graph']['params']``.
    """
    if layout not in _LAYOUTS:
        raise ValueError(f"Provide a valid layout, one of {_LAYOUTS}.")
    adata = adata.copy() if copy else adata
    if adjacency is None:
        adjacency = _choose_graph(adata, obsp, neighbors_key)
    if init_pos in adata.obsm:
        init_coords = adata.obsm[init_pos]
    elif init_pos == "paga" or init_pos:
        init_coords = get_init_pos_from_paga(adata, adjacency, random_state=random_state,
                                             neighbors_key=neighbors_key, obsp=obsp)
    else:
        np.random.seed(random_state)
        init_coords = np.random.random((adjacency.shape[0], 2))
    layout = coerce_fa2_layout(layout)
    if layout == "fa":
        positions = np.array(fa2_positions(adjacency, init_coords, **kwds))
    else:
        # igraph uses Python's `random` module, not numpy's seed
        random.seed(random_state)
        g = get_igraph_from_adjacency(adjacency)
        if layout in {"fr", "drl", "kk", "grid_fr"}:
            ig_layout = g.layout(layout, seed=init_coords.tolist(), **kwds)
        elif "rt" in layout:
            if root is not None:
                root = [root]
            ig_layout = g.layout(layout, root=root, **kwds)
        else:
            ig_layout = g.layout(layout, **kwds)
        positions = np.array(ig_layout.coords)
    adata.uns["draw_graph"] = {"params": dict(layout=layout, random_state=random_state)}
    adata.obsm[f"X_draw_graph_{key_added_ext or layout}"] = positions
    return adata if copy else None
