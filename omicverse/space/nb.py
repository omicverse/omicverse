"""Statistics of how spatial neighbours are arranged — ``ov.space.nb``.

The graph and what lives on it. Build the graph once with
:func:`~omicverse.space.spatial_neighbors`, then ask questions of it::

    ov.space.spatial_neighbors(adata, n_neighs=6, coord_type='grid')
    ov.space.nb.enrichment(adata, 'cell_type')
    ov.space.nb.co_occurrence(adata, 'cell_type')

Every name here is also available flat on ``ov.space`` under its original
spelling, so existing code keeps working.
"""
from ._svg import spatial_neighbors as neighbors
from ._neighborhood import (
    centrality_scores as centrality,
    co_occurrence,
    interaction_matrix,
    mask_graph as mask,
    nhood_enrichment as enrichment,
    ripley,
    sepal,
    sliding_window,
    var_by_distance,
)

__all__ = [
    "neighbors",
    "enrichment",
    "co_occurrence",
    "interaction_matrix",
    "centrality",
    "ripley",
    "sepal",
    "mask",
    "sliding_window",
    "var_by_distance",
]
