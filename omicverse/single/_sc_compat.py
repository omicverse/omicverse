"""Scanpy-equivalent helpers built on omicverse internals.

Kept for the ``from ._sc_compat import ...`` imports in ``omicverse.single``;
the implementations live in :mod:`omicverse._sc_compat`.
"""

from .._sc_compat import *  # noqa: F401,F403
from .._sc_compat import (  # noqa: F401  (names without __all__ / leading underscore)
    _n_vars_used, default_20, palette_28, palette_102, scale_X, pca_X, neighbors, leiden,
    umap, rank_genes_groups, rank_genes_groups_df, paga, paga_pos,
)
