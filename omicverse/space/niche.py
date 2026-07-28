"""Recurrent multicellular structure — ``ov.space.niche``.

Three ways to ask the same question, kept apart because they are not the same
question::

    ov.space.niche.neighborhood(adata, 'cell_type')   # cluster the local mix
    ov.space.niche.utag(adata, use_rep='X_pca')       # cluster smoothed expression
    ov.space.niche.cellcharter(adata, ...)            # GMM on the aggregated graph
    ov.space.niche.molecular(adata, n_factors=10)     # NMF over deconvolved abundance

See :mod:`omicverse.space._niche` for what separates them.
"""
from ._niche import molecular, neighborhood, utag
from ._cellcharter import cellcharter

__all__ = ["neighborhood", "utag", "cellcharter", "molecular"]
