r"""``ov.tl.bonsai`` — public entry point for the Bonsai tree reconstruction.

The work lives in :mod:`omicverse.external.bonsai`; this module only registers
the function and records the citation, mirroring how ``ov.single.samap_integrate``
fronts the vendored SAMap.
"""
from __future__ import annotations

from typing import Optional

from .._settings import add_reference
from .._registry import register_function

__all__ = ["bonsai"]


@register_function(
    aliases=["bonsai", "Bonsai树", "bonsai tree", "细胞树重建"],
    category="tl",
    description="Reconstruct a maximum-likelihood Bonsai tree over cells, "
                "propagating per-feature measurement error.",
    examples=["ov.tl.bonsai(adata, use_rep='scaled|original|X_pca')"],
    related=["ov.pl.bonsai", "ov.pp.umap", "ov.pp.mde"],
    produces={"uns": ["bonsai"]},
)
def bonsai(adata, **kwargs):
    r"""Reconstruct a Bonsai tree over the cells of ``adata``.

    Bonsai infers the maximum-likelihood tree whose leaves are the observed
    cells, treating each cell as a Gaussian measurement in feature space. Where
    UMAP and t-SNE compress the data into two dimensions and let noise spread
    cells apart, Bonsai keeps the uncertainty: a cell measured imprecisely sits
    on a short branch near its parent instead of being pushed into a corner of
    the plot.

    .. warning::
       The vendored Bonsai core is licensed **CC BY-NC 4.0**, which prohibits
       use by commercial entities, including for research. See
       ``omicverse/external/bonsai/LICENSE-CC-BY-NC-4.0.md``.

    Parameters
    ----------
    adata
        Annotated data matrix; cells become the leaves of the tree.
    **kwargs
        Forwarded to :func:`omicverse.external.bonsai.run_bonsai` — notably
        ``use_rep``, ``sd_key``, ``n_cores``, ``results_dir`` and ``key_added``.
        See that function for the full signature.

    Returns
    -------
    :class:`anndata.AnnData` or None
        Writes ``adata.uns['bonsai']`` (``newick``, ``edges``, ``edge_lengths``,
        ``vert_ind``, ``vert_name``, ``leaf_of_obs``, ``results_dir``). Returns
        a copy when ``copy=True``.

    Examples
    --------
    >>> import omicverse as ov
    >>> ov.pp.pca(adata, layer='scaled', n_pcs=50)
    >>> ov.tl.bonsai(adata, use_rep='scaled|original|X_pca', n_cores=4)
    >>> ov.pl.bonsai(adata, color='leiden')
    """
    from ..external.bonsai import run_bonsai

    out = run_bonsai(adata, **kwargs)
    add_reference(adata if out is None else out, 'Bonsai',
                  'cell-state tree reconstruction with Bonsai')
    return out
