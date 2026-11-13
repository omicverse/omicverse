"""Scanpy-equivalent helpers built on omicverse internals.

Used by ``omicverse.space``, ``omicverse.bulk2single``, ``omicverse.genetics``
and ``omicverse.tl``. Several scanpy calls in those modules have public
omicverse counterparts with different defaults or storage conventions (e.g.
``ov.pp.scale`` writes ``layers['scaled']`` while ``sc.pp.scale`` overwrites
``X``; ``ov.pp.leiden`` defaults to the brisc backend while ``sc.tl.leiden``
uses leidenalg). The helpers below reproduce the scanpy *results* with
omicverse code so call sites keep their behaviour without importing scanpy.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


def scale_X(adata, *, zero_center: bool = True, max_value: Optional[float] = None):
    """``sc.pp.scale(adata)``: z-score ``adata.X`` in place."""
    from .pp._scale import scale_anndata
    scale_anndata(adata, zero_center=zero_center, max_value=max_value)


def pca_X(adata, n_comps: Optional[int] = None, *, svd_solver: Optional[str] = None,
          zero_center: bool = True, random_state=0, **kwargs):
    """``sc.pp.pca(adata)`` / ``sc.tl.pca(adata)`` on ``adata.X``.

    Writes ``obsm['X_pca']``, ``varm['PCs']`` and ``uns['pca']`` like scanpy
    (``var['highly_variable']`` is used as mask when present).
    ``svd_solver=None`` maps to scanpy's default, ``'arpack'``.
    """
    from .pp._pca import pca as _pca
    if svd_solver is None:
        svd_solver = "arpack"
    _pca(adata, n_comps=n_comps, layer=None, zero_center=zero_center,
         svd_solver=svd_solver, random_state=random_state, **kwargs)


def neighbors(adata, n_neighbors: int = 15, *, use_rep=None, n_pcs=None,
              random_state=0, **kwargs):
    """``sc.pp.neighbors`` with scanpy defaults (umap connectivities)."""
    from .pp import neighbors as _neighbors
    _neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, n_pcs=n_pcs,
               random_state=random_state, method="umap", **kwargs)


def leiden(adata, resolution: float = 1.0, *, key_added: str = "leiden",
           random_state=0, flavor: str = "leidenalg", n_iterations: int = -1,
           **kwargs):
    """``sc.tl.leiden`` with scanpy defaults.

    ``ov.pp.leiden`` would pick the brisc backend; ``flavor='leidenalg'`` and
    ``n_iterations=-1`` give scanpy's partition (exact C++ port of leidenalg
    when anndataoom-fast is available).
    """
    from .pp import leiden as _leiden
    _leiden(adata, resolution=resolution, random_state=random_state,
            key_added=key_added, flavor=flavor, n_iterations=n_iterations,
            **kwargs)


def umap(adata, **kwargs):
    """``sc.tl.umap`` (min_dist=0.5, spread=1, random_state=0)."""
    from .pp import umap as _umap
    kwargs.setdefault("random_state", 0)
    _umap(adata, **kwargs)


def _n_vars_used(adata, use_raw: Optional[bool], layer: Optional[str]) -> int:
    if use_raw is None:
        use_raw = adata.raw is not None and layer is None
    if use_raw and adata.raw is not None:
        return adata.raw.n_vars
    return adata.n_vars


def rank_genes_groups(
    adata,
    groupby: str,
    *,
    method: str = "t-test",
    groups: Union[str, Sequence[str]] = "all",
    reference: str = "rest",
    n_genes: Optional[int] = None,
    key_added: Optional[str] = None,
    use_raw: Optional[bool] = None,
    layer: Optional[str] = None,
    corr_method: str = "benjamini-hochberg",
    rankby_abs: bool = False,
    tie_correct: bool = False,
    pts: bool = False,
    **kwargs,
):
    """``sc.tl.rank_genes_groups`` with scanpy defaults, via
    :func:`omicverse.single.find_markers` (all genes kept, ``pts=False``,
    ``method='t-test'``)."""
    from .single._markers import find_markers
    if n_genes is None:
        n_genes = _n_vars_used(adata, use_raw, layer)
    find_markers(
        adata,
        groupby=groupby,
        method=method,
        n_genes=n_genes,
        key_added=key_added if key_added is not None else "rank_genes_groups",
        use_raw=use_raw,
        layer=layer,
        groups=groups,
        reference=reference,
        corr_method=corr_method,
        rankby_abs=rankby_abs,
        tie_correct=tie_correct,
        pts=pts,
        **kwargs,
    )


def paga(adata, groups: Optional[str] = None, *, model: str = "v1.2",
         neighbors_key: Optional[str] = None) -> None:
    """``sc.tl.paga(adata, groups)`` (connectivities and tree)."""
    from .utils._compat import PAGA
    if groups is None:
        for k in ("leiden", "louvain"):
            if k in adata.obs.columns:
                groups = k
                break
    if groups is None:
        raise ValueError(
            "You need to run `ov.pp.leiden` or `ov.pp.louvain` to compute "
            "community labels, or specify `groups='an_existing_key'`")
    if groups not in adata.obs.columns:
        raise KeyError(f"`groups` key {groups!r} not found in `adata.obs`.")
    if not isinstance(adata.obs[groups].dtype, pd.CategoricalDtype):
        adata.obs[groups] = adata.obs[groups].astype("category")
    p = PAGA(adata, groups, model=model, neighbors_key=neighbors_key)
    if "paga" not in adata.uns:
        adata.uns["paga"] = {}
    p.compute_connectivities()
    adata.uns["paga"]["connectivities"] = p.connectivities
    adata.uns["paga"]["connectivities_tree"] = p.connectivities_tree
    adata.uns[f"{groups}_sizes"] = np.array(p.ns)
    adata.uns["paga"]["groups"] = groups
