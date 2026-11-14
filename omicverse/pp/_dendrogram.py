"""Hierarchical clustering of groups (native port of ``scanpy.tl.dendrogram``).

Ported from scanpy ``scanpy/tools/_dendrogram.py`` (BSD-3-Clause,
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab). Writes the same
``adata.uns['dendrogram_<groupby>']`` dictionary consumed by dot/matrix/heatmap
plots.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_numeric_dtype
from scipy import sparse

from .._registry import register_function


def _choose_representation(adata, *, use_rep=None, n_pcs=None):
    """scanpy ``_choose_representation`` semantics (PCA if n_vars > 50)."""
    if use_rep is None and n_pcs == 0:
        use_rep = "X"
    if use_rep is None:
        if adata.n_vars > 50:
            if "X_pca" in adata.obsm:
                if n_pcs is not None and n_pcs > adata.obsm["X_pca"].shape[1]:
                    raise ValueError("`X_pca` does not have enough PCs.")
                return adata.obsm["X_pca"][:, :n_pcs]
            from ._pca import pca as _pca
            n_comps = n_pcs if n_pcs is not None else 50
            _pca(adata, n_comps=min(n_comps, min(adata.shape) - 1))
            return adata.obsm["X_pca"][:, :n_pcs]
        return adata.X
    if use_rep in adata.obsm and n_pcs is not None:
        if n_pcs > adata.obsm[use_rep].shape[1]:
            raise ValueError(
                f"{use_rep} does not have enough Dimensions. Provide a Representation "
                "with equal or more dimensions than`n_pcs` or lower `n_pcs` ")
        return adata.obsm[use_rep][:, :n_pcs]
    if use_rep in adata.obsm:
        return adata.obsm[use_rep]
    if use_rep == "X":
        return adata.X
    raise ValueError(f"Did not find {use_rep} in `.obsm.keys()`. You need to compute it first.")


def _prepare_dataframe(adata, var_names, groupby, *, use_raw=None, layer=None):
    """Categories x var_names frame (port of scanpy.plotting._anndata._prepare_dataframe)."""
    if use_raw is None:
        use_raw = adata.raw is not None and layer is None
    if isinstance(var_names, str):
        var_names = [var_names]
    if isinstance(groupby, str):
        groupby = [groupby]
    var_unique = list(np.unique(var_names))
    src_names = adata.raw.var_names if use_raw else adata.var_names
    X = adata.raw.X if use_raw else (adata.layers[layer] if layer is not None else adata.X)
    idx = src_names.get_indexer(var_unique)
    if (idx < 0).any():
        missing = [g for g, i in zip(var_unique, idx) if i < 0]
        raise KeyError(f"Could not find keys {missing} in var_names.")
    sub = X[:, idx]
    sub = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
    obs_tidy = pd.DataFrame(sub, index=adata.obs_names, columns=var_unique)
    if len(groupby) == 1 and is_numeric_dtype(adata.obs[groupby[0]]):
        categorical = pd.cut(adata.obs[groupby[0]], 7)
    elif len(groupby) == 1:
        categorical = adata.obs[groupby[0]].astype("category")
        categorical.name = groupby[0]
    else:
        from itertools import product
        gdf = adata.obs[groupby]
        categorical = gdf.astype(str).apply("_".join, axis=1).astype("category")
        categorical.name = "_".join(groupby)
        order = {"_".join(k): i for i, k in enumerate(
            product(*(gdf[g].cat.categories for g in groupby)))}
        categorical = categorical.cat.reorder_categories(
            sorted(categorical.cat.categories, key=lambda x: order[x]))
    obs_tidy = obs_tidy[list(var_names)].set_index(pd.CategoricalIndex(categorical))
    return obs_tidy.index.categories, obs_tidy


@register_function(
    aliases=["dendrogram", "层次聚类树", "cluster_dendrogram", "group_dendrogram"],
    category="preprocessing",
    description=(
        "Hierarchical clustering of the categories of an obs column on mean PCA (or "
        "gene) profiles; stores uns['dendrogram_<groupby>'] used to order dot/matrix plots."
    ),
    prerequisites={"optional_functions": ["pca"]},
    requires={},
    produces={"uns": ["dendrogram_<groupby>"]},
    auto_fix="none",
    examples=["ov.pp.dendrogram(adata, groupby='leiden')"],
    related=["pl.dotplot", "pl.heatmap"],
)
def dendrogram(
    adata,
    groupby: str | Sequence[str],
    *,
    n_pcs: int | None = None,
    use_rep: str | None = None,
    var_names: Sequence[str] | None = None,
    use_raw: bool | None = None,
    cor_method: str = "pearson",
    linkage_method: str = "complete",
    optimal_ordering: bool = False,
    key_added: str | None = None,
    inplace: bool = True,
) -> dict[str, Any] | None:
    """Hierarchical clustering of ``groupby`` categories (drop-in for ``scanpy.tl.dendrogram``).

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        Categorical ``adata.obs`` column(s).
    n_pcs, use_rep
        Representation to average (default ``X_pca`` if ``n_vars > 50``).
    var_names
        If given, average gene expression instead of a representation. As in
        scanpy, all genes (of ``adata.raw`` if ``use_raw`` else ``adata``) are
        then used, not only the listed ones.
    use_raw
        Use ``adata.raw`` with ``var_names``.
    cor_method
        Correlation method for :meth:`pandas.DataFrame.corr`.
    linkage_method
        :func:`scipy.cluster.hierarchy.linkage` method.
    optimal_ordering
        Passed to ``linkage``.
    key_added
        ``adata.uns`` key (default ``'dendrogram_' + '_'.join(groupby)``).
    inplace
        Store in ``adata.uns`` (else return the dict).

    Returns
    -------
    Dict with ``linkage``, ``groupby``, ``use_rep``, ``cor_method``,
    ``linkage_method``, ``categories_ordered``, ``categories_idx_ordered``,
    ``dendrogram_info``, ``correlation_matrix``.
    """
    if isinstance(groupby, str):
        groupby = [groupby]
    for group in groupby:
        if group not in adata.obs:
            raise ValueError(
                "groupby has to be a valid observation. "
                f"Given value: {group}, valid observations: {adata.obs.columns.tolist()}")
        if not isinstance(adata.obs[group].dtype, CategoricalDtype):
            raise ValueError(
                "groupby has to be a categorical observation. "
                f"Given value: {group}, Column type: {adata.obs[group].dtype}")

    if var_names is None:
        rep = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
        if sparse.issparse(rep):
            rep = rep.toarray()
        rep_df = pd.DataFrame(np.asarray(rep))
        categorical = adata.obs[groupby[0]]
        if len(groupby) > 1:
            for group in groupby[1:]:
                categorical = (categorical.astype(str) + "_"
                               + adata.obs[group].astype(str)).astype("category")
        categorical.name = "_".join(groupby)
        rep_df.set_index(categorical, inplace=True)
        categories = rep_df.index.categories
    else:
        # NB: scanpy (<=1.11) passes *all* gene names here, not ``var_names``
        # (``var_names`` only switches to the gene-expression path). Mirrored
        # for result parity with ``scanpy.tl.dendrogram``.
        gene_names = adata.raw.var_names if use_raw else adata.var_names
        categories, rep_df = _prepare_dataframe(adata, gene_names, groupby, use_raw=use_raw)

    mean_df = rep_df.groupby(level=0, observed=True).mean().loc[categories]

    import scipy.cluster.hierarchy as sch
    from scipy.spatial import distance

    corr_matrix = mean_df.T.corr(method=cor_method).clip(-1, 1)
    corr_condensed = distance.squareform(1 - corr_matrix)
    z_var = sch.linkage(corr_condensed, method=linkage_method, optimal_ordering=optimal_ordering)
    dendro_info = sch.dendrogram(z_var, labels=list(categories), no_plot=True)

    dat = dict(
        linkage=z_var,
        groupby=groupby,
        use_rep=use_rep,
        cor_method=cor_method,
        linkage_method=linkage_method,
        categories_ordered=dendro_info["ivl"],
        categories_idx_ordered=dendro_info["leaves"],
        dendrogram_info=dendro_info,
        correlation_matrix=corr_matrix.values,
    )
    if inplace:
        if key_added is None:
            key_added = f"dendrogram_{'_'.join(groupby)}"
        adata.uns[key_added] = dat
        return None
    return dat
