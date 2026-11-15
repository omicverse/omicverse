"""Per-cell / per-gene QC metrics (native port of ``scanpy.pp.calculate_qc_metrics``).

Ported from scanpy ``scanpy/preprocessing/_qc.py`` (BSD-3-Clause,
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab). Output column
names, dtypes and values are identical to scanpy; the top-k segment proportions
are vectorised with NumPy (row-wise sort of the CSR data) instead of the numba
per-row kernel.
"""

from __future__ import annotations

from typing import Collection

import numpy as np
import pandas as pd
from scipy import sparse

from .._registry import register_function


def _choose_mtx_rep(adata, *, use_raw: bool = False, layer: str | None = None):
    if use_raw and layer is not None:
        raise ValueError(
            "Cannot use expression from both layer and raw. You provided:"
            f"use_raw={use_raw!r} and layer={layer!r}"
        )
    if layer is not None:
        return adata.layers[layer]
    if use_raw:
        return adata.raw.X
    return adata.X


def _axis_nnz(x, axis: int) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.getnnz(axis=axis)) if hasattr(x, "getnnz") else np.diff(
            (x.tocsr() if axis == 1 else x.tocsc()).indptr)
    return np.count_nonzero(x, axis=axis)


def _axis_sum(x, axis: int) -> np.ndarray:
    return np.ravel(np.asarray(x.sum(axis=axis)))


def _mean_axis0(x) -> np.ndarray:
    # scanpy's _get_mean_var: float64 accumulation
    if sparse.issparse(x):
        return np.ravel(np.asarray(x.sum(axis=0, dtype=np.float64))) / x.shape[0]
    return np.mean(x, axis=0, dtype=np.float64)


def _top_segment_proportions(mtx, ns) -> np.ndarray:
    """Fraction of each row's total held by its top ``n`` entries, for each n."""
    ns = np.sort(np.asarray(ns, dtype=np.int64))
    if not (ns.max() <= mtx.shape[1] and ns.min() > 0):
        raise IndexError("Positions outside range of features.")
    maxn = int(ns[-1])
    n_obs = mtx.shape[0]
    if sparse.issparse(mtx):
        mtx = sparse.csr_matrix(mtx)
        data = mtx.data
        rows = np.repeat(np.arange(n_obs), np.diff(mtx.indptr))
        sums = np.bincount(rows, weights=data, minlength=n_obs)
        # sort entries by (row asc, value desc) and keep the first maxn per row
        order = np.lexsort((-data, rows))
        rank = np.arange(data.size) - np.repeat(mtx.indptr[:-1], np.diff(mtx.indptr))
        keep = rank < maxn
        top = np.zeros((n_obs, maxn), dtype=np.float64)
        top[rows[order][keep], rank[keep]] = data[order][keep]
    else:
        mtx = np.asarray(mtx)
        sums = mtx.sum(axis=1, dtype=np.float64)
        top = np.empty((n_obs, maxn), dtype=np.float64)
        step = max(1, (1 << 24) // max(1, mtx.shape[1]))
        for s in range(0, n_obs, step):
            blk = mtx[s:s + step]
            part = -np.partition(-blk, maxn - 1, axis=1)[:, :maxn] if maxn < blk.shape[1] else blk
            top[s:s + step] = -np.sort(-part, axis=1)[:, :maxn]
    cum = np.cumsum(top, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cum[:, ns - 1] / sums[:, None]


def describe_obs(
    adata,
    *,
    expr_type: str = "counts",
    var_type: str = "genes",
    qc_vars: Collection[str] = (),
    percent_top: Collection[int] | None = (50, 100, 200, 500),
    layer: str | None = None,
    use_raw: bool = False,
    log1p: bool | None = True,
    inplace: bool = False,
    x=None,
) -> pd.DataFrame | None:
    """Per-observation QC metrics (see :func:`calculate_qc_metrics`)."""
    if x is None:
        x = _choose_mtx_rep(adata, use_raw=use_raw, layer=layer)
        if isinstance(x, sparse.coo_matrix):
            x = sparse.csr_matrix(x)
        if sparse.issparse(x):
            x.eliminate_zeros()
    obs_metrics = pd.DataFrame(index=adata.obs_names)
    obs_metrics[f"n_{var_type}_by_{expr_type}"] = _axis_nnz(x, axis=1)
    if log1p:
        obs_metrics[f"log1p_n_{var_type}_by_{expr_type}"] = np.log1p(
            obs_metrics[f"n_{var_type}_by_{expr_type}"])
    obs_metrics[f"total_{expr_type}"] = _axis_sum(x, axis=1)
    if log1p:
        obs_metrics[f"log1p_total_{expr_type}"] = np.log1p(obs_metrics[f"total_{expr_type}"])
    if percent_top:
        percent_top = sorted(percent_top)
        proportions = _top_segment_proportions(x, percent_top)
        for i, n in enumerate(percent_top):
            obs_metrics[f"pct_{expr_type}_in_top_{n}_{var_type}"] = proportions[:, i] * 100
    for qc_var in qc_vars:
        obs_metrics[f"total_{expr_type}_{qc_var}"] = _axis_sum(
            x[:, np.asarray(adata.var[qc_var].values, dtype=bool)], axis=1)
        if log1p:
            obs_metrics[f"log1p_total_{expr_type}_{qc_var}"] = np.log1p(
                obs_metrics[f"total_{expr_type}_{qc_var}"])
        obs_metrics[f"pct_{expr_type}_{qc_var}"] = (
            obs_metrics[f"total_{expr_type}_{qc_var}"] / obs_metrics[f"total_{expr_type}"] * 100)
    if inplace:
        adata.obs[obs_metrics.columns] = obs_metrics
        return None
    return obs_metrics


def describe_var(
    adata,
    *,
    expr_type: str = "counts",
    var_type: str = "genes",
    layer: str | None = None,
    use_raw: bool = False,
    inplace: bool = False,
    log1p: bool = True,
    x=None,
) -> pd.DataFrame | None:
    """Per-variable QC metrics (see :func:`calculate_qc_metrics`)."""
    if x is None:
        x = _choose_mtx_rep(adata, use_raw=use_raw, layer=layer)
        if isinstance(x, sparse.coo_matrix):
            x = sparse.csr_matrix(x)
        if sparse.issparse(x):
            x.eliminate_zeros()
    var_names = adata.raw.var_names if (use_raw and x.shape[1] != adata.n_vars) else adata.var_names
    var_metrics = pd.DataFrame(index=var_names)
    var_metrics[f"n_cells_by_{expr_type}"] = _axis_nnz(x, axis=0)
    var_metrics[f"mean_{expr_type}"] = _mean_axis0(x)
    if log1p:
        var_metrics[f"log1p_mean_{expr_type}"] = np.log1p(var_metrics[f"mean_{expr_type}"])
    var_metrics[f"pct_dropout_by_{expr_type}"] = (
        1 - var_metrics[f"n_cells_by_{expr_type}"] / x.shape[0]) * 100
    var_metrics[f"total_{expr_type}"] = _axis_sum(x, axis=0)
    if log1p:
        var_metrics[f"log1p_total_{expr_type}"] = np.log1p(var_metrics[f"total_{expr_type}"])
    if inplace:
        adata.var[var_metrics.columns] = var_metrics
        return None
    return var_metrics


@register_function(
    aliases=["calculate_qc_metrics", "qc指标计算", "质控指标", "calc_qc_metrics"],
    category="preprocessing",
    description=(
        "Calculate scanpy-compatible quality-control metrics (n_genes_by_counts, "
        "total_counts, pct_counts_in_top_N_genes, pct_counts_<qc_var>, per-gene "
        "n_cells_by_counts / mean_counts / pct_dropout_by_counts) without scanpy."
    ),
    prerequisites={},
    requires={},
    produces={
        "obs": ["n_genes_by_counts", "total_counts"],
        "var": ["n_cells_by_counts", "mean_counts", "pct_dropout_by_counts", "total_counts"],
    },
    auto_fix="none",
    examples=[
        "adata.var['mt'] = adata.var_names.str.startswith('MT-')",
        "ov.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)",
    ],
    related=["pp.qc", "pp.qc_metrics"],
)
def calculate_qc_metrics(
    adata,
    *,
    expr_type: str = "counts",
    var_type: str = "genes",
    qc_vars: Collection[str] | str = (),
    percent_top: Collection[int] | None = (50, 100, 200, 500),
    layer: str | None = None,
    use_raw: bool = False,
    inplace: bool = False,
    log1p: bool = True,
    parallel: bool | None = None,
):
    """Calculate quality control metrics (drop-in for ``scanpy.pp.calculate_qc_metrics``).

    Parameters
    ----------
    adata
        Annotated data matrix.
    expr_type
        Name of kind of values in X (used in column names).
    var_type
        The kind of thing the variables are (used in column names).
    qc_vars
        Boolean columns of ``adata.var`` to compute proportions for, e.g. ``'mt'``.
    percent_top
        Ranks for which the cumulative proportion of counts held by the top
        genes is reported. ``None`` / empty to skip.
    layer
        Layer to use instead of ``adata.X``.
    use_raw
        Use ``adata.raw.X``.
    inplace
        Write metrics into ``adata.obs`` / ``adata.var`` instead of returning them.
    log1p
        Also add ``log1p_*`` versions of the count metrics.
    parallel
        Deprecated, no effect.

    Returns
    -------
    ``(obs_metrics, var_metrics)`` DataFrames if ``inplace=False`` else ``None``.
    Columns: ``n_genes_by_counts``, ``total_counts``,
    ``pct_counts_in_top_{n}_genes``, ``total_counts_{qc_var}``,
    ``pct_counts_{qc_var}`` (+ ``log1p_*``) in obs and ``n_cells_by_counts``,
    ``mean_counts``, ``pct_dropout_by_counts``, ``total_counts`` (+ ``log1p_*``)
    in var.
    """
    if parallel is not None:
        import warnings
        warnings.warn("Argument `parallel` is deprecated, and currently has no effect.",
                      FutureWarning, stacklevel=2)
    x = _choose_mtx_rep(adata, use_raw=use_raw, layer=layer)
    if isinstance(x, sparse.coo_matrix):
        x = sparse.csr_matrix(x)
    if sparse.issparse(x):
        x.eliminate_zeros()
    elif not isinstance(x, np.ndarray):
        # backed / OOM arrays: materialise once
        x = x.to_memory() if hasattr(x, "to_memory") else np.asarray(x)
    if isinstance(qc_vars, str):
        qc_vars = [qc_vars]
    obs_metrics = describe_obs(adata, expr_type=expr_type, var_type=var_type,
                               qc_vars=qc_vars, percent_top=percent_top,
                               inplace=inplace, x=x, log1p=log1p)
    var_metrics = describe_var(adata, expr_type=expr_type, var_type=var_type,
                               inplace=inplace, x=x, log1p=log1p, use_raw=use_raw)
    if not inplace:
        return obs_metrics, var_metrics
    return None
