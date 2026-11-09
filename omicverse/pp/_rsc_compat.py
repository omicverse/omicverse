"""Calls into rapids-singlecell that work across its releases.

omicverse's GPU mode calls rapids-singlecell (``rsc``), whose API changed
between releases:

* ``pp.filter_cells`` / ``pp.filter_genes``: up to 0.11 they take
  ``qc_var`` / ``min_count`` / ``max_count`` and filter on an existing
  ``.obs`` / ``.var`` column (which may be stale after earlier filtering);
  0.12 and later take ``min_genes`` / ``max_genes`` / ``min_counts`` /
  ``max_counts`` (and ``min_cells`` / ``max_cells`` for genes), like Scanpy.
* ``pp.normalize_total``: ``exclude_highly_expressed`` / ``max_fraction``
  exist from 0.14 on, and ``target_sum=None`` means the median of the
  non-zero per-cell counts from 0.14 on but the median over all cells
  (zeros included) before; the wrapper always uses the non-zero median.
* ``pp.filter_cells`` / ``pp.filter_genes`` (0.12+) accept one threshold
  per call, as in Scanpy.
* ``pp.pca``: ``use_highly_variable`` is rejected from 0.17 on (use
  ``mask_var``).

Each wrapper calls rsc directly when the installed release has the
argument, and otherwise computes the same result on the GPU itself (Scanpy
semantics), so omicverse behaves the same with any rsc version.
"""
from __future__ import annotations

import inspect
from functools import lru_cache

import numpy as np


def _rsc():
    import rapids_singlecell as rsc
    return rsc


@lru_cache(maxsize=None)
def _params(qualname: str) -> frozenset:
    obj = _rsc()
    for part in qualname.split("."):
        obj = getattr(obj, part)
    try:
        return frozenset(inspect.signature(obj).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        return frozenset()


def _counts(X, axis: int, nonzero: bool):
    """Per-row (axis=1) or per-column (axis=0) sums or nonzero counts, on the GPU."""
    import cupy as cp
    import cupyx.scipy.sparse as csp
    if csp.issparse(X):
        Y = X.copy()
        if nonzero:
            Y.data = (Y.data > 0).astype(cp.float32)
        out = Y.sum(axis=axis)
    else:
        out = (X > 0).sum(axis=axis) if nonzero else X.sum(axis=axis)
    return cp.asnumpy(cp.asarray(out)).ravel()


def _bounds_mask(values, lo, hi):
    keep = np.ones(values.shape[0], dtype=bool)
    if lo is not None:
        keep &= values >= lo
    if hi is not None:
        keep &= values <= hi
    return keep


def filter_cells(adata, *, min_genes=None, max_genes=None, min_counts=None, max_counts=None):
    """``rsc.pp.filter_cells`` with Scanpy arguments, for any rsc release."""
    if "min_genes" in _params("pp.filter_cells"):
        # one threshold per call; each cell's fate depends on its own counts only
        for k, v in dict(min_genes=min_genes, max_genes=max_genes,
                         min_counts=min_counts, max_counts=max_counts).items():
            if v is not None:
                _rsc().pp.filter_cells(adata, **{k: v})
        return
    keep = np.ones(adata.n_obs, dtype=bool)
    if min_genes is not None or max_genes is not None:
        keep &= _bounds_mask(_counts(adata.X, 1, True), min_genes, max_genes)
    if min_counts is not None or max_counts is not None:
        keep &= _bounds_mask(_counts(adata.X, 1, False), min_counts, max_counts)
    if not keep.all():
        adata._inplace_subset_obs(keep)


def filter_genes(adata, *, min_cells=None, max_cells=None, min_counts=None, max_counts=None):
    """``rsc.pp.filter_genes`` with Scanpy arguments, for any rsc release."""
    if "min_cells" in _params("pp.filter_genes"):
        for k, v in dict(min_cells=min_cells, max_cells=max_cells,
                         min_counts=min_counts, max_counts=max_counts).items():
            if v is not None:
                _rsc().pp.filter_genes(adata, **{k: v})
        return
    keep = np.ones(adata.n_vars, dtype=bool)
    if min_cells is not None or max_cells is not None:
        keep &= _bounds_mask(_counts(adata.X, 0, True), min_cells, max_cells)
    if min_counts is not None or max_counts is not None:
        keep &= _bounds_mask(_counts(adata.X, 0, False), min_counts, max_counts)
    if not keep.all():
        adata._inplace_subset_var(keep)


def normalize_total(adata, *, target_sum=None, exclude_highly_expressed=False,
                    max_fraction=0.05, layer=None):
    """``rsc.pp.normalize_total`` incl. ``exclude_highly_expressed``, for any rsc release.

    Without native support the size factors are computed as Scanpy does: a
    gene is excluded if it exceeds ``max_fraction`` of the total counts of
    any cell; each cell is scaled to ``target_sum`` (default: the median of
    the non-zero size factors).
    """
    params = _params("pp.normalize_total")
    native = "exclude_highly_expressed" in params
    if not exclude_highly_expressed and not native and target_sum is None:
        # releases before 0.14 take the median over all cells, zeros included
        import cupy as cp
        X = adata.X if layer is None else adata.layers[layer]
        totals = cp.asarray(X.sum(axis=1)).ravel()
        nz = totals[totals > 0]
        target_sum = float(cp.median(nz)) if nz.size else 1.0
    if not exclude_highly_expressed or native:
        kw = {"target_sum": target_sum}
        if layer is not None:
            kw["layer"] = layer
        if exclude_highly_expressed:
            kw.update(exclude_highly_expressed=True, max_fraction=max_fraction)
        _rsc().pp.normalize_total(adata, **kw)
        return
    import cupy as cp
    import cupyx.scipy.sparse as csp
    X = adata.X if layer is None else adata.layers[layer]
    totals = cp.asarray(X.sum(axis=1)).ravel()
    if csp.issparse(X):
        coo = X.tocoo()
        hi = coo.data > max_fraction * totals[coo.row]
        bad = cp.zeros(X.shape[1], dtype=bool)
        bad[coo.col[hi]] = True
        # counts over the kept genes = totals - counts of the excluded genes
        excl = bad[coo.col]
        counts = totals - cp.bincount(coo.row[excl], weights=coo.data[excl].astype(cp.float64),
                                      minlength=X.shape[0])
    else:
        keep = ~(X > max_fraction * totals[:, None]).any(axis=0)
        counts = X[:, keep].sum(axis=1)
    counts = counts.astype(cp.float64)
    if target_sum is None:
        nz = counts[counts > 0]
        target_sum = float(cp.median(nz)) if nz.size else 1.0
    scale = cp.where(counts > 0, target_sum / cp.where(counts > 0, counts, 1.0), 1.0)
    if csp.issparse(X):
        X = csp.diags(scale.astype(X.dtype)) @ X
    else:
        X = X * scale[:, None].astype(X.dtype)
    if layer is None:
        adata.X = X
    else:
        adata.layers[layer] = X


def pca_kwargs(kwargs: dict) -> dict:
    """Translate ``use_highly_variable`` for rsc releases that only take ``mask_var``."""
    kwargs = dict(kwargs)
    if "use_highly_variable" in kwargs and "use_highly_variable" not in _params("pp.pca"):
        use_hv = kwargs.pop("use_highly_variable")
        if use_hv and "mask_var" not in kwargs:
            kwargs["mask_var"] = "highly_variable"
        elif use_hv is False and "mask_var" not in kwargs:
            kwargs["mask_var"] = None
    return kwargs
