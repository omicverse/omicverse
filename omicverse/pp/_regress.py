"""Native linear regress-out of covariates (replaces ``scanpy.pp.regress_out``).

Algorithm ported from scanpy (``scanpy/preprocessing/_simple.py``,
BSD-3-Clause, Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab),
vectorised with NumPy instead of a per-gene statsmodels GLM loop. The Gaussian
GLM with identity link that scanpy fits per gene is ordinary least squares, so
the residuals are the projection of each gene onto the orthogonal complement of
the design matrix:

* one categorical key   -> residual = x - (per-category mean of x)
* continuous keys       -> design = [1, obs[keys]]; residual = x - R (R^+ x)

Constant genes on the GLM path (categorical / singular design) are left
untouched exactly like scanpy does.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import issparse


def _to_float_dense(x):
    if issparse(x):
        x = x.toarray()
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32 if x.dtype.itemsize <= 4 else np.float64)
    return x


def _residuals(x: np.ndarray, design: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """``x - design @ pinv(design) @ x`` computed in column blocks."""
    pinv = np.linalg.pinv(design.astype(np.float64))
    out = np.empty_like(x)
    for start in range(0, x.shape[1], chunk):
        blk = x[:, start:start + chunk].astype(np.float64, copy=False)
        out[:, start:start + chunk] = blk - design @ (pinv @ blk)
    return out


def regress_out(
    adata,
    keys: str | Sequence[str],
    *,
    layer: str | None = None,
    n_jobs: int | None = None,
    copy: bool = False,
    inplace: bool = True,
):
    """Regress out (mostly) unwanted sources of variation.

    Same semantics and results as ``scanpy.pp.regress_out``: simple linear
    regression per gene, inspired by Seurat's ``regressOut``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    keys
        ``adata.obs`` column(s) to regress on. A single categorical column
        regresses on the per-category gene means.
    layer
        Layer to regress (``None`` uses ``adata.X``).
    n_jobs
        Accepted for API compatibility; the vectorised solver is BLAS-parallel.
    copy
        Return a corrected copy of ``adata``.
    inplace
        If ``False`` return the corrected dense matrix without modifying
        ``adata`` (overrides ``copy``).

    Returns
    -------
    ``None``, a copied AnnData (``copy=True``) or an ``ndarray``
    (``inplace=False``).
    """
    if isinstance(keys, str):
        keys = [keys]
    keys = list(keys)
    if not inplace:
        target = adata
    else:
        target = adata.copy() if copy else adata
        if getattr(target, "is_view", False):
            target._init_as_actual(target.copy())

    x = target.X if layer is None else target.layers[layer]
    x = _to_float_dense(x)

    obs = target.obs
    if keys and keys[0] in obs and isinstance(obs[keys[0]].dtype, pd.CategoricalDtype):
        if len(keys) > 1:
            raise ValueError(
                "If providing categorical variable, only a single one is allowed. "
                "For this one we regress on the mean for each category."
            )
        codes = obs[keys[0]].cat.codes.to_numpy()
        n_cat = len(obs[keys[0]].cat.categories)
        x = np.array(x, copy=True)
        res = x.astype(np.float64)
        for c in range(n_cat):
            m = codes == c
            if m.any():
                # scanpy stores the category means as float32 regressors
                means = x[m].mean(axis=0).astype(np.float32)
                res[m] -= means
        const = ~(x != x[0]).any(axis=0)
        res[:, const] = x[:, const]
        res = res.astype(x.dtype, copy=False)
    else:
        reg = obs[keys] if keys else obs.copy()
        reg = reg.copy()
        reg.insert(0, "ones", 1.0)
        design = reg.to_numpy(dtype=np.float64)
        if np.linalg.det(design.T @ design) != 0:
            res = _residuals(x, design).astype(x.dtype, copy=False)
        else:
            res = _residuals(x, design)
            const = ~(x != x[0]).any(axis=0)
            res[:, const] = x[:, const]

    if not inplace:
        return res
    if layer is None:
        target.X = res
    else:
        target.layers[layer] = res
    return target if copy else None
