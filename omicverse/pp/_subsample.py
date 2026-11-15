"""Random subsampling of observations (native port of ``scanpy.pp.subsample``).

Ported from scanpy ``scanpy/preprocessing/_simple.py`` (BSD-3-Clause,
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab). Uses the same
legacy NumPy global RNG draw (``np.random.seed(random_state)`` then
``np.random.choice(n_obs, n, replace=False)``), so the selected cells are
identical to scanpy for the same ``random_state``.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from .._registry import register_function


@register_function(
    aliases=["subsample", "下采样", "downsample_cells", "随机抽样细胞", "sample_cells"],
    category="preprocessing",
    description="Randomly subsample cells to a fraction or a fixed number (scanpy-compatible RNG).",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "ov.pp.subsample(adata, fraction=0.1)",
        "adata_small = ov.pp.subsample(adata, n_obs=2000, copy=True)",
    ],
    related=["pp.filter_cells"],
)
def subsample(
    data,
    fraction: float | None = None,
    *,
    n_obs: int | None = None,
    random_state=0,
    copy: bool = False,
):
    """Subsample to a fraction or number of observations (drop-in for ``scanpy.pp.subsample``).

    Parameters
    ----------
    data
        AnnData or an array / sparse matrix (cells x genes).
    fraction
        Keep this fraction of observations.
    n_obs
        Keep this number of observations.
    random_state
        Seed (int), ``np.random.RandomState`` or ``None``.
    copy
        For AnnData input, return a subsampled copy instead of subsetting in place.

    Returns
    -------
    ``None`` (in-place AnnData), a subsampled AnnData copy, or
    ``(X[obs_indices], obs_indices)`` for array input.
    """
    old_n = data.shape[0]
    if fraction is None and n_obs is None:
        raise TypeError("Either `fraction` or `n` must be set.")
    if fraction is not None and n_obs is not None:
        raise TypeError("Providing both `fraction` and `n` is not allowed.")
    if n_obs is None:
        if fraction < 0:
            raise ValueError(f"`fraction={fraction}` needs to be nonnegative.")
        if fraction > 1:
            raise ValueError(f"If `replace=False`, `fraction={fraction}` needs to be within [0, 1].")
        n_obs = int(fraction * old_n)
    if not copy and isinstance(data, AnnData) and data.isbacked:
        raise NotImplementedError(
            "Inplace sampling (`copy=False`) is not implemented for backed objects.")

    if isinstance(random_state, np.random.RandomState):
        np.random.set_state(random_state.get_state(legacy=False))
        rs = random_state
    else:
        if random_state is not None:
            np.random.seed(random_state)
        rs = np.random.mtrand._rand
    indices = rs.choice(old_n, size=n_obs, replace=False)

    if isinstance(data, AnnData):
        if not copy:
            data._inplace_subset_obs(indices)
            return None
        subset = data[indices]
        return subset.to_memory() if data.isbacked else subset.copy()
    subset = data[indices]
    if copy:
        subset = subset.copy()
    return subset, indices
