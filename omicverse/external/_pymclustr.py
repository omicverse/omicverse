"""Shared adapter for the optional :mod:`pymclustR` backend.

This replaces the ``rpy2`` bridge the spatial wrappers used to embed, so an R
installation is no longer required. For the ``EEE`` model those wrappers all
default to, the substitution is numerically transparent: on a real 700-cell
PCA embedding, ``pymclustR`` and R ``mclust`` 6.1.2 agree on log-likelihood
and BIC to six decimals and place every cell in the same cluster (ARI 1.0) at
both G=5 and G=7.

That does not hold for every covariance model. ``pymclustR``'s own R-parity
suite holds ``EVE`` and ``VVE`` only to a 50% log-likelihood tolerance and a
55% label-agreement floor, because those two share one orientation matrix
across components and their likelihood surface has several stationary points —
R's Browne-McNicholas MM optimiser and ``pymclustR``'s Stiefel gradient
descent can converge to different maxima. Callers who pass ``EVE`` or ``VVE``
through ``model_names`` should not expect their previous R-derived labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np


def fit_pymclustr(
    data: Any,
    n_components: int,
    model_names: Union[str, Sequence[str]] = "EEE",
    random_state: Optional[int] = None,
    **kwargs: Any,
):
    """Fit pymclustR and return its 1-based classifications and fit object."""
    if isinstance(n_components, bool) or not isinstance(
        n_components, (int, np.integer)
    ):
        raise TypeError("n_components must be a positive integer")
    n_components = int(n_components)
    if n_components < 1:
        raise ValueError("n_components must be a positive integer")

    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("pymclustR input data must be a two-dimensional matrix")
    if values.shape[0] < n_components:
        raise ValueError(
            "n_components cannot exceed the number of observations"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("pymclustR input data must contain numeric values")
    values = values.astype(float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("pymclustR input data must contain only finite values")

    if isinstance(model_names, str):
        normalized_models = [model_names]
    else:
        normalized_models = list(model_names)
    if not normalized_models or not all(
        isinstance(model, str) and model for model in normalized_models
    ):
        raise ValueError("model_names must contain at least one model name")

    try:
        from mclust_py import Mclust
    except ImportError as exc:
        raise ImportError(
            "pymclustR>=0.2.1 is required for this clustering backend. "
            "Install it with `pip install pymclustR>=0.2.1`."
        ) from exc

    random_state_snapshot = np.random.get_state()
    try:
        if random_state is not None:
            np.random.seed(random_state)
        fit = Mclust(
            values,
            G=[n_components],
            model_names=normalized_models,
            **kwargs,
        )
    finally:
        np.random.set_state(random_state_snapshot)

    raw_labels = np.asarray(fit.classification)
    if raw_labels.shape != (values.shape[0],):
        raise RuntimeError(
            "pymclustR returned classifications with an unexpected shape"
        )
    labels = raw_labels.astype(int)
    if not np.array_equal(raw_labels, labels):
        raise RuntimeError("pymclustR returned non-integer classifications")
    if labels.size and (labels.min() < 1 or labels.max() > int(fit.G)):
        raise RuntimeError(
            "pymclustR returned classifications outside its 1-based range"
        )

    return labels, fit
