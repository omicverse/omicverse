"""Control-gene removal of unwanted variation (RUVg matrix method).

Reference: Risso et al., Nature Biotechnology (2014), doi:10.1038/nbt.2931.
Numerical contract checked against drisso/RUVSeq, commit
f560e84eb493f0c2d72279612f9e4ca99493b434 (RUVSeq 1.33.1).
"""

from __future__ import annotations

import warnings
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import sparse

from .._registry import register_function


@register_function(
    aliases=["RUVg", "control gene normalization"], category="bulk",
    description="Estimate unwanted-variation covariates from negative control genes",
)
def ruvg(expression, controls, k=1, *, drop=0, center=True,
         round_counts=True, epsilon=1.0, tolerance=1e-8, is_log=False):
    """Estimate RUVg factors from a genes-by-samples expression matrix.

    Parameters
    ----------
    expression : pandas.DataFrame or two-dimensional array or scipy sparse matrix
        Genes on rows and independent samples on columns. Counts must be finite,
        nonnegative integers. With ``is_log=True``, supply already transformed
        finite expression instead. Sparse input is explicitly densified: the
        transformed/centered matrix and returned correction are dense.
    controls : sequence
        Negative control gene names for a DataFrame, zero-based row indices for
        an array, or a boolean row mask. DataFrame controls are always labels.
        Controls should not respond to the biological condition of interest.
    k : int
        Number of leading control-gene singular vectors considered, before
        removing ``drop`` vectors. Must be less than the number of samples.
    drop : int
        Discard this many leading vectors; the output has at most k-drop factors.
    center : bool
        Center each transformed gene across samples before learning factors.
    round_counts : bool
        For count input, round reconstructed counts and clip negatives to zero.
        If False, negative reconstructed values are retained, as in RUVSeq.
    epsilon : float
        Positive offset in the natural-log transformation log(counts + epsilon).
    tolerance : float
        Absolute singular-value threshold used to reduce k on deficient controls.
    is_log : bool
        Skip log transformation and return correction on the supplied scale.

    Returns
    -------
    dict
        ``W``: samples-by-factors; ``normalized_counts``: genes-by-samples.
        Both are DataFrames with preserved axes for DataFrame input, otherwise
        ndarrays. ``k_used`` is the number of returned factors. Input is unchanged.

    Notes
    -----
    For count-based DE, use W as covariates alongside the biological design and
    the original counts. The reconstructed matrix is not a substitute for raw
    counts in a count likelihood. Confounded biology and unwanted variation
    cannot be separated by this procedure without suitable controls.

    This is the RUVg matrix algorithm, not full RUVSeq (RUVr/RUVs, EDASeq objects,
    or downstream DE). Unlike R's warning-and-continue behavior, fractional raw
    counts and a control subspace with no retained factors are rejected.
    Singular-vector signs are arbitrary; compare W @ W.T across implementations.
    """
    frame = isinstance(expression, pd.DataFrame)
    if frame and (not expression.index.is_unique or not expression.columns.is_unique):
        raise ValueError("Gene and sample labels must be unique.")
    values = expression.to_numpy() if frame else expression
    values = values.toarray() if sparse.issparse(values) else np.asarray(values)
    if (values.ndim != 2 or min(values.shape) < 1
            or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values) or not np.isfinite(values).all()):
        raise ValueError("expression must be a nonempty finite numeric matrix.")
    for name, flag in (("center", center), ("round_counts", round_counts), ("is_log", is_log)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean.")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or not 0 < k < values.shape[1]:
        raise ValueError("k must be a positive integer smaller than the sample count.")
    if isinstance(drop, (bool, np.bool_)) or not isinstance(drop, Integral) or not 0 <= drop < k:
        raise ValueError("drop must be an integer with 0 <= drop < k.")
    for name, value, positive in (("epsilon", epsilon, True), ("tolerance", tolerance, False)):
        if (not np.isscalar(value) or isinstance(value, (str, bool))
                or not np.isreal(value) or not np.isfinite(value)
                or (value <= 0 if positive else value < 0)):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")
    if not is_log and (np.any(values < 0) or
                       not np.all(np.abs(values - np.rint(values)) < np.sqrt(np.finfo(float).eps))):
        raise ValueError("Supply nonnegative integer counts, or set is_log=True for transformed data.")
    selected = np.asarray(controls)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("controls must be a nonempty one-dimensional sequence.")
    if selected.dtype.kind == "b":
        if len(selected) != values.shape[0]:
            raise ValueError("The control mask must have one entry per gene.")
        indices = np.flatnonzero(selected)
    elif frame:
        indices = expression.index.get_indexer(selected)
        if np.any(indices < 0):
            raise ValueError("All control gene labels must occur in expression.index.")
    else:
        if selected.dtype.kind not in "iu":
            raise ValueError("Array controls must be zero-based integer row indices or a boolean mask.")
        indices = selected
        if np.any(indices < 0) or np.any(indices >= values.shape[0]):
            raise ValueError("Control indices are out of bounds.")
    if not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("Select at least one control gene without duplicates.")

    transformed = values.astype(np.float64, copy=True).T
    if not is_log:
        transformed = np.log(transformed + epsilon)
    control_matrix = transformed[:, indices]
    if center:
        control_matrix = control_matrix - control_matrix.mean(axis=0)
    vectors, singular_values, _ = np.linalg.svd(control_matrix, full_matrices=False)
    effective = min(k, int(np.count_nonzero(singular_values > tolerance)))
    if effective <= drop:
        raise ValueError("Control matrix rank leaves no factors after drop; revise controls, k or drop.")
    if effective < k:
        warnings.warn(f"Control matrix rank reduces k from {k} to {effective}.", UserWarning, stacklevel=2)
    factors = vectors[:, drop:effective]
    # Orthogonal projection is invariant to SVD sign flips and basis rotations.
    corrected = transformed - factors @ (factors.T @ transformed)
    if not is_log:
        with np.errstate(over="raise", invalid="raise"):
            corrected = np.exp(corrected) - epsilon
        if round_counts:
            corrected = np.maximum(np.rint(corrected), 0)
    corrected = corrected.T
    if frame:
        factors = pd.DataFrame(factors, index=expression.columns,
                               columns=[f"W_{i + 1}" for i in range(factors.shape[1])])
        corrected = pd.DataFrame(corrected, index=expression.index, columns=expression.columns)
    return {"W": factors, "normalized_counts": corrected, "k_used": effective - drop}
