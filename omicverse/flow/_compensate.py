r"""Compensation — undoing fluorescence spillover.

Every fluorochrome emits into detectors other than its own. A spillover matrix
``S`` records that: ``S[i, j]`` is the fraction of dye ``i``'s signal that lands
in detector ``j``, with ones on the diagonal. Observed values are ``true @ S``,
so recovering the true values is ``observed @ inv(S)``.

Two things about this are worth stating plainly, because both are silent when
wrong:

* **Uncompensated fluorescence is not wrong-looking, it is wrong.** A CD4-PE
  spillover into the FITC detector produces a diagonal smear that looks exactly
  like a real double-positive population, and people gate on it.
* **Double compensation is worse than none**, and nothing in the numbers says
  it happened. So every function here records what it did in
  ``uns['flow']['compensated']`` and refuses to do it twice.

Estimating a matrix from single-stain controls is deliberately NOT here yet —
that is a separate job with its own failure modes, and applying a matrix
correctly is the part everything else depends on.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._registry import register_function

__all__ = ["compensate", "spillover_to_compensation", "spillover_spreading_matrix"]


def _resolve_columns(adata: Any, names: Sequence[str]) -> list:
    """Map spillover column labels onto AnnData columns.

    A matrix names DETECTORS (``FITC-A``), while a var index built by
    ``ov.io.read_fcs`` uses the MARKER where the file named one (``CD3``). Both
    have to work, or a matrix straight out of the file fails to apply to the
    AnnData that same file produced.
    """
    idx = list(adata.var.index.astype(str))
    chan = [str(c) for c in adata.var["channel"]] if "channel" in adata.var.columns else []
    out = []
    for n in names:
        n = str(n)
        if n in idx:
            out.append(idx.index(n))
        elif chan and n in chan:
            out.append(chan.index(n))
        else:
            raise KeyError(
                f"spillover names detector {n!r}, which is not a channel or marker "
                f"in this object. Present: {idx[:12]}{'...' if len(idx) > 12 else ''}"
            )
    return out


def spillover_to_compensation(spillover: pd.DataFrame) -> pd.DataFrame:
    """Invert a spillover matrix into a compensation matrix.

    Kept separate and exposed because the two are constantly confused — vendors
    export both, they are inverses, and applying the wrong one produces
    plausible garbage rather than an error.
    """
    m = np.asarray(spillover, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("a spillover matrix must be square")
    if np.linalg.matrix_rank(m) < m.shape[0]:
        raise np.linalg.LinAlgError(
            "this spillover matrix is singular and cannot be inverted — usually "
            "a duplicated or all-zero detector row."
        )
    return pd.DataFrame(np.linalg.inv(m), index=spillover.index, columns=spillover.columns)


@register_function(
    aliases=["compensate", "补偿", "荧光补偿", "spillover_correction", "compensation"],
    category="flow",
    description=(
        "Apply fluorescence compensation to a cytometry AnnData using a "
        "spillover matrix — the file's own $SPILLOVER by default, or one "
        "supplied explicitly. Only the detectors the matrix names are touched, "
        "so scatter and time pass through untouched. Refuses to compensate "
        "twice, and records what it did in uns['flow']."
    ),
    examples=[
        "ov.flow.compensate(adata)                      # use the file's $SPILLOVER",
        "ov.flow.compensate(adata, spillover=my_matrix)",
        "ov.flow.compensate(adata, matrix_type='compensation')",
    ],
    related=["io.read_fcs", "flow.spillover_spreading_matrix"],
)
def compensate(
    adata: Any,
    spillover: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    *,
    matrix_type: str = "spillover",
    channels: Optional[Sequence[str]] = None,
    layer_original: Optional[str] = "uncompensated",
    inplace: bool = True,
) -> Any:
    r"""Compensate ``adata`` in place (or on a copy).

    Arguments
    ---------
    spillover
        The matrix. ``None`` uses ``uns['fcs']['spillover']``, i.e. what the
        instrument wrote. Note that the acquisition matrix is frequently wrong —
        most labs recompute it from single-stain controls — so passing one
        explicitly is the norm rather than the exception.
    matrix_type
        ``'spillover'`` (default) or ``'compensation'`` for an already-inverted
        matrix. Getting this backwards produces plausible garbage, not an error,
        which is why it is an explicit argument and not inferred.
    channels
        Restrict to these detectors. Rarely wanted: the matrix is defined over
        the detector set it was measured on.
    layer_original
        Keep the pre-compensation values in this layer. ``None`` skips it — but
        keeping them is what lets someone check whether compensation is what
        moved a population.
    """
    if matrix_type not in ("spillover", "compensation"):
        raise ValueError("matrix_type must be 'spillover' or 'compensation'")
    if not inplace:
        adata = adata.copy()

    flow_uns = adata.uns.setdefault("flow", {})
    if flow_uns.get("compensated") or (adata.uns.get("fcs", {}) or {}).get("compensated"):
        raise ValueError(
            "this object is already compensated. Compensating twice is silently "
            "destructive — the numbers stay plausible — so it has to be explicit: "
            "re-read the file if you meant to apply a different matrix."
        )

    if spillover is None:
        spillover = (adata.uns.get("fcs", {}) or {}).get("spillover")
        if spillover is None:
            raise ValueError(
                "no spillover matrix given and this object carries none "
                "(uns['fcs']['spillover'] is absent). Unstained runs and mass "
                "cytometry legitimately have none — pass one explicitly, or skip "
                "compensation."
            )
    if not isinstance(spillover, pd.DataFrame):
        raise TypeError(
            "spillover must be a labelled DataFrame — a bare array cannot say "
            "which detector each row belongs to, and guessing the order is how "
            "channels get compensated against the wrong dye."
        )

    if channels is not None:
        keep = [c for c in spillover.columns if str(c) in {str(x) for x in channels}]
        spillover = spillover.loc[keep, keep]

    comp = (spillover if matrix_type == "compensation"
            else spillover_to_compensation(spillover))

    cols = _resolve_columns(adata, list(spillover.columns))
    X = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype=float)
    if layer_original:
        adata.layers[layer_original] = adata.X.copy()

    # Only the detectors the matrix names. Scatter and time are not fluorescence
    # and must not be dragged through the inverse.
    X[:, cols] = X[:, cols] @ np.asarray(comp, dtype=float)
    adata.X = X.astype(adata.X.dtype if hasattr(adata.X, "dtype") else np.float32)

    flow_uns["compensated"] = True
    flow_uns["spillover"] = spillover
    flow_uns["compensated_channels"] = [str(c) for c in spillover.columns]
    return adata


@register_function(
    aliases=["spillover_spreading_matrix", "SSM", "溢出扩散矩阵", "spreading_error"],
    category="flow",
    description=(
        "Compute the spillover spreading matrix (SSM) from single-stain "
        "controls — the standard deviation of spreading error each dye "
        "contributes to each other detector. Unlike the spillover matrix, "
        "spreading cannot be compensated away: it is the panel-design "
        "diagnostic that says which marker pairs will never resolve."
    ),
    examples=[
        "ssm = ov.flow.spillover_spreading_matrix(controls, unstained)",
    ],
    related=["flow.compensate"],
)
def spillover_spreading_matrix(
    controls: Mapping[str, Any],
    unstained: Optional[Any] = None,
    *,
    n_bins: int = 8,
) -> pd.DataFrame:
    r"""Spreading error, per (spilling dye -> receiving detector) pair.

    Compensation removes the MEAN of the spillover; it cannot remove the
    variance the extra photons added. That residual — spreading — is why a dim
    population can be unresolvable on a detector that a bright dye spills into,
    no matter how good the compensation is. It is a property of the PANEL, and
    the only fix is to move a marker to another dye.

    Computed as the slope of ``variance`` vs ``intensity`` across bins of the
    spilling channel, following the usual formulation: for each single-stain
    control, bin events by the stained detector, take the variance of the
    receiving detector in each bin, and fit. The square root of the slope is
    reported, so units are standard deviations and the values are comparable
    across pairs.

    Arguments
    ---------
    controls
        ``{detector_name: AnnData}`` — one COMPENSATED single-stain control per
        dye. Uncompensated input gives a spreading estimate contaminated by the
        spillover itself.
    unstained
        Optional unstained control; its per-detector variance is subtracted as
        the baseline.
    """
    if not controls:
        raise ValueError("no single-stain controls given")

    detectors: list = []
    for ad_ in controls.values():
        for name in ad_.var.index.astype(str):
            if name not in detectors:
                detectors.append(name)

    base = {}
    if unstained is not None:
        Xu = np.asarray(unstained.X.todense() if hasattr(unstained.X, "todense") else unstained.X, float)
        for i, name in enumerate(unstained.var.index.astype(str)):
            base[name] = float(np.var(Xu[:, i]))

    out = pd.DataFrame(0.0, index=list(controls), columns=detectors, dtype=float)
    for stain, ad_ in controls.items():
        X = np.asarray(ad_.X.todense() if hasattr(ad_.X, "todense") else ad_.X, float)
        names = list(ad_.var.index.astype(str))
        if str(stain) not in names:
            raise KeyError(f"control {stain!r} has no detector called {stain!r}")
        si = names.index(str(stain))
        x = X[:, si]
        # Equal-count bins: fluorescence is heavily skewed, so equal-WIDTH bins
        # put almost every event in the first one and fit noise.
        order = np.argsort(x)
        chunks = np.array_split(order, n_bins)
        centres = np.array([x[c].mean() for c in chunks])
        for j, det in enumerate(names):
            if det == str(stain):
                continue
            var = np.array([np.var(X[c, j]) for c in chunks]) - base.get(det, 0.0)
            good = np.isfinite(var) & np.isfinite(centres)
            if good.sum() < 2:
                continue
            slope = np.polyfit(centres[good], var[good], 1)[0]
            out.loc[stain, det] = float(np.sqrt(slope)) if slope > 0 else 0.0
    return out
