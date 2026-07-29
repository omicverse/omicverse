r"""Classifier-evaluation plots: ROC curves and confusion matrices.

``ov.pl`` had no way to draw the two figures that every classification result
ends up needing. Both functions here are data-first — arrays, a ``DataFrame``,
or an ``AnnData`` (``.obs``) all work — and both handle the multi-model and
multi-class cases that come up when comparing annotation methods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._registry import register_function
from ._plot_backend import style_axes
from ._stats_common import as_frame, font_size

__all__ = ["roc_auc_ci", "roc", "confusion"]


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def _midrank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged — the ``tiedrank`` of the DeLong papers."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


@register_function(
    aliases=["AUC置信区间", "roc_auc_ci", "delong", "AUC区间", "auc_ci"],
    category="pl",
    description=(
        "AUC with a DeLong confidence interval — the analytic interval R's pROC computes, not a bootstrap approximation"
    ),
    examples=[
        'res = ov.pl.roc_auc_ci(y_true, y_score)',
        'print(res["auc"], res["lower"], res["upper"])',
        '# one class against the rest',
        'ov.pl.roc_auc_ci(labels, scores, pos_label="tumour")',
    ],
    related=["pl.roc", "pl.confusion"],
)
def roc_auc_ci(y_true: Sequence[Any],
               y_score: Sequence[float],
               *,
               alpha: float = 0.05,
               pos_label: Any = None) -> Dict[str, float]:
    r"""AUC with a DeLong confidence interval.

    Uses the fast midrank formulation of DeLong's variance estimator
    (Sun & Xu 2014), which is what R's ``pROC::ci.auc(method='delong')``
    computes — and is validated against it in the test suite.

    Returns ``{'auc', 'se', 'lower', 'upper'}``.
    """
    from scipy.stats import norm

    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    if pos_label is None:
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(
                f"AUC needs exactly two classes, found {classes.size}. "
                "Pass `pos_label=` to pick one against the rest."
            )
        pos_label = classes[1]
    positive = y == pos_label

    pos, neg = s[positive], s[~positive]
    m, n = pos.size, neg.size
    if m == 0 or n == 0:
        raise ValueError("Both classes must be present to compute an AUC.")

    tx = _midrank(pos)
    ty = _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    s01 = np.var(v01, ddof=1) / m if m > 1 else 0.0
    s10 = np.var(v10, ddof=1) / n if n > 1 else 0.0
    se = float(np.sqrt(s01 + s10))
    z = norm.ppf(1.0 - alpha / 2.0)
    return {
        "auc": float(auc),
        "se": se,
        "lower": float(np.clip(auc - z * se, 0.0, 1.0)),
        "upper": float(np.clip(auc + z * se, 0.0, 1.0)),
    }


def _resolve_scores(data, y_true, y_score, score_names=None):
    """Normalise the many ways of passing (labels, scores) into one shape.

    Returns ``(y, {name: scores}, mode)`` where ``mode`` is ``'binary'`` or
    ``'multiclass'``.
    """
    frame = as_frame(data)

    if isinstance(y_score, Mapping):
        scores = {str(k): np.asarray(v, dtype=float) for k, v in y_score.items()}
    elif isinstance(y_score, str):
        if frame is None:
            raise TypeError("`y_score` names a column but no table was given.")
        scores = {y_score: frame[y_score].to_numpy(dtype=float)}
    elif isinstance(y_score, (list, tuple)) and all(isinstance(v, str) for v in y_score):
        if frame is None:
            raise TypeError("`y_score` names columns but no table was given.")
        scores = {c: frame[c].to_numpy(dtype=float) for c in y_score}
    else:
        arr = np.asarray(y_score, dtype=float)
        if arr.ndim == 1:
            scores = {"score": arr}
        elif arr.ndim == 2:
            names = (list(score_names) if score_names is not None
                     else [f"class {i}" for i in range(arr.shape[1])])
            scores = {str(nm): arr[:, i] for i, nm in enumerate(names)}
        else:
            raise ValueError("`y_score` must be 1- or 2-dimensional.")

    if isinstance(y_true, str):
        if frame is None:
            raise TypeError("`y_true` names a column but no table was given.")
        y = frame[y_true].to_numpy()
    else:
        y = np.asarray(y_true)

    for name, values in scores.items():
        if len(values) != len(y):
            raise ValueError(
                f"`y_score[{name!r}]` has {len(values)} values but `y_true` "
                f"has {len(y)}."
            )
    return y, scores


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------


@register_function(
    aliases=["ROC曲线", "roc", "roc_curve", "AUC", "auc_plot", "受试者工作特征"],
    category="pl",
    description=(
        "ROC curves with AUC and DeLong confidence intervals; overlays several "
        "models, or expands a multi-class problem into one-vs-rest curves"
    ),
    examples=[
        "# One model",
        "ax = ov.pl.roc(y_true=y_test, y_score=probabilities)",
        "# Several models on one axes",
        "ax = ov.pl.roc(y_true=y_test, y_score={'RF': p_rf, 'SVM': p_svm})",
        "# From a DataFrame of predictions",
        "ax = ov.pl.roc(pred_df, 'label', ['rf_prob', 'svm_prob'])",
        "# Multi-class, one-vs-rest plus the macro average",
        "ax = ov.pl.roc(y_true=y, y_score=proba_matrix, score_names=clf.classes_)",
    ],
    related=["pl.confusion", "pl.forest", "pl.savefig"],
)
def roc(data: Any = None,
        y_true: Any = None,
        y_score: Any = None,
        *,
        score_names: Optional[Sequence[str]] = None,
        pos_label: Any = None,
        multiclass: bool = False,
        average: Optional[str] = "macro",
        ax=None,
        figsize: Tuple[float, float] = (4.0, 4.0),
        palette=None,
        ci: Optional[str] = "delong",
        ci_alpha: float = 0.05,
        n_boot: int = 500,
        random_state: int = 0,
        chance_line: bool = True,
        aspect: Any = "equal",
        xlabel: str = "False positive rate",
        ylabel: str = "True positive rate",
        title: Optional[str] = None,
        legend: bool = True,
        legend_loc: str = "lower right",
        fontsize: Optional[float] = None,
        linewidth: float = 1.6,
        return_stats: bool = False):
    r"""Plot one or more ROC curves.

    Arguments
    ---------
    data
        Optional table; ``y_true`` / ``y_score`` may then be column names.
        Calling ``roc(y_true, y_score)`` with two arrays also works.
    y_true
        Ground-truth labels.
    y_score
        Continuous scores. One of: a 1-D array; a dict ``{model: scores}``;
        a list of column names; or a 2-D array of per-class probabilities
        (set ``multiclass=True``, or pass ``score_names`` matching the
        classes).
    pos_label
        Which label counts as positive in the binary case. Defaults to the
        greater of the two labels after sorting, matching scikit-learn — so
        ``{0, 1}`` gives 1 and ``{'control', 'tumour'}`` gives ``'tumour'``,
        regardless of row order.
    multiclass
        Treat a 2-D ``y_score`` as class probabilities and draw one-vs-rest
        curves. Inferred automatically when ``y_true`` has more than two
        levels and the score matrix is aligned to them.
    average
        With ``multiclass``, also draw ``'macro'`` (unweighted mean of the
        per-class curves) and/or ``'micro'``. Pass ``None`` to skip, or
        ``'both'``.
    ci
        ``'delong'`` (default) puts an analytic 95% CI on each AUC in the
        legend; ``'bootstrap'`` additionally shades a band around the curve;
        ``None`` reports the point estimate only.
    n_boot
        Bootstrap resamples when ``ci='bootstrap'``.
    aspect
        Axes aspect. ``'equal'`` (default) is the convention for a standalone
        ROC — a unit square where the chance line runs at 45°. Pass ``'auto'``
        when the ROC is a panel of a larger figure: ``'equal'`` makes
        matplotlib shrink the axes to a square inside whatever rectangle the
        layout assigned, so the panel stops filling its cell and its x-axis
        leaves the row's baseline. ``None`` leaves the aspect untouched.
    return_stats
        Return ``(ax, stats)`` with the AUC, CI and curve of every model.

    Returns
    -------
    The ``Axes`` (or ``(Axes, dict)``).
    """
    from sklearn.metrics import roc_curve

    # roc(y_true, y_score) with two bare arrays
    if data is not None and as_frame(data) is None:
        data, y_true, y_score = None, data, y_true
    if y_true is None or y_score is None:
        raise TypeError("Both `y_true` and `y_score` are required.")

    y, scores = _resolve_scores(data, y_true, y_score, score_names)
    classes = list(pd.unique(pd.Series(y).dropna()))
    is_multiclass = multiclass or (len(classes) > 2 and len(scores) == len(classes))
    if len(classes) > 2 and not is_multiclass:
        raise ValueError(
            f"`y_true` has {len(classes)} classes but only {len(scores)} score "
            "column(s). Pass a probability matrix with one column per class, "
            "or binarise the labels first."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    curves: Dict[str, Dict[str, Any]] = {}
    if is_multiclass:
        names = list(scores.keys())
        ordered = names if score_names is None else [str(c) for c in score_names]
        for name in ordered:
            binary = (pd.Series(y).astype(str) == str(name)).to_numpy()
            if binary.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(binary.astype(int), scores[name])
            curves[str(name)] = {"fpr": fpr, "tpr": tpr,
                                 **roc_auc_ci(binary.astype(int), scores[name],
                                              alpha=ci_alpha, pos_label=1)}
        if average in {"macro", "both"} and curves:
            grid = np.unique(np.concatenate([c["fpr"] for c in curves.values()]))
            mean_tpr = np.mean(
                [np.interp(grid, c["fpr"], c["tpr"]) for c in curves.values()],
                axis=0,
            )
            from sklearn.metrics import auc as _auc

            curves["macro-average"] = {"fpr": grid, "tpr": mean_tpr,
                                       "auc": float(_auc(grid, mean_tpr)),
                                       "se": np.nan, "lower": np.nan,
                                       "upper": np.nan}
        if average in {"micro", "both"} and curves:
            flat_y = np.concatenate(
                [(pd.Series(y).astype(str) == str(n)).to_numpy().astype(int)
                 for n in ordered if str(n) in scores]
            )
            flat_s = np.concatenate([scores[str(n)] for n in ordered
                                     if str(n) in scores])
            fpr, tpr, _ = roc_curve(flat_y, flat_s)
            curves["micro-average"] = {"fpr": fpr, "tpr": tpr,
                                       **roc_auc_ci(flat_y, flat_s,
                                                    alpha=ci_alpha, pos_label=1)}
    else:
        if pos_label is None and len(classes) == 2:
            # sorted, not order-of-appearance: the positive class must not
            # depend on which row happens to come first (scikit-learn's rule)
            try:
                pos_label = sorted(classes)[-1]
            except TypeError:
                pos_label = classes[-1]
        for name, values in scores.items():
            fpr, tpr, _ = roc_curve(y, values, pos_label=pos_label)
            curves[name] = {"fpr": fpr, "tpr": tpr,
                            **roc_auc_ci(y, values, alpha=ci_alpha,
                                         pos_label=pos_label)}

    from ._survival import _default_palette

    colors = _default_palette(len(curves), palette)
    rng = np.random.default_rng(random_state)
    for (name, curve), color in zip(curves.items(), colors):
        label = f"{name} (AUC {curve['auc']:.3f}"
        if ci and np.isfinite(curve.get("lower", np.nan)):
            label += f", {100 * (1 - ci_alpha):.0f}% CI {curve['lower']:.3f}-{curve['upper']:.3f}"
        label += ")"
        dashed = name.endswith("-average")
        ax.plot(curve["fpr"], curve["tpr"], color=color, linewidth=linewidth,
                linestyle="--" if dashed else "-", label=label)

        if ci == "bootstrap" and not dashed:
            grid = np.linspace(0, 1, 101)
            boots = []
            source = scores.get(name)
            if source is None:
                continue
            binary = (y == pos_label).astype(int) if not is_multiclass else \
                (pd.Series(y).astype(str) == str(name)).to_numpy().astype(int)
            for _ in range(n_boot):
                idx = rng.integers(0, len(binary), len(binary))
                if binary[idx].sum() in (0, len(idx)):
                    continue
                f, t, _ = roc_curve(binary[idx], source[idx])
                boots.append(np.interp(grid, f, t))
            if boots:
                arr = np.vstack(boots)
                lo = np.percentile(arr, 100 * ci_alpha / 2, axis=0)
                hi = np.percentile(arr, 100 * (1 - ci_alpha / 2), axis=0)
                ax.fill_between(grid, lo, hi, color=color, alpha=0.15,
                                linewidth=0)
                curve["band"] = (grid, lo, hi)

    if chance_line:
        ax.plot([0, 1], [0, 1], color="0.6", linewidth=0.9, linestyle=":",
                zorder=0)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    # 'equal' is the convention for a standalone ROC, but it overrides the
    # axes rectangle a layout engine assigned: matplotlib shrinks the axes to
    # square inside it, so the panel no longer fills its cell and its x-axis no
    # longer sits on the row's baseline. `aspect='auto'` keeps the given
    # rectangle, which is what a panel in a multi-panel figure needs.
    if aspect is not None:
        ax.set_aspect(aspect, adjustable="box")
    ax.set_xlabel(xlabel, fontsize=font_size(fontsize, "label"))
    ax.set_ylabel(ylabel, fontsize=font_size(fontsize, "label"))
    if title:
        ax.set_title(title, fontsize=font_size(fontsize, "title"))
    ax.tick_params(labelsize=font_size(fontsize))
    style_axes(ax)
    if legend:
        ax.legend(loc=legend_loc, frameon=False, fontsize=font_size(fontsize, "tick", -1))

    return (ax, {"curves": curves}) if return_stats else ax


@register_function(
    aliases=["混淆矩阵", "confusion", "confusion_matrix", "分类矩阵", "confusionplot"],
    category="pl",
    description=(
        "Confusion-matrix heatmap with counts and/or row percentages, plus "
        "accuracy, balanced accuracy, Cohen's kappa and per-class F1"
    ),
    examples=[
        "ax = ov.pl.confusion(y_true=truth, y_pred=predicted)",
        "# Row-normalised, so class imbalance does not hide the errors",
        "ax = ov.pl.confusion(y_true=truth, y_pred=predicted, normalize='true')",
        "# Compare an annotation method against manual labels in .obs",
        "ax = ov.pl.confusion(adata, 'manual_celltype', 'scsa_celltype',",
        "                     normalize='true', metrics=True)",
    ],
    related=["pl.roc", "pl.heatmap", "pl.savefig"],
)
def confusion(data: Any = None,
              y_true: Any = None,
              y_pred: Any = None,
              *,
              labels: Optional[Sequence[Any]] = None,
              normalize: Optional[str] = None,
              ax=None,
              figsize: Optional[Tuple[float, float]] = None,
              cmap: str = "Blues",
              annot: bool = True,
              annot_fmt: Optional[str] = None,
              counts_and_percent: bool = False,
              metrics: bool = False,
              colorbar: bool = True,
              grid: bool = True,
              xlabel: str = "Predicted",
              ylabel: str = "True",
              title: Optional[str] = None,
              fontsize: Optional[float] = None,
              return_stats: bool = False):
    r"""Plot a confusion matrix.

    Arguments
    ---------
    data, y_true, y_pred
        A table plus column names, two bare arrays, or an ``AnnData`` plus two
        ``.obs`` keys.
    labels
        Class order. Defaults to the sorted union of the two label sets, so
        the diagonal is meaningful even when a class is never predicted.
    normalize
        ``None`` for raw counts, ``'true'`` for row-wise (recall) fractions,
        ``'pred'`` for column-wise (precision), ``'all'`` for the whole matrix.
        ``'true'`` is usually the honest choice with imbalanced classes.
    counts_and_percent
        Annotate each cell with the count *and* the row percentage.
    metrics
        Append a right-hand strip with per-class precision / recall / F1.
    return_stats
        Return ``(ax, stats)`` with the matrix and the summary metrics.

    Returns
    -------
    The ``Axes`` (or ``(Axes, dict)``).
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 cohen_kappa_score, confusion_matrix,
                                 precision_recall_fscore_support)

    if data is not None and as_frame(data) is None:
        data, y_true, y_pred = None, data, y_true
    if y_true is None or y_pred is None:
        raise TypeError("Both `y_true` and `y_pred` are required.")

    frame = as_frame(data)
    yt = frame[y_true].to_numpy() if isinstance(y_true, str) else np.asarray(y_true)
    yp = frame[y_pred].to_numpy() if isinstance(y_pred, str) else np.asarray(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"`y_true` has {len(yt)} entries, `y_pred` {len(yp)}.")
    yt = pd.Series(yt).astype(str).to_numpy()
    yp = pd.Series(yp).astype(str).to_numpy()

    if labels is None:
        labels = sorted(set(yt) | set(yp))
    else:
        labels = [str(v) for v in labels]

    counts = confusion_matrix(yt, yp, labels=labels)
    matrix = counts.astype(float)
    if normalize == "true":
        totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, totals, out=np.zeros_like(matrix),
                           where=totals > 0)
    elif normalize == "pred":
        totals = matrix.sum(axis=0, keepdims=True)
        matrix = np.divide(matrix, totals, out=np.zeros_like(matrix),
                           where=totals > 0)
    elif normalize == "all":
        matrix = matrix / matrix.sum() if matrix.sum() else matrix
    elif normalize is not None:
        raise ValueError(
            f"`normalize` must be None, 'true', 'pred' or 'all', got {normalize!r}."
        )

    n = len(labels)
    if ax is None:
        if figsize is None:
            side = max(3.0, 0.45 * n + 1.8)
            figsize = (side + (2.2 if metrics else 0.0), side)
        _, ax = plt.subplots(figsize=figsize)

    image = ax.imshow(matrix, cmap=cmap, aspect="equal",
                      vmin=0, vmax=matrix.max() if matrix.size else 1)
    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=font_size(fontsize))
    ax.set_yticks(range(n), labels, fontsize=font_size(fontsize))
    ax.set_xlabel(xlabel, fontsize=font_size(fontsize, "label"))
    ax.set_ylabel(ylabel, fontsize=font_size(fontsize, "label"))

    if annot:
        if annot_fmt is None:
            annot_fmt = "{:.0f}" if normalize is None else "{:.2f}"
        threshold = matrix.max() * 0.6 if matrix.size else 0
        row_totals = counts.sum(axis=1)
        for i in range(n):
            for j in range(n):
                value = matrix[i, j]
                text = annot_fmt.format(value)
                if counts_and_percent:
                    pct = counts[i, j] / row_totals[i] * 100 if row_totals[i] else 0
                    text = f"{counts[i, j]:d}\n{pct:.1f}%"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=font_size(fontsize, "tick", -1),
                        color="white" if value > threshold else "black")

    if grid:
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    precision, recall, f1, support = precision_recall_fscore_support(
        yt, yp, labels=labels, zero_division=0
    )
    stats = {
        "matrix": pd.DataFrame(counts, index=labels, columns=labels),
        "normalized": pd.DataFrame(matrix, index=labels, columns=labels),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "kappa": float(cohen_kappa_score(yt, yp)),
        "per_class": pd.DataFrame(
            {"precision": precision, "recall": recall, "f1": f1,
             "support": support}, index=labels),
    }

    if metrics:
        strip = ax.inset_axes([1.04, 0.0, 0.34, 1.0])
        values = np.column_stack([precision, recall, f1])
        strip.imshow(values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        strip.set_xticks(range(3), ["Prec", "Rec", "F1"], rotation=45,
                         ha="right", fontsize=font_size(fontsize, "tick", -1))
        strip.set_yticks([])
        for i in range(n):
            for j in range(3):
                strip.text(j, i, f"{values[i, j]:.2f}", ha="center",
                           va="center", fontsize=font_size(fontsize, "tick", -2),
                           color="white" if values[i, j] > 0.6 else "black")
        for spine in strip.spines.values():
            spine.set_visible(False)
        strip.tick_params(length=0)

    if colorbar:
        if metrics:
            # place it explicitly beyond the metrics strip — the default
            # `pad=` puts it straight on top of the strip
            cbar = ax.figure.colorbar(image, cax=ax.inset_axes(
                [1.48, 0.0, 0.045, 1.0]))
        else:
            cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=font_size(fontsize) - 1)
        cbar.set_label("fraction" if normalize else "count", fontsize=font_size(fontsize))

    if title is None:
        title = (f"accuracy {stats['accuracy']:.3f} | balanced "
                 f"{stats['balanced_accuracy']:.3f} | kappa {stats['kappa']:.3f}")
    ax.set_title(title, fontsize=font_size(fontsize), pad=8)

    return (ax, stats) if return_stats else ax
