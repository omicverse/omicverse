r"""Forest plots — effect sizes with confidence intervals, and meta-analysis.

A forest plot is the natural display for any list of "effect ± CI" rows:
odds/hazard ratios per cohort, per-gene coefficients from a regression, Cox
results across subgroups, or the studies of a meta-analysis. ``ov.pl`` had one
only inside the Mendelian-randomisation code (``ov.genetics.mr_forest``); this
is the general version.

When ``meta=`` is given the function also pools the rows — fixed effect
(inverse variance) and/or DerSimonian-Laird random effects — and draws the
summary diamond with Q, I² and tau².
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._registry import register_function
from ._stats_common import as_frame, resolve_columns

__all__ = ["meta_analysis", "forest"]


@register_function(
    aliases=["元分析", "meta_analysis", "meta分析", "效应量合并", "随机效应模型"],
    category="pl",
    description=(
        "Pool effect estimates by inverse variance — fixed effect or DerSimonian-Laird random effects — with Q, I-squared and tau-squared"
    ),
    examples=[
        'pooled = ov.pl.meta_analysis(res["logHR"], res["se"], method="random")',
        'print(pooled["estimate"], pooled["I2"])',
        '# fixed effect, when the studies really are replicates',
        'ov.pl.meta_analysis(est, se, method="fixed")',
    ],
    related=["pl.forest", "pl.survival"],
)
def meta_analysis(estimate: Sequence[float],
                  se: Sequence[float],
                  *,
                  method: str = "random",
                  ci_level: float = 0.95) -> Dict[str, float]:
    r"""Pool effect estimates by inverse variance.

    Arguments
    ---------
    estimate
        Effect sizes on the scale they are to be pooled on — log(OR),
        log(HR), mean difference, ...
    se
        Standard errors of ``estimate``.
    method
        ``'fixed'`` for inverse-variance fixed effect, ``'random'`` for
        DerSimonian-Laird random effects (the default, and the safer choice
        whenever the studies are not exact replicates).
    ci_level
        Confidence level of the pooled interval.

    Returns
    -------
    dict with ``estimate``, ``se``, ``lower``, ``upper``, ``zvalue``,
    ``pvalue``, and the heterogeneity statistics ``Q``, ``df``, ``Q_pvalue``,
    ``tau2`` and ``I2``.

    Notes
    -----
    Matches ``statsmodels.stats.meta_analysis.combine_effects`` with
    ``method_re='dl'``; that agreement is asserted in the test suite.
    """
    from scipy.stats import chi2 as chi2_dist
    from scipy.stats import norm

    y = np.asarray(estimate, dtype=float)
    v = np.asarray(se, dtype=float) ** 2
    if y.size != v.size:
        raise ValueError("`estimate` and `se` must have the same length.")
    if np.any(v <= 0):
        raise ValueError("Standard errors must be positive.")

    w = 1.0 / v
    theta_f = float(np.sum(w * y) / np.sum(w))
    var_f = float(1.0 / np.sum(w))

    Q = float(np.sum(w * (y - theta_f) ** 2))
    df = int(y.size - 1)
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - df) / C) if C > 0 and df > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q) if Q > 0 and df > 0 else 0.0

    if method == "fixed":
        theta, var = theta_f, var_f
    elif method == "random":
        w_star = 1.0 / (v + tau2)
        theta = float(np.sum(w_star * y) / np.sum(w_star))
        var = float(1.0 / np.sum(w_star))
    else:
        raise ValueError(f"`method` must be 'fixed' or 'random', got {method!r}.")

    se_pooled = float(np.sqrt(var))
    z = norm.ppf(0.5 + ci_level / 2.0)
    zval = theta / se_pooled if se_pooled > 0 else np.nan
    return {
        "estimate": theta,
        "se": se_pooled,
        "lower": theta - z * se_pooled,
        "upper": theta + z * se_pooled,
        "zvalue": float(zval),
        "pvalue": float(2 * norm.sf(abs(zval))) if np.isfinite(zval) else np.nan,
        "Q": Q,
        "df": df,
        "Q_pvalue": float(chi2_dist.sf(Q, df)) if df > 0 else np.nan,
        "tau2": tau2,
        "I2": I2,
        "method": method,
    }


def _format_effect(value: float, lo: float, hi: float, digits: int) -> str:
    return f"{value:.{digits}f} ({lo:.{digits}f}, {hi:.{digits}f})"


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.3g}"


@register_function(
    aliases=["森林图", "forest", "forest_plot", "森林", "meta分析图", "effect_plot"],
    category="pl",
    description=(
        "Forest plot of effect sizes with confidence intervals, optional "
        "subgroups, and fixed/random-effects meta-analysis with I-squared"
    ),
    examples=[
        "# Hazard ratios per cohort, on a log axis",
        "ax = ov.pl.forest(res, 'HR', lower='HR_low', upper='HR_high',",
        "                  label='cohort', log_scale=True)",
        "# From coefficients and standard errors",
        "ax = ov.pl.forest(res, 'beta', se='se', label='gene')",
        "# Pool the studies and show the diamond",
        "ax, stats = ov.pl.forest(studies, 'logOR', se='se', label='study',",
        "                         meta='random', log_scale=True,",
        "                         return_stats=True)",
        "print(stats['meta']['I2'])",
    ],
    related=["pl.survival", "pl.volcano", "pl.savefig"],
)
def forest(data: Any = None,
           estimate: Any = None,
           *,
           lower: Any = None,
           upper: Any = None,
           se: Any = None,
           label: Any = None,
           group: Any = None,
           weight: Any = None,
           pvalue: Any = None,
           ci_level: float = 0.95,
           log_scale: bool = False,
           already_log: bool = False,
           ref_line: Optional[float] = None,
           meta: Optional[str] = None,
           sort: Optional[str] = None,
           ax=None,
           figsize: Optional[Tuple[float, float]] = None,
           color: str = "#1F577B",
           summary_color: str = "#CB3E35",
           marker: str = "s",
           markersize: float = 40.0,
           scale_marker: bool = True,
           annotate: bool = True,
           annotate_x: float = 1.04,
           digits: int = 2,
           xlabel: Optional[str] = None,
           title: Optional[str] = None,
           xlim: Optional[Tuple[float, float]] = None,
           fontsize: float = 9,
           return_stats: bool = False):
    r"""Draw a forest plot.

    Arguments
    ---------
    data, estimate
        A table plus the column holding the effect size, or a bare array.
    lower, upper
        Confidence-interval bounds. Give these **or** ``se``.
    se
        Standard error; the interval is then built at ``ci_level``.
    label
        Row labels (column name or array). Defaults to the table index.
    group
        Optional subgroup column — rows are blocked under a bold header, and
        with ``meta`` each subgroup gets its own summary.
    weight
        Marker area weights. Defaults to inverse variance when ``se`` is
        available, which is the convention in meta-analysis figures.
    pvalue
        Optional column shown in the right-hand annotation.
    log_scale
        Effects are ratios (OR/HR/RR): plot on a log x-axis and centre the
        reference line on 1. Pooling is then done in log space and converted
        back.
    already_log
        Set with ``log_scale`` when the *input* is already log-transformed —
        the axis is then exponentiated for display only.
    ref_line
        Null-effect line. Defaults to 1 for ``log_scale``, else 0.
    meta
        ``'fixed'``, ``'random'`` or ``'both'`` to pool the rows and draw the
        summary diamond(s).
    sort
        ``'estimate'`` or ``'label'`` to reorder rows; ``None`` keeps input
        order (top row first).
    annotate
        Print ``effect (low, high)`` — and the P value when given — to the
        right of the axes.
    return_stats
        Return ``(ax, stats)`` including the pooled result and heterogeneity.

    Returns
    -------
    The ``Axes`` (or ``(Axes, dict)``).
    """
    from scipy.stats import norm

    if data is not None and as_frame(data) is None:
        data, estimate = None, data
    if estimate is None:
        raise TypeError("`estimate` is required.")
    if lower is None and upper is None and se is None:
        raise TypeError("Give either `lower`/`upper` or `se`.")

    frame_in = as_frame(data)
    if label is None and frame_in is not None:
        label = np.asarray(frame_in.index).astype(str)

    frame, names = resolve_columns(data, estimate=estimate, lower=lower,
                                   upper=upper, se=se, label=label,
                                   group=group, weight=weight, pvalue=pvalue,
                                   dropna=False, require=("estimate",))

    est = frame["estimate"].to_numpy(dtype=float)
    z = norm.ppf(0.5 + ci_level / 2.0)

    # everything is computed on the additive scale, then displayed
    work = np.log(est) if (log_scale and not already_log) else est
    if "se" in frame:
        se_work = frame["se"].to_numpy(dtype=float)
        lo_work, hi_work = work - z * se_work, work + z * se_work
    else:
        lo_raw = frame["lower"].to_numpy(dtype=float)
        hi_raw = frame["upper"].to_numpy(dtype=float)
        lo_work = np.log(lo_raw) if (log_scale and not already_log) else lo_raw
        hi_work = np.log(hi_raw) if (log_scale and not already_log) else hi_raw
        se_work = (hi_work - lo_work) / (2.0 * z)

    labels = (frame["label"].astype(str).to_numpy() if "label" in frame
              else np.array([str(i + 1) for i in range(len(est))]))
    groups = frame["group"].astype(str).to_numpy() if "group" in frame else None
    pvals = frame["pvalue"].to_numpy(dtype=float) if "pvalue" in frame else None

    if "weight" in frame:
        weights = frame["weight"].to_numpy(dtype=float)
    elif np.all(np.isfinite(se_work)) and np.all(se_work > 0):
        weights = 1.0 / se_work ** 2
    else:
        weights = np.ones_like(work)

    order = np.arange(len(work))
    if sort == "estimate":
        order = np.argsort(work)
    elif sort == "label":
        order = np.argsort(labels)
    elif sort is not None:
        raise ValueError("`sort` must be 'estimate', 'label' or None.")
    if groups is not None:
        order = order[np.argsort(pd.Categorical(groups[order],
                                                categories=pd.unique(groups),
                                                ordered=True).codes,
                                 kind="stable")]

    # ---- build the row layout (data rows, group headers, summary rows) ----
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    current_group = None
    for idx in order:
        if groups is not None and groups[idx] != current_group:
            current_group = groups[idx]
            rows.append({"kind": "header", "label": current_group})
        rows.append({
            "kind": "study", "label": labels[idx], "value": work[idx],
            "lower": lo_work[idx], "upper": hi_work[idx], "se": se_work[idx],
            "weight": weights[idx],
            "pvalue": pvals[idx] if pvals is not None else np.nan,
        })

    methods = {"fixed": ["fixed"], "random": ["random"],
               "both": ["fixed", "random"], None: []}
    if meta not in methods:
        raise ValueError("`meta` must be 'fixed', 'random', 'both' or None.")
    for method in methods[meta]:
        if not np.all(np.isfinite(se_work)) or np.any(se_work <= 0):
            raise ValueError(
                "Meta-analysis needs a usable standard error for every row; "
                "pass `se=` or finite `lower`/`upper`."
            )
        pooled = meta_analysis(work, se_work, method=method, ci_level=ci_level)
        stats.setdefault("meta", {})[method] = pooled
        rows.append({
            "kind": "summary", "label": f"{method.capitalize()} effect",
            "value": pooled["estimate"], "lower": pooled["lower"],
            "upper": pooled["upper"], "pvalue": pooled["pvalue"],
        })

    n_rows = len(rows)
    if ax is None:
        if figsize is None:
            figsize = (6.4, max(2.2, 0.34 * n_rows + 1.0))
        _, ax = plt.subplots(figsize=figsize)

    if scale_marker:
        w = np.array([r.get("weight", np.nan) for r in rows], dtype=float)
        finite = np.isfinite(w)
        if finite.sum() and np.nanmax(w) > np.nanmin(w[finite]):
            scaled = (w - np.nanmin(w[finite])) / (np.nanmax(w) - np.nanmin(w[finite]))
        else:
            scaled = np.full(n_rows, 0.5)
    else:
        scaled = np.full(n_rows, 0.5)

    # Rows live on the additive (log) scale; ratios are drawn on a log axis so
    # matplotlib picks readable ticks instead of exp() of evenly spaced ones.
    def _x(value):
        return float(np.exp(value)) if log_scale else float(value)

    yticks, yticklabels = [], []
    for i, row in enumerate(rows):
        y = n_rows - 1 - i
        yticks.append(y)
        if row["kind"] == "header":
            yticklabels.append(row["label"])
            continue
        yticklabels.append(row["label"])
        is_summary = row["kind"] == "summary"
        row_color = summary_color if is_summary else color
        lo_x, hi_x, mid_x = _x(row["lower"]), _x(row["upper"]), _x(row["value"])
        ax.plot([lo_x, hi_x], [y, y], color=row_color,
                linewidth=1.4, solid_capstyle="butt", zorder=2)
        if is_summary:
            # diamond spanning the pooled interval
            ax.fill(
                [lo_x, mid_x, hi_x, mid_x],
                [y, y + 0.32, y, y - 0.32],
                color=row_color, zorder=3,
            )
        else:
            size = markersize * (0.5 + 1.5 * scaled[i])
            ax.scatter([mid_x], [y], s=size, marker=marker,
                       color=row_color, zorder=3, edgecolors="none")
    if log_scale:
        ax.set_xscale("log")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=fontsize)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row["kind"] == "header":
            tick.set_fontweight("bold")
        elif row["kind"] == "summary":
            tick.set_fontstyle("italic")
    ax.set_ylim(-0.8, n_rows - 0.2)

    if ref_line is None:
        ref_line = 1.0 if log_scale else 0.0
    ax.axvline(float(ref_line), color="0.5", linewidth=0.9, linestyle="--",
               zorder=1)

    if annotate:
        transform = ax.get_yaxis_transform()
        for i, row in enumerate(rows):
            if row["kind"] == "header":
                continue
            y = n_rows - 1 - i
            value, lo, hi = row["value"], row["lower"], row["upper"]
            if log_scale:
                # rows are held on the log scale internally; report ratios
                value, lo, hi = np.exp(value), np.exp(lo), np.exp(hi)
            text = _format_effect(value, lo, hi, digits)
            p_text = _format_p(row.get("pvalue", np.nan))
            if p_text:
                text += f"   P={p_text}"
            ax.text(annotate_x, y, text, transform=transform, va="center",
                    ha="left", fontsize=fontsize - 0.5,
                    fontstyle="italic" if row["kind"] == "summary" else "normal")

    if meta and "meta" in stats:
        first = next(iter(stats["meta"].values()))
        het = (f"Q = {first['Q']:.2f} (df={first['df']}, "
               f"P={_format_p(first['Q_pvalue'])});  "
               f"I$^2$ = {100 * first['I2']:.0f}%;  "
               f"$\\tau^2$ = {first['tau2']:.3f}")
        # placed in the blank strip below the last row, inside the axes, so it
        # can never collide with the tick labels however tall the figure is
        ax.text(0.005, -0.62, het, transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=fontsize - 1, color="0.35")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if log_scale:
        from matplotlib.ticker import FuncFormatter, LogLocator

        ax.xaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0),
                       numticks=10)
        )
        # ScalarFormatter picks one decimal count for the whole axis and so
        # renders 0.7 and 1.5 as "1"; %g keeps each tick readable instead
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(axis="x", labelsize=fontsize)

    if xlabel is None:
        xlabel = names.get("estimate", "Effect size")
        if log_scale:
            xlabel = f"{xlabel} (ratio scale)"
    ax.set_xlabel(xlabel, fontsize=fontsize + 1)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    stats["rows"] = pd.DataFrame(
        [r for r in rows if r["kind"] != "header"]
    )
    return (ax, stats) if return_stats else ax
