r"""Relational plots: scatter, line and regression.

Every scatter in ``ov.pl`` today is an embedding — it needs an ``AnnData`` and
a ``basis``. There was no way to simply plot y against x from a table, and no
regression plot at all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._registry import register_function
from ._plot_backend import style_axes
from ._stats_common import (as_frame, default_palette, font_size,
                            group_levels, resolve_columns)

__all__ = ["scatterplot", "lineplot", "regplot"]


def _new_axes(ax, figsize, default):
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=figsize or default)
    return ax


def _correlation(x: np.ndarray, y: np.ndarray, method: str):
    from scipy import stats

    if method == "pearson":
        res = stats.pearsonr(x, y)
        return float(res[0]), float(res[1]), "r"
    if method == "spearman":
        res = stats.spearmanr(x, y)
        return float(res[0]), float(res[1]), r"$\rho$"
    if method == "kendall":
        res = stats.kendalltau(x, y)
        return float(res[0]), float(res[1]), r"$\tau$"
    raise ValueError(
        f"`corr` must be 'pearson', 'spearman', 'kendall' or None, got "
        f"{method!r}."
    )


def _format_corr(value: float, pvalue: float, symbol: str) -> str:
    from ._stats_tests import format_pvalue

    return f"{symbol} = {value:.3f}\n{format_pvalue(pvalue, 'value')}"


def _shift(data, *fields):
    """Allow ``f(x_array, y_array, ...)`` when the first positional is data."""
    if data is not None and as_frame(data) is None:
        return (None, data) + fields[:-1]
    return (data,) + fields


@register_function(
    aliases=["散点图", "scatterplot", "scatter", "点图", "相关散点图"],
    category="pl",
    description=(
        "Scatter plot of y against x from a table, with categorical or "
        "continuous colour, point sizing, density shading and a correlation "
        "annotation"
    ),
    examples=[
        "ax = ov.pl.scatterplot(df, 'total_counts', 'n_genes')",
        "# Colour by a category, annotate Spearman's rho",
        "ax = ov.pl.scatterplot(df, 'x', 'y', hue='cell_type', corr='spearman')",
        "# Continuous colour and size, for a big table",
        "ax = ov.pl.scatterplot(df, 'log2FC', 'neglog10P', hue='baseMean',",
        "                       size='n', cmap='viridis', rasterized=True)",
        "# Colour by local point density instead of a variable",
        "ax = ov.pl.scatterplot(df, 'x', 'y', density=True)",
    ],
    related=["pl.regplot", "pl.lineplot", "pl.embedding", "pl.volcano"],
)
def scatterplot(data: Any = None,
                x: Any = None,
                y: Any = None,
                *,
                hue: Any = None,
                size: Any = None,
                hue_order: Optional[Sequence[Any]] = None,
                palette=None,
                cmap: str = "viridis",
                density: bool = False,
                s: float = 18,
                size_range: Tuple[float, float] = (8.0, 90.0),
                alpha: float = 0.8,
                edgecolor: str = "none",
                corr: Optional[str] = None,
                diagonal: bool = False,
                rasterized: bool = False,
                log_x: bool = False,
                log_y: bool = False,
                ax=None,
                figsize: Optional[Tuple[float, float]] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                title: Optional[str] = None,
                legend: bool = True,
                legend_title: Optional[str] = None,
                colorbar_label: Optional[str] = None,
                fontsize: Optional[float] = None,
                return_stats: bool = False):
    r"""Plain y-against-x scatter.

    Arguments
    ---------
    hue
        Column or array. Categorical values get one colour each from
        ``palette``; numeric values get ``cmap`` and a colourbar.
    size
        Column or array mapped onto marker area between ``size_range``.
    density
        Colour each point by the local point density (Gaussian KDE) instead of
        a variable — the fix for an overplotted cloud where every point looks
        the same.
    corr
        ``'pearson'``, ``'spearman'`` or ``'kendall'`` to annotate the
        coefficient and its P value.
    diagonal
        Draw the identity line, for when x and y are the same quantity
        measured two ways.
    rasterized
        Rasterise the points but keep axes and text as vectors — the right
        trade for a 100,000-point figure that still has to be an editable PDF.

    Returns
    -------
    ``Axes``, or ``(Axes, dict)``.
    """
    data, x, y, hue = _shift(data, x, y, hue)
    frame, names = resolve_columns(data, x=x, y=y, hue=hue, size=size,
                                   require=("x", "y"))
    dropped = frame.attrs.get("n_dropped", 0)
    if dropped:
        print(f"scatterplot: dropped {dropped} rows with missing values.")
    xs = frame["x"].to_numpy(dtype=float)
    ys = frame["y"].to_numpy(dtype=float)

    ax = _new_axes(ax, figsize, (4.0, 3.6))
    sizes = s
    if "size" in frame:
        raw = frame["size"].to_numpy(dtype=float)
        span = np.nanmax(raw) - np.nanmin(raw)
        norm = (raw - np.nanmin(raw)) / span if span else np.full_like(raw, 0.5)
        sizes = size_range[0] + norm * (size_range[1] - size_range[0])

    results: Dict[str, Any] = {"n": len(frame)}
    mappable = None

    if density:
        from scipy.stats import gaussian_kde

        try:
            weights = gaussian_kde(np.vstack([xs, ys]))(np.vstack([xs, ys]))
        except np.linalg.LinAlgError:
            weights = np.zeros_like(xs)
        order = np.argsort(weights)  # densest points drawn last
        mappable = ax.scatter(xs[order], ys[order], c=weights[order], s=sizes,
                              cmap=cmap, alpha=alpha, linewidths=0,
                              rasterized=rasterized)
        colorbar_label = colorbar_label or "point density"
    elif "hue" in frame:
        values = frame["hue"]
        numeric = pd.api.types.is_numeric_dtype(values) and values.nunique() > 12
        if numeric:
            mappable = ax.scatter(xs, ys, c=values.to_numpy(dtype=float),
                                  s=sizes, cmap=cmap, alpha=alpha,
                                  edgecolor=edgecolor, linewidths=0,
                                  rasterized=rasterized)
            colorbar_label = colorbar_label or names.get("hue")
        else:
            levels = group_levels(values, hue_order)
            colors = default_palette(len(levels), palette)
            labels = values.to_numpy()
            for level, colour in zip(levels, colors):
                mask = labels == level
                point_sizes = (sizes if np.isscalar(sizes) else sizes[mask])
                ax.scatter(xs[mask], ys[mask], s=point_sizes, color=colour,
                           alpha=alpha, edgecolor=edgecolor, linewidths=0,
                           label=str(level), rasterized=rasterized)
            if legend:
                ax.legend(title=legend_title or names.get("hue"), frameon=False,
                          fontsize=font_size(fontsize), title_fontsize=font_size(fontsize))
    else:
        ax.scatter(xs, ys, s=sizes, color="#1F577B", alpha=alpha,
                   edgecolor=edgecolor, linewidths=0, rasterized=rasterized)

    if mappable is not None:
        bar = ax.figure.colorbar(mappable, ax=ax, fraction=0.04, pad=0.03)
        bar.set_label(colorbar_label or "", fontsize=font_size(fontsize))
        bar.ax.tick_params(labelsize=font_size(fontsize) - 1)

    if corr:
        value, pvalue, symbol = _correlation(xs, ys, corr)
        results.update(correlation=value, pvalue=pvalue, method=corr)
        ax.text(0.03, 0.97, _format_corr(value, pvalue, symbol),
                transform=ax.transAxes, va="top", ha="left", fontsize=font_size(fontsize))
    if diagonal:
        lo = min(xs.min(), ys.min())
        hi = max(xs.max(), ys.max())
        ax.plot([lo, hi], [lo, hi], color="0.55", linewidth=0.9,
                linestyle="--", zorder=0)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel if xlabel is not None else names.get("x", ""),
                  fontsize=font_size(fontsize, "label"))
    ax.set_ylabel(ylabel if ylabel is not None else names.get("y", ""),
                  fontsize=font_size(fontsize, "label"))
    if title:
        ax.set_title(title, fontsize=font_size(fontsize, "title"))
    ax.tick_params(labelsize=font_size(fontsize))
    style_axes(ax)
    return (ax, results) if return_stats else ax


def _aggregate(frame: pd.DataFrame, estimator: str, errorbar, ci_level: float):
    from ._categorical import _error, _ESTIMATORS

    reduce = _ESTIMATORS[estimator]
    rows = []
    for value, part in frame.groupby("x", observed=False, sort=True):
        ys = part["y"].to_numpy(dtype=float)
        low, high = _error(ys, errorbar, ci_level)
        rows.append({"x": value, "centre": float(reduce(ys)), "low": low,
                     "high": high, "n": ys.size})
    return pd.DataFrame(rows).sort_values("x")


@register_function(
    aliases=["折线图", "lineplot", "line_chart", "趋势图", "时间序列图"],
    category="pl",
    description=(
        "Line plot with automatic aggregation over replicates and an error "
        "band — for time courses, dose responses and titration curves"
    ),
    examples=[
        "ax = ov.pl.lineplot(df, 'hour', 'expression', hue='gene')",
        "# Replicates averaged with a 95% CI band",
        "ax = ov.pl.lineplot(df, 'dose', 'viability', hue='cell_line',",
        "                    errorbar='ci')",
        "# Draw each replicate instead of aggregating",
        "ax = ov.pl.lineplot(df, 'hour', 'value', hue='gene', units='replicate')",
    ],
    related=["pl.scatterplot", "pl.regplot", "pl.dynamic_trends"],
)
def lineplot(data: Any = None,
             x: Any = None,
             y: Any = None,
             *,
             hue: Any = None,
             units: Any = None,
             hue_order: Optional[Sequence[Any]] = None,
             estimator: str = "mean",
             errorbar: Optional[str] = "ci",
             ci_level: float = 0.95,
             band_alpha: float = 0.18,
             markers: bool = False,
             linewidth: float = 1.6,
             palette=None,
             log_x: bool = False,
             log_y: bool = False,
             ax=None,
             figsize: Optional[Tuple[float, float]] = None,
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
             title: Optional[str] = None,
             legend: bool = True,
             legend_title: Optional[str] = None,
             fontsize: Optional[float] = None,
             return_stats: bool = False):
    r"""Line plot, aggregating repeated measurements at each x.

    Arguments
    ---------
    estimator
        ``'mean'`` (default), ``'median'``, ``'sum'``, ``'max'``, ``'min'``.
    errorbar
        ``'ci'`` (t-based, default), ``'se'``, ``'sd'``, ``'pi'`` (percentile
        interval) or ``None``.
    units
        Column identifying individual replicates. Given this, every replicate
        is drawn as its own thin line and no aggregation happens — honest when
        n is small enough that a mean would hide the spread.

    Returns
    -------
    ``Axes``, or ``(Axes, DataFrame)`` with the aggregated values.
    """
    data, x, y, hue = _shift(data, x, y, hue)
    frame, names = resolve_columns(data, x=x, y=y, hue=hue, units=units,
                                   require=("x", "y"))
    dropped = frame.attrs.get("n_dropped", 0)
    if dropped:
        print(f"lineplot: dropped {dropped} rows with missing values.")
    hues = group_levels(frame["hue"], hue_order) if "hue" in frame else [None]

    ax = _new_axes(ax, figsize, (4.4, 3.2))
    colors = default_palette(max(len(hues), 1), palette)
    tables = []
    for hue_level, colour in zip(hues, colors):
        part = (frame if hue_level is None
                else frame[frame["hue"].to_numpy() == hue_level])
        if part.empty:
            continue
        label = str(hue_level) if hue_level is not None else None
        if "units" in part:
            for unit, one in part.groupby("units", observed=False):
                one = one.sort_values("x")
                ax.plot(one["x"], one["y"], color=colour, linewidth=0.8,
                        alpha=0.55, zorder=2)
            summary = _aggregate(part, estimator, None, ci_level)
            ax.plot(summary["x"], summary["centre"], color=colour,
                    linewidth=linewidth + 0.6, label=label, zorder=3)
        else:
            summary = _aggregate(part, estimator, errorbar, ci_level)
            ax.plot(summary["x"], summary["centre"], color=colour,
                    linewidth=linewidth, label=label,
                    marker="o" if markers else None, markersize=4, zorder=3)
            if errorbar and (summary["low"].any() or summary["high"].any()):
                ax.fill_between(summary["x"],
                                summary["centre"] - summary["low"],
                                summary["centre"] + summary["high"],
                                color=colour, alpha=band_alpha, linewidth=0,
                                zorder=1)
        summary = summary.assign(hue=hue_level)
        tables.append(summary)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel if xlabel is not None else names.get("x", ""),
                  fontsize=font_size(fontsize, "label"))
    ax.set_ylabel(ylabel if ylabel is not None else names.get("y", ""),
                  fontsize=font_size(fontsize, "label"))
    if title:
        ax.set_title(title, fontsize=font_size(fontsize, "title"))
    ax.tick_params(labelsize=font_size(fontsize))
    style_axes(ax)
    if legend and len(hues) > 1:
        ax.legend(title=legend_title or names.get("hue"), frameon=False,
                  fontsize=font_size(fontsize), title_fontsize=font_size(fontsize))

    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return (ax, result) if return_stats else ax


@register_function(
    aliases=["回归图", "regplot", "regression_plot", "拟合图", "相关回归图"],
    category="pl",
    description=(
        "Scatter with a fitted trend and confidence band — linear, "
        "polynomial or LOWESS — annotated with slope, R-squared and P"
    ),
    examples=[
        "ax = ov.pl.regplot(df, 'gene_A', 'gene_B')",
        "# A non-parametric trend instead of a straight line",
        "ax = ov.pl.regplot(df, 'pseudotime', 'expression', fit='lowess')",
        "# One fit per group",
        "ax = ov.pl.regplot(df, 'dose', 'response', hue='cell_line', fit='poly',",
        "                   order=2)",
    ],
    related=["pl.scatterplot", "pl.lineplot", "pl.volcano"],
)
def regplot(data: Any = None,
            x: Any = None,
            y: Any = None,
            *,
            hue: Any = None,
            hue_order: Optional[Sequence[Any]] = None,
            fit: str = "linear",
            order: int = 2,
            lowess_frac: float = 0.4,
            ci: Optional[float] = 0.95,
            scatter: bool = True,
            s: float = 16,
            alpha: float = 0.7,
            band_alpha: float = 0.2,
            linewidth: float = 1.8,
            palette=None,
            annotate: bool = True,
            corr: str = "pearson",
            ax=None,
            figsize: Optional[Tuple[float, float]] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            title: Optional[str] = None,
            legend: bool = True,
            legend_title: Optional[str] = None,
            fontsize: Optional[float] = None,
            return_stats: bool = False):
    r"""Scatter plus a fitted trend.

    Arguments
    ---------
    fit
        ``'linear'`` (default), ``'poly'`` (degree ``order``) or ``'lowess'``
        (locally weighted, span ``lowess_frac``).
    ci
        Confidence level of the band around the fit. For ``'linear'`` and
        ``'poly'`` this is the analytic interval for the mean response; for
        LOWESS no band is drawn, because the honest one needs a bootstrap that
        this function does not run.
    annotate
        Print slope, R² and the correlation with its P value. Only for a
        single ungrouped linear fit — with several groups the box would be
        larger than the plot.

    Returns
    -------
    ``Axes``, or ``(Axes, dict)`` keyed by group.
    """
    data, x, y, hue = _shift(data, x, y, hue)
    frame, names = resolve_columns(data, x=x, y=y, hue=hue, require=("x", "y"))
    dropped = frame.attrs.get("n_dropped", 0)
    if dropped:
        print(f"regplot: dropped {dropped} rows with missing values.")
    hues = group_levels(frame["hue"], hue_order) if "hue" in frame else [None]
    if fit not in {"linear", "poly", "lowess"}:
        raise ValueError("`fit` must be 'linear', 'poly' or 'lowess'.")

    ax = _new_axes(ax, figsize, (4.0, 3.6))
    colors = default_palette(max(len(hues), 1), palette)
    results: Dict[Any, Dict[str, float]] = {}

    for hue_level, colour in zip(hues, colors):
        part = (frame if hue_level is None
                else frame[frame["hue"].to_numpy() == hue_level])
        xs = part["x"].to_numpy(dtype=float)
        ys = part["y"].to_numpy(dtype=float)
        if xs.size < 3:
            continue
        label = str(hue_level) if hue_level is not None else None
        if scatter:
            ax.scatter(xs, ys, s=s, color=colour, alpha=alpha, linewidths=0,
                       label=label, zorder=2)
        grid = np.linspace(xs.min(), xs.max(), 200)

        if fit == "lowess":
            import statsmodels.api as sm

            smoothed = sm.nonparametric.lowess(ys, xs, frac=lowess_frac,
                                               return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color=colour,
                    linewidth=linewidth, zorder=3,
                    label=None if scatter else label)
            results[hue_level] = {"n": int(xs.size), "fit": "lowess"}
            continue

        degree = 1 if fit == "linear" else int(order)
        coefficients = np.polyfit(xs, ys, degree)
        fitted = np.polyval(coefficients, grid)
        ax.plot(grid, fitted, color=colour, linewidth=linewidth, zorder=3,
                label=None if scatter else label)

        residuals = ys - np.polyval(coefficients, xs)
        dof = xs.size - (degree + 1)
        entry: Dict[str, float] = {"n": int(xs.size), "slope": float(coefficients[-2]),
                                   "intercept": float(coefficients[-1])}
        if dof > 0:
            sigma2 = float(residuals @ residuals / dof)
            total = float(((ys - ys.mean()) ** 2).sum())
            entry["r_squared"] = 1.0 - float(residuals @ residuals) / total if total else np.nan
            if ci is not None:
                from scipy import stats

                design = np.vander(xs, degree + 1)
                grid_design = np.vander(grid, degree + 1)
                xtx_inv = np.linalg.pinv(design.T @ design)
                se_fit = np.sqrt(np.maximum(
                    sigma2 * np.einsum("ij,jk,ik->i", grid_design, xtx_inv,
                                       grid_design), 0.0))
                t = stats.t.ppf(0.5 + ci / 2, dof)
                ax.fill_between(grid, fitted - t * se_fit, fitted + t * se_fit,
                                color=colour, alpha=band_alpha, linewidth=0,
                                zorder=1)
        value, pvalue, symbol = _correlation(xs, ys, corr)
        entry.update(correlation=value, pvalue=pvalue)
        results[hue_level] = entry

    if annotate and len(hues) == 1 and hues[0] is None and results:
        entry = results[None]
        lines = []
        if "slope" in entry:
            lines.append(f"slope = {entry['slope']:.3g}")
        if "r_squared" in entry:
            lines.append(f"$R^2$ = {entry['r_squared']:.3f}")
        if "correlation" in entry:
            from ._stats_tests import format_pvalue

            lines.append(f"{corr} = {entry['correlation']:.3f}")
            lines.append(format_pvalue(entry["pvalue"], "value"))
        if lines:
            ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes,
                    va="top", ha="left", fontsize=font_size(fontsize))

    ax.set_xlabel(xlabel if xlabel is not None else names.get("x", ""),
                  fontsize=font_size(fontsize, "label"))
    ax.set_ylabel(ylabel if ylabel is not None else names.get("y", ""),
                  fontsize=font_size(fontsize, "label"))
    if title:
        ax.set_title(title, fontsize=font_size(fontsize, "title"))
    ax.tick_params(labelsize=font_size(fontsize))
    style_axes(ax)
    if legend and len(hues) > 1:
        ax.legend(title=legend_title or names.get("hue"), frameon=False,
                  fontsize=font_size(fontsize), title_fontsize=font_size(fontsize))
    return (ax, results) if return_stats else ax
