r"""Distribution plots: histograms, densities, ridgelines and Q-Q plots.

``ov.pl`` could draw a violin of an ``AnnData`` column but had no general
histogram, no density plot, no ridgeline, and a Q-Q plot only inside the GWAS
code. These are the container-free versions — a table and column names, bare
arrays, or an ``AnnData`` (its ``.obs``) all work.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._registry import register_function
from ._plot_backend import style_axes
from ._stats_common import (as_frame, default_palette, font_size,
                            group_levels, kde_curve, resolve_columns)

__all__ = ["histplot", "kdeplot", "ridgeplot", "qqplot"]


def _values_and_groups(data, x, hue, order, hue_order, *, name="x"):
    if data is not None and as_frame(data) is None:
        data, x, hue = None, data, x
    if x is None:
        raise TypeError("`x` is required.")
    frame, names = resolve_columns(data, x=x, hue=hue, require=("x",))
    hues = group_levels(frame["hue"], hue_order) if "hue" in frame else [None]
    return frame, hues, names, frame.attrs.get("n_dropped", 0)





@register_function(
    aliases=["直方图", "histplot", "histogram", "频数分布图", "分布直方图"],
    category="pl",
    description=(
        "Histogram with count / density / probability scaling, grouped "
        "overlays (layer, stack, dodge, fill) and an optional density curve"
    ),
    examples=[
        "ax = ov.pl.histplot(df, 'n_genes')",
        "# One distribution per batch, as densities so sizes are comparable",
        "ax = ov.pl.histplot(df, 'n_genes', hue='batch', stat='density',",
        "                    multiple='layer', kde=True)",
        "# From adata.obs, log x-axis",
        "ax = ov.pl.histplot(adata, 'total_counts', log_scale=True)",
    ],
    related=["pl.kdeplot", "pl.ridgeplot", "pl.violinplot", "pl.qc"],
)
def histplot(data: Any = None,
             x: Any = None,
             *,
             hue: Any = None,
             hue_order: Optional[Sequence[Any]] = None,
             bins: Union[int, str, Sequence[float]] = "auto",
             binrange: Optional[Tuple[float, float]] = None,
             stat: str = "count",
             multiple: str = "layer",
             cumulative: bool = False,
             kde: bool = False,
             bw_method: Any = None,
             log_scale: bool = False,
             palette=None,
             alpha: float = 0.6,
             edgecolor: str = "white",
             linewidth: float = 0.5,
             ax=None,
             figsize: Optional[Tuple[float, float]] = None,
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
             title: Optional[str] = None,
             legend: bool = True,
             legend_title: Optional[str] = None,
             fontsize: Optional[float] = None,
             return_stats: bool = False):
    r"""Histogram of one variable, optionally split by a factor.

    Arguments
    ---------
    bins
        Bin count, a NumPy binning rule (``'auto'``, ``'fd'``, ``'sturges'``,
        ...) or explicit edges. Edges are computed once over the pooled data so
        grouped histograms stay comparable.
    stat
        ``'count'`` (default), ``'density'`` (integrates to 1),
        ``'probability'`` (sums to 1) or ``'percent'``.
    multiple
        How groups are combined: ``'layer'`` (translucent overlay, default),
        ``'stack'``, ``'dodge'`` or ``'fill'`` (stacked and normalised to 1,
        which shows composition per bin).
    kde
        Overlay a Gaussian kernel density per group. Only meaningful with
        ``stat='density'``, and forced there if you ask for it.
    log_scale
        Bin on a log10 x-axis. Non-positive values are dropped, and how many
        is reported.

    Returns
    -------
    ``Axes``, or ``(Axes, DataFrame)`` with the bin edges and heights.
    """
    frame, hues, names, dropped = _values_and_groups(data, x, hue, None,
                                                     hue_order)
    if dropped:
        print(f"histplot: dropped {dropped} rows with missing values.")
    if stat not in {"count", "density", "probability", "percent"}:
        raise ValueError(
            "`stat` must be 'count', 'density', 'probability' or 'percent'."
        )
    if multiple not in {"layer", "stack", "dodge", "fill"}:
        raise ValueError("`multiple` must be 'layer', 'stack', 'dodge' or 'fill'.")

    values = frame["x"].to_numpy(dtype=float)
    if log_scale:
        positive = values > 0
        if not positive.all():
            print(f"histplot: dropped {int((~positive).sum())} non-positive "
                  f"values for the log axis.")
        frame = frame[positive].reset_index(drop=True)
        values = np.log10(frame["x"].to_numpy(dtype=float))

    if binrange is None:
        binrange = (float(values.min()), float(values.max()))
    edges = np.histogram_bin_edges(values, bins=bins, range=binrange)
    centres = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    ax = _new_axes(ax, figsize, (4.4, 3.2))
    colors = default_palette(max(len(hues), 1), palette)
    if kde and stat != "density":
        print("histplot: `kde=True` forces stat='density' so the curve and the "
              "bars share a scale.")
        stat = "density"

    heights = []
    for hue_level in hues:
        mask = (np.ones(len(frame), dtype=bool) if hue_level is None
                else frame["hue"].to_numpy() == hue_level)
        counts, _ = np.histogram(values[mask], bins=edges)
        counts = counts.astype(float)
        total = counts.sum()
        if stat == "density":
            counts = counts / (total * widths) if total else counts
        elif stat == "probability":
            counts = counts / total if total else counts
        elif stat == "percent":
            counts = counts / total * 100 if total else counts
        if cumulative:
            counts = np.cumsum(counts * (widths if stat == "density" else 1))
        heights.append(counts)
    matrix = np.vstack(heights) if heights else np.zeros((0, len(centres)))

    if multiple == "fill":
        totals = matrix.sum(axis=0)
        matrix = np.divide(matrix, totals, out=np.zeros_like(matrix),
                           where=totals > 0)

    base = np.zeros(len(centres))
    for index, (hue_level, colour) in enumerate(zip(hues, colors)):
        row = matrix[index]
        if multiple in {"stack", "fill"}:
            ax.bar(centres, row, bottom=base, width=widths, color=colour,
                   alpha=1.0 if multiple == "fill" else alpha,
                   edgecolor=edgecolor, linewidth=linewidth,
                   label=str(hue_level) if hue_level is not None else None)
            base = base + row
        elif multiple == "dodge":
            step = widths / max(len(hues), 1)
            ax.bar(centres - widths / 2 + step * (index + 0.5), row,
                   width=step * 0.95, color=colour, alpha=alpha,
                   edgecolor=edgecolor, linewidth=linewidth,
                   label=str(hue_level) if hue_level is not None else None)
        else:
            ax.bar(centres, row, width=widths, color=colour, alpha=alpha,
                   edgecolor=edgecolor, linewidth=linewidth,
                   label=str(hue_level) if hue_level is not None else None)
        if kde:
            mask = (np.ones(len(frame), dtype=bool) if hue_level is None
                    else frame["hue"].to_numpy() == hue_level)
            grid, curve = kde_curve(values[mask], cut=2.0, gridsize=200,
                                    bw_method=bw_method)
            ax.plot(grid, curve, color=colour, linewidth=1.5, zorder=3)

    label = xlabel if xlabel is not None else names.get("x", "")
    if log_scale:
        label = f"log10({label})" if label else "log10(value)"
    ax.set_xlabel(label, fontsize=font_size(fontsize, "label"))
    default_y = {"count": "count", "density": "density",
                 "probability": "proportion", "percent": "percent"}[stat]
    if multiple == "fill":
        default_y = "fraction of bin"
    ax.set_ylabel(ylabel if ylabel is not None else default_y,
                  fontsize=font_size(fontsize, "label"))
    if title:
        ax.set_title(title, fontsize=font_size(fontsize, "title"))
    ax.tick_params(labelsize=font_size(fontsize))
    style_axes(ax)
    if legend and len(hues) > 1:
        ax.legend(title=legend_title or names.get("hue"), frameon=False,
                  fontsize=font_size(fontsize))

    table = pd.DataFrame(matrix.T, index=pd.Index(centres, name="bin_centre"),
                         columns=[str(h) for h in hues])
    return (ax, table) if return_stats else ax


def _new_axes(ax, figsize, default):
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=figsize or default)
    return ax


@register_function(
    aliases=["核密度图", "kdeplot", "密度曲线", "density_plot", "kernel_density"],
    category="pl",
    description=(
        "Kernel density estimate — one curve per group in 1-D, or filled "
        "contours of a joint density in 2-D"
    ),
    examples=[
        "ax = ov.pl.kdeplot(df, 'score', hue='cell_type')",
        "# Shade under the curves, and mark the group means",
        "ax = ov.pl.kdeplot(df, 'score', hue='arm', fill=True, rug=True)",
        "# Joint density of two variables",
        "ax = ov.pl.kdeplot(df, 'UMAP1', y='UMAP2', levels=8, fill=True)",
    ],
    related=["pl.histplot", "pl.ridgeplot", "pl.violinplot", "pl.add_density_contour"],
)
def kdeplot(data: Any = None,
            x: Any = None,
            *,
            y: Any = None,
            hue: Any = None,
            hue_order: Optional[Sequence[Any]] = None,
            bw_method: Any = None,
            gridsize: int = 200,
            cut: float = 3.0,
            fill: bool = False,
            alpha: float = 0.3,
            linewidth: float = 1.6,
            cumulative: bool = False,
            rug: bool = False,
            levels: int = 10,
            cmap: str = "Blues",
            palette=None,
            ax=None,
            figsize: Optional[Tuple[float, float]] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            title: Optional[str] = None,
            legend: bool = True,
            legend_title: Optional[str] = None,
            fontsize: Optional[float] = None,
            return_stats: bool = False):
    r"""Kernel density estimate in one or two dimensions.

    Arguments
    ---------
    y
        Give this for a **2-D** joint density, drawn as contours. ``hue`` is
        not supported in 2-D — overlapping contour sets are unreadable; draw
        one panel per group instead.
    cut
        How far past the data the curve extends, in bandwidths. Set ``cut=0``
        for a bounded quantity so the density does not spill past 0 or 1.
    rug
        Draw a tick per observation along the axis, which shows where the
        density is actually supported by data.

    Returns
    -------
    ``Axes``, or ``(Axes, dict)``.
    """
    if y is not None and hue is not None:
        raise ValueError(
            "A 2-D density with `hue` would overlay unreadable contour sets — "
            "plot one group per axes instead."
        )
    frame, hues, names, dropped = _values_and_groups(data, x, hue, None,
                                                     hue_order)
    if dropped:
        print(f"kdeplot: dropped {dropped} rows with missing values.")
    ax = _new_axes(ax, figsize, (4.4, 3.2))
    curves: Dict[Any, Any] = {}

    if y is not None:
        from scipy.stats import gaussian_kde

        pair, pair_names = resolve_columns(data, x=x, y=y, require=("x", "y"))
        xs = pair["x"].to_numpy(dtype=float)
        ys = pair["y"].to_numpy(dtype=float)
        gx = np.linspace(xs.min(), xs.max(), gridsize)
        gy = np.linspace(ys.min(), ys.max(), gridsize)
        mesh_x, mesh_y = np.meshgrid(gx, gy)
        kernel = gaussian_kde(np.vstack([xs, ys]), bw_method=bw_method)
        density = kernel(np.vstack([mesh_x.ravel(), mesh_y.ravel()]))
        density = density.reshape(mesh_x.shape)
        if fill:
            ax.contourf(mesh_x, mesh_y, density, levels=levels, cmap=cmap)
        ax.contour(mesh_x, mesh_y, density, levels=levels, cmap=cmap,
                   linewidths=linewidth * 0.6)
        ax.set_xlabel(xlabel if xlabel is not None else pair_names.get("x", ""),
                      fontsize=font_size(fontsize, "label"))
        ax.set_ylabel(ylabel if ylabel is not None else pair_names.get("y", ""),
                      fontsize=font_size(fontsize, "label"))
        curves["density"] = density
    else:
        colors = default_palette(max(len(hues), 1), palette)
        values = frame["x"].to_numpy(dtype=float)
        for hue_level, colour in zip(hues, colors):
            mask = (np.ones(len(frame), dtype=bool) if hue_level is None
                    else frame["hue"].to_numpy() == hue_level)
            sample = values[mask]
            if sample.size < 2:
                continue
            grid, density = kde_curve(sample, cut=cut, gridsize=gridsize,
                                      bw_method=bw_method)
            if cumulative:
                density = np.cumsum(density) * np.gradient(grid)
            ax.plot(grid, density, color=colour, linewidth=linewidth,
                    label=str(hue_level) if hue_level is not None else None)
            if fill:
                ax.fill_between(grid, density, color=colour, alpha=alpha,
                                linewidth=0)
            if rug:
                ax.plot(sample, np.zeros_like(sample), "|", color=colour,
                        markersize=5, markeredgewidth=0.8, alpha=0.6,
                        clip_on=False)
            curves[hue_level] = (grid, density)
        ax.set_xlabel(xlabel if xlabel is not None else names.get("x", ""),
                      fontsize=font_size(fontsize, "label"))
        ax.set_ylabel(ylabel if ylabel is not None
                      else ("cumulative density" if cumulative else "density"),
                      fontsize=font_size(fontsize, "label"))
        ax.set_ylim(bottom=0)
        if legend and len(hues) > 1:
            ax.legend(title=legend_title or names.get("hue"), frameon=False,
                      fontsize=font_size(fontsize))

    if title:
        ax.set_title(title, fontsize=font_size(fontsize, "title"))
    ax.tick_params(labelsize=font_size(fontsize))
    style_axes(ax)
    return (ax, curves) if return_stats else ax


@register_function(
    aliases=["山脊图", "ridgeplot", "joyplot", "岭线图", "ridgeline"],
    category="pl",
    description=(
        "Ridgeline (joy) plot — one overlapping density per group, the "
        "readable way to compare a dozen distributions at once"
    ),
    examples=[
        "ax = ov.pl.ridgeplot(df, 'score', 'cell_type')",
        "# Ordered by median, coloured by it too",
        "ax = ov.pl.ridgeplot(df, 'score', 'cell_type', order='median',",
        "                     color_by='median', cmap='rocket')",
        "# Tighter stacking",
        "ax = ov.pl.ridgeplot(df, 'expression', 'sample', overlap=0.8)",
    ],
    related=["pl.kdeplot", "pl.violinplot", "pl.histplot"],
)
def ridgeplot(data: Any = None,
              x: Any = None,
              group: Any = None,
              *,
              order: Union[str, Sequence[Any], None] = "median",
              ascending: bool = True,
              overlap: float = 0.6,
              bw_method: Any = None,
              gridsize: int = 300,
              cut: float = 2.0,
              fill: bool = True,
              alpha: float = 0.85,
              linewidth: float = 0.9,
              edgecolor: str = "white",
              palette=None,
              color_by: Optional[str] = None,
              cmap: str = "viridis",
              quantile_lines: Optional[Sequence[float]] = None,
              ax=None,
              figsize: Optional[Tuple[float, float]] = None,
              xlabel: Optional[str] = None,
              title: Optional[str] = None,
              fontsize: Optional[float] = None,
              return_stats: bool = False):
    r"""Stacked, overlapping densities — one per group.

    Arguments
    ---------
    order
        ``'median'`` (default), ``'mean'``, ``'name'``, ``None`` to keep the
        natural order, or an explicit list.
    overlap
        How far each ridge reaches into the one above, as a fraction of the
        row height. 0 gives separated densities, 0.9 a dense ridge.
    color_by
        ``'median'`` or ``'mean'`` to map ridge colour onto that statistic via
        ``cmap`` — useful when the ordering statistic is the point.
    quantile_lines
        Draw vertical marks at these quantiles inside each ridge, e.g.
        ``[0.25, 0.5, 0.75]``.

    Returns
    -------
    ``Axes``, or ``(Axes, DataFrame)`` of per-group summaries.
    """
    if group is None:
        raise TypeError("`group` is required — it says what each ridge is.")
    if data is not None and as_frame(data) is None:
        data, x, group = None, data, x
    frame, names = resolve_columns(data, x=x, group=group,
                                   require=("x", "group"))
    values = frame["x"].to_numpy(dtype=float)
    labels = frame["group"].to_numpy()

    summary = (pd.DataFrame({"value": values, "group": labels})
               .groupby("group", observed=False)["value"]
               .agg(["median", "mean", "count"]))
    if isinstance(order, str):
        if order in {"median", "mean"}:
            levels = list(summary.sort_values(order, ascending=ascending).index)
        elif order == "name":
            levels = sorted(summary.index)
        else:
            raise ValueError("`order` must be 'median', 'mean', 'name', a list "
                             "or None.")
    elif order is None:
        levels = group_levels(frame["group"])
    else:
        levels = list(order)
    summary = summary.reindex(levels)

    ax = _new_axes(ax, figsize, (5.0, max(2.4, 0.42 * len(levels) + 1.0)))
    if color_by is not None:
        if color_by not in {"median", "mean"}:
            raise ValueError("`color_by` must be 'median', 'mean' or None.")
        stat = summary[color_by].to_numpy(dtype=float)
        span = np.nanmax(stat) - np.nanmin(stat)
        norm = (stat - np.nanmin(stat)) / span if span else np.zeros_like(stat)
        colors = [plt.get_cmap(cmap)(v) for v in norm]
    else:
        colors = default_palette(len(levels), palette)

    step = 1.0
    height = step / max(1.0 - overlap, 1e-3)
    peak = 0.0
    profiles = []
    for level in levels:
        sample = values[labels == level]
        if sample.size < 2:
            profiles.append(None)
            continue
        grid, density = kde_curve(sample, cut=cut, gridsize=gridsize,
                                  bw_method=bw_method)
        peak = max(peak, float(density.max()))
        profiles.append((grid, density, sample))

    for index, (level, profile, colour) in enumerate(
            zip(levels, profiles, colors)):
        baseline = (len(levels) - 1 - index) * step
        if profile is None:
            continue
        grid, density, sample = profile
        scaled = baseline + density / (peak or 1.0) * height
        if fill:
            ax.fill_between(grid, baseline, scaled, color=colour, alpha=alpha,
                            linewidth=linewidth, edgecolor=edgecolor,
                            zorder=len(levels) - index)
        ax.plot(grid, scaled, color=edgecolor if fill else colour,
                linewidth=linewidth, zorder=len(levels) - index)
        if quantile_lines:
            for q in quantile_lines:
                position = float(np.quantile(sample, q))
                top = np.interp(position, grid, scaled)
                ax.plot([position, position], [baseline, top], color="0.25",
                        linewidth=0.8, zorder=len(levels) - index + 0.5)

    ax.set_yticks([(len(levels) - 1 - i) * step for i in range(len(levels))])
    ax.set_yticklabels([str(level) for level in levels], fontsize=font_size(fontsize))
    ax.set_ylim(-0.2 * step, (len(levels) - 1) * step + height * 1.05)
    ax.set_xlabel(xlabel if xlabel is not None else names.get("x", ""),
                  fontsize=font_size(fontsize, "label"))
    if title:
        ax.set_title(title, fontsize=font_size(fontsize, "title"))
    ax.tick_params(axis="y", length=0)
    ax.tick_params(labelsize=font_size(fontsize))
    style_axes(ax, spines=False)
    ax.spines["bottom"].set_visible(True)
    return (ax, summary) if return_stats else ax


@register_function(
    aliases=["QQ图", "qqplot", "分位图", "quantile_plot", "qq_plot", "正态性检验图"],
    category="pl",
    description=(
        "Q-Q plot against a theoretical distribution or a second sample, with "
        "a reference line, confidence band, and genomic inflation for P values"
    ),
    examples=[
        "# Is this residual normal?",
        "ax = ov.pl.qqplot(residuals)",
        "# P values against uniform, on the -log10 scale, with lambda_GC",
        "ax = ov.pl.qqplot(deg['pvalue'], dist='uniform', log=True)",
        "# Two samples against each other",
        "ax = ov.pl.qqplot(sample_a, other=sample_b)",
    ],
    related=["pl.histplot", "pl.kdeplot", "pl.volcano"],
)
def qqplot(data: Any = None,
           x: Any = None,
           *,
           other: Any = None,
           dist: str = "norm",
           log: bool = False,
           line: Optional[str] = "quartile",
           ci: Optional[float] = 0.95,
           point_size: float = 10,
           color: str = "#1F577B",
           line_color: str = "0.4",
           ax=None,
           figsize: Optional[Tuple[float, float]] = None,
           xlabel: Optional[str] = None,
           ylabel: Optional[str] = None,
           title: Optional[str] = None,
           fontsize: Optional[float] = None,
           return_stats: bool = False):
    r"""Quantile-quantile plot.

    Arguments
    ---------
    data, x
        A table plus a column, or the values directly.
    other
        A second sample. Given this, quantiles of the two samples are plotted
        against each other and ``dist`` is ignored.
    dist
        Reference distribution — any name in ``scipy.stats``. ``'norm'``
        (default) answers "is this normal?"; ``'uniform'`` with ``log=True``
        is the P-value Q-Q that reports genomic inflation.
    log
        Plot ``-log10`` of both axes. Meant for P values.
    line
        ``'quartile'`` (default; robust, through the first and third
        quartiles), ``'45'`` (the identity), ``'ols'``, or ``None``.
    ci
        Pointwise confidence band from the Beta order statistics, at this
        level. ``None`` to skip.

    Returns
    -------
    ``Axes``, or ``(Axes, dict)`` — the latter includes ``lambda_gc`` when the
    reference is uniform.
    """
    from scipy import stats as scipy_stats

    if data is not None and as_frame(data) is None:
        data, x = None, data
    frame, names = resolve_columns(data, x=x, require=("x",))
    sample = np.sort(frame["x"].to_numpy(dtype=float))
    n = sample.size
    if n < 2:
        raise ValueError("A Q-Q plot needs at least two observations.")

    quantiles = (np.arange(1, n + 1) - 0.5) / n
    results: Dict[str, Any] = {"n": n}

    if other is not None:
        reference = np.sort(np.asarray(other, dtype=float))
        theoretical = np.quantile(reference, quantiles)
        x_label = xlabel if xlabel is not None else "second sample quantiles"
    else:
        try:
            distribution = getattr(scipy_stats, dist)
        except AttributeError as exc:
            raise ValueError(
                f"`dist={dist!r}` is not a scipy.stats distribution."
            ) from exc
        if dist == "norm":
            theoretical = distribution.ppf(quantiles)
        elif dist == "uniform":
            theoretical = quantiles
        else:
            theoretical = distribution.ppf(quantiles, *distribution.fit(sample))
        x_label = xlabel if xlabel is not None else f"theoretical quantiles ({dist})"

    if dist == "uniform" and other is None:
        # genomic inflation: median chi2 of the observed p values over the null
        with np.errstate(divide="ignore"):
            chi2 = scipy_stats.chi2.isf(sample, 1)
        results["lambda_gc"] = float(np.median(chi2) / scipy_stats.chi2.ppf(0.5, 1))

    plot_x, plot_y = theoretical, sample
    if log:
        with np.errstate(divide="ignore"):
            plot_x = -np.log10(np.clip(theoretical, 1e-300, None))
            plot_y = -np.log10(np.clip(sample, 1e-300, None))
        x_label = f"expected $-\\log_{{10}}(P)$"

    ax = _new_axes(ax, figsize, (3.8, 3.8))

    # The reference line defines the location-scale map from theoretical units
    # to sample units, and the confidence band has to travel through the same
    # map. For `dist='norm'` the theoretical axis is the *standard* normal while
    # the sample keeps its own units, so a band left in theoretical units lands
    # around zero while the points sit around the sample mean — visibly detached
    # (sepal_length: band -3.6..3.6 against points 4.3..7.9).
    def _reference_line():
        if line == "45":
            return 1.0, 0.0
        qx = np.quantile(plot_x, [0.25, 0.75])
        qy = np.quantile(plot_y, [0.25, 0.75])
        slope = (qy[1] - qy[0]) / (qx[1] - qx[0]) if qx[1] != qx[0] else 1.0
        return slope, qy[0] - slope * qx[0]

    if ci is not None and other is None:
        ranks = np.arange(1, n + 1)
        lo_q = scipy_stats.beta.ppf((1 - ci) / 2, ranks, n - ranks + 1)
        hi_q = scipy_stats.beta.ppf(1 - (1 - ci) / 2, ranks, n - ranks + 1)
        if dist == "uniform":
            # p-value scale on both axes: the band is already in sample units.
            band_lo, band_hi = lo_q, hi_q
        elif dist == "norm":
            slope, intercept = _reference_line()
            band_lo = slope * distribution.ppf(lo_q) + intercept
            band_hi = slope * distribution.ppf(hi_q) + intercept
        else:
            # theoretical quantiles came from parameters fitted to the sample,
            # so the fitted ppf puts the band in sample units directly.
            params = distribution.fit(sample)
            band_lo = distribution.ppf(lo_q, *params)
            band_hi = distribution.ppf(hi_q, *params)
        if band_lo is not None:
            if log:
                with np.errstate(divide="ignore"):
                    band_lo = -np.log10(np.clip(band_lo, 1e-300, None))
                    band_hi = -np.log10(np.clip(band_hi, 1e-300, None))
            ax.fill_between(plot_x, band_lo, band_hi, color="0.85",
                            linewidth=0, zorder=0)

    ax.scatter(plot_x, plot_y, s=point_size, color=color, linewidths=0,
               zorder=2)

    if line == "45":
        lo = min(plot_x.min(), plot_y.min())
        hi = max(plot_x.max(), plot_y.max())
        ax.plot([lo, hi], [lo, hi], color=line_color, linewidth=1.0,
                linestyle="--", zorder=1)
    elif line == "quartile":
        qx = np.quantile(plot_x, [0.25, 0.75])
        qy = np.quantile(plot_y, [0.25, 0.75])
        slope = (qy[1] - qy[0]) / (qx[1] - qx[0]) if qx[1] != qx[0] else 1.0
        intercept = qy[0] - slope * qx[0]
        edges = np.array([plot_x.min(), plot_x.max()])
        ax.plot(edges, slope * edges + intercept, color=line_color,
                linewidth=1.0, linestyle="--", zorder=1)
        results.update(slope=float(slope), intercept=float(intercept))
    elif line == "ols":
        slope, intercept = np.polyfit(plot_x, plot_y, 1)
        edges = np.array([plot_x.min(), plot_x.max()])
        ax.plot(edges, slope * edges + intercept, color=line_color,
                linewidth=1.0, linestyle="--", zorder=1)
        results.update(slope=float(slope), intercept=float(intercept))
    elif line is not None:
        raise ValueError("`line` must be 'quartile', '45', 'ols' or None.")

    ax.set_xlabel(x_label, fontsize=font_size(fontsize, "label"))
    default_y = "observed $-\\log_{10}(P)$" if log else (
        f"sample quantiles ({names.get('x', '')})".strip())
    ax.set_ylabel(ylabel if ylabel is not None else default_y,
                  fontsize=font_size(fontsize, "label"))
    if "lambda_gc" in results:
        ax.text(0.03, 0.96, f"$\\lambda_{{GC}}$ = {results['lambda_gc']:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=font_size(fontsize))
    if title:
        ax.set_title(title, fontsize=font_size(fontsize, "title"))
    ax.tick_params(labelsize=font_size(fontsize))
    style_axes(ax)
    return (ax, results) if return_stats else ax
