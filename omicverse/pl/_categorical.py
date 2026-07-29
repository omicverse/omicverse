r"""Categorical plots that take a table, not an ``AnnData``.

``ov.pl`` has a violin plot and a stacked-proportion plot, but both start from
an ``AnnData``. These are the container-free versions, in the shape a
statistician expects: a long-form table plus the names of the category column
and the value column.

All of them accept ``test=`` and route it through
:func:`~omicverse.pl.compare_groups`, so the significance brackets are drawn
from the same numbers you get back — the annotation cannot drift away from the
statistics.

The ``...plot`` suffix marks this generic layer and keeps it clear of the
existing omics-specific names (``ov.pl.violin`` still takes an ``AnnData``;
``ov.pl.violinplot`` takes a frame).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._registry import register_function
from ._plot_backend import style_axes
from ._stats_common import (as_frame, default_palette, font_size,
                            group_levels, kde_curve, resolve_columns)

__all__ = [
    "barplot", "stripplot", "violinplot", "stackplot", "lollipopplot",
    "pieplot", "donutplot", "slopeplot", "sankey",
]


# --------------------------------------------------------------------------
# shared layout helpers
# --------------------------------------------------------------------------


def _long_frame(data, x, y, hue, order, hue_order, *, require_y=True):
    """Normalise (data, x, y, hue) into a tidy frame plus level orders."""
    if data is not None and as_frame(data) is None:
        data, x, y, hue = None, data, x, y
    if x is None:
        raise TypeError("`x` (the category) is required.")
    if require_y and y is None:
        raise TypeError("`y` (the value) is required.")

    frame, names = resolve_columns(data, x=x, y=y, hue=hue, require=("x",))
    dropped = frame.attrs.get("n_dropped", 0)
    levels = group_levels(frame["x"], order)
    hues = group_levels(frame["hue"], hue_order) if "hue" in frame else [None]
    return frame, levels, hues, names, dropped


def _offsets(n_hue: int, width: float) -> np.ndarray:
    """Dodge offsets for ``n_hue`` sub-categories inside one slot."""
    if n_hue <= 1:
        return np.array([0.0])
    step = width / n_hue
    return (np.arange(n_hue) - (n_hue - 1) / 2.0) * step


def _finish(ax, *, levels, names, xlabel, ylabel, title, fontsize, orient,
            rotation=0, categorical_axis=True):
    labels = [str(level) for level in levels]
    if categorical_axis:
        if orient == "v":
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(labels, fontsize=font_size(fontsize),
                               rotation=rotation,
                               ha="right" if rotation else "center")
        else:
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels(labels, fontsize=font_size(fontsize))
    cat_label = xlabel if xlabel is not None else names.get("x", "")
    val_label = ylabel if ylabel is not None else names.get("y", "")
    if orient == "v":
        ax.set_xlabel(cat_label, fontsize=font_size(fontsize, "label"))
        ax.set_ylabel(val_label, fontsize=font_size(fontsize, "label"))
    else:
        ax.set_ylabel(cat_label, fontsize=font_size(fontsize, "label"))
        ax.set_xlabel(val_label, fontsize=font_size(fontsize, "label"))
    if title:
        ax.set_title(title, fontsize=font_size(fontsize, "title"))
    ax.tick_params(labelsize=font_size(fontsize))
    style_axes(ax)


def _hue_legend(ax, hues, colors, title, fontsize, show):
    if not show or len(hues) <= 1 or hues[0] is None:
        return
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(facecolor=c, label=str(h))
                       for h, c in zip(hues, colors)],
              title=title, frameon=False, fontsize=font_size(fontsize),
              title_fontsize=font_size(fontsize))


def _maybe_annotate(ax, frame, levels, test, correction, style, hide_ns,
                    orient, fontsize):
    """Run the requested comparison and draw brackets. Returns the table."""
    if not test or test in ("none", False):
        return None
    from ._stats_tests import add_stat_annotation

    _, results = add_stat_annotation(
        ax, value=frame["y"].to_numpy(dtype=float),
        group=frame["x"].to_numpy(), order=levels, test=test,
        correction=correction, style=style, hide_ns=hide_ns, orient=orient,
        fontsize=font_size(fontsize), return_stats=True,
    )
    return results


def _series_colors(levels, hues, palette):
    """One colour per hue level, or — with no hue — one per category.

    Colouring by category when there is nothing else to encode is what makes a
    four-group violin readable at a glance, and it is what the ov palettes
    exist for.
    """
    if len(hues) > 1 or hues[0] is not None:
        return default_palette(len(hues), palette), False
    return default_palette(len(levels), palette), True


def _new_axes(ax, figsize, default):
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=figsize or default)
    return ax


# --------------------------------------------------------------------------
# bar / strip / violin
# --------------------------------------------------------------------------


_ESTIMATORS = {"mean": np.mean, "median": np.median, "sum": np.sum,
               "count": len, "max": np.max, "min": np.min}


def _error(values: np.ndarray, kind, ci_level: float) -> Tuple[float, float]:
    if kind in (None, "none") or values.size < 2:
        return 0.0, 0.0
    if kind == "sd":
        half = float(np.std(values, ddof=1))
        return half, half
    if kind == "se":
        half = float(np.std(values, ddof=1) / np.sqrt(values.size))
        return half, half
    if kind == "ci":
        from scipy import stats

        se = np.std(values, ddof=1) / np.sqrt(values.size)
        half = float(stats.t.ppf(0.5 + ci_level / 2, values.size - 1) * se)
        return half, half
    if kind == "pi":  # percentile interval of the raw values
        lo, hi = np.percentile(values, [(1 - ci_level) * 50,
                                        100 - (1 - ci_level) * 50])
        centre = np.mean(values)
        return float(centre - lo), float(hi - centre)
    raise ValueError(
        f"`errorbar` must be 'sd', 'se', 'ci', 'pi' or None, got {kind!r}."
    )


@register_function(
    aliases=["柱状图", "barplot", "条形图", "bar_chart", "均值柱图"],
    category="pl",
    description=(
        "Bar plot of a summary statistic per category with error bars, "
        "optional dodged groups, raw-point overlay and significance brackets"
    ),
    examples=[
        "ax = ov.pl.barplot(df, 'cell_type', 'score')",
        "# Dodged by condition, with the raw points on top",
        "ax = ov.pl.barplot(df, 'cell_type', 'score', hue='condition', dots=True)",
        "# Median with a bootstrap-free percentile interval, and a test",
        "ax = ov.pl.barplot(df, 'arm', 'value', estimator='median',",
        "                   errorbar='pi', test='mannwhitney')",
    ],
    related=["pl.stripplot", "pl.violinplot", "pl.compare_groups", "pl.boxplot"],
)
def barplot(data: Any = None,
            x: Any = None,
            y: Any = None,
            *,
            hue: Any = None,
            order: Optional[Sequence[Any]] = None,
            hue_order: Optional[Sequence[Any]] = None,
            estimator: str = "mean",
            errorbar: Optional[str] = "se",
            ci_level: float = 0.95,
            dots: bool = False,
            dot_size: float = 12,
            dot_jitter: float = 0.12,
            dot_alpha: float = 0.7,
            orient: str = "v",
            width: float = 0.8,
            palette=None,
            edgecolor: str = "none",
            ax=None,
            figsize: Optional[Tuple[float, float]] = None,
            test: Optional[str] = None,
            correction: Optional[str] = "holm",
            annot_style: str = "star",
            hide_ns: bool = False,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            title: Optional[str] = None,
            legend: bool = True,
            legend_title: Optional[str] = None,
            rotation: float = 0,
            fontsize: Optional[float] = None,
            random_state: int = 0,
            return_stats: bool = False):
    r"""Bars of a per-category summary, with error bars.

    Arguments
    ---------
    data, x, y
        Long-form table plus the category and value columns; or bare arrays.
    hue
        Optional second factor — bars are dodged within each category.
    estimator
        ``'mean'`` (default), ``'median'``, ``'sum'``, ``'count'``, ``'max'``,
        ``'min'``.
    errorbar
        ``'se'`` (default), ``'sd'``, ``'ci'`` (t-based at ``ci_level``),
        ``'pi'`` (percentile interval of the raw values), or ``None``.
    dots
        Draw the raw observations over the bars. Worth turning on: a bar with
        n=3 looks exactly like a bar with n=300 until you can see the points.
    test
        Run a comparison and draw brackets — any test accepted by
        :func:`~omicverse.pl.compare_groups`, e.g. ``'mannwhitney'``,
        ``'welch'``, or ``'auto'`` (which is Mann-Whitney U, fixed, not
        picked from the data). Ignored when ``hue`` is set.
    return_stats
        Return ``(ax, results)`` with the summary table and, if a test ran,
        its results.

    Returns
    -------
    ``Axes``, or ``(Axes, dict)``.
    """
    frame, levels, hues, names, dropped = _long_frame(data, x, y, hue, order,
                                                      hue_order)
    if dropped:
        print(f"barplot: dropped {dropped} rows with missing values.")
    if estimator not in _ESTIMATORS:
        raise ValueError(
            f"`estimator` must be one of {sorted(_ESTIMATORS)}, got {estimator!r}."
        )
    reduce = _ESTIMATORS[estimator]

    ax = _new_axes(ax, figsize, (max(3.2, 0.7 * len(levels) + 1.2), 3.4))
    colors, by_category = _series_colors(levels, hues, palette)
    offsets = _offsets(len(hues), width)
    bar_width = width / max(len(hues), 1) * 0.9
    rng = np.random.default_rng(random_state)

    summary = []
    for h_index, hue_level in enumerate(hues):
        for x_index, level in enumerate(levels):
            mask = frame["x"].to_numpy() == level
            if hue_level is not None:
                mask &= frame["hue"].to_numpy() == hue_level
            values = frame.loc[mask, "y"].to_numpy(dtype=float)
            if values.size == 0:
                continue
            centre = float(reduce(values))
            low, high = _error(values, errorbar, ci_level)
            position = x_index + offsets[h_index]
            summary.append({"x": level, "hue": hue_level, "n": values.size,
                            estimator: centre, "err_low": low, "err_high": high})
            colour = colors[x_index if by_category else h_index]
            if orient == "v":
                ax.bar(position, centre, width=bar_width, color=colour,
                       edgecolor=edgecolor, zorder=2)
                if low or high:
                    ax.errorbar(position, centre, yerr=[[low], [high]],
                                fmt="none", ecolor="0.25", elinewidth=1.0,
                                capsize=2.5, zorder=3)
            else:
                ax.barh(position, centre, height=bar_width, color=colour,
                        edgecolor=edgecolor, zorder=2)
                if low or high:
                    ax.errorbar(centre, position, xerr=[[low], [high]],
                                fmt="none", ecolor="0.25", elinewidth=1.0,
                                capsize=2.5, zorder=3)
            if dots:
                jitter = rng.uniform(-dot_jitter, dot_jitter, values.size)
                if orient == "v":
                    ax.scatter(position + jitter, values, s=dot_size,
                               color="0.2", alpha=dot_alpha, linewidths=0,
                               zorder=4)
                else:
                    ax.scatter(values, position + jitter, s=dot_size,
                               color="0.2", alpha=dot_alpha, linewidths=0,
                               zorder=4)

    _finish(ax, levels=levels, names=names, xlabel=xlabel, ylabel=ylabel,
            title=title, fontsize=fontsize, orient=orient, rotation=rotation)
    _hue_legend(ax, hues, colors, legend_title or names.get("hue"), fontsize,
                legend and not by_category)

    results = None
    if test and len(hues) == 1:
        results = _maybe_annotate(ax, frame, levels, test, correction,
                                  annot_style, hide_ns, orient, fontsize)
    elif test:
        print("barplot: `test=` is ignored when `hue` is set — call "
              "ov.pl.compare_groups() per hue level instead.")

    if return_stats:
        return ax, {"summary": pd.DataFrame(summary), "test": results}
    return ax


@register_function(
    aliases=["stripplot", "散点分布图", "抖动散点", "jitter_plot", "逐点分布图"],
    category="pl",
    description=(
        "Strip plot: every observation as a jittered point per category, "
        "optionally over a summary line, with significance brackets"
    ),
    examples=[
        "ax = ov.pl.stripplot(df, 'cell_type', 'score')",
        "# Show the median as a crossbar and test the groups",
        "ax = ov.pl.stripplot(df, 'arm', 'value', summary='median',",
        "                     test='mannwhitney')",
        "# Split by a second factor",
        "ax = ov.pl.stripplot(df, 'cell_type', 'score', hue='batch')",
    ],
    related=["pl.barplot", "pl.violinplot", "pl.compare_groups"],
)
def stripplot(data: Any = None,
              x: Any = None,
              y: Any = None,
              *,
              hue: Any = None,
              order: Optional[Sequence[Any]] = None,
              hue_order: Optional[Sequence[Any]] = None,
              jitter: float = 0.18,
              size: float = 14,
              alpha: float = 0.75,
              summary: Optional[str] = "mean",
              summary_width: float = 0.5,
              orient: str = "v",
              width: float = 0.8,
              palette=None,
              ax=None,
              figsize: Optional[Tuple[float, float]] = None,
              test: Optional[str] = None,
              correction: Optional[str] = "holm",
              annot_style: str = "star",
              hide_ns: bool = False,
              xlabel: Optional[str] = None,
              ylabel: Optional[str] = None,
              title: Optional[str] = None,
              legend: bool = True,
              legend_title: Optional[str] = None,
              rotation: float = 0,
              fontsize: Optional[float] = None,
              random_state: int = 0,
              return_stats: bool = False):
    r"""Every observation as a jittered point.

    Arguments
    ---------
    jitter
        Half-width of the horizontal scatter, in category units.
    summary
        Draw a crossbar at the ``'mean'`` (default), ``'median'``, or ``None``.
    test, correction, annot_style, hide_ns
        As in :func:`barplot`.

    Returns
    -------
    ``Axes``, or ``(Axes, dict)`` when ``return_stats=True``.
    """
    frame, levels, hues, names, dropped = _long_frame(data, x, y, hue, order,
                                                      hue_order)
    if dropped:
        print(f"stripplot: dropped {dropped} rows with missing values.")

    ax = _new_axes(ax, figsize, (max(3.2, 0.7 * len(levels) + 1.2), 3.4))
    colors, by_category = _series_colors(levels, hues, palette)
    offsets = _offsets(len(hues), width)
    rng = np.random.default_rng(random_state)

    for h_index, hue_level in enumerate(hues):
        for x_index, level in enumerate(levels):
            mask = frame["x"].to_numpy() == level
            if hue_level is not None:
                mask &= frame["hue"].to_numpy() == hue_level
            values = frame.loc[mask, "y"].to_numpy(dtype=float)
            if values.size == 0:
                continue
            position = x_index + offsets[h_index]
            noise = rng.uniform(-jitter, jitter, values.size)
            if orient == "v":
                ax.scatter(position + noise, values, s=size, alpha=alpha,
                           color=colors[x_index if by_category else h_index],
                           linewidths=0, zorder=2)
            else:
                ax.scatter(values, position + noise, s=size, alpha=alpha,
                           color=colors[x_index if by_category else h_index],
                           linewidths=0, zorder=2)
            if summary:
                if summary not in {"mean", "median"}:
                    raise ValueError(
                        f"`summary` must be 'mean', 'median' or None, got "
                        f"{summary!r}."
                    )
                centre = float(np.mean(values) if summary == "mean"
                               else np.median(values))
                half = summary_width / 2 / max(len(hues), 1)
                if orient == "v":
                    ax.plot([position - half, position + half], [centre] * 2,
                            color="0.1", linewidth=1.6, zorder=3,
                            solid_capstyle="butt")
                else:
                    ax.plot([centre] * 2, [position - half, position + half],
                            color="0.1", linewidth=1.6, zorder=3,
                            solid_capstyle="butt")

    _finish(ax, levels=levels, names=names, xlabel=xlabel, ylabel=ylabel,
            title=title, fontsize=fontsize, orient=orient, rotation=rotation)
    _hue_legend(ax, hues, colors, legend_title or names.get("hue"), fontsize,
                legend and not by_category)

    results = None
    if test and len(hues) == 1:
        results = _maybe_annotate(ax, frame, levels, test, correction,
                                  annot_style, hide_ns, orient, fontsize)
    return (ax, {"test": results}) if return_stats else ax





def _violin_kde(data: Any = None,
               x: Any = None,
               y: Any = None,
               *,
               hue: Any = None,
               order: Optional[Sequence[Any]] = None,
               hue_order: Optional[Sequence[Any]] = None,
               split: bool = False,
               inner: Optional[str] = "box",
               cut: float = 2.0,
               clip: Optional[Tuple[Optional[float], Optional[float]]] = None,
               bw_method: Any = None,
               gridsize: int = 200,
               scale: str = "width",
               stripplot: bool = False,
               strip_size: float = 2.0,
               strip_alpha: float = 0.5,
               orient: str = "v",
               width: float = 0.8,
               palette=None,
               alpha: float = 0.9,
               linewidth: float = 1.1,
               edgecolor: str = "black",
               ax=None,
               figsize: Optional[Tuple[float, float]] = None,
               test: Optional[str] = None,
               correction: Optional[str] = "holm",
               annot_style: str = "star",
               hide_ns: bool = False,
               xlabel: Optional[str] = None,
               ylabel: Optional[str] = None,
               title: Optional[str] = None,
               legend: bool = True,
               legend_title: Optional[str] = None,
               rotation: float = 0,
               fontsize: Optional[float] = None,
               random_state: int = 0,
               return_stats: bool = False):
    r"""Kernel-density violins per category.

    Arguments
    ---------
    split
        With exactly two hue levels, mirror them in one violin instead of
        dodging — half the width for twice the comparison.
    inner
        ``'box'`` (quartile box + median dot, the default), ``'point'`` (raw
        observations), ``'stick'`` (a tick per observation), or ``None``.
    cut
        How far past the extreme observations the density is drawn, in
        **kernel bandwidths** — the same meaning as in seaborn and R.
        ``cut=0`` stops at the data range, which is what a bounded quantity
        like a proportion needs.
    clip
        Hard ``(low, high)`` limits on the density support, e.g. ``(0, None)``
        for a non-negative quantity. Either end may be ``None``.
    scale
        ``'width'`` (equal maximum width, the default and what
        :func:`~omicverse.pl.violin` uses), ``'area'`` (equal area) or
        ``'count'`` (width proportional to n).
    stripplot
        Overlay the raw observations. Off by default here because this
        function is routinely handed a hundred thousand rows; turn it on for
        the small-n plots where a violin alone hides the sample size.
    edgecolor, linewidth
        Outline of each violin. A visible outline is most of what separates a
        readable violin from a coloured smudge.

    Returns
    -------
    ``Axes``, or ``(Axes, dict)``.
    """
    frame, levels, hues, names, dropped = _long_frame(data, x, y, hue, order,
                                                      hue_order)
    if dropped:
        print(f"violinplot: dropped {dropped} rows with missing values.")
    if split and len(hues) != 2:
        raise ValueError(
            f"`split=True` needs exactly two hue levels, got {len(hues)}."
        )
    if scale not in {"area", "width", "count"}:
        raise ValueError("`scale` must be 'area', 'width' or 'count'.")

    ax = _new_axes(ax, figsize, (max(3.2, 0.8 * len(levels) + 1.2), 3.6))
    colors, by_category = _series_colors(levels, hues, palette)
    offsets = np.zeros(len(hues)) if split else _offsets(len(hues), width)
    slot = width if split else width / max(len(hues), 1)
    rng = np.random.default_rng(random_state)

    profiles = []
    for h_index, hue_level in enumerate(hues):
        for x_index, level in enumerate(levels):
            mask = frame["x"].to_numpy() == level
            if hue_level is not None:
                mask &= frame["hue"].to_numpy() == hue_level
            values = frame.loc[mask, "y"].to_numpy(dtype=float)
            if values.size == 0:
                continue
            grid, density = kde_curve(values, cut=cut, gridsize=gridsize,
                                      bw_method=bw_method, clip=clip)
            profiles.append({"x": level, "hue": hue_level, "n": values.size,
                             "grid": grid, "density": density,
                             "position": x_index + offsets[h_index],
                             "color": colors[x_index if by_category else h_index],
                             "values": values,
                             "half": h_index})

    peak = max((p["density"].max() for p in profiles), default=1.0) or 1.0
    counts = max((p["n"] for p in profiles), default=1)
    for profile in profiles:
        density = profile["density"]
        if scale == "area":
            scaled = density / peak
        elif scale == "width":
            local = density.max() or 1.0
            scaled = density / local
        else:
            scaled = density / peak * (profile["n"] / counts)
        half_width = scaled * slot / 2

        position = profile["position"]
        grid = profile["grid"]
        if split:
            sign = -1.0 if profile["half"] == 0 else 1.0
            edge_lo = position if sign > 0 else position - half_width
            edge_hi = position + half_width if sign > 0 else position
        else:
            edge_lo, edge_hi = position - half_width, position + half_width

        if orient == "v":
            ax.fill_betweenx(grid, edge_lo, edge_hi, color=profile["color"],
                             alpha=alpha, linewidth=linewidth,
                             edgecolor=edgecolor, zorder=2)
        else:
            ax.fill_between(grid, edge_lo, edge_hi, color=profile["color"],
                            alpha=alpha, linewidth=linewidth,
                            edgecolor=edgecolor, zorder=2)

        values = profile["values"]
        if stripplot:
            noise = rng.uniform(-slot * 0.10, slot * 0.10, values.size)
            if orient == "v":
                ax.scatter(position + noise, values, s=strip_size,
                           color="0.15", alpha=strip_alpha, linewidths=0,
                           zorder=2.5)
            else:
                ax.scatter(values, position + noise, s=strip_size,
                           color="0.15", alpha=strip_alpha, linewidths=0,
                           zorder=2.5)
        centre = position
        if split:
            centre = position + (-slot / 6 if profile["half"] == 0 else slot / 6)
        if inner == "box":
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            iqr = q3 - q1
            lo = max(values.min(), q1 - 1.5 * iqr)
            hi = min(values.max(), q3 + 1.5 * iqr)
            if orient == "v":
                ax.plot([centre] * 2, [lo, hi], color="0.15", linewidth=0.9,
                        zorder=3)
                ax.plot([centre] * 2, [q1, q3], color="0.15", linewidth=3.5,
                        solid_capstyle="butt", zorder=3)
                ax.scatter([centre], [med], s=9, color="white", zorder=4,
                           linewidths=0)
            else:
                ax.plot([lo, hi], [centre] * 2, color="0.15", linewidth=0.9,
                        zorder=3)
                ax.plot([q1, q3], [centre] * 2, color="0.15", linewidth=3.5,
                        solid_capstyle="butt", zorder=3)
                ax.scatter([med], [centre], s=9, color="white", zorder=4,
                           linewidths=0)
        elif inner == "point":
            noise = rng.uniform(-slot * 0.12, slot * 0.12, values.size)
            if orient == "v":
                ax.scatter(centre + noise, values, s=5, color="0.15",
                           alpha=0.6, linewidths=0, zorder=3)
            else:
                ax.scatter(values, centre + noise, s=5, color="0.15",
                           alpha=0.6, linewidths=0, zorder=3)
        elif inner == "stick":
            tick = slot * 0.18
            for value in values:
                if orient == "v":
                    ax.plot([centre - tick, centre + tick], [value] * 2,
                            color="0.15", linewidth=0.5, alpha=0.6, zorder=3)
                else:
                    ax.plot([value] * 2, [centre - tick, centre + tick],
                            color="0.15", linewidth=0.5, alpha=0.6, zorder=3)
        elif inner is not None:
            raise ValueError(
                f"`inner` must be 'box', 'point', 'stick' or None, got {inner!r}."
            )

    _finish(ax, levels=levels, names=names, xlabel=xlabel, ylabel=ylabel,
            title=title, fontsize=fontsize, orient=orient, rotation=rotation)
    _hue_legend(ax, hues, colors, legend_title or names.get("hue"), fontsize,
                legend and not by_category)

    results = None
    if test and len(hues) == 1:
        results = _maybe_annotate(ax, frame, levels, test, correction,
                                  annot_style, hide_ns, orient, fontsize)
    return (ax, {"test": results}) if return_stats else ax


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------


@register_function(
    aliases=["stackplot", "堆叠柱状图", "stacked_bar", "表格组成图",
             "非adata堆叠图"],
    category="pl",
    description=(
        "Stacked bars of composition per category — counts or proportions — "
        "the container-free version of ov.pl.cellproportion"
    ),
    examples=[
        "# Composition of each sample, as fractions",
        "ax = ov.pl.stackplot(df, 'sample', hue='cell_type')",
        "# Raw counts, horizontal, ordered by size",
        "ax = ov.pl.stackplot(df, 'sample', hue='cell_type', normalize=False,",
        "                     orient='h')",
        "# Pre-aggregated values instead of raw rows",
        "ax = ov.pl.stackplot(counts, 'sample', 'n', hue='cell_type')",
    ],
    related=["pl.cellproportion", "pl.cellstackarea", "pl.barplot", "pl.pieplot"],
)
def stackplot(data: Any = None,
              x: Any = None,
              y: Any = None,
              *,
              hue: Any = None,
              order: Optional[Sequence[Any]] = None,
              hue_order: Optional[Sequence[Any]] = None,
              normalize: bool = True,
              orient: str = "v",
              width: float = 0.8,
              palette=None,
              edgecolor: str = "white",
              linewidth: float = 0.6,
              ax=None,
              figsize: Optional[Tuple[float, float]] = None,
              xlabel: Optional[str] = None,
              ylabel: Optional[str] = None,
              title: Optional[str] = None,
              legend: bool = True,
              legend_title: Optional[str] = None,
              legend_out: bool = True,
              rotation: float = 45,
              fontsize: Optional[float] = None,
              return_stats: bool = False):
    r"""Stacked composition bars.

    Arguments
    ---------
    x
        The category whose composition is shown — one bar each.
    hue
        The parts that make up each bar. Required.
    y
        Optional pre-aggregated size column. Left out, rows are counted, which
        is the usual case for a per-cell table.
    normalize
        Scale each bar to 1 (default) so composition is comparable across
        categories of different size. ``False`` keeps raw counts.

    Returns
    -------
    ``Axes``, or ``(Axes, DataFrame)`` with the plotted matrix.
    """
    if hue is None:
        raise TypeError("`hue` is required — it names the parts of each bar.")
    frame, levels, hues, names, dropped = _long_frame(data, x, y, hue, order,
                                                      hue_order,
                                                      require_y=False)
    if dropped:
        print(f"stackplot: dropped {dropped} rows with missing values.")

    if "y" in frame:
        matrix = frame.pivot_table(index="x", columns="hue", values="y",
                                   aggfunc="sum", observed=False)
    else:
        matrix = (frame.groupby(["x", "hue"], observed=False).size()
                  .unstack(fill_value=0))
    matrix = matrix.reindex(index=levels, columns=hues).fillna(0.0)
    if normalize:
        totals = matrix.sum(axis=1).replace(0, np.nan)
        matrix = matrix.div(totals, axis=0).fillna(0.0)

    ax = _new_axes(ax, figsize, (max(3.4, 0.55 * len(levels) + 2.0), 3.6))
    colors = default_palette(len(hues), palette)
    base = np.zeros(len(levels))
    positions = np.arange(len(levels))
    for hue_level, colour in zip(hues, colors):
        values = matrix[hue_level].to_numpy(dtype=float)
        if orient == "v":
            ax.bar(positions, values, bottom=base, width=width, color=colour,
                   edgecolor=edgecolor, linewidth=linewidth, label=str(hue_level))
        else:
            ax.barh(positions, values, left=base, height=width, color=colour,
                    edgecolor=edgecolor, linewidth=linewidth,
                    label=str(hue_level))
        base = base + values

    value_label = ylabel
    if value_label is None:
        value_label = "fraction" if normalize else (names.get("y") or "count")
    _finish(ax, levels=levels, names=names, xlabel=xlabel, ylabel=value_label,
            title=title, fontsize=fontsize, orient=orient, rotation=rotation)
    if orient == "v":
        ax.set_xlim(-0.5 - (1 - width) / 2, len(levels) - 0.5 + (1 - width) / 2)
        if normalize:
            ax.set_ylim(0, 1)
    else:
        ax.set_ylim(-0.5, len(levels) - 0.5)
        if normalize:
            ax.set_xlim(0, 1)

    if legend:
        if legend_out:
            from ._layout import take_legend_out

            take_legend_out(ax, loc="right", title=legend_title or names.get("hue"),
                            fontsize=fontsize)
        else:
            ax.legend(title=legend_title or names.get("hue"), frameon=False,
                      fontsize=font_size(fontsize))
    return (ax, matrix) if return_stats else ax


@register_function(
    aliases=["棒棒糖图", "lollipopplot", "lollipop", "棒棒糖", "dot_bar"],
    category="pl",
    description=(
        "Lollipop chart — a stem and a dot per label, the readable "
        "alternative to a long bar chart of ranked values"
    ),
    examples=[
        "ax = ov.pl.lollipopplot(auc_df, 'cell_type', 'AUC')",
        "# Ranked, with a reference line at the null",
        "ax = ov.pl.lollipopplot(res, 'pathway', 'NES', sort='value',",
        "                        reference=0)",
        "# Colour the dots by significance",
        "ax = ov.pl.lollipopplot(res, 'gene', 'log2FC', color_by=res['padj'],",
        "                        cmap='viridis_r')",
    ],
    related=["pl.barplot", "pl.forest", "pl.dotplot"],
)
def lollipopplot(data: Any = None,
                 x: Any = None,
                 y: Any = None,
                 *,
                 sort: Optional[str] = "value",
                 ascending: bool = True,
                 top_n: Optional[int] = None,
                 orient: str = "h",
                 color: str = "#1F577B",
                 color_by: Any = None,
                 cmap: str = "viridis",
                 size: float = 60,
                 stem_width: float = 1.2,
                 reference: Optional[float] = None,
                 baseline: Optional[float] = None,
                 ax=None,
                 figsize: Optional[Tuple[float, float]] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 title: Optional[str] = None,
                 colorbar_label: Optional[str] = None,
                 fontsize: Optional[float] = None,
                 return_stats: bool = False):
    r"""Stem-and-dot chart of one value per label.

    Arguments
    ---------
    sort
        ``'value'`` (default), ``'label'`` or ``None`` to keep input order.
    top_n
        Keep only the ``top_n`` largest by absolute value — and say so, rather
        than silently truncating.
    orient
        ``'h'`` (default) puts labels on the y-axis, which is what makes this
        readable for long names.
    reference
        Draw a dashed line at this value (0 for a log fold change, 1 for a
        ratio).
    baseline
        Where the stems start. Defaults to ``reference``, else 0.

    Returns
    -------
    ``Axes``, or ``(Axes, DataFrame)``.
    """
    frame, levels, _, names, dropped = _long_frame(data, x, y, None, None, None)
    if dropped:
        print(f"lollipopplot: dropped {dropped} rows with missing values.")
    table = frame[["x", "y"]].copy()
    table["y"] = table["y"].astype(float)
    if color_by is not None:
        table["c"] = np.asarray(color_by, dtype=float)[: len(table)]

    if top_n is not None and top_n < len(table):
        keep = table["y"].abs().nlargest(top_n).index
        print(f"lollipopplot: showing the {top_n} largest of {len(table)} "
              f"entries by |value|.")
        table = table.loc[keep]
    if sort == "value":
        table = table.sort_values("y", ascending=ascending)
    elif sort == "label":
        table = table.sort_values("x", ascending=ascending)
    elif sort is not None:
        raise ValueError("`sort` must be 'value', 'label' or None.")
    table = table.reset_index(drop=True)

    start = baseline if baseline is not None else (reference or 0.0)
    positions = np.arange(len(table))
    ax = _new_axes(ax, figsize,
                   (5.0, max(2.4, 0.28 * len(table) + 1.0)) if orient == "h"
                   else (max(3.4, 0.32 * len(table) + 1.4), 3.6))

    scatter_kw: Dict[str, Any] = {"s": size, "zorder": 3, "linewidths": 0}
    if "c" in table:
        scatter_kw.update(c=table["c"].to_numpy(), cmap=cmap)
    else:
        scatter_kw.update(color=color)

    if orient == "h":
        for position, value in zip(positions, table["y"]):
            ax.plot([start, value], [position] * 2, color=color, alpha=0.55,
                    linewidth=stem_width, zorder=2)
        points = ax.scatter(table["y"], positions, **scatter_kw)
        if reference is not None:
            ax.axvline(reference, color="0.55", linestyle="--", linewidth=0.9)
    else:
        for position, value in zip(positions, table["y"]):
            ax.plot([position] * 2, [start, value], color=color, alpha=0.55,
                    linewidth=stem_width, zorder=2)
        points = ax.scatter(positions, table["y"], **scatter_kw)
        if reference is not None:
            ax.axhline(reference, color="0.55", linestyle="--", linewidth=0.9)

    _finish(ax, levels=list(table["x"]), names=names, xlabel=xlabel,
            ylabel=ylabel, title=title, fontsize=fontsize, orient=orient,
            rotation=0 if orient == "h" else 45)
    if "c" in table:
        bar = ax.figure.colorbar(points, ax=ax, fraction=0.035, pad=0.03)
        bar.set_label(colorbar_label or "", fontsize=font_size(fontsize))
        bar.ax.tick_params(labelsize=font_size(fontsize) - 1)
    ax.spines["left" if orient == "h" else "bottom"].set_visible(orient != "h")
    table = table.rename(columns={"x": "label", "y": "value", "c": "color_by"})
    return (ax, table) if return_stats else ax


def _pie(values, labels, colors, *, hole, ax, start_angle, percent, counts,
         label_style, fontsize, explode, edgecolor, linewidth, pct_distance,
         label_distance):
    total = float(np.sum(values))
    texts = []
    for label, value in zip(labels, values):
        share = value / total if total else 0.0
        parts = [str(label)]
        if label_style in {"percent", "both"}:
            parts.append(f"{100 * share:.1f}%")
        if label_style in {"count", "both"}:
            parts.append(f"n={value:g}")
        texts.append("\n".join(parts) if label_style != "none" else "")

    wedges, _ = ax.pie(
        values, labels=texts if label_style != "none" else None,
        colors=colors, startangle=start_angle,
        explode=explode, labeldistance=label_distance,
        wedgeprops=dict(width=1.0 - hole, edgecolor=edgecolor,
                        linewidth=linewidth),
        textprops=dict(fontsize=font_size(fontsize)),
    )
    if percent:
        for wedge, value in zip(wedges, values):
            angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
            radius = 1.0 - (1.0 - hole) * (1 - pct_distance)
            ax.text(radius * np.cos(angle), radius * np.sin(angle),
                    f"{100 * value / total:.0f}%", ha="center", va="center",
                    fontsize=font_size(fontsize, "tick", -1), color="white", weight="bold")
    return wedges


@register_function(
    aliases=["饼图", "pieplot", "pie_chart", "圆饼图"],
    category="pl",
    description=(
        "Pie chart of a composition, from raw rows or pre-aggregated counts, "
        "with percentage labels"
    ),
    examples=[
        "# Count the rows of each category",
        "ax = ov.pl.pieplot(df, 'cell_type')",
        "# Pre-aggregated",
        "ax = ov.pl.pieplot(counts, 'cell_type', 'n')",
        "# Bare values",
        "ax = ov.pl.pieplot(labels=['T', 'B', 'NK'], values=[50, 30, 20])",
    ],
    related=["pl.donutplot", "pl.stackplot", "pl.barplot"],
)
def pieplot(data: Any = None,
            labels: Any = None,
            values: Any = None,
            *,
            order: Optional[Sequence[Any]] = None,
            hole: float = 0.0,
            other_threshold: float = 0.0,
            other_label: str = "other",
            legend: bool = False,
            legend_loc: str = "center left",
            palette=None,
            explode: Optional[Sequence[float]] = None,
            start_angle: float = 90,
            label_style: str = "both",
            percent_inside: bool = False,
            pct_distance: float = 0.6,
            label_distance: float = 1.08,
            edgecolor: str = "white",
            linewidth: float = 1.0,
            centre_text: Optional[str] = None,
            ax=None,
            figsize: Optional[Tuple[float, float]] = None,
            title: Optional[str] = None,
            fontsize: Optional[float] = None,
            return_stats: bool = False):
    r"""Pie chart of a composition.

    Arguments
    ---------
    data, labels, values
        A table plus a category column (rows are counted) and optionally a
        size column; or bare ``labels``/``values`` arrays.
    hole
        Fraction of the radius left empty in the middle — ``0`` is a pie,
        ``0.55`` is a donut (see :func:`donutplot`).
    other_threshold
        Merge every category below this share into one wedge. A pie with ten
        categories, three of them under 2%, is unreadable however carefully
        the labels are placed; ``other_threshold=0.03`` is the usual fix.
    legend
        Put the names in a legend instead of around the wedges. With more than
        about six categories this is the only layout that stays legible.
    label_style
        ``'both'`` (name + percent, default), ``'percent'``, ``'count'``,
        ``'name'`` or ``'none'``.
    percent_inside
        Also write the percentage inside each wedge.

    Returns
    -------
    ``Axes``, or ``(Axes, Series)`` with the plotted shares.
    """
    if label_style not in {"both", "percent", "count", "name", "none"}:
        raise ValueError(
            "`label_style` must be 'both', 'percent', 'count', 'name' or 'none'."
        )
    frame, levels, _, names, _ = _long_frame(data, labels, values, None, order,
                                             None, require_y=False)
    if "y" in frame:
        series = frame.groupby("x", observed=False)["y"].sum()
    else:
        series = frame.groupby("x", observed=False).size()
    series = series.reindex(levels).fillna(0.0)
    series = series[series > 0]
    if series.empty:
        raise ValueError("Nothing to plot — every category is zero or absent.")

    if other_threshold > 0:
        shares = series / series.sum()
        small = shares[shares < other_threshold].index
        if len(small) > 1:
            merged = float(series[small].sum())
            series = series.drop(index=small)
            series[other_label] = merged
            print(f"pieplot: merged {len(small)} categories below "
                  f"{other_threshold:.0%} into {other_label!r}: "
                  f"{', '.join(map(str, small))}")

    ax = _new_axes(ax, figsize, (4.2, 4.2))
    colors = default_palette(len(series), palette)
    wedges = _pie(series.to_numpy(dtype=float), list(series.index), colors,
                  hole=hole, ax=ax, start_angle=start_angle,
                  percent=percent_inside, counts=True,
                  label_style="none" if legend else label_style,
                  fontsize=font_size(fontsize), explode=explode, edgecolor=edgecolor,
                  linewidth=linewidth, pct_distance=pct_distance,
                  label_distance=label_distance)
    if legend:
        total = float(series.sum())
        ax.legend(wedges,
                  [f"{name}  ({100 * value / total:.1f}%)"
                   for name, value in series.items()],
                  loc=legend_loc, bbox_to_anchor=(1.0, 0.5), frameon=False,
                  fontsize=font_size(fontsize))
    if centre_text:
        ax.text(0, 0, centre_text, ha="center", va="center",
                fontsize=font_size(fontsize, "title"))
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=font_size(fontsize, "title"))
    shares = series / series.sum()
    return (ax, shares) if return_stats else ax


@register_function(
    aliases=["环形图", "donutplot", "圆环图", "donut_chart", "甜甜圈图"],
    category="pl",
    description=(
        "Donut chart — a pie with the middle cut out, leaving room for the "
        "total or another summary in the centre"
    ),
    examples=[
        "ax = ov.pl.donutplot(df, 'cell_type')",
        "ax = ov.pl.donutplot(counts, 'cell_type', 'n', centre_text='12,481\\ncells')",
    ],
    related=["pl.pieplot", "pl.stackplot"],
)
def donutplot(data: Any = None,
              labels: Any = None,
              values: Any = None,
              *,
              hole: float = 0.55,
              centre_text: Optional[str] = None,
              **kwargs: Any):
    r"""A pie chart with a hole. See :func:`pieplot` for every argument.

    The centre is the point: put the total there, so the reader gets the
    absolute size as well as the composition.
    """
    if centre_text is None and "return_stats" not in kwargs:
        frame = as_frame(data)
        if frame is not None and values is None and isinstance(labels, str):
            centre_text = f"n = {len(frame):,}"
    return pieplot(data, labels, values, hole=hole, centre_text=centre_text,
                   **kwargs)


@register_function(
    aliases=["斜率图", "slopeplot", "slope_chart", "配对变化图", "before_after"],
    category="pl",
    description=(
        "Slope chart: one line per subject between two or more conditions, "
        "with a paired test — the honest way to show a before/after effect"
    ),
    examples=[
        "ax = ov.pl.slopeplot(df, 'timepoint', 'value', subject='patient')",
        "# Colour by direction and run a Wilcoxon signed-rank test",
        "ax = ov.pl.slopeplot(df, 'timepoint', 'value', subject='patient',",
        "                     color_by='direction', test='wilcoxon')",
        "# 'auto' is the paired default — Wilcoxon signed-rank; res['test'] names it",
        "ax, res = ov.pl.slopeplot(df, 'timepoint', 'value', subject='patient',",
        "                          test='auto', return_stats=True)",
    ],
    related=["pl.violinplot", "pl.barplot", "pl.compare_groups"],
)
def slopeplot(data: Any = None,
              x: Any = None,
              y: Any = None,
              *,
              subject: Any = None,
              order: Optional[Sequence[Any]] = None,
              color_by: str = "direction",
              palette=None,
              up_color: str = "#CB3E35",
              down_color: str = "#1F577B",
              flat_color: str = "0.6",
              linewidth: float = 0.9,
              alpha: float = 0.6,
              marker: str = "o",
              markersize: float = 16,
              summary: Optional[str] = "mean",
              label_points: bool = False,
              test: Optional[str] = None,
              ax=None,
              figsize: Optional[Tuple[float, float]] = None,
              xlabel: Optional[str] = None,
              ylabel: Optional[str] = None,
              title: Optional[str] = None,
              fontsize: Optional[float] = None,
              return_stats: bool = False):
    r"""One line per subject across conditions.

    Arguments
    ---------
    subject
        Column identifying the repeated unit (patient, donor, well). Required —
        without it there is nothing to connect.
    color_by
        ``'direction'`` (default: up / down / flat), ``'subject'``, or the name
        of a column to colour by.
    summary
        Overlay the group ``'mean'`` or ``'median'`` as a heavy line.
    test
        ``'wilcoxon'`` or ``'ttest_rel'`` for a paired comparison of the first
        and last condition. ``'auto'`` is accepted for consistency with the
        other plots and means ``'wilcoxon'`` here — the paired counterpart of
        the Mann-Whitney U that :func:`~omicverse.pl.compare_groups` resolves
        ``'auto'`` to; like there, it is a fixed default and nothing about the
        data is inspected. The test that actually ran is printed above the axes
        and returned as ``stats['test']``. Subjects missing either end are
        excluded, and the number excluded is reported.

    Returns
    -------
    ``Axes``, or ``(Axes, dict)``.
    """
    if subject is None:
        raise TypeError("`subject` is required — it says which points connect.")
    frame, names = resolve_columns(data, x=x, y=y, subject=subject,
                                   require=("x", "y", "subject"))
    levels = group_levels(frame["x"], order)
    positions = {level: index for index, level in enumerate(levels)}

    wide = frame.pivot_table(index="subject", columns="x", values="y",
                             aggfunc="mean", observed=False)
    wide = wide.reindex(columns=levels)

    ax = _new_axes(ax, figsize, (max(2.8, 1.3 * len(levels)), 4.0))
    first, last = levels[0], levels[-1]
    complete = wide[[first, last]].dropna()
    if color_by not in {"direction", "subject"} and color_by not in frame:
        raise ValueError(
            f"`color_by` must be 'direction', 'subject' or a column; got "
            f"{color_by!r}."
        )
    subject_colors = (dict(zip(wide.index, default_palette(len(wide), palette)))
                      if color_by == "subject" else {})

    for name, row in wide.iterrows():
        values = row.to_numpy(dtype=float)
        valid = ~np.isnan(values)
        if valid.sum() < 2:
            continue
        if color_by == "subject":
            colour = subject_colors[name]
        else:
            ends = values[valid]
            delta = ends[-1] - ends[0]
            colour = (up_color if delta > 0 else
                      down_color if delta < 0 else flat_color)
        xs = [positions[level] for level, keep in zip(levels, valid) if keep]
        ax.plot(xs, values[valid], color=colour, linewidth=linewidth,
                alpha=alpha, marker=marker,
                markersize=np.sqrt(markersize), zorder=2)
        if label_points:
            ax.annotate(str(name), (xs[-1], values[valid][-1]),
                        textcoords="offset points", xytext=(4, 0),
                        fontsize=font_size(fontsize, "tick", -2), va="center")

    if summary:
        if summary not in {"mean", "median"}:
            raise ValueError("`summary` must be 'mean', 'median' or None.")
        centre = (wide.mean() if summary == "mean" else wide.median())
        ax.plot(range(len(levels)), centre.to_numpy(dtype=float), color="black",
                linewidth=2.4, marker="o", markersize=5, zorder=4,
                label=summary)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(level) for level in levels], fontsize=font_size(fontsize))
    ax.set_xlim(-0.35, len(levels) - 0.65)
    ax.set_xlabel(xlabel if xlabel is not None else names.get("x", ""),
                  fontsize=font_size(fontsize, "label"))
    ax.set_ylabel(ylabel if ylabel is not None else names.get("y", ""),
                  fontsize=font_size(fontsize, "label"))
    if title:
        ax.set_title(title, fontsize=font_size(fontsize, "title"))
    ax.tick_params(labelsize=font_size(fontsize))
    style_axes(ax)

    stats: Dict[str, Any] = {"wide": wide, "n_complete": int(len(complete))}
    if test:
        dropped = len(wide) - len(complete)
        if dropped:
            print(f"slopeplot: {dropped} subject(s) lack {first} or {last} and "
                  f"are excluded from the paired test.")
        from ._stats_tests import _resolve_test, _run_test, format_pvalue

        # `test` reaches _run_test raw, and _run_test knows concrete names
        # only — so 'auto' has to be translated here, exactly as
        # compare_groups does it for the unpaired plots. This plot is paired
        # by construction (`subject` is required), hence paired=True: 'auto'
        # is Wilcoxon signed-rank, not Mann-Whitney U. `label` below is
        # derived from the resolved name, so the figure and the returned
        # dict always say which test ran.
        statistic, pvalue, label = _run_test(
            complete[first].to_numpy(dtype=float),
            complete[last].to_numpy(dtype=float),
            _resolve_test(test, paired=True))
        stats.update(statistic=statistic, pvalue=pvalue, test=label)
        ax.text(0.5, 1.01, f"{label}: {format_pvalue(pvalue, 'value')} "
                           f"(n={len(complete)})",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=font_size(fontsize))
    return (ax, stats) if return_stats else ax


# --------------------------------------------------------------------------
# flow / alluvial
# --------------------------------------------------------------------------


def _smoothstep(n: int) -> np.ndarray:
    """A cubic S-curve on ``[0, 1]`` sampled at ``n`` points.

    ``3t^2 - 2t^3`` is flat at both ends, so ribbons leave a node horizontally
    and arrive horizontally — the shape that reads as a smooth flow rather than
    a slanted parallelogram. pySankey/cnsplots reach the same curve by
    convolving a step twice with a box; a closed-form smoothstep is the same
    idea without the resampling, and it lets us place the polygon vertices
    exactly, which is what keeps mass conservation checkable from the artists.
    """
    t = np.linspace(0.0, 1.0, n)
    return t * t * (3.0 - 2.0 * t)


def _stage_layout(counts: pd.Series, levels: list, gap: float):
    """Top and height of every node in one stage, stacked top-to-bottom.

    Heights are counts normalised so the bars plus the ``gap``s between them
    fill ``[0, 1]``. Stacking downward from ``y = 1`` puts the first level of
    ``levels`` at the top, so the row order the caller chose is the order a
    reader sees.
    """
    total = float(counts.sum())
    n = len(levels)
    span = 1.0 - gap * (n - 1) if n > 1 else 1.0
    if span <= 0:
        raise ValueError(
            f"`gap={gap}` leaves no room for {n} nodes in a stage; use a "
            f"smaller gap (gap * (n_nodes - 1) must stay below 1)."
        )
    tops, heights = {}, {}
    cursor = 1.0
    for level in levels:
        height = float(counts.get(level, 0.0)) / total * span if total else 0.0
        tops[level] = cursor
        heights[level] = height
        cursor -= height + gap
    return tops, heights


@register_function(
    aliases=["桑基图", "河流图", "sankey", "alluvial", "冲积图"],
    category="pl",
    description=(
        "Sankey / alluvial diagram of the flow between two or more categorical "
        "columns of a table — the general, container-free counterpart of "
        "ov.pl.perturb_sankey"
    ),
    examples=[
        "# Two-stage flow: how each day splits by sex",
        "ax = ov.pl.sankey(df, ['day', 'sex'])",
        "# Three-stage alluvial, coloured from a fixed palette",
        "ax = ov.pl.sankey(df, ['day', 'sex', 'smoker'], palette='tab10')",
        "# Pin the level order of one column",
        "ax = ov.pl.sankey(df, ['day', 'sex'], order={'day': ['Thur', 'Fri', 'Sat', 'Sun']})",
    ],
    related=["pl.perturb_sankey", "pl.stackplot", "pl.slopeplot"],
)
def sankey(data: Any = None,
           columns: Optional[Sequence[Any]] = None,
           *,
           ax=None,
           palette=None,
           gap: float = 0.02,
           node_width: float = 0.03,
           flow_alpha: float = 0.6,
           order: Optional[Any] = None,
           label: bool = True,
           label_fontsize: Optional[float] = None,
           figsize: Tuple[float, float] = (4, 4)):
    r"""Sankey / alluvial diagram of the flow between categorical columns.

    A node is a category within one column (stage); its height is the number
    of rows in that category. A ribbon joins a category in stage ``i`` to a
    category in stage ``i + 1``, and its width is the number of rows that fall
    in both — so the picture accounts for every row, and the widths leaving a
    node always sum back to the node's height (this is the property that makes
    a Sankey a Sankey, and it holds here by construction).

    Ribbon geometry follows cnsplots (Farid Rashidi, BSD-3-Clause) — a
    smoothstep between the two node edges — and is implemented independently.

    Arguments
    ---------
    data
        A ``DataFrame`` (or anything with an ``.obs`` frame, e.g. an
        ``AnnData``). Rows missing any of ``columns`` are dropped and the count
        is reported.
    columns
        Two or more column names, drawn left to right. Three or more columns
        give a multi-stage alluvial, each consecutive pair joined by ribbons.
    palette
        Colours for the nodes of each stage, anything
        :func:`~omicverse.pl.barplot` accepts (an ov palette by default, a
        colormap name, or an explicit list). Ribbons take the colour of their
        **left** node, so a source can be followed downstream.
    gap
        Vertical space between nodes in a stage, in axes units (the whole
        diagram is one unit tall).
    node_width
        Horizontal width of the node bars, in stage-spacing units.
    flow_alpha
        Opacity of the ribbons; the bars are drawn opaque.
    order
        Level order per column. Either a mapping ``{column: [levels...]}`` for
        specific columns, or a single sequence applied to every column. Columns
        left unspecified fall back to categorical order, else a plain sort.
    label
        Draw the category name beside each node.

    Returns
    -------
    The ``Axes`` the diagram was drawn into.
    """
    frame = as_frame(data)
    if frame is None:
        raise TypeError(
            "`data` must be a DataFrame or an object with an `.obs` frame "
            "(e.g. AnnData); got " + type(data).__name__ + "."
        )
    if columns is None or len(list(columns)) < 2:
        raise TypeError(
            "`columns` must name at least two categorical columns to draw a "
            "flow between."
        )
    columns = list(columns)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        available = ", ".join(map(str, frame.columns[:30]))
        raise KeyError(
            f"Column(s) {missing} are not in the table. Available columns: "
            f"{available}" + (" ..." if frame.shape[1] > 30 else "")
        )

    stage = frame[columns].copy()
    before = len(stage)
    stage = stage.dropna()
    dropped = before - len(stage)
    if dropped:
        print(f"sankey: dropped {dropped} rows with missing values.")
    if stage.empty:
        raise ValueError("Nothing to plot — every row is missing a value.")

    # Per-column level order. A dict targets named columns; a bare sequence is
    # applied to all of them (each column keeps only the levels it has).
    def _levels_for(column):
        explicit = None
        if isinstance(order, Mapping):
            explicit = order.get(column)
        elif order is not None:
            present = set(pd.unique(stage[column]))
            explicit = [lv for lv in order if lv in present]
            if not explicit:
                explicit = None
        return group_levels(stage[column], explicit)

    levels = [_levels_for(c) for c in columns]

    ax = _new_axes(ax, figsize, figsize)
    n_stages = len(columns)

    # Node layout and colours, one entry per stage.
    tops, heights, colours = [], [], []
    for i, column in enumerate(columns):
        counts = stage[column].value_counts()
        top, height = _stage_layout(counts, levels[i], gap)
        tops.append(top)
        heights.append(height)
        palette_i = default_palette(len(levels[i]), palette)
        colours.append({lv: palette_i[j] for j, lv in enumerate(levels[i])})

    # Node bars. Rectangles (not fill_between) so their heights are readable
    # straight off the artist in a test, and gid-tagged for the same reason.
    from matplotlib.patches import Polygon, Rectangle

    for i in range(n_stages):
        for level in levels[i]:
            h = heights[i][level]
            if h <= 0:
                continue
            bar = Rectangle((i - node_width / 2, tops[i][level] - h),
                            node_width, h, facecolor=colours[i][level],
                            edgecolor="none", zorder=3)
            bar.set_gid(f"node:{i}:{level}")
            ax.add_patch(bar)

    # Ribbons between each consecutive pair of stages. For a left node, the
    # outgoing widths are laid out top-to-bottom in right-label order; for a
    # right node, the incoming widths are laid out in left-label order — which
    # falls out of iterating left (outer) then right (inner). The left-side
    # width of every ribbon is count / total scaled by the LEFT stage's span,
    # so the widths leaving a node sum exactly to that node's height.
    npts = 60
    s = _smoothstep(npts)
    for i in range(n_stages - 1):
        left_col, right_col = columns[i], columns[i + 1]
        pair = (stage.groupby([left_col, right_col], observed=True)
                .size())
        total = float(len(stage))
        span_l = 1.0 - gap * (len(levels[i]) - 1) if len(levels[i]) > 1 else 1.0
        span_r = (1.0 - gap * (len(levels[i + 1]) - 1)
                  if len(levels[i + 1]) > 1 else 1.0)
        left_cursor = dict(tops[i])
        right_cursor = dict(tops[i + 1])
        x_left = i + node_width / 2
        x_right = (i + 1) - node_width / 2
        xs = x_left + (x_right - x_left) * s
        for left in levels[i]:
            for right in levels[i + 1]:
                count = float(pair.get((left, right), 0.0))
                if count <= 0:
                    continue
                w_left = count / total * span_l
                w_right = count / total * span_r
                l_top = left_cursor[left]
                l_bot = l_top - w_left
                r_top = right_cursor[right]
                r_bot = r_top - w_right
                left_cursor[left] = l_bot
                right_cursor[right] = r_bot

                y_top = l_top + (r_top - l_top) * s
                y_bot = l_bot + (r_bot - l_bot) * s
                verts = np.concatenate([
                    np.column_stack([xs, y_top]),
                    np.column_stack([xs[::-1], y_bot[::-1]]),
                ])
                ribbon = Polygon(verts, closed=True,
                                 facecolor=colours[i][left], edgecolor="none",
                                 alpha=flow_alpha, zorder=2)
                ribbon.set_gid(f"flow:{i}:{left}:{right}")
                ax.add_patch(ribbon)

    if label:
        size = font_size(label_fontsize)
        pad = node_width + 0.02 * max(n_stages - 1, 1)
        for i in range(n_stages):
            for level in levels[i]:
                h = heights[i][level]
                if h <= 0:
                    continue
                centre = tops[i][level] - h / 2
                if i == 0:
                    ax.text(i - node_width / 2 - pad, centre, str(level),
                            ha="right", va="center", fontsize=size)
                elif i == n_stages - 1:
                    ax.text(i + node_width / 2 + pad, centre, str(level),
                            ha="left", va="center", fontsize=size)
                else:
                    ax.text(i, tops[i][level] + 0.005, str(level),
                            ha="center", va="bottom", fontsize=size)

    ax.set_xlim(-0.4, (n_stages - 1) + 0.4)
    ax.set_ylim(-0.02, 1.05)
    ax.axis("off")
    return ax


@register_function(
    aliases=["violinplot", "表格小提琴图", "dataframe小提琴图", "long_form_violin",
             "非adata小提琴图"],
    category="pl",
    description=(
        "Violin plot from a long-form DataFrame or bare arrays. For an "
        "AnnData with genes, layers and .raw use ov.pl.violin instead — this "
        "is the container-free version, with split/clip/inner and tests"
    ),
    examples=[
        "ax = ov.pl.violinplot(df, 'cell_type', 'score')",
        "# Two conditions mirrored in one violin",
        "ax = ov.pl.violinplot(df, 'cell_type', 'score', hue='condition',",
        "                      split=True)",
        "# With a Kruskal-Wallis omnibus and corrected pairwise brackets",
        "ax, res = ov.pl.violinplot(df, 'arm', 'value', test='auto',",
        "                           return_stats=True)",
    ],
    related=["pl.violin", "pl.stripplot", "pl.barplot", "pl.compare_groups"],
)
def violinplot(data: Any = None, x: Any = None, y: Any = None, **kwargs: Any):
    r"""Deprecated: use :func:`omicverse.pl.violin`.

    ``violin`` now covers everything this did — it accepts a DataFrame as well
    as an AnnData, and gained ``hue`` / ``split`` / ``cut`` / ``clip`` /
    ``inner`` / ``orient`` / ``test``. Two names for one plot is what made
    ``ov.pl.violin`` and this function look like they came from different
    libraries.

    The translation is just the argument names: ``x`` is ``groupby`` and ``y``
    is ``keys``. Rendering is unchanged — this always routes through the
    kernel-density engine, so an existing call produces the same figure it did
    before, with a warning.
    """
    from warnings import warn

    warn(
        "`ov.pl.violinplot` is deprecated; use `ov.pl.violin`, which now takes "
        "a DataFrame and supports hue/split/cut/clip/inner/orient/test. "
        "`violinplot(df, x, y)` becomes `violin(df, keys=y, groupby=x)`.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ._violin import violin

    # Keep this function's own defaults rather than inheriting violin's.
    # The two signatures disagree on four of them, and every one changes the
    # picture:
    #   scale      forces the kernel-density engine (violin defaults to the
    #              matplotlib one)
    #   inner      violinplot drew the quartile box; violin defaults to none
    #   stripplot  violin draws the raw points; violinplot did not
    #   fontsize   violin pins 13; violinplot follows ov.plot_set
    for name, value in (("scale", "width"), ("inner", "box"),
                        ("stripplot", False), ("fontsize", None)):
        kwargs.setdefault(name, value)
    return violin(data, keys=y, groupby=x, **kwargs)
