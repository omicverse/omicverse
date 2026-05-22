"""Plotting for single-extracellular-vesicle (single-EV) proteomics.

Only the EV-specific plot lives here:

* :func:`misev_marker_plot` — MISEV2023 positive vs contaminant marker
  levels (the EV-preparation purity view).

Generic single-cell plots — embedding scatters, marker dotplots,
mean-signal heatmaps and stacked-composition bars — are omicverse-native;
use :func:`omicverse.pl.embedding`, :func:`omicverse.pl.dotplot` /
:func:`omicverse.pl.markers_dotplot`, :func:`omicverse.pl.marker_heatmap`
and :func:`omicverse.pl.cellproportion` instead.

The function is ``@register_function``-decorated (``category='ev'``) and
returns the :class:`matplotlib.axes.Axes`.
"""
from __future__ import annotations

import numpy as np

from ..._registry import register_function


def _ax(ax, figsize):
    """Create a fresh Axes when none is supplied."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


@register_function(
    aliases=[
        "misev_marker_plot", "ev_misev_marker_plot", "plot_misev_markers",
        "MISEV标志物图", "EV纯度图",
    ],
    category="ev",
    description=(
        "Bar plot of MISEV2023 positive EV markers versus contaminant / "
        "non-EV markers — the EV-preparation purity view. Positive markers "
        "are drawn in one colour, contaminants in another, sorted by mean "
        "signal."
    ),
    examples=[
        "ov.single.ev.plotting.misev_marker_plot(adata)",
    ],
    related=["single.ev.misev_report"],
)
def misev_marker_plot(
    adata,
    *,
    ax=None,
    figsize=(7, 4),
    title: str = "MISEV2023 marker panel",
):
    """Bar plot of MISEV positive vs contaminant marker levels.

    Parameters
    ----------
    adata
        Single-EV AnnData; classification uses ``var['misev_category']``
        when present, otherwise the built-in canonical marker sets.
    ax, figsize
        Standard matplotlib styling controls.
    title
        Plot title.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    from ._report import misev_report

    ax = _ax(ax, figsize)
    markers = misev_report(adata, as_frame=True)
    shown = markers[markers["misev_class"].isin(["positive", "contaminant"])]
    shown = shown.sort_values(
        ["misev_class", "mean_signal"], ascending=[True, False]
    )
    if shown.empty:
        ax.set_title(title + " (no MISEV markers found)")
        return ax
    colors = {"positive": "#2C7FB8", "contaminant": "#D95F02"}
    bar_colors = [colors[c] for c in shown["misev_class"]]
    x = np.arange(len(shown))
    ax.bar(x, shown["mean_signal"].values, color=bar_colors, width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(shown["protein"], rotation=90, fontsize=8)
    ax.set_ylabel("mean signal")
    ax.set_title(title)
    handles = [
        plt_patch(colors["positive"], "positive EV marker"),
        plt_patch(colors["contaminant"], "contaminant / non-EV"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def plt_patch(color, label):
    """Build a solid-colour legend patch."""
    from matplotlib.patches import Patch

    return Patch(facecolor=color, label=label)
