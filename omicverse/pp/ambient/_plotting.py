"""Plotting helpers for ambient-RNA decontamination diagnostics.

A single light helper, :func:`plot_contamination`, draws the per-cell
contamination-fraction distribution.  For embedding overlays or grouped
violins, prefer the general omicverse plotting surface
(``ov.pl.embedding(adata, color='ambient_contamination')`` or
``ov.pl.violin``) — this module deliberately stays minimal.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..._registry import register_function


@register_function(
    aliases=[
        "plot_contamination", "ambient_contamination_plot",
        "污染分布图", "环境RNA污染图",
    ],
    category="preprocessing",
    description=(
        "Plot the per-cell ambient-RNA contamination fraction written by "
        "ov.pp.ambient.remove_ambient — a histogram, or a per-group "
        "boxplot when groupby is given. For UMAP overlays use "
        "ov.pl.embedding(adata, color='ambient_contamination')."
    ),
    examples=[
        "ov.pp.ambient.plot_contamination(adata)",
        "ov.pp.ambient.plot_contamination(adata, groupby='cell_type')",
    ],
    related=["pp.ambient.remove_ambient", "pp.ambient.contamination_report"],
)
def plot_contamination(
    adata,
    *,
    obs_key: str = "ambient_contamination",
    groupby: Optional[str] = None,
    bins: int = 40,
    figsize: tuple = (5, 3.5),
    color: str = "#4C72B0",
    ax=None,
):
    """Plot the per-cell contamination-fraction distribution.

    Parameters
    ----------
    adata
        AnnData processed by :func:`~omicverse.pp.ambient.remove_ambient`.
    obs_key
        ``obs`` column with the per-cell contamination fraction.
    groupby
        Optional ``obs`` column — when given, draw a per-group boxplot
        instead of a histogram.
    bins
        Histogram bin count (ignored when ``groupby`` is set).
    figsize
        Figure size when ``ax`` is not supplied.
    color
        Bar / box face colour.
    ax
        Existing matplotlib axes to draw into.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the diagnostic was drawn on.
    """
    import matplotlib.pyplot as plt

    if obs_key not in adata.obs:
        raise KeyError(
            f"obs['{obs_key}'] not found — run ov.pp.ambient.remove_ambient "
            "(or estimate_contamination) first.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    rho = adata.obs[obs_key].to_numpy(dtype=float)

    if groupby is None:
        ax.hist(rho, bins=bins, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.mean(rho)), color="#C44E52", linestyle="--",
                   linewidth=1.5, label=f"mean = {np.mean(rho):.3f}")
        ax.set_xlabel("contamination fraction")
        ax.set_ylabel("number of cells")
        ax.legend(frameon=False)
    else:
        if groupby not in adata.obs:
            raise KeyError(f"groupby '{groupby}' not in adata.obs.")
        labels = adata.obs[groupby].astype(str)
        groups = sorted(labels.unique())
        data = [rho[labels.to_numpy() == g] for g in groups]
        bp = ax.boxplot(data, labels=groups, patch_artist=True,
                        showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("contamination fraction")
        ax.set_xlabel(groupby)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")

    ax.set_title("Ambient-RNA contamination")
    ax.spines[["top", "right"]].set_visible(False)
    return ax
