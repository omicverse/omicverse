"""Plotting for single-cell immune-repertoire (AIRR) analysis.

Matplotlib-based plots — an omicverse-style reimplementation of scirpy's
``pl`` module: clonotype-network plot, clonal-expansion plot, spectratype,
V(D)J usage, repertoire-overlap heatmap and group-abundance bars.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .._registry import register_function


def _ax(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


@register_function(
    aliases=["clonotype_network_plot", "plot_clonotype_network", "克隆型网络图"],
    category="airr",
    description=(
        "Scatter the clonotype-network layout (obsm['X_clonotype_network']) "
        "with points coloured by an obs column."
    ),
    requires={"obsm": ["X_clonotype_network"]},
    examples=[
        "ov.airr.plotting.clonotype_network_plot(adata, color='clonal_expansion')",
    ],
    related=["airr.clonotype_network"],
)
def clonotype_network_plot(
    adata,
    *,
    color: Optional[str] = None,
    ax=None,
    figsize=(6, 6),
    size: float = 25,
    title: str = "Clonotype network",
):
    """Plot the clonotype-network layout.

    Parameters
    ----------
    adata
        AnnData with ``obsm['X_clonotype_network']`` from
        :func:`omicverse.airr.clonotype_network`.
    color
        ``obs`` column used to colour the nodes.
    ax, figsize, size, title
        Standard matplotlib styling controls.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    import matplotlib.pyplot as plt

    if "X_clonotype_network" not in adata.obsm:
        raise KeyError("Run ov.airr.clonotype_network first.")
    ax = _ax(ax, figsize)
    coords = np.asarray(adata.obsm["X_clonotype_network"], dtype=float)
    mask = ~np.isnan(coords[:, 0])
    if color is not None and color in adata.obs:
        vals = adata.obs[color][mask]
        cats = pd.Categorical(vals)
        cmap = plt.get_cmap("tab20")
        for i, c in enumerate(cats.categories):
            sel = (cats == c)
            pts = coords[mask][sel]
            ax.scatter(pts[:, 0], pts[:, 1], s=size, color=cmap(i % 20),
                       label=str(c), edgecolors="white", linewidths=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=8, frameon=False)
    else:
        ax.scatter(coords[mask, 0], coords[mask, 1], s=size,
                   color="#4878CF", edgecolors="white", linewidths=0.3)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


@register_function(
    aliases=["clonal_expansion_plot", "plot_clonal_expansion", "克隆扩增图"],
    category="airr",
    description=(
        "Stacked-bar plot of clonal-expansion categories per cell group."
    ),
    requires={"obs": ["clonal_expansion"]},
    examples=[
        "ov.airr.plotting.clonal_expansion_plot(adata, groupby='group')",
    ],
    related=["airr.clonal_expansion"],
)
def clonal_expansion_plot(
    adata,
    groupby: str,
    *,
    key: str = "clonal_expansion",
    normalize: bool = True,
    ax=None,
    figsize=(6, 4),
):
    """Stacked-bar plot of clonal expansion per group.

    Parameters
    ----------
    adata
        AnnData with ``obs[key]`` from
        :func:`omicverse.airr.clonal_expansion`.
    groupby
        ``obs`` column for the x-axis groups.
    key
        Clonal-expansion column (default ``'clonal_expansion'``).
    normalize
        Plot fractions (default) instead of counts.
    ax, figsize
        Matplotlib controls.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    import matplotlib.pyplot as plt

    ax = _ax(ax, figsize)
    sub = adata.obs.dropna(subset=[key])
    tab = pd.crosstab(sub[groupby], sub[key])
    if normalize:
        tab = tab.div(tab.sum(axis=1), axis=0)
    tab.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", width=0.8)
    ax.set_ylabel("fraction of cells" if normalize else "n cells")
    ax.set_title("Clonal expansion")
    ax.legend(title=key, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, frameon=False)
    return ax


@register_function(
    aliases=["spectratype_plot", "plot_spectratype", "谱型图"],
    category="airr",
    description="Line / area plot of the CDR3-length spectratype per group.",
    examples=[
        "ov.airr.plotting.spectratype_plot(adata, groupby='group')",
    ],
    related=["airr.spectratype"],
)
def spectratype_plot(
    adata,
    groupby: Optional[str] = None,
    *,
    chain: str = "VDJ_1",
    sequence: str = "aa",
    ax=None,
    figsize=(6, 4),
):
    """Plot the CDR3-length spectratype.

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData.
    groupby, chain, sequence
        Passed to :func:`omicverse.airr.spectratype`.
    ax, figsize
        Matplotlib controls.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    from ._metrics import spectratype

    ax = _ax(ax, figsize)
    tab = spectratype(adata, groupby, chain=chain, sequence=sequence)
    for g in tab.index:
        ax.plot(tab.columns, tab.loc[g].values, marker="o", label=str(g))
    ax.set_xlabel("CDR3 length")
    ax.set_ylabel("n cells")
    ax.set_title(f"Spectratype ({chain})")
    ax.legend(fontsize=8, frameon=False)
    return ax


@register_function(
    aliases=["vdj_usage_plot", "plot_vdj_usage", "gene_usage_plot", "基因使用图"],
    category="airr",
    description="Bar plot of V/D/J gene-segment usage frequencies per group.",
    examples=[
        "ov.airr.plotting.vdj_usage_plot(adata, gene='v', groupby='group')",
    ],
    related=["airr.vdj_usage"],
)
def vdj_usage_plot(
    adata,
    *,
    gene: str = "v",
    chain: str = "VDJ_1",
    groupby: Optional[str] = None,
    top: int = 15,
    ax=None,
    figsize=(8, 4),
):
    """Bar plot of V/D/J gene-segment usage.

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData.
    gene, chain, groupby
        Passed to :func:`omicverse.airr.vdj_usage`.
    top
        Plot only the ``top`` most-used genes.
    ax, figsize
        Matplotlib controls.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    from ._metrics import vdj_usage

    ax = _ax(ax, figsize)
    tab = vdj_usage(adata, gene=gene, chain=chain, groupby=groupby,
                    normalize=True)
    order = tab.sum(axis=0).sort_values(ascending=False).index[:top]
    tab = tab[order]
    tab.T.plot(kind="bar", ax=ax, colormap="tab10", width=0.8)
    ax.set_ylabel("usage frequency")
    ax.set_xlabel(f"{gene.upper()} gene")
    ax.set_title(f"{gene.upper()} gene usage ({chain})")
    ax.legend(fontsize=8, frameon=False)
    return ax


@register_function(
    aliases=["repertoire_overlap_plot", "plot_repertoire_overlap", "组库重叠热图"],
    category="airr",
    description="Heatmap of the pairwise repertoire-overlap matrix.",
    examples=[
        "ov.airr.plotting.repertoire_overlap_plot(adata, groupby='sample')",
    ],
    related=["airr.repertoire_overlap"],
)
def repertoire_overlap_plot(
    adata,
    groupby: str,
    *,
    target_col: str = "clone_id",
    metric: str = "jaccard",
    ax=None,
    figsize=(5, 4),
    cmap: str = "viridis",
):
    """Heatmap of the repertoire-overlap matrix.

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData.
    groupby, target_col, metric
        Passed to :func:`omicverse.airr.repertoire_overlap`.
    ax, figsize, cmap
        Matplotlib controls.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    import matplotlib.pyplot as plt

    from ._metrics import repertoire_overlap

    ax = _ax(ax, figsize)
    mat = repertoire_overlap(adata, groupby, target_col=target_col,
                             metric=metric)
    im = ax.imshow(mat.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_title(f"Repertoire overlap ({metric})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


@register_function(
    aliases=["group_abundance_plot", "plot_group_abundance", "分组丰度图"],
    category="airr",
    description="Stacked-bar plot of clonotype/category abundance per group.",
    examples=[
        "ov.airr.plotting.group_abundance_plot(adata, groupby='group')",
    ],
    related=["airr.group_abundance"],
)
def group_abundance_plot(
    adata,
    groupby: str,
    *,
    target_col: str = "clone_id",
    normalize: bool = True,
    max_cols: int = 10,
    ax=None,
    figsize=(7, 4),
):
    """Stacked-bar plot of group abundance.

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData.
    groupby, target_col, normalize, max_cols
        Passed to :func:`omicverse.airr.group_abundance`.
    ax, figsize
        Matplotlib controls.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    from ._metrics import group_abundance

    ax = _ax(ax, figsize)
    tab = group_abundance(adata, groupby, target_col=target_col,
                          normalize=normalize, max_cols=max_cols)
    tab.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.8)
    ax.set_ylabel("fraction" if normalize else "n cells")
    ax.set_title(f"Group abundance ({target_col})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
              frameon=False)
    return ax
