"""Visualization functions for metabolism scores."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Union, List
import anndata
import logging

logger = logging.getLogger(__name__)


def _zissou1_cmap():
    colors = ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"]
    return mcolors.LinearSegmentedColormap.from_list("Zissou1", colors, N=256)


def dimplot_metabolism(
    adata: anndata.AnnData,
    pathway: str,
    reduction: str = "umap",
    rerun: bool = False,
    size: float = 1.0,
    ax=None,
    **kwargs,
):
    """Visualize metabolism score on dimensionality reduction.

    Parameters
    ----------
    adata : AnnData
        AnnData object with metabolism scores in .obsm['X_metabolism'] and
        dimensionality reduction in .obsm['X_umap'] or .obsm['X_tsne'].
    pathway : str
        Pathway name to plot.
    reduction : str
        'umap' or 'tsne'.
    rerun : bool
        Whether to re-run dimensionality reduction (not implemented).
    size : float
        Dot size.
    ax : matplotlib Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib Axes
    """
    logger.info("\nPlease Cite: \nYingcheng Wu, Qiang Gao, et al. Cancer Discovery. 2021. \nhttps://pubmed.ncbi.nlm.nih.gov/34417225/   \n")

    if "X_metabolism" not in adata.obsm:
        raise ValueError("Metabolism scores not found. Run sc_metabolism_anndata first.")
    pathway_names = adata.uns.get("metabolism_pathways", [])
    if pathway not in pathway_names:
        raise ValueError(f"Pathway '{pathway}' not found. Available: {pathway_names}")

    pathway_idx = pathway_names.index(pathway)
    scores = adata.obsm["X_metabolism"][:, pathway_idx]

    if reduction == "umap":
        if "X_umap" not in adata.obsm:
            raise ValueError("UMAP coordinates not found in adata.obsm['X_umap']")
        emb = adata.obsm["X_umap"]
        xlabel, ylabel = "UMAP 1", "UMAP 2"
    elif reduction == "tsne":
        if "X_tsne" not in adata.obsm:
            raise ValueError("tSNE coordinates not found in adata.obsm['X_tsne']")
        emb = adata.obsm["X_tsne"]
        xlabel, ylabel = "tSNE 1", "tSNE 2"
    else:
        raise ValueError("reduction must be 'umap' or 'tsne'")

    cmap = _zissou1_cmap()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    sc = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=scores,
        s=size,
        cmap=cmap,
        edgecolors="none",
        **kwargs,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(pathway, fontsize=12)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    return ax


def dotplot_metabolism(
    adata: anndata.AnnData,
    pathways: List[str],
    phenotype: str,
    norm: str = "y",
    ax=None,
    **kwargs,
):
    """Dot plot of metabolism scores across groups.

    Parameters
    ----------
    adata : AnnData
        AnnData with metabolism scores and phenotype in .obs.
    pathways : list of str
        Pathway names to plot.
    phenotype : str
        Column name in adata.obs for grouping.
    norm : str
        Normalization direction: 'x', 'y', or 'na'.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Axes
    """
    logger.info("\nPlease Cite: \nYingcheng Wu, Qiang Gao, et al. Cancer Discovery. 2021. \nhttps://pubmed.ncbi.nlm.nih.gov/34417225/   \n")

    if "X_metabolism" not in adata.obsm:
        raise ValueError("Metabolism scores not found.")
    pathway_names = adata.uns.get("metabolism_pathways", [])
    for p in pathways:
        if p not in pathway_names:
            raise ValueError(f"Pathway '{p}' not found.")

    if phenotype not in adata.obs:
        raise ValueError(f"Phenotype '{phenotype}' not in adata.obs.")

    pathway_indices = [pathway_names.index(p) for p in pathways]
    scores = adata.obsm["X_metabolism"][:, pathway_indices]
    groups = adata.obs[phenotype].astype(str).values

    records = []
    for i, (group, score_row) in enumerate(zip(groups, scores)):
        for j, pathway in enumerate(pathways):
            records.append({
                "group": group,
                "pathway": pathway,
                "score": score_row[j],
            })
    df = pd.DataFrame(records)

    median_df = df.groupby(["group", "pathway"], as_index=False)["score"].median()

    if norm == "y":
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
        median_df["score_norm"] = median_df.groupby("pathway")["score"].transform(normalize)
    elif norm == "x":
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
        median_df["score_norm"] = median_df.groupby("group")["score"].transform(normalize)
    elif norm == "na":
        median_df["score_norm"] = median_df["score"]
    else:
        raise ValueError('norm must be "x", "y", or "na"')

    pivot = median_df.pivot(index="pathway", columns="group", values="score_norm")

    unique_groups = list(pivot.columns)
    unique_pathways = list(pivot.index)
    n_groups = len(unique_groups)
    n_pathways = len(unique_pathways)

    cmap = _zissou1_cmap()

    if ax is None:
        fig_width = max(4, n_groups * 0.8 + 2.5)
        fig_height = max(3, n_pathways * 0.5 + 1)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_positions = []
    y_positions = []
    color_values = []
    for i, pathway in enumerate(unique_pathways):
        for j, group in enumerate(unique_groups):
            val = pivot.loc[pathway, group]
            if pd.isna(val):
                continue
            x_positions.append(j)
            y_positions.append(i)
            color_values.append(val)

    color_values = np.array(color_values)

    vmin = color_values.min() if len(color_values) > 0 else 0
    vmax = color_values.max() if len(color_values) > 0 else 1
    if vmax <= vmin:
        vmax = vmin + 1.0

    norm_for_size = (color_values - vmin) / (vmax - vmin)

    min_dot_size = 10
    max_dot_size = 200
    scaled_sizes = min_dot_size + norm_for_size * (max_dot_size - min_dot_size)

    sc = ax.scatter(
        x_positions,
        y_positions,
        c=color_values,
        s=scaled_sizes,
        cmap=cmap,
        edgecolors="none",
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Value", fontsize=12)

    legend_vals = np.linspace(vmin, vmax, 5)
    legend_norm = (legend_vals - vmin) / (vmax - vmin)
    legend_sizes = min_dot_size + legend_norm * (max_dot_size - min_dot_size)

    legend_handles = []
    for v, s in zip(legend_vals, legend_sizes):
        handle = plt.scatter([], [], c='gray', s=s, edgecolors='none')
        legend_handles.append(handle)

    size_legend = ax.legend(
        legend_handles,
        [f"{v:.2f}" for v in legend_vals],
        title="Value",
        loc='upper left',
        bbox_to_anchor=(1.35, 1.0),
        frameon=True,
        fontsize=9,
    )
    size_legend.get_title().set_fontsize(10)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(unique_groups, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(n_pathways))
    ax.set_yticklabels(unique_pathways, fontsize=10)

    ax.set_xlabel(phenotype, fontsize=12)
    ax.set_ylabel("Metabolic Pathway", fontsize=12)

    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.grid(False)

    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylim(-0.5, n_pathways - 0.5)

    return ax


def boxplot_metabolism(
    adata: anndata.AnnData,
    pathways: List[str],
    phenotype: str,
    ncol: int = 1,
    ax=None,
    **kwargs,
):
    """Box plot of metabolism scores across groups.

    Parameters
    ----------
    adata : AnnData
        AnnData with metabolism scores and phenotype.
    pathways : list of str
        Pathway names.
    phenotype : str
        Column in adata.obs for grouping.
    ncol : int
        Number of columns in faceted plot.
    ax : matplotlib Axes, optional
        Not used if multiple facets.

    Returns
    -------
    matplotlib Figure
    """
    logger.info("\nPlease Cite: \nYingcheng Wu, Qiang Gao, et al. Cancer Discovery. 2021. \nhttps://pubmed.ncbi.nlm.nih.gov/34417225/   \n")

    if "X_metabolism" not in adata.obsm:
        raise ValueError("Metabolism scores not found.")
    pathway_names = adata.uns.get("metabolism_pathways", [])
    for p in pathways:
        if p not in pathway_names:
            raise ValueError(f"Pathway '{p}' not found.")

    if phenotype not in adata.obs:
        raise ValueError(f"Phenotype '{phenotype}' not in adata.obs.")

    pathway_indices = [pathway_names.index(p) for p in pathways]
    scores = adata.obsm["X_metabolism"][:, pathway_indices]
    groups = adata.obs[phenotype].astype(str).values

    records = []
    for i, (group, score_row) in enumerate(zip(groups, scores)):
        for j, pathway in enumerate(pathways):
            records.append({
                "group": group,
                "pathway": pathway,
                "score": score_row[j],
            })
    df = pd.DataFrame(records)

    unique_groups = sorted(df["group"].unique())
    n_groups = len(unique_groups)
    n_pathways = len(pathways)
    nrow = int(np.ceil(n_pathways / ncol))

    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(ncol * 4, nrow * 3.5),
        squeeze=False,
    )

    cmap_groups = plt.cm.Set2(np.linspace(0, 1, max(n_groups, 1)))
    group_colors = {g: cmap_groups[i] for i, g in enumerate(unique_groups)}

    for idx, pathway in enumerate(pathways):
        row_idx = idx // ncol
        col_idx = idx % ncol
        ax_sub = axes[row_idx, col_idx]

        sub_df = df[df["pathway"] == pathway]

        box_data = []
        labels = []
        colors = []
        for g in unique_groups:
            g_scores = sub_df[sub_df["group"] == g]["score"].values
            box_data.append(g_scores)
            labels.append(g)
            colors.append(group_colors[g])

        bp = ax_sub.boxplot(
            box_data,
            patch_artist=True,
            showfliers=False,
            widths=0.6,
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set_color("black")
                item.set_linewidth(0.8)

        ax_sub.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax_sub.set_title(pathway, fontsize=11)
        ax_sub.set_xlabel(phenotype, fontsize=10)
        ax_sub.set_ylabel("Metabolic Pathway", fontsize=10)

        ax_sub.set_facecolor("white")
        for spine in ax_sub.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)
        ax_sub.grid(False)

    for idx in range(n_pathways, nrow * ncol):
        row_idx = idx // ncol
        col_idx = idx % ncol
        axes[row_idx, col_idx].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=group_colors[g], edgecolor="black")
        for g in unique_groups
    ]
    fig.legend(
        handles, unique_groups,
        title=phenotype,
        loc='upper left',
        bbox_to_anchor=(1.0, 1.0),
        fontsize=10,
    )

    plt.tight_layout()
    return fig
