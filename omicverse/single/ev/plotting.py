"""Plotting for single-extracellular-vesicle (single-EV) proteomics.

Matplotlib-based visualization of single-EV proteomic AnnData:

* :func:`embedding_plot` — UMAP/PCA scatter coloured by cluster / marker /
  condition.
* :func:`marker_dotplot` — per-subpopulation mean-expression bubble plot.
* :func:`subpopulation_composition_plot` — stacked-bar EV-subpopulation
  composition per sample / condition.
* :func:`marker_heatmap` — protein x subpopulation mean-signal heatmap.
* :func:`misev_marker_plot` — MISEV positive vs contaminant marker levels
  (the EV-purity view).

Every function is ``@register_function``-decorated (``category='ev'``) and
returns the :class:`matplotlib.axes.Axes`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..._registry import register_function


def _ax(ax, figsize):
    """Create a fresh Axes when none is supplied."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _dense(adata):
    """Return ``adata.X`` as a dense float ndarray."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=float)


@register_function(
    aliases=[
        "embedding_plot", "ev_embedding_plot", "plot_ev_embedding",
        "EV降维图", "EV嵌入散点图",
    ],
    category="ev",
    description=(
        "Scatter a single-EV 2-D embedding (UMAP / PCA from obsm) with EVs "
        "coloured by a cluster / condition obs column or by a protein "
        "marker's signal."
    ),
    examples=[
        "ov.single.ev.plotting.embedding_plot(adata, color='leiden')",
        "ov.single.ev.plotting.embedding_plot(adata, color='CD63', "
        "basis='X_umap')",
    ],
    related=["single.ev.plotting.marker_dotplot"],
)
def embedding_plot(
    adata,
    *,
    color: Optional[str] = None,
    basis: str = "X_umap",
    ax=None,
    figsize=(6, 5),
    size: float = 12,
    cmap: str = "viridis",
    title: Optional[str] = None,
):
    """Scatter a single-EV UMAP / PCA embedding.

    Parameters
    ----------
    adata
        Single-EV AnnData with the 2-D embedding in ``obsm[basis]``.
    color
        ``obs`` column (categorical cluster / condition) or a ``var`` name
        (protein marker) used to colour the EVs; ``None`` draws plain
        points.
    basis
        ``obsm`` key of the embedding (default ``'X_umap'``).
    ax, figsize, size, cmap
        Standard matplotlib styling controls.
    title
        Plot title; defaults to ``color``.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    import matplotlib.pyplot as plt

    if basis not in adata.obsm:
        raise KeyError(
            f"obsm[{basis!r}] not found — compute an embedding first."
        )
    ax = _ax(ax, figsize)
    xy = np.asarray(adata.obsm[basis], dtype=float)[:, :2]

    if color is not None and color in adata.obs and not \
            pd.api.types.is_numeric_dtype(adata.obs[color]):
        cats = pd.Categorical(adata.obs[color].astype(str))
        cmap_obj = plt.get_cmap("tab20")
        for i, c in enumerate(cats.categories):
            sel = np.asarray(cats == c)
            ax.scatter(xy[sel, 0], xy[sel, 1], s=size,
                       color=cmap_obj(i % 20), label=str(c),
                       edgecolors="none")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
                  frameon=False, markerscale=2)
    elif color is not None and (
        color in adata.var_names or (
            color in adata.obs
            and pd.api.types.is_numeric_dtype(adata.obs[color])
        )
    ):
        if color in adata.var_names:
            vals = _dense(adata)[:, list(adata.var_names).index(color)]
        else:
            vals = np.asarray(adata.obs[color], dtype=float)
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=size, c=vals, cmap=cmap,
                        edgecolors="none")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=size, color="#4878CF",
                   edgecolors="none")

    ax.set_xlabel(f"{basis}-1")
    ax.set_ylabel(f"{basis}-2")
    ax.set_title(title if title is not None else (color or "EV embedding"))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


@register_function(
    aliases=[
        "marker_dotplot", "ev_marker_dotplot", "plot_ev_dotplot",
        "EV标志物点图", "亚群气泡图",
    ],
    category="ev",
    description=(
        "Per-EV-subpopulation marker-protein dot/bubble plot: dot colour "
        "encodes mean signal, dot size encodes the fraction of EVs in which "
        "the protein is detected."
    ),
    examples=[
        "ov.single.ev.plotting.marker_dotplot(adata, groupby='leiden', "
        "proteins=['CD9', 'CD63', 'CD81'])",
    ],
    related=["single.ev.rank_markers", "single.ev.plotting.marker_heatmap"],
)
def marker_dotplot(
    adata,
    *,
    groupby: str,
    proteins: Optional[list] = None,
    ax=None,
    figsize=None,
    cmap: str = "Reds",
    title: str = "Marker dotplot",
):
    """Per-subpopulation marker-protein dot/bubble plot.

    Parameters
    ----------
    adata
        Single-EV AnnData.
    groupby
        ``obs`` column with the EV-subpopulation labels.
    proteins
        Proteins to show; ``None`` uses all ``var`` names.
    ax, figsize, cmap
        Standard matplotlib styling controls; ``figsize`` defaults to a
        size derived from the grid shape.
    title
        Plot title.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    import matplotlib.pyplot as plt

    if groupby not in adata.obs:
        raise KeyError(f"obs[{groupby!r}] not found.")
    proteins = list(proteins) if proteins is not None else list(adata.var_names)
    idx = [list(adata.var_names).index(p) for p in proteins]
    X = _dense(adata)[:, idx]
    labels = adata.obs[groupby].astype(str).values
    groups = list(pd.unique(labels))

    mean_mat = np.zeros((len(groups), len(proteins)))
    frac_mat = np.zeros((len(groups), len(proteins)))
    for i, g in enumerate(groups):
        sub = X[labels == g]
        if sub.shape[0]:
            mean_mat[i] = sub.mean(axis=0)
            frac_mat[i] = (sub > 0).mean(axis=0)

    if figsize is None:
        figsize = (0.55 * len(proteins) + 3, 0.5 * len(groups) + 2)
    ax = _ax(ax, figsize)
    mmax = mean_mat.max() if mean_mat.max() > 0 else 1.0
    for i in range(len(groups)):
        for j in range(len(proteins)):
            ax.scatter(
                j, i, s=20 + 300 * frac_mat[i, j],
                c=[mean_mat[i, j]], cmap=cmap, vmin=0, vmax=mmax,
                edgecolors="grey", linewidths=0.4,
            )
    ax.set_xticks(range(len(proteins)))
    ax.set_xticklabels(proteins, rotation=90, fontsize=8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_xlim(-0.5, len(proteins) - 0.5)
    ax.set_ylim(-0.5, len(groups) - 0.5)
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=mmax))
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="mean signal")
    return ax


@register_function(
    aliases=[
        "subpopulation_composition_plot", "ev_composition_plot",
        "plot_ev_composition", "EV亚群组成图", "亚群堆叠柱状图",
    ],
    category="ev",
    description=(
        "Stacked-bar plot of EV-subpopulation composition per sample or per "
        "condition — the fraction (or count) of EVs in each subpopulation "
        "for every group."
    ),
    examples=[
        "ov.single.ev.plotting.subpopulation_composition_plot(adata, "
        "groupby='sample', cluster_key='leiden')",
    ],
    related=["single.ev.differential_subpopulation"],
)
def subpopulation_composition_plot(
    adata,
    *,
    groupby: str,
    cluster_key: str,
    normalize: bool = True,
    ax=None,
    figsize=(7, 4),
    cmap: str = "tab20",
):
    """Stacked-bar EV-subpopulation composition per group.

    Parameters
    ----------
    adata
        Single-EV AnnData.
    groupby
        ``obs`` column for the x-axis groups (sample / condition).
    cluster_key
        ``obs`` column with the EV-subpopulation labels.
    normalize
        Plot per-group fractions (default) instead of EV counts.
    ax, figsize, cmap
        Standard matplotlib styling controls.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    for k in (groupby, cluster_key):
        if k not in adata.obs:
            raise KeyError(f"obs[{k!r}] not found.")
    ax = _ax(ax, figsize)
    tab = pd.crosstab(
        adata.obs[groupby].astype(str),
        adata.obs[cluster_key].astype(str),
    )
    if normalize:
        tab = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    tab.plot(kind="bar", stacked=True, ax=ax, colormap=cmap, width=0.8)
    ax.set_ylabel("fraction of EVs" if normalize else "n EVs")
    ax.set_xlabel(groupby)
    ax.set_title("EV-subpopulation composition")
    ax.legend(title=cluster_key, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, frameon=False)
    return ax


@register_function(
    aliases=[
        "marker_heatmap", "ev_marker_heatmap", "plot_ev_heatmap",
        "EV标志物热图", "蛋白亚群热图",
    ],
    category="ev",
    description=(
        "Protein x EV-subpopulation heatmap of mean (optionally z-scored) "
        "marker signal — the per-subpopulation protein-signature view."
    ),
    examples=[
        "ov.single.ev.plotting.marker_heatmap(adata, groupby='leiden')",
        "ov.single.ev.plotting.marker_heatmap(adata, groupby='flowsom', "
        "proteins=['CD9', 'CD63'], standard_scale=True)",
    ],
    related=["single.ev.rank_markers", "single.ev.plotting.marker_dotplot"],
)
def marker_heatmap(
    adata,
    *,
    groupby: str,
    proteins: Optional[list] = None,
    standard_scale: bool = False,
    ax=None,
    figsize=None,
    cmap: str = "viridis",
    title: str = "Marker heatmap",
):
    """Protein x EV-subpopulation mean-signal heatmap.

    Parameters
    ----------
    adata
        Single-EV AnnData.
    groupby
        ``obs`` column with the EV-subpopulation labels.
    proteins
        Proteins to show; ``None`` uses all ``var`` names.
    standard_scale
        z-score each protein row across subpopulations before plotting.
    ax, figsize, cmap
        Standard matplotlib styling controls.
    title
        Plot title.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    import matplotlib.pyplot as plt

    if groupby not in adata.obs:
        raise KeyError(f"obs[{groupby!r}] not found.")
    proteins = list(proteins) if proteins is not None else list(adata.var_names)
    idx = [list(adata.var_names).index(p) for p in proteins]
    X = _dense(adata)[:, idx]
    labels = adata.obs[groupby].astype(str).values
    groups = list(pd.unique(labels))

    mat = np.zeros((len(proteins), len(groups)))
    for j, g in enumerate(groups):
        sub = X[labels == g]
        if sub.shape[0]:
            mat[:, j] = sub.mean(axis=0)
    if standard_scale:
        sd = mat.std(axis=1, keepdims=True)
        mat = (mat - mat.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)

    if figsize is None:
        figsize = (0.55 * len(groups) + 3, 0.32 * len(proteins) + 2)
    ax = _ax(ax, figsize)
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels(proteins, fontsize=7)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="z-score" if standard_scale else "mean signal")
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
