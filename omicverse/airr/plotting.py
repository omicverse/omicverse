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
    aliases=["kmer_motif_plot", "plot_kmer_motif", "motif_logo", "K-mer基序图"],
    category="airr",
    description=(
        "Sequence-logo / stacked-bar plot of a per-position k-mer motif "
        "profile (PFM / PPM / PWM) from ov.airr.kmer_motif."
    ),
    examples=[
        "prof = ov.airr.kmer_motif(ov.airr.kmer_analysis(immdata, k=5))",
        "ov.airr.plotting.kmer_motif_plot(prof)",
    ],
    related=["airr.kmer_motif", "airr.kmer_analysis"],
)
def kmer_motif_plot(
    profile,
    *,
    ax=None,
    figsize=(8, 4),
    cmap: str = "tab20",
    title: str = "K-mer motif",
):
    """Stacked-bar sequence-logo of a k-mer motif profile.

    Parameters
    ----------
    profile
        A per-position amino-acid profile :class:`pandas.DataFrame` from
        :func:`omicverse.airr.kmer_motif` — amino acids on the rows,
        k-mer positions on the columns.
    ax, figsize, cmap, title
        Standard matplotlib styling controls.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    import matplotlib.pyplot as plt

    ax = _ax(ax, figsize)
    mat = profile.fillna(0.0)
    positions = list(mat.columns)
    x = np.arange(len(positions))
    cmap_obj = plt.get_cmap(cmap)
    bottom = np.zeros(len(positions))
    for i, aa in enumerate(mat.index):
        vals = mat.loc[aa].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.85,
               color=cmap_obj(i % cmap_obj.N), label=str(aa))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_xlabel("k-mer position")
    ax.set_ylabel("profile value")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
              frameon=False, ncol=1)
    return ax


@register_function(
    aliases=["cdr3_aa_profile_plot", "plot_cdr3_aa_profile",
             "cdr3_property_plot", "CDR3理化谱图"],
    category="airr",
    description=(
        "Line / stacked-bar plot of a per-position CDR3 amino-acid "
        "composition or physicochemical property profile from "
        "ov.airr.cdr3_aa_properties."
    ),
    examples=[
        "prof = ov.airr.cdr3_aa_properties(immdata, sample='MS1')",
        "ov.airr.plotting.cdr3_aa_profile_plot(prof)",
    ],
    related=["airr.cdr3_aa_properties", "airr.plotting.kmer_motif_plot"],
)
def cdr3_aa_profile_plot(
    profile,
    *,
    ax=None,
    figsize=(8, 4),
    cmap: str = "tab20",
    title: str = "CDR3 amino-acid profile",
):
    """Plot a per-position CDR3 amino-acid / property profile.

    A single-row profile (a physicochemical property) is drawn as a line;
    a multi-row profile (per-position amino-acid composition) is drawn as a
    stacked bar.

    Parameters
    ----------
    profile
        A per-position profile :class:`pandas.DataFrame` from
        :func:`omicverse.airr.cdr3_aa_properties` — positions on the
        columns.
    ax, figsize, cmap, title
        Standard matplotlib styling controls.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    import matplotlib.pyplot as plt

    ax = _ax(ax, figsize)
    mat = profile.fillna(0.0)
    positions = list(mat.columns)
    x = np.arange(len(positions))
    if len(mat.index) == 1:
        ax.plot(x, mat.iloc[0].to_numpy(dtype=float), marker="o",
                color="#4878CF")
        ax.set_ylabel(str(mat.index[0]))
    else:
        cmap_obj = plt.get_cmap(cmap)
        bottom = np.zeros(len(positions))
        for i, aa in enumerate(mat.index):
            vals = mat.loc[aa].to_numpy(dtype=float)
            ax.bar(x, vals, bottom=bottom, width=0.85,
                   color=cmap_obj(i % cmap_obj.N), label=str(aa))
            bottom += vals
        ax.set_ylabel("composition")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
                  frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_xlabel("CDR3 position")
    ax.set_title(title)
    return ax


@register_function(
    aliases=["gene_usage_analysis_plot", "plot_gene_usage_analysis",
             "基因使用降维图"],
    category="airr",
    description=(
        "Scatter the 2-D sample embedding from ov.airr.gene_usage_analysis "
        "(or ov.airr.overlap_analysis), coloured by cluster label."
    ),
    examples=[
        "res = ov.airr.gene_usage_analysis(gu, reduction='mds')",
        "ov.airr.plotting.gene_usage_analysis_plot(res)",
    ],
    related=["airr.gene_usage_analysis", "airr.overlap_analysis"],
)
def gene_usage_analysis_plot(
    result,
    *,
    ax=None,
    figsize=(6, 5),
    size: float = 80,
    cmap: str = "tab10",
    label_points: bool = True,
    title: str = "Gene-usage analysis",
):
    """Scatter the gene-usage / overlap sample embedding.

    Parameters
    ----------
    result
        The dict returned by :func:`omicverse.airr.gene_usage_analysis`
        (keys ``'embedding'`` / ``'clusters'``) or by
        :func:`omicverse.airr.overlap_analysis` (keys ``'coords'`` /
        ``'clusters'``).
    ax, figsize, size, cmap
        Standard matplotlib styling controls.
    label_points
        Annotate each point with its sample name.
    title
        Plot title.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    import matplotlib.pyplot as plt

    ax = _ax(ax, figsize)
    coords = result.get("embedding")
    if coords is None:
        coords = result.get("coords")
    if coords is None:
        raise KeyError(
            "result must hold an 'embedding' or 'coords' DataFrame."
        )
    xy = np.asarray(coords.iloc[:, :2], dtype=float)
    names = list(coords.index)
    clusters = result.get("clusters")
    if clusters is not None:
        cats = pd.Categorical(pd.Series(clusters).reindex(names))
        cmap_obj = plt.get_cmap(cmap)
        for i, c in enumerate(cats.categories):
            sel = (cats == c)
            ax.scatter(xy[sel, 0], xy[sel, 1], s=size,
                       color=cmap_obj(i % cmap_obj.N), label=str(c),
                       edgecolors="white", linewidths=0.5)
        ax.legend(title="cluster", fontsize=8, frameon=False)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=size, color="#4878CF",
                   edgecolors="white", linewidths=0.5)
    if label_points:
        for (px, py), name in zip(xy, names):
            ax.annotate(str(name), (px, py), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(str(coords.columns[0]))
    ax.set_ylabel(str(coords.columns[1]))
    ax.set_title(title)
    return ax


@register_function(
    aliases=["clonal_abundance_plot", "plot_clonal_abundance",
             "rank_abundance_plot", "克隆丰度曲线"],
    category="airr",
    description=(
        "Rank-abundance curve plot — clone abundance vs rank with "
        "bootstrap CI ribbons — from ov.airr.clonal_abundance."
    ),
    examples=[
        "ab = ov.airr.clonal_abundance(db, group='sample_id')",
        "ov.airr.plotting.clonal_abundance_plot(ab)",
    ],
    related=["airr.clonal_abundance", "airr.hill_diversity"],
)
def clonal_abundance_plot(
    abundance,
    *,
    ax=None,
    figsize=(6, 4),
    cmap: str = "tab10",
    ci_alpha: float = 0.2,
    title: str = "Rank abundance",
):
    """Plot a clonal rank-abundance distribution with CI ribbons.

    Parameters
    ----------
    abundance
        A :class:`pyalakazam.AbundanceCurve` from
        :func:`omicverse.airr.clonal_abundance`, or its tidy
        ``.abundance`` DataFrame (columns ``rank`` / ``p`` /
        ``lower`` / ``upper`` and an optional group column).
    ax, figsize, cmap
        Standard matplotlib styling controls.
    ci_alpha
        Opacity of the confidence-interval ribbon.
    title
        Plot title.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    import matplotlib.pyplot as plt

    ax = _ax(ax, figsize)
    df = getattr(abundance, "abundance", abundance)
    group_col = getattr(abundance, "group_by", None)
    if group_col is None or group_col not in df.columns:
        group_col = next((c for c in df.columns
                          if c not in ("clone_id", "p", "p_sd", "lower",
                                       "upper", "rank")), None)
    cmap_obj = plt.get_cmap(cmap)
    groups = ([("all", df)] if group_col is None
              else list(df.groupby(group_col)))
    for i, (name, sub) in enumerate(groups):
        sub = sub.sort_values("rank")
        rank = sub["rank"].to_numpy(dtype=float)
        p = sub["p"].to_numpy(dtype=float)
        color = cmap_obj(i % cmap_obj.N)
        ax.plot(rank, p, color=color, label=str(name))
        if "lower" in sub.columns and "upper" in sub.columns:
            ax.fill_between(rank, sub["lower"].to_numpy(dtype=float),
                            sub["upper"].to_numpy(dtype=float),
                            color=color, alpha=ci_alpha)
    ax.set_xscale("log")
    ax.set_xlabel("clone rank")
    ax.set_ylabel("clone abundance")
    ax.set_title(title)
    if group_col is not None:
        ax.legend(title=group_col, fontsize=8, frameon=False)
    return ax


@register_function(
    aliases=["group_box_plot", "plot_group_box", "grouped_boxplot", "分组箱线图"],
    category="airr",
    description=(
        "Grouped boxplot with a jittered scatter overlay — compares a numeric "
        "value across cell / sample groups (e.g. a diversity estimator across "
        "conditions). Takes a tidy DataFrame with one numeric value column "
        "and a categorical group column."
    ),
    examples=[
        "ov.airr.plotting.group_box_plot(div, value='chao1', group='group')",
        "ov.airr.plotting.group_box_plot(div, value='d50', order=['Healthy', 'MS'])",
    ],
    related=["airr.hill_diversity", "airr.plotting.group_abundance_plot"],
)
def group_box_plot(
    table,
    *,
    value: str,
    group: str = "group",
    order: Optional[list] = None,
    ax=None,
    figsize=(4.5, 3.5),
    title: str = "",
    palette: Optional[dict] = None,
):
    """Grouped boxplot with a jittered per-point scatter overlay.

    Draws one box per group of a tidy table and overlays every observation
    as a jittered point — the standard way to compare a per-sample metric
    (a diversity estimator, a clonality score …) across conditions.

    Parameters
    ----------
    table
        A tidy :class:`pandas.DataFrame` with a numeric ``value`` column and
        a categorical ``group`` column.
    value
        Column name of the numeric value to plot on the y-axis.
    group
        Column name of the categorical grouping variable (default
        ``'group'``).
    order
        Explicit group order on the x-axis. When ``None`` the groups are
        taken in first-appearance order.
    ax, figsize
        Standard matplotlib styling controls.
    title
        Plot title.
    palette
        Optional ``{group: color}`` mapping for the scatter points; groups
        absent from the mapping fall back to a default colour.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    if value not in table.columns:
        raise KeyError(f"column {value!r} not found in table.")
    if group not in table.columns:
        raise KeyError(f"column {group!r} not found in table.")
    ax = _ax(ax, figsize)
    if order is None:
        order = list(pd.unique(table[group]))
    grp = [
        pd.to_numeric(
            table.loc[table[group] == g, value], errors="coerce"
        ).dropna().values
        for g in order
    ]
    ax.boxplot(grp, labels=[str(g) for g in order])
    palette = palette or {}
    for i, g in enumerate(order):
        vals = grp[i]
        ax.scatter(
            np.full(len(vals), i + 1), vals,
            color=palette.get(g, "#4878CF"), s=30, zorder=3,
        )
    ax.set_ylabel(value)
    ax.set_title(title)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
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
