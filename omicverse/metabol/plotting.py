r"""Metabolomics-specific plots.

Three core plots every metabolomics paper has:

- **Volcano plot** — log2FC vs -log10(padj), metabolite labels on
  significant hits.
- **S-plot** — OPLS-DA signature plot of p(corr) vs p(cov) used to
  interpret which metabolites drive the group separation.
- **VIP bar** — top-N metabolites by VIP score.

These intentionally return ``(fig, ax)`` tuples in omicverse's
convention, so downstream code can layer additional annotations and
ship a single publication-ready figure.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._plsda import PLSDAResult

from .._registry import register_function


@register_function(
    aliases=[
        'volcano',
        'metabol_volcano',
        '火山图代谢',
    ],
    category='metabolomics',
    description='Metabolomics volcano plot — log2FC vs -log10(padj/pvalue). Supports use_pvalue=True for small-n LC-MS where BH is too strict.',
    examples=[
        'ov.metabol.volcano(deg, padj_thresh=0.10, log2fc_thresh=0.3, label_top_n=8)',
    ],
    related=[
        'metabol.differential',
    ],
)
def volcano(
    deg: pd.DataFrame,
    *,
    padj_thresh: float = 0.05,
    log2fc_thresh: float = 1.0,
    label_top_n: int = 10,
    use_pvalue: bool = False,
    clip_log2fc: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (5.5, 4.5),
):
    """Metabolomics volcano plot — log2FC vs -log10(padj) (or pvalue).

    Parameters
    ----------
    use_pvalue
        Plot against the raw ``pvalue`` column instead of the BH-adjusted
        ``padj``. Useful for small-n untargeted LC-MS where the 5000+
        feature FDR burden means no peak survives FDR; raw p-value is
        the honest axis for the volcano in that regime.
    clip_log2fc
        Clip the x-axis to ±this value. On LC-MS data with below-detection
        zeros, a handful of features can have log2fc up to ±25 after
        pseudo-count logging and completely dominate the plot. Set
        e.g. ``clip_log2fc=5`` to keep the volcano interpretable.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    x = deg["log2fc"].to_numpy(dtype=float)
    stat_col = "pvalue" if use_pvalue else "padj"
    y = -np.log10(np.clip(deg[stat_col].to_numpy(dtype=float), 1e-300, 1.0))
    is_sig = (deg[stat_col] < padj_thresh) & (deg["log2fc"].abs() >= log2fc_thresh)
    up = is_sig & (deg["log2fc"] > 0)
    down = is_sig & (deg["log2fc"] < 0)
    # Clip if requested — keeps the x-axis readable on LC-MS-zero-inflated data
    if clip_log2fc is not None:
        x = np.clip(x, -clip_log2fc, clip_log2fc)
    ax.scatter(x[~is_sig], y[~is_sig], c="#bfbfbf", s=8, alpha=0.6, label="n.s.")
    ax.scatter(x[up], y[up], c="#c0392b", s=14, alpha=0.85, label="up")
    ax.scatter(x[down], y[down], c="#2980b9", s=14, alpha=0.85, label="down")
    ax.axhline(-np.log10(padj_thresh), ls="--", c="grey", lw=0.8)
    ax.axvline(+log2fc_thresh, ls="--", c="grey", lw=0.8)
    ax.axvline(-log2fc_thresh, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("log$_2$ fold-change")
    ax.set_ylabel("-log$_{10}$ (pvalue)" if use_pvalue else "-log$_{10}$ (padj)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    # Label top hits
    if label_top_n > 0 and is_sig.any():
        top = deg[is_sig].assign(_y=y[is_sig]).nlargest(label_top_n, "_y")
        for name, row in top.iterrows():
            x_lab = row["log2fc"]
            if clip_log2fc is not None:
                x_lab = max(min(x_lab, clip_log2fc), -clip_log2fc)
            ax.text(x_lab, -np.log10(max(row[stat_col], 1e-300)),
                    str(name), fontsize=7, ha="left", va="bottom")
    return fig, ax


@register_function(
    aliases=[
        'pathway_bar',
        '通路柱状图',
    ],
    category='metabolomics',
    description='Horizontal bar of -log10(p-value) for pathway enrichment results (msea_ora / msea_gsea / lion_enrichment).',
    examples=[
        'ov.metabol.pathway_bar(ora_result, top_n=10)',
    ],
    related=[
        'metabol.pathway_dot',
        'metabol.msea_ora',
    ],
)
def pathway_bar(
    enrichment: pd.DataFrame,
    *,
    term_col: Optional[str] = None,
    score_col: str = "pvalue",
    top_n: int = 15,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (6.0, 5.0),
):
    """Horizontal bar chart of pathway enrichment p-values.

    Works for both ``msea_ora`` output (``pathway`` / ``pvalue``) and
    ``lion_enrichment`` output (``term`` / ``pvalue``); auto-detects
    the term column if ``term_col`` isn't given.

    Parameters
    ----------
    enrichment
        DataFrame returned by :func:`msea_ora`, :func:`msea_gsea`, or
        :func:`lion_enrichment`.
    term_col
        Column holding the pathway / term name. Auto-picks from
        ``pathway`` / ``term`` / ``Term`` / the index if None.
    score_col
        Column to plot. ``"pvalue"`` (default) → ``-log10(pvalue)`` bars;
        ``"padj"`` for FDR-adjusted; ``"nes"`` for GSEA normalized
        enrichment score (plotted as-is, signed).
    top_n
        Show the top-``top_n`` terms by the score.
    """
    df = enrichment.copy()
    if term_col is None:
        for candidate in ("pathway", "term", "Term"):
            if candidate in df.columns:
                term_col = candidate
                break
        else:
            df = df.reset_index().rename(columns={"index": "pathway"})
            term_col = "pathway"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if score_col in ("pvalue", "padj", "pval", "fdr"):
        df["_score"] = -np.log10(np.clip(df[score_col].astype(float), 1e-300, 1.0))
        xlabel = f"-log$_{{10}}$ ({score_col})"
    else:
        df["_score"] = df[score_col].astype(float)
        xlabel = score_col

    df = df.sort_values("_score", ascending=False).head(top_n)
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in df["_score"]]
    ax.barh(df[term_col][::-1], df["_score"][::-1], color=colors[::-1])
    ax.set_xlabel(xlabel)
    return fig, ax


@register_function(
    aliases=[
        'pathway_dot',
        'dotplot',
        '代谢通路点图',
    ],
    category='metabolomics',
    description='Dot plot of pathway enrichment (size=overlap, x=odds_ratio or NES, color=-log10 p). The canonical metabolomics enrichment figure.',
    examples=[
        "ov.metabol.pathway_dot(ora_result, size_col='overlap', x_col='odds_ratio', color_col='pvalue', top_n=10)",
    ],
    related=[
        'metabol.pathway_bar',
        'metabol.msea_ora',
        'metabol.msea_gsea',
    ],
)
def pathway_dot(
    enrichment: pd.DataFrame,
    *,
    term_col: Optional[str] = None,
    size_col: str = "overlap",
    x_col: str = "odds_ratio",
    color_col: str = "pvalue",
    top_n: int = 15,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (6.5, 5.5),
):
    """Dot plot of pathway enrichment — the de-facto standard figure.

    Matches the layout of `scanpy`'s / `clusterProfiler`'s dot plot:
    each row is a pathway, dot **size** encodes overlap count, **x**
    position encodes fold-enrichment (odds ratio or NES), **color**
    encodes -log10(p-value).

    Parameters
    ----------
    size_col
        Column mapped to dot size. Default ``"overlap"`` (ORA output);
        use ``"matched_size"`` for GSEA output.
    x_col
        Column mapped to x position. Default ``"odds_ratio"`` (ORA);
        use ``"nes"`` for GSEA.
    color_col
        Column mapped to color (via ``-log10``). Default ``"pvalue"``.
    """
    df = enrichment.copy()
    if term_col is None:
        for candidate in ("pathway", "term", "Term"):
            if candidate in df.columns:
                term_col = candidate
                break
        else:
            df = df.reset_index().rename(columns={"index": "pathway"})
            term_col = "pathway"

    for col in (size_col, x_col, color_col):
        if col not in df.columns:
            raise KeyError(
                f"column {col!r} not in enrichment DataFrame "
                f"(columns: {list(df.columns)})"
            )

    df["_neglog10"] = -np.log10(np.clip(df[color_col].astype(float), 1e-300, 1.0))
    df = df.sort_values("_neglog10", ascending=False).head(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y = np.arange(len(df))
    sizes = df[size_col].astype(float).to_numpy()
    sizes_scaled = 30 + (sizes - sizes.min()) / max(sizes.max() - sizes.min(), 1) * 200
    scatter = ax.scatter(
        df[x_col].astype(float), y, s=sizes_scaled,
        c=df["_neglog10"], cmap="viridis", edgecolor="black", linewidth=0.4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df[term_col])
    ax.invert_yaxis()
    ax.set_xlabel(x_col)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.04)
    cbar.set_label(f"-log$_{{10}}$ ({color_col})")
    # Size legend
    for s in np.percentile(sizes, [25, 75, 100]):
        ax.scatter([], [], s=30 + (s - sizes.min()) / max(sizes.max() - sizes.min(), 1) * 200,
                   c="grey", label=f"{size_col}={int(s)}")
    ax.legend(loc="center left", bbox_to_anchor=(1.35, 0.5), fontsize=8,
              frameon=False, title=size_col)
    return fig, ax


@register_function(
    aliases=[
        's_plot',
        'Wiklund_s_plot',
    ],
    category='metabolomics',
    description='OPLS-DA S-plot (Wiklund 2008) — p(cov) vs p(corr) scatter colored by VIP, the canonical OPLS-DA interpretation figure.',
    examples=[
        'ov.metabol.s_plot(opls_result, adata, label_top_n=10)',
    ],
    related=[
        'metabol.opls_da',
        'metabol.vip_bar',
    ],
)
def s_plot(
    result: PLSDAResult,
    adata,
    *,
    label_top_n: int = 15,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (5.5, 4.5),
):
    """OPLS-DA S-plot: p(cov) vs p(corr), i.e. covariance vs correlation
    between each feature and the predictive component.

    This is the classic visualization for interpreting OPLS-DA loadings
    (Wiklund et al 2008). Features in the two "arms" of the S
    (high-covariance, high-correlation) are the strongest drivers of
    the case/control separation.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    X = np.asarray(adata.X, dtype=np.float64)
    t = result.scores[:, 0]
    # p(cov)_j = cov(X_j, t) / sd(t)
    t_centered = t - t.mean()
    x_centered = X - X.mean(axis=0)
    cov = (x_centered.T @ t_centered) / (t_centered.size - 1)
    pcov = cov / (t.std(ddof=1) if t.std(ddof=1) > 0 else 1.0)
    # p(corr)_j = cov(X_j, t) / (sd(X_j) * sd(t))
    sd_x = x_centered.std(axis=0, ddof=1)
    pcorr = np.where(sd_x > 0, cov / (sd_x * (t.std(ddof=1) if t.std(ddof=1) > 0 else 1.0)), 0.0)

    ax.scatter(pcov, pcorr, c=np.abs(result.vip), cmap="viridis", s=14, alpha=0.75)
    ax.axhline(0, c="grey", lw=0.6); ax.axvline(0, c="grey", lw=0.6)
    ax.set_xlabel("p(cov) — covariance w/ predictive component")
    ax.set_ylabel("p(corr) — correlation w/ predictive component")
    # Label tallest-|pcorr| × |pcov| points
    if label_top_n > 0:
        score = np.abs(pcov) * np.abs(pcorr)
        idx = np.argsort(-score)[:label_top_n]
        for i in idx:
            ax.text(pcov[i], pcorr[i], str(adata.var_names[i]),
                    fontsize=7, ha="left", va="bottom")
    return fig, ax


@register_function(
    aliases=[
        'vip_bar',
        'VIP条形图',
    ],
    category='metabolomics',
    description='Horizontal bar chart of the top-N VIP (Variable Importance in Projection) metabolites from a PLS-DA or OPLS-DA model.',
    examples=[
        'ov.metabol.vip_bar(opls_result, adata.var_names, top_n=15)',
    ],
    related=[
        'metabol.plsda',
        'metabol.opls_da',
    ],
)
def vip_bar(
    result: PLSDAResult,
    var_names,
    *,
    top_n: int = 15,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (5.0, 5.0),
):
    """Horizontal bar chart of top-``top_n`` VIP metabolites.

    Bars sorted by VIP score (descending). Colour encodes the sign of
    the PLS-DA regression coefficient — red for upregulated in the
    positive class, blue for downregulated — so the figure simultaneously
    shows importance and direction. The dashed grey line at VIP=1 is
    Wold's classical importance cutoff (features above 1 are retained).

    Parameters
    ----------
    result : PLSDAResult
        Output of :func:`omicverse.metabol.plsda` or :func:`opls_da`.
    var_names :
        Iterable of metabolite names — typically ``adata.var_names`` —
        used to label the bars.
    top_n : int, default 15
        Number of top-VIP metabolites to plot.
    ax, figsize :
        Standard matplotlib hooks.

    Returns
    -------
    (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    tbl = result.to_vip_table(var_names).head(top_n)
    # Color by sign of the regression coefficient
    colors = ["#c0392b" if c > 0 else "#2980b9" for c in tbl["coef"]]
    ax.barh(tbl.index[::-1], tbl["vip"].iloc[::-1], color=colors[::-1])
    ax.axvline(1.0, c="grey", lw=0.6, ls="--")   # Wold VIP>1 threshold
    ax.set_xlabel("VIP score")
    ax.set_ylabel("")
    return fig, ax


# ---------------------------------------------------------------------------
# sample-QC scatter — Hotelling T² vs DModX
# ---------------------------------------------------------------------------
@register_function(
    aliases=['sample_qc_plot', 'plot_sample_qc',
             'hotelling_dmodx_scatter'],
    category='metabolomics',
    description='Scatter of Hotelling T² vs DModX with the alpha-level '
                'critical boundaries, highlighting flagged outlier samples. '
                'Pair with metabol.sample_qc.',
    examples=[
        "qc = ov.metabol.sample_qc(adata); "
        "ov.metabol.sample_qc_plot(qc)",
    ],
    related=['metabol.sample_qc'],
)
def sample_qc_plot(
    qc_df,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (5.0, 4.0),
    normal_color: str = "#2980b9",
    outlier_color: str = "#c0392b",
):
    """Scatter of Hotelling T² vs DModX with critical-value lines.

    Standard SIMCA-style outlier diagnostic. Each sample is one point;
    the dashed grey lines are the alpha-level (default 0.95) critical
    values from :func:`omicverse.metabol.sample_qc`. Samples above
    *either* line are flagged — Hotelling T² captures distance from the
    PCA centre (within-model outliers), DModX captures distance to the
    PCA hyperplane (off-model outliers).

    Parameters
    ----------
    qc_df : pd.DataFrame
        Output of :func:`omicverse.metabol.sample_qc` — must contain
        ``T2``, ``DModX``, ``T2_crit``, ``DModX_crit`` and ``is_outlier``.
    normal_color, outlier_color :
        Hex / named colours for the two populations.
    ax, figsize :
        Standard matplotlib hooks.

    Returns
    -------
    (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    flag = qc_df["is_outlier"].to_numpy()
    ax.scatter(qc_df["T2"][~flag], qc_df["DModX"][~flag],
               c=normal_color, s=28, edgecolor="k", alpha=0.7,
               label="normal")
    ax.scatter(qc_df["T2"][flag], qc_df["DModX"][flag],
               c=outlier_color, s=55, edgecolor="k", label="outlier")
    ax.axvline(qc_df["T2_crit"].iloc[0], color="grey", ls="--", lw=1)
    ax.axhline(qc_df["DModX_crit"].iloc[0], color="grey", ls="--", lw=1)
    ax.set_xlabel("Hotelling T²")
    ax.set_ylabel("DModX")
    ax.legend(loc="best", frameon=False)
    return fig, ax


# ---------------------------------------------------------------------------
# DGCA class counts
# ---------------------------------------------------------------------------
@register_function(
    aliases=['dgca_class_bar', 'plot_dgca_classes', 'dc_class_counts'],
    category='metabolomics',
    description='Bar chart of the DGCA class distribution '
                '(counts of +/+, +/0, +/-, -/+, ...) returned by '
                'metabol.dgca.',
    examples=[
        "dc = ov.metabol.dgca(adata, group_col='group'); "
        "ov.metabol.dgca_class_bar(dc)",
    ],
    related=['metabol.dgca', 'metabol.corr_network_plot'],
)
def dgca_class_bar(
    dc_df,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (6.0, 3.5),
    log: bool = True,
):
    """Bar chart of DC-class counts (+/+, +/0, +/-, -/+, ...).

    Summarises the rewiring profile from :func:`omicverse.metabol.dgca`.
    Each pair of metabolites is assigned a class encoding the sign of
    their correlation in group A and group B (`+`, `-`, `0` for
    significant positive, significant negative, or non-significant).
    Symmetric reversals (``+/-`` and ``-/+``) get the same red colour
    because they're the strongest rewiring signal; ``+/+`` / ``-/-``
    are concordant (no rewiring); ``0/0`` (no signal in either group)
    is greyed out and pushed to the right.

    Parameters
    ----------
    dc_df : pd.DataFrame
        Output of :func:`omicverse.metabol.dgca` — needs the ``dc_class``
        column.
    log : bool, default True
        Log-scale the y-axis (counts often span orders of magnitude).
    ax, figsize :
        Standard matplotlib hooks.

    Returns
    -------
    (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    counts = dc_df["dc_class"].value_counts()
    order = [c for c in counts.index if c != "0/0"] + (
        ["0/0"] if "0/0" in counts.index else [])
    counts = counts.reindex(order)
    palette = {
        "+/+": "#27ae60", "-/-": "#8e44ad",
        "+/-": "#c0392b", "-/+": "#c0392b",
        "+/0": "#2980b9", "0/+": "#3498db",
        "-/0": "#e67e22", "0/-": "#f39c12",
        "0/0": "#bdc3c7",
    }
    colors = [palette.get(c, "#7f8c8d") for c in counts.index]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="k")
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("number of pairs")
    ax.set_xlabel("DC class (A=group_a / B=group_b)")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)
    return fig, ax


# ---------------------------------------------------------------------------
# Correlation network plot
# ---------------------------------------------------------------------------
@register_function(
    aliases=['corr_network_plot', 'plot_correlation_network',
             '相关网络图'],
    category='metabolomics',
    description='Draw a metabolite-correlation edge DataFrame (from '
                'metabol.corr_network or a filtered metabol.dgca) as a '
                'NetworkX spring-layout plot. Edge color encodes sign of '
                'r; width scales with |r|.',
    examples=[
        "edges = ov.metabol.corr_network(adata, group_col='group', "
        "group='case'); ov.metabol.corr_network_plot(edges)",
    ],
    related=['metabol.corr_network', 'metabol.dgca'],
)
def corr_network_plot(
    edges_df,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (6.5, 5.5),
    layout: str = "spring",
    node_size: int = 70,
    node_color: str = "white",
    node_edge_color: str = "#34495e",
    edge_width_scale: float = 2.5,
    r_column: str = "r",
    seed: int = 0,
    with_labels: bool = True,
    label_font_size: int = 7,
):
    """Draw an edge DataFrame as a NetworkX spring-layout plot.

    Renders the output of :func:`omicverse.metabol.corr_network` as a
    correlation graph: nodes are metabolites, edges are pairs whose
    Spearman / Pearson ``|r|`` exceeded the network's threshold. Edge
    colour encodes the sign of the correlation (red = positive,
    blue = negative); edge width is proportional to ``|r|``.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Output of :func:`corr_network` — needs columns ``feature_a``,
        ``feature_b`` and ``r`` (or whatever ``r_column`` points at).
    layout : {'spring', 'circular', 'kamada_kawai'}, default 'spring'
        NetworkX layout algorithm.
    node_size, node_color, node_edge_color :
        Standard styling. White-fill / dark-edge nodes read better in
        publications than filled ones because labels overlay legibly.
    edge_width_scale : float
        Multiplier on ``|r|`` for edge width (so ``|r|=1`` is the
        thickest edge in the figure).
    r_column : str, default 'r'
        Column to read for edge correlations. Switch to ``'r_a'`` /
        ``'r_b'`` if you're plotting one half of a DGCA result.
    with_labels : bool, default True
        Toggle metabolite labels.
    label_font_size : int, default 7
        Label size — reduce for dense graphs.
    seed : int
        RNG seed for the spring-layout (only matters when ``layout='spring'``).

    Returns
    -------
    (fig, ax) tuple. If ``edges_df`` is empty, draws a centred
    "no edges" placeholder and returns the empty axes.
    """
    import networkx as nx

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if edges_df.empty:
        ax.text(0.5, 0.5, "no edges above the threshold",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax
    G = nx.from_pandas_edgelist(
        edges_df, source="feature_a", target="feature_b",
        edge_attr=True,
    )
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed, k=0.7)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)

    edge_colors = []
    edge_widths = []
    for u, v, data in G.edges(data=True):
        r = float(data.get(r_column, 0.0))
        edge_colors.append("#c0392b" if r > 0 else "#2980b9")
        edge_widths.append(max(abs(r), 0.1) * edge_width_scale)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                           width=edge_widths, alpha=0.75, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_size,
                           node_color=node_color,
                           edgecolors=node_edge_color, ax=ax)
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=label_font_size, ax=ax)
    ax.set_axis_off()
    return fig, ax


# ---------------------------------------------------------------------------
# ASCA variance-explained bar
# ---------------------------------------------------------------------------
@register_function(
    aliases=['asca_variance_bar', 'plot_asca_effects', 'ASCA方差'],
    category='metabolomics',
    description='Horizontal bar chart of per-effect variance-explained '
                'fractions from an ASCAResult, plus the residual.',
    examples=[
        "res = ov.metabol.asca(adata, factors=['treatment', 'time']); "
        "ov.metabol.asca_variance_bar(res)",
    ],
    related=['metabol.asca'],
)
def asca_variance_bar(
    asca_result,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (5.5, 3.0),
):
    """Horizontal bars of per-effect variance-explained fractions.

    Visualises the partition of total sum-of-squares from
    :func:`omicverse.metabol.asca` across the named effects (factors
    and their interactions) plus the residual. The grey "residual"
    bar at the bottom shows what the model didn't explain — a useful
    sanity check: if residual variance dominates, the planned factors
    don't capture the dominant signal and the model needs revision.

    Parameters
    ----------
    asca_result : ASCAResult
        Output of :func:`asca`. Reads ``.effects[name].variance_explained``
        for each named effect plus ``.residual_ss`` / ``.total_ss``.
    ax, figsize :
        Standard matplotlib hooks.

    Returns
    -------
    (fig, ax) tuple. Bar labels show variance percentage with one
    decimal place.
    """
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    names = list(asca_result.effects.keys())
    fracs = [asca_result.effects[n].variance_explained for n in names]
    residual_frac = (
        asca_result.residual_ss / asca_result.total_ss
        if asca_result.total_ss > 0 else 0.0
    )
    names.append("residual")
    fracs.append(residual_frac)
    colors = ["#2980b9"] * (len(names) - 1) + ["#bdc3c7"]
    y = np.arange(len(names))
    ax.barh(y, fracs, color=colors, edgecolor="k")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("variance explained (fraction of total SS)")
    for i, f in enumerate(fracs):
        ax.text(f + 0.005, i, f"{f*100:.1f}%",
                va="center", ha="left", fontsize=8)
    return fig, ax


@register_function(
    aliases=['acyl_chain_map', 'lipid_map', 'lipidomeR_map', '酰基链图'],
    category='metabolomics',
    description=(
        "lipidomeR-style acyl-chain map — per lipid class, a heatmap of "
        "total chain length (x) vs total double bonds (y) coloured by a "
        "per-lipid statistic (e.g. log2 fold-change). Reveals which "
        "chain-length / unsaturation regions shift between conditions."
    ),
    examples=[
        "ov.metabol.acyl_chain_map(de_table, value_col='log2FC')",
    ],
    related=['metabol.parse_lipid', 'metabol.annotate_lipids'],
)
def acyl_chain_map(
    data,
    *,
    value_col: str = "log2FC",
    name_col: Optional[str] = None,
    classes: Optional[list] = None,
    n_cols: int = 4,
    cmap: str = "RdBu_r",
    vmax: Optional[float] = None,
    figsize: Optional[tuple] = None,
):
    """lipidomeR-style total-carbon × double-bond map, faceted by lipid class.

    Parameters
    ----------
    data
        A DataFrame with one row per lipid. It must carry a lipid-name
        column (the index, or ``name_col``) and a ``value_col`` numeric
        statistic. Lipid names are parsed with :func:`parse_lipid`.
    value_col
        Column to colour each (carbon, double-bond) cell by — typically
        ``log2FC`` from a differential test, but any per-lipid number
        works.
    name_col
        Column holding the lipid name. ``None`` (default) uses the
        DataFrame index.
    classes
        Restrict to these lipid classes; ``None`` plots every class
        present.
    n_cols
        Facet-grid column count.
    cmap, vmax
        Colour map and symmetric colour limit (auto if ``None``).

    Returns
    -------
    (fig, axes)
    """
    import pandas as pd
    from ._lipidomics import parse_lipid

    df = data.copy()
    names = df[name_col].astype(str) if name_col else df.index.astype(str)
    parsed = [parse_lipid(n) for n in names]
    rows = []
    for n, p, v in zip(names, parsed, df[value_col].to_numpy()):
        if p is None:
            continue
        rows.append((p.lipid_class, p.total_carbons, p.total_db, float(v)))
    if not rows:
        raise ValueError("no lipid names could be parsed from the input")
    lipdf = pd.DataFrame(rows, columns=["lipid_class", "C", "DB", "value"])

    present = (classes if classes is not None
               else sorted(lipdf["lipid_class"].unique()))
    present = [c for c in present if (lipdf["lipid_class"] == c).any()]
    n = len(present)
    ncol = min(n_cols, n)
    nrow = int(np.ceil(n / ncol))
    if figsize is None:
        figsize = (3.2 * ncol, 2.8 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(lipdf["value"]), 98)) or 1.0

    im = None
    for idx, cls in enumerate(present):
        ax = axes[idx // ncol][idx % ncol]
        sub = lipdf[lipdf["lipid_class"] == cls]
        # mean value per (C, DB) cell
        grid = sub.groupby(["DB", "C"])["value"].mean().reset_index()
        c_vals = sorted(sub["C"].unique())
        db_vals = sorted(sub["DB"].unique())
        M = np.full((len(db_vals), len(c_vals)), np.nan)
        ci = {c: i for i, c in enumerate(c_vals)}
        di = {d: i for i, d in enumerate(db_vals)}
        for _, r in grid.iterrows():
            M[di[r["DB"]], ci[r["C"]]] = r["value"]
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                       origin="lower", interpolation="nearest")
        ax.set_xticks(range(len(c_vals)))
        ax.set_xticklabels(c_vals, fontsize=6, rotation=90)
        ax.set_yticks(range(len(db_vals)))
        ax.set_yticklabels(db_vals, fontsize=6)
        ax.set_title(f"{cls}  (n={len(sub)})", fontsize=9)
        ax.set_xlabel("total carbons", fontsize=7)
        ax.set_ylabel("double bonds", fontsize=7)
    # blank unused facets
    for idx in range(n, nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, label=value_col, fraction=0.025, pad=0.02)
    return fig, axes
