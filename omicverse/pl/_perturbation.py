r"""Plotting helpers for in-silico gene-perturbation results.

Targets the dictionary returned by :meth:`omicverse.llm.SCLLMManager.perturb_genes`
(see also :meth:`omicverse.llm.geneformer_model.GeneformerModel.perturb_genes`),
which has the schema:

    {
        "cosine_similarities": DataFrame (cell × gene),
        "stats":                DataFrame (gene × {mean, std, n_cells_perturbed, ...}),
        "original_embeddings":  ndarray (n_cells × d),
        "perturbed_embeddings": dict[gene → ndarray (n_cells × d)],
        ...
    }

Two helpers:

* :func:`perturbation_shift_violin` — per-gene violin of per-cell
  cosine-similarity drops. Lower bars = stronger perturbation effect.
* :func:`perturbation_embedding_shift` — UMAP / PCA scatter overlaid with
  arrows from each cell's original embedding position to its post-perturbation
  position, projected onto an existing 2-D basis (``adata.obsm[basis]``).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._registry import register_function


# --------------------------------------------------------------------------- #
# violin: per-gene cosine-similarity distribution                              #
# --------------------------------------------------------------------------- #


@register_function(
    aliases=["扰动小提琴图", "perturbation_shift_violin"],
    category="pl",
    description=(
        "Violin plot of per-cell cosine similarity (original ↔ perturbed) "
        "for each target gene returned by ov.llm.SCLLMManager.perturb_genes. "
        "Lower values = bigger embedding shift = stronger perturbation effect."
    ),
    examples=[
        "result = manager.perturb_genes(adata, ['CD3D', 'PTPRC'], perturb_type='delete')",
        "ov.pl.perturbation_shift_violin(result)",
    ],
    related=["llm.SCLLMManager.perturb_genes", "pl.perturbation_embedding_shift"],
)
def perturbation_shift_violin(
    result: Dict,
    *,
    adata: Optional[AnnData] = None,
    groupby: Optional[str] = None,
    figsize: tuple[float, float] = (6.0, 3.5),
    color: str = "#5B8FF9",
    order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
):
    r"""Violin of per-cell cosine similarity for each perturbed gene.

    Parameters
    ----------
    result : dict
        The dictionary returned by ``manager.perturb_genes(...)``. Must
        contain ``'cosine_similarities'`` — a ``DataFrame`` with one column
        per perturbed gene (cells × genes).
    adata, groupby : AnnData and str
        Optional — when both are passed and ``groupby`` is a column in
        ``adata.obs``, the violin for each gene is split into one violin
        per category (e.g. ``groupby='cell_type'`` reveals lineage-
        specific perturbation effects).
    figsize : tuple
        Figure size in inches.
    color : str
        Default violin fill colour when ``groupby`` is not provided.
    order : sequence of str or None
        Gene plot order. ``None`` = sort by mean cosine ascending (strongest
        effect leftmost).
    title : str or None
        Optional title.
    ax : matplotlib Axes or None
        Pre-allocated axes; ``None`` creates a new figure.

    Returns
    -------
    fig, ax
    """
    cos = result.get("cosine_similarities")
    if cos is None or not hasattr(cos, "columns"):
        raise KeyError(
            "result['cosine_similarities'] is missing or not a DataFrame — "
            "this helper expects ov.llm.SCLLMManager.perturb_genes() output."
        )
    df = pd.DataFrame(cos).dropna(how="all")
    if df.empty:
        raise ValueError("All per-cell cosine similarities are NaN — no cell carried any of the requested genes.")

    if order is None:
        order = df.mean(axis=0, skipna=True).sort_values().index.tolist()
    df = df[order]
    # Drop genes that ended up entirely NaN (no cell carried the gene — typical
    # for narrowly-expressed TFs when max_ncells is small).
    nonempty = [c for c in df.columns if df[c].notna().any()]
    skipped = [c for c in df.columns if c not in nonempty]
    if skipped:
        import warnings as _warnings
        _warnings.warn(
            f"Skipping genes with no perturbed cells: {skipped} "
            f"(increase max_ncells or pick more broadly-expressed targets)."
        )
    df = df[nonempty]
    order = nonempty
    if df.empty:
        raise ValueError("No genes have any perturbed cells to plot.")

    # If a groupby is requested, render one violin per category per gene.
    if groupby is not None and adata is not None:
        if groupby not in adata.obs.columns:
            raise KeyError(f"adata.obs[{groupby!r}] not found")
        from ._palette import palette_28
        group = adata.obs[groupby].astype(str)
        # Align to the cells scored by perturb_genes (which used obs_names index)
        cell_idx = result.get("cell_indices")
        if cell_idx is not None:
            group = group.iloc[cell_idx].reset_index(drop=True)
            df_aligned = df.reset_index(drop=True)
        else:
            df_aligned = df.copy()
            group = group.loc[df_aligned.index]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        cats = sorted(group.unique())
        pal = list(palette_28)
        cat_to_color = {c: pal[i % len(pal)] for i, c in enumerate(cats)}
        n_g = len(order)
        n_c = len(cats)
        width = 0.8 / max(n_c, 1)
        # Cap each gene's group set to top-K most-perturbed categories (by mean
        # |1-cos| over that gene), so the plot stays readable with many categories.
        max_groups_per_gene = 6
        for gi, g in enumerate(order):
            shifts = (1 - df_aligned[g]).abs()
            by_cat = shifts.groupby(group).mean().dropna()
            top_cats = by_cat.sort_values(ascending=False).head(max_groups_per_gene).index.tolist()
            for ci, c in enumerate(top_cats):
                vals = df_aligned[g][(group == c) & df_aligned[g].notna()].values
                if vals.size < 2:
                    continue
                pos = gi + (ci - (len(top_cats) - 1) / 2) * width
                parts = ax.violinplot(
                    [vals], positions=[pos], widths=width * 0.9,
                    showmeans=True, showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(cat_to_color[c])
                    body.set_edgecolor("#333333")
                    body.set_alpha(0.75)
                if "cmeans" in parts:
                    parts["cmeans"].set_color("#222222")
            # legend (once)
            if gi == 0:
                for c in top_cats:
                    ax.scatter([], [], s=30, c=cat_to_color[c], label=str(c))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_xlabel("Perturbed gene")
        ax.set_ylabel("Cosine similarity\n(original ↔ perturbed)")
        ax.axhline(1.0, color="#888888", linewidth=0.5, linestyle="--")
        for spine_name in ("top", "right"):
            ax.spines[spine_name].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(
                loc="center left", bbox_to_anchor=(1.02, 0.5),
                fontsize=7, frameon=False, title=groupby,
            )
        if title:
            ax.set_title(title, fontsize=11)
        elif result.get("perturb_type"):
            ax.set_title(
                f"Geneformer in-silico {result['perturb_type']} — by {groupby}",
                fontsize=11,
            )
        return fig, ax

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    data_per_gene = [df[c].dropna().values for c in order]
    parts = ax.violinplot(data_per_gene, showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.7)
    if "cmeans" in parts:
        parts["cmeans"].set_color("#222222")

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel("Cosine similarity\n(original ↔ perturbed)")
    ax.set_xlabel("Perturbed gene")
    ax.axhline(1.0, color="#888888", linewidth=0.5, linestyle="--")
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11)
    elif result.get("perturb_type"):
        ax.set_title(
            f"Geneformer in-silico {result['perturb_type']} — per-cell cosine similarity",
            fontsize=11,
        )

    return fig, ax


# --------------------------------------------------------------------------- #
# downstream genes: bar plot of top-shifted genes                              #
# --------------------------------------------------------------------------- #


@register_function(
    aliases=["扰动下游基因", "perturbation_top_downstream_genes", "perturbation_top_genes"],
    category="pl",
    description=(
        "Bar plot of the genes whose contextual embedding shifted most after "
        "in-silico knockout of a target transcription factor — i.e. candidate "
        "downstream targets ranked by 1 - cos(orig_emb, perturbed_emb)."
    ),
    examples=[
        "result = manager.perturb_genes(adata, ['PAX5'], perturb_type='delete')",
        "ov.pl.perturbation_top_downstream_genes(result, gene='PAX5', top_n=20)",
    ],
    related=["llm.SCLLMManager.perturb_genes", "pl.perturbation_shift_violin"],
)
def perturbation_top_downstream_genes(
    result: Dict,
    *,
    gene: str,
    top_n: int = 20,
    figsize: tuple[float, float] = (6.5, 5.0),
    bar_color: str = "#cd3a3a",
    min_cells: int = 5,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
):
    r"""Plot the top-N most-shifted downstream genes after a TF knockout.

    Requires ``manager.perturb_genes(..., compute_gene_shifts=True)`` (the
    default). The shift metric per downstream gene is
    ``mean_cells(1 - cos(emb_orig, emb_perturbed))`` — averaged over the
    cells where both the target and the downstream gene were present in
    the rank-encoded input. Higher = larger context shift = stronger
    model-predicted downstream effect.

    Parameters
    ----------
    result : dict
        Output of ``manager.perturb_genes(...)``. Must contain
        ``'gene_shifts'`` — a dict keyed by target-gene label.
    gene : str
        Which target gene's perturbation to summarise.
    top_n : int
        Number of downstream genes to display.
    figsize, bar_color, title, ax
        Standard matplotlib styling.
    min_cells : int
        Minimum number of cells in which a downstream gene must be present
        for it to enter the ranking. Filters out noise from rare genes.

    Returns
    -------
    fig, ax
    """
    shifts_by_target = result.get("gene_shifts") or {}
    if gene not in shifts_by_target:
        raise KeyError(
            f"result['gene_shifts'] has no entry for {gene!r} (keys: "
            f"{list(shifts_by_target.keys())}). Re-run perturb_genes with "
            f"compute_gene_shifts=True (default)."
        )
    df = pd.DataFrame(shifts_by_target[gene])
    if df.empty:
        raise ValueError(f"No downstream-gene shifts recorded for {gene!r}.")
    df = df[df["n_cells"] >= min_cells].sort_values("mean_shift", ascending=False).head(top_n)
    if df.empty:
        raise ValueError(f"No downstream genes pass min_cells={min_cells} filter.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y_pos = np.arange(len(df))[::-1]  # tallest bar at the top
    labels = df["gene"].astype(str).tolist()
    ax.barh(y_pos, df["mean_shift"].values, color=bar_color, alpha=0.8, edgecolor="#333333")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean 1 − cos(orig, perturbed)\n(higher = more shifted)")
    ax.set_ylabel("Downstream gene")
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    ax.set_title(title or f"Top {len(df)} downstream genes shifted by {result.get('perturb_type', 'perturbation')} of {gene}", fontsize=11)
    return fig, ax


# --------------------------------------------------------------------------- #
# embedding shift: arrows on a 2-D basis                                       #
# --------------------------------------------------------------------------- #


@register_function(
    aliases=["扰动嵌入位移", "perturbation_embedding_shift"],
    category="pl",
    description=(
        "Project each cell's original and perturbed Geneformer embedding onto "
        "an existing 2-D basis (e.g. adata.obsm['X_umap']) and draw an arrow "
        "from the original to the perturbed position. Shows the *direction* of "
        "cell-state shift in response to the in-silico knockout."
    ),
    examples=[
        "ov.pl.perturbation_embedding_shift(adata, result, gene='CD3D', basis='X_umap')",
    ],
    related=["llm.SCLLMManager.perturb_genes", "pl.perturbation_shift_violin"],
)
def perturbation_embedding_shift(
    adata: AnnData,
    result: Dict,
    *,
    gene: str,
    basis: str = "X_umap",
    color: Optional[str] = None,
    figsize: tuple[float, float] = (5.0, 5.0),
    arrow_color: str = "#cd3a3a",
    arrow_alpha: float = 0.5,
    arrow_lw: float = 0.6,
    point_size: float = 6.0,
    max_arrows: int = 300,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
):
    r"""Plot per-cell embedding shift arrows on an existing 2-D basis.

    Projects the (n_cells × d) Geneformer original / perturbed embeddings onto
    ``adata.obsm[basis]`` by least-squares (one linear map fit on the original
    pair so both endpoints share the same projection). Then draws an arrow
    from each cell's original position to its post-perturbation position.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obsm[basis]`` (e.g. precomputed UMAP).
    result : dict
        Output of ``manager.perturb_genes(...)`` — needs both
        ``original_embeddings`` and ``perturbed_embeddings[gene]``.
    gene : str
        Which target gene's perturbation to visualise (key in
        ``result['perturbed_embeddings']``).
    basis : str
        ``adata.obsm`` key for the 2-D layout. Default ``'X_umap'``.
    color : str or None
        Optional ``adata.obs`` column to colour the scatter background.
    figsize : tuple
        Figure size in inches.
    arrow_color / arrow_alpha / arrow_lw : float
        Arrow styling.
    point_size : float
        Scatter marker size for the cell positions.
    max_arrows : int
        Cap on number of arrows drawn (random subsample to keep plot readable).
    ax : matplotlib Axes or None
        Pre-allocated axes.
    title : str or None
        Optional title.

    Returns
    -------
    fig, ax
    """
    if basis not in adata.obsm:
        raise KeyError(f"adata.obsm[{basis!r}] not found")
    perturbed = result.get("perturbed_embeddings", {})
    if gene not in perturbed:
        raise KeyError(f"result['perturbed_embeddings'] has no {gene!r} (keys: {list(perturbed.keys())})")
    orig = np.asarray(result["original_embeddings"])
    pert = np.asarray(perturbed[gene])
    cell_idx = np.asarray(result.get("cell_indices", np.arange(orig.shape[0])))

    layout = np.asarray(adata.obsm[basis])
    if layout.shape[1] < 2:
        raise ValueError(f"adata.obsm[{basis!r}] must be ≥ 2-D")
    Y = layout[cell_idx, :2]

    # Solve X · W = Y on original embeddings; reuse W to project perturbed.
    # Centre both sides so the projection has no bias term.
    Xc = orig - orig.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    W, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
    proj_orig = (orig - orig.mean(axis=0)) @ W + Y.mean(axis=0)
    proj_pert = (pert - orig.mean(axis=0)) @ W + Y.mean(axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Background scatter — either uniform grey or coloured by adata.obs[color].
    if color is not None and color in adata.obs:
        from ._palette import palette_28

        cats = pd.Categorical(adata.obs[color].iloc[cell_idx])
        pal = adata.uns.get(f"{color}_colors")
        if pal is None or len(pal) < len(cats.categories):
            pal = list(palette_28)
        c_to_color = {c: pal[i % len(pal)] for i, c in enumerate(cats.categories)}
        colors = [c_to_color.get(c, "#bdbdbd") for c in cats]
        ax.scatter(Y[:, 0], Y[:, 1], s=point_size, c=colors, alpha=0.6, linewidths=0)
        # legend
        for c, col in c_to_color.items():
            ax.scatter([], [], s=20, c=col, label=str(c))
        ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False
        )
    else:
        ax.scatter(Y[:, 0], Y[:, 1], s=point_size, c="#bdbdbd", alpha=0.6, linewidths=0)

    # Arrows — sub-sample if too many.
    rng = np.random.default_rng(0)
    n = proj_orig.shape[0]
    if n > max_arrows:
        sel = rng.choice(n, max_arrows, replace=False)
    else:
        sel = np.arange(n)
    # Only draw arrows for cells that actually had the gene (cosine != NaN).
    if "cosine_similarities" in result and gene in result["cosine_similarities"].columns:
        cs = np.asarray(result["cosine_similarities"][gene].values)
        sel = sel[~np.isnan(cs[sel])]

    dx = proj_pert[sel, 0] - proj_orig[sel, 0]
    dy = proj_pert[sel, 1] - proj_orig[sel, 1]
    ax.quiver(
        proj_orig[sel, 0], proj_orig[sel, 1], dx, dy,
        angles="xy", scale_units="xy", scale=1,
        color=arrow_color, alpha=arrow_alpha, width=0.003,
        linewidth=arrow_lw, headwidth=4, headlength=5,
    )

    ax.set_xlabel(f"{basis} 1")
    ax.set_ylabel(f"{basis} 2")
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    ax.set_title(title or f"In-silico {result.get('perturb_type', 'perturbation')} of {gene}", fontsize=11)
    return fig, ax


# --------------------------------------------------------------------------- #
# Tier B helpers for ov.single.perturb (sctenifoldknk / cell_oracle backends) #
# --------------------------------------------------------------------------- #


@register_function(
    aliases=["perturb_quiver", "扰动箭头图", "perturbation_quiver"],
    category="pl",
    description=(
        "Aggregated quiver plot of per-cell perturbation arrows on a 2-D "
        "embedding (UMAP / FA). Reproduces CellOracle's "
        "`plot_simulation_flow_on_grid` for both sctenifoldknk + cell_oracle "
        "backends — feed it a PerturbResult and an AnnData."
    ),
)
def perturb_quiver(
    adata,
    result,
    *,
    embedding_name: str = "X_umap",
    grid_size: int = 25,
    min_mass: float = 1.0,
    color: str = "k",
    cluster_col: Optional[str] = None,
    cluster_palette: Optional[Dict[str, str]] = None,
    background_size: float = 18.0,
    background_alpha: float = 0.55,
    arrow_target_length: float = 0.022,
    arrow_width: float = 0.006,
    arrow_headwidth: float = 5.0,
    figsize=(7, 6),
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
):
    """CellOracle-style aggregated arrow plot of the perturbation flow.

    The per-cell ΔX is converted to a 2-D embedding shift, then
    aggregated onto a regular ``grid_size × grid_size`` grid by
    Gaussian-weighted averaging. Grid cells with mass below
    ``min_mass`` are dropped.

    ``arrow_target_length`` is the desired length of the median arrow
    expressed as a fraction of the embedding bounding-box diagonal — the
    arrow magnitudes are scaled so the *median* arrow has this length,
    which makes the picture readable regardless of the raw ΔX scale.
    """
    delta_emb = result.delta_embedding(adata=adata, embedding_name=embedding_name)
    emb = np.asarray(adata.obsm[embedding_name])
    xrange = emb[:, 0].max() - emb[:, 0].min()
    yrange = emb[:, 1].max() - emb[:, 1].min()
    bbox_diag = float(np.hypot(xrange, yrange))

    # Build grid + Gaussian-weighted aggregation
    xs = np.linspace(emb[:, 0].min(), emb[:, 0].max(), grid_size)
    ys = np.linspace(emb[:, 1].min(), emb[:, 1].max(), grid_size)
    GX, GY = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    from scipy.spatial import cKDTree
    tree = cKDTree(emb)
    radius = float(np.linalg.norm([xs[1] - xs[0], ys[1] - ys[0]]))
    UV = np.zeros((grid_pts.shape[0], 2))
    mass = np.zeros(grid_pts.shape[0])
    for g_i, p in enumerate(grid_pts):
        ix = tree.query_ball_point(p, r=radius * 1.5)
        if not ix:
            continue
        d = np.linalg.norm(emb[ix] - p, axis=1)
        w = np.exp(-(d ** 2) / (2 * (radius / 2) ** 2))
        mass[g_i] = w.sum()
        if mass[g_i] > 0:
            UV[g_i] = (delta_emb[ix] * w[:, None]).sum(axis=0) / mass[g_i]
    keep = mass >= min_mass

    # Auto-rescale arrows using CellOracle's formula: 90th-percentile arrow
    # norm divided by ``plot_diag * arrow_target_length`` — arrows stay small
    # and reproducible, outliers don't blow up the picture. ``scale`` here is
    # matplotlib quiver's "data units per plot unit", so SMALLER scale = bigger
    # arrows.
    arrow_norms = np.linalg.norm(UV[keep], axis=1) if keep.any() else np.array([1.0])
    arrows_scale = float(np.percentile(arrow_norms[arrow_norms > 0], 90)) \
        if (arrow_norms > 0).any() else 1.0
    scale = arrows_scale / max(bbox_diag * arrow_target_length, 1e-9)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Coloured cell background (per cluster if available) — otherwise lightgray
    if cluster_col is not None and cluster_col in adata.obs:
        labels = pd.Categorical(adata.obs[cluster_col])
        cats = list(labels.categories)
        cluster_palette = _resolve_cluster_palette(adata, cluster_col, cats, cluster_palette)
        c_arr = [cluster_palette[c] for c in labels.astype(str)]
        ax.scatter(emb[:, 0], emb[:, 1], s=background_size, c=c_arr,
                   alpha=background_alpha, linewidths=0, rasterized=True)
        # legend
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(color=cluster_palette[c], label=str(c)) for c in cats],
            loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8,
        )
    else:
        ax.scatter(emb[:, 0], emb[:, 1], s=background_size, c="lightgray",
                   alpha=background_alpha, linewidths=0, rasterized=True)

    if keep.any():
        ax.quiver(
            grid_pts[keep, 0], grid_pts[keep, 1],
            UV[keep, 0], UV[keep, 1],
            angles="xy", scale_units="xy", scale=scale,
            color=color, width=arrow_width,
            headwidth=arrow_headwidth, headlength=arrow_headwidth + 1.5,
            alpha=0.9, zorder=3,
        )
    ax.set_xlabel(f"{embedding_name} 1")
    ax.set_ylabel(f"{embedding_name} 2")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title or f"Perturbation flow ({result.target} {result.mode}, {result.backend})",
                 fontsize=11)
    fig.tight_layout()
    return fig, ax


@register_function(
    aliases=["perturb_sankey", "扰动桑基图", "perturbation_sankey"],
    category="pl",
    description=(
        "Cluster-to-cluster transition Sankey for ov.single.perturb. Calls "
        "result.cluster_transitions(...) and renders the aggregated flow as "
        "a Sankey diagram (matplotlib-only, no plotly dependency)."
    ),
)
def perturb_sankey(
    result,
    adata,
    *,
    cluster_col: str = "leiden",
    min_flow: float = 0.02,
    palette: Optional[Dict[str, str]] = None,
    figsize=(8, 5),
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
):
    """Two-column Sankey of cluster→cluster post-perturbation flow.

    Draws each cluster as a vertical bar on left (source) and right
    (destination); ribbons connect source→target with width
    proportional to transition probability. Off-diagonal flows below
    ``min_flow`` are dropped.
    """
    ct = result.cluster_transitions(adata=adata, cluster_col=cluster_col)
    cats = list(ct.index)
    n = len(cats)
    if palette is None:
        cmap = plt.get_cmap("tab20")
        palette = {c: cmap(i % 20) for i, c in enumerate(cats)}

    # Equal-height source bars so small clusters (e.g. Mk) stay visible.
    # Each row of `ct` is already row-stochastic (out-distribution).
    src_sizes = np.full(n, 1.0 / n, dtype=float)
    # Destination heights = total inflow probability mass per destination.
    right_in = ct.sum(axis=0).to_numpy()
    right_in = right_in / right_in.sum()
    right_sizes = right_in

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y_pad = 0.005
    left_tops = np.cumsum(src_sizes + y_pad)
    left_bottoms = left_tops - src_sizes
    right_tops = np.cumsum(right_sizes + y_pad)
    right_bottoms = right_tops - right_sizes

    for i, c in enumerate(cats):
        ax.barh(left_bottoms[i] + src_sizes[i] / 2, height=src_sizes[i] - y_pad / 2,
                width=0.04, left=0.0,
                color=palette[c], edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(-0.01, left_bottoms[i] + src_sizes[i] / 2, str(c),
                ha="right", va="center", fontsize=9)

    x_right = 1.0
    for j, c in enumerate(cats):
        ax.barh(right_bottoms[j] + right_sizes[j] / 2, height=right_sizes[j] - y_pad / 2,
                width=0.04, left=x_right - 0.04,
                color=palette[c], edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(x_right + 0.01, right_bottoms[j] + right_sizes[j] / 2, str(c),
                ha="left", va="center", fontsize=9)

    # Each source bar splits proportionally by row probability.
    src_y_offsets = np.copy(left_bottoms)
    dst_y_offsets = np.copy(right_bottoms)
    for i, src in enumerate(cats):
        # absolute height occupied by source bar
        for j, dst in enumerate(cats):
            p = float(ct.iloc[i, j])
            if p < min_flow:
                continue
            src_flow = p * src_sizes[i]
            # destination flow occupies in_p[i,j] / right_in[j] fraction
            dst_flow = ct.iloc[i, j] / right_in[j] * right_sizes[j] if right_in[j] > 0 else src_flow
            y0_src = src_y_offsets[i]
            y0_dst = dst_y_offsets[j]
            verts = _make_sankey_ribbon(
                x0=0.04, x1=x_right - 0.04,
                y0_src=y0_src, y1_src=y0_src + src_flow,
                y0_dst=y0_dst, y1_dst=y0_dst + dst_flow,
            )
            # Highlight off-diagonal flows (the perturbation signal)
            alpha = 0.85 if i != j else 0.22
            ax.fill(verts[:, 0], verts[:, 1], color=palette[src],
                    alpha=alpha, edgecolor="none", zorder=2)
            src_y_offsets[i] += src_flow
            dst_y_offsets[j] += dst_flow

    ax.set_xlim(-0.15, x_right + 0.15)
    ax.set_ylim(-0.02, max(left_tops[-1], right_tops[-1]) + 0.02)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title(
        title or f"Cluster-level flow after {result.target} {result.mode} ({result.backend})",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, ax


def _resolve_cluster_palette(adata, cluster_col, cats, cluster_palette=None):
    """Pick a cluster colour palette.

    Resolution order:
      1. explicit ``cluster_palette`` argument
      2. ``adata.uns[f"{cluster_col}_colors"]`` if the user already
         plotted those clusters with a specific palette
      3. ``omicverse.utils.pyomic_palette()`` — the package-wide default
      4. matplotlib ``tab20`` fallback
    """
    if cluster_palette is not None:
        return cluster_palette
    key = f"{cluster_col}_colors"
    if adata is not None and key in getattr(adata, "uns", {}):
        colors = list(adata.uns[key])
        if len(colors) >= len(cats):
            return {c: colors[i] for i, c in enumerate(cats)}
    try:
        from .. import utils as ov_utils
        pal = list(ov_utils.pyomic_palette())
        if pal:
            return {c: pal[i % len(pal)] for i, c in enumerate(cats)}
    except Exception:  # pragma: no cover
        pass
    cmap = plt.get_cmap("tab20")
    return {c: cmap(i % 20) for i, c in enumerate(cats)}


def _make_sankey_ribbon(*, x0, x1, y0_src, y1_src, y0_dst, y1_dst, n_pts: int = 30):
    """Smooth ribbon polygon between two vertical bars (source/destination).

    Uses a sigmoid-shaped curve so both edges of the ribbon look
    natural. Returns an ``(n_pts*2, 2)`` polygon to ``ax.fill``.
    """
    xs = np.linspace(0, 1, n_pts)
    # smoothstep
    sigm = xs * xs * (3 - 2 * xs)
    x_line = x0 + (x1 - x0) * xs
    upper = y1_src + (y1_dst - y1_src) * sigm
    lower = y0_src + (y0_dst - y0_src) * sigm
    poly = np.vstack([
        np.column_stack([x_line, upper]),
        np.column_stack([x_line[::-1], lower[::-1]]),
    ])
    return poly


@register_function(
    aliases=[
        "perturb_ps_grid", "PS热图", "perturbation_score_heatmap",
        "perturbation_inner_product_heatmap",
    ],
    category="pl",
    description=(
        "Perturbation Score (PS) on a 2-D grid — pcolormesh of the raw "
        "dot product between simulation flow and developmental gradient "
        "(CellOracle's `plot_inner_product_on_grid` headline figure). "
        "Green = perturbation promotes development along pseudotime; "
        "pink = perturbation blocks it."
    ),
    requires={"obsm": ["{embedding_name}"]},
    auto_fix="none",
    examples=[
        "fig, ax = ov.pl.perturb_inner_product_on_grid(",
        "    adata, result, pseudotime='Pseudotime',",
        "    cluster_col='main_cluster', grid_size=30,",
        ")",
    ],
    related=[
        "single.PerturbResult.perturbation_score",
        "pl.perturb_celloracle_layout",
        "pl.perturb_development_layout",
    ],
)
def perturb_inner_product_on_grid(
    adata,
    result,
    *,
    pseudotime: "str | np.ndarray",
    embedding_name: str = "X_umap",
    grid_size: int = 30,
    min_mass: float = 1.0,
    vmax: Optional[float] = None,
    cmap: str = "PiYG",
    overlay_arrows: bool = True,
    arrow_target_length: float = 0.018,
    arrow_width: float = 0.005,
    arrow_headwidth: float = 5.0,
    n_neighbors: int = 30,
    cluster_col: Optional[str] = None,
    cluster_palette: Optional[Dict[str, str]] = None,
    background_size: float = 14.0,
    figsize=(7, 6),
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
):
    """Plot the per-grid-cell Perturbation Score as a colored heatmap.

    Implements the CellOracle paper's headline figure (Kamimoto 2023,
    Fig. 2/3): the inner product of the perturbation embedding-shift
    vector with the local pseudotime gradient, aggregated onto a
    digitised grid. Negative = perturbation blocks differentiation
    along this trajectory; positive = it promotes.

    Optionally overlays the simulation vector field via ``overlay_arrows``.
    """
    # Grid-level PS computed exactly the CellOracle way: aggregate the
    # per-cell delta_embedding and the per-cell pseudotime gradient onto
    # the same grid, then take the RAW dot product per grid point
    # (no cosine normalisation).
    grid = result.perturbation_score(
        adata=adata, pseudotime=pseudotime, embedding_name=embedding_name,
        n_neighbors=n_neighbors,
        grid_size=grid_size, min_mass=min_mass,
        level="grid",
    )
    grid_pts = grid["grid_pts"]
    UV = grid["flow_grid"]
    PS_grid = grid["ps_grid"]
    keep = grid["keep"]
    emb = np.asarray(adata.obsm[embedding_name])
    bbox_diag = float(np.hypot(np.ptp(emb[:, 0]), np.ptp(emb[:, 1])))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Optional cluster background dots
    if cluster_col is not None and cluster_col in adata.obs:
        labels = pd.Categorical(adata.obs[cluster_col])
        cats = list(labels.categories)
        cluster_palette = _resolve_cluster_palette(adata, cluster_col, cats, cluster_palette)
        c_arr = [cluster_palette[c] for c in labels.astype(str)]
        ax.scatter(emb[:, 0], emb[:, 1], s=background_size, c=c_arr,
                   alpha=0.35, linewidths=0, rasterized=True)

    # PS heatmap via pcolormesh so the grid renders cleanly regardless of
    # subplot size — `scatter(marker='s')` doesn't auto-size and produces
    # stripes inside small subplots (e.g. the 6-panel layout).
    valid = keep & np.isfinite(PS_grid)
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(PS_grid[valid]), 95)) if valid.any() else 1.0
        vmax = max(vmax, 1e-6)
    PS2d = PS_grid.reshape(grid_size, grid_size)
    mask2d = (~valid).reshape(grid_size, grid_size)
    PS2d_masked = np.ma.array(PS2d, mask=mask2d)
    xs = np.linspace(emb[:, 0].min(), emb[:, 0].max(), grid_size)
    ys = np.linspace(emb[:, 1].min(), emb[:, 1].max(), grid_size)
    sc = ax.pcolormesh(
        xs, ys, PS2d_masked,
        cmap=cmap, vmin=-vmax, vmax=vmax,
        shading="nearest", alpha=0.85, zorder=2,
    )
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                 label="Perturbation Score")

    if overlay_arrows:
        norms = np.linalg.norm(UV[valid], axis=1)
        if (norms > 0).any():
            arrows_scale = float(np.percentile(norms[norms > 0], 90))
            scale = arrows_scale / max(bbox_diag * arrow_target_length, 1e-9)
            ax.quiver(
                grid_pts[valid, 0], grid_pts[valid, 1],
                UV[valid, 0], UV[valid, 1],
                angles="xy", scale_units="xy", scale=scale,
                color="k", width=arrow_width,
                headwidth=arrow_headwidth, headlength=arrow_headwidth + 1.5,
                alpha=0.9, zorder=3,
            )

    ax.set_xlabel(f"{embedding_name} 1")
    ax.set_ylabel(f"{embedding_name} 2")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title or f"Perturbation Score ({result.target} {result.mode}, {result.backend})",
                 fontsize=11)
    fig.tight_layout()
    return fig, ax


@register_function(
    aliases=["perturb_cell_quiver", "per-cell扰动箭头", "perturbation_cell_quiver"],
    category="pl",
    description=(
        "Per-cell (not aggregated) quiver of the perturbation embedding-"
        "shift vectors. Equivalent to CellOracle's `plot_quiver`."
    ),
)
def perturb_cell_quiver(
    adata,
    result,
    *,
    embedding_name: str = "X_umap",
    arrow_target_length: float = 0.025,
    arrow_width: float = 0.0045,
    arrow_headwidth: float = 5.0,
    cluster_col: Optional[str] = None,
    cluster_palette: Optional[Dict[str, str]] = None,
    background_size: float = 22.0,
    background_alpha: float = 0.55,
    arrow_color: str = "0.15",
    arrow_alpha: float = 0.8,
    subsample: int = 1,
    ax: Optional[Axes] = None,
    figsize=(7, 6),
    title: Optional[str] = None,
):
    """Per-cell arrow plot of the perturbation flow.

    One arrow per cell (or per ``subsample``-th cell), arrow length
    proportional to the per-cell embedding-shift magnitude. The cell-level
    counterpart to :func:`perturb_quiver`.
    """
    delta_emb = result.delta_embedding(adata=adata, embedding_name=embedding_name)
    emb = np.asarray(adata.obsm[embedding_name])
    sl = slice(None, None, max(1, subsample))
    xrange = emb[:, 0].max() - emb[:, 0].min()
    yrange = emb[:, 1].max() - emb[:, 1].min()
    bbox_diag = float(np.hypot(xrange, yrange))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if cluster_col is not None and cluster_col in adata.obs:
        labels = pd.Categorical(adata.obs[cluster_col])
        cats = list(labels.categories)
        cluster_palette = _resolve_cluster_palette(adata, cluster_col, cats, cluster_palette)
        c_arr = [cluster_palette[c] for c in labels.astype(str)]
        ax.scatter(emb[:, 0], emb[:, 1], s=background_size, c=c_arr,
                   alpha=background_alpha, linewidths=0, rasterized=True)
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(color=cluster_palette[c], label=str(c)) for c in cats],
            loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8,
        )
    else:
        ax.scatter(emb[:, 0], emb[:, 1], s=background_size, c="lightgray",
                   alpha=background_alpha, linewidths=0, rasterized=True)

    norms = np.linalg.norm(delta_emb[sl], axis=1)
    arrows_scale = float(np.percentile(norms[norms > 0], 90)) if (norms > 0).any() else 1.0
    scale = arrows_scale / max(bbox_diag * arrow_target_length, 1e-9)
    ax.quiver(
        emb[sl, 0], emb[sl, 1],
        delta_emb[sl, 0], delta_emb[sl, 1],
        angles="xy", scale_units="xy", scale=scale,
        color=arrow_color, alpha=arrow_alpha,
        width=arrow_width,
        headwidth=arrow_headwidth, headlength=arrow_headwidth + 1.5,
        zorder=3,
    )
    ax.set_xlabel(f"{embedding_name} 1")
    ax.set_ylabel(f"{embedding_name} 2")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title or f"Per-cell perturbation flow ({result.target} {result.mode}, {result.backend})",
                 fontsize=11)
    fig.tight_layout()
    return fig, ax


@register_function(
    aliases=[
        "perturb_markov_endpoints", "马尔可夫终点条形图",
        "perturbation_markov_endpoint_distribution",
        "perturbation_lineage_redirection",
    ],
    category="pl",
    description=(
        "Bar plot of the endpoint cluster distribution from an n-step "
        "Markov walk on `result.trajectory_shift`. Reveals where a "
        "perturbed lineage ends up (e.g. Gata1 KO redirects Mk → GMP)."
    ),
    requires={"obs": ["{cluster_col}"]},
    auto_fix="none",
    examples=[
        "mep_cells = adata.obs_names[adata.obs['main_cluster'] == 'MEP'][:30]",
        "fig, _ = ov.pl.perturb_markov_endpoints(",
        "    result, adata=adata,",
        "    start_cells=list(mep_cells),",
        "    cluster_col='main_cluster',",
        "    n_steps=15, n_walks_per_cell=50,",
        ")",
    ],
    related=[
        "single.PerturbResult.run_markov",
        "single.PerturbResult.cluster_transitions",
        "pl.perturb_sankey",
    ],
)
def perturb_markov_endpoints(
    result,
    adata,
    *,
    start_cells,
    cluster_col: str,
    n_steps: int = 15,
    n_walks_per_cell: int = 50,
    figsize=(5, 3),
    color: str = "C3",
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
):
    """One-call wrapper around `result.run_markov` that plots the endpoint
    cluster distribution as a horizontal bar chart."""
    walks = result.run_markov(
        start_cells=list(start_cells), n_steps=n_steps,
        n_walks_per_cell=n_walks_per_cell, adata=adata,
    )
    end_ix = walks.values.ravel()
    end_clusters = adata.obs[cluster_col].iloc[end_ix]
    counts = end_clusters.value_counts()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    counts.plot.barh(ax=ax, color=color, alpha=0.75, edgecolor="white")
    ax.set_xlabel(f"# walks ending in cluster "
                  f"({len(start_cells)} starts × {n_walks_per_cell} walks)")
    ax.set_title(
        title or f"Markov-walk endpoints ({result.target} {result.mode})",
        fontsize=11,
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig, ax


@register_function(
    aliases=["perturb_volcano", "扰动火山图", "perturbation_volcano"],
    category="pl",
    description=(
        "Volcano plot of |Δ| (x) vs −log10(p) (y) from a PerturbResult, with "
        "the top-N most significant genes labelled."
    ),
)
def perturb_volcano(
    result,
    *,
    top_n: int = 20,
    p_col: Optional[str] = None,
    delta_col: str = "delta",
    sig_threshold: float = 0.05,
    ax: Optional[Axes] = None,
    figsize=(7, 5),
    title: Optional[str] = None,
):
    """Volcano of delta vs −log10(p) for the per-gene table."""
    df = result.delta_expr.copy()
    if p_col is None:
        # prefer adjusted p, fall back to raw p
        p_col = "adjusted p-value" if "adjusted p-value" in df.columns else (
            "adj_p_value" if "adj_p_value" in df.columns else (
                "p-value" if "p-value" in df.columns else (
                    "p_value" if "p_value" in df.columns else None
                )
            )
        )
    if p_col is None:
        raise ValueError(
            "delta_expr has no p-value column. Call result.add_significance(...) "
            "first for the cell_oracle backend."
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    p = np.asarray(df[p_col].astype(float)).clip(min=1e-300)
    neglog = -np.log10(p)
    # Cap -log10(p) at 50 so the y-axis doesn't get hijacked by one super-significant gene
    y_cap = 50.0
    neglog_plot = np.minimum(neglog, y_cap)
    sig = df[p_col] < sig_threshold
    ax.scatter(df.loc[~sig, delta_col], neglog_plot[~sig.to_numpy()],
               s=10, c="lightgray", alpha=0.6, linewidths=0)
    ax.scatter(df.loc[sig, delta_col], neglog_plot[sig.to_numpy()],
               s=14, c="C3", alpha=0.85, linewidths=0)
    # Rank top-N by |Δ| × −log10(p) — picks both significant AND large-effect genes
    df["_neglog"] = neglog_plot
    score = np.abs(df[delta_col].to_numpy()) * neglog_plot
    top_idx = np.argsort(-score)[:top_n]
    top = df.iloc[top_idx]
    try:
        from adjustText import adjust_text  # type: ignore
        texts = [
            ax.text(row[delta_col], row["_neglog"], str(row.get("gene", "")),
                    fontsize=8)
            for _, row in top.iterrows()
        ]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4))
    except ImportError:
        for _, row in top.iterrows():
            ax.annotate(
                str(row.get("gene", "")),
                xy=(row[delta_col], row["_neglog"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=8, ha="left", va="bottom",
            )
    ax.axhline(-np.log10(sig_threshold), color="0.5", lw=0.5, ls="--")
    ax.set_xlabel(delta_col)
    ax.set_ylabel(f"−log10({p_col})")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title or f"{result.target} {result.mode} ({result.backend})",
                 fontsize=11)
    return fig, ax


def _build_oracle_adapter(adata, result, cluster_column_name=None):
    """Wrap a non-CellOracle PerturbResult (e.g. sctenifoldknk) in a freshly
    built celloracle.Oracle so we can still run CellOracle's downstream
    pipeline (`Oracle_development_module.load_perturb_simulation_data`,
    `visualize_development_module_layout_0`, …) on it.

    Needed attributes on the Oracle (read by Gradient_calculator +
    Oracle_development_module):
        embedding, delta_embedding, delta_embedding_random,
        colorandum, cluster_column_name, embedding_name, adata.
    """
    import celloracle as co
    if result.delta_X is None:
        raise ValueError(
            "PerturbResult has no per-cell delta_X — cannot build an "
            "Oracle adapter."
        )
    oracle = co.Oracle()
    oracle.adata = adata.copy()
    oracle.embedding_name = "X_draw_graph_fa" if "X_draw_graph_fa" in adata.obsm else "X_umap"
    oracle.embedding = np.asarray(adata.obsm[oracle.embedding_name])
    if cluster_column_name is None:
        cluster_column_name = "louvain_annot" if "louvain_annot" in adata.obs else None
    oracle.cluster_column_name = cluster_column_name

    # Per-cell delta_embedding — same formulation as PerturbResult.delta_embedding
    tp = np.asarray(result.trajectory_shift)
    diffs = oracle.embedding[None, :, :] - oracle.embedding[:, None, :]
    norms = np.linalg.norm(diffs, axis=-1, keepdims=True)
    unit_vecs = np.where(norms > 1e-12, diffs / np.maximum(norms, 1e-12), 0.0)
    oracle.delta_embedding = np.einsum("ij,ijk->ik", tp, unit_vecs)
    # Random control: sign-flip the per-cell delta as a quick null
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=oracle.delta_embedding.shape[0])
    oracle.delta_embedding_random = oracle.delta_embedding * signs[:, None]

    # Colorandum: per-cell RGB(A) array from cluster palette
    if cluster_column_name is not None and cluster_column_name in adata.obs:
        labels = pd.Categorical(adata.obs[cluster_column_name])
        palette = _resolve_cluster_palette(adata, cluster_column_name, list(labels.categories), None)
        from matplotlib.colors import to_rgba
        oracle.colorandum = np.asarray(
            [to_rgba(palette[c]) for c in labels.astype(str)]
        )
    else:
        oracle.colorandum = np.tile([[0.5, 0.5, 0.5, 1.0]], (oracle.embedding.shape[0], 1))

    return oracle


@register_function(
    aliases=[
        "perturb_celloracle_layout", "CellOracle原版六面板",
        "perturbation_celloracle_development_module_layout",
        "perturbation_score_official_layout",
    ],
    category="pl",
    description=(
        "**Run CellOracle's own** `Oracle_development_module."
        "visualize_development_module_layout_0` on a PerturbResult — "
        "guarantees 1:1 output with the published Gata1-KO Paul15 figure. "
        "For `backend='cell_oracle'` uses the cached Oracle; for any "
        "other backend builds an Oracle-compatible adapter from "
        "`result.delta_X`."
    ),
    requires={
        "obsm": ["{embedding_obsm_key}"],
        "obs": ["{pseudotime_key}", "{cluster_column_name}"],
    },
    auto_fix="none",
    examples=[
        "fig, dev = ov.pl.perturb_celloracle_layout(",
        "    adata, result,",
        "    pseudotime_key='Pseudotime',",
        "    cluster_column_name='louvain_annot',",
        "    vm=0.02,",
        ")",
    ],
    related=[
        "single.perturb", "single.lineage_pseudotime",
        "pl.perturb_inner_product_on_grid",
        "pl.perturb_development_layout",
    ],
)
def perturb_celloracle_layout(
    adata,
    result,
    *,
    pseudotime_key: str = "Pseudotime",
    cluster_column_name: Optional[str] = None,
    embedding_obsm_key: str = "X_draw_graph_fa",
    n_grid: int = 40,
    smooth: float = 0.8,
    n_neighbors_p_mass: int = 200,
    min_mass: float = 0.01,
    n_knn: int = 50,
    vm: float = 0.02,
    s: float = 5.0,
    s_grid: float = 20.0,
    scale_for_simulation: "float | str" = "auto",
    scale_for_pseudotime: "float | str" = "auto",
    figsize=(15, 10),
):
    """Run CellOracle's own development-module pipeline on this
    PerturbResult and return its `visualize_development_module_layout_0`
    figure — 1:1 with the published Paul15 Gata1-KO tutorial.

    Requires ``result.backend == 'cell_oracle'`` (the cached Oracle
    object is stashed at ``result.meta['oracle']``). The pseudotime
    column ``adata.obs[pseudotime_key]`` should be lineage-specific —
    use ``co.applications.Pseudotime_calculator`` to build it.
    """
    try:
        import celloracle as co
        from celloracle.applications import (
            Gradient_calculator, Oracle_development_module,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "perturb_celloracle_layout needs CellOracle installed."
        ) from exc

    oracle = result.meta.get("oracle")
    if oracle is None:
        # sctenifoldknk (or any non-cell_oracle backend) — wrap the result
        # in a freshly-built Oracle so we can still run CellOracle's own
        # downstream pipeline + visualize_development_module_layout_0.
        oracle = _build_oracle_adapter(adata, result, cluster_column_name)

    # Mirror oracle.adata.obs['Pseudotime'] from the user-supplied column
    if pseudotime_key not in oracle.adata.obs:
        if pseudotime_key in adata.obs:
            oracle.adata.obs[pseudotime_key] = adata.obs[pseudotime_key].values
        else:
            raise KeyError(
                f"pseudotime column {pseudotime_key!r} not in adata.obs"
            )

    # Compute simulation flow on grid via CellOracle's own pipeline.
    # The methods come from VelocytoLoom: calculate_grid_arrows builds
    # flow_grid + flow; calculate_p_mass / calculate_mass_filter build
    # mass_filter_simulation (used by plot_simulation_flow_on_grid).
    # `calculate_grid_arrows` only builds `flow_rndm` if `corrcoef_random`
    # exists on the object — we stub it for the adapter path so the random
    # control field is also populated.
    if not hasattr(oracle, "flow_grid"):
        if not hasattr(oracle, "corrcoef_random") and hasattr(oracle, "delta_embedding_random"):
            oracle.corrcoef_random = np.zeros((oracle.embedding.shape[0], 1))
        oracle.calculate_grid_arrows(
            smooth=smooth, steps=(n_grid, n_grid),
            n_neighbors=n_neighbors_p_mass,
        )
    if not hasattr(oracle, "flow_rndm"):
        oracle.flow_rndm = oracle.flow.copy()
    if not hasattr(oracle, "mass_filter_simulation"):
        oracle.calculate_p_mass(
            smooth=smooth, n_grid=n_grid, n_neighbors=n_neighbors_p_mass,
        )
        oracle.calculate_mass_filter(min_mass=min_mass)

    # Developmental gradient on the same grid
    gradient = Gradient_calculator(
        oracle_object=oracle, pseudotime_key=pseudotime_key,
    )
    gradient.calculate_p_mass(
        smooth=smooth, n_grid=n_grid, n_neighbors=n_neighbors_p_mass,
    )
    gradient.calculate_mass_filter(min_mass=min_mass)
    gradient.transfer_data_into_grid(args={"method": "knn", "n_knn": n_knn})
    gradient.calculate_gradient()

    # Oracle_development_module ties simulation + gradient together
    dev = Oracle_development_module()
    dev.load_differentiation_reference_data(gradient_object=gradient)
    dev.load_perturb_simulation_data(oracle_object=oracle)
    dev.calculate_inner_product()
    try:
        dev.calculate_digitized_ip(n_bins=10)
    except Exception:
        pass

    # Auto-scale per CellOracle's formula
    # (90th-percentile flow norm / (plot diagonal × 0.0025)) so the arrows
    # come out visually similar regardless of whether delta_X is in
    # CellOracle's (large) or sct's (small) magnitude range.
    def _auto_scale(flow_arr):
        mask = ~oracle.mass_filter_simulation if hasattr(oracle, "mass_filter_simulation") else slice(None)
        norms = np.linalg.norm(flow_arr[mask], axis=1)
        norms = norms[norms > 0]
        if not len(norms):
            return 30.0
        plot_diag = np.linalg.norm(
            np.max(oracle.flow_grid, axis=0) - np.min(oracle.flow_grid, axis=0)
        )
        return float(np.percentile(norms, 90) / max(plot_diag * 0.0025, 1e-12))

    if scale_for_simulation == "auto":
        scale_for_simulation = _auto_scale(oracle.flow)
    if scale_for_pseudotime == "auto":
        scale_for_pseudotime = _auto_scale(gradient.ref_flow)

    # CellOracle's own composite figure — guaranteed 1:1
    plt.rcParams["figure.figsize"] = figsize
    dev.visualize_development_module_layout_0(
        s=s, scale_for_simulation=scale_for_simulation,
        s_grid=s_grid, scale_for_pseudotime=scale_for_pseudotime,
        vm=vm,
    )
    fig = plt.gcf()
    return fig, dev


@register_function(
    aliases=[
        "perturb_development_layout", "扰动六面板综合图",
        "perturbation_development_module_layout",
    ],
    category="pl",
    description=(
        "Backend-agnostic six-panel composite reproducing CellOracle's "
        "`visualize_development_module_layout_0`: clusters / dev-gradient / "
        "simulation-flow / PS-heatmap / PS-vs-pseudotime / PS-by-bin. "
        "Works for both sctenifoldknk + cell_oracle backends."
    ),
)
def perturb_development_layout(
    adata,
    result,
    *,
    pseudotime: "str | np.ndarray",
    cluster_col: str,
    embedding_name: str = "X_umap",
    grid_size: int = 40,
    min_mass: float = 1.0,
    cluster_palette: Optional[Dict[str, str]] = None,
    vm: Optional[float] = None,
    n_pseudotime_bins: int = 10,
    background_color: str = "lightgray",
    background_size: float = 5.0,
    cell_label_fontsize: float = 9.0,
    figsize=(15, 10),
):
    """Reproduce CellOracle's ``visualize_development_module_layout_0`` figure
    panel-by-panel.

    A 2×3 grid:

    ``(0,0)`` cell-type clusters on the embedding (`plot_cluster_cells_use`)
        ``(0,1)`` developmental gradient on grid (`plot_reference_flow_on_grid`)
        ``(0,2)`` simulation flow on grid (`plot_simulation_flow_on_grid`)
    ``(1,0)`` PS heatmap on grid (`plot_inner_product_on_grid`)
        ``(1,1)`` PS vs pseudotime — **grid-level** scatter, colored by PS
        ``(1,2)`` PS box-plot — **pseudotime-binned**, not by cluster

    Style matches CellOracle: light-gray cell background everywhere
    except (0,0); axis off; cluster names rendered at cluster centroids.
    """
    if isinstance(pseudotime, str):
        if pseudotime not in adata.obs:
            raise ValueError(f"pseudotime {pseudotime!r} not in adata.obs")
        pt = np.asarray(adata.obs[pseudotime].values, dtype=np.float64)
    else:
        pt = np.asarray(pseudotime, dtype=np.float64)

    # Grid PS + flow + ref_flow + grid pseudotime (all aggregated on the same grid)
    grid = result.perturbation_score(
        adata=adata, pseudotime=pseudotime, embedding_name=embedding_name,
        grid_size=grid_size, min_mass=min_mass,
        level="grid",
    )
    grid_pts = grid["grid_pts"]
    flow_grid = grid["flow_grid"]
    ref_flow_grid = grid["ref_flow_grid"]
    ps_grid = grid["ps_grid"]
    mass = grid["mass"]
    keep = grid["keep"]
    # Aggregate pseudotime per grid point
    from scipy.spatial import cKDTree
    emb = np.asarray(adata.obsm[embedding_name])
    bbox_diag = float(np.hypot(np.ptp(emb[:, 0]), np.ptp(emb[:, 1])))
    xs = np.linspace(emb[:, 0].min(), emb[:, 0].max(), grid_size)
    ys = np.linspace(emb[:, 1].min(), emb[:, 1].max(), grid_size)
    radius = float(np.linalg.norm([xs[1] - xs[0], ys[1] - ys[0]]))
    tree = cKDTree(emb)
    pt_grid = np.full(grid_pts.shape[0], np.nan)
    for g_i, p in enumerate(grid_pts):
        ix = tree.query_ball_point(p, r=radius * 1.5)
        if not ix:
            continue
        d = np.linalg.norm(emb[ix] - p, axis=1)
        w = np.exp(-(d ** 2) / (2 * (radius / 2) ** 2))
        if w.sum() > 0:
            pt_grid[g_i] = (pt[ix] * w).sum() / w.sum()

    labels = pd.Categorical(adata.obs[cluster_col])
    cats = list(labels.categories)
    cluster_palette = _resolve_cluster_palette(adata, cluster_col, cats, cluster_palette)

    # Helper to draw lightgray cell background on a panel and turn axis off
    def _bg(ax):
        ax.scatter(emb[:, 0], emb[:, 1], s=background_size,
                   c=background_color, alpha=0.55, linewidths=0,
                   rasterized=True)

    def _hide_axes(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ("top", "right", "bottom", "left"):
            ax.spines[sp].set_visible(False)

    # --- 90th-percentile-based arrow scale (CellOracle's `arrows_scale`) -----
    def _quiver_scale(UV, target_frac=0.018):
        norms = np.linalg.norm(UV[keep], axis=1)
        if (norms > 0).any():
            arrows_scale = float(np.percentile(norms[norms > 0], 90))
            return arrows_scale / max(bbox_diag * target_frac, 1e-9)
        return 1.0

    if vm is None:
        valid_ps = ps_grid[keep & np.isfinite(ps_grid)]
        vm = float(np.nanpercentile(np.abs(valid_ps), 95)) if valid_ps.size else 0.01
        vm = max(vm, 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # ---- (0,0) clusters on the embedding + labels at centroids ----
    ax = axes[0, 0]
    for c in cats:
        m = (labels.astype(str) == str(c))
        ax.scatter(emb[m, 0], emb[m, 1], s=background_size,
                   c=[cluster_palette[c]], alpha=0.85, linewidths=0,
                   rasterized=True)
    # Cluster labels at centroids
    for c in cats:
        m = (labels.astype(str) == str(c))
        if not m.any():
            continue
        cx, cy = emb[m, 0].mean(), emb[m, 1].mean()
        ax.text(cx, cy, str(c), ha="center", va="center",
                fontsize=cell_label_fontsize,
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none",
                          boxstyle="round,pad=0.2"))
    ax.set_title("Clusters", fontsize=10)
    _hide_axes(ax)

    # ---- (0,1) developmental gradient (CellOracle's ref_flow on grid) ----
    ax = axes[0, 1]
    _bg(ax)
    scale_dev = _quiver_scale(ref_flow_grid, target_frac=0.022)
    ax.quiver(grid_pts[keep, 0], grid_pts[keep, 1],
              ref_flow_grid[keep, 0], ref_flow_grid[keep, 1],
              angles="xy", scale_units="xy", scale=scale_dev,
              color="0.15", alpha=0.9,
              width=0.005, headwidth=5, headlength=6.5, zorder=3)
    ax.set_title("Developmental gradient", fontsize=10)
    _hide_axes(ax)

    # ---- (0,2) simulation flow (KO arrows on grid) ----
    ax = axes[0, 2]
    _bg(ax)
    scale_sim = _quiver_scale(flow_grid, target_frac=0.022)
    ax.quiver(grid_pts[keep, 0], grid_pts[keep, 1],
              flow_grid[keep, 0], flow_grid[keep, 1],
              angles="xy", scale_units="xy", scale=scale_sim,
              color="0.15", alpha=0.9,
              width=0.005, headwidth=5, headlength=6.5, zorder=3)
    ax.set_title("Simulation flow", fontsize=10)
    _hide_axes(ax)

    # ---- (1,0) PS heatmap on grid + arrows ----
    ax = axes[1, 0]
    _bg(ax)
    PS2d = ps_grid.reshape(grid_size, grid_size)
    mask2d = (~(keep & np.isfinite(ps_grid))).reshape(grid_size, grid_size)
    PS2d_m = np.ma.array(PS2d, mask=mask2d)
    sc = ax.pcolormesh(
        xs, ys, PS2d_m, cmap="PiYG",
        vmin=-vm, vmax=vm, shading="nearest", alpha=0.85, zorder=2,
    )
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                 label="Perturbation Score")
    ax.quiver(grid_pts[keep, 0], grid_pts[keep, 1],
              flow_grid[keep, 0], flow_grid[keep, 1],
              angles="xy", scale_units="xy", scale=scale_sim,
              color="k", alpha=0.9,
              width=0.005, headwidth=5, headlength=6.5, zorder=3)
    ax.set_title("Perturbation Score (PS)", fontsize=10)
    _hide_axes(ax)

    # ---- (1,1) grid-PS vs grid-pseudotime, coloured by PS (CellOracle style) ----
    ax = axes[1, 1]
    valid = keep & np.isfinite(ps_grid) & np.isfinite(pt_grid)
    sc2 = ax.scatter(
        pt_grid[valid], ps_grid[valid], c=ps_grid[valid],
        cmap="PiYG", vmin=-vm, vmax=vm, s=25,
        linewidths=0, alpha=0.95,
    )
    ax.axhline(0.0, color="lightgray", lw=0.8)
    ax.set_xlabel("pseudotime")
    ax.set_ylabel("inner product score")
    ax.set_ylim(-vm * 1.1, vm * 1.1)
    ax.set_title("PS vs pseudotime", fontsize=10)
    fig.colorbar(sc2, ax=ax, fraction=0.04, pad=0.02)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- (1,2) pseudotime-binned PS boxplot (CellOracle's plot_inner_product_as_box) ----
    ax = axes[1, 2]
    if valid.any():
        bins = np.linspace(np.nanmin(pt_grid[valid]),
                           np.nanmax(pt_grid[valid]),
                           n_pseudotime_bins + 1)
        digitised = np.digitize(pt_grid[valid], bins) - 1
        digitised = np.clip(digitised, 0, n_pseudotime_bins - 1)
        data_by_bin = [ps_grid[valid][digitised == i]
                       for i in range(n_pseudotime_bins)]
        positions = np.arange(1, n_pseudotime_bins + 1)
        ax.boxplot(
            data_by_bin, positions=positions, showfliers=False,
            widths=0.65, patch_artist=True,
            boxprops=dict(facecolor="white", edgecolor="0.3"),
            medianprops=dict(color="black", lw=1.0),
            whiskerprops=dict(color="0.3"),
            capprops=dict(color="0.3"),
        )
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("digitised pseudotime")
    ax.set_ylabel("inner product score")
    ax.set_ylim(-vm * 1.1, vm * 1.1)
    ax.set_title("PS by pseudotime bin", fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle(
        f"Development module layout — {result.target} {result.mode} ({result.backend})",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    return fig, axes
