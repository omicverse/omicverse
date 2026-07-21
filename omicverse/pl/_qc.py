"""QC-metric distribution plots — inspect before you filter (issue #808).

The classic scRNA QC workflow is *look, then cut*: view the distribution of
each QC metric (optionally per sample, since quality differs between samples)
and only then choose thresholds. omicverse computes the metrics
(:func:`omicverse.pp.qc_metrics` / :func:`omicverse.pp.qc`) and filters
(:func:`omicverse.pp.qc`), but previously left the plotting to the user.
``ov.pl.qc`` fills that gap.
"""
from __future__ import annotations

from numbers import Real
from typing import Optional, Sequence, Union

import numpy as np

from .._registry import register_function

# Preferred metric columns, with fallbacks to scanpy's names, and the
# ov.pp.qc `tresh` key each maps to (for drawing threshold guide lines).
_DEFAULT_METRICS = [
    # (candidate obs columns, nice label, tresh-key, tresh-direction)
    (["nUMIs", "total_counts"], "UMIs per cell", "nUMIs", "min"),
    (["detected_genes", "n_genes_by_counts"], "Genes per cell", "detected_genes", "min"),
    (["mito_perc", "pct_counts_mt"], "Mito %", "mito_perc", "max"),
    (["pct_counts_ribo"], "Ribo %", None, None),
    (["pct_counts_hb"], "Hb %", None, None),
    (["doublet_score"], "Doublet score", None, None),
]
_UMI_METRICS = {"nUMIs", "total_counts"}
_GENE_METRICS = {"detected_genes", "n_genes_by_counts"}


def _validate_clip_at(value, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive numeric value or None.")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive numeric value or None.")
    return value


def _metric_clip_at(
    col: str,
    *,
    umi_clip_at: Optional[float],
    gene_clip_at: Optional[float],
) -> Optional[float]:
    if col in _UMI_METRICS:
        return umi_clip_at
    if col in _GENE_METRICS:
        return gene_clip_at
    return None


def _clip_values(values, clip_at: Optional[float]):
    if clip_at is None:
        return values
    return values.clip(upper=clip_at)


def _label_with_clip(label: str, clip_at: Optional[float]) -> str:
    if clip_at is None:
        return label
    return f"{label} (clipped at {clip_at:g})"


def _resolve_metrics(adata, metrics):
    """Return [(col, label, tresh_key, direction), ...] present in obs."""
    if metrics is not None:
        out = []
        for m in metrics:
            if m in adata.obs:
                # map known cols to a tresh key for guide lines
                key, direction = None, None
                for cands, _lab, k, d in _DEFAULT_METRICS:
                    if m in cands:
                        key, direction = k, d
                        break
                out.append((m, m, key, direction))
        return out
    out = []
    for cands, label, key, direction in _DEFAULT_METRICS:
        for c in cands:
            if c in adata.obs:
                out.append((c, label, key, direction))
                break
    return out


@register_function(
    aliases=["qc_plot", "qc", "plot_qc", "质控绘图", "质控分布图"],
    category="plotting",
    description=(
        "Plot the distribution of per-cell QC metrics (UMIs, genes, mito%, "
        "ribo%, hb%, doublet score) as histograms or violins, optionally per "
        "sample, with threshold guide lines — to choose QC cutoffs before "
        "filtering with ov.pp.qc."
    ),
    examples=[
        "ov.pl.qc(adata)",
        "ov.pl.qc(adata, tresh={'mito_perc': 0.13, 'nUMIs': 800, 'detected_genes': 500})",
        "ov.pl.qc(adata, umi_clip_at=50000, gene_clip_at=8000)",
        "ov.pl.qc(adata, batch_key='sample', kind='violin')",
    ],
    related=["pp.qc_metrics", "pp.qc"],
)
def qc(
    adata,
    *,
    metrics: Optional[Sequence[str]] = None,
    kind: str = "hist",
    batch_key: Optional[str] = None,
    tresh: Optional[dict] = None,
    bins: int = 50,
    log: Union[bool, str] = "auto",
    ncols: int = 3,
    figsize: Optional[tuple] = None,
    color: str = "#4C72B0",
    palette: Optional[Sequence[str]] = None,
    umi_clip_at: Optional[float] = None,
    gene_clip_at: Optional[float] = None,
):
    """Plot QC-metric distributions to guide threshold choice.

    Parameters
    ----------
    adata
        AnnData with QC metrics in ``obs`` (run :func:`omicverse.pp.qc_metrics`
        or :func:`omicverse.pp.qc` first, or scanpy's ``calculate_qc_metrics``).
    metrics
        ``obs`` columns to plot. Default: auto-detect the standard metrics
        (UMIs, genes, mito%, ribo%, hb%, doublet score) that are present.
    kind
        ``'hist'`` (default) or ``'violin'``. Violin is most useful with
        ``batch_key`` to compare per-sample quality.
    batch_key
        ``obs`` column to split by (e.g. sample). Histograms overlay one
        curve per group; violins show one violin per group.
    tresh
        Optional ``ov.pp.qc``-style dict (keys ``'nUMIs'``, ``'detected_genes'``,
        ``'mito_perc'``) — drawn as dashed guide lines on the matching panels.
        ``mito_perc`` is a fraction (0.13 → 13%) to match ``ov.pp.qc``.
    bins
        Histogram bin count (default 50).
    log
        Log-scale the metric axis. ``'auto'`` logs counts metrics
        (UMIs/genes) but not percentages.
    umi_clip_at
        Upper clip for plotted UMI/count metrics (``nUMIs`` and
        ``total_counts`` fallback). Clipping is display-only and does not
        modify ``adata.obs`` or QC/filtering thresholds.
    gene_clip_at
        Upper clip for plotted gene-count metrics (``detected_genes`` and
        ``n_genes_by_counts`` fallback). Clipping is display-only and does
        not modify ``adata.obs`` or QC/filtering thresholds.
    ncols, figsize, color, palette
        Layout / styling controls.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    umi_clip_at = _validate_clip_at(umi_clip_at, "umi_clip_at")
    gene_clip_at = _validate_clip_at(gene_clip_at, "gene_clip_at")
    resolved = _resolve_metrics(adata, metrics)
    if not resolved:
        raise ValueError(
            "No QC metrics found in adata.obs. Run ov.pp.qc_metrics(adata) "
            "(or ov.pp.qc / sc.pp.calculate_qc_metrics) first."
        )
    tresh = tresh or {}
    groups = None
    if batch_key is not None:
        if batch_key not in adata.obs:
            raise KeyError(f"obs[{batch_key!r}] not found.")
        groups = list(adata.obs[batch_key].astype("category").cat.categories)

    n = len(resolved)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.2 * ncols, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    if palette is None:
        cmap = plt.get_cmap("tab20")
        palette = [cmap(i % 20) for i in range(len(groups))] if groups else None

    for ax, (col, label, key, direction) in zip(axes, resolved):
        vals = adata.obs[col].astype(float)
        clip_at = _metric_clip_at(
            col,
            umi_clip_at=umi_clip_at,
            gene_clip_at=gene_clip_at,
        )
        plot_label = _label_with_clip(label, clip_at)
        plot_vals = _clip_values(vals, clip_at)
        do_log = (log is True) or (log == "auto" and direction == "min")

        if kind == "violin":
            if groups:
                data = [adata.obs.loc[adata.obs[batch_key] == g, col]
                        .astype(float).pipe(_clip_values, clip_at).dropna().values
                        for g in groups]
                parts = ax.violinplot(data, showmedians=True, widths=0.85)
                ax.set_xticks(range(1, len(groups) + 1))
                ax.set_xticklabels(groups, rotation=90, fontsize=7)
                if palette:
                    for pc, c in zip(parts["bodies"], palette):
                        pc.set_facecolor(c); pc.set_alpha(0.75)
            else:
                parts = ax.violinplot(plot_vals.dropna().values, showmedians=True)
                for pc in parts["bodies"]:
                    pc.set_facecolor(color); pc.set_alpha(0.8)
                ax.set_xticks([])
            ax.set_ylabel(plot_label)
            if do_log:
                ax.set_yscale("log")
        else:  # histogram
            if groups:
                for g, c in zip(groups, palette):
                    gv = adata.obs.loc[adata.obs[batch_key] == g, col].astype(float).dropna()
                    gv = _clip_values(gv, clip_at)
                    ax.hist(gv, bins=bins, histtype="step", color=c,
                            label=str(g), linewidth=1.2,
                            log=do_log if False else False, density=True)
                if len(groups) <= 12:
                    ax.legend(fontsize=6, frameon=False)
            else:
                ax.hist(plot_vals.dropna(), bins=bins, color=color, alpha=0.85)
            ax.set_xlabel(plot_label)
            ax.set_ylabel("density" if groups else "cells")
            if do_log:
                ax.set_xscale("log")

        # threshold guide line
        if key in tresh:
            t = tresh[key]
            if key == "mito_perc" and col == "pct_counts_mt":
                t = t * 100.0  # tresh is a fraction; pct col is 0-100
            line = ax.axhline if kind == "violin" else ax.axvline
            line(t, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(plot_label, fontsize=10)

    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig
