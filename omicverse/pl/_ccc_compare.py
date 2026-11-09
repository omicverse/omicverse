"""Plots for :func:`omicverse.single.compare_ccc` (agreement between communication methods)."""
from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._registry import register_function

_MATRICES = {"jaccard": "Jaccard index", "recall": "Share of row method's calls\nalso made by column method",
             "overlap": "Shared significant interactions", "score_corr": "Spearman (scores)",
             "lr_corr": "Spearman (cell pairs per LR)", "cellpair_corr": "Pearson (interactions per cell pair)"}
_KEYS = {"jaccard": "jaccard", "recall": "recall", "overlap": "overlap", "score_corr": "score_spearman",
         "lr_corr": "lr_spearman", "cellpair_corr": "cellpair_pearson"}


def _matrix_plot(mat: pd.DataFrame, title: str, ax, cmap: str, fmt: str):
    values = mat.to_numpy(float)
    finite = values[np.isfinite(values)]
    vmax = 1.0 if finite.size == 0 or finite.max() <= 1 else finite.max()
    vmin = min(0.0, finite.min()) if finite.size else 0.0
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0]), mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = values[i, j]
            txt = "NA" if not np.isfinite(v) else format(v, fmt)
            colour = "white" if np.isfinite(v) and (v - vmin) > 0.6 * (vmax - vmin) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=colour)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _upset(sets: Mapping[str, set], ax_bar, ax_dot, ax_size, max_bars: int, color: str):
    methods = list(sets)
    universe = set().union(*sets.values())
    membership = {}
    for item in universe:
        key = tuple(m for m in methods if item in sets[m])
        membership[key] = membership.get(key, 0) + 1
    combos = sorted(membership.items(), key=lambda kv: (-kv[1], len(kv[0])))[:max_bars]
    x = np.arange(len(combos))
    ax_bar.bar(x, [n for _, n in combos], color=color, width=0.6)
    for xi, (_, n) in zip(x, combos):
        ax_bar.text(xi, n, str(n), ha="center", va="bottom", fontsize=8)
    ax_bar.set_ylabel("Significant interactions")
    ax_bar.set_xticks([])
    ax_bar.spines[["top", "right"]].set_visible(False)
    for yi, m in enumerate(methods):
        ax_dot.scatter(x, [yi] * len(x), color="#d9d9d9", s=60, zorder=1)
    for xi, (combo, _) in zip(x, combos):
        ys = [methods.index(m) for m in combo]
        ax_dot.scatter([xi] * len(ys), ys, color=color, s=60, zorder=2)
        if len(ys) > 1:
            ax_dot.plot([xi, xi], [min(ys), max(ys)], color=color, lw=2, zorder=2)
    ax_dot.set_yticks([])
    ax_dot.set_xticks([])
    ax_dot.set_ylim(-0.5, len(methods) - 0.5)
    ax_dot.invert_yaxis()
    for side in ("top", "right", "bottom", "left"):
        ax_dot.spines[side].set_visible(False)
    sizes = [len(sets[m]) for m in methods]
    ax_size.barh(range(len(methods)), sizes, color="#8c8c8c", height=0.5)
    ax_size.set_ylim(-0.5, len(methods) - 0.5)
    ax_size.invert_yaxis()
    ax_size.invert_xaxis()
    # Method names sit between the set-size bars and the dot matrix.
    ax_size.yaxis.tick_right()
    ax_size.set_yticks(range(len(methods)), methods)
    ax_size.tick_params(axis="y", length=0, pad=6)
    ax_size.set_xlabel("Set size")
    ax_size.spines[["top", "left"]].set_visible(False)


@register_function(
    aliases=["ccc_compare", "细胞通讯方法比较图", "compare_ccc plot", "communication method agreement plot"],
    category="pl",
    description="Plot agreement between communication methods from ov.single.compare_ccc "
                "(UpSet of significant interactions, Jaccard/recall/correlation heatmaps, top LR pairs).",
    examples=["res = ov.single.compare_ccc(adata)", "ov.pl.ccc_compare(res, plot_type='upset')",
              "ov.pl.ccc_compare(res, plot_type='jaccard')", "ov.pl.ccc_compare(res, plot_type='top_lr')"],
    related=["single.compare_ccc", "single.run_ccc", "pl.ccc_heatmap"],
)
def ccc_compare(
    result: Mapping,
    plot_type: Literal["upset", "jaccard", "recall", "overlap", "score_corr", "lr_corr", "cellpair_corr",
                       "top_lr", "cellpair"] = "upset",
    *,
    top_n: int = 15,
    max_bars: int = 15,
    cmap: str = "Reds",
    color: str = "#3b6ea8",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    ax=None,
    show: bool = False,
    save: str | None = None,
):
    """Visualise :func:`omicverse.single.compare_ccc` output.

    Parameters
    ----------
    result
        Dict returned by ``ov.single.compare_ccc``.
    plot_type
        ``'upset'`` intersections of significant interactions;
        ``'jaccard'`` / ``'recall'`` / ``'overlap'`` pairwise set agreement;
        ``'score_corr'`` / ``'lr_corr'`` / ``'cellpair_corr'`` rank correlations;
        ``'top_lr'`` number of significant cell pairs for the top LR pairs of
        every method; ``'cellpair'`` significant interactions per sender-receiver
        pair for each method side by side.
    top_n
        LR pairs shown with ``'top_lr'`` (union of each method's top ``top_n``).
    max_bars
        Intersections shown in the UpSet plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    methods = list(result["methods"])
    fig = None
    if plot_type == "upset":
        n_bars = min(max_bars, max(1, len(set().union(*result["sets"].values()))))
        fig = plt.figure(figsize=figsize or (3.2 + 0.5 * n_bars, 2.4 + 0.4 * len(methods) + 1.6))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, max(2.0, 0.5 * n_bars)],
                              height_ratios=[3, max(1, 0.35 * len(methods))], hspace=0.05, wspace=0.45)
        _upset(result["sets"], fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 0]),
               max_bars, color)
        fig.suptitle(title or "Significant interactions shared between methods", fontsize=11)
    elif plot_type in _MATRICES:
        mat = result[_KEYS[plot_type]]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (1.2 + 1.0 * len(methods), 0.8 + 0.9 * len(methods)))
        else:
            fig = ax.figure
        _matrix_plot(mat, title or _MATRICES[plot_type], ax, cmap, "d" if plot_type == "overlap" else ".2f")
    elif plot_type == "top_lr":
        lrm = result["lr_counts"]
        keep = []
        for m in methods:
            keep += list(lrm[m].sort_values(ascending=False, kind="stable").index[:top_n])
        keep = list(dict.fromkeys(keep))
        sub = lrm.loc[keep, methods]
        sub = sub.loc[sub.sum(axis=1).sort_values(ascending=False, kind="stable").index]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (1.5 + 0.9 * len(methods), 0.8 + 0.28 * len(sub)))
        else:
            fig = ax.figure
        yy, xx = np.meshgrid(np.arange(len(sub)), np.arange(len(methods)), indexing="ij")
        vals = sub.to_numpy(float)
        size = 300 * vals / max(vals.max(), 1)
        sca = ax.scatter(xx.ravel(), yy.ravel(), s=size.ravel(), c=vals.ravel(), cmap=cmap, edgecolor="grey",
                         linewidth=0.3, vmin=0)
        ax.set_xticks(range(len(methods)), methods, rotation=45, ha="right")
        ax.set_yticks(range(len(sub)), sub.index)
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.set_ylim(len(sub) - 0.5, -0.5)
        ax.set_title(title or "Significant cell pairs per LR pair", fontsize=11)
        plt.colorbar(sca, ax=ax, fraction=0.05, pad=0.04, label="cell pairs")
    elif plot_type == "cellpair":
        counts = result["cellpair_counts"]
        n = len(methods)
        fig, axes = plt.subplots(1, n, figsize=figsize or (3.4 * n + 0.6, 3.6), squeeze=False)
        for a, m in zip(axes[0], methods):
            mat = counts[m]
            im = a.imshow(mat.to_numpy(float), cmap=cmap, vmin=0)
            a.set_xticks(range(mat.shape[1]), mat.columns, rotation=90, fontsize=7)
            a.set_yticks(range(mat.shape[0]), mat.index if a is axes[0][0] else [""] * mat.shape[0], fontsize=7)
            a.set_title(m, fontsize=10)
            a.set_xlabel("receiver")
            plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        axes[0][0].set_ylabel("sender")
        fig.suptitle(title or "Significant interactions per cell pair", fontsize=11)
        fig.tight_layout()
    else:
        raise ValueError(f"Unknown plot_type '{plot_type}'.")
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig
