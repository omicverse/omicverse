r"""Plotting helpers for :class:`omicverse.single.CNV` outputs.

Three functions, all reading the unified ``adata.obsm['X_cnv']`` /
``adata.uns['cnv']`` schema that both CopyKAT and inferCNV write into:

* :func:`cnv_heatmap` — cells × genomic-position heatmap with a
  chromosome ideogram bar on top (alternating black/grey rectangles).
* :func:`cnv_summary` — per-bin mean CN as a step plot, gain (red)
  filled above zero and loss (blue) filled below.
* :func:`cnv_umap` — thin wrapper around :func:`omicverse.pl.embedding`
  that reads ``adata.obs['cnv_prediction']`` / ``adata.obs['cnv_score']``.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .._registry import register_function


# ----------------------------------------------------------------------
# colour / ordering helpers
# ----------------------------------------------------------------------


_CNV_CMAP = LinearSegmentedColormap.from_list(
    "ov_cnv", ["#3a5fcd", "#88aedd", "#ffffff", "#e89090", "#cd3a3a"], N=256
)


def _natural_chrom_key(c: str) -> tuple[int, str]:
    """Sort chromosomes 1, 2, ..., 22, X, Y, M naturally."""
    s = str(c).replace("chr", "")
    try:
        return (int(s), "")
    except ValueError:
        order = {"X": 100, "Y": 101, "M": 102, "MT": 102}
        return (order.get(s.upper(), 200), s)


_STANDARD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}
_STANDARD_CHROMS |= {f"{i}" for i in range(1, 23)} | {"X", "Y", "M"}


def _is_standard_chromosome(name: str) -> bool:
    return str(name) in _STANDARD_CHROMS


def _get_cnv_data(adata: AnnData) -> tuple[np.ndarray, dict[str, int], dict]:
    """Read X_cnv + chr_pos from the agreed-upon adata slots."""
    if "X_cnv" not in adata.obsm:
        raise KeyError(
            "adata.obsm['X_cnv'] not found — run ov.single.CNV(...).run() first."
        )
    uns = adata.uns.get("cnv", {})
    chr_pos = uns.get("chr_pos")
    if chr_pos is None or len(chr_pos) == 0:
        raise KeyError(
            "adata.uns['cnv']['chr_pos'] not found — run ov.single.CNV(...).run() first."
        )
    X = np.asarray(adata.obsm["X_cnv"])
    return X, dict(chr_pos), dict(uns)


def _build_chr_segments(
    chr_pos: dict[str, int],
    n_bins: int,
    *,
    standard_only: bool = True,
) -> tuple[list[tuple[str, int, int]], "slice | np.ndarray"]:
    """Sorted [(chrom, start_bin, end_bin), ...] plus the column selector to render.

    The returned segments ALWAYS tile [0, number-of-selected-columns) with no
    holes: segment coordinates are rebased onto the compacted axis of exactly
    the columns picked by the second return value. This holds even when
    non-standard scaffolds / alt contigs are interleaved *between* standard
    chromosomes — previously such interleaving left an uncovered hole in the
    middle of the rendered range, so a chromosome past the gap rendered only
    partially / shifted to one side.

    When ``standard_only`` is True (default), unplaced scaffolds / alt contigs
    are dropped from BOTH the ideogram and the rendered columns. The second
    return is a selector applied to the ``X_cnv`` columns: a ``slice`` when
    nothing is dropped, else an integer index array of the kept columns.
    """
    pairs = sorted(chr_pos.items(), key=lambda kv: kv[1])
    spans: list[tuple[str, int, int]] = []
    for i, (chrom, start) in enumerate(pairs):
        end = pairs[i + 1][1] if i + 1 < len(pairs) else n_bins
        spans.append((str(chrom), int(start), int(end)))

    if not standard_only:
        return spans, slice(0, n_bins)

    kept = [sp for sp in spans if _is_standard_chromosome(sp[0])]
    if not kept:
        return spans, slice(0, n_bins)

    # Compact to exactly the kept chromosomes' columns; rebase segment
    # coordinates onto that compacted axis so segments tile [0, kept_width)
    # with no holes (robust to interleaved non-standard scaffolds). The old
    # min/max-envelope "contiguous" check was tautological and returned
    # segments that did not tile the rendered columns when a scaffold sat
    # between two standard chromosomes.
    mask = np.zeros(int(n_bins), dtype=bool)
    for _, s, e in kept:
        mask[s:e] = True
    col_index = np.flatnonzero(mask)

    rebased: list[tuple[str, int, int]] = []
    run = 0
    for c, s, e in kept:
        w = int(e - s)
        rebased.append((c, run, run + w))
        run += w
    # Common case (nothing dropped): return a slice so the caller keeps a view
    # of X_cnv instead of triggering a full-matrix copy via fancy indexing.
    if col_index.size == n_bins:
        return rebased, slice(0, n_bins)
    return rebased, col_index


_CENTROMERE_DIR = Path(__file__).parent / "_data"


_GENOME_ALIASES = {
    "hg38": "hg38", "grch38": "hg38", "hg20": "hg38",
    "hg19": "hg19", "grch37": "hg19",
    "mm10": "mm10", "grcm38": "mm10",
}


@functools.lru_cache(maxsize=8)
def _load_centromeres(genome: str) -> dict[str, int]:
    """Per-chromosome centromere bp for p/q arm splitting (empty if unknown).

    Data vendored under ``_data/`` (see its README + ``generate_centromeres.py``).
    Bundled: hg38/GRCh38, hg19/GRCh37, mm10/GRCm38. Any other genome returns an
    empty mapping so the caller falls back to whole-chromosome blocks.
    """
    key = _GENOME_ALIASES.get(str(genome).strip().lower())
    if key is None:
        return {}
    path = _CENTROMERE_DIR / f"centromere_{key}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {str(c): int(v) for c, v in zip(df["chromosome"], df["centromere"])}


def _centromere_for(chrom: str, centromeres: dict[str, int]):
    """Look up a chromosome's centromere, tolerating 'chr'-prefixed/bare names."""
    c = str(chrom)
    if c in centromeres:
        return centromeres[c]
    alt = c[3:] if c.startswith("chr") else f"chr{c}"
    return centromeres.get(alt)


def _split_segments_by_arm(
    segments: Sequence[tuple[str, int, int]],
    bin_starts: np.ndarray,
    centromeres: dict[str, int],
) -> list[tuple[str, int, int]]:
    """Split each chromosome segment at its centromere into p/q arm segments.

    ``segments`` are ``(chrom, start, end)`` in the *rendered* (compacted) column
    space; ``bin_starts`` is the per-rendered-bin genomic start (same length /
    order as the rendered columns). Segments whose chromosome has no known
    centromere — or that fall entirely on one arm (e.g. acrocentric q-only) —
    are returned as a single arm-labelled segment. The result still tiles
    ``[0, width)`` with no holes.
    """
    out: list[tuple[str, int, int]] = []
    for chrom, s, e in segments:
        cen = _centromere_for(chrom, centromeres)
        if cen is None:
            out.append((chrom, s, e))
            continue
        starts = np.asarray(bin_starts[s:e])
        n_p = int((starts < cen).sum())
        if n_p <= 0:
            out.append((f"{chrom}q", s, e))
        elif n_p >= (e - s):
            out.append((f"{chrom}p", s, e))
        else:
            out.append((f"{chrom}p", s, s + n_p))
            out.append((f"{chrom}q", s + n_p, e))
    return out


def _draw_ideogram(
    ax: Axes,
    segments: Sequence[tuple[str, int, int]],
    *,
    arm_segments: Sequence[tuple[str, int, int]] | None = None,
    height: float = 1.0,
    fontsize: int = 8,
    label_short: bool = True,
) -> None:
    """Render an ideogram-style chromosome bar onto ``ax``.

    ``segments`` are chromosome-level blocks (one rectangle + label each). When
    ``arm_segments`` is given, a faint centromere tick is drawn at each p/q
    boundary inside a chromosome and tiny ``p`` / ``q`` labels are placed,
    without splitting the chromosome rectangle.
    """
    total_width = segments[-1][2] - segments[0][1]
    ax.set_xlim(segments[0][1], segments[-1][2])
    ax.set_ylim(0, height)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    light = "#d8d8d8"
    dark = "#444444"
    # Skip labels on segments narrower than this fraction of the total width —
    # avoids overlapping "20 21 22" labels at the right edge.
    min_label_frac = 0.018
    for i, (chrom, start, end) in enumerate(segments):
        color = light if i % 2 == 0 else dark
        ax.add_patch(
            Rectangle(
                (start, 0), end - start, height, facecolor=color, edgecolor="none"
            )
        )
        if end - start <= 0:
            continue
        if (end - start) / max(total_width, 1) < min_label_frac:
            continue
        label = chrom.replace("chr", "") if label_short else chrom
        ax.text(
            (start + end) / 2,
            height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="white" if i % 2 else "black",
        )

    # Centromere ticks + tiny p/q labels inside each chromosome (no split).
    if arm_segments is not None:
        chrom_ends = {e for _, _, e in segments}
        for chrom, start, end in arm_segments:
            if (end - start) / max(total_width, 1) < min_label_frac / 1.5:
                continue
            arm = chrom[-1] if chrom[-1:] in ("p", "q") else ""
            if arm:
                ax.text(
                    (start + end) / 2, height * 0.5, arm,
                    ha="center", va="center", fontsize=max(fontsize - 3, 4),
                    color="0.25",
                )
            if end not in chrom_ends:  # internal centromere boundary
                ax.axvline(end, color="0.2", linewidth=0.5, alpha=0.7)


def _order_rows_by_group(
    adata: AnnData, groupby: Optional[str]
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Permutation that buckets rows by groupby (preserving within-bucket order).

    Returns ``(row_order, group_segments)`` where ``group_segments`` is a list
    of ``(label, row_start, row_end)`` triples (for drawing group separators).
    """
    if groupby is None or groupby not in adata.obs:
        order = np.arange(adata.n_obs)
        return order, [("", 0, adata.n_obs)]
    series = adata.obs[groupby]
    # Stable groupby preserves NaN at the end.
    cats = pd.Categorical(series).categories.tolist()
    order: list[int] = []
    segs: list[tuple[str, int, int]] = []
    for c in cats:
        idx = np.where(series.values == c)[0]
        if idx.size == 0:
            continue
        start = len(order)
        order.extend(idx.tolist())
        segs.append((str(c), start, start + idx.size))
    # Any cells with NaN groupby value go last
    nan_idx = np.where(series.isna().values)[0]
    if nan_idx.size:
        start = len(order)
        order.extend(nan_idx.tolist())
        segs.append(("NA", start, start + nan_idx.size))
    return np.asarray(order, dtype=int), segs


def _to_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _categorical_palette(
    series: pd.Series, adata: AnnData, key: str, default_palette
) -> dict:
    """Pick a {category → colour} mapping for a categorical column.

    Re-uses ``adata.uns[f'{key}_colors']`` if scanpy / omicverse already
    assigned colours to this column; else samples from the supplied palette.
    """
    cats = pd.Categorical(series).categories.tolist()
    cat_colors_key = f"{key}_colors"
    if cat_colors_key in adata.uns:
        existing = list(adata.uns[cat_colors_key])
        if len(existing) >= len(cats):
            return {c: existing[i] for i, c in enumerate(cats)}
    pal = list(default_palette)
    return {c: pal[i % len(pal)] for i, c in enumerate(cats)}


def _has_marsilea() -> bool:
    try:
        import marsilea  # noqa: F401
        return True
    except ImportError:
        return False


# ----------------------------------------------------------------------
# cnv_heatmap
# ----------------------------------------------------------------------


@register_function(
    aliases=["拷贝数热图", "cnv_heatmap", "chromosome_heatmap"],
    category="pl",
    description=(
        "Genome-wide CNV heatmap: cells × ordered genomic bins, gain in red "
        "and loss in blue, with an alternating chromosome ideogram bar on top, "
        "row ordering by `groupby` (one adata.obs column) plus optional extra "
        "coloured annotation strips for `annotations` columns. Uses marsilea "
        "when available; matplotlib fallback otherwise."
    ),
    examples=[
        "ov.pl.cnv_heatmap(adata)",
        "ov.pl.cnv_heatmap(adata, groupby='cnv_prediction')",
        "# rows ordered by cnv_prediction; cell_type drawn as a 2nd colour strip",
        "ov.pl.cnv_heatmap(adata, groupby='cnv_prediction', annotations=['cell_type'])",
    ],
    related=["pl.cnv_summary", "pl.cnv_umap", "single.CNV"],
)
def cnv_heatmap(
    adata: AnnData,
    *,
    groupby: Optional[str] = None,
    annotations: Union[str, Sequence[str], None] = None,
    max_value: Optional[float] = None,
    figsize: tuple[float, float] = (8.0, 4.5),
    cmap=_CNV_CMAP,
    standard_chromosomes_only: bool = True,
    split_arms: bool = False,
    genome: str = "hg38",
    backend: str = "auto",
    title: Optional[str] = None,
    show: bool = True,
):
    r"""Plot the genome-wide CNV heatmap.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obsm['X_cnv']`` and ``adata.uns['cnv']`` —
        populated by :class:`omicverse.single.CNV`.
    groupby : str or None
        Single column in ``adata.obs`` that controls **row ordering** (cells
        are bucketed by this category, preserving within-bucket order) and
        the group-divider lines. To compare cells under multiple labellings,
        call ``cnv_heatmap`` once per labelling rather than mixing them
        here; alternatively use ``annotations`` (below) to layer extra
        colour strips without affecting the order.
    annotations : str, list of str, or None
        Additional ``adata.obs`` columns drawn as coloured strips on the
        **left** of the heatmap, in the order given. These never change the
        row order — they are decorative overlays so the reader can see how a
        secondary category (e.g. ``cell_type``) lines up against the primary
        grouping. ``groupby`` is automatically prepended if not already in
        the list.
    max_value : float or None
        Symmetric colour limit. If ``None``, set to the 99th percentile of
        ``|X_cnv|``.
    figsize : tuple
        Figure size in inches.
    cmap : matplotlib Colormap
        Diverging colormap. Default is a blue/white/red palette.
    standard_chromosomes_only : bool, default True
        Drop unplaced scaffolds / alt contigs from the plot.
    split_arms : bool, default False
        Split each chromosome at its centromere into separate ``p`` / ``q`` arm
        segments (labelled e.g. ``8p`` / ``8q``), so arm-level CNAs are
        distinguishable. Requires per-bin genomic coordinates in
        ``adata.uns['cnv']['bin_meta']`` (written by ``ov.single.CNV``) and a
        centromere table for ``genome``; falls back to whole-chromosome blocks
        if either is missing.
    genome : str, default 'hg38'
        Genome build for centromere coordinates when ``split_arms=True``.
        Bundled: ``'hg38'``/``'GRCh38'``, ``'hg19'``/``'GRCh37'``,
        ``'mm10'``/``'GRCm38'`` (add more via ``_data/generate_centromeres.py``).
        **Must match your data** — a mismatched build mis-places arm boundaries,
        and an unknown build is silently skipped (whole chromosomes drawn). Note
        mouse (mm10) is telocentric, so its p arm is negligible.
    backend : {'auto', 'marsilea', 'matplotlib'}
        ``'auto'`` picks marsilea when installed (recommended for clean
        categorical legends), else falls back to the matplotlib renderer.
    title : str or None
        Optional title.
    show : bool, default True
        Whether to render the figure (marsilea only).

    Returns
    -------
    Marsilea backend: ``marsilea.Heatmap`` (call ``.figure`` for the Figure).
    Matplotlib backend: ``(fig, axes_dict)``.

    Examples
    --------
    >>> import omicverse as ov
    >>> cnv = ov.single.CNV(adata, method='copykat').run()
    >>> # rows ordered by aneuploid / diploid:
    >>> ov.pl.cnv_heatmap(adata, groupby='cnv_prediction')
    >>> # same ordering, with cell_type overlaid as a 2nd colour strip:
    >>> ov.pl.cnv_heatmap(adata, groupby='cnv_prediction',
    ...                   annotations=['cell_type'])
    """
    X, chr_pos, uns = _get_cnv_data(adata)
    n_cells, n_bins_total = X.shape
    chrom_segments, col_slice = _build_chr_segments(
        chr_pos, n_bins_total, standard_only=standard_chromosomes_only
    )
    X = X[:, col_slice]
    n_bins = X.shape[1]

    # Optionally compute p/q arm sub-segments (centromere split) for an arm
    # annotation strip. Column GROUPING stays at the chromosome level, so the
    # two arms of one chromosome render contiguously (no gap) and only different
    # chromosomes are separated — the arm boundary is shown as a thin line /
    # colour band inside each chromosome.
    arm_segments = None
    if split_arms:
        bin_meta = uns.get("bin_meta")
        centromeres = _load_centromeres(genome)
        if (
            bin_meta is not None
            and centromeres
            and len(bin_meta) == n_bins_total
            and "start" in getattr(bin_meta, "columns", [])
        ):
            bm = bin_meta.iloc[col_slice]
            arm_segments = _split_segments_by_arm(
                chrom_segments, bm["start"].to_numpy(), centromeres
            )

    # Annotation strip order: groupby first (so it visually anchors the row
    # ordering), then the rest of `annotations` (any duplicate removed).
    ann_list = _to_list(annotations)
    strips: list[str] = []
    if groupby is not None:
        strips.append(groupby)
    for a in ann_list:
        if a != groupby and a not in strips:
            strips.append(a)

    row_order, group_segs = _order_rows_by_group(adata, groupby)
    X_ord = X[row_order]

    if max_value is None:
        max_value = float(np.nanpercentile(np.abs(X_ord), 99))
        if max_value <= 0:
            max_value = 1.0

    use_marsilea = backend == "marsilea" or (backend == "auto" and _has_marsilea())
    if backend == "marsilea" and not _has_marsilea():
        raise ImportError(
            "backend='marsilea' but the marsilea package is not installed.\n"
            "Install with `pip install marsilea` (or pass backend='matplotlib')."
        )

    if use_marsilea:
        return _cnv_heatmap_marsilea(
            adata,
            X_ord=X_ord,
            chrom_segments=chrom_segments,
            arm_segments=arm_segments,
            row_order=row_order,
            strips=strips,
            max_value=max_value,
            figsize=figsize,
            cmap=cmap,
            title=title,
            show=show,
        )

    # --- matplotlib fallback ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 1, height_ratios=(0.5, 9.5), hspace=0.05, top=0.92, bottom=0.10
    )
    ideo_ax = fig.add_subplot(gs[0, 0])
    hm_ax = fig.add_subplot(gs[1, 0], sharex=ideo_ax)

    _draw_ideogram(ideo_ax, chrom_segments, arm_segments=arm_segments)

    norm = Normalize(vmin=-max_value, vmax=max_value)
    hm_ax.imshow(
        X_ord,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=(0, n_bins, n_cells, 0),
    )

    # Chromosome dividers — thin white lines between different chromosomes only.
    for _, _, end in chrom_segments[:-1]:
        hm_ax.axvline(end, color="white", linewidth=0.6, alpha=0.8)
    # Centromere dividers — faint dotted lines at the p/q boundary inside a
    # chromosome (arm-split mode); the two arms stay contiguous (no gap).
    if arm_segments is not None:
        chrom_ends = {e for _, _, e in chrom_segments}
        for _, _, end in arm_segments[:-1]:
            if end not in chrom_ends:
                hm_ax.axvline(end, color="0.45", linewidth=0.4, alpha=0.6, linestyle=(0, (1, 1)))

    # Group dividers — horizontal black lines + left-side labels.
    if groupby is not None and len(group_segs) > 1:
        for _, _, row_end in group_segs[:-1]:
            hm_ax.axhline(row_end, color="black", linewidth=0.8)
        for label, row_start, row_end in group_segs:
            hm_ax.text(
                -n_bins * 0.01,
                (row_start + row_end) / 2,
                label,
                ha="right",
                va="center",
                fontsize=9,
                rotation=0,
            )

    hm_ax.set_xticks([])
    hm_ax.set_yticks([])
    hm_ax.set_xlabel("Ordered genomic positions")
    hm_ax.set_ylabel(f"{n_cells:,} cells")
    for spine in hm_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    if title:
        fig.suptitle(title, fontsize=11, y=0.96)

    # Colour bar on the right.
    cax = fig.add_axes(
        [hm_ax.get_position().x1 + 0.01, hm_ax.get_position().y0,
         0.012, hm_ax.get_position().height * 0.4]
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="CN (log ratio)"
    )

    return fig, {"heatmap": hm_ax, "ideogram": ideo_ax, "cbar": cax}


def _cnv_heatmap_marsilea(
    adata: AnnData,
    *,
    X_ord: np.ndarray,
    chrom_segments: list[tuple[str, int, int]],
    arm_segments: list[tuple[str, int, int]] | None,
    row_order: np.ndarray,
    strips: list[str],
    max_value: float,
    figsize: tuple[float, float],
    cmap,
    title: Optional[str],
    show: bool,
):
    """Marsilea-backed heatmap with chromosome ideogram + multi-strip row annotations.

    Columns are grouped by CHROMOSOME (a thin separator between different
    chromosomes, original-API style). When ``arm_segments`` is given, a p/q
    colour band is drawn inside each chromosome so the two arms stay contiguous
    (no inter-arm gap) while the centromere boundary remains visible.
    """
    import marsilea as ma
    import marsilea.plotter as mp

    from ._palette import palette_28, palette_56

    n_cells, n_bins = X_ord.shape

    # Per-bin chromosome (drives grouping, alternating colour strip, labels).
    chrom_per_bin = np.empty(n_bins, dtype=object)
    for chrom, start, end in chrom_segments:
        chrom_per_bin[start:end] = chrom.replace("chr", "")
    ideo_palette: dict[str, str] = {}
    for i, (chrom, _, _) in enumerate(chrom_segments):
        ideo_palette[chrom.replace("chr", "")] = "#d8d8d8" if i % 2 == 0 else "#444444"

    h = ma.Heatmap(
        X_ord,
        width=float(figsize[0]),
        height=float(figsize[1]),
        cmap=cmap,
        vmin=-max_value,
        vmax=max_value,
        label="CN (log ratio)",
    )

    # Arm band (closest to the heatmap): p/q two-tone inside each chromosome.
    if arm_segments is not None:
        arm_per_bin = np.empty(n_bins, dtype=object)
        for chrom, start, end in arm_segments:
            arm_per_bin[start:end] = chrom[-1] if chrom[-1:] in ("p", "q") else "·"
        h.add_top(
            mp.Colors(arm_per_bin, palette={"p": "#c9c9c9", "q": "#7a7a7a", "·": "#ffffff"},
                      label="arm"),
            size=0.10,
            pad=0.0,
            legend=True,
        )

    # Chromosome alternating colour strip + chunk labels.
    h.add_top(
        mp.Colors(chrom_per_bin, palette=ideo_palette),
        size=0.16,
        pad=0.0,
        legend=False,
    )
    chunk_labels = [c.replace("chr", "") for c, _, _ in chrom_segments]
    # Group by chromosome with a thin separator (original-API style): arms of
    # the same chromosome stay together; only different chromosomes are split.
    h.group_cols(chrom_per_bin, order=chunk_labels, spacing=0.006)
    h.add_top(
        mp.Chunk(chunk_labels, fill_colors=None, fontsize=8, rotation=0),
        size=0.12,
        pad=0.02,
    )

    # Left: one colour strip per requested column. The first one is `groupby`
    # (which controls row ordering) — anchored flush against the heatmap;
    # subsequent strips are decorative `annotations`.
    default_pal_28 = list(palette_28)
    default_pal_56 = list(palette_56)
    for i, key in enumerate(strips):
        if key not in adata.obs:
            continue
        values = adata.obs[key].astype(object).fillna("NA").values[row_order]
        n_cats = len(pd.unique(values))
        base_palette = default_pal_28 if n_cats <= len(default_pal_28) else default_pal_56
        cat_palette = _categorical_palette(pd.Series(values), adata, key, base_palette)
        h.add_left(
            mp.Colors(values, palette=cat_palette, label=key),
            size=0.18,
            pad=0.0 if i == 0 else 0.02,
            legend=True,
        )

    h.add_legends()
    if title:
        h.add_title(title, fontsize=11)
    h.render()
    return h


# ----------------------------------------------------------------------
# cnv_summary
# ----------------------------------------------------------------------


@register_function(
    aliases=["拷贝数概览", "cnv_summary", "chromosome_heatmap_summary"],
    category="pl",
    description=(
        "Per-bin mean CN as a step plot — gain (red, above zero) and loss "
        "(blue, below zero) filled to baseline; companion to cnv_heatmap. "
        "Use groupby='cnv_prediction' to plot the aneuploid mean."
    ),
    examples=[
        "ov.pl.cnv_summary(adata)",
        "ov.pl.cnv_summary(adata, groupby='cnv_prediction', subset='aneuploid')",
    ],
    related=["pl.cnv_heatmap", "pl.cnv_umap"],
)
def cnv_summary(
    adata: AnnData,
    *,
    groupby: Optional[str] = None,
    subset: Union[str, Sequence[str], None] = None,
    figsize: tuple[float, float] = (8.0, 2.5),
    gain_color: str = "#cd3a3a",
    loss_color: str = "#3a5fcd",
    ylim: Optional[tuple[float, float]] = None,
    standard_chromosomes_only: bool = True,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> tuple[Figure, dict[str, Axes]]:
    r"""Plot mean CN per genomic bin as a filled step plot.

    Parameters
    ----------
    adata : AnnData
        Output of :class:`omicverse.single.CNV`.
    groupby : str or None
        ``adata.obs`` column. If given, cells are filtered to ``subset``.
    subset : str or list of str or None
        Category value(s) in ``adata.obs[groupby]`` to restrict to (e.g.
        ``'aneuploid'``). If ``None``, the mean is over all cells.
    figsize : tuple
        Figure size in inches.
    gain_color / loss_color : str
        Fill colours for above/below zero deviations.
    ylim : tuple or None
        Y-axis limits. Defaults to symmetric around the largest |mean|.
    ax : matplotlib Axes or None
        Pre-allocated axes. If ``None``, a new figure is created.
    title : str or None
        Optional title.

    Returns
    -------
    fig, axes
    """
    X, chr_pos, _ = _get_cnv_data(adata)
    n_bins_total = X.shape[1]
    segments, col_slice = _build_chr_segments(
        chr_pos, n_bins_total, standard_only=standard_chromosomes_only
    )
    X = X[:, col_slice]
    n_bins = X.shape[1]

    rows = np.arange(adata.n_obs)
    if groupby is not None and subset is not None:
        if groupby not in adata.obs:
            raise KeyError(f"adata.obs[{groupby!r}] not found")
        target = {subset} if isinstance(subset, str) else set(subset)
        rows = np.where(adata.obs[groupby].astype(str).isin(target).values)[0]
        if rows.size == 0:
            raise ValueError(f"no cells with {groupby}={subset}")

    mean_cn = np.nanmean(X[rows, :], axis=0)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2, 1, height_ratios=(0.5, 9.5), hspace=0.05, top=0.90, bottom=0.18
        )
        ideo_ax = fig.add_subplot(gs[0, 0])
        line_ax = fig.add_subplot(gs[1, 0], sharex=ideo_ax)
    else:
        line_ax = ax
        fig = line_ax.figure
        bbox = line_ax.get_position()
        ideo_ax = fig.add_axes(
            [bbox.x0, bbox.y1 + 0.005, bbox.width, bbox.height * 0.07]
        )

    _draw_ideogram(ideo_ax, segments)

    x = np.arange(n_bins)
    pos = np.maximum(mean_cn, 0)
    neg = np.minimum(mean_cn, 0)
    line_ax.fill_between(x, 0, pos, step="post", color=gain_color, alpha=0.85, linewidth=0)
    line_ax.fill_between(x, 0, neg, step="post", color=loss_color, alpha=0.85, linewidth=0)
    line_ax.step(x, mean_cn, where="post", color="black", linewidth=0.4)
    line_ax.axhline(0, color="black", linewidth=0.5)

    # Chromosome dividers.
    for _, _, end in segments[:-1]:
        line_ax.axvline(end, color="grey", linewidth=0.3, alpha=0.5)

    if ylim is None:
        m = float(np.nanmax(np.abs(mean_cn))) or 1.0
        ylim = (-m * 1.15, m * 1.15)
    line_ax.set_ylim(*ylim)
    line_ax.set_xlim(0, n_bins)
    line_ax.set_xticks([])
    line_ax.set_xlabel("Ordered genomic positions")
    line_ax.set_ylabel("Mean CN (log ratio)")
    for spine_name in ("top", "right"):
        line_ax.spines[spine_name].set_visible(False)
    line_ax.spines["left"].set_linewidth(0.5)
    line_ax.spines["bottom"].set_linewidth(0.5)

    if title:
        fig.suptitle(title, fontsize=11, y=0.97)

    return fig, {"line": line_ax, "ideogram": ideo_ax}


# ----------------------------------------------------------------------
# cnv_umap
# ----------------------------------------------------------------------


@register_function(
    aliases=["拷贝数umap", "cnv_umap"],
    category="pl",
    description=(
        "UMAP coloured by CNV outputs — convenience wrapper around "
        "ov.pl.embedding that defaults to colouring by ['cnv_prediction', "
        "'cnv_score'] (the two columns ov.single.CNV writes)."
    ),
    examples=[
        "ov.pl.cnv_umap(adata)",
        "ov.pl.cnv_umap(adata, color='cnv_score', cmap='Purples')",
    ],
    related=["pl.cnv_heatmap", "pl.cnv_summary", "single.CNV", "pl.embedding"],
)
def cnv_umap(
    adata: AnnData,
    *,
    color: Union[str, Sequence[str], None] = None,
    basis: str = "X_umap",
    cmap: str = "Purples",
    **kwargs,
):
    r"""UMAP coloured by CNV outputs.

    Defaults to side-by-side panels of ``cnv_prediction`` (tumour vs normal
    classification, when present) and ``cnv_score`` (per-cell mean |CN|).
    Forwards everything else to :func:`omicverse.pl.embedding`.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obsm[basis]`` (default ``X_umap``) and any
        of ``cnv_prediction`` / ``cnv_score`` in ``adata.obs``.
    color : str, list of str, or None
        Override the default colouring (``['cnv_prediction', 'cnv_score']``
        when both are present, else whichever exists).
    basis : str
        Embedding key (``adata.obsm`` key). Default ``'X_umap'``.
    cmap : str
        Colormap for continuous (``cnv_score``) panels.
    **kwargs
        Forwarded to :func:`omicverse.pl.embedding`.

    Returns
    -------
    Whatever :func:`omicverse.pl.embedding` returns (``(fig, ax)`` or ``ax``,
    depending on its current signature).
    """
    from ._single import embedding

    if color is None:
        candidates = []
        if "cnv_prediction" in adata.obs and not adata.obs["cnv_prediction"].isna().all():
            candidates.append("cnv_prediction")
        if "cnv_score" in adata.obs:
            candidates.append("cnv_score")
        if not candidates:
            raise KeyError(
                "neither 'cnv_prediction' nor 'cnv_score' found in adata.obs — "
                "run ov.single.CNV(...).run() first."
            )
        color = candidates if len(candidates) > 1 else candidates[0]

    kwargs.setdefault("cmap", cmap)
    return embedding(adata, basis=basis, color=color, **kwargs)
