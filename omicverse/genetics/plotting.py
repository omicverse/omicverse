"""Plotting for ``ov.genetics`` — matplotlib visualisations.

Standard statistical-genetics figures: the genome-wide Manhattan plot
(:func:`manhattan`), the Q-Q plot with genomic inflation
(:func:`qqplot`), the LocusZoom-style regional association plot
(:func:`regional_plot`), the colocalization plot (:func:`coloc_plot`),
the Mendelian-randomization scatter / forest plots
(:func:`mr_scatter`, :func:`mr_forest`) and the SuSiE fine-mapping plot
(:func:`finemap_plot`). The MR and fine-mapping plots delegate to the
backends' own plotting routines.

Pipeline composites — multi-panel figures assembled for the GWAS
follow-up workflow — live here too so that tutorials stay one-liners:
the sample-QC distributions (:func:`sample_qc_plot`), the genotype-PCA
structure view (:func:`pca_structure_plot`), the fine-mapping locus view
(:func:`finemap_locus_plot`), the gene-level TWAS Manhattan
(:func:`twas_manhattan`) and the scDRS score-by-cell-type panels
(:func:`scdrs_celltype_plot`), plus the MR exposure-vs-outcome scatter
(:func:`mr_effect_plot`) for callers who have raw effect arrays rather
than a backend MRInput.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .._registry import register_function


def _coerce_assoc(data, snp, chrom, pos, pvalue):
    """Pull SNP / chromosome / position / p-value columns from a DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    cols = {c.lower(): c for c in data.columns}

    def _pick(explicit, candidates, required=True):
        if explicit is not None:
            return explicit
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        if required:
            raise KeyError(
                f"could not find a column among {candidates}; pass it "
                "explicitly."
            )
        return None

    p_col = _pick(pvalue, ["pvalue", "p", "pval", "p.value", "p_value"])
    snp_col = _pick(snp, ["snp", "snps", "rsid", "variant", "id"], required=False)
    chr_col = _pick(chrom, ["chr", "chrom", "chromosome"], required=False)
    pos_col = _pick(pos, ["bp", "pos", "position"], required=False)
    return snp_col, chr_col, pos_col, p_col


@register_function(
    aliases=[
        "manhattan", "manhattan_plot", "gwas_plot", "曼哈顿图", "曼哈顿绘图",
    ],
    category="genetics",
    description=(
        "Genome-wide Manhattan plot of association p-values for GWAS, "
        "eQTL or TWAS results. Plots -log10(p) against genomic position, "
        "alternating colours by chromosome, with an optional "
        "genome-wide-significance line. Accepts any results DataFrame "
        "with a p-value column (and optional chromosome / position). "
        "matplotlib."
    ),
    examples=[
        "ov.genetics.manhattan(gwas_res)",
        "ov.genetics.manhattan(gwas_res, chrom='CHR', pos='BP', pvalue='P')",
    ],
    related=["ov.genetics.qqplot", "ov.genetics.regional_plot",
             "ov.genetics.gwas_association"],
)
def manhattan(
    data: pd.DataFrame,
    *,
    snp: Optional[str] = None,
    chrom: Optional[str] = None,
    pos: Optional[str] = None,
    pvalue: Optional[str] = None,
    sig_line: float = 5e-8,
    suggestive_line: Optional[float] = 1e-5,
    ax=None,
    title: Optional[str] = None,
    colors=("#3b6fb6", "#9bbce0"),
):
    """Draw a Manhattan plot.

    Parameters
    ----------
    data
        Association results — a DataFrame with a p-value column (plus
        optional chromosome / position columns).
    snp, chrom, pos, pvalue
        Column names; auto-detected when not given.
    sig_line
        Genome-wide-significance threshold (drawn as a dashed line).
    suggestive_line
        Optional suggestive-significance threshold.
    ax
        Existing matplotlib Axes; a new one is created if ``None``.
    title
        Optional plot title.
    colors
        Two alternating chromosome colours.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes.
    """
    import matplotlib.pyplot as plt

    snp_col, chr_col, pos_col, p_col = _coerce_assoc(
        data, snp, chrom, pos, pvalue
    )
    df = data.copy()
    df = df[np.isfinite(pd.to_numeric(df[p_col], errors="coerce"))]
    pvals = pd.to_numeric(df[p_col], errors="coerce").to_numpy()
    pvals = np.clip(pvals, np.finfo(float).tiny, 1.0)
    logp = -np.log10(pvals)

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))

    if chr_col is not None:
        df = df.assign(_logp=logp)
        chrom_vals = df[chr_col].astype(str)
        # Sort chromosomes numerically where possible.
        def _chrom_key(c):
            c = c.replace("chr", "")
            return (0, int(c)) if c.isdigit() else (1, c)
        order = sorted(chrom_vals.unique(), key=_chrom_key)
        x = np.zeros(len(df))
        offset = 0.0
        ticks, ticklabels = [], []
        for i, c in enumerate(order):
            mask = (chrom_vals == c).to_numpy()
            n = int(mask.sum())
            if pos_col is not None:
                pp = pd.to_numeric(df.loc[mask, pos_col], errors="coerce")
                pp = pp.fillna(pp.median() if n else 0).to_numpy()
                pp = pp - pp.min() if n else pp
            else:
                pp = np.arange(n, dtype=float)
            span = (pp.max() - pp.min()) if (n and pp.max() > pp.min()) else max(n, 1)
            x[mask] = offset + pp
            ticks.append(offset + span / 2.0)
            ticklabels.append(c)
            ax.scatter(x[mask], df["_logp"].to_numpy()[mask], s=8,
                       c=colors[i % 2], rasterized=True)
            offset += span * 1.05
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=8)
        ax.set_xlabel("Chromosome")
    else:
        x = np.arange(len(logp), dtype=float)
        ax.scatter(x, logp, s=8, c=colors[0], rasterized=True)
        ax.set_xlabel("Variant index")

    if sig_line:
        ax.axhline(-np.log10(sig_line), color="#d62728", ls="--", lw=1,
                   label=f"genome-wide (p={sig_line:g})")
    if suggestive_line:
        ax.axhline(-np.log10(suggestive_line), color="#7f7f7f", ls=":",
                   lw=1, label=f"suggestive (p={suggestive_line:g})")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


@register_function(
    aliases=[
        "qqplot", "qq_plot", "quantile_quantile_plot", "QQ图", "分位数图",
    ],
    category="genetics",
    description=(
        "Quantile-quantile (Q-Q) plot of GWAS / eQTL association "
        "p-values against the uniform null, annotated with the "
        "genomic-inflation factor lambda GC. Departure of the bulk of "
        "points from the diagonal indicates inflation; an early "
        "departure at the tail indicates true signal. matplotlib."
    ),
    examples=[
        "ov.genetics.qqplot(gwas_res['pvalue'])",
        "ov.genetics.qqplot(gwas_res, pvalue='P')",
    ],
    related=["ov.genetics.manhattan", "ov.genetics.genomic_inflation"],
)
def qqplot(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    *,
    pvalue: Optional[str] = None,
    ax=None,
    title: Optional[str] = None,
    color: str = "#3b6fb6",
):
    """Draw a Q-Q plot of association p-values.

    Parameters
    ----------
    data
        A p-value vector, or a DataFrame with a p-value column.
    pvalue
        Column name when ``data`` is a DataFrame (auto-detected if
        ``None``).
    ax
        Existing matplotlib Axes; a new one is created if ``None``.
    title
        Optional plot title.
    color
        Point colour.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes.
    """
    import matplotlib.pyplot as plt

    from ._gwas import genomic_inflation

    if isinstance(data, pd.DataFrame):
        _, _, _, p_col = _coerce_assoc(data, None, None, None, pvalue)
        pvals = pd.to_numeric(data[p_col], errors="coerce").to_numpy()
    else:
        pvals = np.asarray(data, dtype=float).ravel()

    pvals = pvals[np.isfinite(pvals)]
    pvals = np.clip(pvals, np.finfo(float).tiny, 1.0)
    pvals = np.sort(pvals)
    n = pvals.size
    if n == 0:
        raise ValueError("no finite p-values to plot.")

    expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    observed = -np.log10(pvals)
    lam = genomic_inflation(pvals, statistic="pvalue")

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    lim = max(expected.max(), observed.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#d62728", ls="--", lw=1)
    ax.scatter(expected, observed, s=10, c=color, rasterized=True)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.set_title(title or f"Q-Q plot  ($\\lambda_{{GC}}$ = {lam:.3f})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


# --------------------------------------------------------------------------- #
# LocusZoom helpers — the classic discrete r^2 bins and the gene track.        #
# --------------------------------------------------------------------------- #
# The canonical LocusZoom LD-bin colour scheme (lead = purple diamond).
_LD_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_LD_COLORS = ["#1f2f86", "#6fc2ec", "#5cba5c", "#f4a23b", "#e23b30"]
_LD_LABELS = ["0.0 – 0.2", "0.2 – 0.4", "0.4 – 0.6", "0.6 – 0.8", "0.8 – 1.0"]
_LD_NA_COLOR = "#9b9b9b"


def _ld_bin_color(r2_val):
    """Map an r^2 value onto its LocusZoom discrete-bin colour."""
    if r2_val is None or not np.isfinite(r2_val):
        return _LD_NA_COLOR
    for i in range(len(_LD_COLORS)):
        if r2_val <= _LD_BINS[i + 1] or i == len(_LD_COLORS) - 1:
            return _LD_COLORS[i]
    return _LD_COLORS[-1]


def _draw_gene_track(ax, genes, start, end, *, chrom=None,
                     gene_col="gene", chrom_col="chrom",
                     start_col="start", end_col="end", strand_col="strand"):
    """Draw a LocusZoom gene-model track (arrowed boxes + symbols)."""
    g = genes.copy()
    if chrom is not None and chrom_col in g.columns:
        g = g[g[chrom_col].astype(str) == str(chrom)]
    gs = pd.to_numeric(g[start_col], errors="coerce")
    ge = pd.to_numeric(g[end_col], errors="coerce")
    # Keep any gene overlapping the plotted window.
    g = g[(ge >= start) & (gs <= end)].copy()
    g = g.assign(_s=gs[g.index].clip(lower=start),
                 _e=ge[g.index].clip(upper=end))
    g = g.sort_values("_s").reset_index(drop=True)

    span = max(end - start, 1.0)
    rows_end: list = []  # right-edge bp of the last gene placed on each row
    row_of = []
    for _, row in g.iterrows():
        placed = False
        for ri, redge in enumerate(rows_end):
            if row["_s"] > redge + 0.04 * span:
                rows_end[ri] = row["_e"]
                row_of.append(ri)
                placed = True
                break
        if not placed:
            rows_end.append(row["_e"])
            row_of.append(len(rows_end) - 1)
    n_rows = max(len(rows_end), 1)

    for (_, row), ri in zip(g.iterrows(), row_of):
        y = n_rows - 1 - ri
        strand = str(row.get(strand_col, "+"))
        ax.add_patch(plt_rect((row["_s"], y - 0.18),
                              row["_e"] - row["_s"], 0.36,
                              facecolor="#2f6db4", edgecolor="#1b3f6e",
                              linewidth=0.5))
        # Strand arrow at the gene's transcription-start side.
        amark = ">" if strand == "+" else "<"
        ax_x = row["_e"] if strand == "+" else row["_s"]
        ax.plot([ax_x], [y], marker=amark, ms=5, color="#1b3f6e")
        mid = 0.5 * (row["_s"] + row["_e"])
        ax.text(mid, y + 0.30, str(row[gene_col]), ha="center",
                va="bottom", fontsize=6.5, style="italic")
    ax.set_xlim(start, end)
    ax.set_ylim(-0.7, n_rows - 0.1)
    ax.set_yticks([])
    ax.set_ylabel("Genes", fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)


def plt_rect(xy, width, height, **kw):
    """Thin wrapper around matplotlib's Rectangle (keeps imports local)."""
    from matplotlib.patches import Rectangle
    return Rectangle(xy, width, height, **kw)


@register_function(
    aliases=[
        "regional_plot", "locuszoom", "regional_association_plot",
        "区域关联图", "局部关联图",
    ],
    category="genetics",
    description=(
        "Publication-grade LocusZoom regional-association plot. Zooms into "
        "a single locus and plots -log10(p) against base-pair position. "
        "Given an LD vector it colours SNPs by their r^2 to the lead "
        "variant in the classic five discrete bins (navy -> cyan -> green "
        "-> orange -> red) with an r^2 legend box and draws the lead SNP "
        "as a labelled purple diamond. Given a recombination map it "
        "overlays the recombination-rate (cM/Mb) line on a right-hand "
        "axis; given a gene table it draws an arrowed gene-model track "
        "beneath the scatter. All of those inputs are optional — with "
        "just summary statistics it still draws a plain regional plot. "
        "matplotlib."
    ),
    examples=[
        "ov.genetics.regional_plot(gwas_res, chrom='1', start=1e6, end=2e6)",
        "ov.genetics.regional_plot(gwas_res, lead_snp='rs123', r2=ld, "
        "recomb_map=rmap, genes=gene_models)",
    ],
    related=["ov.genetics.manhattan", "ov.genetics.finemap_plot",
             "ov.genetics.compute_ld_to_lead", "ov.genetics.finemap_locus_plot"],
)
def regional_plot(
    data: pd.DataFrame,
    *,
    chrom: Optional[str] = None,
    pos: Optional[str] = None,
    pvalue: Optional[str] = None,
    snp: Optional[str] = None,
    region_chrom: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    lead_snp: Optional[str] = None,
    r2: Optional[Union[pd.Series, np.ndarray, dict]] = None,
    ld: Optional[Union[pd.Series, np.ndarray, dict]] = None,
    recomb_map: Optional[pd.DataFrame] = None,
    genes: Optional[pd.DataFrame] = None,
    ax=None,
    title: Optional[str] = None,
):
    """Draw a publication LocusZoom regional-association plot.

    Parameters
    ----------
    data
        Association results — must have position and p-value columns.
    chrom, pos, pvalue, snp
        Column names; auto-detected when not given.
    region_chrom, start, end
        Optional region filter (chromosome + base-pair window).
    lead_snp
        Optional lead-variant SNP id (highlighted as a purple diamond
        labelled with its rsID).
    r2, ld
        Optional per-SNP LD (r^2) to the lead variant — a Series / array
        aligned to ``data``, or a ``{snp: r2}`` dict (e.g. from
        :func:`ov.genetics.compute_ld_to_lead`). SNPs are then coloured
        by the classic five discrete r^2 bins. ``ld`` is an alias of
        ``r2``.
    recomb_map
        Optional recombination map — a DataFrame with ``position`` and a
        rate column (``rate_cM_per_Mb`` / ``rate`` / ``recomb_rate``);
        drawn as a line on a twin right axis labelled "Recombination
        rate (cM/Mb)".
    genes
        Optional gene-model table (``gene`` / ``chrom`` / ``start`` /
        ``end`` / ``strand``); drawn as an arrowed gene-track panel
        beneath the scatter (a 2-row gridspec is created).
    ax
        Existing matplotlib Axes for the scatter panel. Ignored when
        ``genes`` is given (a fresh 2-panel figure is built).
    title
        Optional plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The scatter (association) axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    snp_col, chr_col, pos_col, p_col = _coerce_assoc(
        data, snp, chrom, pos, pvalue
    )
    if pos_col is None:
        raise KeyError("regional_plot needs a position column.")
    df = data.copy()
    if region_chrom is not None and chr_col is not None:
        df = df[df[chr_col].astype(str) == str(region_chrom)]
    bp = pd.to_numeric(df[pos_col], errors="coerce")
    if start is not None:
        df = df[bp >= start]
        bp = pd.to_numeric(df[pos_col], errors="coerce")
    if end is not None:
        df = df[bp <= end]
        bp = pd.to_numeric(df[pos_col], errors="coerce")

    pvals = pd.to_numeric(df[p_col], errors="coerce").to_numpy()
    pvals = np.clip(pvals, np.finfo(float).tiny, 1.0)
    logp = -np.log10(pvals)
    bp = bp.to_numpy()
    if bp.size == 0:
        raise ValueError("regional_plot: no variants in the requested region.")
    win_lo = float(start) if start is not None else float(np.nanmin(bp))
    win_hi = float(end) if end is not None else float(np.nanmax(bp))
    if win_hi <= win_lo:
        win_hi = win_lo + 1.0

    r2 = r2 if r2 is not None else ld
    gene_chrom = region_chrom
    if gene_chrom is None and chr_col is not None and len(df):
        gene_chrom = str(df[chr_col].astype(str).iloc[0])

    # Build the figure: a 2-row gridspec when a gene track is requested.
    gene_ax = None
    if genes is not None:
        fig = plt.figure(figsize=(9, 6.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.12)
        ax = fig.add_subplot(gs[0])
        gene_ax = fig.add_subplot(gs[1], sharex=ax)
    elif ax is None:
        _, ax = plt.subplots(figsize=(9, 4.4))

    # Recombination-rate line on a twin right axis (drawn first, behind).
    if recomb_map is not None and len(recomb_map):
        rate_col = next(
            (c for c in ("rate_cM_per_Mb", "rate", "recomb_rate",
                         "cM_per_Mb")
             if c in recomb_map.columns), None)
        pos_rc = next(
            (c for c in ("position", "pos", "bp") if c in recomb_map.columns),
            None)
        if rate_col is not None and pos_rc is not None:
            rm = recomb_map[(recomb_map[pos_rc] >= win_lo)
                            & (recomb_map[pos_rc] <= win_hi)]
            rm = rm.sort_values(pos_rc)
            ax_r = ax.twinx()
            ax_r.fill_between(rm[pos_rc].to_numpy(),
                              rm[rate_col].to_numpy(),
                              color="#9ecae1", alpha=0.35, zorder=0)
            ax_r.plot(rm[pos_rc].to_numpy(), rm[rate_col].to_numpy(),
                      color="#4a90d9", lw=0.9, zorder=1)
            ax_r.set_ylabel("Recombination rate (cM/Mb)", color="#4a90d9")
            ax_r.tick_params(axis="y", colors="#4a90d9")
            ax_r.set_ylim(bottom=0)
            ax_r.spines[["top"]].set_visible(False)

    # Association scatter — discrete LD bins or a flat colour.
    if r2 is not None and snp_col is not None:
        if isinstance(r2, dict) or isinstance(r2, pd.Series):
            r2_vals = df[snp_col].astype(str).map(dict(r2)).to_numpy(
                dtype=float)
        else:
            r2_vals = np.asarray(r2, dtype=float)
        colors = [_ld_bin_color(v) for v in r2_vals]
        ax.scatter(bp, logp, c=colors, s=34, edgecolors="black",
                   linewidths=0.35, zorder=3)
        handles = [Patch(facecolor=c, edgecolor="black", label=l)
                   for c, l in zip(reversed(_LD_COLORS),
                                   reversed(_LD_LABELS))]
        ld_legend = ax.legend(handles=handles, title=r"$r^2$",
                              fontsize=7, title_fontsize=8,
                              loc="upper left", framealpha=0.95,
                              borderpad=0.5, labelspacing=0.25)
        ax.add_artist(ld_legend)
    else:
        ax.scatter(bp, logp, s=24, c="#3b6fb6", edgecolors="grey",
                   linewidths=0.3, zorder=3)

    # Lead SNP — the classic purple diamond with the rsID label.
    if lead_snp is not None and snp_col is not None:
        lead = df[df[snp_col].astype(str) == str(lead_snp)]
        if len(lead):
            lx = float(pd.to_numeric(lead[pos_col], errors="coerce").iloc[0])
            lp = float(-np.log10(np.clip(
                pd.to_numeric(lead[p_col], errors="coerce").iloc[0],
                np.finfo(float).tiny, 1.0)))
            ax.scatter([lx], [lp], marker="D", s=95, c="#7b3fa0",
                       edgecolors="black", linewidths=0.6, zorder=6)
            ax.annotate(str(lead_snp), (lx, lp),
                        textcoords="offset points", xytext=(8, 6),
                        fontsize=8, fontweight="bold", color="#4a2069")

    ax.set_xlim(win_lo, win_hi)
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title)
    ax.spines[["top"]].set_visible(False)

    if gene_ax is not None:
        ax.tick_params(labelbottom=False)
        ax.spines[["bottom"]].set_visible(True)
        _draw_gene_track(gene_ax, genes, win_lo, win_hi, chrom=gene_chrom)
        gene_ax.set_xlabel(
            f"Chromosome {gene_chrom} position (Mb)" if gene_chrom
            else "Position (Mb)")
        gene_ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}"))
    else:
        ax.set_xlabel(
            f"Chromosome {gene_chrom} position (Mb)" if gene_chrom
            else "Position (Mb)")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}"))
    return ax


@register_function(
    aliases=[
        "coloc_plot", "colocalization_plot", "coloc_pp_plot",
        "共定位图", "共定位绘图",
    ],
    category="genetics",
    description=(
        "Plot a colocalization result — a bar chart of the five "
        "posterior probabilities PP.H0..PP.H4 (H4 = a shared causal "
        "variant). Accepts the result object from "
        ":func:`ov.genetics.colocalize`. matplotlib."
    ),
    examples=[
        "ov.genetics.coloc_plot(coloc_result)",
    ],
    related=["ov.genetics.colocalize", "ov.genetics.coloc_sensitivity"],
)
def coloc_plot(result, *, ax=None, title: Optional[str] = None):
    """Plot the PP.H0..H4 posterior probabilities of a coloc result.

    Parameters
    ----------
    result
        A result from :func:`ov.genetics.colocalize` (``method='abf'``) —
        anything carrying a ``summary`` with ``PP.H*.abf`` entries.
    ax
        Existing matplotlib Axes.
    title
        Optional plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes.
    """
    import matplotlib.pyplot as plt

    # pycoloc's ColocABF is a dict subclass — prefer ['summary'], then a
    # ``.summary`` attribute, else treat the object itself as the summary.
    if hasattr(result, "keys") and "summary" in result:
        summary = result["summary"]
    else:
        summary = getattr(result, "summary", result)
    if isinstance(summary, dict):
        summary = pd.Series(summary)
    keys = [f"PP.H{i}.abf" for i in range(5)]
    pp = []
    for k in keys:
        if k in summary:
            pp.append(float(summary[k]))
        else:
            alt = k.replace(".abf", "")
            pp.append(float(summary[alt]) if alt in summary else np.nan)
    pp = np.asarray(pp)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    labels = ["H0\nno assoc", "H1\ntrait 1", "H2\ntrait 2",
              "H3\ndistinct", "H4\nshared"]
    colors = ["#bdbdbd", "#74a9cf", "#74c476", "#fdae6b", "#d62728"]
    ax.bar(labels, pp, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(pp):
        if np.isfinite(v):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Posterior probability")
    ax.set_ylim(0, 1.1)
    ax.set_title(title or "Colocalization posterior probabilities")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


@register_function(
    aliases=[
        "mr_scatter", "mr_scatter_plot", "mendelian_randomization_scatter",
        "MR散点图", "孟德尔随机化散点图",
    ],
    category="genetics",
    description=(
        "Mendelian-randomization scatter plot — SNP-outcome effects "
        "against SNP-exposure effects, with the fitted causal-effect "
        "slope. Delegates to :func:`pytwosamplemr.mr_scatter`."
    ),
    examples=[
        "ov.genetics.mr_scatter(mr_input)",
    ],
    related=["ov.genetics.mendelian_randomization", "ov.genetics.mr_forest"],
)
def mr_scatter(mr_input, **kwargs):
    """MR scatter plot (delegates to the pytwosamplemr backend).

    Parameters
    ----------
    mr_input
        An :class:`pytwosamplemr.MRInput`.
    **kwargs
        Forwarded to :func:`pytwosamplemr.mr_scatter`.

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        The backend figure.
    """
    try:
        import pytwosamplemr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.genetics.mr_scatter requires pytwosamplemr: "
            "`pip install pytwosamplemr`."
        ) from exc
    return pytwosamplemr.mr_scatter(mr_input, **kwargs)


@register_function(
    aliases=[
        "mr_forest", "mr_forest_plot", "mendelian_randomization_forest",
        "MR森林图", "孟德尔随机化森林图",
    ],
    category="genetics",
    description=(
        "Mendelian-randomization forest plot — per-SNP (Wald-ratio) "
        "causal estimates with confidence intervals, alongside the "
        "combined estimate. Delegates to :func:`pytwosamplemr.mr_forest`."
    ),
    examples=[
        "ov.genetics.mr_forest(mr_input)",
    ],
    related=["ov.genetics.mendelian_randomization", "ov.genetics.mr_scatter"],
)
def mr_forest(mr_input, **kwargs):
    """MR forest plot (delegates to the pytwosamplemr backend).

    Parameters
    ----------
    mr_input
        An :class:`pytwosamplemr.MRInput`.
    **kwargs
        Forwarded to :func:`pytwosamplemr.mr_forest`.

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        The backend figure.
    """
    try:
        import pytwosamplemr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.genetics.mr_forest requires pytwosamplemr: "
            "`pip install pytwosamplemr`."
        ) from exc
    return pytwosamplemr.mr_forest(mr_input, **kwargs)


@register_function(
    aliases=[
        "finemap_plot", "susie_plot", "finemapping_plot",
        "精细定位图", "SuSiE图",
    ],
    category="genetics",
    description=(
        "Fine-mapping summary plot of a fitted SuSiE model — per-variant "
        "posterior inclusion probabilities (PIPs), colour-coded by "
        "credible set. Delegates to :func:`pysusie.susie_plot`."
    ),
    examples=[
        "ov.genetics.finemap_plot(susie_fit)",
        "ov.genetics.finemap_plot(susie_fit, y='PIP')",
    ],
    related=["ov.genetics.finemap", "ov.genetics.get_pip"],
)
def finemap_plot(fit, *, y: str = "PIP", **kwargs):
    """Fine-mapping plot (delegates to the pysusie backend).

    Parameters
    ----------
    fit
        A :class:`pysusie.SusieFit`.
    y
        Quantity to plot — ``'PIP'`` (default) or ``'z_original'`` etc.
    **kwargs
        Forwarded to :func:`pysusie.susie_plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The backend axes.
    """
    try:
        import pysusie
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.genetics.finemap_plot requires pysusie: "
            "`pip install pysusie`."
        ) from exc
    return pysusie.susie_plot(fit, y=y, **kwargs)


@register_function(
    aliases=[
        "sample_qc_plot", "qc_plot", "individual_qc_plot",
        "样本质控图", "个体质控图",
    ],
    category="genetics",
    description=(
        "Two-panel sample (per-individual) quality-control figure — a "
        "histogram of the per-sample call rate with the call-rate "
        "threshold marked, and a histogram of the per-sample "
        "heterozygosity with the mean +/- 3 SD outlier bounds marked. "
        "Accepts the DataFrame from :func:`ov.genetics.sample_qc_metrics`. "
        "matplotlib."
    ),
    examples=[
        "ov.genetics.sample_qc_plot(qc)",
        "ov.genetics.sample_qc_plot(qc, call_rate=0.98)",
    ],
    related=["ov.genetics.sample_qc_metrics", "ov.genetics.gwas_qc"],
)
def sample_qc_plot(
    qc: pd.DataFrame,
    *,
    call_rate: float = 0.98,
    het_bounds: Optional[tuple] = None,
    axes=None,
    title: Optional[str] = "Sample QC distributions",
):
    """Two-panel sample-QC distribution plot.

    Parameters
    ----------
    qc
        Per-sample QC table (from :func:`ov.genetics.sample_qc_metrics`) —
        needs ``call_rate`` and ``heterozygosity`` columns.
    call_rate
        Call-rate threshold drawn on the call-rate panel.
    het_bounds
        ``(low, high)`` heterozygosity outlier bounds; taken from
        ``qc.attrs['het_bounds']`` when ``None``.
    axes
        Optional pair of existing Axes; a 1x2 grid is created otherwise.
    title
        Optional figure suptitle.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        The two panel axes.
    """
    import matplotlib.pyplot as plt

    if het_bounds is None:
        het_bounds = qc.attrs.get("het_bounds")
        if het_bounds is None:
            mu, sd = qc["heterozygosity"].mean(), qc["heterozygosity"].std()
            het_bounds = (mu - 3 * sd, mu + 3 * sd)
    het_lo, het_hi = het_bounds

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(qc["call_rate"], bins=40, color="#3b6fb6")
    axes[0].axvline(call_rate, color="#d62728", ls="--",
                    label=f"call-rate {call_rate:g}")
    axes[0].set(xlabel="per-sample call rate", ylabel="samples")
    axes[1].hist(qc["heterozygosity"], bins=40, color="#3b6fb6")
    axes[1].axvline(het_lo, color="#d62728", ls="--")
    axes[1].axvline(het_hi, color="#d62728", ls="--", label="mean +/- 3 SD")
    axes[1].set(xlabel="per-sample heterozygosity", ylabel="samples")
    for ax in axes:
        ax.legend(fontsize=8)
    if title:
        axes[0].figure.suptitle(title)
    return axes


@register_function(
    aliases=[
        "pca_structure_plot", "genotype_pca_plot", "structure_plot",
        "群体结构图", "基因型PCA图",
    ],
    category="genetics",
    description=(
        "Two-panel genotype-PCA population-structure figure — a scree "
        "plot of the variance explained by each principal component (to "
        "choose how many PCs to carry) and a PC1-vs-PC2 scatter, "
        "optionally coloured by a (sub)population label so that real "
        "ancestry structure is visible. matplotlib."
    ),
    examples=[
        "ov.genetics.pca_structure_plot(pcs, var_ratio)",
        "ov.genetics.pca_structure_plot(pcs, var_ratio, labels=pop)",
    ],
    related=["ov.genetics.genotype_pca", "ov.genetics.gwas_association"],
)
def pca_structure_plot(
    pcs: np.ndarray,
    variance_ratio: np.ndarray,
    *,
    labels=None,
    axes=None,
    title: Optional[str] = None,
):
    """Two-panel genotype-PCA structure plot (scree + PC1/PC2 scatter).

    Parameters
    ----------
    pcs
        ``samples x n_comps`` PC-score matrix (from
        :func:`ov.genetics.genotype_pca`).
    variance_ratio
        Per-component variance-explained vector.
    labels
        Optional per-sample (sub)population labels colouring the scatter.
    axes
        Optional pair of existing Axes; a 1x2 grid is created otherwise.
    title
        Optional figure suptitle.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        The two panel axes.
    """
    import matplotlib.pyplot as plt

    pcs = np.asarray(pcs)
    vr = np.asarray(variance_ratio)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(np.arange(1, len(vr) + 1), vr, "o-", color="#3b6fb6")
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("variance ratio")
    axes[0].set_title("Scree plot — choosing the number of PCs")

    if labels is not None:
        labels = np.asarray(labels).astype(str)
        for p in sorted(np.unique(labels)):
            m = labels == p
            axes[1].scatter(pcs[m, 0], pcs[m, 1], s=8, label=p, alpha=0.6)
        axes[1].legend(fontsize=8)
    else:
        axes[1].scatter(pcs[:, 0], pcs[:, 1], s=8, color="#3b6fb6", alpha=0.6)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("Genotype PCA — PC1 / PC2")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    if title:
        axes[0].figure.suptitle(title)
    return axes


@register_function(
    aliases=[
        "finemap_locus_plot", "susie_locus_plot", "finemapping_locus_plot",
        "精细定位位点图", "SuSiE位点图",
    ],
    category="genetics",
    description=(
        "Fine-mapping locus view — the top panel is the publication "
        "LocusZoom regional association plot (-log10 p vs position, lead "
        "SNP as a purple diamond, optional LD-binned colouring and "
        "recombination-rate line) and the bottom panel is the SuSiE "
        "posterior-inclusion-probability (PIP) track with the 95% "
        "credible-set SNPs highlighted. When a gene table is supplied a "
        "gene-model track is inserted between the two. Ties the "
        "association peak to the fine-mapped credible set in one figure. "
        "matplotlib."
    ),
    examples=[
        "ov.genetics.finemap_locus_plot(locus, pip, credible)",
        "ov.genetics.finemap_locus_plot(locus, pip, credible, lead_snp='rs1', "
        "r2=ld, recomb_map=rmap, genes=gene_models)",
    ],
    related=["ov.genetics.finemap", "ov.genetics.regional_plot",
             "ov.genetics.get_credible_sets", "ov.genetics.compute_ld_to_lead"],
)
def finemap_locus_plot(
    locus: pd.DataFrame,
    pip: np.ndarray,
    credible: dict,
    *,
    chrom: str = "chrom",
    pos: str = "pos",
    pvalue: str = "pvalue",
    snp: str = "snp",
    lead_snp: Optional[str] = None,
    r2: Optional[Union[pd.Series, np.ndarray, dict]] = None,
    ld: Optional[Union[pd.Series, np.ndarray, dict]] = None,
    recomb_map: Optional[pd.DataFrame] = None,
    genes: Optional[pd.DataFrame] = None,
    axes=None,
    title: Optional[str] = None,
):
    """Fine-mapping locus view (LocusZoom regional p-values + SuSiE PIP).

    Parameters
    ----------
    locus
        Per-SNP association table for one locus — needs position and
        p-value columns; row order must match ``pip``.
    pip
        Per-SNP posterior inclusion probabilities (from
        :func:`ov.genetics.get_pip`), aligned to ``locus``.
    credible
        The credible-set object from
        :func:`ov.genetics.get_credible_sets` — its ``'cs'`` entry lists
        the within-locus index sets.
    chrom, pos, pvalue, snp
        ``locus`` column names.
    lead_snp
        Optional lead-SNP id, drawn as a purple diamond on the regional
        panel.
    r2, ld
        Optional per-SNP LD (r^2) to the lead variant, for the LocusZoom
        discrete-bin colouring (see :func:`regional_plot`).
    recomb_map
        Optional recombination map drawn as the cM/Mb track on the
        regional panel.
    genes
        Optional gene-model table; when given an arrowed gene-model track
        is inserted between the regional and PIP panels.
    axes
        Optional pair of existing Axes (regional + PIP); ignored when
        ``genes`` is supplied (a fresh multi-panel figure is built).
    title
        Optional title for the regional (top) panel.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        The panel axes (regional, PIP — and the gene track when drawn).

    Notes
    -----
    Reading a LocusZoom: SNPs near the lead variant share its LD (warm
    colours, top of the r^2 legend) and cluster at the association peak;
    a recombination hotspot (a tall cM/Mb spike) marks the boundary of
    the LD block; the gene track shows which genes the credible set
    physically falls on.
    """
    import matplotlib.pyplot as plt

    pip = np.asarray(pip, dtype=float)
    r2 = r2 if r2 is not None else ld
    bp = pd.to_numeric(locus[pos], errors="coerce").to_numpy()

    if genes is not None:
        # Three-panel layout: LocusZoom scatter / gene track / PIP track.
        fig = plt.figure(figsize=(9, 9))
        gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.7, 1.6],
                              hspace=0.28)
        ax_reg = fig.add_subplot(gs[0])
        ax_gene = fig.add_subplot(gs[1], sharex=ax_reg)
        ax_pip = fig.add_subplot(gs[2], sharex=ax_reg)
        regional_plot(
            locus, chrom=chrom, pos=pos, pvalue=pvalue, snp=snp,
            lead_snp=lead_snp, r2=r2, recomb_map=recomb_map, ax=ax_reg,
            title=title or "Regional association",
        )
        ax_reg.tick_params(labelbottom=False)
        ax_reg.set_xlabel("")
        win_lo, win_hi = ax_reg.get_xlim()
        chrom_val = (str(locus[chrom].astype(str).iloc[0])
                     if chrom in locus.columns and len(locus) else None)
        _draw_gene_track(ax_gene, genes, win_lo, win_hi, chrom=chrom_val)
        ax_gene.tick_params(labelbottom=False)
        ax_gene.set_xlabel("")
        axes = np.array([ax_reg, ax_pip])
        gene_track = ax_gene
    else:
        if axes is None:
            _, axes = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True)
        regional_plot(
            locus, chrom=chrom, pos=pos, pvalue=pvalue, snp=snp,
            lead_snp=lead_snp, r2=r2, recomb_map=recomb_map, ax=axes[0],
            title=title or "Regional association",
        )
        gene_track = None

    in_cs = np.zeros(len(pip), dtype=bool)
    for idx in (credible.get("cs") or []):
        in_cs[list(idx)] = True
    ax_pip = axes[1]
    ax_pip.scatter(bp[~in_cs], pip[~in_cs], s=18, c="#bdbdbd",
                   label="not in credible set")
    ax_pip.scatter(bp[in_cs], pip[in_cs], s=45, c="#d62728",
                   edgecolors="black", label="95% credible set")
    ax_pip.set_xlabel("position (Mb)")
    ax_pip.set_ylabel("PIP")
    ax_pip.set_title("SuSiE fine-mapping — posterior inclusion probability")
    ax_pip.legend(fontsize=8)
    ax_pip.spines[["top", "right"]].set_visible(False)
    ax_pip.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v / 1e6:.2f}"))
    if gene_track is not None:
        return np.array([axes[0], gene_track, ax_pip], dtype=object)
    return axes


@register_function(
    aliases=[
        "twas_manhattan", "twas_plot", "gene_manhattan",
        "TWAS曼哈顿图", "基因关联图",
    ],
    category="genetics",
    description=(
        "Gene-level TWAS Manhattan / bar plot — one bar per gene of "
        "-log10(p) from a transcriptome-wide association study, with the "
        "gene-level Bonferroni line drawn and an optional gene of "
        "interest highlighted. Summarises which predicted-expression "
        "profiles track the trait. matplotlib."
    ),
    examples=[
        "ov.genetics.twas_manhattan(twas_res)",
        "ov.genetics.twas_manhattan(twas_res, highlight='GENE0007')",
    ],
    related=["ov.genetics.twas", "ov.genetics.manhattan"],
)
def twas_manhattan(
    twas_res: pd.DataFrame,
    *,
    gene: str = "gene",
    pvalue: str = "pvalue",
    highlight: Optional[str] = None,
    ax=None,
    title: Optional[str] = "Gene-level TWAS",
):
    """Gene-level TWAS Manhattan (bar) plot.

    Parameters
    ----------
    twas_res
        Per-gene TWAS results table (from :func:`ov.genetics.twas`).
    gene, pvalue
        Gene-id and p-value column names.
    highlight
        Optional gene id drawn in red (e.g. the candidate causal gene).
    ax
        Existing matplotlib Axes.
    title
        Optional plot title.

    Returns
    -------
    tuple of (matplotlib.axes.Axes, float)
        The plot axes and the Bonferroni threshold used.
    """
    import matplotlib.pyplot as plt

    order = (twas_res.sort_values(pvalue, na_position="last")
                     .reset_index(drop=True))
    n_tested = int(twas_res[pvalue].notna().sum())
    threshold = 0.05 / max(n_tested, 1)
    logp = -np.log10(order[pvalue].clip(lower=1e-300))
    colors = ["#d62728" if g == highlight else "#3b6fb6"
              for g in order[gene]]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(order)), logp, color=colors, edgecolor="black",
           linewidth=0.4)
    ax.axhline(-np.log10(threshold), color="#d62728", ls="--",
               label=f"Bonferroni (p = {threshold:.1e})")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order[gene], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"$-\log_{10}(p)$")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return ax, threshold


@register_function(
    aliases=[
        "scdrs_celltype_plot", "scdrs_plot", "disease_score_celltype_plot",
        "scDRS细胞类型图", "疾病评分细胞类型图",
    ],
    category="genetics",
    description=(
        "Two-panel single-cell disease-relevance (scDRS) summary by cell "
        "type — a violin plot of the per-cell disease-relevance score "
        "across cell types and a bar plot of the fraction of "
        "significantly disease-associated cells per type. Highlights the "
        "cell type carrying the trait's genetic signal. matplotlib."
    ),
    examples=[
        "ov.genetics.scdrs_celltype_plot(adata)",
        "ov.genetics.scdrs_celltype_plot(adata, score='scdrs_score', "
        "pval='scdrs_pval', cell_type='cell_type')",
    ],
    related=["ov.genetics.disease_relevance_score",
             "ov.genetics.score_downstream"],
)
def scdrs_celltype_plot(
    adata,
    *,
    cell_type: str = "cell_type",
    score: str = "scdrs_score",
    pval: str = "scdrs_pval",
    sig: float = 0.05,
    axes=None,
    title: Optional[str] = None,
):
    """Two-panel scDRS score-by-cell-type figure.

    Parameters
    ----------
    adata
        scRNA-seq AnnData scored by
        :func:`ov.genetics.disease_relevance_score` — ``.obs`` must carry
        the cell-type, score and p-value columns.
    cell_type, score, pval
        ``.obs`` column names.
    sig
        Significance threshold for the per-cell-type significant-fraction
        panel.
    axes
        Optional pair of existing Axes; a 1x2 grid is created otherwise.
    title
        Optional figure suptitle.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        The two panel axes.
    """
    import matplotlib.pyplot as plt

    obs = adata.obs
    ct = obs[cell_type].astype("category")
    cats = list(ct.cat.categories)
    by_ct_mean = (obs.groupby(cell_type, observed=True)[score].mean())
    top_ct = by_ct_mean.idxmax()
    ct_colors = ["#d62728" if c == top_ct else "#3b6fb6" for c in cats]
    score_by_ct = [obs.loc[ct == c, score].to_numpy() for c in cats]
    sig_frac = (obs.assign(_sig=obs[pval] < sig)
                   .groupby(cell_type, observed=True)["_sig"].mean()
                   .reindex(cats))

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4))
    parts = axes[0].violinplot(score_by_ct, showmeans=True)
    for pc, col in zip(parts["bodies"], ct_colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    axes[0].set_xticks(np.arange(1, len(cats) + 1))
    axes[0].set_xticklabels(cats, rotation=20)
    axes[0].set(ylabel="scDRS disease-relevance score",
                title="Disease-relevance score by cell type")

    axes[1].bar([str(c) for c in cats], sig_frac.to_numpy(),
                color=ct_colors, edgecolor="black", linewidth=0.4)
    axes[1].set(ylabel=f"fraction of cells with scDRS p < {sig:g}",
                title="Significantly disease-associated cells")
    axes[1].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    if title:
        axes[0].figure.suptitle(title)
    return axes


@register_function(
    aliases=[
        "gene_celltype_expression", "gene_celltype_barplot",
        "gene_expression_by_celltype", "基因细胞类型表达图",
        "细胞类型基因表达图",
    ],
    category="genetics",
    description=(
        "Barplot of one gene's mean expression across cell types — the "
        "panel-c style figure used to show which cell type expresses a "
        "candidate disease gene (e.g. the top scDRS gene). Given an "
        "scRNA-seq AnnData, a gene name and a cell-type ``.obs`` column, "
        "it averages the gene's expression within each cell type and "
        "draws one bar per type, highlighting the highest-expressing "
        "cell type. matplotlib."
    ),
    examples=[
        "ov.genetics.gene_celltype_expression(adata, 'CD3D')",
        "ov.genetics.gene_celltype_expression(adata, gene, "
        "cell_type='cell_type')",
    ],
    related=["ov.genetics.scdrs_celltype_plot",
             "ov.genetics.disease_relevance_score"],
)
def gene_celltype_expression(
    adata,
    gene: str,
    *,
    cell_type: str = "cell_type",
    layer: Optional[str] = None,
    ax=None,
    title: Optional[str] = None,
):
    """Barplot of a gene's mean expression across cell types.

    Parameters
    ----------
    adata
        scRNA-seq AnnData — ``.obs`` must carry the cell-type column and
        ``gene`` must be one of ``.var_names``.
    gene
        Gene name (a ``var_names`` entry) to summarise.
    cell_type
        ``.obs`` column holding the cell-type labels.
    layer
        Optional ``.layers`` key to read expression from; ``.X`` is used
        when ``None``.
    ax
        Existing matplotlib Axes.
    title
        Optional plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes.
    """
    import matplotlib.pyplot as plt

    if gene not in adata.var_names:
        raise KeyError(f"gene {gene!r} is not in adata.var_names.")
    if cell_type not in adata.obs.columns:
        raise KeyError(f"{cell_type!r} is not an .obs column.")
    sub = adata[:, gene]
    expr = sub.layers[layer] if layer is not None else sub.X
    expr = expr.toarray() if hasattr(expr, "toarray") else np.asarray(expr)
    expr = np.asarray(expr, dtype=float).ravel()

    by_ct = (pd.Series(expr, index=adata.obs[cell_type].to_numpy())
               .groupby(level=0).mean().sort_values(ascending=False))
    cats = [str(c) for c in by_ct.index]
    top = cats[0] if cats else None
    colors = ["#d62728" if c == top else "#3b6fb6" for c in cats]

    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 0.7 * len(cats)), 4))
    ax.bar(cats, by_ct.to_numpy(), color=colors, edgecolor="black",
           linewidth=0.4)
    ax.set_ylabel(f"mean {gene} expression")
    ax.set_title(title or f"{gene} expression by cell type")
    ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


@register_function(
    aliases=[
        "mr_effect_plot", "mr_effect_scatter", "mr_ivw_scatter",
        "MR效应散点图", "孟德尔随机化效应图",
    ],
    category="genetics",
    description=(
        "Mendelian-randomization effect scatter from raw instrument "
        "arrays — plots each instrument's SNP-outcome effect against its "
        "SNP-exposure effect (with error bars) and overlays the IVW "
        "causal-effect slope through the origin. Use this when you have "
        "the four effect / SE arrays directly; :func:`mr_scatter` is the "
        "MRInput-based equivalent. matplotlib."
    ),
    examples=[
        "ov.genetics.mr_effect_plot(bx, bxse, by, byse, slope=ivw.estimate)",
    ],
    related=["ov.genetics.mendelian_randomization", "ov.genetics.mr_scatter",
             "ov.genetics.mr_forest"],
)
def mr_effect_plot(
    bx: np.ndarray,
    bxse: np.ndarray,
    by: np.ndarray,
    byse: np.ndarray,
    *,
    slope: float,
    exposure_label: str = "SNP effect on exposure",
    outcome_label: str = "SNP effect on outcome",
    ax=None,
    title: Optional[str] = "Mendelian randomization — exposure vs outcome",
):
    """MR exposure-vs-outcome effect scatter with the IVW slope.

    Parameters
    ----------
    bx, bxse
        Per-instrument SNP-exposure effect sizes and standard errors.
    by, byse
        Per-instrument SNP-outcome effect sizes and standard errors.
    slope
        The IVW causal-effect estimate, drawn as a line through the origin.
    exposure_label, outcome_label
        Axis labels for the exposure (x) and outcome (y).
    ax
        Existing matplotlib Axes.
    title
        Optional plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes.
    """
    import matplotlib.pyplot as plt

    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(bx, by, xerr=np.asarray(bxse, dtype=float),
                yerr=np.asarray(byse, dtype=float), fmt="o", color="#3b6fb6",
                ecolor="#bdbdbd", capsize=2, label="instruments")
    xx = np.linspace(min(bx.min(), 0.0), bx.max() * 1.05, 50)
    ax.plot(xx, slope * xx, color="#d62728", lw=2,
            label=f"IVW slope = {slope:.3f}")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(exposure_label)
    ax.set_ylabel(outcome_label)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return ax
