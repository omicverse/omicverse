"""Plotting for ``ov.genetics`` — matplotlib visualisations.

Standard statistical-genetics figures: the genome-wide Manhattan plot
(:func:`manhattan`), the Q-Q plot with genomic inflation
(:func:`qqplot`), the LocusZoom-style regional association plot
(:func:`regional_plot`), the colocalization plot (:func:`coloc_plot`),
the Mendelian-randomization scatter / forest plots
(:func:`mr_scatter`, :func:`mr_forest`) and the SuSiE fine-mapping plot
(:func:`finemap_plot`). The MR and fine-mapping plots delegate to the
backends' own plotting routines.
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


@register_function(
    aliases=[
        "regional_plot", "locuszoom", "regional_association_plot",
        "区域关联图", "局部关联图",
    ],
    category="genetics",
    description=(
        "LocusZoom-style regional association plot — zooms into a single "
        "locus and plots -log10(p) against base-pair position, optionally "
        "colouring SNPs by their LD (r^2) with a lead variant. Useful for "
        "inspecting a GWAS / eQTL peak before fine-mapping. matplotlib."
    ),
    examples=[
        "ov.genetics.regional_plot(gwas_res, chrom='1', start=1e6, end=2e6)",
        "ov.genetics.regional_plot(gwas_res, lead_snp='rs123', r2=ld_to_lead)",
    ],
    related=["ov.genetics.manhattan", "ov.genetics.finemap_plot"],
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
    ax=None,
    title: Optional[str] = None,
):
    """Draw a regional (LocusZoom-style) association plot.

    Parameters
    ----------
    data
        Association results — must have position and p-value columns.
    chrom, pos, pvalue, snp
        Column names; auto-detected when not given.
    region_chrom, start, end
        Optional region filter (chromosome + base-pair window).
    lead_snp
        Optional lead-variant SNP id (highlighted as a diamond).
    r2
        Optional per-SNP LD (r^2) to the lead variant — a Series / array
        aligned to ``data``, or a ``{snp: r2}`` dict.
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

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    if r2 is not None and snp_col is not None:
        if isinstance(r2, dict):
            r2_vals = df[snp_col].astype(str).map(r2).to_numpy(dtype=float)
        else:
            r2_vals = np.asarray(r2, dtype=float)
        sc = ax.scatter(bp, logp, c=r2_vals, cmap="YlOrRd", vmin=0, vmax=1,
                        s=22, edgecolors="grey", linewidths=0.3)
        cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(r"$r^2$ to lead")
    else:
        ax.scatter(bp, logp, s=20, c="#3b6fb6", edgecolors="grey",
                   linewidths=0.3)

    if lead_snp is not None and snp_col is not None:
        lead = df[df[snp_col].astype(str) == str(lead_snp)]
        if len(lead):
            lx = pd.to_numeric(lead[pos_col], errors="coerce").to_numpy()
            lp = -np.log10(np.clip(
                pd.to_numeric(lead[p_col], errors="coerce").to_numpy(),
                np.finfo(float).tiny, 1.0))
            ax.scatter(lx, lp, marker="D", s=70, c="#8c2d04",
                       edgecolors="black", zorder=5, label=f"lead: {lead_snp}")
            ax.legend(fontsize=8)

    ax.set_xlabel("Position (bp)")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    if title:
        ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
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
