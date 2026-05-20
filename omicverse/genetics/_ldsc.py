"""LD score regression for ``ov.genetics`` — LDSC backend.

Wraps the standalone :mod:`pyldsc` package (Bulik-Sullivan *et al.*
2015; Finucane *et al.* 2015, 2018). Covers SNP-heritability
(:func:`heritability`), genetic correlation (:func:`genetic_correlation`),
partitioned heritability (:func:`partitioned_heritability`), the
cell-type-specific LDSC-SEG analysis (:func:`ldsc_cell_type`) and
summary-statistics munging (:func:`munge_sumstats`).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

from .._registry import register_function


def _require_ldsc():
    """Import :mod:`pyldsc` with a friendly error if it is missing."""
    try:
        import pyldsc  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.genetics.heritability requires the pyldsc backend: "
            "`pip install pyldsc` (or `pip install omicverse[genetics]`)."
        ) from exc
    return pyldsc


@register_function(
    aliases=[
        "heritability", "ldsc", "estimate_h2", "snp_heritability",
        "遗传力", "SNP遗传力", "LD得分回归",
    ],
    category="genetics",
    description=(
        "SNP-heritability estimation by LD score regression (Bulik-"
        "Sullivan 2015). Regresses GWAS chi-square statistics on LD "
        "scores to estimate the proportion of phenotypic variance "
        "explained by common SNPs, while the regression intercept "
        "separates polygenic signal from confounding / inflation. Takes "
        "the path to a ``.sumstats`` file plus reference-panel and "
        "regression-weight LD-score filesets. Wraps "
        ":func:`pyldsc.estimate_h2`."
    ),
    examples=[
        "ov.genetics.heritability('trait.sumstats', ref_ld='baseline', w_ld='weights')",
        "ov.genetics.heritability('trait.sumstats', ref_ld='ldsc', w_ld='w', samp_prev=0.5, pop_prev=0.01)",
    ],
    related=["ov.genetics.genetic_correlation", "ov.genetics.partitioned_heritability",
             "ov.genetics.ldsc_cell_type", "ov.genetics.munge_sumstats"],
)
def heritability(
    sumstats: str,
    ref_ld: str,
    w_ld: str,
    *,
    M: Optional[str] = None,
    intercept_h2: Optional[float] = None,
    no_intercept: bool = False,
    n_blocks: int = 200,
    chisq_max: Optional[float] = None,
    samp_prev: Optional[float] = None,
    pop_prev: Optional[float] = None,
    **kwargs,
):
    """Estimate SNP-heritability with LD score regression.

    Parameters
    ----------
    sumstats
        Path to a ``.sumstats(.gz)`` GWAS summary-statistics file.
    ref_ld
        Reference-panel LD-score fileset prefix(es), comma-separated.
    w_ld
        Regression-weight LD-score fileset prefix.
    M
        Optional override of the SNP count.
    intercept_h2
        Constrain the regression intercept (e.g. to 1).
    no_intercept
        Equivalent to ``intercept_h2=1``.
    n_blocks
        Number of block-jackknife blocks.
    chisq_max
        Drop SNPs with chi-square above this value.
    samp_prev, pop_prev
        Sample / population prevalence — supplying both converts the
        observed-scale h2 to the liability scale.
    **kwargs
        Forwarded to :func:`pyldsc.estimate_h2`.

    Returns
    -------
    pyldsc.regressions.Hsq
        Fitted heritability object (``.tot``, ``.tot_se``, ``.intercept``,
        ``.lambda_gc``, ``.mean_chisq`` ...).
    """
    ldsc = _require_ldsc()
    return ldsc.estimate_h2(
        sumstats, ref_ld, w_ld, M=M, intercept_h2=intercept_h2,
        no_intercept=no_intercept, n_blocks=n_blocks, chisq_max=chisq_max,
        samp_prev=samp_prev, pop_prev=pop_prev, log=None, **kwargs,
    )


@register_function(
    aliases=[
        "genetic_correlation", "estimate_rg", "rg", "genetic_corr",
        "遗传相关", "遗传相关性",
    ],
    category="genetics",
    description=(
        "Genome-wide genetic correlation (rg) between two or more traits "
        "by cross-trait LD score regression (Bulik-Sullivan 2015) — "
        "quantifies the shared polygenic architecture of traits. Takes a "
        "list of ``.sumstats`` paths (the first is the reference trait) "
        "plus LD-score filesets. Wraps :func:`pyldsc.estimate_rg`."
    ),
    examples=[
        "ov.genetics.genetic_correlation(['t1.sumstats', 't2.sumstats'], ref_ld='ldsc', w_ld='w')",
    ],
    related=["ov.genetics.heritability", "ov.genetics.partitioned_heritability"],
)
def genetic_correlation(
    sumstats_list: Sequence[str],
    ref_ld: str,
    w_ld: str,
    *,
    n_blocks: int = 200,
    chisq_max: Optional[float] = None,
    intercept_h2: Optional[float] = None,
    intercept_gencov: Optional[float] = None,
    no_intercept: bool = False,
    **kwargs,
):
    """Estimate genetic correlation between traits.

    Parameters
    ----------
    sumstats_list
        Paths to two or more ``.sumstats(.gz)`` files; the first is the
        reference trait, the rest are each correlated against it.
    ref_ld, w_ld
        Reference-panel and regression-weight LD-score fileset prefixes.
    n_blocks
        Number of block-jackknife blocks.
    chisq_max
        Drop SNPs with chi-square above this value.
    intercept_h2, intercept_gencov
        Optionally constrain the h2 / genetic-covariance intercepts.
    no_intercept
        Constrain all intercepts (h2 to 1, gencov to 0).
    **kwargs
        Forwarded to :func:`pyldsc.estimate_rg`.

    Returns
    -------
    list of pyldsc.regressions.RG
        One genetic-correlation object per trait pair.
    """
    ldsc = _require_ldsc()
    return ldsc.estimate_rg(
        list(sumstats_list), ref_ld, w_ld, n_blocks=n_blocks,
        chisq_max=chisq_max, intercept_h2=intercept_h2,
        intercept_gencov=intercept_gencov, no_intercept=no_intercept,
        log=None, **kwargs,
    )


@register_function(
    aliases=[
        "partitioned_heritability", "partitioned_h2", "stratified_ldsc",
        "分层遗传力", "分区遗传力",
    ],
    category="genetics",
    description=(
        "Partitioned (stratified) heritability — splits SNP-heritability "
        "across functional annotation categories to find which genomic "
        "features (enhancers, conserved regions, etc.) are enriched for a "
        "trait's heritability (Finucane 2015). Takes a ``.sumstats`` path "
        "plus annotation-stratified LD-score filesets. Wraps "
        ":func:`pyldsc.partitioned_h2`."
    ),
    examples=[
        "ov.genetics.partitioned_heritability('trait.sumstats', ref_ld='baseline', w_ld='w', frqfile='1000G')",
    ],
    related=["ov.genetics.heritability", "ov.genetics.ldsc_cell_type"],
)
def partitioned_heritability(
    sumstats: str,
    ref_ld: str,
    w_ld: str,
    *,
    frqfile: Optional[str] = None,
    overlap_annot: bool = True,
    n_blocks: int = 200,
    chisq_max: Optional[float] = None,
    **kwargs,
):
    """Estimate partitioned (category-stratified) heritability.

    Parameters
    ----------
    sumstats
        Path to a ``.sumstats(.gz)`` file.
    ref_ld
        Annotation-stratified reference LD-score fileset prefix(es).
    w_ld
        Regression-weight LD-score fileset prefix.
    frqfile
        Allele-frequency fileset prefix (used with ``overlap_annot``).
    overlap_annot
        Apply the overlapping-category correction.
    n_blocks
        Number of block-jackknife blocks.
    chisq_max
        Drop SNPs with chi-square above this value.
    **kwargs
        Forwarded to :func:`pyldsc.partitioned_h2`.

    Returns
    -------
    pyldsc.regressions.Hsq
        Fitted object with per-category h2, enrichment and coefficients.
    """
    ldsc = _require_ldsc()
    return ldsc.partitioned_h2(
        sumstats, ref_ld, w_ld, frqfile=frqfile,
        overlap_annot=overlap_annot, n_blocks=n_blocks,
        chisq_max=chisq_max, log=None, **kwargs,
    )


@register_function(
    aliases=[
        "ldsc_cell_type", "ldsc_seg", "ldsc_cts", "cell_type_ldsc",
        "细胞类型遗传力富集", "LDSC-SEG", "组织特异性遗传力",
    ],
    category="genetics",
    description=(
        "LDSC-SEG cell-type / tissue specific heritability enrichment "
        "(Finucane 2018). For each cell type's specifically-expressed-gene "
        "LD-score set, fits a heritability regression on top of the "
        "baseline model and reports the cell-type coefficient with a "
        "one-sided p-value — ranking which cell types / tissues are most "
        "relevant to a trait. Wraps :func:`pyldsc.ldsc_seg`."
    ),
    examples=[
        "ov.genetics.ldsc_cell_type('trait.sumstats', ref_ld_cts='tissues.ldcts', ref_ld='baseline', w_ld='w')",
    ],
    related=["ov.genetics.partitioned_heritability", "ov.genetics.disease_relevance_score"],
)
def ldsc_cell_type(
    sumstats: str,
    ref_ld_cts: Union[str, List[Tuple[str, str]]],
    ref_ld: str,
    w_ld: str,
    *,
    n_blocks: int = 200,
    chisq_max: Optional[float] = None,
    **kwargs,
):
    """Run the LDSC-SEG cell-type heritability-enrichment analysis.

    Parameters
    ----------
    sumstats
        Path to a ``.sumstats(.gz)`` file.
    ref_ld_cts
        Either a path to a ``--ref-ld-chr-cts`` file (two columns: name
        and LD-score prefix) or a list of ``(name, ld_prefix)`` tuples.
    ref_ld
        The baseline-model reference LD-score fileset prefix.
    w_ld
        Regression-weight LD-score fileset prefix.
    n_blocks
        Number of block-jackknife blocks.
    chisq_max
        Drop SNPs with chi-square above this value.
    **kwargs
        Forwarded to :func:`pyldsc.ldsc_seg`.

    Returns
    -------
    pandas.DataFrame
        Per-cell-type coefficient, standard error and one-sided p-value.
    """
    ldsc = _require_ldsc()
    return ldsc.ldsc_seg(
        sumstats, ref_ld_cts, ref_ld, w_ld, n_blocks=n_blocks,
        chisq_max=chisq_max, log=None, **kwargs,
    )


@register_function(
    aliases=[
        "munge_sumstats", "munge", "format_sumstats", "标准化汇总统计",
        "GWAS汇总统计格式化",
    ],
    category="genetics",
    description=(
        "Munge a raw GWAS summary-statistics file into the standardised "
        "``.sumstats`` format used by LD score regression — harmonises "
        "column names, filters on INFO / MAF / sample size and aligns "
        "alleles. Wraps :func:`pyldsc.munge_sumstats`."
    ),
    examples=[
        "ov.genetics.munge_sumstats('raw_gwas.txt', out='trait', N=100000)",
    ],
    related=["ov.genetics.heritability", "ov.genetics.read_sumstats"],
)
def munge_sumstats(
    sumstats: str,
    *,
    out: Optional[str] = None,
    N: Optional[float] = None,
    info_min: float = 0.9,
    maf_min: float = 0.01,
    write: bool = True,
    **kwargs,
):
    """Munge a raw GWAS file to ``.sumstats`` format.

    Parameters
    ----------
    sumstats
        Path to the raw GWAS file (optionally ``.gz`` / ``.bz2``).
    out
        Output prefix; ``out.sumstats.gz`` is written when ``write`` is
        ``True``.
    N
        Sample size (used when the file lacks an ``N`` column).
    info_min, maf_min
        Minimum INFO score and minor-allele frequency to keep a SNP.
    write
        Whether to write the munged file to disk.
    **kwargs
        Forwarded to :func:`pyldsc.munge_sumstats`.

    Returns
    -------
    pandas.DataFrame
        The munged summary-statistics table.
    """
    ldsc = _require_ldsc()
    return ldsc.munge_sumstats(
        sumstats, out=out, N=N, info_min=info_min,
        maf_min=maf_min, write=write, log=None, **kwargs,
    )
