"""Single-cell disease relevance scoring for ``ov.genetics`` — scDRS backend.

Wraps the standalone :mod:`pyscdrs` package (Zhang *et al.* 2022). Given
an scRNA-seq :class:`~anndata.AnnData` and a GWAS-derived gene set,
:func:`disease_relevance_score` assigns each cell a disease-relevance
score (and an empirical p-value). :func:`score_downstream` runs the
group-level, covariate-correlation and gene-level downstream analyses.

This is the one ``ov.genetics`` task that is genuinely AnnData-native —
scDRS operates on single-cell expression directly.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._registry import register_function


def _require_scdrs():
    """Import :mod:`pyscdrs` with a friendly error if it is missing."""
    try:
        import pyscdrs  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.genetics.disease_relevance_score requires the pyscdrs "
            "backend: `pip install pyscdrs` "
            "(or `pip install omicverse[genetics]`)."
        ) from exc
    return pyscdrs


@register_function(
    aliases=[
        "disease_relevance_score", "scdrs", "scDRS", "disease_score",
        "单细胞疾病相关性", "疾病相关性打分", "细胞疾病评分",
    ],
    category="genetics",
    description=(
        "Single-cell disease-relevance scoring with scDRS (Zhang 2022). "
        "Links a GWAS to single-cell expression: each cell receives a "
        "normalised disease-relevance score and an empirical p-value "
        "against Monte-Carlo control gene sets, revealing which cell "
        "types / states are enriched for a disease's heritability. Takes "
        "an scRNA-seq AnnData plus a GWAS-derived gene set (a list of "
        "genes, or a (genes, weights) tuple from a MAGMA ``.gs`` file). "
        "Runs scDRS preprocessing then control-matched scoring. Wraps the "
        "pyscdrs backend."
    ),
    examples=[
        "ov.genetics.disease_relevance_score(adata, gene_set=disease_genes)",
        "ov.genetics.disease_relevance_score(adata, gene_set=(genes, weights), n_ctrl=1000)",
    ],
    related=["ov.genetics.score_downstream", "ov.genetics.heritability"],
)
def disease_relevance_score(
    adata,
    gene_set: Union[Sequence[str], tuple],
    *,
    gene_weight: Optional[Sequence[float]] = None,
    cov: Optional[pd.DataFrame] = None,
    n_ctrl: int = 1000,
    weight_opt: str = "vs",
    ctrl_match_key: str = "mean_var",
    n_genebin: int = 200,
    random_seed: int = 0,
    copy: bool = False,
    return_raw: bool = False,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Score every cell for disease relevance with scDRS.

    Parameters
    ----------
    adata
        Single-cell :class:`~anndata.AnnData` (cells x genes), normalised
        and log-transformed.
    gene_set
        The GWAS-derived gene set — either a list of gene names, or a
        ``(genes, weights)`` tuple (as returned by :func:`load_gs`).
    gene_weight
        Optional per-gene weights (used when ``gene_set`` is a plain list).
    cov
        Optional cell-level covariate DataFrame regressed out during
        preprocessing.
    n_ctrl
        Number of Monte-Carlo control gene sets for the empirical null.
    weight_opt
        Gene-weighting scheme — ``'vs'`` (variance-stabilised), ``'inv_std'``,
        ``'od'`` or ``'uniform'``.
    ctrl_match_key
        Cell-property key used to match control genes (default
        ``'mean_var'``).
    n_genebin
        Number of expression bins for control-gene matching.
    random_seed
        Seed for the Monte-Carlo control draws.
    copy
        If ``True``, preprocess a copy and leave ``adata`` untouched.
    return_raw
        If ``True``, also return the raw / control scores.
    verbose
        Print backend progress.
    **kwargs
        Forwarded to :func:`pyscdrs.score_cell`.

    Returns
    -------
    pandas.DataFrame
        Per-cell score table with ``raw_score``, ``norm_score``,
        ``mc_pval``, ``pval``, ``nlog10_pval`` and ``zscore``.
    """
    scdrs = _require_scdrs()

    # Resolve a (genes, weights) tuple vs a plain gene list.
    if isinstance(gene_set, tuple) and len(gene_set) == 2:
        genes, weights = gene_set
        genes = list(genes)
        weights = None if weights is None else list(weights)
    else:
        genes = list(gene_set)
        weights = None if gene_weight is None else list(gene_weight)

    work = adata.copy() if copy else adata
    scdrs.preprocess(work, cov=cov, copy=False)
    df_score = scdrs.score_cell(
        work, gene_list=genes, gene_weight=weights,
        ctrl_match_key=ctrl_match_key, n_ctrl=n_ctrl,
        n_genebin=n_genebin, weight_opt=weight_opt,
        random_seed=random_seed,
        return_ctrl_raw_score=return_raw,
        return_ctrl_norm_score=return_raw,
        verbose=verbose, **kwargs,
    )
    return df_score


@register_function(
    aliases=[
        "score_downstream", "scdrs_downstream", "disease_score_downstream",
        "疾病评分下游分析", "scDRS下游分析",
    ],
    category="genetics",
    description=(
        "Downstream analysis of scDRS disease-relevance scores. "
        "``analysis`` selects the task: ``'group'`` tests which cell "
        "groups (e.g. cell types) are disease-enriched and heterogeneous; "
        "``'corr'`` correlates the score with continuous cell covariates; "
        "``'gene'`` ranks genes by correlation with the score. Wraps "
        "pyscdrs' downstream functions."
    ),
    examples=[
        "ov.genetics.score_downstream(adata, df_score, analysis='group', group_cols=['cell_type'])",
        "ov.genetics.score_downstream(adata, df_score, analysis='corr', var_cols=['gradient'])",
        "ov.genetics.score_downstream(adata, df_score, analysis='gene')",
    ],
    related=["ov.genetics.disease_relevance_score"],
)
def score_downstream(
    adata,
    df_score: pd.DataFrame,
    *,
    analysis: str = "group",
    group_cols: Optional[List[str]] = None,
    var_cols: Optional[List[str]] = None,
    fdr_thresholds: List[float] = [0.05, 0.1, 0.2],
):
    """Run a scDRS downstream analysis.

    Parameters
    ----------
    adata
        The AnnData scored by :func:`disease_relevance_score`.
    df_score
        The per-cell score DataFrame from :func:`disease_relevance_score`.
    analysis
        ``'group'`` (cell-group enrichment / heterogeneity), ``'corr'``
        (covariate correlation), or ``'gene'`` (gene-level association).
        ``analysis='group'`` needs a precomputed neighbour graph — run
        ``sc.pp.neighbors(adata)`` before calling it.
    group_cols
        For ``analysis='group'``: the ``adata.obs`` columns defining the
        cell groups.
    var_cols
        For ``analysis='corr'``: the continuous ``adata.obs`` columns to
        correlate with the score.
    fdr_thresholds
        For ``analysis='group'``: FDR cutoffs for the proportion of
        significant cells.

    Returns
    -------
    dict or pandas.DataFrame
        The backend downstream result.
    """
    scdrs = _require_scdrs()
    key = str(analysis).lower().strip()

    if key == "group":
        if not group_cols:
            raise ValueError("analysis='group' requires group_cols=.")
        return scdrs.downstream_group_analysis(
            adata, df_score, group_cols=list(group_cols),
            fdr_thresholds=list(fdr_thresholds),
        )
    if key == "corr":
        if not var_cols:
            raise ValueError("analysis='corr' requires var_cols=.")
        return scdrs.downstream_corr_analysis(
            adata, df_score, var_cols=list(var_cols),
        )
    if key == "gene":
        return scdrs.downstream_gene_analysis(adata, df_score)

    raise ValueError(
        f"analysis must be 'group', 'corr' or 'gene', got {analysis!r}"
    )
