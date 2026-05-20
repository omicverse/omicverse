"""Data-prep helpers for ``ov.genetics`` — genetics-specific wrangling.

The post-GWAS pipeline strings together several methods that each want
their inputs shaped a particular way: genotype PCA needs a scaled
matrix, locus definition needs LD clumping, colocalization wants a
summary-statistics dict, TWAS wants a prediction model, eQTL mapping
wants features x samples matrices. None of that is *analysis* — it is
plumbing — and it should not be re-typed inline in every tutorial or
script. This module collects those small, registered helpers so a
notebook can call one function instead of a dozen lines of reshaping.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .._registry import register_function


# --------------------------------------------------------------------------- #
# Sample QC                                                                    #
# --------------------------------------------------------------------------- #
@register_function(
    aliases=[
        "sample_qc_metrics", "per_sample_qc", "individual_qc_metrics",
        "样本质控指标", "个体质控指标",
    ],
    category="genetics",
    description=(
        "Compute per-sample (per-individual) GWAS quality-control metrics "
        "from a genotype AnnData — the call rate (fraction of non-missing "
        "genotypes) and the mean heterozygosity (fraction of heterozygous "
        "calls). A low call rate flags a poorly-genotyped DNA sample; a "
        "heterozygosity outlier (mean +/- 3 SD) flags contamination or "
        "inbreeding. Returns a tidy per-sample DataFrame and records the "
        "heterozygosity outlier bounds. Pure numpy."
    ),
    examples=[
        "ov.genetics.sample_qc_metrics(geno)",
        "qc = ov.genetics.sample_qc_metrics(geno, het_sd=3.0)",
    ],
    related=["ov.genetics.gwas_qc", "ov.genetics.sample_qc_plot"],
)
def sample_qc_metrics(adata, *, het_sd: float = 3.0) -> pd.DataFrame:
    """Per-sample call rate and heterozygosity for sample QC.

    Parameters
    ----------
    adata
        Genotype AnnData of ``samples x SNPs`` (0/1/2 dosages in ``.X``).
    het_sd
        Number of standard deviations for the heterozygosity outlier
        bounds (default 3).

    Returns
    -------
    pandas.DataFrame
        One row per sample with columns ``call_rate`` and
        ``heterozygosity``. ``.attrs['het_bounds']`` holds the
        ``(low, high)`` heterozygosity outlier bounds.
    """
    X = np.asarray(adata.X, dtype=float)
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray().astype(float)
    call_rate = 1.0 - np.isnan(X).mean(axis=1)
    het = np.nanmean(X == 1, axis=1)
    qc = pd.DataFrame(
        {"call_rate": call_rate, "heterozygosity": het},
        index=adata.obs_names,
    )
    mu, sd = qc["heterozygosity"].mean(), qc["heterozygosity"].std()
    qc.attrs["het_bounds"] = (float(mu - het_sd * sd), float(mu + het_sd * sd))
    return qc


# --------------------------------------------------------------------------- #
# Genotype PCA (population structure)                                          #
# --------------------------------------------------------------------------- #
@register_function(
    aliases=[
        "genotype_pca", "structure_pca", "ancestry_pca",
        "基因型PCA", "群体结构PCA",
    ],
    category="genetics",
    description=(
        "Principal-component analysis of a QC'd genotype matrix to capture "
        "population structure. Scales the genotypes (standard for genotype "
        "PCA) and runs PCA, returning the sample PC scores and the "
        "variance explained by each component. The top PCs are then used "
        "as covariates in ``gwas_association`` to correct for population "
        "stratification. Wraps scanpy's PCA."
    ),
    examples=[
        "pcs, var_ratio = ov.genetics.genotype_pca(geno_qc, n_comps=10)",
        "pcs, vr = ov.genetics.genotype_pca(geno_qc, n_comps=20, max_value=10)",
    ],
    related=["ov.genetics.gwas_association", "ov.genetics.pca_structure_plot"],
)
def genotype_pca(
    adata,
    *,
    n_comps: int = 10,
    max_value: float = 10.0,
):
    """Genotype PCA for population-structure correction.

    Parameters
    ----------
    adata
        QC'd genotype AnnData of ``samples x SNPs``.
    n_comps
        Number of principal components to compute.
    max_value
        Clip value passed to :func:`scanpy.pp.scale`.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        ``(pcs, variance_ratio)`` — the ``samples x n_comps`` PC-score
        matrix and the per-component variance-explained vector.
    """
    import scanpy as sc

    work = adata.copy()
    sc.pp.scale(work, max_value=max_value)
    sc.tl.pca(work, n_comps=n_comps)
    pcs = np.asarray(work.obsm["X_pca"])
    var_ratio = np.asarray(work.uns["pca"]["variance_ratio"])
    return pcs, var_ratio


# --------------------------------------------------------------------------- #
# Locus definition (LD clumping) and grading                                   #
# --------------------------------------------------------------------------- #
@register_function(
    aliases=[
        "clump_loci", "define_loci", "ld_clump", "clump",
        "定义位点", "LD聚类", "位点聚类",
    ],
    category="genetics",
    description=(
        "Define independent association loci from a GWAS results table by "
        "LD clumping. Keeps every SNP that reaches genome-wide "
        "significance, then collapses correlated SNPs to one lead SNP per "
        "LD block (the block-level clump that stands in for the PLINK "
        "``--clump`` LD window). Returns one row per independent locus, "
        "led by its most-significant SNP. Pure pandas."
    ),
    examples=[
        "loci = ov.genetics.clump_loci(res_adj)",
        "loci = ov.genetics.clump_loci(res_adj, sig=5e-8, block='block')",
    ],
    related=["ov.genetics.gwas_association", "ov.genetics.grade_loci",
             "ov.genetics.manhattan"],
)
def clump_loci(
    results: pd.DataFrame,
    *,
    sig: float = 5e-8,
    block: str = "block",
    snp: str = "snp",
    pvalue: str = "pvalue",
) -> pd.DataFrame:
    """Define independent loci by LD-block clumping.

    Parameters
    ----------
    results
        GWAS results table — needs SNP, p-value and LD-block columns.
    sig
        Genome-wide-significance threshold (default ``5e-8``).
    block
        LD-block column used as the clumping unit.
    snp
        SNP-id column; renamed to ``lead_snp`` in the output.
    pvalue
        p-value column.

    Returns
    -------
    pandas.DataFrame
        One row per independent locus (the lead SNP of each
        genome-wide-significant LD block), sorted by p-value.
    """
    hits = results[results[pvalue] < sig].sort_values(pvalue)
    loci = (hits.drop_duplicates(block)
                .rename(columns={snp: "lead_snp"})
                .reset_index(drop=True))
    return loci


@register_function(
    aliases=[
        "grade_loci", "evaluate_loci", "score_loci", "locus_recovery",
        "位点评估", "位点回收率",
    ],
    category="genetics",
    description=(
        "Grade discovered GWAS loci against a known ground-truth signal "
        "set — for simulated cohorts or replication benchmarks. Splits the "
        "lead SNPs of a clumped locus table into true positives (a planted "
        "causal SNP or a known instrument) and false positives, and "
        "reports the counts. Pure-Python set arithmetic."
    ),
    examples=[
        "ov.genetics.grade_loci(loci, causal_snps=truth['causal_snps'])",
        "ov.genetics.grade_loci(loci, causal_snps=cs, instruments=inst)",
    ],
    related=["ov.genetics.clump_loci", "ov.genetics.simulate_gwas_study"],
)
def grade_loci(
    loci: pd.DataFrame,
    *,
    causal_snps,
    instruments=None,
    lead_col: str = "lead_snp",
) -> dict:
    """Grade discovered loci against a ground-truth signal set.

    Parameters
    ----------
    loci
        A clumped locus table (from :func:`clump_loci`).
    causal_snps
        The planted / known direct causal SNPs.
    instruments
        Optional additional true-signal SNPs (e.g. causal-gene eQTLs that
        are genuine secondary trait loci).
    lead_col
        Column holding each locus' lead SNP.

    Returns
    -------
    dict
        Keys: ``n_loci``, ``recovered_causal`` / ``recovered_instruments``
        (sorted lists), ``false_positives`` (sorted list), and the matching
        ``n_*`` counts.
    """
    causal = set(causal_snps)
    inst = set(instruments) if instruments is not None else set()
    lead = set(loci[lead_col])
    rec_causal = lead & causal
    rec_inst = lead & inst
    false = lead - causal - inst
    return {
        "n_loci": int(len(lead)),
        "recovered_causal": sorted(rec_causal),
        "n_recovered_causal": int(len(rec_causal)),
        "recovered_instruments": sorted(rec_inst),
        "n_recovered_instruments": int(len(rec_inst)),
        "false_positives": sorted(false),
        "n_false_positives": int(len(false)),
    }


# --------------------------------------------------------------------------- #
# Colocalization dataset construction                                          #
# --------------------------------------------------------------------------- #
@register_function(
    aliases=[
        "make_coloc_dataset", "build_coloc_dataset", "coloc_dataset",
        "构建共定位数据集", "共定位数据集",
    ],
    category="genetics",
    description=(
        "Assemble a colocalization input dataset (the dict that "
        "``ov.genetics.colocalize`` expects) from per-SNP summary "
        "statistics. Packs the effect sizes, their variances "
        "(SE squared), SNP ids, minor-allele frequencies and sample size "
        "into the coloc schema for a quantitative or case/control trait. "
        "Pure pandas / numpy."
    ),
    examples=[
        "d = ov.genetics.make_coloc_dataset(gwas_locus, beta='BETA', se='SE', "
        "snps=locus_snps, n=2000, maf=maf)",
        "d = ov.genetics.make_coloc_dataset(eqtl_locus, n=2000, maf=maf)",
    ],
    related=["ov.genetics.colocalize", "ov.genetics.coloc_plot"],
)
def make_coloc_dataset(
    stats: pd.DataFrame,
    *,
    snps,
    n: int,
    maf,
    beta: str = "beta",
    se: str = "se",
    trait_type: str = "quant",
    sdY: float = 1.0,
) -> dict:
    """Build a coloc summary-statistics dataset dict.

    Parameters
    ----------
    stats
        Per-SNP summary statistics indexed (or alignable) by SNP id.
    snps
        SNP ids defining the locus and the row order of the dataset.
    n
        Sample size of the study the statistics come from.
    maf
        Per-SNP minor-allele frequency vector, aligned to ``snps``.
    beta, se
        Effect-size and standard-error column names.
    trait_type
        ``'quant'`` (quantitative trait) or ``'cc'`` (case/control).
    sdY
        Trait standard deviation (quantitative traits).

    Returns
    -------
    dict
        A coloc dataset with keys ``beta``, ``varbeta``, ``snp``,
        ``type``, ``N``, ``MAF`` and ``sdY``.
    """
    sub = stats.loc[list(snps)]
    se_vals = sub[se].to_numpy(dtype=float)
    return {
        "beta": sub[beta].to_numpy(dtype=float),
        "varbeta": se_vals ** 2,
        "snp": list(snps),
        "type": trait_type,
        "N": int(n),
        "MAF": np.asarray(maf, dtype=float),
        "sdY": float(sdY),
    }


# --------------------------------------------------------------------------- #
# eQTL input reshaping                                                         #
# --------------------------------------------------------------------------- #
@register_function(
    aliases=[
        "make_eqtl_matrices", "eqtl_inputs", "prepare_eqtl_inputs",
        "构建eQTL输入", "eQTL输入矩阵",
    ],
    category="genetics",
    description=(
        "Reshape a genotype AnnData and an expression AnnData into the "
        "features x samples matrices (and SNP / gene position tables) that "
        "``ov.genetics.eqtl_map`` (Matrix eQTL) expects. AnnData is "
        "samples x features, so this transposes both matrices and builds "
        "the SNP / gene position tables that drive the cis / trans split. "
        "Pure pandas."
    ),
    examples=[
        "geno_mat, expr_mat, snp_pos, gene_pos = "
        "ov.genetics.make_eqtl_matrices(geno_qc, expr)",
    ],
    related=["ov.genetics.eqtl_map", "ov.genetics.build_twas_model"],
)
def make_eqtl_matrices(geno, expr):
    """Reshape genotype + expression AnnData for Matrix eQTL.

    Parameters
    ----------
    geno
        Genotype AnnData of ``samples x SNPs`` — ``.var`` must carry
        ``chrom`` and ``pos``.
    expr
        Expression AnnData of ``samples x genes`` — ``.var`` must carry
        ``chrom`` and ``pos``.

    Returns
    -------
    tuple
        ``(geno_mat, expr_mat, snp_pos, gene_pos)`` — the two
        ``features x samples`` DataFrames and the SNP / gene position
        tables (``snp``/``chr``/``pos`` and ``geneid``/``chr``/``left``/
        ``right``).
    """
    geno_mat = pd.DataFrame(
        np.asarray(geno.X).T, index=geno.var_names, columns=geno.obs_names,
    )
    expr_mat = pd.DataFrame(
        np.asarray(expr.X).T, index=expr.var_names, columns=expr.obs_names,
    )
    snp_pos = geno.var[["chrom", "pos"]].reset_index()
    snp_pos.columns = ["snp", "chr", "pos"]
    gene_pos = expr.var[["chrom", "pos"]].reset_index()
    gene_pos.columns = ["geneid", "chr", "left"]
    gene_pos["right"] = gene_pos["left"] + 1
    return geno_mat, expr_mat, snp_pos, gene_pos


# --------------------------------------------------------------------------- #
# TWAS prediction-model construction                                           #
# --------------------------------------------------------------------------- #
@register_function(
    aliases=[
        "build_twas_model", "twas_model_from_eqtl", "make_predixcan_model",
        "构建TWAS模型", "TWAS预测模型",
    ],
    category="genetics",
    description=(
        "Build a PrediXcan-style gene-expression prediction model from "
        "lead cis-eQTLs — a single-SNP-per-gene weight table that "
        "``ov.genetics.twas`` (``method='predixcan'``) can use directly. "
        "Each gene's top cis-eQTL becomes its predictor, weighted by the "
        "eQTL effect size: a realistic minimal elastic-net-style model. "
        "Wraps the pytwas PredictionModel container."
    ),
    examples=[
        "model = ov.genetics.build_twas_model(lead_eqtl)",
        "model = ov.genetics.build_twas_model(lead_eqtl, snp_col='snps', "
        "weight_col='beta')",
    ],
    related=["ov.genetics.twas", "ov.genetics.eqtl_map",
             "ov.genetics.make_eqtl_matrices"],
)
def build_twas_model(
    lead_eqtl: pd.DataFrame,
    *,
    snp_col: str = "snps",
    weight_col: str = "beta",
    gene_col: Optional[str] = None,
    effect_allele: str = "A",
    non_effect_allele: str = "G",
):
    """Build a PrediXcan prediction model from lead cis-eQTLs.

    Parameters
    ----------
    lead_eqtl
        Lead-cis-eQTL table — one row per gene. The gene id is taken from
        the index unless ``gene_col`` is given.
    snp_col
        Column holding each gene's lead cis-eQTL SNP id.
    weight_col
        Column holding the eQTL effect size (the prediction weight).
    gene_col
        Optional explicit gene-id column (otherwise the index is used).
    effect_allele, non_effect_allele
        Effect / non-effect alleles for every weight.

    Returns
    -------
    pytwas.PredictionModel
        A single-SNP-per-gene prediction model ready for
        :func:`ov.genetics.twas` (``method='predixcan'``).
    """
    try:
        import pytwas
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.genetics.build_twas_model requires pytwas: "
            "`pip install pytwas` (or omicverse[genetics])."
        ) from exc

    genes = (lead_eqtl[gene_col] if gene_col is not None
             else lead_eqtl.index).astype(str)
    weights = pd.DataFrame({
        "rsid": lead_eqtl[snp_col].astype(str).to_numpy(),
        "gene": genes.to_numpy(),
        "weight": lead_eqtl[weight_col].astype(float).to_numpy(),
        "non_effect_allele": non_effect_allele,
        "effect_allele": effect_allele,
    })
    extra = pd.DataFrame({"gene": weights["gene"].unique()})
    extra["gene_name"] = extra["gene"]
    return pytwas.PredictionModel(
        weights=weights[["rsid", "gene", "weight",
                         "non_effect_allele", "effect_allele"]],
        extra=extra,
    )
