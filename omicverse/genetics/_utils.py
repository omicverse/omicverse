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
# LD to a lead SNP (LocusZoom colouring)                                       #
# --------------------------------------------------------------------------- #
@register_function(
    aliases=[
        "compute_ld_to_lead", "ld_to_lead", "lead_snp_ld", "locuszoom_ld",
        "计算LD", "前导SNP连锁不平衡",
    ],
    category="genetics",
    description=(
        "Compute the LD (r^2) between a lead SNP and every other SNP from "
        "an individual-level genotype AnnData — the colouring track of a "
        "publication LocusZoom regional-association plot. r^2 is the "
        "squared Pearson correlation of the 0/1/2 allele dosages across "
        "individuals; missing genotypes are mean-imputed per SNP. Pass "
        "the result straight to :func:`ov.genetics.regional_plot` as its "
        "``r2=`` argument. Pure numpy."
    ),
    examples=[
        "ld = ov.genetics.compute_ld_to_lead(geno, 'chr22:23456789')",
        "ld = ov.genetics.compute_ld_to_lead(geno, lead, snps=locus_snps)",
    ],
    related=["ov.genetics.regional_plot", "ov.genetics.finemap_locus_plot"],
)
def compute_ld_to_lead(genotype, lead_snp, *, snps=None) -> pd.Series:
    """Compute r^2 between a lead SNP and other SNPs from a genotype AnnData.

    Parameters
    ----------
    genotype
        Genotype AnnData of ``individuals x SNPs`` (0/1/2 allele dosages
        in ``.X``); ``.var_names`` are the SNP ids.
    lead_snp
        SNP id of the lead variant — must be present in
        ``genotype.var_names``.
    snps
        Optional subset / order of SNP ids to score; defaults to every SNP
        in ``genotype``. SNP ids absent from the genotype are returned as
        ``NaN``.

    Returns
    -------
    pandas.Series
        Per-SNP LD ``r^2`` to the lead SNP, indexed by SNP id (the lead
        SNP itself is ``1.0``).
    """
    var_names = list(map(str, genotype.var_names))
    pos = {s: i for i, s in enumerate(var_names)}
    lead_snp = str(lead_snp)
    if lead_snp not in pos:
        raise KeyError(
            f"lead SNP {lead_snp!r} is not in the genotype's var_names."
        )
    X = genotype.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = X.astype(float)
    # Mean-impute missing genotypes per SNP.
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

    target = [lead_snp] if snps is None else list(map(str, snps))
    lead_vec = X[:, pos[lead_snp]]
    lead_c = lead_vec - lead_vec.mean()
    lead_ss = float(np.sqrt(np.sum(lead_c ** 2)))

    out = {}
    for s in (var_names if snps is None else target):
        j = pos.get(s)
        if j is None:
            out[s] = np.nan
            continue
        v = X[:, j]
        vc = v - v.mean()
        denom = lead_ss * float(np.sqrt(np.sum(vc ** 2)))
        if denom == 0.0:
            out[s] = np.nan
        else:
            r = float(np.sum(lead_c * vc) / denom)
            out[s] = r * r
    return pd.Series(out, name="r2")


# --------------------------------------------------------------------------- #
# cis-eQTL gene screen                                                         #
# --------------------------------------------------------------------------- #
@register_function(
    aliases=[
        "scan_cis_genes", "cis_gene_screen", "screen_cis_eqtl",
        "cis基因筛选", "cis-eQTL筛选",
    ],
    category="genetics",
    description=(
        "Fast cis-eQTL screen across an expression panel — for every gene, "
        "find the single most strongly associated SNP within a cis window "
        "of its transcription start site and rank the genes by that best "
        "p-value. A quick way to pick a gene with a strong regulatory "
        "signal before a full association scan / fine-mapping. The genotype "
        "and expression AnnData must share the same individuals; the "
        "expression ``.var`` must carry a TSS column. Pure numpy / scipy."
    ),
    examples=[
        "screen = ov.genetics.scan_cis_genes(geno_qc, expr)",
        "screen = ov.genetics.scan_cis_genes(geno_qc, expr, cis_dist=5e5)",
    ],
    related=["ov.genetics.gwas_association", "ov.genetics.eqtl_map"],
)
def scan_cis_genes(
    geno,
    expr,
    *,
    cis_dist: float = 1e6,
    tss_col: str = "tss",
    pos_col: str = "pos",
    symbol_col: str = "gene_symbol",
    min_cis_snps: int = 5,
):
    """Screen an expression panel for genes with a strong cis-eQTL.

    Parameters
    ----------
    geno
        Genotype AnnData of ``individuals x SNPs`` (0/1/2 dosages).
        ``.var`` must carry the SNP base-pair position (``pos_col``).
    expr
        Expression AnnData of the **same** individuals x genes. ``.var``
        must carry the transcription start site (``tss_col``).
    cis_dist
        cis window half-width in base pairs (default 1 Mb).
    tss_col, pos_col
        Column names of the gene TSS and the SNP position.
    symbol_col
        Optional gene-symbol column in ``expr.var`` (used if present).
    min_cis_snps
        Skip a gene with fewer than this many SNPs in its cis window.

    Returns
    -------
    pandas.DataFrame
        One row per screened gene — ``gene``, ``symbol``, ``n_cis``,
        ``best_snp``, ``r`` (the best Pearson r) and ``p`` — sorted by
        ascending best p-value.
    """
    from scipy import stats as _stats

    if not (geno.obs_names == expr.obs_names).all():
        raise ValueError(
            "scan_cis_genes: geno and expr must share the same individuals "
            "in the same order."
        )
    X = np.asarray(geno.X, dtype=float)
    if hasattr(geno.X, "toarray"):
        X = geno.X.toarray().astype(float)
    snp_pos = geno.var[pos_col].to_numpy()
    has_symbol = symbol_col in expr.var.columns

    rows = []
    for gi, gene in enumerate(expr.var_names):
        tss = float(expr.var[tss_col].iloc[gi])
        y = np.asarray(expr[:, gene].X).ravel().astype(float)
        if y.std() == 0:
            continue
        cis = np.where(np.abs(snp_pos - tss) < cis_dist)[0]
        if len(cis) < min_cis_snps:
            continue
        best_p, best_snp, best_r = 1.0, None, 0.0
        for si in cis:
            x = X[:, si]
            if x.std() == 0:
                continue
            r, p = _stats.pearsonr(x, y)
            if p < best_p:
                best_p, best_snp, best_r = p, geno.var_names[si], r
        if best_snp is None:
            continue
        sym = expr.var[symbol_col].iloc[gi] if has_symbol else gene
        rows.append((gene, sym, int(len(cis)), best_snp, best_r, best_p))
    out = pd.DataFrame(
        rows, columns=["gene", "symbol", "n_cis", "best_snp", "r", "p"],
    )
    return out.sort_values("p").reset_index(drop=True)


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
# Distance-based LD pruning (instrument selection)                             #
# --------------------------------------------------------------------------- #
@register_function(
    aliases=[
        "prune_by_distance", "distance_prune", "ld_prune_distance",
        "select_instruments", "距离剪枝", "工具变量筛选",
    ],
    category="genetics",
    description=(
        "Distance-prune a ranked variant table to an approximately "
        "independent subset — the standard quick way to choose Mendelian-"
        "randomization instruments when a full LD matrix is not at hand. "
        "Walks the table in priority order (e.g. ascending p-value) and "
        "keeps a variant only if it is at least ``min_dist`` base pairs "
        "from every variant already kept. Pure pandas."
    ),
    examples=[
        "inst = ov.genetics.prune_by_distance(sig_eqtl, min_dist=1e4)",
        "inst = ov.genetics.prune_by_distance(hits, pos='BP', min_dist=5e5)",
    ],
    related=["ov.genetics.mendelian_randomization", "ov.genetics.clump_loci"],
)
def prune_by_distance(
    variants: pd.DataFrame,
    *,
    pos: str = "BP",
    min_dist: float = 1e4,
    rank_by: Optional[str] = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """Distance-prune a variant table to a near-independent subset.

    Parameters
    ----------
    variants
        Variant table — one row per variant, with a base-pair position.
    pos
        Base-pair position column.
    min_dist
        Minimum base-pair separation between any two kept variants.
    rank_by
        Optional column to sort by before pruning (e.g. ``'pvalue'``);
        if ``None`` the table's existing order is used as the priority.
    ascending
        Sort direction for ``rank_by``.

    Returns
    -------
    pandas.DataFrame
        The pruned subset, in priority order.
    """
    table = variants if rank_by is None else variants.sort_values(
        rank_by, ascending=ascending)
    kept, used = [], []
    for _, row in table.iterrows():
        bp = float(row[pos])
        if all(abs(bp - u) > min_dist for u in used):
            kept.append(row)
            used.append(bp)
    return pd.DataFrame(kept).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Colocalization scan over many genes                                          #
# --------------------------------------------------------------------------- #
@register_function(
    aliases=[
        "coloc_scan", "scan_colocalization", "colocalize_genes",
        "共定位扫描", "多基因共定位",
    ],
    category="genetics",
    description=(
        "Scan a GWAS locus against the cis-eQTLs of many genes and rank "
        "the genes by colocalization evidence (posterior PP.H4). For every "
        "eGene with enough variants shared with the GWAS, it builds the two "
        "coloc datasets and runs ``ov.genetics.colocalize`` — the right way "
        "to pick the gene a locus acts through (highest PP.H4), instead of "
        "guessing from proximity. Returns one row per gene with PP.H3 / "
        "PP.H4 and the shared-variant count, sorted by PP.H4."
    ),
    examples=[
        "tab = ov.genetics.coloc_scan(gwas_locus, eqtl_locus, n_gwas=173000, "
        "n_eqtl=670)",
    ],
    related=["ov.genetics.colocalize", "ov.genetics.make_coloc_dataset",
             "ov.genetics.coloc_plot"],
)
def coloc_scan(
    gwas: pd.DataFrame,
    eqtl: pd.DataFrame,
    *,
    n_gwas: int,
    n_eqtl: int,
    gene_col: str = "gene",
    variant_col: str = "variant",
    gwas_beta: str = "BETA",
    gwas_se: str = "SE",
    eqtl_beta: str = "beta",
    eqtl_se: str = "se",
    gwas_maf: str = "EAF",
    eqtl_maf: str = "maf",
    min_shared: int = 20,
) -> pd.DataFrame:
    """Scan a GWAS locus against many genes' eQTLs and rank by PP.H4.

    Parameters
    ----------
    gwas
        GWAS summary statistics at the locus — needs the variant id,
        effect size, standard error and (optionally) allele frequency.
    eqtl
        cis-eQTL summary statistics for several genes — needs the gene id,
        variant id, effect size, standard error and MAF.
    n_gwas, n_eqtl
        Sample sizes of the GWAS and the eQTL study.
    gene_col, variant_col
        Gene-id and variant-id column names.
    gwas_beta, gwas_se, eqtl_beta, eqtl_se
        Effect-size / standard-error column names in each table.
    gwas_maf, eqtl_maf
        Allele-frequency columns; the eQTL MAF is used, falling back to
        ``min(EAF, 1-EAF)`` from the GWAS.
    min_shared
        Skip a gene with fewer than this many variants shared with the
        GWAS.

    Returns
    -------
    pandas.DataFrame
        One row per gene — ``gene``, ``n_shared``, ``PP_H3``, ``PP_H4`` —
        sorted by descending ``PP_H4``.
    """
    from ._coloc import colocalize

    gwas_vars = set(gwas[variant_col])
    rows = []
    for gene, sub in eqtl.groupby(gene_col):
        shared = sorted(set(sub[variant_col]) & gwas_vars)
        if len(shared) < min_shared:
            continue
        merged = gwas.merge(
            sub, on=variant_col, suffixes=("_gwas", "_eqtl"),
        ).set_index(variant_col)
        snps = [s for s in shared if s in merged.index]
        if len(snps) < min_shared:
            continue
        emaf = merged.loc[snps, eqtl_maf].to_numpy(dtype=float)
        gmaf = np.minimum(merged.loc[snps, gwas_maf],
                          1.0 - merged.loc[snps, gwas_maf]).to_numpy()
        maf = np.where(np.isfinite(emaf), emaf, gmaf)
        d_gwas = make_coloc_dataset(
            merged, snps=snps, n=int(n_gwas), maf=maf,
            beta=gwas_beta, se=gwas_se,
        )
        d_eqtl = make_coloc_dataset(
            merged, snps=snps, n=int(n_eqtl), maf=maf,
            beta=eqtl_beta, se=eqtl_se,
        )
        co = colocalize(d_gwas, d_eqtl, method="abf")
        rows.append({
            "gene": gene, "n_shared": len(snps),
            "PP_H3": float(co["summary"]["PP.H3.abf"]),
            "PP_H4": float(co["summary"]["PP.H4.abf"]),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("PP_H4", ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Cross-trait colocalization (shared genetic factors / pleiotropy)             #
# --------------------------------------------------------------------------- #
@register_function(
    aliases=[
        "cross_trait_coloc", "multi_trait_coloc", "pleiotropy_coloc",
        "跨性状共定位", "多性状共定位",
    ],
    category="genetics",
    description=(
        "Find shared genetic factors across many traits by colocalization. "
        "Given GWAS summary statistics for multiple traits, scans every "
        "locus carrying a genome-wide-significant signal in two or more "
        "traits and runs pairwise coloc.abf (Giambartolomei 2014) between "
        "those traits at the locus, reporting the posterior probability of "
        "one shared causal variant (PP.H4) versus two distinct ones "
        "(PP.H3). This is the rigorous test for pleiotropy / a shared "
        "genetic factor between traits: two traits being independently "
        "genome-wide-significant at the same locus is NOT a shared factor "
        "— that is exactly the H3-vs-H4 question colocalization resolves. "
        "Summary-statistics-only — coloc.abf needs no LD reference and no "
        "genotypes. Returns one row per (locus, trait pair) ranked by "
        "PP.H4, with a colocalized flag. The trait-vs-trait analogue of "
        "ov.genetics.coloc_scan (GWAS-vs-eQTL)."
    ),
    examples=[
        "ov.genetics.cross_trait_coloc(sumstats, trait_col='trait', n=10000)",
        "ov.genetics.cross_trait_coloc(sumstats, n=n_by_trait, sdY=sd_by_trait, "
        "maf_col='EAF')",
    ],
    related=["ov.genetics.colocalize", "ov.genetics.coloc_scan",
             "ov.genetics.make_coloc_dataset", "ov.genetics.clump_loci"],
)
def cross_trait_coloc(
    stats,
    *,
    trait_col: str = "trait",
    variant_col: str = "variant",
    beta_col: str = "beta",
    se_col: str = "se",
    chrom_col: Optional[str] = None,
    position_col: Optional[str] = None,
    maf_col: Optional[str] = None,
    n=None,
    trait_type="quant",
    sdY=None,
    p_col: Optional[str] = None,
    p_threshold: float = 5e-8,
    locus_window: float = 1_000_000,
    min_shared: int = 20,
    h4_threshold: float = 0.8,
    p1: float = 1e-4,
    p2: float = 1e-4,
    p12: float = 1e-5,
) -> pd.DataFrame:
    """Find shared genetic factors across traits by pairwise colocalization.

    A "shared genetic factor" between two traits is a locus where both
    traits are driven by the **same causal variant** — pleiotropy. Two
    traits being independently genome-wide-significant at one locus does
    not establish that: they may have distinct causal variants in the
    same LD block. This function settles it the rigorous way — pairwise
    Bayesian colocalization (coloc.abf) of every trait pair at every
    multi-trait locus.

    Parameters
    ----------
    stats
        Long-format summary statistics — one row per (variant, trait) —
        or a ``dict`` ``{trait: per-variant DataFrame}``. Needs a
        variant id, a trait label, an effect size and its standard
        error; a chromosome + position (or a ``chrom:pos:...`` variant
        id); and, optionally, an allele-frequency column.
    trait_col, variant_col, beta_col, se_col
        Column names for the trait label, variant id, effect size and
        standard error.
    chrom_col, position_col
        Chromosome and base-position columns. If omitted, both are
        parsed from a ``chrom:pos:...`` style ``variant_col``.
    maf_col
        Optional allele-frequency column (effect-allele frequency is
        accepted; it is folded to MAF). If omitted, ``sdY`` carries the
        quantitative-trait scale instead.
    n
        Per-trait sample size — an ``int`` (same for all traits) or a
        ``dict`` ``{trait: n}``. Required.
    trait_type
        ``'quant'`` / ``'cc'`` — a string, or a ``dict`` ``{trait: type}``.
    sdY
        Trait standard deviation for a quantitative trait — a ``float``
        or a ``dict`` ``{trait: sdY}``. Supplying it (e.g. the SD of the
        measured phenotype) sharpens the quantitative-trait coloc.
    p_col
        Optional p-value column; if omitted, p is computed from
        ``beta / se`` via a two-sided z-test.
    p_threshold
        Genome-wide-significance threshold used to find the loci that
        carry a signal (default ``5e-8``).
    locus_window
        Base-pair window — significant variants within this distance
        are merged into one locus.
    min_shared
        Minimum variants shared by a trait pair at a locus to attempt
        colocalization.
    h4_threshold
        PP.H4 cutoff for the ``colocalized`` flag (default ``0.8``).
    p1, p2, p12
        coloc prior probabilities (trait 1, trait 2, both).

    Returns
    -------
    pandas.DataFrame
        One row per (locus, trait pair) — ``locus``, ``chrom``,
        ``start``, ``end``, ``trait_a``, ``trait_b``, ``n_shared``,
        ``PP_H3``, ``PP_H4``, ``colocalized`` — sorted by descending
        ``PP_H4``. The colocalized pairs are the shared genetic factors.
    """
    import itertools

    from scipy.stats import norm

    from ._coloc import colocalize

    if n is None:
        raise ValueError(
            "cross_trait_coloc requires per-trait sample size(s) — pass "
            "n=<int> or n={trait: n}.")

    # --- coerce input to a long DataFrame ------------------------------
    if isinstance(stats, dict):
        frames = []
        for tr, sub in stats.items():
            s = sub.copy()
            s[trait_col] = tr
            frames.append(s)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = stats.copy()

    df = df.dropna(subset=[variant_col, trait_col, beta_col, se_col])
    df = df[df[se_col].astype(float) > 0].copy()

    # --- chromosome / position -----------------------------------------
    if chrom_col is not None and position_col is not None:
        df["_chrom"] = df[chrom_col].astype(str)
        df["_pos"] = pd.to_numeric(df[position_col], errors="coerce")
    else:
        parts = df[variant_col].astype(str).str.split(":", expand=True)
        df["_chrom"] = parts[0].astype(str)
        df["_pos"] = pd.to_numeric(parts[1], errors="coerce")
    if df["_pos"].isna().any():
        raise ValueError(
            "cross_trait_coloc could not resolve variant positions — pass "
            "chrom_col and position_col explicitly.")

    # --- p-value -------------------------------------------------------
    if p_col is not None:
        df["_p"] = pd.to_numeric(df[p_col], errors="coerce")
    else:
        z = df[beta_col].astype(float) / df[se_col].astype(float)
        df["_p"] = 2.0 * norm.sf(np.abs(z))

    # --- define loci from genome-wide-significant variants -------------
    sig = df[df["_p"] < p_threshold]
    cols = ["locus", "chrom", "start", "end", "trait_a", "trait_b",
            "n_shared", "PP_H3", "PP_H4", "colocalized"]
    if sig.empty:
        return pd.DataFrame(columns=cols)
    loci = []
    for chrom, csig in sig.groupby("_chrom"):
        pos = np.sort(csig["_pos"].unique())
        start = prev = pos[0]
        for p in pos[1:]:
            if p - prev > locus_window:
                loci.append((chrom, start, prev))
                start = p
            prev = p
        loci.append((chrom, start, prev))

    def _resolve(spec, tr, default):
        if spec is None:
            return default
        if isinstance(spec, dict):
            return spec.get(tr, default)
        return spec

    pad = locus_window / 2.0
    rows = []
    for chrom, start, end in loci:
        region = df[(df["_chrom"] == chrom)
                    & (df["_pos"] >= start - pad)
                    & (df["_pos"] <= end + pad)]
        sig_traits = sorted(
            region.loc[region["_p"] < p_threshold, trait_col].unique())
        if len(sig_traits) < 2:
            continue
        per_trait = {
            tr: g.drop_duplicates(variant_col).set_index(variant_col)
            for tr, g in region[region[trait_col].isin(sig_traits)]
            .groupby(trait_col)
        }
        for ta, tb in itertools.combinations(sig_traits, 2):
            A, B = per_trait[ta], per_trait[tb]
            shared = sorted(set(A.index) & set(B.index))
            if len(shared) < min_shared:
                continue

            def _dataset(tbl, tr):
                if maf_col is not None:
                    eaf = tbl.loc[shared, maf_col].to_numpy(dtype=float)
                    maf = np.minimum(eaf, 1.0 - eaf)
                else:
                    maf = np.full(len(shared), 0.5)
                return make_coloc_dataset(
                    tbl, snps=shared, n=int(_resolve(n, tr, 0)),
                    maf=maf, beta=beta_col, se=se_col,
                    trait_type=_resolve(trait_type, tr, "quant"),
                    sdY=float(_resolve(sdY, tr, 1.0)),
                )

            try:
                co = colocalize(_dataset(A, ta), _dataset(B, tb),
                                method="abf", p1=p1, p2=p2, p12=p12)
                summ = co["summary"]
                h3 = float(summ["PP.H3.abf"])
                h4 = float(summ["PP.H4.abf"])
            except Exception:
                continue
            rows.append({
                "locus": f"{chrom}:{int(start)}-{int(end)}",
                "chrom": chrom, "start": float(start), "end": float(end),
                "trait_a": ta, "trait_b": tb, "n_shared": int(len(shared)),
                "PP_H3": h3, "PP_H4": h4,
                "colocalized": bool(h4 >= h4_threshold),
            })
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.sort_values("PP_H4", ascending=False).reset_index(drop=True)
    return out


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
        Effect / non-effect alleles for every weight. Each may be a fixed
        string (one allele for the whole model) **or** the name of a
        column in ``lead_eqtl`` carrying per-SNP alleles — the latter is
        what real eQTL tables (GTEx, eQTL Catalogue) provide.

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

    def _allele(spec):
        # A column name -> per-SNP alleles; otherwise a fixed string.
        if spec in lead_eqtl.columns:
            return lead_eqtl[spec].astype(str).to_numpy()
        return spec

    weights = pd.DataFrame({
        "rsid": lead_eqtl[snp_col].astype(str).to_numpy(),
        "gene": genes.to_numpy(),
        "weight": lead_eqtl[weight_col].astype(float).to_numpy(),
        "non_effect_allele": _allele(non_effect_allele),
        "effect_allele": _allele(effect_allele),
    })
    extra = pd.DataFrame({"gene": weights["gene"].unique()})
    extra["gene_name"] = extra["gene"]
    return pytwas.PredictionModel(
        weights=weights[["rsid", "gene", "weight",
                         "non_effect_allele", "effect_allele"]],
        extra=extra,
    )
