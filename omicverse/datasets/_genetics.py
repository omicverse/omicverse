"""Real public genetics datasets for the ``ov.genetics`` GWAS tutorials.

Each loader downloads a dataset asset from the ``omicverse-data`` GitHub
release (``genetics-v1``) on first use, caches it under ``dir``, and
returns it ready for the ``ov.genetics`` post-GWAS workflow.

| Loader | Returns | Source |
|---|---|---|
| :func:`geuvadis_genotype` | AnnData (individuals × SNPs) | EBI ArrayExpress E-GEUV-1 |
| :func:`geuvadis_expression` | AnnData (samples × genes) | EBI ArrayExpress E-GEUV-1 |
| :func:`gwas_sumstats` | DataFrame (GWAS summary stats) | NHGRI-EBI GWAS Catalog GCST004627 |
| :func:`gtex_eqtl` | DataFrame (cis-eQTL) | GTEx v8 whole blood |
| :func:`genetics_scrna` | AnnData (cells × genes) | 10x Genomics PBMC 3k |
| :func:`recombination_map` | DataFrame (cM/Mb track) | SHAPEIT4 b37 / 1000 Genomes genetic map |
| :func:`gene_annotation` | DataFrame (gene models) | GENCODE v19 (GRCh37/hg19) |

All datasets are real, published human-genetics data redistributed at
tutorial scale (one chromosome / a thinned genome-wide set) from the
EBI ArrayExpress GEUVADIS project, the NHGRI-EBI GWAS Catalog, the GTEx
Consortium and 10x Genomics — every file retains its original open
license. The story is biologically coherent: GEUVADIS profiles
lymphoblastoid cell lines, the GWAS trait is **lymphocyte count**, and
the single-cell atlas is **PBMCs** — all immune-cell biology.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .._registry import register_function
from ._datasets import download_data

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData

_RELEASE = (
    "https://github.com/omicverse/omicverse-data/releases/download/genetics-v1"
)


def _fetch(asset: str, dir: str) -> str:
    """Download ``asset`` from the genetics-v1 release; return local path."""
    return download_data(f"{_RELEASE}/{asset}", file_path=asset, dir=dir)


@register_function(
    aliases=["geuvadis_genotype", "geuvadis_geno", "GEUVADIS基因型"],
    category="datasets",
    description=(
        "Real individual-level GEUVADIS genotypes — 462 unrelated "
        "1000-Genomes individuals from 5 populations (CEU/FIN/GBR/TSI = "
        "European, YRI = African) × 8000 common biallelic chr22 SNPs. "
        "Source: EBI ArrayExpress E-GEUV-1, 1000 Genomes phase 1 "
        "imputed genotypes. Returned as an AnnData (individuals × SNPs); "
        "``.X`` holds 0/1/2 allele dosages, ``.var`` carries chrom / pos "
        "/ effect allele A / other allele G / allele frequency, ``.obs`` "
        "carries the real population label. Anchors the individual-level "
        "GWAS / eQTL / fine-mapping pipeline."
    ),
    examples=["geno = ov.datasets.geuvadis_genotype()"],
)
def geuvadis_genotype(dir: str = "./data") -> "AnnData":
    """Load real GEUVADIS chr22 genotypes (462 individuals, 5 populations)."""
    import anndata as ad
    return ad.read_h5ad(_fetch("geuvadis_chr22_genotype.h5ad", dir))


@register_function(
    aliases=["geuvadis_expression", "geuvadis_expr", "GEUVADIS表达"],
    category="datasets",
    description=(
        "Real GEUVADIS lymphoblastoid-cell-line RNA-seq — 462 samples "
        "(the same individuals as :func:`geuvadis_genotype`) × 633 "
        "expressed chr22 genes, RPKM quantification. Source: EBI "
        "ArrayExpress E-GEUV-1 (GD462.GeneQuantRPKM). Returned as an "
        "AnnData (samples × genes); ``.var`` carries gene symbol / chrom "
        "/ TSS coordinate, ``.obs`` carries the population label. The "
        "real molecular phenotype for the cis-eQTL scan."
    ),
    examples=["expr = ov.datasets.geuvadis_expression()"],
)
def geuvadis_expression(dir: str = "./data") -> "AnnData":
    """Load real GEUVADIS chr22 lymphoblastoid RNA-seq (462 samples)."""
    import anndata as ad
    return ad.read_h5ad(_fetch("geuvadis_chr22_expression.h5ad", dir))


@register_function(
    aliases=["gwas_sumstats", "lymphocyte_gwas", "GWAS统计量"],
    category="datasets",
    description=(
        "Real GWAS summary statistics for blood lymphocyte count "
        "(Astle et al. 2017, Cell; NHGRI-EBI GWAS Catalog GCST004627, "
        "PMID 27863252; N≈173,480 Europeans). ``scope='chr22'`` "
        "(default) returns all ~400k chr22 variants — for "
        "colocalization and Mendelian randomization against the "
        "GEUVADIS chr22 eQTLs; ``scope='genomewide'`` returns a thinned "
        "genome-wide set for LD-score-regression heritability and the "
        "Manhattan overview. Returned as a DataFrame with canonical "
        "columns SNP / CHR / BP / A1 / A2 / BETA / SE / P / Z / N / EAF."
    ),
    examples=[
        "ss = ov.datasets.gwas_sumstats(scope='chr22')",
        "ss = ov.datasets.gwas_sumstats(scope='genomewide')",
    ],
)
def gwas_sumstats(scope: str = "chr22", dir: str = "./data") -> pd.DataFrame:
    """Load the real lymphocyte-count GWAS summary statistics."""
    if scope == "chr22":
        asset = "lymphocyte_count_chr22.tsv.gz"
    elif scope in ("genomewide", "gw"):
        asset = "lymphocyte_count_genomewide_thinned.tsv.gz"
    else:
        raise ValueError("scope must be 'chr22' or 'genomewide'")
    return pd.read_csv(_fetch(asset, dir), sep="\t")


@register_function(
    aliases=["gtex_eqtl", "gtex_wholeblood_eqtl", "GTEx_eQTL"],
    category="datasets",
    description=(
        "Real GTEx v8 whole-blood cis-eQTL summary statistics — the "
        "significant variant-gene pairs on chr22 (~66k pairs, ~398 "
        "genes). Source: GTEx Consortium v8 single-tissue cis-QTL. "
        "Whole blood is the GTEx tissue closest to the GEUVADIS "
        "lymphoblastoid lines, so this provides an independent, "
        "large-sample eQTL track for colocalization. Returned as a "
        "DataFrame: variant_id / gene / CHR / BP / A1 / A2 / maf / "
        "tss_distance / beta / se / pvalue."
    ),
    examples=["eqtl = ov.datasets.gtex_eqtl()"],
)
def gtex_eqtl(dir: str = "./data") -> pd.DataFrame:
    """Load real GTEx v8 whole-blood chr22 cis-eQTL summary statistics."""
    return pd.read_csv(_fetch("gtex_wholeblood_chr22_eqtl.tsv.gz", dir),
                       sep="\t")


@register_function(
    aliases=["genetics_scrna", "pbmc_immune_atlas", "免疫单细胞图谱"],
    category="datasets",
    description=(
        "Real single-cell RNA-seq immune atlas for scDRS — the 10x "
        "Genomics PBMC 3k dataset, 2638 quality-controlled peripheral "
        "blood mononuclear cells × 13,656 genes, raw UMI counts. "
        "``.obs['cell_type']`` carries the canonical PBMC labels "
        "(CD4 T / CD8 T / B / NK cells, CD14+ and FCGR3A+ monocytes, "
        "dendritic cells, megakaryocytes). PBMCs are the cellular "
        "context of the lymphocyte-count GWAS, so this is the natural "
        "atlas for single-cell disease-relevance scoring."
    ),
    examples=["adata = ov.datasets.genetics_scrna()"],
)
def genetics_scrna(dir: str = "./data") -> "AnnData":
    """Load the real PBMC 3k immune atlas (raw counts) for scDRS."""
    import anndata as ad
    return ad.read_h5ad(_fetch("pbmc3k_immune_atlas.h5ad", dir))


@register_function(
    aliases=["recombination_map", "recomb_map", "genetic_map", "重组图谱"],
    category="datasets",
    description=(
        "Real fine-scale recombination map for the LocusZoom "
        "recombination-rate track. Source: the SHAPEIT4 b37 genetic map "
        "(derived from the 1000 Genomes / HapMap recombination maps, "
        "GRCh37/hg19) — the recombination rate (cM/Mb) is the position "
        "derivative of the cumulative genetic map. Returned as a "
        "DataFrame with columns ``position`` (base pairs) and "
        "``rate_cM_per_Mb``; ``chrom='22'`` (default) covers the whole "
        "tutorial chromosome (16.05–51.23 Mb). Overlay it on a regional "
        "association plot with :func:`ov.genetics.regional_plot`."
    ),
    examples=[
        "rmap = ov.datasets.recombination_map()",
        "rmap = ov.datasets.recombination_map(chrom='22')",
    ],
)
def recombination_map(chrom: str = "22", dir: str = "./data") -> pd.DataFrame:
    """Load the real chr22 recombination map (cM/Mb) for the LocusZoom track."""
    if str(chrom) not in ("22",):
        raise ValueError("recombination_map currently ships chrom='22' only.")
    df = pd.read_csv(_fetch("recomb_map_chr22.tsv.gz", dir), sep="\t")
    df["chrom"] = str(chrom)
    return df


@register_function(
    aliases=["gene_annotation", "gene_models", "gene_track", "基因注释"],
    category="datasets",
    description=(
        "Real chr22 gene models for the LocusZoom gene-track panel. "
        "Source: GENCODE v19 (GRCh37/hg19) — the canonical hg19 "
        "annotation, restricted to protein-coding and lincRNA genes "
        "(579 genes on chr22). Returned as a DataFrame with columns "
        "``gene`` (symbol), ``chrom``, ``start``, ``end``, ``strand`` "
        "and ``gene_type``. Pass it to :func:`ov.genetics.regional_plot` "
        "or :func:`ov.genetics.finemap_locus_plot` to draw the arrowed "
        "gene boxes beneath a regional association plot."
    ),
    examples=[
        "genes = ov.datasets.gene_annotation()",
        "genes = ov.datasets.gene_annotation(chrom='22')",
    ],
)
def gene_annotation(chrom: str = "22", dir: str = "./data") -> pd.DataFrame:
    """Load real GENCODE v19 chr22 gene models for the LocusZoom gene track."""
    if str(chrom) not in ("22",):
        raise ValueError("gene_annotation currently ships chrom='22' only.")
    df = pd.read_csv(_fetch("genes_chr22.tsv.gz", dir), sep="\t",
                     dtype={"chrom": str})
    return df
