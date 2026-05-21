"""Real public immune-repertoire (AIRR) datasets for the ``ov.airr`` tutorials.

Each loader downloads a dataset asset from the ``omicverse-data`` GitHub
release (``airr-v1``) on first use, caches it under ``dir``, and returns
it ready for the ``ov.airr`` adaptive-immune-receptor-repertoire workflow
(clonotype calling, clonal expansion, SHM, lineage reconstruction).

The AIRR tutorial chapter spans several modalities, one real dataset each:

| Loader | Returns | Modality | Source |
|---|---|---|---|
| :func:`airr_singlecell` | AnnData (cells x genes) | single-cell TCR + GEX | Wu et al., Nature 2020 |
| :func:`airr_bcr` | DataFrame (AIRR rearrangement) | B-cell BCR (SHM / lineage) | Laserson et al., PNAS 2014 |
| :func:`airr_tcr_antigen` | AnnData (cells x genes) | antigen-labelled scTCR + GEX | 10x Genomics dCODE dextramer |
| :func:`vdjdb_reference` | DataFrame (TCR-epitope) | antigen-specificity reference | VDJdb (antigenomics) |
| *(none)* | -- | bulk TCR | ships inside ``pyimmunarch`` |

The **bulk TCR** modality needs no loader: the real 12-sample TCR-beta
multiple-sclerosis-vs-healthy cohort (``immdata``) is bundled inside the
``pyimmunarch`` package and is loaded directly through the ``ov.airr``
bulk backend via ``pyimmunarch.load_example_immdata()``.

All datasets are real, published immune-repertoire data redistributed at
tutorial scale from open sources — the scverse ``scirpy`` example data
(Wu et al. 2020 tumour-infiltrating T cells) and the Immcantation
framework example data (the Laserson et al. 2014 influenza-vaccination
IgH repertoire) — each retaining its original open license.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .._registry import register_function
from ._datasets import download_data

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData

_RELEASE = (
    "https://github.com/omicverse/omicverse-data/releases/download/airr-v1"
)


def _fetch(asset: str, dir: str) -> str:
    """Download ``asset`` from the airr-v1 release; return local path."""
    return download_data(f"{_RELEASE}/{asset}", file_path=asset, dir=dir)


@register_function(
    aliases=[
        "airr_singlecell", "airr_sctcr", "wu2020", "wu2020_sctcr",
        "single_cell_tcr", "单细胞TCR", "单细胞免疫组库",
    ],
    category="datasets",
    description=(
        "Real single-cell TCR + gene-expression dataset for the "
        "single-cell arm of the AIRR tutorial — Wu et al. (Nature 2020, "
        "579:274-278; PMID 32103181) 10x 5' scTCR-seq + GEX of "
        "tumour-infiltrating T cells. Obtained via "
        "``scirpy.datasets.wu2020()`` and curated to a balanced 5,001-cell "
        "x 13,968-gene tutorial subset spanning 14 patients and 3 tissue "
        "sources (Tumor / NAT / Blood). Returned as an AnnData; ``.X`` "
        "holds raw UMI counts, ``.obsm['airr']`` holds the per-cell TCR "
        "rearrangements in scirpy's awkward-array format, and ``.obs`` "
        "carries ``patient`` / ``sample`` / ``source`` / ``cluster_orig`` "
        "/ ``cell_type`` / ``clonotype_orig``. Feed straight into the "
        "scirpy / ``ov.airr`` single-cell clonotype workflow "
        "(``ir.pp.index_chains``, ``ir.tl.chain_qc``, "
        "``ir.tl.define_clonotypes``)."
    ),
    examples=[
        "adata = ov.datasets.airr_singlecell()",
        "import scirpy as ir; ir.pp.index_chains(adata)",
    ],
)
def airr_singlecell(dir: str = "./data") -> "AnnData":
    """Load the real Wu 2020 single-cell TCR + GEX dataset (5k cells)."""
    import anndata as ad
    return ad.read_h5ad(_fetch("wu2020_sctcr.h5ad", dir))


@register_function(
    aliases=[
        "airr_bcr", "bcr_repertoire", "example_bcr", "immcantation_bcr",
        "B细胞免疫组库", "BCR数据",
    ],
    category="datasets",
    description=(
        "Real B-cell receptor (BCR) repertoire for the SHM / lineage arm "
        "of the AIRR tutorial — the Immcantation ``alakazam::ExampleDb``, "
        "a single subject's influenza-vaccination IgH repertoire from "
        "Laserson et al. (PNAS 2014, 111:4928-4933; PMID 24639495). "
        "1,999 IGH rearrangements across 1,198 Change-O clones and two "
        "timepoints (-1h pre-vaccination, +7d post). Returned as a pandas "
        "DataFrame in AIRR rearrangement format with columns including "
        "``sequence_alignment`` / ``germline_alignment`` / "
        "``germline_alignment_d_mask`` / ``v_call`` / ``j_call`` / "
        "``junction`` / ``clone_id`` / ``c_call`` (isotype) / "
        "``sample_id`` — everything needed for clonal clustering, "
        "somatic-hypermutation quantification and B-cell lineage-tree "
        "reconstruction in the ``ov.airr`` BCR workflow."
    ),
    examples=[
        "bcr = ov.datasets.airr_bcr()",
        "bcr['clone_id'].nunique()",
    ],
)
def airr_bcr(dir: str = "./data") -> pd.DataFrame:
    """Load the real Laserson 2014 B-cell IgH AIRR repertoire (1999 seqs)."""
    return pd.read_csv(_fetch("bcr_repertoire.tsv.gz", dir), sep="\t")


@register_function(
    aliases=[
        "airr_tcr_antigen", "tcr_antigen", "tcr_dextramer", "dextramer_tcr",
        "tcr_specificity", "tcr_gex_antigen", "抗原标记单细胞TCR", "dCODE抗原TCR",
    ],
    category="datasets",
    description=(
        "Real antigen-labelled single-cell TCR + gene-expression dataset for "
        "the TCR-specificity and conga-style TCR+GEX joint-analysis arms of "
        "the AIRR tutorial — the 10x Genomics 'A New Way of Exploring "
        "Immunity' dCODE Dextramer experiment (CD8+ T cells of healthy "
        "donor 1, Chromium 5' v1, Cell Ranger 3.0.2). Every cell carries "
        "paired TCR alpha/beta contigs, gene expression, 14 TotalSeq-C "
        "surface-marker antibodies and 44 dCODE pMHC dextramer UMI counts "
        "with the 10x-published binarised binder calls. Curated to a "
        "6,500-cell x ~2,000-gene tutorial subset (each of the 4 dominant "
        "epitopes capped at 1,200 cells, plus 858 unbound cells). Returned "
        "as an AnnData: ``.X`` holds log-normalised expression, "
        "``.layers['counts']`` raw UMIs, ``.obsm['X_pca']`` / "
        "``['X_umap']`` a precomputed embedding, ``.obs['leiden']`` a "
        "cluster label, ``.obs['antigen']`` / ``antigen_epitope`` / "
        "``antigen_species`` / ``antigen_hla`` / ``is_antigen_bound`` the "
        "dextramer-derived specificity call, the per-cell ``ov.airr`` chain "
        "slots (``VJ_1_*`` / ``VDJ_1_*`` …) in ``.obs``, the raw contig "
        "table in ``.uns['airr_contigs']``, and the dextramer / antibody "
        "UMI matrices in ``.obsm['dextramer_umi']`` / "
        "``.obsm['protein_adt']``. Feeds straight into the ``ov.airr`` "
        "single-cell clonotype + specificity workflow."
    ),
    examples=[
        "adata = ov.datasets.airr_tcr_antigen()",
        "adata.obs['antigen'].value_counts()",
    ],
)
def airr_tcr_antigen(dir: str = "./data") -> "AnnData":
    """Load the real 10x dextramer antigen-labelled scTCR+GEX dataset (6.5k cells)."""
    import anndata as ad
    return ad.read_h5ad(_fetch("tcr_antigen_dextramer.h5ad", dir))


@register_function(
    aliases=[
        "vdjdb_reference", "vdjdb", "vdjdb_human", "tcr_epitope_reference",
        "antigen_reference", "VDJdb参考库", "TCR抗原特异性参考",
    ],
    category="datasets",
    description=(
        "Curated human TCR-epitope antigen-specificity reference table for "
        "the TCR-specificity arm of the AIRR tutorial — VDJdb, the "
        "community-curated database of T-cell-receptor antigen "
        "specificities (antigenomics/vdjdb-db, 'Blooming May' 2026-05-16 "
        "release). Filtered to human TRA / TRB records with a usable CDR3 "
        "and epitope, deduplicated to a compact ~132k-row reference. "
        "Returned as a pandas DataFrame with columns ``gene`` (TRA/TRB), "
        "``cdr3_aa``, ``v_call``, ``j_call``, ``antigen_epitope``, "
        "``antigen_gene``, ``antigen_species`` (CMV / EBV / InfluenzaA / "
        "SARS-CoV-2 / HomoSapiens …), ``mhc`` and ``mhc_class`` — the "
        "reference an observed repertoire is matched against to annotate "
        "antigen specificity (CDR3 / V-J exact or fuzzy match)."
    ),
    examples=[
        "ref = ov.datasets.vdjdb_reference()",
        "ref[ref['antigen_species'] == 'SARS-CoV-2']",
    ],
)
def vdjdb_reference(dir: str = "./data") -> pd.DataFrame:
    """Load the curated human VDJdb TCR-epitope reference table (~132k rows)."""
    return pd.read_csv(_fetch("vdjdb_reference.tsv.gz", dir), sep="\t")
