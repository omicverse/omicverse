"""Real bulk RNA-seq time-course datasets for the ``ov.bulk`` tutorials.

Each loader downloads a dataset asset from the ``omicverse-data`` GitHub
release (``timecourse-v1``) on first use, caches it under ``dir``, and
returns it ready for the ``ov.bulk`` time-course workflow
(``pyDEG.timecourse_deg``, ``temporal_clusters``).

| Loader | Returns | Source |
|---|---|---|
| :func:`fission_timecourse` | AnnData (samples × genes) | Bioconductor ``fission`` package |
| :func:`pombe_genesets` | dict of gene-set dicts | PomBase GO + Chen et al. 2003 CESR |

All datasets are real, published data redistributed from open sources:
the Bioconductor ``fission`` experiment-data package (Leong et al.,
*Nat Commun* 2014), the PomBase Gene Ontology annotation, and the
*S. pombe* Core Environmental Stress Response gene cores of Chen et al.
2003 (*Mol Biol Cell* 14:214-229).
"""
from __future__ import annotations

import gzip
import json
from typing import TYPE_CHECKING

from .._registry import register_function
from ._datasets import download_data

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData

_RELEASE = (
    "https://github.com/omicverse/omicverse-data/releases/download/timecourse-v1"
)


def _fetch(asset: str, dir: str) -> str:
    """Download ``asset`` from the timecourse-v1 release; return local path."""
    return download_data(f"{_RELEASE}/{asset}", file_path=asset, dir=dir)


@register_function(
    aliases=["fission_timecourse", "fission", "时间序列数据", "时序数据"],
    category="datasets",
    description=(
        "Real bulk RNA-seq time-course dataset — the Schizosaccharomyces "
        "pombe oxidative-stress series of the Bioconductor 'fission' "
        "package (Leong et al., Nat Commun 2014). Wild-type vs an "
        "atf21-delta deletion mutant, 6 time points (0/15/30/60/120/180 "
        "min), 3 replicates -> 36 samples × 7039 genes, raw counts. The "
        "canonical DESeq2-vignette two-group time-course dataset. "
        "Returned as an AnnData (samples × genes); ``.obs`` carries "
        "``strain`` (wt/mut), ``minute``, ``replicate`` and ``id``; "
        "``.var`` carries the gene ``symbol`` and ``biotype``. Feed "
        "straight into ``ov.bulk.pyDEG(...).timecourse_deg``."
    ),
    examples=["adata = ov.datasets.fission_timecourse()"],
)
def fission_timecourse(dir: str = "./data") -> "AnnData":
    """Load the real fission-yeast oxidative-stress RNA-seq time course."""
    import anndata as ad
    return ad.read_h5ad(_fetch("fission_timecourse.h5ad", dir))


@register_function(
    aliases=["pombe_genesets", "pombe_go", "fission_genesets", "酵母基因集"],
    category="datasets",
    description=(
        "Schizosaccharomyces pombe gene-set bundle for the time-course "
        "functional-enrichment step. Returns a dict with three keys: "
        "``'GO_BP'`` — ~1950 Gene Ontology biological-process gene sets "
        "built from the PomBase GAF and propagated over the GO DAG "
        "(``{term: [systematic ids]}``); ``'CESR'`` — the Core "
        "Environmental Stress Response induced / repressed gene cores of "
        "Chen et al. 2003 (Mol Biol Cell 14:214-229), the textbook "
        "S. pombe stress program; ``'gene_symbols'`` — a "
        "systematic-id -> symbol map. All identifiers are S. pombe "
        "systematic IDs (SPAC.../SPBC...), matching the "
        ":func:`fission_timecourse` count matrix. Pass ``GO_BP`` (or "
        "``CESR``) straight to ``ov.bulk.geneset_enrichment`` as the "
        "``pathways_dict``."
    ),
    examples=[
        "gs = ov.datasets.pombe_genesets()",
        "enr = ov.bulk.geneset_enrichment(genes, gs['GO_BP'], organism='Yeast')",
    ],
)
def pombe_genesets(dir: str = "./data") -> dict:
    """Load the S. pombe GO + CESR gene-set bundle (systematic IDs)."""
    path = _fetch("pombe_genesets.json.gz", dir)
    with gzip.open(path, "rt") as fh:
        return json.load(fh)
