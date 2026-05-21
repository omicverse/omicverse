r"""
Alignment analysis utilities.

This module provides comprehensive tools for fastq data processing and alignment including:
- Alignment with kb-python (bulk / scRNA-seq)
- RNA velocity analysis with kb-python
- SRA download / conversion / QC / alignment / counting wrappers
- 16S amplicon pipeline: cutadapt + vsearch (merge / filter / derep / UNOISE3 / uchime3 / SINTAX / usearch_global)
"""

from .kb_api import single, ref, count, parallel_fastq_dump
from .simpleaf import simpleaf, simpleaf_index, simpleaf_count, simpleaf_pipeline
from .prefetch import prefetch
from .fq_dump import fqdump
from .fastp import fastp
from .STAR import STAR
from .featureCount import featureCount
from .pipeline import bulk_rnaseq_pipeline

# 16S / amplicon
from .cutadapt import cutadapt
from . import vsearch
from .amplicon_16s import amplicon_16s_pipeline, build_amplicon_anndata
from ._db import fetch_sintax_ref, fetch_silva, fetch_rdp

# Phylogeny (MSA + tree inference)
from .mafft import mafft
from .fasttree import fasttree
from .phylogeny import build_phylogeny

# DADA2 backend (pure-Python via pydada2)
from . import dada2
from .dada2 import dada2_pipeline

__all__ = [
    "single",
    "ref",
    "count",
    "parallel_fastq_dump",
    # simpleaf / salmon / alevin-fry backend
    "simpleaf",
    "simpleaf_index",
    "simpleaf_count",
    "simpleaf_pipeline",
    "prefetch",
    "fqdump",
    "fastp",
    "STAR",
    "featureCount",
    "bulk_rnaseq_pipeline",
    # 16S / amplicon
    "cutadapt",
    "vsearch",
    "amplicon_16s_pipeline",
    "build_amplicon_anndata",
    "fetch_sintax_ref",
    "fetch_silva",
    "fetch_rdp",
    # Phylogeny
    "mafft",
    "fasttree",
    "build_phylogeny",
    # DADA2
    "dada2",
    "dada2_pipeline",
]
