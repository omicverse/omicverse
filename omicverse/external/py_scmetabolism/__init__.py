"""
py_scmetabolism: Python reimplementation of scMetabolism for single-cell metabolism analysis.

This package provides functions to quantify metabolic pathway activity from single-cell RNA-seq data,
compatible with AnnData objects. It supports multiple scoring methods: VISION, AUCell, ssGSEA, GSVA,
and optional imputation using ALRA.

The main functions are:
    - sc_metabolism: compute metabolism scores for a count matrix
    - sc_metabolism_anndata: compute metabolism scores and store in AnnData object
    - dimplot_metabolism: visualize scores on dimensionality reduction
    - dotplot_metabolism: dot plot of pathway scores across groups
    - boxplot_metabolism: box plot of pathway scores across groups

All functions aim to produce results numerically identical to the original R package scMetabolism.
"""

from .compute import sc_metabolism, sc_metabolism_anndata
from .visualize import dimplot_metabolism, dotplot_metabolism, boxplot_metabolism

__version__ = "0.1.0"
__all__ = [
    "sc_metabolism",
    "sc_metabolism_anndata",
    "dimplot_metabolism",
    "dotplot_metabolism",
    "boxplot_metabolism",
]