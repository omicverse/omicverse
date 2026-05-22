# -*- coding: utf-8 -*-
"""
scfea: single-cell Flux Estimation Analysis (vendored sub-package).

This sub-package is vendored from the scFEA project:
    https://github.com/changwn/scFEA
    commit 4c1fb76d52f07bafad84ce7686ad7c3acfcf0126

Reference:
    Alghamdi N, Chang W, Dang P, Lu X, Wan C, Gampala S, Huang Z, Wang J,
    Ma Q, Zang Y, Fishel M, Cao S, Zhang C. "A graph neural network model to
    estimate cell-wise metabolic flux using single-cell RNA-seq data."
    Genome Research, 2021. doi:10.1101/gr.271205.120

License: scFEA is free for academic, non-commercial use (see bundled LICENSE
file). Original author: Wennan Chang.

The upstream command-line tool has been wrapped into a clean Python API,
``run_scfea``, which accepts an in-memory expression DataFrame and returns
flux / balance DataFrames. The GNN architecture, loss and training loop are
kept bit-faithful to the original implementation.
"""

from .scFEA import run_scfea

__all__ = ['run_scfea']
