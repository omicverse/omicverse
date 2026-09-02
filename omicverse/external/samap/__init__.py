"""Vendored SAMap (Tarashansky et al. 2021, eLife) + sam-algorithm.

Source: https://github.com/atarashansky/SAMap (Tarashansky, Musser, Khariton,
Li & Wang, Stanford University). Upstream is MIT-licensed; the license is
carried verbatim beside this file in ``LICENSE`` and applies to every source
file in this directory.

Cross-species single-cell integration via BLAST protein-homology graphs and the
Self-Assembling Manifold algorithm. Vendored (source copied) rather than pip
-installed because upstream pins ``numpy==1.23.5`` / ``scanpy==1.9.3`` /
``numba==0.56.3``, which conflict with omicverse's environment. Heavy runtime
deps (hnswlib, numba, leidenalg) and external BLAST binaries are required only
at call time.

The re-exports below (``pd``/``sp``/``np``/``ut``/``sc``/``q``) reproduce the
upstream ``samap/__init__.py`` namespace that the vendored modules import from.
"""
import warnings

import pandas as pd
import scipy as sp
import numpy as np
import scanpy as sc
import scipy.sparse as _spsparse


def _install_sparse_compat_shim():
    """Restore the ``.A`` / ``.A1`` sparse-matrix attributes that SciPy >=1.13
    removed from ``coo``/``lil``/etc.

    SAMap was written against SciPy <1.13, where every sparse matrix exposed
    ``.A`` (dense ndarray) and ``.A1`` (flattened). Rather than editing ~50 call
    sites across the vendored source, we re-add these harmless, backward-
    compatible properties to the sparse classes at import time.
    """
    classes = []
    for name in ("coo_matrix", "csr_matrix", "csc_matrix", "lil_matrix",
                 "dia_matrix", "bsr_matrix", "dok_matrix"):
        cls = getattr(_spsparse, name, None)
        if cls is not None:
            classes.append(cls)
    for cls in classes:
        if not hasattr(cls, "A"):
            cls.A = property(lambda self: self.toarray())
        if not hasattr(cls, "A1"):
            cls.A1 = property(lambda self: self.toarray().ravel())


_install_sparse_compat_shim()

from .samalg import utilities as ut

__version__ = "1.0.16"


def q(x):
    return np.array(list(x))
