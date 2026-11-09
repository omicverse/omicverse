"""FastCCC: permutation-free cell-cell communication p-values.

Re-implementation of FastCCC 1.0.1 (https://github.com/Svvord/FastCCC, MIT)
by Hou, Ma & Zhou, *Nat Commun* 16, 11428 (2025), without the scanpy and
file-based I/O dependencies of the original package. See ``NOTICE``.
"""
from ._core import (
    FastCCCResult,
    cauchy_combine,
    prepare_input,
    run_fastccc_engine,
)
from ._db import CPDBTables, load_cpdb_tables

__all__ = ["CPDBTables", "FastCCCResult", "cauchy_combine", "load_cpdb_tables", "prepare_input",
           "run_fastccc_engine"]
