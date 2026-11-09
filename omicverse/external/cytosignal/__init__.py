"""CytoSignal: cellular-resolution spatial ligand-receptor signalling.

NumPy/SciPy port of the R package cytosignal 0.5.1 (Liu et al., Nat Genet
2026; GPL >= 3). ``_core.run_pipeline`` reproduces the R vignette workflow
(neighbour graphs, LR-scores, permutation p-values with R's RNG, spatial FDR,
significant cells); ``_group`` summarises the cell-level result per
sender/receiver cell group for ``run_ccc``.
"""
from ._core import run_pipeline
from ._db import CytoSignalDB, load_cytosignal_db

__all__ = ["CytoSignalDB", "load_cytosignal_db", "run_pipeline"]
