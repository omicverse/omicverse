"""SecAct secreted-protein activity and cell-cell communication (Ru et al., Nat Methods 2026).

A NumPy re-implementation of the R package ``SecAct`` (GPL-3) and of its Python
port ``SecActpy`` (MIT): ridge regression of expression changes on the SecAct
signature matrix with a SecAct-identical permutation null, plus the
``SecAct.CCC.scRNAseq`` and ``SecAct.CCC.scST`` communication pipelines.
No scanpy / secactpy dependency; the signature matrix is downloaded on demand.
"""
from ._core import (activity_inference, ccc_scrnaseq, ccc_spatial, find_neighbors, normalize_log, p_adjust_bh,
                    prepare_counts, scrnaseq_activity, spot_activity)
from ._rng import permutation_table, r_sample_replace
from ._ridge import ridge_permutation
from ._signature import cache_dir, fetch_resource, group_signatures, load_signature

__all__ = [
    "activity_inference", "ccc_scrnaseq", "ccc_spatial", "spot_activity", "scrnaseq_activity", "find_neighbors",
    "normalize_log",
    "p_adjust_bh", "prepare_counts", "permutation_table", "r_sample_replace", "ridge_permutation",
    "cache_dir", "fetch_resource", "group_signatures", "load_signature",
]
