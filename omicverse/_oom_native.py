"""Functions that handle a disk-backed ``AnnDataOOM`` themselves.

Every function decorated with :func:`omicverse._registry.register_function`
goes through the OOM guard (:mod:`omicverse._oom_guard`): when it receives an
``AnnDataOOM`` it is run on an in-memory copy and its results are copied back.
The functions listed here already stream the on-disk matrix (or otherwise work
on ``AnnDataOOM`` directly) and bypass that guard.

Names are the *definition* paths (``func.__module__ + '.' + func.__qualname__``),
e.g. ``ov.pp.qc`` is defined as ``omicverse.pp._qc.qc``.  Registered classes are
never wrapped by the registry, so they need no entry (``ov.single.MetaCell``,
``ov.single.Augur`` already work on ``AnnDataOOM``).

Calls made *from inside* a native function (e.g. ``ov.pp.preprocess`` calling
other registered helpers on the same ``AnnDataOOM``) are not guarded either.

To declare a new function native, add its definition path here, or call
:func:`register_native` at runtime.
"""

from __future__ import annotations

OOM_NATIVE: set = {
    # ---- ov.pp: chunked / C++ streaming paths (is_oom branches) ----
    "omicverse.pp._qc.qc",
    "omicverse.pp._qc.filter_cells",
    "omicverse.pp._qc.filter_genes",
    "omicverse.pp._preprocess.preprocess",
    "omicverse.pp._preprocess.identify_robust_genes",
    "omicverse.pp._preprocess.highly_variable_features",
    "omicverse.pp._preprocess.highly_variable_genes",
    "omicverse.pp._preprocess.normalize_pearson_residuals",
    "omicverse.pp._preprocess.scale",
    "omicverse.pp._preprocess.pca",
    "omicverse.pp._preprocess.neighbors",
    "omicverse.pp._preprocess.umap",
    "omicverse.pp._preprocess.leiden",
    "omicverse.pp._preprocess.score_genes_cell_cycle",
    "omicverse.pp._normalization.normalize_total",
    "omicverse.pp._normalization.log1p",
    "omicverse.pp._highly_variable_genes.highly_variable_genes",
    "omicverse.pp._pca.pca",
    "omicverse.pp._scrublet.scrublet",
    # ---- ov.single ----
    # find_markers carries its own guard (streams COSG with backend='cpp',
    # materialises the other methods itself).
    "omicverse.single._markers.find_markers",
    "omicverse.single._markers.get_markers",
    # batch_correction: harmony runs on obsm; other methods materialise inside.
    "omicverse.single._batch.batch_correction",
    # ---- ov.pl: read obs / obsm / obs_vector only ----
    "omicverse.pl._single.embedding",
    "omicverse.pl._single.cellproportion",
    "omicverse.pl._dotplot.dotplot",
    "omicverse.pl._violin.violin",
    "omicverse.pl._spatial.spatial",
    "omicverse.pl._multi.embedding_multi",
    # ---- ov.space ----
    "omicverse.space._svg.spatial_neighbors",
    "omicverse.space._svg.spatial_autocorr",
}


def qualified_name(func) -> str:
    """Definition path ``module.qualname`` used as the allow-list key."""
    return f"{getattr(func, '__module__', '')}.{getattr(func, '__qualname__', getattr(func, '__name__', ''))}"


def is_native(func) -> bool:
    """True if ``func`` handles ``AnnDataOOM`` itself (allow-listed or flagged
    with ``func._oom_native = True``)."""
    if getattr(func, "_oom_native", False):
        return True
    return qualified_name(func) in OOM_NATIVE


def register_native(*names_or_funcs) -> None:
    """Mark functions (callables or definition paths) as OOM-native at runtime."""
    for item in names_or_funcs:
        OOM_NATIVE.add(item if isinstance(item, str) else qualified_name(item))


__all__ = ["OOM_NATIVE", "is_native", "qualified_name", "register_native"]
