"""Helpers for disk-backed ``AnnDataOOM`` objects (``ov.read(path, storage='disk')``)."""

from __future__ import annotations


def to_memory(adata):
    """Return an in-memory :class:`anndata.AnnData` for ``adata``.

    ``AnnDataOOM`` objects (from ``ov.read(path, storage='disk')``) are fully
    loaded into RAM (X, layers, raw, obs/var, obsm/varm/obsp/varp, uns); any
    other object, e.g. an ``anndata.AnnData``, is returned unchanged.  Use it
    before handing the data to third-party tools (scanpy, squidpy, scvi-tools,
    ...) that expect a real ``AnnData``, or once before many omicverse calls
    that would each materialise the disk-backed object.

    Parameters
    ----------
    adata : anndata.AnnData or AnnDataOOM
        Data object.

    Returns
    -------
    anndata.AnnData
        ``adata`` itself if it is already in memory, else a new in-memory copy.

    Examples
    --------
    >>> adata = ov.read('data.h5ad', storage='disk')
    >>> mem = ov.utils.to_memory(adata)
    """
    from .._oom_compat import is_oom
    if not is_oom(adata):
        return adata
    from .._oom_guard import _materialise, estimate_nbytes, available_memory, _gb
    need, avail = estimate_nbytes(adata), available_memory()
    if avail is not None and need > avail:
        raise MemoryError(
            f"to_memory(): the AnnDataOOM ({adata.n_obs:,} x {adata.n_vars:,}) needs "
            f"~{_gb(need)} but only {_gb(avail)} RAM is available; subset it first "
            f"(adata[cells, genes])."
        )
    return _materialise(adata)
