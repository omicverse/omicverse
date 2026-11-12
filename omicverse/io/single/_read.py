import os
import time
from pathlib import Path

import pandas as pd

try:
    from anndata.io import read_h5ad as _anndata_read_h5ad
except ImportError:
    from anndata import read_h5ad as _anndata_read_h5ad

from ..._registry import register_function


def _log_rust_read(path: str, size_mb: float | None, elapsed: float) -> None:
    size_str = f"  ({size_mb:.1f} MB)" if size_mb is not None else ""
    print(
        "📂 Reading with anndata-rs (Rust · out-of-memory)\n"
        f"   {path}{size_str}\n"
        f"   ✓ Loaded in {elapsed:.2f}s\n\n"
        "💡 Data stays on disk. Use ov.pp.* for chunked processing.\n"
        "   adata.close() when done · adata.to_adata() to materialise"
    )


def _h5ad_csr_sorted(path, n_sample_lanes: int = 256) -> bool:
    r"""Quick probe: if ``X`` is a CSR/CSC matrix, are its minor indices sorted
    within each lane? anndata-rs panics on unsorted minor indices with a cryptic
    Rust stack trace, so we pre-flight using ``h5py`` and up to ``n_sample_lanes``
    evenly-spaced lanes. Returns ``True`` when sorted / not sparse / can't be
    checked, ``False`` only when an actual violation is found.
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        return True
    try:
        with h5py.File(str(path), "r") as f:
            if "X" not in f:
                return True
            X = f["X"]
            if not isinstance(X, h5py.Group):
                return True
            enc = X.attrs.get("encoding-type", b"")
            if isinstance(enc, bytes):
                enc = enc.decode(errors="ignore")
            if "csr" not in enc and "csc" not in enc:
                return True
            if "indptr" not in X or "indices" not in X:
                return True
            indptr = np.asarray(X["indptr"][:], dtype=np.int64)
            indices_d = X["indices"]
            n_lanes = indptr.shape[0] - 1
            if n_lanes <= 0:
                return True
            step = max(1, n_lanes // n_sample_lanes)
            for i in range(0, n_lanes, step):
                s, e = int(indptr[i]), int(indptr[i + 1])
                if e - s <= 1:
                    continue
                lane = np.asarray(indices_d[s:e])
                if np.any(np.diff(lane) < 0):
                    return False
        return True
    except Exception:
        return True


def _read_h5ad_rust(path, **kwargs):
    try:
        import anndataoom
    except ImportError:
        raise ImportError(
            "ov.read(storage='disk') needs the 'anndataoom' package (a dependency of "
            "omicverse): pip install anndataoom. Or read into memory with "
            "ov.read(path, storage='memory')."
        ) from None

    if not _h5ad_csr_sorted(path):
        raise ValueError(
            f"Cannot read {path} with storage='disk': its sparse X matrix has "
            "unsorted minor indices, which anndata-rs rejects.\n"
            "Read it into memory instead with ov.read(path, storage='memory'), or "
            "rewrite it with sorted indices, e.g.:\n"
            "    import anndata as ad\n"
            f"    a = ad.read_h5ad('{path}')\n"
            "    a.X.sort_indices()\n"
            "    a.write_h5ad('<fixed>.h5ad')\n"
            "Or use ov.utils.convert_adata_for_rust(a, output_file='<fixed>.h5ad')."
        )

    kwargs.setdefault("backed", "r")
    try:
        size_mb = os.path.getsize(path) / 1024**2
    except OSError:
        size_mb = None

    t0 = time.perf_counter()
    adata = anndataoom.read(str(path), **kwargs)
    _log_rust_read(str(path), size_mb, time.perf_counter() - t0)
    return adata


def _read_h5ad_threaded(path):
    """``anndata.read_h5ad`` with a CSR ``X`` read by anndataoom-fast on all
    threads (int32 indices).  None when it does not apply (backend='python',
    anndataoom-fast without ``read_csr``, non-CSR ``X`` or a ``raw`` section)."""
    from ..._backend import resolve_backend
    if resolve_backend(None) != 'cpp':
        return None
    try:
        import h5py
        from anndataoom_fast.memory import read_csr
    except ImportError:
        return None
    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem
    import anndata
    with h5py.File(path, 'r') as f:
        if 'X' not in f or 'raw' in f or not isinstance(f['X'], h5py.Group):
            return None
        enc = f['X'].attrs.get('encoding-type', b'')
        enc = enc.decode() if isinstance(enc, bytes) else str(enc)
        if enc != 'csr_matrix':
            return None
        parts = {k: read_elem(f[k]) for k in
                 ('obs', 'var', 'uns', 'obsm', 'varm', 'obsp', 'varp', 'layers') if k in f}
    return anndata.AnnData(X=read_csr(path), **parts)


@register_function(
    aliases=["读取数据", "read", "load_data", "数据读取", "file_reader"],
    category="utils",
    description="Universal file reader for common bioinformatics data formats including h5ad, csv, tsv, txt, and gzipped files; h5ad read into memory (default) or kept on disk (storage='disk')",
    examples=[
        "# Read AnnData file into memory",
        "adata = ov.read('data.h5ad')",
        "# Keep the matrix on disk (data larger than RAM / GPU memory)",
        "adata = ov.read('data.h5ad', storage='disk')",
        "# Read CSV file",
        "df = ov.read('data.csv')",
        "# Read TSV file",
        "df = ov.read('data.tsv')",
        "# Read compressed file",
        "df = ov.read('data.csv.gz')",
        "# Pass additional parameters",
        "df = ov.read('data.csv', index_col=0, header=0)",
    ],
    related=["utils.read_csv", "utils.read_h5ad", "pp.preprocess", "settings.gpu_init"]
)
def read(path, storage='memory', *, backend=None, **kwargs):
    r"""
    Read common omics file formats into AnnData or pandas DataFrame.

    For ``.h5ad`` files two choices decide how omicverse runs: where the
    matrix lives (``storage``, here) and where it is computed on
    (``ov.settings.cpu_init()`` / ``cpu_gpu_mixed_init()`` / ``gpu_init()``,
    called *before* ``ov.read``):

    ==============  ======================  ===================================
    mode            ``storage='memory'``    ``storage='disk'``
    ==============  ======================  ===================================
    cpu             AnnData in RAM          AnnDataOOM (anndataoom): streamed
                                            from disk by the C++ kernels
    cpu-gpu-mixed   AnnData in RAM          AnnDataOOM; PCA/kNN/UMAP on GPU
    gpu             AnnData in RAM, moved   Dask row blocks streamed from disk
                    to the GPU (RAPIDS)     to the GPU (``chunk_size`` cells)
    ==============  ======================  ===================================

    ``storage='disk'`` keeps memory low (SEA-AD, 1.38M cells / 91 GB: 13 GB
    on CPU, 5 GB in GPU mode); every step re-reads the file, so it relies on
    disk / page-cache bandwidth. Not every omicverse / scanpy function accepts
    a disk-backed object yet; the default ``storage='memory'`` returns a plain
    ``anndata.AnnData``.

    Parameters
    ----------
    path : str or pathlib.Path
        Input file path.
    storage : {'memory', 'disk'}, default='memory'
        Where the ``.h5ad`` matrix lives; see the table above.
    backend : {'python', 'rust'}, optional
        Deprecated spelling: ``'python'`` = ``storage='memory'``, ``'rust'`` =
        AnnDataOOM whatever the mode.
    **kwargs
        Forwarded to the reader (``chunk_size`` and ``layer`` — read
        ``layers[layer]`` as ``X`` — for GPU ``storage='disk'``;
        ``pandas.read_csv`` options for tables).

    Returns
    -------
    anndata.AnnData, AnnDataOOM or pandas.DataFrame
        Loaded object (for ``.h5ad``) or DataFrame (for table files).

    Raises
    ------
    ImportError
        If ``anndataoom`` (CPU modes) or ``dask`` (GPU mode), both hard
        dependencies, are missing from the environment.
    ValueError
        If ``storage``/``backend`` is invalid, the file suffix is unsupported,
        or the ``.h5ad`` has an unsorted sparse ``X`` (AnnDataOOM; see
        :func:`ov.utils.convert_adata_for_rust`).
    """
    ext = Path(path).suffix.lower()
    if storage in ('python', 'rust') and backend is None:  # read(path, 'rust')
        storage, backend = 'memory', storage

    if ext == '.h5ad':
        if backend is not None:
            import warnings
            if backend not in ('python', 'rust'):
                raise ValueError("backend must be 'python' or 'rust' (deprecated; use storage=)")
            warnings.warn(
                f"ov.read(backend={backend!r}) is deprecated; use "
                f"storage={'memory' if backend == 'python' else 'disk'!r} "
                "(storage='disk' streams to the GPU in ov.settings.gpu_init() mode).",
                FutureWarning, stacklevel=2)
            if backend == 'python':
                return _anndata_read_h5ad(path, **kwargs)
            return _read_h5ad_rust(path, **kwargs)
        if storage == 'memory':
            fast = None if kwargs else _read_h5ad_threaded(path)
            return fast if fast is not None else _anndata_read_h5ad(path, **kwargs)
        if storage == 'disk':
            from ..._settings import settings
            if settings.mode == 'gpu':
                from ...pp._dask import read_h5ad as _read_h5ad_dask
                return _read_h5ad_dask(path, **kwargs)
            kwargs.pop('chunk_size', None)
            if kwargs.get('layer') is not None:
                raise NotImplementedError(
                    "layer= (read a layer as X) is available for storage='disk' in GPU mode")
            kwargs.pop('layer', None)
            return _read_h5ad_rust(path, **kwargs)
        raise ValueError("storage must be 'memory' or 'disk'")

    if ext in {'.csv', '.tsv', '.txt', '.gz'}:
        sep = '\t' if ext in {'.tsv', '.txt'} or path.endswith(('.tsv.gz', '.txt.gz')) else ','
        return pd.read_csv(path, sep=sep, **kwargs)

    raise ValueError('The type is not supported.')
