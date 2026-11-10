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
            "Rust backend requires the 'anndataoom' package. "
            "Install with:  pip install omicverse[rust]   (or  pip install anndataoom)"
        ) from None

    if not _h5ad_csr_sorted(path):
        raise ValueError(
            f"Cannot read {path} with backend='rust': its sparse X matrix has "
            "unsorted minor indices, which anndata-rs rejects.\n"
            "Rewrite with sorted indices, e.g.:\n"
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


@register_function(
    aliases=["读取数据", "read", "load_data", "数据读取", "file_reader"],
    category="utils",
    description="Universal file reader for common bioinformatics data formats including h5ad, csv, tsv, txt, and gzipped files",
    examples=[
        "# Read AnnData file",
        "adata = ov.read('data.h5ad')",
        "# Read CSV file",
        "df = ov.read('data.csv')",
        "# Read TSV file",
        "df = ov.read('data.tsv')",
        "# Read compressed file",
        "df = ov.read('data.csv.gz')",
        "# Pass additional parameters",
        "df = ov.read('data.csv', index_col=0, header=0)",
        "# Chunked GPU mode: X stays a Dask array of row blocks",
        "adata = ov.read('data.h5ad', backend='dask', chunk_size=20000)"
    ],
    related=["utils.read_csv", "utils.read_h5ad", "pp.preprocess"]
)
def read(path, backend='python', **kwargs):
    r"""
    Read common omics file formats into AnnData or pandas DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path
        Input file path.
    backend : {'python', 'rust', 'dask'}, default='python'
        Backend used for ``.h5ad`` reading. ``'rust'`` loads out-of-memory
        via ``anndataoom`` / ``anndata-rs``. ``'dask'`` keeps ``X`` as a Dask
        array of row blocks (``chunk_size`` cells each, on the GPU by
        default) for the chunked GPU mode: ``ov.settings.gpu_init()`` then
        ``qc``/``preprocess``/``scale``/``pca``/``find_markers`` stream the
        blocks, so ``X`` never has to fit in GPU memory. When the file's sparse X has
        unsorted minor indices the call is aborted with a clear
        ``ValueError`` (rather than an anndata-rs panic) pointing at
        :func:`ov.utils.convert_adata_for_rust` for recovery.
    **kwargs
        Additional keyword arguments forwarded to backend readers.

    Returns
    -------
    anndata.AnnData or pandas.DataFrame
        Loaded AnnData object (for ``.h5ad``) or DataFrame (for table files).

    Raises
    ------
    ImportError
        If ``backend='rust'`` is requested but ``anndataoom`` is not installed.
    ValueError
        If ``backend`` is invalid for ``.h5ad`` reading, the file suffix is
        unsupported, or the ``.h5ad`` has an unsorted sparse ``X``.
    """
    ext = Path(path).suffix.lower()

    if ext == '.h5ad':
        if backend == 'python':
            return _anndata_read_h5ad(path, **kwargs)

        if backend == 'rust':
            return _read_h5ad_rust(path, **kwargs)

        if backend == 'dask':
            from ...pp._dask import read_h5ad as _read_h5ad_dask
            return _read_h5ad_dask(path, **kwargs)

        raise ValueError("backend must be 'python', 'rust' or 'dask'")

    if ext in {'.csv', '.tsv', '.txt', '.gz'}:
        sep = '\t' if ext in {'.tsv', '.txt'} or path.endswith(('.tsv.gz', '.txt.gz')) else ','
        return pd.read_csv(path, sep=sep, **kwargs)

    raise ValueError('The type is not supported.')
