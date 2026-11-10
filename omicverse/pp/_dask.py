"""Chunked GPU mode: AnnData whose ``X`` is a Dask array of row blocks.

In ``ov.settings.gpu_init()`` mode, ``ov.read(path, storage='disk')`` opens an
``.h5ad`` lazily: ``X`` becomes a Dask array whose blocks are CSR row slices
read straight from the file and copied to the GPU.  Then
``qc``, ``preprocess``, ``scale``, ``pca`` and ``find_markers(method='cosg')``
stream those blocks through the GPU, so the matrix never has to fit in GPU (or
host) memory; everything after PCA works on the small embedding as usual.
With a ``dask.distributed`` client on a ``dask_cuda.LocalCUDACluster``
(``ov.settings.gpu_init(dask=True)``) the blocks are spread over all GPUs.

Every pass over the matrix is one ``dask.compute`` of per-block partial
results (sums, Gram matrices, ...) that are added up in float64.  Per-gene
statistics are cached by the Dask array's name (a hash of how it was built), so
steps that need the same numbers (gene filtering, robust genes, Pearson HVGs,
scaling) share passes: the standard workflow reads the matrix five times
(QC, gene filter, HVG + scaling statistics, PCA, markers).

The block functions work on cupyx (GPU) and scipy (CPU) blocks alike.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# array helpers
# ---------------------------------------------------------------------------
def is_dask(X) -> bool:
    """True when ``X`` is a Dask array."""
    if type(X).__module__.split(".")[0] != "dask":
        return False
    import dask.array as da
    return isinstance(X, da.Array)


def _on_gpu(a) -> bool:
    return type(a).__module__.startswith(("cupy", "cupyx"))


def _xp(a):
    if _on_gpu(a):
        import cupy
        return cupy
    return np


def _sparse_mod(a):
    if _on_gpu(a):
        import cupyx.scipy.sparse as s
        return s
    import scipy.sparse as s
    return s


def _issparse(a) -> bool:
    return _sparse_mod(a).issparse(a)


def _to_numpy(a):
    return a.get() if hasattr(a, "get") and _on_gpu(a) else np.asarray(a)


def _meta(X):
    return X._meta


def _meta_is_sparse(X) -> bool:
    return _issparse(_meta(X))


def _row_blocks(X):
    """Delayed row blocks of ``X`` (one column chunk), keeping task keys so
    blocks shared by several arrays are read once per compute."""
    if X.numblocks[1] != 1:
        X = X.rechunk({1: -1})
    return list(X.to_delayed(optimize_graph=False).ravel())


def _compute(*tasks):
    """``dask.compute`` on the active scheduler.  Without a distributed client
    the threaded scheduler runs at most ``OV_DASK_WORKERS`` (default 8) blocks
    at once, which keeps one GPU busy without stacking block temporaries."""
    import dask
    try:
        from distributed import default_client
        default_client()
        return dask.compute(*tasks)
    except (ImportError, ValueError):
        pass
    workers = int(os.environ.get("OV_DASK_WORKERS", "8"))
    return dask.compute(*tasks, scheduler="threads", num_workers=workers)


def _persist(*arrays):
    import dask
    try:
        from distributed import default_client
        default_client()
        return dask.persist(*arrays)
    except (ImportError, ValueError):
        pass
    return dask.persist(*arrays, scheduler="threads",
                        num_workers=int(os.environ.get("OV_DASK_WORKERS", "8")))


def _row_ids(indptr, nnz):
    """Row index of every stored value of a CSR block."""
    xp = _xp(indptr)
    return xp.searchsorted(indptr[1:], xp.arange(nnz, dtype=indptr.dtype), side="right")


# ---------------------------------------------------------------------------
# CUDA kernels (one thread per row, float64 atomics; no per-block temporaries)
# ---------------------------------------------------------------------------
_KERNEL_SRC = r"""
template <typename T, typename I>
__global__ void row_gene_stats(const I* indptr, const int* indices, const T* data, long long n_rows,
                               const unsigned char* sets, int n_sets,
                               double* gs, double* gq, double* gz,
                               double* rs, double* rz, double* rset) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    double s = 0, z = 0;
    double acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (I i = indptr[row]; i < indptr[row + 1]; ++i) {
        double v = (double)data[i];
        int j = indices[i];
        s += v;
        if (v != 0) z += 1;
        if (gs) { atomicAdd(gs + j, v); atomicAdd(gq + j, v * v); if (v != 0) atomicAdd(gz + j, 1.0); }
        if (n_sets) {
            unsigned char m = sets[j];
            for (int k = 0; k < n_sets; ++k) if (m & (1u << k)) acc[k] += v;
        }
    }
    if (rs) { rs[row] = s; rz[row] = z; }
    for (int k = 0; k < n_sets; ++k) rset[row * n_sets + k] = acc[k];
}

template <typename T, typename I>
__global__ void normalize_rows(const I* indptr, const T* data, long long n_rows, double target,
                               T* out) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    double s = 0;
    for (I i = indptr[row]; i < indptr[row + 1]; ++i) s += (double)data[i];
    double f = s > 0 ? target / s : 1.0;
    for (I i = indptr[row]; i < indptr[row + 1]; ++i) out[i] = (T)((double)data[i] * f);
}

template <typename T, typename I>
__global__ void pr_stored(const I* indptr, const int* indices, const T* data, long long n_rows,
                          const double* g, double inv_theta, double clip,
                          double* rowsum, double* s1, double* s2) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    double s = 0;
    for (I i = indptr[row]; i < indptr[row + 1]; ++i) s += (double)data[i];
    rowsum[row] = s;
    for (I i = indptr[row]; i < indptr[row + 1]; ++i) {
        int j = indices[i];
        double mu = s * g[j];
        if (!(mu > 0)) continue;
        double inv = rsqrt(mu + mu * mu * inv_theta);
        double r = fmin(fmax(((double)data[i] - mu) * inv, -clip), clip);
        double r0 = fmin(fmax(-mu * inv, -clip), clip);
        atomicAdd(s1 + j, r - r0);
        atomicAdd(s2 + j, r * r - r0 * r0);
    }
}

template <typename T, typename I>
__global__ void group_rows(const I* indptr, const int* indices, const T* data, long long n_rows,
                           const int* codes, long long n_genes, double* gs, double* gq, double* gz) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    long long base = (long long)codes[row] * n_genes;
    for (I i = indptr[row]; i < indptr[row + 1]; ++i) {
        double v = (double)data[i];
        long long j = base + indices[i];
        atomicAdd(gs + j, v); atomicAdd(gq + j, v * v); if (v != 0) atomicAdd(gz + j, 1.0);
    }
}
"""
_KERNELS = {}


def _kernel(name, blk):
    """Compiled ``name<T, I>`` for the dtypes of the CSR block ``blk``."""
    import cupy as cp
    T = "float" if blk.data.dtype == cp.float32 else "double"
    I = "int" if blk.indptr.dtype == cp.int32 else "long long"
    key = f"{name}<{T}, {I}>"
    if not _KERNELS:
        exprs = [f"{n}<{t}, {i}>" for n in ("row_gene_stats", "normalize_rows", "pr_stored", "group_rows")
                 for t in ("float", "double") for i in ("int", "long long")]
        _KERNELS["module"] = cp.RawModule(code=_KERNEL_SRC, options=("-std=c++14",),
                                          name_expressions=exprs)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = _KERNELS["module"].get_function(key)
    return k


def _NULL():
    import numpy
    return numpy.uint64(0)  # a null pointer argument


def _launch(kernel, n_rows, *args):
    threads = 128
    kernel(((n_rows + threads - 1) // threads,), (threads,), args)


def _gpu_csr(blk):
    """A canonical cupyx CSR block with int32 indices and float data."""
    import cupy as cp
    blk = blk.tocsr()
    if blk.data.dtype not in (cp.float32, cp.float64):
        blk = blk.astype(cp.float32)
    if blk.indices.dtype != cp.int32:
        blk.indices = blk.indices.astype(cp.int32)
    return blk


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
_FDS: dict = {}
_CHUNKED: dict = {}
_LOCAL = __import__("threading").local()
_STAGE_BYTES = 64 << 20


def _staging():
    """This thread's pinned host buffer for disk → GPU copies."""
    buf = getattr(_LOCAL, "stage", None)
    if buf is None:
        import cupyx
        buf = _LOCAL.stage = cupyx.empty_pinned((_STAGE_BYTES,), dtype=np.uint8)
    return buf


class _H5Source:
    """Location of a matrix in an h5ad file; picklable for dask workers.

    Contiguous, uncompressed datasets are read with ``os.preadv`` (no HDF5
    lock, so blocks load in parallel); anything else falls back to h5py."""

    def __init__(self, path, key):
        import h5py
        self.path = os.path.realpath(os.fspath(path))
        self.key = key
        st = os.stat(self.path)
        self.stamp = (st.st_mtime_ns, st.st_size)
        with h5py.File(self.path, "r") as f:
            g = f[key]
            self.sparse = isinstance(g, h5py.Group)
            names = ("data", "indices") if self.sparse else (None,)
            self.dtypes, self.offsets = {}, {}
            for k in names:
                ds = g[k] if k else g
                self.dtypes[k] = ds.dtype
                off = None
                if ds.chunks is None and ds.compression is None and not ds._filters:
                    off = ds.id.get_offset()
                self.offsets[k] = off
            if self.sparse:
                enc = g.attrs.get("encoding-type", b"")
                enc = enc.decode() if isinstance(enc, bytes) else str(enc)
                if enc != "csr_matrix":
                    raise NotImplementedError(
                        f"storage='disk' in GPU mode needs a CSR (or dense) {key!r}; found {enc!r}")
                self.shape = tuple(int(x) for x in g.attrs["shape"])
            else:
                self.shape = tuple(int(x) for x in g.shape)

    def __dask_tokenize__(self):
        return ("ov-h5", self.path, self.key, self.stamp)

    def read_device(self, k, a, b):
        """Elements ``[a, b)`` of dataset ``k`` as a cupy array, streamed
        through this thread's fixed pinned buffer (pinned memory stays
        bounded however large the blocks are)."""
        import cupy as cp
        dt = np.dtype(self.dtypes[k])
        off = self.offsets[k]
        if off is None or not hasattr(os, "preadv"):
            return cp.asarray(self.read(k, a, b))
        out = cp.empty(b - a, dtype=dt)
        stage = _staging()
        per = max(1, stage.nbytes // dt.itemsize)
        fd = self._fd()
        dst = out.view(cp.uint8)
        for e0 in range(a, b, per):
            e1 = min(b, e0 + per)
            nbytes = (e1 - e0) * dt.itemsize
            mv = memoryview(stage)[:nbytes]
            done, base = 0, off + e0 * dt.itemsize
            while done < nbytes:
                n = os.preadv(fd, [mv[done:]], base + done)
                if n <= 0:
                    raise OSError(f"short read from {self.path}")
                done += n
            o = (e0 - a) * dt.itemsize
            dst[o:o + nbytes].set(stage[:nbytes])
        return out

    def _fd(self):
        fd = _FDS.get(self.path)
        if fd is None:
            fd = _FDS[self.path] = os.open(self.path, os.O_RDONLY)
        return fd

    def read(self, k, a, b):
        """Elements ``[a, b)`` of the 1-D dataset ``k`` (sparse) as numpy."""
        dt = np.dtype(self.dtypes[k])
        off = self.offsets[k]
        if off is None or not hasattr(os, "preadv"):
            ds = self._chunked(k)
            if ds is not None:  # anndataoom-fast decodes chunks without the HDF5 lock
                return ds.read(a, b).astype(dt, copy=False)
            import h5py
            with h5py.File(self.path, "r") as f:
                return f[self.key][k][a:b]
        out = np.empty(b - a, dtype=dt)
        fd = self._fd()
        mv = memoryview(out).cast("B")
        done, total, base = 0, (b - a) * dt.itemsize, off + a * dt.itemsize
        while done < total:
            n = os.preadv(fd, [mv[done:]], base + done)
            if n <= 0:
                raise OSError(f"short read from {self.path}")
            done += n
        return out

    def _chunked(self, k):
        key = (self.path, self.key, k, self.stamp)
        if key not in _CHUNKED:
            try:
                import h5py
                from anndataoom_fast._h5 import dataset
                with h5py.File(self.path, "r") as f:
                    _CHUNKED[key] = dataset(self.path, f[self.key][k])
            except Exception:  # anndataoom-fast missing or layout unsupported
                _CHUNKED[key] = None
        return _CHUNKED[key]

    def read_dense(self, r0, r1):
        import h5py
        with h5py.File(self.path, "r") as f:
            return f[self.key][r0:r1]


def _read_block(src, indptr, device):
    """CSR rows whose (absolute) indptr slice is ``indptr``."""
    a, b = int(indptr[0]), int(indptr[-1])
    n_rows, n_cols = len(indptr) - 1, src.shape[1]
    ip = np.asarray(indptr, dtype=np.int64) - a
    if device == "gpu":
        import cupy as cp
        import cupyx.scipy.sparse as cps
        data = src.read_device("data", a, b)
        indices = src.read_device("indices", a, b)
        if data.dtype not in (cp.float32, cp.float64):
            data = data.astype(cp.float32)
        indices = cp.asarray(indices).astype(cp.int32, copy=False)
        ip = cp.asarray(ip.astype(np.int32) if b - a < 2**31 else ip)
        return cps.csr_matrix((data, indices, ip), shape=(n_rows, n_cols))
    import scipy.sparse as sp
    data, indices = src.read("data", a, b), src.read("indices", a, b)
    if data.dtype not in (np.float32, np.float64):
        data = data.astype(np.float32)
    return sp.csr_matrix((data, indices.astype(np.int32, copy=False), ip),
                         shape=(n_rows, n_cols))


def _read_dense_block(src, r0, r1, device):
    X = src.read_dense(r0, r1)
    if device == "gpu":
        import cupy as cp
        return cp.asarray(X)
    return X


def _empty_meta(sparse, device, dtype):
    if device == "gpu":
        import cupy as cp
        import cupyx.scipy.sparse as cps
        return cps.csr_matrix(cp.array((1.0,), dtype=dtype)) if sparse else cp.empty((0, 0), dtype)
    import scipy.sparse as sp
    return sp.csr_matrix((0, 0), dtype=dtype) if sparse else np.empty((0, 0), dtype)


def read_h5ad(path, *, chunk_size: int = 20_000, device: str | None = None):
    """Open an ``.h5ad`` with ``X`` as a Dask array of row blocks.

    Arguments:
        path: ``.h5ad`` file; ``X`` must be CSR or dense.
        chunk_size: Cells per block.
        device: ``'gpu'`` (blocks are cupyx CSR matrices; default when cupy is
            importable) or ``'cpu'`` (scipy CSR blocks).

    Returns:
        AnnData with ``obs``, ``var`` and ``uns`` loaded and a lazy ``X``.
        ``layers``, ``obsm``, ``obsp`` and ``raw`` are not read.
    """
    import anndata
    import h5py
    import dask
    import dask.array as da
    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem
    if device is None:
        try:
            import cupy  # noqa: F401
            device = "gpu"
        except ImportError:
            device = "cpu"
    if device not in ("gpu", "cpu"):
        raise ValueError("device must be 'gpu' or 'cpu'")
    src = _H5Source(path, "X")
    n, m = src.shape
    with h5py.File(src.path, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        uns = read_elem(f["uns"]) if "uns" in f else {}
        indptr = f["X"]["indptr"][:] if src.sparse else None
        skipped = [k for k in ("layers", "obsm", "obsp", "varm", "varp", "raw")
                   if k in f and len(f[k])]
    dtype = np.dtype(src.dtypes["data" if src.sparse else None])
    if dtype.kind != "f":
        dtype = np.dtype(np.float32)
    meta = _empty_meta(src.sparse, device, dtype)
    blocks = []
    for r0 in range(0, n, chunk_size):
        r1 = min(r0 + chunk_size, n)
        if src.sparse:
            t = dask.delayed(_read_block, pure=True)(src, indptr[r0:r1 + 1], device)
        else:
            t = dask.delayed(_read_dense_block, pure=True)(src, r0, r1, device)
        blocks.append(da.from_delayed(t, shape=(r1 - r0, m), dtype=dtype, meta=meta))
    X = da.concatenate(blocks, axis=0) if blocks else da.zeros((0, m), dtype=dtype)
    adata = anndata.AnnData(X=X, obs=obs, var=var, uns=uns)
    if skipped:
        print(f"storage='disk' (GPU) reads X, obs, var and uns; skipped: {', '.join(skipped)}")
    return adata


# ---------------------------------------------------------------------------
# per-gene statistics, cached by array name
# ---------------------------------------------------------------------------
_GENE_STATS: "OrderedDict[str, dict]" = OrderedDict()
_SCALED: "OrderedDict[str, tuple]" = OrderedDict()
_CACHE_SIZE = 16


def _remember(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)


def _gene_block(blk):
    """(sum, sum of squares, nonzeros) per gene of one block, float64."""
    xp = _xp(blk)
    G = blk.shape[1]
    if _issparse(blk) and xp is not np:
        import cupy as cp
        blk = _gpu_csr(blk)
        n = blk.shape[0]
        s, q, z = (cp.zeros(G) for _ in range(3))
        _launch(_kernel("row_gene_stats", blk), n, blk.indptr, blk.indices, blk.data, cp.int64(n),
                _NULL(), cp.int32(0), s, q, z, _NULL(), _NULL(), _NULL())
        return s, q, z
    if _issparse(blk):
        blk = blk.tocsr()
        d = blk.data.astype(xp.float64)
        idx = blk.indices
        s = xp.bincount(idx, weights=d, minlength=G)
        q = xp.bincount(idx, weights=d * d, minlength=G)
        z = xp.bincount(idx, weights=(d != 0).astype(xp.float64), minlength=G)
    else:
        d = blk.astype(xp.float64)
        s, q, z = d.sum(0), (d * d).sum(0), (d != 0).sum(0).astype(xp.float64)
    return s, q, z


def _add(parts):
    out = list(parts[0])
    for p in parts[1:]:
        for i, v in enumerate(p):
            out[i] = out[i] + v
    return out


def _stats_dict(n, s, q, z):
    return {"n": int(n), "sum": _to_numpy(s), "sumsq": _to_numpy(q), "nnz": _to_numpy(z)}


def _gene_tasks(X):
    import dask
    return [dask.delayed(_gene_block, pure=True)(b) for b in _row_blocks(X)]


def gene_stats(X):
    """Per-gene ``n``, ``sum``, ``sumsq`` and ``nnz`` of a Dask matrix (cached)."""
    hit = _GENE_STATS.get(X.name)
    if hit is not None:
        return hit
    st = _stats_dict(X.shape[0], *_add(_compute(*_gene_tasks(X))))
    _remember(_GENE_STATS, X.name, st)
    return st


def _carry_var_stats(old_name, new_X, mask):
    """Cached gene statistics survive a gene subset."""
    st = _GENE_STATS.get(old_name)
    if st is not None and is_dask(new_X):
        sub = {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in st.items()}
        _remember(_GENE_STATS, new_X.name, sub)


def subset_var(adata, mask):
    """``adata._inplace_subset_var`` keeping cached gene statistics."""
    mask = np.asarray(mask, dtype=bool)
    before = {k: getattr(v, "name", None) for k, v in _dask_arrays(adata)}
    adata._inplace_subset_var(mask)
    for k, v in _dask_arrays(adata):
        if before.get(k):
            _carry_var_stats(before[k], v, mask)


def _dask_arrays(adata):
    out = []
    if is_dask(adata.X):
        out.append(("X", adata.X))
    for k in adata.layers.keys():
        if is_dask(adata.layers[k]):
            out.append((f"layers/{k}", adata.layers[k]))
    return out


def mean_var(X, ddof=1):
    """Per-gene mean and variance (``ddof=1`` as scanpy) of a Dask matrix."""
    st = gene_stats(X)
    n = st["n"]
    mean = st["sum"] / n
    var = st["sumsq"] / n - mean * mean
    if ddof and n > ddof:
        var = var * (n / (n - ddof))
    return mean, np.maximum(var, 0)


# ---------------------------------------------------------------------------
# quality control
# ---------------------------------------------------------------------------
def _qc_block(blk, masks):
    """Per-cell totals, genes, gene-set totals and per-gene statistics."""
    xp = _xp(blk)
    sps = _sparse_mod(blk)
    G = blk.shape[1]
    if _issparse(blk) and xp is not np and masks.shape[1] <= 8:
        import cupy as cp
        blk = _gpu_csr(blk)
        n, k = blk.shape[0], masks.shape[1]
        bits = (masks.astype(cp.uint8) << cp.arange(k, dtype=cp.uint8)[None, :]).sum(1).astype(cp.uint8) \
            if k else cp.zeros(G, dtype=cp.uint8)
        s, q, z = (cp.zeros(G) for _ in range(3))
        rs, rz, rset = cp.zeros(n), cp.zeros(n), cp.zeros((n, k))
        _launch(_kernel("row_gene_stats", blk), n, blk.indptr, blk.indices, blk.data, cp.int64(n),
                bits, cp.int32(k), s, q, z, rs, rz, rset)
        return rs, rz, rset, s, q, z
    if _issparse(blk):
        blk = blk.tocsr()
        d = blk.data.astype(xp.float64)
        b64 = sps.csr_matrix((d, blk.indices, blk.indptr), shape=blk.shape)
        cell_tot = b64.sum(axis=1).ravel()
        nz = (d != 0).astype(xp.float64)
        cell_genes = sps.csr_matrix((nz, blk.indices, blk.indptr), shape=blk.shape).sum(axis=1).ravel()
        sets = b64 @ masks if masks.shape[1] else xp.zeros((blk.shape[0], 0))
        idx = blk.indices
        s = xp.bincount(idx, weights=d, minlength=G)
        q = xp.bincount(idx, weights=d * d, minlength=G)
        z = xp.bincount(idx, weights=nz, minlength=G)
    else:
        d = blk.astype(xp.float64)
        cell_tot = d.sum(1)
        cell_genes = (d != 0).sum(1).astype(xp.float64)
        sets = d @ masks
        s, q, z = d.sum(0), (d * d).sum(0), (d != 0).sum(0).astype(xp.float64)
    return xp.asarray(cell_tot).ravel(), xp.asarray(cell_genes).ravel(), xp.asarray(sets), s, q, z


def calculate_qc_metrics(adata, qc_vars=(), *, log1p=True):
    """Same ``obs``/``var`` columns as ``rapids_singlecell.pp.calculate_qc_metrics``,
    in one pass over a Dask ``X``."""
    import dask
    X = adata.X
    qc_vars = list(qc_vars)
    M = np.stack([np.asarray(adata.var[k], dtype=np.float64) for k in qc_vars], axis=1) \
        if qc_vars else np.zeros((X.shape[1], 0))
    if _on_gpu(_meta(X)):
        import cupy as cp
        M = cp.asarray(M)
    parts = _compute(*[dask.delayed(_qc_block)(b, M) for b in _row_blocks(X)])
    cell_tot = np.concatenate([_to_numpy(p[0]) for p in parts])
    cell_genes = np.concatenate([_to_numpy(p[1]) for p in parts])
    sets = np.concatenate([_to_numpy(p[2]) for p in parts], axis=0)
    s, q, z = _add([p[3:] for p in parts])
    st = _stats_dict(X.shape[0], s, q, z)
    _remember(_GENE_STATS, X.name, st)
    n = X.shape[0]
    var, obs = adata.var, adata.obs
    var["n_cells_by_counts"] = st["nnz"]
    var["total_counts"] = st["sum"]
    mean = st["sum"] / max(n, 1)
    var["mean_counts"] = mean
    var["pct_dropout_by_counts"] = (1 - st["nnz"] / max(n, 1)) * 100
    if log1p:
        var["log1p_total_counts"] = np.log1p(st["sum"])
        var["log1p_mean_counts"] = np.log1p(mean)
    obs["n_genes_by_counts"] = cell_genes
    obs["total_counts"] = cell_tot
    if log1p:
        obs["log1p_n_genes_by_counts"] = np.log1p(cell_genes)
        obs["log1p_total_counts"] = np.log1p(cell_tot)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, k in enumerate(qc_vars):
            obs[f"total_counts_{k}"] = sets[:, i]
            obs[f"pct_counts_{k}"] = sets[:, i] / cell_tot * 100
            if log1p:
                obs[f"log1p_total_counts_{k}"] = np.log1p(sets[:, i])


def filter_genes(adata, *, min_cells=None, max_cells=None):
    """Keep genes detected in ``[min_cells, max_cells]`` cells (one pass, or
    none when the statistics are cached); sets ``var['n_cells']``."""
    st = gene_stats(adata.X)
    n_cells = st["nnz"].astype(np.int64)
    keep = np.ones(adata.n_vars, dtype=bool)
    if min_cells is not None:
        keep &= n_cells >= min_cells
    if max_cells is not None:
        keep &= n_cells <= max_cells
    adata.var["n_cells"] = n_cells
    if not keep.all():
        subset_var(adata, keep)
    return keep


def identify_robust_genes(adata, percent_cells=0.05):
    """Dask version of :func:`omicverse.pp.identify_robust_genes`."""
    st = gene_stats(adata.X)
    adata.var["n_cells"] = st["nnz"].astype(np.int64)
    subset_var(adata, adata.var["n_cells"].values > 0)
    adata.var["percent_cells"] = adata.var["n_cells"] / adata.shape[0] * 100
    adata.var["robust"] = adata.var["percent_cells"] >= percent_cells
    adata.var["highly_variable_features"] = adata.var["robust"]


# ---------------------------------------------------------------------------
# lazy transforms
# ---------------------------------------------------------------------------
def _normalize_block(blk, target_sum):
    xp = _xp(blk)
    sps = _sparse_mod(blk)
    if _issparse(blk) and xp is not np:
        import cupy as cp
        blk = _gpu_csr(blk)
        out = cp.empty_like(blk.data)
        n = blk.shape[0]
        _launch(_kernel("normalize_rows", blk), n, blk.indptr, blk.data, cp.int64(n),
                cp.float64(target_sum), out)
        return sps.csr_matrix((out, blk.indices, blk.indptr), shape=blk.shape)
    if _issparse(blk):
        blk = blk.tocsr()
        d = blk.data
        rows = _row_ids(blk.indptr, d.shape[0])
        tot = xp.bincount(rows, weights=d.astype(xp.float64), minlength=blk.shape[0])
        f = xp.where(tot > 0, target_sum / xp.where(tot > 0, tot, 1), 1.0)
        return sps.csr_matrix(((d * f[rows]).astype(d.dtype), blk.indices, blk.indptr),
                              shape=blk.shape)
    tot = blk.sum(1, dtype=xp.float64)
    f = xp.where(tot > 0, target_sum / xp.where(tot > 0, tot, 1), 1.0)
    return (blk * f[:, None]).astype(blk.dtype)


def _log1p_block(blk):
    if _issparse(blk):
        return blk.log1p()
    return _xp(blk).log1p(blk)


def normalize_total(X, target_sum):
    """Lazily scale every cell to ``target_sum`` counts (new blocks; the
    input blocks are not modified)."""
    if target_sum is None:
        raise NotImplementedError("storage='disk' in GPU mode needs an explicit target_sum")
    return X.map_blocks(_normalize_block, float(target_sum), dtype=X.dtype, meta=_meta(X))


def log1p(X):
    return X.map_blocks(_log1p_block, dtype=X.dtype, meta=_meta(X))


def _scale_block(blk, mean, std, max_value, zero_center):
    xp = _xp(blk)
    if zero_center:
        D = blk.toarray() if _issparse(blk) else xp.array(blk, copy=True)
        D -= mean
        D /= std
        if max_value is not None:
            xp.clip(D, -max_value, max_value, out=D)
        return D
    sps = _sparse_mod(blk)
    blk = blk.tocsr()
    d = blk.data / std[blk.indices]
    if max_value is not None:
        d = xp.minimum(d, max_value)
    return sps.csr_matrix((d.astype(blk.dtype), blk.indices, blk.indptr), shape=blk.shape)


def scale(X, *, max_value=None, zero_center=True):
    """Lazy z-scores (scanpy/rapids-singlecell semantics: ``ddof=1`` std,
    zero std → 1, clipped to ``±max_value`` when centred)."""
    mean, var = mean_var(X)
    std = np.sqrt(var)
    std[std == 0] = 1
    dt = np.dtype(X.dtype)
    if _on_gpu(_meta(X)):
        import cupy as cp
        mean_d, std_d = cp.asarray(mean, dtype=dt), cp.asarray(std, dtype=dt)
    else:
        mean_d, std_d = mean.astype(dt), std.astype(dt)
    if zero_center:
        meta = _empty_meta(False, "gpu" if _on_gpu(_meta(X)) else "cpu", dt)
    else:
        meta = _meta(X)
    out = X.map_blocks(_scale_block, mean_d, std_d, max_value, zero_center, dtype=dt, meta=meta)
    _remember(_SCALED, out.name, (X, mean, std, max_value, zero_center))
    return out, mean, std


# ---------------------------------------------------------------------------
# Pearson-residual HVGs
# ---------------------------------------------------------------------------
_PR_KERNELS = {}


def _pr_kernels():
    if not _PR_KERNELS:
        import cupy as cp
        pre = """
        __device__ double pr0(double s, double g, double it, double c) {
            double mu = s * g;
            if (!(mu > 0)) return 0.0;
            double r = -mu * rsqrt(mu + mu * mu * it);
            return fmin(fmax(r, -c), c);
        }"""
        for name, expr in (("sum", "pr0(s, g, it, c)"), ("sq", "pr0(s, g, it, c) * pr0(s, g, it, c)")):
            _PR_KERNELS[name] = cp.ReductionKernel(
                "float64 s, float64 g, float64 it, float64 c", "float64 y",
                expr, "a + b", "y = a", "0", f"ov_pr0_{name}", preamble=pre)
    return _PR_KERNELS


def _pr_dense_sums(xp, s, g, inv_theta, clip):
    """Column sums of the zero-count residuals r0(s_i * g_j) and their squares."""
    if xp is not np:
        k = _pr_kernels()
        S, Gm = s[:, None], g[None, :]
        return (k["sum"](S, Gm, inv_theta, clip, axis=0),
                k["sq"](S, Gm, inv_theta, clip, axis=0))
    out1, out2 = np.zeros(g.shape[0]), np.zeros(g.shape[0])
    for j in range(0, g.shape[0], 1024):
        mu = s[:, None] * g[None, j:j + 1024]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(mu > 0, -mu / np.sqrt(mu + mu * mu * inv_theta), 0.0)
        r = np.clip(r, -clip, clip)
        out1[j:j + 1024] = r.sum(0)
        out2[j:j + 1024] = (r * r).sum(0)
    return out1, out2


def _pr_block(blk, codes, g_scaled, clips, inv_theta):
    """Per-batch sums of clipped Pearson residuals and their squares."""
    xp = _xp(blk)
    sps = _sparse_mod(blk)
    blk = blk.tocsr() if _issparse(blk) else sps.csr_matrix(blk)
    nb, G = g_scaled.shape
    S1, S2 = xp.zeros((nb, G)), xp.zeros((nb, G))
    codes = xp.asarray(codes)
    for b in range(nb):
        rows = xp.nonzero(codes == b)[0]
        if rows.shape[0] == 0:
            continue
        sub = blk[rows] if rows.shape[0] != blk.shape[0] else blk
        g = g_scaled[b]
        c = float(clips[b])
        if xp is not np:
            import cupy as cp
            sub = _gpu_csr(sub)
            m = sub.shape[0]
            s, a1, a2 = cp.empty(m), cp.zeros(G), cp.zeros(G)
            _launch(_kernel("pr_stored", sub), m, sub.indptr, sub.indices, sub.data, cp.int64(m),
                    g, cp.float64(inv_theta), cp.float64(c), s, a1, a2)
            d1, d2 = _pr_dense_sums(xp, s, g, inv_theta, c)
            S1[b], S2[b] = a1 + d1, a2 + d2
            continue
        d = sub.data.astype(xp.float64)
        r_id = _row_ids(sub.indptr, d.shape[0])
        s = xp.bincount(r_id, weights=d, minlength=sub.shape[0])
        g = g_scaled[b]
        c = float(clips[b])
        a1, a2 = _pr_dense_sums(xp, s, g, inv_theta, c)
        # stored values: replace the zero-count residual by the real one
        mu = s[r_id] * g[sub.indices]
        den = xp.sqrt(mu + mu * mu * inv_theta)
        ok = mu > 0
        den = xp.where(ok, den, 1.0)
        r = xp.where(ok, xp.clip((d - mu) / den, -c, c), 0.0)
        r0 = xp.where(ok, xp.clip(-mu / den, -c, c), 0.0)
        a1 = a1 + xp.bincount(sub.indices, weights=r - r0, minlength=G)
        a2 = a2 + xp.bincount(sub.indices, weights=r * r - r0 * r0, minlength=G)
        S1[b], S2[b] = a1, a2
    return S1, S2


def _batch_gene_sums_block(blk, codes, nb):
    xp = _xp(blk)
    sps = _sparse_mod(blk)
    codes = xp.asarray(codes)
    onehot = sps.csr_matrix((xp.ones(codes.shape[0]), (codes, xp.arange(codes.shape[0]))),
                            shape=(nb, codes.shape[0]))
    b = blk.tocsr() if _issparse(blk) else sps.csr_matrix(blk)
    b64 = sps.csr_matrix((b.data.astype(xp.float64), b.indices, b.indptr), shape=b.shape)
    return xp.asarray((onehot @ b64).todense())


def _block_codes(X, codes):
    """``codes`` split like the rows of ``X``."""
    out, r = [], 0
    for c in X.chunks[0]:
        out.append(codes[r:r + c])
        r += c
    return out


def highly_variable_genes_pearson(adata, *, layer="counts", n_top_genes=2000, batch_key=None,
                                  theta=100.0, clip=None, also_stats=()):
    """Pearson-residual HVGs (as ``rapids_singlecell`` / scanpy
    ``flavor='pearson_residuals'``) on a Dask matrix.

    The residual variance of each gene is accumulated over row blocks: the
    zero-count residual ``r0(mu_ij)`` is summed densely on the GPU and the
    stored counts replace it by the real residual.  ``also_stats`` are other
    Dask arrays (e.g. the log-normalised ``X``) whose per-gene statistics are
    gathered in the same pass, so a later ``scale`` does not read the matrix
    again.
    """
    import dask
    X = adata.layers[layer] if layer is not None else adata.X
    if theta <= 0:
        raise ValueError("Pearson residuals require theta > 0")
    n = X.shape[0]
    if batch_key is None:
        codes = np.zeros(n, dtype=np.int64)
        batches = np.array([0])
    else:
        cat = pd.Categorical(adata.obs[batch_key].values)
        batches = np.asarray(cat.categories)
        codes = cat.codes.astype(np.int64)
    nb = len(batches)
    n_b = np.bincount(codes, minlength=nb)
    gpu = _on_gpu(_meta(X))
    if gpu:
        import cupy as cp
        dev = cp.asarray
    else:
        dev = np.asarray
    blocks = _row_blocks(X)
    bcodes = _block_codes(X, codes)
    # gene sums per batch (cached statistics cover the single-batch case)
    if nb == 1:
        gsum = gene_stats(X)["sum"][None, :]
    else:
        parts = _compute(*[dask.delayed(_batch_gene_sums_block)(b, dev(c), nb)
                           for b, c in zip(blocks, bcodes)])
        gsum = _to_numpy(sum(parts[1:], parts[0]))
    tot = gsum.sum(axis=1)
    g_scaled = np.where(tot[:, None] > 0, gsum / np.where(tot > 0, tot, 1)[:, None], 0.0)
    clips = np.sqrt(n_b).astype(np.float64) if clip is None else np.full(nb, float(clip))
    if np.any(clips < 0):
        raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")
    g_d = dev(g_scaled)
    pr = [dask.delayed(_pr_block)(b, dev(c), g_d, clips, 1.0 / theta)
          for b, c in zip(blocks, bcodes)]
    extra = [(A, _gene_tasks(A)) for A in also_stats if is_dask(A) and A.name not in _GENE_STATS]
    flat = pr + [t for _, ts in extra for t in ts]
    res = _compute(*flat)
    S1, S2 = _add(res[:len(pr)])
    S1, S2 = _to_numpy(S1), _to_numpy(S2)
    pos = len(pr)
    for A, ts in extra:
        st = _stats_dict(A.shape[0], *_add(res[pos:pos + len(ts)]))
        _remember(_GENE_STATS, A.name, st)
        pos += len(ts)
    nbf = np.maximum(n_b, 1)[:, None].astype(np.float64)
    mean_r = S1 / nbf
    resid_var = np.maximum(S2 / nbf - mean_r * mean_r, 0)
    resid_var[gsum == 0] = 0  # genes absent from a batch (as rapids-singlecell)
    # ranking exactly as rapids_singlecell / scanpy
    ranks = np.argsort(np.argsort(-resid_var, axis=1, kind="stable"), axis=1).astype(np.float32)
    nbatches = np.sum(ranks < n_top_genes, axis=0)
    ranks[ranks >= n_top_genes] = np.nan
    median_rank = np.ma.median(np.ma.masked_invalid(ranks), axis=0).filled(np.nan)
    means, variances = mean_var(X)
    df = pd.DataFrame({
        "means": means, "variances": variances,
        "residual_variances": resid_var.mean(axis=0),
        "highly_variable_rank": median_rank,
        "highly_variable_nbatches": nbatches.astype(np.int64),
        "highly_variable_intersection": nbatches == nb,
    }, index=adata.var_names)
    order = df.sort_values(["highly_variable_nbatches", "highly_variable_rank"],
                           ascending=[False, True], na_position="last").index
    df["highly_variable"] = False
    df.loc[order[:n_top_genes], "highly_variable"] = True
    adata.uns["hvg"] = {"flavor": "pearson_residuals", "computed_on": layer or "adata.X"}
    for col in ("means", "variances", "residual_variances", "highly_variable_rank"):
        adata.var[col] = df[col].values
    if batch_key is not None:
        adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"].values
        adata.var["highly_variable_intersection"] = df["highly_variable_intersection"].values
    adata.var["highly_variable"] = df["highly_variable"].values


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def _dense_block(blk):
    return blk.toarray() if _issparse(blk) else blk


def _gram_block(blk):
    xp = _xp(blk)
    D = _dense_block(blk).astype(xp.float64)
    return D.T @ D, D.sum(0)


def _project_block(blk, mean, comps_t):
    xp = _xp(blk)
    D = _dense_block(blk)
    return ((D - mean) @ comps_t).astype(xp.float32)


def pca(adata, *, layer=None, n_comps=50, mask_var="highly_variable", key_added=None):
    """Covariance-eigendecomposition PCA of a Dask matrix (the HVG columns of
    the scaled data are kept on the device between the two passes).

    Writes ``obsm['X_pca']``, ``varm['PCs']`` and ``uns['pca']`` like
    ``rapids_singlecell.pp.pca``; ``variance`` uses ``n - 1`` and component
    signs follow scikit-learn (largest loading positive)."""
    import dask
    L = adata.layers[layer] if layer is not None else adata.X
    if isinstance(mask_var, str):
        mask = np.asarray(adata.var[mask_var], dtype=bool) if mask_var in adata.var.columns \
            else np.ones(adata.n_vars, dtype=bool)
    elif mask_var is None:
        mask = np.ones(adata.n_vars, dtype=bool)
    else:
        mask = np.asarray(mask_var, dtype=bool)
    src = _SCALED.get(L.name)
    if src is not None:
        X0, mean0, std0, maxv, zc = src
        dt = np.dtype(L.dtype)
        if _on_gpu(_meta(X0)):
            import cupy as cp
            m_d, s_d = cp.asarray(mean0[mask], dtype=dt), cp.asarray(std0[mask], dtype=dt)
        else:
            m_d, s_d = mean0[mask].astype(dt), std0[mask].astype(dt)
        Z = X0[:, mask].map_blocks(_scale_block, m_d, s_d, maxv, zc, dtype=dt, meta=_meta(L))
    else:
        Z = L[:, mask]
    Z = Z.map_blocks(_dense_block, dtype=Z.dtype,
                     meta=_empty_meta(False, "gpu" if _on_gpu(_meta(Z)) else "cpu", Z.dtype))
    (Z,) = _persist(Z)
    n = Z.shape[0]
    blocks = _row_blocks(Z)
    G, s = _add(_compute(*[dask.delayed(_gram_block, pure=True)(b) for b in blocks]))
    xp = _xp(G)
    mean = s / n
    C = G / n - xp.outer(mean, mean)
    ev, V = xp.linalg.eigh(C)
    ev, V = ev[::-1], V[:, ::-1]
    k = int(min(n_comps, V.shape[1]))
    comps = V[:, :k].T.copy()
    flip = xp.sign(comps[xp.arange(k), xp.argmax(xp.abs(comps), axis=1)])
    flip[flip == 0] = 1
    comps *= flip[:, None]
    dt = np.dtype(Z.dtype)
    mean_t, comps_t = mean.astype(dt), comps.T.astype(dt)
    parts = _compute(*[dask.delayed(_project_block)(b, mean_t, comps_t) for b in blocks])
    X_pca = np.concatenate([_to_numpy(p) for p in parts], axis=0)
    del Z
    ev = _to_numpy(ev)
    ev_all = np.maximum(ev, 0)
    variance = ev_all[:k] * (n / (n - 1)) if n > 1 else ev_all[:k]
    ratio = ev_all[:k] / ev_all.sum() if ev_all.sum() > 0 else np.zeros(k)
    obsm_key = key_added or "X_pca"
    varm_key = key_added or "PCs"
    uns_key = key_added or "pca"
    adata.obsm[obsm_key] = X_pca
    PCs = np.zeros((adata.n_vars, k))
    PCs[mask] = _to_numpy(comps).T
    adata.varm[varm_key] = PCs
    adata.uns[uns_key] = {
        "params": {"zero_center": True, "mask_var": mask_var if isinstance(mask_var, str) else None,
                   "use_highly_variable": isinstance(mask_var, str) and mask_var == "highly_variable"},
        "variance": variance, "variance_ratio": ratio,
    }
    return X_pca


# ---------------------------------------------------------------------------
# multi-GPU
# ---------------------------------------------------------------------------
def start_cluster(devices=None, *, managed_memory=True, pool_allocator=True, **kwargs):
    """Start a ``dask_cuda.LocalCUDACluster`` (one worker per GPU) and a client;
    later ``dask.compute`` calls run on it."""
    try:
        from dask_cuda import LocalCUDACluster
        from distributed import Client
    except ImportError as exc:
        raise ImportError("multi-GPU mode needs dask-cuda and distributed "
                          "(both ship with RAPIDS)") from exc
    if devices is not None and not isinstance(devices, (list, tuple)):
        devices = [devices]
    kw = dict(rmm_managed_memory=managed_memory)
    if pool_allocator and not managed_memory:
        kw["rmm_pool_size"] = kwargs.pop("rmm_pool_size", None) or "0.8"
    if devices is not None:
        kw["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
    kw.update(kwargs)
    cluster = LocalCUDACluster(**kw)
    return Client(cluster)
