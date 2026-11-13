"""Minibatch sources for the ``ov.vi`` models.

Every source exposes the same thing: an iterator of minibatches whose tensors
already live on the training device (dense counts ``x``, the coordinates of
its non-zeros, the cell indices and the per-cell covariates).  The source is
chosen from the input:

``device``  in-memory AnnData whose CSR fits on the device: ``indptr`` /
            ``indices`` / ``data`` stay on the device and every minibatch is a
            row gather there (no DataLoader, no per-batch host work).
``host``    in-memory AnnData that does not fit: host CSR row blocks copied
            through pinned staging buffers by a background thread.
``h5``      **AnnDataOOM** (``ov.read(path, storage='disk')`` in CPU / mixed
            mode).  The lazy view chain (QC row subset, gene subset, layer) is
            resolved to the on-disk CSR with ``anndataoom_fast._view.resolve``;
            row blocks are read with anndataoom-fast's threaded ``pread``
            reader (GIL released, no HDF5 lock), the training genes are
            selected at read time on the device.
``dask``    Dask-backed AnnData (``ov.read(path, storage='disk')`` in
            ``ov.settings.gpu_init()`` mode): the dask row chunks are the
            blocks; cupy blocks reach torch zero-copy through DLPack.
``oom``     AnnDataOOM whose view chain cannot be resolved: anndataoom's own
            row reads (slower, same results).

Shuffling
---------
``random``  a global permutation of the training cells (resident sources only;
            what scvi-tools does).
``block``   each epoch permutes the order of the row blocks, fills a shuffle
            buffer with ``buffer_blocks`` blocks, permutes the cells inside the
            buffer and emits minibatches (a minibatch can straddle two buffers),
            while a background thread reads the next buffer.  Host / device
            memory is O(buffer), independent of the number of cells.
``none``    blocks and cells in order (inference).

Which cells / blocks a minibatch contains depends only on the block layout,
the seed and the epoch, never on the source: AnnData, AnnDataOOM and Dask
inputs with the same block layout give identical minibatches.
"""
from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass, field

import numpy as np
import torch

DEFAULT_BLOCK_SIZE = 2048
DEFAULT_BUFFER_BLOCKS = 32
_RESIDENT_FRACTION = 0.5


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def resolve_device(device="auto") -> torch.device:
    """``'auto'`` -> CUDA if available; CUDA devices always carry an index."""
    if isinstance(device, int):
        device = f"cuda:{device}"
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "gpu":
        device = "cuda"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _read_threads() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(int(os.environ.get("OV_VI_READ_THREADS", 8)), n))


def is_oom(adata) -> bool:
    return type(adata).__name__ == "AnnDataOOM" or type(adata).__module__.startswith("anndataoom")


def is_dask_array(X) -> bool:
    """Dask arrays, including anndata's ``DaskArrayView``."""
    return any(c.__module__.split(".")[0] == "dask" for c in type(X).__mro__)


def gene_selection(var, genes) -> np.ndarray | None:
    """Indices of the training genes among ``var`` (None = all)."""
    n = len(var)
    if genes is None:
        return None
    if isinstance(genes, str):
        if genes not in var.columns:
            raise KeyError(f"genes={genes!r} is not a column of adata.var")
        genes = np.asarray(var[genes].values, dtype=bool)
    genes = np.asarray(genes)
    if genes.dtype == bool:
        if genes.shape[0] != n:
            raise ValueError("boolean genes mask has the wrong length")
        sel = np.flatnonzero(genes)
    elif genes.dtype.kind in "iu":
        sel = genes.astype(np.int64)
    else:
        idx = var.index.get_indexer(genes)
        if (idx < 0).any():
            missing = list(np.asarray(genes)[idx < 0][:5])
            raise KeyError(f"genes not found in adata.var_names: {missing}")
        sel = idx.astype(np.int64)
    if sel.size == n and np.array_equal(sel, np.arange(n)):
        return None
    return sel


# ---------------------------------------------------------------------------
# block plan (shared by all sources)
# ---------------------------------------------------------------------------
def make_blocks(n_obs: int, block_size: int):
    starts = np.arange(0, n_obs, block_size, dtype=np.int64)
    ends = np.minimum(starts + block_size, n_obs)
    return starts, ends


def split_train_val(n_obs: int, train_size: float | None = None,
                    validation_size: float | None = None, batch_size: int = 128,
                    seed: int = 0, shuffle: bool = True):
    """scvi-tools' ``DataSplitter`` split (``train_size=None`` means 0.9 and moves a
    tiny last batch into validation)."""
    from math import ceil, floor
    train_size_is_none = train_size is None
    train_size = 0.9 if train_size is None else float(train_size)
    if not 0.0 < train_size <= 1.0:
        raise ValueError("train_size must be in (0, 1]")
    n_train = ceil(train_size * n_obs)
    n_val = n_obs - n_train if validation_size is None else floor(n_obs * validation_size)
    if batch_size:
        r = n_train % batch_size
        if 0 < r < 3 and train_size_is_none:
            n_train -= r
            if n_val > 0:
                n_val += r
    idx = np.arange(n_obs)
    if shuffle:
        idx = np.random.RandomState(seed=seed).permutation(idx)
    val = np.sort(idx[:n_val])
    train = np.sort(idx[n_val:n_val + n_train])
    return train, val


@dataclass
class Load:
    """One shuffle buffer: block ids (sorted) and the cells to emit, in order."""
    blocks: np.ndarray
    rows: np.ndarray


def plan_loads(starts, ends, rows_mask, shuffle, rng, buffer_blocks, rank=0, world=1):
    """Shuffle buffers of one pass (see module docstring)."""
    nb = len(starts)
    order = np.arange(nb) if shuffle == "none" else rng.permutation(nb)
    if world > 1:
        order = order[rank::world]
    k = max(1, int(buffer_blocks))
    for i in range(0, len(order), k):
        g = np.sort(order[i:i + k])
        rows = np.concatenate([np.arange(starts[j], ends[j]) for j in g]) if len(g) else \
            np.zeros(0, np.int64)
        if rows_mask is not None:
            rows = rows[rows_mask[rows]]
        if shuffle != "none" and rows.size:
            rows = rng.permutation(rows)
        if rows.size:
            yield Load(g, rows)


def rows_per_rank(starts, ends, rows_mask, world, perm):
    cs = np.concatenate([[0], np.cumsum(rows_mask.astype(np.int64))])
    per_block = cs[ends] - cs[starts]
    return [int(per_block[perm[r::world]].sum()) for r in range(world)]


# ---------------------------------------------------------------------------
# device buffers and row gather
# ---------------------------------------------------------------------------
@dataclass
class Buffer:
    """A CSR row block resident on the training device.

    Rows are the concatenation of the blocks ``view_starts[j]:view_ends[j]``
    (sorted); ``lens_host`` mirrors the row lengths on the host so minibatch
    nnz totals are known without a device sync."""
    indptr: torch.Tensor          # int64 (m,) row starts (no trailing entry needed)
    lens: torch.Tensor            # int64 (m,)
    indices: torch.Tensor         # int32 / int64
    data: torch.Tensor            # float32
    lens_host: np.ndarray         # int64 (m,)
    view_starts: np.ndarray       # int64 per block
    offsets: np.ndarray           # int64 per block (row offset inside the buffer)
    event: object = None          # CUDA event: buffer ready
    extra: dict = field(default_factory=dict)

    def local(self, view_rows: np.ndarray) -> np.ndarray:
        j = np.searchsorted(self.view_starts, view_rows, side="right") - 1
        return view_rows - self.view_starts[j] + self.offsets[j]

    def tensors(self):
        return (self.indptr, self.lens, self.indices, self.data)


def gather_rows(buf: Buffer, local: torch.Tensor, nnz: int, n_cols: int):
    """Dense rows ``local`` of ``buf`` plus the coordinates of their non-zeros."""
    dev = buf.data.device
    B = int(local.shape[0])
    x = torch.zeros(B, n_cols, device=dev, dtype=torch.float32)
    if nnz == 0:
        e = torch.zeros(0, dtype=torch.long, device=dev)
        return x, e, e, torch.zeros(0, device=dev)
    starts = buf.indptr[local]
    lens = buf.lens[local]
    row = torch.repeat_interleave(torch.arange(B, device=dev), lens, output_size=nnz)
    excl = torch.cumsum(lens, 0) - lens
    off = torch.arange(nnz, device=dev) + (starts - excl)[row]
    col = buf.indices[off].long()
    val = buf.data[off]
    x.view(-1).index_copy_(0, row * n_cols + col, val)
    return x, row, col, val


def _csr_tensors(indptr, indices, data, device, nonzero=True):
    """Host CSR arrays -> Buffer on ``device`` (row lengths mirrored on host)."""
    indptr = np.asarray(indptr, dtype=np.int64)
    lens_host = np.diff(indptr)
    data = np.asarray(data)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    if indices.dtype not in (np.int32, np.int64):
        indices = indices.astype(np.int64)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    return Buffer(t(indptr[:-1]), t(lens_host), t(indices), t(data), lens_host,
                  np.zeros(1, np.int64), np.zeros(1, np.int64))


def compact_rows(indptr_host, indices, data, keep_rows_host, col_map, device):
    """Select rows (``keep_rows_host``, bool on host) and remap columns with
    ``col_map`` (int32 tensor on the device, -1 = drop) of a raw CSR block whose
    ``indices`` / ``data`` already live on the device.  Explicit zeros are
    dropped.  Returns (indptr, lens, indices, data, lens_host) of the kept rows.
    Temporaries are 1-4 bytes per stored value (no per-value row index)."""
    indptr_host = np.asarray(indptr_host, dtype=np.int64)
    indptr_host = indptr_host - indptr_host[0]
    nnz = int(indptr_host[-1])
    lens_raw = np.diff(indptr_host)
    keep = None if keep_rows_host is None or keep_rows_host.all() else \
        np.ascontiguousarray(keep_rows_host, dtype=bool)
    lens_kept = lens_raw if keep is None else lens_raw[keep]

    def out(ind, dat, lh):
        lh = np.ascontiguousarray(lh, dtype=np.int64)
        lens = torch.from_numpy(lh).to(device)
        if ind.dtype != torch.int32:
            ind = ind.to(torch.int32)
        return torch.cumsum(lens, 0) - lens, lens, ind, dat, lh

    if nnz == 0:
        return out(indices[:0], data[:0], lens_kept)
    mask = data != 0
    if col_map is not None:
        newcol = col_map.index_select(0, indices if indices.dtype in (torch.int32, torch.int64)
                                      else indices.long())
        mask &= newcol >= 0
        indices = newcol
    if keep is not None:
        mask &= torch.repeat_interleave(torch.from_numpy(keep).to(device),
                                        torch.from_numpy(lens_raw).to(device), output_size=nnz)
    if bool(mask.all()):
        return out(indices, data, lens_kept)
    cs = torch.cumsum(mask, 0, dtype=torch.int32 if nnz < 2**31 - 1 else torch.int64)
    cs = torch.cat([cs.new_zeros(1), cs])
    ip = torch.from_numpy(indptr_host).to(device)
    lens_all = (cs[ip[1:]] - cs[ip[:-1]]).cpu().numpy()
    lh = lens_all if keep is None else lens_all[keep]
    return out(indices[mask], data[mask], lh)


class _Stage:
    """Pinned host staging buffer for host -> device copies.  The copy out of
    it is asynchronous, so ``wait()`` before refilling (as ``pp/_dask.py``'s
    ``_copy_from_stage``)."""

    def __init__(self, pinned: bool):
        self.pinned = pinned
        self.data = np.zeros(0, np.float32)
        self.indices = np.zeros(0, np.int32)
        self._td = self._ti = None
        self.event = None

    def ensure(self, n: int):
        if self.data.shape[0] >= n:
            return
        n = int(n * 1.25) + 1024 if self.pinned else int(n)
        if self.pinned:
            self._td = torch.empty(n, dtype=torch.float32, pin_memory=True)
            self._ti = torch.empty(n, dtype=torch.int32, pin_memory=True)
            self.data, self.indices = self._td.numpy(), self._ti.numpy()
        else:
            self.data = np.empty(n, np.float32)
            self.indices = np.empty(n, np.int32)

    def wait(self):
        if self.event is not None:
            self.event.synchronize()
            self.event = None

    def to_device(self, nnz: int, device, stream=None):
        if device.type == "cuda":
            td = torch.from_numpy(self.data[:nnz]) if self._td is None else self._td[:nnz]
            ti = torch.from_numpy(self.indices[:nnz]) if self._ti is None else self._ti[:nnz]
            d = td.to(device, non_blocking=self.pinned)
            i = ti.to(device, non_blocking=self.pinned)
            ev = torch.cuda.Event()
            ev.record(stream)
            self.event = ev
            return i, d
        # CPU training: hand the arrays over (no copy); the next load allocates anew
        i, d = torch.from_numpy(self.indices[:nnz]), torch.from_numpy(self.data[:nnz])
        self.data = np.zeros(0, np.float32)
        self.indices = np.zeros(0, np.int32)
        return i, d


# ---------------------------------------------------------------------------
# readers: rows [r0, r1) of an original CSR matrix into staging arrays
# ---------------------------------------------------------------------------
class CSRFile:
    """Picklable description of an on-disk CSR matrix for the threaded reader.

    Built once (h5py opened once, by rank 0); workers rebuild the
    ``anndataoom_fast._core.Dataset`` objects from the chunk tables without
    touching libhdf5, so data reads take no HDF5 lock."""

    def __init__(self, path, key, shape, indptr, data_desc, indices_desc):
        self.path, self.key = path, key
        self.shape = tuple(int(s) for s in shape)
        self.indptr = indptr
        self.data_desc, self.indices_desc = data_desc, indices_desc
        self._ds = None

    @staticmethod
    def _describe(path, ds):
        from anndataoom_fast._h5 import dataset as _dataset  # validates the layout
        _dataset(path, ds)  # raises Unsupported when the C++ reader cannot read it
        from anndataoom_fast._h5 import _dtype_code
        import math
        code = _dtype_code(ds.dtype)
        n = int(ds.shape[0])
        if ds.chunks is None:
            off = ds.id.get_offset()
            return ("contiguous", path, code, n, int(off or 0))
        dcpl = ds.id.get_create_plist()
        filters = [int(dcpl.get_filter(i)[0]) for i in range(dcpl.get_nfilters())]
        ce = int(ds.chunks[0])
        n_chunks = max(1, math.ceil(n / ce)) if n else 0
        addr = np.full(n_chunks, -1, dtype=np.int64)
        nbytes = np.zeros(n_chunks, dtype=np.int64)
        mask = np.zeros(n_chunks, dtype=np.uint32)

        def visit(info):
            c = int(info.chunk_offset[0]) // ce
            addr[c] = info.byte_offset
            nbytes[c] = info.size
            mask[c] = info.filter_mask
        try:
            ds.id.chunk_iter(visit)
        except (AttributeError, NotImplementedError):
            for i in range(ds.id.get_num_chunks()):
                visit(ds.id.get_chunk_info(i))
        return ("chunked", path, code, n, ce, addr, nbytes, mask, filters)

    @classmethod
    def open(cls, path, key="X"):
        import h5py
        path = os.path.realpath(os.fspath(path))
        with h5py.File(path, "r") as f:
            g = f[key]
            if not isinstance(g, h5py.Group):
                raise NotImplementedError(f"{key!r} is dense; ov.vi streams CSR matrices")
            enc = g.attrs.get("encoding-type", b"")
            enc = enc.decode() if isinstance(enc, bytes) else str(enc)
            if enc != "csr_matrix":
                raise NotImplementedError(f"{key!r} has encoding {enc!r}; need csr_matrix")
            shape = tuple(int(x) for x in g.attrs["shape"])
            indptr = np.ascontiguousarray(g["indptr"][:], dtype=np.int64)
            dd = cls._describe(path, g["data"])
            di = cls._describe(path, g["indices"])
        return cls(path, key, shape, indptr, dd, di)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_ds"] = None
        return s

    def _datasets(self):
        if self._ds is None:
            from anndataoom_fast import _core
            def mk(d):
                if d[0] == "contiguous":
                    return _core.Dataset(*d[1:])
                return _core.Dataset(*d[1:])
            self._ds = (mk(self.data_desc), mk(self.indices_desc))
        return self._ds

    def read_into(self, e0, e1, out_data, out_indices, n_threads):
        dd, di = self._datasets()
        dd.read_into(int(e0), int(e1), out_data, n_threads)
        di.read_into(int(e0), int(e1), out_indices, n_threads)


class MemoryCSR:
    """Host scipy CSR exposed through the same ``read_into`` interface."""

    def __init__(self, X):
        self.indptr = np.asarray(X.indptr, dtype=np.int64)
        self.data = X.data
        self.indices = X.indices
        self.shape = X.shape

    def read_into(self, e0, e1, out_data, out_indices, n_threads):
        out_data[:] = self.data[e0:e1]
        out_indices[:] = self.indices[e0:e1]


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
class DataSource:
    """Common minibatch iterator; subclasses provide ``_load(blocks) -> Buffer``."""

    kind = "base"
    resident = False

    def __init__(self, n_obs, n_vars, device, obs: dict, block_size=None):
        self.n_obs, self.n_vars = int(n_obs), int(n_vars)
        self.device = device
        self.obs = {k: v.to(device) for k, v in obs.items()}
        self.starts, self.ends = make_blocks(self.n_obs, block_size or DEFAULT_BLOCK_SIZE)
        self._stream = None

    # -- to override -------------------------------------------------------
    def _load(self, blocks: np.ndarray) -> Buffer:
        raise NotImplementedError

    def close(self):
        pass

    # -- iteration ---------------------------------------------------------
    def _prefetch(self, loads):
        """Yield (load, buffer) with the next buffer read by a background thread."""
        if self.resident:
            for ld in loads:
                yield ld, self._load(ld.blocks)
            return
        q: queue.Queue = queue.Queue(maxsize=1)
        stop = threading.Event()

        def work():
            try:
                if self.device.type == "cuda":
                    torch.cuda.set_device(self.device)
                    if self._stream is None:
                        self._stream = torch.cuda.Stream(self.device)
                    with torch.cuda.stream(self._stream):
                        for ld in loads:
                            if stop.is_set():
                                break
                            q.put((ld, self._load(ld.blocks)))
                else:
                    for ld in loads:
                        if stop.is_set():
                            break
                        q.put((ld, self._load(ld.blocks)))
                q.put(None)
            except BaseException as e:  # propagate to the consumer
                q.put(e)

        t = threading.Thread(target=work, daemon=True, name="ov-vi-prefetch")
        t.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                ld, buf = item
                if buf.event is not None:
                    cur = torch.cuda.current_stream(self.device)
                    cur.wait_event(buf.event)
                    for tns in buf.tensors():
                        tns.record_stream(cur)
                yield ld, buf
        finally:
            stop.set()
            while t.is_alive():
                try:
                    q.get(timeout=0.05)
                except queue.Empty:
                    pass
            t.join()

    def iterate(self, rows=None, *, batch_size=128, shuffle="none", seed=0, epoch=0,
                buffer_blocks=None, rank=0, world=1, n_batches=None, drop_last=False):
        """Minibatches (dicts of device tensors) over the cells ``rows``.

        Arguments:
            rows: Sorted cell indices to include (None = all).
            shuffle: ``'random'``, ``'block'`` or ``'none'``.
            seed, epoch: The shuffle is ``RandomState(seed + epoch)``.
            buffer_blocks: Blocks per shuffle buffer.
            rank, world: Shard blocks ``rank::world`` (distributed training).
            n_batches: Stop after this many minibatches (equal counts per rank).
            drop_last: Drop a final incomplete minibatch.
        """
        mask = None
        if rows is not None:
            mask = np.zeros(self.n_obs, dtype=bool)
            mask[np.asarray(rows, dtype=np.int64)] = True
        rng = np.random.RandomState((int(seed) * 1_000_003 + int(epoch)) % (2**32))
        if shuffle == "random":
            if not self.resident:
                raise ValueError("shuffle='random' needs a device-resident source; use 'block'")
            allrows = np.flatnonzero(mask) if mask is not None else np.arange(self.n_obs)
            perm = rng.permutation(allrows)
            if world > 1:
                perm = perm[rank::world]
            loads = [Load(np.arange(len(self.starts)), perm)]
        else:
            loads = plan_loads(self.starts, self.ends, mask, shuffle, rng,
                               buffer_blocks or (len(self.starts) if self.resident
                                                 else DEFAULT_BUFFER_BLOCKS),
                               rank, world)
        yield from self._batches(self._prefetch(loads), batch_size, n_batches, drop_last)

    def _batches(self, stream, batch_size, n_batches, drop_last):
        dev = self.device
        emitted = 0
        carry = []  # pieces of a minibatch straddling two buffers
        carry_n = 0
        for ld, buf in stream:
            local = buf.local(ld.rows)
            lens = buf.lens_host[local]
            local_t = torch.from_numpy(local).to(dev, non_blocking=True)
            rows_t = torch.from_numpy(ld.rows.astype(np.int64)).to(dev, non_blocking=True)
            pos = 0
            n = len(ld.rows)
            while pos < n:
                take = min(batch_size - carry_n, n - pos)
                nnz = int(lens[pos:pos + take].sum())
                piece = gather_rows(buf, local_t[pos:pos + take], nnz, self.n_vars) + \
                    (rows_t[pos:pos + take],)
                pos += take
                carry.append(piece)
                carry_n += take
                if carry_n == batch_size:
                    yield self._assemble(carry)
                    emitted += 1
                    carry, carry_n = [], 0
                    if n_batches is not None and emitted >= n_batches:
                        return
        if carry and not drop_last and (n_batches is None or emitted < n_batches):
            yield self._assemble(carry)

    def _assemble(self, pieces):
        if len(pieces) == 1:
            x, row, col, val, idx = pieces[0]
        else:
            xs, rs, cs, vs, ids = [], [], [], [], []
            off = 0
            for x, row, col, val, idx in pieces:
                xs.append(x); rs.append(row + off); cs.append(col); vs.append(val); ids.append(idx)
                off += x.shape[0]
            x, row, col, val, idx = (torch.cat(xs), torch.cat(rs), torch.cat(cs),
                                     torch.cat(vs), torch.cat(ids))
        out = {"x": x, "row": row, "col": col, "val": val, "idx": idx,
               "flat": row * self.n_vars + col}
        for k, v in self.obs.items():
            out[k] = v[idx]
        return out

    def n_rows_per_rank(self, rows, world, seed, epoch):
        """Cells per rank for an epoch's block permutation (for equal batch counts)."""
        mask = np.zeros(self.n_obs, dtype=bool)
        mask[np.asarray(rows, dtype=np.int64)] = True
        rng = np.random.RandomState((int(seed) * 1_000_003 + int(epoch)) % (2**32))
        perm = rng.permutation(len(self.starts))
        return rows_per_rank(self.starts, self.ends, mask, world, perm)

    def first_rows(self, n=64):
        """A few dense rows (host numpy) for input validation."""
        ld = Load(np.array([0]), np.arange(min(n, self.ends[0])))
        for _, buf in self._prefetch([ld]):
            x = gather_rows(buf, torch.from_numpy(buf.local(ld.rows)).to(self.device),
                            int(buf.lens_host[buf.local(ld.rows)].sum()), self.n_vars)[0]
            return x.cpu().numpy()
        return np.zeros((0, self.n_vars), np.float32)


class DeviceCSRSource(DataSource):
    """Whole CSR resident on the device; minibatches are row gathers."""

    kind = "device"
    resident = True

    def __init__(self, X, device, obs, block_size=None):
        import scipy.sparse as sp
        X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
        super().__init__(X.shape[0], X.shape[1], device, obs, block_size)
        if X.nnz and (X.data == 0).any():
            X = X.copy()
            X.eliminate_zeros()
        if not X.has_sorted_indices:
            X = X.copy()
            X.sort_indices()
        self._buf = _csr_tensors(X.indptr, X.indices, X.data, device)

    def _load(self, blocks):
        return self._buf


class _StreamedSource(DataSource):
    """Row blocks read on demand into pinned staging, copied to the device and
    compacted there (row filter, gene selection, explicit zeros)."""

    def __init__(self, reader, view_rows, col_map, n_vars, device, obs, block_size=None,
                 n_threads=None):
        n_obs = reader.shape[0] if view_rows is None else len(view_rows)
        super().__init__(n_obs, n_vars, device, obs, block_size)
        self.reader = reader
        self.view_rows = view_rows   # sorted original rows (None = all)
        self.col_map_host = col_map  # int32 original col -> training col / -1 (None = identity)
        self._col_map = None
        self.n_threads = n_threads or _read_threads()
        pinned = device.type == "cuda"
        self._stages = [_Stage(pinned), _Stage(pinned)]
        self._turn = 0

    def _orig_range(self, j):
        s, e = int(self.starts[j]), int(self.ends[j])
        if self.view_rows is None:
            return s, e, None
        r0, r1 = int(self.view_rows[s]), int(self.view_rows[e - 1]) + 1
        keep = np.zeros(r1 - r0, dtype=bool)
        keep[self.view_rows[s:e] - r0] = True
        return r0, r1, (None if keep.all() else keep)

    def _load(self, blocks):
        dev = self.device
        ip = self.reader.indptr
        ranges = [self._orig_range(j) for j in blocks]
        total = sum(int(ip[r1] - ip[r0]) for r0, r1, _ in ranges)
        stage = self._stages[self._turn]
        self._turn ^= 1
        stage.wait()  # the previous copy out of this stage must be complete
        stage.ensure(total)
        indptr_parts, keeps, o = [], [], 0
        for r0, r1, keep in ranges:
            e0, e1 = int(ip[r0]), int(ip[r1])
            self.reader.read_into(e0, e1, stage.data[o:o + e1 - e0],
                                  stage.indices[o:o + e1 - e0], self.n_threads)
            indptr_parts.append(ip[r0:r1] - e0 + o)
            keeps.append(np.ones(r1 - r0, bool) if keep is None else keep)
            o += e1 - e0
        indptr_raw = np.concatenate(indptr_parts + [[o]]).astype(np.int64)
        keep_all = np.concatenate(keeps)
        stream = torch.cuda.current_stream(dev) if dev.type == "cuda" else None
        indices, data = stage.to_device(total, dev, stream)
        if self._col_map is None and self.col_map_host is not None:
            self._col_map = torch.from_numpy(self.col_map_host).to(dev)
        indptr, lens, indices, data, lens_host = compact_rows(
            indptr_raw, indices, data, None if keep_all.all() else keep_all, self._col_map, dev)
        sizes = np.array([int(self.ends[j] - self.starts[j]) for j in blocks], dtype=np.int64)
        buf = Buffer(indptr, lens, indices, data, lens_host, self.starts[blocks].astype(np.int64),
                     np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64))
        if dev.type == "cuda":
            ev = torch.cuda.Event()
            ev.record(stream)
            buf.event = ev
        return buf


class HostCSRSource(_StreamedSource):
    kind = "host"

    def __init__(self, X, device, obs, block_size=None):
        import scipy.sparse as sp
        X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
        if X.data.dtype != np.float32:
            X = X.astype(np.float32)
        super().__init__(MemoryCSR(X), None, None, X.shape[1], device, obs, block_size)


class H5Source(_StreamedSource):
    """AnnDataOOM resolved to an on-disk CSR (anndataoom-fast reader)."""

    kind = "h5"


class OOMSource(DataSource):
    """AnnDataOOM read with anndataoom's own row reads (unresolvable view chains)."""

    kind = "oom"

    def __init__(self, arr, cols, device, obs, block_size=None):
        n_vars = arr.shape[1] if cols is None else len(cols)
        super().__init__(arr.shape[0], n_vars, device, obs, block_size)
        self.arr, self.cols = arr, cols

    def _load(self, blocks):
        import scipy.sparse as sp
        parts = []
        for j in blocks:
            m = self.arr._read_rows(int(self.starts[j]), int(self.ends[j]))
            m = sp.csr_matrix(m) if not sp.issparse(m) else m.tocsr()
            if self.cols is not None:
                m = m[:, self.cols]
            parts.append(m)
        X = sp.vstack(parts).tocsr()
        X.eliminate_zeros()
        buf = _csr_tensors(X.indptr, X.indices, X.data, self.device)
        sizes = np.array([int(self.ends[j] - self.starts[j]) for j in blocks], dtype=np.int64)
        buf.view_starts = self.starts[blocks].astype(np.int64)
        buf.offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        if self.device.type == "cuda":
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.device))
            buf.event = ev
        return buf


class DaskSource(DataSource):
    """Dask-backed AnnData: the dask row chunks are the blocks."""

    kind = "dask"

    def __init__(self, X, cols, device, obs):
        if X.numblocks[1] != 1:
            X = X.rechunk({1: -1})
        if cols is not None:
            X = X[:, cols]
        super().__init__(X.shape[0], X.shape[1], device, obs, block_size=None)
        bounds = np.concatenate([[0], np.cumsum(X.chunks[0])]).astype(np.int64)
        self.starts, self.ends = bounds[:-1], bounds[1:]
        keep = self.ends > self.starts
        self.starts, self.ends = self.starts[keep], self.ends[keep]
        self._block_index = np.flatnonzero(keep)
        self.X = X

    def _compute(self, j):
        import dask
        blk = self.X.blocks[int(self._block_index[j]), 0]
        return dask.compute(blk, scheduler="threads")[0]

    def _block_tensors(self, b):
        """(indptr int64 host, indices tensor, data tensor) of one computed block."""
        dev = self.device
        mod = type(b).__module__
        if mod.startswith("cupy"):
            import cupy as cp
            b = b.tocsr()
            if b.nnz:
                b.sum_duplicates()
            cp.cuda.get_current_stream().synchronize()
            ip = torch.from_dlpack(b.indptr.astype(cp.int64)).to(dev)
            ind = torch.from_dlpack(b.indices).to(dev)
            dat = torch.from_dlpack(b.data.astype(cp.float32, copy=False)).to(dev)
            return ip, ind, dat, None
        import scipy.sparse as sp
        if not sp.issparse(b):
            b = sp.csr_matrix(np.asarray(b))
        b = b.tocsr()
        data = b.data if b.data.dtype == np.float32 else b.data.astype(np.float32)
        return (torch.from_numpy(b.indptr.astype(np.int64)).to(dev),
                torch.from_numpy(b.indices).to(dev), torch.from_numpy(data).to(dev),
                np.asarray(b.indptr, dtype=np.int64))

    def _load(self, blocks):
        dev = self.device
        ips, inds, dats, offs = [], [], [], 0
        hosts = []
        for j in blocks:
            ip, ind, dat, iph = self._block_tensors(self._compute(j))
            ips.append(ip[:-1] + offs)
            hosts.append(iph)
            offs += int(dat.shape[0])
            inds.append(ind)
            dats.append(dat)
        indptr_raw = torch.cat(ips + [torch.tensor([offs], device=dev)])
        indptr_host = indptr_raw.cpu().numpy()
        indices = torch.cat(inds).to(torch.int32)
        data = torch.cat(dats)
        indptr, lens, indices, data, lens_host = compact_rows(indptr_host, indices, data, None,
                                                              None, dev)
        sizes = np.array([int(self.ends[j] - self.starts[j]) for j in blocks], dtype=np.int64)
        buf = Buffer(indptr, lens, indices, data, lens_host, self.starts[blocks].astype(np.int64),
                     np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64))
        if dev.type == "cuda":
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(dev))
            buf.event = ev
        return buf


# ---------------------------------------------------------------------------
# building a source from an AnnData-like object
# ---------------------------------------------------------------------------
def _device_budget(device) -> float:
    if device.type != "cuda":
        return float("inf")
    free, _ = torch.cuda.mem_get_info(device)
    return _RESIDENT_FRACTION * free


def _matrix(adata, layer):
    if layer is None:
        return adata.X
    if layer not in adata.layers.keys():
        raise KeyError(f"layer {layer!r} not found in adata.layers ({list(adata.layers.keys())})")
    return adata.layers[layer]


def _resolve_oom(adata, arr):
    try:
        from anndataoom_fast._view import identify_roots, resolve
    except ImportError:
        return None
    try:
        identify_roots(adata)
        return resolve(arr)
    except Exception:
        return None


def build_source(adata, *, layer=None, genes=None, obs=None, device="auto",
                 block_size=None, resident="auto", source=None):
    """Pick and build the minibatch source for ``adata`` (see module docstring).

    Arguments:
        adata: AnnData, AnnDataOOM or Dask-backed AnnData.
        layer: Raw-count layer (None = ``X``).
        genes: Training genes (mask / names / var column); None = all.
        obs: Dict of per-cell host arrays (codes, covariates) to keep on device.
        device: Training device.
        block_size: Cells per block for block shuffling.
        resident: ``'auto'`` keeps in-memory data on the device when it fits;
            ``True`` forces it (also loads AnnDataOOM / Dask data fully);
            ``False`` forces streaming.
        source: Force a source kind (``'device'``, ``'host'``, ``'h5'``,
            ``'dask'``, ``'oom'``).
    Returns:
        (source, var_index): the source and the indices of the training genes
        among ``adata.var`` (None = all genes).
    """
    import scipy.sparse as sp
    device = resolve_device(device)
    obs = {k: torch.as_tensor(np.ascontiguousarray(v)) for k, v in (obs or {}).items()}
    sel = gene_selection(adata.var, genes)
    X = _matrix(adata, layer)
    if is_dask_array(X):
        if resident is True or source == "device":
            Xc = (X if sel is None else X[:, sel]).compute()
            if type(Xc).__module__.startswith("cupy"):
                Xc = Xc.get()
            return DeviceCSRSource(Xc, device, obs, block_size), sel
        return DaskSource(X, sel, device, obs), sel
    if is_oom(adata):
        view = _resolve_oom(adata, X) if source in (None, "h5") else None
        if view is not None and not view.raw:
            raise ValueError(
                "ov.vi needs raw counts, but this AnnDataOOM matrix carries transforms "
                "(normalisation / log1p / scaling); pass layer='counts' (ov.pp.preprocess "
                "keeps the raw counts there).")
        if view is not None and resident is not True:
            reader = CSRFile.open(view.handle.path, view.handle.key)
            vcols = view.cols if view.cols is not None else np.arange(view.handle.shape[1])
            train_cols = vcols if sel is None else vcols[sel]
            col_map = None
            if not (len(train_cols) == view.handle.shape[1]
                    and np.array_equal(train_cols, np.arange(view.handle.shape[1]))):
                col_map = np.full(view.handle.shape[1], -1, dtype=np.int32)
                col_map[train_cols] = np.arange(len(train_cols), dtype=np.int32)
            src = H5Source(reader, view.rows, col_map, len(train_cols), device, obs, block_size)
            src.view = view
            return src, sel
        if resident is True:
            M = X[:, :] if sel is None else X[:, sel]
            M = M if sp.issparse(M) else sp.csr_matrix(np.asarray(M))
            return DeviceCSRSource(M, device, obs, block_size), sel
        return OOMSource(X, sel, device, obs, block_size), sel
    # in-memory AnnData
    if sel is not None:
        X = X[:, sel]
    if not sp.issparse(X):
        X = sp.csr_matrix(np.asarray(X))
    X = X.tocsr()
    nbytes = X.nnz * 8 + X.shape[0] * 16
    if source == "host" or resident is False or (
            source is None and resident == "auto" and nbytes > _device_budget(device)):
        return HostCSRSource(X, device, obs, block_size), sel
    return DeviceCSRSource(X, device, obs, block_size), sel


__all__ = [
    "build_source", "DataSource", "DeviceCSRSource", "HostCSRSource", "H5Source", "DaskSource",
    "OOMSource", "CSRFile", "split_train_val", "resolve_device", "gather_rows", "make_blocks",
    "DEFAULT_BLOCK_SIZE", "DEFAULT_BUFFER_BLOCKS",
]
