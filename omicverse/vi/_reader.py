"""Torch-free readers of CSR row ranges (used by ``ov.vi`` sources and workers).

Kept free of torch so processes that only read (lock tests, data services)
start fast.
"""
from __future__ import annotations

import os

import numpy as np

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


__all__ = ["CSRFile", "MemoryCSR"]
