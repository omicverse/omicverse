"""Generic AnnDataOOM guard applied by :func:`omicverse._registry.register_function`.

A registered function that is not OOM-native (see :mod:`omicverse._oom_native`)
and receives a disk-backed ``AnnDataOOM`` (first positional argument or
``adata=``) is run on an in-memory copy:

1. the in-memory size of X (+ layers, raw) is estimated and compared with the
   RAM available to this process; above ``MEMORY_FRACTION`` a ``MemoryError``
   is raised before anything is loaded;
2. the object is materialised (``to_adata()``);
3. the function runs on the in-memory ``AnnData``;
4. every new or changed entry of obs, var, uns, obsm, varm, obsp, varp and
   layers is copied back to the ``AnnDataOOM`` (deleted entries are deleted);
5. a returned ``AnnData`` other than the working copy is returned as is.

If the function changed X itself (in place, or by subsetting cells/genes in
place), the disk-backed X cannot be updated: the in-memory ``AnnData`` holding
the result is returned and a warning says so.

Set the environment variable ``OMICVERSE_OOM_GUARD=0`` to disable the guard.
"""

from __future__ import annotations

import contextlib
import io
import os
import threading
import warnings

import numpy as np

#: Fraction of available RAM the materialised object may take.
MEMORY_FRACTION = 0.8

_state = threading.local()

_DENSE_LAZY = ("ScaledBackedArray", "PearsonResidualBackedArray",
               "RegressedBackedArray", "CenteredSparseArray")


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------

def _is_oom(obj) -> bool:
    # Cheap pre-check first so the plain-AnnData path never imports anndataoom.
    if getattr(type(obj), "_is_oom", False) is not True:
        return False
    from ._oom_compat import is_oom
    return bool(is_oom(obj))


def _find_adata(args, kwargs):
    if "adata" in kwargs:
        return kwargs["adata"], "kw"
    if args:
        return args[0], "pos"
    return None, None


def _public_name(func) -> str:
    mod = getattr(func, "__module__", "") or ""
    parts = [p for p in mod.split(".") if p and not p.startswith("_")]
    if parts and parts[0] == "omicverse":
        parts[0] = "ov"
    return ".".join(parts + [getattr(func, "__qualname__", func.__name__)])


# ---------------------------------------------------------------------------
# memory estimate
# ---------------------------------------------------------------------------

def _cgroup_available():
    """Bytes left under this process's cgroup memory limit (Slurm jobs), or None."""
    try:
        with open("/proc/self/cgroup") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    best = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        ctrls, path = parts[1], parts[2]
        if ctrls == "":  # cgroup v2
            base, lim_f, use_f = "/sys/fs/cgroup", "memory.max", "memory.current"
        elif "memory" in ctrls.split(","):
            base, lim_f, use_f = "/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        p = path
        while True:
            d = os.path.join(base, p.lstrip("/"))
            try:
                with open(os.path.join(d, lim_f)) as fh:
                    raw = fh.read().strip()
                with open(os.path.join(d, use_f)) as fh:
                    used = int(fh.read().strip())
                if raw != "max":
                    lim = int(raw)
                    if lim < (1 << 60):
                        free = max(lim - used, 0)
                        best = free if best is None else min(best, free)
            except (OSError, ValueError):
                pass
            if p in ("", "/"):
                break
            p = os.path.dirname(p.rstrip("/")) or "/"
    return best


def available_memory():
    """RAM available to this process in bytes (system and cgroup limit), or None."""
    from .utils._memory import get_available_memory
    vals = [v for v in (get_available_memory(), _cgroup_available()) if v is not None]
    return min(vals) if vals else None


def _file_density(path, key):
    """nnz / (rows * cols) of a sparse matrix group in an .h5ad, 1.0 if dense."""
    if not path:
        return None
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if key not in f:
                return None
            node = f[key]
            if isinstance(node, h5py.Dataset):
                return 1.0
            shape = node.attrs.get("shape")
            if shape is None or "data" not in node:
                return None
            total = float(shape[0]) * float(shape[1])
            return node["data"].shape[0] / total if total else 0.0
    except Exception:
        return None


def _array_nbytes(arr, density):
    from scipy.sparse import issparse
    if arr is None:
        return 0
    if issparse(arr):
        return int(arr.data.nbytes + arr.indices.nbytes + arr.indptr.nbytes)
    if isinstance(arr, np.ndarray):
        return int(arr.nbytes)
    shape = getattr(arr, "shape", None)
    if shape is None or len(shape) != 2:
        return 0
    n = float(shape[0]) * float(shape[1])
    try:
        itemsize = max(np.dtype(arr.dtype).itemsize, 4)
    except Exception:
        itemsize = 4
    dense = int(n * itemsize)
    cls = type(arr).__name__
    if cls in _DENSE_LAZY:
        return dense
    parent = getattr(arr, "_parent", None)
    if parent is not None:  # lazy normalize/log1p keeps sparsity of its parent
        return _array_nbytes(parent, density)
    elem = getattr(arr, "_elem", None)
    if elem is not None and elem is not arr:
        if issparse(elem) or isinstance(elem, np.ndarray):
            return _array_nbytes(elem, density)
        if type(elem).__name__ == "_H5BackedMatrix":
            return dense
    if density is not None:
        return int(density * n * (itemsize + 4) + (shape[0] + 1) * 8)
    return dense


def estimate_nbytes(adata) -> int:
    """Estimated in-memory size of ``adata.X`` + layers + raw after ``to_adata()``."""
    path = getattr(adata, "_origin_file", None)
    total = _array_nbytes(adata.X, _file_density(path, "X"))
    try:
        for k in list(adata.layers.keys()):
            total += _array_nbytes(adata.layers[k], _file_density(path, f"layers/{k}"))
    except Exception:
        pass
    raw = getattr(adata, "_raw", None)
    if raw is not None:
        try:
            rx = getattr(raw, "X", None)
            total += _array_nbytes(rx, _file_density(path, "raw/X"))
        except Exception:
            pass
    return total


def _gb(n):
    return f"{n / 1024 ** 3:.2f} GB"


# ---------------------------------------------------------------------------
# snapshot / copy-back
# ---------------------------------------------------------------------------

def _fingerprint(x):
    from scipy.sparse import issparse
    try:
        if issparse(x):
            data = x.data
            s = float(np.nansum(data, dtype=np.float64)) if data.dtype.kind in "biuf" else None
            return (id(x), x.shape, x.nnz, id(data), s)
        if isinstance(x, np.ndarray):
            s = float(np.nansum(x, dtype=np.float64)) if x.dtype.kind in "biuf" else None
            return (id(x), x.shape, x.dtype.str, s)
    except Exception:
        pass
    return (id(x), getattr(x, "shape", None))


def _snap_mapping(mapping):
    return {k: _fingerprint(mapping[k]) for k in list(mapping.keys())}


def _copy_mapping_back(oom_map, mem_map, before):
    for k in list(mem_map.keys()):
        v = mem_map[k]
        if k not in before or _fingerprint(v) != before[k]:
            oom_map[k] = v
    for k in before:
        if k not in mem_map and k in oom_map:
            del oom_map[k]


def _copy_frame_back(oom, mem, attr):
    o_df, m_df = getattr(oom, attr), getattr(mem, attr)
    same_index = m_df.index.equals(o_df.index)
    for col in m_df.columns:
        s = m_df[col]
        if col in o_df.columns:
            try:
                if same_index and s.equals(o_df[col]):
                    continue
            except Exception:
                pass
        o_df[col] = s.values if len(s) == len(o_df) else s.reindex(o_df.index).values
    for col in [c for c in o_df.columns if c not in m_df.columns]:
        del o_df[col]
    if not same_index and len(m_df.index) == len(o_df.index):
        o_df.index = m_df.index.copy()


def _copy_layers_back(oom, mem, before):
    from scipy.sparse import issparse
    for k in list(mem.layers.keys()):
        v = mem.layers[k]
        if k in before and _fingerprint(v) == before[k]:
            continue
        if issparse(v):  # keep sparse: BackedLayers would densify it
            from anndataoom import BackedArray
            oom.layers[k] = BackedArray(v, shape=v.shape)
        else:
            oom.layers[k] = v
    for k in before:
        if k not in mem.layers and k in oom.layers:
            del oom.layers[k]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _materialise(adata):
    sink = io.StringIO()  # to_adata() prints a long banner; the guard prints one line
    with contextlib.redirect_stdout(sink):
        if hasattr(adata, "to_adata"):
            return adata.to_adata()
        return adata.to_memory()


def call(func, args, kwargs, native):
    """Run ``func(*args, **kwargs)``, materialising an AnnDataOOM if needed."""
    if getattr(_state, "native_depth", 0):
        return func(*args, **kwargs)
    adata, where = _find_adata(args, kwargs)
    if adata is None or not _is_oom(adata) or os.environ.get("OMICVERSE_OOM_GUARD", "1") == "0":
        return func(*args, **kwargs)

    if native:
        _state.native_depth = getattr(_state, "native_depth", 0) + 1
        try:
            return func(*args, **kwargs)
        finally:
            _state.native_depth -= 1

    name = _public_name(func)
    need = estimate_nbytes(adata)
    avail = available_memory()
    shape = f"{adata.n_obs:,} x {adata.n_vars:,}"
    if avail is not None and need > MEMORY_FRACTION * avail:
        raise MemoryError(
            f"{name}() needs the data in memory, but the disk-backed AnnDataOOM ({shape}) "
            f"would take ~{_gb(need)} and only {_gb(avail)} RAM is available "
            f"(limit {int(MEMORY_FRACTION * 100)}%). Options: subset first "
            f"(adata[cells, genes]), read with ov.read(path, storage='memory') on a larger "
            f"machine, use an OOM-native function (ov.pp.*, ov.single.find_markers, ov.pl.*), "
            f"or convert explicitly with ov.utils.to_memory(adata)."
        )
    print(f"[AnnDataOOM] {name}(): loading {shape} into memory (~{_gb(need)}); "
          f"results are copied back to the disk-backed object "
          f"(ov.utils.to_memory(adata) converts once for repeated calls).")

    mem = _materialise(adata)
    x_before = _fingerprint(mem.X)
    shape_before = mem.shape
    snaps = {a: _snap_mapping(getattr(mem, a)) for a in ("obsm", "varm", "obsp", "varp")}
    layers_before = _snap_mapping(mem.layers)

    if where == "kw":
        kwargs = dict(kwargs)
        kwargs["adata"] = mem
        result = func(*args, **kwargs)
    else:
        result = func(mem, *args[1:], **kwargs)

    # uns: values are deep copies, so copying every key back is equivalent to
    # copying the new/changed ones and also catches nested in-place edits.
    for k in list(mem.uns.keys()):
        adata.uns[k] = mem.uns[k]
    for k in [k for k in list(adata.uns.keys()) if k not in mem.uns]:
        del adata.uns[k]

    if mem.shape != shape_before:
        warnings.warn(
            f"{name}() subset the data in place ({shape} -> {mem.shape[0]:,} x {mem.shape[1]:,}); "
            f"the disk-backed object cannot follow, so the in-memory AnnData with the result "
            f"is returned (only .uns was copied back).", UserWarning, stacklevel=3)
        return mem if result is None or result is mem else result

    _copy_frame_back(adata, mem, "obs")
    _copy_frame_back(adata, mem, "var")
    for a in ("obsm", "varm", "obsp", "varp"):
        _copy_mapping_back(getattr(adata, a), getattr(mem, a), snaps[a])
    _copy_layers_back(adata, mem, layers_before)

    if _fingerprint(mem.X) != x_before:
        warnings.warn(
            f"{name}() modified X, which cannot be written back to the disk-backed "
            f"AnnDataOOM; the in-memory AnnData holding the result is returned "
            f"(obs/var/obsm/uns/layers were copied back to the disk-backed object).",
            UserWarning, stacklevel=3)
        return mem if result is None or result is mem else result

    if result is mem:
        return adata
    return result
