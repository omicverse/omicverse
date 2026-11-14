"""Concurrent reads of one h5ad by N processes (spec: multi-GPU section).

    python lock_test.py --path SEAAD_raw.h5ad --procs 4 --mode pread
    python lock_test.py --path SEAAD_raw.h5ad --procs 4 --mode h5py

pread: the chunk table is built once in the parent (``CSRFile``, h5py opened
once); N spawned workers stream X.data / X.indices with anndataoom-fast's
pread reader (as ov.vi's DDP workers do) -- they never open the file with HDF5.
h5py: N workers each open the file with h5py (default HDF5 file locking) and
read the same ranges.  Each worker reads its contiguous share of the rows
(``--split``) or the whole matrix, within ``--max-seconds``; reports bytes,
throughput and errors.  Results are appended to lock_test.csv.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ranges(indptr, rank, procs, split):
    n = len(indptr) - 1
    if split:
        r0, r1 = rank * n // procs, (rank + 1) * n // procs
    else:
        r0, r1 = 0, n
    return int(indptr[r0]), int(indptr[r1])


def _pread_worker(rank, procs, f, split, max_s, chunk, q):
    try:
        e0, e1 = _ranges(f.indptr, rank, procs, split)
        data = np.empty(chunk, np.float32)
        idx = np.empty(chunk, np.int32)
        t = time.perf_counter()
        done = 0
        for a in range(e0, e1, chunk):
            b = min(e1, a + chunk)
            f.read_into(a, b, data[:b - a], idx[:b - a], 4)
            done += b - a
            if time.perf_counter() - t > max_s:
                break
        q.put((rank, done, time.perf_counter() - t, None, e1 - e0))
    except Exception:
        q.put((rank, 0, 0.0, traceback.format_exc(limit=3), -1))


def _h5py_worker(rank, procs, path, key, split, max_s, chunk, q):
    try:
        import h5py
        t = time.perf_counter()
        with h5py.File(path, "r") as fh:  # default HDF5 file locking
            g = fh[key]
            ip = g["indptr"][:]
            e0, e1 = _ranges(ip, rank, procs, split)
            done = 0
            for a in range(e0, e1, chunk):
                b = min(e1, a + chunk)
                g["data"][a:b]
                g["indices"][a:b]
                done += b - a
                if time.perf_counter() - t > max_s:
                    break
        q.put((rank, done, time.perf_counter() - t, None, e1 - e0))
    except Exception:
        q.put((rank, 0, 0.0, traceback.format_exc(limit=3), -1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--key", default="X")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--mode", choices=["pread", "h5py"], default="pread")
    p.add_argument("--split", action="store_true", help="each process reads 1/N of the rows")
    p.add_argument("--max-seconds", type=float, default=600)
    p.add_argument("--chunk", type=int, default=1 << 24)
    p.add_argument("--hold-writer", action="store_true",
                   help="keep the file open for writing (h5py mode 'a') in the parent while "
                        "the readers run, as a rank-0 writer would; use a scratch copy")
    args = p.parse_args()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.perf_counter()
    if args.mode == "pread":
        from omicverse.vi._data import CSRFile
        f = CSRFile.open(args.path, args.key)  # the only HDF5 open
        nnz = int(f.indptr[-1])
        isz = {2: 1, 3: 2, 4: 4, 5: 8}.get(f.indices_desc[2], 8)  # on-disk index width
        procs = [ctx.Process(target=_pread_worker, args=(r, args.procs, f, args.split,
                                                         args.max_seconds, args.chunk, q))
                 for r in range(args.procs)]
    else:
        import h5py
        with h5py.File(args.path, "r") as fh:
            nnz = int(fh[args.key]["data"].shape[0])
            isz = fh[args.key]["indices"].dtype.itemsize
        procs = [ctx.Process(target=_h5py_worker, args=(r, args.procs, args.path, args.key,
                                                        args.split, args.max_seconds,
                                                        args.chunk, q))
                 for r in range(args.procs)]
    writer = None
    if args.hold_writer:
        import h5py
        writer = h5py.File(args.path, "a")  # exclusive HDF5 lock while open
    for pr in procs:
        pr.start()
    res = [q.get() for _ in procs]
    for pr in procs:
        pr.join()
    if writer is not None:
        writer.close()
    wall = time.perf_counter() - t0
    import common
    rows = []
    for rank, done, sec, err, target in sorted(res, key=lambda r: r[0]):
        gb = done * (4 + isz) / 1e9
        r = dict(mode=args.mode, procs=args.procs, split=args.split, rank=rank,
                 file=os.path.basename(args.path), writer_open=args.hold_writer,
                 elements=done, gb_read=round(gb, 2), seconds=round(sec, 1),
                 gb_per_s=round(gb / sec, 3) if sec else 0.0,
                 full=bool(done >= target >= 0),
                 error=(err or "").strip().splitlines()[-1] if err else "", wall_s=round(wall, 1),
                 hdf5_locking=os.environ.get("HDF5_USE_FILE_LOCKING", "default"))
        print(r, flush=True)
        if err:
            print(err)
        rows.append(r)
    common.write_table(rows, "lock_test")


if __name__ == "__main__":
    main()
