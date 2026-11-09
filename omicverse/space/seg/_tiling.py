r"""Tiled inference — segment an image of any size one window at a time.

The image is partitioned into disjoint ``tile_size`` blocks. Each block is
segmented together with a ``context`` margin on every side, and keeps only the
objects whose centroid falls inside the block -- in full, including the pixels
that spill into the margin. Every object therefore has exactly one owner, so
nothing is cut at a seam or labelled twice, as long as ``context`` exceeds the
object radius. When two owners claim the same margin pixel the earlier tile
keeps it.
"""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np


def tile_windows(shape, tile_size: int, context: int) -> Iterator[tuple[tuple, tuple]]:
    """Yield ``(core, window)`` pairs as ``(y0, y1, x0, x1)`` in image pixels.

    The cores partition the image; each window is its core grown by
    ``context`` and clipped to the image.
    """
    H, W = int(shape[0]), int(shape[1])
    tile_size = max(int(tile_size), 1)
    context = max(int(context), 0)
    for y0 in range(0, H, tile_size):
        y1 = min(y0 + tile_size, H)
        for x0 in range(0, W, tile_size):
            x1 = min(x0 + tile_size, W)
            yield (y0, y1, x0, x1), (
                max(0, y0 - context), min(H, y1 + context),
                max(0, x0 - context), min(W, x1 + context),
            )


def _owned_pixels(mask: np.ndarray, core: tuple, min_size: int):
    """Pixels ``(rows, cols, ids)`` of the objects whose centroid lies in ``core``.

    ``core`` is in the coordinates of ``mask``.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    ids = mask[ys, xs].astype(np.int64, copy=False)
    n = int(ids.max()) + 1
    area = np.bincount(ids, minlength=n)
    safe = np.maximum(area, 1)
    cy = np.bincount(ids, weights=ys, minlength=n) / safe
    cx = np.bincount(ids, weights=xs, minlength=n) / safe
    y0, y1, x0, x1 = core
    own = (area > max(int(min_size), 0)) & (cy >= y0) & (cy < y1) & (cx >= x0) & (cx < x1)
    own[0] = False
    keep = own[ids]
    if not keep.any():
        return None
    return ys[keep], xs[keep], ids[keep]


def segment_tiled(
    image,
    segmenter: Callable[[np.ndarray], np.ndarray],
    *,
    tile_size: int = 2048,
    context: int = 128,
    min_size: int = 0,
    sparse: bool = False,
    show_progress: bool = True,
    desc: str = "ov.space.seg",
):
    """Run ``segmenter`` window by window and stitch one global label image.

    Parameters
    ----------
    image
        ``(H, W)`` or ``(H, W, C)`` array, or anything sliceable with a
        ``shape`` (a memmap, a zarr array).
    segmenter
        Maps a window to an integer label image of the window's height and
        width (0 = background).
    tile_size, context
        Block size and the margin segmented around it, in pixels.
    min_size
        Objects of at most this many pixels are dropped.
    sparse
        Return a ``scipy.sparse.csr_matrix`` instead of a dense array; the
        label image is never materialised densely.

    Returns
    -------
    numpy.ndarray or scipy.sparse.csr_matrix
        ``int32`` labels ``1..n`` (0 = background), numbered in tile order.
    """
    H, W = int(image.shape[0]), int(image.shape[1])
    windows = list(tile_windows((H, W), tile_size, context))
    it = windows
    if show_progress and len(windows) > 1:
        try:
            from tqdm.auto import tqdm
            it = tqdm(windows, desc=desc, unit="tile")
        except ImportError:  # pragma: no cover - tqdm ships with omicverse
            pass

    rows, cols, vals = [], [], []
    next_id = 1
    for core, (wy0, wy1, wx0, wx1) in it:
        tile = np.asarray(image[wy0:wy1, wx0:wx1])
        mask = segmenter(tile)
        if mask is None:
            continue
        mask = np.asarray(mask)
        if mask.shape[:2] != tile.shape[:2]:
            raise ValueError(
                f"segmenter returned shape {mask.shape} for a window of shape {tile.shape[:2]}"
            )
        local_core = (core[0] - wy0, core[1] - wy0, core[2] - wx0, core[3] - wx0)
        owned = _owned_pixels(mask, local_core, min_size)
        if owned is None:
            continue
        ys, xs, ids = owned
        uniq, inv = np.unique(ids, return_inverse=True)
        rows.append((ys + wy0).astype(np.int64))
        cols.append((xs + wx0).astype(np.int64))
        vals.append(inv.astype(np.int64) + next_id)
        next_id += len(uniq)

    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        v = np.concatenate(vals)
        # first writer keeps a contested pixel; np.unique keeps the first index
        _, first = np.unique(r * W + c, return_index=True)
        r, c, v = r[first], c[first], v[first]
        # an object can lose every pixel to an earlier one: renumber densely
        _, v = np.unique(v, return_inverse=True)
        v = v.astype(np.int32) + 1
    else:
        r = c = np.zeros(0, dtype=np.int64)
        v = np.zeros(0, dtype=np.int32)

    if sparse:
        from scipy import sparse as sp
        return sp.csr_matrix((v, (r, c)), shape=(H, W), dtype=np.int32)
    out = np.zeros((H, W), dtype=np.int32)
    out[r, c] = v
    return out
