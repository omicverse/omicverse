r"""Tiled, memory-bounded 3-D instance segmentation from three orthogonal 2-D views.

``segment_volume_multiview`` segments every nucleus (or cell) of a
``(Z, Y, X)`` fluorescence volume of any size -- a numpy array, a memmap, a
zarr array or a path to one -- into a 3-D label volume, without ever holding a
whole-volume intermediate in memory.

Why three views and not Cellpose's own 3-D mode or slice stitching
-----------------------------------------------------------------
With anisotropic voxels (light-sheet, spinning disk: z-steps several times the
pixel size) a nucleus is wide in xy but only a few planes deep. Stitching 2-D
xy masks across z by overlap merges stacked nuclei and splits tilted ones;
Cellpose's ``do_3D`` averages raw *flows* from the three views, where a view
that sees two touching nuclei as one pulls the others towards a merge, while
noisy flows split tall nuclei. The indirect method of u-Segment3D (Zhou et
al., Nat. Methods 2025) instead aggregates the three views' *2-D label masks*:

1. each view's 2-D labels vote for a foreground probability (mean of the three
   foregrounds), thresholded by multi-Otsu;
2. each view's per-label 2-D distance transform gives a unit in-plane
   gradient pointing to that 2-D object's centre; every 3-D axis is seen by
   two views, whose gradients are averaged into one 3-D field;
3. the field is rescaled by its divergence (large at object centres) and every
   foreground voxel is advected along it with momentum until it condenses on
   a centre;
4. voxels whose end points fall in the same density cluster form one object.

A nucleus missed or merged in one view is usually resolved in the other two.

Working grid
------------
Each tile is resampled to a near-isotropic working grid before the 2-D passes:
the target voxel is the geometric mean of the in-plane and axial spacing (the
point that halves the anisotropy in log scale for both); xy is block-averaged
by the integer factor nearest to ``target / dxy`` (also raising the
signal-to-noise ratio) and z is linearly up-sampled by the factor nearest to
``dz / target``. Labels are mapped back to the original grid at the end.

Tiling and seams
----------------
The volume is cut into tiles whose *cores* partition it; each tile is
processed with a halo of at least twice the nucleus diameter on every side and
keeps only the objects whose centroid lies in its core, in full (including the
voxels in the halo). Core voxels are written unconditionally, halo voxels only
where still empty, so every object has exactly one owner and nothing is cut or
counted twice at a seam. The tile size is chosen from ``memory_mb``. Finally
each label keeps its largest 26-connected component and objects much smaller
than a nucleus are dropped (chunked passes), and a per-object table is built.
"""
from __future__ import annotations

import math
import os
import time
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ..._registry import register_function

__all__ = ["segment_volume_multiview", "aggregate_orthoviews", "multiview_working_grid"]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _log_fn(verbose):
    if callable(verbose):
        return verbose
    if not verbose:
        return lambda *a: None

    def _p(*a):
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
    return _p


def _peak_rss_mb() -> float:
    """High-water RSS of this process in MB (/proc VmHWM; ru_maxrss is inherited
    from the parent across fork + exec, so it can report a parent's peak)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # pragma: no cover - non-POSIX
        return float("nan")


def _release():
    """Return freed heap to the OS (glibc keeps it otherwise; RSS creeps up tile by tile)."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # pragma: no cover - non-glibc platforms
        pass


def _open_volume(volume):
    if isinstance(volume, (str, os.PathLike)):
        p = str(volume)
        if p.lower().endswith((".tif", ".tiff")):
            import tifffile
            try:
                return tifffile.memmap(p)
            except Exception:
                return tifffile.imread(p)
        import zarr
        return zarr.open(p, mode="r")
    return volume


def _n_threads(n):
    if n:
        return int(n)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:  # pragma: no cover
        return max(1, os.cpu_count() or 1)


@register_function(
    aliases=["多视角工作网格", "各向异性重采样", "multiview working grid",
             "near-isotropic resampling factors"],
    category="space",
    description="Near-isotropic working grid for anisotropic volumes: xy block-mean factor and "
                "z up-sampling factor that meet at the geometric mean of the in-plane and axial "
                "voxel size (used by segment_volume_multiview).",
    requires={},
    produces={},
    auto_fix="none",
    examples=["xy_bin, z_up, wsp = ov.space.seg.multiview_working_grid((2.5, 0.43, 0.43))"],
    related=["space.seg.segment_volume_multiview"],
)
def multiview_working_grid(spacing: Sequence[float], diameter: Optional[float] = None,
                           min_diameter_px: float = 8.0):
    """Return ``(xy_bin, z_up, working_spacing)`` for voxel size ``(dz, dy, dx)``.

    ``xy_bin`` (int >= 1) block-averages y and x, ``z_up`` (>= 1) linearly
    up-samples z. When ``diameter`` (same unit as ``spacing``) is given, the
    xy binning is limited so the object stays at least ``min_diameter_px``
    working pixels across.
    """
    dz, dy, dx = (float(s) for s in spacing)
    dxy = math.sqrt(dy * dx)
    if dz <= dxy * 1.5:                     # (near-)isotropic or finer in z: no resampling
        return 1, 1.0, (dz, dy, dx)
    target = math.sqrt(dz * dxy)
    xy_bin = max(1, int(round(target / dxy)))
    if diameter is not None and diameter > 0:
        xy_bin = max(1, min(xy_bin, int(diameter / dxy // min_diameter_px)))
    wxy = dxy * xy_bin
    z_up = max(1.0, float(round(dz / max(target, wxy))))
    return xy_bin, z_up, (dz / z_up, dy * xy_bin, dx * xy_bin)


def _to_working(img, xy_bin, z_up, smooth):
    Z, Y, X = img.shape
    w = img
    if xy_bin > 1:
        Yc, Xc = Y - Y % xy_bin, X - X % xy_bin
        w = w[:, :Yc, :Xc].reshape(Z, Yc // xy_bin, xy_bin, Xc // xy_bin, xy_bin).mean(axis=(2, 4))
    if z_up != 1.0:
        w = ndi.zoom(w, (z_up, 1, 1), order=1)
    if smooth and smooth > 0:
        w = ndi.gaussian_filter(w, smooth)
    return np.ascontiguousarray(w, dtype=np.float32)


def _from_working(seg, shape, xy_bin, z_up):
    Z, Y, X = shape
    if z_up != 1.0:
        zi = np.clip(np.round((np.arange(Z) + 0.5) * z_up - 0.5).astype(int), 0, seg.shape[0] - 1)
        seg = seg[zi]
    if xy_bin > 1:
        seg = np.repeat(np.repeat(seg, xy_bin, axis=1), xy_bin, axis=2)
    out = np.zeros((Z, Y, X), np.uint32)
    out[:, :min(Y, seg.shape[1]), :min(X, seg.shape[2])] = seg[:, :Y, :X]
    return out


# --------------------------------------------------------------------------- #
# indirect 2-D -> 3-D aggregation (u-Segment3D)
# --------------------------------------------------------------------------- #
def _label_edt_2d(lab):
    """Distance to the nearest voxel of another label or background, per label."""
    fg = lab > 0
    if not fg.any():
        return np.zeros(lab.shape, np.float32)
    seam = np.zeros(lab.shape, bool)
    a, b = lab[:-1, :], lab[1:, :]
    d = (a != b) & (a > 0) & (b > 0)
    seam[:-1, :] |= d
    seam[1:, :] |= d
    a, b = lab[:, :-1], lab[:, 1:]
    d = (a != b) & (a > 0) & (b > 0)
    seam[:, :-1] |= d
    seam[:, 1:] |= d
    return ndi.distance_transform_edt(fg & ~seam).astype(np.float32)


def _view_gradient(stack, axis_out):
    """Unit in-plane gradient of each slice's per-label EDT; ``(2,) + stack.shape``."""
    out = np.zeros((2,) + stack.shape, np.float32)
    for i in range(stack.shape[0]):
        sl = stack[i]
        if not sl.any():
            continue
        g0, g1 = np.gradient(_label_edt_2d(sl))
        n = np.sqrt(g0 * g0 + g1 * g1) + 1e-20
        out[0, i] = g0 / n
        out[1, i] = g1 / n
    return out


def _normalize99(y, lower=0.01, upper=99.0):
    lo, hi = np.percentile(y, lower), np.percentile(y, upper)
    return np.clip((y - lo) / (hi - lo + 1e-20), 0, 1)


@register_function(
    aliases=["正交视角聚合", "二维转三维分割", "u-Segment3D", "aggregate orthogonal views",
             "2D to 3D label aggregation", "three-view 3D segmentation"],
    category="space",
    description="u-Segment3D indirect aggregation: 2-D label stacks from the xy, xz and yz views "
                "of one volume -> one 3-D instance labelling (foreground vote, per-view distance-"
                "transform gradients merged into a 3-D field, divergence rescale, gradient "
                "advection and density clustering).",
    requires={},
    produces={},
    auto_fix="none",
    examples=["seg = ov.space.seg.aggregate_orthoviews(lab_xy, lab_xz, lab_yz)"],
    related=["space.seg.segment_volume_multiview"],
)
def aggregate_orthoviews(lab_xy, lab_xz, lab_yz, *, smooth_binary: float = 1.0,
                         prob_threshold: Optional[float] = None, min_size: int = 30,
                         min_object: int = 5, n_iter: int = 200, step: float = 1.0,
                         momentum: float = 0.98, step_decay: float = 0.1,
                         gradient_sigma: float = 1.0, cluster_sigma: float = 0.5,
                         divergence_rescale: bool = True, border_guard: int = 1,
                         verbose=False):
    """Aggregate three orthogonal 2-D label stacks into 3-D instances.

    Parameters
    ----------
    lab_xy, lab_xz, lab_yz
        Per-slice 2-D labels of the same ``(Z, Y, X)`` volume: ``lab_xy`` is
        ``(Z, Y, X)`` (one xy slice per z), ``lab_xz`` is ``(Y, Z, X)`` and
        ``lab_yz`` is ``(X, Z, Y)``. Label ids need only be unique per slice.
    prob_threshold
        Foreground threshold on the mean of the three (smoothed) foregrounds;
        ``None`` = the upper multi-Otsu level (rounded down to 0.1).
    n_iter, step, momentum, step_decay
        Advection of every foreground voxel along the merged gradient field.
    cluster_sigma
        Gaussian smoothing of the end-point density before clustering.

    Returns
    -------
    numpy.ndarray
        ``uint32`` labels ``(Z, Y, X)``.
    """
    from skimage.filters import threshold_multiotsu
    from skimage.measure import label as cc_label
    from skimage.morphology import ball, binary_closing, binary_dilation, remove_small_objects

    log = _log_fn(verbose)
    lab_xy = np.asarray(lab_xy)
    shape = lab_xy.shape
    lab_xz = np.asarray(lab_xz)
    lab_yz = np.asarray(lab_yz)
    if lab_xz.transpose(1, 0, 2).shape != shape or lab_yz.transpose(1, 2, 0).shape != shape:
        raise ValueError("lab_xz must be (Y, Z, X) and lab_yz (X, Z, Y) of the xy stack's volume")

    # 1. foreground probability
    prob = np.zeros(shape, np.float32)
    prob += ndi.gaussian_filter((lab_xy > 0).astype(np.float32), smooth_binary)
    prob += ndi.gaussian_filter((lab_xz > 0).astype(np.float32), smooth_binary).transpose(1, 0, 2)
    prob += ndi.gaussian_filter((lab_yz > 0).astype(np.float32), smooth_binary).transpose(1, 2, 0)
    prob /= 3.0
    pmax = float(prob.max())
    if pmax <= 0:
        return np.zeros(shape, np.uint32)
    prob /= pmax
    if prob_threshold is None:
        try:
            t = max(0.0, math.floor(float(threshold_multiotsu(prob, 3)[1]) * 10) / 10.0)
        except ValueError:
            t = 0.5
    else:
        t = float(prob_threshold)
    log(f"aggregate: foreground threshold {t:.2f}")
    binary = prob >= t
    del prob
    binary = binary_closing(binary, ball(1))
    if min_size > 0:
        binary = remove_small_objects(binary, min_size, connectivity=2)
    binary = binary_dilation(binary, ball(1))
    if not binary.any():
        return np.zeros(shape, np.uint32)

    # 2. merged 3-D gradient field; each axis is seen by two views
    grad = np.zeros((3,) + shape, np.float32)
    g = _view_gradient(lab_xy, None)                      # (y, x) over z-slices
    del lab_xy
    grad[1] += ndi.gaussian_filter(g[0], gradient_sigma)
    grad[2] += ndi.gaussian_filter(g[1], gradient_sigma)
    del g
    g = _view_gradient(lab_xz, None)                      # (z, x) over y-slices, (Y,Z,X)
    del lab_xz
    grad[0] += ndi.gaussian_filter(g[0], gradient_sigma).transpose(1, 0, 2)
    grad[2] += ndi.gaussian_filter(g[1], gradient_sigma).transpose(1, 0, 2)
    del g
    g = _view_gradient(lab_yz, None)                      # (z, y) over x-slices, (X,Z,Y)
    del lab_yz
    grad[0] += ndi.gaussian_filter(g[0], gradient_sigma).transpose(1, 2, 0)
    grad[1] += ndi.gaussian_filter(g[1], gradient_sigma).transpose(1, 2, 0)
    del g
    grad *= 0.5

    # 3. divergence rescale
    if divergence_rescale:
        H = np.gradient(grad[0], axis=0)
        H += np.gradient(grad[1], axis=1)
        H += np.gradient(grad[2], axis=2)
        H *= 0.5
        grad *= _normalize99(H).astype(np.float32)[None]
        del H

    # 4. advection with momentum and a decaying step
    pts0 = np.argwhere(binary).astype(np.int32)
    del binary
    pts = pts0.astype(np.float32)
    flat = grad.reshape(3, -1)
    v_prev = np.zeros_like(pts)
    lim = (np.asarray(shape, np.float32) - 1)[None, :]
    dims = np.asarray(shape, np.int64)
    for ii in range(n_iter):
        idx = pts[:, 0].astype(np.int64)
        idx *= dims[1]
        idx += pts[:, 1].astype(np.int64)
        idx *= dims[2]
        idx += pts[:, 2].astype(np.int64)
        v = flat[:, idx].T
        del idx
        v_prev *= momentum
        v_prev += step * v
        v_prev *= (1.0 / (step + momentum)) * (step / (1.0 + ii * step_decay))
        pts += v_prev
        np.clip(pts, 0, lim, out=pts)
        v_prev = v
    del flat, grad, v_prev

    # 5. density clustering of the end points, painted back onto the start voxels
    pi = pts.astype(np.int32)
    del pts
    votes = np.zeros(shape, np.float32)
    np.add.at(votes, (pi[:, 0], pi[:, 1], pi[:, 2]), 1.0)
    votes = ndi.gaussian_filter(votes, cluster_sigma)
    if border_guard:
        b = int(border_guard)
        votes[:b] = 0; votes[-b:] = 0
        votes[:, :b] = 0; votes[:, -b:] = 0
        votes[:, :, :b] = 0; votes[:, :, -b:] = 0
    seeds = cc_label(votes > votes.mean(), connectivity=3).astype(np.uint32)
    del votes
    out = np.zeros(shape, np.uint32)
    out[pts0[:, 0], pts0[:, 1], pts0[:, 2]] = seeds[pi[:, 0], pi[:, 1], pi[:, 2]]
    del seeds, pi, pts0
    if min_object > 0 and out.any():
        cnt = np.bincount(out.ravel())
        bad = np.nonzero(cnt < min_object)[0]
        bad = bad[bad > 0]
        if bad.size:
            out[np.isin(out, bad)] = 0
    return out


# --------------------------------------------------------------------------- #
# 2-D segmenters
# --------------------------------------------------------------------------- #
def _slice_normalize(stack, floor, lower=1.0, upper=99.0):
    """Cellpose's per-image percentile normalisation, with the upper level floored.

    ``(x - p_lower) / (p_upper - p_lower)`` per slice, as Cellpose does by
    default, except that ``p_upper`` is never below ``floor`` (the volume's
    foreground level): a slice that is mostly background is then not stretched
    until its noise looks like nuclei (which yields hundreds of spurious
    masks and makes their computation very slow).
    """
    out = np.empty(stack.shape, np.float32)
    for i in range(stack.shape[0]):
        sl = stack[i]
        lo, hi = np.percentile(sl, [lower, upper])
        hi = max(float(hi), float(floor))
        out[i] = (sl - lo) / max(hi - lo, 1e-6)
    return out


class _CellposeView:
    """Cellpose 2-D model applied to a stack of slices, in small batches."""

    def __init__(self, model, *, gpu=False, flow_threshold=0.4, cellprob_threshold=0.0,
                 batch_size=2, chunk=16, normalize=True):
        self.spec = model
        self.gpu = gpu
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.batch_size = batch_size
        self.chunk = chunk
        self.normalize = normalize
        self._m = None

    def model(self):
        if self._m is None:
            if not isinstance(self.spec, str):
                self._m = self.spec
            else:
                from cellpose import models
                builtin = {"cyto", "cyto2", "cyto3", "nuclei", "tissuenet_cp3", "livecell_cp3",
                           "cyto2_cp3", "deepbacs_cp3", "bact_fluor_cp3"}
                if self.spec in builtin and not os.path.exists(self.spec):
                    self._m = models.CellposeModel(gpu=self.gpu, model_type=self.spec)
                else:
                    self._m = models.CellposeModel(gpu=self.gpu, pretrained_model=self.spec)
        return self._m

    @property
    def diam_labels(self):
        m = self.model()
        return float(getattr(m, "diam_labels", getattr(m, "diam_mean", 30.0)))

    def __call__(self, stack, diameter):
        m = self.model()
        out = np.zeros(stack.shape, np.int32)
        for s in range(0, stack.shape[0], self.chunk):
            sl = [stack[i] for i in range(s, min(s + self.chunk, stack.shape[0]))]
            res = m.eval(sl, channels=[0, 0], diameter=float(diameter), do_3D=False,
                         normalize=self.normalize, flow_threshold=self.flow_threshold,
                         cellprob_threshold=self.cellprob_threshold, batch_size=self.batch_size)
            for k, msk in enumerate(res[0]):
                out[s + k] = np.asarray(msk, np.int32)
            del res
        return out


def _as_view(model, **kw):
    if model is None:
        return None
    if callable(model) and not hasattr(model, "eval"):
        return model                                # custom: f(stack, diameter_px) -> labels
    return _CellposeView(model, **kw)


def _run_view(seg, stack, diameter, sig_level=None, min_pixels=0, norm_floor=None):
    """Run a 2-D segmenter on the slices of ``stack`` that carry signal; others stay 0.

    Slices with fewer than ``min_pixels`` voxels above ``sig_level`` are not
    segmented. With ``norm_floor`` the slices are percentile-normalised here
    (upper level floored, see :func:`_slice_normalize`) and the segmenter
    must not normalise again.
    """
    if sig_level is not None and min_pixels > 0:
        active = (stack > sig_level).reshape(stack.shape[0], -1).sum(1) >= min_pixels
    else:
        active = np.ones(stack.shape[0], bool)
    out = np.zeros(stack.shape, np.int32)
    if active.any():
        idx = np.nonzero(active)[0]
        sub = np.ascontiguousarray(stack[idx]) if not active.all() else stack
        if norm_floor is not None:
            sub = _slice_normalize(sub, norm_floor)
        out[idx] = np.asarray(seg(sub, diameter), np.int32)
        del sub
    _release()
    return out


def _eq_diameters(lab_stack):
    d = []
    for sl in lab_stack:
        if sl.max() == 0:
            continue
        a = np.bincount(sl.ravel())[1:]
        a = a[a > 0]
        d.append(2.0 * np.sqrt(a / np.pi))
    return np.concatenate(d) if d else np.zeros(0)


# --------------------------------------------------------------------------- #
# chunked post-processing of the assembled label volume
# --------------------------------------------------------------------------- #
def _blocks(shape, b):
    for z0 in range(0, shape[0], b[0]):
        for y0 in range(0, shape[1], b[1]):
            for x0 in range(0, shape[2], b[2]):
                yield (slice(z0, min(z0 + b[0], shape[0])), slice(y0, min(y0 + b[1], shape[1])),
                       slice(x0, min(x0 + b[2], shape[2])))


def _label_stats(out, block):
    """Per label: count, coordinate sums and bounding box, in one chunked pass."""
    acc = {}
    for sl in _blocks(out.shape, block):
        b = np.asarray(out[sl])
        nz = np.nonzero(b)
        if nz[0].size == 0:
            continue
        labs = b[nz].astype(np.int64)
        u, inv = np.unique(labs, return_inverse=True)
        c = np.bincount(inv)
        off = (sl[0].start, sl[1].start, sl[2].start)
        sums = [np.bincount(inv, weights=nz[k] + off[k]) for k in range(3)]
        mins, maxs = [], []
        for k in range(3):
            mn = np.full(u.size, np.iinfo(np.int64).max)
            mx = np.full(u.size, -1)
            np.minimum.at(mn, inv, nz[k] + off[k])
            np.maximum.at(mx, inv, nz[k] + off[k])
            mins.append(mn); maxs.append(mx)
        for j, l in enumerate(u.tolist()):
            r = acc.get(l)
            if r is None:
                acc[l] = [int(c[j]), sums[0][j], sums[1][j], sums[2][j],
                          [int(mins[0][j]), int(mins[1][j]), int(mins[2][j])],
                          [int(maxs[0][j]), int(maxs[1][j]), int(maxs[2][j])]]
            else:
                r[0] += int(c[j]); r[1] += sums[0][j]; r[2] += sums[1][j]; r[3] += sums[2][j]
                for k in range(3):
                    r[4][k] = min(r[4][k], int(mins[k][j])); r[5][k] = max(r[5][k], int(maxs[k][j]))
    return acc


def _cleanup(out, block, min_volume, log):
    """Each label keeps its largest 26-connected component; drop labels below ``min_volume``."""
    from skimage.measure import label as cc_label
    stats = _label_stats(out, block)
    n_multi = n_small = 0
    for lab, r in stats.items():
        sl = tuple(slice(r[4][k], r[5][k] + 1) for k in range(3))
        sub = np.asarray(out[sl])
        mm = sub == lab
        cc = cc_label(mm, connectivity=3)
        changed = False
        keep = mm
        if cc.max() > 1:
            sizes = np.bincount(cc.ravel()); sizes[0] = 0
            keep = cc == sizes.argmax()
            n_multi += 1
            changed = True
        if min_volume and int(keep.sum()) < min_volume:
            keep = np.zeros_like(mm)
            n_small += 1
            changed = True
        if changed:
            sub[mm & ~keep] = 0
            out[sl] = sub
    log(f"cleanup: {n_multi} labels cut to their largest 26-connected component, "
        f"{n_small} below {min_volume} voxels removed")


def _relabel(out, block):
    mx = 0
    for sl in _blocks(out.shape, block):
        mx = max(mx, int(np.asarray(out[sl]).max()))
    if mx == 0:
        return 0
    present = np.zeros(mx + 1, bool)
    for sl in _blocks(out.shape, block):
        present[np.asarray(out[sl]).ravel()] = True
    present[0] = False
    lut = np.zeros(mx + 1, np.uint32)
    n = int(present.sum())
    lut[present] = np.arange(1, n + 1, dtype=np.uint32)
    for sl in _blocks(out.shape, block):
        b = np.asarray(out[sl])
        if b.any():
            out[sl] = lut[b]
    return n


def _table(out, block):
    stats = _label_stats(out, block)
    rows = []
    for lab in sorted(stats):
        r = stats[lab]
        n = r[0]
        rows.append((lab, r[1] / n, r[2] / n, r[3] / n, n,
                     r[4][0], r[4][1], r[4][2], r[5][0], r[5][1], r[5][2]))
    return pd.DataFrame(rows, columns=["label", "z", "y", "x", "volume_voxels",
                                       "z_min", "y_min", "x_min", "z_max", "y_max", "x_max"])


# --------------------------------------------------------------------------- #
# public driver
# --------------------------------------------------------------------------- #
@register_function(
    aliases=["三维细胞核分割", "光片三维分割", "分块三维分割", "多视角三维分割",
             "3D nuclei segmentation", "tiled 3D segmentation", "light-sheet nuclei",
             "out-of-core 3D segmentation", "orthogonal view Cellpose", "u-Segment3D tiled",
             "memory-limited volume segmentation", "seam-free 3D labels"],
    category="space",
    description="Tiled, memory-bounded 3-D instance segmentation of a (Z,Y,X) volume of any size "
                "(numpy / zarr / tif): per tile, resample to a near-isotropic working grid, run a "
                "2-D model (Cellpose, separate xy and side-view models allowed) over the xy, xz "
                "and yz slice stacks, aggregate the three 2-D label sets into 3-D instances "
                "(u-Segment3D indirect method), keep only objects whose centroid lies in the "
                "tile core (seam-free ownership), then chunked 26-connectivity cleanup and a "
                "per-object table. Writes to a zarr store; peak memory set by memory_mb.",
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "labels, table, info = ov.space.seg.segment_volume_multiview('volume.zarr', "
        "spacing=(dz, dy, dx), model='nuclei', out='results/labels.zarr', memory_mb=2048, "
        "return_info=True)",
        "ov.space.seg.segment_volume_multiview(vol, spacing=sp, model=xy_ckpt, "
        "side_model=xz_yz_ckpt, out='labels.zarr', n_threads=4)",
    ],
    related=["space.seg.aggregate_orthoviews", "space.seg.segment_nuclei", "space.seg.segment"],
)
def segment_volume_multiview(
    volume,
    spacing: Sequence[float],
    *,
    model: Union[str, Callable, object] = "nuclei",
    side_model: Union[str, Callable, object, None] = None,
    out=None,
    diameter: Optional[float] = None,
    memory_mb: Optional[float] = 2048,
    tile: Optional[int] = None,
    halo: Optional[int] = None,
    xy_bin: Optional[int] = None,
    z_up: Optional[float] = None,
    smooth: float = 1.0,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_volume: Optional[int] = None,
    normalize_percentiles: Sequence[float] = (1.0, 99.8),
    skip_empty: bool = True,
    slice_normalize: Union[bool, str] = True,
    diameter_quantile: float = 0.9,
    n_threads: Optional[int] = None,
    gpu: bool = False,
    batch_size: int = 2,
    max_tiles: Optional[int] = None,
    tile_indices: Union[Sequence[int], str, None] = None,
    resume: bool = False,
    relabel: bool = True,
    aggregate_kwargs: Optional[dict] = None,
    verbose=True,
    return_info: bool = False,
):
    """Segment a 3-D volume into instances, tile by tile, from three orthogonal 2-D views.

    Parameters
    ----------
    volume
        ``(Z, Y, X)`` single-channel array, zarr array / memmap, or a path to a
        zarr store or tif. Read tile by tile; never loaded whole.
    spacing
        Voxel size ``(dz, dy, dx)`` in one unit (e.g. um), from the metadata.
    model
        2-D model for the xy view: a Cellpose built-in name (``'nuclei'``,
        ``'cyto3'``), a path to a Cellpose checkpoint, a Cellpose model
        object, or a callable ``f(stack, diameter_px) -> labels`` that labels
        each 2-D slice of a ``(S, H, W)`` stack.
    side_model
        Same, for the xz and yz views (side views look different from xy
        under anisotropic optics; use a model trained on them when there is
        one). ``None`` = ``model``.
    out
        Path of a zarr store to write the ``uint32`` labels to (created; an
        existing store is replaced unless ``resume``), a writable array of the
        volume's shape, or ``None`` = an in-memory numpy array (only for
        volumes that fit in memory).
    diameter
        Typical object diameter (equatorial, i.e. the widest section) in the
        unit of ``spacing``. ``None`` (default): estimated with the xy model
        itself on the most structured xy slices of the most occupied windows:
        the ``diameter_quantile`` of the 2-D equivalent diameters of its masks
        (sections through the middle of an object are the widest; the median
        of all sections is biased low by caps), iterated to a fixed point --
        check it in ``info`` and pass it explicitly once known.
    memory_mb
        The memory limit of the process / container (e.g. 2048), as is: the
        tile size is chosen to stay well below it, with room for the
        interpreter, torch and the models already included -- do not subtract
        your own margin (smaller tiles only cost time). The high-water RSS of
        this process is logged per tile and returned as ``peak_rss_mb``.
        ``None`` = no limit (one tile if ``tile`` is not given).
    tile, halo
        Tile core size and halo in original xy voxels (z is tiled only when a
        full-depth column cannot fit). Default: halo = 2 diameters, tile from
        ``memory_mb``.
    xy_bin, z_up
        Working-grid factors; default from :func:`multiview_working_grid`.
    smooth
        Gaussian sigma (working voxels) applied on the working grid.
    flow_threshold, cellprob_threshold
        Cellpose mask thresholds for every 2-D pass.
    min_volume
        Objects with fewer voxels (original grid) are removed. Default: a
        quarter of the volume of an ellipsoid of the estimated diameter.
    normalize_percentiles
        The volume is scaled to [0, 1] once, from these percentiles of a
        strided whole-volume subsample, so a sparse tile is not stretched.
    skip_empty
        Skip z-planes of a tile, and slices of every view, that carry no
        signal (fewer voxels above the global Otsu level than a quarter of
        one object section).
    slice_normalize
        ``True`` (default): normalise each 2-D slice by its 1st / 99th
        percentiles as Cellpose does (how its models are trained), but with
        the upper level floored at the volume's foreground level, so that a
        slice that is mostly background is not stretched into spurious
        objects. ``'cellpose'``: Cellpose's own unfloored normalisation.
        ``False``: the globally scaled intensities.
    n_threads
        Torch / BLAS threads; default = CPUs available to the process.
    max_tiles, tile_indices
        Process only the first N tiles, or only these tile indices, or
        ``'densest'`` (the tile with most foreground: the slowest and the one
        with the highest peak memory) -- for a timing / memory dry run.
    resume
        Continue a run interrupted after some tiles (progress is kept in the
        zarr store's attributes).
    aggregate_kwargs
        Passed to :func:`aggregate_orthoviews`.

    Returns
    -------
    labels
        The label store / array (``uint32``, 0 = background).
    table
        ``pandas.DataFrame`` with ``label, z, y, x, volume_voxels`` (centroid
        in voxel coordinates) and the bounding box.
    info
        (``return_info=True``) parameters used, tile timings, peak RSS (MB).
    """
    log = _log_fn(verbose)
    t_start = time.time()
    vol = _open_volume(volume)
    if len(vol.shape) != 3:
        raise ValueError(f"volume must be (Z, Y, X); got shape {vol.shape}")
    Z, Y, X = (int(s) for s in vol.shape)
    dz, dy, dx = (float(s) for s in spacing)
    nth = _n_threads(n_threads)
    try:
        import torch
        torch.set_num_threads(nth)
        # CPU convolutions on dim (near-zero) slices hit denormal floats and run several
        # times slower; flushing them to zero does not change the result measurably
        torch.set_flush_denormal(True)
    except ImportError:  # pragma: no cover - custom segmenters need no torch
        pass

    if slice_normalize not in (True, False, "cellpose"):
        raise ValueError("slice_normalize must be True, False or 'cellpose'")
    view_kw = dict(gpu=gpu, flow_threshold=flow_threshold, normalize=(slice_normalize == "cellpose"),
                   cellprob_threshold=cellprob_threshold, batch_size=batch_size)
    seg_xy = _as_view(model, **view_kw)
    seg_side = _as_view(side_model, **view_kw) if side_model is not None else seg_xy

    # ---- global normalisation from a strided subsample
    step = max(1, int(round((Y * X / 2.5e5) ** 0.5)))
    zstep = max(1, Z // 64)
    sub = np.asarray(vol[::zstep, ::step, ::step]).astype(np.float32)
    p_lo, p_hi = (float(np.percentile(sub, q)) for q in normalize_percentiles)
    if p_hi <= p_lo:
        p_hi = p_lo + 1.0
    subn = np.clip((sub - p_lo) / (p_hi - p_lo), 0, 1)
    try:
        from skimage.filters import threshold_otsu
        sig_level = float(threshold_otsu(subn))
    except ValueError:
        sig_level = 0.5
    del sub
    nfloor = sig_level if slice_normalize is True else None
    log(f"input {vol.shape} spacing {tuple(spacing)}; normalisation {p_lo:.1f}..{p_hi:.1f}, "
        f"signal level {sig_level:.3f}; {nth} threads")

    def norm(a):
        return np.clip((np.asarray(a, np.float32) - p_lo) / (p_hi - p_lo), 0, 1)

    # ---- diameter (unit of spacing)
    dxy = math.sqrt(dy * dx)
    info_diam = {}
    if diameter is None:
        xb0, zu0, _ = multiview_working_grid(spacing)
        xb0 = xy_bin or xb0
        zu0 = z_up or zu0
        # most occupied sample windows from the subsample
        win = min(Y, X, 192)
        occ = (subn > sig_level)
        cand = []
        for yy in range(0, max(1, Y - win + 1), max(1, win // 2)):
            for xx in range(0, max(1, X - win + 1), max(1, win // 2)):
                o = occ[:, yy // step:(yy + win) // step, xx // step:(xx + win) // step].mean()
                cand.append((o, yy, xx))
        cand.sort(reverse=True)
        slices = []
        for _, yy, xx in cand[:3]:
            w = _to_working(norm(vol[:, yy:yy + win, xx:xx + win]), xb0, zu0, smooth)
            sd = w.reshape(w.shape[0], -1).std(1)
            for i in np.argsort(-sd)[:4]:
                slices.append(w[i])
        stack = np.stack(slices)
        d = float(getattr(seg_xy, "diam_labels", 30.0)) if hasattr(seg_xy, "diam_labels") else 30.0
        hist = []
        for _ in range(4):
            eq = _eq_diameters(_run_view(seg_xy, stack, d, norm_floor=nfloor))
            if eq.size < 3:
                break
            dn = float(np.quantile(eq, diameter_quantile))
            hist.append((round(d, 2), round(dn, 2), int(eq.size)))
            conv = abs(dn - d) / d < 0.05
            d = dn
            if conv:
                break
        diameter = d * dxy * xb0
        info_diam = {"estimate_history_px": hist, "estimate_grid_xy_bin": xb0}
        log(f"diameter estimated: {diameter:.3g} (spacing units) = {d:.2f} working px; history {hist}")

    xb, zu, wsp = multiview_working_grid(spacing, diameter)
    if xy_bin:
        xb = int(xy_bin)
    if z_up:
        zu = float(z_up)
    wsp = (dz / zu, dy * xb, dx * xb)
    diam_w = diameter / math.sqrt(wsp[1] * wsp[2])
    diam_xy_vox = diameter / dxy
    diam_z_vox = diameter / dz
    if halo is None:
        halo = int(math.ceil(2.0 * diam_xy_vox))
    halo_z = int(math.ceil(2.0 * max(diam_z_vox, 1.0)))
    if min_volume is None:
        min_volume = int(0.25 * (math.pi / 6.0) * diam_xy_vox ** 2 * max(diam_z_vox, 1.0))
    log(f"working grid: xy bin {xb}, z up {zu:g} -> voxel {tuple(round(s, 3) for s in wsp)}; "
        f"diameter {diam_w:.1f} working px; halo {halo} xy / {halo_z} z voxels; "
        f"min volume {min_volume} voxels")

    # ---- tile size from the memory budget
    # measured: ~110 B per working voxel for the three views + aggregation, plus ~700 MB
    # for the interpreter, torch and the model(s)
    per_wvox = 110.0
    base_mb = 700.0
    tile_z = Z
    if tile is None:
        if memory_mb is None:
            tile = max(Y, X)
        else:
            budget = max(64.0, float(memory_mb) - base_mb) * 2 ** 20 / per_wvox
            tile = None
            for tz in [Z] + [max(4 * halo_z, Z // k) for k in range(2, 64)]:
                wz = (min(Z, tz + 2 * halo_z) if tz < Z else Z) * zu
                side = math.sqrt(budget / max(wz, 1.0)) * xb      # window side, original voxels
                core = int(side - 2 * halo)
                if core >= max(2 * halo, 32):
                    tile, tile_z = core, tz
                    break
            if tile is None:
                raise MemoryError("memory_mb is too small for one tile of this volume")
            tile = int(min(tile, max(Y, X)))
    tile = int(tile)
    # balance tiles so the last one is not a sliver
    ny = max(1, math.ceil(Y / tile)); nx = max(1, math.ceil(X / tile)); nz = max(1, math.ceil(Z / tile_z))
    ty, tx, tz_ = math.ceil(Y / ny), math.ceil(X / nx), math.ceil(Z / nz)
    cores = [(z0, min(z0 + tz_, Z), y0, min(y0 + ty, Y), x0, min(x0 + tx, X))
             for z0 in range(0, Z, tz_) for y0 in range(0, Y, ty) for x0 in range(0, X, tx)]
    occ_tiles = []
    for (z0, z1, y0, y1, x0, x1) in cores:
        blk = subn[z0 // zstep:max(z0 // zstep + 1, z1 // zstep), y0 // step:max(y0 // step + 1, y1 // step),
                   x0 // step:max(x0 // step + 1, x1 // step)]
        occ_tiles.append(float((blk > sig_level).mean()) if blk.size else 0.0)
    del subn
    if isinstance(tile_indices, str):
        if tile_indices != "densest":
            raise ValueError("tile_indices must be a list of indices or 'densest'")
        tile_indices = [int(np.argmax(occ_tiles))]
    log(f"tiles: {len(cores)} ({nz} z x {ny} y x {nx} x), core {tz_}x{ty}x{tx}, "
        f"halo {halo_z}/{halo}/{halo}; foreground fraction per tile "
        f"{[round(o, 3) for o in occ_tiles]}. A tile keeps ('owns') only the objects whose "
        f"centroid lies in its core, so a sparse or border tile can own none.")

    # ---- output
    chunks = (min(Z, 32), min(Y, 128), min(X, 128))
    store = None
    done, next_label = [], 1
    if out is None:
        labels = np.zeros((Z, Y, X), np.uint32)
    elif isinstance(out, (str, os.PathLike)):
        import zarr
        store = str(out)
        if resume and os.path.exists(store):
            labels = zarr.open(store, mode="r+")
            done = list(labels.attrs.get("tiles_done", []))
            next_label = int(labels.attrs.get("next_label", 1))
            log(f"resume: {len(done)} tiles done, next label {next_label}")
        else:
            if os.path.exists(store):
                import shutil
                shutil.rmtree(store)
            labels = zarr.open(store, mode="w", shape=(Z, Y, X), chunks=chunks, dtype="uint32")
    else:
        labels = out

    agg_kw = dict(aggregate_kwargs or {})
    tile_log = []
    for ti, (z0, z1, y0, y1, x0, x1) in enumerate(cores):
        if max_tiles is not None and ti >= max_tiles:
            break
        if tile_indices is not None and ti not in set(tile_indices):
            continue
        if ti in done:
            continue
        t0 = time.time()
        rz0, rz1 = (max(0, z0 - halo_z), min(Z, z1 + halo_z)) if nz > 1 else (0, Z)
        ry0, ry1 = max(0, y0 - halo), min(Y, y1 + halo)
        rx0, rx1 = max(0, x0 - halo), min(X, x1 + halo)
        img = norm(vol[rz0:rz1, ry0:ry1, rx0:rx1])
        masks = np.zeros(img.shape, np.uint32)
        za, zb = 0, img.shape[0]
        if skip_empty:
            area = 0.25 * math.pi * (0.5 * diam_xy_vox) ** 2
            occ_z = (img > sig_level).reshape(img.shape[0], -1).sum(1) >= max(1.0, 0.25 * area)
            if occ_z.any():
                zs = np.nonzero(occ_z)[0]
                pad = int(math.ceil(max(diam_z_vox, 1.0)))
                za, zb = max(0, zs[0] - pad), min(img.shape[0], zs[-1] + pad + 1)
            else:
                za = zb = 0
        n_obj = 0
        rss_views = rss_agg = None
        if zb - za >= 2:
            w = _to_working(img[za:zb], xb, zu, smooth)
            del img
            # a slice with less signal than a quarter of one object section is not segmented
            # (per-slice normalisation would stretch its noise into objects)
            mp = 0.25 * math.pi * (0.5 * diam_w) ** 2 if skip_empty else 0
            views = [_run_view(seg_xy, w, diam_w, sig_level, mp, nfloor),
                     _run_view(seg_side, np.ascontiguousarray(w.transpose(1, 0, 2)), diam_w, sig_level, mp, nfloor),
                     _run_view(seg_side, np.ascontiguousarray(w.transpose(2, 0, 1)), diam_w, sig_level, mp, nfloor)]
            del w
            rss_views = _peak_rss_mb()
            seg = aggregate_orthoviews(views.pop(0), views.pop(0), views.pop(0), **agg_kw)
            rss_agg = _peak_rss_mb()
            masks[za:zb] = _from_working(seg, (zb - za,) + masks.shape[1:], xb, zu)
            del seg
        else:
            del img
        # ownership by centroid in the core (per-plane sums: no whole-tile coordinate arrays)
        n_own = 0
        K = int(masks.max())
        if K > 0:
            cnt = np.zeros(K + 1); sz = np.zeros(K + 1); sy = np.zeros(K + 1); sx = np.zeros(K + 1)
            for k in range(masks.shape[0]):
                m = masks[k]
                ys, xs = np.nonzero(m)
                if ys.size == 0:
                    continue
                lv = m[ys, xs]
                c = np.bincount(lv, minlength=K + 1)
                cnt += c
                sz += c * k
                sy += np.bincount(lv, weights=ys, minlength=K + 1)
                sx += np.bincount(lv, weights=xs, minlength=K + 1)
            present = cnt > 0
            present[0] = False
            n_obj = int(present.sum())
            safe = np.maximum(cnt, 1)
            cz = sz / safe + rz0; cy = sy / safe + ry0; cx = sx / safe + rx0
            keep = (present & (cz >= z0) & (cz < z1) & (cy >= y0) & (cy < y1) & (cx >= x0) & (cx < x1)
                    & (cnt >= max(1, min_volume)))
            lut = np.zeros(K + 1, np.uint32)
            n_own = int(keep.sum())
            lut[keep] = np.arange(next_label, next_label + n_own, dtype=np.uint32)
            next_label += n_own
            # write slab by slab: core voxels unconditionally, halo voxels only where empty
            cy0, cy1, cx0, cx1 = y0 - ry0, y1 - ry0, x0 - rx0, x1 - rx0
            for s0 in range(0, masks.shape[0], 16):
                s1 = min(s0 + 16, masks.shape[0])
                o = lut[masks[s0:s1]]
                if not o.any():
                    continue
                region = np.asarray(labels[rz0 + s0:rz0 + s1, ry0:ry1, rx0:rx1])
                wr = (o > 0) & (region == 0)
                zlo, zhi = max(s0, z0 - rz0), min(s1, z1 - rz0)
                if zhi > zlo:
                    oc = o[zlo - s0:zhi - s0, cy0:cy1, cx0:cx1]
                    wr[zlo - s0:zhi - s0, cy0:cy1, cx0:cx1] |= oc > 0
                region[wr] = o[wr]
                labels[rz0 + s0:rz0 + s1, ry0:ry1, rx0:rx1] = region
                del o, region, wr
        del masks
        _release()
        done.append(ti)
        if store is not None:
            labels.attrs["tiles_done"] = done
            labels.attrs["next_label"] = next_label
        dt = time.time() - t0
        tile_log.append({"tile": ti, "core": (z0, z1, y0, y1, x0, x1), "z_run": (za + rz0, zb + rz0),
                         "foreground_fraction": round(occ_tiles[ti], 4),
                         "objects": n_obj, "owned": n_own, "seconds": round(dt, 1),
                         "peak_rss_after_views_mb": rss_views, "peak_rss_after_aggregate_mb": rss_agg})
        log(f"tile {ti + 1}/{len(cores)} z {z0}:{z1} y {y0}:{y1} x {x0}:{x1} "
            f"(foreground {occ_tiles[ti]:.3f}): {n_obj} objects, "
            f"{n_own} owned, total {next_label - 1}; {dt:.0f}s; peak RSS {_peak_rss_mb():.0f} MB "
            f"(views {rss_views or 0:.0f}, aggregate {rss_agg or 0:.0f})")

    block = (min(Z, 128), min(Y, 128), min(X, 128))
    _cleanup(labels, block, min_volume, log)
    if relabel:
        _relabel(labels, block)
    table = _table(labels, block)
    if store is not None:
        labels.attrs["tiles_done"] = done
        labels.attrs["complete"] = bool(len(set(done)) >= len(cores))
    peak = _peak_rss_mb()
    log(f"done: {len(table)} objects in {time.time() - t_start:.0f}s; peak RSS {peak:.0f} MB")
    if not return_info:
        return labels, table
    info = {
        "diameter": float(diameter), "diameter_working_px": float(diam_w), **info_diam,
        "xy_bin": xb, "z_up": zu, "working_spacing": wsp, "tile": tile, "tile_z": tile_z,
        "halo": halo, "halo_z": halo_z, "n_tiles": len(cores), "min_volume": min_volume,
        "normalize": (p_lo, p_hi), "signal_level": sig_level, "tiles": tile_log,
        "seconds": round(time.time() - t_start, 1), "peak_rss_mb": peak, "n_threads": nth,
    }
    return labels, table, info
