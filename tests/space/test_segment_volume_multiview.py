"""Tests for ov.space.seg.segment_volume_multiview / aggregate_orthoviews.

Synthetic anisotropic volumes of ellipsoidal nuclei; a classical per-slice
2-D segmenter stands in for Cellpose so the tests need no model weights.
"""
import numpy as np
import pytest
from scipy import ndimage as ndi

from omicverse.space.seg import (aggregate_orthoviews, multiview_working_grid,
                                 segment_volume_multiview)


def _phantom(shape=(24, 160, 160), n=40, r_xy=7.0, r_z=2.2, seed=0, noise=0.03):
    """Ellipsoids on a jittered lattice (non-overlapping), anisotropic (z coarse)."""
    rng = np.random.default_rng(seed)
    Z, Y, X = shape
    gt = np.zeros(shape, np.int32)
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    centres = []
    step = int(2 * r_xy + 4)
    for cy in range(step // 2 + 2, Y - step // 2, step):
        for cx in range(step // 2 + 2, X - step // 2, step):
            cz = rng.uniform(r_z + 2, Z - r_z - 2)
            centres.append((cz, cy + rng.uniform(-1.5, 1.5), cx + rng.uniform(-1.5, 1.5)))
    rng.shuffle(centres)
    centres = centres[:n]
    for i, (cz, cy, cx) in enumerate(centres, 1):
        m = ((zz - cz) / r_z) ** 2 + ((yy - cy) / r_xy) ** 2 + ((xx - cx) / r_xy) ** 2 <= 1
        gt[m & (gt == 0)] = i
    img = (gt > 0).astype(np.float32)
    img = ndi.gaussian_filter(img, (0.6, 1.0, 1.0))
    img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    return (np.clip(img, 0, None) * 1000).astype(np.uint16), gt


def _slice_seg(stack, diameter):
    """Per-slice Otsu + distance-transform watershed; one label set per slice."""
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    out = np.zeros(stack.shape, np.int32)
    glob = stack[stack > 0]
    t = threshold_otsu(glob) if glob.size and np.ptp(glob) > 0 else 0.5
    for i, sl in enumerate(stack):
        fg = sl > t
        if fg.sum() < 3:
            continue
        d = ndi.distance_transform_edt(fg)
        pk = peak_local_max(d, min_distance=max(2, int(diameter / 3)), labels=ndi.label(fg)[0])
        mk = np.zeros(sl.shape, np.int32)
        mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
        out[i] = watershed(-d, mk, mask=fg)
    return out


def _match(pred, gt, thr=0.5):
    """Number of one-to-one IoU>thr matches."""
    p = pred.ravel().astype(np.int64)
    g = gt.ravel().astype(np.int64)
    m = (p > 0) & (g > 0)
    pairs, cnt = np.unique(np.stack([p[m], g[m]]), axis=1, return_counts=True)
    ps = np.bincount(p)
    gs = np.bincount(g)
    iou = cnt / (ps[pairs[0]] + gs[pairs[1]] - cnt)
    return int((iou > thr).sum())


def test_working_grid_geometric_mean():
    xb, zu, wsp = multiview_working_grid((2.5, 0.43, 0.43))
    assert xb == 2 and zu == 2.0
    assert wsp == pytest.approx((1.25, 0.86, 0.86))
    assert multiview_working_grid((1.0, 1.0, 1.0))[:2] == (1, 1.0)
    # a small object limits the xy binning
    assert multiview_working_grid((4.0, 0.2, 0.2), diameter=2.0)[0] == 1


def test_aggregate_separates_touching_and_keeps_tall():
    Z, Y, X = 30, 40, 60
    gt = np.zeros((Z, Y, X), np.int32)
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    gt[((zz - 15) / 10) ** 2 + ((yy - 20) / 8) ** 2 + ((xx - 18) / 8) ** 2 <= 1] = 1   # tall
    gt[((zz - 15) / 6) ** 2 + ((yy - 20) / 8) ** 2 + ((xx - 40) / 8) ** 2 <= 1] = 2   # touching-ish

    def views(v):
        return v.copy(), v.transpose(1, 0, 2).copy(), v.transpose(2, 0, 1).copy()
    seg = aggregate_orthoviews(*views(gt))
    assert len(np.unique(seg)) - 1 == 2
    assert _match(seg, gt) == 2


def test_tiled_equals_reference_and_seam_free():
    vol, gt = _phantom()
    sp = (2.5, 0.8, 0.8)
    one, t1 = segment_volume_multiview(vol, sp, model=_slice_seg, diameter=11.0,
                                       memory_mb=None, verbose=False)
    tiled, t2, info = segment_volume_multiview(vol, sp, model=_slice_seg, diameter=11.0,
                                               tile=48, verbose=False, return_info=True)
    assert info["n_tiles"] > 4
    n_gt = len(np.unique(gt)) - 1
    m_one, m_tiled = _match(one, gt), _match(tiled, gt)
    assert m_one >= 0.9 * n_gt
    assert m_tiled >= m_one - 1                       # tiling costs (almost) nothing
    assert len(t2) <= n_gt + 2                        # no duplicates at seams
    # labels are 26-connected, table agrees with the volume
    from skimage.measure import label as cc
    for lab in t2["label"][:10]:
        assert cc(tiled == lab, connectivity=3).max() == 1
    vv = np.bincount(tiled.ravel())
    assert np.all(vv[t2["label"].to_numpy()] == t2["volume_voxels"].to_numpy())
    assert set(np.unique(tiled)) - {0} == set(t2["label"])


def test_zarr_out_resume_and_memory_budget(tmp_path):
    zarr = pytest.importorskip("zarr")
    vol, gt = _phantom(seed=1)
    src = zarr.open(str(tmp_path / "vol.zarr"), mode="w", shape=vol.shape, chunks=(8, 64, 64),
                    dtype="uint16")
    src[:] = vol
    sp = (2.5, 0.8, 0.8)
    path = str(tmp_path / "labels.zarr")
    lab, tab = segment_volume_multiview(str(tmp_path / "vol.zarr"), sp, model=_slice_seg,
                                        diameter=11.0, tile=64, out=path, max_tiles=2,
                                        verbose=False)
    assert not lab.attrs["complete"]
    lab, tab, info = segment_volume_multiview(str(tmp_path / "vol.zarr"), sp, model=_slice_seg,
                                              diameter=11.0, tile=64, out=path, resume=True,
                                              verbose=False, return_info=True)
    assert lab.attrs["complete"]
    arr = np.asarray(zarr.open(path, mode="r")[:])
    assert arr.dtype == np.uint32 and arr.shape == vol.shape
    assert _match(arr, gt) >= 0.85 * (len(np.unique(gt)) - 1)
    # memory budget sets the tile size: a small budget means more, smaller tiles
    _, _, small = segment_volume_multiview(vol, sp, model=_slice_seg, diameter=11.0,
                                           memory_mb=720, verbose=False, return_info=True,
                                           max_tiles=1)
    _, _, big = segment_volume_multiview(vol, sp, model=_slice_seg, diameter=11.0,
                                         memory_mb=4000, verbose=False, return_info=True,
                                         max_tiles=1)
    assert small["n_tiles"] > big["n_tiles"]


def test_diameter_estimated_from_segmenter():
    vol, gt = _phantom(seed=2)
    _, tab, info = segment_volume_multiview(vol, (2.5, 0.8, 0.8), model=_slice_seg,
                                            memory_mb=None, verbose=False, return_info=True)
    # true xy diameter 14 px * 0.8 = 11.2; section medians are a bit smaller
    assert 6.0 < info["diameter"] < 13.0
    assert len(tab) > 0.8 * (len(np.unique(gt)) - 1)
