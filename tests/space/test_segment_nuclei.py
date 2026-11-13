"""``ov.space.seg.segment_nuclei`` on drawn images with known nuclei.

Every scene is synthetic: nuclei are ellipses / discs with a soft edge, a
known focal plane in z, Poisson-like noise and a background offset. The
checks are instance-level (matched by IoU), as a nucleus mask is scored.
"""
from __future__ import annotations

import numpy as np
import pytest

from omicverse.space.seg import segment_nuclei


def _iou_match(truth, labels, thr=0.5):
    """Number of truth objects matched 1:1 at IoU >= thr, and unmatched labels."""
    matched_t, matched_l = set(), set()
    for t in np.unique(truth)[1:]:
        tm = truth == t
        vals, counts = np.unique(labels[tm], return_counts=True)
        for v, c in zip(vals, counts):
            if v == 0 or v in matched_l:
                continue
            iou = c / (tm.sum() + (labels == v).sum() - c)
            if iou >= thr:
                matched_t.add(int(t))
                matched_l.add(int(v))
                break
    unmatched = set(np.unique(labels)[1:].tolist()) - matched_l
    return len(matched_t), unmatched


def _ellipse(H, W, cy, cx, ry, rx, angle=0.0):
    yy, xx = np.mgrid[:H, :W].astype(float)
    c, s = np.cos(angle), np.sin(angle)
    u = (xx - cx) * c + (yy - cy) * s
    v = -(xx - cx) * s + (yy - cy) * c
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def _render(H, W, objects, z=1, noise=0.05, seed=0, bg=0.1, axial_sd=1.5):
    """objects: list of (mask, brightness, focal_z or callable(mask)->zmap).

    Returns image (Y, X) if z == 1 else (Z, Y, X), and the truth label image."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    truth = np.zeros((H, W), np.int32)
    vol = np.zeros((z, H, W), float)
    for i, (mask, bright, fz) in enumerate(objects, start=1):
        soft = ndi.gaussian_filter(mask.astype(float), 1.5)
        if z == 1:
            vol[0] = np.maximum(vol[0], bright * soft)
        else:
            zmap = fz(mask) if callable(fz) else np.full((H, W), float(fz))
            zz = np.arange(z)[:, None, None]
            prof = np.exp(-0.5 * ((zz - zmap[None]) / axial_sd) ** 2)
            vol = np.maximum(vol, bright * soft[None] * prof)
        truth[mask & (truth == 0)] = i
    vol = vol + bg
    vol = vol + noise * rng.standard_normal(vol.shape)
    vol = np.clip(vol, 0, None).astype(np.float32)
    return (vol[0] if z == 1 else vol), truth


def test_touching_nuclei_of_different_size_are_split():
    H, W = 260, 360
    big = _ellipse(H, W, 130, 120, 60, 55)
    small = _ellipse(H, W, 130, 205, 32, 32)          # pressed against the big one
    lone = _ellipse(H, W, 60, 300, 40, 45)
    img, truth = _render(H, W, [(big, 1.0, 0), (small, 0.9, 0), (lone, 1.0, 0)])
    labels = segment_nuclei(img)
    n_match, unmatched = _iou_match(truth, labels)
    assert n_match == 3 and not unmatched
    assert labels.dtype == np.int32 and labels.shape == (H, W)


def test_elongated_nucleus_with_waist_is_not_oversplit():
    H, W = 240, 360
    # a long nucleus made of two overlapping ellipses: a shallow waist, one object
    m = _ellipse(H, W, 120, 140, 45, 70) | _ellipse(H, W, 120, 220, 42, 70)
    other = _ellipse(H, W, 50, 60, 35, 35)
    img, truth = _render(H, W, [(m, 1.0, 0), (other, 1.0, 0)], seed=1)
    labels = segment_nuclei(img)
    n_match, unmatched = _iou_match(truth, labels)
    assert n_match == 2 and not unmatched


def test_dim_and_bright_nuclei_both_found():
    H, W = 240, 400
    a = _ellipse(H, W, 120, 90, 45, 50)
    b = _ellipse(H, W, 120, 300, 45, 50)
    img, truth = _render(H, W, [(a, 1.0, 0), (b, 0.35, 0)], noise=0.03, seed=2)
    labels = segment_nuclei(img)
    n_match, unmatched = _iou_match(truth, labels)
    assert n_match == 2 and not unmatched


def _field(H, W, centres, ry=44, rx=48):
    return [_ellipse(H, W, cy, cx, ry, rx) for cy, cx in centres]


def test_projection_overlap_at_different_depths_is_split():
    """One convex 2-D blob that is two nuclei focused at different z: no waist,
    so only the focal depth can separate them."""
    H, W, Z = 400, 520, 11
    yy, xx = np.mgrid[:H, :W]
    blob = _ellipse(H, W, 200, 260, 62, 125)
    lo, hi = blob & (xx < 260), blob & (xx >= 260)
    lone = _field(H, W, [(60, 70), (60, 450), (340, 70), (340, 450)])
    objs = [(lo, 1.0, 2.0), (hi, 1.0, 7.0)] + [(m, 1.0, 3.0 + i) for i, m in enumerate(lone)]
    vol, truth = _render(H, W, objs, z=Z, seed=3)
    flat = segment_nuclei(vol.max(0))                  # the projection cannot do it
    assert flat.max() == 5 and flat[200, 200] == flat[200, 320]   # the pair is one label
    labels, table, info = segment_nuclei(vol, return_info=True)
    n_match, unmatched = _iou_match(truth, labels)
    assert n_match == 6 and not unmatched
    assert info["n_depth_split"] == 1
    assert {"label", "area", "centroid_y", "centroid_x", "focal_depth"} <= set(table.columns)
    assert len(table) == info["n_nuclei"] == labels.max()


def test_tilted_single_nucleus_is_not_depth_split():
    """A nucleus whose focal plane changes smoothly across it (a ramp from z 2 to
    z 7) is one nucleus, even though the depth gap between its halves is large."""
    H, W, Z = 400, 520, 11
    xx = np.mgrid[:H, :W][1].astype(float)
    tilted = _ellipse(H, W, 200, 260, 62, 125)
    ramp = lambda m: 2.0 + 5.0 * (xx - 135) / 250.0
    lone = _field(H, W, [(60, 70), (60, 450), (340, 70), (340, 450)])
    objs = [(tilted, 1.0, ramp)] + [(m, 1.0, 3.0 + i) for i, m in enumerate(lone)]
    vol, truth = _render(H, W, objs, z=Z, seed=4)
    labels, _, info = segment_nuclei(vol, return_info=True)
    n_match, unmatched = _iou_match(truth, labels)
    assert n_match == 5 and not unmatched
    assert info["n_depth_split"] == 0


def test_scale_is_estimated_from_the_image():
    """The same scene at twice the resolution gives the same instances."""
    H, W = 200, 300
    objs = [(_ellipse(H, W, 100, 90, 45, 45), 1.0, 0),
            (_ellipse(H, W, 100, 165, 30, 30), 1.0, 0),
            (_ellipse(H, W, 50, 250, 30, 35), 1.0, 0)]
    img, truth = _render(H, W, objs, seed=5)
    big = np.kron(img, np.ones((2, 2), np.float32))
    tbig = np.kron(truth, np.ones((2, 2), np.int32))
    l1, _, i1 = segment_nuclei(img, return_info=True)
    l2, _, i2 = segment_nuclei(big, return_info=True)
    assert _iou_match(truth, l1) == (3, set())
    assert _iou_match(tbig, l2) == (3, set())
    assert i2["radius"] == pytest.approx(2 * i1["radius"], rel=0.15)


def test_empty_image_and_bad_shape():
    labels = segment_nuclei(np.zeros((64, 64), np.float32))
    assert labels.max() == 0
    with pytest.raises(ValueError):
        segment_nuclei(np.zeros((2, 2, 2, 2)))
