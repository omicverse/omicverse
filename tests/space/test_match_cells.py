"""``ov.space.geom.match_cells`` on synthetic two-round imaging of one section.

Ground truth is built, not borrowed: cells are placed with a minimum spacing,
the second round sees them through a similarity transform, a smooth
non-rigid deformation, centroid noise, partial overlap, and independent
segmentation dropout plus spurious detections on both sides.
"""
from __future__ import annotations

import numpy as np
import pytest

from omicverse.space._cellmatch import hematoxylin_channel, match_cells


def _cells(n, size, min_dist, rng):
    pts = []
    while len(pts) < n:
        p = rng.uniform(0, size, 2)
        if all(np.hypot(*(p - q)) >= min_dist for q in pts):
            pts.append(p)
    return np.array(pts)


def _scene(seed, *, theta=17.0, scale=1.04, flip=False, warp=6.0, noise=1.0,
           drop=0.08, extra=0.08):
    rng = np.random.default_rng(seed)
    tissue = _cells(700, 520, 12.0, rng)                       # the physical cells
    # round 1 (fixed): a 340 x 340 field
    in_f = np.all((tissue > 60) & (tissue < 400), axis=1)
    # round 2 (moving): an offset, larger field -> partial overlap
    in_m = np.all((tissue > 120) & (tissue < 520), axis=1)
    t = np.deg2rad(theta)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) * scale
    if flip:
        R = R @ np.diag([1.0, -1.0])
    def to_moving(p):
        q = p @ R.T + np.array([-40.0, 25.0])
        # smooth deformation: low-frequency sinusoidal field
        q = q + warp * np.c_[np.sin(q[:, 1] / 90.0), np.cos(q[:, 0] / 110.0)]
        return q + rng.normal(0, noise, q.shape)
    keep_f = in_f & (rng.random(len(tissue)) > drop)
    keep_m = in_m & (rng.random(len(tissue)) > drop)
    fixed = tissue[keep_f] + rng.normal(0, noise, (keep_f.sum(), 2))
    moving = to_moving(tissue[keep_m])
    # spurious detections on each side
    fixed = np.vstack([fixed, rng.uniform(60, 400, (int(extra * len(fixed)), 2))])
    moving = np.vstack([moving, to_moving(rng.uniform(120, 520, (int(extra * len(moving)), 2)))])
    f_ids = np.r_[np.where(keep_f)[0], -1 - np.arange(len(fixed) - keep_f.sum())]
    m_ids = np.r_[np.where(keep_m)[0], -100000 - np.arange(len(moving) - keep_m.sum())]
    truth = set(np.where(keep_f & keep_m)[0])
    return fixed, moving, f_ids, m_ids, truth


def _prf(pairs, truth):
    correct = int((pairs.moving_id == pairs.fixed_id).sum())
    precision = correct / max(len(pairs), 1)
    recall = correct / max(len(truth), 1)
    return precision, recall


@pytest.mark.parametrize("seed,kw", [
    (0, {}),
    (1, {"theta": -63.0, "scale": 0.93}),
    (2, {"flip": True, "theta": 140.0}),
    (3, {"warp": 10.0, "noise": 1.5}),
])
def test_recovers_correspondence(seed, kw):
    fixed, moving, f_ids, m_ids, truth = _scene(seed, **kw)
    pairs = match_cells(fixed, moving, fixed_ids=f_ids, moving_ids=m_ids)
    p, r = _prf(pairs, truth)
    assert p >= 0.95 and r >= 0.90, (p, r)


def test_one_to_one_and_capped():
    fixed, moving, f_ids, m_ids, _ = _scene(4)
    pairs, info = match_cells(fixed, moving, fixed_ids=f_ids, moving_ids=m_ids,
                              return_info=True)
    assert pairs.moving_id.is_unique and pairs.fixed_id.is_unique
    assert (pairs.distance <= info["max_distance"]).all()


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        match_cells(np.zeros((5, 3)), np.zeros((5, 2)))


def test_hematoxylin_channel_marks_nuclei():
    img = np.full((20, 20, 3), 235, dtype=np.uint8)           # pale background
    img[5:10, 5:10] = (60, 40, 140)                           # haematoxylin-blue nucleus
    h = hematoxylin_channel(img)
    assert h.shape == (20, 20)
    assert h[7, 7] > 0.8 and h[15, 15] < 0.2
