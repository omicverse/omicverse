"""``ov.space.seg.detect_cilia`` on drawn 3-D scenes with known cilia.

Rods (axoneme marker) and puncta (basal-body marker) are drawn in micrometre
coordinates and blurred with an anisotropic PSF; cilia have a punctum at one
end, distractors do not (or are pieces of a longer fibre, or carry the basal
marker along their length like a centriole pair).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from omicverse.space.seg import detect_cilia


def _scene(spacing, shape, rods, puncta, seed=0, noise=0.01, haze=0.05):
    """rods: list of (p0_um, p1_um, brightness); puncta: list of (p_um, brightness).
    Returns (axoneme, basal) float32 stacks."""
    rng = np.random.default_rng(seed)
    um = np.asarray(spacing, float)
    ax = np.zeros(shape, float)
    bb = np.zeros(shape, float)
    for p0, p1, b in rods:
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        n = int(np.linalg.norm(p1 - p0) / (0.25 * um[1:].min())) + 2
        for s in np.linspace(0, 1, n):
            v = np.round((p0 + s * (p1 - p0)) / um).astype(int)
            if np.all(v >= 0) and np.all(v < shape):
                ax[tuple(v)] = max(ax[tuple(v)], b)
    for p, b in puncta:
        v = np.round(np.asarray(p, float) / um).astype(int)
        bb[tuple(v)] = b
    psf = (0.35 / um[0], 0.12 / um[1], 0.12 / um[2])
    ax = ndi.gaussian_filter(ax, psf) * 40.0
    bb = ndi.gaussian_filter(bb, (0.4 / um[0], 0.2 / um[1], 0.2 / um[2])) * 40.0
    yy, xx = np.mgrid[:shape[1], :shape[2]]
    bg = haze * (1 + np.sin(yy / 40.0) * np.cos(xx / 55.0))
    ax = ax + bg[None] + noise * rng.standard_normal(shape)
    bb = bb + 0.5 * bg[None] + noise * rng.standard_normal(shape)
    return ax.astype(np.float32), bb.astype(np.float32)


def _match(df, truth_centroids_px, radius_px):
    got = list(zip(df["centroid_y"], df["centroid_x"]))
    hit = 0
    for ty, tx in truth_centroids_px:
        if any(np.hypot(gy - ty, gx - tx) <= radius_px for gy, gx in got):
            hit += 1
    return hit


ISO = (0.2, 0.1, 0.1)            # near-isotropic confocal stack
ISO_SHAPE = (30, 300, 300)


def _iso_scene(seed=0):
    z = 3.0
    cilia = [  # (base_um, tip_um): base at the basal body
        ((z, 5.0, 5.0), (z + 0.6, 7.5, 7.0)),
        ((z, 5.0, 20.0), (z + 0.4, 8.0, 21.5)),
        ((z, 18.0, 6.0), (z + 0.8, 21.0, 5.0)),
        ((z, 14.0, 14.0), (z + 0.2, 15.0, 17.5)),
    ]
    rods = [(b, t, 1.0) for b, t in cilia]
    puncta = [(b, 1.0) for b, _ in cilia]
    # distractors: a rod with no basal body; a long fibre passing a punctum
    rods.append(((z, 24.0, 18.0), (z, 26.5, 20.0), 1.0))
    rods.append(((z + 1.0, 10.0, 2.0), (z + 1.0, 11.0, 28.0), 1.0))
    puncta.append(((z + 1.0, 10.4, 12.0), 1.0))
    # centriole pair: short rod with basal marker along all of it
    rods.append(((z, 26.0, 4.0), (z, 26.0, 5.6), 1.0))
    puncta += [((z, 26.0, 4.0 + 0.4 * k), 1.0) for k in range(5)]
    ax, bb = _scene(ISO, ISO_SHAPE, rods, puncta, seed=seed)
    truth = [((b[1] + t[1]) / 2 / ISO[1], (b[2] + t[2]) / 2 / ISO[2]) for b, t in cilia]
    return ax, bb, cilia, truth


def test_detects_cilia_and_rejects_distractors():
    ax, bb, cilia, truth = _iso_scene()
    df, rej = detect_cilia(ax, bb, ISO, return_rejected=True)
    assert _match(df, truth, 10) == len(cilia)
    assert len(df) == len(cilia)
    assert len(rej) and "reason" in rej.columns


def test_base_is_at_the_basal_body_and_length_is_3d():
    ax, bb, cilia, truth = _iso_scene(seed=1)
    df = detect_cilia(ax, bb, ISO)
    um = np.asarray(ISO)
    for b, t in cilia:
        b, t = np.asarray(b), np.asarray(t)
        cy, cx = (b[1] + t[1]) / 2 / um[1], (b[2] + t[2]) / 2 / um[2]
        row = df.iloc[int(np.argmin(np.hypot(df.centroid_y - cy, df.centroid_x - cx)))]
        base = np.array([row.base_z, row.base_y, row.base_x]) * um
        tip = np.array([row.tip_z, row.tip_y, row.tip_x]) * um
        assert np.linalg.norm(base - b) < np.linalg.norm(base - t)      # oriented
        assert np.linalg.norm(base - b) < 0.6 and np.linalg.norm(tip - t) < 0.6
        assert row.length_um == pytest.approx(np.linalg.norm(t - b), abs=0.6)


def test_coarse_z_stack():
    sp = (0.7, 0.07, 0.07)
    shape = (9, 300, 300)
    z = 2.8
    cilia = [((z, 4.0, 4.0), (z + 0.7, 6.5, 6.0)), ((z, 4.0, 14.0), (z, 7.0, 15.0)),
             ((z, 14.0, 5.0), (z + 0.7, 16.0, 8.0))]
    rods = [(b, t, 1.0) for b, t in cilia] + [((z, 15.0, 15.0), (z, 17.5, 17.0), 1.0)]
    puncta = [(b, 1.0) for b, _ in cilia]
    ax, bb = _scene(sp, shape, rods, puncta, seed=2)
    df = detect_cilia(ax, bb, sp)
    truth = [((b[1] + t[1]) / 2 / sp[1], (b[2] + t[2]) / 2 / sp[2]) for b, t in cilia]
    assert len(df) == 3 and _match(df, truth, 15) == 3


def test_one_cilium_per_basal_body_and_empty_input():
    ax, bb, cilia, truth = _iso_scene(seed=3)
    df = detect_cilia(ax, bb, ISO)
    assert df["cilium_id"].tolist() == list(range(1, len(df) + 1))
    empty = detect_cilia(np.zeros((5, 40, 40), np.float32), np.zeros((5, 40, 40), np.float32),
                         ISO)
    assert len(empty) == 0
    with pytest.raises(ValueError):
        detect_cilia(np.zeros((5, 40, 40)), np.zeros((5, 40, 41)), ISO)
