"""Synthetic tests for ov.space.track (no Cellpose / Trackastra needed)."""
import numpy as np
import pandas as pd
import pytest

from omicverse.space.track import (close_gaps, detect_mitoses, lineage_tables,
                                   mitosis_signal, refine_divisions, track_cells)


def _disc(lab, img, cx, cy, r, value, lid, bright=None):
    yy, xx = np.ogrid[:lab.shape[0], :lab.shape[1]]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    lab[m] = lid
    img[m] = value if bright is None else bright


def _movie(T=60, H=120, W=160, div_frame=30, drop=None):
    """Cell A divides at div_frame (bright for 4 frames before); cell B drifts out
    of the right border; cell C is present throughout. `drop` = frames where C is
    missed by the segmentation."""
    rng = np.random.default_rng(0)
    labels = np.zeros((T, H, W), np.int32)
    movie = np.full((T, H, W), 100.0) + rng.normal(0, 2, (T, H, W))
    for t in range(T):
        # A then daughters
        if t <= div_frame:
            bright = 220 if t > div_frame - 4 else None
            _disc(labels[t], movie[t], 40, 40, 7, 130, 1, bright)
        else:
            d = min(12, 3 + (t - div_frame))
            _disc(labels[t], movie[t], 40 - d, 40, 6, 130, 1)
            _disc(labels[t], movie[t], 40 + d, 40, 6, 130, 2)
        # B moves right and leaves
        bx = 100 + 2 * t
        if bx < W - 3:
            _disc(labels[t], movie[t], bx, 90, 7, 130, 3)
        # C static
        if drop is None or t not in drop:
            _disc(labels[t], movie[t], 30, 95, 7, 130, 4)
    return movie.astype(np.float32), labels


def test_overlap_tracking_and_lineage():
    movie, labels = _movie()
    det, tl = track_cells(labels, movie, method="overlap", gate=10)
    res = lineage_tables(det, tl, n_frames=movie.shape[0], frame_shape=movie.shape[1:])
    dv = res["divisions"]
    assert len(dv) == 1
    assert dv.loc[0, "frame"] == 30 and dv.loc[0, "generation"] == 0
    fates = res["cells"]["fate"].value_counts().to_dict()
    assert fates.get("divided") == 1
    assert fates.get("left_field") == 1
    assert fates.get("still_present") == 3  # two daughters + C
    f = res["founders"].set_index("track_id")
    assert sorted(f["divisions"].tolist()) == [0, 0, 1]
    go = res["generation_outcome"]
    assert go.loc[1, "still_present"] == 2
    gw = res["generation_window"]
    assert gw.values.sum() == 1


def test_close_gaps_rejoins_missed_detections():
    movie, labels = _movie(drop={20, 21})
    det, tl = track_cells(labels, movie, method="overlap", gate=10)
    n_before = len(tl)
    det2, tl2 = close_gaps(det, tl, frame_shape=movie.shape[1:], max_gap=5, gate=10)
    assert len(tl2) == n_before - 1
    res = lineage_tables(det2, tl2, n_frames=movie.shape[0], frame_shape=movie.shape[1:])
    assert (res["cells"]["start"] == 0).sum() == 3


def test_mitosis_signal_and_refinement():
    movie, labels = _movie(div_frame=30)
    sig = mitosis_signal(movie, cell_diameter=14, downsample=2)
    ev = detect_mitoses(sig, cell_diameter=14, downsample=2, lag=4)
    assert len(ev) >= 1
    best = ev.sort_values("score").iloc[-1]
    assert abs(best["frame"] - 28) <= 3 and abs(best["x"] - 40) < 8
    det, tl = track_cells(labels, movie, method="overlap", gate=10)
    div = refine_divisions(det, tl, sig, cell_diameter=14, downsample=2, window=6, lag=4)
    assert len(div) == 1
    assert 27 <= div.loc[0, "frame"] <= 30
    assert div.loc[0, "score"] > 0
