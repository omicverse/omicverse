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
    pytest.importorskip("cv2")  # opencv is optional (mitosis_signal)
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


def _det(rows):
    """rows: (frame, x, y, track_id)"""
    d = pd.DataFrame(rows, columns=["frame", "x", "y", "track_id"])
    d["label"] = 1
    d["area"] = 100.0
    d["diameter"] = 10.0
    return d


def test_insert_adopt_and_prune():
    from omicverse.space.track import adopt_orphans, insert_mitoses, prune_divisions
    from omicverse.space.track._track import _summarise
    # track 0 passes through (50,50) for frames 0..19; track 1 appears at frame 11 nearby
    rows = [(t, 50.0, 50.0, 0) for t in range(20)] + [(t, 58.0, 50.0, 1) for t in range(11, 20)]
    det = _det(rows)
    tl = _summarise(det, {})
    ev = pd.DataFrame({"frame": [10], "x": [50.0], "y": [50.0], "score": [5.0]})
    det2, tl2 = insert_mitoses(det, tl, ev, None, cell_diameter=10)
    assert tl2.attrs["inserted"] == 1
    mother = tl2[tl2["n_children"] == 2]
    assert len(mother) == 1 and int(mother["end"].iloc[0]) == 10
    # adopt: a division with one daughter adopts the orphan next to it
    rows = [(t, 50.0, 50.0, 0) for t in range(11)] + [(t, 52.0, 50.0, 2) for t in range(11, 20)] \
        + [(t, 60.0, 50.0, 1) for t in range(12, 20)]
    det = _det(rows)
    tl = _summarise(det, {2: 0})
    div = pd.DataFrame({"mother": [0], "frame": [10], "x": [50.0], "y": [50.0]})
    det3, tl3 = adopt_orphans(det, tl, div, cell_diameter=10)
    assert tl3.attrs["adopted"] == 1 and tl3.loc[1, "parent"] == 0
    # prune: a low-score division among high-score ones is undone
    rows, parent, divs = [], {}, []
    for k in range(12):
        m, a, b = 3 * k, 3 * k + 1, 3 * k + 2
        x = 100.0 * k
        rows += [(t, x, 0.0, m) for t in range(10)]
        rows += [(t, x - 3, 0.0, a) for t in range(10, 20)]
        rows += [(t, x + 3, 0.0, b) for t in range(10, 20)]
        parent[a] = m
        parent[b] = m
        divs.append({"mother": m, "daughters": (a, b), "frame": 9, "x": x, "y": 0.0,
                     "score": 0.1 if k == 0 else 8.0 + 0.1 * k})
    det = _det(rows)
    tl = _summarise(det, parent)
    det4, tl4, kept = prune_divisions(det, tl, pd.DataFrame(divs))
    assert len(kept) == 11 and (tl4["n_children"] >= 2).sum() == 11


def test_edge_grazing_cells_are_not_counted():
    rows = [(t, 2.0, 50.0, 0) for t in range(5)] + [(t, 50.0, 50.0, 1) for t in range(10)]
    det = _det(rows)
    from omicverse.space.track._track import _summarise
    tl = _summarise(det, {})
    res = lineage_tables(det, tl, n_frames=10, frame_shape=(100, 100))
    assert res["cells"].loc[0, "fate"] == "edge"
    assert res["generation_outcome"].loc[0, "still_present"] == 1
