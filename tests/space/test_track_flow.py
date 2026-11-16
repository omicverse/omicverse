"""Synthetic tests for the global flow tracker (ov.space.track.flow_track / track_lineage).

The synthetic movie has founders that divide every ~60 frames (bright rounding just
before the split, daughters touching after it), a cell that walks in
across the border, random segmentation drop-outs of 1-6 frames and a short false
detection. No Cellpose / Trackastra needed.
"""
import numpy as np
import pandas as pd
import pytest

from omicverse.space.track import (build_tracklets, detections_table, flow_lineage_tables,
                                   flow_track, frame_overlaps, lineage_json, track_lineage)


def _paint(lab, img, cx, cy, r, lid, value):
    yy, xx = np.ogrid[:lab.shape[0], :lab.shape[1]]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    lab[m] = lid
    img[m] = value


def _synthetic(T=150, H=220, W=220, seed=0, drop_rate=0.03):
    rng = np.random.default_rng(seed)
    # cell: dict(id, gen, root, x, y, born, cycle)
    cells, nid, truth_div = [], 0, []

    def add(x, y, gen, root, born, parent):
        nonlocal nid
        c = dict(id=nid, gen=gen, root=root, x=x, y=y, born=born, parent=parent,
                 cycle=int(rng.integers(55, 70)), alive=True, drop=0)
        nid += 1
        cells.append(c)
        return c

    for x, y in [(50, 50), (150, 60), (70, 160)]:
        c = add(x, y, 0, None, 0, -1)
        c["root"] = c["id"]
        c["born"] = -int(rng.integers(0, 40))       # founders are part-way through a cycle
    labels = np.zeros((T, H, W), np.int32)
    movie = np.full((T, H, W), 100.0, np.float32) + rng.normal(0, 2, (T, H, W)).astype(np.float32)
    entrant_t = 40
    for t in range(T):
        if t == entrant_t:
            c = add(W - 3.0, 120.0, 0, None, t, -1)
            c["root"] = c["id"]
            c["vx"] = -1.5
        lid = 1
        for c in list(cells):
            if not c["alive"]:
                continue
            age = t - c["born"]
            if age == c["cycle"]:
                truth_div.append(dict(frame=t - 1, x=c["x"], y=c["y"], generation=c["gen"],
                                      root=c["root"]))
                c["alive"] = False
                ang = rng.uniform(0, np.pi)
                for s in (-1, 1):
                    d = add(c["x"] + s * 4 * np.cos(ang), c["y"] + s * 4 * np.sin(ang),
                            c["gen"] + 1, c["root"], t, c["id"])
                    d["dir"] = (s * np.cos(ang), s * np.sin(ang))
                continue
        for c in cells:
            if not c["alive"]:
                continue
            age = t - c["born"]
            if "vx" in c and c["x"] > W - 40:
                c["x"] += c["vx"]
            elif "dir" in c and age < 8:
                c["x"] += 1.2 * c["dir"][0]
                c["y"] += 1.2 * c["dir"][1]
            else:
                c["x"] = float(np.clip(c["x"] + rng.normal(0, 0.6), 12, W - 12))
                c["y"] = float(np.clip(c["y"] + rng.normal(0, 0.6), 12, H - 12))
            if c["drop"] > 0:
                c["drop"] -= 1
                continue
            if rng.random() < drop_rate and 3 < t < T - 3:
                c["drop"] = int(rng.integers(1, 6))
            if 0 <= c["cycle"] - age <= 3:
                _paint(labels[t], movie[t], c["x"], c["y"], 7, lid, 230.0)   # rounded, bright
            else:
                _paint(labels[t], movie[t], c["x"], c["y"], 9, lid, 140.0)
            lid += 1
        if t in (70, 71):   # a short false detection
            _paint(labels[t], movie[t], 200, 200, 5, lid, 150.0)
    truth = pd.DataFrame(truth_div)
    founders = {c["id"]: int((truth.root == c["id"]).sum()) if len(truth) else 0
                for c in cells if c["parent"] < 0 and c["born"] <= 0}
    return movie, labels, truth, founders, cells


def _match(pred, truth, ft=3, dx=12):
    used, hits, gen_ok = set(), 0, 0
    for r in truth.itertuples():
        best = None
        for j, p in enumerate(pred.itertuples()):
            if j in used or abs(p.frame - r.frame) > ft or np.hypot(p.x - r.x, p.y - r.y) > dx:
                continue
            best = j
            break
        if best is not None:
            used.add(best)
            hits += 1
            gen_ok += int(pred.iloc[best].generation == r.generation)
    return hits, gen_ok


@pytest.fixture(scope="module")
def synthetic():
    return _synthetic()


def test_frame_overlaps_and_tracklets(synthetic):
    movie, labels, truth, founders, cells = synthetic
    ovl = frame_overlaps(labels)
    assert {"frame", "label_a", "label_b", "inter"} <= set(ovl.columns)
    assert (ovl.inter > 0).all()
    det = detections_table(labels, movie)
    det = build_tracklets(det, ovl)
    assert "tracklet" in det and det.tracklet.min() == 0
    # every tracklet holds consecutive frames only
    gaps = det.sort_values("frame").groupby("tracklet").frame.diff().dropna()
    assert (gaps == 1).all()
    # tracklets are cut after the bright mitotic frames
    assert det.post_peak.any()


def test_flow_track_recovers_lineage(synthetic):
    movie, labels, truth, founders, cells = synthetic
    res = track_lineage(labels, movie, signal=None, verbose=False, long_gap=60,
                        max_gap=10, windows=3)
    dv = res["divisions"]
    hits, gen_ok = _match(dv, truth)
    assert hits >= len(truth) - 1
    assert len(dv) <= len(truth) + 1
    assert gen_ok >= hits - 1
    # founders: the three cells of frame 0 with their lineage sizes
    f = res["founders"]
    assert len(f) == 3
    assert sorted(f.divisions.tolist()) == sorted(founders.values())
    # the walk-in cell is generation 0 and not a founder
    C = res["cells"]
    assert ((C.parent < 0) & (C.start > 3)).sum() >= 1
    # the false detection does not become a cell
    assert not ((C.x0 > 190) & (C.y0 > 190)).any()
    out = lineage_json(res)
    assert set(out) == {"divisions", "founders", "generation_outcome",
                        "generation_window_divisions"}
    assert sum(sum(v) for v in out["generation_window_divisions"].values()) == len(dv)


def test_minimum_cycle_constraint(synthetic):
    """A spurious split right after a birth is removed by the refractory rule."""
    movie, labels, truth, founders, cells = synthetic
    det = detections_table(labels, movie)
    det = build_tracklets(det, frame_overlaps(labels))
    T, H, W = labels.shape
    cells_, info = flow_track(det, n_frames=T, frame_shape=(H, W), verbose=False,
                              max_gap=10, long_gap=60)
    res = flow_lineage_tables(cells_, n_frames=T, frame_shape=(H, W))
    C = res["cells"]
    kids = C[(C.parent >= 0) & (C.fate == "divided")]
    assert ((kids.end - kids.start) >= 0.5 * info["median_cycle"]).all()
    assert info["diameter"] > 10
