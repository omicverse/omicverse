r"""Linking per-frame segmentations into tracks with divisions."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..._registry import register_function


def detections_table(labels: np.ndarray, movie: Optional[np.ndarray] = None) -> pd.DataFrame:
    r"""One row per labelled object per frame: frame, label, x, y, area (+ intensity)."""
    from skimage.measure import regionprops_table
    rows = []
    for t in range(labels.shape[0]):
        lab = np.asarray(labels[t])
        if lab.max() == 0:
            continue
        props = ["label", "centroid", "area", "equivalent_diameter_area"]
        kw = {}
        if movie is not None:
            props.append("intensity_mean")
            kw["intensity_image"] = np.asarray(movie[t])
        try:
            p = regionprops_table(lab, properties=props, **kw)
        except Exception:  # older scikit-image names
            props = [q.replace("equivalent_diameter_area", "equivalent_diameter")
                     .replace("intensity_mean", "mean_intensity") for q in props]
            p = regionprops_table(lab, properties=props, **kw)
        d = pd.DataFrame(p).rename(columns={"centroid-0": "y", "centroid-1": "x",
                                            "equivalent_diameter": "diameter",
                                            "equivalent_diameter_area": "diameter",
                                            "mean_intensity": "intensity",
                                            "intensity_mean": "intensity"})
        d["frame"] = t
        rows.append(d)
    det = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["label", "y", "x", "area", "diameter", "frame"])
    det["label"] = det["label"].astype(int)
    return det[["frame", "label", "x", "y", "area", "diameter"] +
               (["intensity"] if "intensity" in det else [])]


def _tracklets_from_links(det: pd.DataFrame, links) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """det rows indexed 0..n-1; links = iterable of (i, j) parent-detection -> child-detection."""
    n = len(det)
    succ = [[] for _ in range(n)]
    pred = [[] for _ in range(n)]
    for i, j in links:
        succ[i].append(j)
        pred[j].append(i)
    tid = np.full(n, -1, int)
    parent_of = {}
    order = np.argsort(det["frame"].values, kind="stable")
    nxt = 0
    for i in order:
        if tid[i] >= 0:
            continue
        p = pred[i]
        # start of a tracklet: no predecessor, or predecessor divides / merges
        tid[i] = nxt
        if len(p) == 1 and len(succ[p[0]]) >= 2:
            parent_of[nxt] = tid[p[0]]
        cur = i
        while len(succ[cur]) == 1 and len(pred[succ[cur][0]]) == 1:
            cur = succ[cur][0]
            tid[cur] = nxt
        nxt += 1
    det = det.copy()
    det["track_id"] = tid
    tl = _summarise(det, parent_of)
    return det, tl


def _summarise(det: pd.DataFrame, parent_of: dict) -> pd.DataFrame:
    g = det.sort_values("frame").groupby("track_id")
    first = g.first()
    last = g.last()
    tl = pd.DataFrame({
        "start": first["frame"].astype(int), "end": last["frame"].astype(int),
        "x_start": first["x"], "y_start": first["y"],
        "x_end": last["x"], "y_end": last["y"],
        "n_det": g.size(), "diameter": g["diameter"].median(),
    })
    tl["parent"] = [int(parent_of.get(t, -1)) for t in tl.index]
    ch = pd.Series(tl["parent"].values, index=tl.index)
    counts = ch[ch >= 0].value_counts()
    tl["n_children"] = [int(counts.get(t, 0)) for t in tl.index]
    tl.index.name = "track_id"
    return tl


def _link_trackastra(labels, movie, model, mode, device, model_dir, delta_t=1,
                     threshold=0.5, cache=None):
    import os
    import pickle
    import torch  # noqa: F401
    from trackastra.model import Trackastra
    from trackastra.tracking.tracking import build_graph, track_greedy
    preds = None
    if cache is not None and os.path.exists(cache):
        with open(cache, "rb") as fh:
            preds = pickle.load(fh)
    if preds is None:
        if device == "auto":
            device = "automatic"
        if model_dir is not None:
            from pathlib import Path
            tr = Trackastra.from_folder(Path(model_dir), device=device)
        else:
            tr = Trackastra.from_pretrained(model, device=device)
        imgs = np.asarray(movie, dtype=np.float32)
        masks = np.asarray(labels).astype(np.int32)
        preds = tr._predict(imgs, masks)
        if cache is not None:
            with open(cache, "wb") as fh:
                pickle.dump(preds, fh)
    if mode == "ilp":
        from trackastra.tracking.ilp import track_ilp
        G = build_graph(nodes=preds["nodes"], weights=preds["weights"], max_distance=256,
                        max_neighbors=10, delta_t=delta_t)
        return track_ilp(G, ilp_config="gt")
    G = build_graph(nodes=preds["nodes"], weights=preds["weights"], max_distance=256,
                    max_neighbors=10, delta_t=delta_t)
    return track_greedy(G, allow_divisions=(mode != "greedy_nodiv"), threshold=threshold)


def _link_overlap(det: pd.DataFrame, gate: float):
    """Frame-to-frame Hungarian matching on centroid distance, plus divisions.

    An unmatched object in frame t+1 becomes the second daughter of the nearest
    object in frame t when that object was matched and both daughters lie within the
    gate of it.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    links = []
    byf = {t: d.index.values for t, d in det.groupby("frame")}
    frames = sorted(byf)
    xy = det[["x", "y"]].values
    for a, b in zip(frames[:-1], frames[1:]):
        if b != a + 1:
            continue
        ia, ib = byf[a], byf[b]
        D = cdist(xy[ia], xy[ib])
        C = np.where(D <= gate, D, 1e6)
        r, c = linear_sum_assignment(C)
        ok = C[r, c] < 1e6
        matched_b = set(c[ok])
        out_deg = {}
        for rr, cc in zip(r[ok], c[ok]):
            links.append((ia[rr], ib[cc]))
            out_deg[rr] = 1
        for cc in range(len(ib)):
            if cc in matched_b:
                continue
            rr = int(np.argmin(D[:, cc]))
            if D[rr, cc] <= gate and out_deg.get(rr, 0) == 1:
                links.append((ia[rr], ib[cc]))
                out_deg[rr] = 2
    return links


@register_function(
    aliases=["细胞追踪", "track cells", "cell tracking", "link segmentations over time",
             "trackastra", "lineage tracking", "多帧细胞关联"],
    category="space",
    description="Link per-frame cell segmentations into tracks with divisions. Default "
                "'trackastra' (pretrained transformer that scores associations over a "
                "window of frames, then greedy/ILP linking with divisions; Gallusser & "
                "Weigert, ECCV 2024); fallback 'overlap' (Hungarian centroid matching + "
                "division by an extra nearby daughter). Returns per-detection table with "
                "track ids and a tracklet table (start, end, parent, n_children)",
    examples=["det, tl = ov.space.track.track_cells(labels, movie)",
              "det, tl = ov.space.track.track_cells(labels, movie, method='overlap')"],
    related=["space.track.segment_movie", "space.track.close_gaps",
             "space.track.lineage_tables"],
)
def track_cells(labels: np.ndarray, movie: Optional[np.ndarray] = None, *,
                method: str = "trackastra", model: str = "general_2d", mode: str = "greedy",
                device: str = "auto", model_dir: Optional[str] = None,
                gate: Optional[float] = None, delta_t: int = 1, threshold: float = 0.5,
                cache: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""Track segmented cells through a movie.

    Parameters
    ----------
    labels
        ``(T, H, W)`` label movie (:func:`segment_movie`).
    movie
        ``(T, H, W)`` raw movie (needed by ``'trackastra'``; adds an intensity column).
    method
        ``'trackastra'`` or ``'overlap'``.
    model, mode
        Trackastra pretrained model (``'general_2d'``, ``'ctc'``) and linker
        (``'greedy'``, ``'greedy_nodiv'``, ``'ilp'``).
    model_dir
        Folder of an already-downloaded Trackastra model (offline use).
    gate
        ``'overlap'`` only: maximum centroid displacement between frames; default the
        median object diameter.
    delta_t
        ``'trackastra'``: frame distance of candidate links. The pretrained models
        (v0.3) only score consecutive frames, so values > 1 add nothing; missed
        detections are bridged by :func:`close_gaps` instead.
    threshold
        ``'trackastra'``: minimum association probability for a link (greedy).
    cache
        ``'trackastra'``: pickle file for the model's association predictions, so the
        linking can be re-run with other settings without the network.

    Returns
    -------
    det
        One row per object and frame: ``frame, label, x, y, area, diameter[,
        intensity], track_id``.
    tracklets
        One row per tracklet (cell between two events): ``start, end, x_start,
        y_start, x_end, y_end, n_det, diameter, parent, n_children``.
    """
    det = detections_table(labels, movie)
    if method == "trackastra":
        if movie is None:
            raise ValueError("trackastra needs the raw movie")
        graph = _link_trackastra(labels, movie, model, mode, device, model_dir,
                                 delta_t=delta_t, threshold=threshold, cache=cache)
        key = {(int(f), int(l)): i for i, (f, l) in enumerate(zip(det["frame"], det["label"]))}
        links = []
        for u, v in graph.edges():
            nu, nv = graph.nodes[u], graph.nodes[v]
            a = key.get((int(nu["time"]), int(nu["label"])))
            b = key.get((int(nv["time"]), int(nv["label"])))
            if a is not None and b is not None:
                links.append((a, b))
    elif method == "overlap":
        g = float(det["diameter"].median()) if gate is None else float(gate)
        links = _link_overlap(det, g)
    else:
        raise ValueError(f"unknown method {method!r}")
    return _tracklets_from_links(det, links)


def _relabel(det: pd.DataFrame, tl: pd.DataFrame, merge: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """merge: {later_track: earlier_track} -> continue earlier track with later's detections."""
    def root(t):
        while t in merge:
            t = merge[t]
        return t
    det = det.copy()
    det["track_id"] = [root(t) for t in det["track_id"]]
    parent_of = {}
    for t, row in tl.iterrows():
        if row["parent"] >= 0:
            parent_of[root(t)] = root(int(row["parent"]))
    parent_of = {k: v for k, v in parent_of.items() if k != v}
    return det, _summarise(det, parent_of)


@register_function(
    aliases=["轨迹断点连接", "gap closing", "close track gaps", "relink broken tracks"],
    category="space",
    description="Gap closing: join a tracklet that ends without dividing (and away from the "
                "border) to a parent-less tracklet that starts within max_gap frames nearby, "
                "by a global Hungarian assignment on distance. Repairs tracks broken by "
                "missed detections, which would otherwise appear as new generation-0 cells",
    examples=["det, tl = ov.space.track.close_gaps(det, tl, frame_shape=movie.shape[1:])"],
)
def close_gaps(det: pd.DataFrame, tracklets: pd.DataFrame, *, frame_shape=None,
               max_gap: int = 5, gate: Optional[float] = None,
               border: Optional[float] = None, n_frames: Optional[int] = None,
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""Join broken tracks.

    Parameters
    ----------
    max_gap
        Largest number of frames bridged (end frame to start frame).
    gate
        Distance limit for a one-frame gap, growing as ``sqrt(gap)``; default the
        median object diameter.
    border
        Tracklets ending / starting within this many pixels of the frame edge are not
        joined (they left / entered); default half the median diameter.
    """
    from scipy.optimize import linear_sum_assignment
    tl = tracklets
    diam = float(det["diameter"].median())
    gate = diam if gate is None else float(gate)
    border = diam / 2 if border is None else float(border)
    T = int(det["frame"].max()) + 1 if n_frames is None else int(n_frames)

    def near_border(x, y):
        if frame_shape is None:
            return False
        H, W = frame_shape[:2]
        return x < border or y < border or x > W - 1 - border or y > H - 1 - border

    ends = tl[(tl["n_children"] == 0) & (tl["end"] < T - 1)]
    ends = ends[[not near_border(x, y) for x, y in zip(ends["x_end"], ends["y_end"])]]
    starts = tl[(tl["parent"] < 0) & (tl["start"] > 0)]
    starts = starts[[not near_border(x, y) for x, y in zip(starts["x_start"], starts["y_start"])]]
    if len(ends) == 0 or len(starts) == 0:
        return det, tl
    E = ends[["end", "x_end", "y_end"]].values
    S = starts[["start", "x_start", "y_start"]].values
    dt = S[None, :, 0] - E[:, None, 0]
    d = np.hypot(S[None, :, 1] - E[:, None, 1], S[None, :, 2] - E[:, None, 2])
    ok = (dt >= 1) & (dt <= max_gap) & (d <= gate * np.sqrt(np.maximum(dt, 1)))
    C = np.where(ok, d + 1e-3 * dt, 1e9)
    r, c = linear_sum_assignment(C)
    merge = {}
    for rr, cc in zip(r, c):
        if C[rr, cc] < 1e9:
            merge[int(starts.index[cc])] = int(ends.index[rr])
    if not merge:
        return det, tl
    return _relabel(det, tl, merge)


@register_function(
    aliases=["合并-分离轨迹连接", "close gaps through merges", "occlusion gap closing",
             "merge split resolution"],
    category="space",
    description="Re-link tracks interrupted by a temporary merge: when a track ends because "
                "its cell was absorbed into a neighbour's mask and a parent-less track later "
                "emerges from that same neighbour, join the two (global assignment, shortest "
                "interruption first). Complements close_gaps, which only bridges short "
                "missed-detection gaps by distance",
    examples=["det, tl = ov.space.track.close_merge_gaps(det, tl, frame_shape=movie.shape[1:])"],
)
def close_merge_gaps(det: pd.DataFrame, tracklets: pd.DataFrame, *, frame_shape=None,
                     max_gap: int = 100, gate: Optional[float] = None,
                     border: Optional[float] = None, n_frames: Optional[int] = None,
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""Join a track that disappeared into a neighbour to the track that re-emerges.

    Parameters
    ----------
    max_gap
        Longest merge (frames) that is resolved.
    gate
        Distance from the ending cell to the absorbing neighbour (and from the
        emerging cell to the emitting neighbour); default the median diameter.
    border
        Ends / starts this close to the frame edge are left alone (default half the
        median diameter).
    """
    from scipy.optimize import linear_sum_assignment
    tl = tracklets
    diam = float(det["diameter"].median())
    gate = diam if gate is None else float(gate)
    border = diam / 2 if border is None else float(border)
    T = int(det["frame"].max()) + 1 if n_frames is None else int(n_frames)
    byf = {t: g for t, g in det.groupby("frame")}

    def near_border(x, y):
        if frame_shape is None:
            return False
        H, W = frame_shape[:2]
        return x < border or y < border or x > W - 1 - border or y > H - 1 - border

    def neighbour(t, x, y, exclude):
        g = byf.get(t)
        if g is None:
            return -1
        dd = np.hypot(g["x"].values - x, g["y"].values - y)
        ok = g["track_id"].values != exclude
        if not ok.any():
            return -1
        k = np.argmin(np.where(ok, dd, np.inf))
        return int(g["track_id"].values[k]) if dd[k] <= gate else -1

    ends = tl[(tl["n_children"] == 0) & (tl["end"] < T - 1)]
    starts = tl[(tl["parent"] < 0) & (tl["start"] > 0)]
    E, S = [], []
    for t, r in ends.iterrows():
        if near_border(r["x_end"], r["y_end"]):
            continue
        a = neighbour(int(r["end"]) + 1, r["x_end"], r["y_end"], t)
        if a >= 0:
            E.append((t, int(r["end"]), a))
    for t, r in starts.iterrows():
        if near_border(r["x_start"], r["y_start"]):
            continue
        e = neighbour(int(r["start"]) - 1, r["x_start"], r["y_start"], t)
        if e >= 0:
            S.append((t, int(r["start"]), e))
    if not E or not S:
        return det, tl
    # the absorbing track may itself be continued by other tracklets; compare by lineage
    # identity: the absorber's id at the end frame vs the emitter's id at the start frame
    C = np.full((len(E), len(S)), 1e9)
    for i, (te, fe, a) in enumerate(E):
        for j, (ts, fs, e) in enumerate(S):
            if a == e and 1 <= fs - fe <= max_gap:
                C[i, j] = fs - fe
    if not np.isfinite(C).any() or C.min() >= 1e9:
        return det, tl
    r, c = linear_sum_assignment(C)
    merge = {int(S[j][0]): int(E[i][0]) for i, j in zip(r, c) if C[i, j] < 1e9}
    return _relabel(det, tl, merge) if merge else (det, tl)
