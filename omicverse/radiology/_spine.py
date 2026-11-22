r"""Per-level vertebral alignment (spondylolisthesis) on a sagittal spine series.

:func:`measure_spine_alignment` segments the vertebral bodies (and the
sacrum) on a sagittal MR or CT slice without a trained model, orders them
along the column, labels them from the sacrum upward when the sacrum can be
identified, and measures for every disc level the translation of the upper
body relative to the lower one — the quantity graded by Meyerding and
expressed as a percentage by Taillard.

Measurement definitions (literature)
------------------------------------
* **Meyerding grade** — the slip is the displacement of the posterior-inferior
  corner of the upper body relative to the posterior-superior corner of the
  lower body, along the lower body's superior endplate, as a fraction of that
  endplate's antero-posterior length: grade I ≤ 25 %, II ≤ 50 %, III ≤ 75 %,
  IV ≤ 100 %, V > 100 % (spondyloptosis). Meyerding HW, *Surg Gynecol
  Obstet* 1932; 54:371-377; see also StatPearls "Spondylolisthesis"
  (NBK430767) and Radiopaedia "Spondylolisthesis".
* **Taillard index** — the same displacement as a continuous percentage of the
  AP length of the lower endplate (Taillard W, *Acta Orthop Scand* 1954) —
  reported here as ``slip_pct``.
* **Posterior vertebral body line (George's line)** — the posterior walls of
  consecutive bodies form a smooth line; a step is ante- or retrolisthesis.
  The default measurement intersects a line fitted to each posterior wall with
  a line fitted to the adjacent endplate ("virtual corners"), which is the
  posterior-line reading of the Meyerding offset and is insensitive to the
  rounded corners and small osteophytes that move a single corner pixel.
* **Abnormal** — an offset of at least ``min_slip_mm`` (default 3 mm, the
  threshold most often used to define degenerative spondylolisthesis in
  epidemiological work; see the review by Wang et al., arXiv:1610.06608).

Segmentation and labelling (methods, reimplemented)
---------------------------------------------------
Classical vertebra segmentation on MRI uses intensity clustering / multilevel
thresholds followed by morphology (e.g. fuzzy c-means + morphology,
arXiv:1605.02460), and notes that edges alone are unreliable because of the
wide range of edge strengths, so regions come first and boundaries are
refined afterwards. Here: the slice is divided by a smoothed local mean
(coil-shading correction), connected components are collected over a grid
of intensity windows (bright *and* intermediate windows, so T1, T2 and CT
all work), kept when they are compact and box-like at vertebral-body size,
de-duplicated, and the longest chain of similar-sized boxes stacked along a
smooth curve is taken as the column (a dynamic programme over candidates in
craniocaudal order, in the spirit of the part-based detection + chain model
of Lootus et al., *Vertebrae detection and labelling in lumbar MR images*,
MICCAI-CSI 2013). The sacrum is the elongated bony component directly below
the lowest body; levels are counted upward from it, which is also how
radiologists number lumbar levels on sagittal images. Transitional
lumbosacral anatomy (present in roughly a third of people; SPIDER dataset
paper, arXiv:2306.12217) makes any count from the sacrum alone uncertain,
so a ``label_confidence`` is reported and labels are relative (``V1`` =
lowest mobile body) whenever the sacrum is not identified.

Candidate comparison on synthetic phantoms (see the tests): fitted-edge
virtual corners (default), extreme corner points of the segmented body,
and body-centroid offset were compared across noise, slice thickness,
bias field, lordosis and slip grade 0-III; the fitted-edge method had the
lowest error and is reported, the other two are returned as cross-checks and
drive the QC flag when they disagree.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._registry import register_function
from ._dicom import DicomSeries, DicomStudy

__all__ = ["measure_spine_alignment", "DEFAULT_LEVEL_NAMES"]

#: Names from the sacrum upward used when the sacrum is identified; pass
#: ``level_names`` to number differently (e.g. a transitional segment).
DEFAULT_LEVEL_NAMES = ("S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10", "T9", "T8", "T7")


# ----------------------------------------------------------------- geometry
def _frame(inst) -> Tuple[np.ndarray, np.ndarray]:
    """Superior and posterior unit vectors in in-plane mm coordinates (y=row, x=col)."""
    col_cos, row_cos = np.asarray(inst.col_cos, float), np.asarray(inst.row_cos, float)
    sup = np.array([col_cos[2], row_cos[2]])
    post = np.array([col_cos[1], row_cos[1]])
    if np.linalg.norm(sup) < 0.3:
        raise ValueError("the slice does not contain the craniocaudal axis (not sagittal)")
    sup /= np.linalg.norm(sup)
    post = post - (post @ sup) * sup
    if np.linalg.norm(post) < 1e-6:
        post = np.array([-sup[1], sup[0]])
    return sup, post / np.linalg.norm(post)


def _min_rect(P: np.ndarray, axis: np.ndarray, span_deg: float = 45.0) -> Dict:
    """Minimum-area rectangle of points ``P`` (mm) with one side within
    ``span_deg`` of ``axis``. Returns centre, unit ``u`` (≈ axis), ``e``,
    side lengths ``h`` (along u) and ``d`` (along e)."""
    a0 = np.arctan2(axis[1], axis[0])
    best = None
    for a in a0 + np.deg2rad(np.arange(-span_deg, span_deg + 0.1, 1.0)):
        u = np.array([np.cos(a), np.sin(a)])
        e = np.array([-u[1], u[0]])
        pu, pe = P @ u, P @ e
        h, d = np.ptp(pu), np.ptp(pe)
        if best is None or h * d < best[0]:
            c = u * (pu.max() + pu.min()) / 2 + e * (pe.max() + pe.min()) / 2
            best = (h * d, c, u, e, h, d)
    _, c, u, e, h, d = best
    return {"c": c, "u": u, "e": e, "h": float(h), "d": float(d)}


def _line_fit(P: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(P) < 3:
        return None
    m = P.mean(0)
    _, _, vt = np.linalg.svd(P - m)
    return m, vt[0]


def _intersect(l1, l2) -> Optional[np.ndarray]:
    (p, r), (q, s) = l1, l2
    A = np.array([r, -s]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, q - p)
    return p + t[0] * r


# ------------------------------------------------------------- segmentation
def _normalise(img: np.ndarray, ps: float, sigma_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    from scipy import ndimage as ndi
    img = np.asarray(img, float)
    body = img > 0.05 * np.percentile(img, 99)
    body = ndi.binary_fill_holes(ndi.binary_closing(body, iterations=3))
    s = sigma_mm / ps
    g = ndi.gaussian_filter(np.where(body, img, 0), s) / np.maximum(ndi.gaussian_filter(body.astype(float), s), 1e-3)
    n = np.where(body, img / np.maximum(g, 1e-6), 0.0)
    return ndi.median_filter(n, 3), body


def _region_props(mask_idx: np.ndarray, shape, ps2: np.ndarray, axis: np.ndarray, area_mm2: float) -> Optional[Dict]:
    from scipy.spatial import ConvexHull
    rr, cc = np.unravel_index(mask_idx, shape)
    P = np.stack([rr * ps2[0], cc * ps2[1]], 1)
    try:
        hull = ConvexHull(P)
    except Exception:  # noqa: BLE001 - degenerate region
        return None
    H = P[hull.vertices]
    rect = _min_rect(H, axis)
    solidity = area_mm2 / max(hull.volume, 1e-6)
    extent = area_mm2 / max(rect["h"] * rect["d"], 1e-6)
    return dict(rect, solidity=float(solidity), extent=float(extent), area=float(area_mm2),
                centroid=P.mean(0), idx=mask_idx)


def _candidates(n: np.ndarray, body: np.ndarray, ps2: np.ndarray, axis: np.ndarray, *, size_mm, n_levels: int,
                min_extent: float, min_solidity: float, max_aspect: float) -> Tuple[List[Dict], List[Dict]]:
    """Body-like components over a grid of intensity windows, and all large
    components (for the sacrum search)."""
    from scipy import ndimage as ndi
    vals = n[body]
    qs = np.percentile(vals, np.linspace(15, 92, n_levels))
    his = [np.inf] + list(np.percentile(vals, [75, 85, 93]))
    r_open = max(1, int(round(1.2 / float(np.mean(ps2)))))
    st = ndi.generate_binary_structure(2, 1)
    px_area = float(ps2[0] * ps2[1])
    amin, amax = size_mm[0] ** 2 * 0.6, size_mm[1] ** 2
    bodies, large = [], []
    seen = set()
    for hi in his:
        for lo in qs:
            if lo >= hi:
                continue
            m = (n > lo) & (n < hi) & body
            m = ndi.binary_opening(m, st, iterations=r_open)
            lab, k = ndi.label(m)
            if k == 0:
                continue
            idx = np.arange(1, k + 1)
            area = ndi.sum(m, lab, idx) * px_area
            objs = ndi.find_objects(lab)
            for j in np.where((area >= amin) & (area <= 6 * amax))[0]:
                sl = objs[j]
                sub = lab[sl] == j + 1
                sub = ndi.binary_fill_holes(sub)
                rr, cc = np.nonzero(sub)
                flat = np.ravel_multi_index((rr + sl[0].start, cc + sl[1].start), n.shape)
                key = (int(flat.size), int(flat[0]), int(flat[-1]))
                if key in seen:
                    continue
                seen.add(key)
                pr = _region_props(flat, n.shape, ps2, axis, float(flat.size * px_area))
                if pr is None:
                    continue
                pr["window"] = (float(lo), float(hi))
                if area[j] > 1.2 * amin:
                    large.append(pr)
                h, d = pr["h"], pr["d"]
                if (size_mm[0] <= min(h, d) and max(h, d) <= size_mm[1] and max(h, d) / max(min(h, d), 1e-6) <= max_aspect
                        and pr["extent"] >= min_extent and pr["solidity"] >= min_solidity and area[j] <= amax):
                    pr["score"] = pr["extent"] * pr["solidity"]
                    bodies.append(pr)
    # non-maximum suppression (same body found in several windows)
    bodies.sort(key=lambda r: -r["score"])
    kept: List[Dict] = []
    for b in bodies:
        sb = set(b["idx"].tolist()) if len(kept) < 200 else None
        ok = True
        for k_ in kept:
            if np.linalg.norm(b["centroid"] - k_["centroid"]) > 0.75 * (np.sqrt(b["area"]) + np.sqrt(k_["area"])):
                continue
            inter = len(sb & k_["_set"])
            if inter / min(len(sb), len(k_["_set"])) > 0.3:
                ok = False
                break
        if ok:
            b["_set"] = sb
            kept.append(b)
    return kept, large


def _chain(c: List[Dict], sup: np.ndarray, max_turn_deg: float, spacing: Tuple[float, float]) -> List[int]:
    """Longest smooth chain of similar boxes, ordered inferior → superior."""
    if not c:
        return []
    order = sorted(range(len(c)), key=lambda i: float(c[i]["centroid"] @ sup))
    pos = {i: k for k, i in enumerate(order)}
    n = len(c)
    edges: Dict[int, List[int]] = {i: [] for i in range(n)}   # i (lower) -> j (upper)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = c[j]["centroid"] - c[i]["centroid"]
            along = float(d @ sup)
            if along <= 0:
                continue
            s = 0.5 * (np.sqrt(c[i]["area"]) + np.sqrt(c[j]["area"]))
            L = float(np.linalg.norm(d))
            if not (spacing[0] * s <= L <= spacing[1] * s):
                continue
            if np.degrees(np.arccos(np.clip(along / L, -1, 1))) > 60:
                continue
            ra = c[i]["area"] / c[j]["area"]
            if not (0.45 <= ra <= 2.2):
                continue
            edges[i].append(j)
    # DP over (prev, cur) with a turn limit; score = n_bodies + mean quality bonus - turn penalty
    best: Dict[Tuple[int, int], Tuple[float, List[int]]] = {}
    for i in sorted(range(n), key=lambda k: pos[k]):
        for j in edges[i]:
            sc = 1.0 + c[i]["score"] + c[j]["score"]
            cand = (sc, [i, j])
            for (a, b), (v, path) in list(best.items()):
                if b != i:
                    continue
                d1 = c[i]["centroid"] - c[a]["centroid"]
                d2 = c[j]["centroid"] - c[i]["centroid"]
                cosang = float(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9))
                turn = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if turn > max_turn_deg or not (0.6 <= l2 / l1 <= 1.6):
                    continue
                v2 = v + 1.0 + c[j]["score"] - turn / (3 * max_turn_deg)
                if v2 > cand[0]:
                    cand = (v2, path + [j])
            if (i, j) not in best or cand[0] > best[(i, j)][0]:
                best[(i, j)] = cand
    if not best:
        i = max(range(n), key=lambda k: c[k]["score"])
        return [i]
    return max(best.values(), key=lambda t: t[0])[1]


def _sacrum(large: List[Dict], bodies: List[Dict], h_med: float, d_med: float, ps2, shape,
            min_elongation: float) -> Tuple[Optional[Dict], int]:
    """Elongated bony component directly below the lowest body.

    Returns the sacrum (rectangle of its top ``h_med``, i.e. the S1 segment)
    and how many chain bodies were absorbed into it: when a window splits the
    S1 segment off as a box, the lowest "body" lies inside an elongated
    component that continues caudally, and it is the sacrum's top instead.
    """
    best, best_drop = None, 0
    for drop in (0, 1):
        if drop >= len(bodies) - 1 and drop:
            break
        low = bodies[drop - 1] if drop else bodies[0]
        u = low["u"]
        bottom = float(low["c"] @ u - low["h"] / 2)
        low_set = set(np.asarray(low["idx"]).tolist()) if drop else None
        ref = bodies[drop] if drop else low      # the body that must sit on top of the sacrum
        ref_bottom = float(ref["c"] @ u - ref["h"] / 2)
        for r in large:
            rr, cc = np.unravel_index(r["idx"], shape)
            P = np.stack([rr * ps2[0], cc * ps2[1]], 1)
            pu = P @ u
            top = float(pu.max())
            if drop:
                inter = len(low_set.intersection(np.asarray(r["idx"]).tolist()))
                if inter < 0.8 * len(low_set):
                    continue
            gap = ref_bottom - top
            if not (0.5 <= gap <= 1.2 * h_med):
                continue
            length = float(np.ptp(pu))
            if length < min_elongation * h_med:
                continue
            top_part = P[pu >= top - 0.9 * h_med]
            if len(top_part) < 20:
                continue
            rect = _min_rect(top_part, u, span_deg=40)
            if not (0.5 * d_med <= rect["d"] <= 1.8 * d_med) or \
                    abs(float(rect["c"] @ ref["e"]) - float(ref["c"] @ ref["e"])) > 0.8 * d_med:
                continue
            sc = length / h_med
            if best is None or sc > best["elongation"]:
                best = dict(rect, elongation=sc, idx=r["idx"], P=top_part, score=r.get("extent", 0.0),
                            solidity=r["solidity"], area=r["area"], centroid=P.mean(0), length_mm=length, gap_mm=gap)
                best_drop = drop
        if best is not None:
            break
    return best, best_drop


def _body_geometry(P: np.ndarray, contour: np.ndarray, sup: np.ndarray, post: np.ndarray, axis: np.ndarray,
                   rect: Optional[Dict] = None, top_only: bool = False) -> Dict:
    """Corners of one body: extreme-point corners and fitted-edge virtual corners."""
    r = rect if rect is not None else _min_rect(P, axis, span_deg=40)
    u, e = r["u"], r["e"]
    if u @ sup < 0:
        u = -u
    if e @ post > 0:          # e points anteriorly
        e = -e
    c, h, d = r["c"], r["h"], r["d"]
    C = contour
    lu, le = (C - c) @ u, (C - c) @ e
    out = {"c": c, "u": u, "e": e, "h": h, "d": d}
    # extreme-point corners: contour point nearest each rectangle corner
    for name, su, se in (("PS", 1, -1), ("AS", 1, 1), ("PI", -1, -1), ("AI", -1, 1)):
        k = c + su * h / 2 * u + se * d / 2 * e
        out[name + "_corner"] = C[int(np.argmin(np.linalg.norm(C - k, axis=1)))]
    # fitted edges on the middle 60 % of each side
    def side(sel):
        return _line_fit(C[sel]) if sel.sum() >= 4 else None
    sup_l = side((lu > 0.25 * h) & (np.abs(le) < 0.3 * d))
    inf_l = side((lu < -0.25 * h) & (np.abs(le) < 0.3 * d))
    post_l = side((le < -0.25 * d) & (np.abs(lu) < (0.15 * h if top_only else 0.3 * h)) if not top_only else
                  (le < -0.25 * d) & (lu > 0.0) & (lu < 0.35 * h))
    ant_l = side((le > 0.25 * d) & (np.abs(lu) < (0.3 * h)) if not top_only else
                 (le > 0.25 * d) & (lu > 0.0) & (lu < 0.35 * h))
    out["lines"] = {"sup": sup_l, "inf": inf_l, "post": post_l, "ant": ant_l}
    for name, a, b in (("PS", "sup", "post"), ("AS", "sup", "ant"), ("PI", "inf", "post"), ("AI", "inf", "ant")):
        la, lb = out["lines"][a], out["lines"][b]
        v = _intersect(la, lb) if la is not None and lb is not None else None
        if v is None or np.linalg.norm(v - out[name + "_corner"]) > 0.25 * max(h, d):
            v = out[name + "_corner"]
            out.setdefault("edge_fallback", []).append(name)
        out[name] = v
    out["centroid"] = P.mean(0)
    return out


def _contour(idx: np.ndarray, shape, ps2) -> Tuple[np.ndarray, np.ndarray]:
    from scipy import ndimage as ndi
    from skimage.measure import find_contours
    rr, cc = np.unravel_index(idx, shape)
    r0, r1, c0, c1 = rr.min() - 3, rr.max() + 4, cc.min() - 3, cc.max() + 4
    m = np.zeros((r1 - r0, c1 - c0), float)
    m[rr - r0, cc - c0] = 1
    cs = find_contours(ndi.gaussian_filter(m, 0.7), 0.5)
    C = max(cs, key=len) if cs else np.stack([rr - r0, cc - c0], 1).astype(float)
    C = (C + [r0, c0]) * ps2
    P = np.stack([rr * ps2[0], cc * ps2[1]], 1)
    return P, C


def _segment_slice(img: np.ndarray, ps2: np.ndarray, sup: np.ndarray, post: np.ndarray, *, size_mm, shading_mm,
                   n_levels, min_extent, min_solidity, max_aspect, max_turn_deg, spacing,
                   min_sacrum_elongation) -> Dict:
    ps = float(np.mean(ps2))
    n, body = _normalise(img, ps, shading_mm)
    cands, large = _candidates(n, body, ps2, sup, size_mm=size_mm, n_levels=n_levels, min_extent=min_extent,
                               min_solidity=min_solidity, max_aspect=max_aspect)
    ch = _chain(cands, sup, max_turn_deg, spacing)
    bodies = []
    for i in ch:
        P, C = _contour(cands[i]["idx"], n.shape, ps2)
        g = _body_geometry(P, C, sup, post, sup, rect=cands[i])
        g.update(score=cands[i]["score"], solidity=cands[i]["solidity"], extent=cands[i]["extent"],
                 area=cands[i]["area"], kind="body")
        g["idx"] = cands[i]["idx"]
        bodies.append(g)
    sac = None
    if bodies:
        h_med = float(np.median([b["h"] for b in bodies]))
        d_med = float(np.median([b["d"] for b in bodies]))
        s, drop = _sacrum(large, bodies, h_med, d_med, ps2, n.shape, min_sacrum_elongation)
        if s is not None:
            bodies = bodies[drop:]
            Pall, Call = _contour(s["idx"], n.shape, ps2)
            pu = Call @ bodies[0]["u"]
            Ctop = Call[pu >= pu.max() - 0.9 * h_med]
            g = _body_geometry(s["P"], Ctop, sup, post, bodies[0]["u"], rect=s, top_only=True)
            g.update(score=s["score"], solidity=s["solidity"], extent=np.nan, area=s["area"], kind="sacrum",
                     elongation=s["elongation"], length_mm=s["length_mm"])
            sac = g
    return {"bodies": bodies, "sacrum": sac, "n_candidates": len(cands)}


# --------------------------------------------------------------- measuring
def _meyerding(pct: float) -> int:
    a = abs(pct)
    if a <= 0:
        return 0
    return int(min(5, np.ceil(a / 25.0)))


def _level_rows(seg: Dict, *, min_slip_mm: float, agree_mm: float) -> List[Dict]:
    chain = ([seg["sacrum"]] if seg["sacrum"] is not None else []) + seg["bodies"]
    rows = []
    d_med = float(np.median([b["d"] for b in seg["bodies"]])) if seg["bodies"] else np.nan
    for k in range(1, len(chain)):
        lo, up = chain[k - 1], chain[k]
        e = lo["AS"] - lo["PS"]
        L = float(np.linalg.norm(e))
        e = e / max(L, 1e-9)
        nrm = np.array([-e[1], e[0]])
        if nrm @ lo["u"] < 0:
            nrm = -nrm
        off = float((up["PI"] - lo["PS"]) @ e)
        off_corner = float((up["PI_corner"] - lo["PS_corner"]) @ e)
        Lc = float(np.linalg.norm(lo["AS_corner"] - lo["PS_corner"]))
        off_centroid = float((up["centroid"] - lo["centroid"]) @ e) if lo["kind"] == "body" else np.nan
        dh_a = float((up["AI"] - lo["AS"]) @ nrm)
        dh_p = float((up["PI"] - lo["PS"]) @ nrm)
        mid_u, mid_l = (up["PI"] + up["AI"]) / 2, (lo["PS"] + lo["AS"]) / 2
        pct = 100.0 * off / max(L, 1e-9)
        flags = []
        if abs(off - off_corner) > max(agree_mm, 0.1 * L):
            flags.append("corner_and_edge_methods_disagree")
        if not (0.6 * d_med <= L <= 1.5 * d_med):
            flags.append("lower_endplate_length_atypical")
        if min(dh_a, dh_p) <= 0:
            flags.append("disc_height_not_positive")
        if min(up["score"], lo["score"] if lo["kind"] == "body" else 1.0) < 0.8:
            flags.append("body_shape_poor")
        if set(up.get("edge_fallback", [])) & {"PI", "AI"} or set(lo.get("edge_fallback", [])) & {"PS", "AS"}:
            flags.append("edge_fit_fallback_to_corner")
        abnormal = abs(off) >= min_slip_mm
        rows.append({
            "_lo": lo, "_up": up, "offset_mm": off, "slip_mm": abs(off), "slip_pct": pct,
            "direction": ("antero" if off > 0 else "retro") if abnormal else "none",
            "meyerding_grade": _meyerding(pct) if abnormal else 0,
            "lower_endplate_mm": L, "disc_height_mm": 0.5 * (dh_a + dh_p),
            "disc_height_anterior_mm": dh_a, "disc_height_posterior_mm": dh_p,
            "disc_height_middle_mm": float(np.linalg.norm(mid_u - mid_l)),
            "offset_corner_mm": off_corner, "slip_pct_corner": 100 * off_corner / max(Lc, 1e-9),
            "offset_centroid_mm": off_centroid, "abnormal": bool(abnormal),
            "qc": "ok" if not flags else "check", "qc_flags": flags,
            "_disc_centre": (mid_u + mid_l) / 2,
        })
    return rows


def _resolve_series(obj, series) -> DicomSeries:
    if isinstance(obj, DicomSeries):
        return obj
    if isinstance(obj, DicomStudy):
        if series is not None:
            return obj[series]
        sag = obj.diagnostic(plane="SAG")
        if not sag:
            raise ValueError("no sagittal diagnostic series in the study; pass `series`")
        pref = {"T1": 0, "other": 1, "PD": 2, "fluid_sensitive": 3}
        return sorted(sag, key=lambda s: (pref.get(s.weighting, 4), -len(s)))[0]
    raise TypeError("pass a DicomSeries or a DicomStudy")


@register_function(
    aliases=["脊柱滑脱测量", "椎体滑移", "腰椎滑脱分级", "spondylolisthesis measurement", "vertebral slip",
             "meyerding grade", "spine alignment", "anterolisthesis retrolisthesis", "taillard index"],
    category="radiology",
    description="Segment vertebral bodies and sacrum on a sagittal MR/CT series, label levels from the sacrum "
                "upward, and measure per disc level the antero/retrolisthesis offset (mm, patient coordinates), "
                "slip percent of the lower endplate (Taillard), Meyerding grade, disc height, landmarks and QC",
    requires={},
    produces={},
    auto_fix="none",
    examples=["a = ov.radiology.measure_spine_alignment(st)  # picks the sagittal T1 series and mid slice",
              "a['table'][['level', 'offset_mm', 'slip_pct', 'meyerding_grade', 'direction', 'qc']]",
              "ov.radiology.ledger_add_measurement('ledger.json', st, a)"],
)
def measure_spine_alignment(series: Union[DicomSeries, DicomStudy], instance: Optional[int] = None, *,
                            study_series: Optional[Union[str, int]] = None,
                            level_names: Sequence[str] = DEFAULT_LEVEL_NAMES,
                            min_slip_mm: float = 3.0, body_size_mm: Tuple[float, float] = (14.0, 65.0),
                            shading_mm: float = 25.0, n_levels: int = 12, min_extent: float = 0.78,
                            min_solidity: float = 0.90, max_aspect: float = 2.3, max_turn_deg: float = 50.0,
                            min_sacrum_elongation: float = 1.6,
                            spacing: Tuple[float, float] = (0.75, 1.9), search: Optional[int] = None,
                            agree_mm: float = 2.0, decimals: int = 1) -> Dict:
    """Per-level vertebral alignment on a sagittal series.

    Parameters
    ----------
    series
        A sagittal :class:`DicomSeries`, or a :class:`DicomStudy` (the
        sagittal diagnostic series is chosen: T1 first, as bodies and discs
        separate best there; or pass ``study_series``).
    instance
        Mid-sagittal slice to measure. ``None``: the slices within ``search``
        of the series centre (default: the middle half) are segmented and
        the one with the longest, best-shaped column is measured.
    level_names
        Names from the sacrum upward, used only when the sacrum is identified.
    min_slip_mm
        Offset at or above which a level is reported as abnormal.
    body_size_mm, min_extent, min_solidity, max_aspect
        Acceptance of a component as a vertebral body: side lengths within
        ``body_size_mm``, area / bounding-rectangle area ≥ ``min_extent``,
        convexity ≥ ``min_solidity``, long / short side ≤ ``max_aspect``.
    shading_mm
        Scale of the local-mean division that removes coil shading.
    min_sacrum_elongation
        The component below the lowest body is taken as the sacrum when its
        craniocaudal extent is at least this many median body heights
        (``label_confidence`` ``high`` from 2.4).
    max_turn_deg, spacing
        Chain constraints: bend between consecutive inter-body vectors, and
        centre spacing relative to body size.
    agree_mm
        Tolerance between the edge-fit and corner-point offsets before the
        level's ``qc`` becomes ``"check"``.

    Returns
    -------
    dict with ``found``; ``series`` / ``series_instance_uid``; ``instance`` /
    ``sop_instance_uid`` (measured slice); ``sacrum_found``,
    ``label_confidence`` (``high`` / ``medium`` / ``low`` / ``none``);
    ``levels`` (list of dicts, also as ``table`` DataFrame) with ``level``,
    ``upper`` / ``lower``, ``offset_mm`` (signed, + = upper body anterior),
    ``slip_mm``, ``slip_pct`` (of the lower endplate), ``direction``
    (``antero`` / ``retro`` / ``none``), ``meyerding_grade`` (0 = below
    ``min_slip_mm``), ``lower_endplate_mm``, ``disc_height_mm`` (mean of
    anterior / posterior), ``disc_height_anterior_mm``,
    ``disc_height_posterior_mm``, ``disc_height_middle_mm``,
    ``offset_corner_mm`` / ``offset_centroid_mm`` (cross-checks),
    ``xyz_mm`` (disc centre), ``landmarks_xyz_mm`` and ``landmarks_rowcol``
    (``upper_PI``, ``upper_AI``, ``lower_PS``, ``lower_AS``), ``abnormal``,
    ``qc`` (``ok`` / ``check``), ``qc_flags``, ``n_slices_agree``; ``bodies``
    (label, corners in mm); ``slices`` (per-slice summary); ``warnings``.
    """
    s = _resolve_series(series, study_series)
    ids = s.instance_numbers
    if instance is None:
        k0 = (len(ids) - 1) / 2
        w = max(1, int(round(len(ids) / 4))) if search is None else int(search)
        cand = [i for k, i in enumerate(ids) if abs(k - k0) <= w + 0.01]
    else:
        cand = [int(instance)]
    i0 = s.instance(cand[0])
    ps2 = np.asarray(i0.ps, float)
    sup, post = _frame(i0)
    kw = dict(size_mm=body_size_mm, shading_mm=shading_mm, n_levels=n_levels, min_extent=min_extent,
              min_solidity=min_solidity, max_aspect=max_aspect, max_turn_deg=max_turn_deg, spacing=spacing,
              min_sacrum_elongation=min_sacrum_elongation)
    per = {}
    for i in cand:
        try:
            per[i] = _segment_slice(np.asarray(s.pixels(i), float), ps2, sup, post, **kw)
        except Exception as e:  # noqa: BLE001 - keep other slices
            per[i] = {"bodies": [], "sacrum": None, "error": str(e)}

    def quality(i):
        g = per[i]
        nb = len(g["bodies"]) + (1 if g["sacrum"] is not None else 0)
        q = float(np.mean([b["score"] for b in g["bodies"]])) if g["bodies"] else 0.0
        dmean = float(np.mean([b["d"] for b in g["bodies"]])) if g["bodies"] else 0.0
        # mid-sagittal: most bodies, best shapes, deepest bodies (elliptic cross-sections), then centre
        return (g["sacrum"] is not None, nb, round(q, 1), round(dmean), -abs(ids.index(i) - (len(ids) - 1) / 2))
    best = max(cand, key=quality)
    seg = per[best]
    slices = [{"instance": int(i), "n_bodies": len(per[i]["bodies"]), "sacrum": per[i]["sacrum"] is not None,
               "error": per[i].get("error")} for i in cand]
    if len(seg["bodies"]) < 1 or (len(seg["bodies"]) < 2 and seg["sacrum"] is None):
        return {"found": False, "series": s.description, "series_instance_uid": s.uid, "slices": slices,
                "warnings": ["fewer than two vertebrae found on any searched slice"]}
    warn = []
    sac = seg["sacrum"]
    if sac is not None:
        conf = "high" if sac["elongation"] >= 2.4 else "medium"
        names = list(level_names)
        if len(names) < len(seg["bodies"]) + 1:
            names += [f"V{k}" for k in range(len(names), len(seg["bodies"]) + 1)]
        body_names = names[1:len(seg["bodies"]) + 1]
        lowest = names[0]
    else:
        conf = "none"
        body_names = [f"V{k}" for k in range(1, len(seg["bodies"]) + 1)]
        lowest = None
        warn.append("sacrum not identified: levels are relative (V1 = lowest body found), not anatomical")
    if sac is not None and conf != "high":
        warn.append("sacrum identification uncertain (short component below the lowest body)")
    warn.append("numbering from the sacrum assumes no lumbosacral transitional vertebra; confirm on a "
                "larger field of view when the level matters")
    labels = ([lowest] if sac is not None else []) + body_names
    rows = _level_rows(seg, min_slip_mm=min_slip_mm, agree_mm=agree_mm)

    # neighbouring slices: is the same abnormality seen there?
    def offsets(i):
        g = per[i]
        if not g["bodies"]:
            return []
        return [r["offset_mm"] for r in _level_rows(g, min_slip_mm=min_slip_mm, agree_mm=agree_mm)]
    ref_n = len(rows)
    other = {i: offsets(i) for i in cand if i != best}
    inst = s.instance(best)

    def to_xyz(p):
        return np.round(s.pixel_to_patient(best, p[0] / ps2[0], p[1] / ps2[1]), decimals).tolist()

    def to_rc(p):
        return [round(float(p[0] / ps2[0]), 1), round(float(p[1] / ps2[1]), 1)]

    levels = []
    for k, r in enumerate(rows):
        lo_name, up_name = labels[k], labels[k + 1]
        agree = sum(1 for v in other.values() if len(v) == ref_n and r["abnormal"]
                    and abs(v[k]) >= min_slip_mm and np.sign(v[k]) == np.sign(r["offset_mm"]))
        lm = {"upper_PI": r["_up"]["PI"], "upper_AI": r["_up"]["AI"], "lower_PS": r["_lo"]["PS"],
              "lower_AS": r["_lo"]["AS"]}
        row = {"level": f"{up_name}-{lo_name}", "upper": up_name, "lower": lo_name}
        for key in ("offset_mm", "slip_mm", "slip_pct", "lower_endplate_mm", "disc_height_mm",
                    "disc_height_anterior_mm", "disc_height_posterior_mm", "disc_height_middle_mm",
                    "offset_corner_mm", "slip_pct_corner", "offset_centroid_mm"):
            v = r[key]
            row[key] = None if v is None or not np.isfinite(v) else round(float(v), decimals)
        row.update({"direction": r["direction"], "meyerding_grade": r["meyerding_grade"],
                    "abnormal": r["abnormal"], "qc": r["qc"], "qc_flags": list(r["qc_flags"]),
                    "n_slices_agree": int(agree),
                    "xyz_mm": to_xyz(r["_disc_centre"]),
                    "landmarks_xyz_mm": {k_: to_xyz(v) for k_, v in lm.items()},
                    "landmarks_rowcol": {k_: to_rc(v) for k_, v in lm.items()}})
        # distances recomputed from the rounded patient coordinates (checker-consistent)
        A, B = np.asarray(row["landmarks_xyz_mm"]["upper_PI"]), np.asarray(row["landmarks_xyz_mm"]["lower_PS"])
        Cc = np.asarray(row["landmarks_xyz_mm"]["lower_AS"])
        ee = (Cc - B) / max(np.linalg.norm(Cc - B), 1e-9)
        row["offset_mm"] = round(float((A - B) @ ee), decimals)
        row["slip_mm"] = abs(row["offset_mm"])
        row["lower_endplate_mm"] = round(float(np.linalg.norm(Cc - B)), decimals)
        row["slip_pct"] = round(100 * row["offset_mm"] / max(row["lower_endplate_mm"], 1e-9), decimals)
        levels.append(row)
    bodies_out = []
    chain = ([sac] if sac is not None else []) + seg["bodies"]
    for name, b in zip(labels, chain):
        bodies_out.append({"label": name, "kind": b["kind"],
                           "corners_xyz_mm": {c: to_xyz(b[c]) for c in ("PS", "AS", "PI", "AI")},
                           "centre_xyz_mm": to_xyz(b["c"]), "height_mm": round(b["h"], decimals),
                           "depth_mm": round(b["d"], decimals)})
    tab = pd.DataFrame(levels)
    return {"found": True, "series": s.description, "series_instance_uid": s.uid, "plane": s.plane,
            "weighting": s.weighting, "instance": int(best), "sop_instance_uid": inst.sop_uid,
            "sacrum_found": sac is not None, "label_confidence": conf, "min_slip_mm": float(min_slip_mm),
            "method": "posterior-wall / endplate line intersections (virtual corners)",
            "levels": levels, "table": tab, "bodies": bodies_out, "slices": slices, "warnings": warn}
