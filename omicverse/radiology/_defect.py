r"""Defect of a band-like structure measured against the band's own surroundings.

A tendon, ligament or fascia seen lengthwise is a dark band on an anatomic
(T1 / PD) image. Where it is torn, the band's low signal is lost and a
fluid-sensitive image is bright. :func:`measure_band_defect` measures such a
defect along a reader-supplied course and is built to tolerate the things that
made single-profile thresholds unreliable on real studies:

* **receive-coil shading** — surface coils make the same tissue two or three
  times brighter at one end of the field of view than at the other, so a band
  level taken over the whole course is wrong at both ends. Every anatomic
  pixel is divided by the local bright-tissue envelope (fat / marrow within a
  few millimetres), which leaves band darkness on a common 0–1 scale;
* **which slice** — a band a few millimetres thick is cut squarely by one or
  two slices; neighbours graze it and blur the defect. The slices on which the
  traced band is most conspicuous (darkest against its surroundings) are
  measured and the median of those is reported, not the median of all
  neighbours;
* **where the defect ends** — the two stumps of a torn tendon differ (one is
  usually thickened and darker), so each end is placed where the abnormality
  falls half-way back to the level of *its own* side (two-sided half-maximum),
  which keeps frayed, tapering stump ends with the stump;
* **false defects at the attachment** — where the course meets bone or fat at
  the reference end the trace leaves the band and the signal rises; a run that
  reaches the reference end is not a gap between two stumps and is never
  chosen.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._registry import register_function
from ._dicom import DicomSeries

__all__ = ["band_flatfield", "measure_band_defect"]


def _body_mask(img: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    # air is near zero; dark tissue (tendon, cortex) is a few percent of the bright tissue at least
    m = img > 0.04 * np.percentile(img, 99)
    m = ndi.binary_closing(m, iterations=3)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return ndi.binary_fill_holes(lab == (int(np.argmax(sizes)) + 1))


@register_function(
    aliases=["局部亮度归一化", "线圈不均匀校正", "flat field mri", "local envelope normalisation",
             "coil shading correction"],
    category="radiology",
    description="Divide an MR / CT slice by its local bright-tissue envelope (inside the body) so that "
                "dark structures read on a common 0-1 scale across the field of view, independent of "
                "receive-coil shading",
    requires={},
    produces={},
    auto_fix="none",
    examples=["n = ov.radiology.band_flatfield(st['Sag T1'].pixels(11), 0.35)"],
)
def band_flatfield(img: np.ndarray, pixel_mm: float, *, envelope_mm: float = 12.0) -> np.ndarray:
    """Local-envelope normalised image (1 ≈ surrounding bright tissue).

    The envelope is the maximum of the lightly smoothed image within
    ``envelope_mm``, smoothed again; outside the body (air) the result is 1 so
    that air never looks like a dark band.
    """
    from scipy import ndimage as ndi
    img = np.asarray(img, float)
    body = _body_mask(img)
    x = np.where(body, img, 0.0)
    w = max(3, int(round(envelope_mm / pixel_mm)))
    env = ndi.maximum_filter(ndi.gaussian_filter(x, 1.0 / pixel_mm), size=w)
    env = ndi.gaussian_filter(env, w / 2.0)
    return np.where(body, img / np.maximum(env, 1e-6), 1.0)


def _trace(n: np.ndarray, ps: float, wp, corridor_mm: float, stiffness: float, band_mm: float) -> np.ndarray:
    from scipy import ndimage as ndi
    from skimage.graph import route_through_array
    ns = ndi.gaussian_filter(n, 0.75 * band_mm / ps)
    rr, cc = np.mgrid[0:n.shape[0], 0:n.shape[1]]
    path: List[Tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(wp[:-1], wp[1:]):
        a = np.array([r0, c0]) * ps
        ab = np.array([r1, c1]) * ps - a
        P = np.stack([rr * ps, cc * ps], -1) - a
        t = np.clip((P @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
        dist = np.linalg.norm(P - t[..., None] * ab, axis=-1)
        cost = np.where(dist <= corridor_mm, np.clip(ns, 0, 2) + 0.02 + stiffness * dist / corridor_mm, 1e6)
        seg, _ = route_through_array(cost, (int(round(r0)), int(round(c0))), (int(round(r1)), int(round(c1))),
                                     fully_connected=True, geometric=True)
        seg = [tuple(p) for p in seg]
        path.extend(seg if not path else seg[1:])
    return np.asarray(path, float)


def _extend(wp, which: str, px: float, shape) -> list:
    if which == "first":
        p0, p1 = np.asarray(wp[0], float), np.asarray(wp[1], float)
    else:
        p0, p1 = np.asarray(wp[-1], float), np.asarray(wp[-2], float)
    d = (p0 - p1) / (np.linalg.norm(p0 - p1) + 1e-9)
    q = tuple(np.clip(p0 + d * px, [0, 0], [shape[0] - 1, shape[1] - 1]))
    return [q] + list(wp) if which == "first" else list(wp) + [q]


def _rolling_median(v: np.ndarray, k: int) -> np.ndarray:
    return pd.Series(v).rolling(2 * k + 1, center=True, min_periods=1).median().to_numpy()


def _profile(anat: DicomSeries, fluid: DicomSeries, i: int, wp, n: np.ndarray, *, corridor_mm, stiffness,
             band_mm) -> Dict:
    from scipy import ndimage as ndi
    ps = float(np.mean(anat.instance(i).ps))
    P = _trace(n, ps, wp, corridor_mm, stiffness, band_mm)
    T = np.gradient(P, axis=0) if len(P) > 2 else np.tile(P[-1] - P[0], (len(P), 1))
    T = ndi.gaussian_filter1d(T, max(1, int(3 / ps)), axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    off = np.arange(-band_mm / ps, band_mm / ps + 1e-9, 0.5)
    a = ndi.map_coordinates(n, [P[:, :1] + off * N[:, :1], P[:, 1:] + off * N[:, 1:]],
                            order=1, mode="nearest").mean(1)
    xyz = np.array([anat.pixel_to_patient(i, r, c) for r, c in P])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    f, cache = [], {}
    for p in xyz:
        m = fluid.patient_to_pixel(p)
        k = m["instance_number"]
        if k not in cache:
            cache[k] = np.asarray(fluid.pixels(k), float)
        fps = float(np.mean(fluid.instance(k).ps))
        h = max(1, int(round(band_mm / fps)))
        ri, ci = int(round(m["row"])), int(round(m["col"]))
        img = cache[k]
        ok = 0 <= ri < img.shape[0] and 0 <= ci < img.shape[1] and m["out_of_plane_mm"] < 5 * fps + 5
        f.append(float(img[max(0, ri - h):ri + h + 1, max(0, ci - h):ci + h + 1].mean()) if ok else np.nan)
    return {"P": P, "xyz": xyz, "arc": arc, "a": a, "f": np.asarray(f, float)}


@register_function(
    aliases=["条带缺损两端测量", "肌腱撕裂间隙", "撕裂到止点距离", "band defect measurement",
             "tendon tear gap", "tear gap and distance to insertion"],
    category="radiology",
    description="Measure a defect (tear gap) of a band-like structure in patient mm: trace the band on "
                "coil-shading-normalised anatomic slices, pick the slice where the band is most "
                "conspicuous, find the run where the band's darkness is lost and the fluid-sensitive "
                "signal is high, place each end at the half-way level of its own side, and give the "
                "distance from the defect midpoint to the reference (attachment) end",
    requires={},
    produces={},
    auto_fix="none",
    examples=["d = ov.radiology.measure_band_defect(st['Sag T1'], st['Sag STIR'], 11, "
              "[(60, 385), (200, 380), (330, 390), (425, 421)])",
              "d['measurement_mm'], d['reference_distance_mm'], d['landmarks_xyz_mm']"],
)
def measure_band_defect(anat: DicomSeries, fluid: DicomSeries, instance: int,
                        waypoints: Sequence[Tuple[float, float]], *, search: int = 2, pool: int = 3,
                        near_xyz: Optional[Sequence[float]] = None, corridor_mm: float = 6.0,
                        stiffness: float = 0.5, band_mm: float = 2.0, envelope_mm: float = 12.0,
                        smooth_mm: float = 2.5, screen: float = 1.5, level: float = 0.5,
                        baseline_mm: float = 40.0, end_margin_mm: float = 5.0,
                        extend_mm: float = 30.0, decimals: int = 1) -> Dict:
    """Defect length and midpoint-to-reference distance of a band, in patient mm.

    Parameters
    ----------
    anat, fluid
        Anatomic series on which the band is dark (T1 / PD) and a
        fluid-sensitive series (STIR / fat-suppressed PD / T2); the fluid
        series is sampled through patient coordinates, so it may have its own
        geometry.
    instance
        Slice of ``anat`` the waypoints were placed on.
    waypoints
        ``(row, col)`` pixels along the structure: the first on intact band
        beyond the defect, the last on the landmark the reference distance is
        measured to (e.g. where the structure attaches to bone).
    search, pool
        Slices ``instance ± search`` are traced with the same waypoints and
        ranked by how conspicuous the band is on them (lowest lower-quartile
        normalised band signal); of the ``pool`` most conspicuous slices with
        a defect, the one with the median defect length is reported (one
        slice alone is noisy; grazing slices are left out). ``search=0``
        measures ``instance`` only.
    near_xyz
        Optional patient point near the lesion (e.g. a survey location); when
        several runs qualify, the one closest to it is measured. Without it the
        most pronounced run (length × mean abnormality) is used.
    corridor_mm, stiffness
        The trace is a minimum-cost path through the normalised image that may
        leave the waypoint polyline by at most ``corridor_mm``.
    band_mm
        Half-width over which the band signal is averaged across the trace.
    smooth_mm, screen, level, baseline_mm
        The abnormality ``sqrt(anat_rel * fluid_rel)`` (both relative to the
        band's intact level) is smoothed over ``smooth_mm``; runs ≥ ``screen``
        are candidates; each end is where the abnormality falls to
        ``baseline + level * (peak - baseline)``, the baseline being the lowest
        abnormality within ``baseline_mm`` on that side.
    end_margin_mm
        Runs ending within this distance of the reference end are not defects
        (the trace left the band at the attachment).
    extend_mm
        If the defect reaches the first waypoint, the trace is prolonged there
        by this much (up to twice) and measured again.

    Returns
    -------
    dict with ``instance``, ``sop_instance_uid``, ``landmarks_xyz_mm`` (the two
    defect ends, rounded), ``landmarks_rowcol``, ``measurement_mm``,
    ``midpoint_xyz_mm``, ``reference_xyz_mm`` / ``reference_rowcol`` (trace end
    = last waypoint), ``reference_distance_mm`` (recomputed from the rounded
    coordinates), ``peak`` (abnormality at the defect centre),
    ``slices`` (conspicuity and defect per traced slice), ``warnings`` and
    ``found`` (False when no defect run qualifies).
    """
    ids = anat.instance_numbers
    k0 = ids.index(int(instance))
    cand = ids[max(0, k0 - search):k0 + search + 1]
    wp0 = [tuple(map(float, w)) for w in waypoints]
    if len(wp0) < 2:
        raise ValueError("need at least two waypoints")

    def measure_slice(i: int) -> Dict:
        ps = float(np.mean(anat.instance(i).ps))
        n = band_flatfield(anat.pixels(i), ps, envelope_mm=envelope_mm)
        wp = list(wp0)
        extended = 0
        out: Dict = {"instance": int(i)}
        for attempt in range(3):
            pr = _profile(anat, fluid, i, wp, n, corridor_mm=corridor_mm, stiffness=stiffness, band_mm=band_mm)
            arc, L = pr["arc"], float(pr["arc"][-1])
            step = float(np.nanmedian(np.diff(arc))) if len(arc) > 1 else 1.0
            kk = max(1, int(round(smooth_mm / max(step, 1e-6))))
            a = _rolling_median(pr["a"], kk)
            f = _rolling_median(pr["f"], kk)
            a0 = float(np.nanpercentile(a, 25))
            intact = a <= np.nanpercentile(a, 40)
            f0 = float(np.nanmedian(f[intact])) if np.isfinite(f[intact]).any() else float(np.nanmedian(f))
            c = np.sqrt(np.clip(a / max(a0, 1e-6), 1e-3, None) * np.clip(f / max(f0, 1e-6), 1e-3, None))
            c = np.nan_to_num(c, nan=1.0)
            out.update({"conspicuity": a0, "trace_length_mm": L})
            flag = c >= screen
            runs, s = [], None
            for j, v in enumerate(flag):
                if v and s is None:
                    s = j
                if (not v or j == len(flag) - 1) and s is not None:
                    runs.append((s, j if v else j - 1))
                    s = None
            runs = [(p, q) for p, q in runs if arc[q] - arc[p] >= 2.0]
            out["run_at_reference_end"] = any(L - arc[q] <= end_margin_mm for p, q in runs)
            runs = [(p, q) for p, q in runs if L - arc[q] > end_margin_mm]
            if not runs:
                out["found"] = False
                return out
            xyz = pr["xyz"]
            if near_xyz is not None:
                d = [float(np.min(np.linalg.norm(xyz[p:q + 1] - np.asarray(near_xyz, float), axis=1)))
                     for p, q in runs]
                p, q = runs[int(np.argmin(d))]
            else:
                p, q = max(runs, key=lambda t: (arc[t[1]] - arc[t[0]]) * float(c[t[0]:t[1] + 1].mean()))
            pk = p + int(np.argmax(c[p:q + 1]))
            peak = float(c[pk])
            left = np.where((arc < arc[pk]) & (arc >= arc[pk] - baseline_mm))[0]
            right = np.where((arc > arc[pk]) & (arc <= arc[pk] + baseline_mm))[0]
            bl = float(np.min(c[left])) if len(left) else 1.0
            br = float(np.min(c[right])) if len(right) else 1.0
            tl, tr = bl + level * (peak - bl), br + level * (peak - br)
            lo = pk
            while lo > 0 and c[lo - 1] >= tl:
                lo -= 1
            hi = pk
            while hi < len(c) - 1 and c[hi + 1] >= tr:
                hi += 1
            if arc[lo] <= 3.0 and attempt < 2:
                wp = _extend(wp, "first", extend_mm / ps, n.shape)
                extended += 1
                continue
            break
        A = np.round(xyz[lo], decimals)
        B = np.round(xyz[hi], decimals)
        R = np.round(xyz[-1], decimals)
        M = (A + B) / 2
        out.update({
            "found": True, "sop_instance_uid": anat.instance(i).sop_uid,
            "landmarks_xyz_mm": [A.tolist(), B.tolist()],
            "landmarks_rowcol": [[int(pr["P"][lo, 0]), int(pr["P"][lo, 1])], [int(pr["P"][hi, 0]), int(pr["P"][hi, 1])]],
            "measurement_mm": round(float(np.linalg.norm(A - B)), decimals),
            "midpoint_xyz_mm": np.round(M, decimals).tolist(),
            "reference_xyz_mm": R.tolist(),
            "reference_rowcol": [int(pr["P"][-1, 0]), int(pr["P"][-1, 1])],
            "reference_distance_mm": round(float(np.linalg.norm(M - R)), decimals),
            "peak": round(peak, 2), "beyond_defect_mm": round(float(L - arc[hi]), 1),
            "before_defect_mm": round(float(arc[lo]), 1), "trace_extended": extended,
        })
        return out

    per = []
    for i in cand:
        try:
            per.append(measure_slice(int(i)))
        except Exception as e:  # noqa: BLE001 - waypoint outside a slice etc.
            per.append({"instance": int(i), "found": False, "error": str(e)})
    traced = [r for r in per if "conspicuity" in r]
    if not traced:
        return {"found": False, "slices": per, "warnings": ["the band could not be traced on any slice"]}
    ranked = sorted(traced, key=lambda r: r["conspicuity"])
    found = [r for r in ranked[:max(1, pool)] if r.get("found")]
    if found:
        found = sorted(found, key=lambda r: r["measurement_mm"])
        best = found[(len(found) - 1) // 2]
    else:
        best = ranked[0]
    res = dict(best)
    warn = []
    if not best.get("found"):
        warn.append(f"no defect run on the {pool} most conspicuous slices")
        if any(r.get("run_at_reference_end") for r in ranked[:max(1, pool)]):
            warn.append("an abnormal run reaches the last waypoint: the last waypoint is probably on or just "
                        "beyond the defect, not on the attachment - move it to where the structure meets the bone")
    else:
        if best["beyond_defect_mm"] < 10.0:
            warn.append(f"the reference end lies only {best['beyond_defect_mm']} mm beyond the defect: check "
                        "that the last waypoint is on the attachment and not on the defect's edge")
        if best["before_defect_mm"] < 5.0:
            warn.append("the defect starts at the first waypoint: put the first waypoint on intact band")
    res["warnings"] = warn
    res["slices"] = [{k: r.get(k) for k in ("instance", "conspicuity", "found", "measurement_mm",
                                            "reference_distance_mm", "error") if k in r} for r in per]
    return res
