r"""Deterministic reading survey of a DICOM study.

``structure_survey`` turns "look at the whole study, then at every structure
on every sequence" into a fixed list of image sheets with a manifest, so the
reading procedure does not depend on which images an agent decides to draw.

Two phases, same function:

1. **Regional survey** (no ``structures``): for every imaging plane, the
   fluid-sensitive series (fluid / oedema bright — where most lesions show) is
   split into overlapping regions of the body's field of view and each region
   is montaged over all its slices at about native resolution, every tile
   carrying a lettered cell grid (``inst 11 B2``).
2. **Structure sheets** (``structures=[...]``): for each named structure with
   a location — a patient point, or a ``(sheet, instance, cell)`` read off a
   regional sheet — a fixed crop around it is montaged over the consecutive
   slices through it on **every** diagnostic series (all planes, anatomic and
   fluid-sensitive), so each structure is seen in cross-section and along its
   course, on both contrasts.

No anatomy is encoded: structure names come from the caller (e.g. the task's
own vocabulary) and locations come from the images (the reader's answer on a
regional sheet). The number of sheets is bounded (``max_sheets``) and image
sizes are capped, which bounds vision latency.

``sheet_cell_to_patient`` converts a reader's ``(instance, cell)`` answer to
patient mm; ``ledger_concordance`` summarises a findings ledger (how many
slices / series agree on each abnormality of each structure).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._registry import register_function
from ._dicom import DicomSeries, DicomStudy
from ._render import montage, region_sheets

__all__ = ["structure_survey", "sheet_cell_to_patient", "ledger_concordance", "ledger_add_read",
           "ledger_hotspots", "ledger_principal", "draft_principal",
           "DEFAULT_SEVERITY_TERMS"]

_PLANE_ORDER = {"SAG": 0, "COR": 1, "AX": 2}
_W_ORDER = {"fluid_sensitive": 0, "T1": 1, "PD": 2, "other": 3}


def _ordered_series(study: DicomStudy) -> List[DicomSeries]:
    """Diagnostic series in a fixed order: plane (SAG, COR, AX), then
    fluid-sensitive before anatomic, then series number."""
    ser = [s for s in study.series.values() if not s.is_localizer and s.plane in _PLANE_ORDER]
    return sorted(ser, key=lambda s: (_PLANE_ORDER[s.plane], _W_ORDER.get(s.weighting, 9),
                                      s.number if s.number is not None else 1e9, s.uid))


def _slice_spacing(s: DicomSeries) -> float:
    pos = s.slice_positions()["position_mm"].to_numpy()
    return float(np.median(np.abs(np.diff(np.sort(pos))))) if len(pos) > 1 else 1.0


def _cell_centre_rc(crop, cells: int, cell: str):
    col = ord(cell[0].upper()) - 65
    row = int(cell[1:]) - 1
    if not (0 <= col < cells and 0 <= row < cells):
        raise ValueError(f"cell {cell!r} outside a {cells}x{cells} grid")
    r0, r1, c0, c1 = crop
    return (r0 + (row + 0.5) * (r1 - r0) / cells, c0 + (col + 0.5) * (c1 - c0) / cells)


@register_function(
    aliases=["网格单元转坐标", "sheet cell to patient", "reader cell to mm", "montage cell location"],
    category="radiology",
    description="Convert a reader's (instance, grid cell) answer on a survey sheet into patient mm",
    requires={},
    produces={},
    auto_fix="none",
    examples=["xyz = ov.radiology.sheet_cell_to_patient(st, sheet, 11, 'B2')"],
)
def sheet_cell_to_patient(study: DicomStudy, sheet: Dict, instance: int, cell: str) -> List[float]:
    """Patient coordinates (mm) of the centre of ``cell`` on tile ``instance``
    of a survey sheet (an entry of the manifest returned by
    :func:`structure_survey`)."""
    s = study[sheet["series_instance_uid"]]
    r, c = _cell_centre_rc(sheet["crop"], int(sheet.get("cells") or 3), cell)
    return np.round(s.pixel_to_patient(int(instance), r, c), 1).tolist()


def _locate(study: DicomStudy, spec: Dict, manifest: Sequence[Dict]) -> Optional[np.ndarray]:
    if spec.get("xyz_mm") is not None:
        return np.asarray(spec["xyz_mm"], float)
    if spec.get("sheet") is not None and spec.get("instance") is not None:
        ref = spec["sheet"]
        sh = ref if isinstance(ref, dict) else next(
            (m for m in manifest if m.get("id") == ref or m.get("path") == ref
             or os.path.basename(m.get("path", "")) == os.path.basename(str(ref))), None)
        if sh is None:
            raise KeyError(f"sheet {ref!r} not in the manifest")
        return np.asarray(sheet_cell_to_patient(study, sh, spec["instance"], spec.get("cell", "B2")))
    if spec.get("series") is not None and spec.get("instance") is not None and spec.get("rowcol") is not None:
        r, c = spec["rowcol"]
        return study[spec["series"]].pixel_to_patient(int(spec["instance"]), r, c)
    return None


@register_function(
    aliases=["结构化阅片巡检", "全检查结构巡检", "structure survey", "deterministic radiology survey",
             "reading sheets for every structure", "mri survey manifest"],
    category="radiology",
    description="Fixed, bounded set of reading sheets for a DICOM study with a JSON manifest: a regional "
                "survey of each plane's fluid-sensitive series (cell-gridded, native resolution, all slices) "
                "and, for caller-named structures located on the images, consecutive-slice crops on every "
                "diagnostic series",
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "sheets = ov.radiology.structure_survey(st, 'survey')            # phase 1: regional survey",
        "sheets2 = ov.radiology.structure_survey(st, 'survey', structures=[\n"
        "    {'name': '<vocabulary term>', 'sheet': 'R03', 'instance': 11, 'cell': 'C1'}])  # phase 2",
    ],
)
def structure_survey(study: DicomStudy, out_dir: str = "survey", *,
                     structures: Optional[Sequence[Union[str, Dict]]] = None,
                     region_mm: float = 90.0, crop_mm: float = 80.0, depth_mm: float = 25.0,
                     cells: int = 3, max_sheets: int = 24, tile_px: int = 320,
                     max_tiles: int = 24, max_px: int = 1800) -> List[Dict]:
    """Write reading sheets + ``manifest.json`` and return the sheets in order.

    Parameters
    ----------
    structures
        ``None`` → phase 1 (regional survey). Otherwise a list of structure
        names (strings: recorded as *unlocalised*, nothing rendered) or dicts
        ``{'name': ..., <location>, 'crop_mm': .., 'depth_mm': ..}`` with the
        location given as ``xyz_mm`` (patient mm), or ``sheet`` (manifest id /
        path) + ``instance`` + ``cell`` from a regional sheet, or ``series`` +
        ``instance`` + ``rowcol``. Each located structure gets one sheet per
        diagnostic series: a ``crop_mm`` square around it on every slice within
        ``depth_mm`` of it.
    region_mm
        Phase-1 region size; enlarged automatically if the survey would exceed
        ``max_sheets``.
    max_sheets
        Upper bound on the sheets written by one call (vision latency). In
        phase 2 every located structure gets the same share
        (``max(2, max_sheets // n_structures)`` series), taken in the order
        SAG-fluid, COR-fluid, AX-fluid, SAG-anatomic, ...; the remaining
        (structure, series) pairs are listed in the manifest's ``skipped``.

    Returns
    -------
    list of dicts (also in ``<out_dir>/manifest.json`` / ``manifest_structures.json``):
    ``id``, ``kind`` ('region' | 'structure'), ``path``, ``series_instance_uid``,
    ``description``, ``plane``, ``weighting``, ``instances``, ``crop``,
    ``cells``, ``pixel_spacing_mm``, ``slice_spacing_mm``, ``directions``,
    ``centre_xyz_mm``, ``structure``.
    """
    os.makedirs(out_dir, exist_ok=True)
    ordered = _ordered_series(study)
    sheets: List[Dict] = []

    def _meta(s: DicomSeries, d: Dict) -> Dict:
        i0 = s.instances[0]
        d.update({"series_instance_uid": s.uid, "description": s.description, "plane": s.plane,
                  "weighting": s.weighting,
                  "pixel_spacing_mm": [round(float(i0.ps[0]), 4), round(float(i0.ps[1]), 4)],
                  "slice_spacing_mm": round(_slice_spacing(s), 3)})
        return d

    if structures is None:
        # one survey series per plane: fluid-sensitive (most slices) else first anatomic
        per_plane: Dict[str, DicomSeries] = {}
        for s in ordered:
            cur = per_plane.get(s.plane)
            if cur is None or (s.weighting == "fluid_sensitive" and
                               (cur.weighting != "fluid_sensitive" or len(s) > len(cur))):
                per_plane[s.plane] = s
        rmm = region_mm
        for _ in range(6):
            sheets = []
            for plane in sorted(per_plane, key=_PLANE_ORDER.get):
                s = per_plane[plane]
                i0 = s.instances[0]
                tp = int(min(tile_px, max(160, rmm / float(np.mean(i0.ps)))))
                res = region_sheets(s, region_mm=rmm, max_tiles=max_tiles, ncols=6, tile_px=tp,
                                    out_prefix=os.path.join(out_dir, f"region_{plane}"), cells=cells)
                for r in res:
                    sheets.append(_meta(s, {"kind": "region", "structure": None, **r}))
            if len(sheets) <= max_sheets:
                break
            rmm *= 1.3
        for k, sh in enumerate(sheets):
            sh["id"] = f"R{k + 1:02d}"
        man = os.path.join(out_dir, "manifest.json")
    else:
        prior = []
        mp = os.path.join(out_dir, "manifest.json")
        if os.path.exists(mp):
            prior = json.load(open(mp))["sheets"]
        unloc, skipped = [], []
        specs = []
        for spec in structures:
            spec = {"name": spec} if isinstance(spec, str) else dict(spec)
            p = _locate(study, spec, prior)
            if p is None:
                unloc.append(spec["name"])
            else:
                specs.append((spec, p))
        # Fair budget: every located structure gets the same number of series,
        # interleaving planes with fluid-sensitive first (SAG-fluid, COR-fluid,
        # AX-fluid, SAG-anatomic, ...), so no structure is crowded out by the
        # ones listed before it.
        by_plane: Dict[str, List[DicomSeries]] = {}
        for s in ordered:
            by_plane.setdefault(s.plane, []).append(s)
        inter: List[DicomSeries] = []
        for k in range(max((len(v) for v in by_plane.values()), default=0)):
            for plane in sorted(by_plane, key=_PLANE_ORDER.get):
                if k < len(by_plane[plane]):
                    inter.append(by_plane[plane][k])
        per = max(2, max_sheets // max(1, len(specs))) if specs else 0
        for spec, p in specs:
            cm = float(spec.get("crop_mm", crop_mm))
            dm = float(spec.get("depth_mm", depth_mm))
            used = 0
            for s in inter:
                if used >= per:
                    skipped.append({"structure": spec["name"], "series": s.description})
                    continue
                m = s.patient_to_pixel(p)
                if not m["inside"]:
                    continue
                pos = s.slice_positions()
                c = float(np.dot(p, s.normal))
                ids = pos.loc[(pos.position_mm - c).abs() <= dm + 1e-6, "instance_number"].tolist()
                if not ids:
                    ids = [m["instance_number"]]
                ids = ids[:max_tiles]
                inst = s.instance(m["instance_number"])
                hr = int(min(cm / 2 / inst.ps[0], inst.rows / 2))
                hc = int(min(cm / 2 / inst.ps[1], inst.cols / 2))
                # keep the full crop inside the image (shift instead of clipping)
                r0 = int(np.clip(round(m["row"]) - hr, 0, inst.rows - 2 * hr))
                c0 = int(np.clip(round(m["col"]) - hc, 0, inst.cols - 2 * hc))
                crop = (r0, r0 + 2 * hr, c0, c0 + 2 * hc)
                # at least native size, enlarged up to tile_px for small crops
                tp = int(max(tile_px, min(2 * hr, 450)))
                safe = "".join(ch if ch.isalnum() else "_" for ch in spec["name"])[:40]
                out = os.path.join(out_dir, f"struct_{len(sheets) + 1:02d}_{safe}_{s.plane}_"
                                            f"{''.join(ch for ch in s.description if ch.isalnum())[:12]}.png")
                res = montage(s, ids, crop=crop, ncols=min(6, len(ids)), tile_px=tp, out=out,
                              cells=cells, max_px=max_px)
                sheets.append(_meta(s, {"kind": "structure", "structure": spec["name"], "path": out,
                                        "instances": ids, "crop": crop, "cells": cells,
                                        "directions": res["directions"],
                                        "centre_xyz_mm": np.round(p, 1).tolist(),
                                        "centre_instance": m["instance_number"]}))
                used += 1
        for k, sh in enumerate(sheets):
            sh["id"] = f"S{k + 1:02d}"
        if skipped:
            print(f"structure_survey: {len(skipped)} (structure, series) pairs over the sheet budget "
                  f"({per} series per structure); listed in manifest 'skipped'")
        if unloc:
            print(f"structure_survey: not located (give sheet+instance+cell or xyz_mm): {unloc}")
        man = os.path.join(out_dir, "manifest_structures.json")
        with open(man, "w") as f:
            json.dump({"sheets": sheets, "unlocalised": unloc, "skipped": skipped}, f, indent=1)
        return sheets
    with open(man, "w") as f:
        json.dump({"sheets": sheets}, f, indent=1)
    return sheets


@register_function(
    aliases=["阅片记录一致性", "findings ledger concordance", "ledger concordance", "rank findings"],
    category="radiology",
    description="Summarise a findings ledger: per (structure, abnormality) the number of distinct slices "
                "and series that report it, concordance flag, and the slices / sheets involved",
    requires={},
    produces={},
    auto_fix="none",
    examples=["ov.radiology.ledger_concordance('ledger.json')"],
)
def ledger_concordance(ledger: Union[str, Sequence[Dict]], *, min_slices: int = 2,
                       min_series: int = 2) -> pd.DataFrame:
    """Concordance table of a findings ledger.

    The ledger is a list (or a JSON file of a list) of reads, one per
    (sheet, instance, structure): ``{'sheet', 'series', 'instance',
    'structure', 'abnormality', 'acute' (true/false/null), 'confidence'
    ('high'/'medium'/'low'), 'normal' (bool)}``. Reads with ``normal`` true
    are counted as contradicting reads of the same structure on the same
    instance. ``concordant`` is True when an abnormality is reported on at
    least ``min_slices`` distinct slices of one series or on at least
    ``min_series`` series.
    """
    rows = json.load(open(ledger)) if isinstance(ledger, str) else list(ledger)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c, d in (("normal", False), ("acute", None), ("confidence", "medium"), ("series", ""),
                 ("abnormality", ""), ("sheet", "")):
        if c not in df:
            df[c] = d
    df["normal"] = df["normal"].map(lambda v: bool(v) if v is not None and v == v else False)
    ab = df[~df.normal & (df.abnormality.astype(str) != "")]
    out = []
    for (st, a), g in ab.groupby(["structure", "abnormality"]):
        per_series = g.groupby("series")["instance"].nunique()
        same = df[(df.structure == st) & df.normal]
        contra = same.merge(g[["series", "instance"]], on=["series", "instance"]).shape[0]
        out.append({"structure": st, "abnormality": a,
                    "n_slices_max_series": int(per_series.max()), "n_series": int(per_series.size),
                    "n_reads": int(len(g)),
                    "acute_votes": int((g.acute == True).sum()),  # noqa: E712
                    "high_conf": int((g.confidence == "high").sum()),
                    "contradicting_reads_same_slice": int(contra),
                    "concordant": bool(per_series.max() >= min_slices or per_series.size >= min_series),
                    "series": sorted(map(str, per_series.index)),
                    "instances": sorted({(str(r.series), int(r.instance)) for r in g.itertuples()}),
                    "sheets": sorted(set(map(str, g.sheet)))})
    res = pd.DataFrame(out)
    return res.sort_values(["concordant", "n_series", "n_slices_max_series", "acute_votes"],
                           ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Ledger from reader text, and location hotspots
# --------------------------------------------------------------------------

import re as _re

#: Generic triage vocabulary for free-text reads (not anatomy, not diagnoses):
#: class 3 = loss of continuity, 2 = acute-type signal, 1 = other abnormal.
DEFAULT_SEVERITY_TERMS = {
    3: ["interrupt", "discontinu", "absent", "gap", "tear", "torn", "ruptur", "retract",
        "defect", "fractur", "cortical break", "replaced", "loss of the normal", "loss of normal",
        "not visible", "frayed", "stump", "disrupt"],
    2: ["oedema", "edema", "haematoma", "hematoma", "blurred", "bright core", "fluid-filled",
        "fluid filled", "hyperintense", "high signal", "bright signal"],
    1: ["fluid", "effusion", "thicken", "intermediate", "collection", "enlarged", "irregular",
        "mass", "cyst", "bright"],
}
_NEG = _re.compile(r"\b(no|not|without|absence of|negative for|nor)\b[^.;,]{0,40}$", _re.I)


def _severity(text: str, terms: Dict[int, List[str]]) -> int:
    t = text.lower()
    for level in sorted(terms, reverse=True):
        for w in terms[level]:
            for m in _re.finditer(_re.escape(w), t):
                if not _NEG.search(t[max(0, m.start() - 45):m.start()]):
                    return int(level)
    return 0


def _segments(text: str):
    """Split a reader's answer into (instance, segment text) pieces."""
    parts = list(_re.finditer(r"\binst(?:ance)?\.?\s*#?\s*(\d+)", text, _re.I))
    for k, m in enumerate(parts):
        end = parts[k + 1].start() if k + 1 < len(parts) else len(text)
        seg = text[m.end():end]
        # a paragraph break ends the tile's read (summaries / notes follow)
        seg = _re.split(r"\n\s*\n|\n\s*(?:\*\*|#)", seg, maxsplit=1)[0]
        yield int(m.group(1)), seg


@register_function(
    aliases=["阅片记录写入", "ledger add read", "parse vision read into ledger", "record survey read"],
    category="radiology",
    description="Parse a reader's free-text answer on a survey sheet into ledger entries (instance, cell, "
                "patient mm, text, severity class, confidence, normal flag) and append them to a JSON ledger",
    requires={},
    produces={},
    auto_fix="none",
    examples=["ov.radiology.ledger_add_read('ledger.json', st, sheet, vision_answer_text)"],
)
def ledger_add_read(ledger_path: str, study: DicomStudy, sheet: Dict, text: str, *,
                    source: Optional[str] = None,
                    severity_terms: Optional[Dict[int, List[str]]] = None) -> List[Dict]:
    """Append the reads in ``text`` (answer to a sheet of :func:`structure_survey`)
    to ``ledger_path``; returns the new entries.

    Every ``inst N`` segment becomes one entry per grid cell it names (or one
    entry at the tile centre if it names none). ``severity`` is the highest
    class of :data:`DEFAULT_SEVERITY_TERMS` found without a preceding negation
    (3 continuity loss, 2 acute-type signal, 1 other abnormal, 0 none);
    segments that start with 'none' / 'normal' are recorded as normal.
    The structure is left for the reader to name (``structure: null``).
    """
    terms = severity_terms or DEFAULT_SEVERITY_TERMS
    if isinstance(text, dict):
        text = text.get("analysis", "")
    text = str(text)
    try:
        js = json.loads(text)
        if isinstance(js, dict) and "analysis" in js:
            text = js["analysis"]
    except Exception:
        pass
    text = text.replace("\\n", "\n")
    cells = int(sheet.get("cells") or 3)
    valid = set(sheet["instances"])
    first = _re.search(r"\binst(?:ance)?\.?\s*#?\s*\d+", text, _re.I)
    header = text[:first.start()].strip(" *#:\n\t") if first else ""
    new = []
    for inst, seg in _segments(text):
        if inst not in valid:
            continue
        seg_clean = seg.strip(" :*-–—\n\t")
        normal = bool(_re.match(r"^(none|normal|no abnormal|unremarkable|nil)\b", seg_clean, _re.I))
        conf = next((c for c in ("high", "medium", "low") if c in seg_clean.lower()), "medium")
        sev = 0 if normal else _severity(seg_clean, terms)
        found = sorted(set(m.upper() for m in _re.findall(r"\b([A-Ca-c][1-3])\b", seg_clean)
                           if ord(m[0].upper()) - 65 < cells and int(m[1]) <= cells))
        for cell in (found or [None]):
            if cell is None:
                s = study[sheet["series_instance_uid"]]
                r0, r1, c0, c1 = sheet["crop"]
                xyz = np.round(s.pixel_to_patient(inst, (r0 + r1) / 2, (c0 + c1) / 2), 1).tolist()
            else:
                xyz = sheet_cell_to_patient(study, sheet, inst, cell)
            new.append({"sheet": sheet.get("id"), "series": sheet.get("description"),
                        "series_instance_uid": sheet.get("series_instance_uid"),
                        "plane": sheet.get("plane"), "weighting": sheet.get("weighting"),
                        "instance": inst, "cell": cell, "xyz_mm": xyz, "text": seg_clean[:400],
                        "severity": int(sev), "confidence": conf, "normal": normal or sev == 0,
                        "structure": sheet.get("structure"), "sheet_note": header[:200],
                        "abnormality": None, "acute": None,
                        "source": source or sheet.get("kind")})
    led = json.load(open(ledger_path)) if os.path.exists(ledger_path) else []
    led.extend(new)
    with open(ledger_path, "w") as f:
        json.dump(led, f, indent=1)
    return new


@register_function(
    aliases=["异常热点", "ledger hotspots", "abnormality hotspots", "cluster survey reads by location"],
    category="radiology",
    description="Cluster abnormal ledger reads by patient position and rank locations by severity and by "
                "how many slices and series report them — where to look next, independent of any history",
    requires={},
    produces={},
    auto_fix="none",
    examples=["hs = ov.radiology.ledger_hotspots('ledger.json')",
              "ov.radiology.structure_survey(st, 'survey', structures=[{'name': f'hotspot_{k}', "
              "'xyz_mm': r.xyz_mm} for k, r in hs.head(3).iterrows()])"],
)
def ledger_hotspots(ledger: Union[str, Sequence[Dict]], *, radius_mm: float = 15.0,
                    min_severity: int = 1) -> pd.DataFrame:
    """Rank locations of abnormal reads.

    Abnormal entries (``severity >= min_severity``) are clustered greedily
    (single linkage, ``radius_mm``) on ``xyz_mm``. Each hotspot reports its
    centre, maximum and mean severity, the number of distinct slices
    ``(series, instance)`` and of series / planes that report it, the number of
    normal reads within the radius (contradictions), and its member texts. The
    ranking key is (max severity, series and slices reporting that maximum
    severity, all series, all slices) — agreement across independent views
    outranks a single strong read.
    """
    rows = json.load(open(ledger)) if isinstance(ledger, str) else list(ledger)
    ab = [r for r in rows if r.get("xyz_mm") is not None and int(r.get("severity") or 0) >= min_severity
          and not r.get("normal")]
    nm = [r for r in rows if r.get("xyz_mm") is not None and r.get("normal")]
    if not ab:
        return pd.DataFrame(columns=["xyz_mm", "max_severity", "n_slices", "n_series"])
    P = np.array([r["xyz_mm"] for r in ab], float)
    lab = -np.ones(len(ab), int)
    k = 0
    for i in range(len(ab)):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], k
        while stack:
            j = stack.pop()
            d = np.linalg.norm(P - P[j], axis=1)
            for q in np.where((d <= radius_mm) & (lab < 0))[0]:
                lab[q] = k
                stack.append(q)
        k += 1
    Q = np.array([r["xyz_mm"] for r in nm], float) if nm else np.zeros((0, 3))
    out = []
    for c in range(k):
        mem = [ab[i] for i in np.where(lab == c)[0]]
        ctr = P[lab == c].mean(0)
        sev = [int(m["severity"]) for m in mem]
        slices = {(m["series"], int(m["instance"])) for m in mem}
        contra = int((np.linalg.norm(Q - ctr, axis=1) <= radius_mm).sum()) if len(Q) else 0
        top = max(mem, key=lambda m: (int(m["severity"]), m.get("confidence") == "high"))
        hi = [m for m in mem if int(m["severity"]) >= max(sev)]
        # centre on the most severe reads (single linkage chains large clusters,
        # whose mean can drift away from the lesion)
        ctr_hi = np.mean([m["xyz_mm"] for m in hi], axis=0)
        out.append({"xyz_mm": np.round(ctr_hi, 1).tolist(), "cluster_centre_mm": np.round(ctr, 1).tolist(),
                    "max_severity": max(sev),
                    "n_slices_max": len({(m["series"], int(m["instance"])) for m in hi}),
                    "n_series_max": len({m["series"] for m in hi}),
                    "mean_severity": round(float(np.mean(sev)), 2), "n_slices": len(slices),
                    "n_series": len({m["series"] for m in mem}),
                    "n_planes": len({m.get("plane") for m in mem}), "n_normal_nearby": contra,
                    "best_read": {"sheet": top["sheet"], "series": top["series"],
                                  "instance": top["instance"], "cell": top.get("cell"),
                                  "text": top["text"][:200]},
                    "slices": sorted(slices), "texts": [m["text"][:160] for m in mem][:12]})
    res = pd.DataFrame(out).sort_values(["max_severity", "n_series_max", "n_slices_max", "n_series",
                                         "n_slices"], ascending=False)
    return res.reset_index(drop=True)


_NOT_IN_PLANE = _re.compile(r"not in (?:the )?plane|out of plane|not (?:seen|included|visible) (?:in|on) this", _re.I)
_NOT_IN_PLANE_SPAN = _re.compile(r"[^.;]*(?:not in (?:the )?plane|out of plane|not (?:seen|included|visible) (?:in|on) this)[^.;]*[.;]?", _re.I)
_DISCONT = _re.compile(r"interrupt|discontinu|absent|\bgap\b|torn|tear|ruptur|retract|stump|not traceable|"
                       r"difficult to trace|replaced by (?:bright|fluid)", _re.I)


@register_function(
    aliases=["主要发现候选", "ledger principal", "principal finding from ledger", "top lesion from reads"],
    category="radiology",
    description="Principal-finding candidate from a findings ledger: the top hotspot, the structure the reader "
                "named at that location, and per series the runs of consecutive slices read as discontinuous",
    requires={},
    produces={},
    auto_fix="none",
    examples=["p = ov.radiology.ledger_principal('ledger.json')"],
)
def ledger_principal(ledger: Union[str, Sequence[Dict]], *, radius_mm: float = 15.0,
                     near_mm: float = 40.0) -> Dict:
    """Summarise the first row of :func:`ledger_hotspots` for reporting.

    Returns the hotspot centre, the structure names the reader gave on sheets
    centred within ``near_mm`` (most frequent first), and for each series the
    longest run of consecutive slices whose reads describe a discontinuity
    (interrupted / absent / gap / torn / replaced by fluid), with the slices and
    texts. Reads elsewhere (other levels, other locations) are not part of the
    evidence for this location and cannot overrule it.
    """
    rows = json.load(open(ledger)) if isinstance(ledger, str) else list(ledger)
    hs = ledger_hotspots(rows, radius_mm=radius_mm)
    if hs.empty:
        return {"found": False}
    top = hs.iloc[0]
    c = np.asarray(top["xyz_mm"], float)
    near = [r for r in rows if r.get("xyz_mm") is not None
            and np.linalg.norm(np.asarray(r["xyz_mm"], float) - c) <= near_mm]
    names: Dict[str, int] = {}
    for r in near:
        note = (r.get("sheet_note") or "").strip()
        if note and r.get("source") == "structure":
            key = _re.sub(r"^(central structure|structure at (the )?centre[^:]*)\s*:?\s*", "", note,
                          flags=_re.I).strip(" .*")[:80]
            names[key] = names.get(key, 0) + 1
    runs = {}
    by_series: Dict[str, Dict[int, List[str]]] = {}
    for r in near:
        if r.get("normal"):
            continue
        txt = r.get("text", "")
        if _NOT_IN_PLANE.search(txt):
            txt = _NOT_IN_PLANE_SPAN.sub("", txt)
        if _DISCONT.search(txt) and int(r.get("severity") or 0) >= 3:
            by_series.setdefault(r["series"], {}).setdefault(int(r["instance"]), []).append(r["text"][:160])
    for ser, d in by_series.items():
        ins = sorted(d)
        best, cur = [], [ins[0]]
        for a, b in zip(ins[:-1], ins[1:]):
            if b - a == 1:
                cur.append(b)
            else:
                best, cur = (cur if len(cur) > len(best) else best), [b]
        best = cur if len(cur) > len(best) else best
        runs[ser] = {"slices": best, "n": len(best), "texts": [d[i][0] for i in best][:6]}
    return {"found": True, "xyz_mm": c.round(1).tolist(), "max_severity": int(top["max_severity"]),
            "n_series": int(top["n_series"]), "n_slices": int(top["n_slices"]),
            "structure_names": sorted(names, key=names.get, reverse=True)[:5],
            "discontinuity_runs": dict(sorted(runs.items(), key=lambda kv: -kv[1]["n"])),
            "discontinuity_on_consecutive_slices": any(v["n"] >= 2 for v in runs.values())}


def _best_term(name: str, terms: Sequence[str]) -> Optional[str]:
    import difflib
    key = _re.sub(r"\(.*?\)", "", name or "").lower().strip()
    key = _re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    if not key:
        return None
    toks = set(key.split("_"))
    scored = []
    for t in terms:
        tt = set(t.lower().split("_"))
        overlap = len(toks & tt) / max(1, len(tt))
        scored.append((overlap, difflib.SequenceMatcher(None, key, t.lower()).ratio(), t))
    scored.sort(reverse=True)
    return scored[0][2] if scored and (scored[0][0] > 0 or scored[0][1] > 0.6) else None


def _pick(terms: Sequence[str], *patterns: str) -> Optional[str]:
    for pat in patterns:
        for t in terms:
            if pat in t.lower():
                return t
    return None


@register_function(
    aliases=["主要发现草稿", "draft principal finding", "principal finding draft from ledger"],
    category="radiology",
    description="Draft the principal finding (structure / finding type / acuity terms from the caller's "
                "vocabulary, location, measurement slice, image citations) from ledger_principal by fixed rules",
    requires={},
    produces={},
    auto_fix="none",
    examples=["d = ov.radiology.draft_principal(st, 'ledger.json', json.load(open('vocabularies.json')))"],
)
def draft_principal(study: DicomStudy, ledger: Union[str, Sequence[Dict]], vocabulary: Dict, *,
                    radius_mm: float = 15.0) -> Dict:
    """Rule-based draft of the principal finding.

    ``vocabulary`` is the task's own controlled vocabulary (keys ``structure``,
    ``finding_type``, ``acuity``); terms are matched by name, nothing is
    hard-coded. Rules on the reads at the :func:`ledger_principal` location:

    - structure: over the whole ledger, the structure (reader's name on the
      sheets centred on it, matched to the closest vocabulary term) with the
      most slices read as discontinuous (ties: most series); reads that report
      no discontinuity do not vote. Location and discontinuity runs are then
      taken from that structure's reads;
    - finding type: discontinuity on consecutive slices in >= 2 series, or on
      >= 3 consecutive slices of one series → the vocabulary's
      full-thickness / complete tear term, unless 'partial' reads outnumber
      'absent / gap / interrupted' reads (→ partial-thickness term);
      no discontinuity run → the tendinosis-type term if reads say
      thickened / heterogeneous, else the first matching inflammatory term;
    - acuity: 'acute' if the reads describe fluid / bright / oedema in or
      around the defect and none describe scar or low-signal fibrous tissue,
      'chronic' for the reverse, else 'indeterminate';
    - measurement slice: the middle slice of the longest run in a series
      whose plane shows the structure lengthwise (most slices in the run);
    - evidence: the nearest in-plane instance of the two series with the
      longest runs.
    Returns the draft plus the counts behind every decision.
    """
    p = ledger_principal(ledger, radius_mm=radius_mm)
    if not p.get("found"):
        return {"found": False}
    vs, vf, va = vocabulary.get("structure", []), vocabulary.get("finding_type", []), vocabulary.get("acuity", [])
    rows = json.load(open(ledger)) if isinstance(ledger, str) else list(ledger)
    # Structure = the one with the most slices read as DISCONTINUOUS on the
    # sheets the reader centred on it (whole ledger; one vote per slice;
    # ties by number of series). Reads that do not report a discontinuity,
    # and structures merely visible near a lesion, do not vote.
    term_of_note: Dict[str, Optional[str]] = {}
    agg: Dict[str, Dict] = {}
    for r in rows:
        note = (r.get("sheet_note") or "").strip()
        if not note or r.get("source") != "structure" or r.get("xyz_mm") is None:
            continue
        if note not in term_of_note:
            key = _re.sub(r"^(central structure|centre structure|structure( at (the )?cent(re|er)[^:]*)?)\s*:?\s*",
                          "", note, flags=_re.I)
            term_of_note[note] = _best_term(key, vs)
        t = term_of_note[note]
        if not t or r.get("normal") or int(r.get("severity") or 0) < 3:
            continue
        txt = _NOT_IN_PLANE_SPAN.sub("", r.get("text", ""))
        if not _DISCONT.search(txt):
            continue
        a_ = agg.setdefault(t, {"slices": {}, "series": set()})
        a_["slices"].setdefault((r["series"], int(r["instance"])), []).append(r)
        a_["series"].add(r["series"])
    tally = {t: len(v["slices"]) for t, v in agg.items()}
    struct = max(agg, key=lambda t: (len(agg[t]["slices"]), len(agg[t]["series"]))) if agg else None
    if struct is None:
        for nm in p["structure_names"]:
            struct = _best_term(nm, vs)
            if struct:
                break
    if struct in agg:
        sel = [np.asarray(rr["xyz_mm"], float) for v in agg[struct]["slices"].values() for rr in v]
        by_series: Dict[str, Dict[int, str]] = {}
        for (ser, ins), v in agg[struct]["slices"].items():
            by_series.setdefault(ser, {})[ins] = v[0]["text"][:160]
        runs_s = {}
        for ser, d in by_series.items():
            ins = sorted(d)
            best, cur = [], [ins[0]]
            for x, y in zip(ins[:-1], ins[1:]):
                if y - x == 1:
                    cur.append(y)
                else:
                    best, cur = (cur if len(cur) > len(best) else best), [y]
            best = cur if len(cur) > len(best) else best
            runs_s[ser] = {"slices": best, "n": len(best), "texts": [d[x] for x in best][:6]}
        p = dict(p, xyz_mm=np.median(sel, axis=0).round(1).tolist(),
                 discontinuity_runs=dict(sorted(runs_s.items(), key=lambda kv: -kv[1]["n"])))
    runs = p["discontinuity_runs"]
    texts = ([rr["text"] for v in agg[struct]["slices"].values() for rr in v[:1]] if struct in agg
             else [t for r in runs.values() for t in r["texts"]])   # one read per slice
    allt = " ".join(texts).lower()
    n_partial = sum("partial" in t.lower() for t in texts)
    n_full = sum(bool(_re.search(r"absent|\bgap\b|interrupt|discontinu|ruptur|retract", t, _re.I))
                 and "partial" not in t.lower() for t in texts)
    multi = sum(1 for r in runs.values() if r["n"] >= 2) >= 2 or any(r["n"] >= 3 for r in runs.values())
    if multi:
        if n_partial > n_full:
            ftype = _pick(vf, "partial")
        else:
            ftype = _pick(vf, "full_thickness", "full thickness", "complete", "rupture") or _pick(vf, "tear")
    else:
        ftype = _pick(vf, "tendinosis", "tendinopathy") if _re.search(r"thicken|heterogen", allt) else \
            _pick(vf, "tenosynovitis", "effusion", "bursitis", "edema", "oedema")
    acute_cues = len(_re.findall(r"fluid|bright|oedema|edema|haemat|hemat", allt))
    chronic_cues = len(_re.findall(r"scar|fibrous|low[- ]signal (?:tissue|filling)", allt))
    if acute_cues and not chronic_cues:
        acuity = _pick(va, "acute")
    elif chronic_cues and not acute_cues:
        acuity = _pick(va, "chronic")
    else:
        acuity = _pick(va, "indeterminate")
    xyz = np.asarray(p["xyz_mm"], float)
    ranked = sorted(runs.items(), key=lambda kv: -kv[1]["n"])
    meas = None
    evidence = []
    for ser, r in ranked:
        s = study[ser] if ser in [x.description for x in study] else None
        if s is None:
            continue
        mid = r["slices"][len(r["slices"]) // 2]
        if meas is None:
            meas = {"series": ser, "series_instance_uid": s.uid, "instance": int(mid), "run": r["slices"]}
        if len(evidence) < 2 and s.uid not in [e["series_instance_uid"] for e in evidence]:
            evidence.append({"series_instance_uid": s.uid, "sop_instance_uid": s.instance(int(mid)).sop_uid})
    cands = []
    for ser, r in ranked:
        s = next((x for x in study if x.description == ser), None)
        if s is None:
            continue
        m = s.patient_to_pixel(xyz)
        i0 = s.instance(m["instance_number"])
        h = int(round(50.0 / i0.ps[0])); w = int(round(50.0 / i0.ps[1]))
        r0 = int(np.clip(round(m["row"]) - h, 0, max(0, i0.rows - 2 * h)))
        c0 = int(np.clip(round(m["col"]) - w, 0, max(0, i0.cols - 2 * w)))
        cands.append({"series": ser, "plane": s.plane, "weighting": s.weighting,
                      "run_slices": r["slices"], "instance_at_location": m["instance_number"],
                      "rowcol_at_location": [round(m["row"], 1), round(m["col"], 1)],
                      "crop_100mm": [r0, min(i0.rows, r0 + 2 * h), c0, min(i0.cols, c0 + 2 * w)]})
    return {"found": True, "structure": struct, "finding_type": ftype, "acuity": acuity,
            "location_xyz_mm": xyz.round(1).tolist(), "measurement_slice": meas,
            "measurement_candidates": cands, "evidence": evidence,
            "basis": {"structure_votes": tally, "structure_names": p["structure_names"], "runs": {k: v["n"] for k, v in runs.items()},
                      "full_reads": n_full, "partial_reads": n_partial, "acute_cues": acute_cues,
                      "chronic_cues": chronic_cues}}
