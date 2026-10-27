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

__all__ = ["structure_survey", "sheet_cell_to_patient", "ledger_concordance"]

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
                     region_mm: float = 90.0, crop_mm: float = 60.0, depth_mm: float = 15.0,
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
        phase 2 structures are processed in the given order and series in the
        fixed order (plane, fluid-sensitive first) until the bound is reached;
        skipped (structure, series) pairs are listed in the manifest.

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
        n = 0
        for spec in structures:
            spec = {"name": spec} if isinstance(spec, str) else dict(spec)
            p = _locate(study, spec, prior)
            if p is None:
                unloc.append(spec["name"])
                continue
            cm = float(spec.get("crop_mm", crop_mm))
            dm = float(spec.get("depth_mm", depth_mm))
            for s in ordered:
                if n >= max_sheets:
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
                n += 1
        for k, sh in enumerate(sheets):
            sh["id"] = f"S{k + 1:02d}"
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
