r"""Rendering DICOM slices for visual reading.

Two failure modes make a multimodal model misread an MRI study:

* tiny thumbnails — a 22-slice contact sheet shrinks a 512² image to ~100 px,
  below the size at which a tendon or a cortical line can be followed;
* lost geometry — pixels shown without their aspect ratio, orientation or a
  coordinate reference, so the reader cannot say *where* something is, and
  positions read off the picture cannot be turned into millimetres.

``render_slices`` therefore draws aspect-correct crops at a useful size, side
by side across sequences at the same patient location, with a labelled pixel
grid (so a reader can report ``(row, col)``) and the patient directions on the
image edges. ``montage`` gives an overview of a whole series with every tile
labelled by instance number, for choosing where to look closer.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .._registry import register_function
from ._dicom import DicomSeries, DicomStudy

__all__ = ["render_slices", "montage"]


def _window(img: np.ndarray, lo: float = 1.0, hi: float = 99.5) -> Tuple[float, float]:
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(v, [lo, hi])
    return float(a), float(b if b > a else a + 1)


def _resolve(study: Optional[DicomStudy], s) -> DicomSeries:
    if isinstance(s, DicomSeries):
        return s
    if study is None:
        raise ValueError("pass a DicomStudy to refer to series by uid / number / description")
    return study[s]


def _crop_on(target: DicomSeries, tinst: int, ref: DicomSeries, rinst: int,
             crop: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    """Map a (r0, r1, c0, c1) crop of the reference panel onto another series."""
    r0, r1, c0, c1 = crop
    ti = target.instance(tinst)
    pts = [ref.pixel_to_patient(rinst, r, c) for r in (r0, r1) for c in (c0, c1)]
    rr, cc = [], []
    for p in pts:
        d = p - ti.ipp
        cc.append(float(np.dot(d, ti.row_cos)) / ti.ps[1])
        rr.append(float(np.dot(d, ti.col_cos)) / ti.ps[0])
    return (max(0, int(np.floor(min(rr)))), min(ti.rows, int(np.ceil(max(rr)))),
            max(0, int(np.floor(min(cc)))), min(ti.cols, int(np.ceil(max(cc)))))


@register_function(
    aliases=["MRI 切片渲染", "DICOM 阅片图", "render dicom slices", "side by side mri sequences",
             "mri slice crop with grid", "radiology reading image"],
    category="radiology",
    description="Aspect-correct, gridded (pixel row/col) and orientation-labelled crops of DICOM "
                "slices, side by side across sequences at the same patient location, with optional "
                "landmark markers — images sized for visual reading by a vision model",
    requires={},
    produces={},
    auto_fix="none",
    examples=["ov.radiology.render_slices(st, ['Sag T1', 'Sag STIR'], instance=11, "
              "crop=(0, 300, 300, 512), out='sag11_post.png')",
              "ov.radiology.render_slices(st, [uid], instance=11, points={'A': (162, 386)}, out='check.png')"],
)
def render_slices(study: Optional[DicomStudy], series: Sequence[Union[str, int, DicomSeries]],
                  instance: Optional[int] = None, *, instances: Optional[Sequence[int]] = None,
                  crop: Optional[Tuple[float, float, float, float]] = None,
                  points: Optional[Dict[str, Tuple[float, float]]] = None,
                  paths: Optional[Dict[str, Sequence[Tuple[float, float]]]] = None,
                  out: str = "slices.png", panel_px: int = 900, grid: Optional[int] = None,
                  window: Tuple[float, float] = (1.0, 99.5), title: Optional[str] = None,
                  max_px: int = 1800) -> Dict:
    """Render one slice per series, side by side.

    Parameters
    ----------
    study
        The :class:`DicomStudy` (may be ``None`` if ``series`` are objects).
    series
        Series uids / numbers / descriptions / objects. The first is the
        *reference* panel.
    instance
        Instance number in the reference series. Other panels show the
        instance of their own series nearest to the reference slice (by
        patient position), unless ``instances`` gives one per panel.
    crop
        ``(row0, row1, col0, col1)`` in the reference panel's pixels. The same
        patient region is cropped from every other panel.
    points
        ``{label: (row, col)}`` in reference pixels, drawn on every panel
        (mapped through patient space) to check landmark placement.
    paths
        ``{label: [(row, col), ...]}`` polylines in reference pixels (e.g. the
        ``row``/``col`` columns of :func:`path_profile`), drawn on every panel.
    panel_px
        Longest side of each panel in output pixels (crops are enlarged —
        do not go below ~600 for fine structures).
    grid
        Grid step in reference pixels; default ≈ 1/10 of the crop.
    max_px
        Cap on the output's longest side; panels shrink to fit. Very large
        images are slow for (and may time out in) a vision model — prefer fewer
        panels or a tighter crop to shrinking below ~600 px per panel.

    Returns
    -------
    dict with the output path and, per panel, series / instance / SOP UID /
    crop and the out-of-plane distance to the reference slice.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sers = [_resolve(study, s) for s in series]
    ref = sers[0]
    rinst = instance if instance is not None else ref.instance_numbers[len(ref) // 2]
    ri = ref.instance(rinst)
    if crop is None:
        crop = (0, ri.rows, 0, ri.cols)
    r0, r1, c0, c1 = [int(round(v)) for v in crop]
    span = max(r1 - r0, c1 - c0)
    step = grid or max(5, int(round(span / 10 / 5.0)) * 5)
    centre = ref.pixel_to_patient(rinst, (r0 + r1) / 2, (c0 + c1) / 2)

    panels = []
    for k, s in enumerate(sers):
        if instances is not None:
            ti = int(instances[k])
            oop = abs(float(np.dot(centre - s.instance(ti).ipp, s.normal)))
        elif k == 0:
            ti, oop = rinst, 0.0
        else:
            m = s.patient_to_pixel(centre)
            ti, oop = m["instance_number"], m["out_of_plane_mm"]
        cr = (r0, r1, c0, c1) if k == 0 else _crop_on(s, ti, ref, rinst, (r0, r1, c0, c1))
        panels.append((s, ti, cr, oop))

    n = len(panels)
    # size each panel from the crop's physical aspect: longest side = panel_px
    h_mm, w_mm = (r1 - r0) * ri.ps[0], (c1 - c0) * ri.ps[1]
    ph, pw = (panel_px, panel_px * w_mm / h_mm) if h_mm >= w_mm else (panel_px * h_mm / w_mm, panel_px)
    fw, fh = n * (pw + 80.0), ph + 100.0
    k = min(1.0, max_px / max(fw, fh))
    fig, axes = plt.subplots(1, n, figsize=(fw * k / 100.0, fh * k / 100.0), squeeze=False)
    meta = []
    for ax, (s, ti, (a0, a1, b0, b1), oop) in zip(axes[0], panels):
        inst = s.instance(ti)
        img = inst.pixels()[a0:a1, b0:b1]
        lo, hi = _window(img, *window)
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="lanczos",
                  extent=(b0 - 0.5, b1 - 0.5, a1 - 0.5, a0 - 0.5),
                  aspect=inst.ps[0] / inst.ps[1])
        # grid in this panel's own pixel coordinates
        g = max(1, int(round(step * (ri.ps[0] / inst.ps[0]))))
        rows = np.arange((a0 // g + 1) * g, a1, g)
        cols = np.arange((b0 // g + 1) * g, b1, g)
        for r in rows:
            ax.axhline(r, color="yellow", lw=0.4, alpha=0.45)
        for c in cols:
            ax.axvline(c, color="yellow", lw=0.4, alpha=0.45)
        ax.set_yticks(rows)
        ax.set_xticks(cols)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("col (pixel)", fontsize=8)
        ax.set_ylabel("row (pixel)", fontsize=8)
        d = s.anatomical_directions(ti)
        kw = dict(color="cyan", fontsize=11, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.995, d["up"], ha="center", va="top", **kw)
        ax.text(0.5, 0.005, d["down"], ha="center", va="bottom", **kw)
        ax.text(0.005, 0.5, d["left"], ha="left", va="center", **kw)
        ax.text(0.995, 0.5, d["right"], ha="right", va="center", **kw)
        mm = (inst.ps[0] + inst.ps[1]) / 2
        bar = 10.0 / mm
        ax.plot([b0 + 0.05 * (b1 - b0), b0 + 0.05 * (b1 - b0) + bar], [a1 - 0.04 * (a1 - a0)] * 2,
                color="white", lw=2)
        ax.text(b0 + 0.05 * (b1 - b0), a1 - 0.06 * (a1 - a0), "10 mm", color="white", fontsize=7)
        if points:
            for lab, (pr, pc) in points.items():
                xyz = ref.pixel_to_patient(rinst, pr, pc)
                m = s.patient_to_pixel(xyz)
                ax.plot(m["col"], m["row"], "+", color="red", ms=12, mew=1.5)
                ax.text(m["col"] + 0.01 * (b1 - b0), m["row"], str(lab), color="red", fontsize=9)
        if paths:
            for lab, pl in paths.items():
                pl = np.asarray(pl, float)
                mm_ = [s.patient_to_pixel(ref.pixel_to_patient(rinst, a, b)) for a, b in pl]
                ax.plot([m["col"] for m in mm_], [m["row"] for m in mm_], "-", color="lime",
                        lw=0.8, alpha=0.7)
                ax.text(mm_[0]["col"], mm_[0]["row"], str(lab), color="lime", fontsize=8)
        ax.set_title(f"{s.description} | {s.plane} | inst {ti}"
                     + (f" | {oop:.1f} mm from ref" if oop > 0.05 else ""), fontsize=9)
        meta.append({"series_instance_uid": s.uid, "description": s.description,
                     "instance_number": ti, "sop_instance_uid": inst.sop_uid,
                     "crop": (a0, a1, b0, b1), "out_of_plane_mm": round(oop, 2),
                     "directions": d})
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return {"path": out, "panels": meta}


@register_function(
    aliases=["MRI 序列总览", "DICOM 缩略图", "dicom montage", "mri series overview",
             "contact sheet of slices"],
    category="radiology",
    description="Overview grid of a DICOM series (optionally a crop), every tile labelled with its "
                "instance number, for choosing slices to inspect at full size",
    requires={},
    produces={},
    auto_fix="none",
    examples=["ov.radiology.montage(st['Sag STIR'], out='stir_all.png')",
              "ov.radiology.montage(st['Sag T1'], crop=(0, 300, 300, 512), instances=range(6, 17))"],
)
def montage(series: DicomSeries, instances: Optional[Iterable[int]] = None, *,
            crop: Optional[Tuple[int, int, int, int]] = None, ncols: int = 4,
            tile_px: int = 400, out: str = "montage.png",
            window: Tuple[float, float] = (1.0, 99.5), max_px: int = 1800) -> Dict:
    """Grid of slices of one series. Keep tiles ≥ ~300 px: split long series
    into several montages (``instances=``, ≤ 12–16 per sheet) rather than
    shrinking the tiles; the sheet is capped at ``max_px`` on its longest side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = list(instances) if instances is not None else series.instance_numbers
    nrow = int(np.ceil(len(ids) / ncols))
    # keep the whole sheet under max_px: shrink tiles (better: fewer instances per sheet)
    tile_px = int(min(tile_px, max_px / max(ncols, nrow)))
    fig, axes = plt.subplots(nrow, ncols, figsize=(ncols * tile_px / 100, nrow * tile_px / 100 + 0.4),
                             squeeze=False)
    imgs = []
    for i in ids:
        px = series.pixels(i)
        if crop is not None:
            px = px[crop[0]:crop[1], crop[2]:crop[3]]
        imgs.append(px)
    lo, hi = _window(np.concatenate([x.ravel() for x in imgs]), *window)
    d = series.anatomical_directions()
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i, px in zip(axes.ravel(), ids, imgs):
        inst = series.instance(i)
        ax.imshow(px, cmap="gray", vmin=lo, vmax=hi, aspect=inst.ps[0] / inst.ps[1])
        ax.set_title(f"inst {i}", fontsize=9, color="black")
    fig.suptitle(f"{series.description} ({series.plane}); image right = {d['right']}, "
                 f"down = {d['down']}" + (f"; crop rows {crop[0]}-{crop[1]}, cols {crop[2]}-{crop[3]}"
                                          if crop else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return {"path": out, "instances": ids, "directions": d}
