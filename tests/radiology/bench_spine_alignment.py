"""Compare edge-fit / corner / centroid slip measurement on random phantoms.

Usage: python tests/radiology/bench_spine_alignment.py [n_configs] [workdir]
Not collected by pytest (slow); the numbers are quoted in the PR.
"""
import os, sys, tempfile, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, spine_phantom as sp
from omicverse.radiology import measure_spine_alignment, read_dicom_study
rng = np.random.default_rng(7)
rows = []
grades = {"0": 0, "I": 15, "II": 35, "III": 60}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
for k in range(N):
    cfg = dict(contrast=["T1", "T2", "CT"][k % 3], noise=float(rng.choice([0.02, 0.05, 0.08])),
               thickness=float(rng.choice([3.0, 6.0])), bias=float(rng.choice([0.0, 1.2])),
               lordosis_deg=float(rng.choice([30.0, 60.0])), slip_level=int(rng.choice([0, 1, 2])),
               retro=bool(rng.random() < 0.2))
    gname = list(grades)[k % 4]
    cfg["slip_mm"] = grades[gname] / 100 * 38.0 * (0.5 if cfg["retro"] else 1)
    root = pathlib.Path(tempfile.mkdtemp(dir=sys.argv[2] if len(sys.argv) > 2 else None))
    g = sp.make_study(root, seed=k, **cfg)
    a = measure_spine_alignment(read_dicom_study(root))
    tr = g["truth"]
    base = dict(cfg, grade=gname, found=a["found"], conf=a.get("label_confidence"))
    if not a["found"] or len(a["levels"]) != len(tr):
        rows.append(dict(base, labels_ok=False)); continue
    lv = a["levels"]
    for j, (t, r) in enumerate(zip(tr, lv)):
        rows.append(dict(base, labels_ok=True, level=j, is_slip=(j == cfg["slip_level"] and cfg["slip_mm"] > 0),
                         truth=t["offset_mm"], truth_pct=t["slip_pct"], edge=r["offset_mm"], corner=r["offset_corner_mm"],
                         centroid=r["offset_centroid_mm"], pct=r["slip_pct"], grade_est=r["meyerding_grade"],
                         abnormal=r["abnormal"], qc=r["qc"], dh=r["disc_height_mm"], dh_true=t["disc_height_mm"]))
df = pd.DataFrame(rows)
df["is_slip"] = df.is_slip.astype(object).where(df.is_slip.notna(), False).astype(bool)
ok = df[df.labels_ok.astype(bool)]
print("configs", N, "found+all levels", df.groupby(["contrast"]).labels_ok.mean().round(2).to_dict(),
      "label conf", df.drop_duplicates(subset=["contrast","noise","thickness","bias","lordosis_deg","slip_level","grade"]).conf.value_counts().to_dict())
for m in ("edge", "corner", "centroid"):
    e = (ok[m] - ok.truth).abs()
    print(f"{m:9s} MAE {e.mean():.2f} mm  P90 {e.quantile(.9):.2f}  max {e.max():.2f}  | slip levels MAE {e[ok.is_slip].mean():.2f}")
    for f in ("noise", "thickness", "bias", "lordosis_deg", "grade", "contrast"):
        print("   ", f, (ok.assign(err=e).groupby(f).err.mean().round(2)).to_dict())
tp = ok[ok.is_slip & (ok.truth.abs() >= 3)]
fp = ok[~ok.is_slip]
print("detection sens", tp.abnormal.mean().round(3), "false-pos rate", fp.abnormal.mean().round(3),
      "qc ok frac", ok.qc.eq("ok").mean().round(2))
from omicverse.radiology._spine import _meyerding
gt = ok[ok.is_slip].truth_pct.map(_meyerding)
print("meyerding grade exact", (ok[ok.is_slip].grade_est == gt).mean().round(3))
print("disc height MAE", (ok.dh - ok.dh_true).abs().mean().round(2))
