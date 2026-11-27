"""ov.radiology.measure_spine_alignment on synthetic sagittal spine phantoms with known slip."""

import json

import numpy as np
import pytest

pytest.importorskip("pydicom")
pytest.importorskip("skimage")

from omicverse import radiology as rad

from .spine_phantom import make_study

NAMES = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2"]


def _oblique(deg):
    t = np.deg2rad(deg)
    # sagittal, tilted about the left-right axis: rows run posterior, columns inferior
    return (0.0, np.cos(t), np.sin(t), 0.0, np.sin(t), -np.cos(t))


@pytest.fixture(scope="module")
def t1_slip(tmp_path_factory):
    root = tmp_path_factory.mktemp("spine_t1")
    geo = make_study(root, contrast="T1", slip_level=0, slip_mm=8.0, noise=0.05, bias=1.0, iop=_oblique(8))
    return rad.read_dicom_study(root), geo


def test_alignment_finds_levels_from_sacrum_and_measures_slip(t1_slip):
    st, geo = t1_slip
    a = rad.measure_spine_alignment(st)
    assert a["found"] and a["sacrum_found"] and a["label_confidence"] == "high"
    assert [lv["level"] for lv in a["levels"]] == NAMES
    truth = geo["truth"]
    for lv, t in zip(a["levels"], truth):
        assert lv["offset_mm"] == pytest.approx(t["offset_mm"], abs=1.5)
        assert lv["disc_height_mm"] == pytest.approx(t["disc_height_mm"], abs=3.5)
    l5 = a["levels"][0]
    assert l5["abnormal"] and l5["direction"] == "antero" and l5["meyerding_grade"] == 1
    assert l5["slip_pct"] == pytest.approx(truth[0]["slip_pct"], abs=5)
    assert l5["qc"] == "ok" and l5["n_slices_agree"] >= 1
    assert not any(lv["abnormal"] for lv in a["levels"][1:])
    # landmarks are patient mm on the measured slice; the offset is recomputable from them
    s = st[a["series_instance_uid"]]
    lm = {k: np.asarray(v) for k, v in l5["landmarks_xyz_mm"].items()}
    e = (lm["lower_AS"] - lm["lower_PS"]) / np.linalg.norm(lm["lower_AS"] - lm["lower_PS"])
    assert float((lm["upper_PI"] - lm["lower_PS"]) @ e) == pytest.approx(l5["offset_mm"], abs=0.05)
    for p in lm.values():
        q = s.patient_to_pixel(p)
        assert q["instance_number"] == a["instance"] and q["out_of_plane_mm"] < 0.1
    # anterior in LPS is -y: the slipped upper corner is anterior of the lower one
    assert lm["upper_PI"][1] < lm["lower_PS"][1]
    assert list(a["table"].columns[:3]) == ["level", "upper", "lower"]


@pytest.mark.parametrize("contrast,level,slip,retro,kw", [
    ("T2", 1, 6.0, True, dict(thickness=6.0, noise=0.05, bias=1.2)),        # retrolisthesis, thick slices
    ("CT", 0, 15.0, False, dict(lordosis_deg=30.0, noise=0.03)),            # grade II, flat lordosis
    ("T1", 2, 23.0, False, dict(lordosis_deg=60.0, noise=0.08, bias=1.2)),  # grade III, noisy, shaded
])
def test_alignment_robust_to_contrast_and_acquisition(tmp_path, contrast, level, slip, retro, kw):
    geo = make_study(tmp_path, contrast=contrast, slip_level=level, slip_mm=slip, retro=retro, seed=level, **kw)
    a = rad.measure_spine_alignment(rad.read_dicom_study(tmp_path))
    assert a["found"] and [lv["level"] for lv in a["levels"]] == NAMES
    t = geo["truth"][level]
    lv = a["levels"][level]
    assert lv["offset_mm"] == pytest.approx(t["offset_mm"], abs=1.5)
    assert lv["direction"] == ("retro" if retro else "antero")
    assert lv["meyerding_grade"] == int(np.ceil(abs(t["slip_pct"]) / 25))
    assert [x["abnormal"] for x in a["levels"]] == [k == level for k in range(5)]


def test_alignment_without_sacrum_labels_relative(tmp_path):
    from .spine_phantom import spine_geometry  # noqa: F401 - documents where the geometry lives
    make_study(tmp_path, contrast="T1", slip_level=1, slip_mm=7.0, noise=0.03)
    st = rad.read_dicom_study(tmp_path)
    a = rad.measure_spine_alignment(st, min_sacrum_elongation=50.0)   # nothing qualifies as sacrum
    assert a["found"] and not a["sacrum_found"] and a["label_confidence"] == "none"
    assert a["levels"][0]["level"] == "V2-V1" and a["levels"][0]["abnormal"]
    assert any("sacrum not identified" in w for w in a["warnings"])


def _reads(st, xyz, text, severity=3, instance=3):
    s = st.diagnostic("SAG")[0]
    return {"sheet": "s1", "series": s.description, "series_instance_uid": s.uid, "plane": "SAG",
            "weighting": s.weighting, "instance": instance, "cell": "B2", "xyz_mm": list(map(float, xyz)),
            "text": text, "severity": severity, "confidence": "high", "normal": False,
            "structure": None, "sheet_note": "Central structure: L4-L5 disc", "abnormality": None,
            "acute": None, "source": "structure"}


VOC = {"structure": ["L3-L4", "L4-L5", "L5-S1"],
       "finding_type": ["disc_herniation", "spondylolisthesis", "retrolisthesis", "full_thickness_tear"],
       "acuity": ["acute", "chronic", "indeterminate"], "grade": ["grade_I", "grade_II", "grade_III"]}


def test_ledger_measurement_promotes_slip_to_principal(tmp_path, t1_slip):
    st, _ = t1_slip
    a = rad.measure_spine_alignment(st)
    led = str(tmp_path / "ledger.json")
    # a reader's read at another level, severe (the failure mode: herniation chosen at the wrong level)
    other = np.asarray(a["levels"][1]["xyz_mm"])
    rows = [_reads(st, other + [0, 0, d], "disc extruded, torn annulus, absent posterior margin, high")
            for d in (0, 1, 2)]
    json.dump(rows, open(led, "w"))
    before = rad.draft_principal(st, led, VOC)
    new = rad.ledger_add_measurement(led, st, a)
    assert len(new) == 5 and new[0]["kind"] == "alignment" and new[0]["severity"] == 3
    assert sum(not e["normal"] for e in new) == 1
    d = rad.draft_principal(st, led, VOC)
    assert d["found"] and d["basis"]["principal_source"] == "measured_alignment"
    assert (d["structure"], d["finding_type"], d["grade"]) == ("L5-S1", "spondylolisthesis", "grade_I")
    assert d["level"] == "L5-S1" and d["measurement"]["meyerding_grade"] == 1
    assert d["evidence"][0]["sop_instance_uid"] == a["sop_instance_uid"]
    assert d["basis"]["reads_only_draft"]["structure"] == before["structure"]
    # opt-outs and vocabularies without an alignment term keep the reads-only draft
    off = rad.draft_principal(st, led, VOC, alignment="off")
    assert off["structure"] == before["structure"] and off["finding_type"] == before["finding_type"]
    voc2 = dict(VOC, finding_type=["disc_herniation", "full_thickness_tear"])
    d2 = rad.draft_principal(st, led, voc2)
    assert d2["finding_type"] == before["finding_type"] and d2["basis"]["alignment"]["promoted"] is False
    # 'corroborated' needs a read describing a step at the measured level
    assert rad.draft_principal(st, led, VOC, alignment="corroborated")["basis"].get("principal_source") is None
    rows = json.load(open(led)) + [_reads(st, a["levels"][0]["xyz_mm"], "step-off of L5 on S1, medium", 1)]
    d3 = rad.draft_principal(st, rows, VOC, alignment="corroborated")
    assert d3["structure"] == "L5-S1" and d3["basis"]["alignment"]["corroborating_reads"] == 1


def test_ledger_measurement_normal_alignment_does_not_promote(tmp_path):
    make_study(tmp_path / "s", contrast="T1", slip_level=None, noise=0.03)
    st = rad.read_dicom_study(tmp_path / "s")
    a = rad.measure_spine_alignment(st)
    assert a["found"] and not any(lv["abnormal"] for lv in a["levels"])
    led = str(tmp_path / "l.json")
    new = rad.ledger_add_measurement(led, st, a)
    assert all(e["normal"] and e["severity"] == 0 for e in new)
    assert rad.ledger_hotspots(led).empty
    assert rad.draft_principal(st, led, VOC) == {"found": False}
