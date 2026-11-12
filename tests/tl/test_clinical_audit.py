"""Synthetic tests for omicverse.tl clinical data-entry audit (no real records, no task constants)."""
import numpy as np
import pandas as pd

from omicverse.tl._clinical_audit import audit_clinical_tables, infer_clinical_schema


def _cohort(seed=0, n_pat=60):
    rng = np.random.default_rng(seed)
    pats = pd.DataFrame({"_row_id": [f"p{i}" for i in range(n_pat)],
                         "subject_id": [str(100 + i) for i in range(n_pat)],
                         "gender": rng.choice(["M", "F"], n_pat),
                         "anchor_age": rng.integers(30, 85, n_pat).astype(str)})
    rows, chart = [], []
    t0 = pd.Timestamp("2100-01-01")
    k = 0
    for _, p in pats.iterrows():
        for j in range(12):
            t = str(t0 + pd.Timedelta(hours=6 * j))
            spec = str(5000 + k)
            # analyte A: positive, log-normal; reference band depends on sex
            a = float(np.round(rng.lognormal(0, 0.2), 2))
            lo, hi = ("0.5", "1.2") if p.gender == "M" else ("0.4", "1.1")
            rows.append([f"l{k}", p.subject_id, spec, "A", t, a, "mg/dL", lo, hi]); k += 1
            # analyte B: around 140, sex-neutral
            b = float(np.round(rng.normal(140, 3)))
            rows.append([f"l{k}", p.subject_id, spec, "B", t, b, "mEq/L", "135", "145"]); k += 1
            # the chart table transcribes analyte B at the same instant
            chart.append([f"c{k}", p.subject_id, "CB", t, b, "mEq/L"]); k += 1
    lab = pd.DataFrame(rows, columns=["_row_id", "subject_id", "specimen_id", "itemid", "charttime",
                                      "valuenum", "valueuom", "ref_range_lower", "ref_range_upper"])
    ch = pd.DataFrame(chart, columns=["_row_id", "subject_id", "itemid", "charttime", "valuenum", "valueuom"])
    rx = pd.DataFrame({"_row_id": [f"r{i}" for i in range(n_pat * 3)],
                       "subject_id": np.repeat(pats.subject_id.values, 3),
                       "drug": ["Drugx"] * (n_pat * 3),
                       "dose_val_rx": ["40"] * (n_pat * 3), "dose_unit_rx": ["mg"] * (n_pat * 3)})
    for c in lab.columns:
        lab[c] = lab[c].astype(str)
    for c in ch.columns:
        ch[c] = ch[c].astype(str)
    return {"patients": pats, "labevents": lab, "chartevents": ch, "prescriptions": rx}


def _flags(f):
    return set(zip(f["table"], f["row_id"]))


def test_schema_inference():
    s = infer_clinical_schema(_cohort())
    assert s["subject"] == "subject_id" and s["row_id"] == "_row_id"
    assert s["demographics"]["table"] == "patients" and s["demographics"]["sex"] == "gender"
    assert s["measurements"]["labevents"]["specimen"] == "specimen_id"
    assert "prescriptions" in s["prescriptions"]


def test_clean_cohort_has_few_flags():
    T = _cohort()
    f = audit_clinical_tables(T, external_reference=False, verbose=False)
    n_rows = sum(len(d) for d in T.values())
    assert f.drop_duplicates(["table", "row_id"]).shape[0] < 0.01 * n_rows


def test_injected_errors_are_found():
    T = _cohort()
    lab, ch, rx, pats = T["labevents"], T["chartevents"], T["prescriptions"], T["patients"]
    # 1 decimal slip, 2 unit label, 3 duplicate disagreeing, 4 lab/chart conflict, 5 dose x100,
    # 6 sex flipped, 7 isolated age
    lab.loc[lab.index[lab.itemid == "A"][5], "valuenum"] = "100.0"
    lab.loc[lab.index[lab.itemid == "B"][7], "valueuom"] = "ng/mL"
    src = lab[lab.itemid == "B"].iloc[20].copy()
    src["_row_id"] = "dup1"
    src["valuenum"] = str(float(src["valuenum"]) * 1.3)
    T["labevents"] = lab = pd.concat([lab, src.to_frame().T], ignore_index=True)
    ch.loc[ch.index[40], "valuenum"] = str(float(ch.loc[ch.index[40], "valuenum"]) + 25)
    rx.loc[rx.index[10], "dose_val_rx"] = "4000"
    flip = pats.index[3]
    pats.loc[flip, "gender"] = "F" if pats.loc[flip, "gender"] == "M" else "M"
    pats.loc[pats.index[8], "anchor_age"] = "3"
    f = audit_clinical_tables(T, external_reference=False, verbose=False)
    fl = _flags(f)
    assert ("labevents", lab[(lab.itemid == "A") & (lab.valuenum == "100.0")]["_row_id"].iloc[0]) in fl
    assert ("labevents", lab[lab.valueuom == "ng/mL"]["_row_id"].iloc[0]) in fl
    assert ("labevents", "dup1") in fl
    assert ("chartevents", ch.loc[ch.index[40], "_row_id"]) in fl
    assert ("prescriptions", rx.loc[rx.index[10], "_row_id"]) in fl
    assert ("patients", pats.loc[flip, "_row_id"]) in fl
    assert ("patients", pats.loc[pats.index[8], "_row_id"]) in fl
    # one incident per sex conflict, evidence rows share its cluster
    sex = f[f["rule"] == "sex_vs_records"]
    assert sex["cluster"].nunique() == 1
    assert (f["cluster"] == sex["cluster"].iloc[0]).sum() > 1
