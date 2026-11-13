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


def test_deliverable_written(tmp_path):
    T = _cohort()
    T["labevents"].loc[T["labevents"].index[T["labevents"].itemid == "A"][5], "valuenum"] = "100.0"
    p = tmp_path / "sub" / "flagged.csv"
    f = audit_clinical_tables(T, external_reference=False, verbose=False, out=str(p))
    d = pd.read_csv(p, dtype=str)
    assert list(d.columns) == ["table", "_row_id"]
    assert len(d) == f.drop_duplicates(["table", "row_id"]).shape[0]


def _synthea_like(seed=0, n=60):
    """Event-table cohort: patients with birth / death dates, visits, observations, medications."""
    rng = np.random.default_rng(seed)
    pats, encs, obs, meds = [], [], [], []
    for i in range(n):
        pid = f"p{i}"
        sex = "F" if i % 2 else "M"
        birth = pd.Timestamp("1950-01-01") + pd.Timedelta(days=int(rng.integers(0, 15000)))
        death = birth + pd.Timedelta(days=30000) if i % 5 == 0 else None
        pats.append([f"P{i}", pid, str(birth.date()), "" if death is None else str(death.date()), sex,
                     "a" if i % 3 else "b"])
        for k in range(8):
            t = birth + pd.Timedelta(days=3000 + 1000 * k)
            if death is not None and t > death:
                break
            eid = f"e{i}_{k}"
            encs.append([f"E{i}_{k}", eid, pid, str(t), str(t + pd.Timedelta(hours=1)), "checkup"])
            s = float(rng.normal(120, 8)); d_ = float(s - rng.normal(40, 5))
            obs.append([f"O{i}_{k}a", pid, eid, str(t), "SYS", f"{s:.0f}", "mmHg"])
            obs.append([f"O{i}_{k}b", pid, eid, str(t), "DIA", f"{d_:.0f}", "mmHg"])
            obs.append([f"O{i}_{k}c", pid, eid, str(t), "ANS", "a" if i % 3 else "b", ""])
            obs.append([f"O{i}_{k}d", pid, eid, str(t), "FEM" if sex == "F" else "MAL", "1", "{#}"])
            bc, dp = float(rng.uniform(1, 50)), int(rng.integers(1, 20))
            meds.append([f"M{i}_{k}", pid, eid, str(t), "", "drug", f"{bc:.2f}", str(dp), f"{bc * dp:.2f}"])
        if death is not None:     # a death-certification visit after death is a source convention
            t = death + pd.Timedelta(days=1)
            encs.append([f"E{i}_d", f"e{i}_d", pid, str(t), str(t), "certification"])
    T = {"patients": pd.DataFrame(pats, columns=["_row_id", "patient_id", "birthdate", "deathdate", "gender", "race"]),
         "encounters": pd.DataFrame(encs, columns=["_row_id", "encounter_id", "patient_id", "start", "stop", "code"]),
         "observations": pd.DataFrame(obs, columns=["_row_id", "patient_id", "encounter_id", "date", "code", "value", "units"]),
         "medications": pd.DataFrame(meds, columns=["_row_id", "patient_id", "encounter_id", "start", "stop", "code",
                                                    "base_cost", "dispenses", "total_cost"])}
    return T


def test_event_rules():
    T = _synthea_like()
    obs, encs, meds, pats = T["observations"], T["encounters"], T["medications"], T["patients"]
    i_sys = obs.index[obs.code == "SYS"][3]; i_dia = obs.index[obs.code == "DIA"][3]
    obs.loc[i_sys, "value"], obs.loc[i_dia, "value"] = obs.loc[i_dia, "value"], obs.loc[i_sys, "value"]  # swap
    meds.loc[meds.index[5], "total_cost"] = "9999.99"                                                     # arithmetic
    p0 = pats.index[pats.deathdate != ""][0]
    pid0 = pats.loc[p0, "patient_id"]
    e_late = encs[encs.patient_id == pid0].iloc[0].copy()
    e_late["_row_id"], e_late["encounter_id"], e_late["start"] = "ELATE", "elate", "2200-01-01 10:00:00"
    T["encounters"] = encs = pd.concat([encs, e_late.to_frame().T], ignore_index=True)                    # after death
    o = obs.index[(obs.code == "ANS") & (obs.patient_id == "p7")]
    obs.loc[o, "value"] = "b" if pats.loc[7, "race"] == "a" else "a"                                    # race vs answers
    j = obs.index[(obs.code == "FEM")][0]
    T["observations"] = obs = pd.concat([obs, obs.loc[[j]].assign(_row_id="DUP")], ignore_index=True)     # exact dup
    f = audit_clinical_tables(T, external_reference=False, verbose=False)
    fl = _flags(f)
    assert ("observations", obs.loc[i_sys, "_row_id"]) in fl or ("observations", obs.loc[i_dia, "_row_id"]) in fl
    assert ("medications", meds.loc[meds.index[5], "_row_id"]) in fl
    assert ("encounters", "ELATE") in fl
    assert ("patients", pats.loc[7, "_row_id"]) in fl
    assert ("observations", "DUP") in fl
    # the routine after-death certification visits are not flagged
    assert not any(t == "encounters" and r.startswith("E") and r.endswith("_d") for t, r in fl)
