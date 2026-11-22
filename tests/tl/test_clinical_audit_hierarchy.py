"""Synthetic ICU cohorts with nested ids (person > hospital admission > ICU stay) in two layouts:

* ``date``   -- absolute timestamps, death as a date of death (MIMIC-like shape);
* ``offset`` -- event times as minute offsets from ICU admission, death as a discharge status
  ("Expired") plus the offset of that discharge (eICU-like shape, generic column names).

Errors of known kind are injected and returned as truth.  No real records, no task constants.
"""
import numpy as np
import pandas as pd

from omicverse.tl._clinical_audit import audit_clinical_tables, infer_clinical_schema

ETH = ["white", "black", "asian", "hispanic", "other"]
LABS = {"hgb": (12.0, 1.5, "g/dL"), "na": (140.0, 3.0, "mmol/L"), "cr": (None, 0.3, "mg/dL")}


def make_icu_cohort(seed=0, layout="offset", n_person=200, inject=True):
    """Return ``(tables, truth)``; ``truth`` maps an error category to a set of (table, row_id)."""
    rng = np.random.default_rng(seed)
    persons = []
    for i in range(n_person):
        sex = "M" if rng.random() < 0.5 else "F"
        persons.append(dict(pid=f"{1000 + i}", sex=sex, eth=rng.choice(ETH, p=[.6, .15, .1, .1, .05]),
                            height=rng.normal(176, 7) if sex == "M" else rng.normal(163, 6),
                            w0=float(np.exp(rng.normal(np.log(80), 0.18))),
                            birth=2100 - int(rng.integers(20, 85)) - float(rng.random()),
                            n_adm=int(rng.choice([1, 2, 3], p=[.5, .32, .18])), died=rng.random() < 0.15))
    stays, labs, treats = [], [], []
    k_stay = k_adm = k_lab = 0
    for p in persons:
        yrs = np.sort(rng.choice(np.arange(10), p["n_adm"], replace=False))
        for a_i, y in enumerate(yrs):
            k_adm += 1
            adm_id = f"{5000 + k_adm}"
            n_st = int(rng.choice([1, 2], p=[.75, .25]))
            adm_start = pd.Timestamp(f"{2100 + int(y)}-01-01") + pd.Timedelta(days=int(rng.integers(0, 300)))
            t_cursor = adm_start + pd.Timedelta(minutes=int(rng.integers(60, 3000)))
            for s_i in range(n_st):
                k_stay += 1
                last = p["died"] and a_i == len(yrs) - 1 and s_i == n_st - 1
                los = int(rng.integers(1000, 8000))
                hadm_off = -int((t_cursor - adm_start).total_seconds() // 60)
                hdis_off = los if last else los + int(rng.integers(500, 5000))
                age_y = (t_cursor - pd.Timestamp("2100-01-01")).days / 365.25 + 2100 - p["birth"]
                wa = p["w0"] * float(np.exp(rng.normal(0, 0.04)))
                st = dict(stay=f"{200000 + k_stay}", adm=adm_id, pid=p["pid"], sex=p["sex"], eth=p["eth"],
                          age=int(age_y), height=round(p["height"] + rng.normal(0, 0.8), 1), wa=round(wa, 1),
                          wd=round(wa * float(np.exp(rng.normal(0, 0.02))), 1), hadm_off=hadm_off, udis_off=los,
                          ustatus="Expired" if last else "Alive", hdis_off=hdis_off,
                          hstatus="Expired" if last else "Alive", year=adm_start.year, intime=t_cursor,
                          adm_start=adm_start, last=last)
                stays.append(st)
                # distinct draw instants per stay (two draws of one item at one minute would be a
                # genuine duplicate conflict)
                for off in rng.choice(np.arange(max(hadm_off, -600), hdis_off), 12, replace=False):
                    k_lab += 1
                    it = rng.choice(list(LABS))
                    mu, sd, u = LABS[it]
                    v = round(float(np.exp(rng.normal(0, sd))) if mu is None else float(rng.normal(mu, sd)), 2)
                    off = int(off)
                    labs.append(dict(lid=f"{k_lab}", stay=st["stay"], adm=adm_id, pid=p["pid"], off=off,
                                     item=it, v=v, u=u, t=t_cursor + pd.Timedelta(minutes=off)))
                for _ in range(3):
                    off = int(rng.integers(0, los))
                    treats.append(dict(stay=st["stay"], off=off, name=rng.choice(["abx", "vent", "pressor"])))
                t_cursor += pd.Timedelta(minutes=hdis_off + int(rng.integers(100, 2000)))
    S, L = pd.DataFrame(stays), pd.DataFrame(labs)
    truth = {k: set() for k in ("after_death", "demographic_change", "age_shift", "height_unit",
                                "weight_lb", "duplicate")}

    if layout == "offset":
        T = {"unit_stays": pd.DataFrame({
                "_row_id": "S" + S.index.astype(str), "unit_stay_id": S.stay, "hosp_adm_id": S.adm,
                "person_id": S.pid, "gender": S.sex.map({"M": "Male", "F": "Female"}), "age": S.age.astype(str),
                "ethnicity": S.eth, "height_cm": S.height.astype(str), "admit_weight": S.wa.astype(str),
                "discharge_weight": S.wd.astype(str), "hosp_admit_offset": S.hadm_off.astype(str),
                "unit_discharge_offset": S.udis_off.astype(str), "unit_discharge_status": S.ustatus,
                "hosp_discharge_offset": S.hdis_off.astype(str), "hosp_discharge_status": S.hstatus,
                "discharge_year": S.year.astype(str)}),
             "lab": pd.DataFrame({"_row_id": "L" + L.index.astype(str), "lab_id": L.lid, "unit_stay_id": L.stay,
                                  "lab_result_offset": L.off.astype(str), "lab_name": L["item"],
                                  "lab_result": L.v.astype(str)}),
             "treatment": pd.DataFrame({"_row_id": [f"T{i}" for i in range(len(treats))],
                                        "treatment_id": [f"{9000 + i}" for i in range(len(treats))],
                                        "unit_stay_id": [t["stay"] for t in treats],
                                        "treatment_offset": [str(t["off"]) for t in treats],
                                        "treatment_name": [t["name"] for t in treats]})}
        stay_t, eth_c, age_c, h_c, w_c, lab_t = "unit_stays", "ethnicity", "age", "height_cm", "admit_weight", "lab"
        multi_stay = S.pid.map(S.pid.value_counts()) >= 2
    else:
        P = S.groupby("pid").agg(sex=("sex", "first"), age=("age", "min")).reset_index()
        dod = S[S["last"]].set_index("pid").apply(lambda r: r.intime + pd.Timedelta(minutes=r.hdis_off), axis=1)
        A = S.groupby("adm").agg(pid=("pid", "first"), eth=("eth", "first"), start=("adm_start", "first"),
                                 end=("intime", "max")).reset_index()
        T = {"patients": pd.DataFrame({"_row_id": "P" + P.index.astype(str), "subject_id": P.pid, "gender": P.sex,
                                       "anchor_age": P.age.astype(str),
                                       "dod": P.pid.map(dod).map(lambda x: "" if pd.isna(x) else str(x.date()))}),
             "admissions": pd.DataFrame({"_row_id": "A" + A.index.astype(str), "hadm_id": A.adm, "subject_id": A.pid,
                                         "admittime": A.start.astype(str),
                                         "dischtime": (A.end + pd.Timedelta(days=12)).astype(str), "race": A.eth}),
             "icustays": pd.DataFrame({"_row_id": "I" + S.index.astype(str), "stay_id": S.stay, "hadm_id": S.adm,
                                       "subject_id": S.pid, "intime": S.intime.astype(str),
                                       "outtime": (S.intime + pd.to_timedelta(S.udis_off, unit="m")).astype(str),
                                       "admission_age": S.age.astype(str), "height": S.height.astype(str),
                                       "weight_admit": S.wa.astype(str), "weight_discharge": S.wd.astype(str)}),
             "labevents": pd.DataFrame({"_row_id": "L" + L.index.astype(str), "subject_id": L.pid, "hadm_id": L.adm,
                                        "stay_id": L.stay, "itemid": L["item"], "charttime": L.t.astype(str),
                                        "valuenum": L.v.astype(str), "valueuom": L.u}),
             "d_labitems": pd.DataFrame({"_row_id": ["D0", "D1", "D2"], "itemid": list(LABS),
                                         "label": ["hemoglobin", "sodium", "creatinine"]})}
        stay_t, eth_c, age_c, h_c, w_c, lab_t = "icustays", "race", "admission_age", "height", "weight_admit", "labevents"
        multi_stay = S.pid.map(S.pid.value_counts()) >= 2
    if not inject:
        return T, truth

    ST, LB = T[stay_t], T[lab_t]
    # 1. events after death
    dead_stays = set(S.stay[S["last"]])
    cand = L.index[L.stay.isin(dead_stays)]
    for i in rng.choice(cand, 3, replace=False):
        st = S[S.stay == L.loc[i, "stay"]].iloc[0]
        if layout == "offset":
            LB.loc[i, "lab_result_offset"] = str(st.hdis_off + int(rng.integers(600, 4000)))
        else:
            LB.loc[i, "charttime"] = str(st.intime + pd.Timedelta(minutes=int(st.hdis_off))
                                         + pd.Timedelta(days=int(rng.integers(2, 10))))
        truth["after_death"].add((lab_t, LB.loc[i, "_row_id"]))
    # 2. demographic attributes changing between stays / admissions of one person
    if layout == "offset":
        idx = rng.choice(ST.index[multi_stay.values], 5, replace=False)
        for j, i in enumerate(idx):
            if j < 3:
                ST.loc[i, eth_c] = rng.choice([e for e in ETH if e != ST.loc[i, eth_c]])
            else:
                ST.loc[i, "gender"] = "Female" if ST.loc[i, "gender"] == "Male" else "Male"
            truth["demographic_change"].add((stay_t, ST.loc[i, "_row_id"]))
    else:
        AD = T["admissions"]
        multi_adm = AD.subject_id.map(AD.subject_id.value_counts()) >= 2
        for i in rng.choice(AD.index[multi_adm.values], 3, replace=False):
            AD.loc[i, eth_c] = rng.choice([e for e in ETH if e != AD.loc[i, eth_c]])
            truth["demographic_change"].add(("admissions", AD.loc[i, "_row_id"]))
    pool = list(rng.permutation(ST.index[multi_stay.values]))
    # 3. age inconsistent with the person's other stays
    for i in pool[:2]:
        ST.loc[i, age_c] = str(int(ST.loc[i, age_c]) + int(rng.choice([-1, 1]) * rng.integers(12, 25)))
        truth["age_shift"].add((stay_t, ST.loc[i, "_row_id"]))
    # 4. height recorded in inches, 5. admission weight recorded in pounds
    rest = [i for i in rng.permutation(ST.index) if i not in pool[:2]]
    for i in rest[:2]:
        ST.loc[i, h_c] = str(round(float(ST.loc[i, h_c]) / 2.54, 1))
        truth["height_unit"].add((stay_t, ST.loc[i, "_row_id"]))
    for i in rest[2:5]:
        ST.loc[i, w_c] = str(round(float(ST.loc[i, w_c]) * 2.2046, 1))
        truth["weight_lb"].add((stay_t, ST.loc[i, "_row_id"]))
    # 6. exact duplicate event rows (fresh row id and fresh table-local surrogate id)
    dup = LB.loc[rng.choice(LB.index, 5, replace=False)].copy()
    dup["_row_id"] = [f"DUP{i}" for i in range(len(dup))]
    if "lab_id" in dup:
        dup["lab_id"] = [f"{900000 + i}" for i in range(len(dup))]
    T[lab_t] = pd.concat([LB, dup], ignore_index=True)
    truth["duplicate"] |= {(lab_t, r) for r in dup["_row_id"]}
    return T, truth


def score(findings, truth):
    fl = set(zip(findings["table"], findings["row_id"]))
    allt = set().union(*truth.values())
    rec = {k: (len(v & fl) / len(v) if v else float("nan")) for k, v in truth.items()}
    return rec, (len(fl & allt) / len(fl) if fl else float("nan")), len(fl)


def test_offset_schema():
    T, _ = make_icu_cohort(0, "offset", inject=False)
    s = infer_clinical_schema(T)
    assert s["subject"] == "unit_stay_id"
    assert s["person"] == "person_id"
    assert s["id_hierarchy"] == ["unit_stay_id", "hosp_adm_id", "person_id"]
    assert s["stays"]["unit_stays"]["key"] == "unit_stay_id"
    assert s["events"]["lab"]["kind"] == "offset" and s["events"]["lab"]["time"] == "lab_result_offset"
    pairs = {(d["status"], d["time"]) for d in s["demographics"]["death_status"]}
    assert ("hosp_discharge_status", "hosp_discharge_offset") in pairs
    assert s["measurements"]["lab"]["item"] == "lab_name"


def test_date_schema_hierarchy():
    T, _ = make_icu_cohort(0, "date", inject=False)
    s = infer_clinical_schema(T)
    assert s["subject"] == "subject_id" and s["person"] == "subject_id"
    assert s["id_hierarchy"] == ["stay_id", "hadm_id", "subject_id"]
    assert set(s["stays"]) == {"admissions", "icustays"}


def _check(layout, seed):
    T, truth = make_icu_cohort(seed, layout)
    f = audit_clinical_tables(T, external_reference=False, verbose=False)
    rec, prec, n = score(f, truth)
    for k in ("after_death", "demographic_change", "height_unit", "weight_lb", "duplicate", "age_shift"):
        assert rec[k] >= (0.6 if k in ("demographic_change", "age_shift") else 0.99), (layout, seed, k, rec)
    assert prec >= 0.5, (layout, seed, prec, n)
    return f


def test_offset_layout_injected():
    f = _check("offset", 1)
    assert {"after_death", "person_constant", "person_numeric", "unit_factor", "exact_duplicate"} <= set(f["rule"])
    assert (f.loc[f["rule"] == "exact_duplicate", "family"] == "duplicate").all()


def test_date_layout_injected():
    _check("date", 2)


def test_clean_cohort_quiet():
    for layout in ("offset", "date"):
        T, _ = make_icu_cohort(3, layout, inject=False)
        f = audit_clinical_tables(T, external_reference=False, verbose=False)
        n_rows = sum(len(d) for d in T.values())
        assert f.drop_duplicates(["table", "row_id"]).shape[0] <= 0.005 * n_rows, f["rule"].value_counts()


def test_severity_rate_and_cap():
    T, _ = make_icu_cohort(4, "offset")
    f = audit_clinical_tables(T, external_reference=False, verbose=False)
    assert {"severity", "rule_rate", "rule_rank"} <= set(f.columns)
    assert f["rule_rate"].between(0, 1).all() and (f["rule_rank"] >= 1).all()
    g = audit_clinical_tables(T, external_reference=False, verbose=False, max_rule_rate=1e-9)
    # logical rules (NaN severity) survive the cap; graded rules keep their top-ranked row per table
    assert set(f.loc[f["severity"].isna(), "rule"]) <= set(g["rule"])
    graded = g[g["severity"].notna()]
    assert (graded["rule_rank"] <= 1).all()


def test_without_shared_row_id():
    T, truth = make_icu_cohort(5, "offset")
    T = {n: d.drop(columns="_row_id") for n, d in T.items()}
    f = audit_clinical_tables(T, external_reference=False, verbose=False)
    assert "after_death" in set(f["rule"])
