r"""Data-entry error audit for linked clinical record tables (EHR extracts).

A clinical-data-quality audit flags *rows* whose recorded value is wrong for
that patient: physically impossible, on the wrong scale or unit, or in
contradiction with other records of the same patient / event.  Every rule
here derives its reference from the data (or from a public database queried
at run time), never from a hand-written table of analytes, drugs or limits:

==========================  =================================================
family / rule               what is compared
==========================  =================================================
impossible_value
  ``unit_label``            unit string vs. the unit the rest of that item's
                            rows carry (rare deviations only)
  ``item_extreme``          value vs. the robust spread of the same item over
                            the whole corpus (log scale for positive items),
                            and only when it is isolated from every other
                            value of the item
  ``sign_flip``             negative value of an item that is otherwise never
                            negative
  ``subject_jump``          value vs. the same patient's own history of the
                            item (power-of-ten slips, unit-conversion factors)
  ``dose_outlier``          a prescribed dose vs. every other prescription of
                            the same drug (power-of-ten slips, extremes)
inconsistency
  ``duplicate_conflict``    two rows of one observation (same specimen + item,
                            or same patient + item + instant) with different
                            values
  ``cross_table_conflict``  the same quantity recorded in two tables at one
                            instant; the item pairs are *discovered* as pairs
                            whose co-timed values agree almost always
  ``time_order``            end before start inside one row
demographic_conflict
  ``sex_reference_band``    recorded sex vs. the sex-specific reference band
                            attached to the patient's own lab rows (band↔sex
                            association learned from the cohort)
  ``sex_only_item``         recorded sex vs. items / drugs seen in only one
                            sex across the cohort
  ``sex_external``          recorded sex vs. the reporting sex ratio of the
                            drug in FDA adverse-event reports (openFDA)
  ``age_isolated``          an age isolated from the rest of the cohort by a
                            gap far larger than the cohort's typical spacing
  ``person_constant``       a person-level category (sex, ethnicity) that
                            changes between the stays of one person
  ``person_numeric``        height / weight / age-minus-elapsed-time that
                            jumps between the stays of one person
impossible_value (time)
  ``after_death``           event after death: a date of death, or a death
                            *status* ("Expired") at a dated / offset discharge
  ``outside_stay``          offset-stamped event outside the subject's window
  ``unit_factor``           value a unit factor (lb/kg, in/cm, x10) away from
                            its column, a sibling column or the person's
                            other stays
duplicate
  ``exact_duplicate``       identical copy of another row (ids excluded)
==========================  =================================================

Nested identifiers (person > hospital admission > ICU stay) are detected from the data
(``schema['id_hierarchy']``: an id whose values each map to one value of a coarser id), and
relative times (``*offset``, ``minutes_from_*``) are recognised as ``kind='offset'`` events.

Every finding also carries ``severity`` (robust z of the flagged value against its reference;
NaN for logical rules) and ``rule_rate`` (share of the table's rows flagged by that rule);
``max_rule_rate`` keeps only the most severe rows of a rule that fires too broadly.

Every finding carries a ``cluster`` id: rows that are evidence for the same
incident (a patient's sex contradicted by 40 lab rows, the two rows of a
disagreeing duplicate) share it, so a reviewer reads incidents rather than
rows.  The thresholds are screening thresholds: the audit is tuned to catch
(missing an incident is costly, an extra flag is cheap), and ``min_score``
or per-rule filtering trades recall for precision afterwards.

Background: the rule families follow the conformance / plausibility (atemporal,
temporal, uniqueness) categories of Kahn et al. 2016 (eGEMs 4:1244); the
after-death / outside-stay rules are the data-derived counterparts of the OHDSI
Data Quality Dashboard ``plausibleDuringLife`` / ``plausibleBeforeDeath`` checks,
extended to relative (offset) time encodings, and the person-level rules check
attributes that must not change across a person's stays (cross-stay
consistency) on an id hierarchy discovered from the data.
"""
from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .._registry import register_function

__all__ = ["load_clinical_tables", "infer_clinical_schema", "audit_clinical_tables",
           "drug_sex_reference"]

_FINDING_COLS = ["table", "row_id", "family", "rule", "cluster", "score", "evidence", "severity",
                 "rule_rate", "rule_rank"]


# ---------------------------------------------------------------------------
# loading and schema inference
# ---------------------------------------------------------------------------

@register_function(
    aliases=["加载临床表", "load EHR tables", "load clinical tables", "read EHR csv directory"],
    category="utils",
    description="Read every table of a clinical / EHR extract from a directory (csv, csv.gz, tsv, "
                "parquet) as strings with empty cells kept as '' (no NA coercion), keyed by table name.",
    examples=["tables = ov.tl.load_clinical_tables('/data/csv')"],
    related=["tl.audit_clinical_tables"],
)
def load_clinical_tables(path, pattern: str = "*") -> Dict[str, pd.DataFrame]:
    """Return ``{table_name: DataFrame}`` for every tabular file under ``path``.

    Values are read as strings and empty cells stay ``''`` so that the audit
    sees the exact recorded text (unit strings, blank reference ranges).
    """
    path = Path(path)
    out: Dict[str, pd.DataFrame] = {}
    for f in sorted(path.glob(pattern)):
        name = f.name
        for suf in (".csv.gz", ".tsv.gz", ".csv", ".tsv", ".parquet", ".txt.gz", ".txt"):
            if name.endswith(suf):
                key = name[: -len(suf)]
                if suf == ".parquet":
                    df = pd.read_parquet(f).astype(str)
                else:
                    sep = "\t" if "tsv" in suf else ","
                    df = pd.read_csv(f, dtype=str, keep_default_na=False, sep=sep)
                out[key] = df
                break
    if not out:
        raise FileNotFoundError(f"no tabular files under {path}")
    return out


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _first(cols: Sequence[str], patterns: Sequence[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for p in patterns:
        for lc, c in low.items():
            if re.fullmatch(p, lc):
                return c
    return None


_SEX_VALUES = {"m", "f", "male", "female", "man", "woman", "1", "2"}

# Name patterns (lower-cased column names) used for the relative-time / hierarchy roles.
# They describe *kinds* of columns, not any particular dataset:
#   _ID_RE       an identifier column (``..._id``, ``...id``, ``..._key``)
#   _OFFSET_RE   a relative time: minutes/hours/days from an anchor (``*offset``,
#                ``*_rel_time``, ``minutes_from_*``, ``*_mins_since_*``)
#   _TIMEISH_RE  any time-like column (absolute or relative); never an attribute
#   _DEATH_RE    status values that mean the patient died
_ID_RE = r"(id|key)$"
_OFFSET_RE = (r"(.*offset(_?(min(ute)?s?|hours?|hrs?|days?))?|.*rel(ative)?_?time.*"
              r"|.*(min(ute)?s?|hours?|hrs?|days?)_?(from|since|after)_.*)")
_TIMEISH_RE = r"(time|date|offset|year|_dt$|^dt_|dttm|_rel_|minutes|hours|days)"
_DEATH_RE = r"(expired|dead|deceased|died|death)"
_PERSONISH_RE = r"(subject|patient|person|member|mrn|pid|uid|health|hadm|admission|encounter|visit|stay)"
_FACILITY_RE = r"(hospital|ward|site|cent(er|re)|facility|provider|caregiver|physician|org|dept|department)"
_MISSING = {"", "nan", "none", "null", "na", "n/a", "unknown", "unk", "?"}


def _is_numeric_col(s: pd.Series, min_share: float = 0.9) -> bool:
    x = s.astype(str).str.strip()
    x = x[~x.str.lower().isin(_MISSING)]
    if x.empty:
        return False
    return _num(x).notna().mean() >= min_share


def _infer_id_hierarchy(tables: Mapping[str, pd.DataFrame], row_id, subject, skip=(),
                        min_ratio: float = 0.2) -> List[str]:
    """Nested identifier levels connected to ``subject``, finest first.

    ``a`` is nested in ``b`` when, in every table holding both, each value of ``a`` maps to
    exactly one value of ``b`` and ``b`` has fewer distinct values.  Facility-type ids (hospital,
    ward) are also functions of a stay id but group hundreds of stays each; a person-type level
    groups a few, so ``b`` must keep at least ``min_ratio`` of ``a``'s distinct values,
    unless its name says it is a person / patient / admission level (``_PERSONISH_RE``); facility
    names (``_FACILITY_RE``) are never accepted as a level.
    """
    if not subject:
        return []
    idc = sorted({c for n, df in tables.items() if n not in skip for c in df.columns
                  if c != row_id and re.search(_ID_RE, c.lower())})
    # a table's own surrogate row key (unique there, referenced nowhere else) is not a level
    holders = {c: [df for n, df in tables.items() if n not in skip and c in df.columns] for c in idc}
    idc = [c for c in idc if c == subject or not (len(holders[c]) == 1 and holders[c][0][c].is_unique)]
    nuniq = {c: len(set().union(*[set(df[c].astype(str)) for n, df in tables.items()
                                   if n not in skip and c in df.columns]) - {"", "nan"})
             for c in idc}
    edges = {}
    for a in idc:
        for b in idc:
            if a == b or nuniq[b] >= nuniq[a] or nuniq[b] < 2:
                continue
            bl = b.lower()
            personish = re.search(_PERSONISH_RE, bl)
            facility = re.search(_FACILITY_RE, bl) and not personish
            if facility or (nuniq[b] < min_ratio * nuniq[a] and not personish):
                continue
            ok, seen = True, False
            for n, df in tables.items():
                if n in skip or a not in df.columns or b not in df.columns:
                    continue
                seen = True
                pair = df[[a, b]].astype(str)
                pair = pair[(pair[a] != "") & (pair[b] != "")]
                if pair.groupby(a)[b].nunique().max() > 1:
                    ok = False
                    break
            if ok and seen:
                edges.setdefault(a, set()).add(b)
                edges.setdefault(b, set()).add(a)
    comp, todo = {subject}, [subject]
    while todo:
        c = todo.pop()
        for d in edges.get(c, ()):
            if d not in comp:
                comp.add(d)
                todo.append(d)
    # keep a chain: every level must be nested in the next coarser one
    chain = sorted(comp, key=lambda c: -nuniq.get(c, 0))
    return chain if len(chain) > 1 else [subject]


def _death_pairs(df: pd.DataFrame, exclude=()) -> List[dict]:
    """(status column, time column) pairs encoding death in one table.

    A status column is categorical (<= 12 distinct values) with at least one value matching
    ``_DEATH_RE`` and at least one that does not ("Alive", "Home" ...).  Its time column is
    the date / offset column sharing the longest name prefix with it (>= 3 characters:
    ``unit_discharge_status`` <-> ``unit_discharge_offset``).
    """
    out = []
    cols = [c for c in df.columns if c not in exclude]
    timec = [c for c in cols if re.fullmatch(_OFFSET_RE, c.lower()) or re.search(r"(time|date)", c.lower())]
    for c in cols:
        if c in timec:
            continue
        vals = set(df[c].astype(str).str.strip().str.lower()) - _MISSING
        if not (2 <= len(vals) <= 12):
            continue
        dead = {v for v in vals if re.search(_DEATH_RE, v) and not re.search(r"(not|non|alive)", v)}
        if not dead or dead == vals:
            continue
        best, blen = None, 2
        for t in timec:
            k = len(os.path.commonprefix([c.lower(), t.lower()]))
            if k > blen:
                best, blen = t, k
        if best is None:
            continue
        kind = "offset" if re.fullmatch(_OFFSET_RE, best.lower()) or _is_numeric_col(df[best]) else "date"
        out.append({"status": c, "dead_values": sorted(dead), "time": best, "kind": kind})
    return out


@register_function(
    aliases=["推断临床表结构", "infer EHR schema", "clinical schema inference"],
    category="utils",
    description="Infer the roles of the tables and columns of a linked clinical extract: patient key, "
                "demographics table (sex / age columns), measurement tables (item, numeric value, unit, "
                "time, specimen, reference range), dictionaries, prescriptions (drug, dose, dose unit) "
                "and start/end time pairs. Every role can be overridden.",
    examples=["schema = ov.tl.infer_clinical_schema(tables)"],
    related=["tl.audit_clinical_tables"],
)
def infer_clinical_schema(tables: Mapping[str, pd.DataFrame], overrides: Optional[dict] = None) -> dict:
    """Guess column roles from names and contents; ``overrides`` wins.

    Returns a dict with ``row_id``, ``subject``, ``demographics`` {table, sex, age},
    ``measurements`` {table: {item, value, unit, time, specimen, ref_lo, ref_hi, text}},
    ``dictionaries`` {table: {item, label}}, ``prescriptions`` {table: {drug, dose, dose_unit,
    item}} and ``time_pairs`` {table: [(start, end), ...]}.
    """
    overrides = dict(overrides or {})
    names = list(tables)
    allcols = [set(df.columns) for df in tables.values()]
    common = set.intersection(*allcols) if allcols else set()
    row_id = overrides.get("row_id") or next((c for c in ("_row_id", "row_id", "rowid")
                                              if c in common), None)
    # patient key: id-like column shared by most tables, unique in exactly one small table
    counts: Dict[str, int] = {}
    for df in tables.values():
        for c in df.columns:
            if re.search(r"(subject|patient|person|member|mrn)", c.lower()) and c != row_id:
                counts[c] = counts.get(c, 0) + 1
    subject = overrides.get("subject") or (max(counts, key=counts.get) if counts else None)
    # nested ids (person > admission > ICU stay): when the name-matched key lives in a minority of
    # the tables, the tables are linked by a finer level (a stay id nested in it) -- use that
    # level as the linking subject and keep the coarse one as ``person``.
    if subject and not overrides.get("subject"):
        n_with = sum(subject in df.columns for df in tables.values())
        if n_with < 0.5 * len(tables):
            hier = _infer_id_hierarchy(tables, row_id, subject)
            finer = [c for c in hier[:hier.index(subject)]] if subject in hier else []
            if finer:
                use = {c: sum(c in df.columns for df in tables.values()) for c in finer}
                best = max(finer, key=lambda c: (use[c], -finer.index(c)))
                if use[best] > n_with:
                    subject = best

    demo = overrides.get("demographics")
    if demo is None and subject:
        best = None
        for n, df in tables.items():
            if subject not in df.columns or not df[subject].is_unique:
                continue
            sexc = None
            for c in df.columns:
                if re.search(r"(sex|gender)", c.lower()):
                    vals = set(df[c].astype(str).str.strip().str.lower()) - {""}
                    if vals and vals <= _SEX_VALUES | {"u", "unknown", "o", "other"}:
                        sexc = c
            agec = _first(df.columns, [r"(anchor_)?age(_years)?", r".*_age", r"age.*"])
            if sexc or agec:
                best = {"table": n, "sex": sexc, "age": agec}
                break
        demo = best

    meas, dicts, rx, pairs = {}, {}, {}, {}
    for n, df in tables.items():
        cols = list(df.columns)
        item = _first(cols, [r"itemid", r"item_id", r"test_?(id|code)", r"(lab|loinc|concept)_?(id|code)",
                             r"code"])
        label = _first(cols, [r"label", r"name", r"(test|item)_?name", r"description"])
        is_dict = item is not None and df[item].is_unique and label is not None and subject not in cols
        if is_dict:
            dicts[n] = {"item": item, "label": label}
            continue
        drug = _first(cols, [r"drug", r"drug_?name", r"medication", r"med_?name"])
        if drug and subject in cols:
            rx[n] = {"drug": drug,
                     "dose": _first(cols, [r"dose_val_rx", r"dose(_val(ue)?)?", r"dose_amount", r"amount"]),
                     "dose_unit": _first(cols, [r"dose_unit_rx", r"dose_?unit", r"unit"]),
                     "item": _first(cols, [r"ndc", r"gsn", r"formulary_drug_cd"])}
        value = _first(cols, [r"valuenum", r"value_?num(eric)?", r"result_?num(eric)?",
                              r"value_as_number"])
        if value is None and item is not None:
            value = _first(cols, [r"value", r"result"])
        if value is None and subject in cols:
            # long tables keyed by a test *name* with a numeric result column (``lab_name`` /
            # ``lab_result``): accepted only when the result column is mostly numeric
            it2 = _first(cols, [r"(lab|test|item|obs(ervation)?|measure(ment)?)_?name"])
            v2 = _first(cols, [r"(lab|test|obs(ervation)?)_?result(_?num(eric)?)?", r"(lab|test)_?value"])
            if it2 and v2 and _is_numeric_col(df[v2].head(2000), 0.5):
                item, value = it2, v2
        if item and value and subject in cols:
            tm = _first(cols, [r"charttime", r"(obs|result|event|measure(ment)?)_?(time|date|datetime)",
                               r"time", r"datetime", r"date"])
            tkind = "date"
            if tm is None:
                tm = next((c for c in cols if re.fullmatch(_OFFSET_RE, c.lower())
                           and _is_numeric_col(df[c].head(2000))), None)
                tkind = "offset" if tm else None
            meas[n] = {
                "item": item, "value": value,
                "text": _first(cols, [r"value", r"result"]) if value.lower() != "value" else None,
                "unit": _first(cols, [r"valueuom", r"unit(s)?", r"uom", r"value_?unit", r"unit_?of_?measure"]),
                "time": tm, "time_kind": tkind,
                "specimen": _first(cols, [r"specimen_?id", r"sample_?id", r"order_?id", r"accession"]),
                "ref_lo": _first(cols, [r"ref_range_lower", r"(ref|normal)_?(low|lower|min)"]),
                "ref_hi": _first(cols, [r"ref_range_upper", r"(ref|normal)_?(high|upper|max)"]),
            }
        tp = []
        for a, b in (("admittime", "dischtime"), ("intime", "outtime"), ("starttime", "stoptime"),
                     ("starttime", "endtime"), ("start_date", "end_date"), ("edregtime", "edouttime"),
                     ("charttime", "storetime"), ("start", "stop"), ("start", "end")):
            if a in cols and b in cols:
                tp.append((a, b))
        if tp:
            pairs[n] = tp
    if demo:
        D = tables[demo["table"]]
        demo.setdefault("birth", _first(list(D.columns), [r"birth_?date", r"dob", r"date_?of_?birth", r"birth_?datetime"]))
        demo.setdefault("death", _first(list(D.columns), [r"death_?date", r"dod", r"date_?of_?death", r"death_?datetime",
                                                          r"deceased_?date"]))
        cats = []
        for c in D.columns:
            if c in (row_id, subject, demo.get("sex"), demo.get("age"), demo.get("birth"), demo.get("death")):
                continue
            nun = D[c].astype(str).nunique()
            if 2 <= nun <= 30 and not re.search(r"(id|date|time|year|zip|postal|address|name)$", c.lower()):
                cats.append(c)
        demo.setdefault("categories", cats)
        # death recorded as a status ("Expired") plus the time of that status (date or offset)
        demo.setdefault("death_status", _death_pairs(D, exclude=(row_id, subject)))
    # event tables: subject + a date / time column; visit tables: an id unique there and repeated elsewhere
    events, visits = {}, {}
    for n, df in tables.items():
        cols = list(df.columns)
        if subject not in cols or (demo and n == demo["table"]) or n in dicts:
            continue
        t0 = _first(cols, [r"charttime", r"start(time|_?date|_?datetime)?", r"admittime", r"intime", r"date",
                           r"(event|obs|observation|service|visit)_?(date|time|datetime)", r"datetime", r"time"])
        kind = "date"
        if not t0:
            # relative times: minutes (hours, days) from the subject's anchor (e.g. unit admission)
            offs = [c for c in cols if re.fullmatch(_OFFSET_RE, c.lower()) and _is_numeric_col(df[c].head(2000))]
            if not offs:
                continue
            t0, kind = offs[0], "offset"
        t1 = _first(cols, [r"stop(time)?", r"end(time|_?date)?", r"dischtime", r"outtime"]) if kind == "date" else \
            next((c for c in cols if c != t0 and re.fullmatch(_OFFSET_RE, c.lower())
                  and re.search(r"(stop|end|out|disch)", c.lower())), None)
        events[n] = {"time": t0, "end": t1, "kind": kind,
                     "item": _first(cols, [r"itemid", r"code", r"item_?id", r"concept_?id"]),
                     "label": _first(cols, [r"description", r"label", r"drug", r"name"])}
        if kind == "offset" and events[n]["item"] is None:
            events[n]["item"] = _first(cols, [r".*(name|string|type|code)"])
    for n, df in tables.items():
        for c in df.columns:
            if c in (row_id, subject) or not re.search(r"(^|_)(id|key)$", c.lower()) or not df[c].is_unique:
                continue
            users = [m for m, d2 in tables.items() if m != n and c in d2.columns]
            if users and subject in df.columns and n in events and events[n].get("kind") == "date":
                visits[n] = {"key": c, "start": events[n]["time"], "end": events[n]["end"], "users": users}
    hier = overrides.get("id_hierarchy") or _infer_id_hierarchy(tables, row_id, subject, skip=set(dicts))
    person = overrides.get("person") or (hier[-1] if hier else subject)
    # stay tables: one row per level-of-hierarchy unit (a level id unique there) with the person id,
    # so attributes can be compared between the stays of one person
    stays = {}
    for n, df in tables.items():
        if n in dicts or person not in df.columns:
            continue
        lev = next((c for c in hier if c != person and c in df.columns and df[c].is_unique), None)
        if lev and not df[person].is_unique:
            stays[n] = {"key": lev, "person": person}
    # relative-time window of each subject: offset columns of the subject-level table
    offset_window = None
    if demo and subject:
        D = tables[demo["table"]]
        oc = [c for c in D.columns if re.fullmatch(_OFFSET_RE, c.lower()) and _is_numeric_col(D[c])]
        if oc:
            offset_window = {"table": demo["table"], "columns": oc}
    schema = {"row_id": row_id, "subject": subject, "demographics": demo, "measurements": meas,
              "dictionaries": dicts, "prescriptions": rx, "time_pairs": pairs, "events": events,
              "visits": visits, "id_hierarchy": hier, "person": person, "stays": stays,
              "offset_window": offset_window}
    for k, v in overrides.items():
        if k in ("measurements", "dictionaries", "prescriptions", "time_pairs", "events", "visits",
                 "stays") and isinstance(v, dict):
            schema[k].update(v)
        else:
            schema[k] = v
    return schema


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _Findings:
    def __init__(self):
        self.parts: List[pd.DataFrame] = []

    def add(self, table, row_ids, family, rule, cluster, score, evidence, severity=np.nan):
        row_ids = pd.Series(row_ids).astype(str).reset_index(drop=True)
        if row_ids.empty:
            return
        n = len(row_ids)

        def _vec(x):
            if isinstance(x, (pd.Series, np.ndarray, list)):
                return pd.Series(x).reset_index(drop=True)
            return pd.Series([x] * n)

        self.parts.append(pd.DataFrame({
            "table": table, "row_id": row_ids, "family": family, "rule": rule,
            "cluster": _vec(cluster).astype(str), "score": _vec(score).astype(float),
            "evidence": _vec(evidence).astype(str), "severity": _vec(severity).astype(float)}))

    def frame(self) -> pd.DataFrame:
        if not self.parts:
            return pd.DataFrame(columns=_FINDING_COLS)
        out = pd.concat(self.parts, ignore_index=True)
        out["rule_rate"] = np.nan
        out["rule_rank"] = np.nan
        return out[_FINDING_COLS]


def _rank_findings(out: pd.DataFrame, tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach ``rule_rate`` (share of the table's rows the rule flagged: a rule that fires on many
    rows is a weak, low-precision screen there) and ``rule_rank`` (1 = most severe finding of that
    rule in that table, by ``severity`` then ``score``)."""
    if out.empty:
        return out
    n_rows = {t: max(len(df), 1) for t, df in tables.items()}
    nflag = out.groupby(["table", "rule"])["row_id"].transform("nunique")
    out["rule_rate"] = nflag / out["table"].map(n_rows).astype(float)
    key = out["severity"].fillna(np.inf)
    tmp = out.assign(_k=key, _s=out["score"])
    tmp = tmp.sort_values(["_k", "_s"], ascending=False)
    tmp["rule_rank"] = tmp.groupby(["table", "rule"]).cumcount() + 1
    out["rule_rank"] = tmp["rule_rank"].reindex(out.index).astype(float)
    return out


def _robust_center_scale(v: pd.Series, by: pd.Series):
    g = v.groupby(by)
    med = g.transform("median")
    q1 = g.transform(lambda s: s.quantile(0.25))
    q3 = g.transform(lambda s: s.quantile(0.75))
    mad = (v - med).abs().groupby(by).transform("median") * 1.4826
    iqr = (q3 - q1) / 1.349
    scale = np.maximum(mad.fillna(0), iqr.fillna(0))
    return med, scale


def _isolation_ratio(v: pd.Series, by: pd.Series) -> pd.Series:
    """Distance of each value to its nearest neighbour within the group, in units of the
    group's median nearest-neighbour spacing of distinct values (large = isolated)."""
    df = pd.DataFrame({"v": v.values, "g": by.values}, index=v.index)
    df = df.sort_values(["g", "v"])
    prev = df.groupby("g")["v"].shift(1)
    nxt = df.groupby("g")["v"].shift(-1)
    gap = pd.concat([(df["v"] - prev).abs(), (nxt - df["v"]).abs()], axis=1).min(axis=1)
    distinct = df.drop_duplicates(["g", "v"])
    sp = distinct.groupby("g")["v"].apply(lambda s: np.median(np.diff(s.values)) if len(s) > 2 else np.nan)
    ref = df["g"].map(sp)
    ref = ref.where(ref > 0)
    return (gap / ref).reindex(v.index)


def _norm_drug(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    toks = [t for t in s.split() if t and not t.isdigit()]
    return toks[0] if toks else ""


# ---------------------------------------------------------------------------
# public database: drug reporting sex ratio (openFDA FAERS)
# ---------------------------------------------------------------------------

@register_function(
    aliases=["药物性别特异性", "drug sex specificity", "openFDA sex ratio", "FAERS sex reference"],
    category="utils",
    description="Per-drug reporting counts by patient sex (and optionally age group) from FDA adverse-"
                "event reports (openFDA FAERS) -- a public reference for drugs used almost only in one "
                "sex or age group. Two count queries for the most-reported drugs, then one query per "
                "requested drug that the lists leave undecided.",
    examples=["ref = ov.tl.drug_sex_reference(['Tamsulosin', 'Levonorgestrel'])"],
    related=["tl.audit_clinical_tables"],
)
def drug_sex_reference(drugs: Optional[Iterable[str]] = None, *, timeout: float = 60.0,
                       max_single_queries: int = 60, api_key: Optional[str] = None,
                       cache: Optional[str] = None) -> pd.DataFrame:
    """Return a DataFrame indexed by normalised drug token with columns
    ``male``, ``female`` (report counts, NaN = unknown), ``male_share`` and
    ``source``. Offline / failed queries return an empty frame.

    ``drugs`` (optional) are free-text names from the records; each is reduced
    to its first word (``"Tamsulosin HCl 0.4mg"`` -> ``tamsulosin``) and matched
    to the first word of FAERS generic names.
    """
    base = "https://api.fda.gov/drug/event.json"
    key = f"&api_key={api_key}" if api_key else ""
    if cache and Path(cache).exists():
        try:
            return pd.read_csv(cache, index_col=0)
        except Exception:
            pass

    def _get(url):
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())

    rows: Dict[str, dict] = {}
    floors = {}
    try:
        for sex, code in (("male", 1), ("female", 2)):
            res = _get(f"{base}?search=patient.patientsex:{code}"
                       f"&count=patient.drug.openfda.generic_name.exact&limit=999{key}")["results"]
            floors[sex] = min(r["count"] for r in res)
            for r in res:
                tok = _norm_drug(r["term"])
                d = rows.setdefault(tok, {"male": 0.0, "female": 0.0})
                d[sex] += r["count"]
    except Exception:
        return pd.DataFrame(columns=["male", "female", "male_share", "source"])
    ref = pd.DataFrame.from_dict(rows, orient="index")
    ref["source"] = "faers_top"
    # a drug missing from one list has fewer reports than that list's floor: unknown, refine below
    for sex in ("male", "female"):
        ref.loc[ref[sex] == 0, sex] = np.nan
    want = sorted({_norm_drug(d) for d in (drugs or [])} - {""})
    todo = [d for d in want if d not in ref.index or ref.loc[d, ["male", "female"]].isna().any()]
    for tok in todo[:max_single_queries]:
        try:
            q = urllib.parse.quote(f'patient.drug.openfda.generic_name:"{tok}"')
            res = _get(f"{base}?search={q}&count=patient.patientsex{key}")["results"]
            c = {int(r["term"]): r["count"] for r in res}
            ref.loc[tok, ["male", "female", "source"]] = [c.get(1, 0.0), c.get(2, 0.0), "faers_drug"]
        except Exception:
            continue
    ref["male"] = ref["male"].astype(float)
    ref["female"] = ref["female"].astype(float)
    ref["male_share"] = ref["male"] / (ref["male"] + ref["female"])
    if cache:
        try:
            ref.to_csv(cache)
        except Exception:
            pass
    return ref


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------

@register_function(
    aliases=["临床数据质量审计", "EHR 错误检测", "病历数据录入错误", "EHR data quality audit",
             "clinical data-entry error detection", "audit EHR tables", "flag erroneous EHR rows"],
    category="utils",
    description="Audit linked clinical / EHR tables for data-entry errors and return one finding per "
                "flagged row with family, rule, incident cluster and evidence: unit-label deviations, "
                "per-item corpus extremes and sign flips, jumps against the patient's own history "
                "(power-of-ten / unit-factor slips), drug-dose outliers, disagreeing duplicate "
                "observations, cross-table conflicts on automatically discovered transcribed item pairs, "
                "end-before-start times, and demographic contradictions (sex vs. sex-specific reference "
                "bands learned from the cohort, single-sex items/drugs, FAERS drug sex ratios; isolated "
                "ages). All references are derived from the data or public databases at run time.",
    examples=["tables = ov.tl.load_clinical_tables('/workspace/data/csv')",
              "f = ov.tl.audit_clinical_tables(tables)",
              "f.drop_duplicates(['table','row_id'])[['table','row_id']]"],
    related=["tl.load_clinical_tables", "tl.infer_clinical_schema", "tl.drug_sex_reference"],
)
def audit_clinical_tables(
    tables: Mapping[str, pd.DataFrame],
    schema: Optional[dict] = None,
    *,
    rules: Optional[Sequence[str]] = None,
    external_reference: bool = True,
    drug_reference: Optional[pd.DataFrame] = None,
    min_item_rows: int = 20,
    rare_unit_share: float = 0.02,
    extreme_z: float = 8.0,
    isolation: float = 5.0,
    jump_ratio: float = 8.0,
    pair_min_n: int = 20,
    pair_min_agree: float = 0.9,
    min_sex_evidence: int = 3,
    unit_factors: Sequence[float] = (2.2046, 2.54, 10.0),
    unit_z: float = 4.0,
    person_tol: float = 4.0,
    max_duplicate_share: float = 0.05,
    max_rule_rate: Optional[float] = None,
    out: Optional[str] = None,
    out_id_column: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every rule family and return the findings table.

    Parameters
    ----------
    tables
        ``{name: DataFrame}`` as returned by :func:`load_clinical_tables`.
    schema
        Output of :func:`infer_clinical_schema` (inferred when omitted).
        Inspect ``result.attrs['schema']`` and override roles that were guessed wrongly.
    rules
        Subset of rule names to run (default: all).
    external_reference
        Query openFDA once for drug reporting sex ratios (needs network; skipped
        silently when unavailable). ``drug_reference`` passes a precomputed table.
    min_item_rows, rare_unit_share, extreme_z, isolation, jump_ratio
        Screening thresholds (see module docstring); defaults favour recall.
    pair_min_n, pair_min_agree
        A lab/chart (any two measurement tables) item pair is treated as "the same
        quantity" when at least ``pair_min_n`` co-timed values of the two items exist and
        at least ``pair_min_agree`` of them are equal.
    min_sex_evidence
        Minimum number of sex-informative rows before a patient's recorded sex is judged.
    unit_factors, unit_z
        ``unit_factor`` rule: physical unit-conversion constants (default lb/kg 2.2046, in/cm
        2.54, a decimal shift 10).  A value is flagged when it sits ``>= unit_z`` robust SDs
        (log scale) from its column / item and dividing (multiplying) it by a factor puts it
        back inside the bulk (``|z| <= 1.5``), or when it is a factor away from a sibling
        column of the same quantity in the same row (admission vs discharge weight) or from
        the same person's other stays.
    person_tol
        Person-level rules over nested ids (person > admission > stay): a numeric attribute
        that is stable within a person (height, age adjusted for elapsed time, weight) is
        flagged when it deviates from the person's other stays by more than ``person_tol``
        times the 90th percentile of within-person deviations seen in the cohort.
    max_duplicate_share
        ``exact_duplicate`` is reported for a table when at most this share of its rows are
        byte-identical copies (more is a storage convention, not an entry error).
    max_rule_rate
        Precision control. Every finding carries ``rule_rate`` (share of the table's rows the
        rule flagged) and ``severity`` (robust z of the flagged value against its reference;
        NaN for logical rules, which are kept).  When set, a rule whose ``rule_rate`` exceeds
        ``max_rule_rate`` in a table keeps only its ``max_rule_rate * len(table)`` most severe
        rows there (default ``None``: keep everything, the screening behaviour).
    out, out_id_column
        Optional path of the deliverable: the distinct ``(table, row id)`` pairs are written
        there as CSV with columns ``table`` and ``out_id_column`` (default: the schema's row-id
        column name).

    Returns
    -------
    DataFrame with columns ``table, row_id, family, rule, cluster, score, evidence, severity,
    rule_rate, rule_rank`` (one row per (finding, row); a row can carry several findings;
    ``rule_rank`` 1 = most severe finding of the rule in that table). ``attrs`` holds
    ``schema`` and ``summary`` (per-rule row and cluster counts). Deliver
    ``drop_duplicates(['table', 'row_id'])``.
    """
    out_path = out
    schema = schema or infer_clinical_schema(tables)
    if not schema.get("row_id"):
        # no row id shared by all tables: address rows by their position in each table
        tables = {n: df.assign(_row_id=df.index.astype(str)) for n, df in tables.items()}
        schema = dict(schema, row_id="_row_id")
    rid = schema["row_id"]
    subj = schema["subject"]
    run = set(rules) if rules else None
    F = _Findings()
    log = print if verbose else (lambda *a, **k: None)

    def on(rule):
        return run is None or rule in run

    labels = {}
    for dn, d in schema["dictionaries"].items():
        dd = tables[dn]
        labels.update(dict(zip(dd[d["item"]].astype(str), dd[d["label"]].astype(str))))

    # ---------------- measurement tables ----------------
    prepared = {}
    for tn, m in schema["measurements"].items():
        df = tables[tn]
        w = pd.DataFrame({"rid": df[rid].astype(str), "subj": df[subj].astype(str),
                          "item": df[m["item"]].astype(str), "v": _num(df[m["value"]])})
        w["unit"] = df[m["unit"]].astype(str).str.strip() if m.get("unit") else ""
        w["time"] = df[m["time"]].astype(str) if m.get("time") else ""
        if m.get("specimen"):
            w["spec"] = df[m["specimen"]].astype(str)
        prepared[tn] = w

        # unit label vs the item's own modal unit
        if on("unit_label") and m.get("unit"):
            u = w[(w["unit"] != "") & (w["unit"].str.lower() != "nan") & w["v"].notna()]
            cnt = u.groupby("item")["unit"].transform("size")
            modal = u.groupby("item")["unit"].agg(lambda s: s.value_counts().idxmax())
            n_same = u.groupby(["item", "unit"])["unit"].transform("size")
            share = n_same / cnt
            # a slip is a handful of rows; a unit used by dozens of rows is a second convention
            odd = u[(cnt >= min_item_rows) & (u["unit"] != u["item"].map(modal)) & (share < rare_unit_share)
                    & ((n_same <= 3) | (share < 0.002))]
            F.add(tn, odd["rid"], "impossible_value", "unit_label", tn + "|unit|" + odd["rid"],
                  1 - share.loc[odd.index],
                  "unit '" + odd["unit"] + "' vs usual '" + odd["item"].map(modal) + "' for item "
                  + odd["item"] + " " + odd["item"].map(labels).fillna(""))
            log(f"  {tn}: unit_label {len(odd)}")

        x = w[w["v"].notna()].copy()
        if x.empty:
            continue
        n_item = x.groupby("item")["v"].transform("size")
        x = x[n_item >= min_item_rows].copy()

        # corpus extremes per item (log scale for positive items), isolated only
        if on("item_extreme") and not x.empty:
            pos = x.groupby("item")["v"].transform(lambda s: (s > 0).mean() > 0.99)
            t = np.where(pos & (x["v"] > 0), np.log10(x["v"].clip(lower=1e-12)), x["v"])
            t = pd.Series(t, index=x.index)
            med, sc = _robust_center_scale(t, x["item"])
            z = (t - med) / sc.replace(0, np.nan)
            iso = _isolation_ratio(t, x["item"])
            far = z.abs() >= extreme_z
            # isolated from the rest of the item, or one of a handful of such values (several
            # independent slips of one analyte sit near each other on the log scale)
            n_far = far.groupby(x["item"]).transform("sum")
            few = n_far <= np.maximum(3, 0.002 * x.groupby("item")["v"].transform("size"))
            hit = x[far & ((iso >= isolation) | iso.isna() | few)]
            F.add(tn, hit["rid"], "impossible_value", "item_extreme", tn + "|ext|" + hit["rid"],
                  z.loc[hit.index].abs(),
                  "value " + hit["v"].astype(str) + " robust z=" + z.loc[hit.index].round(1).astype(str)
                  + " for item " + hit["item"] + " " + hit["item"].map(labels).fillna(""),
                  severity=z.loc[hit.index].abs())
            log(f"  {tn}: item_extreme {len(hit)}")

        if on("percent_bound") and not x.empty and m.get("unit"):
            pc = x[x["unit"].str.strip() == "%"]
            if len(pc):
                n_over = (pc["v"] > 100).groupby(pc["item"]).transform("sum")
                n_all = pc.groupby("item")["v"].transform("size")
                hit = pc[(pc["v"] > 100) & (n_over <= np.maximum(1, 0.005 * n_all))]
                F.add(tn, hit["rid"], "impossible_value", "percent_bound", tn + "|pct|" + hit["rid"], 1.0,
                      "percentage " + hit["v"].astype(str) + " > 100 for item " + hit["item"])
                log(f"  {tn}: percent_bound {len(hit)}")

        if on("sign_flip") and not x.empty:
            nonneg = x.groupby("item")["v"].transform(lambda s: (s >= 0).mean())
            neg = x[(x["v"] < 0) & (nonneg >= 0.999)]
            F.add(tn, neg["rid"], "impossible_value", "sign_flip", tn + "|neg|" + neg["rid"], 1.0,
                  "negative value " + neg["v"].astype(str) + " for never-negative item " + neg["item"]
                  + " " + neg["item"].map(labels).fillna(""))
            log(f"  {tn}: sign_flip {len(neg)}")

        # jump against the patient's own history
        if on("subject_jump"):
            y = w[w["v"].notna() & (w["v"] > 0)].copy()
            key = y["subj"] + "|" + y["item"]
            n = y.groupby(key)["v"].transform("size")
            y = y[n >= 4].copy()
            key = y["subj"] + "|" + y["item"]
            lv = np.log10(y["v"])
            g = lv.groupby(key)
            s1 = g.transform("sum")
            cnt = g.transform("size")
            # leave-one-out median approximated by group median of the others: use median and
            # check that no other reading of the patient sits near the suspicious one
            med = g.transform("median")
            dev = lv - med
            # nearest other reading of the same patient and item (log10 distance): a sustained
            # level has close neighbours, a slip stands alone (two unrelated slips do too)
            o = pd.DataFrame({"k": key, "lv": lv}).sort_values(["k", "lv"])
            gap = pd.concat([(o["lv"] - o.groupby("k")["lv"].shift(1)).abs(),
                             (o.groupby("k")["lv"].shift(-1) - o["lv"]).abs()], axis=1).min(axis=1)
            nn = gap.reindex(y.index)
            hit = y[(dev.abs() >= math.log10(jump_ratio)) & (nn >= math.log10(2))]
            # severity: deviation in units of the item's typical within-patient spread
            wmad = (lv - med).abs().groupby(key).transform("median").groupby(y["item"]).transform("median")
            wsev = dev.abs() / (1.4826 * wmad).clip(lower=0.01)
            # a sustained level is the patient's state; a single excursion is a slip
            F.add(tn, hit["rid"], "impossible_value", "subject_jump", tn + "|jump|" + hit["rid"],
                  dev.loc[hit.index].abs(),
                  "value " + hit["v"].astype(str) + " is x" + (10 ** dev.loc[hit.index]).round(3).astype(str)
                  + " the patient's median for item " + hit["item"] + " " + hit["item"].map(labels).fillna(""),
                  severity=wsev.loc[hit.index])
            log(f"  {tn}: subject_jump {len(hit)}")

        # disagreeing duplicates of one observation
        if on("duplicate_conflict"):
            # the observation is the specimen when the table has one (several specimens of one
            # analyte at one instant are legitimate); otherwise patient + item + instant
            keys = []
            if "spec" in x.columns:
                keys.append(["spec", "item"])
            elif m.get("time"):
                keys.append(["subj", "item", "time"])
            for k in keys:
                z_ = w[w["v"].notna()]
                if "spec" in k:
                    z_ = z_[(z_["spec"] != "") & (z_["spec"].str.lower() != "nan")]
                nun = z_.groupby(k)["v"].transform("nunique")
                # items for which disagreeing same-instant rows are routine are a source convention
                gk = z_.assign(_c=nun > 1).drop_duplicates(k)
                conv_rate = gk.groupby("item")["_c"].mean()
                conv = set(conv_rate[(conv_rate > 0.05) & (gk.groupby("item")["_c"].sum() >= 3)].index)
                d = z_[(nun > 1) & ~z_["item"].isin(conv)]
                clus = tn + "|dup|" + d[k].astype(str).agg("|".join, axis=1)
                rng = d.groupby(k)["v"].transform(lambda s: (s.max() - s.min()) / max(abs(s).max(), 1e-12))
                F.add(tn, d["rid"], "inconsistency", "duplicate_conflict", clus, rng,
                      "rows of one observation (" + "+".join(k) + ") disagree; item " + d["item"]
                      + " " + d["item"].map(labels).fillna("") + " value " + d["v"].astype(str))
                log(f"  {tn}: duplicate_conflict[{'+'.join(k)}] {len(d)}")

    # cross-table conflicts on discovered transcribed pairs
    if on("cross_table_conflict"):
        names = [n for n, m in schema["measurements"].items() if m.get("time")]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                A = prepared[a][prepared[a]["v"].notna()][["rid", "subj", "time", "item", "v"]]
                B = prepared[b][prepared[b]["v"].notna()][["rid", "subj", "time", "item", "v"]]
                mm = A.merge(B, on=["subj", "time"], suffixes=("_a", "_b"))
                if mm.empty:
                    continue
                mm["eq"] = np.isclose(mm["v_a"], mm["v_b"], rtol=1e-6, atol=1e-9)
                st = mm.groupby(["item_a", "item_b"])["eq"].agg(["size", "mean"])
                # an a-item maps to the b-item(s) it agrees with; require dominance
                good = st[(st["size"] >= pair_min_n) & (st["mean"] >= pair_min_agree)].index
                if len(good) == 0:
                    continue
                cand = mm.set_index(["item_a", "item_b"]).loc[mm.set_index(["item_a", "item_b"]).index.isin(good)].reset_index()
                # an instance is a conflict when no value of the partner item at that instant agrees
                anyeq_a = cand.groupby(["rid_a", "item_b"])["eq"].transform("any")
                anyeq_b = cand.groupby(["rid_b", "item_a"])["eq"].transform("any")
                bad = cand[~anyeq_a & ~anyeq_b]
                if bad.empty:
                    continue
                rel = (bad["v_a"] - bad["v_b"]).abs() / np.maximum(bad[["v_a", "v_b"]].abs().max(axis=1), 1e-12)
                clus = "xtab|" + bad["rid_a"] + "|" + bad["rid_b"]
                ev = (a + " item " + bad["item_a"] + "=" + bad["v_a"].astype(str) + " vs " + b + " item "
                      + bad["item_b"] + "=" + bad["v_b"].astype(str) + " at the same instant")
                F.add(a, bad["rid_a"], "inconsistency", "cross_table_conflict", clus, rel, ev)
                F.add(b, bad["rid_b"], "inconsistency", "cross_table_conflict", clus, rel, ev)
                log(f"  {a}<->{b}: {len(good)} transcribed item pairs, {len(bad)} conflicts")

    # ---------------- prescriptions ----------------
    for tn, r in schema["prescriptions"].items():
        df = tables[tn]
        if on("dose_outlier") and r.get("dose"):
            w = pd.DataFrame({"rid": df[rid].astype(str), "drug": df[r["drug"]].astype(str).str.lower().str.strip(),
                              "unit": df[r["dose_unit"]].astype(str) if r.get("dose_unit") else "",
                              "v": _num(df[r["dose"]])})
            w = w[w["v"].notna() & (w["v"] > 0)]
            key = w["drug"] + "|" + w["unit"]
            n = w.groupby(key)["v"].transform("size")
            w = w[n >= 2].copy()
            key = w["drug"] + "|" + w["unit"]
            lv = np.log10(w["v"])
            g = lv.groupby(key)
            cnt = g.transform("size")
            med = g.transform("median")
            dec = lv - med
            # (1) power-of-ten slip of a dose that IS recorded for the same drug: value / 10^k
            # (k = 2, 3, 4, either direction) equals another prescription's dose
            seen = w.groupby(key)["v"].apply(lambda s: set(np.round(s.values, 6)))
            vk = pd.DataFrame({"key": key, "v": w["v"]})
            slip = pd.Series(False, index=w.index)
            for k in (2, 3, 4, -2, -3, -4):
                cand = np.round(w["v"] / 10.0 ** k, 6)
                slip |= pd.Series([c in seen.get(kk, ()) for kk, c in zip(key, cand)], index=w.index)
            # (2) far outside the spread of every other prescription of the drug
            q1 = g.transform(lambda s: s.quantile(0.25))
            q3 = g.transform(lambda s: s.quantile(0.75))
            iqr = (q3 - q1)
            ext = ((lv > q3 + 3 * np.maximum(iqr, math.log10(2))) & (dec >= 1.0)) | \
                  ((lv < q1 - 3 * np.maximum(iqr, math.log10(2))) & (dec <= -1.0))
            n_same = w.groupby([key, w["v"]])["v"].transform("size")
            rare = (n_same == 1) | (n_same / cnt <= 0.05)
            slip &= (dec.abs() >= 1.0)
            ext &= (cnt >= 5)
            hit = w[(slip | ext) & rare]
            dsev = dec.abs() / (iqr / 1.349).clip(lower=math.log10(2) / 1.349)
            F.add(tn, hit["rid"], "impossible_value", "dose_outlier", tn + "|dose|" + hit["rid"],
                  dec.loc[hit.index].abs(),
                  "dose " + hit["v"].astype(str) + " " + hit["unit"] + " is x"
                  + (10 ** dec.loc[hit.index]).round(3).astype(str) + " the usual dose of " + hit["drug"],
                  severity=dsev.loc[hit.index])
            log(f"  {tn}: dose_outlier {len(hit)}")

    # ---------------- time order within a row ----------------
    if on("time_order"):
        for tn, prs in schema["time_pairs"].items():
            df = tables[tn]
            for a, b in prs:
                if (a, b) == ("charttime", "storetime"):
                    continue
                ta = pd.to_datetime(df[a], errors="coerce")
                tb = pd.to_datetime(df[b], errors="coerce")
                bad = df[(ta.notna() & tb.notna() & (tb < ta))]
                # conventions that run through a whole table are not errors
                if len(bad) > 0.01 * max(len(df), 1):
                    continue
                F.add(tn, bad[rid], "impossible_value", "time_order", tn + "|time|" + bad[rid].astype(str), 1.0,
                      f"{b} before {a}")
                log(f"  {tn}: time_order[{a}>{b}] {len(bad)}")

    # ---------------- orphan foreign keys ----------------
    if on("orphan_key"):
        # a key column is a foreign key wherever it repeats, and its parent is the table in
        # which it is unique (patients for the patient key, admissions for an admission key ...)
        keycols = {c for df in tables.values() for c in df.columns
                   if c != rid and re.search(r"(^|_)(id|key)$", c.lower())}
        for c in sorted(keycols):
            parents = [n for n, df in tables.items() if c in df.columns and df[c].is_unique
                       and n not in schema["dictionaries"] and len(df) > 1]
            if len(parents) != 1:
                continue
            par = set(tables[parents[0]][c].astype(str)) - {""}
            for n, df in tables.items():
                if n == parents[0] or c not in df.columns or n in schema["dictionaries"]:
                    continue
                v = df[c].astype(str)
                bad = df[(v != "") & (v.str.lower() != "nan") & ~v.isin(par)]
                if 0 < len(bad) <= 0.01 * len(df):
                    F.add(n, bad[rid], "inconsistency", "orphan_key", n + "|fk|" + bad[rid].astype(str), 1.0,
                          c + " '" + bad[c].astype(str) + "' has no row in " + parents[0])
                    log(f"  {n}: orphan_key[{c}] {len(bad)}")

    # ---------------- demographics ----------------
    demo = schema.get("demographics")
    if demo and subj:
        P = tables[demo["table"]]
        prow = dict(zip(P[subj].astype(str), P[rid].astype(str)))
        sex = {}
        if demo.get("sex"):
            sx = P[demo["sex"]].astype(str).str.strip().str.upper().str[0]
            sex = {s: v for s, v in zip(P[subj].astype(str), sx) if v in ("M", "F")}
        evid = []   # (table, rid, subj, implied_sex, rule, text)

        if sex:
            # (a) sex-specific reference bands learned from the cohort
            if on("sex_reference_band"):
                for tn, m in schema["measurements"].items():
                    if not (m.get("ref_lo") and m.get("ref_hi")):
                        continue
                    df = tables[tn]
                    b = pd.DataFrame({"rid": df[rid].astype(str), "subj": df[subj].astype(str),
                                      "item": df[m["item"]].astype(str),
                                      "band": df[m["ref_lo"]].astype(str) + "|" + df[m["ref_hi"]].astype(str)})
                    b = b[(b["band"] != "|") & b["subj"].isin(sex)]
                    b["sex"] = b["subj"].map(sex)
                    # band -> sex purity counted over patients (not rows)
                    pb = b.drop_duplicates(["item", "band", "subj"])
                    ct = pb.groupby(["item", "band", "sex"]).size().unstack(fill_value=0)
                    for c in ("M", "F"):
                        if c not in ct:
                            ct[c] = 0
                    tot = ct["M"] + ct["F"]
                    maj = np.where(ct["M"] >= ct["F"], "M", "F")
                    pur = np.maximum(ct["M"], ct["F"]) / tot
                    ct = ct.assign(maj=maj, pur=pur, tot=tot)
                    ok = ct[(ct["tot"] >= 5) & (ct["pur"] >= 0.85)]
                    # item must have informative bands of both sexes
                    both = ok.reset_index().groupby("item")["maj"].nunique()
                    items = set(both[both == 2].index)
                    ok = ok.reset_index()
                    ok = ok[ok["item"].isin(items)].set_index(["item", "band"])["maj"]
                    bb = b.set_index(["item", "band"])
                    bb = bb[bb.index.isin(ok.index)].copy()
                    bb["implied"] = ok.reindex(bb.index).values
                    bb = bb.reset_index()
                    for r_ in bb.itertuples():
                        evid.append((tn, r_.rid, r_.subj, r_.implied, "sex_reference_band",
                                     f"reference band {r_.band} of item {r_.item} {labels.get(r_.item, '')} is the {r_.implied} band"))
                    log(f"  {tn}: {len(items)} items with sex-specific reference bands")

            # (b) items / drugs observed in one sex only, across enough patients
            if on("sex_only_item"):
                srcs = [(tn, m["item"]) for tn, m in schema["measurements"].items()]
                srcs += [(tn, r["drug"]) for tn, r in schema["prescriptions"].items()]
                srcs += [(tn, e["item"]) for tn, e in schema.get("events", {}).items()
                         if e.get("item") and (tn, e["item"]) not in srcs]
                for tn, col in srcs:
                    df = tables[tn]
                    vals = df[col].astype(str)
                    if col == schema["prescriptions"].get(tn, {}).get("drug"):
                        vals = vals.map(_norm_drug)
                    b = pd.DataFrame({"rid": df[rid].astype(str), "subj": df[subj].astype(str), "it": vals})
                    b = b[b["subj"].isin(sex)]
                    b["sex"] = b["subj"].map(sex)
                    pp = b.drop_duplicates(["it", "subj"])
                    ct = pp.groupby(["it", "sex"]).size().unstack(fill_value=0)
                    for c in ("M", "F"):
                        if c not in ct:
                            ct[c] = 0
                    # one-sex items: >= 5 patients of one sex, at most one of the other
                    # minority allowance: one patient, or 5 % of the majority (the minority may be
                    # exactly the patients whose recorded sex is wrong)
                    allow_m = np.where(ct["F"] >= 8, np.maximum(2, np.floor(0.1 * ct["F"])), 1)
                    allow_f = np.where(ct["M"] >= 8, np.maximum(2, np.floor(0.1 * ct["M"])), 1)
                    cand = ct[((ct["M"] >= 5) & (ct["F"] <= allow_f)) | ((ct["F"] >= 5) & (ct["M"] <= allow_m))]
                    imp = pd.Series(np.where(cand["M"] > cand["F"], "M", "F"), index=cand.index)
                    hit = b[b["it"].isin(imp.index)].copy()
                    hit["implied"] = hit["it"].map(imp)
                    for r_ in hit.itertuples():
                        evid.append((tn, r_.rid, r_.subj, r_.implied, "sex_only_item",
                                     f"{col} '{r_.it}' is recorded only for {r_.implied} patients in this cohort"))

            # (c) public database: FAERS reporting sex ratio of each prescribed drug
            if on("sex_external") and schema["prescriptions"] and (external_reference or drug_reference is not None):
                alld = pd.concat([tables[tn][r["drug"]].astype(str) for tn, r in schema["prescriptions"].items()])
                ref = drug_reference
                if ref is None:
                    # only drugs whose recipients' recorded sex could be contradicted need a lookup
                    ref = drug_sex_reference(sorted(set(alld)), max_single_queries=0)
                if ref is not None and len(ref):
                    ms = ref["male_share"] if "male_share" in ref else ref["male"] / (ref["male"] + ref["female"])
                    tot = ref[["male", "female"]].sum(axis=1, min_count=1)
                    # a list miss means fewer reports than that list's floor: bound the share
                    for tn, r in schema["prescriptions"].items():
                        df = tables[tn]
                        tok = df[r["drug"]].astype(str).map(_norm_drug)
                        b = pd.DataFrame({"rid": df[rid].astype(str), "subj": df[subj].astype(str), "tok": tok,
                                          "drug": df[r["drug"]].astype(str)})
                        b = b[b["subj"].isin(sex) & b["tok"].isin(ref.index)]
                        share = b["tok"].map(ms)
                        mlo = b["tok"].map(ref["male"])
                        flo = b["tok"].map(ref["female"])
                        # missing count in one list -> that sex is rare for the drug
                        imp = pd.Series(None, index=b.index, dtype=object)
                        imp[(flo.notna() & mlo.isna()) | (share <= 0.02)] = "F"
                        imp[(mlo.notna() & flo.isna()) | (share >= 0.98)] = "M"
                        b["implied"] = imp
                        b = b[b["implied"].notna()]
                        # single-drug queries for the contradicted ones, to confirm the bound
                        contra = b[b["implied"] != b["subj"].map(sex)]
                        if len(contra) and drug_reference is None:
                            conf = drug_sex_reference(sorted(set(contra["drug"])), max_single_queries=30)
                            if len(conf):
                                cs = conf["male_share"]
                                keep = []
                                for r_ in contra.itertuples():
                                    s_ = cs.get(r_.tok, np.nan)
                                    if np.isnan(s_) or (r_.implied == "F" and s_ <= 0.05) or (r_.implied == "M" and s_ >= 0.95):
                                        keep.append(r_.Index)
                                b = pd.concat([b[b["implied"] == b["subj"].map(sex)], contra.loc[keep]])
                        for r_ in b.itertuples():
                            evid.append((tn, r_.rid, r_.subj, r_.implied, "sex_external",
                                         f"drug '{r_.drug}' is reported almost only in {r_.implied} patients (openFDA FAERS)"))

            if evid:
                E = pd.DataFrame(evid, columns=["table", "rid", "subj", "implied", "rule", "text"])
                E["rec"] = E["subj"].map(sex)
                E["contra"] = E["implied"] != E["rec"]
                per = E.groupby("subj").agg(n=("contra", "size"), k=("contra", "sum"),
                                            strong=("rule", lambda s: s.isin(["sex_external", "sex_only_item"]).sum()))
                # judge the recorded sex: majority of the evidence rows contradict it, or a
                # one-sex drug / item contradicts it (a single prescription can be the error)
                major = per[(per["n"] >= min_sex_evidence) & (per["k"] / per["n"] >= 0.6)].index
                ext_contra = E[E["contra"] & E["rule"].isin(["sex_external", "sex_only_item"])]
                minor = set(ext_contra["subj"]) - set(major)
                for s in major:
                    e = E[(E["subj"] == s) & E["contra"]]
                    clus = "sex|" + s
                    F.add(demo["table"], [prow[s]], "demographic_conflict", "sex_vs_records", clus,
                          per.loc[s, "k"] / per.loc[s, "n"],
                          f"recorded sex {sex[s]} contradicted by {int(per.loc[s, 'k'])}/{int(per.loc[s, 'n'])} sex-informative rows")
                    for tn, g in e.groupby("table"):
                        F.add(tn, g["rid"], "demographic_conflict", g["rule"].iloc[0], clus, 1.0, g["text"])
                for s in minor:
                    # the records agree with the recorded sex except a one-sex drug / item: that row
                    # (or the sex itself) is wrong -- flag both, one incident
                    e = ext_contra[ext_contra["subj"] == s]
                    clus = "sex|" + s
                    F.add(demo["table"], [prow[s]], "demographic_conflict", "sex_vs_records", clus, 0.5,
                          f"recorded sex {sex[s]} contradicted by {len(e)} sex-specific drug/item rows")
                    for tn, g in e.groupby("table"):
                        F.add(tn, g["rid"], "demographic_conflict", g["rule"].iloc[0], clus, 1.0, g["text"])
                log(f"  demographics: sex contradicted for {len(major)} patients by the majority of their "
                    f"records, {len(minor)} by a one-sex drug/item")

        # (d) isolated ages
        if on("age_isolated") and demo.get("age"):
            age = _num(P[demo["age"]])
            ok = age.notna()
            a = age[ok].sort_values()
            if len(a) >= 10:
                vals = a.values
                typical = np.median(np.diff(np.unique(vals))) if a.nunique() > 2 else 0.0
                k = min(6, len(vals) - 2)
                flag_idx = []
                # the extreme value is separated from the rest by a gap far larger than the cohort's
                # typical spacing and than the gaps among its nearest neighbours
                lo_gap, lo_next = vals[1] - vals[0], np.max(np.diff(vals[1:1 + k]))
                hi_gap, hi_next = vals[-1] - vals[-2], np.max(np.diff(vals[-1 - k:-1]))
                if typical > 0 and lo_gap >= isolation * typical and lo_gap >= 2 * lo_next:
                    flag_idx.append(a.index[0])
                if typical > 0 and hi_gap >= isolation * typical and hi_gap >= 2 * hi_next:
                    flag_idx.append(a.index[-1])
                for i in flag_idx:
                    F.add(demo["table"], [P.loc[i, rid]], "demographic_conflict", "age_isolated",
                          "age|" + str(P.loc[i, subj]), 1.0,
                          f"age {age[i]:g} is isolated from the rest of the cohort "
                          f"(range of the others {a.drop(i).min():g}-{a.drop(i).max():g})")
                log(f"  demographics: {len(flag_idx)} isolated ages")

    _event_rules(tables, schema, F, on, log, isolation, max_duplicate_share)
    _offset_rules(tables, schema, F, on, log)
    _person_rules(tables, schema, F, on, log, person_tol, unit_factors)
    if on("unit_factor"):
        _unit_factor_rules(tables, schema, F, log, unit_factors, unit_z, min_item_rows)
    out = _rank_findings(F.frame(), tables)
    if max_rule_rate is not None and len(out):
        cap = out["table"].map({t: max(1, int(max_rule_rate * len(df))) for t, df in tables.items()})
        keep = out["severity"].isna() | (out["rule_rate"] <= max_rule_rate) | (out["rule_rank"] <= cap)
        out = out[keep].reset_index(drop=True)
    summ = (out.groupby(["family", "rule"]).agg(rows=("row_id", "nunique"), clusters=("cluster", "nunique"),
                                                rule_rate=("rule_rate", "max"))
            if len(out) else pd.DataFrame())
    out.attrs["schema"] = schema
    out.attrs["summary"] = summ
    flags = out.drop_duplicates(["table", "row_id"])[["table", "row_id"]]
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        flags.rename(columns={"row_id": out_id_column or rid or "row_id"}).to_csv(out_path, index=False)
        from ._meta_selection import record_deliverable
        record_deliverable(out_path, "omicverse.tl.audit_clinical_tables", rows=len(flags))
    if verbose and len(out):
        log(summ.to_string())
        log(f"  total distinct flagged rows: {len(flags)}"
            + (f" -> written to {out_path} (fingerprinted: finish with ov.tl.verify_deliverable('{out_path}'), "
               f"which must say 'unchanged')" if out_path is not None else ""))
        log("  Deliver ALL of these rows. Every rule already requires a second signature (isolation, a "
            "power-of-ten / unit relation, a disagreeing record of the same event, or the patient's "
            "other records); small disagreements (duplicates 10-50 % apart, a lab vs chart value 1.3x "
            "off) are the typical data-entry errors. Post-filtering by your own thresholds loses "
            "incidents; add rows from extra checks if you like, never remove these.")
    return out


def _dt(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.replace("T", " ", regex=False).str.replace("Z", "", regex=False)
    return pd.to_datetime(x.where(x != ""), errors="coerce")


def _event_rules(tables, schema, F, on, log, isolation, max_duplicate_share=0.01):
    """Rules over event tables (subject + date): life-span, visit consistency, exact duplicates,
    demographic answers vs. recorded demographics, arithmetic relations."""
    rid, subj = schema["row_id"], schema["subject"]
    demo = schema.get("demographics") or {}
    # date-based rules only see date-stamped events; relative offsets are handled by _offset_rules
    events = {n: e for n, e in schema.get("events", {}).items() if e.get("kind", "date") == "date"}
    visits = schema.get("visits", {})
    P = tables[demo["table"]] if demo.get("table") else None

    # ---- exact duplicate rows (all columns but the row id and the table's own surrogate key)
    if on("exact_duplicate"):
        for tn, df in tables.items():
            # a surrogate key: id-named, unique in this table, referenced by no other table
            surr = {c for c in df.columns if c != rid and re.search(_ID_RE, c.lower()) and df[c].is_unique
                    and not any(c in d2.columns for m, d2 in tables.items() if m != tn)}
            body = [c for c in df.columns if c != rid and c not in surr]
            if len(body) < 3 or len(df) < 2:
                continue
            dmask = df.duplicated(body, keep="first")
            d = df[dmask]
            if 0 < len(d) <= max_duplicate_share * len(df):
                first = df[~dmask].drop_duplicates(body).set_index(body)[rid]
                orig = pd.Series(list(map(tuple, d[body].values)) if len(body) > 1 else d[body[0]].values,
                                 index=d.index).map(lambda k: first.get(k, "?"))
                F.add(tn, d[rid], "duplicate", "exact_duplicate", tn + "|xdup|" + orig.astype(str), 1.0,
                      "identical copy of row " + orig.astype(str) + " (all columns but ids)")
                log(f"  {tn}: exact_duplicate {len(d)}")

    # ---- life span: events after death / before birth; birth date contradicted by events
    dpairs = [d for d in (demo.get("death_status") or []) if d.get("kind") == "date"]
    if P is not None and (demo.get("death") or demo.get("birth") or dpairs):
        birth = _dt(P[demo["birth"]]) if demo.get("birth") else pd.Series(pd.NaT, index=P.index)
        death = _dt(P[demo["death"]]) if demo.get("death") else pd.Series(pd.NaT, index=P.index)
        for dp in dpairs:   # death as a status ("Expired") at a dated discharge
            dead = P[dp["status"]].astype(str).str.strip().str.lower().isin(dp["dead_values"])
            death = death.where(death.notna(), _dt(P[dp["time"]]).where(dead))
        if dpairs:
            demo = dict(demo, death=demo.get("death") or dpairs[0]["time"])
        birth.index = death.index = P[subj].astype(str)
        firsts, lasts = [], []
        conv_visits = set()
        # visit types that routinely happen after death (certification, autopsy ...) are conventions
        for vn, v in visits.items():
            V = tables[vn]
            it = events[vn].get("item") or events[vn].get("label")
            if not it or not demo.get("death"):
                continue
            d = _dt(V[v["start"]])
            dd = V[subj].astype(str).map(death)
            aft = dd.notna() & (d.dt.normalize() > dd.dt.normalize())
            dead = V[dd.notna()]
            rate = aft[dd.notna()].groupby(dead[it]).mean()
            npat = V[aft].groupby(it)[subj].nunique()
            conv_types = set(rate[(rate >= 0.5)].index) & set(npat[npat >= 2].index)
            conv_visits |= set(V.loc[V[it].isin(conv_types), v["key"]].astype(str))
        vkeys = {v["key"] for v in visits.values()}
        for tn, e in events.items():
            df = tables[tn]
            d = _dt(df[e["time"]])
            sid = df[subj].astype(str)
            firsts.append(d.groupby(sid).min())
            lasts.append(d.groupby(sid).max())
            dd, bd = sid.map(death), sid.map(birth)
            inconv = pd.Series(False, index=df.index)
            for k in vkeys & set(df.columns):
                inconv |= df[k].astype(str).isin(conv_visits)
            it = e.get("item")
            if on("after_death") and demo.get("death"):
                aft = dd.notna() & d.notna() & (d.dt.normalize() > dd.dt.normalize()) & ~inconv
                # per-item convention (an item recorded after death for most dead patients)
                if it and aft.any():
                    dead = dd.notna()
                    rate = aft[dead].groupby(df.loc[dead, it]).mean()
                    conv_items = set(rate[rate >= 0.5].index)
                    aft &= ~df[it].isin(conv_items)
                if aft.sum() and aft.mean() <= 0.05:
                    hit = df[aft]
                    F.add(tn, hit[rid], "impossible_value", "after_death", tn + "|death|" + sid[aft], 1.0,
                          "dated " + d[aft].astype(str) + " after the patient's death " + dd[aft].astype(str))
                    log(f"  {tn}: after_death {int(aft.sum())}")
            if on("before_birth") and demo.get("birth"):
                bef = bd.notna() & d.notna() & (d.dt.normalize() < bd.dt.normalize())
                if bef.sum() and bef.mean() <= 0.05:
                    hit = df[bef]
                    F.add(tn, hit[rid], "impossible_value", "before_birth", tn + "|birth|" + hit[rid].astype(str),
                          1.0, "dated " + d[bef].astype(str) + " before the patient's birth " + bd[bef].astype(str))
                    log(f"  {tn}: before_birth {int(bef.sum())}")
        if on("birth_vs_events") and demo.get("birth") and firsts:
            first = pd.concat(firsts, axis=1).min(axis=1)
            last = pd.concat(lasts, axis=1).max(axis=1)
            b = birth
            # the birth date itself is wrong when MOST of the patient's records precede it
            nb = {}
            for tn, e in events.items():
                df = tables[tn]
                d = _dt(df[e["time"]])
                sid = df[subj].astype(str)
                bef = (d < sid.map(b))
                g = bef.groupby(sid).agg(["sum", "size"])
                nb[tn] = g
            bad = (pd.concat([g["sum"] for g in nb.values()]).groupby(level=0).sum() if nb else None)
            many_before = set(bad[(bad >= 3)].index) if bad is not None else set()
            ages = ((last - b.reindex(last.index)).dt.days / 365.25).dropna()
            iso = set()
            if len(ages) >= 10:
                a = ages.sort_values()
                vals = a.values
                typical = np.median(np.diff(np.unique(np.round(vals, 2)))) if a.nunique() > 2 else 0.0
                k = min(6, len(vals) - 2)
                hi_gap, hi_next = vals[-1] - vals[-2], np.max(np.diff(vals[-1 - k:-1]))
                if typical > 0 and hi_gap >= isolation * typical and hi_gap >= 2 * hi_next:
                    iso.add(a.index[-1])
            for s_ in many_before | iso:
                r_ = P.loc[P[subj].astype(str) == s_, rid]
                why = ("most of the patient's records precede the recorded birth date" if s_ in many_before else
                       f"age at last record {ages.get(s_, np.nan):.0f} y is isolated from the cohort")
                F.add(demo["table"], r_, "demographic_conflict", "birth_vs_events", "birth|" + s_, 1.0, why)
            log(f"  demographics: birth date contradicted for {len(many_before | iso)} patients")

    # ---- visits: rows outside their visit's time window or of another patient
    for vn, v in visits.items():
        V = tables[vn]
        key = v["key"]
        vs = _dt(V[v["start"]])
        ve = _dt(V[v["end"]]) if v.get("end") else vs
        vs.index = ve.index = V[key].astype(str)
        vsub = pd.Series(V[subj].astype(str).values, index=V[key].astype(str))
        for tn in v["users"]:
            if tn not in events or tn == vn:
                continue
            df = tables[tn]
            k_ = df[key].astype(str)
            has = k_.isin(vsub.index)
            if on("visit_subject") and subj in df.columns:
                mis = has & (k_.map(vsub) != df[subj].astype(str))
                if 0 < mis.sum() <= 0.01 * len(df):
                    hit = df[mis]
                    F.add(tn, hit[rid], "inconsistency", "visit_subject", tn + "|vsub|" + hit[rid].astype(str), 1.0,
                          "the row's patient differs from its visit's patient")
                    log(f"  {tn}: visit_subject {int(mis.sum())}")
            if on("outside_visit"):
                d = _dt(df[events[tn]["time"]])
                lag_lo = (d - k_.map(vs)).dt.total_seconds() / 86400
                lag_hi = (d - k_.map(ve).fillna(k_.map(vs))).dt.total_seconds() / 86400
                ok = has & d.notna()
                if ok.sum() < 20:
                    continue
                # tolerance learned from the data: how far rows usually sit outside their visit
                lo_tol = max(1.0, -float(np.nanquantile(lag_lo[ok], 0.001)) * 1.5)
                hi_tol = max(1.0, float(np.nanquantile(lag_hi[ok], 0.999)) * 1.5)
                # results may be filed after a visit; nothing can be recorded before it started
                out_ = ok & (lag_lo < -lo_tol)
                if 0 < out_.sum() <= 0.01 * len(df):
                    hit = df[out_]
                    F.add(tn, hit[rid], "inconsistency", "outside_visit", tn + "|vwin|" + hit[rid].astype(str), 1.0,
                          "dated outside its visit's time window (by " + lag_lo[out_].round(1).astype(str) + " d)")
                    log(f"  {tn}: outside_visit {int(out_.sum())}")

    # ---- recorded demographic categories vs the patient's own answers recorded as items
    if on("demographic_answer") and P is not None:
        sub_cat = {c: pd.Series(P[c].astype(str).str.strip().str.lower().values, index=P[subj].astype(str))
                   for c in (demo.get("categories") or []) + ([demo["sex"]] if demo.get("sex") else [])}
        for tn, m in schema["measurements"].items():
            df = tables[tn]
            it, val = m["item"], (m.get("text") or m["value"])
            if val not in df.columns:
                val = m["value"]
            txt = df[val].astype(str).str.strip().str.lower()
            nonnum = _num(df[val]).isna() & (txt != "")
            if not nonnum.any():
                continue
            sub = df[nonnum]
            for col, cat in sub_cat.items():
                for item_id, g in sub.groupby(it):
                    sids = g[subj].astype(str)
                    if sids.nunique() < 20:
                        continue
                    ans = txt.loc[g.index]
                    # the answer's category: learned as the recorded category most patients with
                    # that answer have (a mapping from the data, no vocabulary table)
                    rec = sids.map(cat)
                    ct = pd.crosstab(ans, rec)
                    if ct.shape[0] < 2 or ct.shape[0] > 30:
                        continue
                    mapping = ct.idxmax(axis=1)
                    purity = (ct.max(axis=1).sum() / ct.values.sum())
                    base = ct.sum(axis=0).max() / ct.values.sum()
                    # the answers must explain the recorded category far beyond its majority class
                    lift = (purity - base) / max(1 - base, 1e-9)
                    if purity < 0.95 or lift < 0.8 or mapping.nunique() < 2:
                        continue
                    per = pd.DataFrame({"s": sids, "a": ans.map(mapping)}).groupby("s")["a"].agg(
                        lambda x: x.value_counts().index[0] if x.value_counts().iloc[0] >= 0.6 * len(x) else None)
                    per = per.dropna()
                    bad = per[per != per.index.map(cat)]
                    if len(bad) and len(bad) <= 0.05 * len(per):
                        for s_ in bad.index:
                            r_ = P.loc[P[subj].astype(str) == s_, rid]
                            F.add(demo["table"], r_, "demographic_conflict", "demographic_answer", f"dem|{col}|{s_}",
                                  1.0, f"recorded {col} '{cat.get(s_)}' contradicts the patient's own answers "
                                       f"to item {item_id} ('{bad[s_]}')")
                        log(f"  demographics: {col} contradicted by answers to item {item_id} for {len(bad)} patients")

    # ---- arithmetic relations discovered from the data
    if on("arithmetic"):
        _arithmetic_rules(tables, schema, F, log)


def _arithmetic_rules(tables, schema, F, log, min_rows: int = 50):
    rid, subj = schema["row_id"], schema["subject"]
    # (a) wide: numeric columns of one table with c = a * b (or a / b) on almost every row
    for tn, df in tables.items():
        numc = []
        for c in df.columns:
            if c == rid or re.search(r"(id|code|key)$", c.lower()):
                continue
            smp = df[c].astype(str).head(500)
            if smp.str.fullmatch(r"\s*-?\d+(\.\d+)?([eE][-+]?\d+)?\s*").mean() < 0.9:
                continue
            xc = _num(df[c])
            if xc.notna().mean() > 0.9 and xc.nunique() > 10:
                numc.append(c)
        if len(numc) < 3 or len(numc) > 12:
            continue
        X = {c: _num(df[c]) for c in numc}
        for c in numc:
            for a in numc:
                for b in numc:
                    if len({a, b, c}) < 3 or a > b:
                        continue
                    for op in ("mul", "div"):
                        pred = X[a] * X[b] if op == "mul" else X[a] / X[b].replace(0, np.nan)
                        ok = pred.notna() & X[c].notna() & (X[c].abs() > 0)
                        if ok.sum() < min_rows:
                            continue
                        rel = ((X[c] - pred).abs() / X[c].abs().clip(lower=1e-9))[ok]
                        if (rel < 0.01).mean() < 0.95:
                            continue
                        bad = ok & ((X[c] - pred).abs() > 0.01 * X[c].abs().clip(lower=1))
                        if 0 < bad.sum() <= 0.01 * len(df):
                            hit = df[bad]
                            F.add(tn, hit[rid], "inconsistency", "arithmetic", tn + "|arith|" + hit[rid].astype(str),
                                  1.0, f"{c} != {a} {'*' if op == 'mul' else '/'} {b}, a relation that holds on "
                                       f"{(rel < 0.01).mean():.1%} of rows")
                            log(f"  {tn}: arithmetic {c}={a}{'*' if op == 'mul' else '/'}{b}: {int(bad.sum())}")
    # (b) long: items recorded together (same patient and instant)
    for tn, m in schema["measurements"].items():
        if not m.get("time"):
            continue
        df = tables[tn]
        w = pd.DataFrame({"rid": df[rid].astype(str), "s": df[subj].astype(str), "t": df[m["time"]].astype(str),
                          "it": df[m["item"]].astype(str), "v": _num(df[m["value"]])}).dropna(subset=["v"])
        w = w.drop_duplicates(["s", "t", "it"], keep=False)
        top = w["it"].value_counts()
        top = list(top[top >= 200].index[:25])
        w = w[w["it"].isin(top)]
        units = {}
        if m.get("unit"):
            uu = pd.DataFrame({"it": df[m["item"]].astype(str), "u": df[m["unit"]].astype(str)})
            units = uu[uu["it"].isin(top)].groupby("it")["u"].agg(lambda x: x.value_counts().index[0]).to_dict()
        V = w.pivot_table(index=["s", "t"], columns="it", values="v", aggfunc="first")
        R = w.pivot_table(index=["s", "t"], columns="it", values="rid", aggfunc="first")
        cols = list(V.columns)
        # ordered pairs: a > b on (almost) every co-recorded instant
        for a in cols:
            for b in cols:
                if a == b:
                    continue
                ok = V[a].notna() & V[b].notna()
                if ok.sum() < min_rows:
                    continue
                gt = (V.loc[ok, a] > V.loc[ok, b]).mean()
                same_unit = units.get(a) is not None and units.get(a) == units.get(b)
                if gt >= 0.995 and same_unit and (V.loc[ok, a] / V.loc[ok, b].replace(0, np.nan)).median() < 10:
                    bad = ok & (V[a] <= V[b])
                    if 0 < bad.sum():
                        ids = list(R.loc[bad, a]) + list(R.loc[bad, b])
                        F.add(tn, ids, "inconsistency", "ordered_pair", [f"{tn}|ord|{x}" for x in
                              list(R.loc[bad, a]) * 2], 1.0, f"item {a} <= item {b}, while {a} > {b} at "
                              f"{gt:.1%} of shared instants")
                        log(f"  {tn}: ordered_pair {a}>{b}: {int(bad.sum())}")
        # products: log c = log k + p log a + q log b, constant across instants
        L = np.log(V.where(V > 0))
        for c in cols:
            for i_, a in enumerate(cols):
                for b in cols[i_ + 1:]:
                    if c in (a, b):
                        continue
                    ok = L[c].notna() & L[a].notna() & L[b].notna()
                    if ok.sum() < min_rows:
                        continue
                    for p_ in (1, -1, 2, -2):
                        for q_ in (1, -1, 2, -2):
                            res = (L[c] - p_ * L[a] - q_ * L[b])[ok]
                            med = res.median()
                            mad = (res - med).abs().median()
                            if mad > 0.005:
                                continue
                            dev = (L[c] - p_ * L[a] - q_ * L[b] - med).abs()
                            bad = ok & (dev > 0.05)
                            if 0 < bad.sum() <= 0.02 * ok.sum():
                                F.add(tn, R.loc[bad, c], "inconsistency", "arithmetic",
                                      [f"{tn}|prod|{x}" for x in R.loc[bad, c]], 1.0,
                                      f"item {c} != k * {a}^{p_} * {b}^{q_} (holds on the other instants)")
                                log(f"  {tn}: arithmetic {c}~{a}^{p_}*{b}^{q_}: {int(bad.sum())}")


# ---------------------------------------------------------------------------
# relative-time (offset) layouts, nested ids, unit factors
# ---------------------------------------------------------------------------

def _offset_rules(tables, schema, F, on, log):
    """Events stamped with offsets (minutes / hours from the subject's anchor, e.g. unit admission).

    ``after_death``  an event later than the subject's death offset.  Death is a status column
                     ("Expired", "Dead", "Deceased" ...) with the offset of that status
                     (``demographics.death_status``); the earliest dead status wins.
    ``outside_stay`` an event outside the subject's window [min(0, offsets), max(offsets)] spanned
                     by the offset columns of the subject-level table (admission ... discharge).
    Tolerances are learned from the data: how far events of *alive* subjects run past the end of
    their window (after_death), and the 0.1 % / 99.9 % quantiles of all events (outside_stay).
    """
    rid, subj = schema["row_id"], schema["subject"]
    demo = schema.get("demographics") or {}
    offs = {n: e for n, e in schema.get("events", {}).items() if e.get("kind") == "offset"}
    if not offs or not demo.get("table") or not subj:
        return
    D = tables[demo["table"]]
    sid = D[subj].astype(str).values
    lo = hi = None
    ow = schema.get("offset_window")
    if ow and ow.get("columns"):
        W = pd.DataFrame({c: _num(D[c]).values for c in ow["columns"]}, index=sid)
        lo = W.min(axis=1).clip(upper=0)
        hi = W.max(axis=1).clip(lower=0)
        lo = lo[~lo.index.duplicated()]
        hi = hi[~hi.index.duplicated()]
    death = np.full(len(D), np.nan)
    for dp in demo.get("death_status") or []:
        if dp.get("kind") != "offset":
            continue
        dead = D[dp["status"]].astype(str).str.strip().str.lower().isin(dp["dead_values"]).values
        t = _num(D[dp["time"]]).values
        death = np.where(dead & ~np.isnan(t), np.fmin(death, t), death)
    death = pd.Series(death, index=sid)
    death = death[~death.index.duplicated()]
    for tn, e in offs.items():
        df = tables[tn]
        s_ = df[subj].astype(str)
        t = _num(df[e["time"]])
        it = e.get("item")
        dd = s_.map(death)
        if on("after_death") and dd.notna().any():
            tol = 0.0
            if hi is not None:
                alive = dd.isna() & t.notna() & s_.isin(hi.index)
                ex = (t - s_.map(hi))[alive]
                if len(ex) >= 20:
                    tol = max(0.0, 1.5 * float(np.nanquantile(ex, 0.999)))
            aft = dd.notna() & t.notna() & (t > dd + tol)
            if it and it in df.columns and aft.any():
                # items recorded after death for most dead subjects having them are a convention
                dead = dd.notna()
                rate = aft[dead].groupby(df.loc[dead, it]).mean()
                aft &= ~df[it].isin(set(rate[rate >= 0.5].index))
            if aft.sum() and aft.mean() <= 0.05:
                hit = df[aft]
                F.add(tn, hit[rid], "impossible_value", "after_death", tn + "|death|" + s_[aft], 1.0,
                      e["time"] + "=" + t[aft].astype(str) + " after the death offset " + dd[aft].astype(str)
                      + " (" + tn + " relative times)")
                log(f"  {tn}: after_death[offset] {int(aft.sum())}")
        if on("outside_stay") and lo is not None:
            has = t.notna() & s_.isin(lo.index)
            if has.sum() < 20:
                continue
            dlo = (t - s_.map(lo))[has]
            dhi = (t - s_.map(hi))[has]
            lo_tol = max(0.0, -1.5 * float(np.nanquantile(dlo, 0.001)))
            hi_tol = max(0.0, 1.5 * float(np.nanquantile(dhi, 0.999)))
            bad = pd.Series(False, index=df.index)
            bad[has] = (dlo < -lo_tol) | (dhi > hi_tol)
            if 0 < bad.sum() <= 0.01 * len(df):
                hit = df[bad]
                F.add(tn, hit[rid], "inconsistency", "outside_stay", tn + "|swin|" + hit[rid].astype(str), 1.0,
                      e["time"] + "=" + t[bad].astype(str) + " outside the stay window ["
                      + s_[bad].map(lo).astype(str) + ", " + s_[bad].map(hi).astype(str) + "]")
                log(f"  {tn}: outside_stay {int(bad.sum())}")


def _robust_scale(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3:
        return float("nan")
    mad = 1.4826 * float((x - x.median()).abs().median())
    iqr = float(x.quantile(0.75) - x.quantile(0.25)) / 1.349
    return max(mad, iqr)


def _numeric_values(s: pd.Series) -> pd.Series:
    """Leading number of each cell ("> 89" -> 89, "72.5 kg" -> 72.5)."""
    return _num(s.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False))


def _match_factor(ratio: pd.Series, factors, tol: float = 0.03) -> pd.Series:
    """The unit factor f with |log10 ratio| ~= log10 f (within ``tol``), else NaN."""
    lr = np.log10(ratio.where(ratio > 0)).abs()
    out = pd.Series(np.nan, index=ratio.index)
    for f in factors:
        out = out.where(out.notna() | ((lr - math.log10(f)).abs() > tol), f)
    return out


def _stay_time_years(df: pd.DataFrame) -> Optional[pd.Series]:
    """Time of each stay row in years (absolute start date, else a calendar-year column)."""
    for c in df.columns:
        lc = c.lower()
        if re.search(r"(adm|in|start|begin)", lc) and re.search(r"(time|date)", lc) and not re.fullmatch(_OFFSET_RE, lc):
            d = _dt(df[c])
            if d.notna().mean() >= 0.9:
                return (d - d.min()).dt.total_seconds() / (365.25 * 86400)
    for c in df.columns:
        if re.search(r"year$", c.lower()) and _is_numeric_col(df[c]):
            return _num(df[c])
    return None


def _others_median(s: pd.Series) -> pd.Series:
    """Reference of each row within its person: the other row for two rows, the person's median
    for three or more (one wrong row cannot move it, unlike a leave-one-out median of two)."""
    if len(s) == 2:
        return pd.Series(s.values[::-1], index=s.index)
    return pd.Series(np.median(s.values), index=s.index)


def _person_rules(tables, schema, F, on, log, person_tol=4.0, unit_factors=(2.2046, 2.54, 10.0)):
    """Attributes that must agree between the stays of one person (nested ids person > admission >
    stay, ``schema['stays']``).

    ``person_constant`` a categorical column is person-level when the persons with several stays
                        almost always keep one value (>= 90 %) *far more often than chance*
                        (lift >= 0.8 over the constancy expected from the value frequencies --
                        discharge status or unit type fail this); the minority value of a person
                        is flagged (all of the person's rows on a tie, score 0.5).
    ``person_numeric``  a numeric column is person-stable when the 90th percentile of within-person
                        deviations is at most half the population robust SD (height, weight, age
                        after subtracting the elapsed time between stays); a stay deviating from
                        the median of the person's other stays by more than ``person_tol`` times
                        that percentile is flagged.  With two stays the one farther from the
                        population is flagged (both when undecidable).
    ``unit_factor``     added when the deviating value is a unit factor away from the others.
    """
    rid = schema["row_id"]
    demo = schema.get("demographics") or {}
    skip = set(schema.get("id_hierarchy") or []) | {rid, schema.get("subject")}
    for d in demo.get("death_status") or []:
        skip |= {d["status"], d["time"]}
    for tn, st in (schema.get("stays") or {}).items():
        df = tables[tn]
        per = df[st["person"]].astype(str)
        okp = ~per.str.lower().isin(_MISSING)
        multi = okp & per.map(per[okp].value_counts()).fillna(0).ge(2)
        if per[multi].nunique() < 10:
            continue
        ty = _stay_time_years(df)
        for c in df.columns:
            lc = c.lower()
            if c in skip or re.search(_ID_RE, lc) or re.search(_TIMEISH_RE, lc):
                continue
            raw = df[c].astype(str).str.strip()
            present = ~raw.str.lower().isin(_MISSING)
            if present.mean() < 0.5:
                continue
            numeric = _is_numeric_col(df[c])
            nun = raw[present].str.lower().nunique()
            if (not numeric or nun <= 5) and on("person_constant"):
                v = raw.str.lower().where(present)
                m = multi & present
                g = v[m].groupby(per[m])
                k, n_i = g.nunique(), g.size()
                k = k[n_i >= 2]
                if len(k) < 10 or not (2 <= nun <= 30):
                    continue
                p = v[present].value_counts(normalize=True)
                const = float((k == 1).mean())
                exp = float(np.mean([(p ** n).sum() for n in n_i[k.index]]))
                lift = (const - exp) / max(1 - exp, 1e-9)
                bad_p = list(k[k > 1].index)
                if const < 0.9 or lift < 0.8 or not bad_p:
                    continue
                nflag = 0
                for pp in bad_p:
                    rows = v[m & (per == pp)]
                    cnt = rows.value_counts()
                    tie = len(cnt) > 1 and cnt.iloc[0] == cnt.iloc[1]
                    sel = rows if tie else rows[rows != cnt.index[0]]
                    F.add(tn, df.loc[sel.index, rid], "demographic_conflict", "person_constant",
                          f"person|{c}|{pp}", 0.5 if tie else 1.0,
                          f"{c} '" + df.loc[sel.index, c].astype(str) + f"' differs between stays of person {pp} "
                          f"(values {dict(cnt)}); {c} is constant within {const:.0%} of multi-stay persons")
                    nflag += len(sel)
                log(f"  {tn}: person_constant[{c}] {len(bad_p)} persons, {nflag} rows")
            elif numeric and nun > 5 and on("person_numeric"):
                x = _numeric_values(df[c])
                is_age = bool(re.search(r"(^|_)age($|_)|^age", lc)) or c == demo.get("age")
                resid = x - ty if (is_age and ty is not None) else x
                m = multi & resid.notna()
                if m.sum() < 20:
                    continue
                xm, pm = resid[m], per[m]
                om = xm.groupby(pm).transform(_others_median)
                dev = xm - om
                s_w = float(np.nanquantile(dev.abs(), 0.9))
                s_p = _robust_scale(resid)
                if not (s_p > 0) or s_w > 0.5 * s_p:
                    continue
                tol = max(person_tol * s_w, 0.1 * s_p)
                cand = dev.abs() > tol
                if not cand.any() or cand.mean() > 0.05:
                    continue
                zp = (resid - resid.median()).abs() / s_p
                nper = pm.map(pm.value_counts())
                flag = cand & (nper >= 3)
                score = pd.Series(1.0, index=xm.index)
                two = cand & (nper == 2)
                for pp in pm[two].unique():
                    ix = list(xm.index[pm == pp])
                    za, zb = zp[ix[0]], zp[ix[1]]
                    if abs(za - zb) >= 1:
                        flag[ix[0] if za > zb else ix[1]] = True
                    else:
                        flag[ix] = True
                        score[ix] = 0.5
                hit = xm.index[flag]
                if not len(hit):
                    continue
                sev = 1.645 * dev[hit].abs() / max(s_w, 1e-9)
                raw_o = x[m].groupby(pm).transform(_others_median)
                fac = _match_factor(x[hit] / raw_o[hit], unit_factors)
                what = "age minus elapsed years" if resid is not x else c
                F.add(tn, df.loc[hit, rid], "demographic_conflict", "person_numeric", "person|" + c + "|" + pm[hit],
                      score[hit], f"{c}=" + x[hit].astype(str) + f" vs the person's other stays ({what} differs by "
                      + dev[hit].round(2).astype(str) + f", cohort within-person p90 {s_w:.3g})", severity=sev)
                uf = fac.notna() & (not is_age)
                if uf.any() and on("unit_factor"):
                    h2 = fac.index[uf]
                    F.add(tn, df.loc[h2, rid], "impossible_value", "unit_factor", "person|" + c + "|" + pm[h2],
                          1.0, f"{c}=" + x[h2].astype(str) + " is x" + fac[h2].round(4).astype(str)
                          + " (or 1/x) the person's other stays: unit slip", severity=sev[h2])
                log(f"  {tn}: person_numeric[{c}] {len(hit)} rows ({int(uf.sum())} unit factors)")


def _unit_factor_rules(tables, schema, F, log, factors=(2.2046, 2.54, 10.0), unit_z=4.0, min_rows=20):
    """Unit slips (lb recorded as kg, inches as cm, a decimal shift) in wide numeric columns.

    (a) population: the value is ``>= unit_z`` robust SDs (log scale) from its column and dividing
        (multiplying) it by a factor brings it back inside the bulk (|z| <= 1.5); only when few
        values of the column are that far (a second convention is not a slip) or it is isolated.
    (b) sibling columns: two columns recording the same quantity (they agree within ~25 % on 90 %
        of rows, e.g. admission / discharge weight) and differ by a unit factor in one row.
    Long measurement / prescription tables are covered per item by item_extreme / subject_jump.
    """
    rid = schema["row_id"]
    skip_t = set(schema.get("dictionaries", {})) | set(schema.get("measurements", {})) | \
        set(schema.get("prescriptions", {}))
    idcols = set(schema.get("id_hierarchy") or []) | {rid, schema.get("subject")}
    for tn, df in tables.items():
        if tn in skip_t or len(df) < max(min_rows, 30):
            continue
        cols = {}
        for c in df.columns:
            lc = c.lower()
            if c in idcols or re.search(_ID_RE, lc) or re.search(_TIMEISH_RE, lc) or not _is_numeric_col(df[c]):
                continue
            x = _numeric_values(df[c])
            xp = x[x.notna()]
            if len(xp) < max(min_rows, 30) or xp.nunique() <= 10 or (xp > 0).mean() < 0.99:
                continue
            cols[c] = np.log10(x.where(x > 0))
        n_hit = 0
        for c, lv in cols.items():
            med, sc = lv.median(), _robust_scale(lv)
            if not (sc > 0):
                continue
            z = (lv - med) / sc
            far = z.abs() >= unit_z
            few = far.sum() <= max(3, 0.002 * lv.notna().sum())
            iso = _isolation_ratio(lv.dropna(), pd.Series("g", index=lv.dropna().index)).reindex(lv.index)
            back = pd.Series(np.nan, index=lv.index)
            for f in factors:
                lf = math.log10(f)
                ok_hi = (z >= unit_z) & (((lv - lf - med) / sc).abs() <= 1.5)
                ok_lo = (z <= -unit_z) & (((lv + lf - med) / sc).abs() <= 1.5)
                back = back.where(back.notna() | ~(ok_hi | ok_lo), f)
            hit = back.notna() & (few | (iso >= 5))
            if hit.any():
                h = hit[hit].index
                F.add(tn, df.loc[h, rid], "impossible_value", "unit_factor", tn + "|unit|" + c + "|" + df.loc[h, rid].astype(str),
                      z[h].abs(), f"{c}=" + df.loc[h, c].astype(str) + " robust z=" + z[h].round(1).astype(str)
                      + "; divided/multiplied by " + back[h].astype(str) + " it is typical for the column",
                      severity=z[h].abs())
                n_hit += len(h)
        names = list(cols)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                r = cols[a] - cols[b]
                ok = r.notna()
                if ok.sum() < max(min_rows, 30):
                    continue
                medr = r[ok].median()
                spread = float((r[ok] - medr).abs().quantile(0.9))
                if abs(medr) >= 0.05 or spread >= 0.1:
                    continue
                d = (r - medr).abs()
                fac = _match_factor(10 ** d, factors, tol=max(0.03, spread))
                hit = ok & fac.notna() & (d >= 4 * max(spread, 1e-3))
                if hit.any():
                    h = hit[hit].index
                    sev = 1.645 * d[h] / max(spread, 1e-3)
                    F.add(tn, df.loc[h, rid], "impossible_value", "unit_factor", tn + "|unitpair|" + df.loc[h, rid].astype(str),
                          1.0, f"{a}=" + df.loc[h, a].astype(str) + f" vs {b}=" + df.loc[h, b].astype(str)
                          + " differ by a unit factor " + fac[h].astype(str) + f" (same quantity on most rows)",
                          severity=sev)
                    n_hit += len(h)
        if n_hit:
            log(f"  {tn}: unit_factor {n_hit}")
