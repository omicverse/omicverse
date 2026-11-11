r"""Write and audit a cluster → label deliverable from a scorer result.

:func:`write_cluster_labels` writes the table straight from the ``label``
column of :func:`omicverse.space.score_cluster_signatures` (one result, or
one per dataset) in the column layout the caller asks for.  Manual changes
go only through ``overrides={cluster: (label, evidence)}`` and only for
clusters the scorer marked ``review = 'open ...'``; overrides of fixed
clusters are refused.  Every cluster's scorer label, written label, source
and evidence go to a sidecar audit file next to the deliverable.

:func:`audit_cluster_labels` reads an existing deliverable back and reports
every row that differs from the scorer without an accepted, recorded
override (and every cluster missing or duplicated).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from .._registry import register_function

__all__ = ["write_cluster_labels", "audit_cluster_labels"]

_ResultT = Union[pd.DataFrame, Mapping[str, pd.DataFrame]]


def _audit_path(path) -> Path:
    p = Path(path)
    return p.with_name(p.stem + ".audit.csv")


def _as_results(res: _ResultT) -> Dict[Optional[str], pd.DataFrame]:
    if isinstance(res, pd.DataFrame):
        return {None: res}
    return {str(k): v for k, v in res.items()}


def _key(group, cluster) -> Tuple[Optional[str], str]:
    return (None if group is None else str(group), str(cluster))


def _norm_overrides(overrides, groups) -> Dict[Tuple[Optional[str], str], Tuple[str, str]]:
    out = {}
    for k, v in (overrides or {}).items():
        if isinstance(k, tuple) and len(k) == 2:
            g, c = k
        elif len(groups) == 1:
            g, c = next(iter(groups)), k
        else:
            raise ValueError("with several results, override keys must be (group, cluster)")
        if isinstance(v, str) or not isinstance(v, Sequence) or len(v) != 2:
            raise ValueError("an override is (label, evidence); evidence must state the "
                             "marker values that justify it")
        lab, ev = v
        if not str(ev).strip():
            raise ValueError(f"override {k}: evidence is empty")
        out[_key(g, c)] = (str(lab), str(ev))
    return out


REVIEW_BELOW = 0.0       # the default review threshold (dataset mean of lineage evidence)


def _default_review(rr) -> str:
    """Open / fixed under the DEFAULT policy, recomputed from the result's
    evidence columns (``lineage_z``, ``alt_lineage_z``) — independent of the
    arguments the scorer was called with.  Without those columns: fixed."""
    if "lineage_z" not in rr.index:
        return "fixed (no evidence columns)"
    try:
        lz = float(rr["lineage_z"])
    except (TypeError, ValueError):
        lz = float("nan")
    alt = float(rr["alt_lineage_z"]) if "alt_lineage_z" in rr.index and rr["alt_lineage_z"] == rr["alt_lineage_z"] \
        and rr["alt_lineage_z"] is not None else float("nan")
    if lz != lz or lz < REVIEW_BELOW:
        return f"open: winner evidence z={lz:.1f} below {REVIEW_BELOW}"
    if alt == alt and alt > lz:
        return f"open: another candidate's evidence z={alt:.1f} > winner z={lz:.1f}"
    return "fixed"


def _sort_clusters(idx):
    vals = [str(i) for i in idx]
    try:
        return sorted(vals, key=lambda x: float(x))
    except ValueError:
        return sorted(vals)


def _out_cluster(c: str):
    try:
        f = float(c)
        return int(f) if f == int(f) else c
    except ValueError:
        return c


@register_function(
    aliases=["写出簇标签", "簇注释交付文件", "write cluster labels", "cluster annotation deliverable"],
    category="space",
    description="Write a cluster -> label deliverable directly from score_cluster_signatures "
                "results in the caller's column layout; manual overrides only for clusters the "
                "scorer marked open, with evidence, recorded in a sidecar audit file",
    requires={}, produces={}, auto_fix="none",
    examples=[
        "tab, diff = ov.space.write_cluster_labels(res, 'results/cluster_annotations.csv', "
        "columns=('cluster', 'cell_class'))",
        "tab, diff = ov.space.write_cluster_labels({'d1': r1, 'd2': r2}, 'results/annotations.csv', "
        "columns=('dataset', 'cluster', 'Annotation'))",
    ],
)
def write_cluster_labels(
    res: _ResultT,
    path,
    columns: Sequence[str] = ("cluster", "label"),
    *,
    label_map: Optional[Mapping[str, str]] = None,
    overrides: Optional[Mapping] = None,
    extra: Optional[Mapping[str, object]] = None,
    order: Optional[Sequence[str]] = None,
    allowed: Optional[Sequence[str]] = None,
    label_column: str = "label",
):
    r"""Write the cluster → label deliverable from scorer results.

    Arguments:
        res: a :func:`score_cluster_signatures` result (index = cluster), or
            ``{group: result}`` (e.g. one per dataset) written into one file.
        path: deliverable path (CSV).  The audit goes to
            ``<stem>.audit.csv`` next to it.
        columns: output column names, copied verbatim from the task:
            ``(cluster_col, label_col)``, or ``(group_col, cluster_col,
            label_col)`` when ``res`` is a mapping.
        label_map: optional ``{scorer label: deliverable label}`` (spelling
            only; unmapped labels are written as they are).
        overrides: ``{cluster: (label, evidence)}`` (``{(group, cluster):
            ...}`` for a mapping).  Accepted only for clusters that are open
            under the DEFAULT review policy, recomputed here from the
            result's ``lineage_z`` / ``alt_lineage_z`` (winner evidence below
            the dataset mean, unmeasured, or a stronger other candidate) —
            whatever ``review_below`` the scorer was called with; refused
            (with a printed reason) otherwise, and for results without those
            columns.  ``evidence`` must state the marker values relied on.
        extra: constant columns to add (``{name: value}``).
        order: group order for a mapping (default: as given).
        allowed: optional allowed label set; a written label outside it
            raises.
        label_column: the scorer column holding the labels.

    Returns:
        ``(table, diff)`` — the written table and the rows whose written
        label differs from the scorer's (accepted overrides).
    """
    results = _as_results(res)
    groups = list(order) if order is not None else list(results)
    cols = list(columns)
    grouped = len(results) > 1 or next(iter(results)) is not None
    if grouped and len(cols) != 3:
        raise ValueError("with several results, columns = (group_col, cluster_col, label_col)")
    if not grouped and len(cols) != 2:
        raise ValueError("columns = (cluster_col, label_col)")
    ov = _norm_overrides(overrides, groups)
    lm = dict(label_map or {})
    rows, audit, used = [], [], set()
    for g in groups:
        r = results[g]
        if label_column not in r.columns:
            raise KeyError(f"result for {g!r} has no {label_column!r} column")
        if "lineage_forced" in r.columns and r["lineage_forced"].astype(bool).any():
            bad = [str(c) for c in r.index[r["lineage_forced"].astype(bool)]]
            raise ValueError(f"result {g or ''} carries lineage changes of fixed clusters "
                             f"{bad} (score_cluster_signatures(..., review_below=None)); rerun "
                             "the scorer with the default review policy and pass manual changes "
                             "as overrides for open clusters")
        for c in _sort_clusters(r.index):
            rr = r.loc[c] if c in r.index else r.loc[[i for i in r.index if str(i) == c][0]]
            scorer = lm.get(str(rr[label_column]), str(rr[label_column]))
            review = _default_review(rr)
            k = _key(g, c)
            lab, src, ev = scorer, "scorer", ""
            if k in ov:
                used.add(k)
                want, ev = ov[k]
                want = lm.get(want, want)
                if want == scorer:
                    src = "scorer (override equal)"
                elif review.startswith("open"):
                    lab, src = want, "override"
                else:
                    src = "override refused (fixed)"
                    print(f"[write_cluster_labels] override of {g + ' ' if g else ''}cluster {c} "
                          f"-> {want!r} refused: the scorer marked it {review!r}; keeping {scorer!r}")
            if allowed is not None and lab not in set(allowed):
                raise ValueError(f"label {lab!r} for cluster {c} is not in the allowed label set")
            row = ([g] if grouped else []) + [_out_cluster(c), lab]
            rows.append(row)
            audit.append(dict(group=g, cluster=c, scorer_label=scorer, written_label=lab,
                              source=src, review=review, evidence=ev))
    for k in set(ov) - used:
        print(f"[write_cluster_labels] override for unknown cluster {k} ignored")
    tab = pd.DataFrame(rows, columns=cols)
    for name, val in (extra or {}).items():
        tab[name] = val
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(path, index=False)
    aud = pd.DataFrame(audit)
    aud.to_csv(_audit_path(path), index=False)
    diff = aud[aud["scorer_label"] != aud["written_label"]].reset_index(drop=True)
    print(f"[write_cluster_labels] wrote {len(tab)} rows to {path}; "
          f"{len(diff)} accepted override(s); audit {_audit_path(path)}")
    return tab, diff


@register_function(
    aliases=["核对簇标签", "簇注释审计", "audit cluster labels", "check cluster annotation deliverable"],
    category="space",
    description="Read a cluster -> label deliverable back and report every row that differs from "
                "the scorer result without an accepted override recorded by write_cluster_labels",
    requires={}, produces={}, auto_fix="none",
    examples=["bad = ov.space.audit_cluster_labels('results/cluster_annotations.csv', res, "
              "columns=('cluster', 'cell_class'))"],
)
def audit_cluster_labels(
    path,
    res: _ResultT,
    columns: Sequence[str] = ("cluster", "label"),
    *,
    label_map: Optional[Mapping[str, str]] = None,
    label_column: str = "label",
) -> pd.DataFrame:
    r"""Report deliverable rows not backed by the scorer or a recorded override.

    Arguments:
        path: the deliverable written (by :func:`write_cluster_labels` or
            otherwise).
        res / columns / label_map / label_column: as for
            :func:`write_cluster_labels`.

    Returns:
        ``DataFrame`` with one row per problem (``group``, ``cluster``,
        ``scorer_label``, ``file_label``, ``problem``): a label that differs
        from the scorer without an accepted override in the audit file, a
        missing cluster, a duplicated cluster, a cluster not in the scorer
        result, or a scorer result carrying lineage changes of fixed
        clusters (``review_below=None``).  Empty means the deliverable is exactly the scorer's
        output plus recorded overrides.
    """
    results = _as_results(res)
    cols = list(columns)
    grouped = len(results) > 1 or next(iter(results)) is not None
    lm = dict(label_map or {})
    df = pd.read_csv(path, dtype=str)
    low = {c.strip().lower(): c for c in df.columns}
    try:
        use = [low[c.strip().lower()] for c in cols]
    except KeyError as e:
        raise KeyError(f"deliverable lacks column {e}; header is {list(df.columns)}")
    gcol, ccol, lcol = (use if grouped else [None] + use)
    accepted = {}
    ap = _audit_path(path)
    if ap.exists():
        a = pd.read_csv(ap, dtype=str).fillna("")
        for r in a.itertuples(index=False):
            if r.source == "override":
                accepted[_key(r.group or None, r.cluster)] = r.written_label
    probs = []
    seen = {}
    for r in df.itertuples(index=False):
        rec = dict(zip(df.columns, r))
        g = str(rec[gcol]).strip() if gcol else None
        c = str(_out_cluster(str(rec[ccol]).strip()))
        k = _key(g, c)
        seen.setdefault(k, []).append(str(rec[lcol]).strip())
    for g, r in results.items():
        if "lineage_forced" in r.columns:
            for c in r.index[r["lineage_forced"].astype(bool)]:
                probs.append(dict(group=g, cluster=str(c), scorer_label=None, file_label=None,
                                  problem="scorer result carries a forced lineage change "
                                          "(review_below=None)"))
        for c in _sort_clusters(r.index):
            rr = r.loc[c] if c in r.index else r.loc[[i for i in r.index if str(i) == c][0]]
            scorer = lm.get(str(rr[label_column]), str(rr[label_column]))
            k = _key(g, str(_out_cluster(c)))
            got = seen.pop(k, None)
            if got is None:
                probs.append(dict(group=g, cluster=c, scorer_label=scorer, file_label=None,
                                  problem="missing"))
                continue
            if len(set(got)) > 1 or len(got) > 1:
                probs.append(dict(group=g, cluster=c, scorer_label=scorer, file_label=";".join(got),
                                  problem="duplicated"))
            lab = got[0]
            if lab != scorer and accepted.get(_key(g, c)) != lab:
                probs.append(dict(group=g, cluster=c, scorer_label=scorer, file_label=lab,
                                  problem="differs from scorer without a recorded override"))
    for (g, c), got in seen.items():
        probs.append(dict(group=g, cluster=c, scorer_label=None, file_label=";".join(got),
                          problem="not in scorer result"))
    out = pd.DataFrame(probs, columns=["group", "cluster", "scorer_label", "file_label", "problem"])
    print(f"[audit_cluster_labels] {len(out)} unrecorded difference(s) in {path}")
    if len(out):
        print(out.to_string(index=False))
    return out
