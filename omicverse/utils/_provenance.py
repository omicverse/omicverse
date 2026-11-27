r"""Provenance of validated defaults: record when a caller overrides them.

Several omicverse methods expose parameters whose defaults were chosen on
independent benchmarks (simulated generators with known truth, public data
sets with labels) and documented as such -- e.g. an estimator's shrinkage
level, a fusion gain, a decision rule.  A caller may have a good reason to
change one; a caller tuning it on the data at hand (a simulator built from
the same data, a development sweep scored on the same samples) usually does
not, and such changes were the most frequent cause of failures in end-to-end
agent runs.

:func:`validated_defaults` marks those parameters on a function.  When a call
passes a value different from the default **without** ``override_reason=``,
the call still runs, but

* a :class:`DefaultOverrideWarning` is emitted naming the parameter, the
  default and the value;
* a record ``{function, parameter, default, value, reason, time}`` is appended
  to the session log (:func:`provenance_log`) and attached to the result
  (``AnnData.uns['ov_provenance']``, ``DataFrame.attrs['ov_provenance']``, a
  ``dict`` result's ``'ov_provenance'`` key) so that whoever reads the result
  -- a report, a deliverable checker -- can see it.

With ``override_reason="..."`` the record is kept (with the reason) and no
warning is raised.  :func:`provenance_report` summarises the session log;
:func:`write_provenance` writes it as JSON next to a deliverable.

This is the "decision record" idea of computational provenance (W3C PROV-DM;
Davison 2012 "Automated capture of experiment context") applied narrowly to
default choices: the mechanism does not block anything, it makes an
unexplained departure from a validated default visible.

**Deliverable drift.**  Changing an argument is only one way to leave a
validated method; the other is to call it, then replace its output with a
hand-written re-implementation (own per-group models, own thresholds) tuned on
the same data.  A function declared with ``validated_defaults(..., output=)``
therefore also writes its output (key columns + value columns) to a ledger on
disk (``$OV_PROVENANCE_DIR``, default ``~/.cache/omicverse/provenance``), so
the record survives across processes.  :func:`check_deliverable` compares a
final deliverable (file or DataFrame) with every recorded validated output
whose key columns it carries and reports, per value column, the share of rows
that still agree.  A departure with no reason declared through
:func:`declare_departure` is reported as ``UNEXPLAINED``: deliver the
validated output, or state the independent evidence (data not used to tune
the replacement) that justifies the change.
"""
from __future__ import annotations

import functools
import inspect
import json
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = ["DefaultOverrideWarning", "validated_defaults", "provenance_log",
           "provenance_report", "write_provenance", "clear_provenance",
           "provenance_dir", "record_output", "declare_departure", "check_deliverable",
           "compare_to_default", "audit_override"]

_LOG: List[Dict[str, Any]] = []


class DefaultOverrideWarning(UserWarning):
    """A validated default was changed without ``override_reason``."""


def _same(a, b) -> bool:
    try:
        if a is b:
            return True
        r = a == b
        if isinstance(r, bool):
            return r
        import numpy as np
        return bool(np.all(r))
    except Exception:
        return False


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except Exception:
        return repr(v)[:200]


def _attach(result, records):
    if not records:
        return result
    try:
        from anndata import AnnData
        if isinstance(result, AnnData):
            result.uns.setdefault("ov_provenance", [])
            result.uns["ov_provenance"] = list(result.uns["ov_provenance"]) + records
            return result
    except Exception:
        pass
    try:
        import pandas as pd
        if isinstance(result, (pd.DataFrame, pd.Series)):
            result.attrs["ov_provenance"] = list(result.attrs.get("ov_provenance", [])) + records
            return result
    except Exception:
        pass
    if isinstance(result, dict):
        result["ov_provenance"] = list(result.get("ov_provenance", [])) + records
    return result


def validated_defaults(*names: str, note: str = "",
                       output: Optional[Callable] = None,
                       protocol: Optional[Callable] = None) -> Callable:
    """Decorator: record (and warn about) overrides of the named parameters.

    ``names``: parameters whose defaults are validated.  ``note``: one line
    saying how they were validated (shown in the warning).  The wrapped
    function gains an optional keyword ``override_reason`` (popped before the
    call).  Only parameters passed explicitly with a value different from the
    default are recorded.  ``output``: optional ``f(result, arguments) ->
    (table, keys, values)`` (``arguments``: all bound arguments with defaults
    applied); the table is written to the ledger by :func:`record_output` so
    that :func:`check_deliverable` can later compare a deliverable with it.
    ``protocol``: optional ``f(result, arguments) -> dict`` describing the
    robustness protocol the function validated itself with (``name``,
    ``groups`` -- the split unit per labelled sample, ``scores``); stored in
    the record so an audit (:func:`compare_to_default`) re-uses the same
    splits.  Record format: ``docs/evolution/override-audit-contract.md``
    (schema ``verse.provenance.v1``).
    """
    def deco(func):
        sig = inspect.signature(func)
        defaults = {n: sig.parameters[n].default for n in names if n in sig.parameters}
        missing = [n for n in names if n not in sig.parameters]
        if missing:
            raise ValueError(f"validated_defaults: {func.__name__} has no parameter(s) {missing}")

        @functools.wraps(func)
        def wrapper(*args, override_reason: Optional[str] = None, **kwargs):
            try:
                bound = sig.bind_partial(*args, **kwargs)
            except TypeError:
                return func(*args, **kwargs)
            records = []
            for n, dflt in defaults.items():
                if n in bound.arguments and not _same(bound.arguments[n], dflt):
                    rec = {"function": f"{func.__module__}.{func.__qualname__}", "parameter": n,
                           "default": _jsonable(dflt), "value": _jsonable(bound.arguments[n]),
                           "reason": override_reason, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    records.append(rec)
            if records and not override_reason:
                desc = ", ".join(f"{r['parameter']}={r['value']!r} (default {r['default']!r})" for r in records)
                warnings.warn(
                    f"{func.__name__}: validated default(s) overridden without override_reason: {desc}. "
                    f"{note + ' ' if note else ''}Defaults tuned on the data at hand (own simulator, "
                    f"development sweep) usually generalise worse; keep them, or pass "
                    f"override_reason='<independent evidence>'. Recorded in ov.utils.provenance_log().",
                    DefaultOverrideWarning, stacklevel=2)
            _LOG.extend(records)
            _append_ledger("overrides.jsonl", records)
            out = func(*args, **kwargs)
            if output is not None:
                try:
                    full = sig.bind(*args, **kwargs)
                    full.apply_defaults()
                    spec = output(out, dict(full.arguments))
                    if spec is not None:
                        table, keys, values = spec
                        proto = protocol(out, dict(full.arguments)) if protocol else None
                        record_output(f"{func.__module__}.{func.__qualname__}", table, keys,
                                      values, overrides=records, protocol=proto,
                                      inputs=dict(full.arguments))
                except Exception as e:   # provenance must never break the analysis
                    warnings.warn(f"provenance: output not recorded ({type(e).__name__}: {e})")
            return _attach(out, records)

        wrapper.__validated_defaults__ = dict(defaults)
        return wrapper
    return deco


def provenance_log() -> List[Dict[str, Any]]:
    """All override records of this session (a copy)."""
    return [dict(r) for r in _LOG]


def clear_provenance() -> None:
    _LOG.clear()


def provenance_report() -> str:
    """Human-readable summary: one line per override, unexplained ones first."""
    if not _LOG:
        return "ov provenance: all validated defaults kept."
    rows = sorted(_LOG, key=lambda r: (r["reason"] is not None, r["function"]))
    lines = [f"ov provenance: {sum(r['reason'] is None for r in rows)} unexplained / {len(rows)} overrides"]
    for r in rows:
        why = r["reason"] if r["reason"] else "NO REASON GIVEN"
        lines.append(f"  {r['function']}: {r['parameter']}={r['value']!r} (default {r['default']!r}) -- {why}")
    return "\n".join(lines)


def write_provenance(path: str) -> str:
    """Write the session log as JSON (e.g. next to a deliverable); returns the path."""
    with open(path, "w") as fh:
        json.dump({"overrides": provenance_log()}, fh, indent=1)
    return path


# ── persistent ledger: validated outputs and declared departures ─────────────

def provenance_dir() -> str:
    """Run-local ledger directory: ``$OMICOS_PROVENANCE_DIR``, else
    ``$OV_PROVENANCE_DIR``, else ``<working directory>/.omicos/provenance``."""
    import os
    d = os.environ.get("OMICOS_PROVENANCE_DIR") or os.environ.get("OV_PROVENANCE_DIR") or \
        os.path.join(os.getcwd(), ".omicos", "provenance")
    os.makedirs(d, exist_ok=True)
    return d


def _ledger_dirs(near: Optional[str] = None) -> List[str]:
    """Ledger directories to read: the configured one, plus
    ``<ancestor>/.omicos/provenance`` of ``near`` (a deliverable path), so a
    record written by a script run elsewhere in the workspace is found."""
    import os
    dirs = [provenance_dir()]
    if near:
        p = os.path.dirname(os.path.abspath(str(near)))
        while True:
            c = os.path.join(p, ".omicos", "provenance")
            if os.path.isdir(c) and c not in dirs:
                dirs.append(c)
            q = os.path.dirname(p)
            if q == p:
                break
            p = q
    return dirs


def _append_ledger(name: str, records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        return
    import os
    try:
        with open(os.path.join(provenance_dir(), name), "a") as fh:
            for r in records:
                fh.write(json.dumps({k: _jsonable(v) for k, v in r.items()}) + "\n")
    except OSError:
        pass


def _read_ledger(name: str, near: Optional[str] = None) -> List[Dict[str, Any]]:
    import os
    out = []
    for d in _ledger_dirs(near):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        r["_ledger"] = d
                        out.append(r)
                    except ValueError:
                        pass
    return out


def _hash_obj(v) -> Optional[str]:
    """Content hash of an argument / output (DataFrames by values+index)."""
    import hashlib
    try:
        import pandas as pd
        import numpy as np
        if isinstance(v, (pd.DataFrame, pd.Series)):
            h = pd.util.hash_pandas_object(v, index=True).to_numpy()
            return hashlib.sha256(h.tobytes()).hexdigest()[:16]
        if isinstance(v, np.ndarray):
            return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]
        if isinstance(v, dict) and v and all(isinstance(x, (pd.DataFrame, pd.Series)) for x in v.values()):
            return hashlib.sha256("".join(f"{k}:{_hash_obj(x)}" for k, x in sorted(v.items())).encode()).hexdigest()[:16]
        return hashlib.sha256(json.dumps(_jsonable(v), sort_keys=True, default=str).encode()).hexdigest()[:16]
    except Exception:
        return None


# ── omicos.provenance/1 (runtime contract, omicos-core docs/override-audit.md) ─

def _omicos_write(payload: Dict[str, Any]) -> Optional[str]:
    """Write a signed ``omicos.provenance/1`` envelope to ``<store>/records``.
    No-op unless the runtime set ``$OMICOS_PROVENANCE_KEY``."""
    import hashlib
    import hmac
    import os
    import uuid
    key = os.environ.get("OMICOS_PROVENANCE_KEY")
    if not key:
        return None
    payload.setdefault("id", uuid.uuid4().hex)
    text = json.dumps(payload, ensure_ascii=False, default=str)
    mac = hmac.new(key.encode(), text.encode(), hashlib.sha256).hexdigest()
    d = os.path.join(provenance_dir(), "records")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{payload['id']}.json"), "w") as fh:
        json.dump({"schema": "omicos.provenance/1", "payload": text, "mac": mac}, fh)
    return payload["id"]


def _snapshot(path: str) -> str:
    import hashlib
    import os
    data = open(path, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    d = os.path.join(provenance_dir(), "objects")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, sha), "wb") as fh:
        fh.write(data)
    return sha


def _scores_dict(scores) -> Dict[str, Any]:
    """Protocol scores as a flat dict (means of numeric per-row scores)."""
    import numpy as np
    if isinstance(scores, dict):
        return scores
    out: Dict[str, Any] = {}
    if isinstance(scores, list) and scores and isinstance(scores[0], dict):
        for k in scores[0]:
            vals = []
            for r in scores:
                try:
                    v = r.get(k)
                    v = float(v) if not isinstance(v, dict) else np.nan
                    if np.isfinite(v):
                        vals.append(v)
                except (TypeError, ValueError):
                    pass
            if vals:
                out[f"mean_{k}"] = float(np.mean(vals))
    return out


def record_output(function: str, table, keys: Sequence[str], values: Sequence[str], *,
                  overrides: Optional[Sequence[Dict[str, Any]]] = None,
                  protocol: Optional[Dict[str, Any]] = None,
                  inputs: Optional[Dict[str, Any]] = None,
                  partition_values: Optional[Sequence[str]] = None) -> Optional[str]:
    """Write a validated function's output (``keys`` + ``values`` columns of
    ``table``) to the ledger as a ``verse.provenance.v1`` record; returns the
    output file path.  ``partition_values``: value columns that are arbitrary
    cluster labels (only the grouping matters); :func:`check_deliverable`
    compares those by the adjusted Rand index instead of equal strings."""
    import os
    import uuid
    import pandas as pd
    keys, values = list(keys), [v for v in values if v in table.columns]
    if not keys or not values or any(k not in table.columns for k in keys):
        return None
    stamp = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
    fname = f"output_{stamp}.csv"
    path = os.path.join(provenance_dir(), fname)
    out = pd.DataFrame(table[keys + values]).astype(str)
    out.to_csv(path, index=False)
    lib = function.split(".")[0]
    try:
        import importlib
        version = getattr(importlib.import_module(lib), "__version__", None)
    except Exception:
        version = None
    proto = dict(protocol or {})
    if "groups" in proto and proto["groups"] is not None:
        proto["groups"] = [str(x) for x in proto["groups"]]
    rec = {"schema": "verse.provenance.v1", "record_id": stamp, "function": function,
           "library": lib, "version": version, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "inputs": {k: _hash_obj(v) for k, v in (inputs or {}).items()},
           "output_file": fname, "output_hash": _hash_obj(out), "keys": keys, "values": values,
           "n_rows": int(len(out)), "protocol": proto or None,
           "partition": [v for v in (partition_values or []) if v in values] or None,
           "overrides": [r.get("parameter") for r in (overrides or [])]}
    _append_ledger("outputs.jsonl", [rec])
    try:
        sha = _snapshot(path)
        payload = {
            "kind": "library_default", "id": stamp,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "library": {"name": lib, "version": str(version) if version else None},
            "function": function,
            "inputs": [{"name": k, "sha256": v} for k, v in rec["inputs"].items()],
            "output": {"path": path, "sha256": sha, "format": "table",
                       "table": {"key": keys, "columns": values, "delimiter": ","}},
            "compare": {"max_disagreement": 0.0},
        }
        if proto:
            payload["robustness"] = {"protocol": proto.get("name"),
                                     "split": {"group_key": proto.get("group_key"),
                                               "groups_sha256": _hash_obj(proto.get("groups"))},
                                     "scores": _scores_dict(proto.get("scores"))}
        payload["audit"] = {"function": "ov.utils.audit_override",
                            "hint": f"ov.utils.audit_override('{stamp}', '<deliverable>', y=..., "
                                    "fit_predict_default=..., fit_predict_override=...)"}
        _omicos_write(payload)
    except Exception as e:   # provenance must never break the analysis
        warnings.warn(f"provenance: omicos record not written ({type(e).__name__}: {e})")
    return path


def declare_departure(function: str, reason: str, *, columns: Optional[Sequence[str]] = None) -> None:
    """State why a deliverable departs from a validated function's output.

    ``function``: (suffix of) the function name, e.g. ``"recover_sample_annotations"``.
    ``reason`` should name independent evidence -- data or ground truth that
    was NOT used to tune the replacement.  Recorded in the ledger and shown by
    :func:`check_deliverable`."""
    if not reason or not str(reason).strip():
        raise ValueError("declare_departure needs a non-empty reason")
    _append_ledger("departures.jsonl", [{"function": function, "reason": str(reason),
                                         "columns": list(columns) if columns else None,
                                         "time": time.strftime("%Y-%m-%dT%H:%M:%S")}])


def check_deliverable(deliverable, *, sep: Optional[str] = None, tolerance: float = 0.0):
    """Compare a deliverable with the validated outputs recorded in the ledger.

    ``deliverable``: a DataFrame or a CSV / TSV path.  For each recorded
    output whose key columns the deliverable carries (the latest record per
    function and key set), rows are matched on the keys and each shared value
    column is scored by agreement (share of equal values).  Returns a
    DataFrame (``function, column, n_rows, agreement, status, reason``) whose
    ``attrs['summary']`` is a printable verdict.  ``status``: ``match``
    (agreement >= 1 - ``tolerance``), ``departs`` (a reason was declared) or
    ``UNEXPLAINED``."""
    import pandas as pd
    if isinstance(deliverable, pd.DataFrame):
        d = deliverable.astype(str)
        name = "deliverable"
    else:
        name = str(deliverable)
        if sep is None:
            sep = "\t" if name.endswith((".tsv", ".tab", ".txt")) else ","
        d = pd.read_csv(name, sep=sep, dtype=str, keep_default_na=False)
    near = None if isinstance(deliverable, pd.DataFrame) else name
    outs = _read_ledger("outputs.jsonl", near)
    deps = _read_ledger("departures.jsonl", near)
    audits = [a for a in _read_ledger("audits.jsonl", near) if a.get("verdict") == "override_supported"]
    latest: Dict[Any, Dict[str, Any]] = {}
    for o in outs:
        if all(k in d.columns for k in o["keys"]):
            latest[(o["function"], tuple(o["keys"]))] = o
    rows = []
    for (fn, keys), o in latest.items():
        try:
            import os
            f = o.get("output_file") or o.get("file")
            lib = pd.read_csv(f if os.path.isabs(f) else os.path.join(o["_ledger"], f),
                              dtype=str, keep_default_na=False)
        except OSError:
            continue
        m = d.merge(lib, on=list(keys), how="inner", suffixes=("", "__lib"))
        for v in o["values"]:
            if v not in d.columns or len(m) == 0:
                continue
            if v in (o.get("partition") or []):
                agree = _partition_agreement(m[v].astype(str), m[v + "__lib"].astype(str))
            else:
                agree = float((m[v].astype(str) == m[v + "__lib"].astype(str)).mean())
            why = [x["reason"] for x in deps if fn.endswith(x["function"])
                   and (not x.get("columns") or v in x["columns"])]
            why += [f"audit {x['audit_id']} passed" for x in audits
                    if fn.endswith(str(x.get("function"))) and (not x.get("columns") or v in x["columns"])]
            status = "match" if agree >= 1.0 - tolerance else ("departs" if why else "UNEXPLAINED")
            rows.append({"function": fn, "column": v, "n_rows": int(len(m)),
                         "agreement": round(agree, 4), "status": status,
                         "reason": why[-1] if why else None,
                         "default_overrides": ",".join(o.get("overrides") or []) or None})
    res = pd.DataFrame(rows, columns=["function", "column", "n_rows", "agreement", "status",
                                      "reason", "default_overrides"])
    if res.empty:
        summary = f"check_deliverable({name}): no recorded validated output shares its key columns."
    else:
        bad = res[res.status == "UNEXPLAINED"]
        lines = [f"check_deliverable({name}): {len(res)} column(s) traced to validated outputs, "
                 f"{int((res.status == 'match').sum())} match, {len(bad)} UNEXPLAINED departure(s)."]
        for r in res.itertuples():
            lines.append(f"  {r.function.split('.')[-1]}:{r.column} agreement {r.agreement:.1%} "
                         f"over {r.n_rows} rows -> {r.status}"
                         + (f" ({r.reason})" if r.reason else ""))
        if len(bad):
            lines.append("  Deliver the validated output for these columns, or call "
                         "ov.utils.declare_departure(function, reason) naming independent "
                         "evidence (data not used to tune the replacement).")
        summary = "\n".join(lines)
    res.attrs["summary"] = summary
    return res


def _partition_agreement(a, b) -> float:
    """Adjusted Rand index of two labelings of the same rows (1 = same grouping,
    whatever the cluster names); identical singletons-only labelings give 1."""
    from sklearn.metrics import adjusted_rand_score
    a, b = list(a), list(b)
    if len(set(a)) == len(a) and len(set(b)) == len(b):
        return 1.0
    return float(adjusted_rand_score(a, b))


# ── should an override replace the validated result? ─────────────────────────

def _cramers_v(a, b) -> float:
    import numpy as np
    import pandas as pd
    t = pd.crosstab(pd.Series(list(a), dtype=str), pd.Series(list(b), dtype=str)).to_numpy(float)
    if t.size == 0 or min(t.shape) < 2:
        return 0.0
    n = t.sum()
    e = t.sum(1, keepdims=True) @ t.sum(0, keepdims=True) / n
    chi2 = float(((t - e) ** 2 / np.where(e > 0, e, 1)).sum())
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))


def _sign(rec: Dict[str, Any]) -> Dict[str, Any]:
    """sha256 digest of the canonical record; plus HMAC-SHA256 when the runtime
    provides ``$OMICOS_AUDIT_KEY`` (the key is the runtime's, so a record
    cannot be forged by editing the ledger)."""
    import hashlib
    import hmac
    import os
    body = json.dumps({k: _jsonable(v) for k, v in rec.items()}, sort_keys=True, default=str)
    rec["digest"] = hashlib.sha256(body.encode()).hexdigest()
    key = os.environ.get("OMICOS_AUDIT_KEY")
    rec["hmac"] = hmac.new(key.encode(), body.encode(), hashlib.sha256).hexdigest() if key else None
    return rec


def compare_to_default(y, fit_predict_default: Callable, fit_predict_override: Callable, *,
                       groups=None, default_record: Optional[Dict[str, Any]] = None,
                       confounder=None, n_candidates: int = 1,
                       metric: str = "balanced_accuracy", min_effect: float = 0.0,
                       max_group_drop: float = 0.05, n_boot: int = 2000, alpha: float = 0.05,
                       function: Optional[str] = None, columns: Optional[Sequence[str]] = None,
                       seed: int = 0) -> Dict[str, Any]:
    """Audit an override against the validated default on the default's own protocol.

    An agent's own check of a replacement is usually a random hold-out of the
    same reference.  That hold-out shares every confound of the reference (a
    technical batch coupled with the label), so it rewards predictors that
    read the batch; and with many candidates on a small hold-out the best one
    wins by chance (winner's curse; Cawley & Talbot, JMLR 2010).  Here default
    and override are refitted on the **same leave-one-group-out splits**
    (``groups``, or ``default_record['protocol']['groups']``: batch, cohort,
    site -- the unit the result must generalise across; Roberts et al.,
    Ecography 2017), and the override passes only if

    1. **effect**: the bootstrap lower bound of its paired improvement, at
       level ``alpha / n_candidates`` (Bonferroni over every alternative
       tried), exceeds ``min_effect``;
    2. **no collapse**: its worst held-out group scores no more than
       ``max_group_drop`` below the default's worst group;
    3. **no confound** (with ``confounder``, a per-sample technical
       covariate): Cramér's V between its predictions and the confounder is
       at most the V between the true labels and the confounder + 0.1.

    ``fit_predict_*(train_idx, test_idx) -> predictions for test_idx``.
    Returns the audit record (``verdict`` ``override_supported`` /
    ``keep_default``, ``reasons``, scores, CI, per-group table), also written
    to ``audits.jsonl`` in the run-local ledger with a digest (and an HMAC
    when the runtime sets ``$OMICOS_AUDIT_KEY``).  Contract:
    ``docs/evolution/override-audit-contract.md``.
    """
    import uuid
    import numpy as np
    import pandas as pd
    from sklearn import metrics as _m
    y = np.asarray(list(y)).astype(str)
    if groups is None and default_record:
        groups = ((default_record.get("protocol") or {}).get("groups"))
    if groups is None:
        raise ValueError("groups (or default_record with protocol.groups) are required")
    g = np.asarray(list(groups)).astype(str)
    if len(y) != len(g):
        raise ValueError("y and groups must have the same length")
    score = {"balanced_accuracy": _m.balanced_accuracy_score,
             "accuracy": _m.accuracy_score}[metric]
    ug = np.unique(g)
    if len(ug) < 2:
        raise ValueError("need >= 2 groups for leave-one-group-out evaluation")
    p_def = np.empty(len(y), dtype=object)
    p_ovr = np.empty(len(y), dtype=object)
    per = []
    for grp in ug:
        te, tr = np.flatnonzero(g == grp), np.flatnonzero(g != grp)
        a = np.asarray(fit_predict_default(tr, te)).astype(str)
        b = np.asarray(fit_predict_override(tr, te)).astype(str)
        p_def[te], p_ovr[te] = a, b
        per.append({"group": grp, "n": int(len(te)),
                    "default": float(_m.accuracy_score(y[te], a)),
                    "override": float(_m.accuracy_score(y[te], b))})
    p_def, p_ovr = p_def.astype(str), p_ovr.astype(str)
    s_def, s_ovr = float(score(y, p_def)), float(score(y, p_ovr))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, len(y), len(y))
        diffs[i] = score(y[k], p_ovr[k]) - score(y[k], p_def[k])
    lvl = alpha / max(1, int(n_candidates))
    lo, hi = float(np.quantile(diffs, lvl)), float(np.quantile(diffs, 1 - lvl))
    per_df = pd.DataFrame(per)
    checks, reasons = {}, []
    checks["effect"] = lo > min_effect
    if not checks["effect"]:
        reasons.append(f"effect: {metric} {s_ovr:.3f} vs default {s_def:.3f} under leave-one-group-out; "
                       f"{100 * (1 - 2 * lvl):.1f}% CI of the difference [{lo:+.3f}, {hi:+.3f}] "
                       f"does not exceed {min_effect} (n_candidates={n_candidates})")
    worst_d, worst_o = float(per_df["default"].min()), float(per_df["override"].min())
    checks["no_collapse"] = worst_o >= worst_d - max_group_drop
    if not checks["no_collapse"]:
        reasons.append(f"collapse: worst held-out group accuracy {worst_o:.3f} vs default {worst_d:.3f}")
    v_pred = v_true = None
    if confounder is not None:
        c = np.asarray(list(confounder)).astype(str)
        v_true, v_pred = _cramers_v(y, c), _cramers_v(p_ovr, c)
        checks["no_confound"] = v_pred <= v_true + 0.1
        if not checks["no_confound"]:
            reasons.append(f"confound: predictions follow the covariate (Cramér's V {v_pred:.2f}) "
                           f"more than the true labels do ({v_true:.2f})")
    ok = all(checks.values())
    if ok:
        reasons.append(f"{metric} {s_ovr:.3f} vs default {s_def:.3f} under leave-one-group-out, "
                       f"CI [{lo:+.3f}, {hi:+.3f}]")
    rec = {"schema": "verse.audit.v1", "audit_id": uuid.uuid4().hex[:12],
           "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "function": function,
           "columns": list(columns) if columns else None,
           "default_record_id": (default_record or {}).get("record_id"),
           "protocol": "leave-one-group-out", "splits_hash": _hash_obj(g), "n": int(len(y)),
           "n_groups": int(len(ug)), "metric": metric, "default_score": s_def,
           "override_score": s_ovr, "diff_ci": [lo, hi], "alpha": alpha,
           "n_candidates": int(n_candidates), "min_effect": min_effect, "checks": checks,
           "cramers_v_pred": v_pred, "cramers_v_true": v_true,
           "verdict": "override_supported" if ok else "keep_default", "reasons": reasons}
    _append_ledger("audits.jsonl", [_sign(dict(rec))])
    rec["per_group"] = per_df
    return rec


def _find_record(record_id: str) -> Optional[Dict[str, Any]]:
    for r in _read_ledger("outputs.jsonl"):
        if r.get("record_id") == record_id:
            return r
    return None


def audit_override(record_id: str, candidate_path: str, *, y, fit_predict_default: Callable,
                   fit_predict_override: Callable, groups=None, confounder=None,
                   n_candidates: int = 1, **kwargs) -> Dict[str, Any]:
    """Trusted audit routine for the omicos override gate (``omicos.provenance/1``).

    Runs :func:`compare_to_default` on the default's own protocol (the
    record's leave-one-group-out ``groups`` unless given) and writes a signed
    ``override_audit`` record bound to the candidate file's current sha256 and
    to ``record_id``.  ``verdict`` is ``pass`` only when the override clears
    the effect (Bonferroni over ``n_candidates``), collapse and confound
    checks.  ``y`` / ``groups`` / ``confounder`` are the labelled reference
    rows the record's protocol used; ``fit_predict_*(train_idx, test_idx)``
    refit each method on the training rows.  Returns the audit dict.
    """
    import hashlib
    rec = _find_record(record_id)
    if rec is None:
        raise ValueError(f"no library_default record {record_id!r} in {provenance_dir()}")
    res = compare_to_default(y, fit_predict_default, fit_predict_override, groups=groups,
                             default_record=rec, confounder=confounder,
                             n_candidates=n_candidates, function=rec["function"], **kwargs)
    proto = rec.get("protocol") or {}
    sha = hashlib.sha256(open(candidate_path, "rb").read()).hexdigest()
    payload = {"kind": "override_audit", "default_id": record_id,
               "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "candidate": {"path": candidate_path, "sha256": sha},
               "protocol": proto.get("name") or "leave_one_group_out",
               "split": {"group_key": proto.get("group_key"),
                         "groups_sha256": _hash_obj(proto.get("groups"))},
               "scores": {"default": res["default_score"], "candidate": res["override_score"],
                          "diff_ci": res["diff_ci"]},
               "checks": res["checks"], "n_candidates": int(n_candidates),
               "verdict": "pass" if res["verdict"] == "override_supported" else "fail",
               "reason": "; ".join(res["reasons"]),
               "library": {"name": rec.get("library"), "version": rec.get("version")},
               "function": "ov.utils.audit_override"}
    res["omicos_audit_id"] = _omicos_write(payload)
    res["candidate_sha256"] = sha
    return res
