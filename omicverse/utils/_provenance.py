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
           "provenance_dir", "record_output", "declare_departure", "check_deliverable"]

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
                       output: Optional[Callable] = None) -> Callable:
    """Decorator: record (and warn about) overrides of the named parameters.

    ``names``: parameters whose defaults are validated.  ``note``: one line
    saying how they were validated (shown in the warning).  The wrapped
    function gains an optional keyword ``override_reason`` (popped before the
    call).  Only parameters passed explicitly with a value different from the
    default are recorded.  ``output``: optional ``f(result, arguments) ->
    (table, keys, values)`` (``arguments``: all bound arguments with defaults
    applied); the table is written to the ledger by :func:`record_output` so
    that :func:`check_deliverable` can later compare a deliverable with it.
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
                        record_output(f"{func.__module__}.{func.__qualname__}", table, keys,
                                      values, overrides=records)
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
    """Ledger directory (``$OV_PROVENANCE_DIR`` or ``~/.cache/omicverse/provenance``)."""
    import os
    d = os.environ.get("OV_PROVENANCE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "omicverse", "provenance")
    os.makedirs(d, exist_ok=True)
    return d


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


def _read_ledger(name: str) -> List[Dict[str, Any]]:
    import os
    p = os.path.join(provenance_dir(), name)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def record_output(function: str, table, keys: Sequence[str], values: Sequence[str], *,
                  overrides: Optional[Sequence[Dict[str, Any]]] = None) -> Optional[str]:
    """Write a validated function's output (``keys`` + ``values`` columns of
    ``table``) to the ledger; returns the file path."""
    import os
    import uuid
    import pandas as pd
    keys, values = list(keys), [v for v in values if v in table.columns]
    if not keys or not values or any(k not in table.columns for k in keys):
        return None
    stamp = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
    path = os.path.join(provenance_dir(), f"output_{stamp}.csv")
    pd.DataFrame(table[keys + values]).astype(str).to_csv(path, index=False)
    _append_ledger("outputs.jsonl", [{
        "function": function, "file": path, "keys": keys, "values": values,
        "n_rows": int(len(table)), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "overrides": [r.get("parameter") for r in (overrides or [])]}])
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
    outs = _read_ledger("outputs.jsonl")
    deps = _read_ledger("departures.jsonl")
    latest: Dict[Any, Dict[str, Any]] = {}
    for o in outs:
        if all(k in d.columns for k in o["keys"]):
            latest[(o["function"], tuple(o["keys"]))] = o
    rows = []
    for (fn, keys), o in latest.items():
        try:
            lib = pd.read_csv(o["file"], dtype=str, keep_default_na=False)
        except OSError:
            continue
        m = d.merge(lib, on=list(keys), how="inner", suffixes=("", "__lib"))
        for v in o["values"]:
            if v not in d.columns or len(m) == 0:
                continue
            agree = float((m[v].astype(str) == m[v + "__lib"].astype(str)).mean())
            why = [x["reason"] for x in deps if fn.endswith(x["function"])
                   and (not x.get("columns") or v in x["columns"])]
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
