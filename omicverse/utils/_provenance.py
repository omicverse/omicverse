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
"""
from __future__ import annotations

import functools
import inspect
import json
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = ["DefaultOverrideWarning", "validated_defaults", "provenance_log",
           "provenance_report", "write_provenance", "clear_provenance"]

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


def validated_defaults(*names: str, note: str = "") -> Callable:
    """Decorator: record (and warn about) overrides of the named parameters.

    ``names``: parameters whose defaults are validated.  ``note``: one line
    saying how they were validated (shown in the warning).  The wrapped
    function gains an optional keyword ``override_reason`` (popped before the
    call).  Only parameters passed explicitly with a value different from the
    default are recorded.
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
            out = func(*args, **kwargs)
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
