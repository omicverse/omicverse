r"""Shared input handling for the data-first plots in ``ov.pl``.

The statistical plots added alongside the omics-specific ones (survival, ROC,
confusion, forest, ...) deliberately do **not** require an ``AnnData``. They
accept, interchangeably:

* a ``pandas.DataFrame`` plus column names,
* bare arrays / Series / lists,
* an ``AnnData``-like object, in which case ``.obs`` is used as the frame,
* a plain ``dict`` of columns.

:func:`resolve_columns` normalises all four into one tidy frame, so each
plotting function only has to deal with arrays.
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["as_frame", "default_palette", "group_levels", "kde_curve",
           "resolve_columns"]


def as_frame(data: Any) -> Optional[pd.DataFrame]:
    """Return a tabular view of ``data``, or ``None`` if it is not tabular.

    Recognises ``DataFrame``, ``dict``, and anything exposing an ``.obs``
    frame (``AnnData``, ``MuData``, and the out-of-core equivalents).
    """
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data
    obs = getattr(data, "obs", None)
    if isinstance(obs, pd.DataFrame):
        return obs
    if isinstance(data, Mapping):
        return pd.DataFrame({k: np.asarray(v).ravel() for k, v in data.items()})
    return None


def group_levels(group: Any, explicit: Optional[Any] = None) -> list:
    """Deterministic ordering of the levels of a grouping variable.

    The first level is the reference for a hazard ratio, so its identity must
    not depend on which row happens to come first in the file. Order is taken
    from, in priority order: an explicit ``explicit`` list, the categories of a
    pandas ``Categorical``, or a plain sort of the observed values.
    """
    series = pd.Series(group)
    present = list(pd.unique(series.dropna()))
    if explicit is not None:
        missing = [lv for lv in explicit if lv not in present]
        if missing:
            raise ValueError(
                f"`groups={list(explicit)}` names levels that are not present: "
                f"{missing}. Observed levels: {present}."
            )
        return list(explicit)
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [c for c in series.cat.categories if c in present]
    try:
        return sorted(present)
    except TypeError:  # mixed types — fall back to order of appearance
        return present


def _lookup(frame: pd.DataFrame, name: str, arg: str):
    if name in frame.columns:
        # keep the Series (and therefore its dtype, e.g. Categorical) — the
        # category order decides which group is the reference downstream
        return frame[name].reset_index(drop=True)
    hint = get_close_matches(name, [str(c) for c in frame.columns], n=3, cutoff=0.6)
    suffix = f" Did you mean: {', '.join(hint)}?" if hint else ""
    raise KeyError(
        f"`{arg}={name!r}` is not a column of the input table.{suffix} "
        f"Available columns: {', '.join(map(str, frame.columns[:20]))}"
        + (" ..." if frame.shape[1] > 20 else "")
    )


def resolve_columns(data: Any = None,
                    *,
                    dropna: bool = True,
                    require: Tuple[str, ...] = (),
                    **specs: Any) -> Tuple[pd.DataFrame, Dict[str, str]]:
    r"""Turn ``data`` + column specs into one aligned frame.

    Each keyword is either a column name in ``data``, an array-like of values,
    or ``None`` (meaning "not supplied").

    Arguments
    ---------
    data
        Table to read column names from — ``DataFrame``, ``dict``, ``AnnData``
        (uses ``.obs``), or ``None`` when every spec is already an array.
    dropna
        Drop rows where any resolved column is missing. Rows removed this way
        are reported by the caller, not silently ignored.
    require
        Names that must resolve to something; a clear error is raised if not.
    **specs
        ``key=column_name_or_array`` pairs.

    Returns
    -------
    ``(frame, names)`` — the aligned frame keyed by the spec names, and a map
    from spec name to a human-readable label (the original column name where
    one was given), suitable for axis labels.
    """
    frame = as_frame(data)
    # With an AnnData the name may be a gene rather than an obs column, so
    # resolve through PlotData, which checks metadata first and features next.
    getter = None
    if (data is not None and not isinstance(data, (pd.DataFrame, Mapping))
            and hasattr(data, "var_names")):
        from ._plotdata import as_plotdata

        getter = as_plotdata(data).values

    columns: Dict[str, np.ndarray] = {}
    names: Dict[str, str] = {}

    for key, spec in specs.items():
        if spec is None:
            continue
        if isinstance(spec, str):
            if frame is None:
                raise TypeError(
                    f"`{key}={spec!r}` names a column, but no table was passed "
                    f"as the first argument. Either pass a DataFrame/AnnData, "
                    f"or give `{key}` as an array of values."
                )
            if getter is not None and spec not in frame.columns:
                columns[key] = getter(spec)
            else:
                columns[key] = _lookup(frame, spec, key)
            names[key] = spec
        elif isinstance(spec, pd.Series):
            columns[key] = spec.reset_index(drop=True)
            names[key] = spec.name or key
        else:
            values = np.asarray(spec)
            columns[key] = values.ravel() if values.ndim > 1 and 1 in values.shape else values
            names[key] = key

    for key in require:
        if key not in columns:
            raise TypeError(f"`{key}` is required.")

    if not columns:
        raise TypeError("No data columns were supplied.")

    lengths = {k: len(v) for k, v in columns.items()}
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{k}={n}" for k, n in lengths.items())
        raise ValueError(f"Columns have different lengths: {detail}.")

    out = pd.DataFrame(columns)
    if dropna:
        before = len(out)
        out = out.dropna()
        if len(out) < before:
            out.attrs["n_dropped"] = before - len(out)
    out = out.reset_index(drop=True)
    return out, names


def default_palette(n: int, palette: Any = None) -> list:
    """``n`` distinct colours: the ov palette, a named colormap, or a list."""
    import matplotlib.pyplot as plt

    if palette is None:
        from ._palette import sc_color

        base = list(sc_color)
    elif isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        base = [cmap(i / max(n - 1, 1)) for i in range(n)]
    else:
        base = list(palette)
    if not base:
        raise ValueError("`palette` is empty.")
    if len(base) < n:
        base = (base * (n // len(base) + 1))[:n]
    return base[:n]


def kde_curve(values, *, cut: float = 2.0, gridsize: int = 200,
              bw_method: Any = None, clip=None):
    """Gaussian KDE of ``values`` on a grid extended by ``cut`` *bandwidths*.

    ``cut`` means what it means in seaborn and R: how far past the extreme
    observations the density is drawn, measured in kernel bandwidths. An
    earlier version of this code measured it as a fraction of the data range
    instead, which put a long thin tail on every violin — visually the whole
    difference between these plots and the ones in ``ov.pl.violin``.

    Returns ``(grid, density)``; a degenerate sample (fewer than two distinct
    values) yields a flat zero density rather than raising.
    """
    import numpy as _np

    values = _np.asarray(values, dtype=float)
    values = values[_np.isfinite(values)]
    if values.size < 2 or _np.allclose(values, values[0]):
        centre = float(values[0]) if values.size else 0.0
        grid = _np.linspace(centre - 1.0, centre + 1.0, gridsize)
        return grid, _np.zeros_like(grid)

    from scipy.stats import gaussian_kde

    kernel = gaussian_kde(values, bw_method=bw_method)
    bandwidth = float(kernel.factor * _np.std(values, ddof=1))
    low = values.min() - cut * bandwidth
    high = values.max() + cut * bandwidth
    if clip is not None:
        low = max(low, clip[0]) if clip[0] is not None else low
        high = min(high, clip[1]) if clip[1] is not None else high
    grid = _np.linspace(low, high, gridsize)
    return grid, kernel(grid)
