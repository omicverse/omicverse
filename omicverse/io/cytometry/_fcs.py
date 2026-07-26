r"""FCS (Flow Cytometry Standard) reader for OmicVerse I/O.

WHY THIS LIVES IN ``ov.io``
---------------------------
Reading an FCS file is I/O, not an assay. The only FCS reader omicverse had was
``ov.single.ev.read_fcs``, buried inside the single-extracellular-vesicle
module, and it was shaped for that assay: it stamped ``uns['ev']``, named every
row ``event000123``, and — the part that actually loses information — it took
whatever DataFrame the backend handed back and threw the FCS metadata away.
That means it dropped the two things a cytometry file exists to carry:

* the ``$PnN`` / ``$PnS`` distinction — the DETECTOR name (``FITC-A``) versus
  the MARKER on it (``CD3``). Confusing the two is the classic FCS mistake:
  the same antibody sits on a different detector between panels, and the same
  detector carries a different antibody between experiments.
* ``$SPILLOVER`` — the acquisition-time spillover matrix. Without it,
  compensation cannot be applied later, and uncompensated fluorescence data is
  not wrong-looking, just wrong.

INTEROPERABILITY
----------------
The ``var`` schema here is deliberately identical to `pytometry
<https://github.com/scverse/pytometry>`_ (scverse, Apache-2.0): same index rule
(marker when present, else channel) and the same ``n / channel / marker / PnB /
PnE / PnG / PnR`` columns. An AnnData from this function is therefore a drop-in
input to ``pytometry.pp.compensate`` and ``pytometry.tl.normalize_logicle``
without a conversion step. The test suite pins this, though those tests skip
when pytometry is absent — so it is checked, not merely asserted, wherever
pytometry is installed.

The parsing itself uses `flowio <https://github.com/whitews/FlowIO>`_ (BSD-3),
which is the parser underneath both pytometry and FlowKit. Depending on it
directly keeps the output contract stable no matter which higher-level
cytometry package the user happens to have installed.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from ..._registry import register_function

__all__ = ["read_fcs", "parse_spillover"]

_INSTALL_HINT = (
    "Reading FCS files needs `flowio` (BSD-3, a few hundred KB):\n"
    "    pip install flowio\n"
    "or install the fuller cytometry stack:\n"
    "    pip install pytometry"
)


def _flowio():
    try:
        import flowio  # noqa: F401
        return flowio
    except ImportError as exc:                                # pragma: no cover
        raise ImportError(f"{exc}\n{_INSTALL_HINT}") from exc


def parse_spillover(text: str, channels: Sequence[str]) -> Optional[pd.DataFrame]:
    r"""Parse an FCS ``$SPILLOVER`` / ``$SPILL`` string into a square DataFrame.

    The keyword is a flat, comma-delimited string: a channel count ``n``, then
    ``n`` channel names, then ``n*n`` matrix values in row-major order. Both
    pytometry and FlowKit hand this back as the raw string; parsing it here
    means the matrix is usable immediately and a malformed one is caught at read
    time rather than at compensation time.

    Returns ``None`` when the keyword is absent or does not parse — a missing
    spillover matrix is completely normal (unstained runs, mass cytometry) and
    must not be an error.
    """
    if not text:
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip() != ""]
    if not parts:
        return None
    try:
        n = int(float(parts[0]))
    except (TypeError, ValueError):
        return None
    if n <= 0 or len(parts) < 1 + n + n * n:
        return None

    names = parts[1: 1 + n]
    try:
        values = [float(v) for v in parts[1 + n: 1 + n + n * n]]
    except (TypeError, ValueError):
        return None

    matrix = np.asarray(values, dtype=float).reshape(n, n)
    # Some vendors write detector INDEXES rather than names. Map them back so
    # the frame is labelled the same way as `var['channel']` either way.
    if all(re.fullmatch(r"\d+", nm) for nm in names) and channels:
        try:
            names = [channels[int(nm) - 1] for nm in names]
        except (IndexError, ValueError):                      # pragma: no cover
            pass
    return pd.DataFrame(matrix, index=names, columns=names)


def _channel_frame(channels: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """Build the `var` frame from flowio's per-channel keyword dict."""
    rows = []
    for idx in sorted(channels, key=int):
        c = channels[idx]
        pnn = (c.get("pnn") or "").strip()
        pns = (c.get("pns") or "").strip()
        rows.append({
            "n": int(idx),
            "channel": pnn,
            "marker": pns,
            "PnB": c.get("pnb"),
            "PnE": c.get("pne"),
            "PnG": c.get("png"),
            "PnR": c.get("pnr"),
        })
    var = pd.DataFrame(rows)
    # Index by the MARKER when the file names one, else the detector. This is
    # pytometry's rule and the one a biologist expects: you asked for CD3, not
    # for whatever detector CD3 happened to sit on in this panel.
    var.index = pd.Index(
        [m if m else c for m, c in zip(var["marker"], var["channel"])],
        name=None,
    )
    # A panel can legitimately put the same marker on two detectors; make the
    # index unique by falling back to the detector name for the duplicates
    # rather than silently returning an AnnData that cannot be sliced by name.
    if var.index.duplicated().any():
        seen: Dict[str, int] = {}
        fixed = []
        for name, chan in zip(var.index, var["channel"]):
            if name in seen:
                seen[name] += 1
                fixed.append(f"{name} ({chan})" if chan else f"{name}.{seen[name]}")
            else:
                seen[name] = 0
                fixed.append(name)
        var.index = pd.Index(fixed)
    return var


@register_function(
    aliases=["read_fcs", "read_flow", "读取fcs", "流式细胞术读取", "fcs", "flow_cytometry_read"],
    category="io",
    description=(
        "Read a Flow Cytometry Standard (.fcs) file into an AnnData: events x "
        "channels, with the detector name ($PnN) and the marker on it ($PnS) "
        "kept as separate var columns, the full FCS keyword dictionary in "
        "uns['fcs'], and the acquisition $SPILLOVER matrix parsed into a "
        "labelled DataFrame ready for compensation. Works for conventional "
        "flow, spectral flow and mass cytometry (CyTOF) files alike. The var "
        "schema matches pytometry, so the result feeds pytometry.pp.compensate "
        "and pytometry.tl.normalize_logicle unchanged."
    ),
    examples=[
        "adata = ov.io.read_fcs('sample.fcs')",
        "adata.var[['channel', 'marker']]           # FITC-A -> CD3",
        "adata.uns['fcs']['spillover']              # parsed, square DataFrame",
        "adata = ov.io.read_fcs('sample.fcs', markers_only=True)",
    ],
    related=["io.read", "single.ev.read_fcs"],
)
def read_fcs(
    path: str,
    *,
    channels: Optional[Sequence[str]] = None,
    markers_only: bool = False,
    sample: Optional[str] = None,
    compensated: bool = False,
    preprocess: bool = True,
    dtype: Union[str, np.dtype] = "float32",
) -> AnnData:
    r"""Read an ``.fcs`` file.

    Arguments
    ---------
    path
        Path to a Flow Cytometry Standard file. FCS **2.0 / 3.0 / 3.1** —
        the versions the flowio parser supports; 3.2 is not claimed.
    channels
        Keep only these channels. Matched against BOTH the detector name
        (``$PnN``, e.g. ``'FITC-A'``) and the marker (``$PnS``, e.g. ``'CD3'``),
        so either vocabulary works.
    markers_only
        Keep only channels that carry a ``$PnS`` marker — i.e. drop scatter and
        time. Convenient, but note that FSC/SSC are exactly what the first
        gates need, so leave it ``False`` for a gating workflow.
    sample
        Value for ``obs['sample']``. Defaults to the filename stem, which makes
        concatenating several files give a usable sample column for free.
    preprocess
        Convert the stored CHANNEL values into SCALE values — apply ``$PnG``
        gain, decode ``$PnE`` log-amplified channels, and scale time by
        ``$TIMESTEP``. On by default because scale values are what every
        downstream step assumes; set ``False`` only if you specifically want the
        raw ADC buffer.
    compensated
        Apply the file's own ``$SPILLOVER`` matrix on read. ``False`` by default
        and deliberately so: the acquisition matrix is frequently wrong, most
        labs recompute it from single-stain controls, and applying it silently
        would leave no way to tell whether it had been applied. The matrix is
        always available in ``uns['fcs']['spillover']``.
    dtype
        Storage dtype for ``X``. ``float32`` halves memory on a 10^7-event file
        and is well past the precision of a 18-bit ADC.

    Returns
    -------
    :class:`~anndata.AnnData`
        ``X`` events x channels; ``var`` with ``n / channel / marker / PnB /
        PnE / PnG / PnR``, indexed by marker where one exists; ``obs`` with
        ``sample``; ``uns['fcs']`` with ``keywords``, ``spillover`` (a square
        DataFrame or ``None``), ``version`` and ``compensated``.

    Notes
    -----
    Nothing here is assay-specific. ``ov.single.ev.read_fcs`` wraps this for the
    single-EV convention (it adds ``uns['ev']``).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"FCS file not found: {path}")

    flowio = _flowio()
    fd = flowio.FlowData(str(path))

    var = _channel_frame(fd.channels)
    # `as_array(preprocess=True)`, NOT the raw `fd.events` buffer. The DATA
    # segment holds CHANNEL values; turning them into SCALE values means
    # applying `$PnG` (amplifier gain), decoding `$PnE` log-amplified channels
    # as ``10 ** (decades * x / range) * log0``, and scaling the time channel by
    # `$TIMESTEP`. Reading the raw buffer silently returns a gain-2.0 channel at
    # twice its true value and a log-amplified channel off by orders of
    # magnitude — every downstream gate would then be drawn in the wrong place.
    values = fd.as_array(preprocess=preprocess)

    keywords = dict(fd.text)
    spill = parse_spillover(
        keywords.get("spillover") or keywords.get("spill") or "",
        list(var["channel"]),
    )

    if compensated:
        if spill is None:
            raise ValueError(
                "compensated=True but this file carries no $SPILLOVER matrix. "
                "Pass the matrix explicitly to your compensation step instead."
            )
        values = _apply_spillover(values, list(var["channel"]), spill)

    # Subset AFTER compensation: spillover is defined over the full detector
    # set, so dropping channels first would silently compensate against a
    # different matrix than the instrument recorded.
    keep = np.ones(len(var), dtype=bool)
    if markers_only:
        keep &= var["marker"].astype(str).str.len().to_numpy() > 0
    if channels is not None:
        wanted = {str(c) for c in channels}
        keep &= (var["channel"].isin(wanted) | var["marker"].isin(wanted)).to_numpy()
        missing = wanted - set(var["channel"]) - set(var["marker"])
        if missing:
            raise KeyError(
                f"channel(s) not in this file: {sorted(missing)}. "
                f"Available detectors: {sorted(var['channel'])}"
            )
    if not keep.any():
        raise ValueError("no channels left after filtering")
    values = values[:, keep]
    var = var.loc[keep].copy()

    obs = pd.DataFrame(index=pd.RangeIndex(values.shape[0]).astype(str))
    obs["sample"] = sample if sample is not None else os.path.splitext(os.path.basename(path))[0]

    adata = AnnData(X=np.ascontiguousarray(values, dtype=dtype), obs=obs, var=var)
    # `uns['meta']` is pytometry's contract and NOT decoration: `pp.compensate`
    # does `adata.uns["meta"]["spill"]` and reads `.columns` off it, so the key
    # must exist and hold the PARSED matrix (or None), not the raw keyword
    # string. Matching the var schema alone was not enough to be a drop-in —
    # this cost two failed interop runs to find, and the test suite pins it.
    adata.uns["meta"] = dict(keywords)
    adata.uns["meta"]["spill"] = spill
    adata.uns["fcs"] = {
        # Same object, not a copy: `uns['meta']` is the interop name and
        # `uns['fcs']['keywords']` the discoverable one, for one dict.
        "keywords": keywords,
        "spillover": spill,
        "version": getattr(fd, "version", None),
        "compensated": bool(compensated),
        "preprocessed": bool(preprocess),
        "path": os.path.abspath(path),
    }
    return adata


def _apply_spillover(values: np.ndarray, channels: Sequence[str],
                     spill: pd.DataFrame) -> np.ndarray:
    """Right-multiply the fluorescence block by the inverse spillover matrix.

    Only the channels the matrix names are touched — scatter and time must pass
    through untouched, and a matrix that names a channel the file does not have
    is a genuine mismatch worth refusing.
    """
    idx = []
    for name in spill.columns:
        if name not in list(channels):
            raise ValueError(
                f"$SPILLOVER names channel {name!r}, which is not in this file. "
                "The matrix does not belong to these events."
            )
        idx.append(list(channels).index(name))
    block = values[:, idx]
    values = values.copy()
    values[:, idx] = block @ np.linalg.inv(spill.to_numpy())
    return values
