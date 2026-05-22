"""Preprocessing for single-extracellular-vesicle (single-EV) proteomics.

Single-EV proteomics produces an EV x protein matrix (vesicle ~ cell, protein
marker ~ gene). The standard single-cell stack transfers, but the
normalization step must branch on the *value type* of the assay:

* ``count``     — PBA / DBS-Pro sequencing reads. Centered-log-ratio (CLR),
  the CITE-seq antibody-derived-tag (ADT) convention, is the right transform.
* ``intensity`` — NanoFCM / ExoView flow-imaging fluorescence. ``arcsinh``
  (the CyTOF convention, configurable cofactor) or ``log2``.
* ``binary``    — droplet digital presence/absence. Passes through unchanged.

The value type is read from ``adata.uns['ev']['value_type']`` so that
``normalize(adata, method='auto')`` does the right thing automatically.

Only the EV-specific normalization lives here. The generic single-cell
preprocessing steps that follow it — scaling, PCA and the kNN graph — are
omicverse-native and should be run with :func:`omicverse.pp.scale`,
:func:`omicverse.pp.pca` and :func:`omicverse.pp.neighbors`.

Functions
---------
* :func:`normalize`  — value-type-aware normalization (CLR / arcsinh / log2),
  with an optional per-EV size-factor correction for vesicle size.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..._registry import register_function


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dense(x):
    """Return a dense float64 ndarray from a (possibly sparse) matrix."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def _value_type(adata) -> str:
    """Read the EV value type from ``uns['ev']``, defaulting to ``'count'``."""
    ev = adata.uns.get("ev", {}) if hasattr(adata, "uns") else {}
    return str(ev.get("value_type", "count")).lower()


def _clr(mat: np.ndarray, axis: int = 1, pseudocount: float = 1.0) -> np.ndarray:
    """Centered-log-ratio transform along ``axis`` (per-EV when ``axis=1``).

    ``clr(x) = log(x + pc) - mean_axis(log(x + pc))`` — the CITE-seq ADT
    normalization. The geometric-mean centering removes the per-EV
    composition / depth effect.
    """
    logm = np.log1p(mat) if pseudocount == 1.0 else np.log(mat + pseudocount)
    return logm - logm.mean(axis=axis, keepdims=True)


def _size_factors(mat: np.ndarray) -> np.ndarray:
    """Per-EV size factor = (total signal) / (median total signal).

    Dividing each EV by its size factor controls for vesicle size /
    total surface-protein abundance before the value transform.
    """
    totals = mat.sum(axis=1)
    med = np.median(totals[totals > 0]) if np.any(totals > 0) else 1.0
    sf = totals / med
    sf[sf <= 0] = 1.0
    return sf


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------
@register_function(
    aliases=["ev_normalize", "normalize_ev", "EV标准化", "单囊泡标准化"],
    category="ev",
    description=(
        "Value-type-aware normalization of a single-EV proteomics AnnData. "
        "method='auto' reads uns['ev']['value_type']: CLR (centered-log-"
        "ratio) for count data, arcsinh or log2 for intensity data, and a "
        "pass-through for binary data. An optional per-EV size-factor "
        "correction controls for vesicle size / total surface protein."
    ),
    examples=[
        "ov.single.ev.normalize(adata)",
        "ov.single.ev.normalize(adata, method='clr')",
        "ov.single.ev.normalize(adata, method='arcsinh', cofactor=150)",
        "ov.single.ev.normalize(adata, method='log2', size_factor=True)",
    ],
    related=["pp.scale", "pp.pca"],
)
def normalize(
    adata,
    *,
    method: str = "auto",
    cofactor: float = 5.0,
    pseudocount: float = 1.0,
    size_factor: bool = False,
    layer: Optional[str] = None,
    key_added: Optional[str] = None,
):
    """Value-type-aware normalization for single-EV proteomics.

    Parameters
    ----------
    adata
        EV x protein AnnData. Raw values are taken from ``layers['counts']``
        if present, otherwise from ``adata.X``.
    method
        ``'auto'`` (default) picks from ``uns['ev']['value_type']``:
        ``count`` -> ``'clr'``, ``intensity`` -> ``'arcsinh'``,
        ``binary`` -> ``'none'``. May also be set explicitly to ``'clr'``,
        ``'arcsinh'``, ``'log2'`` or ``'none'``.
    cofactor
        Cofactor for the ``arcsinh`` transform, ``arcsinh(x / cofactor)``
        (the CyTOF convention; 5 for mass cytometry, ~150 for fluorescence).
    pseudocount
        Pseudocount added before the ``clr`` / ``log2`` logarithm.
    size_factor
        If ``True``, divide each EV by its size factor (total signal over
        the median total signal) *before* the value transform, to control
        for vesicle size / total surface-protein abundance.
    layer
        Input layer; ``None`` uses ``layers['counts']`` or ``X``.
    key_added
        If given, the normalized matrix is written to ``layers[key_added]``
        and ``X`` is left untouched; otherwise it overwrites ``adata.X``.

    Returns
    -------
    :class:`anndata.AnnData`
        The same object, with the normalized matrix in ``X`` (or in
        ``layers[key_added]``) and a record under ``uns['ev']['normalize']``.

    Notes
    -----
    The raw input is preserved in ``layers['counts']`` (created on first
    call) so the transform is always reproducible / re-runnable.
    """
    valid = {"auto", "clr", "arcsinh", "log2", "none"}
    if method not in valid:
        raise ValueError(f"method must be one of {sorted(valid)}, got {method!r}")

    # resolve raw input matrix
    if layer is not None:
        mat = _dense(adata.layers[layer])
    elif "counts" in adata.layers:
        mat = _dense(adata.layers["counts"])
    else:
        mat = _dense(adata.X)
        adata.layers["counts"] = mat.copy()

    vtype = _value_type(adata)
    if method == "auto":
        method = {
            "count": "clr",
            "intensity": "arcsinh",
            "binary": "none",
        }.get(vtype, "clr")

    if size_factor and method != "none":
        sf = _size_factors(mat)
        mat = mat / sf[:, None]
    else:
        sf = None

    if method == "clr":
        out = _clr(mat, axis=1, pseudocount=pseudocount)
    elif method == "arcsinh":
        out = np.arcsinh(mat / float(cofactor))
    elif method == "log2":
        out = np.log2(mat + pseudocount)
    else:  # none — binary / pass-through
        out = mat.astype(np.float64)

    if key_added is not None:
        adata.layers[key_added] = out
    else:
        adata.X = out

    ev = adata.uns.setdefault("ev", {})
    ev["normalize"] = {
        "method": method,
        "value_type": vtype,
        "cofactor": float(cofactor),
        "pseudocount": float(pseudocount),
        "size_factor": bool(size_factor),
        "layer_out": key_added,
    }
    return adata

