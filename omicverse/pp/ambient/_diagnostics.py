"""Diagnostics for ambient-RNA decontamination.

These helpers verify that an ambient-removal run behaved sensibly — the
"count-integrity" and "negative-marker" checks stressed by the 2026
ambient-RNA benchmark (Janssen *et al.*).  None of them modify the data;
they are read-only audits of a corrected :class:`anndata.AnnData`.

* :func:`ambient_negative_marker_check` — a marker that a cell type does
  *not* express should drop to ~0 in that cell type after correction.
* :func:`count_integrity_check` — corrected counts must not exceed the
  raw counts and the matrix must not have been wholesale rewritten.
* :func:`contamination_report` — a compact per-method summary table.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..._registry import register_function


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_dense_1d(x) -> np.ndarray:
    """Flatten a (possibly sparse) column/row vector to a dense 1-D array."""
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x).ravel()


def _get_matrix(adata, layer: Optional[str]):
    """Return the count matrix from ``.X`` or a named layer."""
    m = adata.layers[layer] if layer is not None else adata.X
    return m


# ---------------------------------------------------------------------------
# 1. negative-marker check
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "ambient_negative_marker_check", "negative_marker_check",
        "阴性标志物检查", "环境RNA阴性标志物",
    ],
    category="preprocessing",
    description=(
        "Diagnostic for ambient-RNA correction: confirm a marker gene "
        "drops toward zero in cell types that should NOT express it after "
        "decontamination. Compares the raw vs corrected mean expression of "
        "the marker in the off-target cell types and reports the fold "
        "reduction. A large drop indicates the soup was removed correctly."
    ),
    examples=[
        "ov.pp.ambient.ambient_negative_marker_check(adata, 'HBB', 'cell_type')",
        "df = ov.pp.ambient.ambient_negative_marker_check(adata, 'HBB', "
        "'cell_type', positive_celltypes=['Erythrocyte'])",
    ],
    related=["pp.ambient.remove_ambient", "pp.ambient.count_integrity_check"],
)
def ambient_negative_marker_check(
    adata,
    marker: str,
    celltype_key: str,
    *,
    raw_layer: Optional[str] = None,
    corrected_layer: Optional[str] = None,
    positive_celltypes: Optional[list] = None,
):
    """Check a negative marker drops to ~0 after ambient correction.

    Ambient ("soup") RNA spreads a few highly-expressed transcripts across
    every droplet.  A correct decontamination should drive a cell-type
    specific marker back toward zero in the cell types that biologically
    do **not** express it.  This audits exactly that.

    Parameters
    ----------
    adata
        AnnData carrying both the raw and the corrected counts (as ``.X``
        and a layer, or two layers).
    marker
        Gene name (in ``adata.var_names``) expected to be specific to one
        or a few cell types.
    celltype_key
        Column of ``adata.obs`` holding cell-type labels.
    raw_layer
        Layer with the raw (pre-correction) counts. ``None`` means the
        raw counts live in a layer named ``'ambient_raw'`` if present,
        else ``.X`` is used as raw.
    corrected_layer
        Layer with the corrected counts. ``None`` uses ``.X``.
    positive_celltypes
        Cell types that legitimately express ``marker``. They are excluded
        from the "should-be-zero" set. If ``None``, the cell type with the
        highest raw mean expression is taken as the positive one.

    Returns
    -------
    :class:`pandas.DataFrame`
        Per cell type: raw mean, corrected mean, fold reduction and an
        ``is_negative`` flag. ``.attrs['summary']`` holds the aggregate
        result over the negative cell types.
    """
    if marker not in adata.var_names:
        raise KeyError(f"marker '{marker}' not in adata.var_names.")
    if celltype_key not in adata.obs:
        raise KeyError(f"celltype_key '{celltype_key}' not in adata.obs.")

    gi = adata.var_names.get_loc(marker)

    # resolve raw / corrected layers
    if raw_layer is None and "ambient_raw" in adata.layers:
        raw_layer = "ambient_raw"
    raw = _get_matrix(adata, raw_layer)
    corr = _get_matrix(adata, corrected_layer)

    raw_g = _as_dense_1d(raw[:, gi])
    corr_g = _as_dense_1d(corr[:, gi])

    labels = adata.obs[celltype_key].astype(str).to_numpy()
    rows = []
    for ct in pd.unique(labels):
        mask = labels == ct
        raw_mean = float(raw_g[mask].mean())
        corr_mean = float(corr_g[mask].mean())
        fold = raw_mean / corr_mean if corr_mean > 0 else np.inf
        rows.append({
            "celltype": ct,
            "n_cells": int(mask.sum()),
            "raw_mean": raw_mean,
            "corrected_mean": corr_mean,
            "fold_reduction": fold,
        })
    df = pd.DataFrame(rows).set_index("celltype")

    # decide which cell types are the "positive" (legitimate) ones
    if positive_celltypes is None:
        positive_celltypes = [df["raw_mean"].idxmax()]
    df["is_negative"] = ~df.index.isin(set(positive_celltypes))

    neg = df[df["is_negative"]]
    summary = {
        "marker": marker,
        "positive_celltypes": list(positive_celltypes),
        "n_negative_celltypes": int(neg.shape[0]),
        "neg_raw_mean": float(neg["raw_mean"].mean()) if len(neg) else 0.0,
        "neg_corrected_mean": (
            float(neg["corrected_mean"].mean()) if len(neg) else 0.0),
    }
    summary["neg_fold_reduction"] = (
        summary["neg_raw_mean"] / summary["neg_corrected_mean"]
        if summary["neg_corrected_mean"] > 0 else np.inf)
    df.attrs["summary"] = summary
    return df


# ---------------------------------------------------------------------------
# 2. count-integrity check
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "count_integrity_check", "ambient_count_integrity",
        "计数完整性检查", "校正计数完整性",
    ],
    category="preprocessing",
    description=(
        "Diagnostic for ambient-RNA correction: verify the corrected count "
        "matrix never exceeds the raw counts and was not wholesale "
        "rewritten (the 2026 ambient-RNA benchmark count-integrity "
        "criterion). Reports the number of entries that increased, the "
        "fraction of entries left unchanged and the total counts removed."
    ),
    examples=[
        "ov.pp.ambient.count_integrity_check(raw_adata, corrected_adata)",
        "report = ov.pp.ambient.count_integrity_check(raw, corrected, "
        "raise_on_fail=True)",
    ],
    related=["pp.ambient.remove_ambient", "pp.ambient.contamination_report"],
)
def count_integrity_check(
    raw,
    corrected,
    *,
    raw_layer: Optional[str] = None,
    corrected_layer: Optional[str] = None,
    tol: float = 1e-6,
    raise_on_fail: bool = False,
):
    """Verify corrected counts respect the raw counts.

    A trustworthy ambient correction only *subtracts* contamination: no
    entry may grow, and only a minority of entries (the contaminated
    genes) should change at all. This is the count-integrity criterion of
    the 2026 ambient-RNA benchmark.

    Parameters
    ----------
    raw
        AnnData (or matrix) of raw, pre-correction counts.
    corrected
        AnnData (or matrix) of corrected counts. Must match ``raw`` in
        shape.
    raw_layer, corrected_layer
        Optional layer keys (ignored when a bare matrix is passed).
    tol
        Numerical tolerance below which a difference counts as "unchanged".
    raise_on_fail
        When ``True`` raise a :class:`ValueError` if any entry increased.

    Returns
    -------
    dict
        ``passed`` (bool), ``n_increased`` (entries that grew),
        ``max_increase``, ``frac_unchanged``, ``frac_changed``,
        ``total_raw``, ``total_corrected``, ``total_removed`` and
        ``removed_fraction``.
    """
    def _mat(x, layer):
        if hasattr(x, "obs") and hasattr(x, "X"):
            return _get_matrix(x, layer)
        return x

    rm = _mat(raw, raw_layer)
    cm = _mat(corrected, corrected_layer)

    rm = rm.toarray() if sp.issparse(rm) else np.asarray(rm)
    cm = cm.toarray() if sp.issparse(cm) else np.asarray(cm)
    if rm.shape != cm.shape:
        raise ValueError(
            f"raw {rm.shape} and corrected {cm.shape} shape mismatch.")
    rm = rm.astype(np.float64)
    cm = cm.astype(np.float64)

    diff = cm - rm
    increased = diff > tol
    n_increased = int(increased.sum())
    max_increase = float(diff.max()) if diff.size else 0.0
    n_total = int(rm.size)
    n_unchanged = int((np.abs(diff) <= tol).sum())

    total_raw = float(rm.sum())
    total_corr = float(cm.sum())
    total_removed = total_raw - total_corr

    passed = n_increased == 0
    report = {
        "passed": bool(passed),
        "n_entries": n_total,
        "n_increased": n_increased,
        "max_increase": max_increase,
        "n_unchanged": n_unchanged,
        "frac_unchanged": n_unchanged / n_total if n_total else 1.0,
        "frac_changed": 1.0 - (n_unchanged / n_total if n_total else 1.0),
        "total_raw": total_raw,
        "total_corrected": total_corr,
        "total_removed": total_removed,
        "removed_fraction": total_removed / total_raw if total_raw else 0.0,
    }
    if raise_on_fail and not passed:
        raise ValueError(
            f"count integrity FAILED: {n_increased} entries increased "
            f"(max +{max_increase:.4g}). The corrected matrix must never "
            "exceed the raw counts.")
    return report


# ---------------------------------------------------------------------------
# 3. compact contamination report
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "contamination_report", "ambient_report", "污染报告",
        "环境RNA报告",
    ],
    category="preprocessing",
    description=(
        "Compact summary of an ambient-RNA decontamination run: the "
        "per-cell contamination fraction (mean / median / range), the "
        "number of genes corrected, the method used and the "
        "count-integrity statistics, read out of adata.obs / adata.uns "
        "populated by ov.pp.ambient.remove_ambient."
    ),
    examples=[
        "ov.pp.ambient.contamination_report(adata)",
        "summary = ov.pp.ambient.contamination_report(adata)",
    ],
    related=["pp.ambient.remove_ambient", "pp.ambient.count_integrity_check"],
)
def contamination_report(adata, *, uns_key: str = "ambient"):
    """Compact summary of an ambient-correction run.

    Reads the metadata that :func:`~omicverse.pp.ambient.remove_ambient`
    writes into ``adata.uns[uns_key]`` and ``adata.obs`` and returns it as
    a tidy one-row :class:`pandas.DataFrame`.

    Parameters
    ----------
    adata
        AnnData processed by :func:`remove_ambient`.
    uns_key
        ``adata.uns`` key holding the ambient metadata. Default
        ``'ambient'``.

    Returns
    -------
    :class:`pandas.DataFrame`
        A single-row summary. ``.attrs['raw']`` holds the underlying
        ``uns`` dict.
    """
    info = adata.uns.get(uns_key, {})
    if not info:
        raise KeyError(
            f"adata.uns['{uns_key}'] is empty — run ov.pp.ambient."
            "remove_ambient first.")

    method = info.get("method", "unknown")
    frac_col = info.get("contamination_obs_key", "ambient_contamination")
    frac = None
    if frac_col in adata.obs:
        frac = adata.obs[frac_col].to_numpy(dtype=float)

    row = {
        "method": method,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "genes_corrected": info.get("n_genes_corrected", np.nan),
    }
    if frac is not None and frac.size:
        row.update({
            "mean_contamination": float(np.nanmean(frac)),
            "median_contamination": float(np.nanmedian(frac)),
            "min_contamination": float(np.nanmin(frac)),
            "max_contamination": float(np.nanmax(frac)),
        })
    else:
        row.update({
            "mean_contamination": info.get("contamination_fraction", np.nan),
            "median_contamination": np.nan,
            "min_contamination": np.nan,
            "max_contamination": np.nan,
        })

    integ = info.get("count_integrity", {})
    if integ:
        row["integrity_passed"] = integ.get("passed")
        row["removed_fraction"] = integ.get("removed_fraction")
        row["frac_genes_changed"] = integ.get("frac_changed")

    df = pd.DataFrame([row])
    df.attrs["raw"] = dict(info)
    return df
