"""MISEV2023 / MIFlowCyt-EV characterization reporting for single-EV data.

The Minimal Information for Studies of Extracellular Vesicles (MISEV2023)
guidelines ask every EV study to document which positive EV markers and which
contaminant ("non-EV") markers were assessed, plus particle/protein and
single-EV characterization metadata.

* :func:`misev_report` — assemble a MISEV2023-aligned characterization
  report (positive markers present, contaminants present, summaries).
* :func:`ev_summary` — a compact run summary (n EVs / proteins /
  subpopulations, value type, platform, QC pass rates).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..._registry import register_function

# MISEV2023 marker categories — used as the default classification when
# ``var['misev_category']`` has not been filled by an annotation step.
#: canonical positive transmembrane / lumen EV markers
MISEV_POSITIVE = {
    "CD9", "CD63", "CD81", "CD82", "TSPAN8", "TSPAN14",
    "FLOT1", "FLOT2", "TFR2", "SDCBP", "PDCD6IP", "ALIX",
    "TSG101", "CD47", "ITGB1", "ITGA4", "ANXA5", "HSPA8",
    "HSP90AB1", "ACTB", "GAPDH", "EGFR", "BSG",
}
#: canonical contaminant / non-EV co-isolate markers
MISEV_CONTAMINANT = {
    "ALB", "APOA1", "APOB", "APOE", "CALR", "CANX", "HSP90B1",
    "GRP94", "HIST1H1C", "HISTH2B", "RPL", "RPS", "GM130", "GOLGA2",
    "CYC1", "TOMM20", "ACTN4", "AGO2", "UMOD", "FN1",
}
#: MISEV "topology" categories that count as positive evidence
_POSITIVE_CATS = {"transmembrane", "cytosolic", "positive", "ev_positive"}
_CONTAMINANT_CATS = {"contaminant", "non_ev", "negative", "ev_negative"}


def _dense(adata):
    """Return ``adata.X`` as a dense float ndarray."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def _classify(adata):
    """Return a Series mapping each protein to 'positive'/'contaminant'/'other'.

    Uses ``var['misev_category']`` when present; otherwise falls back to the
    built-in canonical marker name sets (case-insensitive).
    """
    proteins = list(adata.var_names)
    if "misev_category" in adata.var:
        cats = adata.var["misev_category"].astype(str)
        out = {}
        for p in proteins:
            c = str(cats.get(p, "")).strip().lower()
            if c in _POSITIVE_CATS:
                out[p] = "positive"
            elif c in _CONTAMINANT_CATS:
                out[p] = "contaminant"
            else:
                out[p] = "other"
        return pd.Series(out)
    out = {}
    for p in proteins:
        u = str(p).upper()
        if u in MISEV_POSITIVE:
            out[p] = "positive"
        elif u in MISEV_CONTAMINANT or any(
            u.startswith(pre) for pre in ("RPL", "RPS")
        ):
            out[p] = "contaminant"
        else:
            out[p] = "other"
    return pd.Series(out)


@register_function(
    aliases=[
        "misev_report", "ev_misev_report", "misev2023_report",
        "MISEV报告", "EV表征报告",
    ],
    category="ev",
    description=(
        "Assemble a MISEV2023-aligned single-EV characterization report: "
        "which positive EV markers and which contaminant (non-EV) markers "
        "are present and at what level, a particle/protein summary, the "
        "single-EV characterization metadata (value type, platform) and an "
        "overall purity score (positive vs contaminant signal)."
    ),
    examples=[
        "rep = ov.single.ev.misev_report(adata)",
        "rep = ov.single.ev.misev_report(adata, as_frame=True)",
    ],
    related=["single.ev.ev_summary", "single.ev.plotting.misev_marker_plot"],
)
def misev_report(adata, *, as_frame: bool = False):
    """Assemble a MISEV2023-aligned EV characterization report.

    Parameters
    ----------
    adata
        Single-EV AnnData. ``var['misev_category']`` is used to classify
        markers when present; otherwise the built-in canonical marker sets
        are used.
    as_frame
        When ``True`` return the per-marker classification table only;
        otherwise return the full structured report dict.

    Returns
    -------
    dict or :class:`pandas.DataFrame`
        With ``as_frame=False`` a dict with keys ``meta``, ``markers``
        (per-protein table), ``positive_markers``, ``contaminant_markers``,
        ``summary`` (counts / purity score) and ``misev_checklist``. With
        ``as_frame=True`` just the ``markers`` table.
    """
    cls = _classify(adata)
    X = _dense(adata)
    mean_signal = X.mean(axis=0)
    frac_pos = (X > 0).mean(axis=0)
    proteins = list(adata.var_names)

    markers = pd.DataFrame({
        "protein": proteins,
        "misev_class": [cls[p] for p in proteins],
        "mean_signal": mean_signal,
        "frac_ev_positive": frac_pos,
    })
    if "misev_category" in adata.var:
        markers["misev_category"] = adata.var["misev_category"].astype(
            str
        ).reindex(proteins).values
    markers = markers.sort_values(
        ["misev_class", "mean_signal"], ascending=[True, False]
    ).reset_index(drop=True)

    if as_frame:
        return markers

    pos = markers[markers["misev_class"] == "positive"]
    con = markers[markers["misev_class"] == "contaminant"]
    pos_signal = float(pos["mean_signal"].sum())
    con_signal = float(con["mean_signal"].sum())
    purity = (
        pos_signal / (pos_signal + con_signal)
        if (pos_signal + con_signal) > 0 else float("nan")
    )
    ev_uns = dict(adata.uns.get("ev", {}))
    meta = {
        "n_evs": int(adata.n_obs),
        "n_proteins": int(adata.n_vars),
        "value_type": ev_uns.get("value_type", "unknown"),
        "platform": ev_uns.get("platform", "unknown"),
    }
    summary = {
        "n_positive_markers": int(len(pos)),
        "n_contaminant_markers": int(len(con)),
        "n_other_markers": int((markers["misev_class"] == "other").sum()),
        "positive_signal": pos_signal,
        "contaminant_signal": con_signal,
        "purity_score": purity,
        "mean_proteins_per_ev": float((X > 0).sum(axis=1).mean()),
        "mean_total_signal_per_ev": float(X.sum(axis=1).mean()),
    }
    checklist = {
        "positive_ev_markers_assessed": bool(len(pos) > 0),
        "contaminant_markers_assessed": bool(len(con) > 0),
        "single_ev_characterization": True,
        "value_type_documented": meta["value_type"] != "unknown",
        "platform_documented": meta["platform"] != "unknown",
        "particle_count_reported": meta["n_evs"] > 0,
    }
    return {
        "meta": meta,
        "markers": markers,
        "positive_markers": list(pos["protein"]),
        "contaminant_markers": list(con["protein"]),
        "summary": summary,
        "misev_checklist": checklist,
    }


@register_function(
    aliases=[
        "ev_summary", "single_ev_summary", "ev_run_summary",
        "EV运行摘要", "EV数据概览",
    ],
    category="ev",
    description=(
        "Compact run-level summary of a single-EV proteomics AnnData: number "
        "of EVs, proteins and subpopulations, value type, platform, samples "
        "and QC pass rates (EVs meeting minimum protein / signal thresholds)."
    ),
    examples=[
        "s = ov.single.ev.ev_summary(adata)",
        "s = ov.single.ev.ev_summary(adata)  # returns a one-row DataFrame",
    ],
    related=["single.ev.misev_report"],
)
def ev_summary(
    adata,
    *,
    cluster_key: Optional[str] = None,
    sample_key: str = "sample",
    min_proteins: int = 1,
    min_signal: float = 0.0,
):
    """Compact run summary of a single-EV proteomics dataset.

    Parameters
    ----------
    adata
        Single-EV AnnData.
    cluster_key
        ``obs`` column with EV-subpopulation labels; auto-detected from
        ``'leiden'`` / ``'flowsom'`` / ``'cluster'`` when ``None``.
    sample_key
        ``obs`` column with sample ids (default ``'sample'``).
    min_proteins
        An EV passes QC when it detects at least this many proteins.
    min_signal
        An EV passes QC when its total signal exceeds this value.

    Returns
    -------
    :class:`pandas.DataFrame`
        A one-row summary table — ``n_evs``, ``n_proteins``,
        ``n_subpopulations``, ``n_samples``, ``value_type``, ``platform``,
        ``mean_proteins_per_ev``, ``median_total_signal``,
        ``qc_pass_rate``.
    """
    X = _dense(adata)
    n_prot_per_ev = (X > 0).sum(axis=1)
    total_signal = X.sum(axis=1)

    if cluster_key is None:
        for cand in ("leiden", "flowsom", "cluster", "louvain"):
            if cand in adata.obs:
                cluster_key = cand
                break
    n_subpop = (
        int(adata.obs[cluster_key].astype(str).nunique())
        if cluster_key is not None and cluster_key in adata.obs else 0
    )
    n_samples = (
        int(adata.obs[sample_key].astype(str).nunique())
        if sample_key in adata.obs else 0
    )
    qc_pass = (n_prot_per_ev >= min_proteins) & (total_signal > min_signal)
    ev_uns = dict(adata.uns.get("ev", {}))

    row = {
        "n_evs": int(adata.n_obs),
        "n_proteins": int(adata.n_vars),
        "n_subpopulations": n_subpop,
        "n_samples": n_samples,
        "value_type": ev_uns.get("value_type", "unknown"),
        "platform": ev_uns.get("platform", "unknown"),
        "mean_proteins_per_ev": float(n_prot_per_ev.mean()),
        "median_total_signal": float(np.median(total_signal)),
        "qc_pass_rate": float(qc_pass.mean()),
    }
    return pd.DataFrame([row])
