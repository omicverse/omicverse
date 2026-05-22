"""Quality control for single-extracellular-vesicle (single-EV) proteomics.

These functions operate on the shared ``ov.single.ev`` AnnData schema (see
:mod:`omicverse.single.ev.io`): an EV x protein matrix with the raw values in
``layers['counts']`` and the value type recorded in ``uns['ev']``.

Single-EV QC differs from single-cell QC in the artifacts it targets:

* **fragments / noise** — EVs (rows) with too few detected proteins are likely
  membrane fragments, free antibody or background, and are removed by
  :func:`qc`.
* **doublets / swarm / barcode collisions** — EVs with implausibly high total
  signal are two-or-more vesicles read as one, capped by :func:`qc` and flagged
  by :func:`detect_doublets`.
* **background binding** — non-specific antibody binding is estimated from
  isotype / IgG / buffer-blank controls by :func:`subtract_isotype`.
* **co-isolated contaminants** — lipoproteins, soluble albumin and organelle
  proteins (MISEV2023 negative markers) are quantified by
  :func:`contaminant_score` as a preparation-purity metric.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..._registry import register_function
from .io import refresh_ev_metrics


# ---------------------------------------------------------------------------
# MISEV2023 contaminant marker panels
# ---------------------------------------------------------------------------
#: Co-isolated non-vesicular contaminants (lipoproteins + soluble protein).
_LIPOPROTEIN_MARKERS = ("APOA1", "APOA2", "APOB", "APOB100", "APOE", "APOC3")
_ALBUMIN_MARKERS = ("ALB", "ALBUMIN", "SERUM ALBUMIN")
#: Intracellular / organelle contaminants — should be absent from pure EVs.
_ORGANELLE_MARKERS = (
    "CANX", "CALNEXIN",          # endoplasmic reticulum
    "GOLGA2", "GM130",            # Golgi
    "CYCS", "CYTOCHROME C",       # mitochondria (apoptosis)
    "TOMM20", "TOM20",            # mitochondria
    "HSPA5", "GRP78", "BIP",      # ER chaperone
)


def _as_dense(X):
    """Return ``X`` as a dense float ndarray."""
    import scipy.sparse as sp

    if sp.issparse(X):
        return np.asarray(X.todense(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _match_markers(var_names: Sequence[str], panel: Sequence[str]) -> list[str]:
    """Case-insensitive match of a marker panel against ``var_names``."""
    upper = {str(v).upper(): str(v) for v in var_names}
    found: list[str] = []
    for m in panel:
        key = str(m).upper()
        if key in upper and upper[key] not in found:
            found.append(upper[key])
    return found


def _ev_uns(adata) -> dict:
    """Return ``adata.uns['ev']`` (a tolerant empty dict if absent)."""
    ev = adata.uns.get("ev", {})
    return dict(ev) if isinstance(ev, dict) else {}


# ---------------------------------------------------------------------------
# Core QC
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "qc", "ev_qc", "quality_control", "EV质控", "单囊泡质控",
    ],
    category="ev",
    description=(
        "Single-EV quality control: removes EVs with too few detected "
        "proteins (membrane fragments / noise), caps or removes EVs with "
        "implausibly high total signal (doublets / barcode collisions / "
        "swarm) and drops proteins detected in too few EVs. Writes per-EV and "
        "per-protein QC metrics into obs/var and returns the filtered AnnData."
    ),
    requires={"layers": ["counts"]},
    produces={
        "obs": ["total_signal", "n_proteins", "qc_pass"],
        "var": ["n_ev_detected", "frac_ev_detected"],
    },
    examples=[
        "adata = ov.single.ev.qc(adata, min_proteins=2, min_ev_frac=0.01)",
        "adata = ov.single.ev.qc(adata, max_total_signal='auto')",
    ],
    related=["single.ev.detect_doublets", "single.ev.contaminant_score"],
)
def qc(
    adata,
    *,
    min_proteins: int = 2,
    min_total_signal: float = 0.0,
    max_total_signal=None,
    max_signal_mad: float = 5.0,
    min_ev_count: int = 3,
    min_ev_frac: float = 0.0,
    cap_high_signal: bool = False,
    inplace: bool = False,
):
    """Run single-EV quality control.

    Parameters
    ----------
    adata
        Single-EV :class:`~anndata.AnnData` in the ``ov.single.ev`` schema.
    min_proteins
        Minimum number of detected (non-zero) proteins an EV must have;
        EVs below this are dropped as fragments / noise.
    min_total_signal
        Minimum per-EV total signal (row sum). EVs below this are dropped.
    max_total_signal
        Upper bound on per-EV total signal. ``None`` derives the cut from
        ``max_signal_mad``; a float sets it explicitly; the string ``'auto'``
        is equivalent to ``None``.
    max_signal_mad
        When ``max_total_signal`` is not given, the high-signal cut is
        ``median + max_signal_mad * MAD`` of ``total_signal`` (robust
        doublet / swarm threshold).
    min_ev_count
        Minimum number of EVs a protein must be detected in to be kept.
    min_ev_frac
        Minimum fraction of EVs a protein must be detected in to be kept
        (combined with ``min_ev_count`` — the stricter bound wins).
    cap_high_signal
        If ``True``, EVs above the high-signal cut are *kept* but have their
        ``X`` row down-scaled to the cut (signal capping); if ``False``
        (default) they are removed.
    inplace
        If ``True`` filter ``adata`` in place where possible; otherwise a
        filtered copy is returned.

    Returns
    -------
    :class:`~anndata.AnnData`
        QC-filtered AnnData. ``obs`` gains ``qc_pass`` (bool) and refreshed
        ``total_signal`` / ``n_proteins``; ``var`` gains ``n_ev_detected``
        and ``frac_ev_detected``.
    """
    adata = adata if inplace else adata.copy()
    refresh_ev_metrics(adata)

    n_ev = adata.n_obs
    total = adata.obs["total_signal"].to_numpy(dtype=np.float64)
    n_prot = adata.obs["n_proteins"].to_numpy(dtype=np.int64)

    # high-signal (doublet / swarm) threshold
    if max_total_signal is None or max_total_signal == "auto":
        med = float(np.median(total))
        mad = float(np.median(np.abs(total - med))) or float(total.std() or 1.0)
        high_cut = med + max_signal_mad * 1.4826 * mad
    else:
        high_cut = float(max_total_signal)

    keep_ev = (
        (n_prot >= int(min_proteins))
        & (total >= float(min_total_signal))
    )
    high_mask = total > high_cut
    if cap_high_signal:
        # down-scale offending rows to the cut, keep the EV
        X = _as_dense(adata.X)
        for i in np.where(high_mask & keep_ev)[0]:
            row_sum = X[i].sum()
            if row_sum > 0:
                X[i] *= high_cut / row_sum
        adata.X = X
    else:
        keep_ev = keep_ev & (~high_mask)

    adata.obs["qc_pass"] = keep_ev

    # per-protein detection
    Xd = _as_dense(adata.X)
    n_ev_detected = (Xd > 0).sum(axis=0).astype(np.int64)
    frac_detected = n_ev_detected / max(1, n_ev)
    adata.var["n_ev_detected"] = n_ev_detected
    adata.var["frac_ev_detected"] = frac_detected
    keep_var = (
        (n_ev_detected >= int(min_ev_count))
        & (frac_detected >= float(min_ev_frac))
    )

    n_drop_ev = int((~keep_ev).sum())
    n_drop_var = int((~keep_var).sum())
    out = adata[keep_ev, keep_var].copy()
    refresh_ev_metrics(out)
    out.var["n_ev_detected"] = (_as_dense(out.X) > 0).sum(axis=0).astype(
        np.int64
    )
    out.var["frac_ev_detected"] = out.var["n_ev_detected"] / max(1, out.n_obs)

    ev_uns = _ev_uns(out)
    ev_uns["qc"] = {
        "n_ev_in": int(n_ev),
        "n_ev_out": int(out.n_obs),
        "n_ev_removed": n_drop_ev,
        "n_proteins_in": int(adata.n_vars),
        "n_proteins_out": int(out.n_vars),
        "n_proteins_removed": n_drop_var,
        "high_signal_cut": high_cut,
        "n_high_signal": int(high_mask.sum()),
        "min_proteins": int(min_proteins),
        "cap_high_signal": bool(cap_high_signal),
    }
    out.uns["ev"] = ev_uns
    return out


# ---------------------------------------------------------------------------
# Isotype / blank background subtraction
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "subtract_isotype", "isotype_control", "background_subtract",
        "同型对照扣除", "本底扣除",
    ],
    category="ev",
    description=(
        "Isotype / IgG-control / buffer-blank background subtraction for "
        "single-EV data. Estimates non-specific binding from control markers "
        "or control EVs and subtracts it from X (control-based positivity "
        "calling); optionally writes a boolean obs/var positivity layer."
    ),
    requires={"layers": ["counts"]},
    produces={"layers": ["bg_subtracted"]},
    examples=[
        "adata = ov.single.ev.subtract_isotype(adata, isotype_markers=['IgG1'])",
        "adata = ov.single.ev.subtract_isotype(adata, blank_obs_key='is_blank')",
    ],
    related=["single.ev.qc", "single.ev.contaminant_score"],
)
def subtract_isotype(
    adata,
    *,
    isotype_markers: Optional[Sequence[str]] = None,
    blank_obs_key: Optional[str] = None,
    blank_value=True,
    quantile: float = 0.95,
    n_mad: float = 3.0,
    clip_negative: bool = True,
    add_positivity: bool = True,
    inplace: bool = False,
):
    """Subtract isotype / blank background and call positivity.

    Two background sources are supported:

    * **isotype markers** — one or more antibody columns that report
      non-specific binding (an IgG / isotype control target). Their per-EV
      signal defines a background level subtracted from every protein.
    * **blank EVs** — control EVs (buffer / no-antibody droplets) flagged in
      ``obs``; their per-protein distribution defines a per-protein
      background threshold.

    Parameters
    ----------
    adata
        Single-EV :class:`~anndata.AnnData`.
    isotype_markers
        Column names of isotype / IgG-control markers in ``var``. If given,
        the per-EV mean of these columns is the background level.
    blank_obs_key
        ``obs`` column flagging blank / buffer control EVs. If given, the
        per-protein ``quantile`` of the blank EVs is the background level.
    blank_value
        Value in ``obs[blank_obs_key]`` that marks a blank EV.
    quantile
        Quantile of the blank-EV signal used as the per-protein background
        when ``blank_obs_key`` is given.
    n_mad
        When neither control is supplied, the background is the per-protein
        ``median + n_mad * MAD`` (robust noise floor).
    clip_negative
        Clip post-subtraction negative values to 0.
    add_positivity
        If ``True`` store a boolean ``layers['positive']`` (signal above
        background).
    inplace
        Filter / annotate in place if ``True``.

    Returns
    -------
    :class:`~anndata.AnnData`
        AnnData with ``layers['bg_subtracted']`` (and ``layers['positive']``
        if requested); ``X`` is replaced by the background-subtracted matrix.
    """
    adata = adata if inplace else adata.copy()
    X = _as_dense(adata.X)
    n_ev, n_var = X.shape

    if isotype_markers:
        cols = _match_markers(adata.var_names, isotype_markers)
        if not cols:
            raise ValueError(
                f"none of isotype_markers={list(isotype_markers)} found in var."
            )
        idx = [adata.var_names.get_loc(c) for c in cols]
        # per-EV background level broadcast over all proteins
        bg = X[:, idx].mean(axis=1, keepdims=True)
        background = np.repeat(bg, n_var, axis=1)
        source = f"isotype:{cols}"
    elif blank_obs_key is not None:
        if blank_obs_key not in adata.obs.columns:
            raise KeyError(f"obs[{blank_obs_key!r}] not found.")
        blank_mask = (adata.obs[blank_obs_key].to_numpy() == blank_value)
        if blank_mask.sum() == 0:
            raise ValueError(
                f"no blank EVs (obs[{blank_obs_key!r}] == {blank_value!r})."
            )
        per_prot = np.quantile(X[blank_mask], quantile, axis=0)
        background = np.repeat(per_prot[None, :], n_ev, axis=0)
        source = f"blank:{blank_obs_key}"
    else:
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        per_prot = med + n_mad * 1.4826 * mad
        background = np.repeat(per_prot[None, :], n_ev, axis=0)
        source = "robust-noise-floor"

    subtracted = X - background
    if add_positivity:
        adata.layers["positive"] = (subtracted > 0).astype(np.float32)
    if clip_negative:
        subtracted = np.clip(subtracted, 0.0, None)

    adata.layers["bg_subtracted"] = subtracted.astype(np.float32)
    adata.X = subtracted.astype(np.float32)
    refresh_ev_metrics(adata)

    ev_uns = _ev_uns(adata)
    ev_uns["isotype_subtraction"] = {
        "source": source,
        "mean_background": float(np.mean(background)),
        "clip_negative": bool(clip_negative),
    }
    adata.uns["ev"] = ev_uns
    return adata


# ---------------------------------------------------------------------------
# Contaminant scoring (MISEV2023 negative markers)
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "contaminant_score", "purity_score", "misev_contaminant",
        "污染评分", "纯度评分",
    ],
    category="ev",
    description=(
        "Score each EV and the whole preparation for MISEV2023 co-isolated "
        "contaminant markers — lipoproteins (ApoA1/ApoB/ApoE), soluble "
        "albumin (ALB) and organelle contaminants (calnexin CANX, GM130/"
        "GOLGA2, cytochrome c CYCS, TOMM20). Writes per-EV contaminant scores "
        "into obs and a preparation-level purity summary into uns['ev']."
    ),
    examples=[
        "adata = ov.single.ev.contaminant_score(adata)",
        "score = adata.uns['ev']['contaminant']['purity']",
    ],
    related=["single.ev.qc", "single.ev.subtract_isotype"],
)
def contaminant_score(
    adata,
    *,
    lipoprotein_markers: Optional[Sequence[str]] = None,
    albumin_markers: Optional[Sequence[str]] = None,
    organelle_markers: Optional[Sequence[str]] = None,
    flag_quantile: float = 0.9,
    inplace: bool = False,
):
    """Score MISEV2023 co-isolated / organelle contamination.

    Parameters
    ----------
    adata
        Single-EV :class:`~anndata.AnnData`.
    lipoprotein_markers, albumin_markers, organelle_markers
        Override the built-in MISEV2023 negative-marker panels. ``None`` uses
        the defaults (``APOA1/APOB/APOE...``, ``ALB``, ``CANX/GOLGA2/CYCS/
        TOMM20...``). Matching against ``var_names`` is case-insensitive.
    flag_quantile
        EVs whose total contaminant signal exceeds this quantile are flagged
        ``obs['contaminant_flag'] == True``.
    inplace
        Annotate ``adata`` in place if ``True``.

    Returns
    -------
    :class:`~anndata.AnnData`
        ``obs`` gains ``lipoprotein_score``, ``albumin_score``,
        ``organelle_score``, ``contaminant_score`` (sum) and a boolean
        ``contaminant_flag``. ``uns['ev']['contaminant']`` holds the
        preparation-level summary, including a ``purity`` value in
        ``[0, 1]`` (1 = no contaminant signal).
    """
    adata = adata if inplace else adata.copy()
    X = _as_dense(adata.X)
    total = X.sum(axis=1)
    total_safe = np.where(total > 0, total, 1.0)

    panels = {
        "lipoprotein": lipoprotein_markers or _LIPOPROTEIN_MARKERS,
        "albumin": albumin_markers or _ALBUMIN_MARKERS,
        "organelle": organelle_markers or _ORGANELLE_MARKERS,
    }
    matched: dict[str, list[str]] = {}
    scores: dict[str, np.ndarray] = {}
    for name, panel in panels.items():
        cols = _match_markers(adata.var_names, panel)
        matched[name] = cols
        if cols:
            idx = [adata.var_names.get_loc(c) for c in cols]
            # contaminant signal as a fraction of each EV's total signal
            scores[name] = X[:, idx].sum(axis=1) / total_safe
        else:
            scores[name] = np.zeros(adata.n_obs)
        adata.obs[f"{name}_score"] = scores[name]

    contam = scores["lipoprotein"] + scores["albumin"] + scores["organelle"]
    adata.obs["contaminant_score"] = contam
    cut = float(np.quantile(contam, flag_quantile)) if contam.size else 0.0
    adata.obs["contaminant_flag"] = contam > cut

    purity = float(np.clip(1.0 - np.mean(contam), 0.0, 1.0))
    ev_uns = _ev_uns(adata)
    ev_uns["contaminant"] = {
        "markers_found": matched,
        "mean_lipoprotein": float(np.mean(scores["lipoprotein"])),
        "mean_albumin": float(np.mean(scores["albumin"])),
        "mean_organelle": float(np.mean(scores["organelle"])),
        "mean_contaminant": float(np.mean(contam)),
        "purity": purity,
        "flag_cut": cut,
        "n_flagged": int(adata.obs["contaminant_flag"].sum()),
    }
    adata.uns["ev"] = ev_uns
    return adata


# ---------------------------------------------------------------------------
# Doublet / barcode-collision detection
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "detect_doublets", "ev_doublets", "barcode_collision",
        "检测双联体", "条形码碰撞检测",
    ],
    category="ev",
    description=(
        "Flag multi-vesicle / barcode-collision artifacts in single-EV data. "
        "An EV is flagged as a putative doublet when its total signal AND its "
        "number of detected proteins are both robustly elevated above the "
        "population (median + n_mad * MAD), the signature of two-or-more "
        "vesicles read as one tag. Writes obs['ev_doublet'] and a score."
    ),
    produces={"obs": ["ev_doublet", "doublet_score"]},
    examples=[
        "adata = ov.single.ev.detect_doublets(adata, n_mad=4.0)",
        "singlets = adata[~adata.obs['ev_doublet']].copy()",
    ],
    related=["single.ev.qc", "single.ev.contaminant_score"],
)
def detect_doublets(
    adata,
    *,
    n_mad: float = 4.0,
    expected_doublet_rate: Optional[float] = None,
    require_both: bool = True,
    inplace: bool = False,
):
    """Flag multi-vesicle / barcode-collision artifacts.

    Parameters
    ----------
    adata
        Single-EV :class:`~anndata.AnnData`.
    n_mad
        An EV is elevated on a metric when it exceeds
        ``median + n_mad * 1.4826 * MAD`` of that metric.
    expected_doublet_rate
        If given (e.g. ``0.05``), the threshold is instead set so the top
        ``expected_doublet_rate`` fraction of EVs by ``doublet_score`` is
        flagged — useful when a load-based collision rate is known.
    require_both
        If ``True`` (default) an EV must be elevated on *both* total signal
        and detected-protein count to be flagged (a stricter, more specific
        rule); if ``False`` either metric suffices.
    inplace
        Annotate ``adata`` in place if ``True``.

    Returns
    -------
    :class:`~anndata.AnnData`
        ``obs`` gains ``doublet_score`` (a 0-1 combined rank score) and the
        boolean ``ev_doublet``. ``uns['ev']['doublets']`` records the rule
        and the flagged count.
    """
    adata = adata if inplace else adata.copy()
    refresh_ev_metrics(adata)

    total = adata.obs["total_signal"].to_numpy(dtype=np.float64)
    n_prot = adata.obs["n_proteins"].to_numpy(dtype=np.float64)

    def _z(v):
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        scale = 1.4826 * mad if mad > 0 else (float(v.std()) or 1.0)
        return (v - med) / scale

    z_total = _z(total)
    z_prot = _z(n_prot)
    # combined doublet score — rank-normalised mean of the two robust z-scores
    combined = (z_total + z_prot) / 2.0
    order = combined.argsort().argsort()
    score = order / max(1, len(order) - 1)
    adata.obs["doublet_score"] = score

    if expected_doublet_rate is not None:
        rate = float(np.clip(expected_doublet_rate, 0.0, 1.0))
        cut = np.quantile(score, 1.0 - rate) if score.size else 1.0
        flag = score >= cut
        rule = f"top {rate:.3f} by doublet_score"
    else:
        hi_total = z_total > n_mad
        hi_prot = z_prot > n_mad
        flag = (hi_total & hi_prot) if require_both else (hi_total | hi_prot)
        rule = (
            f"{'both' if require_both else 'either'} metric > "
            f"{n_mad} MAD"
        )

    adata.obs["ev_doublet"] = flag

    ev_uns = _ev_uns(adata)
    ev_uns["doublets"] = {
        "rule": rule,
        "n_doublets": int(np.sum(flag)),
        "doublet_fraction": float(np.mean(flag)) if flag.size else 0.0,
        "require_both": bool(require_both),
    }
    adata.uns["ev"] = ev_uns
    return adata
