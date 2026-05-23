r"""MS-specific quality control.

Untargeted LC-MS data without proper QC gives unreliable downstream
results — this module implements the three filters that every serious
metabolomics pipeline does:

1. **CV%-based feature filter** (``cv_filter``) — drop features whose
   technical variation (measured on pooled QC samples) exceeds a
   threshold. 30% is the community standard; 20% for very stringent
   analysis.
2. **Signal-drift correction** (``drift_correct``) — LC-MS response
   drifts over long injection sequences; we fit a LOESS curve against
   injection order using the QC samples and divide it out.
3. **Blank filtering** (``blank_filter``) — features whose sample
   intensity isn't > ``ratio`` times the blank intensity are likely
   contamination peaks and get dropped.

Each function takes an AnnData, operates on ``.X``, and returns a
*new* AnnData (subset or transformed). Nothing is done in-place, so
the caller can chain these without surprises.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats

from .._registry import register_function


@register_function(
    aliases=[
        'cv_filter',
        'CV过滤',
        'qc_cv_filter',
    ],
    category='metabolomics',
    description='Drop metabolite features with coefficient-of-variation above a threshold. Default computes CV across pooled QC samples (metabolomics-specific); pass `across="all"` to compute across every sample (for NMR / non-QC-pool studies).',
    examples=[
        "ov.metabol.cv_filter(adata, qc_mask='is_qc', cv_threshold=0.30)",
        "ov.metabol.cv_filter(adata, across='all', cv_threshold=1.5)",
    ],
    related=[
        'metabol.drift_correct',
        'metabol.blank_filter',
    ],
)
def cv_filter(
    adata: AnnData,
    *,
    qc_mask: "str | np.ndarray | None" = None,
    cv_threshold: float = 0.30,
    across: str = "qc",
) -> AnnData:
    """Drop features with coefficient-of-variation above ``cv_threshold``.

    Parameters
    ----------
    qc_mask
        Either the name of a boolean column in ``adata.obs`` (True = QC
        pool sample), or a boolean array of length ``adata.n_obs``.
        Required when ``across='qc'`` (the default). Ignored when
        ``across='all'``.
    cv_threshold
        Features with ``std/mean > cv_threshold`` are dropped. Default
        0.30 is the community standard for untargeted LC-MS with QC
        pools; for ``across='all'`` on biological samples a much
        higher threshold (1–2) is typical because biology itself
        adds variance.
    across
        - ``"qc"`` (default) — compute CV across QC-pool samples only
          (the MetaboAnalyst convention for LC-MS); requires
          ``qc_mask``.
        - ``"all"`` — compute CV across every sample. Use for NMR
          datasets or any workflow without pooled QC injections.
          MTBLS1-style studies fall in this bucket.

    Returns
    -------
    AnnData
        Subset to features that passed. ``adata.var['qc_cv']`` carries
        the CV values on the kept features.
    """
    if across == "qc":
        if qc_mask is None:
            raise ValueError(
                "cv_filter(across='qc') requires a qc_mask (bool column or "
                "array marking QC-pool samples). For studies without QC "
                "pools pass `across='all'`."
            )
        mask = _resolve_sample_mask(adata, qc_mask)
        X_sub = adata.X[mask]
        if X_sub.shape[0] < 3:
            raise ValueError(
                f"Need ≥3 QC samples for a meaningful CV filter, got "
                f"{X_sub.shape[0]}"
            )
    elif across == "all":
        X_sub = np.asarray(adata.X)
        if X_sub.shape[0] < 3:
            raise ValueError(
                f"Need ≥3 samples for a meaningful CV filter, got "
                f"{X_sub.shape[0]}"
            )
    else:
        raise ValueError(
            f"across must be 'qc' or 'all', got {across!r}"
        )
    mu = np.nanmean(X_sub, axis=0)
    sd = np.nanstd(X_sub, axis=0, ddof=1)
    # Avoid divide-by-zero on all-zero features
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mu > 0, sd / mu, np.inf)
    keep = cv <= cv_threshold
    out = adata[:, keep].copy()
    out.var["qc_cv"] = cv[keep]
    return out


@register_function(
    aliases=[
        'drift_correct',
        'LOESS漂移校正',
    ],
    category='metabolomics',
    description='Correct systematic LC-MS intensity drift across injection order via LOESS regression on pooled QC samples.',
    examples=[
        "ov.metabol.drift_correct(adata, injection_order='run_order', qc_mask='is_qc')",
    ],
    related=[
        'metabol.cv_filter',
        'metabol.blank_filter',
    ],
)
def drift_correct(
    adata: AnnData,
    *,
    injection_order: str | np.ndarray,
    qc_mask: str | np.ndarray,
    frac: float = 0.5,
) -> AnnData:
    """Correct systematic signal drift using LOESS regression on QC samples.

    Fits ``log1p(intensity) ~ injection_order`` per feature on QC
    samples only, then divides the raw intensities in both QC and real
    samples by the fitted drift curve.

    Parameters
    ----------
    injection_order
        Name of a numeric column in ``adata.obs`` (or a 1-D array) that
        gives the run order of each sample.
    qc_mask
        Bool column/array marking QC pool samples.
    frac
        LOESS smoothing fraction (``statsmodels.lowess`` ``frac``
        argument). 0.5 is robust for runs of <500 samples; tighten to
        0.3 for larger runs.

    Returns
    -------
    AnnData
        Intensity matrix corrected in place on a copy; ``uns['qc_drift']``
        records the per-feature LOESS fit for reproducibility.
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "drift_correct needs statsmodels; `pip install statsmodels`."
        ) from exc

    order = _resolve_sample_array(adata, injection_order).astype(float)
    qmask = _resolve_sample_mask(adata, qc_mask)

    # Warn if real samples fall outside the QC injection range — np.interp
    # clamps at the boundaries rather than extrapolating, so the drift
    # correction for those edge samples is effectively ignored.
    qc_min, qc_max = order[qmask].min(), order[qmask].max()
    outside = ((order < qc_min) | (order > qc_max)) & ~qmask
    if outside.any():
        import warnings
        warnings.warn(
            f"{int(outside.sum())} real sample(s) were injected outside the "
            f"QC range [{qc_min:g}, {qc_max:g}]. Their drift correction uses "
            "the nearest-edge LOESS value (np.interp clamps; no extrapolation). "
            "Consider bracketing your run with QC pools.",
            UserWarning, stacklevel=2,
        )

    out = adata.copy()
    X = out.X.astype(np.float64, copy=True)
    fits = np.zeros_like(X)
    for j in range(X.shape[1]):
        y_log = np.log1p(np.where(X[qmask, j] > 0, X[qmask, j], 0))
        smooth = lowess(
            endog=y_log, exog=order[qmask], frac=frac, it=0, return_sorted=False
        )
        # Evaluate at all sample injection orders
        fit_all = np.interp(order, np.sort(order[qmask]),
                            smooth[np.argsort(order[qmask])])
        # Divide by drift curve (centered at QC median so magnitudes stay comparable)
        ref = np.median(fit_all[qmask])
        fits[:, j] = np.expm1(fit_all) / max(np.expm1(ref), 1e-6)
    X_corrected = np.where(fits > 0, X / fits, X)
    out.X = X_corrected
    out.uns["qc_drift"] = {"frac": frac, "ref": "median_qc"}
    return out


@register_function(
    aliases=[
        'blank_filter',
        '空白过滤',
    ],
    category='metabolomics',
    description="Drop features whose sample-mean intensity isn't at least ratio× the blank-mean intensity (contamination filter).",
    examples=[
        "ov.metabol.blank_filter(adata, blank_mask='is_blank', ratio=3.0)",
    ],
    related=[
        'metabol.cv_filter',
        'metabol.drift_correct',
    ],
)
def blank_filter(
    adata: AnnData,
    *,
    blank_mask: str | np.ndarray,
    ratio: float = 3.0,
) -> AnnData:
    """Drop features whose sample-mean intensity isn't at least ``ratio``\ ×
    the blank-mean intensity.

    Parameters
    ----------
    blank_mask
        Column name (bool) or bool array marking blank / extraction-control
        samples in ``adata.obs``.
    ratio
        Features with ``mean(sample)/mean(blank) < ratio`` are dropped.
        3× is the community default; 5× for more stringent filtering.
    """
    bmask = _resolve_sample_mask(adata, blank_mask)
    if bmask.sum() == 0:
        raise ValueError("No blank samples found — pass a valid blank_mask.")
    real = ~bmask
    sample_mean = np.nanmean(adata.X[real], axis=0)
    blank_mean = np.nanmean(adata.X[bmask], axis=0)
    # Features never detected in blanks pass automatically.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(blank_mean > 0, sample_mean / blank_mean, np.inf)
    keep = r >= ratio
    out = adata[:, keep].copy()
    out.var["blank_ratio"] = r[keep]
    return out


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _resolve_sample_mask(adata: AnnData, mask_or_col: str | np.ndarray) -> np.ndarray:
    if isinstance(mask_or_col, str):
        if mask_or_col not in adata.obs.columns:
            raise KeyError(f"adata.obs has no column {mask_or_col!r}")
        return adata.obs[mask_or_col].astype(bool).to_numpy()
    arr = np.asarray(mask_or_col, dtype=bool)
    if arr.shape != (adata.n_obs,):
        raise ValueError(
            f"mask shape {arr.shape} does not match n_obs={adata.n_obs}"
        )
    return arr


def _resolve_sample_array(adata: AnnData, col_or_arr: str | np.ndarray) -> np.ndarray:
    if isinstance(col_or_arr, str):
        if col_or_arr not in adata.obs.columns:
            raise KeyError(f"adata.obs has no column {col_or_arr!r}")
        return adata.obs[col_or_arr].to_numpy()
    arr = np.asarray(col_or_arr)
    if arr.shape != (adata.n_obs,):
        raise ValueError(
            f"array shape {arr.shape} does not match n_obs={adata.n_obs}"
        )
    return arr


# ---------------------------------------------------------------------------
# Sample-level outlier detection — Hotelling T² + DModX on PCA
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        'sample_qc',
        'hotelling_t2',
        'dmodx',
        '离群样本',
    ],
    category='metabolomics',
    description='Per-sample outlier flags via Hotelling T-squared (inside the PCA model) plus DModX (distance to the PCA residual plane). The standard SIMCA-style sample-level diagnostic for metabolomics runs.',
    examples=[
        "ov.metabol.sample_qc(adata, n_components=2)",
        "ov.metabol.sample_qc(adata, n_components=3, alpha=0.99)",
    ],
    related=[
        'metabol.cv_filter',
        'metabol.serrf',
        'metabol.plsda',
    ],
)
def sample_qc(
    adata: AnnData,
    *,
    n_components: int = 2,
    alpha: float = 0.95,
    center: bool = True,
    scale: bool = True,
    layer: str | None = None,
) -> "pd.DataFrame":
    """Hotelling T-squared + DModX sample-level outlier detection.

    Fits PCA on ``adata.X`` (after mean-centre + unit-variance scale by
    default) and returns per-sample diagnostics.

    - **Hotelling T-squared** ``= Σ (t_a / s_a)^2`` — quadratic-form
      distance from the sample to the model origin inside the PC
      subspace. Critical value at level ``alpha`` is the
      ``(1-alpha)``-quantile of a scaled F distribution (Hotelling
      1947); flagged samples are deep *within* the model space.
    - **DModX** ``= sqrt(||residual||² / (p - A))`` — standardised
      distance from the sample to the residual subspace. Flagged
      samples are *outside* the model space. DModX critical value is
      based on an F approximation (Eriksson 2013, Ch. 7).

    A sample flagged by either metric should be inspected before
    downstream stats — T-squared catches unusual profiles that still
    "look like" the training set; DModX catches novel profiles that
    don't fit the PCA subspace at all.

    Parameters
    ----------
    n_components
        Number of PCs to retain. Default 2 — enough for a 2-D score
        plot, too few for detecting outliers in high-dimensional
        data. Try 3–5 for real studies.
    alpha
        Significance level for flagging. Default 0.95.
    center, scale
        Pre-processing. Default: mean-centre + scale to unit variance
        (matches SIMCA / MetaboAnalyst convention).
    layer
        AnnData layer (default ``None`` → ``adata.X``).

    Returns
    -------
    pd.DataFrame
        Indexed by sample name, columns:
        ``T2`` (Hotelling T-squared), ``DModX``,
        ``T2_crit`` / ``DModX_crit`` (critical values at ``alpha``),
        ``T2_flag`` / ``DModX_flag`` (bools), ``is_outlier``
        (flagged by either).

        Also attaches ``attrs['variance_explained']`` (array of length
        ``n_components``) and ``attrs['n_components']``.
    """
    from sklearn.decomposition import PCA

    Xraw = adata.X if layer is None else adata.layers[layer]
    X = np.asarray(Xraw, dtype=np.float64)
    n, p = X.shape

    # Impute NaNs with column median so PCA doesn't choke
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    nan_idx = np.isnan(X)
    if nan_idx.any():
        X = np.where(nan_idx, np.broadcast_to(col_med, X.shape), X)

    if center:
        X = X - X.mean(axis=0, keepdims=True)
    if scale:
        sd = X.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, 1.0)
        X = X / sd

    A = min(n_components, min(X.shape) - 1)
    if A < 1:
        raise ValueError(
            f"need at least one PC, got n_components={n_components} "
            f"on data shape {X.shape}"
        )
    pca = PCA(n_components=A)
    T = pca.fit_transform(X)          # (n, A)
    P = pca.components_.T             # (p, A)
    var_explained = pca.explained_variance_ratio_

    # Hotelling T² — sum of squared standardised scores
    score_var = T.var(axis=0, ddof=1)
    score_var = np.where(score_var > 0, score_var, 1.0)
    t2 = ((T ** 2) / score_var).sum(axis=1)
    # Critical value (Hotelling 1947, F-scaled)
    f_crit = stats.f.ppf(alpha, A, max(n - A, 1))
    t2_crit = A * (n - 1) / max(n - A, 1) * f_crit

    # DModX — RMSE of residual vector per sample
    X_hat = T @ P.T
    residuals = X - X_hat
    if p > A:
        dmodx = np.sqrt((residuals ** 2).sum(axis=1) / (p - A))
    else:
        dmodx = np.zeros(n)
    # Critical value — F approximation (Eriksson 2013)
    # s0 = total residual variance (per-row), s_crit = s0 * sqrt(F_alpha)
    s0 = dmodx.mean() if dmodx.size else 1.0
    dmodx_crit = s0 * np.sqrt(stats.f.ppf(alpha, max(p - A, 1),
                                           max((n - A - 1) * (p - A), 1)))

    t2_flag = t2 > t2_crit
    dmodx_flag = dmodx > dmodx_crit
    out = pd.DataFrame({
        "T2": t2,
        "DModX": dmodx,
        "T2_crit": float(t2_crit),
        "DModX_crit": float(dmodx_crit),
        "T2_flag": t2_flag,
        "DModX_flag": dmodx_flag,
        "is_outlier": t2_flag | dmodx_flag,
    }, index=adata.obs_names.copy())
    out.attrs["n_components"] = A
    out.attrs["variance_explained"] = var_explained.tolist()
    out.attrs["alpha"] = alpha
    return out
