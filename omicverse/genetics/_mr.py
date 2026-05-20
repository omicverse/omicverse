"""Mendelian randomization for ``ov.genetics`` — TwoSampleMR backend.

Wraps the standalone :mod:`pytwosamplemr` package. :func:`mendelian_randomization`
is a single dispatcher over the full library of two-sample MR
estimators (IVW, MR-Egger, weighted median, mode-based, maximum
likelihood, dIVW, contamination mixture, MR-Lasso, MR-cML).
:func:`harmonize`, :func:`mr_steiger`, :func:`mr_heterogeneity` and
:func:`mr_pleiotropy` cover data preparation and sensitivity analyses.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._registry import register_function


def _require_mr():
    """Import :mod:`pytwosamplemr` with a friendly error if it is missing."""
    try:
        import pytwosamplemr  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.genetics.mendelian_randomization requires the pytwosamplemr "
            "backend: `pip install pytwosamplemr` "
            "(or `pip install omicverse[genetics]`)."
        ) from exc
    return pytwosamplemr


_MR_METHODS = {
    "ivw": "mr_ivw",
    "egger": "mr_egger",
    "median": "mr_median",
    "mode": "mr_mbe",
    "mbe": "mr_mbe",
    "maxlik": "mr_maxlik",
    "divw": "mr_divw",
    "conmix": "mr_conmix",
    "lasso": "mr_lasso",
    "cml": "mr_cml",
}


def _coerce_input(mr, mr_input_or_dataframes, bx, bxse, by, byse, snps):
    """Build an :class:`pytwosamplemr.MRInput` from the flexible first arg."""
    obj = mr_input_or_dataframes
    # Already an MRInput.
    if obj is not None and obj.__class__.__name__ == "MRInput":
        return obj
    # Harmonised TwoSampleMR-style DataFrame.
    if isinstance(obj, pd.DataFrame):
        cols = {c.lower(): c for c in obj.columns}
        def _col(*cands):
            for cand in cands:
                if cand in cols:
                    return obj[cols[cand]].to_numpy(dtype=float)
            raise KeyError(
                f"DataFrame is missing one of {cands}; pass an MRInput or "
                f"explicit bx/bxse/by/byse instead."
            )
        snp_col = next((obj[cols[c]].astype(str).tolist()
                        for c in ("snp", "snps") if c in cols), None)
        return mr.mr_input(
            bx=_col("beta.exposure", "bx", "beta_exposure"),
            bxse=_col("se.exposure", "bxse", "se_exposure"),
            by=_col("beta.outcome", "by", "beta_outcome"),
            byse=_col("se.outcome", "byse", "se_outcome"),
            snps=snp_col,
        )
    # Explicit arrays.
    if bx is not None and bxse is not None and by is not None and byse is not None:
        return mr.mr_input(bx=np.asarray(bx, dtype=float),
                           bxse=np.asarray(bxse, dtype=float),
                           by=np.asarray(by, dtype=float),
                           byse=np.asarray(byse, dtype=float),
                           snps=snps)
    raise ValueError(
        "Provide an MRInput, a harmonised DataFrame, or all four of "
        "bx / bxse / by / byse."
    )


@register_function(
    aliases=[
        "mendelian_randomization", "mr", "mendelian randomization",
        "孟德尔随机化", "MR分析", "因果推断",
    ],
    category="genetics",
    description=(
        "Two-sample Mendelian randomization to estimate the causal effect "
        "of an exposure on an outcome from GWAS summary statistics. "
        "``method`` selects the estimator: ``'ivw'`` (inverse-variance "
        "weighted, default), ``'egger'`` (MR-Egger, pleiotropy-robust), "
        "``'median'`` (weighted median), ``'mode'`` (mode-based estimate), "
        "``'maxlik'`` (maximum likelihood), ``'divw'`` (debiased IVW), "
        "``'conmix'`` (contamination mixture), ``'lasso'`` (MR-Lasso), "
        "``'cml'`` (MR-cML), or ``'all'`` to run every estimator and "
        "return a comparison table. The first argument may be an MRInput, "
        "a harmonised DataFrame, or explicit bx/bxse/by/byse arrays. "
        "Wraps the pytwosamplemr backend."
    ),
    examples=[
        "ov.genetics.mendelian_randomization(mr_input, method='ivw')",
        "ov.genetics.mendelian_randomization(mr_input, method='egger')",
        "ov.genetics.mendelian_randomization(mr_input, method='all')",
        "ov.genetics.mendelian_randomization(bx=bx, bxse=bxse, by=by, byse=byse)",
    ],
    related=["ov.genetics.harmonize", "ov.genetics.mr_steiger",
             "ov.genetics.mr_heterogeneity", "ov.genetics.mr_pleiotropy",
             "ov.genetics.mr_scatter"],
)
def mendelian_randomization(
    mr_input_or_dataframes=None,
    *,
    method: str = "ivw",
    bx: Optional[Sequence[float]] = None,
    bxse: Optional[Sequence[float]] = None,
    by: Optional[Sequence[float]] = None,
    byse: Optional[Sequence[float]] = None,
    snps: Optional[Sequence[str]] = None,
    n: Optional[int] = None,
    **kwargs,
):
    """Run two-sample Mendelian randomization.

    Parameters
    ----------
    mr_input_or_dataframes
        An :class:`pytwosamplemr.MRInput`, or a harmonised DataFrame with
        columns ``beta.exposure`` / ``se.exposure`` / ``beta.outcome`` /
        ``se.outcome`` (or the ``bx`` / ``bxse`` / ``by`` / ``byse``
        short forms).
    method
        Estimator selector — one of ``'ivw'``, ``'egger'``, ``'median'``,
        ``'mode'``, ``'maxlik'``, ``'divw'``, ``'conmix'``, ``'lasso'``,
        ``'cml'``, or ``'all'``.
    bx, bxse, by, byse
        Explicit SNP-exposure / SNP-outcome effect sizes and standard
        errors (an alternative to passing an MRInput / DataFrame).
    snps
        Optional SNP identifiers.
    n
        Sample size — required for ``method='cml'``.
    **kwargs
        Forwarded to the backend estimator.

    Returns
    -------
    object or pandas.DataFrame
        A single MR result object, or — for ``method='all'`` — a tidy
        DataFrame comparing every estimator.
    """
    mr = _require_mr()
    obj = _coerce_input(mr, mr_input_or_dataframes, bx, bxse, by, byse, snps)
    key = str(method).lower().strip()

    if key == "all":
        return mr.mr_allmethods(obj, **kwargs)
    if key not in _MR_METHODS:
        raise ValueError(
            f"method must be 'all' or one of {sorted(_MR_METHODS)}, "
            f"got {method!r}"
        )
    fn = getattr(mr, _MR_METHODS[key])
    if key == "cml":
        if n is None:
            raise ValueError("method='cml' requires the sample size n=.")
        return fn(obj, n=n, **kwargs)
    return fn(obj, **kwargs)


@register_function(
    aliases=["harmonize", "harmonise", "harmonise_data", "数据协调", "等位基因协调"],
    category="genetics",
    description=(
        "Harmonise SNP-exposure and SNP-outcome GWAS effects onto the "
        "same effect allele before Mendelian randomization — resolves "
        "strand and allele mismatches and flags ambiguous palindromic "
        "SNPs. Wraps :func:`pytwosamplemr.harmonise_data`."
    ),
    examples=[
        "ov.genetics.harmonize(exposure_df, outcome_df)",
        "ov.genetics.harmonize(exposure_df, outcome_df, action=2)",
    ],
    related=["ov.genetics.mendelian_randomization"],
)
def harmonize(
    exposure: pd.DataFrame,
    outcome: pd.DataFrame,
    *,
    tolerance: float = 0.08,
    action: int = 2,
) -> pd.DataFrame:
    """Harmonise exposure and outcome GWAS effects.

    Parameters
    ----------
    exposure, outcome
        DataFrames with columns ``SNP``, ``beta``, ``se``,
        ``effect_allele``, ``other_allele`` (``eaf`` optional).
    tolerance
        Max allele-frequency difference for palindromic-SNP resolution.
    action
        Allele-harmonisation strictness (``1`` / ``2`` / ``3``).

    Returns
    -------
    pandas.DataFrame
        Harmonised SNP table ready for :func:`mendelian_randomization`.
    """
    mr = _require_mr()
    return mr.harmonise_data(exposure, outcome,
                             tolerance=tolerance, action=action)


@register_function(
    aliases=["mr_steiger", "steiger", "directionality_test", "方向性检验"],
    category="genetics",
    description=(
        "MR-Steiger directionality test — checks that the instrument "
        "explains more variance in the exposure than in the outcome, "
        "i.e. that the assumed causal direction is correct. Wraps "
        ":func:`pytwosamplemr.mr_steiger`."
    ),
    examples=[
        "ov.genetics.mr_steiger(p_exp, p_out, n_exp, n_out)",
    ],
    related=["ov.genetics.mendelian_randomization"],
)
def mr_steiger(p_exp, p_out, n_exp, n_out, *, r_exp=None, r_out=None) -> dict:
    """Run the MR-Steiger directionality test.

    Parameters
    ----------
    p_exp, p_out
        SNP-exposure and SNP-outcome association p-values.
    n_exp, n_out
        Exposure / outcome GWAS sample sizes.
    r_exp, r_out
        Optional pre-computed SNP-trait correlations.

    Returns
    -------
    dict
        Steiger result — correlations, the directionality verdict and its
        p-value.
    """
    mr = _require_mr()
    return mr.mr_steiger(p_exp, p_out, n_exp, n_out, r_exp=r_exp, r_out=r_out)


@register_function(
    aliases=["mr_heterogeneity", "heterogeneity", "cochran_q", "异质性检验"],
    category="genetics",
    description=(
        "Cochran's Q heterogeneity test for a Mendelian-randomization "
        "analysis — excess heterogeneity across instruments flags "
        "pleiotropy or invalid instruments. Wraps "
        ":func:`pytwosamplemr.mr_heterogeneity`."
    ),
    examples=[
        "ov.genetics.mr_heterogeneity(mr_input)",
        "ov.genetics.mr_heterogeneity(mr_input, methods=('ivw', 'egger'))",
    ],
    related=["ov.genetics.mendelian_randomization", "ov.genetics.mr_pleiotropy"],
)
def mr_heterogeneity(
    dat,
    *,
    methods: Sequence[str] = ("ivw", "egger"),
    bx=None, bxse=None, by=None, byse=None, snps=None,
) -> pd.DataFrame:
    """Cochran's Q heterogeneity test.

    Parameters
    ----------
    dat
        An MRInput / harmonised DataFrame (or pass bx/bxse/by/byse).
    methods
        Which estimators to compute Q for.
    bx, bxse, by, byse, snps
        Explicit arrays, if ``dat`` is not given.

    Returns
    -------
    pandas.DataFrame
        Per-method Q statistic, degrees of freedom and p-value.
    """
    mr = _require_mr()
    obj = _coerce_input(mr, dat, bx, bxse, by, byse, snps)
    return mr.mr_heterogeneity(obj, methods=tuple(methods))


@register_function(
    aliases=["mr_pleiotropy", "pleiotropy", "egger_intercept", "多效性检验"],
    category="genetics",
    description=(
        "MR-Egger intercept test for directional (horizontal) pleiotropy "
        "— a non-zero intercept indicates the instruments violate the "
        "exclusion-restriction assumption. Wraps "
        ":func:`pytwosamplemr.mr_pleiotropy_test`."
    ),
    examples=[
        "ov.genetics.mr_pleiotropy(mr_input)",
    ],
    related=["ov.genetics.mendelian_randomization", "ov.genetics.mr_heterogeneity"],
)
def mr_pleiotropy(
    dat,
    *,
    bx=None, bxse=None, by=None, byse=None, snps=None,
) -> pd.DataFrame:
    """MR-Egger intercept (pleiotropy) test.

    Parameters
    ----------
    dat
        An MRInput / harmonised DataFrame (or pass bx/bxse/by/byse).
    bx, bxse, by, byse, snps
        Explicit arrays, if ``dat`` is not given.

    Returns
    -------
    pandas.DataFrame
        The MR-Egger intercept, its standard error and p-value.
    """
    mr = _require_mr()
    obj = _coerce_input(mr, dat, bx, bxse, by, byse, snps)
    return mr.mr_pleiotropy_test(obj)
