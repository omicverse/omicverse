"""Statistical fine-mapping for ``ov.genetics`` — SuSiE backend.

Wraps the standalone :mod:`pysusie` package (the Sum of Single Effects
model, Wang *et al.* 2020). :func:`finemap` dispatches between SuSiE on
individual-level data (``method='susie'``) and SuSiE-RSS on GWAS
summary statistics + an LD matrix (``method='susie_rss'``).
:func:`get_credible_sets` and :func:`get_pip` summarise a fitted model.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .._registry import register_function


def _require_susie():
    """Import :mod:`pysusie` with a friendly error if it is missing."""
    try:
        import pysusie  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.genetics.finemap requires the pysusie backend: "
            "`pip install pysusie` (or `pip install omicverse[genetics]`)."
        ) from exc
    return pysusie


@register_function(
    aliases=[
        "finemap", "fine_mapping", "susie", "susie_rss",
        "精细定位", "精细映射", "因果变异定位",
    ],
    category="genetics",
    description=(
        "Statistical fine-mapping with SuSiE (Wang 2020) to resolve which "
        "variants in an associated locus are causal. ``method`` selects the "
        "backend: ``'susie'`` fits the Sum of Single Effects model on "
        "individual-level genotype ``X`` and phenotype ``y``; "
        "``'susie_rss'`` (default) fits from GWAS summary statistics — a "
        "z-score vector ``z`` (or ``bhat`` / ``shat``) plus an LD "
        "correlation matrix ``R``. Returns a fitted SuSiE model carrying "
        "per-variant posterior inclusion probabilities and 95% credible "
        "sets. Wraps the pysusie backend."
    ),
    examples=[
        "ov.genetics.finemap(z=z, R=ld, n=5000, method='susie_rss')",
        "ov.genetics.finemap(X=geno, y=pheno, method='susie', L=10)",
        "fit = ov.genetics.finemap(z=z, R=ld, n=5000); ov.genetics.get_credible_sets(fit, R=ld)",
    ],
    related=["ov.genetics.colocalize", "ov.genetics.get_pip",
             "ov.genetics.get_credible_sets", "ov.genetics.finemap_plot"],
)
def finemap(
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    *,
    method: str = "susie_rss",
    z: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    n: Optional[int] = None,
    bhat: Optional[np.ndarray] = None,
    shat: Optional[np.ndarray] = None,
    L: int = 10,
    coverage: float = 0.95,
    min_abs_corr: float = 0.5,
    max_iter: int = 100,
    **kwargs,
):
    """Fine-map a locus with SuSiE.

    Parameters
    ----------
    X, y
        For ``method='susie'``: individual-level genotype matrix
        (``samples x SNPs``) and phenotype vector (``samples``).
    method
        ``'susie'`` (individual data) or ``'susie_rss'`` (summary stats).
    z
        For ``method='susie_rss'``: SNP z-score vector.
    R
        For ``method='susie_rss'``: SNP-SNP LD correlation matrix.
    n
        GWAS sample size (recommended for ``susie_rss``).
    bhat, shat
        Alternative to ``z`` for ``susie_rss``: marginal effect sizes and
        their standard errors.
    L
        Maximum number of causal effects (single effects) to fit.
    coverage
        Target coverage of the credible sets (default 0.95).
    min_abs_corr
        Minimum absolute correlation for a "pure" credible set.
    max_iter
        Maximum number of IBSS iterations.
    **kwargs
        Forwarded to the backend (:func:`pysusie.susie` /
        :func:`pysusie.susie_rss`).

    Returns
    -------
    pysusie.SusieFit
        Fitted model; use :func:`get_pip` / :func:`get_credible_sets`.
    """
    ps = _require_susie()
    key = str(method).lower().strip()

    if key == "susie":
        if X is None or y is None:
            raise ValueError("method='susie' requires individual-level X and y.")
        return ps.susie(
            np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel(),
            L=L, coverage=coverage, min_abs_corr=min_abs_corr,
            max_iter=max_iter, **kwargs,
        )

    if key == "susie_rss":
        if z is None and (bhat is None or shat is None):
            raise ValueError(
                "method='susie_rss' requires either z, or both bhat and shat."
            )
        if R is None:
            raise ValueError("method='susie_rss' requires the LD matrix R.")
        return ps.susie_rss(
            z=None if z is None else np.asarray(z, dtype=float).ravel(),
            R=np.asarray(R, dtype=float),
            n=n,
            bhat=None if bhat is None else np.asarray(bhat, dtype=float).ravel(),
            shat=None if shat is None else np.asarray(shat, dtype=float).ravel(),
            L=L, coverage=coverage, min_abs_corr=min_abs_corr,
            max_iter=max_iter, **kwargs,
        )

    raise ValueError(
        f"method must be 'susie' or 'susie_rss', got {method!r}"
    )


@register_function(
    aliases=["get_credible_sets", "credible_sets", "susie_cs", "可信集"],
    category="genetics",
    description=(
        "Extract the 95% credible sets from a fitted SuSiE model — the "
        "minimal variant groups that jointly capture each causal signal "
        "at the requested coverage. Wraps :func:`pysusie.susie_get_cs`."
    ),
    examples=[
        "ov.genetics.get_credible_sets(fit, R=ld)",
        "ov.genetics.get_credible_sets(fit, X=geno, coverage=0.9)",
    ],
    related=["ov.genetics.finemap", "ov.genetics.get_pip"],
)
def get_credible_sets(
    fit,
    *,
    X: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    coverage: float = 0.95,
    min_abs_corr: float = 0.5,
    **kwargs,
):
    """Return the credible sets of a SuSiE fit.

    Parameters
    ----------
    fit
        A :class:`pysusie.SusieFit` returned by :func:`finemap`.
    X
        Individual-level genotype matrix used to compute purity (for
        models fit with ``method='susie'``).
    R
        SNP-SNP LD correlation matrix used to compute purity (for
        ``method='susie_rss'`` models).
    coverage, min_abs_corr
        Credible-set coverage and the purity threshold.
    **kwargs
        Forwarded to :func:`pysusie.susie_get_cs`.

    Returns
    -------
    dict
        Credible-set summary (``cs``, ``purity``, ``coverage`` ...).
    """
    ps = _require_susie()
    return ps.susie_get_cs(
        fit, X=X, Xcorr=R, coverage=coverage,
        min_abs_corr=min_abs_corr, **kwargs,
    )


@register_function(
    aliases=["get_pip", "pip", "posterior_inclusion_probability", "后验包含概率"],
    category="genetics",
    description=(
        "Extract per-variant posterior inclusion probabilities (PIPs) "
        "from a fitted SuSiE model — the probability that each variant is "
        "causal. Wraps :func:`pysusie.susie_get_pip`."
    ),
    examples=[
        "ov.genetics.get_pip(fit)",
        "ov.genetics.get_pip(fit, prune_by_cs=True)",
    ],
    related=["ov.genetics.finemap", "ov.genetics.get_credible_sets"],
)
def get_pip(fit, *, prune_by_cs: bool = False, **kwargs) -> np.ndarray:
    """Return the per-variant PIP vector of a SuSiE fit.

    Parameters
    ----------
    fit
        A :class:`pysusie.SusieFit` returned by :func:`finemap`.
    prune_by_cs
        If ``True``, zero out variants not in any credible set.
    **kwargs
        Forwarded to :func:`pysusie.susie_get_pip`.

    Returns
    -------
    numpy.ndarray
        Length-``n_snps`` vector of posterior inclusion probabilities.
    """
    ps = _require_susie()
    return ps.susie_get_pip(fit, prune_by_cs=prune_by_cs, **kwargs)
