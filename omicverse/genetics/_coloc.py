"""Colocalization analysis for ``ov.genetics`` — coloc backend.

Wraps the standalone :mod:`pycoloc` package (Giambartolomei *et al.*
2014; Wallace 2020/2021). :func:`colocalize` tests whether two traits
(e.g. a GWAS signal and an eQTL) share a causal variant at a locus.
``method`` dispatches between the single-causal-variant ABF model,
the SuSiE-based multi-signal model, and the conditional/masking
``coloc_signals`` model.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .._registry import register_function


def _require_coloc():
    """Import :mod:`pycoloc` with a friendly error if it is missing."""
    try:
        import pycoloc  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.genetics.colocalize requires the pycoloc backend: "
            "`pip install pycoloc` (or `pip install omicverse[genetics]`)."
        ) from exc
    return pycoloc


@register_function(
    aliases=[
        "colocalize", "coloc", "colocalization", "coloc_abf",
        "共定位", "共定位分析",
    ],
    category="genetics",
    description=(
        "Bayesian colocalization of two traits — tests whether a GWAS "
        "signal and a molecular QTL (eQTL / pQTL) are driven by the same "
        "causal variant. ``method`` selects the model: ``'abf'`` (default; "
        "single causal variant per locus, Giambartolomei 2014), "
        "``'susie'`` (SuSiE fine-mapping allowing multiple causal "
        "variants, Wallace 2021), or ``'signals'`` "
        "(conditional/masking iteration over multiple signals). Each "
        "dataset is a dict of summary statistics (keys such as ``beta``, "
        "``varbeta``/``MAF``, ``snp``, ``type``, ``N``, ``sdY``). Returns "
        "posterior probabilities PP.H0..PP.H4 (H4 = shared causal "
        "variant). Wraps the pycoloc backend."
    ),
    examples=[
        "ov.genetics.colocalize(d1, d2, method='abf')",
        "ov.genetics.colocalize(d1, d2, method='susie')",
        "ov.genetics.colocalize(d1, d2, method='signals', p12=1e-6)",
    ],
    related=["ov.genetics.finemap", "ov.genetics.finemap_abf",
             "ov.genetics.coloc_sensitivity", "ov.genetics.coloc_plot"],
)
def colocalize(
    dataset1: dict,
    dataset2: dict,
    *,
    method: str = "abf",
    MAF: Optional[np.ndarray] = None,
    p1: float = 1e-4,
    p2: float = 1e-4,
    p12: float = 1e-5,
    **kwargs,
):
    """Colocalize two GWAS / QTL traits.

    Parameters
    ----------
    dataset1, dataset2
        Summary-statistics dicts for the two traits. Recognised keys
        include ``beta``, ``varbeta`` (or ``MAF`` + ``N``), ``snp``,
        ``position``, ``type`` (``'quant'`` / ``'cc'``), ``N``, ``sdY``,
        and ``LD`` (required for ``method='susie'``).
    method
        ``'abf'`` (single causal variant), ``'susie'`` (multiple causal
        variants via SuSiE), or ``'signals'`` (conditional/masking).
    MAF
        Optional minor-allele-frequency vector attached to datasets that
        lack ``MAF``.
    p1, p2, p12
        Prior probabilities that a SNP associates with trait 1, trait 2,
        or both.
    **kwargs
        Forwarded to the backend coloc function.

    Returns
    -------
    object
        The backend result — for ``'abf'`` a :class:`pycoloc.ColocABF`
        with a ``summary`` (``PP.H0..PP.H4.abf``) and per-SNP ``results``.
    """
    cl = _require_coloc()
    key = str(method).lower().strip()

    if key == "abf":
        return cl.coloc_abf(dataset1, dataset2, MAF=MAF,
                            p1=p1, p2=p2, p12=p12, **kwargs)
    if key == "susie":
        p12_susie = kwargs.pop("p12", 5e-6) if p12 == 1e-5 else p12
        return cl.coloc_susie(dataset1, dataset2,
                              p1=p1, p2=p2, p12=p12_susie, **kwargs)
    if key in ("signals", "signal"):
        return cl.coloc_signals(dataset1, dataset2, MAF=MAF,
                                p1=p1, p2=p2, p12=p12, **kwargs)

    raise ValueError(
        f"method must be one of 'abf', 'susie', 'signals', got {method!r}"
    )


@register_function(
    aliases=["finemap_abf", "abf_finemap", "近似贝叶斯因子精细定位"],
    category="genetics",
    description=(
        "Single-trait fine-mapping with Approximate Bayes Factors "
        "(Wakefield 2009) — computes a per-SNP posterior probability of "
        "causality for one GWAS / QTL dataset. Wraps "
        ":func:`pycoloc.finemap_abf`."
    ),
    examples=[
        "ov.genetics.finemap_abf(dataset)",
        "ov.genetics.finemap_abf(dataset, p1=1e-4)",
    ],
    related=["ov.genetics.colocalize", "ov.genetics.finemap"],
)
def finemap_abf(dataset: dict, *, p1: float = 1e-4):
    """ABF fine-mapping of a single dataset.

    Parameters
    ----------
    dataset
        Summary-statistics dict (as for :func:`colocalize`).
    p1
        Prior probability a SNP is causal.

    Returns
    -------
    pandas.DataFrame
        Per-SNP table with the approximate Bayes factor and posterior
        probability of association.
    """
    cl = _require_coloc()
    return cl.finemap_abf(dataset, p1=p1)


@register_function(
    aliases=["coloc_sensitivity", "sensitivity", "coloc敏感性分析", "共定位敏感性"],
    category="genetics",
    description=(
        "Prior-sensitivity analysis for a colocalization result — shows "
        "how the posterior probabilities respond to the choice of priors "
        "p1 / p2 / p12, and whether a colocalization rule (e.g. "
        "``'H4 > 0.8'``) is robust. Wraps :func:`pycoloc.sensitivity`."
    ),
    examples=[
        "ov.genetics.coloc_sensitivity(res, rule='H4 > 0.5')",
        "ov.genetics.coloc_sensitivity(res, rule='H4 > 0.8', dataset1=d1, dataset2=d2)",
    ],
    related=["ov.genetics.colocalize", "ov.genetics.coloc_plot"],
)
def coloc_sensitivity(
    obj,
    *,
    rule: str = "",
    dataset1: Optional[dict] = None,
    dataset2: Optional[dict] = None,
    doplot: bool = True,
    **kwargs,
):
    """Run a prior-sensitivity analysis on a coloc result.

    Parameters
    ----------
    obj
        A result from :func:`colocalize` (``method='abf'``).
    rule
        Colocalization rule to evaluate, e.g. ``'H4 > 0.8'``.
    dataset1, dataset2
        The original input datasets (needed to redraw Manhattan panels).
    doplot
        Whether to draw the sensitivity figure.
    **kwargs
        Forwarded to :func:`pycoloc.sensitivity`.

    Returns
    -------
    object
        The backend sensitivity result.
    """
    cl = _require_coloc()
    return cl.sensitivity(obj, rule=rule, dataset1=dataset1,
                          dataset2=dataset2, doplot=doplot, **kwargs)
