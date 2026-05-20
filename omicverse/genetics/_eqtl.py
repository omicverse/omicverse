"""eQTL mapping for ``ov.genetics`` — Matrix eQTL backend.

Exposes :func:`eqtl_map`, a dispatcher over the linear / ANOVA /
linear-cross models of `Matrix eQTL <https://github.com/andreyshabalin/MatrixEQTL>`_
(Shabalin 2012). The heavy lifting lives in the standalone
:mod:`pymatrixeqtl` package — this module is a thin, registry-aware
wrapper that translates plain DataFrames / arrays into the backend's
:class:`SlicedData` containers.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .._registry import register_function


def _require_matrixeqtl():
    """Import :mod:`pymatrixeqtl` with a friendly error if it is missing."""
    try:
        import pymatrixeqtl  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.genetics.eqtl_map requires the pymatrixeqtl backend: "
            "`pip install pymatrixeqtl` (or `pip install omicverse[genetics]`)."
        ) from exc
    return pymatrixeqtl


_MODEL_ALIASES = {
    "linear": "modelLINEAR",
    "anova": "modelANOVA",
    "linear_cross": "modelLINEAR_CROSS",
    "cross": "modelLINEAR_CROSS",
}


@register_function(
    aliases=[
        "eqtl_map", "eqtl", "matrix_eqtl", "eQTL mapping",
        "表达数量性状位点", "eQTL分析", "eQTL映射",
    ],
    category="genetics",
    description=(
        "eQTL mapping via Matrix eQTL (Shabalin 2012). Scans every "
        "SNP-gene pair for a genotype-expression association. ``model`` "
        "selects the test: ``'linear'`` (additive dosage), ``'anova'`` "
        "(genotype as a 3-level factor) or ``'linear_cross'`` (genotype x "
        "covariate interaction). Genotype / expression / covariates are "
        "given as features x samples DataFrames or numpy arrays; optional "
        "``snp_pos`` / ``gene_pos`` enable a cis / trans split. Wraps the "
        "pymatrixeqtl backend; returns a result object whose ``.cis`` / "
        "``.trans`` / ``.all`` tables hold beta, t-stat, p-value and FDR."
    ),
    examples=[
        "ov.genetics.eqtl_map(geno, expr, model='linear')",
        "ov.genetics.eqtl_map(geno, expr, covariates=cov, model='anova')",
        "ov.genetics.eqtl_map(geno, expr, snp_pos=spos, gene_pos=gpos, cis_dist=1e6)",
    ],
    related=["ov.genetics.finemap", "ov.genetics.colocalize", "ov.genetics.manhattan"],
)
def eqtl_map(
    genotype: Union[pd.DataFrame, np.ndarray, str],
    expression: Union[pd.DataFrame, np.ndarray, str],
    covariates: Optional[Union[pd.DataFrame, np.ndarray, str]] = None,
    *,
    snp_pos: Optional[Union[pd.DataFrame, str]] = None,
    gene_pos: Optional[Union[pd.DataFrame, str]] = None,
    model: str = "linear",
    cis_dist: float = 1e6,
    pv_threshold: float = 1e-5,
    pv_threshold_cis: float = 0.0,
    n_anova_groups: int = 3,
    verbose: bool = False,
    **kwargs,
):
    """Map eQTLs with Matrix eQTL.

    Parameters
    ----------
    genotype
        Genotype dosages — ``SNPs x samples``. A DataFrame (index = SNP
        IDs, columns = sample IDs), a 2-D numpy array, a TSV path, or a
        :class:`pymatrixeqtl.SlicedData`.
    expression
        Expression matrix — ``genes x samples``, same accepted types.
    covariates
        Optional covariate matrix — ``covariates x samples``.
    snp_pos, gene_pos
        Optional SNP / gene position tables. When both are supplied the
        result is split into ``cis`` (within ``cis_dist``) and ``trans``.
    model
        ``'linear'`` (additive), ``'anova'`` (3-level genotype factor), or
        ``'linear_cross'`` (genotype x first-covariate interaction).
    cis_dist
        Maximum SNP-gene distance (bp) counted as ``cis``. Default 1e6.
    pv_threshold
        p-value cutoff for reported trans (or all) associations.
    pv_threshold_cis
        p-value cutoff for reported cis associations; ``0`` disables the
        cis / trans split unless positions are given.
    n_anova_groups
        Number of genotype levels for ``model='anova'``.
    verbose
        Print backend progress.
    **kwargs
        Forwarded to :func:`pymatrixeqtl.eqtl`.

    Returns
    -------
    pymatrixeqtl.MatrixEQTLResult
        Result object exposing ``.all`` / ``.cis`` / ``.trans`` DataFrames
        (columns: ``snps``, ``gene``, ``beta``, ``statistic``, ``pvalue``,
        ``FDR``).
    """
    meq = _require_matrixeqtl()

    key = str(model).lower().strip()
    if key not in _MODEL_ALIASES:
        raise ValueError(
            f"model must be one of {sorted(_MODEL_ALIASES)}, got {model!r}"
        )
    model_const = getattr(meq, _MODEL_ALIASES[key])

    # Cis/trans split only when both position tables are supplied; otherwise
    # report a single combined table at ``pv_threshold``.
    use_cis = snp_pos is not None and gene_pos is not None
    pv_cis = pv_threshold_cis if (use_cis and pv_threshold_cis > 0) else (
        pv_threshold if use_cis else 0.0
    )

    result = meq.eqtl(
        genotype, expression, covariates,
        model=model_const,
        pv_threshold=pv_threshold,
        pv_threshold_cis=pv_cis,
        snpspos=snp_pos,
        genepos=gene_pos,
        cis_dist=cis_dist,
        n_anova_groups=n_anova_groups,
        verbose=verbose,
        **kwargs,
    )
    return result
