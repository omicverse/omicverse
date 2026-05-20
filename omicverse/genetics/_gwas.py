"""GWAS core for ``ov.genetics`` — pure numpy / scipy / statsmodels.

Unlike the rest of ``ov.genetics`` this module has no external backend:
the standard genome-wide-association primitives are implemented
directly. Covers per-SNP / per-sample quality control (:func:`gwas_qc`),
the per-SNP association scan (:func:`gwas_association`, linear or
logistic) and the genomic-inflation factor (:func:`genomic_inflation`).
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .._registry import register_function


# --------------------------------------------------------------------------- #
# Hardy-Weinberg exact test                                                    #
# --------------------------------------------------------------------------- #
def _hwe_exact_p(n_aa: int, n_ab: int, n_bb: int) -> float:
    """Wigginton-Cutler-Abecasis (2005) exact HWE test p-value.

    Parameters
    ----------
    n_aa, n_ab, n_bb
        Counts of homozygous-major, heterozygous and homozygous-minor
        genotypes.

    Returns
    -------
    float
        Two-sided exact-test p-value (1.0 for an empty / degenerate site).
    """
    n_aa, n_ab, n_bb = int(n_aa), int(n_ab), int(n_bb)
    n = n_aa + n_ab + n_bb
    if n == 0:
        return 1.0
    # Minor-allele count.
    n_a = 2 * min(n_aa, n_bb) + n_ab
    n_genotypes = n
    # Probability mass over all heterozygote counts of the same parity.
    het_probs = np.zeros(n_a + 1)
    mid = n_a * (2 * n_genotypes - n_a) // (2 * n_genotypes)
    if mid % 2 != n_a % 2:
        mid += 1
    het_probs[mid] = 1.0
    s = het_probs[mid]
    # Downward recursion from mid.
    curr_hets = mid
    curr_homr = (n_a - mid) // 2
    curr_homc = n_genotypes - curr_hets - curr_homr
    while curr_hets > 1:
        het_probs[curr_hets - 2] = (
            het_probs[curr_hets] * curr_hets * (curr_hets - 1.0)
            / (4.0 * (curr_homr + 1.0) * (curr_homc + 1.0))
        )
        s += het_probs[curr_hets - 2]
        curr_homr += 1
        curr_homc += 1
        curr_hets -= 2
    # Upward recursion from mid.
    curr_hets = mid
    curr_homr = (n_a - mid) // 2
    curr_homc = n_genotypes - curr_hets - curr_homr
    while curr_hets <= n_a - 2:
        het_probs[curr_hets + 2] = (
            het_probs[curr_hets] * 4.0 * curr_homr * curr_homc
            / ((curr_hets + 2.0) * (curr_hets + 1.0))
        )
        s += het_probs[curr_hets + 2]
        curr_homr -= 1
        curr_homc -= 1
        curr_hets += 2
    if s <= 0:
        return 1.0
    het_probs /= s
    obs_hets = n_ab
    p = het_probs[het_probs <= het_probs[obs_hets] + 1e-12].sum()
    return float(min(1.0, max(0.0, p)))


def _as_genotype_matrix(genotype):
    """Coerce a genotype input to a (samples, snps) float array + SNP names."""
    if isinstance(genotype, pd.DataFrame):
        return genotype.to_numpy(dtype=float), list(genotype.columns.astype(str))
    arr = np.asarray(genotype, dtype=float)
    if arr.ndim != 2:
        raise ValueError("genotype must be a 2-D (samples x SNPs) array.")
    return arr, [f"snp{i}" for i in range(arr.shape[1])]


@register_function(
    aliases=[
        "gwas_qc", "genotype_qc", "snp_qc", "gwas_quality_control",
        "GWAS质控", "基因型质控", "SNP质控",
    ],
    category="genetics",
    description=(
        "Quality control for a GWAS genotype matrix — applies the "
        "standard per-SNP and per-sample filters: SNP call rate, minor-"
        "allele frequency (MAF), the Hardy-Weinberg-equilibrium exact "
        "test (Wigginton 2005), and per-sample missingness. Operates on "
        "an AnnData of samples x SNPs (dosage / 0-1-2 genotype in ``.X``) "
        "and returns a filtered AnnData plus the SNP / sample QC metrics. "
        "Pure numpy / scipy — no external backend."
    ),
    examples=[
        "ov.genetics.gwas_qc(adata, maf=0.01, call_rate=0.95, hwe=1e-6)",
        "adata_qc = ov.genetics.gwas_qc(adata, sample_call_rate=0.98)",
    ],
    related=["ov.genetics.gwas_association", "ov.genetics.read_plink"],
)
def gwas_qc(
    adata,
    *,
    call_rate: float = 0.95,
    maf: float = 0.01,
    hwe: float = 1e-6,
    sample_call_rate: float = 0.95,
    copy: bool = True,
):
    """Quality-control a GWAS genotype AnnData.

    Parameters
    ----------
    adata
        AnnData of ``samples x SNPs``; ``.X`` holds genotype dosages
        (0 / 1 / 2, with NaN for missing calls).
    call_rate
        Minimum per-SNP call rate (fraction of non-missing genotypes).
    maf
        Minimum minor-allele frequency.
    hwe
        Hardy-Weinberg-equilibrium exact-test p-value cutoff — SNPs below
        it are dropped.
    sample_call_rate
        Minimum per-sample call rate.
    copy
        If ``True`` (default) return a filtered copy; if ``False`` filter
        in place.

    Returns
    -------
    AnnData
        The QC-filtered AnnData. ``.var`` gains ``call_rate``, ``maf``,
        ``hwe_p``; ``.obs`` gains ``sample_call_rate``.
    """
    X = np.asarray(adata.X, dtype=float)
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray().astype(float)

    n_samples, n_snps = X.shape
    missing = np.isnan(X)

    # Per-SNP metrics.
    snp_call = 1.0 - missing.mean(axis=0)
    with np.errstate(invalid="ignore"):
        allele_freq = np.nanmean(X, axis=0) / 2.0
    snp_maf = np.minimum(allele_freq, 1.0 - allele_freq)

    hwe_p = np.ones(n_snps)
    for j in range(n_snps):
        col = X[:, j]
        col = col[~np.isnan(col)]
        if col.size == 0:
            continue
        rounded = np.rint(col)
        n_aa = int(np.sum(rounded == 0))
        n_ab = int(np.sum(rounded == 1))
        n_bb = int(np.sum(rounded == 2))
        hwe_p[j] = _hwe_exact_p(n_aa, n_ab, n_bb)

    # Per-sample metrics.
    sample_call = 1.0 - missing.mean(axis=1)

    snp_keep = (
        (snp_call >= call_rate)
        & (snp_maf >= maf)
        & (hwe_p >= hwe)
    )
    sample_keep = sample_call >= sample_call_rate

    out = adata.copy() if copy else adata
    out.var["call_rate"] = snp_call
    out.var["maf"] = snp_maf
    out.var["hwe_p"] = hwe_p
    out.obs["sample_call_rate"] = sample_call
    out = out[sample_keep, snp_keep].copy()
    out.uns["gwas_qc"] = {
        "n_snps_in": int(n_snps),
        "n_snps_kept": int(snp_keep.sum()),
        "n_samples_in": int(n_samples),
        "n_samples_kept": int(sample_keep.sum()),
        "thresholds": {
            "call_rate": call_rate, "maf": maf, "hwe": hwe,
            "sample_call_rate": sample_call_rate,
        },
    }
    return out


@register_function(
    aliases=[
        "gwas_association", "gwas_scan", "association_test", "gwas",
        "全基因组关联分析", "GWAS关联分析", "关联扫描",
    ],
    category="genetics",
    description=(
        "Per-SNP genome-wide association scan. Regresses a phenotype on "
        "each SNP's genotype (with optional covariates): ``model='linear'`` "
        "uses ordinary least squares for a quantitative trait, "
        "``model='logistic'`` uses logistic regression for a binary "
        "(case / control) trait. Returns a tidy per-SNP DataFrame with "
        "the effect size (beta / log-OR), standard error, test statistic "
        "and p-value. Pure numpy / scipy / statsmodels — no external "
        "backend."
    ),
    examples=[
        "ov.genetics.gwas_association(genotype, phenotype, model='linear')",
        "ov.genetics.gwas_association(genotype, phenotype, covariates=pcs, model='logistic')",
    ],
    related=["ov.genetics.gwas_qc", "ov.genetics.genomic_inflation",
             "ov.genetics.manhattan", "ov.genetics.qqplot"],
)
def gwas_association(
    genotype: Union[pd.DataFrame, np.ndarray],
    phenotype: Union[pd.Series, np.ndarray],
    covariates: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    *,
    model: str = "linear",
) -> pd.DataFrame:
    """Run a per-SNP GWAS association scan.

    Parameters
    ----------
    genotype
        Genotype matrix — ``samples x SNPs`` (DataFrame or array). Values
        are dosages / 0-1-2 genotypes; NaN entries are dropped per SNP.
    phenotype
        Per-sample phenotype — quantitative (``model='linear'``) or binary
        0/1 (``model='logistic'``).
    covariates
        Optional ``samples x covariates`` matrix (e.g. genotype PCs, age,
        sex) added to every per-SNP regression.
    model
        ``'linear'`` (OLS, quantitative trait) or ``'logistic'``
        (logistic regression, binary trait).

    Returns
    -------
    pandas.DataFrame
        One row per SNP, columns ``snp``, ``beta``, ``se``, ``stat``,
        ``pvalue``, ``n`` — sorted by p-value.
    """
    key = str(model).lower().strip()
    if key not in ("linear", "logistic"):
        raise ValueError(f"model must be 'linear' or 'logistic', got {model!r}")

    X, snp_names = _as_genotype_matrix(genotype)
    y = np.asarray(phenotype, dtype=float).ravel()
    if y.size != X.shape[0]:
        raise ValueError(
            f"phenotype has length {y.size}, expected {X.shape[0]} samples."
        )
    if covariates is not None:
        C = (covariates.to_numpy(dtype=float)
             if isinstance(covariates, pd.DataFrame)
             else np.asarray(covariates, dtype=float))
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != X.shape[0]:
            raise ValueError("covariates must have the same number of samples.")
    else:
        C = None

    n_snps = X.shape[1]
    beta = np.full(n_snps, np.nan)
    se = np.full(n_snps, np.nan)
    stat = np.full(n_snps, np.nan)
    pval = np.full(n_snps, np.nan)
    n_used = np.zeros(n_snps, dtype=int)

    if key == "linear":
        from scipy import stats as _st
        for j in range(n_snps):
            g = X[:, j]
            keep = ~np.isnan(g) & ~np.isnan(y)
            if C is not None:
                keep &= ~np.isnan(C).any(axis=1)
            ng = int(keep.sum())
            if ng < 3 or np.ptp(g[keep]) == 0:
                continue
            cols = [np.ones(ng), g[keep]]
            if C is not None:
                cols.append(C[keep])
            design = np.column_stack(cols)
            yk = y[keep]
            # OLS via least squares.
            coef, _, rank, _ = np.linalg.lstsq(design, yk, rcond=None)
            if rank < design.shape[1]:
                continue
            resid = yk - design @ coef
            dof = ng - design.shape[1]
            if dof <= 0:
                continue
            sigma2 = float(resid @ resid) / dof
            try:
                xtx_inv = np.linalg.inv(design.T @ design)
            except np.linalg.LinAlgError:
                continue
            b = coef[1]
            b_se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
            if b_se == 0:
                continue
            t = b / b_se
            beta[j] = b
            se[j] = b_se
            stat[j] = t
            pval[j] = float(2.0 * _st.t.sf(abs(t), dof))
            n_used[j] = ng
    else:  # logistic
        try:
            import statsmodels.api as sm
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "model='logistic' requires statsmodels: "
                "`pip install statsmodels` (or omicverse[genetics])."
            ) from exc
        from scipy import stats as _st
        for j in range(n_snps):
            g = X[:, j]
            keep = ~np.isnan(g) & ~np.isnan(y)
            if C is not None:
                keep &= ~np.isnan(C).any(axis=1)
            ng = int(keep.sum())
            if ng < 5 or np.ptp(g[keep]) == 0:
                continue
            cols = [g[keep]]
            if C is not None:
                cols.append(C[keep])
            design = sm.add_constant(np.column_stack(cols), has_constant="add")
            yk = y[keep]
            try:
                fit = sm.Logit(yk, design).fit(disp=False, maxiter=100)
            except Exception:
                continue
            b = float(fit.params[1])
            b_se = float(fit.bse[1])
            if not np.isfinite(b_se) or b_se == 0:
                continue
            z = b / b_se
            beta[j] = b
            se[j] = b_se
            stat[j] = z
            pval[j] = float(2.0 * _st.norm.sf(abs(z)))
            n_used[j] = ng

    res = pd.DataFrame({
        "snp": snp_names,
        "beta": beta,
        "se": se,
        "stat": stat,
        "pvalue": pval,
        "n": n_used,
    })
    return res.sort_values("pvalue", na_position="last").reset_index(drop=True)


@register_function(
    aliases=[
        "genomic_inflation", "lambda_gc", "inflation_factor",
        "基因组膨胀因子", "lambdaGC",
    ],
    category="genetics",
    description=(
        "Genomic-inflation factor (lambda GC) — the ratio of the median "
        "observed association chi-square to its expected value under the "
        "null. A value near 1 indicates well-calibrated test statistics; "
        "values much above 1 flag population stratification or "
        "cryptic relatedness. Accepts p-values, z-scores or chi-square "
        "statistics. Pure numpy / scipy."
    ),
    examples=[
        "ov.genetics.genomic_inflation(res['pvalue'])",
        "ov.genetics.genomic_inflation(zscores, statistic='z')",
    ],
    related=["ov.genetics.gwas_association", "ov.genetics.qqplot"],
)
def genomic_inflation(
    values: Union[pd.Series, np.ndarray],
    *,
    statistic: str = "pvalue",
) -> float:
    """Compute the genomic-inflation factor lambda GC.

    Parameters
    ----------
    values
        Per-SNP association statistics — p-values, z-scores or
        chi-square values (see ``statistic``).
    statistic
        ``'pvalue'`` (default), ``'z'`` (z-scores) or ``'chi2'``
        (chi-square statistics).

    Returns
    -------
    float
        The genomic-inflation factor (1 df).
    """
    from scipy import stats as _st

    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")

    key = str(statistic).lower().strip()
    if key == "pvalue":
        v = np.clip(v, np.finfo(float).tiny, 1.0)
        chi2 = _st.chi2.isf(v, df=1)
    elif key == "z":
        chi2 = v ** 2
    elif key == "chi2":
        chi2 = v
    else:
        raise ValueError(
            f"statistic must be 'pvalue', 'z' or 'chi2', got {statistic!r}"
        )

    chi2 = chi2[np.isfinite(chi2)]
    if chi2.size == 0:
        return float("nan")
    return float(np.median(chi2) / _st.chi2.ppf(0.5, df=1))
