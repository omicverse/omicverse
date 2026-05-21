from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import patsy
from scipy import optimize, special, stats
from statsmodels.nonparametric.smoothers_lowess import lowess


@dataclass
class EList:
    """Container for limma ``EList`` fields produced by ``voom``."""

    E: np.ndarray
    weights: np.ndarray
    design: np.ndarray
    targets: pd.DataFrame
    genes: Optional[pd.DataFrame] = None
    span: Optional[float] = None
    voom_xy: Optional[dict[str, np.ndarray | str | float]] = None
    voom_line: Optional[dict[str, np.ndarray]] = None
    other: dict[str, np.ndarray | None] = field(default_factory=dict)
    gene_names: list[str] = field(default_factory=list)
    sample_names: list[str] = field(default_factory=list)


@dataclass
class MArrayLM:
    """Container mirroring limma's ``MArrayLM`` object for implemented fields."""

    coefficients: np.ndarray
    stdev_unscaled: np.ndarray
    sigma: np.ndarray
    df_residual: np.ndarray
    Amean: np.ndarray
    design: np.ndarray
    cov_coefficients: np.ndarray
    cov_coefficients_by_gene: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    fitted_values: Optional[np.ndarray] = None
    coef_names: list[str] = field(default_factory=list)
    gene_names: list[str] = field(default_factory=list)
    t: Optional[np.ndarray] = None
    p_value: Optional[np.ndarray] = None
    F: Optional[np.ndarray] = None
    F_p_value: Optional[np.ndarray] = None
    s2_post: Optional[np.ndarray] = None
    df_prior: Optional[float | np.ndarray] = None
    s2_prior: Optional[float | np.ndarray] = None
    df_total: Optional[np.ndarray] = None
    block: Optional[np.ndarray] = None
    correlation: Optional[float] = None
    ndups: int = 1
    spacing: int = 1
    treat_lfc: Optional[float] = None


@dataclass
class DuplicateCorrelation:
    """Result from ``duplicate_correlation``."""

    consensus_correlation: float
    cor: float
    atanh_correlations: np.ndarray


@dataclass
class TestResults:
    """limma-style -1/0/1 test result matrix."""

    values: np.ndarray
    levels: tuple[int, int, int] = (-1, 0, 1)
    labels: tuple[str, str, str] = ("Down", "NotSig", "Up")
    gene_names: list[str] = field(default_factory=list)
    coef_names: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, index=self.gene_names, columns=self.coef_names)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                coef: [np.sum(col == lev) for lev in self.levels]
                for coef, col in zip(self.coef_names, self.values.T)
            },
            index=self.labels,
        )


def _as_matrix(x: np.ndarray | pd.DataFrame, *, rows_prefix: str, cols_prefix: str):
    if isinstance(x, pd.DataFrame):
        return (
            x.to_numpy(dtype=float),
            list(x.index.astype(str)),
            list(x.columns.astype(str)),
        )
    arr = np.asarray(x, dtype=float)
    return (
        arr,
        [f"{rows_prefix}{i}" for i in range(arr.shape[0])],
        [f"{cols_prefix}{j}" for j in range(arr.shape[1])],
    )


def _as_weight_matrix(
    weights: np.ndarray | pd.DataFrame | pd.Series | float | None,
    shape: tuple[int, int],
    *,
    sample_names: list[str],
    gene_names: list[str],
) -> np.ndarray | None:
    if weights is None:
        return None
    if isinstance(weights, pd.DataFrame):
        if list(weights.index.astype(str)) == gene_names and list(weights.columns.astype(str)) == sample_names:
            arr = weights.to_numpy(dtype=float)
        elif list(weights.index.astype(str)) == sample_names and list(weights.columns.astype(str)) == gene_names:
            arr = weights.T.to_numpy(dtype=float)
        else:
            arr = weights.to_numpy(dtype=float)
    elif isinstance(weights, pd.Series):
        if list(weights.index.astype(str)) == sample_names:
            arr = weights.to_numpy(dtype=float)[None, :]
        elif list(weights.index.astype(str)) == gene_names:
            arr = weights.to_numpy(dtype=float)[:, None]
        else:
            arr = weights.to_numpy(dtype=float)
    else:
        arr = np.asarray(weights, dtype=float)

    if arr.ndim == 0:
        return np.full(shape, float(arr))
    if arr.ndim == 1:
        if arr.size == shape[1]:
            return np.broadcast_to(arr[None, :], shape).copy()
        if arr.size == shape[0]:
            return np.broadcast_to(arr[:, None], shape).copy()
        if arr.size == np.prod(shape):
            return arr.reshape(shape)
        raise ValueError(
            "one-dimensional weights must have length matching samples, genes, or M.size."
        )
    if arr.shape == shape:
        return arr.astype(float, copy=True)
    if arr.shape == (shape[1], shape[0]):
        return arr.T.astype(float, copy=True)
    raise ValueError(f"weights shape {arr.shape} is incompatible with M shape {shape}.")


def _unwrap_dups(M: np.ndarray, ndups: int = 2, spacing: int = 1) -> np.ndarray:
    """Match limma's internal ``unwrapdups`` matrix reshaping."""

    if ndups < 1:
        raise ValueError("ndups must be at least 1.")
    if spacing < 1:
        raise ValueError("spacing must be at least 1.")
    arr = np.asarray(M)
    if ndups == 1:
        return arr.copy()
    if arr.ndim != 2:
        raise ValueError("M must be two-dimensional.")
    n_spots, n_slides = arr.shape
    group_size = ndups * spacing
    if n_spots % group_size:
        raise ValueError("number of rows must be divisible by ndups * spacing.")
    n_groups = n_spots // group_size
    reshaped = arr.reshape((spacing, ndups, n_groups, n_slides), order="F")
    unwrapped = np.transpose(reshaped, (0, 2, 1, 3))
    return unwrapped.reshape((spacing * n_groups, ndups * n_slides), order="F")


def _duplicate_spot_correlation_matrix(narrays: int, ndups: int, correlation: float) -> np.ndarray:
    if abs(float(correlation)) >= 1:
        raise ValueError("correlation is 1 or -1, so the model is degenerate.")
    v = np.kron(np.eye(narrays) * float(correlation), np.ones((ndups, ndups)))
    np.fill_diagonal(v, 1.0)
    return v


def lm_fit(
    M: np.ndarray | pd.DataFrame,
    design: np.ndarray | pd.DataFrame,
    weights: np.ndarray | pd.DataFrame | pd.Series | float | None = None,
    block: np.ndarray | pd.Series | list | None = None,
    correlation: float | None = None,
    ndups: int = 1,
    spacing: int = 1,
) -> MArrayLM:
    """Fit limma-style row-wise linear models.

    Rows are genes/features and columns are samples. Missing values and
    non-positive weights are omitted per row, matching limma's default fit path.
    If ``block`` and ``correlation`` are supplied, a limma-style generalized
    least-squares fit is used for correlated samples.
    """

    y, gene_names, sample_names = _as_matrix(M, rows_prefix="gene_", cols_prefix="S")
    if isinstance(design, pd.DataFrame):
        x = design.to_numpy(dtype=float)
        coef_names = list(design.columns.astype(str))
    else:
        x = np.asarray(design, dtype=float)
        coef_names = [f"coef_{i}" for i in range(x.shape[1])]

    if y.ndim != 2 or x.ndim != 2:
        raise ValueError("M and design must be two-dimensional.")
    if x.shape[0] != y.shape[1]:
        raise ValueError(
            f"design has {x.shape[0]} rows but M has {y.shape[1]} columns."
        )

    if ndups < 1:
        raise ValueError("ndups must be at least 1.")
    if spacing < 1:
        raise ValueError("spacing must be at least 1.")
    if ndups > 1 and block is not None:
        raise NotImplementedError("duplicate spots with sample blocks are not implemented.")

    w = _as_weight_matrix(weights, y.shape, sample_names=sample_names, gene_names=gene_names)
    block_arr = None if block is None else np.asarray(block)
    if ndups > 1:
        if correlation is None:
            raise ValueError("correlation must be supplied when ndups > 1.")
        original_narrays = y.shape[1]
        row_index = _unwrap_dups(np.arange(y.shape[0])[:, None], ndups=ndups, spacing=spacing)[:, 0].astype(int)
        y = _unwrap_dups(y, ndups=ndups, spacing=spacing).astype(float, copy=False)
        if w is not None:
            w = _unwrap_dups(w, ndups=ndups, spacing=spacing).astype(float, copy=False)
        gene_names = [gene_names[i] for i in row_index]
        sample_names = [name for name in sample_names for _ in range(ndups)]
        x = np.repeat(x, ndups, axis=0)
        cormatrix = _duplicate_spot_correlation_matrix(original_narrays, ndups, float(correlation))
    else:
        cormatrix = None
    if block_arr is not None:
        if block_arr.size != y.shape[1]:
            raise ValueError("block length must match number of samples.")
        if correlation is None:
            raise ValueError("correlation must be supplied when block is used.")
        if abs(float(correlation)) >= 1:
            raise ValueError("correlation is 1 or -1, so the model is degenerate.")
        cormatrix = _block_correlation_matrix(block_arr, float(correlation))
    n_genes, n_coef = y.shape[0], x.shape[1]
    coefficients = np.full((n_genes, n_coef), np.nan)
    stdev_unscaled = np.full((n_genes, n_coef), np.nan)
    cov_by_gene = np.full((n_genes, n_coef, n_coef), np.nan)
    fitted_values = np.full_like(y, np.nan, dtype=float)
    sigma = np.full(n_genes, np.nan)
    df_residual = np.zeros(n_genes, dtype=float)

    if cormatrix is None:
        xtx_inv = np.linalg.pinv(x.T @ x)
    else:
        l_full = np.linalg.cholesky(cormatrix)
        x_full = np.linalg.solve(l_full, x)
        xtx_inv = np.linalg.pinv(x_full.T @ x_full)
    for i in range(n_genes):
        keep = np.isfinite(y[i])
        if w is not None:
            keep &= np.isfinite(w[i]) & (w[i] > 0)
        if not keep.any():
            continue
        xi = x[keep]
        yi = y[i, keep]
        if cormatrix is None and w is None:
            x_work = xi
            y_work = yi
            rss_weights = None
        elif cormatrix is None:
            sw = np.sqrt(w[i, keep])
            x_work = xi * sw[:, None]
            y_work = yi * sw
            rss_weights = w[i, keep]
        else:
            v = cormatrix[np.ix_(keep, keep)].copy()
            if w is not None:
                wrs = 1.0 / np.sqrt(w[i, keep])
                v = wrs[:, None] * v * wrs[None, :]
            l = np.linalg.cholesky(v)
            x_work = np.linalg.solve(l, xi)
            y_work = np.linalg.solve(l, yi)
            rss_weights = None
        rank_i = np.linalg.matrix_rank(x_work)
        df_i = yi.size - rank_i
        xtx_inv_i = np.linalg.pinv(x_work.T @ x_work)
        beta_i = xtx_inv_i @ x_work.T @ y_work
        resid_i = yi - xi @ beta_i
        coefficients[i] = beta_i
        fitted_values[i, keep] = xi @ beta_i
        if cormatrix is not None:
            resid_work = y_work - x_work @ beta_i
            rss = float(np.sum(resid_work * resid_work))
        elif rss_weights is None:
            rss = float(np.sum(resid_i * resid_i))
        else:
            rss = float(np.sum(rss_weights * resid_i * resid_i))
        sigma[i] = np.sqrt(rss / df_i) if df_i > 0 else np.nan
        df_residual[i] = df_i
        cov_by_gene[i] = xtx_inv_i
        stdev_unscaled[i] = np.sqrt(np.maximum(np.diag(xtx_inv_i), 0.0))

    return MArrayLM(
        coefficients=coefficients,
        stdev_unscaled=stdev_unscaled,
        sigma=sigma,
        df_residual=df_residual,
        Amean=np.nanmean(y, axis=1),
        design=x,
        cov_coefficients=xtx_inv,
        cov_coefficients_by_gene=cov_by_gene,
        weights=w,
        fitted_values=fitted_values,
        coef_names=coef_names,
        gene_names=gene_names,
        block=block_arr,
        correlation=correlation,
        ndups=ndups,
        spacing=spacing,
    )


def _block_correlation_matrix(block: np.ndarray, correlation: float) -> np.ndarray:
    block = np.asarray(block)
    same = block[:, None] == block[None, :]
    v = np.where(same, correlation, 0.0).astype(float)
    np.fill_diagonal(v, 1.0)
    return v


def remove_batch_effect(
    x: np.ndarray | pd.DataFrame,
    batch=None,
    batch2=None,
    covariates=None,
    design: np.ndarray | pd.DataFrame | None = None,
    group=None,
    **kwargs,
):
    """Remove fitted batch/covariate effects while preserving design effects."""

    y, gene_names, sample_names = _as_matrix(x, rows_prefix="gene_", cols_prefix="S")
    batch_terms = []
    batch_names = []
    if batch is not None:
        mat, names = _sum_contrast_matrix(batch, prefix="batch")
        batch_terms.append(mat)
        batch_names.extend(names)
    if batch2 is not None:
        mat, names = _sum_contrast_matrix(batch2, prefix="batch2")
        batch_terms.append(mat)
        batch_names.extend(names)
    if covariates is not None:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim == 1:
            cov = cov[:, None]
        if cov.shape[0] != y.shape[1]:
            raise ValueError("covariates rows must match number of samples.")
        cov = cov - np.nanmean(cov, axis=0, keepdims=True)
        batch_terms.append(cov)
        batch_names.extend([f"covariate_{i}" for i in range(cov.shape[1])])
    if not batch_terms:
        return pd.DataFrame(y.copy(), index=gene_names, columns=sample_names) if isinstance(x, pd.DataFrame) else y.copy()
    x_batch = np.column_stack(batch_terms)

    if group is not None:
        keep_design, keep_names = _treatment_design(group, prefix="group")
    elif design is None:
        keep_design = np.ones((y.shape[1], 1), dtype=float)
        keep_names = ["Intercept"]
    elif isinstance(design, pd.DataFrame):
        keep_design = design.to_numpy(dtype=float)
        keep_names = list(design.columns.astype(str))
    else:
        keep_design = np.asarray(design, dtype=float)
        keep_names = [f"coef_{i}" for i in range(keep_design.shape[1])]
    if keep_design.shape[0] != y.shape[1]:
        raise ValueError("design rows must match number of samples.")

    full_design = pd.DataFrame(
        np.column_stack([keep_design, x_batch]),
        index=sample_names,
        columns=keep_names + batch_names,
    )
    fit = lm_fit(pd.DataFrame(y, index=gene_names, columns=sample_names), full_design, **kwargs)
    beta = fit.coefficients[:, keep_design.shape[1] :].copy()
    beta[~np.isfinite(beta)] = 0.0
    adjusted = y - beta @ x_batch.T
    return pd.DataFrame(adjusted, index=gene_names, columns=sample_names) if isinstance(x, pd.DataFrame) else adjusted


def _sum_contrast_matrix(values, prefix: str):
    cats = pd.Categorical(values)
    levels = list(cats.categories.astype(str))
    n = len(cats)
    if len(levels) <= 1:
        return np.zeros((n, 0), dtype=float), []
    mat = np.zeros((n, len(levels) - 1), dtype=float)
    codes = cats.codes
    for j in range(len(levels) - 1):
        mat[codes == j, j] = 1.0
    mat[codes == len(levels) - 1, :] = -1.0
    return mat, [f"{prefix}{lev}" for lev in levels[:-1]]


def _treatment_design(values, prefix: str):
    cats = pd.Categorical(values)
    levels = list(cats.categories.astype(str))
    if len(levels) <= 1:
        return np.ones((len(cats), 1), dtype=float), ["Intercept"]
    mat = np.column_stack([np.ones(len(cats)), *[(cats == lev).astype(float) for lev in levels[1:]]])
    return mat, ["Intercept"] + [f"{prefix}{lev}" for lev in levels[1:]]


def duplicate_correlation(
    M: np.ndarray | pd.DataFrame,
    design: np.ndarray | pd.DataFrame | None = None,
    block: np.ndarray | pd.Series | list | None = None,
    trim: float = 0.15,
    weights: np.ndarray | pd.DataFrame | pd.Series | float | None = None,
    ndups: int = 1,
    spacing: int = 1,
) -> DuplicateCorrelation:
    """Estimate an intra-block sample correlation.

    This is a Python-native method-of-moments approximation to limma's
    mixed-model ``duplicateCorrelation`` for the common sample-block case.
    It returns the same result fields as limma and is intended to provide a
    practical starting value for ``lm_fit(..., block=..., correlation=...)``.
    """

    y, gene_names, sample_names = _as_matrix(M, rows_prefix="gene_", cols_prefix="S")
    if design is None:
        x = np.ones((y.shape[1], 1), dtype=float)
    elif isinstance(design, pd.DataFrame):
        x = design.to_numpy(dtype=float)
    else:
        x = np.asarray(design, dtype=float)
    if ndups < 1:
        raise ValueError("ndups must be at least 1.")
    if spacing < 1:
        raise ValueError("spacing must be at least 1.")
    w = _as_weight_matrix(weights, y.shape, sample_names=sample_names, gene_names=gene_names)
    if block is None and ndups > 1:
        narrays = y.shape[1]
        y = _unwrap_dups(y, ndups=ndups, spacing=spacing).astype(float, copy=False)
        if w is not None:
            w = _unwrap_dups(w, ndups=ndups, spacing=spacing).astype(float, copy=False)
        x = np.repeat(x, ndups, axis=0)
        block_arr = np.repeat(np.arange(narrays), ndups)
    elif block is None:
        return DuplicateCorrelation(0.0, 0.0, np.zeros(y.shape[0]))
    else:
        if ndups > 1:
            raise NotImplementedError("duplicate spots with sample blocks are not implemented.")
        block_arr = np.asarray(block)
    if block_arr.size != y.shape[1]:
        raise ValueError("block length must match number of samples.")
    if pd.Series(block_arr).value_counts().max() <= 1:
        return DuplicateCorrelation(0.0, 0.0, np.zeros(y.shape[0]))
    rho = np.full(y.shape[0], np.nan)
    for i in range(y.shape[0]):
        keep = np.isfinite(y[i])
        if w is not None:
            keep &= np.isfinite(w[i]) & (w[i] > 0)
        if np.sum(keep) <= x.shape[1] + 2:
            continue
        yi = y[i, keep]
        xi = x[keep]
        bi = np.linalg.pinv(xi.T @ xi) @ xi.T @ yi
        resid = yi - xi @ bi
        blocks = block_arr[keep]
        vals = []
        for lev in pd.unique(blocks):
            r = resid[blocks == lev]
            if r.size >= 2:
                vals.extend((r[:, None] * r[None, :])[np.triu_indices(r.size, k=1)])
        if not vals:
            continue
        denom = np.mean(resid * resid)
        if denom > 0:
            rho[i] = float(np.mean(vals) / denom)
    max_block = int(pd.Series(block_arr).value_counts().max())
    rhomin = 1.0 / (1.0 - max_block) + 0.01
    rho = np.clip(rho, rhomin, 0.99)
    arho = np.arctanh(rho)
    good = np.isfinite(arho)
    if not np.any(good):
        consensus = 0.0
    else:
        consensus = float(np.tanh(stats.trim_mean(arho[good], proportiontocut=trim)))
    return DuplicateCorrelation(consensus, consensus, arho)


def contrasts_fit(fit: MArrayLM, contrasts: np.ndarray | pd.DataFrame) -> MArrayLM:
    """Apply limma ``contrasts.fit`` for vector or matrix contrasts."""

    c = contrasts.to_numpy(dtype=float) if isinstance(contrasts, pd.DataFrame) else np.asarray(contrasts, dtype=float)
    contrast_names = list(contrasts.columns.astype(str)) if isinstance(contrasts, pd.DataFrame) else None
    if c.ndim == 1:
        c = c[:, None]
    if c.shape[0] != fit.coefficients.shape[1]:
        raise ValueError("contrast rows must match fitted coefficient columns.")

    fit.coefficients = fit.coefficients @ c
    if fit.cov_coefficients_by_gene is not None:
        cov_by_gene = np.einsum("ib,gbc,cj->gij", c.T, fit.cov_coefficients_by_gene, c)
        fit.cov_coefficients_by_gene = cov_by_gene
        fit.stdev_unscaled = np.sqrt(
            np.maximum(np.diagonal(cov_by_gene, axis1=1, axis2=2), 0.0)
        )
    else:
        cov_by_gene = None
        su = None
    cov_c = c.T @ fit.cov_coefficients @ c
    fit.cov_coefficients = cov_c
    if cov_by_gene is None:
        su = np.sqrt(np.maximum(np.diag(cov_c), 0.0))
        fit.stdev_unscaled = np.broadcast_to(su, fit.coefficients.shape).copy()
    fit.coef_names = contrast_names or [f"contrast_{i}" for i in range(c.shape[1])]
    fit.t = None
    fit.p_value = None
    fit.F = None
    fit.F_p_value = None
    return fit


def choose_lowess_span(
    n: int = 1000,
    *,
    small_n: int = 50,
    min_span: float = 0.3,
    power: float = 1.0 / 3.0,
) -> float:
    """Return limma's adaptive LOWESS span."""

    if n <= 0:
        raise ValueError("n must be positive.")
    return float(min(min_span + (1.0 - min_span) * (small_n / n) ** power, 1.0))


def _interp_rule2(x: np.ndarray, y: np.ndarray, new_x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if unique_x.size != x_sorted.size:
        y_sum = np.zeros(unique_x.size, dtype=float)
        y_count = np.zeros(unique_x.size, dtype=float)
        np.add.at(y_sum, inverse, y_sorted)
        np.add.at(y_count, inverse, 1.0)
        y_sorted = y_sum / y_count
        x_sorted = unique_x
    return np.interp(new_x, x_sorted, y_sorted, left=y_sorted[0], right=y_sorted[-1])


def voom(
    counts: np.ndarray | pd.DataFrame,
    design: np.ndarray | pd.DataFrame | None = None,
    lib_size: np.ndarray | pd.Series | None = None,
    offset: np.ndarray | pd.DataFrame | None = None,
    offset_prior: np.ndarray | pd.DataFrame | None = None,
    normalize_method: str = "none",
    weights: np.ndarray | pd.DataFrame | pd.Series | float | None = None,
    block: np.ndarray | pd.Series | list | None = None,
    correlation: float | None = None,
    span: float = 0.5,
    adaptive_span: bool = True,
    save_plot: bool = False,
) -> EList:
    """Compute limma ``voom`` logCPM values and observation weights.

    Implemented coverage follows the matrix-counts path with
    ``normalize_method='none'``. Between-array normalization methods are
    intentionally not implemented yet.
    """

    cts, gene_names, sample_names = _as_matrix(counts, rows_prefix="gene_", cols_prefix="S")
    if cts.ndim != 2:
        raise ValueError("counts must be two-dimensional.")
    if cts.shape[0] < 2:
        raise ValueError("Need at least two genes to fit a mean-variance trend.")
    if np.isnan(cts).any():
        raise ValueError("NA counts not allowed.")
    if np.min(cts) < 0:
        raise ValueError("Negative counts not allowed.")
    if normalize_method.lower() != "none":
        raise NotImplementedError("Only normalize_method='none' is implemented.")

    if design is None:
        x = np.ones((cts.shape[1], 1), dtype=float)
        coef_names = ["GrandMean"]
    elif isinstance(design, pd.DataFrame):
        x = design.to_numpy(dtype=float)
        coef_names = list(design.columns.astype(str))
    else:
        x = np.asarray(design, dtype=float)
        coef_names = [f"coef_{i}" for i in range(x.shape[1])]
    if x.ndim != 2 or x.shape[0] != cts.shape[1]:
        raise ValueError("design rows must match count columns.")

    if lib_size is None:
        lib = cts.sum(axis=0).astype(float)
    elif isinstance(lib_size, pd.Series):
        if list(lib_size.index.astype(str)) == sample_names:
            lib = lib_size.to_numpy(dtype=float)
        else:
            lib = lib_size.to_numpy(dtype=float)
    else:
        lib = np.asarray(lib_size, dtype=float)
    if lib.ndim != 1 or lib.size != cts.shape[1]:
        raise ValueError("lib_size must be a vector matching the number of samples.")

    lib_size_matrix = np.broadcast_to(lib[None, :], cts.shape).astype(float, copy=True)
    offset_arr = None if offset is None else _as_matrix(offset, rows_prefix="gene_", cols_prefix="S")[0]
    offset_prior_arr = None if offset_prior is None else _as_matrix(offset_prior, rows_prefix="gene_", cols_prefix="S")[0]
    if offset_arr is not None and offset_prior_arr is None:
        if offset_arr.shape != cts.shape:
            raise ValueError("counts and offset must have equal dimensions.")
        offset_prior_arr = offset_arr - np.nanmean(offset_arr, axis=1)[:, None]
    elif offset_prior_arr is not None:
        if offset_prior_arr.shape != cts.shape:
            raise ValueError("counts and offset_prior must have equal dimensions.")
    if offset_prior_arr is not None:
        lib_size_matrix = np.exp(np.log(lib_size_matrix) + offset_prior_arr)

    actual_span = choose_lowess_span(cts.shape[0], small_n=50, min_span=0.3, power=1.0 / 3.0) if adaptive_span else span

    y = np.log2((cts + 0.5) / (lib_size_matrix + 1.0) * 1e6)
    fit = lm_fit(
        pd.DataFrame(y, index=gene_names, columns=sample_names),
        pd.DataFrame(x, index=sample_names, columns=coef_names),
        weights=weights,
        block=block,
        correlation=correlation,
    )
    n_with_reps = int(np.sum(fit.df_residual > 0))
    targets = pd.DataFrame({"lib.size": lib}, index=sample_names)
    if n_with_reps < 2:
        return EList(
            E=y,
            weights=np.ones_like(y),
            design=x,
            targets=targets,
            span=actual_span if adaptive_span else None,
            other={"offset.prior": offset_prior_arr},
            gene_names=gene_names,
            sample_names=sample_names,
        )

    sx = fit.Amean + np.mean(np.log2(lib + 1.0)) - np.log2(1e6)
    sy = np.sqrt(fit.sigma)
    allzero = np.sum(cts, axis=1) == 0
    sx_fit = sx[~allzero]
    sy_fit = sy[~allzero]
    ok = np.isfinite(sx_fit) & np.isfinite(sy_fit)
    line = lowess(sy_fit[ok], sx_fit[ok], frac=actual_span, it=3, delta=0.01 * (np.max(sx_fit[ok]) - np.min(sx_fit[ok])), return_sorted=True)

    fitted_values = fit.coefficients @ x.T
    fitted_cpm = 2.0**fitted_values
    fitted_count = 1e-6 * fitted_cpm * (lib_size_matrix + 1.0)
    fitted_logcount = np.log2(fitted_count)
    trend = _interp_rule2(line[:, 0], line[:, 1], fitted_logcount)
    obs_weights = 1.0 / trend**4

    voom_xy = None
    voom_line = None
    if save_plot:
        voom_xy = {
            "x": sx_fit,
            "y": sy_fit,
            "xlab": "log2( count size + 0.5 )",
            "ylab": "Sqrt( standard deviation )",
            "pch": 16,
            "cex": 0.25,
        }
        voom_line = {"x": line[:, 0], "y": line[:, 1]}

    return EList(
        E=y,
        weights=obs_weights,
        design=x,
        targets=targets,
        span=actual_span if adaptive_span else None,
        voom_xy=voom_xy,
        voom_line=voom_line,
        other={"offset.prior": offset_prior_arr},
        gene_names=gene_names,
        sample_names=sample_names,
    )


def _trigamma_inverse(x: float, max_iter: int = 50, atol: float = 1e-8) -> float:
    if x <= 0:
        return float("inf")
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(max_iter):
        tg = special.polygamma(1, y)
        step = tg * (1.0 - tg / x) / special.polygamma(2, y)
        y += step
        if abs(step / y) < atol:
            break
    return float(y)


def _squeeze_var(
    var: np.ndarray,
    df: np.ndarray,
    covariate: np.ndarray | None = None,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
) -> tuple[float, np.ndarray | float, np.ndarray, np.ndarray]:
    var = np.asarray(var, dtype=float).copy()
    df = np.asarray(df, dtype=float)
    if robust:
        dfp = df[np.isfinite(df) & (df > 0)]
        if covariate is None and dfp.size and np.nanmin(dfp) == np.nanmax(dfp):
            return _squeeze_var_robust_equal_df(var, df, winsor_tail_p=winsor_tail_p)
        # Fall back to the non-robust estimator for currently unsupported
        # unequal-df/trended robust hyperparameter fits.
    good = np.isfinite(var) & (var > -1e-15) & np.isfinite(df) & (df > 1e-15)
    if not np.any(good):
        raise ValueError("No rows with positive residual variance.")
    x = np.maximum(var[good], 0.0)
    med = np.median(x)
    if med == 0:
        med = 1.0
    x = np.maximum(x, 1e-5 * med)
    df_good = df[good]
    z = np.log(x)
    e = z - special.digamma(df_good / 2.0) + np.log(df_good / 2.0)
    if covariate is None:
        emean_good = np.full_like(e, float(np.mean(e)))
        evar = float(np.var(e, ddof=1))
        emean_all = float(np.mean(e))
    else:
        cov = np.asarray(covariate, dtype=float)
        cov_good = cov[good]
        splinedf = 1 + int(x.size >= 3) + int(x.size >= 6) + int(x.size >= 30)
        splinedf = min(splinedf, np.unique(cov_good).size)
        if splinedf < 2:
            return _squeeze_var(var, df, covariate=None)
        design = np.asarray(patsy.dmatrix(f"cr(x, df={splinedf}) - 1", {"x": cov_good}, return_type="dataframe"))
        beta = np.linalg.pinv(design.T @ design) @ design.T @ e
        fitted = design @ beta
        rank = np.linalg.matrix_rank(design)
        resid = e - fitted
        evar = float(np.sum(resid * resid) / max(x.size - rank, 1))
        design_all = np.asarray(patsy.build_design_matrices([patsy.dmatrix(f"cr(x, df={splinedf}) - 1", {"x": cov_good}, return_type="dataframe").design_info], {"x": cov})[0])
        emean_full = design_all @ beta
        emean_good = fitted
        emean_all = emean_full
    evar = evar - float(np.mean(special.polygamma(1, df_good / 2.0)))
    if evar > 0:
        d0 = 2.0 * _trigamma_inverse(evar)
        s20_good = np.exp(emean_good + special.digamma(d0 / 2.0) - np.log(d0 / 2.0))
        if covariate is None:
            s20: np.ndarray | float = float(s20_good[0])
        else:
            s20 = np.exp(np.asarray(emean_all) + special.digamma(d0 / 2.0) - np.log(d0 / 2.0))
    else:
        d0 = float("inf")
        if covariate is None:
            s20 = float(np.mean(x))
        else:
            s20 = np.exp(np.asarray(emean_all))
    if np.isfinite(d0):
        s2_post = (df * var + d0 * np.asarray(s20)) / (df + d0)
    else:
        s2_post = np.broadcast_to(np.asarray(s20), var.shape).astype(float).copy()
    df_total = np.minimum(d0 + df, np.sum(df_good))
    return d0, s20, s2_post, df_total


def _trimmed_mean(x: np.ndarray, proportiontocut: float) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    cut = int(np.floor(proportiontocut * n))
    if cut > 0:
        x = x[cut:-cut]
    return float(np.mean(x))


def _f_q(p: np.ndarray | float, df1: float, df2: float) -> np.ndarray | float:
    if np.isinf(df2):
        return stats.chi2.ppf(p, df1) / df1
    return stats.f.ppf(p, df1, df2)


def _f_pdf(x: np.ndarray, df1: float, df2: float) -> np.ndarray:
    if np.isinf(df2):
        return df1 * stats.chi2.pdf(x * df1, df1)
    return stats.f.pdf(x, df1, df2)


def _f_sf(x: np.ndarray, df1: float, df2: float) -> np.ndarray:
    if np.isinf(df2):
        return stats.chi2.sf(x * df1, df1)
    return stats.f.sf(x, df1, df2)


def _f_logsf(x: np.ndarray, df1: float, df2: float) -> np.ndarray:
    if np.isinf(df2):
        return stats.chi2.logsf(x * df1, df1)
    return stats.f.logsf(x, df1, df2)


def _winsorized_f_moments(df1: float, df2: float, winsor_tail_p: tuple[float, float]) -> tuple[float, float]:
    p_low, p_right = winsor_tail_p
    probs = np.array([p_low, 1.0 - p_right], dtype=float)
    fq = np.asarray(_f_q(probs, df1, df2), dtype=float)
    zq = np.log(fq)
    q = fq / (1.0 + fq)
    nodes0, weights0 = np.polynomial.legendre.leggauss(128)
    nodes0 = (nodes0 + 1.0) / 2.0
    weights0 = weights0 / 2.0
    nodes = q[0] + (q[1] - q[0]) * nodes0
    fnodes = nodes / (1.0 - nodes)
    znodes = np.log(fnodes)
    dens = _f_pdf(fnodes, df1, df2) / (1.0 - nodes) ** 2
    q21 = q[1] - q[0]
    mean = q21 * np.sum(weights0 * dens * znodes) + np.sum(zq * np.asarray(winsor_tail_p))
    var = q21 * np.sum(weights0 * dens * (znodes - mean) ** 2) + np.sum(
        (zq - mean) ** 2 * np.asarray(winsor_tail_p)
    )
    return float(mean), float(var)


def _squeeze_var_robust_equal_df(
    var: np.ndarray,
    df: np.ndarray,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    good = np.isfinite(var) & np.isfinite(df) & (df > 1e-6)
    if np.sum(good) < 3:
        return _squeeze_var(var, df, robust=False)
    x = np.asarray(var, dtype=float).copy()
    xg = x[good].copy()
    dfg = np.asarray(df[good], dtype=float)
    df1 = float(np.max(dfg))
    med = float(np.median(xg))
    if med <= 0:
        raise ValueError("Variances are mostly <= 0.")
    xg[xg < med * 1e-12] = med * 1e-12
    nonrobust_d0, nonrobust_scale, _, _ = _squeeze_var(xg, np.full_like(xg, df1), robust=False)
    z = np.log(xg)
    ztrend = _trimmed_mean(z, winsor_tail_p[1])
    zresid = z - ztrend
    zq = np.quantile(zresid, [winsor_tail_p[0], 1.0 - winsor_tail_p[1]], method="linear")
    zwins = np.minimum(np.maximum(zresid, zq[0]), zq[1])
    zwmean = float(np.mean(zwins))
    zwvar = float(np.var(zwins, ddof=1))
    mean_inf, var_inf = _winsorized_f_moments(df1, np.inf, winsor_tail_p)
    funval_inf = np.log(zwvar / var_inf)
    n = xg.size
    if funval_inf <= 0:
        d0 = float("inf")
        ztrend_corrected = ztrend + zwmean - mean_inf
        scale = float(np.exp(ztrend_corrected))
        fstat = np.exp(z - ztrend_corrected)
        tail_p = stats.chi2.sf(fstat * df1, df1)
        empirical_tail = (n - stats.rankdata(fstat, method="average") + 0.5) / n
        prob_not_outlier = np.minimum(tail_p / empirical_tail, 1.0)
        df2_shrunk = np.full(n, np.inf)
        outlier = prob_not_outlier < 1
        if np.any(outlier):
            df_pooled = n * df1
            df2_shrunk[outlier] = prob_not_outlier[outlier] * df_pooled
            order = np.argsort(tail_p, kind="mergesort")
            df2_shrunk[order] = np.maximum.accumulate(df2_shrunk[order])
    else:
        if np.isinf(nonrobust_d0):
            d0 = float("inf")
        else:
            def fun(link_x):
                df2 = link_x / (1.0 - link_x)
                _, v = _winsorized_f_moments(df1, df2, winsor_tail_p)
                return np.log(zwvar / v)

            low = nonrobust_d0 / (1.0 + nonrobust_d0)
            fun_low = fun(low)
            if fun_low >= 0:
                d0 = float(nonrobust_d0)
            else:
                root = optimize.brentq(fun, low, 1.0 - 1e-12, xtol=1e-8)
                d0 = float(root / (1.0 - root))
        mean_d0, _ = _winsorized_f_moments(df1, d0, winsor_tail_p)
        ztrend_corrected = ztrend + zwmean - mean_d0
        scale = float(np.exp(ztrend_corrected))
        fstat = np.exp(z - ztrend_corrected)
        log_tail_p = _f_logsf(fstat, df1, d0)
        tail_p = np.exp(log_tail_p)
        log_empirical_tail = np.log(n - stats.rankdata(fstat, method="average") + 0.5) - np.log(n)
        log_prob_not_outlier = np.minimum(log_tail_p - log_empirical_tail, 0.0)
        prob_not_outlier = np.exp(log_prob_not_outlier)
        prob_outlier = -np.expm1(log_prob_not_outlier)
        if np.any(log_prob_not_outlier < 0):
            min_log_tail_p = float(np.min(log_tail_p))
            if min_log_tail_p == -np.inf:
                df2_outlier = 0.0
                df2_shrunk = prob_not_outlier * d0
            else:
                df2_outlier = np.log(0.5) / min_log_tail_p * d0
                new_log_tail_p = float(_f_logsf(np.array([np.max(fstat)]), df1, df2_outlier)[0])
                df2_outlier = np.log(0.5) / new_log_tail_p * df2_outlier
                df2_shrunk = prob_not_outlier * d0 + prob_outlier * df2_outlier
            order = np.argsort(log_tail_p, kind="mergesort")
            ordered = df2_shrunk[order]
            running = np.cumsum(ordered) / np.arange(1, n + 1)
            imin = int(np.argmin(running))
            ordered[: imin + 1] = running[imin]
            df2_shrunk[order] = np.maximum.accumulate(ordered)
        else:
            df2_shrunk = np.full(n, d0)

    df_prior = np.full(var.shape, d0, dtype=float)
    df_prior[good] = df2_shrunk
    scale_all = float(scale)
    with np.errstate(invalid="ignore"):
        s2_post = (df * var + df_prior * scale_all) / (df + df_prior)
    inf_prior = ~np.isfinite(df_prior)
    if np.any(inf_prior):
        s2_post[inf_prior] = scale_all
    df_total = np.minimum(df_prior + df, np.sum(df[good]))
    return df_prior, scale_all, s2_post, df_total


def ebayes(
    fit: MArrayLM,
    stdev_coef_lim: tuple[float, float] = (0.1, 4.0),
    trend: bool | np.ndarray = False,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
) -> MArrayLM:
    """Run limma's empirical-Bayes moderated t calculation."""

    s2 = np.asarray(fit.sigma, dtype=float) ** 2
    df = np.asarray(fit.df_residual, dtype=float)
    if isinstance(trend, bool):
        covariate = fit.Amean if trend else None
    else:
        covariate = np.asarray(trend, dtype=float)
    d0, s20, s2_post, df_total = _squeeze_var(
        s2,
        df,
        covariate=covariate,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
    )

    # The moderated t uses the RAW stdev_unscaled. R limma applies
    # `stdev.coef.lim` only inside the B-statistic (lods) calculation,
    # never to the t-statistic. Clipping stdev_unscaled here deflated
    # the moderated t for any design column with a small
    # stdev_unscaled — e.g. a continuous predictor — pushing every
    # gene below significance. (A two-group contrast has
    # stdev_unscaled ~sqrt(1/n1+1/n2) ≈ 0.5, inside the [0.1,4] clip,
    # so the bug stayed latent there.)
    su = np.asarray(fit.stdev_unscaled, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = fit.coefficients / (su * np.sqrt(s2_post)[:, None])
    p = 2.0 * stats.t.sf(np.abs(t), df_total[:, None])

    fit.df_prior = d0
    fit.s2_prior = s20
    fit.s2_post = s2_post
    fit.df_total = df_total
    fit.t = t
    fit.p_value = p
    f_stat, df1 = _classify_tests_f_stat(t, fit.cov_coefficients)
    fit.F = f_stat
    fit.F_p_value = stats.f.sf(f_stat, df1, df_total)
    return fit


def treat(
    fit: MArrayLM,
    fc: float = 1.2,
    lfc: float | None = None,
    trend: bool | np.ndarray = False,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
    upshot: bool = False,
) -> MArrayLM:
    """Run limma ``treat`` moderated tests against a logFC threshold.

    Current coverage implements the non-robust ``treat`` paths with the same
    variance moderation used by :func:`ebayes`.
    """

    if lfc is None:
        lfc = float(np.log2(fc))
    lfc = abs(float(lfc))
    ebayes(
        fit,
        stdev_coef_lim=(0.0, np.inf),
        trend=trend,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
    )
    coef = np.asarray(fit.coefficients, dtype=float)
    se = np.asarray(fit.stdev_unscaled, dtype=float) * np.sqrt(fit.s2_post)[:, None]
    acoef = np.abs(coef)
    lfc_for_t = lfc
    if upshot and lfc > 0:
        nodes, weights = np.polynomial.legendre.leggauss(16)
        nodes = lfc * nodes
        weights = weights / 2.0
        p = np.zeros_like(coef, dtype=float)
        df = fit.df_total[:, None]
        for lfci, weight in zip(nodes[8:], weights[8:]):
            with np.errstate(divide="ignore", invalid="ignore"):
                tstat_right_i = (acoef - lfci) / se
                tstat_left_i = (acoef + lfci) / se
            p += weight * (stats.t.sf(tstat_right_i, df) + stats.t.sf(tstat_left_i, df))
        p = 2.0 * p
        lfc_for_t = lfc / 2.0
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat_right = (acoef - lfc) / se
            tstat_left = (acoef + lfc) / se
        p = stats.t.sf(tstat_right, fit.df_total[:, None]) + stats.t.sf(tstat_left, fit.df_total[:, None])
    t_out = np.zeros_like(coef)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat_right = (acoef - lfc_for_t) / se
    t_pos = np.maximum(tstat_right, 0.0)
    t_out[coef > lfc_for_t] = t_pos[coef > lfc_for_t]
    t_out[coef < -lfc_for_t] = -t_pos[coef < -lfc_for_t]
    fit.treat_lfc = lfc
    fit.t = t_out
    fit.p_value = p
    fit.F = None
    fit.F_p_value = None
    return fit


def decide_tests(
    fit: MArrayLM | np.ndarray | pd.DataFrame,
    method: str = "separate",
    adjust_method: str = "BH",
    p_value: float = 0.05,
    lfc: float = 0.0,
) -> TestResults:
    """Classify tests as down/not-significant/up, like limma ``decideTests``.

    Implemented methods are ``separate``, ``global``, ``hierarchical`` and
    ``nestedF``.
    """

    method = method.lower()
    if method not in {"separate", "global", "hierarchical", "nestedf"}:
        raise NotImplementedError("Unknown decide_tests method.")
    if isinstance(fit, MArrayLM):
        if fit.p_value is None:
            ebayes(fit)
        p = np.asarray(fit.p_value, dtype=float).copy()
        coef = np.asarray(fit.coefficients, dtype=float)
        gene_names = fit.gene_names
        coef_names = fit.coef_names
        if method == "hierarchical":
            out = _decide_tests_hierarchical_marraylm(fit, adjust_method, p_value)
            if lfc > 0:
                out = out * (np.abs(coef) > lfc)
            return TestResults(out.astype(int), gene_names=gene_names, coef_names=coef_names)
        if method == "nestedf":
            out = _decide_tests_nested_f(fit, adjust_method, p_value)
            if lfc > 0:
                out = out * (np.abs(coef) > lfc)
            return TestResults(out.astype(int), gene_names=gene_names, coef_names=coef_names)
    else:
        if method == "nestedf":
            raise ValueError("nestedF adjust method requires an MArrayLM object.")
        if isinstance(fit, pd.DataFrame):
            p = fit.to_numpy(dtype=float)
            gene_names = list(fit.index.astype(str))
            coef_names = list(fit.columns.astype(str))
        else:
            p = np.asarray(fit, dtype=float)
            gene_names = [f"gene_{i}" for i in range(p.shape[0])]
            coef_names = [f"coef_{i}" for i in range(p.shape[1])]
        coef = np.ones_like(p)
    if method == "separate":
        adj = np.empty_like(p)
        for j in range(p.shape[1]):
            adj[:, j] = _p_adjust(p[:, j], adjust_method)
    elif method == "global":
        adj = _p_adjust(p.ravel(), adjust_method).reshape(p.shape)
    else:
        adj = _decide_tests_hierarchical_pvalues(p, adjust_method, p_value)
    out = np.sign(coef).astype(int) * (adj < p_value).astype(int)
    if lfc > 0 and isinstance(fit, MArrayLM):
        out = out * (np.abs(coef) > lfc)
    return TestResults(out.astype(int), gene_names=gene_names, coef_names=coef_names)


def top_table(
    fit: MArrayLM,
    coef: int | str = 0,
    number: int | float = 10,
    sort_by: str = "P",
    adjust_method: str = "BH",
) -> pd.DataFrame:
    """Return a limma-like top table for one coefficient."""

    if fit.t is None or fit.p_value is None:
        raise ValueError("Run ebayes/eBayes before top_table.")
    coef_i = fit.coef_names.index(coef) if isinstance(coef, str) else int(coef)
    p = fit.p_value[:, coef_i]
    order = np.argsort(p if sort_by.upper().startswith("P") else -np.abs(fit.t[:, coef_i]))
    if np.isfinite(number):
        order = order[: int(number)]
    adj = _p_adjust_bh(p) if adjust_method.upper() == "BH" else p
    return pd.DataFrame(
        {
            "gene": np.asarray(fit.gene_names)[order],
            "logFC": fit.coefficients[order, coef_i],
            "AveExpr": fit.Amean[order],
            "t": fit.t[order, coef_i],
            "P.Value": p[order],
            "adj.P.Val": adj[order],
        }
    )


def _classify_tests_f_stat(tstat: np.ndarray, cov_coefficients: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    tstat = np.asarray(tstat, dtype=float)
    if tstat.ndim == 1:
        tstat = tstat[:, None]
    ntests = tstat.shape[1]
    if ntests == 1:
        return np.squeeze(tstat * tstat), 1
    if cov_coefficients is None:
        r = ntests
        q = np.eye(r) / np.sqrt(r)
    else:
        cov = np.asarray(cov_coefficients, dtype=float).copy()
        diag = np.diag(cov)
        diag = np.where(diag == 0, 1.0, diag)
        cor = cov / np.sqrt(np.outer(diag, diag))
        vals, vecs = np.linalg.eigh(cor)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        r = int(np.sum(vals / vals[0] > 1e-8)) if vals[0] > 0 else 1
        q = (vecs[:, :r] / np.sqrt(vals[:r])[None, :]) / np.sqrt(r)
    z = tstat @ q
    return np.sum(z * z, axis=1), r


def _classify_tests_f(
    tstat: np.ndarray,
    cov_coefficients: np.ndarray | None,
    df: np.ndarray,
    p_value: float,
) -> np.ndarray:
    tstat = np.asarray(tstat, dtype=float)
    if tstat.ndim == 1:
        tstat = tstat[:, None]
    ngenes, ntests = tstat.shape
    if ntests == 1:
        p = 2.0 * stats.t.sf(np.abs(tstat), df[:, None])
        return np.sign(tstat).astype(int) * (p < p_value).astype(int)

    if cov_coefficients is None:
        r = ntests
        q = np.eye(r) / np.sqrt(r)
    else:
        cov = np.asarray(cov_coefficients, dtype=float).copy()
        diag = np.diag(cov)
        diag = np.where(diag == 0, 1.0, diag)
        cor = cov / np.sqrt(np.outer(diag, diag))
        vals, vecs = np.linalg.eigh(cor)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        r = int(np.sum(vals / vals[0] > 1e-8)) if vals[0] > 0 else 1
        q = (vecs[:, :r] / np.sqrt(vals[:r])[None, :]) / np.sqrt(r)

    qf = stats.f.isf(p_value, r, df)
    result = np.zeros_like(tstat, dtype=int)
    for i in range(ngenes):
        x = tstat[i].copy()
        if np.any(~np.isfinite(x)):
            result[i] = 0
            continue
        if float(np.sum((x @ q) ** 2)) > qf[i]:
            order = np.argsort(-np.abs(x), kind="mergesort")
            result[i, order[0]] = int(np.sign(x[order[0]]))
            for j in range(1, ntests):
                bigger = order[:j]
                x[bigger] = np.sign(x[bigger]) * abs(x[order[j]])
                if float(np.sum((x @ q) ** 2)) > qf[i]:
                    result[i, order[j]] = int(np.sign(x[order[j]]))
                else:
                    break
    return result


def _classify_tests_p_by_row(tstat: np.ndarray, df: np.ndarray, p_value: float, method: str) -> np.ndarray:
    tstat = np.asarray(tstat, dtype=float)
    if tstat.ndim == 1:
        tstat = tstat[:, None]
    p = 2.0 * stats.t.sf(np.abs(tstat), df[:, None])
    result = np.zeros_like(tstat, dtype=int)
    for i in range(tstat.shape[0]):
        adj = _p_adjust(p[i], method)
        result[i] = np.sign(tstat[i]).astype(int) * (adj < p_value).astype(int)
    return result


def _hierarchical_gate_scale(adj_method: str, selected: np.ndarray) -> float:
    method = _normalize_adjust_method(adj_method)
    i = int(np.sum(selected))
    n = int(np.sum(np.isfinite(selected)))
    if n == 0:
        return 0.0
    if method == "none":
        return 1.0
    if method == "bonferroni":
        return 1.0 / n
    if method == "holm":
        return 1.0 / max(n - i + 1, 1)
    if method == "BH":
        return i / n
    if method == "BY":
        return i / n / np.sum(1.0 / np.arange(1, n + 1))
    raise NotImplementedError(f"p-value adjustment method {adj_method!r} is not implemented.")


def _decide_tests_hierarchical_marraylm(fit: MArrayLM, adjust_method: str, p_value: float) -> np.ndarray:
    if fit.F_p_value is None:
        ebayes(fit)
    f_p = np.asarray(fit.F_p_value, dtype=float)
    if np.any(~np.isfinite(f_p)):
        raise ValueError("Can't handle NA p-values yet.")
    selected = _p_adjust(f_p, adjust_method) < p_value
    scale = _hierarchical_gate_scale(adjust_method, selected)
    result = np.zeros_like(fit.t, dtype=int)
    if np.any(selected):
        df = np.asarray(fit.df_residual, dtype=float)[selected] + float(fit.df_prior)
        result[selected] = _classify_tests_p_by_row(fit.t[selected], df, p_value * scale, adjust_method)
    return result


def _decide_tests_nested_f(fit: MArrayLM, adjust_method: str, p_value: float) -> np.ndarray:
    if fit.F_p_value is None:
        ebayes(fit)
    f_p = np.asarray(fit.F_p_value, dtype=float)
    if np.any(~np.isfinite(f_p)):
        raise ValueError("nestedF method can't handle NA p-values.")
    selected = _p_adjust(f_p, adjust_method) < p_value
    scale = _hierarchical_gate_scale(adjust_method, selected)
    result = np.zeros_like(fit.t, dtype=int)
    if np.any(selected):
        df = np.asarray(fit.df_residual, dtype=float)[selected] + float(fit.df_prior)
        result[selected] = _classify_tests_f(fit.t[selected], fit.cov_coefficients, df, p_value * scale)
    return result


def _decide_tests_hierarchical_pvalues(p: np.ndarray, adjust_method: str, p_value: float) -> np.ndarray:
    p = np.asarray(p, dtype=float).copy()
    ngenes, ncontrasts = p.shape
    simes_multiplier = ncontrasts / np.arange(1, ncontrasts + 1)
    genewise = np.ones(ngenes, dtype=float)
    for g in range(ngenes):
        row = np.sort(p[g][np.isfinite(p[g])])
        if row.size:
            genewise[g] = np.min(row * simes_multiplier[: row.size])
    selected = _p_adjust(genewise, adjust_method) <= p_value
    p[~selected] = 1.0
    for g in np.flatnonzero(selected):
        p[g] = _p_adjust(p[g], adjust_method)
    scale = _hierarchical_gate_scale(adjust_method, selected)
    return p <= p_value * scale


def _p_adjust_bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    ranks = np.argsort(p[ok])
    sorted_p = p[ok][ranks]
    n = sorted_p.size
    adj_sorted = np.minimum.accumulate((sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    ok_idx = np.flatnonzero(ok)
    out[ok_idx[ranks]] = adj_sorted
    return out


def _normalize_adjust_method(method: str) -> str:
    m = method.upper()
    if m == "FDR":
        return "BH"
    if m == "NONE":
        return "none"
    if m == "BONFERRONI":
        return "bonferroni"
    if m == "HOLM":
        return "holm"
    if m == "BY":
        return "BY"
    if m == "BH":
        return "BH"
    raise NotImplementedError(f"p-value adjustment method {method!r} is not implemented.")


def _p_adjust(p: np.ndarray, method: str = "BH") -> np.ndarray:
    norm = _normalize_adjust_method(method)
    if norm == "BH":
        return _p_adjust_bh(p)
    if norm == "none":
        return np.asarray(p, dtype=float)
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    n = int(np.sum(ok))
    if n == 0:
        return out
    vals = p[ok]
    if norm == "bonferroni":
        out[ok] = np.minimum(vals * n, 1.0)
    elif norm == "holm":
        order = np.argsort(vals, kind="mergesort")
        ordered = vals[order]
        adj = np.maximum.accumulate((n - np.arange(n)) * ordered)
        adj = np.minimum(adj, 1.0)
        ok_idx = np.flatnonzero(ok)
        out[ok_idx[order]] = adj
    elif norm == "BY":
        out[ok] = np.minimum(_p_adjust_bh(vals) * np.sum(1.0 / np.arange(1, n + 1)), 1.0)
    else:
        raise NotImplementedError(f"p-value adjustment method {method!r} is not implemented.")
    return out
