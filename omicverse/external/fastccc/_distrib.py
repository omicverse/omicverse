"""Discrete null distributions on the log1p expression grid (FastCCC ``distrib``).

Port of ``fastccc/distrib.py`` (FastCCC 1.0.1, MIT, Hou, Ma & Zhou 2025).
Every operation keeps the original numerical recipe -- same grid, same
FFT convolutions, same normal (CLT) shortcuts, same trimming -- so p-values
match the reference implementation to floating-point precision.

A :class:`Distribution` is either a normal law (``loc``/``scale``) or a PMF on
the grid ``0, 0.01, ..., 13.99``; ``a & b`` is the law of ``a + b`` when both
are non-zero (zero otherwise), ``a / n`` the law of a mean of ``n`` draws and
``a * b`` the law of ``sqrt(a * b)``.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import norm

PRECISION = 0.01
MAX_LOG1P = 14
PMF_BIN_EDGE = np.arange(0, MAX_LOG1P + 1e-10, PRECISION)
PMF_BINS = np.arange(0, MAX_LOG1P - 1e-10, PRECISION)
PMF_INT_BINS = np.arange(len(PMF_BINS))


@lru_cache(maxsize=1)
def _index_mat() -> np.ndarray:
    """Grid index of ``sqrt(x * y)``: first bin ``i`` with ``bins[i]**2 >= x * y``."""
    sq = PMF_BINS ** 2
    prod = np.outer(PMF_BINS, PMF_BINS)
    return np.searchsorted(sq, prod.ravel(), side="left").reshape(prod.shape).astype(np.int16)


def _norm_cdf_func(loc, scale):
    def analytic_func(x):
        return norm.cdf(x, loc, scale)
    return analytic_func


def _norm_minimum_cdf_func(locs, scales):
    def analytic_func(x):
        F = [norm.cdf(x, loc, scale) for loc, scale in zip(locs, scales)]
        return 1 - np.prod(1 - np.array(F), axis=0)
    return analytic_func


class Distribution:
    """Normal or grid-PMF distribution with FastCCC's algebra."""

    __slots__ = ("loc", "scale", "dtype", "is_align", "pmf_array", "is_normal_type", "is_analytic",
                 "is_complex_analytic", "cdf_analytic_func", "min_cdf_non_zero", "min_cdf_one",
                 "support_length", "cdf_array", "ligand", "receptor")

    def __init__(self, dtype="other", loc=None, scale=None, pmf_array=None, is_align=None):
        if dtype in ("normal", "gaussian"):
            if loc is None or scale is None:
                raise ValueError("A normal Distribution needs loc and scale.")
            is_align, dtype = True, "normal"
        else:
            if pmf_array is None or is_align is None:
                raise ValueError("A grid Distribution needs pmf_array and is_align.")
            pmf_array = np.trim_zeros(pmf_array, trim="b")
        self.loc = loc
        self.scale = scale
        self.dtype = dtype
        self.is_align = is_align
        self.pmf_array = pmf_array
        self.is_normal_type = dtype == "normal"
        self.is_analytic = self.is_normal_type
        self.is_complex_analytic = False
        self.cdf_analytic_func = None
        self.cdf_array = None
        self.ligand = self.receptor = None
        if self.is_normal_type:
            self.cdf_analytic_func = _norm_cdf_func(loc, scale)
            self.min_cdf_non_zero = self.loc - 5 * self.scale
            self.min_cdf_one = self.loc + 5 * self.scale
            self.support_length = self.min_cdf_one - self.min_cdf_non_zero

    # ------------------------------------------------------------------ arrays
    @property
    def pmf(self):
        return self.get_pmf_array()

    @property
    def cdf(self):
        return self.get_cdf_array()

    def get_pmf_array(self):
        if self.pmf_array is None:  # normal law discretised on the grid
            cdf = norm.cdf(PMF_BINS, loc=self.loc, scale=self.scale)
            self.pmf_array = np.trim_zeros(np.diff(cdf, prepend=0), trim="b")
        return self.pmf_array

    def get_cdf_array(self):
        if self.cdf_array is None:
            self.cdf_array = np.cumsum(self.get_pmf_array())
        return self.cdf_array

    # ----------------------------------------------------------------- algebra
    def __add__(self, other):
        if self.is_normal_type and other.is_normal_type:
            return Distribution("normal", loc=self.loc + other.loc,
                                scale=np.sqrt(np.square(self.scale) + np.square(other.scale)))
        new = Distribution("other", pmf_array=fftconvolve(self.get_pmf_array(), other.get_pmf_array()),
                           is_align=False)
        if self.is_analytic and other.is_analytic:
            new.is_complex_analytic = True
            new.ligand, new.receptor = self, other
        return new

    def __pow__(self, n):
        """Law of the sum of ``n`` i.i.d. draws (normal for ``n >= 30``)."""
        if n == 1:
            return self
        if n >= 30 or self.is_normal_type:
            return Distribution("normal", loc=self.loc * n, scale=np.sqrt(n) * self.scale)
        arr = self.get_pmf_array()
        out = arr
        for _ in range(1, n):
            out = fftconvolve(out, arr)
        return Distribution("other", pmf_array=out, is_align=False)

    def __truediv__(self, n):
        """Law of ``X / n`` re-binned onto the grid."""
        if n == 1:
            return self
        if self.is_normal_type:
            return Distribution("normal", loc=self.loc / n, scale=self.scale / n)
        cdf = self.get_cdf_array()
        bins = PMF_INT_BINS * n
        n_cut = np.sum(bins < len(cdf))
        avg = Distribution("other", pmf_array=np.diff(np.append(cdf[bins[:n_cut]], 1.0), prepend=0),
                           is_align=True)
        if self.is_complex_analytic:
            avg.is_complex_analytic = True
            avg.ligand, avg.receptor = self.ligand, self.receptor
        return avg

    def __and__(self, other):
        """Law of ``X + Y`` where the sum is 0 if either term is 0."""
        if self.is_normal_type and other.is_normal_type:
            return self.__add__(other)
        p1, p2 = self.get_pmf_array(), other.get_pmf_array()
        pr0 = p1[0] + p2[0] - p1[0] * p2[0]
        arr = np.insert(fftconvolve(p1[1:], p2[1:]), 0, [pr0, 0])
        return Distribution("other", pmf_array=arr, is_align=False)

    def __mul__(self, other):
        """Law of ``sqrt(X * Y)`` (not ``X * Y``)."""
        outer = np.outer(self.pmf, other.pmf)
        h, w = outer.shape
        arr = np.bincount(_index_mat()[:h, :w].ravel(), weights=outer.ravel())
        return Distribution("other", pmf_array=arr, is_align=True)


# --------------------------------------------------------------------------- builders
def distribution_from_samples(samples) -> Distribution:
    """Empirical grid PMF of log1p expression, keeping the sample mean/std for the CLT."""
    samples = np.asarray(samples)
    if np.min(samples) < 0:
        raise ValueError("Expression values must be non-negative.")
    if np.max(samples) > MAX_LOG1P:
        raise ValueError(f"FastCCC expects log1p data (max <= {MAX_LOG1P}); got {np.max(samples)}.")
    pmf, _ = np.histogram(samples, PMF_BIN_EDGE)
    pmf = pmf / np.sum(pmf)
    d = Distribution("other", pmf_array=pmf, is_align=True)
    d.loc, d.scale = np.mean(samples), np.std(samples)
    return d


def minimum_distribution(*dists: Distribution) -> Distribution:
    """Law of the minimum of independent variables (complex ``Minimum`` aggregation)."""
    if len(dists) == 1:
        return dists[0]
    if all(d.is_normal_type for d in dists):
        locs = [d.loc for d in dists]
        scales = [d.scale for d in dists]
        order = np.argsort(locs)
        loc_other = np.array(locs)[order[1:]]
        scale_other = np.array(scales)[order[1:]]
        if locs[order[0]] + 5 * scales[order[0]] <= np.min(loc_other - 5 * scale_other):
            return Distribution("normal", loc=locs[order[0]], scale=scales[order[0]])
        func = _norm_minimum_cdf_func(locs, scales)
        out = Distribution("other", pmf_array=np.diff(func(PMF_BINS), prepend=0), is_align=True)
        out.is_analytic = True
        out.cdf_analytic_func = func
        return out
    arrays = [d.get_pmf_array() for d in dists]
    length = max(a.shape[0] for a in arrays)
    F = np.cumsum(np.vstack([np.pad(a, (0, length - a.shape[0]), "constant") for a in arrays]), axis=1)
    return Distribution("other", pmf_array=np.diff(1 - np.prod(1 - F, axis=0), prepend=0), is_align=True)


def average_distribution(*dists: Distribution) -> Distribution:
    """Law of the subunit average (complex ``Average`` aggregation)."""
    if len(dists) == 1:
        return dists[0]
    out = dists[0]
    for d in dists[1:]:
        out = out + d
    return out / len(dists)


# --------------------------------------------------------------------------- p-values
def _precise_pmf_array(dist: Distribution, n_segment: int = 100):
    domain = np.linspace(dist.min_cdf_non_zero, dist.min_cdf_one, n_segment + 1)
    return np.diff(dist.cdf_analytic_func(domain), prepend=0), domain


def _pvalue_complex_analytic(value, dist: Distribution) -> float:
    value *= 2  # the sum of ligand and receptor
    lo = (dist.ligand.min_cdf_non_zero + dist.receptor.min_cdf_non_zero) / 2
    hi = (dist.ligand.min_cdf_one + dist.receptor.min_cdf_one) / 2
    if value <= lo * 2:
        return 1.0
    if value >= hi * 2:
        return 0.0
    if dist.ligand.support_length > dist.receptor.support_length:
        accurate, discrete = dist.ligand, dist.receptor
    else:
        accurate, discrete = dist.receptor, dist.ligand
    pmf, domain = _precise_pmf_array(discrete)
    p = 0.0
    for x, w in zip(domain, pmf):
        p += w * (1 - accurate.cdf_analytic_func(value - x))
    return p


def pvalue_from_distribution(value, dist: Distribution) -> float:
    """Upper-tail probability ``P(S >= value)`` under the null law ``dist``."""
    if dist.is_analytic:
        return 1 - dist.cdf_analytic_func(value)
    if dist.is_complex_analytic:
        return _pvalue_complex_analytic(value, dist)
    y = dist.get_pmf_array()
    return 1 - np.sum(y[:int(np.ceil(value / PRECISION))])


# --------------------------------------------------------------------------- order statistics
_M = 100000
with np.errstate(divide="ignore", invalid="ignore"):
    _LOG_F = np.log(np.arange(1, _M + 1))
    _LOG_Q = np.log(_M + 1 - np.arange(1, _M + 1))


@lru_cache(maxsize=8)
def _fxi_cdf(n: int, i: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = (i - 1) * _LOG_F + (n - i) * _LOG_Q
        fxi = np.exp(logp - np.max(logp))
        fxi /= np.sum(fxi)
    return np.cumsum(fxi)


def _split_integer_by_probability(N: int, probabilities) -> np.ndarray:
    targets = N * np.asarray(probabilities)
    parts = np.floor(targets).astype(int)
    error = N - np.sum(parts)
    order = np.argsort(-(targets - parts))
    for k in range(error):
        parts[order[k]] += 1
    return parts


def quantile_sections(dist: Distribution) -> np.ndarray:
    """Grid-bin boundaries of ``dist`` on the 1..M rank scale (depends on the gene only)."""
    return np.int32(np.cumsum(_split_integer_by_probability(_M, dist.pmf)))


@lru_cache(maxsize=256)
def _padded_fxi_cdf(n: int, i: int) -> np.ndarray:
    return np.concatenate([[0.0], _fxi_cdf(n, i)])


def quantile_pmf_for_n_iid(dist: Distribution, n: int, quantile: float, sections=None) -> np.ndarray:
    """PMF of the ``quantile`` order statistic of ``n`` i.i.d. draws from ``dist``."""
    i = max(int(quantile * (n - 1) + 1), 1)
    if sections is None:
        sections = quantile_sections(dist)
    cdf = _padded_fxi_cdf(n, i)[sections]
    return np.clip(np.diff(cdf, prepend=0), a_min=0, a_max=None)
