"""Count likelihoods for the ``ov.vi`` models, written as plain tensor formulas.

Parameterisation follows scvi-tools: a negative binomial with mean ``mu`` and
inverse dispersion ``theta``; zero inflation through logits ``pi`` (``ZINB``);
a protein background / foreground NB mixture with mixing logits (``totalVI``).
All functions return element-wise log-probabilities (same shape as ``x``) and
use the same ``eps`` guards as scvi-tools, so values agree to float precision.

Sparse forms
------------
Count matrices are mostly zeros.  For ``x == 0`` the ``lgamma`` / ``x * log``
terms of the NB vanish, so the per-cell sums only need them on the stored
entries.  :func:`nb_rowsum` / :func:`zinb_rowsum` take the dense mean
``mu`` plus the coordinates ``(row, col, val)`` of the non-zeros of ``x`` and
return the per-cell log-likelihood sums with about half the transcendental
work of the dense formula (identical up to float rounding).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

EPS = 1e-8


def _broadcast_theta(theta: torch.Tensor) -> torch.Tensor:
    return theta.view(1, -1) if theta.dim() == 1 else theta


# ---------------------------------------------------------------------------
# element-wise log-probabilities
# ---------------------------------------------------------------------------
def log_nb_positive(x, mu, theta, eps: float = EPS):
    """log NB(x; mu, theta) element-wise (scvi-tools ``log_nb_positive``)."""
    theta = _broadcast_theta(theta)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    return (theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1))


def log_zinb_positive(x, mu, theta, pi, eps: float = EPS):
    """log ZINB(x; mu, theta, zero-inflation logits pi) element-wise."""
    theta = _broadcast_theta(theta)
    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)
    case_zero = F.softplus(pi_theta_log) - softplus_pi
    case_non_zero = (-softplus_pi + pi_theta_log
                     + x * (torch.log(mu + eps) - log_theta_mu_eps)
                     + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1))
    return (x < eps).to(mu.dtype) * case_zero + (x > eps).to(mu.dtype) * case_non_zero


def log_poisson(x, rate, eps: float = 0.0):
    """log Poisson(x; rate) element-wise (as ``torch.distributions.Poisson``)."""
    return torch.xlogy(x, rate) - rate - torch.lgamma(x + 1)


def log_normal(x, loc, scale):
    """log N(x; loc, scale) element-wise."""
    var = scale * scale
    return -((x - loc) ** 2) / (2 * var) - torch.log(scale) - 0.5 * math.log(2 * math.pi)


def log_mixture_nb(x, mu_1, mu_2, theta_1, theta_2, pi_logits, eps: float = EPS):
    """log of the two-component NB mixture used for totalVI protein counts.

    ``pi_logits`` are the logits of the *background* component (``mu_1``).
    With ``theta_2=None`` both components share ``theta_1``."""
    if theta_2 is not None:
        log_nb_1 = log_nb_positive(x, mu_1, theta_1, eps)
        log_nb_2 = log_nb_positive(x, mu_2, theta_2, eps)
    else:
        theta = _broadcast_theta(theta_1)
        log_theta_mu_1_eps = torch.log(theta + mu_1 + eps)
        log_theta_mu_2_eps = torch.log(theta + mu_2 + eps)
        lgamma_x_theta = torch.lgamma(x + theta)
        lgamma_theta = torch.lgamma(theta)
        lgamma_x_plus_1 = torch.lgamma(x + 1)
        log_theta = torch.log(theta + eps)
        log_nb_1 = (theta * (log_theta - log_theta_mu_1_eps)
                    + x * (torch.log(mu_1 + eps) - log_theta_mu_1_eps)
                    + lgamma_x_theta - lgamma_theta - lgamma_x_plus_1)
        log_nb_2 = (theta * (log_theta - log_theta_mu_2_eps)
                    + x * (torch.log(mu_2 + eps) - log_theta_mu_2_eps)
                    + lgamma_x_theta - lgamma_theta - lgamma_x_plus_1)
    logsumexp = torch.logsumexp(torch.stack((log_nb_1, log_nb_2 - pi_logits)), dim=0)
    return logsumexp - F.softplus(-pi_logits)


# ---------------------------------------------------------------------------
# per-cell sums using the sparsity of x
# ---------------------------------------------------------------------------
def _take(t, flat):
    """``t[row, col]`` as a flat gather (its backward is an index_add, much
    cheaper on GPUs than the accumulate-index_put of 2-D advanced indexing)."""
    return t.reshape(-1).index_select(0, flat)


def _flat(t, row, col, flat):
    return flat if flat is not None else row * t.shape[-1] + col


def _nnz_terms(mu, theta, log_theta_mu_eps, row, col, val, eps, flat):
    """sum over stored entries of x*(log(mu+eps)-log(theta+mu+eps)) + lgamma(x+theta)
    - lgamma(theta) - lgamma(x+1), per row."""
    mu_s = _take(mu, flat)
    if theta.shape[0] == 1:  # per-gene theta: lgamma(theta) on G values, then gathered
        th_s = theta[0].index_select(0, col)
        lg_th = torch.lgamma(theta[0]).index_select(0, col)
    else:
        th_s = _take(theta, flat)
        lg_th = torch.lgamma(th_s)
    with torch.no_grad():
        lg_x1 = torch.lgamma(val + 1)
    term = (val * (torch.log(mu_s + eps) - _take(log_theta_mu_eps, flat))
            + torch.lgamma(val + th_s) - lg_th - lg_x1)
    out = torch.zeros(mu.shape[0], dtype=term.dtype, device=mu.device)
    return out.index_add_(0, row, term)


def nb_rowsum(mu, theta, row, col, val, eps: float = EPS, flat=None):
    """Per-cell sum of log NB(x; mu, theta) given the non-zeros of x.

    ``row``/``col``/``val`` are the coordinates and values of the stored
    entries of x (no explicit zeros needed; explicit zeros are harmless);
    ``flat = row * n_genes + col`` may be passed precomputed."""
    theta = _broadcast_theta(theta)
    flat = _flat(mu, row, col, flat)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    dense = (theta * (torch.log(theta + eps) - log_theta_mu_eps)).sum(-1)
    return dense + _nnz_terms(mu, theta, log_theta_mu_eps, row, col, val, eps, flat)


def zinb_rowsum(mu, theta, pi, row, col, val, eps: float = EPS, flat=None):
    """Per-cell sum of log ZINB(x; mu, theta, pi) given the non-zeros of x.

    Entries must have ``val > eps`` (true non-zeros), as in the dense form
    where ``x < eps`` selects the zero branch."""
    theta = _broadcast_theta(theta)
    flat = _flat(mu, row, col, flat)
    softplus_pi = F.softplus(-pi)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (torch.log(theta + eps) - log_theta_mu_eps)
    case_zero = F.softplus(pi_theta_log) - softplus_pi
    # all entries start in the zero branch; stored entries switch to the non-zero branch
    # (non-zero branch = -softplus_pi + pi_theta_log + nnz terms)
    dense = case_zero.sum(-1)
    switch = _take(pi_theta_log, flat) - _take(softplus_pi, flat) - _take(case_zero, flat)
    out = dense.index_add(0, row, switch)
    return out + _nnz_terms(mu, theta, log_theta_mu_eps, row, col, val, eps, flat)


# ---------------------------------------------------------------------------
# Gaussian helpers
# ---------------------------------------------------------------------------
def kl_normal_standard(qm, qv):
    """KL(N(qm, qv) || N(0, 1)) summed over the last dimension (closed form)."""
    return 0.5 * (qv + qm * qm - 1.0 - torch.log(qv)).sum(-1)


def kl_normal_normal(qm, qv, pm, pv):
    """KL(N(qm, qv) || N(pm, pv)) element-wise (variances, not scales)."""
    return 0.5 * (torch.log(pv) - torch.log(qv) + (qv + (qm - pm) ** 2) / pv - 1.0)


# ---------------------------------------------------------------------------
# sampling (for posterior predictive / normalized expression)
# ---------------------------------------------------------------------------
def sample_nb(mu, theta):
    """Draw NB(mu, theta) via the Gamma-Poisson mixture."""
    theta = _broadcast_theta(theta).expand_as(mu)
    rate = torch._standard_gamma(theta.contiguous()) * (mu / theta)
    return torch.poisson(rate.clamp(max=1e8))


def sample_zinb(mu, theta, pi):
    x = sample_nb(mu, theta)
    drop = torch.rand_like(mu) < torch.sigmoid(pi)
    return x.masked_fill(drop, 0.0)


__all__ = [
    "log_nb_positive", "log_zinb_positive", "log_poisson", "log_normal", "log_mixture_nb",
    "nb_rowsum", "zinb_rowsum", "kl_normal_standard", "kl_normal_normal",
    "sample_nb", "sample_zinb",
]
