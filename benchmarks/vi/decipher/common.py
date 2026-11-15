"""Shared, implementation-agnostic metrics for the ov.vi.Decipher vs
scvi.external.Decipher validation.

``elbo_and_nll`` re-derives the single-sample ELBO and an importance-weighted
NLL estimate directly from the four Decipher sub-networks and ``theta`` --
the same maths in both ``ov.vi._decipher.DecipherModule`` and scvi-tools'
``DecipherPyroModule`` (identical attribute names: ``encoder_x_to_z``,
``encoder_zx_to_v``, ``decoder_v_to_z``, ``decoder_z_to_x``, ``theta``), so
calling it on either model's networks is a fair, single-codepath comparison
that does not depend on either training stack's own (possibly differently
scaled) validation bookkeeping.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

_LOG_2PI = math.log(2 * math.pi)


def _log_normal(x, loc, scale):
    return (-0.5 * ((x - loc) / scale) ** 2 - torch.log(scale) - 0.5 * _LOG_2PI).sum(-1)


def _one_sample(enc_x_z, enc_zx_v, dec_v_z, dec_z_x, theta, x, eps):
    xl = torch.log1p(x)
    z_loc, z_s = enc_x_z(xl)
    z_scale = F.softplus(z_s)
    z = z_loc + z_scale * torch.randn_like(z_loc)
    v_loc, v_s = enc_zx_v(torch.cat([z, xl], dim=-1))
    v_scale = F.softplus(v_s)
    v = v_loc + v_scale * torch.randn_like(v_loc)
    log_qz = _log_normal(z, z_loc, z_scale)
    log_qv = _log_normal(v, v_loc, v_scale)
    log_pv = (-0.5 * v ** 2 - 0.5 * _LOG_2PI).sum(-1)
    pz_loc, pz_s = dec_v_z(v)
    pz_scale = F.softplus(pz_s)
    log_pz = _log_normal(z, pz_loc, pz_scale)
    mu = F.softmax(dec_z_x(z), dim=-1)
    lib = x.sum(-1, keepdim=True)
    r = theta + eps
    logits = torch.log(lib * mu + eps) - torch.log(r)
    log_px = (x * F.logsigmoid(logits) + torch.lgamma(r + x) - torch.lgamma(r)
              - torch.lgamma(x + 1) + r * F.logsigmoid(-logits)).sum(-1)
    return log_px, log_pz - log_qz, log_pv - log_qv


@torch.no_grad()
def elbo_and_nll(networks, x, beta, eps=1e-5, n_iw=100):
    """Per-cell ELBO (nats, higher is better; the mean of ``n_iw`` single-sample
    MC draws -- still an unbiased ELBO estimator, just lower-variance than any
    one draw) and an importance-weighted NLL estimate (``logsumexp`` of the same
    ``n_iw`` draws), for one minibatch ``x`` (raw counts), given ``networks =
    (enc_x_z, enc_zx_v, dec_v_z, dec_z_x, theta)``."""
    elbo_terms, lw = [], []
    for _ in range(n_iw):
        log_px, wz, wv = _one_sample(*networks, x, eps)
        elbo_terms.append(log_px + wz + beta * wv)
        lw.append(log_px + wz + wv)
    elbo = torch.stack(elbo_terms, 0).mean(0)
    lw = torch.stack(lw, 0)
    nll = -(torch.logsumexp(lw, 0) - math.log(n_iw))
    return elbo, nll


@torch.no_grad()
def eval_networks(networks, X, device, beta, eps=1e-5, n_iw=100, batch_size=256):
    """Median per-cell ELBO / NLL of ``networks`` over the rows of ``X`` (numpy,
    raw counts), moving ``batch_size`` rows to ``device`` at a time.  The median
    (rather than the mean) is reported because the single/few-sample NB
    log-likelihood occasionally hits a catastrophic-outlier (z, v) draw for one
    cell in a few thousand -- a known heavy MC tail of this likelihood, seen for
    both ov.vi.Decipher and scvi.external.Decipher networks here, that a mean
    lets dominate the summary; the median (and the trimmed distribution) is the
    standard robust fix and does not change which model is being evaluated."""
    elbos, nlls = [], []
    for i in range(0, X.shape[0], batch_size):
        x = torch.as_tensor(np.asarray(X[i:i + batch_size]), dtype=torch.float32, device=device)
        e, n = elbo_and_nll(networks, x, beta, eps=eps, n_iw=n_iw)
        elbos.append(e.cpu().numpy())
        nlls.append(n.cpu().numpy())
    elbos, nlls = np.concatenate(elbos), np.concatenate(nlls)
    return float(np.median(elbos)), float(np.median(nlls))


def best_axis_spearman(v, pt):
    """``max_k |spearman(v[:, k], pt)|`` -- the rotation of the 2-D Decipher
    space is arbitrary, so the best-correlated axis is the one that matters."""
    from scipy.stats import spearmanr
    rhos = [spearmanr(v[:, k], pt).statistic for k in range(v.shape[1])]
    k = int(np.argmax(np.abs(rhos)))
    return float(rhos[k]), k
