"""Latent priors ``p(z)``.

Phase 1 ships the standard normal prior used by scVI / scANVI / totalVI.
A prior exposes ``kl(q_m, q_v, z)`` (per-cell KL of the Gaussian posterior
from the prior: analytic when available, otherwise a single-sample Monte-Carlo
estimate ``log q(z) - log p(z)``), ``log_prob(z)`` and ``sample(n)``.
Mixture priors (MoG, VampPrior) plug in through the same interface.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._distributions import kl_normal_standard


class Prior(nn.Module):
    """Interface of a latent prior."""

    analytic_kl = False

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:  # per cell (summed over dims)
        raise NotImplementedError

    def sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError

    def kl(self, q_m: torch.Tensor, q_v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Per-cell KL(q || p); Monte-Carlo with the posterior sample ``z`` by default."""
        log_q = (-0.5 * ((z - q_m) ** 2 / q_v + torch.log(q_v) + math.log(2 * math.pi))).sum(-1)
        return log_q - self.log_prob(z)


class StandardNormalPrior(Prior):
    """``p(z) = N(0, I)``."""

    analytic_kl = True

    def __init__(self, n_latent: int):
        super().__init__()
        self.n_latent = int(n_latent)

    def log_prob(self, z):
        return (-0.5 * (z * z + math.log(2 * math.pi))).sum(-1)

    def sample(self, n):
        return torch.randn(n, self.n_latent)

    def kl(self, q_m, q_v, z=None):
        return kl_normal_standard(q_m, q_v)


class VampPrior(Prior):
    """VampPrior (Tomczak & Welling, 2018) with trainable pseudo-inputs, as used
    by SysVI (Hrovatin et al., 2023, ``scvi.external.SysVI``).

    The prior is a mixture of ``n_components`` Gaussians whose parameters are
    the *posterior* of trainable pseudo-inputs under ``encoder`` (evaluated in
    eval mode, so dropout / batch-norm statistics do not leak into the prior).
    ``encoder`` is any module called as
    ``encoder(x, *cats, cont=None, sample=False) -> (q_m, q_v, z)`` (the
    model's own encoder, e.g. :class:`~omicverse.vi._nn.Encoder` or the
    ``GaussianBlock`` used by :class:`~omicverse.vi.SysVI`); it is kept as a
    *reference*, so its parameters are trained jointly with the rest of the
    model (and, like scvi-tools, appear twice in ``state_dict()`` — once
    under the encoder's own path, once under the prior's — harmlessly, since
    ``nn.Module.parameters()`` already deduplicates by identity).

    Mirrors ``scvi.external.sysvi._priors.VampPrior`` exactly, including its
    per-dimension (not jointly-summed) mixture: :meth:`log_prob` returns one
    log-density per latent dimension (``N x L``, *not* summed over ``L``) and
    :meth:`kl` sums ``log q - log p`` over ``L`` only at the end, so the
    mixture is really a product, over latent dimensions, of 1-D Gaussian
    mixtures rather than a single joint mixture in ``L`` dimensions.

    Arguments:
        n_components: Number of pseudo-inputs / mixture components.
        encoder: The model's encoder (duck-typed as above).
        x: Pseudo-input expression, ``(n_components, n_input)``.
        n_cat_list: Categories per categorical covariate, batch (system) first
            (as passed to ``encoder``).
        batch_index: Pseudo-input batch codes, ``(n_components,)`` or
            ``(n_components, 1)``.
        cat_list: Pseudo-input codes of the other categorical covariates
            (each ``(n_components,)``).
        cont: Pseudo-input continuous covariates, ``(n_components, n_cont)``
            (or ``None``).
        trainable: Pseudo-inputs are learned (``True``) or fixed at their
            initial value (``False``).
    """

    analytic_kl = False

    def __init__(self, n_components: int, encoder, x: torch.Tensor, n_cat_list,
                 batch_index: torch.Tensor, cat_list, cont: torch.Tensor | None = None,
                 trainable: bool = True):
        super().__init__()
        self.n_components = int(n_components)
        self.encoder = encoder
        self.u = nn.Parameter(x, requires_grad=trainable)
        cats = [batch_index] + list(cat_list)
        self.u_cat = nn.ParameterList([
            nn.Parameter(F.one_hot(c.reshape(-1).long(), int(n)).float(), requires_grad=trainable)
            for c, n in zip(cats, n_cat_list)
        ])
        self.u_cont = None if cont is None else nn.Parameter(cont, requires_grad=trainable)
        self.w = nn.Parameter(torch.zeros(self.n_components, 1, 1))

    def _pseudoinput_params(self):
        """Posterior mean / scale of each pseudo-input under ``encoder`` (eval mode).

        ``scale`` here is the *scale* (std) of the encoder's posterior (as
        anywhere else), but it is what ``scvi.external.sysvi``'s own
        ``get_params`` returns under the name "variance" and callers then
        take *another* square root of (``Normal(m_p, v_p.sqrt())`` in its
        ``log_prob``): the per-component std actually used by the mixture is
        the square root of this scale, not the scale itself. Kept bit-for-bit
        (see :meth:`log_prob`) rather than "fixed", since the goal is parity
        with the reference, including its quirks."""
        was = self.encoder.training
        self.encoder.eval()
        cats = [torch.multinomial(c, 1).squeeze(-1) for c in self.u_cat]
        batch_index, cats = cats[0], cats[1:]
        q_m, q_v, _ = self.encoder(self.u, batch_index, *cats, cont=self.u_cont, sample=False)
        self.encoder.train(was)
        return q_m, q_v.sqrt() + 1e-12  # (K, L) mean, scale

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Per-(cell, latent dim) log density of the mixture, **not** summed
        over latent dimensions (see the class docstring)."""
        z = z.unsqueeze(0)  # 1 x N x L
        m_p, scale_p = self._pseudoinput_params()
        m_p, scale_p = m_p.unsqueeze(1), scale_p.unsqueeze(1)  # K x 1 x L
        s_p = scale_p.sqrt()  # the extra sqrt scvi-tools' VampPrior.log_prob applies
        log_w = torch.log_softmax(self.w, dim=0)  # K x 1 x 1
        log_prob = (-0.5 * ((z - m_p) / s_p) ** 2 - torch.log(s_p)
                   - 0.5 * math.log(2 * math.pi)) + log_w  # K x N x L
        return torch.logsumexp(log_prob, dim=0)  # N x L

    def sample(self, n):  # pragma: no cover - not needed by SysVI training/inference
        raise NotImplementedError("VampPrior.sample is not used by SysVI")

    def kl(self, q_m: torch.Tensor, q_v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``KL(q(z) || p(z))`` exactly as ``scvi.external.sysvi``: elementwise
        ``log q - log p`` (both un-summed over latent dims), summed last."""
        s_q = q_v.sqrt() + 1e-12
        log_q = -0.5 * ((z - q_m) / s_q) ** 2 - torch.log(s_q) - 0.5 * math.log(2 * math.pi)
        return (log_q - self.log_prob(z)).sum(-1)


PRIORS = {"normal": StandardNormalPrior}


def make_prior(name: str, n_latent: int, **kwargs) -> Prior:
    try:
        return PRIORS[name](n_latent, **kwargs)
    except KeyError:
        raise ValueError(f"prior must be one of {sorted(PRIORS)}; got {name!r}") from None


__all__ = ["Prior", "StandardNormalPrior", "VampPrior", "make_prior"]
