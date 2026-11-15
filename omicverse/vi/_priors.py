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


PRIORS = {"normal": StandardNormalPrior}


def make_prior(name: str, n_latent: int, **kwargs) -> Prior:
    try:
        return PRIORS[name](n_latent, **kwargs)
    except KeyError:
        raise ValueError(f"prior must be one of {sorted(PRIORS)}; got {name!r}") from None


__all__ = ["Prior", "StandardNormalPrior", "make_prior"]
