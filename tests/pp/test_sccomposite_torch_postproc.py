"""The torch/GPU post-processing of sccomposite (prob_k0_rna / reliability_rna)
must match the old scipy+float128+3D-array implementation. The rewrite is a
pure performance/OOM change — the doublet calls and consistency are unchanged.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scipy.stats import gamma as _gamma, poisson as _poisson
from scipy.special import logsumexp as _logsumexp

from omicverse.pp import _sccomposite as sc


# ---- self-contained scipy reference (the pre-rewrite maths) ----
def _ref_log_joint(data, theta, alpha, beta, decay, k0):
    a = alpha * (1 + k0 / (1 + np.exp(-decay)))
    lc = np.log(_gamma.pdf(data, a, loc=0, scale=1 / beta))
    return lc.sum(axis=1) + np.log(_poisson.pmf(k0, theta))


def _ref_prob_k0(data, theta, alpha, beta, decay, k0, k=3):
    stacked = np.stack([_ref_log_joint(data, theta, alpha, beta, decay, i)
                        for i in range(k)], axis=1)
    return np.exp(_ref_log_joint(data, theta, alpha, beta, decay, k0)
                  - _logsumexp(stacked, axis=1))


def _ref_reliability(data, theta, alpha, beta, decay, k=3):
    prob_singlet = _ref_prob_k0(data, theta, alpha, beta, decay, 0, k)
    pred = (1 - prob_singlet) > 0.5
    one_ks = np.empty((data.shape[0], data.shape[1], k))
    for i in range(k):
        a = alpha * (1 + i / (1 + np.exp(-decay)))
        one_ks[:, :, i] = np.log(_gamma.pdf(data, a, loc=0, scale=1 / beta))
    rel = 1 - np.exp(one_ks[:, :, 0] - _logsumexp(one_ks, axis=2))
    rel[pred, :] = np.where(rel[pred, :] > 0.5, 1, 0)
    rel[~pred, :] = np.where(rel[~pred, :] < 0.5, 1, 0)
    rel = rel.sum(axis=1) / data.shape[1]
    out = np.zeros((2, data.shape[0]))
    out[0], out[1] = rel, np.where(rel <= 0.5, 1, 0)
    return out


def _fixture(seed=0, n_cells=800, n_genes=60):
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(1.5, 6.0, n_genes)
    beta = rng.uniform(0.5, 2.0, n_genes)
    theta = 0.3
    decay = 0.4
    data = rng.gamma(alpha, 1 / beta, size=(n_cells, n_genes)) + 1e-4
    t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64))
    return data, alpha, beta, theta, decay, t


def test_prob_k0_matches_scipy():
    data, alpha, beta, theta, decay, t = _fixture()
    for k0 in (0, 1, 2):
        got = sc.prob_k0_rna(t(data), t(theta), t(alpha), t(beta), t(decay), k0, k=3)
        ref = _ref_prob_k0(data, theta, alpha, beta, decay, k0, k=3)
        assert np.allclose(got, ref, atol=1e-8), f"prob_k0 mismatch at k0={k0}"


def test_reliability_and_classification_match_scipy():
    data, alpha, beta, theta, decay, t = _fixture(seed=1)
    got = sc.reliability_rna(t(data), t(theta), t(alpha), t(beta), t(decay), k=3)
    ref = _ref_reliability(data, theta, alpha, beta, decay, k=3)
    assert np.allclose(got[0], ref[0], atol=1e-8)     # consistency score
    assert np.array_equal(got[1], ref[1])             # outlier flags

    # doublet classification (from prob_k0) must be identical
    ps_new = sc.prob_k0_rna(t(data), t(theta), t(alpha), t(beta), t(decay), 0, k=3)
    ps_ref = _ref_prob_k0(data, theta, alpha, beta, decay, 0, k=3)
    assert np.array_equal((1 - ps_new) > 0.5, (1 - ps_ref) > 0.5)
