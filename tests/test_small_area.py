"""Tests for omicverse.tl._small_area (small-area estimation toolkit)."""
import math

import numpy as np
import pytest

from omicverse.tl._small_area import (crps_ensemble, energy_score, pit_histogram_tv,
                                      sae_benchmark, sae_grouped_model,
                                      sae_longitudinal_design, sae_reference_prior)


def _population(rng, beta, n_groups=3, per_group=12, dev_sd=0.6, area_sd=0.3, waves=3):
    """Areas with longitudinal auxiliaries; each group's coefficients deviate from beta."""
    n = n_groups * per_group
    aux = rng.normal(size=(n, waves, 2))
    X = sae_longitudinal_design(aux, n_outputs=1, pairwise=False, squares=False,
                                squash=False, time=False)
    groups = np.repeat(np.arange(n_groups), per_group)
    theta = np.empty((n, waves))
    for g in range(n_groups):
        bg = beta + dev_sd * rng.normal(size=beta.size)
        m = groups == g
        theta[m] = X[m] @ bg + area_sd * rng.normal(size=(m.sum(), waves))
    return X, theta, groups


def test_crps_matches_gaussian_closed_form():
    rng = np.random.default_rng(0)
    draws = rng.normal(0.3, 1.0, size=(200000, 1))
    y = np.array([1.1])
    z = (y - 0.3) / 1.0
    pdf = math.exp(-z[0] ** 2 / 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + math.erf(z[0] / math.sqrt(2)))
    exact = z[0] * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi)
    assert crps_ensemble(draws, y)[0] == pytest.approx(exact, abs=5e-3)
    es = energy_score(draws[:2000, None, :], y[None, :])
    assert es.shape == (1,) and es[0] > 0


def test_pit_tv_flags_overconfidence():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=4000)
    good = rng.normal(size=(255, 4000))
    narrow = 0.3 * rng.normal(size=(255, 4000))
    assert pit_histogram_tv(good, truth)[0] < 0.03
    assert pit_histogram_tv(narrow, truth)[0] > 0.2


def test_reference_prior_recovers_coefficients():
    rng = np.random.default_rng(2)
    beta = np.array([0.5, 1.0, -0.7])
    pops = [_population(rng, beta, dev_sd=0.0) for _ in range(4)]
    prior = sae_reference_prior([p[0] for p in pops], [p[1] for p in pops], ridge=(0.01, 1.0))
    assert np.allclose(prior.beta0, beta, atol=0.08)
    assert np.allclose(np.diag(prior.resid_cov), 0.09, rtol=0.35)


def test_grouped_model_beats_prior_and_is_calibrated():
    rng = np.random.default_rng(3)
    beta = np.array([0.5, 1.0, -0.7])
    refs = [_population(rng, beta) for _ in range(5)]
    prior = sae_reference_prior([p[0] for p in refs], [p[1] for p in refs], ridge=1.0)
    crps_model, crps_prior, all_d, all_t = [], [], [], []
    for rep in range(6):
        X, theta, groups = _population(rng, beta)
        n, q = theta.shape
        unsampled = np.zeros(n, bool)
        for g in range(3):                      # 4 of 12 areas per group unsampled
            unsampled[np.flatnonzero(groups == g)[:4]] = True
        D = np.repeat((0.05 ** 2 * np.eye(q))[None], n, 0)
        direct = theta + 0.05 * rng.normal(size=theta.shape)
        direct[unsampled] = np.nan
        res = sae_grouped_model(direct, D, X, groups, prior, holdout=np.arange(n) // 4,
                                n_draws=256, seed=rep)
        flat = sae_grouped_model(direct, D, X, groups, prior, lam=1e8, cov_scale=1.0,
                                 n_draws=256, seed=rep)
        crps_model.append(crps_ensemble(res.draws[:, unsampled], theta[unsampled]).mean())
        crps_prior.append(crps_ensemble(flat.draws[:, unsampled], theta[unsampled]).mean())
        all_d.append(res.draws[:, unsampled].reshape(256, -1))
        all_t.append(theta[unsampled].ravel())
        assert res.lam < 1e6                    # group deviations are real -> adapted
    assert np.mean(crps_model) < 0.8 * np.mean(crps_prior)
    tv, _ = pit_histogram_tv(np.concatenate(all_d, 1), np.concatenate(all_t))
    assert tv < 0.15


def test_benchmark_holds_in_every_draw():
    rng = np.random.default_rng(4)
    draws = rng.normal(size=(64, 10, 2))
    groups = np.repeat([0, 1], 5)
    w = rng.uniform(1, 3, size=(10, 2))
    targets = {(0, 0): 1.0, (0, 1): -2.0, (1, 0): 0.5}
    for alloc in ("weights", "covariance", "variance"):
        out = sae_benchmark(draws, groups, w, targets, allocation=alloc)
        for (g, k), t in targets.items():
            m = groups == g
            ww = w[m, k] / w[m, k].sum()
            assert np.allclose(out[:, m, k] @ ww, t)
        assert np.allclose(out[:, groups == 1, 1], draws[:, groups == 1, 1])   # free key untouched


def test_design_shapes_and_blocks():
    aux = np.random.default_rng(5).normal(size=(7, 4, 3))
    X = sae_longitudinal_design(aux, n_outputs=2, summaries=True)
    assert X.shape[0] == 7 and X.shape[1] == 8
    pb = X.shape[2] // 2
    assert np.all(X[:, 0, pb:] == 0) and np.all(X[:, 1, :pb] == 0)   # output blocks separate


def test_vendored_source_runs_without_omicverse(tmp_path):
    import os, subprocess, sys
    from omicverse.tl._small_area import sae_source
    text = sae_source(tmp_path / "sae_core.py")
    assert "register_function" not in text and "omicverse" not in text.split("\n", 1)[1]
    script = tmp_path / "run.py"
    script.write_text("import numpy as np\nimport sae_core as s\n"
                      "d = np.random.default_rng(0).normal(size=(50, 3, 2))\n"
                      "print(s.crps_ensemble(d, np.zeros((3, 2))).shape)\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "(3, 2)" in out.stdout
