"""Unit tests for ``omicverse.utils.gpuex.scipy.rankdata``.

We compare against ``scipy.stats.rankdata(..., method='average')`` on a
range of shapes / distributions; the GPU path must agree to floating-
point precision (we test 0-difference on fp64 since both backends use
the same searchsorted-derived formula).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as sts


def test_module_imports():
    from omicverse.utils.gpuex.scipy import rankdata

    assert callable(rankdata)


def test_method_not_implemented():
    from omicverse.utils.gpuex.scipy import rankdata

    with pytest.raises(NotImplementedError, match="method"):
        rankdata(np.array([1.0, 2.0]), method="min")


def test_nan_policy_raise():
    from omicverse.utils.gpuex.scipy import rankdata

    with pytest.raises(ValueError, match="NaN"):
        rankdata(np.array([1.0, np.nan, 2.0]), axis=0, nan_policy="raise")


@pytest.mark.parametrize(
    "shape,seed",
    [
        ((5,), 0),
        ((1, 20), 1),
        ((3, 17), 2),
        ((10, 200), 3),
        ((30, 500), 4),
    ],
)
def test_rankdata_matches_scipy(shape, seed):
    """For every shape, GPU output must be bit-identical to scipy on fp64."""
    from omicverse.utils.gpuex.scipy import rankdata

    rng = np.random.default_rng(seed)
    # Mix integer-valued (tie-heavy) and continuous floats — both produce
    # the same rank arithmetic; the integer case exercises the average-tie
    # branch hard.
    a = rng.poisson(2.0, size=shape).astype(float)
    if a.ndim == 1:
        cpu = sts.rankdata(a, method="average")
        gpu = rankdata(a, axis=0)
    else:
        cpu = sts.rankdata(a, axis=1, method="average")
        gpu = rankdata(a, axis=1)
    np.testing.assert_array_equal(cpu, gpu)


def test_rankdata_continuous_no_ties():
    """Continuous floats (no ties) — average == ordinal, should still match."""
    from omicverse.utils.gpuex.scipy import rankdata

    rng = np.random.default_rng(0)
    a = rng.standard_normal(size=(4, 50))
    cpu = sts.rankdata(a, axis=1, method="average")
    gpu = rankdata(a, axis=1)
    np.testing.assert_array_equal(cpu, gpu)


def test_rankdata_descending_via_negation():
    """Common usage in AUCell / UCell: pass -mat to get descending ranks."""
    from omicverse.utils.gpuex.scipy import rankdata

    rng = np.random.default_rng(0)
    a = rng.poisson(3.0, size=(6, 100)).astype(float)
    cpu = sts.rankdata(-a, axis=1, method="average")
    gpu = rankdata(-a, axis=1)
    np.testing.assert_array_equal(cpu, gpu)
