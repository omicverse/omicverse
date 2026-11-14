"""R-reference fixtures and scientific/input contracts for the RUVg subset."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("pyruvseq")
from omicverse.bulk import ruvg


FIXTURES = Path(__file__).parent / "fixtures" / "ruvg"
CASES = {
    "counts": {"k": 2},
    "unrounded": {"k": 2, "round_counts": False},
    "uncentered": {"k": 2, "center": False, "round_counts": False},
    "dropped": {"k": 3, "drop": 1, "round_counts": False},
    "logged": {"k": 2, "is_log": True},
    "offset": {"k": 2, "epsilon": 0.5, "round_counts": False},
    "rank_one": {"k": 3, "round_counts": False},
}


@pytest.mark.parametrize("case", CASES)
def test_pinned_r_matrix_method(case):
    values = np.loadtxt(FIXTURES / f"{case}_input.csv", delimiter=",")
    expected = np.loadtxt(FIXTURES / f"{case}_corrected.csv", delimiter=",")
    projection = np.loadtxt(FIXTURES / f"{case}_projection.csv", delimiter=",")
    if case == "rank_one":
        with pytest.warns(UserWarning, match="rank reduces"):
            result = ruvg(values, np.arange(8), **CASES[case])
    else:
        result = ruvg(values, np.arange(8), **CASES[case])
    w = result["W"]
    np.testing.assert_allclose(w @ w.T, projection, atol=1e-10, rtol=0)
    if case == "counts":
        np.testing.assert_array_equal(result["normalized_counts"], expected)
    else:
        np.testing.assert_allclose(result["normalized_counts"], expected, atol=1e-8, rtol=1e-10)
    residual = result["normalized_counts"].T if case == "logged" else None
    if residual is not None:
        np.testing.assert_allclose(w.T @ residual, 0, atol=1e-10)


def test_dataframe_axes_sparse_and_input_preservation():
    values = np.loadtxt(FIXTURES / "counts_input.csv", delimiter=",")
    frame = pd.DataFrame(values, index=[f"g{i}" for i in range(18)],
                         columns=[f"s{i}" for i in range(12)])
    before = frame.copy(deep=True)
    result = ruvg(frame, frame.index[:8], k=2)
    other = ruvg(sparse.csr_matrix(values), np.arange(8), k=2)
    pd.testing.assert_frame_equal(frame, before)
    assert result["normalized_counts"].index.equals(frame.index)
    assert result["normalized_counts"].columns.equals(frame.columns)
    assert result["W"].index.equals(frame.columns)
    np.testing.assert_array_equal(result["normalized_counts"], other["normalized_counts"])
    # Joint sample permutation must permute the answer, not change it.
    order = np.arange(11, -1, -1)
    permuted = ruvg(frame.iloc[:, order], frame.index[:8], k=2)
    np.testing.assert_array_equal(permuted["normalized_counts"], result["normalized_counts"].iloc[:, order])


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": True}, {"k": 12}, {"drop": 1},
                                   {"epsilon": 0}, {"tolerance": -1}, {"center": "yes"}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ruvg(np.ones((18, 12)), [0, 1], **kwargs)


@pytest.mark.parametrize("controls", [[], [0, 0], [-1], [18], [0.1], [False] * 18])
def test_invalid_controls(controls):
    with pytest.raises(ValueError):
        ruvg(np.ones((18, 12)), controls)


def test_no_control_signal_and_fractional_counts_fail():
    with pytest.raises(ValueError, match="rank"):
        ruvg(np.ones((18, 12)), [0, 1])
    with pytest.raises(ValueError, match="integer counts"):
        ruvg(np.ones((18, 12)) * 1.5, [0, 1])
    with pytest.raises(ValueError, match="finite"):
        ruvg(np.full((18, 12), np.nan), [0, 1])


def test_control_labels_are_not_positions():
    frame = pd.DataFrame(np.ones((18, 12)), index=[f"g{i}" for i in range(18)])
    with pytest.raises(ValueError, match="labels"):
        ruvg(frame, [0, 1])
