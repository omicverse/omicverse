"""Compensation. Every failure mode here is silent, which is why they are all
pinned: uncompensated data looks like a real double-positive population, and
double-compensated data looks like a real single-positive one."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import omicverse.flow as fl

DETECTORS = ["FITC-A", "PE-A", "APC-A"]
SPILL = pd.DataFrame(
    [[1.00, 0.15, 0.02],
     [0.08, 1.00, 0.05],
     [0.01, 0.06, 1.00]],
    index=DETECTORS, columns=DETECTORS,
)


def make(n=2000, seed=0, with_spill=True, markers=None):
    rng = np.random.default_rng(seed)
    true = np.abs(rng.normal(5e3, 1e3, (n, 3)))
    obs = true @ SPILL.to_numpy() if with_spill else true
    scatter = np.abs(rng.normal(6e4, 1e4, (n, 1)))
    X = np.column_stack([scatter, obs]).astype(np.float32)
    idx = ["FSC-A"] + (markers or DETECTORS)
    var = pd.DataFrame({"channel": ["FSC-A"] + DETECTORS, "marker": [""] + (markers or ["", "", ""])},
                       index=idx)
    a = ad.AnnData(X, var=var)
    a.uns["fcs"] = {"spillover": SPILL, "compensated": False}
    return a, true


def test_compensation_recovers_the_true_signal():
    """The only test that really matters: observed = true @ S, so compensating
    must return `true`."""
    a, true = make()
    fl.compensate(a)
    assert np.allclose(np.asarray(a.X)[:, 1:], true, rtol=1e-3, atol=1.0)


def test_scatter_is_never_touched():
    """The matrix names only fluorescence detectors. FSC/SSC are not
    fluorescence and must not be dragged through the inverse."""
    a, _ = make()
    before = np.asarray(a.X)[:, 0].copy()
    fl.compensate(a)
    assert np.allclose(np.asarray(a.X)[:, 0], before)


def test_compensating_twice_is_refused():
    """Double compensation is worse than none and NOTHING in the numbers says
    it happened — they stay plausible. So it has to be refused explicitly."""
    a, _ = make()
    fl.compensate(a)
    with pytest.raises(ValueError, match="already compensated"):
        fl.compensate(a)


def test_a_reader_flagged_file_is_also_refused():
    """`ov.io.read_fcs(compensated=True)` already applied the matrix; this must
    notice that, not just its own flag."""
    a, _ = make()
    a.uns["fcs"]["compensated"] = True
    with pytest.raises(ValueError, match="already compensated"):
        fl.compensate(a)


def test_the_original_values_are_kept():
    a, _ = make()
    before = np.asarray(a.X).copy()
    fl.compensate(a)
    assert np.allclose(np.asarray(a.layers["uncompensated"]), before)


def test_what_was_done_is_recorded():
    a, _ = make()
    fl.compensate(a)
    assert a.uns["flow"]["compensated"] is True
    assert a.uns["flow"]["compensated_channels"] == DETECTORS


def test_a_bare_array_is_refused():
    """An unlabelled matrix cannot say which detector each row is, and guessing
    the order is how channels get compensated against the wrong dye."""
    a, _ = make()
    with pytest.raises(TypeError, match="labelled DataFrame"):
        fl.compensate(a, spillover=SPILL.to_numpy())


def test_missing_matrix_says_why():
    a, _ = make(with_spill=False)
    a.uns["fcs"]["spillover"] = None
    with pytest.raises(ValueError, match="no spillover matrix"):
        fl.compensate(a)


def test_a_matrix_naming_an_absent_detector_is_refused():
    a, _ = make()
    bad = SPILL.rename(index={"APC-A": "BV421-A"}, columns={"APC-A": "BV421-A"})
    with pytest.raises(KeyError, match="BV421-A"):
        fl.compensate(a, spillover=bad)


def test_matrix_columns_match_markers_too():
    """The matrix names detectors; read_fcs indexes var by MARKER. Both have to
    resolve, or a matrix straight out of a file fails on the AnnData that same
    file produced."""
    a, true = make(markers=["CD3", "CD4", "CD8"])
    fl.compensate(a)
    assert np.allclose(np.asarray(a.X)[:, 1:], true, rtol=1e-3, atol=1.0)


def test_compensation_matrix_type_is_explicit():
    """Passing an already-inverted matrix as if it were spillover produces
    plausible garbage, not an error — so the caller must say which it is."""
    a, true = make()
    comp = fl.spillover_to_compensation(SPILL)
    fl.compensate(a, spillover=comp, matrix_type="compensation")
    assert np.allclose(np.asarray(a.X)[:, 1:], true, rtol=1e-3, atol=1.0)


def test_spillover_to_compensation_is_the_inverse():
    comp = fl.spillover_to_compensation(SPILL)
    assert np.allclose(SPILL.to_numpy() @ comp.to_numpy(), np.eye(3), atol=1e-12)
    assert list(comp.columns) == DETECTORS


def test_singular_matrix_says_so():
    bad = SPILL.copy()
    bad.iloc[1] = bad.iloc[0]
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        fl.spillover_to_compensation(bad)


def test_not_inplace_leaves_the_original_alone():
    a, _ = make()
    before = np.asarray(a.X).copy()
    out = fl.compensate(a, inplace=False)
    assert np.allclose(np.asarray(a.X), before)
    assert not np.allclose(np.asarray(out.X), before)


def test_spreading_matrix_finds_the_pair_that_spreads():
    """Spreading is what compensation CANNOT remove. Built so FITC spills into
    PE with real photon noise; the SSM must show that pair and not its reverse.
    """
    rng = np.random.default_rng(0)
    n = 20000
    controls = {}
    for stain in ("FITC-A", "PE-A"):
        x = np.abs(rng.normal(2e4, 8e3, n))
        other = rng.normal(0, 50, n)
        if stain == "FITC-A":
            # variance grows with the spilling channel's intensity
            other = other + rng.normal(0, np.sqrt(np.maximum(x, 1)) * 0.9)
        cols = {"FITC-A": x, "PE-A": other} if stain == "FITC-A" else {"FITC-A": other, "PE-A": x}
        X = np.column_stack([cols["FITC-A"], cols["PE-A"]]).astype(np.float32)
        controls[stain] = ad.AnnData(X, var=pd.DataFrame(index=["FITC-A", "PE-A"]))
    ssm = fl.spillover_spreading_matrix(controls)
    assert ssm.loc["FITC-A", "PE-A"] > 3 * max(ssm.loc["PE-A", "FITC-A"], 1e-9)
