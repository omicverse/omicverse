"""Tests for ``ov.io.read_fcs`` — the general Flow Cytometry Standard reader.

The FCS fixture is WRITTEN BY THE TEST (flowio can create files as well as read
them), so these run in CI without shipping a binary and without depending on a
vendor sample. That matters: the two things this reader exists to preserve —
the $PnN/$PnS distinction and $SPILLOVER — only appear in a file that actually
carries them, and a hand-waved fixture would test neither.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

HAS_FLOWIO = importlib.util.find_spec("flowio") is not None
HAS_PYTOMETRY = importlib.util.find_spec("pytometry") is not None

pytestmark = pytest.mark.skipif(not HAS_FLOWIO, reason="flowio not installed")

from omicverse.io.cytometry import parse_spillover  # noqa: E402

# 4 fluorescence detectors carrying named antibodies, plus scatter with none.
CHANNELS = [
    ("FSC-A", None), ("SSC-A", None),
    ("FITC-A", "CD3"), ("PE-A", "CD4"), ("APC-A", "CD8"), ("PerCP-A", "CD19"),
]
SPILL_NAMES = ["FITC-A", "PE-A", "APC-A", "PerCP-A"]
SPILL = np.array([
    [1.00, 0.08, 0.01, 0.00],
    [0.11, 1.00, 0.05, 0.01],
    [0.02, 0.06, 1.00, 0.09],
    [0.00, 0.01, 0.07, 1.00],
])


@pytest.fixture
def fcs(tmp_path):
    import flowio

    rng = np.random.default_rng(0)
    n = 2000
    data = np.abs(rng.normal(5e4, 1e4, size=(n, len(CHANNELS)))).astype(np.float32)
    opt = {f"$P{i + 1}S": s for i, (_, s) in enumerate(CHANNELS) if s}
    opt["$SPILLOVER"] = ",".join(
        [str(len(SPILL_NAMES))] + SPILL_NAMES + [f"{v:g}" for v in SPILL.flatten()]
    )
    path = tmp_path / "sample_a.fcs"
    with open(path, "wb") as fh:
        flowio.create_fcs(
            fh, data.flatten().tolist(),
            channel_names=[c for c, _ in CHANNELS],
            opt_channel_names=[s for _, s in CHANNELS],
            metadata_dict=opt,
        )
    return str(path)


@pytest.fixture
def fcs_no_spill(tmp_path):
    """A file with no $SPILLOVER — completely normal (unstained, CyTOF)."""
    import flowio

    rng = np.random.default_rng(1)
    data = np.abs(rng.normal(1e3, 2e2, size=(500, 3))).astype(np.float32)
    path = tmp_path / "plain.fcs"
    with open(path, "wb") as fh:
        flowio.create_fcs(fh, data.flatten().tolist(),
                          channel_names=["FSC-A", "SSC-A", "Time"])
    return str(path)


# ── the two things the EV reader threw away ─────────────────────────────────

def test_detector_and_marker_are_kept_apart(fcs):
    """$PnN vs $PnS is THE classic FCS mistake: the same antibody sits on a
    different detector between panels. Both must survive."""
    from omicverse.io import read_fcs

    a = read_fcs(fcs)
    assert list(a.var["channel"]) == [c for c, _ in CHANNELS]
    assert list(a.var["marker"]) == [s or "" for _, s in CHANNELS]
    # Indexed by the marker where one exists, else the detector.
    assert list(a.var.index) == ["FSC-A", "SSC-A", "CD3", "CD4", "CD8", "CD19"]


def test_spillover_is_parsed_into_a_labelled_matrix(fcs):
    from omicverse.io import read_fcs

    a = read_fcs(fcs)
    spill = a.uns["fcs"]["spillover"]
    assert isinstance(spill, pd.DataFrame)
    assert list(spill.columns) == SPILL_NAMES
    assert np.allclose(spill.to_numpy(), SPILL)


def test_a_file_without_spillover_is_not_an_error(fcs_no_spill):
    from omicverse.io import read_fcs

    a = read_fcs(fcs_no_spill)
    assert a.uns["fcs"]["spillover"] is None
    assert a.shape == (500, 3)


# ── parse_spillover, directly ───────────────────────────────────────────────

def test_parse_spillover_round_trips():
    text = ",".join(["2", "A", "B", "1", "0.05", "0.02", "1"])
    m = parse_spillover(text, ["A", "B"])
    assert list(m.columns) == ["A", "B"]
    assert np.allclose(m.to_numpy(), [[1, 0.05], [0.02, 1]])


def test_parse_spillover_maps_detector_indexes_to_names():
    """Some vendors write 1-based detector INDEXES instead of names."""
    m = parse_spillover("2,1,2,1,0.05,0.02,1", ["FITC-A", "PE-A"])
    assert list(m.columns) == ["FITC-A", "PE-A"]


@pytest.mark.parametrize("bad", ["", "not,a,matrix", "3,A,B,1,0", "0", "abc"])
def test_malformed_spillover_returns_None_rather_than_raising(bad):
    # A missing or broken matrix must degrade to "no compensation available",
    # never take down the read of an otherwise fine file.
    assert parse_spillover(bad, ["A", "B"]) is None


# ── selection ───────────────────────────────────────────────────────────────

def test_channels_can_be_named_by_either_vocabulary(fcs):
    from omicverse.io import read_fcs

    by_marker = read_fcs(fcs, channels=["CD3", "CD4"])
    by_detector = read_fcs(fcs, channels=["FITC-A", "PE-A"])
    assert by_marker.shape == by_detector.shape == (2000, 2)
    assert np.allclose(np.asarray(by_marker.X), np.asarray(by_detector.X))


def test_markers_only_drops_scatter(fcs):
    from omicverse.io import read_fcs

    a = read_fcs(fcs, markers_only=True)
    assert list(a.var.index) == ["CD3", "CD4", "CD8", "CD19"]


def test_an_unknown_channel_is_a_loud_error(fcs):
    from omicverse.io import read_fcs

    with pytest.raises(KeyError, match="not in this file"):
        read_fcs(fcs, channels=["CD3", "CD400"])


def test_sample_defaults_to_the_filename(fcs):
    from omicverse.io import read_fcs

    assert read_fcs(fcs).obs["sample"].unique().tolist() == ["sample_a"]
    assert read_fcs(fcs, sample="donor7").obs["sample"].unique().tolist() == ["donor7"]


# ── compensation ────────────────────────────────────────────────────────────

def test_compensation_is_off_by_default_and_recorded_when_on(fcs):
    from omicverse.io import read_fcs

    raw = read_fcs(fcs)
    comp = read_fcs(fcs, compensated=True)
    assert raw.uns["fcs"]["compensated"] is False
    assert comp.uns["fcs"]["compensated"] is True
    assert not np.allclose(np.asarray(raw.X), np.asarray(comp.X))


def test_compensation_leaves_scatter_untouched(fcs):
    """The spillover matrix names only fluorescence detectors; FSC/SSC must not
    be dragged through it."""
    from omicverse.io import read_fcs

    raw = read_fcs(fcs)
    comp = read_fcs(fcs, compensated=True)
    assert np.allclose(np.asarray(raw.X)[:, :2], np.asarray(comp.X)[:, :2])


def test_compensation_without_a_matrix_refuses(fcs_no_spill):
    from omicverse.io import read_fcs

    with pytest.raises(ValueError, match="no \\$SPILLOVER"):
        read_fcs(fcs_no_spill, compensated=True)


def test_subsetting_does_not_change_the_compensation(fcs):
    """Compensation is defined over the full detector set. Subsetting first
    would silently compensate against a different matrix than was recorded."""
    from omicverse.io import read_fcs

    full = read_fcs(fcs, compensated=True)
    subset = read_fcs(fcs, compensated=True, channels=["CD3"])
    col = list(full.var.index).index("CD3")
    assert np.allclose(np.asarray(full.X)[:, col], np.asarray(subset.X)[:, 0])


# ── interop with pytometry (the reason the schema is what it is) ────────────

@pytest.mark.skipif(not HAS_PYTOMETRY, reason="pytometry not installed")
def test_the_result_is_a_drop_in_for_pytometry(fcs):
    """Matching pytometry's `var` schema was NOT sufficient — `pp.compensate`
    reads `uns['meta']['spill']` and wants the parsed matrix there. Both halves
    are pinned here because the first two attempts at this shipped a var schema
    that looked right and a uns that did not work."""
    import pytometry as pm

    from omicverse.io import read_fcs

    a = read_fcs(fcs)
    pm.pp.compensate(a.copy())
    pm.tl.normalize_logicle(a.copy())
    pm.tl.normalize_arcsinh(a.copy(), cofactor=150)
    pm.pp.split_signal(a.copy(), var_key="channel")


@pytest.mark.skipif(not HAS_PYTOMETRY, reason="pytometry not installed")
def test_our_compensation_agrees_with_pytometry_exactly(fcs):
    import pytometry as pm

    from omicverse.io import read_fcs

    ours = read_fcs(fcs, compensated=True)
    theirs = read_fcs(fcs)
    pm.pp.compensate(theirs)
    cols = [list(ours.var["channel"]).index(c) for c in SPILL_NAMES]
    assert np.allclose(np.asarray(ours.X)[:, cols], np.asarray(theirs.X)[:, cols])


@pytest.mark.skipif(not HAS_PYTOMETRY, reason="pytometry not installed")
def test_var_schema_matches_pytometrys_own_reader(fcs):
    import pytometry as pm

    from omicverse.io import read_fcs

    ours = read_fcs(fcs)
    theirs = pm.io.read_fcs(fcs)
    assert list(ours.var.columns) == list(theirs.var.columns)
    assert list(ours.var.index) == list(theirs.var.index)
    assert list(ours.var["channel"]) == list(theirs.var["channel"])


# ── the EV wrapper still works ──────────────────────────────────────────────

def test_ev_reader_keeps_its_contract_and_gains_the_metadata(fcs):
    """`ov.single.ev.read_fcs` now delegates here. Its own conventions must be
    unchanged for existing callers — and it should no longer THROW AWAY the
    spillover matrix and the channel/marker split, which it used to."""
    from omicverse.single.ev.io import read_fcs as ev_read_fcs

    a = ev_read_fcs(fcs)
    assert a.obs_names[0] == "event000000"
    assert a.uns["ev"] == {"value_type": "intensity", "platform": "FCS"}
    assert a.obs["sample"].unique().tolist() == ["sample1"]
    # newly preserved
    assert a.uns["fcs"]["spillover"].shape == (4, 4)
    assert list(a.var["channel"]) == [c for c, _ in CHANNELS]


def test_ev_reader_still_accepts_marker_cols(fcs):
    from omicverse.single.ev.io import read_fcs as ev_read_fcs

    assert ev_read_fcs(fcs, marker_cols=["CD3", "CD4"]).shape == (2000, 2)


def test_missing_file_is_a_FileNotFoundError(tmp_path):
    from omicverse.io import read_fcs

    with pytest.raises(FileNotFoundError):
        read_fcs(str(tmp_path / "nope.fcs"))


# ── channel values vs scale values ──────────────────────────────────────────
# Found by adversarial review AFTER the reader had been written and opened as a
# PR. The original fixture used $PnE = "0,0" and $PnG = 1.0 throughout, so it
# could not have caught this no matter what the reader did.

@pytest.fixture
def fcs_amplified(tmp_path):
    """A file with a real amplifier gain on one channel."""
    import flowio

    raw = np.array([[100.0, 200.0, 1000.0, 5000.0]] * 50, dtype=np.float32)
    path = tmp_path / "amplified.fcs"
    with open(path, "wb") as fh:
        flowio.create_fcs(
            fh, raw.flatten().tolist(),
            channel_names=["FSC-A", "SSC-A", "FITC-A", "PE-A"],
            metadata_dict={"$P4G": "2.0", "$P3R": "1024", "$P4R": "1024"},
        )
    return str(path)


def test_amplifier_gain_is_applied(fcs_amplified):
    """The DATA segment holds CHANNEL values; $PnG turns them into SCALE values.

    Reading the raw buffer returned a gain-2.0 channel at exactly twice its true
    value — silently, and every gate drawn on it would sit in the wrong place.
    """
    from omicverse.io import read_fcs

    a = read_fcs(fcs_amplified)
    assert np.isclose(np.asarray(a.X)[0, 3], 2500.0), "PE-A has $P4G=2.0: 5000 -> 2500"
    assert np.isclose(np.asarray(a.X)[0, 2], 1000.0), "FITC-A has no gain: unchanged"
    assert a.uns["fcs"]["preprocessed"] is True


def test_preprocess_can_be_turned_off(fcs_amplified):
    from omicverse.io import read_fcs

    a = read_fcs(fcs_amplified, preprocess=False)
    assert np.isclose(np.asarray(a.X)[0, 3], 5000.0)
    assert a.uns["fcs"]["preprocessed"] is False


def test_scaling_agrees_with_flowio_itself(fcs_amplified):
    """Pin against the parser's own preprocessing rather than a hand-computed
    number, so $PnE / $TIMESTEP handling stays correct too."""
    import flowio

    from omicverse.io import read_fcs

    expected = flowio.FlowData(fcs_amplified).as_array(preprocess=True)
    assert np.allclose(np.asarray(read_fcs(fcs_amplified).X), expected)
