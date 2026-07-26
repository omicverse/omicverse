"""``ov.datasets.flow_demo`` — the simulated sample the ov.flow tutorials use.

Lives beside the flow tests rather than under tests/datasets/ (which does not
exist) because what these assert is cytometry, not plumbing: that the ground
truth actually describes the events it is attached to, and that the sample has
the properties the tutorials claim about it.

The alignment tests matter more than they look. ``obs['population']`` comes from
a second call to the simulator, so a divergence between the two calls would
shift every label by an unknown amount — and every downstream claim ("the
singlet gate excludes the doublets") would then be false while still looking
entirely plausible. Nothing else in the suite would catch that.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flowio")

import omicverse as ov  # noqa: E402
from omicverse.datasets import FLOW_DEMO_PANEL, FLOW_DEMO_SPILLOVER  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return ov.datasets.flow_demo()


@pytest.fixture(scope="module")
def comp(raw):
    a = raw.copy()
    ov.flow.compensate(a)
    return a


def _median(adata, channel, population):
    values = adata[:, channel].X.ravel()
    mask = adata.obs["population"].to_numpy() == population
    return float(np.median(values[mask]))


# ── shape and provenance ────────────────────────────────────────────────────


def test_it_reads_as_a_cytometry_anndata(raw):
    assert raw.n_obs == 59_600
    assert list(raw.var["channel"]) == list(FLOW_DEMO_PANEL)
    assert raw.uns["flow_demo"]["simulated"] is True


def test_the_spillover_matrix_survives_the_file(raw):
    """It has to come back out of the $SPILLOVER keyword, not be attached in
    Python — teaching compensation from a matrix the loader handed over would
    skip the part that is actually hard."""
    spill = raw.uns["fcs"]["spillover"]
    assert list(spill.columns) == list(FLOW_DEMO_SPILLOVER.columns)
    np.testing.assert_allclose(
        spill.to_numpy(), FLOW_DEMO_SPILLOVER.to_numpy(), rtol=1e-6
    )


def test_both_entry_points_give_the_same_numbers(raw, tmp_path):
    """flowio writes float32. An AnnData built directly in float64 would differ by
    enough to move an event across a gate boundary, so flow_demo() goes out
    through a file and back in — and this is what pins that it still does."""
    path = ov.datasets.flow_demo_fcs(str(tmp_path / "demo.fcs"))
    from_file = ov.io.read_fcs(path)
    assert np.array_equal(np.asarray(raw.X), np.asarray(from_file.X))


def test_it_is_deterministic_and_the_seed_does_something():
    a = ov.datasets.flow_demo()
    b = ov.datasets.flow_demo()
    c = ov.datasets.flow_demo(seed=1)
    assert np.array_equal(np.asarray(a.X), np.asarray(b.X))
    assert not np.array_equal(np.asarray(a.X), np.asarray(c.X))
    # a different seed must not change the biology, only the draw
    assert (a.obs["population"].value_counts().sort_index()
            == c.obs["population"].value_counts().sort_index()).all()


# ── the ground truth describes the events it is attached to ────────────────


def test_scatter_matches_the_population_labels(raw):
    assert _median(raw, "FSC-A", "debris") < 30_000
    assert _median(raw, "FSC-A", "mono") > 120_000
    assert _median(raw, "SSC-A", "mono") > _median(raw, "SSC-A", "CD4 T")


@pytest.mark.parametrize(
    "population,channel,other",
    [
        ("B", "CD19", "CD4 T"),
        ("CD8 T", "CD8", "CD4 T"),
        ("CD4 T", "CD4", "CD8 T"),
        ("dead", "Viability", "CD4 T"),
    ],
)
def test_each_population_is_bright_for_its_own_marker(comp, population, channel, other):
    """On COMPENSATED values. Asserting this on raw data would fail — and
    correctly so: uncompensated, CD4-bright T cells read ~1,500 on the CD19
    detector purely from PE and FITC spillover."""
    mine = _median(comp, channel, population)
    theirs = _median(comp, channel, other)
    assert mine > 50 * theirs, f"{population} {channel}: {mine:.0f} vs {theirs:.0f}"


def test_the_rare_t_cell_subsets_are_what_they_say(comp):
    assert _median(comp, "CD4", "DP T") > 1000 and _median(comp, "CD8", "DP T") > 1000
    assert _median(comp, "CD4", "DN T") < 200 and _median(comp, "CD8", "DN T") < 200
    counts = comp.obs["population"].value_counts()
    assert counts["DN T"] == 1400 and counts["DP T"] == 500


def test_doublets_have_the_pulse_geometry_that_identifies_them(raw):
    """Same area, about half the height. If doublets were merely labelled
    rather than shaped, the singlet gate in the tutorial would be decorative."""
    ratio = raw[:, "FSC-H"].X.ravel() / raw[:, "FSC-A"].X.ravel()
    pop = raw.obs["population"].to_numpy()
    assert np.median(ratio[pop == "doublet"]) < 0.7 * np.median(ratio[pop == "CD4 T"])


# ── the properties the tutorials are built on ──────────────────────────────


def test_compensation_removes_a_spillover_artefact_worth_showing(raw, comp):
    """CD4-bright T cells appear CD19-dim before compensation and do not after.
    A demo sample where compensation changes nothing visible teaches nothing."""
    before = _median(raw, "CD19", "CD4 T")
    after = _median(comp, "CD19", "CD4 T")
    assert before > 500, f"no visible spillover artefact to correct ({before:.0f})"
    assert after < 100, f"compensation did not remove it ({after:.0f})"


def test_compensated_data_goes_below_zero(comp):
    """The reason biexponential display scales exist. A strictly-positive
    simulation makes logicle look like a complication rather than a fix."""
    for channel in ("CD3", "CD4", "CD8", "CD19"):
        below = (comp[:, channel].X.ravel() <= 0).mean()
        assert below > 0.05, f"{channel}: only {below:.1%} at or below zero"


def test_compensation_creates_more_negatives_than_it_started_with(raw, comp):
    """Subtracting spillover from a dim event pushes it under zero — which is
    why compensated data needs the scale even more than raw data does."""
    before = (raw[:, "CD3"].X.ravel() <= 0).mean()
    after = (comp[:, "CD3"].X.ravel() <= 0).mean()
    assert after > before


def test_a_gating_strategy_finds_the_populations_that_are_there(comp):
    """End to end: the sample must be gateable, or it is not a demo sample.
    Checked against obs['population'], not against how the plot looks."""
    lin = ov.flow.Linear(t=262144.0)
    lg = ov.flow.Logicle(t=262144.0, m=4.5, w=1.0, a=0.0)
    at = lambda v: float(lg.apply(np.array([float(v)]))[0])  # noqa: E731

    gs = ov.flow.GatingStrategy("check")
    gs.add_gate(ov.flow.PolygonGate(
        name="Cells", dims=("FSC-A", "SSC-A"),
        transforms={"FSC-A": lin, "SSC-A": lin},
        vertices=np.array([[0.12, 0.01], [0.12, 0.17], [0.40, 0.32],
                           [0.68, 0.32], [0.68, 0.09], [0.38, 0.01]])))
    gs.add_gate(ov.flow.PolygonGate(
        name="Singlets", dims=("FSC-A", "FSC-H"),
        transforms={"FSC-A": lin, "FSC-H": lin},
        vertices=np.array([[0.05, 0.03], [0.05, 0.10], [0.75, 0.86], [0.75, 0.69]])),
        parent="Cells")
    gs.add_gate(ov.flow.RectangleGate(
        name="Live", dims=("Viability",), transforms={"Viability": lg},
        bounds=((None, at(1200)),)), parent="Singlets")
    res = gs.apply(comp.copy(), write_obs=False)

    pop = comp.obs["population"].to_numpy()
    kept_doublets = int((pop[res.masks["Singlets"]] == "doublet").sum())
    kept_dead = int((pop[res.masks["Live"]] == "dead").sum())
    assert kept_doublets < 0.1 * int((pop == "doublet").sum())
    assert kept_dead < 0.1 * int((pop == "dead").sum())
    # and it must not throw the real cells away with them
    assert int((pop[res.masks["Live"]] == "CD4 T").sum()) > 0.9 * int((pop == "CD4 T").sum())
