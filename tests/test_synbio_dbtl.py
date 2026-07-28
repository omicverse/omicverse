"""Build, Test and Learn layers, plus construct tuning.

The assertions that matter here are recovery and monotonicity, not "does it run":

* growth-curve fitting is checked against **synthetic data with known
  parameters** — a fitter that cannot recover µmax from a curve it was handed is
  not a fitter, and the ground truth is the only way to see that.
* the Echo and Opentrons outputs are checked against their **externally specified
  formats**, because a picklist with the wrong column names is not a picklist.
* DoE run counts are checked against the combinatorics they claim
  (Plackett-Burman is N runs for N-1 factors; a resolution-IV design must be a
  fraction of the full factorial), and the response-surface designs are checked
  to actually recover a quadratic.
* burden must *scale* with copy number, since a burden model that returns the
  same number for 1 and 200 copies is worse than no burden model.
"""
import math

import pytest

np = pytest.importorskip("numpy")

import omicverse as ov
from omicverse.synbio._assay import _gompertz

sb = ov.synbio


# ---------------------------------------------------------------------------
# Build — plate layout
# ---------------------------------------------------------------------------

def test_plate_fills_by_row_then_column():
    p = sb.plate_layout(["a", "b", "c"], plate="96", by="row")
    assert p.contents["A1"] == "a" and p.contents["A2"] == "b"
    q = sb.plate_layout(["a", "b", "c"], plate="96", by="column")
    assert q.contents["A1"] == "a" and q.contents["B1"] == "b"


def test_controls_are_placed_before_the_items():
    """So they land somewhere predictable rather than wherever the items run out."""
    p = sb.plate_layout(["v1", "v2"], controls=["blank", "wt"])
    assert p.contents["A1"] == "blank" and p.contents["A2"] == "wt"
    assert p.contents["A3"] == "v1"


def test_replicates_are_named_distinctly():
    p = sb.plate_layout(["v1"], replicates=3)
    assert sorted(p.contents.values()) == ["v1", "v1_r2", "v1_r3"]


def test_start_well_leaves_a_block_free():
    p = sb.plate_layout(["a"], start_well="C5")
    assert p.contents == {"C5": "a"}


def test_overfilling_a_plate_is_an_error():
    with pytest.raises(ValueError, match="放不进|更大的板"):
        sb.plate_layout([f"v{i}" for i in range(100)], plate="96")


def test_well_name_round_trips():
    from omicverse.synbio._build import parse_well, well_name
    for well in ("A1", "B7", "H12", "P24"):
        assert well_name(*parse_well(well)) == well


def test_plate_formats_have_the_right_capacity():
    from omicverse.synbio._build import PLATE_FORMATS
    for name, (rows, cols) in PLATE_FORMATS.items():
        assert rows * cols == int(name)


# ---------------------------------------------------------------------------
# Build — assembly worklist
# ---------------------------------------------------------------------------

def test_equimolar_volumes_account_for_fragment_length():
    """Equal volumes of equal ng/µL are *not* equimolar when lengths differ —
    that conversion is the thing this function exists to get right."""
    wl = sb.assembly_worklist({"long": (3000, 50.0), "short": (500, 50.0)},
                              backbone="long", insert_ratio=1.0)
    vols = {t.item: t.volume_ul for t in wl.transfers}
    assert vols["long"] > vols["short"] * 3, (
        "a 6x longer fragment at the same ng/µL needs ~6x the volume for the "
        f"same moles, got {vols['long']:.3f} vs {vols['short']:.3f}")


def test_dna_nm_conversion():
    from omicverse.synbio._build import dna_nm
    # 50 ng/µL of a 1000 bp fragment ~ 77 nM
    assert dna_nm(50.0, 1000) == pytest.approx(76.9, rel=0.02)
    assert dna_nm(50.0, 2000) == pytest.approx(dna_nm(50.0, 1000) / 2, rel=1e-6)


def test_worklist_reaches_the_final_volume():
    wl = sb.assembly_worklist({"bb": (2700, 50.0), "ins": (900, 30.0)},
                              final_volume_ul=20.0)
    assert wl.total_volume_ul == pytest.approx(20.0, abs=0.01)


def test_master_mix_and_water_are_included():
    wl = sb.assembly_worklist({"bb": (2700, 50.0), "ins": (900, 30.0)})
    items = {t.item for t in wl.transfers}
    assert "water" in items
    assert any("master_mix" in i for i in items)


def test_unpipettable_volumes_are_reported_with_the_dilution():
    """A volume below the pipette's floor is not a volume — say what to do."""
    wl = sb.assembly_worklist({"bb": (2700, 50.0), "conc": (500, 5000.0)})
    assert any("稀释" in n for n in wl.notes)


def test_backbone_defaults_to_the_longest_and_says_so():
    wl = sb.assembly_worklist({"small": (500, 20.0), "big": (5000, 20.0)})
    assert any("big" in n for n in wl.notes)


def test_assembly_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        sb.assembly_worklist({"a": (100, 10.0)}, method="ligation")


# ---------------------------------------------------------------------------
# Build — Echo pick list
# ---------------------------------------------------------------------------

@pytest.fixture
def worklist_and_source():
    wl = sb.assembly_worklist({"bb": (2700, 50.0), "ins": (900, 30.0)})
    src = sb.plate_layout(["bb", "ins", "golden_gate_master_mix", "water"],
                          name="src")
    return wl, src


def test_echo_columns_are_the_documented_ones(worklist_and_source):
    """Echo Cherry Pick reads specific column names; approximations do not load."""
    from omicverse.synbio._build import ECHO_COLUMNS
    wl, src = worklist_and_source
    header = sb.echo_picklist(wl, source_plate=src).splitlines()[0]
    assert header == ",".join(ECHO_COLUMNS)
    for required in ("Source Plate Name", "Source Well",
                     "Destination Plate Name", "Destination Well",
                     "Transfer Volume"):
        assert required in header


def test_echo_volumes_are_nanolitres_snapped_to_resolution(worklist_and_source):
    """The Echo dispenses in 2.5 nL droplets — a volume it cannot make is not a
    volume, and µL here would be a 1000x error."""
    wl, src = worklist_and_source
    rows = sb.echo_picklist(wl, source_plate=src, resolution_nl=2.5).splitlines()[1:]
    volumes = [float(r.split(",")[5]) for r in rows if r.strip()]
    assert volumes and all(v > 100 for v in volumes), "should be nL, not µL"
    for v in volumes:
        assert abs(v / 2.5 - round(v / 2.5)) < 1e-9


def test_echo_splits_transfers_above_the_ceiling(worklist_and_source):
    """Count rows per item rather than against the worklist length: the pick list
    now excludes the enzyme master mix and the water (they are outside every
    calibrated Echo fluid class and belong on a tip-based handler), so the
    worklist length is no longer the right baseline."""
    wl, src = worklist_and_source
    rows = [r for r in sb.echo_picklist(wl, source_plate=src,
                                        max_volume_nl=500.0).splitlines()[1:]
            if r.strip()]
    per_item: dict = {}
    for row in rows:
        fields = row.split(",")
        per_item[fields[6]] = per_item.get(fields[6], 0) + 1
    assert per_item, "no transfers emitted"
    assert any(n > 1 for n in per_item.values()), (
        f"a transfer above 500 nL must be split: {per_item}")
    for item, n_rows in per_item.items():
        volumes = [float(r.split(",")[5]) for r in rows if r.split(",")[6] == item]
        assert all(v <= 500.0 + 1e-9 for v in volumes), (item, volumes)


def test_echo_excludes_the_enzyme_master_mix(worklist_and_source):
    """A glycerol-containing enzyme mix is outside every Echo fluid class."""
    wl, src = worklist_and_source
    with pytest.warns(UserWarning, match="枪头式移液器"):
        rows = sb.echo_picklist(wl, source_plate=src).splitlines()[1:]
    items = {r.split(",")[6] for r in rows if r.strip()}
    assert "water" not in items
    assert not any("master_mix" in i for i in items), items


def test_echo_source_plate_name_matches_where_the_well_came_from(worklist_and_source):
    """The CSV named a plate ('reagents') the well number did not come from."""
    wl, src = worklist_and_source
    rows = sb.echo_picklist(wl, source_plate=src).splitlines()[1:]
    for row in rows:
        if row.strip():
            assert row.split(",")[0] == src.name, row


def test_echo_never_emits_scientific_notation(worklist_and_source):
    """`f"{v:g}"` printed a 1.25 mL transfer as '1.25e+06'."""
    wl, src = worklist_and_source
    rows = sb.echo_picklist(wl, source_plate=src,
                            max_volume_nl=2e6).splitlines()[1:]
    for row in rows:
        if row.strip():
            assert "e+" not in row.split(",")[5].lower(), row


def test_echo_refuses_transfers_it_cannot_place():
    wl = sb.assembly_worklist({"bb": (2700, 50.0), "ins": (900, 30.0)})
    with pytest.raises(ValueError, match="没有源孔位|source_plate"):
        sb.echo_picklist(wl)


def test_echo_writes_a_file(tmp_path, worklist_and_source):
    wl, src = worklist_and_source
    out = tmp_path / "pick.csv"
    sb.echo_picklist(wl, source_plate=src, out=str(out))
    assert out.is_file() and out.read_text().startswith("Source Plate Name")


# ---------------------------------------------------------------------------
# Build — Opentrons protocol
# ---------------------------------------------------------------------------

def test_opentrons_script_has_the_required_api_v2_structure(worklist_and_source):
    wl, src = worklist_and_source
    script = sb.opentrons_protocol(wl, source_plate=src)
    for required in ('from opentrons import protocol_api',
                     '"protocolName"', '"robotType"', '"apiLevel"',
                     'def run(protocol: protocol_api.ProtocolContext) -> None:',
                     'load_labware', 'load_instrument', '.transfer('):
        assert required in script, f"missing {required!r}"


def test_opentrons_script_is_valid_python(worklist_and_source):
    """A generated protocol that does not parse is not a protocol."""
    import ast
    wl, src = worklist_and_source
    ast.parse(sb.opentrons_protocol(wl, source_plate=src))


def test_pipette_is_chosen_from_the_volumes():
    """Asking a p300 for 0.5 µL is the usual way an automated run quietly fails,
    so the instrument follows the largest transfer rather than a default."""
    import re

    small = sb.assembly_worklist({"a": (1000, 500.0), "b": (1000, 500.0)},
                                 final_volume_ul=5.0, master_mix_fraction=0.2)
    big = sb.assembly_worklist({"a": (1000, 50.0), "b": (1000, 50.0)},
                               final_volume_ul=200.0, master_mix_fraction=0.5)
    src = sb.plate_layout(["a", "b", "golden_gate_master_mix", "water"])

    def pipette_of(wl):
        script = sb.opentrons_protocol(wl, source_plate=src)
        return re.search(r'load_instrument\("([^"]+)"', script).group(1)

    assert max(t.volume_ul for t in small.transfers) <= 20.0
    assert "p20" in pipette_of(small)
    assert max(t.volume_ul for t in big.transfers) > 20.0
    assert "p300" in pipette_of(big)


def test_opentrons_flags_volumes_below_the_pipette_minimum():
    wl = sb.assembly_worklist({"a": (1000, 800.0), "b": (1000, 800.0)},
                              final_volume_ul=20.0)
    src = sb.plate_layout(["a", "b", "golden_gate_master_mix", "water"])
    script = sb.opentrons_protocol(wl, source_plate=src, pipette="p300_single_gen2")
    assert "WARNING" in script and "minimum" in script


def test_opentrons_rejects_unknown_robot(worklist_and_source):
    wl, src = worklist_and_source
    with pytest.raises(ValueError, match="robot must be"):
        sb.opentrons_protocol(wl, source_plate=src, robot="OT-3")


# ---------------------------------------------------------------------------
# Build — PCR program
# ---------------------------------------------------------------------------

def test_annealing_follows_the_lower_primer_tm():
    prog = sb.pcr_protocol(tm_forward=68.0, tm_reverse=60.0, amplicon_bp=1000)
    assert prog.annealing_C == pytest.approx(63.0)     # 60 + 3 for Q5


def test_taq_anneals_below_the_tm_and_q5_above():
    q5 = sb.pcr_protocol(tm_forward=62.0, tm_reverse=62.0, amplicon_bp=1000,
                         polymerase="q5")
    taq = sb.pcr_protocol(tm_forward=62.0, tm_reverse=62.0, amplicon_bp=1000,
                          polymerase="taq")
    assert q5.annealing_C > 62.0 > taq.annealing_C


def test_extension_scales_with_amplicon_length():
    short = sb.pcr_protocol(tm_forward=62.0, tm_reverse=62.0, amplicon_bp=500)
    long = sb.pcr_protocol(tm_forward=62.0, tm_reverse=62.0, amplicon_bp=5000)
    assert long.extension_s > short.extension_s * 5


def test_mismatched_primer_tms_are_flagged():
    prog = sb.pcr_protocol(tm_forward=70.0, tm_reverse=55.0, amplicon_bp=1000)
    assert any("Tm 差" in n for n in prog.notes)


def test_pcr_rejects_unknown_polymerase():
    with pytest.raises(ValueError, match="polymerase must be one of"):
        sb.pcr_protocol(tm_forward=62.0, tm_reverse=62.0, amplicon_bp=100,
                        polymerase="pfu")


# ---------------------------------------------------------------------------
# Test — growth curves against known ground truth
# ---------------------------------------------------------------------------

TRUE = {"A": 1.2, "mu": 0.45, "lag": 2.5}
BLANK = 0.05


@pytest.fixture(scope="module")
def synthetic_curve():
    t = np.linspace(0, 24, 49)
    clean = _gompertz(t, TRUE["A"], TRUE["mu"], TRUE["lag"])
    noise = np.random.default_rng(0).normal(0, 0.01, t.size)
    return t, clean + noise + BLANK


def test_fit_recovers_the_carrying_capacity(synthetic_curve):
    t, od = synthetic_curve
    fit = sb.fit_growth_curve(t, od, blank=BLANK)
    assert fit.carrying_capacity == pytest.approx(TRUE["A"], rel=0.05)


def test_fit_recovers_the_lag(synthetic_curve):
    t, od = synthetic_curve
    fit = sb.fit_growth_curve(t, od, blank=BLANK)
    assert fit.lag == pytest.approx(TRUE["lag"], abs=0.3)


def test_fit_recovers_the_sigmoid_slope(synthetic_curve):
    """``slope_max`` is the quantity the Gompertz µ parameter *is* — dOD/dt."""
    t, od = synthetic_curve
    fit = sb.fit_growth_curve(t, od, blank=BLANK)
    assert fit.slope_max == pytest.approx(TRUE["mu"], rel=0.10)


def test_fit_is_good(synthetic_curve):
    t, od = synthetic_curve
    assert sb.fit_growth_curve(t, od, blank=BLANK).good_fit


@pytest.mark.parametrize("model", ["gompertz", "logistic", "richards", "baranyi"])
def test_every_model_reports_a_specific_rate_not_its_own_parameter(
        synthetic_curve, model):
    """The published parameterisations disagree about what µ means: Gompertz and
    logistic use dOD/dt (OD/h), Baranyi uses a specific rate (1/h). Reading each
    model's own parameter gave 0.43 and 1.43 for identical growth. mu_max is now
    the specific rate at the inflection for all of them, and slope_max carries
    the sigmoid's slope."""
    t, od = synthetic_curve
    fit = sb.fit_growth_curve(t, od, model=model, blank=BLANK)
    assert fit.slope_max == pytest.approx(TRUE["mu"], rel=0.15), model
    assert 0.3 < fit.mu_max < 3.0, f"{model}: µspec={fit.mu_max}"


def test_specific_rates_agree_across_models_to_within_the_inflection_difference(
        synthetic_curve):
    """The residual spread is real — Gompertz inflects at A/e, logistic at A/2 —
    and is documented rather than averaged away."""
    t, od = synthetic_curve
    rates = [sb.fit_growth_curve(t, od, model=m, blank=BLANK).mu_max
             for m in ("gompertz", "logistic", "richards", "baranyi")]
    assert max(rates) / min(rates) < 2.0, rates


def test_doubling_time_comes_from_the_specific_rate(synthetic_curve):
    t, od = synthetic_curve
    fit = sb.fit_growth_curve(t, od, blank=BLANK)
    assert fit.doubling_time == pytest.approx(math.log(2) / fit.mu_max)


def test_minutes_are_detected_and_converted(synthetic_curve):
    """A trace in minutes fitted as hours puts µmax out by 60x."""
    t, od = synthetic_curve
    hours = sb.fit_growth_curve(t, od, blank=BLANK)
    minutes = sb.fit_growth_curve(t * 60, od, blank=BLANK)
    assert any("分钟" in n for n in minutes.notes)
    assert minutes.mu_max == pytest.approx(hours.mu_max, rel=0.02)


def test_model_comparison_ranks_the_generating_model_first(synthetic_curve):
    """The data came from a Gompertz curve, so Gompertz should win on AIC."""
    t, od = synthetic_curve
    df = sb.compare_growth_models(t, od, blank=BLANK)
    assert df.index[0] in ("gompertz", "richards"), df.index.tolist()
    assert "slope_max" in df.columns


def test_a_flat_trace_is_reported_not_fitted():
    t = np.linspace(0, 24, 25)
    fit = sb.fit_growth_curve(t, np.full(25, 0.05), blank=0.05)
    assert not fit.converged and fit.mu_max == 0.0


def test_too_few_points_is_an_error():
    with pytest.raises(ValueError, match="至少需要 5 个"):
        sb.fit_growth_curve([0, 1, 2], [0.1, 0.2, 0.3])


def test_unknown_growth_model_rejected():
    with pytest.raises(ValueError, match="model must be one of"):
        sb.fit_growth_curve(np.arange(10), np.arange(10) * 0.1, model="monod")


# ---------------------------------------------------------------------------
# Test — plates, blanks, dose response
# ---------------------------------------------------------------------------

@pytest.fixture
def plate_frame():
    import pandas as pd
    t = np.linspace(0, 24, 33)
    data = {"time_h": t}
    rng = np.random.default_rng(1)
    for i, mu in enumerate([0.3, 0.45, 0.6], start=1):
        data[f"A{i}"] = (_gompertz(t, 1.0, mu, 2.0)
                         + rng.normal(0, 0.008, t.size) + BLANK)
    data["A12"] = np.full(t.size, BLANK) + rng.normal(0, 0.004, t.size)
    return pd.DataFrame(data)


def test_plate_fit_orders_wells_by_growth(plate_frame):
    df = sb.fit_growth_curves(plate_frame, blanks=["A12"])
    assert list(df.index) == ["A1", "A2", "A3"]
    assert df["slope_max"].is_monotonic_increasing if "slope_max" in df else True
    assert df.loc["A3", "mu_max"] > df.loc["A1", "mu_max"]


def test_blank_wells_are_subtracted_not_fitted(plate_frame):
    df = sb.fit_growth_curves(plate_frame, blanks=["A12"])
    assert "A12" not in df.index
    assert df.attrs["blank"] == pytest.approx(BLANK, abs=0.01)


def test_poor_fits_are_flagged_not_dropped(plate_frame):
    df = sb.fit_growth_curves(plate_frame, blanks=["A12"])
    assert "good_fit" in df.columns and len(df) == 3


def test_missing_time_column_is_an_error(plate_frame):
    with pytest.raises(ValueError, match="找不到时间列"):
        sb.fit_growth_curves(plate_frame, time_col="Time")


def test_dose_response_recovers_a_known_ic50():
    conc = np.array([0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    resp = 1.0 / (1.0 + (conc / 0.8) ** 1.6)
    resp[0] = 1.0
    fit = sb.dose_response(conc, resp)
    assert fit.inhibitory
    assert fit.ec50 == pytest.approx(0.8, rel=0.05)
    assert fit.ic50 == fit.ec50
    assert fit.mic is not None and fit.mic > 0


def test_dose_response_marks_a_stimulatory_curve_as_not_inhibitory():
    conc = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    resp = 1.0 / (1.0 + (0.5 / conc) ** 1.5)
    fit = sb.dose_response(conc, resp)
    assert not fit.inhibitory and fit.ic50 is None


def test_zero_dose_is_used_not_dropped():
    conc = np.array([0, 0.1, 0.3, 1.0, 3.0, 10.0])
    resp = np.array([1.0, 0.95, 0.8, 0.5, 0.2, 0.05])
    fit = sb.dose_response(conc, resp)
    assert any("零浓度" in n for n in fit.notes)


def test_dose_response_needs_four_points():
    with pytest.raises(ValueError, match="至少需要 4 个"):
        sb.dose_response([1, 2, 3], [1, 2, 3])


def test_read_plate_reader_round_trip(tmp_path, plate_frame):
    path = tmp_path / "od.csv"
    plate_frame.to_csv(path, index=False)
    got = sb.read_plate_reader(str(path))
    assert "time_h" in got.columns and len(got) == len(plate_frame)


def test_read_plate_reader_detects_minutes(tmp_path, plate_frame):
    import pandas as pd
    frame = plate_frame.copy()
    frame["time_h"] = frame["time_h"] * 60
    path = tmp_path / "min.csv"
    frame.to_csv(path, index=False)
    got = sb.read_plate_reader(str(path))
    assert got.attrs["time_unit_detected"] == "min"
    assert got["time_h"].max() == pytest.approx(24.0, rel=0.01)


# ---------------------------------------------------------------------------
# Test — the coupling to the metabolic model
# ---------------------------------------------------------------------------

def test_measured_growth_sits_beside_the_model_prediction():
    cobra = pytest.importorskip("cobra")
    model = sb.load_gem("textbook")
    df = sb.compare_growth_to_model({"wt": 0.60, "slow": 0.20}, model)
    assert set(df.index) == {"wt", "slow"}
    assert "fba" in df.columns and "measured_over_fba" in df.columns
    assert df.loc["wt", "fba"] == pytest.approx(0.8739, abs=1e-3)


def test_measurements_above_the_fba_bound_are_flagged():
    """FBA is a stoichiometric upper bound — measuring above it means something
    is wrong with the medium constraints, the OD-to-DW conversion, or the units."""
    cobra = pytest.importorskip("cobra")
    model = sb.load_gem("textbook")
    df = sb.compare_growth_to_model({"impossible": 5.0}, model)
    assert "warning" in df.attrs and "FBA" in df.attrs["warning"]


def test_knockout_labels_are_compared_against_their_own_prediction():
    cobra = pytest.importorskip("cobra")
    model = sb.load_gem("textbook")
    df = sb.compare_growth_to_model({"pfk_ko": 0.10}, model,
                                    knockouts={"pfk_ko": ["PFK"]})
    assert "moma" in df.columns
    assert df.loc["pfk_ko", "fba"] < 0.8739


# ---------------------------------------------------------------------------
# Learn — designs
# ---------------------------------------------------------------------------

FACTORS = {"promoter": (0.05, 1.0), "rbs": (1.0, 1000.0), "copies": (1.0, 20.0)}


def test_full_factorial_is_two_to_the_k():
    assert sb.doe_design(FACTORS, design="full_factorial").n_runs == 8


def test_fractional_factorial_is_a_fraction_of_the_full_one():
    """The whole argument for a screening design: 8 factors is 256 runs full and
    16 at resolution IV."""
    d = sb.doe_design({f"f{i}": (0, 1) for i in range(8)},
                      design="fractional_factorial", resolution=4)
    assert d.n_runs < 2 ** 8
    assert d.n_runs == 16


@pytest.mark.parametrize("k", [3, 5, 7, 11])
def test_plackett_burman_has_more_runs_than_factors(k):
    """N runs for up to N-1 factors. Dropping a row too gave N-1 runs, so three
    factors got three observations for three main effects plus an intercept — an
    under-determined design that still looked like one."""
    d = sb.doe_design({f"f{i}": (0, 1) for i in range(k)},
                      design="plackett_burman")
    assert d.n_runs > k, f"{k} factors got only {d.n_runs} runs"
    assert d.n_runs % 4 == 0


def test_definitive_screening_is_two_k_plus_one():
    d = sb.doe_design(FACTORS, design="definitive_screening")
    assert d.n_runs == 2 * len(FACTORS) + 1


def test_definitive_screening_has_three_levels():
    """Curvature is only estimable with a middle level."""
    d = sb.doe_design(FACTORS, design="definitive_screening")
    coded = np.asarray(d.coded)
    assert len(np.unique(coded)) >= 3


def test_box_behnken_has_no_corner_points():
    """No run combines the extremes of every factor — which matters when those
    conditions kill the culture."""
    coded = np.asarray(sb.doe_design(FACTORS, design="box_behnken").coded)
    corners = np.all(np.abs(coded) == 1.0, axis=1)
    assert not corners.any()


def test_central_composite_has_axial_points_beyond_the_factorial_core():
    """Axial points must sit outside the factorial core — that is what makes
    curvature estimable — but inside the stated factor range, which is what
    makes them buildable. The design is rescaled so the axial points land *on*
    the bounds; asserting max|coded| > 1 encoded the old behaviour, where a third
    of the runs asked for settings beyond the range the caller declared
    (including negative plasmid copy numbers)."""
    design = sb.doe_design(FACTORS, design="central_composite")
    coded = np.asarray(design.coded)
    core = np.abs(coded[np.count_nonzero(coded, axis=1) == coded.shape[1]]).max()
    assert np.abs(coded).max() > core + 1e-9, "no axial points beyond the core"
    assert np.abs(coded).max() <= 1.0 + 1e-9, "axial points outside the bounds"

    natural = design.to_frame()
    for name, (lo, hi) in FACTORS.items():
        assert natural[name].min() >= lo - 1e-9, f"{name} below its low bound"
        assert natural[name].max() <= hi + 1e-9, f"{name} above its high bound"


def test_central_composite_unbounded_is_available_and_does_leave_the_box():
    coded = np.asarray(sb.doe_design(FACTORS, design="central_composite",
                                     bounded=False).coded)
    assert np.abs(coded).max() > 1.0


def test_natural_units_respect_the_factor_bounds():
    df = sb.doe_design(FACTORS, design="full_factorial").to_frame()
    for name, (lo, hi) in FACTORS.items():
        assert df[name].min() >= lo - 1e-9
        assert df[name].max() <= hi + 1e-9


def test_latin_hypercube_run_count_is_settable():
    d = sb.doe_design(FACTORS, design="latin_hypercube", n_runs=25)
    assert d.n_runs == 25


def test_unknown_design_rejected():
    with pytest.raises(ValueError, match="design must be one of"):
        sb.doe_design(FACTORS, design="taguchi")


def test_degenerate_factor_bounds_rejected():
    with pytest.raises(ValueError, match="low, high"):
        sb.doe_design({"a": (1.0, 1.0)})


# ---------------------------------------------------------------------------
# Learn — DoE analysis
# ---------------------------------------------------------------------------

def _quadratic_response(X, rng):
    return (3.0 * X[:, 0] + 0.002 * X[:, 1]
            - 0.05 * (X[:, 2] - 10.0) ** 2 + rng.normal(0, 0.2, len(X)))


def test_analysis_recovers_a_quadratic_from_a_response_surface_design():
    """central_composite exists *to* fit curvature. Without quadratic terms the
    analysis fitted a plane through a bowl and reached R² = 0.14 while still
    reporting the main effects as significant."""
    des = sb.doe_design(FACTORS, design="central_composite")
    X = des.to_frame().values
    y = _quadratic_response(X, np.random.default_rng(1))
    res = sb.analyse_doe(des, y)
    assert res.r_squared > 0.95
    assert "copies^2" in res.quadratics
    assert res.ranked_effects[0][0] == "copies^2"


def test_quadratic_terms_are_skipped_on_a_two_level_design():
    des = sb.doe_design(FACTORS, design="full_factorial")
    y = np.arange(des.n_runs, dtype=float)
    res = sb.analyse_doe(des, y)
    assert res.quadratics == {}
    assert any("三水平" in n for n in res.notes)


def test_main_effect_signs_follow_the_response():
    des = sb.doe_design({"x": (0.0, 1.0), "y": (0.0, 1.0)},
                        design="full_factorial")
    X = np.asarray(des.coded)
    res = sb.analyse_doe(des, 5.0 * X[:, 0] - 3.0 * X[:, 1])
    assert res.main_effects["x"] > 0 > res.main_effects["y"]


def test_lenth_significance_is_reported_without_replication():
    des = sb.doe_design({f"f{i}": (0, 1) for i in range(7)},
                        design="plackett_burman")
    X = np.asarray(des.coded)
    y = 10.0 * X[:, 0] + np.random.default_rng(0).normal(0, 0.1, len(X))
    res = sb.analyse_doe(des, y)
    assert "f0" in res.significant
    assert any("Lenth" in n for n in res.notes)


def test_mismatched_response_length_is_an_error():
    des = sb.doe_design(FACTORS, design="full_factorial")
    with pytest.raises(ValueError, match="响应给了"):
        sb.analyse_doe(des, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Learn — Bayesian optimisation and the campaign
# ---------------------------------------------------------------------------

@pytest.fixture
def observed():
    des = sb.doe_design(FACTORS, design="central_composite")
    X = des.to_frame().values
    return X, _quadratic_response(X, np.random.default_rng(2))


def test_proposals_lie_inside_the_factor_bounds(observed):
    pytest.importorskip("sklearn")
    X, y = observed
    prop = sb.bayesian_optimize(X, y, FACTORS, batch=4)
    points = np.asarray(prop.points)
    for j, (lo, hi) in enumerate(FACTORS.values()):
        assert points[:, j].min() >= lo - 1e-9
        assert points[:, j].max() <= hi + 1e-9


def test_a_batch_is_not_the_same_point_four_times(observed):
    """The constant-liar pass is what stops that."""
    pytest.importorskip("sklearn")
    X, y = observed
    points = np.asarray(sb.bayesian_optimize(X, y, FACTORS, batch=4).points)
    assert len({tuple(np.round(p, 6)) for p in points}) == 4


@pytest.mark.parametrize("acq", ["ei", "ucb", "poi", "greedy"])
def test_every_acquisition_produces_a_proposal(observed, acq):
    pytest.importorskip("sklearn")
    X, y = observed
    prop = sb.bayesian_optimize(X, y, FACTORS, batch=2, acquisition=acq)
    assert len(prop.predicted_mean) == 2 and prop.acquisition == acq


def test_optimiser_reports_its_own_model_quality(observed):
    pytest.importorskip("sklearn")
    X, y = observed
    prop = sb.bayesian_optimize(X, y, FACTORS, batch=2)
    assert 0.0 <= prop.model_r_squared <= 1.0


def test_too_few_observations_is_an_error():
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError, match="只有 2 个观测|高斯过程"):
        sb.bayesian_optimize([[0.1, 1, 1], [0.2, 2, 2]], [1.0, 2.0], FACTORS)


def test_unknown_acquisition_rejected(observed):
    X, y = observed
    with pytest.raises(ValueError, match="acquisition must be one of"):
        sb.bayesian_optimize(X, y, FACTORS, acquisition="thompson")


def test_campaign_accumulates_rounds(observed):
    X, y = observed
    camp = sb.dbtl_campaign(FACTORS, objective="titre")
    camp.record(X, y)
    camp.record(X[:3], y[:3] + 1.0)
    assert camp.n_rounds == 2
    assert camp.n_experiments == len(y) + 3
    assert len(camp.to_frame()) == camp.n_experiments


def test_campaign_best_tracks_the_maximum(observed):
    X, y = observed
    camp = sb.dbtl_campaign(FACTORS)
    camp.record(X, y)
    best, settings = camp.best
    assert best == pytest.approx(float(np.max(y)))
    assert set(settings) == set(FACTORS)


def test_campaign_knows_when_it_has_stopped_improving(observed):
    """Two flat rounds is the stop signal, not a fixed round count."""
    X, y = observed
    camp = sb.dbtl_campaign(FACTORS)
    camp.record(X, y)
    camp.record(X[:3], y[:3] - 10.0)          # a worse round
    assert not camp.improving
    camp.record(X[:3], y[:3] + 10.0)          # a better one
    assert camp.improving


def test_campaign_proposes_from_all_rounds(observed):
    pytest.importorskip("sklearn")
    X, y = observed
    camp = sb.dbtl_campaign(FACTORS)
    camp.record(X, y)
    prop = camp.propose(batch=2)
    assert len(prop.predicted_mean) == 2


def test_campaign_rejects_a_wrong_shaped_round():
    camp = sb.dbtl_campaign(FACTORS)
    with pytest.raises(ValueError, match="campaign 定义了"):
        camp.record([[1.0, 2.0]], [1.0])


def test_empty_campaign_cannot_propose():
    camp = sb.dbtl_campaign(FACTORS)
    with pytest.raises(ValueError, match="还没有任何一轮"):
        camp.propose()


# ---------------------------------------------------------------------------
# tuning — tAI
# ---------------------------------------------------------------------------

def test_tai_is_between_zero_and_one():
    res = sb.tai(sb.GAPDH_CDS)
    assert 0.0 < res.tai <= 1.0


def test_tai_differs_between_hosts():
    """A species-specific index that gives the same answer for every species is
    not species-specific."""
    coli = sb.tai(sb.GAPDH_CDS, host="e_coli").tai
    yeast = sb.tai(sb.GAPDH_CDS, host="s_cerevisiae").tai
    assert coli != yeast


def test_cai_optimisation_can_lower_tai():
    """The case the docstring exists for: codon usage and tRNA availability do
    not always agree, and when they disagree tAI is the one tied to decoding."""
    pytest.importorskip("dnachisel", reason="needs omicverse[synbio] (dnachisel)")
    native = sb.tai(sb.GAPDH_CDS, host="e_coli").tai
    optimised = sb.tai(sb.codon_optimize(sb.GAPDH_CDS, host="e_coli").sequence,
                       host="e_coli").tai
    assert optimised != native


def test_missing_trna_does_not_zero_the_index():
    res = sb.tai(sb.GAPDH_CDS)
    assert res.tai > 0
    assert all(w > 0 for w in res.weights.values())


def test_bottleneck_codons_are_the_worst_weighted():
    res = sb.tai(sb.GAPDH_CDS)
    worst = res.bottleneck_codons(5)
    assert len(worst) == 5
    assert worst[0][2] <= worst[-1][2]


def test_custom_trna_counts_are_used():
    counts = dict(sb.TRNA_COPY_NUMBERS["e_coli"])
    res = sb.tai(sb.GAPDH_CDS, trna_copies=counts)
    assert res.host == "custom"
    assert res.tai == pytest.approx(sb.tai(sb.GAPDH_CDS, host="e_coli").tai)


def test_unknown_host_says_to_supply_counts():
    with pytest.raises(ValueError, match="trna_copies|内置"):
        sb.tai(sb.GAPDH_CDS, host="b_subtilis")


def test_tai_rejects_a_non_triplet_cds():
    with pytest.raises(ValueError, match="不是 3 的倍数"):
        sb.tai("ATGGC")


# ---------------------------------------------------------------------------
# tuning — graded libraries
# ---------------------------------------------------------------------------

def test_rbs_library_spans_orders_of_magnitude():
    """Scoring one RBS is rbs_strength's job; a graded set is this one's."""
    pytest.importorskip("RNA", reason="needs omicverse[synbio] (ViennaRNA)")
    lib = sb.rbs_library(sb.GAPDH_CDS, n=6, n_candidates=300,
                         target_range=(10, 10000))
    assert len(lib.parts) == 6
    assert lib.dynamic_range > 50, f"only {lib.dynamic_range:.0f}x"
    assert lib.predicted == sorted(lib.predicted)


def test_rbs_library_reports_its_coverage():
    pytest.importorskip("RNA", reason="needs omicverse[synbio] (ViennaRNA)")
    lib = sb.rbs_library(sb.GAPDH_CDS, n=5, n_candidates=250,
                         target_range=(10, 10000))
    assert 0.0 <= lib.coverage <= 1.0


def test_rbs_library_needs_at_least_two_members():
    """n < 2 is rejected before any scoring, so this needs no optional deps."""
    with pytest.raises(ValueError, match="梯度库"):
        sb.rbs_library(sb.GAPDH_CDS, n=1)


def test_promoter_library_is_honest_about_a_narrow_range():
    """The built-in σ70 scorer is a bounded consensus similarity, not a rate
    model, so perturbing the boxes cannot span decades — and the library says so
    rather than presenting 1.4x as a gradient."""
    lib = sb.promoter_library(n=4, n_candidates=200)
    assert len(lib.parts) == 4
    if lib.dynamic_range < 3.0:
        assert any("动态范围有限" in n for n in lib.notes)


# ---------------------------------------------------------------------------
# tuning — integration sites, burden, toehold
# ---------------------------------------------------------------------------

def test_integration_sites_rank_toward_the_requested_expression():
    high = sb.integration_sites(target_expression=2.5)[0]
    low = sb.integration_sites(target_expression=0.4)[0]
    assert high.relative_expression > low.relative_expression


def test_essential_neighbourhoods_are_penalised_and_explained():
    sites = sb.integration_sites(target_expression=2.6, avoid_essential=True)
    oric = [s for s in sites if "oriC" in s.name][0]
    assert oric.essential_nearby
    assert any("必需基因" in n for n in oric.notes)


def test_mechanism_filter():
    sites = sb.integration_sites(mechanism="attB")
    assert sites and all(s.mechanism == "attB" for s in sites)


def test_unknown_host_needs_custom_sites():
    with pytest.raises(ValueError, match="sites=|基因组特异"):
        sb.integration_sites(host="b_subtilis")


def test_burden_scales_with_copy_number_and_expression():
    """A burden model that returns the same number for 1 and 200 copies is worse
    than none. It did, because the default proteome budget was not binding on
    e_coli_core, so subtracting from it changed nothing."""
    cobra = pytest.importorskip("cobra")
    model = sb.load_gem("textbook")
    light = sb.plasmid_burden(model, copy_number=1,
                              expressed_protein_fraction=0.02)
    heavy = sb.plasmid_burden(model, copy_number=200,
                              expressed_protein_fraction=0.50)
    assert heavy.burden > light.burden + 0.1
    assert heavy.verdict != "negligible"


def test_burden_says_when_the_budget_was_not_binding():
    cobra = pytest.importorskip("cobra")
    model = sb.load_gem("textbook")
    est = sb.plasmid_burden(model, copy_number=100,
                            expressed_protein_fraction=0.30)
    assert any("不构成约束" in n for n in est.notes)


def test_high_burden_suggests_what_to_change():
    cobra = pytest.importorskip("cobra")
    model = sb.load_gem("textbook")
    est = sb.plasmid_burden(model, copy_number=200,
                            expressed_protein_fraction=0.50)
    assert any("拷贝数" in n or "整合" in n for n in est.notes)


def test_toehold_recognises_the_trigger_three_prime_end():
    trigger = "GGGAUUUAGCUCAGUUGGGAGAGCGCCAGA"
    sw = sb.toehold_switch(trigger, toehold_length=12)
    assert sb.reverse_complement(sw.toehold.replace("U", "T")) == \
        trigger[-12:].replace("U", "T")


def test_toehold_sequesters_the_rbs_and_start_codon():
    sw = sb.toehold_switch("GGGAUUUAGCUCAGUUGGGAGAGCGCCAGA")
    assert sw.rbs in sw.switch_rna
    assert "AUG" in sw.switch_rna
    assert sw.start_codon_position > len(sw.toehold)


def test_toehold_rejects_a_trigger_that_is_too_short():
    with pytest.raises(ValueError, match="至少|缩短"):
        sb.toehold_switch("GGGAUUU", toehold_length=12, stem_length=9)


def test_toehold_flags_a_weak_stem():
    sw = sb.toehold_switch("AAAAAAAAAAAAAAAAAAAAAAAA")
    assert any("GC" in n for n in sw.notes)


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def _agg():
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    return plt


def test_plot_plate(_agg):
    fig, ax = sb.plot_plate(sb.plate_layout(["a", "b"], controls=["blank"]),
                            highlight=["blank"])
    _agg.close(fig)


def test_plot_growth_curves_single(_agg, synthetic_curve):
    t, od = synthetic_curve
    fit = sb.fit_growth_curve(t, od, blank=BLANK)
    fig, ax = sb.plot_growth_curves(t, od, fit, blank=BLANK)
    _agg.close(fig)


def test_plot_growth_curves_plate(_agg, plate_frame):
    fig, ax = sb.plot_growth_curves(plate_frame, blank=BLANK)
    _agg.close(fig)


def test_plot_dose_response(_agg):
    conc = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    resp = 1.0 / (1.0 + (conc / 0.8) ** 1.6)
    fig, ax = sb.plot_dose_response(conc, resp, sb.dose_response(conc, resp))
    _agg.close(fig)


def test_plot_doe_effects(_agg):
    des = sb.doe_design(FACTORS, design="central_composite")
    y = _quadratic_response(des.to_frame().values, np.random.default_rng(3))
    fig, axes = sb.plot_doe_effects(sb.analyse_doe(des, y))
    assert len(axes) == 2
    _agg.close(fig)


def test_plot_optimization_progress(_agg, observed):
    X, y = observed
    camp = sb.dbtl_campaign(FACTORS)
    camp.record(X, y)
    camp.record(X[:3], y[:3] + 1.0)
    fig, axes = sb.plot_optimization_progress(camp)
    assert len(axes) == 2
    _agg.close(fig)


def test_plot_expression_library(_agg):
    pytest.importorskip("RNA", reason="needs omicverse[synbio] (ViennaRNA)")
    lib = sb.rbs_library(sb.GAPDH_CDS, n=4, n_candidates=200)
    fig, ax = sb.plot_expression_library(lib)
    _agg.close(fig)


def test_plot_integration_sites(_agg):
    fig, ax = sb.plot_integration_sites(sb.integration_sites())
    _agg.close(fig)


# ---------------------------------------------------------------------------
# the real dataset the tutorial fits
# ---------------------------------------------------------------------------

def _growth_dataset():
    pytest.importorskip("pyreadr", reason="the dataset is an R .rda file")
    try:
        return sb.fetch_growth_dataset()
    except RuntimeError as exc:                      # no network on this runner
        pytest.skip(str(exc)[:80])


def test_real_dataset_has_the_expected_shape():
    od = _growth_dataset()
    assert od.shape[1] == 97, "time column plus 96 wells"
    assert od["time_h"].min() == 0.0 and od["time_h"].max() == pytest.approx(24.0)


def test_real_dataset_has_no_blank_wells():
    """Documented because it is a trap: 95 of the 96 wells grow six- to
    thirteen-fold, column 12 included. Nominating a column as the blank and
    subtracting it drops the median R² from 0.998 to 0.39."""
    od = _growth_dataset()
    wells = [c for c in od.columns if c != "time_h"]
    grew = [w for w in wells
            if od[w].iloc[-1] > 3.0 * od[w].iloc[0]]
    assert len(grew) >= 90, f"only {len(grew)} wells grew — is this the right set?"
    for well in (f"{r}12" for r in "ABCDEFGH"):
        assert od[well].iloc[-1] > 3.0 * od[well].iloc[0], (
            f"{well} is not a blank in this dataset")


def test_every_well_of_the_real_run_fits_well():
    """The claim the tutorial rests on. A fitter validated only on curves from
    its own equations has not been validated."""
    od = _growth_dataset()
    baseline = float(od.iloc[0, 1:].mean())
    fits = sb.fit_growth_curves(od, blank=baseline)
    assert len(fits) == 96
    assert fits["r_squared"].median() > 0.99
    assert int(fits["good_fit"].sum()) >= 90


def test_real_fits_give_biologically_plausible_parameters():
    od = _growth_dataset()
    fits = sb.fit_growth_curves(od, blank=float(od.iloc[0, 1:].mean()))
    assert (fits["doubling_time_h"] > 0.1).all()
    assert (fits["lag_h"] >= 0).all()
    assert (fits["carrying_capacity"] > 0).all()
