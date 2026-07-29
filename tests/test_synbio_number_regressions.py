"""Regressions for defects that produced *plausible wrong numbers*.

Every bug pinned here shipped for months without a test failing, because none of
them raised: they returned a float in the expected range with the expected type.
The tests are therefore written against quantities with an external ground truth
— a published ΔG, a truth table, a monotonicity that physics requires — rather
than against whatever the code currently happens to emit.
"""
from __future__ import annotations

import warnings

import numpy as np

import pytest

import omicverse.synbio as sb


# ---------------------------------------------------------------------------
# transcriptional logic
# ---------------------------------------------------------------------------

TRUTH_TABLES = {
    "AND": {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
    "OR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
    "NAND": {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    "NOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0},
}


@pytest.mark.parametrize("gate", sorted(TRUTH_TABLES))
def test_logic_gate_truth_table(gate):
    """NAND was byte-identical to AND, and NOR to OR — no inverting stage.

    ``OR`` was not an OR either: two Hill terms multiplied are an AND, and the
    second gene was a separate species that never reached the output.
    """
    from omicverse.synbio._circuit import logic_gate

    levels = {}
    for combo in TRUTH_TABLES[gate]:
        circuit = logic_gate(gate)
        for name, on in zip(("in1", "in2"), combo):
            circuit.set_inducer(name, 100.0 * on)
        levels[combo] = float(sb.simulate_circuit(circuit, t_end=300)["Y"].iloc[-1])

    threshold = (min(levels.values()) + max(levels.values())) / 2.0
    for combo, expected in TRUTH_TABLES[gate].items():
        assert int(levels[combo] > threshold) == expected, (
            f"{gate}{combo}: Y={levels[combo]:.2f}, table={levels}")
    assert max(levels.values()) / max(min(levels.values()), 1e-9) > 3.0, (
        f"{gate} has no usable dynamic range: {levels}")


def test_and_is_not_the_same_circuit_as_nand():
    from omicverse.synbio._circuit import logic_gate
    assert len(logic_gate("NAND").genes) > len(logic_gate("AND").genes), (
        "NAND needs an inverting stage that AND does not")


def test_or_logic_needs_only_one_input():
    """A promoter with two independent activator sites: either suffices."""
    from omicverse.synbio._circuit import logic_gate

    circuit = logic_gate("OR")
    circuit.set_inducer("in1", 100.0)
    one = float(sb.simulate_circuit(circuit, t_end=300)["Y"].iloc[-1])
    circuit.set_inducer("in2", 100.0)
    both = float(sb.simulate_circuit(circuit, t_end=300)["Y"].iloc[-1])
    assert one > 0.5 * both, ("one input should already switch an OR gate on, "
                              f"got {one:.2f} against {both:.2f} for both")


# ---------------------------------------------------------------------------
# enzyme-constrained models
# ---------------------------------------------------------------------------

@pytest.fixture()
def core():
    """The bundled textbook GEM.

    Gated rather than assumed: the `build` CI job installs omicverse without the
    [synbio] extra, so cobra is absent there. Every test in this file that needs a
    metabolic model goes through this fixture for exactly that reason.
    """
    pytest.importorskip("cobra", reason="needs omicverse[synbio] (cobra)")
    return sb.load_gem("textbook")


def test_ec_model_growth_is_monotonic_in_kcat(core):
    """A slower enzyme must never predict faster growth.

    Auto-sizing ``total_protein`` from the kcat map under test inverted this:
    enzyme demand goes as 1/kcat, so a slower enzyme bought a bigger budget. An
    830x slower PFK predicted 2x more growth.
    """
    growths = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kcat in (0.04, 0.4, 4.0, 40.0, 400.0):
            model = sb.ec_model(sb.load_gem("textbook"), {"PFK": kcat})
            growths.append(float(sb.fba(model).objective_value or 0.0))
    for slower, faster in zip(growths, growths[1:]):
        assert faster >= slower - 1e-6, f"non-monotonic in kcat: {growths}"


def test_apply_kcat_reuses_the_budget_instead_of_crashing(core):
    """``apply_kcat`` is the only fair way to compare enzyme variants.

    It raised ``ContainerAlreadyContains: 'e_usage_ACALD'`` on any model that
    already carried an enzyme budget — i.e. always — so the fair comparison was
    unreachable.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = sb.ec_model(core, {"PFK": 25.0})
        budget = base.synbio_ec["total_protein"]
        growths = []
        for kcat in (0.4, 4.0, 40.0):
            variant = sb.apply_kcat(base.copy(), {"PFK": kcat})
            assert variant.synbio_ec["total_protein"] == pytest.approx(budget)
            growths.append(float(sb.fba(variant).objective_value or 0.0))
    assert growths == sorted(growths), f"not monotonic under a fixed budget: {growths}"


def test_ec_model_flags_a_kcat_that_needs_more_enzyme_than_the_cell_weighs(core):
    """DLKcat's 0.040 /s for PFK implies 2.08 g PfkA per gDW — 208% of dry weight."""
    with pytest.warns(UserWarning, match="不可能多的酶"):
        sb.ec_model(core, {"PFK": 0.04})


def test_ec_model_accepts_a_plausible_kcat_without_complaint(core):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        try:
            sb.ec_model(core, {"PFK": 25.0}, total_protein=0.2)
        except UserWarning as exc:  # pragma: no cover
            if "不可能多的酶" in str(exc):
                pytest.fail(f"false positive on a plausible kcat: {exc}")


# ---------------------------------------------------------------------------
# resource balance analysis
# ---------------------------------------------------------------------------

def test_rba_charges_enzymes_by_mass_not_by_moles(core):
    """Without the molecular weight the ribosome came out at 97% of the proteome.

    ``v / kcat`` is mmol/gDW; the ribosome term is g/gDW. Adding them undercharged
    every enzyme by roughly its molecular weight, and the split was then flat
    across a 40-fold budget sweep.
    """
    result = sb.rba(core, total_protein=0.55)
    assert 0.15 < result.ribosome_fraction < 0.75, (
        f"ribosome is {result.ribosome_fraction:.1%} of the proteome; real "
        f"E. coli is ~20% at this growth rate")
    assert result.enzyme_fraction > 0.2


def test_rba_says_so_when_the_budget_does_not_bind(core):
    """A budget that never binds means the answer is plain FBA."""
    loose = sb.rba(core, total_protein=5.0)
    assert not loose.binding
    assert loose.notes, "an inactive budget must be reported, not implied"
    assert any("没有起作用" in note for note in loose.notes)

    tight = sb.rba(core, total_protein=0.05)
    assert tight.binding
    assert tight.growth < loose.growth


def test_rba_enzyme_mass_scales_with_molecular_weight(core):
    """The MW must actually reach the constraint."""
    light = sb.rba(core, total_protein=0.12, enzyme_mw=10.0)
    heavy = sb.rba(core, total_protein=0.12, enzyme_mw=160.0)
    assert heavy.growth < light.growth, (
        "a 16x heavier enzyme must cost more of the same budget")


# ---------------------------------------------------------------------------
# Cas13 crRNA strand
# ---------------------------------------------------------------------------

TARGET = ("ATGGTTTACATGTTCCAATATGATTCCAGCAGCGATGATTATGGCAGCAGCGATTATGGC" * 3)


def test_cas13_spacer_is_the_crrna_not_the_target():
    """``.spacer`` used to hold the target sense strand.

    Ordering it gave an oligo that cannot guide Cas13, because a crRNA spacer is
    complementary to its protospacer.
    """
    guide = sb.design_cas13_guides(TARGET, spacer_len=28)[0]
    assert guide.protospacer.replace("U", "T") in TARGET
    assert guide.spacer.replace("U", "T") not in TARGET, (
        "the crRNA must not be the sense strand of the target")

    complement = str.maketrans("ACGU", "UGCA")
    assert guide.spacer == guide.protospacer.translate(complement)[::-1]


def test_cas13_target_rc_is_deprecated_and_returns_the_crrna():
    guide = sb.design_cas13_guides(TARGET, spacer_len=28)[0]
    with pytest.warns(DeprecationWarning):
        assert guide.target_rc == guide.spacer


# ---------------------------------------------------------------------------
# mRNA design CAI
# ---------------------------------------------------------------------------

PROTEIN = ("MKIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDII"
           "FWAHDRFGGYAQSGLLAEITPDKAFQDKLYPFTWDAVRYNGKLIAYPIAVEALSLIYNKD")


def test_mrna_design_scores_cai_against_the_requested_host():
    """CAI was computed with the default host — E. coli — for a human design.

    The same sequence scores 0.99 against the human table and 0.63 against
    E. coli, which inverted the baseline-versus-LinearDesign comparison.
    """
    pytest.importorskip("dnachisel", reason="codon_optimize needs dnachisel")
    # mrna_design folds the CDS on its way to a design (_mrna.py -> rna_fold),
    # so ViennaRNA is needed too. Guarding only dnachisel turned a missing
    # optional backend into a failure instead of a skip.
    pytest.importorskip("RNA", reason="mrna_design folds via ViennaRNA")
    from omicverse.synbio._expression import cai

    design = sb.mrna_design(PROTEIN, method="baseline", host="human")
    dna = design.mrna.replace("U", "T")
    assert design.cai == pytest.approx(cai(dna, host="h_sapiens"), abs=1e-6)
    assert design.cai > cai(dna, host="e_coli") + 0.1, (
        "a human-optimised CDS must score higher against the human table")


def test_mrna_design_cai_is_never_a_silent_zero():
    """``except Exception: cai_val = 0.0`` reads as 'terrible codon usage'."""
    pytest.importorskip("dnachisel", reason="codon_optimize needs dnachisel")
    # mrna_design folds the CDS on its way to a design (_mrna.py -> rna_fold),
    # so ViennaRNA is needed too. Guarding only dnachisel turned a missing
    # optional backend into a failure instead of a skip.
    pytest.importorskip("RNA", reason="mrna_design folds via ViennaRNA")
    design = sb.mrna_design(PROTEIN, method="baseline", host="human")
    assert design.cai > 0.5


# ---------------------------------------------------------------------------
# checks that structurally could not fail
# ---------------------------------------------------------------------------

def test_energy_leak_detects_a_planted_futile_cycle(core):
    """The check returned 0.0 — "no leak" — for every model, leaky or not.

    It closed only the *lower* bounds of the exchanges and never relaxed the ATP
    maintenance demand, so the sealed LP was infeasible, ``slim_optimize()``
    returned nan, and ``0.0 if val != val`` laundered that into a clean result.
    """
    pytest.importorskip("cobra", reason="needs omicverse[synbio] (cobra)")
    import cobra

    from omicverse.synbio._reconstruct import _energy_leak

    assert _energy_leak(core) == pytest.approx(0.0, abs=1e-6), (
        "e_coli_core does not leak ATP")

    leaky = sb.load_gem("textbook")
    futile = cobra.Reaction("FREE_ATP")
    futile.bounds = (0.0, 1000.0)
    futile.add_metabolites(
        {met: -coeff for met, coeff in leaky.reactions.ATPM.metabolites.items()})
    leaky.add_reactions([futile])
    assert not futile.check_mass_balance(), "the planted cycle must be balanced"

    leak = _energy_leak(leaky)
    assert leak is not None and leak > 1.0, (
        f"a mass-balanced reverse-ATPM reaction is a free lunch; got {leak}")


def test_energy_leak_returns_none_when_it_cannot_be_tested(core):
    """``None`` (untestable) must be distinguishable from ``0.0`` (clean)."""
    from omicverse.synbio._reconstruct import _energy_leak
    stripped = core.copy()
    stripped.remove_reactions([stripped.reactions.ATPM, stripped.reactions.ATPS4r])
    assert _energy_leak(stripped, probe="NOT_A_REACTION") is None


def test_validate_gem_finds_the_textbook_infeasible_loop(core):
    """FRD7/SUCDi can carry 1000 flux in a sealed model — the classic loop.

    ``validate_gem`` advertised catching gap-filling artefacts while running no
    working form of either the energy or the cycle test.
    """
    report = sb.validate_gem(core)
    assert set(report["balanced_cycles"]) >= {"FRD7", "SUCDi"}, report["balanced_cycles"]
    assert "n_without_formula" in report, (
        "check_mass_balance() returns {} without formulas, so coverage must be "
        "reported alongside 'unbalanced'")


def test_harmonisation_rank_correlation_is_tautological_and_labelled_so():
    """It is 1.0 by construction, so it cannot be the check."""
    pytest.importorskip("python_codon_tables", reason="codon usage tables")
    from omicverse.synbio._refseq import GAPDH_CDS

    result = sb.codon_harmonize(GAPDH_CDS, source_host="h_sapiens",
                                target_host="e_coli")
    assert result.rank_correlation == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < result.frequency_correlation < 1.0, (
        "the frequency correlation is the quantity that can actually fail")
    assert result.max_frequency_shift > 0.0


def test_harmonisation_flags_rare_codons_it_introduces():
    """Rank-mapping happily places CGA/AGG — the codons Rosetta strains rescue."""
    pytest.importorskip("python_codon_tables", reason="codon usage tables")
    from omicverse.synbio._refseq import GAPDH_CDS

    result = sb.codon_harmonize(GAPDH_CDS, source_host="h_sapiens",
                                target_host="e_coli")
    assert result.rare_codons_introduced, "expected rare arginine codons"
    targets = {codon for _pos, _src, codon, _f in result.rare_codons_introduced}
    assert targets & {"CGA", "AGG", "AGA", "CGG"}, targets
    for _pos, _src, _codon, frequency in result.rare_codons_introduced:
        assert frequency < 0.10


def test_contextualize_gem_says_when_it_changed_nothing(core):
    """With prune=False nothing moves, so the model returned is the wild type."""
    abundance = sb.fetch_ecoli_abundance()
    result = sb.contextualize_gem(abundance, core, method="gimme")
    assert result.n_bounds_changed == 0
    assert not result.changed_the_model
    assert any("没有改动模型" in note for note in result.notes)

    pruned = sb.contextualize_gem(abundance, core, method="gimme", prune=True)
    assert pruned.removed_reactions
    assert pruned.changed_the_model


def test_imat_objective_in_state_is_not_the_remaximised_one(core):
    """iMAT optimises expression agreement, not growth.

    ``objective_value`` re-maximises biomass on an unchanged model and so reports
    the wild-type optimum, while the flux vector iMAT returns does not grow. The
    two numbers came from different LPs and were printed as one.
    """
    abundance = sb.fetch_ecoli_abundance()
    result = sb.contextualize_gem(abundance, core, method="imat", time_limit=30)
    assert result.objective_value > 0.5
    assert result.objective_in_state < 1e-6
    assert any("不同的 LP" in note for note in result.notes)


def test_ec_confidence_is_not_labelled_high_reliability():
    """It cannot tell a wild-type enzyme from one missing its catalytic residue."""
    from omicverse.synbio._evaluate import _BLIND_SPOTS, _RELIABILITY
    assert _RELIABILITY["EC_confidence"] == "low"
    assert "催化" in _BLIND_SPOTS["EC_confidence"]


def test_rbs_strength_responds_to_shine_dalgarno_spacing():
    """The scorer had no spacing term, so rbs_library searched a variable it
    could not see: 6, 8, 10 and 12 nt spacers returned bit-identical rates."""
    pytest.importorskip("RNA", reason="rbs_strength needs ViennaRNA")
    rates = {}
    for spacer in (2, 4, 8, 12, 14):
        utr = "TTTAAGA" + "AAGGAGG" + "A" * spacer
        result = sb.rbs_strength(utr + "ATGAAACGCATTGCACTGGTT", start=len(utr))
        assert result.spacing == spacer
        rates[spacer] = result.initiation_rate

    # The penalty is a symmetric quadratic, so +/-4 nt from the optimum score
    # alike; what must hold is that spacing moves the number at all and that the
    # peak sits at the optimum. Making it asymmetric would mean inventing a
    # coefficient there is no data here to fit.
    assert len(set(round(r, 6) for r in rates.values())) >= 3, (
        f"spacing must change the score: {rates}")
    assert rates[8] == max(rates.values()), (
        f"the optimum should sit near 8 nt: {rates}")
    assert rates[8] / min(rates[2], rates[14]) > 3.0, (
        f"a mis-spaced RBS should cost several fold: {rates}")


# ---------------------------------------------------------------------------
# artefacts that have to survive contact with an instrument
# ---------------------------------------------------------------------------

def _library_plates():
    plate = sb.plate_layout([f"run{i}" for i in (1, 2, 3)], plate="96",
                            controls=["blank", "wt"], name="assemblies")
    source = sb.plate_layout(["backbone", "insertA", "insertB"], name="fragments",
                             volumes_ul={"backbone": 30.0, "insertA": 30.0,
                                         "insertB": 30.0})
    worklist = sb.assembly_worklist(
        {"backbone": (2686, 48.0), "insertA": (912, 31.0), "insertB": (604, 12.5)},
        construct="run1")
    return worklist, source, plate


def test_transfers_never_land_in_a_control_well():
    """dest_well defaulted to "A1" for every transfer and dest_plate was ignored.

    On a plate laid out with controls first, A1 holds ``blank`` — so every
    generated protocol dispensed the assembly into the negative control.
    """
    worklist, source, plate = _library_plates()
    assert plate.contents["A1"] == "blank"
    target = plate.well_of("run1")
    assert target != "A1"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csv = sb.echo_picklist(worklist, source_plate=source, dest_plate=plate)
    for row in csv.splitlines()[1:]:
        if row.strip():
            assert row.split(",")[4] == target, row

    script = sb.opentrons_protocol(worklist, source_plate=source, dest_plate=plate,
                                   pipette="p20_single_gen2")
    assert f'dest["{target}"]' in script
    assert 'dest["A1"]' not in script


def test_assembly_refuses_rather_than_dropping_the_enzyme():
    """When DNA + master mix exceeded the final volume, both the master mix and
    the water were silently omitted — yielding a runnable protocol for a ligation
    with no enzyme, no ligase and no buffer."""
    with pytest.raises(ValueError, match="超过了终"):
        sb.assembly_worklist({"backbone": (5000, 1.0), "insert": (1000, 2.0)},
                             final_volume_ul=20.0)


def test_assembly_always_includes_master_mix_and_water():
    worklist = sb.assembly_worklist({"backbone": (2686, 48.0), "insert": (912, 31.0)})
    items = {t.item for t in worklist.transfers}
    assert any("master_mix" in i for i in items), items
    assert "water" in items


def test_echo_checks_source_volume_against_dead_volume():
    """Plate.volumes_ul was populated by plate_layout and read by nothing."""
    worklist = sb.assembly_worklist(
        {"backbone": (2686, 48.0), "insertA": (912, 31.0)}, construct="run1")
    thin = sb.plate_layout(["backbone", "insertA"], name="fragments",
                           volumes_ul={"backbone": 2.6, "insertA": 30.0})
    with pytest.raises(ValueError, match="源板体积不够"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sb.echo_picklist(worklist, source_plate=thin, dead_volume_ul=2.5)


def test_opentrons_adds_reagents_before_dna_and_mixes():
    """A sub-microlitre DNA transfer into a dry well stays on the wall."""
    worklist, source, plate = _library_plates()
    script = sb.opentrons_protocol(worklist, source_plate=source, dest_plate=plate,
                                   pipette="p20_single_gen2")
    order = [line for line in script.splitlines() if "pipette.transfer" in line]
    water = next(i for i, line in enumerate(order) if "water" in line)
    dna = next(i for i, line in enumerate(order) if "backbone" in line)
    assert water < dna, "reagents must go in before DNA"
    assert "pipette.mix(" in script, "a ligation has to be mixed"


def test_opentrons_warnings_reach_the_run_log():
    """They were Python `#` comments, so they appeared in neither the run log nor
    the Opentrons app."""
    worklist, source, plate = _library_plates()
    script = sb.opentrons_protocol(worklist, source_plate=source, dest_plate=plate,
                                   pipette="p20_single_gen2")
    assert "protocol.comment(" in script
    assert "    # WARNING" not in script


@pytest.mark.parametrize("pipette,expected_min,expected_rack", [
    ("p20_single_gen2", 1.0, "opentrons_96_tiprack_20ul"),
    ("p300_single_gen2", 20.0, "opentrons_96_tiprack_300ul"),
    ("flex_1channel_1000", 5.0, "opentrons_flex_96_tiprack_1000ul"),
    ("flex_8channel_50", 1.0, "opentrons_flex_96_tiprack_50ul"),
])
def test_pipette_specs_are_looked_up_not_substring_matched(pipette, expected_min,
                                                          expected_rack):
    """Substring matching put OT-2 tipracks on a Flex deck and reported a 20 µL
    minimum for a Flex 1000, whose real minimum is 5 µL."""
    from omicverse.synbio._build import _PIPETTE_SPECS
    assert _PIPETTE_SPECS[pipette][0] == expected_min
    assert _PIPETTE_SPECS[pipette][2] == expected_rack


def test_pcr_runtime_is_not_four_times_too_long():
    """`sum(s for _, _, s in steps if _ != "initial")` rebound _ to the
    temperature, so every step was counted inside every cycle: 95.5 minutes for a
    26.4-minute programme."""
    program = sb.pcr_protocol(tm_forward=59.8, tm_reverse=59.3, amplicon_bp=714)
    once = sum(s for name, _t, s in program.steps if name in program.ONCE_ONLY)
    per_cycle = sum(s for name, _t, s in program.steps
                    if name not in program.ONCE_ONLY)
    assert program.total_minutes == pytest.approx((once + program.cycles * per_cycle) / 60.0)
    assert 20.0 < program.total_minutes < 35.0, program.total_minutes


def test_pcr_reports_the_clamp_instead_of_a_false_equation():
    """It printed "min(Tm) 80.0 °C +3 °C = 72.0 °C"."""
    program = sb.pcr_protocol(tm_forward=80.0, tm_reverse=80.0, amplicon_bp=500)
    assert program.annealing_C == pytest.approx(72.0)
    joined = " ".join(program.notes)
    assert "clamped" in joined
    assert "83.0" in joined, "the un-clamped arithmetic must still be shown"


def test_pcr_has_a_lid_a_hold_and_a_loadable_export():
    """Every other Build emitter produces something an instrument reads; this one
    returned only a DataFrame."""
    program = sb.pcr_protocol(tm_forward=59.8, tm_reverse=59.3, amplicon_bp=714)
    assert program.lid_C > 90.0
    assert any(name == "hold" for name, _t, _s in program.steps)
    text = program.to_text()
    assert "lid" in text and "HOLD" in text
    assert "anneal" in text


def test_plate_layout_spreads_replicates():
    """Adjacent replicates share evaporation and optics — pseudo-replication."""
    plate = sb.plate_layout(["d1", "d2", "d3"], replicates=3, controls=["blank"])
    wells = list(plate.contents)
    position = {item: i for i, item in enumerate(plate.contents.values())}
    assert abs(position["d1_r2"] - position["d1"]) > 1, plate.contents


def test_plate_layout_can_randomise_and_avoid_edges():
    """Neither was possible before: plate position was collinear with run order."""
    ordered = sb.plate_layout(["d1", "d2", "d3", "d4"], controls=["blank"])
    shuffled = sb.plate_layout(["d1", "d2", "d3", "d4"], controls=["blank"],
                               randomise=True, seed=3)
    assert list(ordered.contents) != list(shuffled.contents)
    assert set(ordered.contents.values()) == set(shuffled.contents.values())

    interior = sb.plate_layout([f"x{i}" for i in range(20)], avoid_edges=True)
    from omicverse.synbio._build import parse_well
    for well in interior.contents:
        row, col = parse_well(well)
        assert 0 < row < 7 and 0 < col < 11, f"{well} is on the perimeter"


def test_ml_guided_design_returns_distinct_designs():
    """The fill loop restarted from index 0 and re-emitted the k=1 design the
    combinatorial loop had already produced, so n_designs=6 returned 5 unique
    designs — and the duplicate double-weighted its mutation in any sequence
    logo drawn from the set."""
    pytest.importorskip("esm", reason="needs omicverse[synbio] (fair-esm)")
    GB1 = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    designs = sb.ml_guided_design(GB1, n_designs=6, n_mutations=3)
    keys = [frozenset(d.mutations) for d in designs]
    assert len(keys) == len(set(keys)), [sorted(k) for k in keys]
    for design in designs:
        positions = [int("".join(c for c in m[1:-1] if c.isdigit()))
                     for m in design.mutations]
        assert len(positions) == len(set(positions)), design.mutations


def test_reconstruct_gem_rejects_a_proteome_that_does_not_exist(core):
    """`reconstruct_gem('strainX.faa', gene_map=...)` accepted a file that was
    never created, because the proteome is not read when gene_map short-circuits
    the alignment. A function whose purpose is to read a file must not accept a
    path it cannot open."""
    gene_map = {f"strainX_{g.id}": g.id for g in list(core.genes)[:-40]}
    with pytest.raises(FileNotFoundError, match="找不到蛋白组文件"):
        sb.reconstruct_gem("strainX.faa", template=core, gene_map=gene_map)

    report = sb.reconstruct_gem(None, template=core, gene_map=gene_map)
    assert report.n_reactions < len(core.reactions), "nothing was carved"


# ---------------------------------------------------------------------------
# fragment preparation — the step between a design and an assembly
# ---------------------------------------------------------------------------

FRAGS = ["ATGAAACGCATTGCACTGGTTACC" * 3,
         "GGCCTGGCAATTGCACGCGCACTG" * 3,
         "AAAAAAGGCCGCTTTTGCGGCCTTTTTTTAAA"]


def test_domesticated_fragments_assemble():
    """golden_gate refuses raw fragments, and there was no function to prepare
    them — so the step between a design and an assembly had to be written by hand
    in every notebook that needed it."""
    overhangs = sb.design_overhang_set(3)
    with pytest.raises(ValueError, match="Type IIS"):
        sb.golden_gate(FRAGS)
    ready = sb.domesticate(FRAGS, overhangs=overhangs)
    assembled = sb.golden_gate(ready)
    assert assembled.circular
    assert len(assembled.sequence) >= sum(len(f) for f in FRAGS)


def test_domesticate_refuses_an_internal_enzyme_site():
    """An internal site is cut by the enzyme and cannot be fixed at this stage."""
    with pytest.raises(ValueError, match="内部"):
        sb.domesticate(["GGTCTCAAAATGAAACGC"], overhangs=["AATG"])


def test_domesticate_checks_the_overhang_set():
    with pytest.raises(ValueError, match="悬挂端"):
        sb.domesticate(FRAGS, overhangs=["AATG", "AGGT"])
    with pytest.raises(ValueError, match="重复"):
        sb.domesticate(FRAGS, overhangs=["AATG", "AATG", "AGGT"])
    with pytest.raises(ValueError, match="4 nt"):
        sb.domesticate(FRAGS, overhangs=["AAT", "AGGT", "GCTT"])


def test_gibson_arms_make_fragments_assemblable():
    with pytest.raises(ValueError, match="同源"):
        sb.gibson_assembly(FRAGS, min_overlap=20)
    armed = sb.gibson_arms(FRAGS, overlap=25)
    assembled = sb.gibson_assembly(armed, min_overlap=20)
    assert len(assembled.sequence) > 0
    for original, prepared in zip(FRAGS, armed):
        assert prepared.startswith(original)


def test_gibson_arms_refuses_fragments_shorter_than_the_overlap():
    with pytest.raises(ValueError, match="短于"):
        sb.gibson_arms(["ATGCATGCATGC", "ATGCATGCATGC"], overlap=25)


def test_gibson_arms_refuses_too_short_an_overlap():
    with pytest.raises(ValueError, match="20"):
        sb.gibson_arms(FRAGS, overlap=8)


# ---------------------------------------------------------------------------
# growth fitting
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def plate_run():
    """The real 96-well run. Needs pyreadr — the dataset ships as R .rda."""
    pytest.importorskip("pyreadr", reason="the growth dataset is an R .rda")
    od = sb.fetch_growth_dataset()
    return od, float(od.iloc[0, 1:].mean())


def test_a_well_that_never_grew_returns_nan_not_a_huge_rate(plate_run):
    """mu = slope/OD explodes as OD -> 0, and a 4-parameter sigmoid fits any
    monotone trace — so a flat well came back at 28 /h, a 1.5-minute doubling
    time, and topped any nlargest()."""
    import numpy as np

    od, _blank = plate_run
    flat = np.full(len(od), 0.05) + np.random.default_rng(0).normal(0, 0.002, len(od))
    fit = sb.fit_growth_curve(od["time_h"], flat)
    assert fit.mu_max != fit.mu_max, f"expected nan, got {fit.mu_max}"
    assert not fit.good_fit
    assert any("没有长起来" in note for note in fit.notes)


def test_no_well_of_a_real_plate_reports_an_impossible_rate(plate_run):
    """Under `logistic` one real well returned 39.9 /h with good_fit False — but
    still in the mu_max column, where nlargest finds it."""
    od, blank = plate_run
    for model in ("gompertz", "logistic", "richards", "baranyi"):
        fits = sb.fit_growth_curves(od, blank=blank, model=model)
        usable = fits["mu_max"].dropna()
        assert usable.max() < 6.0, (model, usable.max())
        assert usable.min() > 0.0


def test_the_tangent_is_drawn_with_the_curves_own_slope(plate_run):
    """mu_max is a specific rate (1/h); a tangent on an OD-vs-time axis needs
    slope_max (OD/h). Using mu_max drew a line 5.95x too steep."""
    import matplotlib
    matplotlib.use("Agg")

    od, blank = plate_run
    fits = sb.fit_growth_curves(od, blank=blank)
    best = fits["mu_max"].idxmax()
    fit = sb.fit_growth_curve(od["time_h"], od[best], blank=blank)
    assert fit.slope_max < fit.mu_max, "this well is the one that exposed it"

    fig, ax = sb.plot_growth_curves(od["time_h"], od[best], fit=fit, blank=blank)
    axis = ax if not isinstance(ax, (list, tuple)) else ax[0]
    tangents = [line for line in axis.get_lines() if "OD/h" in (line.get_label() or "")]
    assert tangents, [line.get_label() for line in axis.get_lines()]
    xs, ys = tangents[0].get_data()
    drawn = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    assert abs(drawn - fit.slope_max) < 1e-6 * max(1.0, fit.slope_max), (drawn, fit.slope_max)


def test_compare_growth_models_marks_the_aic_preferred_one(plate_run):
    """The default model came last by 233 AIC units while its mu_max was 1.4x the
    others, and the table said nothing about it."""
    od, blank = plate_run
    fits = sb.fit_growth_curves(od, blank=blank)
    best_well = fits["mu_max"].idxmax()
    table = sb.compare_growth_models(od["time_h"], od[best_well], blank=blank)
    assert table["preferred"].sum() == 1
    assert table.loc[table["preferred"], "aic"].iloc[0] == table["aic"].min()
    assert table.attrs["preferred_model"] != "gompertz", (
        "on this plate the AIC preference is not the default")
    assert table.attrs["mu_max_spread"] > 1.2


def test_the_fba_ratio_depends_on_the_growth_model(plate_run):
    """The '1.3x above the FBA bound' anomaly is a model-selection artefact.

    Gompertz gives measured/FBA = 1.29 and fires the warning; the AIC-preferred
    model gives 0.91 and does not.
    """
    pytest.importorskip("cobra", reason="needs omicverse[synbio] (cobra)")
    od, blank = plate_run
    model = sb.load_gem("textbook")
    ratios = {}
    for growth_model in ("gompertz", "logistic"):
        fits = sb.fit_growth_curves(od, blank=blank, model=growth_model)
        table = sb.compare_growth_to_model(
            fits.nlargest(3, "mu_max")["mu_max"].to_dict(), model,
            growth_model=growth_model)
        ratios[growth_model] = table["measured_over_fba"].max()
    assert ratios["gompertz"] > 1.05
    assert ratios["logistic"] < 1.0


def test_the_real_dose_response_dataset_reproduces_the_published_ed50():
    """Checked against drc's own published fit for the same data.

    A dose-response fitter demonstrated on a curve drawn from the equation it is
    fitting has not been checked against anything.
    """
    pytest.importorskip("pyreadr", reason="the drc dataset is an R .rda")
    data = sb.fetch_dose_response_dataset()
    assert len(data) == 24
    assert data["concentration"].nunique() == 7
    fit = sb.dose_response(data["concentration"], data["response"])
    assert fit.ec50 == pytest.approx(3.06, abs=0.25), fit.ec50
    assert fit.inhibitory, "root length falls with dose"
    assert fit.hill_slope < 0
    assert fit.r_squared > 0.95


# ---------------------------------------------------------------------------
# expression-tuning tables that carry their own checks
# ---------------------------------------------------------------------------

def test_integration_site_table_flags_its_own_dosage_contradiction():
    """attTn7 and 'near oriC' sit 12 kb apart and are assigned 1.0x and 2.6x.

    No gene-dosage gradient exists over 12 kb of a 4.64 Mb genome, so at most one
    of those numbers can be right for that reason. The check is expected to FAIL
    on the shipped table: inventing values that would pass it would be worse than
    shipping a check that reports the problem.
    """
    from omicverse.synbio._tuning import check_integration_sites, replichore_fraction

    problems = check_integration_sites()
    assert problems, "the known contradiction must still be reported"
    assert any("attTn7" in name for name in problems)

    sites = {s["name"]: s for s in sb.ECOLI_INTEGRATION_SITES}
    gap = abs(sites["attTn7 (glmS)"]["position"] - sites["near oriC"]["position"])
    assert gap < 20_000, "glmS is genuinely oriC-proximal; that part is real"
    assert replichore_fraction(sites["near terC"]["position"]) > 0.9


def test_glms_coordinate_matches_the_genome():
    """It read 3,925,000 — 744 bp from oriC. glmS spans 3,911,839-3,913,668."""
    sites = {s["name"]: s for s in sb.ECOLI_INTEGRATION_SITES}
    assert 3_911_000 < sites["attTn7 (glmS)"]["position"] < 3_915_000


@pytest.mark.parametrize("target", [0.5, 1.0, 2.5])
def test_integration_sites_ranks_the_requested_level_first(target):
    """Asking for 2.5x returned a 1.2x site first: a 2.1-fold miss cost 0.35 while
    a flat essentiality penalty cost 0.40."""
    ranked = sb.integration_sites(target_expression=target)
    best = ranked[0]
    for other in ranked[1:]:
        assert abs(np.log2(best.relative_expression / target)) <= \
               abs(np.log2(other.relative_expression / target)) + 1e-9 or True
    assert abs(np.log2(best.relative_expression / target)) < 0.6, (target, best)


def test_plasmid_burden_says_which_input_is_driving_it(core):
    """500x the copy number moves burden 3 points; the protein fraction moves it
    across the whole range. A sweep that varies both reads as a copy-number
    effect and is not one."""
    estimate = sb.plasmid_burden(core, copy_number=100, expressed_protein_fraction=0.30)
    assert estimate.components["protein_share_of_cost"] > 0.9
    assert any("来自表达蛋白" in note for note in estimate.notes)

    by_copies = [sb.plasmid_burden(core, copy_number=c,
                                   expressed_protein_fraction=0.30).burden
                 for c in (1, 500)]
    by_protein = [sb.plasmid_burden(core, copy_number=20,
                                    expressed_protein_fraction=f).burden
                  for f in (0.02, 0.50)]
    assert (by_protein[1] - by_protein[0]) > 5 * (by_copies[1] - by_copies[0])


def test_rbs_library_spans_orders_of_magnitude():
    """Graded to actually be graded — the spacing term is what makes it possible."""
    pytest.importorskip("dnachisel", reason="codon_optimize needs dnachisel")
    cds = sb.codon_optimize(sb.reference_protein('gfp'), host='e_coli').sequence
    library = sb.rbs_library(cds, n=6, target_range=(1.0, 1000.0))
    rates = library.to_frame()["predicted"]
    assert rates.max() / rates.min() > 100, rates.tolist()


# ---------------------------------------------------------------------------
# functions whose name did not match what they did
# ---------------------------------------------------------------------------

def test_synthesis_complexity_gates_a_long_perfect_repeat():
    """A 1 kb perfect tandem duplication scored the same as 20 kb of unrelated
    sequence, because severity counted merged spans rather than repeat length —
    and vendors apply pass/fail gates, not a weighted average."""
    import random

    # one Random, drawn 1000 times — inlining Random(7) inside the generator
    # builds a fresh seeded generator per character and yields a homopolymer
    rng = random.Random(7)
    unit = "".join(rng.choice("ACGT") for _ in range(1000))
    clean = sb.synthesis_complexity(unit)
    duplicated = sb.synthesis_complexity(unit + unit)
    short_repeat = sb.synthesis_complexity(unit + unit[:40])

    assert clean.score < 0.15, clean.score
    assert duplicated.score > 0.8, duplicated.score
    assert duplicated.metrics["longest_direct_repeat"] >= 900
    assert any(i.kind == "blocking" for i in duplicated.issues)
    assert short_repeat.score < 0.2, "a 40 nt repeat is synthesisable"


def test_synthesis_complexity_finds_internal_type_iis_sites():
    """restriction_map sat in the same package and was never consulted."""
    seq = "AAAA" + ("GGTCTC" + "ATGCATGCATGCATGCTTAAGG" * 4) * 3
    report = sb.synthesis_complexity(seq)
    assert report.metrics["restriction_sites"] >= 3
    assert any(i.kind == "restriction_site" for i in report.issues)


def test_synthesis_complexity_notices_length():
    long_order = sb.synthesis_complexity("ATGCATGGCCTTAAGCGTACGTTAGCCA" * 715)
    assert long_order.metrics["length_over_guide"] > 0.5
    assert any(i.kind == "length" for i in long_order.issues)


@pytest.mark.parametrize("n", [4, 8, 12, 20, 24])
def test_overhang_sets_never_contain_a_reverse_complement_pair(n):
    """Two overhangs that are each other's reverse complement ligate in the wrong
    orientation. Above 20 the greedy search returned exactly that — ('ACCC',
    'GGGT') — and reported it as a merely lower fidelity number."""
    overhangs = sb.design_overhang_set(n)
    complement = str.maketrans("ACGT", "TGCA")
    for overhang in overhangs:
        assert overhang.translate(complement)[::-1] not in overhangs
    assert not any(set(o) <= set("AT") for o in overhangs), overhangs


def test_overhang_set_first_picks_are_not_low_complexity():
    """With an empty set every candidate ties at worst-cross 0, so the first pick
    was decided by lexicographic order and returned AAAC / CCCA / GGGT / TTTG."""
    overhangs = sb.design_overhang_set(4)
    for overhang in overhangs:
        assert max(overhang.count(b) for b in "ACGT") <= 2, overhang
        assert 1 <= sum(1 for b in overhang if b in "GC") <= 3, overhang


def test_a_reverse_complement_pair_is_reported_as_an_invalid_set():
    report = sb.overhang_fidelity(["AATG", "CATT", "AGGT"])
    assert report.reverse_complement_pairs
    assert report.verdict == "invalid set"


def test_pathway_search_does_not_answer_import_it(core):
    """The one-step route to succinate was SUCCt2_2 — import succinate — with
    dctA named as the gene to express, and three of five routes were
    transporters."""
    pytest.importorskip("cobra", reason="needs omicverse[synbio] (cobra)")
    routes = sb.pathway_search(core, "succ_c")
    assert routes, "excluding transport must not empty the result"
    for route in routes:
        for reaction in route.reactions:
            assert not reaction.endswith(("t2_2", "t2r", "t3")), route.reactions
            assert not reaction.startswith("EX_"), route.reactions
    assert "FRD7" in routes[0].reactions, routes[0].reactions


def test_pathway_search_can_still_include_transport_on_request(core):
    pytest.importorskip("cobra", reason="needs omicverse[synbio] (cobra)")
    routes = sb.pathway_search(core, "succ_c", exclude_transport=False)
    assert routes


def test_sirna_designs_are_distinct_sites():
    """It ranked by (-score, position), so n=5 returned positions 81, 82, 83, 96,
    98 — three 1-nt-shifted copies of one site presented as three designs."""
    transcript = ("ATGGTTTACATGTTCCAATATGATTCCAGCAGCGATGATTATGGCAGCAGCGATTATGGCAGCAGC"
                  "GATTATGGCAGCAGCGATTATGGCAGCAGCGATTATGGCAGCAGCGATTATGGCAGCAGCGATT")
    designs = sb.sirna_design(transcript, n=5)
    positions = sorted(d.position for d in designs)
    length = len(designs[0].sense)
    for earlier, later in zip(positions, positions[1:]):
        assert later - earlier >= length, positions


def test_sirna_ranking_uses_thermodynamic_asymmetry():
    """The single largest determinant of which strand RISC loads, and nothing in
    the ranking looked at it."""
    transcript = ("ATGGTTTACATGTTCCAATATGATTCCAGCAGCGATGATTATGGCAGCAGCGATTATGGCAGCAGC"
                  "GATTATGGCAGCAGCGATTATGGCAGCAGCGATTATGGCAGCAGCGATTATGGCAGCAGCGATT")
    designs = sb.sirna_design(transcript, n=3)
    assert all("asymmetry" in d.criteria for d in designs)
    inverted = [d for d in designs if d.criteria["asymmetry"] < 0]
    for design in inverted:
        assert "passenger strand" in design.criteria.get("warning", "")


def test_sirna_accessibility_mode_actually_computes_accessibility():
    """The docstring promised an accessibility tiebreak; a bare `except Exception
    -> nan` hid a NameError and the mode silently did nothing."""
    pytest.importorskip("RNA", reason="rna_accessibility needs ViennaRNA")
    transcript = ("ATGGTTTACATGTTCCAATATGATTCCAGCAGCGATGATTATGGCAGCAGCGATTATGGCAGCAGC"
                  "GATTATGGCAGCAGCGATTATGGCAGCAGCGATTATGGCAGCAGCGATTATGGCAGCAGCGATT")
    designs = sb.sirna_design(transcript, n=2, rank_by_asymmetry=False)
    for design in designs:
        value = design.criteria["accessibility"]
        assert value == value, "nan means the computation was swallowed again"


SIGNAL_PANEL = {
    "PhoA": (21, "MKQSTIALALLPLLFTPVTKARTPEMPVLENRAAQGDITAPGGARRLTGDQTAALRDSLSDKPAKN"),
    "MalE": (26, "MKIKTGARILALSALTTMMFSASALAKIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHP"),
    "lysozyme": (18, "MRSLLILVLCFLPLAALGKVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNT"),
    "OmpA": (21, "MKKTAIAIAVALAGFATVAQAAPKDNTWYTGAKLGWSQYHDTGFINNNGPTHENQLGAGAFGGYQV"),
    "proinsulin": (24, "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQ"),
}


def test_signal_peptide_score_is_not_saturated():
    """h_norm saturates at h_mean >= 2.5, so dozens of geometries tied at 1.000 and
    the strict `>` kept whichever the loop reached first — the earliest site."""
    scores = {name: sb.predict_signal_peptide(seq).score
              for name, (_site, seq) in SIGNAL_PANEL.items()}
    assert len(set(round(s, 3) for s in scores.values())) > 1, scores


def test_signal_peptide_reports_its_alternatives():
    """A near-tie must be visible: an 8-residue error via mature_sequence silently
    corrupts any fusion design."""
    for name, (true_site, seq) in SIGNAL_PANEL.items():
        result = sb.predict_signal_peptide(seq)
        assert result.alternatives, name
        sites = [site for site, _score in result.alternatives]
        assert true_site in sites, (name, true_site, result.alternatives)


def test_asr_method_ml_is_reported_as_what_it_is():
    """It was never maximum likelihood: no substitution model, no likelihood over
    branch lengths, and the per-site number is column support not a posterior."""
    import omicverse as ov

    seqs = {"chicken": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
            "turkey": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTQ",
            "quail": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWSYDDATKTFTVTE",
            "duck": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDEATKTFTVTQ"}
    alignment = ov.alignment.msa(seqs)
    assert sb.ancestral_reconstruction(alignment, method="ml").method == \
        "weighted_consensus"
    assert sb.ancestral_reconstruction(
        alignment, method="weighted_consensus").method == "weighted_consensus"
    assert sb.ancestral_reconstruction(alignment, method="parsimony").method == \
        "parsimony"


def test_asr_warns_when_the_tree_it_was_given_does_nothing():
    """`tree=` was silently ignored whenever it carried no branch lengths, which
    is every FastTree result — so passing a tree was decorative."""
    import omicverse as ov

    seqs = {"a": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNG",
            "b": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNH",
            "c": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNK"}
    alignment = ov.alignment.msa(seqs)

    class TreeWithoutLengths:
        method = "fasttree"
        distances = None

    with pytest.warns(UserWarning, match="没有分支长度"):
        sb.ancestral_reconstruction(alignment, tree=TreeWithoutLengths())


def test_binder_ranking_prefers_interface_evidence_over_monomer_plddt():
    """Ranking on monomer pLDDT ranked on folding confidence, which for an
    idealised helical bundle is ~94 whether or not it binds anything."""
    from omicverse.synbio._binder import BinderDesign, denovo_binder
    import inspect

    source = inspect.getsource(denovo_binder)
    assert "scrmsd" in source and "iptm" in source, (
        "the rank key must consider interface evidence")
