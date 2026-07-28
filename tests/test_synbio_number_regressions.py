"""Regressions for defects that produced *plausible wrong numbers*.

Every bug pinned here shipped for months without a test failing, because none of
them raised: they returned a float in the expected range with the expected type.
The tests are therefore written against quantities with an external ground truth
— a published ΔG, a truth table, a monotonicity that physics requires — rather
than against whatever the code currently happens to emit.
"""
from __future__ import annotations

import warnings

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
    from omicverse.synbio._expression import cai

    design = sb.mrna_design(PROTEIN, method="baseline", host="human")
    dna = design.mrna.replace("U", "T")
    assert design.cai == pytest.approx(cai(dna, host="h_sapiens"), abs=1e-6)
    assert design.cai > cai(dna, host="e_coli") + 0.1, (
        "a human-optimised CDS must score higher against the human table")


def test_mrna_design_cai_is_never_a_silent_zero():
    """``except Exception: cai_val = 0.0`` reads as 'terrible codon usage'."""
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
    from omicverse.synbio._refseq import GAPDH_CDS

    result = sb.codon_harmonize(GAPDH_CDS, source_host="h_sapiens",
                                target_host="e_coli")
    assert result.rank_correlation == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < result.frequency_correlation < 1.0, (
        "the frequency correlation is the quantity that can actually fail")
    assert result.max_frequency_shift > 0.0


def test_harmonisation_flags_rare_codons_it_introduces():
    """Rank-mapping happily places CGA/AGG — the codons Rosetta strains rescue."""
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
