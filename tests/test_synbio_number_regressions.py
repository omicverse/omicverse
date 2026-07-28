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
