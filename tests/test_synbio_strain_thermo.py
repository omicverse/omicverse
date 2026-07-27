"""OptForce MUST/FORCE sets, and thermodynamics pushed inside FBA.

Two additions that were genuinely missing from layer A. ``strain_design``
already covered OptKnock, RobustKnock and MCS through ``straindesign``; what it
had no path to was OptForce (``straindesign`` ships no OptForce), and ΔG was
computed only *outside* FBA by ``reaction_dg`` / MDF.

The OptForce assertion worth reading is
:func:`test_force_set_reverses_fumarase_for_succinate` — on ``textbook``, the
search picks reversing FUM, which is the textbook route to succinate
over-production in *E. coli*, and guaranteed product goes 0 → 8.7.

Offline on the bundled ``textbook`` model.
"""
import pytest

cobra = pytest.importorskip("cobra", reason="needs omicverse[synbio] (cobra)")

import omicverse as ov

sb = ov.synbio


@pytest.fixture(scope="module")
def model():
    return sb.load_gem("textbook")


@pytest.fixture(scope="module")
def succinate(model):
    return sb.optforce(model, "EX_succ_e", target_fraction=0.8,
                       max_interventions=3, candidate_limit=20)


# ---------------------------------------------------------------------------
# OptForce — MUST sets
# ---------------------------------------------------------------------------

def test_must_sets_are_not_empty_at_a_demanding_target(succinate):
    """Asking for 80% of the theoretical succinate yield must force rewiring."""
    assert succinate.must, "no reaction had to change to reach 80% of max yield"
    assert succinate.must_decrease, "expected reactions that must be turned down"


def test_must_intervals_really_are_disjoint(succinate):
    """The definition of MUST: wild-type and target ranges cannot overlap."""
    for iv in succinate.must_increase:
        assert iv.target[0] > iv.wild_type[1], (
            f"{iv.reaction} is in MUST↑ but its intervals overlap")
    for iv in succinate.must_decrease:
        assert iv.target[1] < iv.wild_type[0], (
            f"{iv.reaction} is in MUST↓ but its intervals overlap")


def test_target_state_keeps_a_growth_requirement(model):
    """Regression: the target-side FVA originally ran at fraction_of_optimum=0.

    Dropping the growth requirement widens every interval to the network's full
    capability, and two maximally wide intervals almost always overlap — the
    MUST sets came back essentially empty (1 reaction). A viable overproducer is
    the only state for which "cannot stay where it is" means anything.
    """
    res = sb.optforce(model, "EX_succ_e", target_fraction=0.8,
                      max_interventions=1, candidate_limit=10)
    assert len(res.must) >= 5, (
        f"only {len(res.must)} MUST reactions — the target state is probably "
        "unconstrained again")


def test_a_modest_target_forces_less_than_a_demanding_one(model):
    """More demanding production must not require *fewer* changes."""
    easy = sb.optforce(model, "EX_succ_e", target_fraction=0.3,
                       max_interventions=1, candidate_limit=10)
    hard = sb.optforce(model, "EX_succ_e", target_fraction=0.8,
                       max_interventions=1, candidate_limit=10)
    assert len(hard.must) >= len(easy.must)


# ---------------------------------------------------------------------------
# OptForce — FORCE set
# ---------------------------------------------------------------------------

def test_force_set_reverses_fumarase_for_succinate(succinate):
    """The recognisable answer: run FUM backwards toward succinate."""
    assert succinate.force, "FORCE search found nothing"
    reactions = {iv.reaction for iv in succinate.force}
    assert "FUM" in reactions, f"expected FUM in the FORCE set, got {reactions}"


def test_force_set_actually_buys_guaranteed_product(succinate):
    """Without this, a FORCE set is just a list of reactions."""
    assert succinate.baseline_product < 1e-6, (
        "wild-type textbook should not be forced to make succinate")
    assert succinate.guaranteed_product > 1.0, (
        f"FORCE set bought only {succinate.guaranteed_product}")


def test_force_set_is_a_subset_of_must(succinate):
    must = {(i.reaction, i.action) for i in succinate.must}
    for iv in succinate.force:
        assert (iv.reaction, iv.action) in must


def test_force_is_flagged_as_a_search_not_a_proof(succinate):
    """The MUST sets are derived; the FORCE set is a greedy search. Saying so is
    part of the contract — straindesign has no OptForce to defer to."""
    assert succinate.force_is_exact is False


def test_max_interventions_is_respected(model):
    res = sb.optforce(model, "EX_succ_e", target_fraction=0.8,
                      max_interventions=2, candidate_limit=10)
    assert len(res.force) <= 2


def test_to_frame_marks_the_force_set(succinate):
    df = succinate.to_frame()
    assert len(df) == len(succinate.must)
    assert df["in_force_set"].sum() == len(succinate.force)


def test_model_is_not_mutated_by_optforce(model, succinate):
    assert model.slim_optimize() == pytest.approx(0.8739, abs=1e-3)


# ---------------------------------------------------------------------------
# OptForce — errors
# ---------------------------------------------------------------------------

def test_unknown_target_names_how_to_find_one(model):
    with pytest.raises(ValueError, match="不在模型里|EX_"):
        sb.optforce(model, "EX_notAMetabolite_e")


def test_unproducible_target_is_reported(model):
    """A target the medium cannot support must fail clearly, not return zeros."""
    with pytest.raises(ValueError, match="无法产生|理论最大"):
        with model as m:
            m.reactions.EX_glc__D_e.lower_bound = 0.0    # no carbon at all
            sb.optforce(m, "EX_succ_e")


# ---------------------------------------------------------------------------
# thermodynamically constrained FBA
# ---------------------------------------------------------------------------

def test_directionality_finds_dg_data_and_constrains_something(model):
    res = sb.thermo_fba(model, method="directionality")
    assert res.dg_range, "no reaction got a ΔG estimate"
    assert res.method == "directionality"
    assert res.blocked_forward or res.blocked_reverse, "nothing was constrained"


def test_directionality_never_increases_the_objective(model):
    """Constraints can only remove flux states."""
    res = sb.thermo_fba(model, method="directionality")
    assert res.objective_value <= res.unconstrained_objective + 1e-6
    assert res.cost >= -1e-9


def test_tmfa_is_at_least_as_strict_as_per_reaction_directionality(model):
    """The MILP chooses directions and concentrations *together*, so it can rule
    out loops the one-reaction-at-a-time test cannot see."""
    direct = sb.thermo_fba(model, method="directionality")
    tmfa = sb.thermo_fba(model, method="tmfa", time_limit=30)
    assert tmfa.objective_value <= direct.objective_value + 1e-6


def test_tmfa_reports_concentrations_and_dg_prime(model):
    res = sb.thermo_fba(model, method="tmfa", time_limit=30)
    assert res.concentrations, "no concentrations returned"
    assert res.dg_prime, "no ΔG' values returned"
    assert all(1e-9 < c < 1.0 for c in res.concentrations.values())


def test_tmfa_forward_flux_implies_negative_dg(model):
    """The second law, as an assertion: a reaction carrying forward flux in the
    TMFA solution must have ΔG' < 0 at the chosen concentrations."""
    res = sb.thermo_fba(model, method="tmfa", time_limit=30)
    violations = [rid for rid, dg in res.dg_prime.items()
                  if res.fluxes.get(rid, 0.0) > 1e-6 and dg > 1e-6]
    assert not violations, f"forward flux with ΔG' > 0: {violations[:5]}"


def test_explicit_dg0_means_absence_is_no_data(model):
    """Passing a mapping must not have missing reactions treated as ΔG = 0."""
    res = sb.thermo_fba(model, dg0={"PGI": 2.5}, method="directionality")
    assert list(res.dg_range) == ["PGI"], (
        "only the reaction we supplied data for may be constrained")


def test_fixed_concentrations_are_honoured(model):
    res = sb.thermo_fba(model, method="tmfa", fixed={"h2o": 1.0}, time_limit=30)
    if "h2o" in res.concentrations:
        assert res.concentrations["h2o"] == pytest.approx(1.0, rel=1e-3)


def test_thermo_fba_rejects_unknown_method(model):
    with pytest.raises(ValueError, match="method must be one of"):
        sb.thermo_fba(model, method="tfa2")


def test_thermo_fba_does_not_mutate_the_model(model):
    before = model.slim_optimize()
    sb.thermo_fba(model, method="directionality")
    sb.thermo_fba(model, method="tmfa", time_limit=15)
    assert model.slim_optimize() == pytest.approx(before)


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

def test_plot_optforce(succinate):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    fig, axes = sb.plot_optforce(succinate)
    assert len(axes) == 2
    plt.close(fig)


def test_plot_thermo_fba(model):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    res = sb.thermo_fba(model, method="directionality")
    fig, axes = sb.plot_thermo_fba(res)
    assert len(axes) == 2
    plt.close(fig)
