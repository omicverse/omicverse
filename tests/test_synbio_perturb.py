"""Knockout response: FBA re-optimises, MOMA/ROOM stay near the wild type.

The finding these tests pin down is not a subtlety. On COBRApy's ``textbook``
model a PFK deletion reads as:

    fba          growth 0.7040   (81% of wild type — a viable strain)
    linear_moma  growth 0.0000   (dead)
    room         growth 0.2397

FBA does not merely overestimate the knockout, it **inverts the call**. A cell
that just lost pfk still carries the wild-type proteome; it cannot instantly
adopt the globally optimal flux state of the deleted network, which is exactly
what FBA hands it. Any knockout screen run on FBA alone will therefore keep
targets that are actually lethal.

Offline on the bundled ``textbook`` model.
"""
import pytest

cobra = pytest.importorskip("cobra", reason="needs omicverse[synbio] (cobra)")

import omicverse as ov

sb = ov.synbio

WILD_TYPE_GROWTH = 0.8739


@pytest.fixture(scope="module")
def model():
    return sb.load_gem("textbook")


@pytest.fixture(scope="module")
def reference(model):
    return cobra.flux_analysis.pfba(model)


# ---------------------------------------------------------------------------
# the finding
# ---------------------------------------------------------------------------

def test_fba_calls_a_lethal_pfk_knockout_viable(model):
    """The headline: FBA and MOMA disagree about whether the strain lives."""
    df = sb.compare_knockout_methods(model, ["PFK"])
    fba = df.loc["fba", "growth"]
    moma = df.loc["linear_moma", "growth"]

    assert fba > 0.5, f"FBA should call PFK-KO viable, got {fba}"
    assert moma < 1e-6, f"linear MOMA should call PFK-KO dead, got {moma}"


@pytest.mark.parametrize("ko", ["PFK", "TPI"])
def test_fba_never_predicts_less_growth_than_moma(model, ko):
    """FBA's optimum is an upper bound: it optimises over the whole deleted
    network, while MOMA is additionally tied to the wild-type flux vector."""
    df = sb.compare_knockout_methods(model, [ko])
    assert df.loc["fba", "growth"] >= df.loc["linear_moma", "growth"] - 1e-6


def test_methods_agree_on_a_genuinely_essential_reaction(model):
    """ENO is essential — no formulation can rescue it, so all three agree.

    Without this, the tests above could pass on a module that always reports
    zero for MOMA.
    """
    df = sb.compare_knockout_methods(model, ["ENO"])
    for method in ("fba", "linear_moma", "room"):
        assert df.loc[method, "growth"] < 1e-6, f"{method} kept ENO-KO alive"


def test_a_mild_knockout_stays_mild_under_every_method(model):
    """PGI is dispensable — the point is that MOMA is not simply pessimistic."""
    df = sb.compare_knockout_methods(model, ["PGI"])
    for method in ("fba", "linear_moma", "room"):
        assert df.loc[method, "growth"] > 0.5, (
            f"{method} killed a PGI knockout the model tolerates")


# ---------------------------------------------------------------------------
# unit semantics — the trap this module exists to close
# ---------------------------------------------------------------------------

def test_growth_is_biomass_flux_not_the_method_objective(model, reference):
    """linear MOMA's own objective_value is a flux deviation (~71 on this
    knockout). Reading it as growth silently mixes units."""
    res = sb.knockout_flux(model, ["PGI"], method="linear_moma",
                           reference=reference)
    assert 0.0 <= res.growth <= WILD_TYPE_GROWTH + 1e-6
    assert res.method_objective is not None
    assert res.method_objective > 10.0, (
        "expected a large deviation value, which is exactly why it must not be "
        "reported as a growth rate")
    assert "deviation" in res.method_objective_meaning


def test_room_objective_is_a_count_not_a_growth_rate(model, reference):
    res = sb.knockout_flux(model, ["PGI"], method="room", reference=reference)
    assert 0.0 <= res.growth <= WILD_TYPE_GROWTH + 1e-6
    assert "changed" in res.method_objective_meaning


def test_growth_ratio_is_relative_to_the_reference(model, reference):
    res = sb.knockout_flux(model, ["PGI"], method="linear_moma",
                           reference=reference)
    assert res.reference_growth == pytest.approx(WILD_TYPE_GROWTH, abs=1e-3)
    assert res.growth_ratio == pytest.approx(res.growth / res.reference_growth)
    assert 0.0 <= res.growth_ratio <= 1.001


def test_fba_reports_no_method_objective(model, reference):
    """Plain FBA has no deviation to report; the field must be None rather than
    quietly reusing the growth rate."""
    res = sb.knockout_flux(model, ["PGI"], method="fba", reference=reference)
    assert res.method_objective is None
    assert res.method_objective_meaning == ""


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

def test_gene_knockout_goes_through_the_gpr(model, reference):
    """b4025 is PGI's only gene, so deleting it must disable PGI."""
    res = sb.knockout_flux(model, genes=["b4025"], method="fba",
                           reference=reference)
    assert abs(res.fluxes["PGI"]) < 1e-6


def test_model_is_not_mutated(model):
    before = model.slim_optimize()
    sb.knockout_flux(model, ["PFK"], method="linear_moma")
    assert model.slim_optimize() == pytest.approx(before)
    assert model.reactions.PFK.upper_bound > 0, "knockout leaked out of the context"


def test_changed_fluxes_are_relative_to_the_reference(model, reference):
    res = sb.knockout_flux(model, ["PGI"], method="linear_moma",
                           reference=reference)
    changed = res.changed_fluxes(threshold=1e-3)
    assert res.n_changed == len(changed)
    assert "PGI" in changed, "the knocked-out reaction itself must have moved"


def test_unknown_method_rejected(model):
    with pytest.raises(ValueError, match="method must be one of"):
        sb.knockout_flux(model, ["PGI"], method="lmoma")


def test_no_target_is_an_error(model):
    with pytest.raises(ValueError, match="knockouts|genes"):
        sb.knockout_flux(model)


def test_quadratic_moma_explains_the_missing_qp_solver(model, reference):
    """GLPK cannot do QP. The error must name the LP alternative rather than
    surfacing COBRApy's bare SolverNotFound."""
    from cobra.util.solver import solvers
    if any(s in solvers for s in ("cplex", "gurobi", "osqp")):
        pytest.skip("a QP-capable solver is installed, so this path is not taken")
    with pytest.raises(RuntimeError, match="linear_moma"):
        sb.knockout_flux(model, ["PGI"], method="moma", reference=reference)


def test_compare_reports_errors_instead_of_raising(model):
    """A method that cannot run must not take the whole comparison down."""
    df = sb.compare_knockout_methods(model, ["PGI"],
                                     methods=("fba", "moma", "linear_moma"))
    assert "fba" in df.index and "linear_moma" in df.index
    from cobra.util.solver import solvers
    if not any(s in solvers for s in ("cplex", "gurobi", "osqp")):
        assert df.loc["moma", "status"].startswith("error:")


def test_compare_shares_one_reference_across_methods(model):
    """Each method must be scored against the same wild type, or the growth
    ratios are not comparable."""
    df = sb.compare_knockout_methods(model, ["PGI"])
    assert df["growth_ratio"].notna().all()
    assert (df["growth_ratio"] <= 1.001).all()


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

def test_plot_from_dataframe(model):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    df = sb.compare_knockout_methods(model, ["PGI"])
    fig, axes = sb.plot_knockout_response(df)
    assert len(axes) == 2
    plt.close(fig)


def test_plot_from_result(model, reference):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    res = sb.knockout_flux(model, ["PGI"], method="linear_moma",
                           reference=reference)
    fig, axes = sb.plot_knockout_response(res)
    assert len(axes) == 2
    plt.close(fig)
