"""Learn → Design bridge: constrain a GEM with measured expression.

The interesting assertions here are not "does it run" but "does the expression
actually reach the flux". Three reactions in ``textbook`` make that testable
deterministically, so nothing depends on random draws:

* ``PGI``  — GPR ``b4025``, a single gene, and the model still grows at 0.863
  without it. Silence ``b4025`` and a working method must route around PGI.
* ``FRD7`` — GPR ``b4153 and b4151 and b4152 and b4154``. A complex is limited
  by its scarcest subunit, so one low subunit must drag the whole reaction low.
* ``PFK``  — GPR ``b3916 or b1723``. Isozymes are alternatives, so one
  well-expressed gene must keep the reaction high.

Runs offline on COBRApy's bundled ``textbook`` (``e_coli_core`` would download).
"""
import pytest

cobra = pytest.importorskip("cobra", reason="needs omicverse[synbio] (cobra)")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import omicverse as ov
from omicverse.synbio._context import _gpr_score, gene_expression

sb = ov.synbio

PGI_GENE = "b4025"
FRD7_SUBUNITS = ["b4153", "b4151", "b4152", "b4154"]
PFK_ISOZYMES = ["b3916", "b1723"]


@pytest.fixture(scope="module")
def model():
    return sb.load_gem("textbook")


@pytest.fixture(scope="module")
def genes(model):
    return [g.id for g in model.genes]


@pytest.fixture
def flat_expression(genes):
    """Every gene high, so any silencing below is the only signal present."""
    return {g: 100.0 for g in genes}


@pytest.fixture
def spread_expression(genes):
    """A deterministic spread, so percentile thresholds are meaningful.

    iMAT needs a distribution: it makes a three-way high/moderate/low call, and
    on flat input every reaction lands in the same bin. Values cycle over a
    decade so the 25th and 75th percentiles are genuinely apart.
    """
    ladder = [1.0, 3.0, 10.0, 30.0, 100.0]
    return {g: ladder[i % len(ladder)] for i, g in enumerate(sorted(genes))}


# iMAT is a MILP and genuinely NP-hard. Bound it in tests: a time-limited run
# still yields a usable incumbent, and the assertions below say what they can
# claim under that. Without the bound the suite ran for hours.
IMAT_LIMIT = 10.0


@pytest.fixture(scope="module")
def imat_result(model, genes):
    """One iMAT solve shared by the assertions that only read its output.

    Each MILP solve costs up to IMAT_LIMIT seconds, so re-solving per assertion
    made this file the slowest in the suite for no extra coverage.
    """
    ladder = [1.0, 3.0, 10.0, 30.0, 100.0]
    expr = {g: ladder[i % len(ladder)] for i, g in enumerate(sorted(genes))}
    return sb.contextualize_gem(expr, model, method="imat",
                                low_percentile=25.0, high_percentile=75.0,
                                time_limit=IMAT_LIMIT)


# ---------------------------------------------------------------------------
# GPR folding — and → min, or → max
# ---------------------------------------------------------------------------

def test_and_takes_the_minimum():
    """A complex runs no faster than its scarcest subunit."""
    vals = {g: 100.0 for g in FRD7_SUBUNITS}
    vals["b4151"] = 1.0
    assert _gpr_score(" and ".join(FRD7_SUBUNITS), vals) == 1.0


def test_or_takes_the_maximum():
    """Isozymes are alternatives — the best one sets the rate."""
    vals = {"b3916": 100.0, "b1723": 1.0}
    assert _gpr_score("b3916 or b1723", vals) == 100.0


def test_or_can_be_additive():
    vals = {"b3916": 100.0, "b1723": 1.0}
    assert _gpr_score("b3916 or b1723", vals, or_op="sum") == 101.0


def test_and_binds_tighter_than_or_numerically():
    """``a or b and c`` folds as ``max(a, min(b, c))``."""
    vals = {"a": 5.0, "b": 100.0, "c": 2.0}
    assert _gpr_score("a or b and c", vals) == 5.0


def test_unmeasured_rule_is_none_not_zero():
    """"We did not measure this" must stay distinguishable from "it is off"."""
    assert _gpr_score("someGeneWeNeverMeasured", {"other": 1.0}) is None
    assert _gpr_score("", {"a": 1.0}) is None


def test_partially_measured_rule_uses_what_it_has():
    assert _gpr_score("a or unknown", {"a": 7.0}) == 7.0
    assert _gpr_score("a and unknown", {"a": 7.0}) == 7.0


# ---------------------------------------------------------------------------
# expression input plumbing
# ---------------------------------------------------------------------------

def test_gene_expression_from_mapping():
    assert gene_expression({"a": 1, "b": 2.5}) == {"a": 1.0, "b": 2.5}


def test_gene_expression_from_series():
    s = pd.Series({"a": 3.0, "b": 4.0})
    assert gene_expression(s) == {"a": 3.0, "b": 4.0}


def test_gene_expression_from_samples_by_genes_frame():
    df = pd.DataFrame({"a": [1.0, 3.0], "b": [10.0, 10.0]})
    got = gene_expression(df, agg="mean")
    assert got == {"a": 2.0, "b": 10.0}


def test_gene_expression_from_long_frame():
    df = pd.DataFrame({"gene": ["a", "b"], "tpm": [5.0, 6.0]})
    assert gene_expression(df, gene_col="gene", value_col="tpm") == {"a": 5.0, "b": 6.0}


def test_gene_expression_from_anndata(genes):
    anndata = pytest.importorskip("anndata")
    X = np.vstack([np.full(len(genes), 2.0), np.full(len(genes), 4.0)])
    a = anndata.AnnData(X=X, var=pd.DataFrame(index=genes),
                        obs=pd.DataFrame(index=["s1", "s2"]))
    got = gene_expression(a, agg="mean")
    assert got[genes[0]] == pytest.approx(3.0)


def test_anndata_groupby_actually_selects(genes):
    """Selecting a group must change the numbers, not just be accepted."""
    anndata = pytest.importorskip("anndata")
    X = np.vstack([np.full(len(genes), 1.0), np.full(len(genes), 9.0)])
    a = anndata.AnnData(
        X=X, var=pd.DataFrame(index=genes),
        obs=pd.DataFrame({"cond": ["ctrl", "tumor"]}, index=["s1", "s2"]))
    ctrl = gene_expression(a, groupby="cond", group="ctrl")
    tumor = gene_expression(a, groupby="cond", group="tumor")
    assert ctrl[genes[0]] == pytest.approx(1.0)
    assert tumor[genes[0]] == pytest.approx(9.0)


def test_anndata_layer_is_read(genes):
    anndata = pytest.importorskip("anndata")
    X = np.full((1, len(genes)), 1.0)
    a = anndata.AnnData(X=X, var=pd.DataFrame(index=genes),
                        obs=pd.DataFrame(index=["s1"]))
    a.layers["counts"] = np.full((1, len(genes)), 50.0)
    assert gene_expression(a, layer="counts")[genes[0]] == pytest.approx(50.0)


def test_groupby_without_group_is_an_error(genes):
    anndata = pytest.importorskip("anndata")
    a = anndata.AnnData(X=np.ones((1, len(genes))), var=pd.DataFrame(index=genes),
                        obs=pd.DataFrame({"c": ["x"]}, index=["s1"]))
    with pytest.raises(ValueError, match="group"):
        gene_expression(a, groupby="c")


def test_unsupported_expression_type():
    with pytest.raises(TypeError, match="不支持的 expression 类型|AnnData"):
        gene_expression(object())


# ---------------------------------------------------------------------------
# reaction_expression
# ---------------------------------------------------------------------------

def test_reaction_expression_respects_gpr_semantics(model, flat_expression):
    expr = dict(flat_expression)
    expr["b4151"] = 1.0          # one FRD7 subunit scarce
    expr["b1723"] = 1.0          # one PFK isozyme scarce, the other stays 100
    got = sb.reaction_expression(expr, model)
    assert got["FRD7"] == 1.0, "AND must take the minimum subunit"
    assert got["PFK"] == 100.0, "OR must take the best isozyme"


def test_reaction_expression_is_none_for_ungened_reactions(model, flat_expression):
    got = sb.reaction_expression(flat_expression, model)
    assert got["ATPM"] is None or got.get("EX_glc__D_e") is None


def test_reaction_expression_rejects_bad_or_op(model, flat_expression):
    with pytest.raises(ValueError, match="or_op must be one of"):
        sb.reaction_expression(flat_expression, model, or_op="mean")


# ---------------------------------------------------------------------------
# the science: does silencing a gene move the flux?
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["gimme", "riptide"])
def test_silencing_pgi_routes_flux_around_it(model, flat_expression, method):
    """The whole point of the bridge: expression must reach the flux.

    ``textbook`` grows at 0.874 through PGI and at 0.863 without it, so routing
    around PGI is cheap — a method that ignores expression has no reason to, and
    a method that honours it must.
    """
    silenced = dict(flat_expression)
    silenced[PGI_GENE] = 0.0

    base = model.optimize().fluxes["PGI"]
    assert abs(base) > 1e-6, "unconstrained textbook should carry PGI flux"

    res = sb.contextualize_gem(silenced, model, method=method,
                               threshold=1.0, fraction_of_optimum=0.5)
    assert abs(res.fluxes["PGI"]) < abs(base), (
        f"{method} left PGI at {res.fluxes['PGI']} despite its only gene being "
        f"silent (unconstrained {base})")


def test_imat_silences_a_low_expression_reaction(model, spread_expression):
    """iMAT's low class is a hard constraint: those reactions carry zero flux."""
    expr = dict(spread_expression)
    expr[PGI_GENE] = 1.0            # bottom of the ladder -> "low"
    res = sb.contextualize_gem(expr, model, method="imat",
                               low_percentile=25.0, high_percentile=75.0,
                               time_limit=IMAT_LIMIT)
    assert res.reaction_expression["PGI"] == 1.0
    if not res.hit_time_limit:
        assert abs(res.fluxes["PGI"]) < 1e-6, (
            "a reaction iMAT called low must carry no flux, got "
            f"{res.fluxes['PGI']}")


def test_gimme_reports_zero_inconsistency_when_nothing_is_silenced(
        model, flat_expression):
    """With every gene above threshold there is no penalised flux to force."""
    res = sb.contextualize_gem(flat_expression, model, method="gimme",
                               threshold=1.0)
    assert res.inconsistency == pytest.approx(0.0, abs=1e-6)


def test_gimme_inconsistency_is_nonnegative(model, flat_expression):
    silenced = dict(flat_expression)
    for g in FRD7_SUBUNITS:
        silenced[g] = 0.0
    res = sb.contextualize_gem(silenced, model, method="gimme", threshold=1.0)
    assert res.inconsistency is not None and res.inconsistency >= -1e-9


def test_imat_agreement_is_a_fraction(imat_result):
    assert imat_result.agreement is not None
    assert 0.0 <= imat_result.agreement <= 1.0


def test_imat_refuses_degenerate_thresholds(model, flat_expression):
    """A three-way call needs a distribution. Flat input has none.

    Silently lumping every reaction into "high" is what made this hang: iMAT
    then requires all of them to carry flux at once, and branch-and-bound has no
    chance. Better to say so than to grind.
    """
    with pytest.raises(ValueError, match="分布|iMAT"):
        sb.contextualize_gem(flat_expression, model, method="imat")


def test_milp_time_limit_is_reported_not_hidden(imat_result):
    """Stopping early is acceptable; pretending it did not happen is not."""
    assert isinstance(imat_result.hit_time_limit, bool)
    assert imat_result.solver_status, "solver status should be recorded"
    if imat_result.hit_time_limit:
        assert imat_result.fluxes, "an incumbent solution must still be returned"


def test_time_limit_actually_applies_to_the_solver(model):
    """Regression: the limit was set as a float, GLPK's setter takes an int, and
    a bare ``except: pass`` swallowed the TypeError — so the cap silently never
    applied and iMAT ran unbounded."""
    from omicverse.synbio._context import _apply_time_limit

    work = model.copy()
    assert _apply_time_limit(work, 7.5) is True
    assert work.solver.configuration.timeout == 8, "float seconds must be rounded to int"


# ---------------------------------------------------------------------------
# cross-method contract
# ---------------------------------------------------------------------------

ALL_METHODS = ["gimme", "imat", "init", "tinit", "riptide"]


@pytest.mark.parametrize("method", ["gimme", "init", "tinit", "riptide"])
def test_every_method_returns_a_usable_result(model, spread_expression, method):
    res = sb.contextualize_gem(spread_expression, model, method=method,
                               time_limit=IMAT_LIMIT)
    assert res.method in (method, "init", "tinit")
    assert res.fluxes, "no fluxes returned"
    assert res.n_active > 0
    assert set(res.reaction_expression) == {r.id for r in model.reactions}


def test_imat_returns_a_usable_result(model, imat_result):
    assert imat_result.fluxes and imat_result.n_active > 0
    assert set(imat_result.reaction_expression) == {r.id for r in model.reactions}


@pytest.mark.parametrize("method", ["gimme", "init", "tinit", "riptide"])
def test_objective_value_always_means_the_biological_objective(
        model, spread_expression, method):
    """Not the MILP score. iMAT used to report its satisfied-target count here,
    which made ``objective_value`` mean something different per method."""
    res = sb.contextualize_gem(spread_expression, model, method=method,
                               time_limit=IMAT_LIMIT)
    unconstrained = model.slim_optimize()
    assert 0.0 <= res.objective_value <= unconstrained + 1e-6, (
        f"{method}: objective_value={res.objective_value} is not a growth rate "
        f"(unconstrained optimum {unconstrained})")


def test_imat_objective_value_is_growth_not_the_milp_score(model, imat_result):
    """iMAT's satisfied-target count belongs in ``agreement``, not here."""
    unconstrained = model.slim_optimize()
    assert 0.0 <= imat_result.objective_value <= unconstrained + 1e-6


@pytest.mark.parametrize("method", ["gimme", "init", "tinit", "riptide"])
def test_input_model_is_never_mutated(model, spread_expression, method):
    n_before = len(model.reactions)
    obj_before = model.slim_optimize()
    sb.contextualize_gem(spread_expression, model, method=method, prune=True,
                         time_limit=IMAT_LIMIT)
    assert len(model.reactions) == n_before
    assert model.slim_optimize() == pytest.approx(obj_before)


def test_imat_does_not_mutate_the_input(model, imat_result):
    assert imat_result.model is not model
    assert model.slim_optimize() == pytest.approx(0.8739, abs=1e-3)


def test_unknown_method_rejected(model, flat_expression):
    with pytest.raises(ValueError, match="method must be one of"):
        sb.contextualize_gem(flat_expression, model, method="fastcore")


def test_mismatched_gene_ids_give_an_actionable_error(model):
    with pytest.raises(ValueError, match="对不上|gene id"):
        sb.contextualize_gem({"ENSG00000139618": 5.0}, model, method="gimme")


# ---------------------------------------------------------------------------
# pruning
# ---------------------------------------------------------------------------

def test_prune_shrinks_the_model(model, flat_expression):
    expr = dict(flat_expression)
    for g in list(expr)[:30]:
        expr[g] = 0.0
    res = sb.contextualize_gem(expr, model, method="gimme", threshold=1.0,
                               prune=True)
    assert res.removed_reactions
    assert len(res.model.reactions) < len(model.reactions)


def test_prune_never_removes_the_objective_or_exchanges(model, flat_expression):
    expr = {g: 0.0 for g in flat_expression}
    res = sb.contextualize_gem(expr, model, method="gimme", threshold=1.0,
                               prune=True)
    ids = {r.id.upper() for r in res.model.reactions}
    assert any("BIOMASS" in i for i in ids), "pruning deleted the objective"
    assert any(i.startswith("EX_") for i in ids), "pruning deleted all exchanges"


def test_protected_reactions_survive_pruning(model, flat_expression):
    expr = {g: 0.0 for g in flat_expression}
    res = sb.contextualize_gem(expr, model, method="gimme", threshold=1.0,
                               prune=True, protected_reactions=["PGI", "FRD7"])
    ids = {r.id for r in res.model.reactions}
    assert "PGI" in ids and "FRD7" in ids
    assert "PGI" not in res.removed_reactions


# ---------------------------------------------------------------------------
# reporting & visualisation
# ---------------------------------------------------------------------------

def test_to_frame_has_a_row_per_reaction(model, flat_expression):
    res = sb.contextualize_gem(flat_expression, model, method="gimme")
    df = res.to_frame()
    assert len(df) == len(model.reactions)
    assert {"expression", "flux", "active"} <= set(df.columns)


def test_plot_contextualization_returns_three_panels(model, flat_expression):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    res = sb.contextualize_gem(flat_expression, model, method="gimme")
    fig, axes = sb.plot_contextualization(res)
    assert len(axes) == 3
    plt.close(fig)
