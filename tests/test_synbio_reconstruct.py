"""Layer-A entry point: reconstruct a GEM, gap-fill it, validate it.

Everything here runs on COBRApy's **bundled** ``textbook`` model. ``e_coli_core``
is *not* bundled — asking for it downloads from BiGG on every call — so tests
that want to stay offline must say ``textbook``.

The homology path is driven through ``gene_map=`` rather than a real DIAMOND run:
that keeps the test offline and aligner-free while still exercising the part that
decides the model's content — GPR evaluation and carving.
"""
import pytest

cobra = pytest.importorskip("cobra", reason="needs omicverse[synbio] (cobra)")

import omicverse as ov
from omicverse.synbio._reconstruct import _eval_gpr, _is_structural

sb = ov.synbio


@pytest.fixture(scope="module")
def template():
    return sb.load_gem("textbook")


@pytest.fixture(scope="module")
def carved(template):
    """A draft missing the last 40 template genes — enough to stop growth."""
    genes = sorted(g.id for g in template.genes)
    gene_map = {f"q_{g}": g for g in genes[:-40]}
    return sb.reconstruct_gem(None, template=template, gene_map=gene_map)


# ---------------------------------------------------------------------------
# GPR evaluation — the single decision the whole reconstruction rests on
# ---------------------------------------------------------------------------

def test_gpr_single_gene():
    assert _eval_gpr("a", {"a"}) is True
    assert _eval_gpr("a", {"b"}) is False


def test_gpr_and_needs_every_gene():
    assert _eval_gpr("a and b", {"a", "b"}) is True
    assert _eval_gpr("a and b", {"a"}) is False


def test_gpr_or_needs_one():
    assert _eval_gpr("a or b", {"b"}) is True
    assert _eval_gpr("a or b", set()) is False


def test_gpr_and_binds_tighter_than_or():
    """``a or b and c`` is ``a or (b and c)``, not ``(a or b) and c``."""
    assert _eval_gpr("a or b and c", {"b"}) is False
    assert _eval_gpr("a or b and c", {"a"}) is True
    assert _eval_gpr("a or b and c", {"b", "c"}) is True


def test_gpr_parentheses_override_precedence():
    assert _eval_gpr("(a or b) and c", {"a", "c"}) is True
    assert _eval_gpr("(a or b) and c", {"a"}) is False


def test_gpr_gene_named_like_an_operator():
    """A gene id containing 'and'/'or' must not be parsed as an operator.

    This is why the evaluator tokenises instead of calling eval() on the rule.
    """
    assert _eval_gpr("and1 or b", {"and1"}) is True
    assert _eval_gpr("orf1 and orf2", {"orf1"}) is False
    assert _eval_gpr("lambda", {"lambda"}) is True


def test_gpr_empty_rule_is_satisfied():
    """No gene evidence in the template means homology has nothing to say."""
    assert _eval_gpr("", set()) is True
    assert _eval_gpr("   ", set()) is True


# ---------------------------------------------------------------------------
# reconstruct_gem
# ---------------------------------------------------------------------------

def test_structural_reactions_are_recognised(template):
    biomass = [r for r in template.reactions if "BIOMASS" in r.id.upper()]
    assert biomass and all(_is_structural(r) for r in biomass)
    exchanges = [r for r in template.reactions if r.id.startswith("EX_")]
    assert exchanges and all(_is_structural(r) for r in exchanges)


def test_reconstruct_carves_gene_orphaned_reactions(carved, template):
    assert carved.dropped_reactions, "dropping 40 genes should carve something"
    assert carved.n_reactions < carved.template_reactions
    assert 0.0 < carved.coverage < 1.0
    for rid in carved.dropped_reactions:
        assert rid not in {r.id for r in carved.model.reactions}


def test_reconstruct_never_carves_the_objective(carved):
    """Biomass and exchanges have no GPR — a naive rule would delete them all."""
    ids = {r.id.upper() for r in carved.model.reactions}
    assert any("BIOMASS" in i for i in ids), "objective was carved away"
    assert any(i.startswith("EX_") for i in ids), "all exchanges were carved away"


def test_reconstruct_reports_unmapped_template_genes(carved):
    assert len(carved.unmapped_template_genes) == 40
    mapped = set(carved.mapped_genes.values())
    assert not mapped & set(carved.unmapped_template_genes)


def test_reconstruct_leaves_the_template_untouched(template, carved):
    """The draft is a copy — reconstructing must not mutate the template."""
    assert len(template.reactions) == carved.template_reactions
    assert carved.model is not template


def test_keep_orphan_reactions_carves_nothing(template):
    genes = sorted(g.id for g in template.genes)
    gene_map = {f"q_{g}": g for g in genes[:-40]}
    rep = sb.reconstruct_gem(None, template=template, gene_map=gene_map,
                             keep_orphan_reactions=True)
    assert rep.n_reactions == rep.template_reactions
    assert rep.dropped_reactions, "the report still lists what would have gone"


def test_reconstruct_rejects_unknown_method(template):
    """'gapseq' used to be the example here because it was unsupported. It is a
    real backend now, so an unknown method has to be something genuinely absent
    — otherwise this test silently stops testing rejection."""
    with pytest.raises(ValueError, match="method must be one of"):
        sb.reconstruct_gem(None, template=template, method="raven")


def test_all_named_reconstruction_backends_are_accepted(template):
    """Every backend the docs name must at least reach its own dispatch."""
    import shutil

    assert sb.reconstruct_gem(None, template=template,
                              gene_map={}).method == "homology"
    for method, tool in (("carveme", "carve"), ("gapseq", "gapseq")):
        if shutil.which(tool):
            continue
        with pytest.raises(ImportError, match=tool):
            sb.reconstruct_gem(None, template=template, method=method)


def test_homology_without_a_template_proteome_is_an_error(template):
    with pytest.raises(ValueError, match="template_proteome"):
        sb.reconstruct_gem(None, template=template, method="homology")


# ---------------------------------------------------------------------------
# universal_reactions
# ---------------------------------------------------------------------------

def test_universe_strips_gene_rules(template):
    uni = sb.universal_reactions([template])
    assert uni.reactions
    assert all(not r.gene_reaction_rule for r in uni.reactions), \
        "a universal reaction is a chemical possibility, not encoded evidence"


def test_universe_excludes_boundaries_by_default(template):
    uni = sb.universal_reactions([template])
    assert not [r for r in uni.reactions if r.id.startswith("EX_")]
    with_ex = sb.universal_reactions([template], include_exchanges=True)
    assert [r for r in with_ex.reactions if r.id.startswith("EX_")]


def test_universe_deduplicates_across_templates(template):
    once = sb.universal_reactions([template])
    twice = sb.universal_reactions([template, template])
    assert len(once.reactions) == len(twice.reactions)


# ---------------------------------------------------------------------------
# gapfill_model
# ---------------------------------------------------------------------------

def test_carved_draft_cannot_grow(carved):
    """The premise of gap-filling: carving broke the model."""
    assert not carved.grows, f"expected a dead draft, got growth={carved.growth}"


@pytest.mark.parametrize("method", ["lp", "cobra"])
def test_gapfill_restores_growth(carved, template, method):
    uni = sb.universal_reactions([template])
    rep = sb.gapfill_model(carved.model, universe=uni, method=method,
                           growth_threshold=0.05, max_added=15)
    assert rep.added, f"{method} added nothing"
    assert rep.solved, f"{method} did not restore growth: {rep.growth_after}"
    assert rep.growth_after > rep.growth_before


def test_gapfill_lp_handles_multi_reaction_gaps(carved, template):
    """An LP relaxation must solve gaps no single addition can.

    The naive "add whichever one reaction helps most, repeat" never starts here:
    carving removes consecutive pathway steps, so every single-reaction trial
    still yields zero growth and the search has no gradient to follow.
    """
    uni = sb.universal_reactions([template])
    singles = 0
    for rxn in list(uni.reactions)[:40]:
        if rxn.id in {r.id for r in carved.model.reactions}:
            continue
        probe = carved.model.copy()
        probe.add_reactions([rxn.copy()])
        val = probe.slim_optimize()
        if val == val and val > 1e-6:
            singles += 1
    rep = sb.gapfill_model(carved.model, universe=uni, method="lp",
                           growth_threshold=0.05, max_added=15)
    assert rep.solved
    if singles == 0:
        assert len(rep.added) >= 2, \
            "no single reaction helps, so the fill must be a multi-reaction set"


def test_gapfill_does_not_mutate_the_input(carved, template):
    uni = sb.universal_reactions([template])
    before = len(carved.model.reactions)
    rep = sb.gapfill_model(carved.model, universe=uni, method="lp",
                           growth_threshold=0.05)
    assert len(carved.model.reactions) == before, "input model was modified"
    assert len(rep.model.reactions) > before


def test_gapfill_rejects_unknown_method(carved, template):
    uni = sb.universal_reactions([template])
    with pytest.raises(ValueError, match="method must be one of"):
        sb.gapfill_model(carved.model, universe=uni, method="fastgapfill")


def test_gapfill_respects_max_added(carved, template):
    uni = sb.universal_reactions([template])
    rep = sb.gapfill_model(carved.model, universe=uni, method="lp",
                           growth_threshold=0.05, max_added=2)
    assert len(rep.added) <= 2


# ---------------------------------------------------------------------------
# validate_gem
# ---------------------------------------------------------------------------

def test_validate_reports_growth_and_counts(template):
    qc = sb.validate_gem(template, check_blocked=False)
    assert qc["grows"] and qc["growth"] > 0.1
    assert qc["n_reactions"] == len(template.reactions)
    assert qc["n_genes"] == len(template.genes)


def test_validate_finds_no_mass_imbalance_in_a_curated_model(template):
    qc = sb.validate_gem(template, check_blocked=False, check_energy_leak=False)
    assert qc["unbalanced"] == [], f"textbook should balance: {qc['unbalanced']}"


def test_validate_detects_a_dead_model(carved):
    qc = sb.validate_gem(carved.model, check_blocked=False)
    assert not qc["grows"]


def test_energy_leak_is_zero_in_a_curated_model(template):
    """ATP made from nothing means gap-filling introduced an impossible cycle."""
    qc = sb.validate_gem(template, check_blocked=False)
    leak = qc["energy_leak"]
    assert leak is not None
    assert leak < 1e-6, f"textbook should not generate free ATP, got {leak}"


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

def test_plot_reconstruction_two_panels(carved):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    fig, axes = sb.plot_reconstruction(carved)
    assert len(axes) == 2
    plt.close(fig)


def test_plot_reconstruction_adds_a_gapfill_panel(carved, template):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    uni = sb.universal_reactions([template])
    gf = sb.gapfill_model(carved.model, universe=uni, method="lp",
                          growth_threshold=0.05)
    fig, axes = sb.plot_reconstruction(carved, gapfill=gf)
    assert len(axes) == 3
    plt.close(fig)
