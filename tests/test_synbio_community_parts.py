"""Resource allocation, communities, enzyme matching and vector selection.

Three of these were on the original gap list as lower priority ("加分项",
"可以不急") and one — the reaction→enzyme handoff — was the broken joint between
layers A and B.

Offline throughout: the community and RBA tests use COBRApy's bundled
``textbook``, enzyme matching uses a temporary FASTA rather than the network, and
backbone selection needs no network at all by design.
"""
import pytest

cobra = pytest.importorskip("cobra", reason="needs omicverse[synbio] (cobra)")

import omicverse as ov

sb = ov.synbio


@pytest.fixture(scope="module")
def model():
    return sb.load_gem("textbook")


@pytest.fixture(scope="module")
def community(model):
    return sb.community_model({"a": model, "b": model.copy()})


# ---------------------------------------------------------------------------
# RBA
# ---------------------------------------------------------------------------

def test_a_loose_budget_does_not_bind(model):
    res = sb.rba(model, total_protein=0.9)
    assert res.growth == pytest.approx(res.unconstrained_growth, rel=1e-3)
    assert res.cost < 1e-3


def test_tightening_the_budget_costs_growth(model):
    loose = sb.rba(model, total_protein=0.9).growth
    tight = sb.rba(model, total_protein=0.05).growth
    assert tight < loose, f"tight {tight} should be below loose {loose}"
    assert tight > 0


def test_growth_is_monotone_in_the_budget(model):
    growths = [sb.rba(model, total_protein=b).growth
               for b in (0.02, 0.05, 0.1, 0.3)]
    assert growths == sorted(growths), growths


def test_ribosome_competes_for_the_same_pool(model):
    """The distinguishing feature over ec_model: translation is charged too."""
    res = sb.rba(model, total_protein=0.05)
    assert res.ribosome_fraction > 0.0
    assert res.ribosome_fraction + res.enzyme_fraction == pytest.approx(1.0, abs=1e-6)


def test_a_slower_ribosome_costs_more_growth(model):
    fast = sb.rba(model, total_protein=0.05, ribosome_efficiency=20.0).growth
    slow = sb.rba(model, total_protein=0.05, ribosome_efficiency=2.0).growth
    assert slow < fast


def test_rba_does_not_mutate_the_model(model):
    """Regression: the objective was built from the *original* model's flux
    expression and then set on a copy, which tried to import the original's
    solver variables and collided on their names."""
    before = model.slim_optimize()
    sb.rba(model, total_protein=0.2)
    assert model.slim_optimize() == pytest.approx(before)


def test_rba_rejects_a_nonpositive_budget(model):
    with pytest.raises(ValueError, match="total_protein"):
        sb.rba(model, total_protein=0.0)


# ---------------------------------------------------------------------------
# community
# ---------------------------------------------------------------------------

def test_community_merges_both_members(community, model):
    assert len(community.reactions) > len(model.reactions)
    assert community.notes["community_members"] == ["a", "b"]


def test_member_reactions_are_namespaced(community):
    for prefix in ("a__", "b__"):
        assert any(r.id.startswith(prefix) for r in community.reactions)


def test_the_medium_is_shared_not_duplicated(community, model):
    """Regression: leaving the shared exchanges at ±1000 gave each member its
    own unlimited supply — two identical E. coli grew at exactly 2x the
    monoculture rate and there was nothing to compete over."""
    mono = model.slim_optimize()
    com = community.slim_optimize()
    assert com < 1.5 * mono, (
        f"community {com:.4f} against monoculture {mono:.4f} — the members are "
        "not sharing a medium")


def test_community_needs_two_members(model):
    with pytest.raises(ValueError, match="至少需要 2 个"):
        sb.community_model({"only": model})


def test_member_names_may_not_contain_underscores(model):
    with pytest.raises(ValueError, match="下划线"):
        sb.community_model({"a_1": model, "b": model.copy()})


def test_steadycom_finds_a_common_growth_rate(community):
    res = sb.steadycom(community)
    assert res.growth > 0
    assert set(res.abundances) == {"a", "b"}
    assert sum(res.abundances.values()) == pytest.approx(1.0, abs=1e-4)
    assert all(g == pytest.approx(res.growth) for g in res.member_growth.values())


def test_steadycom_reports_a_non_unique_composition(community):
    """Two interchangeable members have no determined ratio: with the abundances
    summing to one, every split is feasible and the LP returns an arbitrary
    vertex. Presenting 0.90/0.10 as a prediction would be wrong."""
    res = sb.steadycom(community)
    assert res.abundance_is_unique is False
    assert res.notes and "不唯一" in res.notes[0]


def test_steadycom_uses_the_recorded_membership(community):
    """Regression: members were recovered from reaction-id prefixes, and the
    shared exchange EX_glc__D_e_u made 'EX_glc' look like a member."""
    res = sb.steadycom(community)
    assert "EX_glc" not in res.abundances


def test_micom_backend_is_explained(model):
    pytest.importorskip  # noqa: B018
    try:
        import micom  # noqa: F401
        pytest.skip("MICOM is installed, so this path is not taken")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="MICOM"):
        sb.micom_grow({"a": model, "b": model.copy()})


# ---------------------------------------------------------------------------
# enzyme matching — the A→B handoff
# ---------------------------------------------------------------------------

@pytest.fixture
def enzyme_db(tmp_path):
    p = tmp_path / "enzymes.faa"
    p.write_text(
        ">ADH1 1.1.1.1 OS=Saccharomyces cerevisiae\n"
        "MSIPETQKGVIFYESHGKLEYKDIPVPKPKANELLINVKYSGVCHTDLHAWHGDWPLPVK\n"
        ">ADH2 1.1.1.2 OS=Escherichia coli\n"
        "MKAAVVTKDHTVAIREVPTPQVGPHDVLIRIAAAGLCHSDLSVINGSRPRPLPMALGHEA\n"
        ">PFKA 2.7.1.11 OS=Escherichia coli\n"
        "MIKKIGVLTSGGDAPGMNAAIRGVVRSALTEGLEVMGIYDGYLGLYEDRMVQLDRYSVSD\n",
        encoding="utf-8")
    return str(p)


def test_exact_ec_match_ranks_first(enzyme_db):
    m = sb.match_enzymes("1.1.1.1", database=enzyme_db)
    assert m.ec_number == "1.1.1.1"
    assert m.best is not None and m.best.identifier == "ADH1"
    assert m.best.score == pytest.approx(1.0)
    assert m.best.has_sequence


def test_near_miss_ec_is_offered_not_discarded(enzyme_db):
    """1.1.1.1 and 1.1.1.2 are the same chemistry with different specificity —
    for a designed pathway the near miss is often the better starting point
    than nothing."""
    m = sb.match_enzymes("1.1.1.1", database=enzyme_db)
    ids = [c.identifier for c in m.candidates]
    assert "ADH2" in ids
    assert m.candidates[ids.index("ADH2")].score < 1.0


def test_min_ec_levels_filters(enzyme_db):
    m = sb.match_enzymes("1.1.1.1", database=enzyme_db, min_ec_levels=4)
    assert [c.identifier for c in m.candidates] == ["ADH1"]


def test_organism_filter(enzyme_db):
    m = sb.match_enzymes("1.1.1.1", database=enzyme_db, min_ec_levels=3,
                         organism="Escherichia")
    assert all("Escherichia" in c.organism for c in m.candidates)


def test_model_genes_are_offered_when_the_reaction_exists(model):
    m = sb.match_enzymes("PFK", model=model)
    assert m.candidates, "PFK has genes in textbook"
    assert all(c.source == "model" for c in m.candidates)


def test_source_is_always_recorded(enzyme_db):
    m = sb.match_enzymes("1.1.1.1", database=enzyme_db)
    assert all(c.source for c in m.candidates)
    assert m.sources_tried


def test_no_source_is_an_actionable_error():
    with pytest.raises(ValueError, match="没有任何可用的酶来源"):
        sb.match_enzymes("1.1.1.1")


def test_missing_database_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        sb.match_enzymes("1.1.1.1", database=str(tmp_path / "nope.faa"))


def test_pathway_matching_reports_the_gaps(enzyme_db):
    """The steps with no candidate are the engineering risk."""
    res = sb.match_pathway_enzymes(["1.1.1.1", "2.7.1.11", "9.9.9.9"],
                                   database=enzyme_db)
    assert res["gaps"] == ["9.9.9.9"]
    assert res["coverage"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# backbone selection
# ---------------------------------------------------------------------------

def test_expression_vector_beats_a_cloning_vector():
    picks = sb.select_backbone(host="e_coli", insert_kb=2.0, expression=True)
    best = picks[0]
    assert not best.disqualified
    assert best.backbone.promoter, f"{best.backbone.name} has no promoter"


def test_wrong_host_is_disqualified_not_downranked():
    """A backbone that cannot replicate in the host is not a weak option."""
    picks = sb.select_backbone(host="e_coli")
    yeast = [c for c in picks if c.backbone.name == "pYES2"][0]
    assert yeast.disqualified
    assert any("宿主" in r for r in yeast.reasons)


def test_oversized_insert_is_disqualified():
    picks = sb.select_backbone(host="e_coli", insert_kb=9.0)
    puc = [c for c in picks if c.backbone.name == "pUC19"][0]
    assert puc.disqualified
    assert any("容量" in r for r in puc.reasons)


def test_incompatible_origins_are_disqualified():
    """Two plasmids sharing an origin cannot be stably co-maintained — the usual
    reason a two-plasmid system quietly loses one."""
    picks = sb.select_backbone(host="e_coli", incompatible_with="pBR322")
    pet = [c for c in picks if c.backbone.name == "pET28a"][0]
    assert pet.disqualified
    assert any("不相容" in r for r in pet.reasons)
    survivors = [c.backbone.name for c in picks if not c.disqualified]
    assert "pACYC184" in survivors, "p15A is compatible with pBR322"


def test_marker_exclusion():
    picks = sb.select_backbone(host="e_coli", avoid_markers=["ampicillin"])
    amp = [c for c in picks if c.backbone.marker == "ampicillin"]
    assert amp and all(c.disqualified for c in amp)


def test_copy_number_preference_is_a_ranking_not_a_filter():
    high = sb.select_backbone(host="e_coli", copy_number="high")
    assert not high[0].disqualified
    assert high[0].backbone.copy_number == "high"


def test_assembly_standard_compatibility():
    picks = sb.select_backbone(host="e_coli", standard="Golden Gate")
    assert "Golden Gate" in picks[0].backbone.standard


def test_secretion_prefers_a_leader_vector():
    picks = sb.select_backbone(host="e_coli", expression=True, secretion=True)
    assert "periplasmic" in picks[0].backbone.notes or \
           "secretion" in picks[0].backbone.notes


def test_rejected_options_stay_visible():
    """A rejected option with its reason is more useful than an absent one."""
    picks = sb.select_backbone(host="e_coli")
    assert any(c.disqualified for c in picks)
    assert all(c.reasons for c in picks if c.disqualified)


def test_registry_query_requires_opting_into_the_network():
    with pytest.raises(ValueError, match="allow_network"):
        sb.query_parts("T7 promoter")


def test_unknown_registry_rejected():
    with pytest.raises(ValueError, match="registry must be one of"):
        sb.query_parts("x", registry="jbei", allow_network=True)


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

def test_plot_community(community):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    fig, axes = sb.plot_community(sb.steadycom(community))
    assert len(axes) == 2
    plt.close(fig)


def test_plot_rba(model):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    fig, axes = sb.plot_rba(model, budgets=[0.05, 0.2, 0.5])
    assert len(axes) == 2
    plt.close(fig)


def test_plot_enzyme_match(enzyme_db):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    res = sb.match_pathway_enzymes(["1.1.1.1", "9.9.9.9"], database=enzyme_db)
    fig, ax = sb.plot_enzyme_match(res)
    plt.close(fig)


def test_plot_backbone_choice():
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    fig, ax = sb.plot_backbone_choice(sb.select_backbone(host="e_coli"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# the table's remaining named backends: named things must exist or say why not
# ---------------------------------------------------------------------------

def test_gapseq_is_a_reconstruction_backend_not_just_an_alias():
    """`gapseq` appeared only as an alias string on gapfill_model while
    reconstruct_gem accepted no such method — the name was present and the
    thing was not."""
    with pytest.raises(ValueError, match="gapseq"):
        sb.reconstruct_gem("x.faa", method="not-a-method")
    import shutil
    if shutil.which("gapseq"):
        pytest.skip("gapseq is installed; the dispatch would run for real")
    with pytest.raises(ImportError, match="gapseq"):
        sb.reconstruct_gem("genome.fna", method="gapseq")


def test_gapseq_rejects_a_proteome():
    """CarveMe eats a proteome, gapseq eats a genome. Passing .faa to gapseq is
    the standard mistake and fails deep inside the pipeline otherwise."""
    import shutil
    if not shutil.which("gapseq"):
        pytest.skip("the ImportError fires before the extension check")
    with pytest.raises(ValueError, match="基因组|genome"):
        sb.reconstruct_gem("proteins.faa", method="gapseq")


def test_me_model_is_separate_from_rba():
    """`ME_model` used to be an alias on rba, which made it look as though
    ov.synbio ships a ME model when it ships a proteome constraint. They answer
    different questions and are now different functions."""
    assert sb.me_model is not sb.rba
    with pytest.raises(ImportError, match="COBRAme|cobrame"):
        sb.me_model()


def test_me_model_error_points_at_the_dependency_free_alternative():
    with pytest.raises(ImportError, match="rba"):
        sb.me_model()


def test_addgene_says_it_has_no_search_api():
    """The registered description advertised Addgene while the code raised
    NotImplementedError — the agent kernel reads that description."""
    with pytest.raises(NotImplementedError, match="Addgene"):
        sb.query_parts("pET28", registry="addgene", allow_network=True)


def test_enzyme_dynamics_is_the_bridge_into_ov_mol():
    """ov.mol has a full MD stack; synbio had no way to hand a design to it."""
    assert callable(sb.enzyme_dynamics)
    doc = sb.enzyme_dynamics.__doc__ or ""
    assert "QM/MM is not implemented" in doc, (
        "the docstring must say plainly that QM/MM is absent rather than imply "
        "mechanistic capability")
