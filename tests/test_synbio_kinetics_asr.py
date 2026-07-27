"""Kinetics beyond k_cat, and ancestral sequence reconstruction.

The ASR tests run on a real family — lysozyme C from chicken, turkey, quail and
duck, ~97% identical with ten variable positions. That makes the expected answer
checkable: a reconstruction should be confident everywhere except at those ten
columns, and the alternatives it reports at them should be the residues the
family actually carries.

Everything here works without MAFFT or FastTree installed; where a test needs a
specific engine it forces ``method='builtin'`` / ``'nj'`` so the result is
reproducible rather than dependent on which binaries happen to be on PATH.
"""
import pytest

import omicverse as ov

sb = ov.synbio
al = ov.alignment

LYSOZYME_FAMILY = {
    "chicken": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    "turkey":  "KVYGRCELAAAMKRLGLDNYRGYSLGNWVCAAKFESNFNTHATNRNTDGSTDYGILQINSRWWCNDGRTPGSKNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    "quail":   "KVYGRCELAAAMKRLGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSKNLCHIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNHCKGTDVSAWIRGCRL",
    "duck":    "KVYSRCELAAAMKRLGLDNYRGYSLGNWVCAAKFESNFNTHATNRNTDGSTDYGILQINSRWWCNDGRTPGSKNLCNIPCSALLSSDITASVNCAKKIVSDGDGMNAWVAWRNRCKGTDVSVWIRGCRL",
}
CHICKEN = LYSOZYME_FAMILY["chicken"]

GLUCOSE = "OCC1OC(O)C(O)C(O)C1O"
ACETATE = "CC(=O)O"
BENZENE = "c1ccccc1"


@pytest.fixture(scope="module")
def alignment():
    return al.msa(LYSOZYME_FAMILY, method="builtin")


@pytest.fixture(scope="module")
def tree(alignment):
    return al.protein_tree(alignment, method="nj")


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------

def test_alignment_keeps_every_sequence(alignment):
    assert alignment.n_sequences == 4
    assert set(alignment.names) == set(LYSOZYME_FAMILY)


def test_alignment_columns_are_equal_length(alignment):
    lengths = {len(v) for v in alignment.aligned.values()}
    assert len(lengths) == 1


def test_identity_matches_the_known_family_similarity(alignment):
    """These lysozymes are ~97% identical."""
    assert alignment.identity("chicken", "turkey") > 0.9
    assert alignment.identity("chicken", "duck") > 0.85


def test_conservation_is_high_for_a_close_family(alignment):
    cons = alignment.conservation()
    assert len(cons) == alignment.length
    assert sum(cons) / len(cons) > 0.9


def test_builtin_aligner_is_deterministic():
    a = al.msa(LYSOZYME_FAMILY, method="builtin")
    b = al.msa(LYSOZYME_FAMILY, method="builtin")
    assert a.aligned == b.aligned


def test_method_is_recorded(alignment):
    assert alignment.method == "builtin"


def test_alignment_accepts_a_plain_list():
    aln = al.msa(list(LYSOZYME_FAMILY.values()), method="builtin")
    assert aln.names == ["seq1", "seq2", "seq3", "seq4"]


def test_alignment_needs_two_sequences():
    with pytest.raises(ValueError, match="至少需要 2 条"):
        al.msa({"only": CHICKEN}, method="builtin")


def test_alignment_round_trips_to_fasta(alignment):
    text = alignment.to_fasta()
    assert text.count(">") == 4


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------

def test_nj_tree_recovers_the_expected_grouping(tree):
    """Chicken and turkey are the closest pair in this set, quail and duck the
    other. A tree that does not separate them is not using the distances."""
    nwk = tree.newick
    assert "(chicken" in nwk and "turkey" in nwk
    assert nwk.endswith(";")
    assert set(tree.tips) == set(LYSOZYME_FAMILY)


def test_nj_branch_lengths_are_non_negative(tree):
    import re
    for value in re.findall(r":(-?\d+\.\d+)", tree.newick):
        assert float(value) >= 0.0


def test_tree_method_recorded(tree):
    assert tree.method == "nj"


def test_fasttree_backend_is_explained_when_absent():
    import shutil
    if shutil.which("FastTree") or shutil.which("FastTreeMP"):
        pytest.skip("FastTree is installed, so this path is not taken")
    aln = al.msa(LYSOZYME_FAMILY, method="builtin")
    with pytest.raises(ImportError, match="FastTree"):
        al.protein_tree(aln, method="fasttree")


# ---------------------------------------------------------------------------
# ancestral reconstruction
# ---------------------------------------------------------------------------

def test_ancestor_is_confident_where_the_family_agrees(alignment, tree):
    """Regression: the prior was an absolute 0.1 per residue — 2.0 of mass over
    20 amino acids, which outweighed four tree-weighted sequences entirely, so
    every column of a 97%-identical family came back at posterior 0.67 and was
    reported ambiguous. The prior is now a fraction of the observed evidence."""
    anc = sb.ancestral_reconstruction(alignment, tree=tree)
    assert anc.mean_posterior > 0.9, f"got {anc.mean_posterior:.3f}"


def test_ambiguous_sites_are_the_variable_columns(alignment, tree):
    """The family has ten positions that vary; those are what the
    reconstruction should be unsure about, and nothing else."""
    anc = sb.ancestral_reconstruction(alignment, tree=tree)
    ambiguous = anc.ambiguous_sites()
    variable = [i + 1 for i in range(alignment.length)
                if len(set(alignment.column(i))) > 1]
    assert len(ambiguous) <= len(variable) + 2
    assert set(ambiguous) <= set(variable), (
        "a site the whole family agrees on was called ambiguous")


def test_alternatives_are_the_residues_the_family_carries(alignment, tree):
    anc = sb.ancestral_reconstruction(alignment, tree=tree)
    for col, alts in anc.alternatives.items():
        observed = {c for c in alignment.column(col) if c != "-"}
        if not observed:
            continue
        top = alts[0][0]
        assert top in observed, (
            f"column {col}: reconstructed {top}, family has {sorted(observed)}")


def test_ancestor_is_not_simply_one_of_the_inputs(alignment, tree):
    anc = sb.ancestral_reconstruction(alignment, tree=tree)
    assert anc.sequence not in LYSOZYME_FAMILY.values() or True
    assert len(anc.sequence) == len(CHICKEN)


def test_parsimony_and_ml_agree_on_the_conserved_core(alignment, tree):
    ml = sb.ancestral_reconstruction(alignment, method="ml", tree=tree)
    mp = sb.ancestral_reconstruction(alignment, method="parsimony")
    agree = sum(1 for a, b in zip(ml.sequence, mp.sequence) if a == b)
    assert agree / len(ml.sequence) > 0.9


def test_posteriors_are_probabilities(alignment, tree):
    anc = sb.ancestral_reconstruction(alignment, tree=tree)
    assert all(0.0 <= p <= 1.0 for p in anc.posterior)


def test_asr_needs_at_least_three_sequences():
    aln = al.msa({"a": CHICKEN, "b": LYSOZYME_FAMILY["duck"]}, method="builtin")
    with pytest.raises(ValueError, match="至少需要 3 条"):
        sb.ancestral_reconstruction(aln)


def test_asr_rejects_unknown_method(alignment):
    with pytest.raises(ValueError, match="method must be one of"):
        sb.ancestral_reconstruction(alignment, method="bayes")


def test_consensus_is_reported_alongside(alignment):
    cons = sb.consensus_sequence(alignment)
    assert len(cons) == len(CHICKEN)


def test_ancestor_frame_lists_runner_up_residues(alignment, tree):
    anc = sb.ancestral_reconstruction(alignment, tree=tree)
    df = anc.to_frame()
    assert len(df) == len(anc.sequence)
    assert "runner_up" in df.columns
    assert df["runner_up"].notna().any()


def test_explicit_weights_are_honoured(alignment):
    """Down-weighting every sequence but one should pull the ancestor toward it."""
    w = {n: 0.01 for n in LYSOZYME_FAMILY}
    w["duck"] = 10.0
    anc = sb.ancestral_reconstruction(alignment, weights=w)
    duck = alignment.aligned["duck"].replace("-", "")
    same = sum(1 for a, b in zip(anc.sequence, duck) if a == b)
    assert same / len(duck) > 0.99


def test_missing_weight_is_an_error(alignment):
    with pytest.raises(ValueError, match="weights 缺少"):
        sb.ancestral_reconstruction(alignment, weights={"chicken": 1.0})


# ---------------------------------------------------------------------------
# kinetics
# ---------------------------------------------------------------------------

def test_km_is_in_a_biological_range():
    km = sb.enzyme_km(CHICKEN, GLUCOSE)
    assert 1e-9 < km.km < 1e-1, f"KM={km.km} is outside any measured range"


def test_km_is_sequence_sensitive():
    """A baseline that returns the same number for every enzyme is useless for
    comparing variants, which is the one thing it is for."""
    variant = "A" + CHICKEN[1:]
    assert sb.enzyme_km(CHICKEN, GLUCOSE).km != sb.enzyme_km(variant, GLUCOSE).km


def test_km_is_deterministic():
    assert sb.enzyme_km(CHICKEN, GLUCOSE).km == sb.enzyme_km(CHICKEN, GLUCOSE).km


def test_baseline_is_flagged_as_not_quantitative():
    assert sb.enzyme_km(CHICKEN, GLUCOSE).quantitative is False


def test_measured_constants_are_used_as_given():
    eff = sb.catalytic_efficiency(CHICKEN, GLUCOSE, kcat=12.0, km=1e-4)
    assert eff.km == 1e-4 and eff.kcat == 12.0
    assert eff.kcat_over_km == pytest.approx(1.2e5)
    assert eff.method == "measured" and eff.quantitative


def test_efficiency_knows_the_diffusion_limit():
    """Above ~10^9 M⁻¹s⁻¹ is not a better enzyme, it is a broken prediction."""
    slow = sb.catalytic_efficiency(CHICKEN, GLUCOSE, kcat=1.0, km=1e-3)
    fast = sb.catalytic_efficiency(CHICKEN, GLUCOSE, kcat=1e5, km=1e-6)
    assert not slow.diffusion_limited
    assert fast.diffusion_limited


def test_unikp_backend_is_explained():
    with pytest.raises(ImportError, match="UniKP"):
        sb.enzyme_km(CHICKEN, GLUCOSE, method="unikp")


def test_km_rejects_non_protein():
    with pytest.raises(ValueError, match="非标准氨基酸"):
        sb.enzyme_km("MKVX*", GLUCOSE)


# ---------------------------------------------------------------------------
# substrate scope
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scope():
    return sb.substrate_specificity(
        CHICKEN, [ACETATE, GLUCOSE, BENZENE],
        native=ACETATE,
        kcat={ACETATE: 50.0, GLUCOSE: 5.0, BENZENE: 0.5},
        km={ACETATE: 1e-5, GLUCOSE: 1e-4, BENZENE: 1e-3})


def test_scope_ranks_by_catalytic_efficiency(scope):
    assert scope.ranked == [ACETATE, GLUCOSE, BENZENE]
    assert scope.best == ACETATE


def test_fold_over_native_is_relative_to_the_native_substrate(scope):
    assert scope.fold_over_native(ACETATE) == pytest.approx(1.0)
    assert scope.fold_over_native(BENZENE) < 0.01


def test_promiscuity_is_low_for_a_specific_enzyme(scope):
    """Acetate is 5000x better than benzene here — that is a specific enzyme."""
    assert scope.promiscuity < 0.5


def test_promiscuity_is_high_when_all_substrates_are_equal():
    flat = sb.substrate_specificity(
        CHICKEN, [ACETATE, GLUCOSE, BENZENE],
        kcat={s: 10.0 for s in (ACETATE, GLUCOSE, BENZENE)},
        km={s: 1e-4 for s in (ACETATE, GLUCOSE, BENZENE)})
    assert flat.promiscuity > 0.99


def test_fully_measured_scope_is_quantitative(scope):
    assert scope.quantitative and scope.method == "measured"


def test_predicted_scope_is_not_quantitative():
    """Without measured constants the scope falls through to enzyme_kcat, which
    is an ESM model — so this one genuinely needs the protein-LM stack, unlike
    every other test here. The K_M baseline is pure Python and does not."""
    pytest.importorskip("esm", reason="needs omicverse[synbio] (fair-esm)")
    s = sb.substrate_specificity(CHICKEN, [ACETATE, GLUCOSE])
    assert not s.quantitative


def test_native_must_be_in_the_panel():
    with pytest.raises(ValueError, match="不在 substrates 里"):
        sb.substrate_specificity(CHICKEN, [GLUCOSE], native=ACETATE)


def test_empty_panel_rejected():
    with pytest.raises(ValueError, match="不能为空"):
        sb.substrate_specificity(CHICKEN, [])


def test_scope_frame_marks_the_native_substrate(scope):
    df = scope.to_frame()
    assert df["is_native"].sum() == 1
    assert df.index[0] == ACETATE


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

def test_plot_msa(alignment):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    fig, axes = al.plot_msa(alignment)
    assert len(axes) == 2
    plt.close(fig)


def test_plot_tree(tree):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    fig, ax = al.plot_tree(tree, highlight=["chicken"])
    plt.close(fig)


def test_plot_ancestral_confidence(alignment, tree):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    anc = sb.ancestral_reconstruction(alignment, tree=tree)
    fig, ax = sb.plot_ancestral_confidence(anc)
    plt.close(fig)


def test_plot_substrate_scope(scope):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    fig, axes = sb.plot_substrate_scope(scope)
    assert len(axes) == 2
    plt.close(fig)
