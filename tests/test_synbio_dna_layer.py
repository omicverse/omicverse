"""Layer C: codon harmonisation, synthesis difficulty, overhang fidelity,
terminators, and truth-table compilation.

Landmarks these tests are anchored on, so a regression shows up as a wrong
biological call rather than a changed number:

* **rrnB T1** — the canonical strong *E. coli* intrinsic terminator. It must
  score ``strong``; a GC hairpin with no U-tract must not score as a terminator
  at all.
* **Harmonisation moves CAI the *wrong* way on purpose.** If harmonised CAI ever
  equals optimised CAI, harmonisation has silently become optimisation and the
  whole point is gone.
* **Golden Gate fidelity must degrade with fragment count** — that is the
  phenomenon the function exists to expose.
* **A single-minterm truth table must compile to that minterm**, not its
  negation.
"""
import pytest

import omicverse as ov

sb = ov.synbio

# E. coli rrnB T1
T1 = "AAAAAAGGCCGCTTTTGCGGCCTTTTTTTAAA"
# same hairpin, no U-tract: pauses polymerase but does not release it
HAIRPIN_NO_U = "AAAAAAGGCCGCTTTTGCGGCCGCACGCACG"

# human GAPDH CDS opening
GAPDH = (
    "ATGGGGAAGGTGAAGGTCGGAGTCAACGGATTTGGTCGTATTGGGCGCCTGGTCACCAGGGCTGCTTTTAACTCTGGTAAA"
    "GTGGATATTGTTGCCATCAATGACCCCTTCATTGACCTCAACTACATGGTTTACATGTTCCAATATGATTCCACCCATGGC"
)

_HAS_TABLES = True
try:
    import python_codon_tables  # noqa: F401
except ImportError:  # pragma: no cover
    _HAS_TABLES = False

needs_tables = pytest.mark.skipif(
    not _HAS_TABLES, reason="needs python_codon_tables")


# ---------------------------------------------------------------------------
# codon harmonisation
# ---------------------------------------------------------------------------

@needs_tables
def test_harmonisation_preserves_the_protein():
    """Replacements come only from the same synonymous family, by construction."""
    from omicverse.synbio._harmonize import _CODON_TABLE

    res = sb.codon_harmonize(GAPDH, "h_sapiens", "e_coli")
    before = [_CODON_TABLE[GAPDH[i:i + 3]] for i in range(0, len(GAPDH), 3)]
    after = [_CODON_TABLE[res.sequence[i:i + 3]]
             for i in range(0, len(res.sequence), 3)]
    assert before == after


@needs_tables
def test_harmonisation_is_not_optimisation():
    """The load-bearing distinction. Harmonised CAI must sit below the CAI-optimal
    rewrite — a rare codon is kept rare on purpose, to preserve the ribosomal
    pauses that let a domain fold before the next one is synthesised."""
    res = sb.codon_harmonize(GAPDH, "h_sapiens", "e_coli")
    assert res.harmonized_cai < res.optimized_cai, (
        f"harmonised CAI {res.harmonized_cai:.3f} reached the optimum "
        f"{res.optimized_cai:.3f} — harmonisation has become optimisation")


@needs_tables
def test_harmonisation_preserves_rarity_rank():
    """Rank in the synonymous family is what is being carried across hosts."""
    res = sb.codon_harmonize(GAPDH, "h_sapiens", "e_coli")
    assert res.rank_correlation > 0.99
    df = res.to_frame()
    assert (df["source_rank"] == df["target_rank"]).all()


@needs_tables
def test_optimisation_reaches_cai_one():
    df = sb.compare_codon_strategies(GAPDH, "h_sapiens", "e_coli")
    assert df.loc["optimized", "cai_in_target"] == pytest.approx(1.0, abs=1e-9)
    assert df.loc["harmonized", "cai_in_target"] < 1.0
    assert df.loc["native", "cai_in_target"] < 1.0


@needs_tables
def test_harmonising_to_the_same_host_changes_nothing():
    res = sb.codon_harmonize(GAPDH, "h_sapiens", "h_sapiens")
    assert res.n_changed == 0
    assert res.sequence == GAPDH


def test_harmonize_rejects_a_non_triplet_cds():
    with pytest.raises(ValueError, match="不是 3 的倍数"):
        sb.codon_harmonize("ATGGC", "h_sapiens", "e_coli")


def test_harmonize_rejects_unknown_host():
    with pytest.raises(ValueError, match="host must be one of"):
        sb.codon_harmonize(GAPDH, "h_sapiens", "e_koli")


def test_harmonize_rejects_non_dna():
    with pytest.raises(ValueError, match="非 DNA 字符"):
        sb.codon_harmonize("ATGXYZ", "h_sapiens", "e_coli")


# ---------------------------------------------------------------------------
# synthesis complexity
# ---------------------------------------------------------------------------

def test_a_homopolymer_is_flagged():
    asm = sb.synthesis_complexity("ATGC" * 10 + "A" * 14 + "GCTA" * 10)
    kinds = {i.kind for i in asm.issues}
    assert "homopolymer" in kinds
    hp = [i for i in asm.issues if i.kind == "homopolymer"][0]
    assert hp.end - hp.start + 1 == 14


def test_extreme_gc_is_flagged():
    asm = sb.synthesis_complexity("GC" * 120)
    assert any(i.kind == "gc_extreme" for i in asm.issues)
    assert asm.difficulty != "routine"


def test_a_repeat_is_reported_once_not_per_window():
    """A 160-nt duplication produced 66 near-identical rows before merging,
    which made the report unreadable and hid the other issues."""
    asm = sb.synthesis_complexity(GAPDH + GAPDH)
    repeats = [i for i in asm.issues if i.kind == "direct_repeat"]
    assert len(repeats) == 1, f"expected one merged repeat, got {len(repeats)}"
    assert repeats[0].end - repeats[0].start > 100


def test_inverted_repeats_are_distinguished_from_direct_ones():
    from omicverse.synbio._harmonize import _revcomp
    seq = GAPDH + "ATGC" * 5 + _revcomp(GAPDH)
    asm = sb.synthesis_complexity(seq)
    assert any(i.kind == "inverted_repeat" for i in asm.issues)


def test_a_benign_sequence_is_routine():
    asm = sb.synthesis_complexity(GAPDH)
    assert asm.difficulty == "routine"
    assert asm.score < 0.25


def test_issues_carry_coordinates():
    """"Fix bases 412-431" is actionable; "this is hard" is not."""
    asm = sb.synthesis_complexity("ATGC" * 10 + "T" * 12 + "ATGC" * 10)
    for issue in asm.issues:
        assert 1 <= issue.start <= issue.end <= asm.sequence_length
        assert issue.detail


def test_synthesis_frame_is_tabular():
    asm = sb.synthesis_complexity(GAPDH + GAPDH)
    df = asm.to_frame()
    assert set(df.columns) == {"kind", "start", "end", "detail", "severity"}


# ---------------------------------------------------------------------------
# Golden Gate overhang fidelity
# ---------------------------------------------------------------------------

def test_a_correct_junction_is_an_overhang_with_itself():
    """Two fragments join when each presents the *same* 4 nt on the same strand.

    Scoring the diagonal as `a` against `revcomp(a)` made the matrix measure
    palindromicity instead, and a perfectly orthogonal set scored 0.30.
    """
    from omicverse.synbio._fidelity import _pair_score
    assert _pair_score("AATG", "AATG") == pytest.approx(1.0)
    assert _pair_score("AATG", "CGCT") < 0.01


def test_an_orthogonal_set_has_high_fidelity():
    rep = sb.overhang_fidelity(["AATG", "AGGT", "GCTT", "CGCT"])
    assert rep.fidelity > 0.95
    assert rep.verdict == "high fidelity"


def test_a_single_mismatch_pair_wrecks_the_set():
    """AATG and AATC differ only at the last position."""
    rep = sb.overhang_fidelity(["AATG", "AATC", "GCTT", "CGCT"])
    assert rep.fidelity < 0.9
    assert rep.cross_reactions, "the offending pair should be named"
    worst = rep.cross_reactions[0]
    assert {worst.left, worst.right} == {"AATG", "AATC"}


def test_duplicate_and_palindromic_overhangs_are_called_invalid():
    dup = sb.overhang_fidelity(["AATG", "AATG", "GCTT", "CGCT"])
    assert dup.duplicates == ["AATG"] and dup.verdict == "invalid set"
    pal = sb.overhang_fidelity(["AATT", "GCTT", "CGCT", "AGGT"])
    assert "AATT" in pal.palindromic and pal.verdict == "invalid set"


def test_fidelity_degrades_as_fragments_are_added():
    """The phenomenon this function exists to expose: 4-nt overhangs run out of
    orthogonal space, so a set that is fine at six fragments is not at twenty."""
    scores = [sb.overhang_fidelity(sb.design_overhang_set(n)).fidelity
              for n in (4, 8, 16, 20)]
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] > 0.99
    assert scores[-1] < scores[0]


def test_designed_sets_are_orthogonal_and_deterministic():
    a = sb.design_overhang_set(8)
    b = sb.design_overhang_set(8)
    assert a == b, "the greedy search must be deterministic"
    assert len(set(a)) == 8
    assert sb.overhang_fidelity(a).fidelity > 0.95


def test_design_excludes_palindromes_and_homopolymers():
    from omicverse.synbio._fidelity import _revcomp
    for oh in sb.design_overhang_set(12):
        assert oh != _revcomp(oh), f"{oh} is palindromic"
        assert len(set(oh)) > 1, f"{oh} is a homopolymer"


def test_design_can_extend_a_committed_set():
    fixed = ["AATG", "GCTT"]
    got = sb.design_overhang_set(6, fixed=fixed)
    assert got[:2] == fixed
    assert len(set(got)) == 6


def test_design_says_when_the_space_runs_out():
    """There are only 4^4 four-mers, minus palindromes and homopolymers."""
    with pytest.raises(ValueError, match="正交容量有限|只能找到"):
        sb.design_overhang_set(300)


def test_or_compiles_through_its_complement():
    """OR expands to three minterms and nine gates in plain sum-of-products —
    one more than the eight-part library — while its complement NOR is a single
    gate, so OR is that gate plus an inverter."""
    cc = sb.compile_circuit(TABLES["OR"])
    assert cc.feasible, cc.failures
    assert cc.n_gates == 2, f"OR should need 2 gates, got {cc.n_gates}"


def test_measured_table_requires_a_table():
    with pytest.raises(ValueError, match="fidelity_table"):
        sb.overhang_fidelity(["AATG", "GCTT"], method="measured")


def test_a_supplied_table_is_used_and_labelled():
    table = {("AATG", "AATG"): 1.0, ("GCTT", "GCTT"): 1.0,
             ("AATG", "GCTT"): 0.9}
    rep = sb.overhang_fidelity(["AATG", "GCTT"], fidelity_table=table)
    assert rep.method == "measured"
    assert rep.fidelity < 0.4, "a 0.9 cross-reaction must dominate"


def test_fallback_method_is_labelled():
    assert sb.overhang_fidelity(["AATG", "GCTT"]).method == "mismatch"


def test_overhang_must_be_four_bases():
    with pytest.raises(ValueError, match="4 个 ACGT"):
        sb.overhang_fidelity(["AAT", "GCTT"])


# ---------------------------------------------------------------------------
# terminators
# ---------------------------------------------------------------------------

def test_rrnb_t1_is_strong():
    t = sb.terminator_strength(T1)
    assert t.classification == "strong", f"got {t.strength:.2f}"
    assert t.has_hairpin
    assert t.u_tract_length >= 6
    assert t.stem_gc > 0.6, "the stem should be the GC-rich one"


def test_the_hairpin_search_does_not_swallow_the_u_tract():
    """Maximising stem stability paired T1's 5' A-tract with its own 3' U-tract
    into a 12-nt 'stem', consuming the features that make it a terminator."""
    t = sb.terminator_strength(T1)
    assert t.stem_length <= 9, f"stem of {t.stem_length} spans the tracts"
    assert t.u_tract_length > 0


def test_a_hairpin_without_a_u_tract_is_not_a_terminator():
    t = sb.terminator_strength(HAIRPIN_NO_U)
    assert t.u_tract_length == 0
    assert t.classification == "not a terminator"
    assert t.readthrough > 0.7


def test_readthrough_complements_strength():
    t = sb.terminator_strength(T1)
    assert t.readthrough == pytest.approx(1.0 - t.strength)


def test_terminator_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        sb.terminator_strength(T1, method="rho")


# ---------------------------------------------------------------------------
# circuit compilation
# ---------------------------------------------------------------------------

TABLES = {
    "NOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0},
    "AND": {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
    "OR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
    "XOR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    "XNOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1},
}


@pytest.mark.parametrize("name", sorted(TABLES))
def test_every_two_input_function_compiles(name):
    cc = sb.compile_circuit(TABLES[name])
    assert cc.feasible, f"{name}: {cc.failures}"
    assert cc.n_gates >= 1
    assert cc.dna, "a feasible circuit must emit DNA"


@pytest.mark.parametrize("name", sorted(TABLES))
def test_the_compiled_network_realises_the_truth_table(name):
    """Boolean check on the synthesised topology, independent of parts.

    A single-minterm table used to gain an extra inverter, so AND compiled to
    NAND and NOR to OR. The level check then reported 'infeasible', which read
    as a parts problem rather than the logic error it was.
    """
    from omicverse.synbio._compile import _gate_truth

    table = TABLES[name]
    cc = sb.compile_circuit(table)
    by_id = {g.gate_id: g for g in cc.gates}
    out_gate = [g for g in cc.gates if g.kind == "OUTPUT"][0]
    for combo, expected in table.items():
        got = _gate_truth(by_id, out_gate, cc.inputs, combo)
        assert int(got) == expected, (
            f"{name}: inputs {combo} gave {int(got)}, expected {expected}")


def test_a_boolean_expression_compiles_the_same_way():
    cc = sb.compile_circuit("A and not B")
    assert cc.feasible
    assert cc.inputs == ["A", "B"]
    from omicverse.synbio._compile import _gate_truth
    by_id = {g.gate_id: g for g in cc.gates}
    out = [g for g in cc.gates if g.kind == "OUTPUT"][0]
    assert _gate_truth(by_id, out, cc.inputs, (1, 0)) is True
    assert _gate_truth(by_id, out, cc.inputs, (1, 1)) is False


def test_each_gate_gets_a_distinct_repressor():
    """Reusing a repressor makes two gates cross-talk into one."""
    cc = sb.compile_circuit(TABLES["XOR"])
    parts = [g.part.name for g in cc.gates if g.part is not None]
    assert len(parts) == len(set(parts))


def test_levels_are_checked_not_just_topology():
    """The check that makes this compilation rather than graph rewriting."""
    cc = sb.compile_circuit(TABLES["XOR"], min_dynamic_range=1e6)
    assert not cc.feasible
    assert cc.failures and "死区" in cc.failures[0]


def test_too_few_parts_is_reported_clearly():
    lib = list(sb.DEFAULT_GATE_LIBRARY)[:2]
    cc = sb.compile_circuit(TABLES["XOR"], library=lib)
    assert not cc.feasible
    assert "门库只有" in cc.failures[0]


def test_library_provenance_is_recorded():
    """The bundled response parameters are illustrative, and the result says so."""
    cc = sb.compile_circuit(TABLES["AND"])
    assert "illustrative" in cc.library_source
    cc2 = sb.compile_circuit(TABLES["AND"], library=list(sb.DEFAULT_GATE_LIBRARY))
    assert cc2.library_source == "user library"


def test_incomplete_truth_table_is_rejected():
    with pytest.raises(ValueError, match="真值表不完整"):
        sb.compile_circuit({(0, 0): 0, (1, 1): 1})


def test_constant_truth_tables_are_rejected():
    with pytest.raises(ValueError, match="恒为 0"):
        sb.compile_circuit({(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0})
    with pytest.raises(ValueError, match="恒为 1"):
        sb.compile_circuit({(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1})


def test_emitted_dna_is_well_formed():
    cc = sb.compile_circuit(TABLES["AND"])
    assert set(cc.dna) <= set("ACGT")
    assert len(cc.dna) > 100


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

def test_plot_synthesis_complexity():
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    fig, axes = sb.plot_synthesis_complexity(GAPDH + GAPDH)
    assert len(axes) == 2
    plt.close(fig)


def test_plot_overhang_fidelity():
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    rep = sb.overhang_fidelity(sb.design_overhang_set(6))
    fig, ax = sb.plot_overhang_fidelity(rep)
    plt.close(fig)


def test_plot_compiled_circuit():
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    cc = sb.compile_circuit(TABLES["XOR"])
    fig, axes = sb.plot_compiled_circuit(cc)
    assert len(axes) == 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# a bare install must still get sensible errors
# ---------------------------------------------------------------------------

def test_bad_host_is_reported_as_a_bad_host_not_a_missing_dependency():
    """Argument validation must not depend on an optional package.

    codon_harmonize deferred host checking to codon_usage, which validated the
    *source* host first — so a typo in the target host was only reported after
    python_codon_tables had been imported. On a CI runner without the synbio
    extra the caller was told to install a package when the real problem was
    their own argument, and that is how this surfaced: green locally, red on the
    bare `build` job.
    """
    from omicverse.synbio._harmonize import _check_hosts

    with pytest.raises(ValueError, match="host must be one of"):
        _check_hosts("h_sapiens", "e_koli")
    _check_hosts("h_sapiens", "e_coli")       # both valid: no exception


def test_host_validation_happens_before_the_codon_tables_are_needed():
    """The order matters, so assert the order rather than the outcome."""
    import inspect
    from omicverse.synbio import _harmonize

    src = inspect.getsource(_harmonize.codon_harmonize)
    assert src.index("_check_hosts(") < src.index("codon_usage("), (
        "hosts must be validated before codon_usage imports python_codon_tables")


@needs_tables
def test_codon_usage_survives_codon_optimize():
    """Regression: DNAchisel writes `log_best_frequencies` and
    `log_codons_frequencies` into the dict python_codon_tables caches and hands
    out, and codon_usage iterated every top-level key — so the log table (itself
    keyed by codon) overwrote all 64 frequencies with negative values and every
    CAI computed afterwards was 0.

    Same process only, so it presented as test-order flakiness: the layer-C file
    passed alone and failed after any suite that called codon_optimize.
    """
    from omicverse.synbio._harmonize import codon_usage

    before = codon_usage("e_coli")
    sb.codon_optimize(GAPDH, host="e_coli")
    after = codon_usage("e_coli")

    assert after == before
    assert len(after) == 64, f"expected 64 codons, got {len(after)}"
    assert all(0.0 <= v <= 1.0 for v in after.values()), "frequencies, not logs"


@needs_tables
def test_cai_survives_codon_optimize():
    pytest.importorskip("dnachisel", reason="needs omicverse[synbio] (dnachisel)")
    sb.codon_optimize(GAPDH, host="e_coli")
    df = sb.compare_codon_strategies(GAPDH, "h_sapiens", "e_coli")
    assert df.loc["optimized", "cai_in_target"] == pytest.approx(1.0, abs=1e-9)
