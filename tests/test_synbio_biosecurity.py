"""Biosecurity screening harness.

The reference sets used here are **innocuous proteins standing in for a
reference**: a GFP fragment plays "sequence of concern" and a lysozyme fragment
plays "benign housekeeping". That is deliberate and it is not a shortcut — what
is under test is the harness (six-frame translation, threshold handling, benign
subtraction, decision logic, and above all the refusal to pass without a
reference), not the biology of any particular hazard. ``ov.synbio`` ships no
sequences of concern and these tests do not introduce any.

The single most important assertion in this file is
:func:`test_no_database_raises_instead_of_passing`.
"""
import os

import pytest

import omicverse as ov
from omicverse.synbio._biosecurity import translate_frames

sb = ov.synbio

# innocuous stand-ins (see module docstring)
CONCERN_PROTEIN = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ"
    "HDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNG"
)
BENIGN_PROTEIN = (
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPC"
    "SALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
)

_HAS_ALIGNER = any(
    os.path.exists(os.path.join(p, exe))
    for p in os.environ.get("PATH", "").split(os.pathsep) if p
    for exe in ("diamond", "blastp")
)
needs_aligner = pytest.mark.skipif(
    not _HAS_ALIGNER, reason="needs DIAMOND or BLAST+ on PATH")


@pytest.fixture
def concern_db(tmp_path):
    p = tmp_path / "concern.faa"
    p.write_text(f">stand_in_concern_01\n{CONCERN_PROTEIN}\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def benign_db(tmp_path):
    p = tmp_path / "benign.faa"
    p.write_text(f">stand_in_housekeeping_01\n{BENIGN_PROTEIN}\n", encoding="utf-8")
    return str(p)


def _back_translate(protein: str) -> str:
    """One arbitrary codon per residue — enough to exercise the DNA path."""
    codons = {
        "A": "GCT", "R": "CGT", "N": "AAT", "D": "GAT", "C": "TGT", "Q": "CAA",
        "E": "GAA", "G": "GGT", "H": "CAT", "I": "ATT", "L": "CTT", "K": "AAA",
        "M": "ATG", "F": "TTT", "P": "CCT", "S": "TCT", "T": "ACT", "W": "TGG",
        "Y": "TAT", "V": "GTT",
    }
    return "".join(codons[a] for a in protein)


# ---------------------------------------------------------------------------
# the rule that matters most
# ---------------------------------------------------------------------------

def test_no_database_raises_instead_of_passing():
    """A screener with nothing to compare against must not report 'clean'.

    A pass recorded on an unconfigured screen is worse than no screen: it gets
    filed, and everyone downstream believes a check happened.
    """
    with pytest.raises(ValueError, match="没有配置任何关注序列库|虚假保证"):
        sb.screen_sequence(CONCERN_PROTEIN, databases=[])


def test_no_database_error_says_where_to_get_one():
    with pytest.raises(ValueError, match="OMICOS_BIOSECURITY_DB"):
        sb.screen_sequence(CONCERN_PROTEIN, databases=[])


def test_opting_out_is_labelled_not_screened(monkeypatch):
    monkeypatch.delenv("OMICOS_BIOSECURITY_DB", raising=False)
    rep = sb.screen_sequence(CONCERN_PROTEIN, databases=[], require_database=False)
    assert rep.decision == "not_screened"
    assert rep.clean is False, "'not_screened' must never read as clean"
    assert rep.notes and any("虚假" in n or "不能作为放行依据" in n for n in rep.notes)


def test_missing_database_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        sb.screen_sequence(CONCERN_PROTEIN,
                           databases=[str(tmp_path / "nope.faa")])


def test_clean_is_only_true_after_a_real_screen():
    rep = sb.screen_sequence(CONCERN_PROTEIN, databases=[],
                             require_database=False)
    assert not rep.clean


# ---------------------------------------------------------------------------
# six-frame translation
# ---------------------------------------------------------------------------

def test_all_six_frames_are_produced():
    frames = translate_frames("ATGGCTTGTAAACGT")
    assert set(frames) == {"+1", "+2", "+3", "-1", "-2", "-3"}


def test_forward_frame_one_translates_correctly():
    assert translate_frames("ATGGCTTGTAAA")["+1"] == "MACK"


def test_reverse_frames_read_the_other_strand():
    """A screen that only reads the annotated strand is evaded by reversing it."""
    dna = _back_translate("MACK")
    rc_frames = translate_frames(dna)
    assert "MACK" in rc_frames["+1"]
    assert "MACK" not in rc_frames["-1"]
    # ...but reverse-complementing the construct moves it to a minus frame
    from omicverse.synbio._biosecurity import _revcomp
    flipped = translate_frames(_revcomp(dna))
    assert any("MACK" in p for p in
               (flipped["-1"], flipped["-2"], flipped["-3"]))


def test_stop_codons_are_marked():
    assert translate_frames("TAA")["+1"] == "*"


def test_unknown_codons_become_x():
    assert "X" in translate_frames("ATGNNNAAA")["+1"]


# ---------------------------------------------------------------------------
# screening a protein
# ---------------------------------------------------------------------------

@needs_aligner
def test_a_matching_protein_is_flagged(concern_db):
    rep = sb.screen_sequence(CONCERN_PROTEIN, databases=[concern_db])
    assert rep.hits, "an exact match to the reference produced no hits"
    assert rep.decision == "flag", f"expected flag, got {rep.decision}"
    assert rep.flagged_hits
    assert max(h.identity for h in rep.hits) > 95.0


@needs_aligner
def test_an_unrelated_protein_passes(concern_db):
    rep = sb.screen_sequence(BENIGN_PROTEIN, databases=[concern_db])
    assert rep.decision == "pass"
    assert not rep.flagged_hits


@needs_aligner
def test_decision_is_review_for_a_weak_hit(concern_db):
    """Between the reporting threshold and the flag threshold, ask a human."""
    rep = sb.screen_sequence(CONCERN_PROTEIN, databases=[concern_db],
                             flag_identity=101.0)
    assert rep.decision == "review"


# ---------------------------------------------------------------------------
# benign subtraction — the Common Mechanism's central idea
# ---------------------------------------------------------------------------

@needs_aligner
def test_benign_homology_downgrades_a_hit(concern_db, benign_db):
    """A region better explained by housekeeping homology is marked, not hidden.

    Without this, domains shared by hazardous and innocuous proteins bury the
    report in noise and the screen stops being read.
    """
    rep = sb.screen_sequence(BENIGN_PROTEIN,
                             databases=[benign_db],
                             benign_databases=[benign_db])
    assert rep.hits, "the stand-in should hit its own reference"
    assert all(h.downgraded for h in rep.hits)
    assert rep.decision == "pass"


@needs_aligner
def test_downgraded_hits_are_still_reported(concern_db, benign_db):
    rep = sb.screen_sequence(BENIGN_PROTEIN, databases=[benign_db],
                             benign_databases=[benign_db])
    assert len(rep.hits) > len(rep.flagged_hits)
    df = rep.to_frame()
    assert df["downgraded"].any()
    assert df["benign_competitor"].notna().any()


# ---------------------------------------------------------------------------
# DNA input
# ---------------------------------------------------------------------------

@needs_aligner
def test_dna_is_translated_before_screening(concern_db):
    dna = _back_translate(CONCERN_PROTEIN)
    rep = sb.screen_sequence(dna, databases=[concern_db])
    assert rep.sequence_type == "dna"
    assert rep.hits, "DNA encoding a reference protein should be caught"
    assert rep.decision == "flag"


@needs_aligner
def test_reverse_complemented_dna_is_still_caught(concern_db):
    """The six-frame screen is what makes this true."""
    from omicverse.synbio._biosecurity import _revcomp
    dna = _revcomp(_back_translate(CONCERN_PROTEIN))
    rep = sb.screen_sequence(dna, databases=[concern_db])
    assert rep.decision == "flag", "reversing the construct evaded the screen"
    assert any(h.query_frame.startswith("-") for h in rep.hits)


def test_sequence_type_can_be_forced(concern_db):
    with pytest.raises(ValueError, match="sequence_type must be one of"):
        sb.screen_sequence("ACGT", databases=[concern_db], sequence_type="rna")


def test_empty_sequence_rejected(concern_db):
    with pytest.raises(ValueError, match="序列为空"):
        sb.screen_sequence("", databases=[concern_db])


def test_unknown_method_rejected(concern_db):
    with pytest.raises(ValueError, match="method must be 'homology'"):
        sb.screen_sequence(CONCERN_PROTEIN, databases=[concern_db], method="hmm")


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

@needs_aligner
def test_report_records_its_provenance(concern_db):
    rep = sb.screen_sequence(CONCERN_PROTEIN, databases=[concern_db])
    assert rep.databases == ["concern.faa"]
    assert rep.thresholds["min_identity"] > 0
    assert rep.thresholds["flag_identity"] > 0
    assert rep.method == "homology"


@needs_aligner
def test_report_always_states_the_limits_of_a_homology_screen(concern_db):
    """The caveat is part of the output, not just the docs — a report gets
    forwarded, the docstring does not."""
    rep = sb.screen_sequence(CONCERN_PROTEIN, databases=[concern_db])
    assert any("必要条件而非充分条件" in n for n in rep.notes)


@needs_aligner
def test_summary_is_human_readable(concern_db):
    rep = sb.screen_sequence(CONCERN_PROTEIN, databases=[concern_db])
    text = rep.summary()
    assert "decision:" in text and "FLAG" in text
    assert "concern.faa" in text


def test_empty_report_frame_has_the_right_columns():
    rep = sb.screen_sequence(CONCERN_PROTEIN, databases=[],
                             require_database=False)
    df = rep.to_frame()
    assert "downgraded" in df.columns and len(df) == 0


@needs_aligner
def test_env_var_supplies_the_database(concern_db, monkeypatch):
    monkeypatch.setenv("OMICOS_BIOSECURITY_DB", concern_db)
    rep = sb.screen_sequence(CONCERN_PROTEIN)
    assert rep.databases == ["concern.faa"]
    assert rep.decision == "flag"


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

@needs_aligner
def test_plot_screening(concern_db):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    rep = sb.screen_sequence(CONCERN_PROTEIN, databases=[concern_db])
    fig, axes = sb.plot_screening(rep)
    assert len(axes) == 2
    plt.close(fig)


def test_plot_handles_an_empty_report():
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    rep = sb.screen_sequence(CONCERN_PROTEIN, databases=[],
                             require_database=False)
    fig, axes = sb.plot_screening(rep)
    assert len(axes) == 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# in-memory references, and the reverse-complement helper the six-frame claim
# rests on
# ---------------------------------------------------------------------------

@needs_aligner
def test_an_in_memory_reference_works_like_a_fasta():
    """A reference is not always a file on disk — it may have come back from a
    registry query or a database cursor. Requiring a path forced callers to
    write a temporary FASTA themselves."""
    rep = sb.screen_sequence(CONCERN_PROTEIN,
                             databases={"reference_entry_01": CONCERN_PROTEIN})
    assert rep.decision == "flag"
    assert rep.hits and rep.hits[0].subject == "reference_entry_01"


@needs_aligner
def test_in_memory_and_file_references_agree(concern_db):
    from_file = sb.screen_sequence(CONCERN_PROTEIN, databases=[concern_db])
    from_memory = sb.screen_sequence(
        CONCERN_PROTEIN, databases={"stand_in_concern_01": CONCERN_PROTEIN})
    assert from_file.decision == from_memory.decision
    assert len(from_file.hits) == len(from_memory.hits)


def test_an_empty_in_memory_entry_is_rejected():
    with pytest.raises(ValueError, match="是空的"):
        sb.screen_sequence(CONCERN_PROTEIN, databases={"bad": ""})


def test_reverse_complement_is_public():
    """It is the operation that makes the six-frame requirement concrete, and
    the tutorial needs it to show that flipping a construct does not evade the
    screen."""
    assert sb.reverse_complement("ATGGCT") == "AGCCAT"
    assert sb.reverse_complement("augGCu") == "AGCCAT"


def test_reverse_complement_is_an_involution():
    seq = "ATGGCTTGTAAACGT"
    assert sb.reverse_complement(sb.reverse_complement(seq)) == seq


def test_reverse_complement_rejects_protein():
    with pytest.raises(ValueError, match="非 DNA 字符"):
        sb.reverse_complement("MKVQ")
