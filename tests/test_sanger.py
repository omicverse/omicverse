"""Tests for ``ov.alignment.sanger``.

The parsing/verdict logic runs everywhere. Anything that actually reads a
chromatogram needs `tracy` on PATH and skips cleanly when absent.
"""
from __future__ import annotations

import os
import shutil

import pytest

from omicverse.alignment.sanger import (
    SangerTrace, _trace_from_tracy_json, read_ab1, align_to_reference,
    verify_construct,
)

HAS_TRACY = shutil.which("tracy") is not None
AB1 = "/tmp/sanger_test/real.ab1"
HAS_AB1 = os.path.exists(AB1)
needs_trace = pytest.mark.skipif(not (HAS_TRACY and HAS_AB1),
                                 reason="needs tracy + a real .ab1")


# ── pure logic ──────────────────────────────────────────────────────────────

def test_tracy_json_maps_onto_the_dataclass():
    """Keys pinned against tracy 0.8.1 — a rename upstream must fail loudly."""
    doc = {"primarySeq": "ACGT", "secondarySeq": "ACNT",
           "basecallQual": [40, 41, 10, 39], "basecallPos": [1, 2, 3, 4],
           "peakA": [1.0, 2.0], "peakC": [0.0, 1.0], "peakG": [0.0, 0.0],
           "peakT": [0.0, 0.0]}
    tr = _trace_from_tracy_json("x.ab1", doc)
    assert tr.sequence == "ACGT" and tr.secondary == "ACNT"
    assert tr.quality == [40, 41, 10, 39]
    assert sorted(tr.peaks) == ["A", "C", "G", "T"] and len(tr.peaks["A"]) == 2
    assert len(tr) == 4


def test_mixed_positions_ignores_N_and_reports_real_disagreement():
    assert SangerTrace("x", sequence="ACGT", secondary="ACNT").mixed_positions == []
    assert SangerTrace("x", sequence="ACGT", secondary="AGGT").mixed_positions == [1]
    assert SangerTrace("x", sequence="ACGT", secondary="").mixed_positions == []


def test_to_fasta_wraps_and_names(tmp_path):
    p = tmp_path / "o.fa"
    SangerTrace("clone_A.ab1", sequence="A" * 130).to_fasta(str(p))
    lines = p.read_text().splitlines()
    assert lines[0] == ">clone_A"
    assert all(len(l) <= 60 for l in lines[1:])
    assert "".join(lines[1:]) == "A" * 130


def test_missing_files_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="trace not found"):
        read_ab1(str(tmp_path / "nope.ab1"))
    with pytest.raises(ValueError, match="engine"):
        read_ab1(__file__, engine="guess")


# ── end-to-end ──────────────────────────────────────────────────────────────

@needs_trace
def test_read_ab1_returns_four_channels_and_a_sequence():
    tr = read_ab1(AB1)
    assert len(tr.sequence) > 100
    assert sorted(tr.peaks) == ["A", "C", "G", "T"]
    assert len(tr.peaks["A"]) > len(tr.sequence)      # samples > bases
    assert len(tr.quality) == len(tr.sequence)
    assert len(tr.basecall_pos) == len(tr.sequence)


@needs_trace
def test_end_decay_does_not_fail_a_correct_clone(tmp_path):
    """The regression that motivated `identity_core`.

    A real trace against its own basecall is a PERFECT clone, yet the last ~50
    bases are all gaps — whole-read identity lands around 0.96 and a
    whole-read verdict would reject a construct that is exactly right.
    """
    tr = read_ab1(AB1)
    ref = tmp_path / "ref.fa"
    tr.to_fasta(str(ref), name="ref")

    aln = align_to_reference(AB1, str(ref))
    assert aln["identity"] < 0.99, "expected end decay in a real trace"
    assert aln["identity_core"] == pytest.approx(1.0)
    assert aln["core_differences"] == []
    # every whole-read difference is an END artifact, and a gap rather than a
    # real substitution
    assert all(d["ref"] == "-" or d["alt"] == "-" for d in aln["differences"])

    rep = verify_construct(AB1, str(ref))
    assert rep["ok"] is True


@needs_trace
def test_a_planted_substitution_is_located(tmp_path):
    tr = read_ab1(AB1)
    seq = list(tr.sequence)
    i = len(seq) // 2
    seq[i] = "A" if seq[i] != "A" else "T"
    ref = tmp_path / "mut.fa"
    ref.write_text(">mut\n" + "".join(seq) + "\n")

    rep = verify_construct(AB1, str(ref))
    core = rep["traces"][0]["core_differences"]
    assert len(core) == 1, core
    assert core[0]["pos"] == i
    assert core[0]["alt"] == tr.sequence[i]     # the trace holds the original base

    # A single base still clears the identity threshold — which is exactly why
    # the docstring tells callers to require an empty `core_differences`.
    clean = rep["ok"] and not any(t["core_differences"] for t in rep["traces"])
    assert clean is False


@needs_trace
def test_verify_accepts_a_list_of_traces(tmp_path):
    tr = read_ab1(AB1)
    ref = tmp_path / "ref.fa"
    tr.to_fasta(str(ref), name="ref")
    rep = verify_construct([AB1, AB1], str(ref))
    assert len(rep["traces"]) == 2 and rep["ok"] is True
