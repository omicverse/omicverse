"""Tests for ``ov.alignment.homology_search`` / ``reciprocal_best_hits``.

The command-construction and parsing tests run everywhere. The end-to-end
tests need DIAMOND or NCBI BLAST+ on PATH and skip cleanly when absent — a
skip is reported, never silently passed.
"""
from __future__ import annotations

import shutil
import textwrap

import pytest

from omicverse.alignment._homology import (
    BLAST_TAB_COLUMNS,
    _align_cmd,
    _PROGRAMS,
    _read_hits,
    homology_search,
    reciprocal_best_hits,
    resolve_method,
)

HAS_DIAMOND = shutil.which("diamond") is not None
HAS_BLAST = shutil.which("blastp") is not None and shutil.which("makeblastdb") is not None

# Real orthologs, trimmed: the pair must come out as a reciprocal best hit.
HUMAN = textwrap.dedent("""\
    >hs_GAPDH
    MGKVKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTVKAENGKLVIN
    GNPITIFQERDPSKIKWGDAGAEYVVESTGVFTTMEKAGAHLQGGAKRVIISAPSADAPMFVMGVNHEKY
    DNSLKIISNASCTTNCLAPLAKVIHDNFGIVEGLMTTVHAITATQKTVDGPSGKLWRDGRGALQNIIPAS
    TGAAKAVGKVIPELNGKLTGMAFRVPTANVSVVDLTCRLEKPAKYDDIKKVVKQASEGPLKGILGYTEHQ
    >hs_TP53
    MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAA
    PPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKT
    CPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRN
    """)
MOUSE = textwrap.dedent("""\
    >mm_Gapdh
    MVKVGVNGFGRIGRLVTRAAFSCGKVDVVAINDPFIDLHYMVYMFQYDSTHGKFNGTVKAENGKLVINGK
    AITIFQERDPANIKWGDAGATYVVESTGVFTTMEKAGAHLKGGAKRVIISAPSADAPMFVMGVNHEKYDN
    SLKIVSNASCTTNCLAPLAKVIHDNFGIVEGLMTTVHAITATQKTVDGPSGKLWRDGRGAAQNIIPASTG
    AAKAVGKVIPELNGKLTGMAFRVPTPNVSVVDLTCRLEKAAKYDDIKKVVKQASEGPLKGILGYTEDQVV
    >mm_Trp53
    MTAMEESQSDISLELPLSQETFSGLWKLLPPEDILPSPHCMDDLLLPQDVEEFFEGPSEALRVSGAPAAQ
    DPVTETPGPVAPAPATPWPLSSFVPSQKTYQGNYGFHLGFLQSGTAKSVMCTYSPPLNKLFCQLAKTCPV
    QLWVSATPPAGSRVRAMAIYKKSQHMTEVVRRCPHHERCSDGDGLAPPQHLIRVEGNLYPEYLEDRQTFR
    """)


@pytest.fixture
def fastas(tmp_path):
    a, b = tmp_path / "human.faa", tmp_path / "mouse.faa"
    a.write_text(HUMAN)
    b.write_text(MOUSE)
    return str(a), str(b)


# ── method resolution ────────────────────────────────────────────────────────

def test_auto_prefers_diamond_when_present(monkeypatch):
    monkeypatch.setattr("omicverse.alignment._homology._which",
                        lambda name, env=None: "/usr/bin/diamond" if name == "diamond" else None)
    assert resolve_method("auto") == "diamond"


def test_auto_falls_back_to_blastp_without_diamond(monkeypatch):
    monkeypatch.setattr("omicverse.alignment._homology._which", lambda name, env=None: None)
    assert resolve_method("auto") == "blastp"


@pytest.mark.parametrize("method", sorted(_PROGRAMS))
def test_every_advertised_method_resolves(method):
    """`method` is the user-facing knob — every documented value must work."""
    assert resolve_method(method) == method


def test_unknown_method_is_rejected_with_the_valid_list():
    with pytest.raises(ValueError, match="unknown method"):
        resolve_method("bowtie2")


def test_nucleotide_programs_use_blast_and_a_nucl_db():
    """DIAMOND has no nucleotide-database mode; those must route to BLAST+."""
    for m in ("blastn", "tblastn", "tblastx"):
        assert _PROGRAMS[m]["engine"] == "blast"
    assert _PROGRAMS["blastn"]["db"] == "nucl"
    assert _PROGRAMS["diamond-blastx"]["engine"] == "diamond"
    assert _PROGRAMS["diamond-blastx"]["db"] == "prot"


# ── command construction (no binaries needed) ────────────────────────────────

def _cmd(method, **kw):
    spec = _PROGRAMS[method]
    exe = {"diamond": "diamond", "makeblastdb": "makeblastdb",
           spec["program"]: spec["program"]}
    kw.setdefault("query", "q.faa"); kw.setdefault("db_prefix", "db")
    kw.setdefault("out_path", "o.tsv"); kw.setdefault("evalue", 1e-5)
    kw.setdefault("max_target_seqs", 10); kw.setdefault("threads", 4)
    kw.setdefault("sensitivity", "sensitive"); kw.setdefault("extra_args", None)
    return _align_cmd(spec, exe=exe, **kw)


def test_diamond_command_shape():
    cmd = _cmd("diamond")
    assert cmd[:2] == ["diamond", "blastp"]
    assert "--outfmt" in cmd and cmd[cmd.index("--outfmt") + 1] == "6"
    assert "--sensitive" in cmd            # sensitivity becomes a flag
    assert "--quiet" in cmd


def test_diamond_fast_is_not_passed_as_a_flag():
    """`fast` is DIAMOND's default; `--fast` would be rejected by older builds."""
    assert "--fast" not in _cmd("diamond", sensitivity="fast")


def test_blast_command_shape_and_ignores_sensitivity():
    cmd = _cmd("blastp", sensitivity="ultra-sensitive")
    assert cmd[0] == "blastp"
    assert "-outfmt" in cmd and cmd[cmd.index("-outfmt") + 1] == "6"
    assert "-max_hsps" in cmd
    assert not any(str(a).startswith("--ultra") for a in cmd)


def test_bad_sensitivity_is_rejected():
    with pytest.raises(ValueError, match="sensitivity"):
        _cmd("diamond", sensitivity="turbo")


def test_extra_args_are_appended():
    assert _cmd("blastp", extra_args=["-seg", "no"])[-2:] == ["-seg", "no"]


# ── parsing ─────────────────────────────────────────────────────────────────

def test_empty_result_yields_an_empty_frame_with_the_right_columns(tmp_path):
    p = tmp_path / "empty.tsv"; p.write_text("")
    df = _read_hits(str(p))
    assert len(df) == 0 and list(df.columns) == BLAST_TAB_COLUMNS


def test_outfmt6_is_parsed_and_typed(tmp_path):
    p = tmp_path / "hits.tsv"
    p.write_text("qA\tsB\t93.7\t120\t7\t0\t1\t120\t1\t120\t8.7e-233\t623\n")
    df = _read_hits(str(p))
    assert list(df.columns) == BLAST_TAB_COLUMNS
    assert df.loc[0, "qseqid"] == "qA" and df.loc[0, "sseqid"] == "sB"
    assert df.loc[0, "pident"] == pytest.approx(93.7)
    assert df.loc[0, "evalue"] == pytest.approx(8.7e-233)
    assert float(df.loc[0, "bitscore"]) == 623.0


def test_missing_input_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="query FASTA"):
        homology_search(str(tmp_path / "nope.faa"), str(tmp_path / "also_nope.faa"))


# ── end-to-end (needs the real aligners) ────────────────────────────────────

@pytest.mark.skipif(not (HAS_DIAMOND or HAS_BLAST), reason="no aligner on PATH")
def test_end_to_end_finds_the_orthologs(fastas):
    human, mouse = fastas
    hits = homology_search(human, mouse, threads=2)
    assert len(hits), "GAPDH/TP53 must hit their mouse orthologs"
    best = hits.groupby("qseqid").head(1).set_index("qseqid")["sseqid"].to_dict()
    assert best["hs_GAPDH"] == "mm_Gapdh"
    assert best["hs_TP53"] == "mm_Trp53"
    assert hits.attrs["engine"] in {"diamond", "blast"}


@pytest.mark.skipif(not (HAS_DIAMOND and HAS_BLAST),
                    reason="needs BOTH engines to cross-check")
def test_both_engines_agree(fastas):
    """DIAMOND is an approximation of BLAST — on unambiguous orthologs the two
    must still call the same pairs at a similar identity."""
    human, mouse = fastas
    d = homology_search(human, mouse, method="diamond", threads=2).groupby("qseqid").head(1)
    b = homology_search(human, mouse, method="blastp", threads=2).groupby("qseqid").head(1)
    dd = d.set_index("qseqid")[["sseqid", "pident"]]
    bb = b.set_index("qseqid")[["sseqid", "pident"]]
    for q in dd.index:
        assert dd.loc[q, "sseqid"] == bb.loc[q, "sseqid"]
        assert abs(float(dd.loc[q, "pident"]) - float(bb.loc[q, "pident"])) < 2.0


@pytest.mark.skipif(not (HAS_DIAMOND or HAS_BLAST), reason="no aligner on PATH")
def test_reciprocal_best_hits_pairs_the_orthologs(fastas):
    human, mouse = fastas
    rbh = reciprocal_best_hits(human, mouse, threads=2)
    pairs = set(zip(rbh["a"], rbh["b"]))
    assert ("hs_GAPDH", "mm_Gapdh") in pairs
    assert ("hs_TP53", "mm_Trp53") in pairs
    # RBH is symmetric by construction — one row per pair, no duplicates.
    assert len(rbh) == len(pairs)


@pytest.mark.skipif(not (HAS_DIAMOND or HAS_BLAST), reason="no aligner on PATH")
def test_out_writes_the_table_the_caller_asked_for(fastas, tmp_path):
    human, mouse = fastas
    out = tmp_path / "nested" / "hits.tsv"
    hits = homology_search(human, mouse, out=str(out), threads=2)
    assert out.exists() and out.stat().st_size > 0
    assert len(_read_hits(str(out))) == len(hits)


@pytest.mark.skipif(not (HAS_DIAMOND or HAS_BLAST), reason="no aligner on PATH")
def test_concurrent_searches_into_one_directory_do_not_collide(fastas, tmp_path):
    """Regression: the search DB must never live beside `out`.

    A reciprocal run fires both directions concurrently with `out=` pointing
    into the SAME directory. With a fixed db name next to `out`, the two runs
    clobbered each other's database and one direction silently searched the
    WRONG subject — it reported mouse-vs-mouse at 100% identity.
    """
    from concurrent.futures import ThreadPoolExecutor

    human, mouse = fastas
    shared = tmp_path / "maps"
    shared.mkdir()

    def run(pair):
        q, s, name = pair
        return homology_search(q, s, out=str(shared / name), threads=2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fwd, rev = list(pool.map(run, [(human, mouse, "h_to_m.txt"),
                                       (mouse, human, "m_to_h.txt")]))

    # Every hit must be cross-species: a query never matches its own file.
    assert len(fwd) and len(rev)
    assert all(q.startswith("hs_") and s.startswith("mm_")
               for q, s in zip(fwd["qseqid"], fwd["sseqid"])), fwd.head().to_dict()
    assert all(q.startswith("mm_") and s.startswith("hs_")
               for q, s in zip(rev["qseqid"], rev["sseqid"])), rev.head().to_dict()
