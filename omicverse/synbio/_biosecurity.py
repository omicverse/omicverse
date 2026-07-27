r"""Biosecurity screening for designed sequences.

Screening a construct before it is ordered or shared has moved from best
practice toward a hard requirement — the international frameworks (the
NTI/IBBIS *Common Mechanism*, SecureDNA) and a growing number of national rules
expect a provider of design services to have done it. ``ov.synbio`` could design
DNA and had no screen at all.

What this module is
-------------------
A screening **harness**: it translates a construct, aligns it against reference
sets *you* configure, subtracts hits that are equally well explained by benign
housekeeping homology, and produces a reviewable report with provenance.

What it deliberately is not
---------------------------
It ships **no sequences of concern**. Distributing such a set inside a
general-purpose analysis package would be both useless (it would go stale) and
irresponsible. You point it at a reference you are entitled to use — your
institution's list, the Common Mechanism's biorisk database, or a provider's —
via ``databases=`` or ``OMICOS_BIOSECURITY_DB``.

The rule that matters most
--------------------------
With no reference configured, :func:`screen_sequence` **raises**. It does not
return "clean". A screener that reports a pass because it had nothing to compare
against manufactures false assurance, which is worse than no screener at all:
the pass gets recorded, and everyone downstream believes a check happened. Pass
``require_database=False`` only if you have some other reason to want the
harness to run, and note that the report then carries
``decision='not_screened'``.

A homology screen is also necessary rather than sufficient. It cannot see
function that homology does not carry, it inherits every gap in its reference,
and it says nothing about intent or context. :attr:`ScreeningReport.decision` is
``'flag'`` / ``'review'`` / ``'pass'`` — an input to human review, never a
clearance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


_DNA = set("ACGTUN")
_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
_COMP = str.maketrans("ACGTN", "TGCAN")

DECISIONS = ("flag", "review", "pass", "not_screened")


@dataclass
class ScreeningHit:
    """One alignment between the query and a reference of concern."""

    query_frame: str                  # '+1'..'+3', '-1'..'-3', or 'protein'
    query_start: int                  # 1-based, in query residue/nt coordinates
    query_end: int
    subject: str
    database: str
    identity: float
    alignment_length: int
    evalue: float
    bitscore: float
    benign_competitor: Optional[str] = None
    benign_bitscore: float = 0.0

    @property
    def downgraded(self) -> bool:
        """True when a benign reference explains this region at least as well."""
        return self.benign_competitor is not None

    def __repr__(self) -> str:  # pragma: no cover
        tag = " (downgraded)" if self.downgraded else ""
        return (f"ScreeningHit({self.subject!r} id={self.identity:.0f}% "
                f"len={self.alignment_length} E={self.evalue:.1e}{tag})")


@dataclass
class ScreeningReport:
    """The reviewable output of a screen."""

    decision: str = "not_screened"
    hits: List[ScreeningHit] = field(default_factory=list)
    sequence_type: str = ""
    length: int = 0
    databases: List[str] = field(default_factory=list)
    benign_databases: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    method: str = "homology"
    notes: List[str] = field(default_factory=list)

    @property
    def flagged_hits(self) -> List[ScreeningHit]:
        return [h for h in self.hits if not h.downgraded]

    @property
    def clean(self) -> bool:
        """Only ever True after an actual screen against a real reference."""
        return self.decision == "pass"

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        if not self.hits:
            return pd.DataFrame(columns=[
                "frame", "query_start", "query_end", "subject", "database",
                "identity", "alignment_length", "evalue", "bitscore",
                "downgraded", "benign_competitor"])
        return pd.DataFrame([{
            "frame": h.query_frame, "query_start": h.query_start,
            "query_end": h.query_end, "subject": h.subject,
            "database": h.database, "identity": h.identity,
            "alignment_length": h.alignment_length, "evalue": h.evalue,
            "bitscore": h.bitscore, "downgraded": h.downgraded,
            "benign_competitor": h.benign_competitor,
        } for h in self.hits])

    def summary(self) -> str:
        lines = [f"decision: {self.decision.upper()}",
                 f"sequence: {self.sequence_type}, {self.length} units",
                 f"databases: {', '.join(self.databases) or '(none)'}"]
        if self.benign_databases:
            lines.append(f"benign sets: {', '.join(self.benign_databases)}")
        flagged = self.flagged_hits
        lines.append(f"hits: {len(self.hits)} total, {len(flagged)} not explained "
                     f"by benign homology")
        for h in flagged[:10]:
            lines.append(f"  - {h.query_frame} {h.query_start}-{h.query_end} "
                         f"-> {h.subject} ({h.identity:.0f}% id, E={h.evalue:.1e})")
        lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"ScreeningReport({self.decision}, {len(self.flagged_hits)} "
                f"flagged of {len(self.hits)} hits)")


def _looks_like_dna(seq: str) -> bool:
    s = set(seq.upper()) - {"\n", " "}
    return bool(s) and s <= _DNA


def _revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def translate_frames(dna: str) -> Dict[str, str]:
    """All six reading frames, as ``{'+1': protein, ..., '-3': protein}``.

    Screening has to cover every frame: a construct's hazardous content may sit
    on the strand or in the frame you did not annotate, and a screen that only
    reads the annotated ORF is trivially evaded by reversing it.
    """
    dna = "".join(str(dna).split()).upper().replace("U", "T")
    out: Dict[str, str] = {}
    for strand, seq in (("+", dna), ("-", _revcomp(dna))):
        for offset in range(3):
            codons = [seq[i:i + 3] for i in range(offset, len(seq) - 2, 3)]
            out[f"{strand}{offset + 1}"] = "".join(
                _CODON_TABLE.get(c, "X") for c in codons)
    return out


def _configured_databases(databases) -> List[str]:
    if databases is None:
        env = os.environ.get("OMICOS_BIOSECURITY_DB", "")
        paths = [p for p in env.split(os.pathsep) if p]
    elif isinstance(databases, str):
        paths = [databases]
    else:
        paths = list(databases)
    return [os.path.expanduser(p) for p in paths]


@register_function(
    aliases=["screen_sequence", "生物安全筛查", "序列筛查", "biosecurity",
             "biosecurity_screen", "危害筛查", "合规筛查", "SecureDNA",
             "common_mechanism"],
    category="synthetic_biology",
    description="设计序列的生物安全筛查:六框翻译 → 与你配置的关注序列库比对 → 扣除能被良性同源(管家蛋白)同等解释的命中 → 出可复核的报告(decision='flag'/'review'/'pass')。本模块不内置任何危害序列,库由 databases= 或 OMICOS_BIOSECURITY_DB 指定;**未配置库时会报错而不是返回通过** —— 无库而报'干净'制造的是虚假保证。同源筛查是必要而非充分条件,结果是人工复核的输入,不是放行凭证。Biosecurity screening harness for designed sequences.",
    examples=[
        "rep = ov.synbio.screen_sequence(dna, databases=['/db/biorisk.faa'])",
        "rep.decision, rep.flagged_hits, print(rep.summary())",
        "rep = ov.synbio.screen_sequence(protein_seq, databases=[...], benign_databases=['/db/housekeeping.faa'])",
    ],
    related=["synbio.plot_screening", "alignment.homology_search",
             "synbio.annotate_construct"],
    requires={},
    produces={},
)
def screen_sequence(
    sequence: str,
    databases: "Optional[str | Sequence[str]]" = None,
    *,
    benign_databases: "Optional[str | Sequence[str]]" = None,
    method: str = "homology",
    min_identity: float = 40.0,
    min_alignment_length: int = 50,
    evalue: float = 1e-5,
    flag_identity: float = 70.0,
    threads: int = 4,
    require_database: bool = True,
    sequence_type: str = "auto",
) -> ScreeningReport:
    """Screen a designed sequence and return a reviewable report.

    Parameters
    ----------
    sequence
        DNA or protein. ``sequence_type='auto'`` decides by alphabet; DNA is
        translated in all six frames before alignment.
    databases
        FASTA path(s) of sequences of concern. Defaults to
        ``OMICOS_BIOSECURITY_DB`` (``os.pathsep``-separated). **No such set is
        bundled.**
    benign_databases
        FASTA path(s) of sequences that are *not* of concern — housekeeping
        proteins, common expression tags, the host proteome. A hit whose region
        is explained at least as well by a benign reference is marked
        ``downgraded`` rather than dropped, so the reviewer can still see it.
        This is the Common Mechanism's central idea: without it, conserved
        domains shared by hazardous and innocuous proteins alike bury the report
        in noise and the screen stops being read.
    method
        ``'homology'`` — protein-level alignment via
        :func:`omicverse.alignment.homology_search` (DIAMOND or BLAST+).
    min_identity, min_alignment_length, evalue
        Reporting thresholds. Deliberately permissive: a screen should
        over-report to a human, not under-report.
    flag_identity
        At or above this identity an undowngraded hit makes the decision
        ``'flag'``; weaker hits give ``'review'``.
    threads
        Passed to the aligner.
    require_database
        When True (default) and no database is configured, raise. See the module
        docstring — this is the point.
    sequence_type
        ``'auto'`` | ``'dna'`` | ``'protein'``.

    Returns
    -------
    ScreeningReport
    """
    if method != "homology":
        raise ValueError(f"method must be 'homology', got {method!r}")
    if sequence_type not in ("auto", "dna", "protein"):
        raise ValueError(
            f"sequence_type must be one of ['auto', 'dna', 'protein'], "
            f"got {sequence_type!r}")

    seq = "".join(str(sequence).split()).upper()
    if not seq:
        raise ValueError("ov.synbio.screen_sequence: 序列为空。")

    dbs = _configured_databases(databases)
    benign = _configured_databases(benign_databases) if benign_databases else []

    if not dbs:
        if require_database:
            raise ValueError(
                "没有配置任何关注序列库,筛查无法进行。\n"
                "本模块**不内置**危害序列 —— 请用 databases=['/path/to/ref.faa'] "
                "或环境变量 OMICOS_BIOSECURITY_DB 指向你有权使用的参考库"
                "(例如机构清单、NTI/IBBIS Common Mechanism 的 biorisk 库,"
                "或服务商提供的库)。\n"
                "这里故意报错而不是返回 'pass':没有参考库却报告'干净',"
                "产生的是被记录下来的虚假保证,比不筛查更糟。"
                "确实只想跑通流程时传 require_database=False,"
                "报告会标成 decision='not_screened'。")
        return ScreeningReport(
            decision="not_screened", sequence_type=sequence_type, length=len(seq),
            method=method,
            notes=["没有配置参考库 —— 这不是一次筛查,不能作为放行依据。"])

    missing = [p for p in dbs + benign if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"参考库文件不存在:{missing}")

    if sequence_type == "auto":
        sequence_type = "dna" if _looks_like_dna(seq) else "protein"

    if sequence_type == "dna":
        queries = translate_frames(seq)
        unit_len = len(seq)
    else:
        queries = {"protein": seq}
        unit_len = len(seq)

    import tempfile
    from ..alignment import homology_search

    hits: List[ScreeningHit] = []
    notes: List[str] = []

    with tempfile.TemporaryDirectory(prefix="ov_screen_") as tmp:
        query_fa = os.path.join(tmp, "query.faa")
        with open(query_fa, "w", encoding="utf-8") as fh:
            for frame, prot in queries.items():
                # split on stops: an alignment cannot run through a stop codon,
                # and keeping the '*' in confuses the aligners' alphabets
                for i, seg in enumerate(prot.split("*")):
                    if len(seg) >= min_alignment_length // 2:
                        fh.write(f">{frame}|seg{i}|off{prot.find(seg)}\n{seg}\n")

        if os.path.getsize(query_fa) == 0:
            notes.append("六框翻译后没有足够长的无终止子片段可比对。")

        for db in dbs:
            try:
                table = homology_search(query_fa, db, evalue=evalue,
                                        threads=threads, max_target_seqs=5)
            except Exception as exc:
                raise RuntimeError(
                    f"比对 {db!r} 失败({type(exc).__name__}: {exc})。"
                    f"同源筛查需要 DIAMOND 或 NCBI BLAST+ 在 PATH 上"
                    f"(conda install -c bioconda diamond)。") from exc
            for _, row in table.iterrows():
                ident = float(row.get("pident", 0.0) or 0.0)
                alen = int(row.get("length", 0) or 0)
                if ident < min_identity or alen < min_alignment_length:
                    continue
                # BLAST tabular outfmt-6 names these qseqid/sseqid, not
                # query/subject — see alignment.BLAST_TAB_COLUMNS. Guessing wrong
                # here silently produced empty frame and subject labels while the
                # decision logic still worked, so a report looked fine and named
                # nothing.
                qid = str(row.get("qseqid", ""))
                frame = qid.split("|")[0] if "|" in qid else qid
                hits.append(ScreeningHit(
                    query_frame=frame,
                    query_start=int(row.get("qstart", 0) or 0),
                    query_end=int(row.get("qend", 0) or 0),
                    subject=str(row.get("sseqid", "")),
                    database=os.path.basename(db),
                    identity=ident, alignment_length=alen,
                    evalue=float(row.get("evalue", 1.0) or 1.0),
                    bitscore=float(row.get("bitscore", 0.0) or 0.0),
                ))

        # benign subtraction
        if benign and hits:
            best_benign: Dict[str, Tuple[str, float]] = {}
            for db in benign:
                try:
                    table = homology_search(query_fa, db, evalue=evalue,
                                            threads=threads, max_target_seqs=5)
                except Exception:
                    notes.append(f"良性库 {os.path.basename(db)} 比对失败,未做扣除。")
                    continue
                for _, row in table.iterrows():
                    qid = str(row.get("qseqid", ""))
                    frame = qid.split("|")[0] if "|" in qid else qid
                    bs = float(row.get("bitscore", 0.0) or 0.0)
                    prev = best_benign.get(frame)
                    if prev is None or bs > prev[1]:
                        best_benign[frame] = (str(row.get("sseqid", "")), bs)
            for h in hits:
                cand = best_benign.get(h.query_frame)
                if cand and cand[1] >= h.bitscore:
                    h.benign_competitor, h.benign_bitscore = cand

    flagged = [h for h in hits if not h.downgraded]
    if not flagged:
        decision = "pass"
    elif max(h.identity for h in flagged) >= flag_identity:
        decision = "flag"
    else:
        decision = "review"

    notes.append(
        "同源筛查是必要条件而非充分条件:它看不到同源性不携带的功能,"
        "继承参考库的所有缺口,也无法判断意图与语境。此结果是人工复核的输入。")

    return ScreeningReport(
        decision=decision, hits=hits, sequence_type=sequence_type,
        length=unit_len, databases=[os.path.basename(p) for p in dbs],
        benign_databases=[os.path.basename(p) for p in benign],
        thresholds={"min_identity": min_identity,
                    "min_alignment_length": float(min_alignment_length),
                    "evalue": evalue, "flag_identity": flag_identity},
        method=method, notes=notes,
    )


@register_function(
    aliases=["plot_screening", "筛查报告图", "生物安全图", "plot_biosecurity",
             "筛查可视化"],
    category="synthetic_biology",
    description="画生物安全筛查报告:命中沿序列的位置(按框分层,被良性同源扣除的用灰色)、身份一致度分布与判定阈值。Plot a biosecurity screening report.",
    examples=["ov.synbio.plot_screening(rep)"],
    related=["synbio.screen_sequence"],
    requires={},
    produces={},
)
def plot_screening(report: ScreeningReport, axes=None):
    """Two panels: hits along the query by frame, and identity vs the flag line."""
    from ._plot import _mpl
    plt = _mpl()

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8),
                                 gridspec_kw={"width_ratios": [2, 1]})
    else:
        fig = axes[0].figure
    axes = list(axes)

    frames = sorted({h.query_frame for h in report.hits})
    ax = axes[0]
    for h in report.hits:
        y = frames.index(h.query_frame)
        ax.plot([h.query_start, h.query_end], [y, y], lw=7,
                solid_capstyle="butt",
                color="#B0B0B0" if h.downgraded else
                ("#E41A1C" if h.identity >= report.thresholds.get("flag_identity", 70)
                 else "#FF7F00"),
                alpha=0.9)
    ax.set_yticks(range(len(frames)))
    ax.set_yticklabels(frames, fontsize=8)
    ax.set_xlabel("position in translated frame")
    ax.set_title(f"decision: {report.decision.upper()}  "
                 f"(grey = explained by benign homology)")
    if not report.hits:
        ax.text(0.5, 0.5, "no hits above threshold", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)

    ax = axes[1]
    idents = [h.identity for h in report.hits]
    if idents:
        ax.hist(idents, bins=10, color="#377EB8", alpha=0.85)
    flag = report.thresholds.get("flag_identity")
    if flag:
        ax.axvline(flag, ls="--", c="#E41A1C", lw=1.2, label=f"flag ≥ {flag:.0f}%")
        ax.legend(fontsize=7)
    ax.set_xlabel("identity to reference (%)")
    ax.set_ylabel("hits")
    ax.set_title("hit strength")

    fig.tight_layout()
    return fig, axes


__all__ = ["screen_sequence", "ScreeningReport", "ScreeningHit",
           "translate_frames", "plot_screening", "DECISIONS"]
