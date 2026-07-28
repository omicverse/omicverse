r"""Codon harmonisation, and whether a designed sequence can actually be made.

Two gaps at the DNA layer that ``codon_optimize`` does not cover.

**Harmonisation is not optimisation.** ``codon_optimize`` maximises CAI: it
replaces every codon with the host's fastest one. That is the right move for a
small, fast-folding protein where yield is the only concern. For a difficult
eukaryotic protein it is often exactly wrong — translation speed is part of the
folding programme, and native mRNAs use rare codons at domain boundaries to
pause the ribosome while the previous domain folds. Flattening those pauses
gives you more protein and less *folded* protein.

:func:`codon_harmonize` preserves the rhythm instead: for each codon it finds
that codon's usage *rank within its synonymous family* in the source organism,
and picks the codon holding the same relative rank in the host. A rare codon
stays rare; a common one stays common. The two strategies pull in opposite
directions, and :func:`compare_codon_strategies` shows by how much.

**Synthesis difficulty.** A designed sequence can be perfectly optimal and still
be un-orderable. :func:`synthesis_complexity` scores the features that make
commercial synthesis fail or cost more — GC extremes, long homopolymers, direct
and inverted repeats, hairpins — and reports them as located problems so they
can be fixed rather than just priced.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


_COMP = str.maketrans("ACGTN", "TGCAN")

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

_SYNONYMS: Dict[str, List[str]] = defaultdict(list)
for _c, _a in _CODON_TABLE.items():
    _SYNONYMS[_a].append(_c)

HOSTS = ("e_coli", "s_cerevisiae", "h_sapiens", "b_subtilis", "c_glutamicum",
         "p_pastoris", "cho")

#: The keys of a codon-usage table that are actually amino acids. Anything else
#: in there was put there by another library (see :func:`codon_usage`).
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY*")


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _clean_dna(seq: str, fn: str) -> str:
    s = "".join(str(seq).split()).upper().replace("U", "T")
    if not s:
        raise ValueError(f"ov.synbio.{fn}: 序列为空。")
    bad = sorted(set(s) - set("ACGTN"))
    if bad:
        raise ValueError(f"ov.synbio.{fn}: 序列含非 DNA 字符 {bad}。")
    return s


def _check_hosts(*hosts: str) -> None:
    """Reject unknown hosts without needing the codon tables to be installed.

    Argument validation must not depend on an optional package: a typo in a host
    name should say so, not report a missing dependency.
    """
    for host in hosts:
        if host not in HOSTS:
            raise ValueError(f"host must be one of {list(HOSTS)}, got {host!r}")


def codon_usage(host: str) -> Dict[str, float]:
    """Relative synonymous codon usage for ``host``, ``{codon: frequency}``.

    Uses ``python_codon_tables`` when available (the same source
    ``codon_optimize`` builds on, so the two agree); otherwise raises with the
    install line. Frequencies are within each amino acid's synonymous family and
    sum to 1 per family.
    """
    if host not in HOSTS:
        raise ValueError(f"host must be one of {list(HOSTS)}, got {host!r}")
    try:
        import python_codon_tables as pct
    except ImportError as exc:
        raise ImportError(
            "密码子使用表需要 python_codon_tables(随 omicverse[synbio] 的 "
            "dnachisel 一起安装)。请 pip install python_codon_tables。") from exc

    table = pct.get_codons_table(host)

    # Read only the amino-acid entries. `python_codon_tables` caches one dict per
    # species and hands out that same object, and DNAchisel writes its own
    # `log_best_frequencies` / `log_codons_frequencies` into it as a side effect
    # of codon_optimize(). Iterating every top-level key therefore picked up
    # `log_codons_frequencies` — itself keyed by codon — and overwrote all 64
    # frequencies with log values, which are negative. Every CAI computed after
    # any call to codon_optimize came back 0, in the same process only, so it
    # looked like test-order flakiness rather than shared-state corruption.
    out: Dict[str, float] = {}
    for aa, codons in table.items():
        if aa not in _AMINO_ACIDS:
            continue
        if not hasattr(codons, "items"):
            continue
        for codon, freq in codons.items():
            key = str(codon).upper().replace("U", "T")
            if len(key) == 3 and set(key) <= set("ACGT"):
                out[key] = float(freq)
    if not out:
        raise RuntimeError(
            f"python_codon_tables 没有给出 {host!r} 的可用密码子频率 —— "
            f"表的结构可能变了。顶层键:{sorted(table)[:8]}")
    return out


def _family_ranks(usage: Dict[str, float]) -> Dict[str, Dict[str, int]]:
    """``{amino_acid: {codon: rank}}`` — rank 0 is the most-used synonym."""
    ranks: Dict[str, Dict[str, int]] = {}
    for aa, codons in _SYNONYMS.items():
        ordered = sorted(codons, key=lambda c: -usage.get(c, 0.0))
        ranks[aa] = {c: i for i, c in enumerate(ordered)}
    return ranks


@dataclass
class HarmonizationResult:
    """A harmonised CDS and how far it moved."""

    sequence: str
    source_host: str
    target_host: str
    n_changed: int = 0
    #: Pearson r of ordinal rank, source against harmonised. **Structurally 1.0**
    #: whenever both hosts use the same genetic code, because the algorithm picks
    #: the codon *at* the source rank — so it is a description of the method, not
    #: a check on it. Use :attr:`frequency_correlation`.
    rank_correlation: float = 0.0
    #: Pearson r of *relative synonymous frequency*, source against harmonised —
    #: the quantity Angov harmonisation is actually meant to preserve, and the one
    #: that can fail.
    frequency_correlation: float = 0.0
    #: Largest absolute shift in relative synonymous frequency at any codon.
    max_frequency_shift: float = 0.0
    #: Codons that are common in the source but rare in the target, as
    #: ``(position, source_codon, target_codon, target_frequency)``. These are the
    #: translational pauses harmonisation is supposed to avoid creating.
    rare_codons_introduced: List[Tuple[int, str, str, float]] = field(
        default_factory=list)
    source_cai: float = 0.0
    harmonized_cai: float = 0.0
    optimized_cai: float = 0.0
    per_codon: List[Tuple[int, str, str, int, int]] = field(default_factory=list)

    @property
    def protein_unchanged(self) -> bool:
        return True   # enforced by construction; see codon_harmonize

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame(
            self.per_codon,
            columns=["position", "source_codon", "target_codon",
                     "source_rank", "target_rank"]).set_index("position")

    def __repr__(self) -> str:  # pragma: no cover
        rare = (f", {len(self.rare_codons_introduced)} rare codons introduced"
                if self.rare_codons_introduced else "")
        return (f"HarmonizationResult({self.source_host}->{self.target_host}, "
                f"{self.n_changed} codons changed, frequency r="
                f"{self.frequency_correlation:.3f}{rare}, "
                f"CAI in target {self.harmonized_cai:.3f} "
                f"(optimise would give {self.optimized_cai:.3f}))")


@register_function(
    aliases=["codon_harmonize", "密码子和谐化", "harmonization", "harmonise",
             "codon_harmonization", "和谐化", "共翻译折叠"],
    category="synthetic_biology",
    description="密码子和谐化:保留源宿主的翻译节奏,而不是一味优化 CAI。每个密码子在其同义家族里的**使用频率排名**被映射到目标宿主的同名次密码子 —— 稀有仍稀有,常用仍常用。对难折叠的真核蛋白,这与单纯优化 CAI 是相反的策略:天然 mRNA 用稀有密码子在结构域边界暂停核糖体,抹平这些停顿会得到更多蛋白但更少**正确折叠**的蛋白。Codon harmonisation — preserve translational rhythm across hosts.",
    examples=[
        "res = ov.synbio.codon_harmonize(cds, source_host='h_sapiens', target_host='e_coli')",
        "res.sequence, res.rank_correlation, res.harmonized_cai",
        "df = ov.synbio.compare_codon_strategies(cds, 'h_sapiens', 'e_coli')",
    ],
    related=["synbio.codon_optimize", "synbio.compare_codon_strategies",
             "synbio.synthesis_complexity", "synbio.cai"],
    requires={},
    produces={},
)
def codon_harmonize(
    cds: str,
    source_host: str = "h_sapiens",
    target_host: str = "e_coli",
) -> HarmonizationResult:
    """Rewrite ``cds`` for ``target_host`` keeping each codon's relative rarity.

    Parameters
    ----------
    cds
        Coding DNA, length a multiple of 3.
    source_host, target_host
        Where the gene comes from and where it is going.

    Returns
    -------
    HarmonizationResult

    Notes
    -----
    The protein is preserved exactly: replacements are only ever drawn from the
    same synonymous family, so the translation is identical by construction.
    """
    seq = _clean_dna(cds, "codon_harmonize")
    if len(seq) % 3:
        raise ValueError(
            f"CDS 长度 {len(seq)} 不是 3 的倍数,无法按密码子处理。")
    # Validate *both* hosts before touching the codon tables. Deferring to
    # codon_usage checked the source host first, so a bad target host was only
    # reported after python_codon_tables had been imported — on a bare install
    # the caller got "install python_codon_tables" for what was actually a typo
    # in their own argument.
    _check_hosts(source_host, target_host)

    src_usage = codon_usage(source_host)
    tgt_usage = codon_usage(target_host)
    src_ranks = _family_ranks(src_usage)
    tgt_ranks = _family_ranks(tgt_usage)

    out_codons: List[str] = []
    per_codon: List[Tuple[int, str, str, int, int]] = []
    changed = 0
    src_r: List[int] = []
    tgt_r: List[int] = []

    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3]
        aa = _CODON_TABLE.get(codon)
        if aa is None:
            out_codons.append(codon)     # ambiguous (contains N) — leave alone
            continue
        rank = src_ranks[aa].get(codon, 0)
        family = sorted(_SYNONYMS[aa], key=lambda c: tgt_ranks[aa][c])
        pick = family[min(rank, len(family) - 1)]
        out_codons.append(pick)
        src_r.append(rank)
        tgt_r.append(tgt_ranks[aa][pick])
        if pick != codon:
            changed += 1
        per_codon.append((i // 3 + 1, codon, pick, rank, tgt_ranks[aa][pick]))

    harmonised = "".join(out_codons)

    # how well the rarity profile survived the transfer.
    # NOTE: the *rank* correlation below is 1.0 by construction — `pick` is the
    # codon at the source rank, and synonymous families are the same size in both
    # hosts because they share the genetic code. It is reported for continuity but
    # it cannot fail, so the frequency correlation is what actually tells you
    # whether the transfer worked.
    corr = 1.0
    if len(src_r) > 2:
        import numpy as np
        if np.std(src_r) > 0 and np.std(tgt_r) > 0:
            corr = float(np.corrcoef(src_r, tgt_r)[0, 1])

    import numpy as np
    src_freq: List[float] = []
    tgt_freq: List[float] = []
    rare: List[Tuple[int, str, str, float]] = []
    for position, source_codon, target_codon, _sr, _tr in per_codon:
        aa = _CODON_TABLE[source_codon]
        family = _SYNONYMS[aa]
        s_total = sum(src_usage.get(c, 0.0) for c in family) or 1.0
        t_total = sum(tgt_usage.get(c, 0.0) for c in family) or 1.0
        fs = src_usage.get(source_codon, 0.0) / s_total
        ft = tgt_usage.get(target_codon, 0.0) / t_total
        src_freq.append(fs)
        tgt_freq.append(ft)
        # A codon that is genuinely rare in the host is a translational pause
        # wherever it lands. CGA and AGG/AGA for arginine are the classic cases —
        # the ones BL21-CodonPlus / Rosetta strains exist to rescue — and
        # rank-mapping will happily place one when the source codon sat at the
        # same ordinal rank. Flagged on the *target* frequency alone, because
        # that is what stalls the ribosome regardless of where it came from.
        if ft < 0.10:
            rare.append((position, source_codon, target_codon, float(ft)))
    freq_corr = 0.0
    max_shift = 0.0
    if len(src_freq) > 2 and np.std(src_freq) > 0 and np.std(tgt_freq) > 0:
        freq_corr = float(np.corrcoef(src_freq, tgt_freq)[0, 1])
        max_shift = float(np.max(np.abs(np.asarray(src_freq) - np.asarray(tgt_freq))))

    return HarmonizationResult(
        sequence=harmonised, source_host=source_host, target_host=target_host,
        n_changed=changed, rank_correlation=corr,
        frequency_correlation=freq_corr, max_frequency_shift=max_shift,
        rare_codons_introduced=rare,
        source_cai=_cai(seq, src_usage),
        harmonized_cai=_cai(harmonised, tgt_usage),
        optimized_cai=_cai(_max_cai(seq, tgt_ranks), tgt_usage),
        per_codon=per_codon,
    )


def _max_cai(seq: str, tgt_ranks) -> str:
    """The CAI-optimal rewrite — what ``codon_optimize`` aims at."""
    out = []
    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3]
        aa = _CODON_TABLE.get(codon)
        if aa is None:
            out.append(codon)
            continue
        best = min(_SYNONYMS[aa], key=lambda c: tgt_ranks[aa][c])
        out.append(best)
    return "".join(out)


def _cai(seq: str, usage: Dict[str, float]) -> float:
    """Codon Adaptation Index — geometric mean of relative adaptiveness."""
    import math
    best: Dict[str, float] = {}
    for aa, codons in _SYNONYMS.items():
        best[aa] = max((usage.get(c, 0.0) for c in codons), default=0.0)
    logs = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i + 3]
        aa = _CODON_TABLE.get(codon)
        if aa is None or aa in ("*", "M", "W"):
            continue                    # single-codon families carry no signal
        w = usage.get(codon, 0.0)
        b = best.get(aa, 0.0)
        if w > 0 and b > 0:
            logs.append(math.log(w / b))
    return float(math.exp(sum(logs) / len(logs))) if logs else 0.0


@register_function(
    aliases=["compare_codon_strategies", "密码子策略比较", "和谐化vs优化",
             "compare_harmonization", "codon_strategy"],
    category="synthetic_biology",
    description="并排比较三种密码子策略:原始序列、CAI 优化(codon_optimize 的目标)、和谐化。给出 CAI、GC、改动数与稀有度排名相关性 —— 用来看清优化和和谐化在同一条序列上是往相反方向拉的。Compare native / CAI-optimised / harmonised codon strategies.",
    examples=["df = ov.synbio.compare_codon_strategies(cds, 'h_sapiens', 'e_coli')"],
    related=["synbio.codon_harmonize", "synbio.codon_optimize"],
    requires={},
    produces={},
)
def compare_codon_strategies(cds: str, source_host: str = "h_sapiens",
                             target_host: str = "e_coli") -> "pd.DataFrame":
    """One row per strategy: CAI in the target host, GC, codons changed."""
    import pandas as pd

    seq = _clean_dna(cds, "compare_codon_strategies")
    _check_hosts(source_host, target_host)
    harm = codon_harmonize(seq, source_host, target_host)
    tgt_usage = codon_usage(target_host)
    tgt_ranks = _family_ranks(tgt_usage)
    opt = _max_cai(seq, tgt_ranks)

    def gc(s: str) -> float:
        return (s.count("G") + s.count("C")) / max(1, len(s))

    def n_diff(a: str, b: str) -> int:
        return sum(1 for i in range(0, min(len(a), len(b)) - 2, 3)
                   if a[i:i + 3] != b[i:i + 3])

    rows = [
        {"strategy": "native", "cai_in_target": _cai(seq, tgt_usage),
         "gc": gc(seq), "codons_changed": 0},
        {"strategy": "optimized", "cai_in_target": _cai(opt, tgt_usage),
         "gc": gc(opt), "codons_changed": n_diff(seq, opt)},
        {"strategy": "harmonized", "cai_in_target": harm.harmonized_cai,
         "gc": gc(harm.sequence), "codons_changed": harm.n_changed},
    ]
    return pd.DataFrame(rows).set_index("strategy")


# ---------------------------------------------------------------------------
# synthesis complexity
# ---------------------------------------------------------------------------

@dataclass
class SynthesisIssue:
    kind: str
    start: int          # 1-based
    end: int
    detail: str
    severity: float     # 0..1

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.kind}({self.start}-{self.end}: {self.detail})"


@dataclass
class SynthesisAssessment:
    sequence_length: int
    score: float                    # 0 = trivial to synthesise, 1 = likely refused
    issues: List[SynthesisIssue] = field(default_factory=list)
    gc_content: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def difficulty(self) -> str:
        if self.score < 0.25:
            return "routine"
        if self.score < 0.5:
            return "moderate"
        if self.score < 0.75:
            return "difficult"
        return "likely to be rejected"

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        if not self.issues:
            return pd.DataFrame(columns=["kind", "start", "end", "detail", "severity"])
        return pd.DataFrame([{
            "kind": i.kind, "start": i.start, "end": i.end,
            "detail": i.detail, "severity": i.severity} for i in self.issues])

    def __repr__(self) -> str:  # pragma: no cover
        return (f"SynthesisAssessment({self.difficulty}, score={self.score:.2f}, "
                f"{len(self.issues)} issues, GC={self.gc_content:.1%})")


@register_function(
    aliases=["synthesis_complexity", "合成难度", "合成复杂度", "可合成性",
             "synthesis_difficulty", "manufacturability_dna", "下单前检查"],
    category="synthetic_biology",
    description="DNA 合成难度评分:GC 极值与局部 GC 漂移、长同聚物、直接重复与反向重复、发夹结构 —— 每一项都定位到序列坐标,是可修的问题而不只是一个价格。设计完直接下单会踩的坑在这里先看到。Score how hard a designed sequence is to synthesise, with located issues.",
    examples=[
        "asm = ov.synbio.synthesis_complexity(dna)",
        "asm.difficulty, asm.issues, asm.to_frame()",
    ],
    related=["synbio.codon_optimize", "synbio.codon_harmonize",
             "synbio.plot_synthesis_complexity"],
    requires={},
    produces={},
)
def synthesis_complexity(
    sequence: str,
    *,
    gc_window: int = 50,
    gc_low: float = 0.25,
    gc_high: float = 0.75,
    homopolymer_length: int = 8,
    repeat_length: int = 20,
    hairpin_stem: int = 10,
    hairpin_loop_max: int = 100,
) -> SynthesisAssessment:
    """Flag the features that make commercial DNA synthesis fail.

    Every check reports coordinates, because the useful output is "fix bases
    412-431", not "this is a hard sequence".

    ``gc_extreme``
        Windows outside ``[gc_low, gc_high]``. Vendors quote or refuse on these.
    ``homopolymer``
        Runs of one base ≥ ``homopolymer_length`` — polymerase slippage.
    ``direct_repeat``
        Repeated ``repeat_length``-mers, which misassemble.
    ``inverted_repeat``
        A ``repeat_length``-mer whose reverse complement also appears — the
        worst case for assembly, and invisible to a direct-repeat scan.
    ``hairpin``
        Stems ≥ ``hairpin_stem`` with a loop up to ``hairpin_loop_max``.
    """
    seq = _clean_dna(sequence, "synthesis_complexity")
    n = len(seq)
    issues: List[SynthesisIssue] = []

    gc_total = (seq.count("G") + seq.count("C")) / n

    # GC windows
    w = min(gc_window, n)
    worst_gc_dev = 0.0
    i = 0
    while i + w <= n:
        win = seq[i:i + w]
        gc = (win.count("G") + win.count("C")) / w
        if gc < gc_low or gc > gc_high:
            dev = max(gc_low - gc, gc - gc_high)
            worst_gc_dev = max(worst_gc_dev, dev)
            issues.append(SynthesisIssue(
                "gc_extreme", i + 1, i + w, f"GC={gc:.0%} in a {w}-nt window",
                min(1.0, dev / 0.25)))
            i += w                       # do not report every shifted window
        else:
            i += max(1, w // 2)

    # homopolymers
    run_start, run_base = 0, seq[0]
    longest_run = 1
    for j in range(1, n + 1):
        base = seq[j] if j < n else None
        if base != run_base:
            length = j - run_start
            longest_run = max(longest_run, length)
            if length >= homopolymer_length:
                issues.append(SynthesisIssue(
                    "homopolymer", run_start + 1, j,
                    f"{length}x{run_base}", min(1.0, length / 15.0)))
            run_start, run_base = j, base
    # Repeats. A single long repeat produces one hit per shifted k-mer window,
    # so consecutive hits are merged into the maximal repeated stretch — the
    # useful output is "bases 175-260 repeat bases 1-86", not 66 near-identical
    # rows that make the report unreadable.
    seen: Dict[str, int] = {}
    direct_spans: List[Tuple[int, int, int]] = []
    for j in range(0, n - repeat_length + 1):
        kmer = seq[j:j + repeat_length]
        if "N" in kmer:
            continue
        if kmer in seen:
            src = seen[kmer]
            if direct_spans and j == direct_spans[-1][1] + 1 \
                    and src == direct_spans[-1][2] + 1:
                s, _, o = direct_spans[-1]
                direct_spans[-1] = (s, j, src)
            else:
                direct_spans.append((j, j, src))
        else:
            seen[kmer] = j
    for s, e, src in direct_spans[:20]:
        issues.append(SynthesisIssue(
            "direct_repeat", s + 1, e + repeat_length,
            f"{e - s + repeat_length} nt repeat of position {src + 1}", 0.5))
    direct = len(direct_spans)

    inverted_spans: List[Tuple[int, int, int]] = []
    for j in range(0, n - repeat_length + 1):
        kmer = seq[j:j + repeat_length]
        if "N" in kmer:
            continue
        pos = seen.get(_revcomp(kmer))
        if pos is not None and pos != j:
            if inverted_spans and j == inverted_spans[-1][1] + 1:
                s, _, o = inverted_spans[-1]
                inverted_spans[-1] = (s, j, o)
            else:
                inverted_spans.append((j, j, pos))
    for s, e, src in inverted_spans[:20]:
        issues.append(SynthesisIssue(
            "inverted_repeat", s + 1, e + repeat_length,
            f"reverse complement of position {src + 1}", 0.7))
    inverted = len(inverted_spans)

    # hairpins: a stem is an inverted repeat separated by a short loop
    hairpins = 0
    for j in range(0, n - hairpin_stem):
        stem = seq[j:j + hairpin_stem]
        if "N" in stem:
            continue
        rc = _revcomp(stem)
        window_end = min(n, j + hairpin_stem + hairpin_loop_max + hairpin_stem)
        hit = seq.find(rc, j + hairpin_stem, window_end)
        if hit != -1:
            hairpins += 1
            if hairpins <= 20:
                issues.append(SynthesisIssue(
                    "hairpin", j + 1, hit + hairpin_stem,
                    f"{hairpin_stem}-nt stem, {hit - j - hairpin_stem}-nt loop",
                    0.6))

    metrics = {
        "gc_content": gc_total,
        "worst_gc_deviation": worst_gc_dev,
        "longest_homopolymer": float(longest_run),
        "direct_repeats": float(direct),
        "inverted_repeats": float(inverted),
        "hairpins": float(hairpins),
    }
    score = min(1.0,
                0.30 * min(1.0, worst_gc_dev / 0.25)
                + 0.20 * min(1.0, max(0, longest_run - homopolymer_length + 1) / 8.0)
                + 0.20 * min(1.0, direct / 10.0)
                + 0.15 * min(1.0, inverted / 5.0)
                + 0.15 * min(1.0, hairpins / 10.0))

    return SynthesisAssessment(
        sequence_length=n, score=float(score), issues=issues,
        gc_content=gc_total, metrics=metrics)


@register_function(
    aliases=["plot_synthesis_complexity", "合成难度图", "plot_synthesis",
             "GC漂移图", "合成问题图"],
    category="synthetic_biology",
    description="画合成难度:沿序列的滑窗 GC 曲线与可接受带,以及各类问题(GC 极值/同聚物/重复/发夹)的位置轨道。Plot synthesis difficulty: GC profile and located issues.",
    examples=["ov.synbio.plot_synthesis_complexity(dna)"],
    related=["synbio.synthesis_complexity"],
    requires={},
    produces={},
)
def plot_synthesis_complexity(target, window: int = 50, axes=None):
    """Accepts a sequence or a :class:`SynthesisAssessment`."""
    from ._plot import _mpl
    plt = _mpl()

    if isinstance(target, SynthesisAssessment):
        asm = target
        seq = None
    else:
        seq = _clean_dna(str(target), "plot_synthesis_complexity")
        asm = synthesis_complexity(seq, gc_window=window)

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.0),
                                 gridspec_kw={"height_ratios": [2, 1]},
                                 sharex=True)
    else:
        fig = axes[0].figure
    axes = list(axes)

    ax = axes[0]
    if seq is not None:
        w = min(window, len(seq))
        xs = list(range(1, len(seq) - w + 2))
        gcs = [(seq[i:i + w].count("G") + seq[i:i + w].count("C")) / w
               for i in range(len(seq) - w + 1)]
        ax.plot(xs, gcs, color="#377EB8", lw=1.2)
        ax.axhspan(0.25, 0.75, color="#4DAF4A", alpha=0.12,
                   label="vendor-friendly band")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    ax.set_ylabel(f"GC ({window}-nt window)")
    ax.set_title(f"{asm.difficulty}  (score {asm.score:.2f}, "
                 f"{len(asm.issues)} issues, {asm.sequence_length} nt)")

    ax = axes[1]
    kinds = ["gc_extreme", "homopolymer", "direct_repeat", "inverted_repeat",
             "hairpin"]
    colours = {"gc_extreme": "#E41A1C", "homopolymer": "#FF7F00",
               "direct_repeat": "#984EA3", "inverted_repeat": "#A65628",
               "hairpin": "#377EB8"}
    for issue in asm.issues:
        if issue.kind not in kinds:
            continue
        y = kinds.index(issue.kind)
        ax.plot([issue.start, issue.end], [y, y], lw=6, solid_capstyle="butt",
                color=colours[issue.kind], alpha=0.85)
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels(kinds, fontsize=8)
    ax.set_xlabel("position (nt)")
    ax.invert_yaxis()
    if not asm.issues:
        ax.text(0.5, 0.5, "no synthesis issues found", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)

    fig.tight_layout()
    return fig, axes


__all__ = [
    "codon_harmonize", "HarmonizationResult", "compare_codon_strategies",
    "codon_usage", "synthesis_complexity", "SynthesisAssessment",
    "SynthesisIssue", "plot_synthesis_complexity", "HOSTS",
]
