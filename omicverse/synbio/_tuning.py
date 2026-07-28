r"""Tuning a construct rather than just building one.

Five gaps that are individually small and collectively the difference between a
design that works and one that works *at the level you wanted*.

* :func:`tai` — the tRNA adaptation index. ``cai`` measures how closely a gene's
  codons match the *codon usage* of highly expressed genes; tAI measures how well
  they match the *tRNA pool that actually decodes them*, wobble interactions
  included. When the two disagree, tAI is usually the better predictor of
  translation rate, and disagreement is exactly the case worth knowing about.
* :func:`rbs_library` — a graded series of RBSs spanning a requested range of
  translation initiation rates. ``rbs_strength`` scores one RBS; tuning a pathway
  needs a set that covers a decade or three.
* :func:`promoter_library` — the same for transcription.
* :func:`integration_sites` — where in the chromosome to put it. Expression from
  different loci spans roughly 25% to 500% of a high-copy plasmid, so the site is
  a design parameter, not a formality.
* :func:`plasmid_burden` — what carrying and expressing the construct costs the
  host, coupled to :func:`~omicverse.synbio.rba` so the cost shows up as a growth
  rate rather than as a vague warning.
* :func:`toehold_switch` — a specific device, where ``rna_inverse_design`` is a
  general solver.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


_COMP = str.maketrans("ACGTUN", "TGCAAN")


def _revcomp(seq: str) -> str:
    return seq.upper().translate(_COMP)[::-1]


# ---------------------------------------------------------------------------
# tRNA adaptation index
# ---------------------------------------------------------------------------

# Wobble penalties from the dos Reis 2004 formulation: the cost of a
# non-Watson-Crick pairing at the third codon position. 0 is a perfect
# interaction, 1 no interaction. These four are the ones the index actually
# uses; they were originally fitted against E. coli and S. cerevisiae expression
# data and are the reason species-specific variants (stAI, gtAI) exist at all —
# so they are exposed as a parameter rather than hard-wired.
#: Wobble penalties from dos Reis, Savva & Wernisch (2004), keyed
#: **codon-base : anticodon-base** — the opposite order to the paper, which
#: writes them anticodon-first. The values below are transposed accordingly.
#:
#: Getting that transposition wrong is easy and silent: an earlier version
#: carried the paper's ``s(G:U) = 0.41`` under the key ``"G:U"`` even though this
#: dict means by that key "codon ending G, read by U34" — which the paper calls
#: ``s(U:G) = 0.68``. The two penalties were therefore swapped, under-weighting
#: every U-ending codon and over-weighting every G-ending one.
DEFAULT_WOBBLE_PENALTIES: Dict[str, float] = {
    "G:U": 0.68,     # codon ending G read by U34 anticodon  = paper's s(U:G)
    "I:C": 0.28,     # codon ending C read by inosine        = paper's s(I:C)
    "I:A": 0.9999,   # codon ending A read by inosine        = paper's s(I:A)
    "U:G": 0.41,     # codon ending U read by G34 anticodon  = paper's s(G:U)
}

#: tRNA gene copy numbers, keyed by anticodon, for two reference organisms.
#: These are genome properties (counted from tRNAscan annotations) rather than
#: fitted parameters. Supply your own for any other host — that is the whole
#: point of a *species-specific* index.
TRNA_COPY_NUMBERS: Dict[str, Dict[str, int]] = {
    # Counted from the RefSeq annotation of *E. coli* K-12 MG1655 (NC_000913.3):
    # every one of the 86 annotated tRNA genes was located, its anticodon read
    # out of the anticodon loop (U33 immediately 5', purine immediately 3'), and
    # checked against the amino acid in the /product qualifier. 82 resolved
    # unambiguously, plus proM (TGG), the two lysidine-modified ileX/ileY (keyed
    # TAT because the modified C34 reads AUA), and selC (excluded).
    #
    # The previous table was wrong for 14 of the 20 amino acids: it put the gene
    # count on the A-starting anticodon where *E. coli* actually uses a G- or
    # T-starting one — ``CTT: 6`` for lysine, when MG1655 has six lysT/V/W/Y/Z/Q
    # genes with anticodon **UUU** and no CUU-anticodon tRNA at all. The
    # per-amino-acid totals gave it away: 7 alanine and 9 glycine genes against a
    # real 5 and 6, and 89 genes against a real 86. The effect was to make the
    # *minor* codon of Lys/Asn/Glu/Asp/Phe/Ile score higher than the major one,
    # so tAI ranked E. coli's preferred AAA as its worst lysine codon.
    "e_coli": {
        "TGC": 3, "GGC": 2,                        # Ala (alaT/U/V, alaW/X)
        "ACG": 4, "CCG": 1, "CCT": 1, "TCT": 1,   # Arg
        "GTT": 4,                                  # Asn (asnT/U/V/W)
        "GTC": 3,                                  # Asp (aspT/U/V)
        "GCA": 1,                                  # Cys (cysT)
        "CTG": 2, "TTG": 2,                        # Gln
        "TTC": 4,                                  # Glu (gltT/U/V/W)
        "GCC": 4, "CCC": 1, "TCC": 1,             # Gly
        "GTG": 1,                                  # His (hisR)
        "GAT": 3, "TAT": 2,                        # Ile (ileT/U/V; ileX/Y lysidine)
        "CAG": 4, "CAA": 1, "TAA": 1, "TAG": 1, "GAG": 1,  # Leu
        "TTT": 6,                                  # Lys (lysT/V/W/Y/Z/Q)
        "CAT": 6,                                  # Met (elongator + initiator)
        "GAA": 2,                                  # Phe (pheU/V)
        "CGG": 1, "GGG": 1, "TGG": 1,             # Pro (proK/L/M)
        "GGA": 2, "CGA": 1, "TGA": 1, "GCT": 1,   # Ser
        "GGT": 2, "CGT": 1, "TGT": 1,             # Thr
        "CCA": 1,                                  # Trp (trpT)
        "GTA": 3,                                  # Tyr (tyrT/U/V)
        "TAC": 5, "GAC": 2,                        # Val
    },
    "s_cerevisiae": {
        "AGC": 11, "GGC": 0, "CGC": 0, "TGC": 5,
        "ACG": 6, "CCG": 1, "CCT": 1, "TCT": 11, "TCG": 0, "GCG": 0,
        "ATT": 10, "GTT": 0,
        "ATC": 15, "GTC": 0,
        "ACA": 4, "GCA": 0,
        "CTG": 9, "TTG": 1,
        "CTC": 2, "TTC": 14,
        "ACC": 16, "CCC": 2, "GCC": 3, "TCC": 3,
        "ATG": 7, "GTG": 0,
        "AAT": 13, "GAT": 0, "CAT": 2,
        "CAA": 7, "TAA": 7, "CAG": 1, "TAG": 3, "GAG": 0,
        "CTT": 14, "TTT": 7,
        "CAT_M": 10,
        "AAA": 10, "GAA": 0,
        "AGG": 10, "CGG": 1, "GGG": 0, "TGG": 2,
        "AGA": 11, "CGA": 1, "GGA": 0, "TGA": 3, "ACT": 4, "GCT": 0,
        "AGT": 11, "CGT": 1, "GGT": 0, "TGT": 4,
        "CCA": 6,
        "ATA": 8, "GTA": 0,
        "AAC": 14, "CAC": 0, "GAC": 2, "TAC": 2,
    },
}

_CODON_TABLE_MIN = {
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


@dataclass
class TAIResult:
    """A tRNA adaptation index and the weights behind it."""

    tai: float
    host: str
    weights: Dict[str, float] = field(default_factory=dict)
    n_codons: int = 0
    per_codon: List[float] = field(default_factory=list)
    wobble_penalties: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def bottleneck_codons(self, n: int = 10) -> List[Tuple[int, str, float]]:
        """The ``n`` slowest-decoded positions: ``(position, codon, weight)``.

        A low tAI spread over the whole gene and a low tAI caused by six terrible
        codons are different problems with different fixes, and only the second
        one is worth a targeted edit.
        """
        indexed = [(i + 1, c, w) for i, (c, w) in enumerate(self.per_codon)]
        return sorted(indexed, key=lambda t: t[2])[:n]

    def __repr__(self) -> str:  # pragma: no cover
        return (f"TAIResult(tAI={self.tai:.4f}, host={self.host!r}, "
                f"{self.n_codons} codons)")


@register_function(
    aliases=["tai", "tAI", "tRNA适应指数", "tRNA_adaptation_index",
             "翻译效率指数", "stAI"],
    category="synthetic_biology",
    description="tRNA 适应指数 (tAI, dos Reis 2004):按 tRNA 基因拷贝数与摆动配对惩罚算每个密码子的相对适应度,再取几何平均。CAI 衡量的是密码子与高表达基因**用法**的接近程度,tAI 衡量的是与实际**解码它的 tRNA 池**的匹配程度 —— 两者不一致时通常 tAI 更能预测翻译速率,而不一致的那些基因正是值得注意的。内置大肠杆菌与酵母的 tRNA 拷贝数,其它宿主自己传(这正是『物种特异』的含义)。tRNA adaptation index.",
    examples=[
        "res = ov.synbio.tai(cds, host='e_coli')",
        "res.tai, res.bottleneck_codons()",
        "res = ov.synbio.tai(cds, trna_copies=my_counts)",
    ],
    related=["synbio.cai", "synbio.codon_optimize", "synbio.codon_harmonize",
             "synbio.plot_codon_usage"],
    requires={},
    produces={},
)
def tai(
    cds: str,
    host: str = "e_coli",
    *,
    trna_copies: Optional[Mapping[str, int]] = None,
    wobble_penalties: Optional[Mapping[str, float]] = None,
    exclude_met_trp: bool = True,
) -> TAIResult:
    """tRNA adaptation index of a coding sequence.

    Parameters
    ----------
    cds
        Coding DNA, length a multiple of 3.
    host
        ``'e_coli'`` or ``'s_cerevisiae'`` for the built-in tRNA gene counts.
    trna_copies
        ``{anticodon: gene_copy_number}``, overriding the built-in table. This is
        the parameter that makes the index species-specific, and supplying it is
        the right move for any host not in the table — a tAI computed with
        another organism's tRNA pool is not a tAI for yours.
    wobble_penalties
        Override :data:`DEFAULT_WOBBLE_PENALTIES`. The published values were
        fitted against *E. coli* and yeast expression data, and the existence of
        stAI/gtAI is precisely because they vary across the tree of life.
    exclude_met_trp
        Leave Met and Trp out of the geometric mean. They have one codon each, so
        they carry no adaptation signal and only pull the mean toward their own
        fixed weight.

    Returns
    -------
    TAIResult
    """
    seq = "".join(str(cds).split()).upper().replace("U", "T")
    if not seq:
        raise ValueError("ov.synbio.tai: 序列为空。")
    bad = sorted(set(seq) - set("ACGTN"))
    if bad:
        raise ValueError(f"ov.synbio.tai: 序列含非 DNA 字符 {bad}。")
    if len(seq) % 3:
        raise ValueError(f"CDS 长度 {len(seq)} 不是 3 的倍数。")

    if trna_copies is None:
        if host not in TRNA_COPY_NUMBERS:
            raise ValueError(
                f"没有内置 {host!r} 的 tRNA 拷贝数。内置的有 "
                f"{sorted(TRNA_COPY_NUMBERS)};其它宿主请传 trna_copies="
                f"{{anticodon: copies}}(可从 tRNAscan-SE 注释统计)。"
                f"用别的物种的 tRNA 池算出来的不是你这个宿主的 tAI。")
        copies = dict(TRNA_COPY_NUMBERS[host])
    else:
        copies = {str(k).upper().replace("U", "T"): int(v)
                  for k, v in trna_copies.items()}

    s = dict(DEFAULT_WOBBLE_PENALTIES)
    if wobble_penalties:
        s.update({k: float(v) for k, v in wobble_penalties.items()})

    # absolute adaptiveness W per codon: sum over anticodons that can read it,
    # each weighted by (1 - wobble penalty) x tRNA gene copies
    W: Dict[str, float] = {}
    for codon, aa in _CODON_TABLE_MIN.items():
        if aa == "*":
            continue
        third = codon[2]
        exact = _revcomp(codon)                       # Watson-Crick anticodon
        w = float(copies.get(exact, 0))               # penalty 0 for exact match
        # wobble contributions, by the third base of the codon
        if third == "T":                              # U3 also read by G34
            w += (1.0 - s["U:G"]) * copies.get(_revcomp(codon[:2] + "C"), 0)
        elif third == "C":                            # C3 also read by I34 (A34 gene)
            w += (1.0 - s["I:C"]) * copies.get(_revcomp(codon[:2] + "T"), 0)
        elif third == "A":                            # A3 also read by I34
            w += (1.0 - s["I:A"]) * copies.get(_revcomp(codon[:2] + "T"), 0)
        elif third == "G":                            # G3 also read by U34
            # The U34 reader's *gene* anticodon starts with T, i.e. it is
            # revcomp(XY + "A") — not revcomp(XY + "T"), which is the A34/inosine
            # gene and cannot pair with G3. Looking up the wrong tRNA left every
            # G-ending codon whose exact anticodon is absent at weight zero.
            w += (1.0 - s["G:U"]) * copies.get(_revcomp(codon[:2] + "A"), 0)
        W[codon] = w

    w_max = max(W.values()) if W else 1.0
    if w_max <= 0:
        raise ValueError(
            "所有密码子的绝对适应度都是 0 —— tRNA 拷贝数表的反密码子命名可能"
            "和这里的约定不一致(这里用 DNA 字母、5'->3' 的反密码子)。")

    # relative adaptiveness; codons with W = 0 get the geometric mean of the
    # non-zero weights, which is the standard treatment — a zero would make the
    # whole geometric mean zero and destroy the index for one missing tRNA.
    nonzero = [v / w_max for v in W.values() if v > 0]
    geo = math.exp(sum(math.log(v) for v in nonzero) / len(nonzero))
    weights = {c: (v / w_max if v > 0 else geo) for c, v in W.items()}

    notes: List[str] = []
    zero_codons = [c for c, v in W.items() if v <= 0]
    if zero_codons:
        notes.append(
            f"{len(zero_codons)} 个密码子没有可解码的 tRNA 基因,权重取非零权重的"
            f"几何平均({geo:.4f})——否则单个缺失 tRNA 会把整个几何平均归零。")

    skip = {"ATG", "TGG"} if exclude_met_trp else set()
    per_codon: List[Tuple[str, float]] = []
    logs: List[float] = []
    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3]
        aa = _CODON_TABLE_MIN.get(codon)
        if aa is None or aa == "*":
            continue
        w = weights.get(codon, geo)
        per_codon.append((codon, w))
        if codon in skip:
            continue
        logs.append(math.log(max(w, 1e-12)))

    value = math.exp(sum(logs) / len(logs)) if logs else 0.0
    return TAIResult(tai=float(value), host=host if trna_copies is None else "custom",
                     weights=weights, n_codons=len(per_codon),
                     per_codon=per_codon, wobble_penalties=s, notes=notes)


# ---------------------------------------------------------------------------
# graded expression libraries
# ---------------------------------------------------------------------------

@dataclass
class ExpressionLibrary:
    """A graded series of regulatory parts."""

    parts: List[str] = field(default_factory=list)
    predicted: List[float] = field(default_factory=list)
    kind: str = "rbs"
    target_range: Tuple[float, float] = (0.0, 0.0)
    method: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def achieved_range(self) -> Tuple[float, float]:
        return ((min(self.predicted), max(self.predicted))
                if self.predicted else (0.0, 0.0))

    @property
    def dynamic_range(self) -> float:
        lo, hi = self.achieved_range
        return hi / lo if lo > 0 else float("inf")

    @property
    def coverage(self) -> float:
        """How much of the requested range the library actually spans, in decades
        of the target over decades achieved."""
        t_lo, t_hi = self.target_range
        a_lo, a_hi = self.achieved_range
        if t_lo <= 0 or t_hi <= 0 or a_lo <= 0 or a_hi <= 0:
            return 0.0
        want = math.log10(t_hi / t_lo)
        got = math.log10(a_hi / a_lo)
        return min(1.0, got / want) if want > 0 else 1.0

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame({"sequence": self.parts,
                             "predicted": self.predicted}).sort_values(
            "predicted").reset_index(drop=True)

    def __repr__(self) -> str:  # pragma: no cover
        lo, hi = self.achieved_range
        return (f"ExpressionLibrary({self.kind}, {len(self.parts)} parts, "
                f"{lo:.3g}–{hi:.3g} ({self.dynamic_range:.0f}x), "
                f"coverage {self.coverage:.0%})")


@register_function(
    aliases=["rbs_library", "RBS梯度库", "RBS文库", "翻译强度梯度",
             "graded_rbs", "rbs_series"],
    category="synthetic_biology",
    description="设计一组**梯度** RBS,覆盖指定的翻译起始速率范围。rbs_strength 只给单条 RBS 打分,而调一条通路需要的是覆盖一两个数量级的一整套 —— 在退化位点上搜索并按预测强度分箱,每箱取一条,所以给出的是均匀铺开的序列而不是一堆强度相近的。Design a graded RBS library spanning a target range of translation rates.",
    examples=[
        "lib = ov.synbio.rbs_library(cds, n=8, target_range=(1, 1000))",
        "lib.to_frame(), lib.dynamic_range, lib.coverage",
    ],
    related=["synbio.rbs_strength", "synbio.promoter_library",
             "synbio.predict_expression", "synbio.doe_design"],
    requires={},
    produces={},
)
def rbs_library(
    cds: str,
    *,
    n: int = 8,
    target_range: Tuple[float, float] = (1.0, 1000.0),
    core: str = "AGGAGG",
    spacer_range: Tuple[int, int] = (4, 12),
    upstream: str = "TTTAAGAAGGAGATATACAT",
    n_candidates: int = 600,
    method: str = "thermodynamic",
    seed: int = 0,
) -> ExpressionLibrary:
    """Search degenerate RBS space for a graded set.

    Parameters
    ----------
    cds
        The coding sequence the RBS will drive — translation initiation depends
        on the 5' end of the CDS, so the same RBS gives different rates in front
        of different genes and the library has to be designed against this one.
    n
        How many library members.
    target_range
        Requested span of relative initiation rates, low to high.
    core
        Shine-Dalgarno core to mutate around.
    spacer_range
        Range of spacer lengths between the core and the start codon. Spacing is
        one of the strongest determinants, which is why it is searched rather
        than fixed.
    n_candidates
        Candidates to score before binning.
    method
        Passed to :func:`~omicverse.synbio.rbs_strength`.

    Returns
    -------
    ExpressionLibrary
    """
    import numpy as np

    if n < 2:
        raise ValueError("n 必须 >= 2 才叫梯度库。")
    lo_t, hi_t = float(target_range[0]), float(target_range[1])
    if not (0 < lo_t < hi_t):
        raise ValueError(f"target_range 必须是 0 < low < high,得到 {target_range!r}")

    seq = "".join(str(cds).split()).upper().replace("U", "T")
    if not seq.startswith("ATG"):
        seq = "ATG" + seq

    from ._expression import rbs_strength

    rng = np.random.default_rng(seed)
    bases = "ACGT"
    seen: set = set()
    scored: List[Tuple[str, float]] = []

    for _ in range(n_candidates):
        variant = list(core)
        # mutate 0-3 positions of the core; 0 keeps the consensus in the pool
        for pos in rng.choice(len(core), size=int(rng.integers(0, 4)),
                              replace=False):
            variant[int(pos)] = bases[int(rng.integers(0, 4))]
        spacer_len = int(rng.integers(spacer_range[0], spacer_range[1] + 1))
        spacer = "".join(bases[int(i)] for i in rng.integers(0, 4, spacer_len))
        utr = upstream + "".join(variant) + spacer
        if utr in seen:
            continue
        seen.add(utr)
        res = rbs_strength(utr + seq[:60], method=method)
        rate = float(res.initiation_rate)
        if rate > 0:
            scored.append((utr, rate))

    notes: List[str] = []
    if len(scored) < n:
        raise RuntimeError(
            f"只评出 {len(scored)} 条有效候选,凑不出 {n} 条的梯度库。"
            f"提高 n_candidates,或放宽 spacer_range。")

    # bin on a log scale and take one member per bin, so the library is spread
    # across the range rather than clustered wherever the search happened to
    # find the most candidates
    rates = np.array([r for _, r in scored])
    edges = np.logspace(math.log10(max(rates.min(), 1e-9)),
                        math.log10(rates.max()), n + 1)
    picked: List[Tuple[str, float]] = []
    for i in range(n):
        in_bin = [(u, r) for u, r in scored if edges[i] <= r <= edges[i + 1]]
        if in_bin:
            mid = math.sqrt(edges[i] * edges[i + 1])
            picked.append(min(in_bin, key=lambda ur: abs(ur[1] - mid)))
    if len(picked) < n:
        remaining = [p for p in scored if p not in picked]
        remaining.sort(key=lambda ur: ur[1])
        step = max(1, len(remaining) // (n - len(picked)))
        picked.extend(remaining[::step][: n - len(picked)])
        notes.append(
            f"部分强度区间里没有候选,已用等间隔补足到 {n} 条 —— "
            f"梯度不会完全均匀。")

    picked.sort(key=lambda ur: ur[1])
    lib = ExpressionLibrary(
        parts=[u for u, _ in picked], predicted=[r for _, r in picked],
        kind="rbs", target_range=(lo_t, hi_t), method=method, notes=notes)
    if lib.coverage < 0.5:
        lib.notes.append(
            f"实际只覆盖了目标范围的 {lib.coverage:.0%}"
            f"(达成 {lib.achieved_range[0]:.3g}–{lib.achieved_range[1]:.3g})。"
            f"要更宽的动态范围,得同时动启动子或拷贝数,单靠 RBS 不够。")
    return lib


@register_function(
    aliases=["promoter_library", "启动子梯度库", "启动子文库",
             "graded_promoter", "promoter_series"],
    category="synthetic_biology",
    description="设计一组梯度 σ70 启动子,覆盖指定的转录强度范围:在 -35/-10 盒的共识序列上做定向偏离,按 promoter_strength 打分后分箱取样。Design a graded σ70 promoter library spanning a target strength range.",
    examples=[
        "lib = ov.synbio.promoter_library(n=6, target_range=(0.05, 1.0))",
        "lib.to_frame()",
    ],
    related=["synbio.promoter_strength", "synbio.rbs_library",
             "synbio.compile_circuit"],
    requires={},
    produces={},
)
def promoter_library(
    *,
    n: int = 6,
    target_range: Tuple[float, float] = (0.02, 1.0),
    n_candidates: int = 800,
    spacer_length: int = 17,
    seed: int = 0,
) -> ExpressionLibrary:
    """Graded σ70 promoters by perturbing the −35 and −10 consensus boxes."""
    import numpy as np

    if n < 2:
        raise ValueError("n 必须 >= 2。")
    lo_t, hi_t = float(target_range[0]), float(target_range[1])
    if not (0 < lo_t < hi_t):
        raise ValueError(f"target_range 必须是 0 < low < high,得到 {target_range!r}")

    from ._expression import promoter_strength

    rng = np.random.default_rng(seed)
    bases = "ACGT"
    m35, m10 = "TTGACA", "TATAAT"
    scored: List[Tuple[str, float]] = []
    seen: set = set()

    for _ in range(n_candidates):
        b35 = list(m35)
        b10 = list(m10)
        for box in (b35, b10):
            for pos in rng.choice(len(box), size=int(rng.integers(0, 3)),
                                  replace=False):
                box[int(pos)] = bases[int(rng.integers(0, 4))]
        spacer = "".join(bases[int(i)] for i in rng.integers(0, 4, spacer_length))
        promoter = ("GCTAGC" + "".join(b35) + spacer + "".join(b10)
                    + "GCTAGCACTAGT")
        if promoter in seen:
            continue
        seen.add(promoter)
        value = float(promoter_strength(promoter)["strength"])
        if value > 0:
            scored.append((promoter, value))

    if len(scored) < n:
        raise RuntimeError(
            f"只评出 {len(scored)} 条候选,凑不出 {n} 条。提高 n_candidates。")

    values = np.array([v for _, v in scored])
    edges = np.logspace(math.log10(max(values.min(), 1e-9)),
                        math.log10(values.max()), n + 1)
    picked: List[Tuple[str, float]] = []
    for i in range(n):
        in_bin = [(p, v) for p, v in scored if edges[i] <= v <= edges[i + 1]]
        if in_bin:
            mid = math.sqrt(edges[i] * edges[i + 1])
            picked.append(min(in_bin, key=lambda pv: abs(pv[1] - mid)))
    if len(picked) < n:
        rest = sorted((p for p in scored if p not in picked), key=lambda pv: pv[1])
        step = max(1, len(rest) // max(1, n - len(picked)))
        picked.extend(rest[::step][: n - len(picked)])

    picked.sort(key=lambda pv: pv[1])
    lib = ExpressionLibrary(
        parts=[p for p, _ in picked], predicted=[v for _, v in picked],
        kind="promoter", target_range=(lo_t, hi_t), method="consensus")
    if lib.dynamic_range < 3.0:
        lib.notes.append(
            f"内置的 σ70 共识打分器动态范围有限(这里只有 "
            f"{lib.dynamic_range:.1f}x)——它按 -35/-10 盒与间距的匹配度给分,"
            f"本质是个有界的相似度,不是转录速率的定量模型。要真正跨数量级的"
            f"启动子梯度,用实测过的部件库(如 Anderson 系列)或改动拷贝数,"
            f"不要指望在共识序列上做扰动。")
    return lib


# ---------------------------------------------------------------------------
# genomic integration sites
# ---------------------------------------------------------------------------

@dataclass
class IntegrationSite:
    """A candidate chromosomal integration locus."""

    name: str
    position: int = 0                  # bp from oriC, or genome coordinate
    score: float = 0.0
    relative_expression: float = 1.0   # fold vs a reference locus
    mechanism: str = ""                # 'attTn7', 'attB', 'CRISPR', …
    essential_nearby: bool = False
    notes: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"IntegrationSite({self.name!r}, {self.mechanism}, "
                f"expr {self.relative_expression:.2f}x, score {self.score:.2f})")


#: Well-characterised *E. coli* integration loci. Positions are MG1655
#: coordinates; ``relative_expression`` is the published direction and rough
#: magnitude of the locus effect, which spans about 25%–500% of a high-copy
#: plasmid across the chromosome — so the site is a design parameter, not a
#: formality.
ECOLI_INTEGRATION_SITES: List[Dict[str, object]] = [
    {"name": "attTn7 (glmS)", "position": 3_925_000, "mechanism": "attTn7",
     "relative_expression": 1.0, "essential_nearby": False,
     "note": "the standard neutral site; downstream of glmS, no polar effects"},
    {"name": "attB(HK022)", "position": 1_100_000, "mechanism": "attB",
     "relative_expression": 1.2, "essential_nearby": False,
     "note": "phage HK022 attachment site"},
    {"name": "attB(lambda)", "position": 806_000, "mechanism": "attB",
     "relative_expression": 1.1, "essential_nearby": False,
     "note": "classic lambda site between gal and bio"},
    {"name": "near oriC", "position": 3_925_744, "mechanism": "CRISPR",
     "relative_expression": 2.6, "essential_nearby": True,
     "note": "highest copy during fast growth (replication gradient), but the "
             "neighbourhood is dense with essential genes"},
    {"name": "near terC", "position": 1_590_000, "mechanism": "CRISPR",
     "relative_expression": 0.4, "essential_nearby": False,
     "note": "lowest copy; useful when the construct is toxic"},
    {"name": "lacZ", "position": 365_000, "mechanism": "CRISPR",
     "relative_expression": 0.9, "essential_nearby": False,
     "note": "disrupts lac — convenient blue/white screen, and removes lactose "
             "utilisation"},
    {"name": "araBAD", "position": 70_000, "mechanism": "CRISPR",
     "relative_expression": 0.8, "essential_nearby": False,
     "note": "disrupts arabinose catabolism; do not combine with pBAD"},
]


@register_function(
    aliases=["integration_sites", "整合位点", "基因组整合", "landing_pad",
             "attB", "attTn7", "安全位点", "safe_harbor"],
    category="synthetic_biology",
    description="给基因组整合位点排序。同一构建体放在染色体不同位置,表达量可以差到 25%–500%(复制梯度:靠 oriC 的位点在快速生长时拷贝数更高),所以位点是**设计参数**而不是形式。按目标表达水平、是否邻近必需基因、可用的整合机制(attTn7/attB/CRISPR)打分,并说明每个位点的副作用(比如整到 lacZ 会顺手废掉乳糖利用)。Rank chromosomal integration loci.",
    examples=[
        "sites = ov.synbio.integration_sites(target_expression=2.0)",
        "sites = ov.synbio.integration_sites(host='e_coli', avoid_essential=True)",
    ],
    related=["synbio.select_backbone", "synbio.plasmid_burden",
             "synbio.design_grnas", "synbio.hdr_arms"],
    requires={},
    produces={},
)
def integration_sites(
    *,
    host: str = "e_coli",
    target_expression: Optional[float] = None,
    avoid_essential: bool = True,
    mechanism: Optional[str] = None,
    sites: Optional[Sequence[Mapping[str, object]]] = None,
) -> List[IntegrationSite]:
    """Score integration loci against what the design needs.

    Parameters
    ----------
    host
        Only ``'e_coli'`` has a built-in table; pass ``sites=`` for anything else.
    target_expression
        Desired expression relative to the neutral attTn7 site. Sites are ranked
        by closeness to this rather than by "highest", because the right answer
        for a toxic product is a *low* site.
    avoid_essential
        Penalise loci whose neighbourhood is dense with essential genes.
    mechanism
        Restrict to one integration mechanism.
    sites
        Custom loci: ``{'name', 'position', 'mechanism', 'relative_expression',
        'essential_nearby', 'note'}``.
    """
    if sites is None:
        if host != "e_coli":
            raise ValueError(
                f"只内置了 e_coli 的整合位点表。{host!r} 请传 sites=[...] —— "
                f"整合位点是基因组特异的,拿别的物种的坐标没有意义。")
        sites = ECOLI_INTEGRATION_SITES

    out: List[IntegrationSite] = []
    for raw in sites:
        name = str(raw["name"])
        expr = float(raw.get("relative_expression", 1.0))
        ess = bool(raw.get("essential_nearby", False))
        mech = str(raw.get("mechanism", ""))
        notes = [str(raw["note"])] if raw.get("note") else []

        if mechanism and mech != mechanism:
            continue

        score = 1.0
        if target_expression is not None:
            if target_expression <= 0:
                raise ValueError("target_expression 必须为正。")
            # closeness on a log scale — 2x too high and 2x too low are equally wrong
            score -= abs(math.log2(expr / target_expression)) / 3.0
        if ess and avoid_essential:
            score -= 0.4
            notes.append("邻近必需基因密集 —— 整合可能影响生长,需实测确认。")
        out.append(IntegrationSite(
            name=name, position=int(raw.get("position", 0)), score=score,
            relative_expression=expr, mechanism=mech, essential_nearby=ess,
            notes=notes))

    if not out:
        raise ValueError(
            f"没有符合条件的位点(mechanism={mechanism!r})。"
            f"可用机制:{sorted({str(s.get('mechanism', '')) for s in sites})}")
    out.sort(key=lambda s: -s.score)
    return out


# ---------------------------------------------------------------------------
# plasmid burden
# ---------------------------------------------------------------------------

@dataclass
class BurdenEstimate:
    """What carrying and expressing a construct costs the host."""

    growth_unburdened: float
    growth_burdened: float
    protein_fraction: float = 0.0
    copy_number: int = 1
    construct_kb: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def burden(self) -> float:
        """Fractional growth-rate loss."""
        if self.growth_unburdened <= 1e-12:
            return 0.0
        return 1.0 - self.growth_burdened / self.growth_unburdened

    @property
    def verdict(self) -> str:
        b = self.burden
        if b < 0.05:
            return "negligible"
        if b < 0.2:
            return "tolerable"
        if b < 0.5:
            return "will be selected against"
        return "likely to lose the construct"

    def __repr__(self) -> str:  # pragma: no cover
        return (f"BurdenEstimate({self.burden:.1%} growth loss, "
                f"{self.growth_unburdened:.4f} -> {self.growth_burdened:.4f} /h, "
                f"{self.verdict})")


@register_function(
    aliases=["plasmid_burden", "代谢负担", "质粒负担", "burden", "拷贝数负担",
             "metabolic_burden", "表达负担"],
    category="synthetic_biology",
    description="估算携带并表达一个构建体给宿主带来的代谢负担,并把它换算成**生长速率**而不是一句含糊的警告 —— 通过 rba 的蛋白质组预算实现:异源蛋白按拷贝数与表达强度占掉预算,剩下的才归宿主自己的酶。负担超过一定程度,培养过程中构建体会被选择性丢弃,这是设计阶段就该看到的。Estimate the growth cost of carrying and expressing a construct.",
    examples=[
        "est = ov.synbio.plasmid_burden(model, copy_number=20, construct_kb=5.0)",
        "est.burden, est.verdict",
    ],
    related=["synbio.rba", "synbio.select_backbone",
             "synbio.integration_sites", "synbio.predict_expression_level"],
    requires={},
    produces={},
)
def plasmid_burden(
    model,
    *,
    copy_number: int = 20,
    construct_kb: float = 5.0,
    expressed_protein_fraction: float = 0.05,
    total_protein: float = 0.55,
    genome_kb: float = 4641.0,
) -> BurdenEstimate:
    """Growth cost of a construct, via the proteome budget.

    Parameters
    ----------
    model
        A :class:`cobra.Model`.
    copy_number
        Plasmid copies per cell — the reason a high-copy vector is not free.
    construct_kb
        Size of the construct, for the DNA replication cost term.
    expressed_protein_fraction
        Fraction of the proteome the heterologous protein occupies at full
        induction. 5% is a modest recombinant; a strongly induced T7 system can
        exceed 30%, and the burden is roughly linear in this.
    total_protein
        Host proteome budget, as in :func:`~omicverse.synbio.rba`.
    genome_kb
        Host genome size, for scaling the replication term.

    Returns
    -------
    BurdenEstimate
    """
    if copy_number < 1:
        raise ValueError("copy_number 必须 >= 1")
    if not (0.0 <= expressed_protein_fraction < 1.0):
        raise ValueError("expressed_protein_fraction 必须在 [0, 1) 内。")

    from ._community import rba

    unburdened = rba(model, total_protein=total_protein)

    # A budget the model never reaches cannot show a burden. On e_coli_core the
    # default 0.55 g/gDW is far above what the network needs, so subtracting even
    # 35% of it left growth untouched and the function reported "negligible" for
    # a 100-copy plasmid expressing 30% of the proteome — a confidently wrong
    # answer. Find the budget that actually binds first, and burden is then
    # measured against that.
    notes: List[str] = []
    binding = total_protein
    if unburdened.growth >= unburdened.unconstrained_growth - 1e-9:
        probe = total_protein
        for _ in range(40):
            probe *= 0.7
            if rba(model, total_protein=probe).growth < \
                    unburdened.unconstrained_growth - 1e-6:
                break
        binding = probe / 0.7
        notes.append(
            f"total_protein={total_protein:g} 对这个模型不构成约束"
            f"(生长已达无约束上限 {unburdened.unconstrained_growth:.4f})。"
            f"负担改以真正起作用的预算 {binding:.4g} g/gDW 为基准 —— "
            f"否则任何负担都会被报成 0。")
        unburdened = rba(model, total_protein=binding)

    # the heterologous protein takes its share off the top of the same budget
    protein_cost = expressed_protein_fraction
    # replication cost: plasmid DNA as a fraction of the genome, which competes
    # for the same precursors and polymerase time
    dna_fraction = (copy_number * construct_kb) / genome_kb
    dna_cost = 0.05 * dna_fraction        # DNA is a small share of biomass cost

    remaining = binding * (1.0 - protein_cost - dna_cost)
    if remaining <= 0:
        remaining = total_protein * 0.01
        notes.append(
            "异源表达与复制成本已吃掉整个蛋白质组预算 —— 这样的构建体在诱导后"
            "基本无法生长。")

    burdened = rba(model, total_protein=remaining)

    notes.append(
        f"proteome: {protein_cost:.1%} to the heterologous protein, "
        f"{dna_cost:.2%} to replicating {copy_number} x {construct_kb} kb, "
        f"against a binding budget of {binding:.4g} g/gDW")
    est = BurdenEstimate(
        growth_unburdened=unburdened.growth, growth_burdened=burdened.growth,
        protein_fraction=expressed_protein_fraction, copy_number=copy_number,
        construct_kb=construct_kb,
        components={"protein": protein_cost, "dna": dna_cost,
                    "budget_left": remaining},
        notes=notes)
    if est.burden > 0.2:
        est.notes.append(
            f"负担 {est.burden:.0%}:培养若干代后未携带构建体的细胞会占优。"
            f"考虑降低拷贝数(select_backbone 的 copy_number=)、"
            f"改用染色体整合(integration_sites),或减弱启动子/RBS。")
    return est


# ---------------------------------------------------------------------------
# toehold switch
# ---------------------------------------------------------------------------

@dataclass
class ToeholdSwitch:
    """A designed toehold switch."""

    switch_rna: str
    trigger_rna: str
    toehold: str
    stem: str
    loop: str
    rbs: str
    start_codon_position: int = 0
    on_off_estimate: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"ToeholdSwitch(toehold {len(self.toehold)} nt, stem "
                f"{len(self.stem)} nt, ON/OFF ~{self.on_off_estimate:.0f}x)")


@register_function(
    aliases=["toehold_switch", "toehold开关", "核酸开关", "riboregulator",
             "toehold", "翻译开关"],
    category="synthetic_biology",
    description="设计 toehold 开关(Green 2014 式的翻译水平核酸调控器):由触发 RNA 序列反推出开关 RNA —— 单链 toehold 区负责起始识别,茎环把 RBS 与起始密码子封在里面,触发链结合后打开茎、露出 RBS。rna_inverse_design 是通用反向折叠求解器,这里是特定器件。Design a toehold switch riboregulator from a trigger sequence.",
    examples=[
        "sw = ov.synbio.toehold_switch(trigger='GGGAUUUAGCUCAGUUGGGA')",
        "sw.switch_rna, sw.on_off_estimate",
    ],
    related=["synbio.rna_inverse_design", "synbio.rna_fold",
             "synbio.rbs_strength", "synbio.rna_duplex"],
    requires={},
    produces={},
)
def toehold_switch(
    trigger: str,
    *,
    toehold_length: int = 12,
    stem_length: int = 9,
    loop: str = "AACAGAGGAGA",
    linker: str = "AACCUGGCGGCAGCGCAAAAG",
    rbs: str = "AGAGGAGA",
) -> ToeholdSwitch:
    """Design a toehold switch that responds to ``trigger``.

    The switch is built so that the trigger's 3' region is complementary to the
    switch's single-stranded toehold, and the region just inside it forms the
    stem that sequesters the RBS and the start codon. Trigger binding nucleates
    at the toehold and then unzips the stem, which is what makes the response
    thresholded rather than graded.

    Parameters
    ----------
    trigger
        The RNA (or DNA) the switch should detect.
    toehold_length
        Single-stranded recognition region. Longer binds faster and leaks more —
        the classic trade in this device.
    stem_length
        Base pairs sequestering the RBS. Longer means a lower OFF state and a
        harder-to-open switch.
    loop, linker, rbs
        Loop holding the RBS, the linker to the reporter, and the RBS itself.

    Returns
    -------
    ToeholdSwitch
    """
    seq = "".join(str(trigger).split()).upper().replace("T", "U")
    if not seq:
        raise ValueError("trigger 序列为空。")
    bad = sorted(set(seq) - set("ACGU"))
    if bad:
        raise ValueError(f"trigger 含非 RNA 字符 {bad}。")
    need = toehold_length + stem_length
    if len(seq) < need:
        raise ValueError(
            f"trigger 只有 {len(seq)} nt,而 toehold({toehold_length}) + "
            f"stem({stem_length}) 需要至少 {need} nt。缩短其中之一,或换更长的触发链。")

    def rc(s: str) -> str:
        return s.translate(str.maketrans("ACGU", "UGCA"))[::-1]

    # the switch's toehold is complementary to the trigger's 3' end, and the
    # stem-forming region is complementary to the segment just 5' of it
    toehold_target = seq[-toehold_length:]
    stem_target = seq[-need:-toehold_length]
    toehold = rc(toehold_target)
    stem_a = rc(stem_target)
    stem_b = rc(stem_a)

    switch = toehold + stem_a + loop + rbs + "AUG" + stem_b + linker
    start_pos = len(toehold + stem_a + loop + rbs) + 1

    gc = sum(1 for c in stem_a if c in "GC") / max(1, len(stem_a))
    # a more stable stem gives a lower OFF state, hence a higher ON/OFF ratio,
    # while a longer toehold speeds binding at the cost of leakage
    on_off = max(1.0, 6.0 * stem_length * (0.5 + gc) / max(toehold_length, 1) * 2.0)

    notes = [
        f"stem GC {gc:.0%};茎越稳 OFF 态越低但越难打开",
        f"toehold {toehold_length} nt:越长结合越快,泄漏也越多",
    ]
    if gc < 0.3:
        notes.append("茎 GC 偏低,OFF 态可能泄漏 —— 换触发链上 GC 更高的一段。")
    if toehold_length > 15:
        notes.append("toehold > 15 nt 通常泄漏明显上升。")

    return ToeholdSwitch(
        switch_rna=switch, trigger_rna=seq, toehold=toehold, stem=stem_a,
        loop=loop, rbs=rbs, start_codon_position=start_pos,
        on_off_estimate=float(on_off), notes=notes)


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

@register_function(
    aliases=["plot_expression_library", "梯度库图", "plot_library",
             "文库分布图"],
    category="synthetic_biology",
    description="画梯度表达库:各成员的预测强度(对数轴)与目标范围带,看出梯度是否真的铺开、还是挤在一处。Plot a graded expression library against its target range.",
    examples=["ov.synbio.plot_expression_library(lib)"],
    related=["synbio.rbs_library", "synbio.promoter_library"],
    requires={},
    produces={},
)
def plot_expression_library(library: ExpressionLibrary, ax=None):
    """Library members on a log strength axis, with the requested band shaded."""
    from ._plot import _mpl
    plt = _mpl()

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
    else:
        fig = ax.figure

    values = library.predicted
    ax.semilogy(range(1, len(values) + 1), values, "o-", color="#377EB8", lw=1.4)
    lo, hi = library.target_range
    if lo > 0 and hi > 0:
        ax.axhspan(lo, hi, color="#4DAF4A", alpha=0.13, label="target range")
    ax.set_xlabel("library member (sorted)")
    ax.set_ylabel(f"predicted {library.kind} strength")
    ax.set_title(f"{len(values)} {library.kind}s, {library.dynamic_range:.0f}x "
                 f"dynamic range, {library.coverage:.0%} of target covered")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


@register_function(
    aliases=["plot_integration_sites", "整合位点图", "plot_loci",
             "位点排序图"],
    category="synthetic_biology",
    description="画整合位点排序:相对表达量与得分,邻近必需基因的位点标红。Plot ranked integration loci by relative expression and score.",
    examples=["ov.synbio.plot_integration_sites(sites)"],
    related=["synbio.integration_sites"],
    requires={},
    produces={},
)
def plot_integration_sites(sites: Sequence[IntegrationSite], ax=None):
    """Loci by relative expression, coloured by essential-gene proximity."""
    from ._plot import _mpl
    plt = _mpl()

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(sites) + 1.8))
    else:
        fig = ax.figure

    names = [s.name for s in sites]
    expr = [s.relative_expression for s in sites]
    colours = ["#E41A1C" if s.essential_nearby else "#4DAF4A" for s in sites]
    ax.barh(range(len(sites)), expr, color=colours)
    ax.axvline(1.0, ls="--", c="k", lw=0.9, label="neutral site (attTn7)")
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("expression relative to the neutral site")
    ax.set_title("red = essential genes nearby")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


#: Codons preferred in highly expressed *E. coli* genes (Ikemura 1981;
#: Sharp & Li 1987). tAI is a tRNA-availability index, not a usage index, so it
#: is *not* expected to reproduce all of these — see :func:`check_trna_table`.
ECOLI_OPTIMAL_CODONS: Dict[str, str] = {
    "A": "GCT", "R": "CGT", "N": "AAC", "D": "GAC", "C": "TGC", "Q": "CAG",
    "E": "GAA", "G": "GGT", "H": "CAC", "I": "ATC", "L": "CTG", "K": "AAA",
    "F": "TTC", "P": "CCG", "T": "ACC", "Y": "TAC", "V": "GTT",
}

#: Number of tRNA genes per amino acid in *E. coli* K-12 MG1655, counted from
#: NC_000913.3 — a hard constraint on :data:`TRNA_COPY_NUMBERS`.
ECOLI_TRNA_TOTALS: Dict[str, int] = {
    "A": 5, "R": 7, "N": 4, "D": 3, "C": 1, "Q": 4, "E": 4, "G": 6, "H": 1,
    "I": 5, "L": 8, "K": 6, "M": 6, "F": 2, "P": 3, "S": 5, "T": 4, "W": 1,
    "Y": 3, "V": 7,
}


def check_trna_table(host: str = "e_coli", min_optimal: int = 12) -> Dict[str, object]:
    """Sanity-check a tRNA copy-number table. Empty result means it passed.

    Two independent checks, both of which the previous *E. coli* table failed:

    * **Gene totals per amino acid** against :data:`ECOLI_TRNA_TOTALS`. This is
      the check that exposes a transposed table without needing any expression
      data — the old one implied 7 alanine and 9 glycine tRNA genes where the
      genome has 5 and 6, and 89 genes in total against a real 86.
    * **Agreement with :data:`ECOLI_OPTIMAL_CODONS`**: how many amino acids have
      their highly-expressed-gene codon ranked first by tAI weight. Currently
      14 of 17. It is deliberately *not* required to be 17: tAI weights count
      tRNA genes, and for the four-fold families (Ala, Gly, Val) *E. coli* has
      more of the tRNA that reads the A-ending codon than the one Ikemura found
      preferred. Requiring 17 would mean tuning the wobble penalties until they
      reproduced a different index, which is the opposite of the point.

    Returns a dict of failures, so ``not check_trna_table()`` is the assertion.
    """
    problems: Dict[str, object] = {}
    if host not in TRNA_COPY_NUMBERS:
        return {"host": f"{host!r} not in TRNA_COPY_NUMBERS"}
    copies = TRNA_COPY_NUMBERS[host]

    if host == "e_coli":
        totals: Dict[str, int] = {}
        for anticodon, n in copies.items():
            aa = _CODON_TABLE_MIN.get(_revcomp(anticodon))
            if aa and aa != "*":
                totals[aa] = totals.get(aa, 0) + n
        wrong = {aa: (totals.get(aa, 0), want)
                 for aa, want in ECOLI_TRNA_TOTALS.items()
                 if totals.get(aa, 0) != want}
        if wrong:
            problems["gene_totals"] = wrong

        weights = _absolute_adaptiveness(copies, DEFAULT_WOBBLE_PENALTIES)
        hit = []
        for aa, best in ECOLI_OPTIMAL_CODONS.items():
            family = {c: weights[c] for c, a in _CODON_TABLE_MIN.items()
                      if a == aa and c in weights}
            if family and max(family, key=family.get) == best:
                hit.append(aa)
        if len(hit) < min_optimal:
            problems["optimal_codons"] = (
                f"{len(hit)}/{len(ECOLI_OPTIMAL_CODONS)} optimal codons rank "
                f"first, expected at least {min_optimal}")
    return problems


def _absolute_adaptiveness(copies: Mapping[str, int],
                           penalties: Mapping[str, float]) -> Dict[str, float]:
    """Per-codon absolute adaptiveness W — the core of :func:`tai`."""
    W: Dict[str, float] = {}
    for codon, aa in _CODON_TABLE_MIN.items():
        if aa == "*":
            continue
        third = codon[2]
        w = float(copies.get(_revcomp(codon), 0))
        if third == "T":
            w += (1.0 - penalties["U:G"]) * copies.get(_revcomp(codon[:2] + "C"), 0)
        elif third == "C":
            w += (1.0 - penalties["I:C"]) * copies.get(_revcomp(codon[:2] + "T"), 0)
        elif third == "A":
            w += (1.0 - penalties["I:A"]) * copies.get(_revcomp(codon[:2] + "T"), 0)
        elif third == "G":
            w += (1.0 - penalties["G:U"]) * copies.get(_revcomp(codon[:2] + "A"), 0)
        W[codon] = w
    return W


__all__ = [
    "tai", "TAIResult", "TRNA_COPY_NUMBERS", "DEFAULT_WOBBLE_PENALTIES",
    "check_trna_table", "ECOLI_OPTIMAL_CODONS", "ECOLI_TRNA_TOTALS",
    "rbs_library", "promoter_library", "ExpressionLibrary",
    "integration_sites", "IntegrationSite", "ECOLI_INTEGRATION_SITES",
    "plasmid_burden", "BurdenEstimate",
    "toehold_switch", "ToeholdSwitch",
    "plot_expression_library", "plot_integration_sites",
]
