r"""RNA design — inverse folding, siRNA and antisense-oligo design.

Where :mod:`_rna` *analyses* a given RNA, this module *designs* one:

* :func:`rna_inverse_design` — the inverse-folding problem: find a sequence
  that folds into a **target** secondary structure (ViennaRNA ``inverse_fold``;
  the Eterna/RNAinverse task). Useful for riboswitches, aptamers, thermometers.
* :func:`sirna_design` — rank 19-mer siRNA candidates against an mRNA with the
  published **Reynolds 2004** (8-criteria) or **Ui-Tei 2004** rational-design
  rules, plus target-site accessibility.
* :func:`aso_design` — design antisense oligonucleotides / gapmers against a
  transcript, scored by target accessibility, Tm and liability motifs.

All three are dependency-light (ViennaRNA only) and CPU-only. Heavier,
deep-learning RNA/mRNA design (LinearDesign, UTR-LM) lives in :mod:`_mrna`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .._registry import register_function

_COMP = str.maketrans("ACGUacgu", "UGCAugca")
_DNA_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def _rna(fn: str):
    try:
        import RNA
        return RNA
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"ov.synbio.{fn} 需要 ViennaRNA。请 pip install ViennaRNA "
            "(或 pip install 'omicverse[synbio]')。"
        ) from exc


def _rc_rna(s: str) -> str:
    return s.upper().replace("T", "U").translate(_COMP)[::-1]


def _rc_dna(s: str) -> str:
    return s.upper().replace("U", "T").translate(_DNA_COMP)[::-1]


# ---------------------------------------------------------------------------
# 1 — inverse folding
# ---------------------------------------------------------------------------
@dataclass
class RNADesign:
    sequence: str
    structure: str          # achieved dot-bracket
    target: str             # requested dot-bracket
    mfe: float              # kcal/mol of the achieved fold
    distance: int           # base-pair distance to target (0 = exact)

    def __repr__(self) -> str:  # pragma: no cover
        ok = "exact" if self.distance == 0 else f"d={self.distance}"
        return f"RNADesign(len={len(self.sequence)}, mfe={self.mfe:.2f}, {ok})"


@register_function(
    aliases=["rna_inverse_design", "RNA反向设计", "反向折叠", "rna_inverse_fold",
             "inverse_fold", "RNA设计", "结构设计序列", "rna_design"],
    category="synthetic_biology",
    description="RNA 反向折叠设计:给定目标二级结构(点括号),设计能折叠成它的序列(ViennaRNA inverse_fold)。用于核糖开关/适配体/RNA 温度计。Inverse RNA folding — design a sequence that folds to a target structure.",
    examples=[
        "designs = ov.synbio.rna_inverse_design('((((....))))', n=3)",
        "designs[0].sequence, designs[0].distance",
    ],
    related=["synbio.rna_fold", "synbio.mrna_design", "synbio.rna_accessibility"],
    requires={},
    produces={},
)
def rna_inverse_design(target_structure: str, n: int = 1,
                       start: Optional[str] = None,
                       gc_target: Optional[float] = None,
                       max_tries: int = 20) -> List[RNADesign]:
    """Design ``n`` sequences that fold into *target_structure* (dot-bracket).

    Uses ViennaRNA ``inverse_fold`` from randomised starts; each result is
    re-folded to report the achieved MFE structure and its base-pair distance
    to the target (0 = the target is the MFE structure)."""
    RNA = _rna("rna_inverse_design")
    tgt = target_structure.strip()
    if tgt.count("(") != tgt.count(")"):
        raise ValueError("target_structure 括号不配对,不是合法的点括号结构。")
    L = len(tgt)

    # deterministic-but-varied starts: cycle a fixed alphabet seeded by index so
    # repeated calls are reproducible without Math.random-style nondeterminism.
    alph = "ACGU"
    paired = "GC" if (gc_target is None or gc_target >= 0.5) else "AU"
    designs: List[RNADesign] = []
    seen = set()
    for i in range(n):
        best: Optional[RNADesign] = None
        for t in range(max_tries):
            if start is not None:
                seed = (start.upper().replace("T", "U") + "N" * L)[:L]
            else:
                # vary the seed by (i, t) so distinct calls explore differently
                seed = "".join(
                    (paired if c in "()" else alph[(j + i * 7 + t * 3) % 4])
                    for j, c in enumerate(tgt))
            seq, _ = RNA.inverse_fold(seed.replace("N", "A"), tgt)
            seq = seq.upper()
            struct, mfe = RNA.fold(seq)
            dist = RNA.bp_distance(struct, tgt)
            cand = RNADesign(sequence=seq, structure=struct, target=tgt,
                             mfe=float(mfe), distance=int(dist))
            if best is None or cand.distance < best.distance:
                best = cand
            if dist == 0 and seq not in seen:
                break
        if best is not None and best.sequence not in seen:
            seen.add(best.sequence)
            designs.append(best)
    return designs


# ---------------------------------------------------------------------------
# 2 — siRNA design (Reynolds 2004 / Ui-Tei 2004)
# ---------------------------------------------------------------------------
@dataclass
class SiRNA:
    position: int           # 0-based start of the 19-nt sense target on the mRNA
    sense: str              # 19-nt sense (== mRNA target), RNA
    antisense: str          # 21-nt guide with UU overhang (5'->3')
    gc: float
    score: int              # rational-design score (higher = better)
    method: str
    criteria: Dict[str, bool] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"SiRNA(@{self.position} {self.method} score={self.score} "
                f"GC={self.gc:.2f} sense={self.sense})")


def _reynolds_score(s: str) -> Tuple[int, Dict[str, bool]]:
    """Reynolds *et al.* 2004 8-criteria score on a 19-nt sense strand (1-based
    positions as in the paper)."""
    gc = (s.count("G") + s.count("C")) / 19.0
    au_15_19 = sum(1 for c in s[14:19] if c in "AU")
    c = {
        "GC_30_52": 0.30 <= gc <= 0.52,          # I
        "AU_15to19>=3": au_15_19 >= 3,           # II
        "A19": s[18] == "A",                     # IV
        "A3": s[2] == "A",                       # V
        "U10": s[9] == "U",                      # VI
        "not_GC_19": s[18] not in "GC",          # VII
        "not_G13": s[12] != "G",                 # VIII
    }
    return sum(c.values()), c


def _uitei_score(s: str) -> Tuple[int, Dict[str, bool]]:
    """Ui-Tei *et al.* 2004 rules on a 19-nt sense strand.  Antisense 5' end
    corresponds to the sense 3' base (position 19)."""
    au_1to7_anti = sum(1 for ch in s[12:19] if ch in "AU")   # antisense 1-7 ~ sense 13-19
    # longest GC run
    run = mx = 0
    for ch in s:
        run = run + 1 if ch in "GC" else 0
        mx = max(mx, run)
    c = {
        "anti5_AU (sense19 A/U)": s[18] in "AU",     # A/U at antisense 5'
        "sense5_GC (sense1 G/C)": s[0] in "GC",      # G/C at sense 5'
        "AU_rich_anti_seed>=4": au_1to7_anti >= 4,
        "no_GC_run>9": mx <= 9,
    }
    return sum(c.values()), c


#: 5' terminal pentamer stability difference, in arbitrary stacking units.
#: Positive means the antisense 5' end is the *less* stable one, which is what
#: loads it as the guide.
_STACK = {"GC": 3.0, "CG": 3.0, "GG": 2.5, "CC": 2.5, "AU": 1.0, "UA": 1.0,
          "AA": 0.9, "UU": 0.9, "AG": 1.8, "GA": 1.8, "AC": 1.8, "CA": 1.8,
          "UG": 1.6, "GU": 1.6, "UC": 1.6, "CU": 1.6}


def _terminal_asymmetry(antisense: str, window: int = 5) -> float:
    """Stability of the antisense 3' end minus its 5' end, over ``window`` nt.

    RISC loads the strand whose 5' end is less thermodynamically stable, so a
    positive value is what a working siRNA needs. Nothing in the ranking used to
    look at it, and a candidate with inverted asymmetry scored identically to a
    correct one.
    """
    seq = antisense.upper().replace("T", "U")
    if len(seq) < 2 * window:
        return 0.0

    def stability(part: str) -> float:
        return sum(_STACK.get(part[i:i + 2], 1.5) for i in range(len(part) - 1))

    return stability(seq[-window:]) - stability(seq[:window])


@register_function(
    aliases=["sirna_design", "siRNA设计", "siRNA", "rnai_design", "小干扰RNA",
             "沉默设计", "knockdown_design"],
    category="synthetic_biology",
    description="siRNA 理性设计:在 mRNA 上扫描 19-mer,用 Reynolds 2004(8 准则)或 Ui-Tei 2004 规则打分,并结合靶位点可及性排序,返回带 UU 突出端的反义链。Rational siRNA design (Reynolds/Ui-Tei rules + accessibility).",
    examples=[
        "sirnas = ov.synbio.sirna_design(mrna, n=10, method='reynolds')",
        "sirnas[0].antisense, sirnas[0].score",
    ],
    related=["synbio.aso_design", "synbio.rna_duplex", "synbio.rna_accessibility"],
    requires={},
    produces={},
)
def sirna_design(mrna: str, n: int = 10, method: str = "reynolds",
                 min_score: int = 0, avoid_start: int = 75,
                 min_spacing: Optional[int] = None,
                 rank_by_asymmetry: bool = True) -> List[SiRNA]:
    """Rank 19-nt siRNA candidates against *mrna*.

    ``method='reynolds'`` uses the Reynolds rules; ``method='uitei'`` the Ui-Tei
    rules. Candidates start after ``avoid_start`` nt.

    Ties on the rule score are broken by **thermodynamic asymmetry** — the
    difference in duplex stability between the two ends, which is the single
    largest determinant of which strand RISC loads (Khvorova, Schwarz 2003). A
    candidate whose asymmetry is inverted will load the passenger strand and
    silence the wrong transcript, and it used to be returned with the same rank as
    a correct one.

    ``min_spacing`` (default: the siRNA length, i.e. no overlap) stops the
    function returning the same site several times over. It previously ranked by
    ``(-score, position)``, so ``n=5`` came back as positions 81, 82, 83, 96, 98 —
    three 1-nt-shifted copies of one site presented as three designs.

    The docstring used to claim ties were broken by target-site accessibility.
    ``rna_accessibility`` was never called; it now is, when ``rank_by_asymmetry``
    is False.
    """
    scorers = {"reynolds": _reynolds_score, "uitei": _uitei_score}
    if method not in scorers:
        raise ValueError(
            f"method must be one of {sorted(scorers)}, got {method!r}")
    m = mrna.upper().replace("T", "U")
    scorer = scorers[method]

    cands: List[SiRNA] = []
    for i in range(avoid_start, len(m) - 19):
        sense = m[i:i + 19]
        if "N" in sense:
            continue
        score, crit = scorer(sense)
        if score < min_score:
            continue
        gc = (sense.count("G") + sense.count("C")) / 19.0
        # antisense guide = reverse complement of the sense 19-mer + UU overhang
        anti = _rc_rna(sense) + "UU"
        cands.append(SiRNA(position=i, sense=sense, antisense=anti, gc=gc,
                           score=score, method=method, criteria=crit))

    # rank: rule score desc, then 5'->3' position (earlier, well-scored sites first)
    # secondary key: asymmetry (or accessibility), not position
    for c in cands:
        c.criteria = dict(c.criteria or {})
        if rank_by_asymmetry:
            c.criteria["asymmetry"] = _terminal_asymmetry(c.antisense)
            c.criteria["secondary"] = c.criteria["asymmetry"]
        else:
            # Import here rather than at module scope so the ViennaRNA dependency
            # stays optional, and let an ImportError propagate: a bare
            # `except Exception -> nan` hid a NameError and made the accessibility
            # mode silently do nothing, which is how the docstring came to promise
            # a ranking the code never performed.
            from ._rna import rna_accessibility
            open_cost = float(rna_accessibility(m, c.position,
                                                c.position + len(c.sense)))
            c.criteria["accessibility"] = open_cost
            c.criteria["secondary"] = -open_cost
    cands.sort(key=lambda c: (-c.score, -c.criteria.get("secondary", 0.0), c.position))
    for c in cands:
        if rank_by_asymmetry and c.criteria.get("asymmetry", 0.0) < 0:
            c.criteria["warning"] = (
                "thermodynamic asymmetry is inverted — RISC will preferentially "
                "load the passenger strand and silence the wrong transcript")

    # enforce spacing so distinct designs are distinct sites
    spacing = len(cands[0].sense) if (min_spacing is None and cands) else (min_spacing or 0)
    spaced: List[SiRNA] = []
    for cand in cands:
        if all(abs(cand.position - kept.position) >= spacing for kept in spaced):
            spaced.append(cand)
    cands = spaced
    return cands[:n]


# ---------------------------------------------------------------------------
# 3 — antisense oligonucleotide (ASO / gapmer) design
# ---------------------------------------------------------------------------
@dataclass
class ASO:
    position: int           # 0-based start on the target transcript
    target: str             # the transcript window (RNA) the ASO binds
    aso: str                # the antisense oligo (DNA, 5'->3')
    tm: float               # nearest-neighbour Tm (°C)
    accessibility: float    # opening energy of the target window (kcal/mol; lower=open)
    score: float            # combined design score (higher = better)
    liabilities: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        flag = f" ⚠{','.join(self.liabilities)}" if self.liabilities else ""
        return (f"ASO(@{self.position} len={len(self.aso)} Tm={self.tm:.1f}°C "
                f"open={self.accessibility:.1f}{flag})")


def _wallace_tm(dna: str) -> float:
    """Nearest-neighbour Tm via Biopython if available, else Wallace rule."""
    try:
        from Bio.SeqUtils import MeltingTemp as mt
        return float(mt.Tm_NN(dna))
    except Exception:
        gc = dna.count("G") + dna.count("C")
        at = dna.count("A") + dna.count("T")
        return 64.9 + 41 * (gc - 16.4) / max(1, len(dna)) if len(dna) > 13 \
            else 2 * at + 4 * gc


def _aso_liabilities(dna: str) -> List[str]:
    liab = []
    if "GGGG" in dna:
        liab.append("G4run")                 # G-quadruplex / aggregation risk
    if "CG" in dna:
        liab.append("CpG")                   # immunostimulatory
    for b in "ACGT":
        if b * 5 in dna:
            liab.append(f"{b}5run")
    return liab


@register_function(
    aliases=["aso_design", "反义寡核苷酸设计", "ASO设计", "gapmer", "antisense_oligo",
             "反义设计", "gapmer_design", "antisense_design"],
    category="synthetic_biology",
    description="反义寡核苷酸(ASO/gapmer)设计:沿转录本扫描窗口,按靶位点可及性、Tm 与风险基序(G四联体/CpG/同聚物)打分,返回排序的 DNA 反义序列。Antisense-oligo / gapmer design scored by accessibility, Tm and liability motifs.",
    examples=[
        "asos = ov.synbio.aso_design(mrna, length=18, n=8)",
        "asos[0].aso, asos[0].tm, asos[0].accessibility",
    ],
    related=["synbio.sirna_design", "synbio.rna_accessibility", "synbio.rna_duplex"],
    requires={},
    produces={},
)
def aso_design(target: str, length: int = 18, n: int = 8,
               tm_range: Tuple[float, float] = (45.0, 65.0),
               step: int = 3, avoid_start: int = 50) -> List[ASO]:
    """Design antisense oligonucleotides against *target* (a transcript).

    Slides a window of ``length`` nt; the ASO is the reverse complement (DNA)
    of each window. Windows are scored by target accessibility (opening energy,
    lower is better), Tm proximity to the middle of ``tm_range``, and penalised
    for liability motifs (G-quadruplex runs, CpG, homopolymers)."""
    from ._rna import rna_accessibility
    m = target.upper().replace("T", "U")
    tm_mid = sum(tm_range) / 2.0

    cands: List[ASO] = []
    for i in range(avoid_start, len(m) - length, max(1, step)):
        win = m[i:i + length]
        if "N" in win:
            continue
        aso = _rc_dna(win)
        tm = _wallace_tm(aso)
        try:
            acc = rna_accessibility(m, i, i + length)
        except Exception:
            acc = 0.0
        liab = _aso_liabilities(aso)
        # score: reward accessibility (open target) & Tm centrality, penalise liabilities
        score = -acc - 0.3 * abs(tm - tm_mid) - 2.0 * len(liab)
        if not (tm_range[0] - 8 <= tm <= tm_range[1] + 8):
            continue
        cands.append(ASO(position=i, target=win, aso=aso, tm=tm,
                         accessibility=acc, score=score, liabilities=liab))

    cands.sort(key=lambda c: -c.score)
    return cands[:n]


__all__ = ["rna_inverse_design", "sirna_design", "aso_design",
           "RNADesign", "SiRNA", "ASO"]
