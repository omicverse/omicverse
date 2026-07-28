r"""Advanced genome editing — prime editing, CRISPRi/a, and Cas13.

Beyond cut-and-repair (:mod:`_crispr`), this module covers the modern editing
modalities:

* :func:`prime_editing_design` — design a **pegRNA** (spacer + reverse-
  transcriptase template + primer-binding site) and the PE3 nicking guide for a
  desired substitution / insertion / deletion (Anzalone *et al.* 2019). The
  ``method='primedesign'`` backend defers to the vendored PrimeDesign tool.
* :func:`crispr_regulation` — CRISPRi (repression) / CRISPRa (activation) guide
  design, scored by position in the optimal window **relative to the TSS**
  rather than by cut efficiency.
* :func:`design_cas13_guides` — Cas13 crRNA design for RNA targeting /
  knockdown, filtered by the 3' protospacer-flanking site (PFS) and target
  accessibility.

The pegRNA / CRISPRi / Cas13 algorithms here are transparent, published-rule
designs (CPU-only); the heavier learned scorers plug in via ``method=``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .._registry import register_function

_COMP = str.maketrans("ACGTN", "TGCAN")


def _rc(s: str) -> str:
    return s.upper().replace("U", "T").translate(_COMP)[::-1]


# ---------------------------------------------------------------------------
# 1 — prime editing (pegRNA + PE3 nick)
# ---------------------------------------------------------------------------
@dataclass
class PegRNA:
    spacer: str                 # 20-nt protospacer (5'->3', the pegRNA 5' end)
    pbs: str                    # primer-binding site (3' extension, 5'->3')
    rtt: str                    # RT template (3' extension, 5'->3')
    extension_3p: str           # full 3' extension = RTT + PBS (5'->3')
    strand: str                 # '+' | '-' (strand the protospacer lies on)
    pam: str
    nick_position: int          # 0-based nick site on the input (+ strand coord)
    nick_to_edit: int           # nt from nick to the edit (smaller = better)
    edit: str                   # human-readable edit description
    pe3_nick: Optional[Dict] = None    # secondary nicking guide (PE3)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"PegRNA({self.strand}@nick{self.nick_position} "
                f"nick→edit={self.nick_to_edit}nt PBS{len(self.pbs)}/RTT{len(self.rtt)} "
                f"spacer={self.spacer})")


def _find_pe_candidates(work: str, edited: str, edit_start: int, alt_len: int,
                        strand: str, pbs_len: int, homology: int,
                        max_nick_to_edit: int, offset_to_plus) -> List[PegRNA]:
    """Enumerate pegRNAs on one strand (``work`` is that strand 5'->3').

    ``edited`` is ``work`` with the edit applied; ``edit_start`` is the 0-based
    index (identical in *work* and *edited* because everything 5' of the nick is
    unchanged). Cas9 nicks 3 nt 5' of an ``NGG`` PAM; the edit must sit 3' of the
    nick and within the RT template's reach."""
    import re
    cands: List[PegRNA] = []
    for m in re.finditer(r"(?=([ACGT]GG))", work):     # NGG PAMs, overlapping
        p = m.start()                                   # PAM start on work
        if p < 21:
            continue
        nick = p - 3                                    # nick between 17|18 of spacer
        if nick < pbs_len or nick > edit_start:
            continue
        if edit_start - nick > max_nick_to_edit:
            continue
        spacer = work[p - 20:p]
        if len(spacer) < 20 or "N" in spacer:
            continue
        rtt_end = edit_start + alt_len + homology
        if rtt_end > len(edited):
            continue
        rtt_region = edited[nick:rtt_end]               # newly synthesised strand
        pbs_region = work[nick - pbs_len:nick]
        rtt = _rc(rtt_region)
        pbs = _rc(pbs_region)
        cands.append(PegRNA(
            spacer=spacer, pbs=pbs, rtt=rtt, extension_3p=rtt + pbs,
            strand=strand, pam=work[p:p + 3],
            nick_position=offset_to_plus(nick), nick_to_edit=edit_start - nick,
            edit="", ))
    return cands


def _pe3_nick(target: str, peg_strand: str, peg_nick_plus: int,
              lo: int = 40, hi: int = 90) -> Optional[Dict]:
    """Find a PE3 secondary-nick sgRNA on the *opposite* strand, 40–90 nt from
    the pegRNA nick (nicking the non-edited strand boosts editing)."""
    import re
    opp = _rc(target)
    n = len(target)
    best = None
    for m in re.finditer(r"(?=([ACGT]GG))", opp):
        p = m.start()
        if p < 21:
            continue
        nick_opp = p - 3
        nick_plus = n - nick_opp                        # map to + coord
        d = abs(nick_plus - peg_nick_plus)
        if lo <= d <= hi and (best is None or d < best[0]):
            best = (d, opp[p - 20:p], nick_plus)
    if best is None:
        return None
    return {"spacer": best[1], "nick_position": best[2], "distance": best[0],
            "strand": "-" if peg_strand == "+" else "+"}


@register_function(
    aliases=["prime_editing_design", "先导编辑", "pegRNA设计", "prime_edit",
             "pegRNA", "引导编辑", "先导编辑设计", "prime_editing"],
    category="synthetic_biology",
    description="Prime editing pegRNA 设计:为指定的替换/插入/缺失设计 pegRNA(spacer + RT 模板 RTT + 引物结合位点 PBS)以及 PE3 二级切口向导(Anzalone 2019)。method='baseline' 理性设计,method='primedesign' 走 vendored PrimeDesign(Pinello 实验室参考实现)。Design a pegRNA + PE3 nick for a desired edit.",
    examples=[
        "pegs = ov.synbio.prime_editing_design(locus, edit_pos=60, ref='A', alt='G')",
        "pegs[0].spacer, pegs[0].extension_3p",
    ],
    related=["synbio.design_grnas", "synbio.base_editor_window", "synbio.hdr_arms"],
    requires={},
    produces={},
)
def prime_editing_design(target: str, edit_pos: int, ref: str = "", alt: str = "",
                         pbs_len: int = 13, rtt_homology: int = 10,
                         max_nick_to_edit: int = 30, top_n: int = 5,
                         method: str = "baseline") -> List[PegRNA]:
    """Design pegRNAs to install an edit at *edit_pos* (0-based) in *target*.

    The edit replaces ``ref`` with ``alt`` starting at ``edit_pos``:
    substitution (``ref='A', alt='G'``), insertion (``ref='', alt='ATG'``) or
    deletion (``ref='ATG', alt=''``). Returns pegRNAs from both strands ranked
    by nick-to-edit distance (shorter edits closer to the nick prime best), each
    with a PE3 secondary nicking guide."""
    if method not in ("baseline", "primedesign"):
        raise ValueError(
            f"method must be one of ['baseline', 'primedesign'], got {method!r}")
    if method == "primedesign":
        from ._primedesign import run_primedesign
        return run_primedesign(target, edit_pos, ref, alt, n_pegrnas=top_n)
    seq = target.upper().replace("U", "T")
    if ref and seq[edit_pos:edit_pos + len(ref)] != ref.upper():
        raise ValueError(
            f"ref='{ref}' 与 target[{edit_pos}:{edit_pos+len(ref)}]="
            f"'{seq[edit_pos:edit_pos+len(ref)]}' 不符。")
    n = len(seq)
    edited_top = seq[:edit_pos] + alt.upper() + seq[edit_pos + len(ref):]
    desc = f"{ref or '-'}>{alt or '-'}@{edit_pos}"

    # + strand: edit_start = edit_pos, coords identity to +.
    plus = _find_pe_candidates(seq, edited_top, edit_pos, len(alt), "+",
                               pbs_len, rtt_homology, max_nick_to_edit,
                               offset_to_plus=lambda x: x)
    # - strand: work on revcomp; edit region maps to the 3' side.
    work_m = _rc(seq)
    edited_m = _rc(edited_top)
    edit_start_m = n - (edit_pos + len(ref))
    minus = _find_pe_candidates(work_m, edited_m, edit_start_m, len(alt), "-",
                                pbs_len, rtt_homology, max_nick_to_edit,
                                offset_to_plus=lambda x: n - x)

    cands = plus + minus
    for c in cands:
        c.edit = desc
        c.pe3_nick = _pe3_nick(seq, c.strand, c.nick_position)
    cands.sort(key=lambda c: (c.nick_to_edit, len(c.rtt)))
    return cands[:top_n]


# ---------------------------------------------------------------------------
# 2 — CRISPRi / CRISPRa (dCas9 regulation)
# ---------------------------------------------------------------------------
@dataclass
class RegGuide:
    spacer: str
    strand: str
    start: int              # 0-based on input
    tss_offset: int         # signed nt from TSS (− = upstream)
    efficiency: float       # on-target heuristic
    window_score: float     # suitability of position for the chosen mode
    score: float            # combined

    def __repr__(self) -> str:  # pragma: no cover
        return (f"RegGuide({self.strand}@{self.start} TSS{self.tss_offset:+d} "
                f"win={self.window_score:.2f} spacer={self.spacer})")


@register_function(
    aliases=["crispr_regulation", "CRISPRi", "CRISPRa", "转录调控向导",
             "dCas9调控", "基因抑制向导", "基因激活向导", "crispri_crispra"],
    category="synthetic_biology",
    description="CRISPRi/a 调控向导设计:相对转录起始位点(TSS)在最优窗口内挑选 dCas9 向导——CRISPRi 抑制窗口约 −50…+300、CRISPRa 激活窗口约 −400…−50——按窗口位置+on-target 打分。CRISPRi/CRISPRa guide design scored by TSS-relative window.",
    examples=[
        "g = ov.synbio.crispr_regulation(promoter, tss=400, mode='crispri')",
    ],
    related=["synbio.design_grnas", "synbio.prime_editing_design"],
    requires={},
    produces={},
)
def crispr_regulation(sequence: str, tss: int, mode: str = "crispri",
                      strand: str = "+", enzyme: str = "SpCas9",
                      top_n: int = 10) -> List[RegGuide]:
    """Design CRISPRi/CRISPRa guides around a transcription start site *tss*.

    ``mode='crispri'`` favours guides in ~[−50, +300] of the TSS (strongest
    dCas9 steric repression near/just downstream of the TSS); ``mode='crispra'``
    favours the ~[−400, −50] upstream window. ``tss`` is a 0-based index and
    ``strand`` is the gene's strand; guides are ranked by window fit × on-target
    efficiency."""
    if mode not in ("crispri", "crispra"):
        raise ValueError(f"mode must be one of ['crispri', 'crispra'], got {mode!r}")
    from ._crispr import design_grnas
    win = (-50, 300) if mode == "crispri" else (-400, -50)
    peak = (win[0] + win[1]) / 2.0
    half = (win[1] - win[0]) / 2.0

    guides = design_grnas(sequence, enzyme=enzyme)
    out: List[RegGuide] = []
    for g in guides:
        # signed offset from TSS in the gene's orientation
        off = (g.start - tss) if strand == "+" else (tss - g.start)
        if not (win[0] - 50 <= off <= win[1] + 50):
            continue
        wscore = max(0.0, 1.0 - abs(off - peak) / half)     # 1 at peak, 0 at edge
        out.append(RegGuide(
            spacer=g.spacer, strand=g.strand, start=g.start, tss_offset=off,
            efficiency=g.efficiency, window_score=wscore,
            score=wscore * g.efficiency))
    out.sort(key=lambda r: r.score, reverse=True)
    return out[:top_n]


# ---------------------------------------------------------------------------
# 3 — Cas13 crRNA (RNA targeting)
# ---------------------------------------------------------------------------
@dataclass
class Cas13Guide:
    """A Cas13 crRNA and the target site it pairs with.

    .. warning::

       ``spacer`` used to hold the **target** sequence, with the actual crRNA
       hidden in a field called ``target_rc``, and the docstring asserted that a
       crRNA spacer *is* the target RNA. It is not — a spacer is complementary to
       its protospacer, so ordering the old ``.spacer`` gave a sense-strand
       oligo that cannot guide Cas13. ``spacer`` is now the crRNA, i.e. the
       sequence to order, and the target lives in :attr:`protospacer`.
    """

    spacer: str             # the crRNA spacer — this is what you order
    protospacer: str        # the target RNA site it base-pairs with (5'->3')
    position: int
    gc: float
    pfs: str                # 3' protospacer-flanking base on the target
    accessibility: float    # opening energy of the target site (lower = open)
    score: float

    @property
    def target_rc(self) -> str:
        """Deprecated. The crRNA is now :attr:`spacer`."""
        import warnings
        warnings.warn(
            "Cas13Guide.target_rc 已废弃:crRNA 现在就是 .spacer,靶序列在 "
            ".protospacer。旧版把两者的名字弄反了。",
            DeprecationWarning, stacklevel=2)
        return self.spacer

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Cas13Guide(crRNA {self.spacer[:12]}… @{self.position} "
                f"GC={self.gc:.2f} PFS={self.pfs} "
                f"open={self.accessibility:.1f})")


@register_function(
    aliases=["design_cas13_guides", "Cas13设计", "Cas13", "RNA靶向向导",
             "cas13_crrna", "RNA敲低向导", "cas13_guides"],
    category="synthetic_biology",
    description="Cas13 crRNA 设计:在靶 RNA 上扫描 spacer(默认 28 nt),按 3' 保护性侧翼位点(PFS,LwaCas13a 偏好非 G)与靶位点可及性筛选打分,用于 RNA 敲低。Cas13 crRNA design for RNA knockdown (PFS + accessibility).",
    examples=[
        "g = ov.synbio.design_cas13_guides(mrna, spacer_len=28)",
    ],
    related=["synbio.sirna_design", "synbio.rna_accessibility"],
    requires={},
    produces={},
)
def design_cas13_guides(target_rna: str, spacer_len: int = 28, n: int = 10,
                        gc_range: Tuple[float, float] = (0.3, 0.7),
                        step: int = 2, avoid_start: int = 30) -> List[Cas13Guide]:
    """Design Cas13 crRNAs against *target_rna*.

    Slides a ``spacer_len``-nt window; keeps sites whose GC is in range and whose
    3' protospacer-flanking site (PFS) is non-G (the LwaCas13a preference), then
    ranks by target-site accessibility (open sites are cut better)."""
    from ._rna import rna_accessibility
    m = target_rna.upper().replace("T", "U")
    out: List[Cas13Guide] = []
    for i in range(avoid_start, len(m) - spacer_len - 1, max(1, step)):
        site = m[i:i + spacer_len]
        if "N" in site:
            continue
        gc = (site.count("G") + site.count("C")) / spacer_len
        if not (gc_range[0] <= gc <= gc_range[1]):
            continue
        pfs = m[i + spacer_len]                    # 3' flanking base on target
        if pfs == "G":                             # LwaCas13a disfavours G PFS
            continue
        try:
            acc = rna_accessibility(m, i, i + spacer_len)
        except Exception:
            acc = 0.0
        out.append(Cas13Guide(
            spacer=_rc(site).replace("T", "U"),      # complementary — order this
            protospacer=site.replace("T", "U"),      # the target site
            position=i, gc=gc, pfs=pfs, accessibility=acc, score=-acc))
    out.sort(key=lambda g: g.score, reverse=True)
    return out[:n]


__all__ = ["prime_editing_design", "crispr_regulation", "design_cas13_guides",
           "PegRNA", "RegGuide", "Cas13Guide"]
