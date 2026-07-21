r"""CRISPR guide-RNA design & genome editing.

Design and score guide RNAs, screen off-targets, and plan edits:

* :func:`design_grnas` — scan a target for PAM sites, extract protospacers,
  score on-target efficiency, flag poly-T terminators, rank.
* :func:`offtarget_search` — find near-matches in a background sequence and
  score them with a seed-weighted mismatch penalty (CRISPOR/CFD-style).
* :func:`base_editor_window` — editable positions & target bases for ABE / CBE
  base editors.
* :func:`hdr_arms` — homology arms for HDR knock-in around a cut site.

Supports SpCas9 (``NGG``), SaCas9 (``NNGRRT``) and Cas12a/Cpf1 (5' ``TTTV``).
The efficiency and off-target scores are **transparent heuristics** (GC / poly-T
/ seed-weighted mismatches) — good for ranking; for calibrated scores use
Azimuth (on-target) and the Doench-2016 CFD matrix / CRISPOR (off-target).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .._registry import register_function

_COMP = str.maketrans("ACGTN", "TGCAN")
_IUPAC = {
    "N": "ACGT", "R": "AG", "Y": "CT", "V": "ACG", "H": "ACT", "D": "AGT",
    "B": "CGT", "W": "AT", "S": "CG", "K": "GT", "M": "AC", "A": "A", "C": "C",
    "G": "G", "T": "T",
}

# enzyme -> (PAM motif, protospacer length, PAM at 3' end?)
ENZYMES = {
    "SpCas9":  ("NGG", 20, True),
    "SaCas9":  ("NNGRRT", 21, True),
    "Cas12a":  ("TTTV", 23, False),   # Cpf1: 5' PAM
}


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _pam_regex(pam: str) -> "re.Pattern":
    import re
    return re.compile("".join(f"[{_IUPAC[b]}]" for b in pam.upper()))


@dataclass
class Guide:
    """A candidate guide RNA."""
    spacer: str           # protospacer (5'->3'), no PAM
    pam: str
    start: int            # 0-based position of protospacer start on the input
    strand: str           # '+' | '-'
    gc: float
    efficiency: float     # 0..1 heuristic on-target score
    poly_t: bool          # contains TTTT (Pol III terminator)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Guide({self.spacer} {self.pam} {self.strand}@{self.start} "
                f"GC={self.gc:.2f} eff={self.efficiency:.2f})")


def _efficiency(spacer: str) -> float:
    """Heuristic on-target efficiency in [0,1].

    Rewards mid-range GC (~40–60%), a G at the PAM-proximal position, and
    penalises poly-T stretches and extreme GC — the dominant, model-agnostic
    determinants of SpCas9 activity."""
    s = spacer.upper()
    gc = (s.count("G") + s.count("C")) / max(1, len(s))
    gc_term = 1.0 - min(1.0, abs(gc - 0.5) / 0.5)      # peak at 50%
    polyt_pen = 0.3 if "TTTT" in s else 0.0
    g20 = 0.1 if s.endswith("G") else 0.0              # PAM-proximal G helps
    homopoly = 0.15 if any(b * 5 in s for b in "ACGT") else 0.0
    score = 0.5 + 0.4 * (gc_term - 0.5) + g20 - polyt_pen - homopoly
    return float(max(0.0, min(1.0, score + 0.25)))


@register_function(
    aliases=["design_grnas", "gRNA设计", "向导RNA", "sgRNA设计", "crispr_guides",
             "guide_design", "CRISPR", "向导设计"],
    category="synthetic_biology",
    description="CRISPR gRNA 设计:在靶序列上扫描 PAM(SpCas9 NGG / SaCas9 / Cas12a),提取原型间隔序列,按 GC/poly-T/效率启发式打分并排序。Design & rank CRISPR guide RNAs for a target sequence.",
    examples=[
        "guides = ov.synbio.design_grnas(target_dna, enzyme='SpCas9')",
        "guides[0].spacer, guides[0].efficiency",
    ],
    related=["synbio.offtarget_search", "synbio.base_editor_window", "synbio.hdr_arms"],
    requires={},
    produces={},
)
def design_grnas(sequence: str, enzyme: str = "SpCas9",
                 pam: Optional[str] = None, min_gc: float = 0.2,
                 max_gc: float = 0.8, top_n: Optional[int] = None) -> List[Guide]:
    """Design guide RNAs targeting *sequence*.

    Scans both strands for PAM sites, extracts protospacers of the enzyme's
    length, filters on GC, and ranks by the heuristic efficiency score.
    """
    import re
    seq = sequence.upper().replace("U", "T")
    if enzyme not in ENZYMES and pam is None:
        raise ValueError(f"未知 enzyme='{enzyme}';可用 {list(ENZYMES)} 或传 pam=")
    motif, plen, three_prime = ENZYMES.get(enzyme, (pam or "NGG", 20, True))
    if pam:
        motif = pam
    rx = _pam_regex(motif)
    plen_pam = len(motif)

    guides: List[Guide] = []
    for strand, s in (("+", seq), ("-", _revcomp(seq))):
        for m in rx.finditer(s):
            p = m.start()
            if three_prime:                # PAM is 3' of protospacer
                sp_start, sp_end = p - plen, p
                pam_seq = s[p:p + plen_pam]
            else:                          # Cas12a: PAM is 5'
                sp_start, sp_end = p + plen_pam, p + plen_pam + plen
                pam_seq = s[p:p + plen_pam]
            if sp_start < 0 or sp_end > len(s):
                continue
            spacer = s[sp_start:sp_end]
            if len(spacer) < plen:
                continue
            gc = (spacer.count("G") + spacer.count("C")) / plen
            if not (min_gc <= gc <= max_gc):
                continue
            # map back to + strand coordinate for reporting
            start = sp_start if strand == "+" else len(seq) - sp_end
            guides.append(Guide(
                spacer=spacer, pam=pam_seq, start=start, strand=strand, gc=gc,
                efficiency=_efficiency(spacer), poly_t=("TTTT" in spacer)))

    guides.sort(key=lambda g: g.efficiency, reverse=True)
    return guides[:top_n] if top_n else guides


@dataclass
class OffTarget:
    site: str
    start: int
    strand: str
    mismatches: int
    cfd: float          # 0..1, higher = more likely cut

    def __repr__(self) -> str:  # pragma: no cover
        return f"OffTarget(@{self.start}{self.strand} mm={self.mismatches} cfd={self.cfd:.3f})"


@register_function(
    aliases=["offtarget_search", "脱靶搜索", "off_target", "脱靶", "cfd",
             "脱靶打分", "特异性"],
    category="synthetic_biology",
    description="脱靶搜索:在背景序列中找 gRNA 的近似匹配位点,用种子区加权的错配罚分(CFD 风格)给出脱靶可能性打分。Off-target search + seed-weighted (CFD-style) scoring.",
    examples=[
        "hits = ov.synbio.offtarget_search(guide.spacer, genome_seq)",
    ],
    related=["synbio.design_grnas"],
    requires={},
    produces={},
)
def offtarget_search(spacer: str, background: str, max_mismatches: int = 4,
                     pam: str = "NGG") -> List[OffTarget]:
    """Find and score potential off-target sites for *spacer* in *background*.

    A seed-weighted mismatch model: mismatches in the PAM-proximal seed
    (last ~10 nt) are penalised far more than distal ones — the CFD/CRISPOR
    principle. Returns hits with ≤ ``max_mismatches`` mismatches, ranked by
    predicted cutting likelihood (``cfd`` in [0,1])."""
    import re
    sp = spacer.upper().replace("U", "T")
    L = len(sp)
    bg = background.upper().replace("U", "T")
    rx = _pam_regex(pam)

    # position weights: PAM-proximal (3') end weighted heavily.
    weights = [0.2 + 0.8 * (i / (L - 1)) for i in range(L)]  # 0.2..1.0 toward PAM

    hits: List[OffTarget] = []
    for strand, s in (("+", bg), ("-", _revcomp(bg))):
        for i in range(0, len(s) - L - len(pam) + 1):
            cand = s[i:i + L]
            pam_seq = s[i + L:i + L + len(pam)]
            if not rx.fullmatch(pam_seq):
                continue
            mism = [j for j in range(L) if cand[j] != sp[j]]
            if len(mism) > max_mismatches:
                continue
            # CFD-style: product of per-mismatch retention (1 - weight*penalty)
            cfd = 1.0
            for j in mism:
                cfd *= max(0.0, 1.0 - weights[j] * 0.75)
            hits.append(OffTarget(site=cand, start=i, strand=strand,
                                  mismatches=len(mism), cfd=float(cfd)))
    hits.sort(key=lambda h: h.cfd, reverse=True)
    return hits


@register_function(
    aliases=["base_editor_window", "碱基编辑", "base_editing", "ABE", "CBE",
             "碱基编辑窗口", "编辑窗口"],
    category="synthetic_biology",
    description="碱基编辑窗口:给定 gRNA 与编辑器(ABE:A→G / CBE:C→T),报告可编辑窗口内的目标碱基与预期产物。Base-editor (ABE/CBE) editable window + target bases.",
    examples=["ov.synbio.base_editor_window(guide.spacer, editor='ABE')"],
    related=["synbio.design_grnas"],
    requires={},
    produces={},
)
def base_editor_window(spacer: str, editor: str = "ABE",
                       window: Tuple[int, int] = (4, 8)) -> Dict:
    """Report editable target bases within a base-editor's activity window.

    Positions are 1-based from the PAM-distal end of the protospacer. ABE edits
    A→G, CBE edits C→T; the canonical window is protospacer positions 4–8."""
    sp = spacer.upper().replace("U", "T")
    target = "A" if editor.upper() == "ABE" else "C"
    product = "G" if editor.upper() == "ABE" else "T"
    lo, hi = window
    edits = []
    for pos in range(lo, hi + 1):
        if pos - 1 < len(sp) and sp[pos - 1] == target:
            edits.append({"position": pos, "from": target, "to": product})
    return {"editor": editor.upper(), "window": window, "target_base": target,
            "n_edits": len(edits), "edits": edits,
            "edited_spacer": _apply_edits(sp, edits, product, lo, hi)}


def _apply_edits(sp, edits, product, lo, hi):
    chars = list(sp)
    for e in edits:
        chars[e["position"] - 1] = product
    return "".join(chars)


@register_function(
    aliases=["hdr_arms", "同源臂", "HDR", "homology_arms", "敲入", "knock_in",
             "同源重组臂"],
    category="synthetic_biology",
    description="HDR 同源臂设计:围绕 Cas 切割位点为敲入/敲除设计左右同源臂(可插入 payload),用于同源定向修复。Design HDR homology arms around a cut site (optional insert).",
    examples=[
        "arms = ov.synbio.hdr_arms(locus_dna, cut_site=500, arm_length=500, insert='ATG...')",
    ],
    related=["synbio.design_grnas"],
    requires={},
    produces={},
)
def hdr_arms(sequence: str, cut_site: int, arm_length: int = 500,
             insert: str = "") -> Dict[str, str]:
    """Design left/right homology arms around *cut_site* for HDR.

    Returns the two arms and the assembled donor (left + insert + right)."""
    seq = sequence.upper()
    left = seq[max(0, cut_site - arm_length):cut_site]
    right = seq[cut_site:cut_site + arm_length]
    return {"left_arm": left, "right_arm": right, "insert": insert.upper(),
            "donor": left + insert.upper() + right,
            "left_len": len(left), "right_len": len(right)}


__all__ = ["design_grnas", "offtarget_search", "base_editor_window", "hdr_arms",
           "Guide", "OffTarget", "ENZYMES"]
