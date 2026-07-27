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
    efficiency: float     # heuristic on-target score, always in [0,1]
    poly_t: bool          # contains TTTT (Pol III terminator)
    context: str = ""     # 30-mer context (4nt + 20nt spacer + PAM + 3nt) for rs3
    rs3_score: Optional[float] = None   # Rule Set 3 z-score; None if not scored

    def __repr__(self) -> str:  # pragma: no cover
        tail = "" if self.rs3_score is None else f" rs3={self.rs3_score:+.2f}"
        return (f"Guide({self.spacer} {self.pam} {self.strand}@{self.start} "
                f"GC={self.gc:.2f} eff={self.efficiency:.2f}{tail})")


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
    description="CRISPR gRNA 设计:扫描 PAM(SpCas9/SaCas9/Cas12a)提取原型间隔序列并排序。method='heuristic'(默认,无额外依赖)按 efficiency([0,1] GC/poly-T 启发式)排序;method='rs3' 额外填 rs3_score(真实 Rule Set 3 z 分数,Azimuth 继任者;首用自动下载 ~20MB 模型)并按它排序,efficiency 语义不变。Design & rank CRISPR guide RNAs (heuristic or Rule Set 3).",
    examples=[
        "guides = ov.synbio.design_grnas(target_dna, enzyme='SpCas9')",
        "guides[0].spacer, guides[0].efficiency",
        "rs3 = ov.synbio.design_grnas(target_dna, method='rs3'); rs3[0].rs3_score",
    ],
    related=["synbio.offtarget_search", "synbio.base_editor_window", "synbio.hdr_arms"],
    requires={},
    produces={},
)
def design_grnas(sequence: str, enzyme: str = "SpCas9",
                 pam: Optional[str] = None, min_gc: float = 0.2,
                 max_gc: float = 0.8, top_n: Optional[int] = None,
                 method: str = "heuristic") -> List[Guide]:
    """Design guide RNAs targeting *sequence*.

    Scans both strands for PAM sites, extracts protospacers of the enzyme's
    length and filters on GC.

    ``method='heuristic'`` (the default) ranks by :attr:`Guide.efficiency`, a
    transparent GC/poly-T score in ``[0,1]`` that needs no download at all.

    ``method='rs3'`` additionally fills :attr:`Guide.rs3_score` with the real
    Rule Set 3 z-score and ranks by *that*; ``efficiency`` keeps its heuristic
    meaning in both cases, so the two are never compared on the same scale.
    Guides too close to the sequence ends to yield a 30-mer context cannot be
    rs3-scored and are ranked last. rs3 cannot be a declared dependency (its
    stale pins would make the whole ``[synbio]`` extra unresolvable), so on
    first use it is fetched with ``--no-deps`` into
    ``$OMICOS_SYNBIO_WEIGHTS/rs3`` — ~20 MB, one network round-trip, see
    :mod:`omicverse.synbio._rs3`.
    """
    import re
    if method not in ("heuristic", "rs3"):
        raise ValueError(f"method must be one of ['heuristic', 'rs3'], got {method!r}")
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
            # 30-mer context for rs3: 4nt 5' + 20nt spacer + 3nt PAM + 3nt 3'
            if three_prime:
                ctx = s[sp_start - 4:sp_end + plen_pam + 3]
            else:
                ctx = s[p - 4:sp_end + 3]
            context = ctx if len(ctx) == 30 else ""
            # map back to + strand coordinate for reporting
            start = sp_start if strand == "+" else len(seq) - sp_end
            guides.append(Guide(
                spacer=spacer, pam=pam_seq, start=start, strand=strand, gc=gc,
                efficiency=_efficiency(spacer), poly_t=("TTTT" in spacer),
                context=context))

    if method == "rs3":
        _score_rs3(guides)
        # rs3 returns a z-score (unbounded, frequently negative) — it lives in
        # its own field and is *not* comparable to the heuristic [0,1] score.
        # Guides with no 30-mer context cannot be scored, so they rank last
        # rather than being mixed in on an incompatible scale.
        guides.sort(key=lambda g: (g.rs3_score is not None, g.rs3_score or 0.0),
                    reverse=True)
    else:
        guides.sort(key=lambda g: g.efficiency, reverse=True)
    return guides[:top_n] if top_n else guides


def _score_rs3(guides: List["Guide"]) -> None:
    """Attach the real Rule Set 3 score (DeWeirdt *et al.* 2021; the successor
    to Azimuth/Rule Set 2) to each guide's ``rs3_score``.

    The heuristic ``efficiency`` is left untouched — the two scores are on
    different scales. Guides lacking a full 30-mer context keep
    ``rs3_score=None``; no flanking sequence is invented to manufacture one."""
    scored = [g for g in guides if len(g.context) == 30]
    if not scored:
        return
    from ._rs3 import predict_rs3

    preds = predict_rs3([g.context for g in scored], sequence_tracr="Hsu2013")
    for g, p in zip(scored, preds):
        g.rs3_score = float(p)


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
                     pam: str = "NGG", method: str = "cfd") -> List[OffTarget]:
    """Find and score potential off-target sites for *spacer* in *background*.

    ``method='cfd'`` (default) uses the **real Doench-2016 CFD** scoring tables
    (vendored from CRISPOR) — the field-standard off-target metric.
    ``method='seed'`` is a dependency-free seed-weighted approximation.
    Returns hits with ≤ ``max_mismatches`` mismatches, ranked by predicted
    cutting likelihood (``cfd`` in [0,1]; PAM-proximal seed mismatches hurt most).
    """
    if method not in ("cfd", "seed"):
        raise ValueError(f"method must be one of ['cfd', 'seed'], got {method!r}")
    sp = spacer.upper().replace("U", "T")
    L = len(sp)
    bg = background.upper().replace("U", "T")
    rx = _pam_regex(pam)

    cfd_fn = None
    if method == "cfd":
        from ._cfd import cfd_score
        cfd_fn = cfd_score
    weights = [0.2 + 0.8 * (i / (L - 1)) for i in range(L)]  # seed model fallback

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
            if cfd_fn is not None:                 # real Doench-2016 CFD
                cfd = cfd_fn(sp, cand, pam_seq[-2:])
            else:                                  # seed-weighted approximation
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


@register_function(
    aliases=["plot_grna_efficiency", "gRNA效率图", "向导效率图", "guide_plot",
             "plot_guides", "gRNA分布图"],
    category="synthetic_biology",
    description="画候选 gRNA 沿靶序列的分布与效率:横轴位置、纵轴效率、正负链分色、点大小=GC。Plot candidate gRNAs along the target by position and efficiency.",
    examples=["ov.synbio.plot_grna_efficiency(guides, target_len=len(dna))"],
    related=["synbio.design_grnas", "synbio.plot_offtargets"],
    requires={},
    produces={},
)
def plot_grna_efficiency(guides: List["Guide"], target_len: Optional[int] = None,
                         ax=None):
    """Scatter candidate guides along the target: x = start, y = efficiency,
    colour = strand, size ∝ GC. Highlights where the best guides sit."""
    from ._plot import _mpl
    plt = _mpl()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
    else:
        fig = ax.figure
    for strand, col in (("+", "#377EB8"), ("-", "#E41A1C")):
        gs = [g for g in guides if g.strand == strand]
        if gs:
            ax.scatter([g.start for g in gs], [g.efficiency for g in gs],
                       s=[30 + 120 * g.gc for g in gs], c=col, alpha=0.7,
                       edgecolors="k", linewidths=0.3, label=f"{strand} strand")
    if guides:
        best = guides[0]
        ax.annotate("best", (best.start, best.efficiency),
                    textcoords="offset points", xytext=(4, 6), fontsize=8)
    if target_len:
        ax.set_xlim(0, target_len)
    ax.set_xlabel("protospacer start (bp on target)")
    ax.set_ylabel("on-target efficiency")
    ax.set_title(f"CRISPR guide landscape ({len(guides)} guides; size ∝ GC)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


@register_function(
    aliases=["plot_offtargets", "脱靶图", "脱靶热图", "offtarget_plot",
             "plot_off_targets", "错配图"],
    category="synthetic_biology",
    description="画脱靶位点的错配矩阵热图:每行一个候选脱靶,每列一个位点,标出错配位置并按 CFD 排序。Plot off-target mismatch matrix (sites × positions), ranked by CFD.",
    examples=["ov.synbio.plot_offtargets(offtargets, spacer=guide.spacer)"],
    related=["synbio.offtarget_search", "synbio.plot_grna_efficiency"],
    requires={},
    produces={},
)
def plot_offtargets(offtargets: List["OffTarget"], spacer: Optional[str] = None,
                    top_n: int = 15, ax=None):
    """Mismatch matrix of the top off-targets (rows = sites, cols = positions).

    Cells are dark where the off-target base differs from the guide; rows are
    ordered by CFD (most concerning first), with the CFD annotated on the right."""
    import numpy as np
    from ._plot import _mpl
    plt = _mpl()

    hits = offtargets[:top_n]
    if not hits:
        raise ValueError("没有脱靶位点可画。")
    L = len(hits[0].site)
    ref = (spacer or hits[0].site).upper()
    mat = np.zeros((len(hits), L))
    for i, h in enumerate(hits):
        for j in range(min(L, len(h.site))):
            mat[i, j] = 1.0 if h.site[j] != ref[j] else 0.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5, 0.28 * L), max(2.5, 0.32 * len(hits))))
    else:
        fig = ax.figure
    ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(0, L, max(1, L // 10)))
    ax.set_yticks(range(len(hits)))
    ax.set_yticklabels([f"cfd={h.cfd:.2f} (mm={h.mismatches})" for h in hits],
                       fontsize=7)
    ax.set_xlabel("protospacer position (PAM-distal → PAM-proximal)")
    ax.set_title("Off-target mismatches (dark = mismatch)")
    fig.tight_layout()
    return fig, ax


__all__ = ["design_grnas", "offtarget_search", "base_editor_window", "hdr_arms",
           "plot_grna_efficiency", "plot_offtargets",
           "Guide", "OffTarget", "ENZYMES"]
