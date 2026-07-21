r"""RNA secondary structure & thermodynamics (ViennaRNA).

RNA folding controls translation (RBS accessibility), riboswitches, sRNA
regulation and sgRNA activity.  Thin, well-typed wrappers over ViennaRNA:

* :func:`rna_fold` — minimum-free-energy structure (dot-bracket) + ΔG.
* :func:`rna_accessibility` — opening energy of a region (e.g. how exposed an
  RBS or a target site is) via the partition function.
* :func:`rna_duplex` — hybridisation energy + structure between two RNAs
  (antisense / sRNA : mRNA, sgRNA : DNA target).
* :func:`gc_content` — quick GC helper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

from .._registry import register_function


def _rna(fn: str):
    try:
        import RNA
        return RNA
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"ov.synbio.{fn} 需要 ViennaRNA。请 pip install ViennaRNA "
            "(或 pip install 'omicverse[synbio]')。"
        ) from exc


@dataclass
class RNAStructure:
    sequence: str
    structure: str    # dot-bracket
    mfe: float        # kcal/mol

    def __repr__(self) -> str:  # pragma: no cover
        return f"RNAStructure(len={len(self.sequence)}, mfe={self.mfe:.2f} kcal/mol)"


@register_function(
    aliases=["rna_fold", "RNA折叠", "二级结构", "rna_structure", "mfe",
             "rna二级结构", "RNAfold"],
    category="synthetic_biology",
    description="RNA 二级结构预测:计算最小自由能结构(点括号表示)与 ΔG(ViennaRNA)。RNA secondary-structure MFE folding → dot-bracket + ΔG.",
    examples=["s = ov.synbio.rna_fold('GGGAAACCC'); s.structure, s.mfe"],
    related=["synbio.rna_accessibility", "synbio.rna_duplex"],
    requires={},
    produces={},
)
def rna_fold(seq: str) -> RNAStructure:
    """MFE secondary structure of an RNA (or DNA, treated as RNA)."""
    RNA = _rna("rna_fold")
    s = seq.upper().replace("T", "U")
    structure, mfe = RNA.fold(s)
    return RNAStructure(sequence=s, structure=structure, mfe=float(mfe))


@register_function(
    aliases=["rna_accessibility", "可及性", "RBS可及性", "accessibility",
             "开放能", "位点可及性"],
    category="synthetic_biology",
    description="RNA 区域可及性:计算打开某段(如 RBS/靶位点)所需的自由能——值越小越暴露、越易被核糖体/sRNA/sgRNA 接近。Opening free energy (accessibility) of an RNA region.",
    examples=["dG = ov.synbio.rna_accessibility(utr, 20, 30)"],
    related=["synbio.rna_fold", "synbio.rbs_strength"],
    requires={},
    produces={},
)
def rna_accessibility(seq: str, start: int, end: int) -> float:
    """Free energy (kcal/mol) to keep region ``[start, end)`` single-stranded.

    Computed as the difference between the constrained ensemble energy (region
    forced open) and the unconstrained ensemble energy — a positive opening
    cost. Lower = more accessible."""
    RNA = _rna("rna_accessibility")
    s = seq.upper().replace("T", "U")

    fc = RNA.fold_compound(s)
    _, g_free = fc.pf()

    fc_c = RNA.fold_compound(s)
    constraint = "." * len(s)
    constraint = ("." * start + "x" * (end - start) +
                  "." * (len(s) - end))[:len(s)]
    fc_c.hc_add_from_db(constraint)
    _, g_open = fc_c.pf()
    return float(g_open - g_free)


@dataclass
class RNADuplex:
    energy: float
    structure: str

    def __repr__(self) -> str:  # pragma: no cover
        return f"RNADuplex(dG={self.energy:.2f} kcal/mol)"


@register_function(
    aliases=["rna_duplex", "RNA杂交", "rna_hybridization", "duplex",
             "sRNA靶向", "反义配对"],
    category="synthetic_biology",
    description="RNA-RNA 杂交能:两条 RNA(反义/sRNA:mRNA、sgRNA:靶标)的最优互补结合自由能与结构(ViennaRNA duplexfold)。RNA:RNA hybridisation energy + structure.",
    examples=["d = ov.synbio.rna_duplex(srna, mrna_target); d.energy"],
    related=["synbio.rna_fold", "synbio.rna_accessibility"],
    requires={},
    produces={},
)
def rna_duplex(a: str, b: str) -> RNADuplex:
    """Best hybridisation between two RNAs (antisense / sRNA : target)."""
    RNA = _rna("rna_duplex")
    sa = a.upper().replace("T", "U")
    sb = b.upper().replace("T", "U")
    dup = RNA.duplexfold(sa, sb)
    return RNADuplex(energy=float(dup.energy), structure=dup.structure)


@register_function(
    aliases=["gc_content", "GC含量", "gc"],
    category="synthetic_biology",
    description="GC 含量:序列的 GC 比例(0–1)。GC fraction of a sequence.",
    examples=["ov.synbio.gc_content('ATGCGC')"],
    related=["synbio.rna_fold", "synbio.codon_optimize"],
    requires={},
    produces={},
)
def gc_content(seq: str) -> float:
    """GC fraction of a nucleotide sequence."""
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    return gc / max(1, len(s))


__all__ = ["rna_fold", "rna_accessibility", "rna_duplex", "gc_content",
           "RNAStructure", "RNADuplex"]
