r"""Regulatory-element strength & expression prediction.

Predict how strongly a construct is transcribed and translated, from sequence:

* :func:`rbs_strength` — translation-initiation rate from a 5'UTR, via a
  thermodynamic model (Salis-style): the Shine–Dalgarno : anti-SD (16S rRNA)
  hybridisation energy minus the cost of unfolding the mRNA around the start
  codon, computed with ViennaRNA.  Reported on a proportional (arbitrary-unit)
  scale, as the RBS Calculator does.
* :func:`promoter_strength` — σ70 promoter strength from how well the −10 /
  −35 boxes and spacer match the *E. coli* consensus.
* :func:`cai` — Codon Adaptation Index (translation-elongation proxy).
* :func:`predict_expression` — a relative protein-expression estimate combining
  promoter × RBS × CAI.

These are **transparent biophysical / consensus baselines** — good for ranking
and design, not calibrated absolute numbers.  For calibrated RBS predictions use
the Salis lab RBS Calculator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

from .._registry import register_function

# E. coli 16S rRNA 3' tail (anti-Shine-Dalgarno), 5'->3'.
_ANTI_SD = "ACCUCCUUA"
# sigma70 consensus boxes.
_MINUS10 = "TATAAT"
_MINUS35 = "TTGACA"
# ideal -35..-10 spacer.
_IDEAL_SPACER = 17
# Boltzmann factor (kcal/mol)^-1 at 37C, scaled as in RBS-calculator-like models.
_BETA = 0.45


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
class RBSResult:
    """Translation-initiation prediction for a 5'UTR + start codon."""
    dG_total: float
    dG_SD: float
    dG_mRNA: float
    initiation_rate: float   # arbitrary units, proportional to true rate

    def __repr__(self) -> str:  # pragma: no cover
        return (f"RBSResult(rate={self.initiation_rate:.1f} a.u., "
                f"dG_total={self.dG_total:.2f} kcal/mol)")


@register_function(
    aliases=["rbs_strength", "RBS强度", "核糖体结合位点", "翻译起始",
             "translation_initiation", "rbs_calculator", "SD序列"],
    category="synthetic_biology",
    description="RBS(核糖体结合位点)强度:用热力学模型(SD:anti-SD 杂交能 − mRNA 起始区解折叠)从 5'UTR 预测翻译起始速率。RBS / translation-initiation-rate prediction (Salis-style, ViennaRNA).",
    examples=[
        "r = ov.synbio.rbs_strength('AGGAGGACAACATG...')",
        "r.initiation_rate",
    ],
    related=["synbio.promoter_strength", "synbio.predict_expression"],
    requires={},
    produces={},
)
def rbs_strength(utr_and_cds: str, start: Optional[int] = None,
                 footprint: int = 35) -> RBSResult:
    """Predict translation-initiation rate for a 5'UTR + start codon.

    Parameters
    ----------
    utr_and_cds
        The 5'UTR followed by the start of the CDS (DNA or RNA). Should contain
        the ribosome-binding site and the start codon.
    start
        Index of the start codon (ATG/AUG). If ``None`` it is located
        automatically (first ATG/AUG after position 15).
    footprint
        Ribosomal footprint window (nt) used when folding the mRNA.

    Returns
    -------
    RBSResult
    """
    import math
    RNA = _rna("rbs_strength")

    seq = utr_and_cds.upper().replace("T", "U")
    if start is None:
        start = seq.find("AUG", 12)
        if start < 0:
            start = max(0, len(seq) // 2)

    # SD:anti-SD hybridisation over the region just 5' of the start codon.
    sd_region = seq[max(0, start - 20):start]
    duplex = RNA.duplexfold(sd_region, _ANTI_SD)
    dG_SD = float(duplex.energy)          # favourable (negative)

    # mRNA structure cost: folding energy of the ribosomal footprint around AUG.
    fp = seq[max(0, start - 20):min(len(seq), start + footprint - 20)]
    _, dG_mRNA = RNA.fold(fp)             # <=0; a strong hairpin blocks the RBS
    dG_mRNA = float(dG_mRNA)

    # total: SD binding must overcome mRNA unfolding.
    dG_total = dG_SD - dG_mRNA
    rate = 100.0 * math.exp(-_BETA * dG_total)
    return RBSResult(dG_total=dG_total, dG_SD=dG_SD, dG_mRNA=dG_mRNA,
                     initiation_rate=rate)


@register_function(
    aliases=["promoter_strength", "启动子强度", "promoter", "启动子",
             "sigma70", "启动子预测"],
    category="synthetic_biology",
    description="σ70 启动子强度:根据 −10/−35 框与间隔长度对大肠杆菌共识的匹配度,给出相对强度打分。σ70 promoter strength from -10/-35 consensus match.",
    examples=["s = ov.synbio.promoter_strength('...TTGACA...TATAAT...')"],
    related=["synbio.rbs_strength", "synbio.predict_expression"],
    requires={},
    produces={},
)
def promoter_strength(promoter: str) -> Dict[str, float]:
    """Relative σ70 promoter strength from −10 / −35 box quality.

    Scans for the best-matching −35 and −10 hexamers with a ~17 nt spacer and
    scores their similarity to the *E. coli* consensus. Returns a dict with the
    matched boxes, spacer, per-box scores, and a combined ``strength`` in
    ``[0, 1]`` (1 = perfect consensus)."""
    seq = promoter.upper().replace("U", "T")

    def _score(hexamer: str, consensus: str) -> float:
        return sum(a == b for a, b in zip(hexamer, consensus)) / len(consensus)

    best = {"strength": 0.0, "minus35": "", "minus10": "", "spacer": 0,
            "score35": 0.0, "score10": 0.0}
    for i in range(0, max(1, len(seq) - 12)):
        m35 = seq[i:i + 6]
        if len(m35) < 6:
            break
        for spacer in range(14, 21):
            j = i + 6 + spacer
            m10 = seq[j:j + 6]
            if len(m10) < 6:
                continue
            s35 = _score(m35, _MINUS35)
            s10 = _score(m10, _MINUS10)
            spacer_pen = 1.0 - abs(spacer - _IDEAL_SPACER) * 0.08
            strength = (0.4 * s35 + 0.6 * s10) * max(0.4, spacer_pen)
            if strength > best["strength"]:
                best.update(strength=strength, minus35=m35, minus10=m10,
                            spacer=spacer, score35=s35, score10=s10)
    return best


@register_function(
    aliases=["cai", "密码子适应指数", "codon_adaptation_index", "CAI"],
    category="synthetic_biology",
    description="密码子适应指数 (CAI):衡量 CDS 密码子对宿主高表达基因偏好的适配度(翻译延伸速率代理)。Codon Adaptation Index of a coding sequence.",
    examples=["ov.synbio.cai(dna_cds, host='e_coli')"],
    related=["synbio.codon_optimize", "synbio.predict_expression"],
    requires={},
    produces={},
)
def cai(cds: str, host: str = "e_coli") -> float:
    """Codon Adaptation Index of *cds* against a host's codon-usage table."""
    import math
    from ._codon import _HOST_TAXID
    try:
        import python_codon_tables as pct
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ov.synbio.cai 需要 python_codon_tables (随 dnachisel 安装)。") from exc

    table = pct.get_codons_table(_HOST_TAXID.get(host, host))
    seq = cds.upper().replace("U", "T")
    codons = [seq[i:i + 3] for i in range(0, len(seq) - 2, 3)]
    # per-amino-acid max usage -> relative adaptiveness w = freq/max_freq
    logs = []
    for c in codons:
        for aa, cod in table.items():
            if c in cod:
                fmax = max(cod.values())
                w = cod[c] / fmax if fmax > 0 else 0.0
                if w > 0:
                    logs.append(math.log(w))
                break
    return float(math.exp(sum(logs) / len(logs))) if logs else 0.0


@register_function(
    aliases=["predict_expression", "表达预测", "蛋白表达预测", "expression_prediction",
             "表达量"],
    category="synthetic_biology",
    description="相对蛋白表达预测:综合 启动子强度 × RBS 起始速率 × CAI,给出相对表达量估计(用于对比不同构建)。Relative protein-expression estimate = promoter × RBS × CAI.",
    examples=[
        "ov.synbio.predict_expression(promoter, utr_cds, cds)",
    ],
    related=["synbio.promoter_strength", "synbio.rbs_strength", "synbio.cai"],
    requires={},
    produces={},
)
def predict_expression(promoter: str, utr_cds: str, cds: Optional[str] = None,
                       host: str = "e_coli") -> Dict[str, float]:
    """Relative protein-expression estimate from a construct's parts.

    Combines transcription (promoter strength), translation initiation (RBS)
    and elongation (CAI) into a single relative score (a.u.)."""
    p = promoter_strength(promoter)["strength"]
    r = rbs_strength(utr_cds).initiation_rate
    c = cai(cds, host) if cds else 1.0
    return {"promoter": p, "rbs_rate": r, "cai": c,
            "relative_expression": p * r * c}


__all__ = ["rbs_strength", "promoter_strength", "cai", "predict_expression",
           "RBSResult"]
