r"""Combinatorial library & directed-evolution design.

Design the *libraries* you screen, and let a sequence model guide them:

* :func:`degenerate_codon` — the minimal IUPAC codon covering a target amino-acid
  set (e.g. NNK for all 20), with its off-target load.
* :func:`saturation_library` — size / diversity of a site-saturation library
  (NNK / NNS / NDT …) over chosen positions.
* :func:`dms_library` — an explicit deep-mutational-scanning library (all single
  substitutions, or a given position set).
* :func:`ml_guided_design` — model-guided directed evolution: score single
  mutations with an ESM saturation scan, build an additive fitness surrogate,
  and greedily combine beneficial mutations into ranked multi-mutant designs.
"""
from __future__ import annotations

import itertools

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set

from .._registry import register_function

_AA20 = list("ACDEFGHIKLMNPQRSTVWY")
_IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "AG", "Y": "CT",
          "S": "CG", "W": "AT", "K": "GT", "M": "AC", "B": "CGT", "D": "AGT",
          "H": "ACT", "V": "ACG", "N": "ACGT"}

# standard codon table (DNA) -> amino acid
_CODON = {}
_bases = "TCAG"
_aas = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
for i, a in enumerate(_aas):
    _CODON["".join(_bases[(i >> s) & 3] for s in (4, 2, 0))] = a


def _expand(deg: str) -> List[str]:
    return ["".join(c) for c in product(*[_IUPAC[b] for b in deg])]


def _codon_aas(deg: str) -> List[str]:
    return [_CODON[c] for c in _expand(deg)]


@dataclass
class DegenerateCodon:
    codon: str                 # IUPAC degenerate codon (e.g. 'NNK')
    covers: List[str]          # amino acids encoded
    n_codons: int              # DNA codons in the mixture
    missing: List[str]         # requested AAs not covered
    off_target: List[str]      # AAs encoded but not requested
    has_stop: bool

    def __repr__(self) -> str:  # pragma: no cover
        return (f"DegenerateCodon({self.codon}: {self.n_codons} codons -> "
                f"{len(self.covers)} AA, missing={self.missing})")


@register_function(
    aliases=["degenerate_codon", "简并密码子", "degenerate", "NNK", "密码子简并",
             "codon_degeneracy"],
    category="synthetic_biology",
    description="简并密码子设计:为一组目标氨基酸找覆盖它们的最优 IUPAC 简并密码子(如 20 种→NNK),报告脱靶氨基酸与终止子。Minimal IUPAC degenerate codon covering a target amino-acid set.",
    examples=[
        "ov.synbio.degenerate_codon(list('ACDEFGHIKLMNPQRSTVWY'))  # -> NNK",
        "ov.synbio.degenerate_codon(['F','Y','W'])                 # aromatics",
    ],
    related=["synbio.saturation_library", "synbio.dms_library"],
    requires={},
    produces={},
)
def degenerate_codon(amino_acids: Sequence[str]) -> DegenerateCodon:
    """Find the degenerate codon best covering *amino_acids*.

    Searches all IUPAC degeneracies at each of the 3 positions and picks the
    codon that covers every requested amino acid with the fewest off-target
    residues and codons (and no stop if avoidable)."""
    want = set(a.upper() for a in amino_acids)
    best = None
    codes = list(_IUPAC.keys())
    for p1 in codes:
        for p2 in codes:
            for p3 in codes:
                deg = p1 + p2 + p3
                aas = _codon_aas(deg)
                aset = set(aas)
                if not want <= (aset - {"*"}):
                    continue
                stop = "*" in aas
                off = sorted((aset - {"*"}) - want)
                score = (len(off), len(_expand(deg)), int(stop))
                if best is None or score < best[0]:
                    best = (score, deg, sorted(aset - {"*"}), off, stop)
    if best is None:
        # nothing covers all -> return NNN as a fallback with the missing set
        deg = "NNN"
        aas = set(_codon_aas(deg)) - {"*"}
        return DegenerateCodon("NNN", sorted(aas), 64,
                               sorted(want - aas), sorted(aas - want), True)
    _, deg, covers, off, stop = best
    return DegenerateCodon(codon=deg, covers=covers, n_codons=len(_expand(deg)),
                           missing=sorted(want - set(covers)), off_target=off,
                           has_stop=stop)


@register_function(
    aliases=["saturation_library", "饱和文库", "site_saturation", "位点饱和",
             "library_size", "文库设计"],
    category="synthetic_biology",
    description="位点饱和文库:对给定位点用某简并方案(NNK/NNS/NDT)设计文库,报告 DNA 多样性、蛋白多样性与建议筛选规模。Site-saturation library size/diversity over positions.",
    examples=[
        "ov.synbio.saturation_library(n_positions=3, scheme='NNK')",
    ],
    related=["synbio.degenerate_codon", "synbio.dms_library"],
    requires={},
    produces={},
)
def saturation_library(n_positions: int, scheme: str = "NNK") -> Dict:
    """Diversity of a site-saturation library.

    Returns DNA diversity (codons^positions), protein diversity (unique AA
    combinations), stop-codon fraction, and the library size to screen for ~95%
    coverage (≈ 3× the DNA diversity)."""
    scheme = scheme.upper()
    aas = _codon_aas(scheme)
    n_dna = len(_expand(scheme))
    n_prot = len(set(a for a in aas if a != "*"))
    stop_frac = aas.count("*") / len(aas)
    dna_div = n_dna ** n_positions
    prot_div = n_prot ** n_positions
    return {"scheme": scheme, "positions": n_positions,
            "codons_per_site": n_dna, "aa_per_site": n_prot,
            "stop_fraction_per_site": round(stop_frac, 3),
            "dna_diversity": dna_div, "protein_diversity": prot_div,
            "screen_for_95pct": int(round(3 * dna_div))}


@register_function(
    aliases=["dms_library", "深度突变文库", "deep_mutational_scanning",
             "DMS文库", "突变文库", "single_mutant_library"],
    category="synthetic_biology",
    description="深度突变扫描(DMS)文库:枚举序列(指定位点或全长)的所有单点氨基酸突变体,给出文库清单与规模。Explicit deep-mutational-scanning single-mutant library.",
    examples=[
        "lib = ov.synbio.dms_library(seq, positions=[10,20,30])",
    ],
    related=["synbio.variant_effect", "synbio.saturation_library"],
    requires={},
    produces={},
)
def dms_library(seq: str, positions: Optional[Sequence[int]] = None) -> Dict:
    """Enumerate all single amino-acid substitutions (a DMS library).

    Returns the list of mutations (``A23V`` style, 1-based) and the size."""
    s = seq.upper()
    positions = list(positions) if positions is not None else list(range(1, len(s) + 1))
    muts = []
    for p in positions:
        wt = s[p - 1]
        for alt in _AA20:
            if alt != wt:
                muts.append(f"{wt}{p}{alt}")
    return {"n_positions": len(positions), "n_variants": len(muts),
            "mutations": muts}


@dataclass
class DesignedVariant:
    mutations: List[str]
    sequence: str
    predicted_score: float

    def __repr__(self) -> str:  # pragma: no cover
        return f"DesignedVariant({'+'.join(self.mutations) or 'WT'}: {self.predicted_score:.2f})"


@register_function(
    aliases=["ml_guided_design", "机器学习引导设计", "ml_directed_evolution",
             "定向进化设计", "guided_evolution", "AI引导进化"],
    category="synthetic_biology",
    description="机器学习引导定向进化:用 ESM 饱和扫描给单点突变打分,构建加性适应度代理,贪心组合有利突变 → 排序的多点突变设计。ESM-guided combinatorial variant design.",
    examples=[
        "designs = ov.synbio.ml_guided_design(seq, n_mutations=3, n_designs=10)",
    ],
    related=["synbio.variant_effect", "synbio.dms_library", "synbio.protein_embed"],
    requires={},
    produces={},
)
def ml_guided_design(seq: str, n_mutations: int = 3, n_designs: int = 10,
                     positions: Optional[Sequence[int]] = None,
                     model: str = "esm1v", device: Optional[str] = None,
                     min_single_score: float = 0.0) -> List[DesignedVariant]:
    """Model-guided directed-evolution design.

    Scores single mutations with an ESM saturation scan (:func:`variant_effect`),
    keeps the beneficial ones (score > ``min_single_score``), and greedily
    combines up to ``n_mutations`` non-conflicting mutations by additive
    predicted score — an epistasis-free surrogate for a first design round.

    Returns ``n_designs`` ranked :class:`DesignedVariant` s.
    """
    from ._variant import variant_effect

    df = variant_effect(seq, model=model, device=device, verbose=False)
    if positions is not None:
        df = df[df["pos"].isin(list(positions))]
    good = df[df["score"] > min_single_score].sort_values("score", ascending=False)

    # greedy: take top beneficial mutations at distinct positions
    chosen = []
    used_pos: Set[int] = set()
    for _, row in good.iterrows():
        if row["pos"] in used_pos:
            continue
        chosen.append((f"{row['wt']}{int(row['pos'])}{row['mut']}",
                       int(row["pos"]), row["mut"], float(row["score"])))
        used_pos.add(int(row["pos"]))
        if len(chosen) >= n_mutations * n_designs:
            break

    # build combinatorial designs of increasing order, ranked by additive score
    designs: List[DesignedVariant] = []
    for k in range(1, n_mutations + 1):
        combo = chosen[:k]
        if not combo:
            break
        muts = [c[0] for c in combo]
        s = list(seq)
        for _, pos, alt, _score in combo:
            s[pos - 1] = alt
        total = sum(c[3] for c in combo)
        designs.append(DesignedVariant(mutations=muts, sequence="".join(s),
                                       predicted_score=float(total)))
    # Fill out n_designs with *distinct* combinations. Emitting single-position
    # designs from index 0 re-emitted the k=1 design the loop above had already
    # produced, so n_designs=6 returned 5 unique designs — every one of them a
    # nested subset of the top design, and the duplicate double-weighted its
    # mutation in any downstream sequence logo.
    seen = {frozenset(d.mutations) for d in designs}
    for size in range(1, n_mutations + 1):
        if len(designs) >= n_designs:
            break
        for combo in itertools.combinations(chosen, size):
            if len(designs) >= n_designs:
                break
            key = frozenset(c[0] for c in combo)
            if key in seen:
                continue
            if len({c[1] for c in combo}) != len(combo):   # one alt per position
                continue
            seen.add(key)
            s = list(seq)
            for _m, pos, alt, _score in combo:
                s[pos - 1] = alt
            designs.append(DesignedVariant(
                mutations=[c[0] for c in combo], sequence="".join(s),
                predicted_score=float(sum(c[3] for c in combo))))
    designs.sort(key=lambda d: d.predicted_score, reverse=True)
    return designs[:n_designs]


__all__ = ["degenerate_codon", "saturation_library", "dms_library",
           "ml_guided_design", "DegenerateCodon", "DesignedVariant"]
