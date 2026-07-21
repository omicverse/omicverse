r"""Layer C — DNA design: codon optimisation (DNAchisel) and PCR primer design
(primer3).

Given a protein or coding sequence, produce a host-optimised DNA sequence
respecting codon usage, GC content and forbidden restriction sites; then design
validated primer pairs to amplify it.  Both are CPU-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from .._registry import register_function

_AA = set("ACDEFGHIKLMNPQRSTVWY*")
_DNA = set("ACGT")

# minimal standard codon table (one representative codon per amino acid) used
# only to back-translate a protein when the user passes amino acids.
_BACKTRANSLATE = {
    "A": "GCT", "R": "CGT", "N": "AAT", "D": "GAT",
    "C": "TGT", "Q": "CAA", "E": "GAA", "G": "GGT", "H": "CAT", "I": "ATT",
    "L": "CTG", "K": "AAA", "M": "ATG", "F": "TTT", "P": "CCG", "S": "AGC",
    "T": "ACC", "W": "TGG", "Y": "TAT", "V": "GTG", "*": "TAA",
}

# map friendly host names to python_codon_tables identifiers understood by
# DNAchisel's CodonOptimize. The bundled tables use these short names; a NCBI
# taxid is accepted for species without a bundled table (fetched online).
_HOST_TAXID = {
    "e_coli": "e_coli",
    "s_cerevisiae": "s_cerevisiae",
    "h_sapiens": "h_sapiens",
    "b_subtilis": "b_subtilis",
    "c_glutamicum": "1718",  # Corynebacterium glutamicum (no bundled table)
}


def _is_protein(seq: str) -> bool:
    s = seq.upper().replace("*", "")
    return bool(s) and set(s) <= (_AA - {"*"}) and not set(s) <= _DNA


def _backtranslate(protein: str) -> str:
    return "".join(_BACKTRANSLATE[a] for a in protein.upper())


@dataclass
class CodonResult:
    """Output of :func:`codon_optimize`."""

    sequence: str
    host: str
    n_edits: int = 0
    gc_content: float = 0.0
    constraints_ok: bool = True
    report: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        return (f"CodonResult(host={self.host!r}, len={len(self.sequence)}, "
                f"edits={self.n_edits}, GC={self.gc_content:.2f}, "
                f"constraints_ok={self.constraints_ok})")


@register_function(
    aliases=[
        "codon_optimize", "密码子优化", "密码子", "codon_optimization",
        "codon", "宿主优化",
    ],
    category="synthetic_biology",
    description="密码子优化:把 AA/DNA 序列按宿主 (e_coli/酵母/人等) 密码子偏好优化,并约束 GC、去除酶切位点。Host codon-optimize a protein/DNA sequence with DNAchisel.",
    examples=[
        "res = ov.synbio.codon_optimize('MKTAYIAKQR', host='e_coli')",
        "res.sequence  # optimised DNA",
    ],
    related=["synbio.design_primers", "synbio.inverse_design"],
    requires={},
    produces={},
)
def codon_optimize(
    seq: str,
    host: str = "e_coli",
    avoid_enzymes: Optional[List[str]] = None,
    gc_min: float = 0.30,
    gc_max: float = 0.70,
    gc_window: int = 50,
) -> CodonResult:
    """Codon-optimise *seq* for *host*.

    Parameters
    ----------
    seq
        Amino-acid sequence or a coding DNA sequence.  Amino-acid input is
        back-translated first.
    host
        One of ``e_coli`` / ``s_cerevisiae`` / ``h_sapiens`` /
        ``c_glutamicum`` / ``b_subtilis`` (or any DNAchisel species name).
    avoid_enzymes
        Restriction-enzyme names whose sites should be removed
        (e.g. ``["EcoRI", "BsaI"]``).
    gc_min, gc_max, gc_window
        GC-content bounds enforced over a sliding window.

    Returns
    -------
    CodonResult
    """
    try:
        from dnachisel import (
            DnaOptimizationProblem, reverse_translate,
            CodonOptimize, EnforceGCContent, AvoidPattern,
            EnforceTranslation,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.synbio.codon_optimize 需要 DNAchisel。请 pip install "
            "'omicverse[synbio]' (或 pip install dnachisel)。"
        ) from exc

    species = _HOST_TAXID.get(host, host)

    if _is_protein(seq):
        try:
            dna = reverse_translate(seq.replace("*", ""))
        except Exception:
            dna = _backtranslate(seq)
        protein_constraint = True
    else:
        dna = seq.upper()
        protein_constraint = len(dna) % 3 == 0

    constraints = [
        EnforceGCContent(mini=gc_min, maxi=gc_max, window=gc_window),
    ]
    if protein_constraint:
        constraints.append(EnforceTranslation())
    for enz in (avoid_enzymes or []):
        try:
            from dnachisel.builtin_specifications import EnzymeSitePattern
            constraints.append(AvoidPattern(EnzymeSitePattern(enz)))
        except Exception:
            # fall back to treating the string as a raw / IUPAC pattern
            constraints.append(AvoidPattern(enz))

    problem = DnaOptimizationProblem(
        sequence=dna,
        constraints=constraints,
        objectives=[CodonOptimize(species=species)],
        logger=None,
    )
    problem.resolve_constraints()
    problem.optimize()

    final = problem.sequence
    gc = (final.count("G") + final.count("C")) / max(len(final), 1)
    n_edits = sum(1 for a, b in zip(dna, final) if a != b)
    ok = problem.all_constraints_pass()
    return CodonResult(
        sequence=final, host=host, n_edits=n_edits, gc_content=gc,
        constraints_ok=ok,
        report=f"{n_edits} nt changed vs input back-translation; "
               f"constraints_pass={ok}",
    )


@dataclass
class PrimerPair:
    """A single primer pair from :func:`design_primers`."""

    left: str
    right: str
    left_tm: float
    right_tm: float
    product_size: int
    penalty: float = 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return (f"PrimerPair(product={self.product_size}bp, "
                f"Tm={self.left_tm:.1f}/{self.right_tm:.1f}, "
                f"L={self.left}, R={self.right})")


@register_function(
    aliases=[
        "design_primers", "引物设计", "引物", "primer_design", "pcr_primers",
        "primer3",
    ],
    category="synthetic_biology",
    description="PCR 引物设计:对 DNA 序列用 primer3 设计并打分引物对 (Tm/GC/产物长度约束)。Design validated PCR primer pairs for a DNA sequence with primer3.",
    examples=[
        "pairs = ov.synbio.design_primers(dna, n_return=3)",
        "pairs[0].left, pairs[0].right",
    ],
    related=["synbio.codon_optimize"],
    requires={},
    produces={},
)
def design_primers(
    seq: str,
    product_size_range=(100, 1000),
    opt_tm: float = 60.0,
    n_return: int = 5,
    opt_size: int = 20,
) -> List[PrimerPair]:
    """Design PCR primer pairs for *seq* with primer3.

    Returns up to ``n_return`` :class:`PrimerPair` objects sorted by primer3's
    penalty (best first)."""
    try:
        import primer3
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.synbio.design_primers 需要 primer3-py。请 pip install "
            "'omicverse[synbio]' (或 pip install primer3-py)。"
        ) from exc

    seq = seq.upper().strip()
    seq_args = {"SEQUENCE_ID": "synbio", "SEQUENCE_TEMPLATE": seq}
    global_args = {
        "PRIMER_OPT_SIZE": opt_size,
        "PRIMER_MIN_SIZE": max(15, opt_size - 5),
        "PRIMER_MAX_SIZE": min(35, opt_size + 7),
        "PRIMER_OPT_TM": opt_tm,
        "PRIMER_MIN_TM": opt_tm - 3.0,
        "PRIMER_MAX_TM": opt_tm + 3.0,
        "PRIMER_MIN_GC": 30.0,
        "PRIMER_MAX_GC": 70.0,
        "PRIMER_NUM_RETURN": n_return,
        "PRIMER_PRODUCT_SIZE_RANGE": [list(product_size_range)],
    }
    # primer3-py exposes either design_primers (new) or designPrimers (old).
    fn = getattr(primer3, "design_primers", None) or getattr(primer3, "designPrimers")
    res = fn(seq_args, global_args)

    pairs: List[PrimerPair] = []
    n = res.get("PRIMER_PAIR_NUM_RETURNED", 0)
    for i in range(n):
        pairs.append(PrimerPair(
            left=res[f"PRIMER_LEFT_{i}_SEQUENCE"],
            right=res[f"PRIMER_RIGHT_{i}_SEQUENCE"],
            left_tm=float(res[f"PRIMER_LEFT_{i}_TM"]),
            right_tm=float(res[f"PRIMER_RIGHT_{i}_TM"]),
            product_size=int(res[f"PRIMER_PAIR_{i}_PRODUCT_SIZE"]),
            penalty=float(res.get(f"PRIMER_PAIR_{i}_PENALTY", 0.0)),
        ))
    return pairs


__all__ = ["codon_optimize", "design_primers", "CodonResult", "PrimerPair"]
