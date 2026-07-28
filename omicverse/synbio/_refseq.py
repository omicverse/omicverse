r"""Reference sequences used as landmarks by the tutorials and the test suite.

Sequence-level predictors can only be judged against sequences whose behaviour
is already known, so the same handful keeps coming up: a protein with a
documented signal-peptide cleavage site, one that is famously soluble, one that
is famously not, and a family close enough for its variable positions to be
countable. Keeping them here rather than pasted into every notebook and test
means a tutorial cell stays one line and the two never drift apart.

These are public reference sequences (UniProt / GenBank), included because they
are small, stable and citable. Contrast :func:`~omicverse.synbio.match_enzymes`,
which deliberately bundles *no* enzyme database — a catalogue goes stale, a
landmark does not.

    >>> import omicverse as ov
    >>> seq = ov.synbio.reference_protein('phoA')
    >>> ov.synbio.predict_signal_peptide(seq).cleavage_site
    21
"""
from __future__ import annotations

from typing import Dict, List

from .._registry import register_function


#: Landmark proteins. Each entry documents *why* it is a landmark — a reference
#: sequence with no known expected behaviour is just a string.
REFERENCE_PROTEINS: Dict[str, Dict[str, str]] = {
    "phoA": {
        "organism": "Escherichia coli",
        "description": "Alkaline phosphatase, periplasmic. Its Sec/SPI signal "
                       "peptide is experimentally cleaved after residue 21 "
                       "(MKQSTIALALLPLLFTPVTKA | RTPEMPVLENR…), which makes it "
                       "the positive control for signal-peptide prediction.",
        "sequence": (
            "MKQSTIALALLPLLFTPVTKARTPEMPVLENRAAQGDITAPGGARRLTGDQTAALRDSLSD"
            "KPAKNIILLIGDGMGDSEITAARNYAEGAGGFFKGIDALPLTGQYTHYALNKKTGKPDYVT"
            "DSAASATAWSTGVKTYNGALGVDIHEKDHPTILEMAKA"),
    },
    "gfp": {
        "organism": "Aequorea victoria",
        "description": "Green fluorescent protein. Cytoplasmic, no signal "
                       "peptide, and famously well expressed and soluble in "
                       "E. coli — the positive control for solubility.",
        "sequence": (
            "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV"
            "TTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNR"
            "IELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQ"
            "QNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"),
    },
    "lacY": {
        "organism": "Escherichia coli",
        "description": "Lactose permease, N-terminal region. A polytopic "
                       "membrane protein — the negative control for solubility "
                       "and the positive one for transmembrane-helix detection.",
        "sequence": (
            "MYYLKNTNFWMFGLFFFFYFFIMGAYFPFFPIWLHDINHISKSDTGIIFAAISLFSLLFQP"
            "LFGLLSDKLGLRKYLLWIITGMLVMFAPFFIFIFGPLLQYNILVGSIVGGIYLGFCFNAGA"
            "PAVEAFIEKVSRRSNFEFGRARMFGCVGWALCASIVGI"),
    },
}

#: Lysozyme C from four birds, ~97% identical with ten variable positions —
#: close enough that an ancestral reconstruction should be confident everywhere
#: except at those ten columns, which makes the posterior checkable.
LYSOZYME_FAMILY: Dict[str, str] = {
    "chicken": (
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSR"
        "WWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQA"
        "WIRGCRL"),
    "turkey": (
        "KVYGRCELAAAMKRLGLDNYRGYSLGNWVCAAKFESNFNTHATNRNTDGSTDYGILQINSR"
        "WWCNDGRTPGSKNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQA"
        "WIRGCRL"),
    "quail": (
        "KVYGRCELAAAMKRLGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSR"
        "WWCNDGRTPGSKNLCHIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNHCKGTDVSA"
        "WIRGCRL"),
    "duck": (
        "KVYSRCELAAAMKRLGLDNYRGYSLGNWVCAAKFESNFNTHATNRNTDGSTDYGILQINSR"
        "WWCNDGRTPGSKNLCNIPCSALLSSDITASVNCAKKIVSDGDGMNAWVAWRNRCKGTDVSV"
        "WIRGCRL"),
}

#: Human GAPDH coding sequence, **opening 162 nt (54 codons)** — a real
#: eukaryotic CDS to harmonise into a bacterial host. Note the length: codon
#: harmonisation exists to reproduce ribosomal pausing at *domain boundaries*,
#: and 54 codons has none, so this is a mechanism demo rather than a design.
GAPDH_CDS = (
    "ATGGGGAAGGTGAAGGTCGGAGTCAACGGATTTGGTCGTATTGGGCGCCTGGTCACCAGGGC"
    "TGCTTTTAACTCTGGTAAAGTGGATATTGTTGCCATCAATGACCCCTTCATTGACCTCAACT"
    "ACATGGTTTACATGTTCCAATATGATTCCACCCATGGC")

#: The E. coli *rrnB* T1 terminator — the canonical strong intrinsic
#: terminator: GC-rich hairpin followed by a poly-U tract.
RRNB_T1 = "AAAAAAGGCCGCTTTTGCGGCCTTTTTTTAAA"

#: The same hairpin with the U-tract removed. Without it a hairpin only pauses
#: polymerase, so this should *not* read as a terminator.
HAIRPIN_NO_U_TRACT = "AAAAAAGGCCGCTTTTGCGGCCGCACGCACG"


@register_function(
    aliases=["reference_protein", "参考序列", "landmark_sequence",
             "reference_sequence", "标准序列"],
    category="synthetic_biology",
    description="取一条参考蛋白序列(phoA / gfp / lacY 等),用作序列级预测器的对照。每条都附带它为什么是地标 —— 比如 phoA 的 Sec 信号肽在第 21 位后被切,是信号肽预测的阳性对照。教程与测试共用同一份,避免两边漂移。Fetch a landmark reference protein sequence.",
    examples=[
        "seq = ov.synbio.reference_protein('phoA')",
        "ov.synbio.reference_protein('lacY', with_metadata=True)",
    ],
    related=["synbio.predict_signal_peptide", "synbio.predict_solubility",
             "synbio.predict_localization"],
    requires={},
    produces={},
)
def reference_protein(name: str, with_metadata: bool = False):
    """A landmark protein sequence by name.

    ``with_metadata=True`` returns the whole record — organism, description and
    sequence — so a notebook can print why the sequence is being used.
    """
    key = str(name)
    if key not in REFERENCE_PROTEINS:
        raise KeyError(
            f"没有名为 {name!r} 的参考序列。可用的有 "
            f"{sorted(REFERENCE_PROTEINS)}。")
    record = REFERENCE_PROTEINS[key]
    return dict(record) if with_metadata else record["sequence"]


@register_function(
    aliases=["reference_family", "参考序列家族", "landmark_family",
             "溶菌酶家族"],
    category="synthetic_biology",
    description="取一个参考序列家族(目前是四种鸟的溶菌酶 C,约 97% 一致、10 个可变位点),用于多序列比对、建树与祖先序列重构的演示与测试。Fetch a landmark sequence family for alignment / tree / ASR work.",
    examples=[
        "family = ov.synbio.reference_family('lysozyme')",
        "aln = ov.alignment.msa(family)",
    ],
    related=["alignment.msa", "alignment.protein_tree",
             "synbio.ancestral_reconstruction"],
    requires={},
    produces={},
)
def reference_family(name: str = "lysozyme") -> Dict[str, str]:
    """A family of related sequences, keyed by organism."""
    families = {"lysozyme": LYSOZYME_FAMILY}
    if name not in families:
        raise KeyError(f"没有名为 {name!r} 的家族。可用的有 {sorted(families)}。")
    return dict(families[name])


__all__ = ["reference_protein", "reference_family", "REFERENCE_PROTEINS",
           "LYSOZYME_FAMILY", "GAPDH_CDS", "RRNB_T1", "HAIRPIN_NO_U_TRACT"]
