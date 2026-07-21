r"""Publication-quality sequence & construct visualisation.

The bar charts elsewhere are for *numbers*; DNA deserves proper genetic maps.
These helpers use `dna_features_viewer <https://edinburgh-genome-foundry.github.io/DnaFeaturesViewer/>`_
(the Edinburgh Genome Foundry library, same lab as DNAchisel) and
``logomaker``:

* :func:`view_primers` — the template with forward / reverse primers drawn as
  annotated binding arrows and the amplicon highlighted (SnapGene-style).
* :func:`view_construct` — an annotated linear **or circular plasmid** map with
  coloured feature arrows (promoters, RBS, CDS, terminators, Type IIS sites …).
* :func:`plot_sequence_logo` — a sequence logo for a set of aligned sequences
  (library members, designed variants, a binding site).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

from .._registry import register_function

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# colour scheme by feature type (SnapGene-ish).
_FEATURE_COLORS = {
    "promoter": "#7DBF6A", "RBS": "#F2C14E", "CDS": "#7EA6E0",
    "terminator": "#E06666", "TypeIIS": "#B39DDB", "primer": "#F08C3A",
    "amplicon": "#CFE8FF", "misc_feature": "#C9C9C9", "ORF": "#7EA6E0",
}


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _dfv(fn: str):
    try:
        import dna_features_viewer as dfv
        return dfv
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"ov.synbio.{fn} 需要 dna_features_viewer。请 pip install "
            "dna_features_viewer (或 pip install 'omicverse[synbio]')。"
        ) from exc


@register_function(
    aliases=["view_primers", "引物图", "引物可视化", "primer_map", "plot_primers",
             "引物结合图"],
    category="synthetic_biology",
    description="引物可视化:把正/反向引物画成在模板上的结合箭头并高亮扩增子(SnapGene 风格)。Draw primers as annotated binding arrows on the template with the amplicon.",
    examples=[
        "primers = ov.synbio.design_primers(dna)",
        "ov.synbio.view_primers(dna, primers[0])",
    ],
    related=["synbio.design_primers", "synbio.view_construct"],
    requires={},
    produces={},
)
def view_primers(template: str, primers, ax=None, figure_width: float = 9):
    """Draw primer(s) as binding arrows on *template* with the amplicon.

    *primers* is a :class:`PrimerPair` (or a list of them). Forward primers are
    located on the top strand, reverse primers by their reverse-complement."""
    dfv = _dfv("view_primers")
    t = template.upper().replace("U", "T")
    if not isinstance(primers, (list, tuple)):
        primers = [primers]

    feats = []
    for i, pp in enumerate(primers):
        left, right = pp.left.upper(), pp.right.upper()
        f0 = t.find(left)
        r_bind = _revcomp(right)
        r0 = t.find(r_bind)
        if f0 < 0 or r0 < 0:
            continue
        r1 = r0 + len(right)
        feats.append(dfv.GraphicFeature(
            start=f0, end=f0 + len(left), strand=+1,
            color=_FEATURE_COLORS["primer"], label=f"Fwd{i+1}"))
        feats.append(dfv.GraphicFeature(
            start=r0, end=r1, strand=-1,
            color=_FEATURE_COLORS["primer"], label=f"Rev{i+1}"))
        feats.append(dfv.GraphicFeature(
            start=f0, end=r1, strand=0, color=_FEATURE_COLORS["amplicon"],
            label=f"amplicon {pp.product_size} bp"))

    record = dfv.GraphicRecord(sequence_length=len(t), features=feats)
    if ax is None:
        ax, _ = record.plot(figure_width=figure_width)
    else:
        record.plot(ax=ax)
    ax.set_title("PCR primers on template", fontsize=11)
    return ax


@register_function(
    aliases=["view_construct", "构建体图", "质粒图", "plasmid_map", "construct_map",
             "序列图谱", "质粒图谱"],
    category="synthetic_biology",
    description="构建体/质粒图谱:带彩色特征箭头的线性或环状注释图(启动子/RBS/CDS/终止子等)。Annotated linear or circular plasmid map with coloured feature arrows.",
    examples=[
        "feats = ov.synbio.annotate_construct(dna)",
        "ov.synbio.view_construct(dna, feats, circular=True)",
    ],
    related=["synbio.annotate_construct", "synbio.view_primers"],
    requires={},
    produces={},
)
def view_construct(sequence: str, features: Optional[List[Dict]] = None,
                   circular: bool = False, title: str = "construct",
                   ax=None, figure_width: float = 8):
    """Draw an annotated genetic map of *sequence*.

    *features* is a list of ``{name, type, start, end, strand}`` (e.g. from
    :func:`annotate_construct`); if ``None`` they are auto-detected. Set
    ``circular=True`` for a plasmid map."""
    dfv = _dfv("view_construct")
    seq = sequence.upper().replace("U", "T")
    if features is None:
        from ._assembly import annotate_construct
        features = annotate_construct(seq)

    gfeats = []
    for f in features:
        col = _FEATURE_COLORS.get(f.get("type", "misc_feature"),
                                  _FEATURE_COLORS["misc_feature"])
        strand = 1 if f.get("strand", "+") == "+" else -1
        gfeats.append(dfv.GraphicFeature(
            start=int(f["start"]), end=int(f["end"]), strand=strand,
            color=col, label=f.get("name", "")))

    RecordClass = dfv.CircularGraphicRecord if circular else dfv.GraphicRecord
    record = RecordClass(sequence_length=len(seq), features=gfeats)
    if ax is None:
        ax, _ = record.plot(figure_width=figure_width)
    else:
        record.plot(ax=ax)
    ax.set_title(f"{title} ({len(seq)} bp{', circular' if circular else ''})",
                 fontsize=11)
    return ax


@register_function(
    aliases=["plot_sequence_logo", "序列logo", "sequence_logo", "logo图",
             "seqlogo", "序列标识"],
    category="synthetic_biology",
    description="序列 logo:对一组等长序列(文库成员/设计变体/结合位点)画信息量堆叠字母图(logomaker)。Sequence logo for a set of aligned sequences.",
    examples=[
        "ov.synbio.plot_sequence_logo([d.sequence for d in designs])",
    ],
    related=["synbio.dms_library", "synbio.ml_guided_design", "synbio.saturation_library"],
    requires={},
    produces={},
)
def plot_sequence_logo(sequences: Sequence[str], alphabet: str = "auto",
                       ax=None, title: str = "sequence logo"):
    """Draw a sequence logo from equal-length *sequences* (DNA or protein)."""
    try:
        import logomaker
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.synbio.plot_sequence_logo 需要 logomaker。请 pip install "
            "logomaker (或 pip install 'omicverse[synbio]')。") from exc
    from ._plot import _mpl
    plt = _mpl()

    seqs = [s.upper() for s in sequences]
    L = min(len(s) for s in seqs)
    seqs = [s[:L] for s in seqs]
    if alphabet == "auto":
        chars = set("".join(seqs))
        alphabet = "ACGT" if chars <= set("ACGTN") else "ACDEFGHIKLMNPQRSTVWY"

    counts = logomaker.alignment_to_matrix(seqs, to_type="information",
                                           characters_to_ignore="NX-")
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 0.25 * L), 2.6))
    else:
        fig = ax.figure
    logomaker.Logo(counts, ax=ax, color_scheme="classic"
                   if set(alphabet) <= set("ACGT") else "chemistry")
    ax.set_ylabel("bits")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig, ax


def _pairs_from_dotbracket(structure: str):
    """Return the list of (i, j) base pairs from a dot-bracket string."""
    stack, pairs = [], []
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            pairs.append((stack.pop(), i))
    return pairs


@register_function(
    aliases=["plot_rna_structure", "RNA结构图", "rna_structure_plot", "弧线图",
             "二级结构图", "arc_diagram", "rna_arc"],
    category="synthetic_biology",
    description="RNA 二级结构弧线图:碱基排成一行,配对以半圆弧连接(按 G-C/A-U/G-U 上色),直观展示折叠。Arc-diagram of an RNA secondary structure (dot-bracket) coloured by pair type.",
    examples=[
        "s = ov.synbio.rna_fold(seq)",
        "ov.synbio.plot_rna_structure(s.sequence, s.structure)",
    ],
    related=["synbio.rna_fold", "synbio.rna_inverse_design", "synbio.mrna_design"],
    requires={},
    produces={},
)
def plot_rna_structure(sequence: str, structure: str, ax=None,
                       title: str = "RNA secondary structure"):
    """Draw an arc diagram of *structure* (dot-bracket) over *sequence*.

    Bases sit on a line; each base pair is a semicircular arc, coloured by pair
    type (G-C blue, A-U orange, G-U/other grey). Compact and dependency-free."""
    import numpy as np
    from ._plot import _mpl
    plt = _mpl()

    seq = sequence.upper().replace("T", "U")
    pairs = _pairs_from_dotbracket(structure)
    L = len(structure)
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 0.14 * L), max(2.0, 0.045 * L + 1.4)))
    else:
        fig = ax.figure
    colours = {"GC": "#2C7FB8", "CG": "#2C7FB8", "AU": "#F0894A",
               "UA": "#F0894A", "GU": "#9E9E9E", "UG": "#9E9E9E"}
    for i, j in pairs:
        pair = (seq[i] + seq[j]) if i < len(seq) and j < len(seq) else "??"
        col = colours.get(pair, "#C9C9C9")
        centre, rad = (i + j) / 2.0, (j - i) / 2.0
        th = np.linspace(0, np.pi, 40)
        ax.plot(centre + rad * np.cos(th), rad * np.sin(th), color=col, lw=1.3)
    ax.plot([0, L - 1], [0, 0], color="k", lw=1.2, zorder=0)
    ax.scatter(range(L), [0] * L, s=6, c="k", zorder=3)
    ax.set_ylim(-0.05 * L - 1, 0.5 * L + 1)
    ax.set_xlim(-1, L)
    ax.set_yticks([])
    ax.set_xlabel(f"position (nt) — {len(pairs)} base pairs")
    ax.set_title(title, fontsize=11)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig, ax


__all__ = ["view_primers", "view_construct", "plot_sequence_logo",
           "plot_rna_structure"]
