r"""In-memory multiple sequence alignment and trees, for protein work.

``mafft`` / ``fasttree`` / ``build_phylogeny`` already existed, and they are
pipeline shaped: FASTA path in, FASTA path out, nucleotide defaults, vsearch
``;size=`` annotations stripped on the way. That is right for the 16S pipeline
they were written for and wrong for protein engineering, where the sequences are
a list in memory, the alphabet is amino acids, and the next step is ancestral
reconstruction rather than a UniFrac matrix.

* :func:`msa` — align sequences given as a dict/list. Uses MAFFT when it is on
  ``PATH``; otherwise falls back to a built-in progressive aligner so the
  function works on a bare install. Which one ran is in ``result.method``.
* :func:`protein_tree` — a tree from an alignment. FastTree with a protein model
  when available, otherwise neighbour-joining on corrected distances.
* :func:`plot_msa` — the alignment coloured by conservation.
* :func:`plot_tree` — the tree as a cladogram with branch lengths.

The built-in fallbacks are honest about what they are: progressive alignment
with a BLOSUM-style score and NJ on Poisson-corrected distances is what
phylogenetics did before the fast heuristics, and it is entirely adequate for
tens of sequences. For hundreds, install MAFFT and FastTree.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


_AA = "ACDEFGHIKLMNPQRSTVWY"
GAP = "-"


def _as_mapping(sequences) -> Dict[str, str]:
    if isinstance(sequences, Mapping):
        out = {str(k): "".join(str(v).split()).upper() for k, v in sequences.items()}
    elif isinstance(sequences, (list, tuple)):
        if sequences and isinstance(sequences[0], (list, tuple)):
            out = {str(k): "".join(str(v).split()).upper() for k, v in sequences}
        else:
            out = {f"seq{i + 1}": "".join(str(s).split()).upper()
                   for i, s in enumerate(sequences)}
    else:
        raise TypeError(
            "sequences 需要是 {name: seq} 字典、序列列表,或 (name, seq) 列表。")
    if len(out) < 2:
        raise ValueError("至少需要 2 条序列才能比对。")
    empty = [k for k, v in out.items() if not v]
    if empty:
        raise ValueError(f"这些序列是空的:{empty}")
    return out


# A compact substitution matrix: identity, conservative substitution, and the
# rest. Enough to drive a progressive alignment; not a replacement for BLOSUM62
# when a real aligner is available.
_GROUPS = ("AGST", "ILVM", "FWY", "KRH", "DENQ", "C", "P")


def _sub_score(a: str, b: str) -> int:
    if a == b:
        return 4
    for g in _GROUPS:
        if a in g and b in g:
            return 1
    return -2


def _pairwise(a: str, b: str, gap_open: int = -10, gap_extend: int = -1
              ) -> Tuple[str, str, float]:
    """Global alignment with affine gaps (Gotoh)."""
    n, m = len(a), len(b)
    neg = float("-inf")
    M = [[neg] * (m + 1) for _ in range(n + 1)]
    Ix = [[neg] * (m + 1) for _ in range(n + 1)]
    Iy = [[neg] * (m + 1) for _ in range(n + 1)]
    ptr = [[0] * (m + 1) for _ in range(n + 1)]
    M[0][0] = 0.0
    for i in range(1, n + 1):
        Ix[i][0] = gap_open + (i - 1) * gap_extend
    for j in range(1, m + 1):
        Iy[0][j] = gap_open + (j - 1) * gap_extend

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = _sub_score(a[i - 1], b[j - 1])
            best_prev = max(M[i - 1][j - 1], Ix[i - 1][j - 1], Iy[i - 1][j - 1])
            M[i][j] = best_prev + s
            Ix[i][j] = max(M[i - 1][j] + gap_open, Ix[i - 1][j] + gap_extend)
            Iy[i][j] = max(M[i][j - 1] + gap_open, Iy[i][j - 1] + gap_extend)
            cells = (M[i][j], Ix[i][j], Iy[i][j])
            ptr[i][j] = cells.index(max(cells))

    i, j = n, m
    out_a: List[str] = []
    out_b: List[str] = []
    state = max(range(3), key=lambda k: (M[n][m], Ix[n][m], Iy[n][m])[k])
    score = max(M[n][m], Ix[n][m], Iy[n][m])
    while i > 0 or j > 0:
        if i > 0 and j > 0 and state == 0:
            out_a.append(a[i - 1])
            out_b.append(b[j - 1])
            state = ptr[i][j]
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or state == 1):
            out_a.append(a[i - 1])
            out_b.append(GAP)
            i -= 1
            state = 1 if i > 0 and Ix[i][j] >= M[i][j] else 0
        else:
            out_a.append(GAP)
            out_b.append(b[j - 1])
            j -= 1
            state = 2 if j > 0 and Iy[i][j] >= M[i][j] else 0
    return "".join(reversed(out_a)), "".join(reversed(out_b)), float(score)


@dataclass
class Alignment:
    """A multiple sequence alignment held in memory."""

    names: List[str] = field(default_factory=list)
    aligned: Dict[str, str] = field(default_factory=dict)
    method: str = "builtin"

    @property
    def length(self) -> int:
        return len(next(iter(self.aligned.values()))) if self.aligned else 0

    @property
    def n_sequences(self) -> int:
        return len(self.aligned)

    def column(self, i: int) -> List[str]:
        return [self.aligned[n][i] for n in self.names]

    def conservation(self) -> List[float]:
        """Per-column fraction of sequences sharing the most common residue.

        Gaps count as a character, so an insertion present in one sequence
        lowers conservation rather than being invisible.
        """
        out = []
        for i in range(self.length):
            col = self.column(i)
            if not col:
                out.append(0.0)
                continue
            best = max(set(col), key=col.count)
            out.append(col.count(best) / len(col))
        return out

    def identity(self, a: str, b: str) -> float:
        """Fraction of aligned, non-gap-in-both columns that match."""
        x, y = self.aligned[a], self.aligned[b]
        pairs = [(p, q) for p, q in zip(x, y) if p != GAP or q != GAP]
        if not pairs:
            return 0.0
        return sum(1 for p, q in pairs if p == q) / len(pairs)

    def to_fasta(self, path: Optional[str] = None) -> str:
        text = "".join(f">{n}\n{self.aligned[n]}\n" for n in self.names)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame([list(self.aligned[n]) for n in self.names],
                            index=self.names,
                            columns=range(1, self.length + 1))

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Alignment({self.n_sequences} sequences x {self.length} columns, "
                f"method={self.method!r})")


@register_function(
    aliases=["msa", "多序列比对", "multiple_alignment", "align_sequences",
             "序列比对", "protein_msa"],
    category="alignment",
    description="内存里的多序列比对(蛋白/核酸皆可):输入 {name: seq} 字典或序列列表,返回 Alignment 对象。装了 MAFFT 就用 MAFFT,否则退回内置渐进比对,method 字段标明用的哪一个 —— 免依赖也能跑。现有的 ov.alignment.mafft 是 FASTA 路径进出、面向 16S 流水线的;蛋白工程需要的是在内存里拿到比对结果。In-memory multiple sequence alignment.",
    examples=[
        "aln = ov.alignment.msa({'wt': seq1, 'v1': seq2, 'v2': seq3})",
        "aln.conservation(), aln.identity('wt', 'v1')",
        "aln = ov.alignment.msa(seqs, method='mafft')",
    ],
    related=["alignment.protein_tree", "alignment.plot_msa", "alignment.mafft",
             "synbio.ancestral_reconstruction"],
)
def msa(sequences, method: str = "auto", *, threads: int = 4,
        mafft_mode: str = "auto") -> Alignment:
    """Align ``sequences`` and return an :class:`Alignment`.

    Parameters
    ----------
    sequences
        ``{name: sequence}``, a list of sequences, or ``(name, sequence)`` pairs.
    method
        ``'auto'`` (MAFFT if installed, else the built-in aligner),
        ``'mafft'`` (require MAFFT), or ``'builtin'`` (force the fallback, which
        is what makes results reproducible without pinning a MAFFT version).
    threads
        MAFFT threads.
    mafft_mode
        MAFFT strategy flag.
    """
    _VALID = ("auto", "mafft", "builtin")
    if method not in _VALID:
        raise ValueError(f"method must be one of {list(_VALID)}, got {method!r}")

    seqs = _as_mapping(sequences)

    have_mafft = shutil.which("mafft") is not None
    if method == "mafft" and not have_mafft:
        raise ImportError(
            "method='mafft' 需要 mafft 在 PATH 上(conda install -c bioconda mafft)。"
            "默认 method='auto' 在没有 MAFFT 时会退回内置渐进比对。")
    use_mafft = have_mafft if method == "auto" else (method == "mafft")

    if use_mafft:
        return _mafft_align(seqs, threads=threads, mode=mafft_mode)
    return _progressive_align(seqs)


def _mafft_align(seqs: Dict[str, str], *, threads: int, mode: str) -> Alignment:
    import tempfile
    from .mafft import mafft as _mafft_fn

    with tempfile.TemporaryDirectory(prefix="ov_msa_") as tmp:
        fa = os.path.join(tmp, "in.faa")
        with open(fa, "w", encoding="utf-8") as fh:
            for name, s in seqs.items():
                fh.write(f">{name}\n{s}\n")
        res = _mafft_fn(fa, output_dir=os.path.join(tmp, "out"), mode=mode,
                        threads=threads, auto_install=False)
        aligned: Dict[str, str] = {}
        name = None
        with open(res["aligned"], "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(">"):
                    name = line[1:].strip().split()[0]
                    aligned[name] = ""
                elif name is not None:
                    aligned[name] += line.strip().upper()
    names = [n for n in seqs if n in aligned]
    return Alignment(names=names, aligned={n: aligned[n] for n in names},
                     method="mafft")


def _progressive_align(seqs: Dict[str, str]) -> Alignment:
    """Progressive alignment: align the closest pair, then add the rest by
    similarity to the growing profile.

    The order matters and is made deterministic — sequences are added in
    descending similarity to the current profile, ties broken by name — so the
    same input always gives the same alignment.
    """
    names = list(seqs)
    # seed with the most similar pair
    best_pair, best_score = (names[0], names[1]), float("-inf")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            _, _, sc = _pairwise(seqs[names[i]], seqs[names[j]])
            if sc > best_score:
                best_score, best_pair = sc, (names[i], names[j])

    a, b = best_pair
    aa, bb, _ = _pairwise(seqs[a], seqs[b])
    aligned: Dict[str, str] = {a: aa, b: bb}
    order = [a, b]

    remaining = [n for n in names if n not in aligned]
    while remaining:
        profile = _consensus(list(aligned.values()))
        scored = sorted(
            ((_pairwise(seqs[n], profile)[2], n) for n in remaining),
            key=lambda t: (-t[0], t[1]))
        _, pick = scored[0]
        new_seq, new_profile, _ = _pairwise(seqs[pick], profile)
        # propagate gaps introduced in the profile back into every aligned row
        aligned = {k: _insert_gaps(v, new_profile) for k, v in aligned.items()}
        aligned[pick] = new_seq
        order.append(pick)
        remaining.remove(pick)

    width = max(len(v) for v in aligned.values())
    aligned = {k: v.ljust(width, GAP) for k, v in aligned.items()}
    return Alignment(names=[n for n in names if n in aligned],
                     aligned=aligned, method="builtin")


def _consensus(rows: Sequence[str]) -> str:
    out = []
    for i in range(len(rows[0])):
        col = [r[i] for r in rows if i < len(r) and r[i] != GAP]
        out.append(max(set(col), key=col.count) if col else GAP)
    return "".join(out)


def _insert_gaps(row: str, gapped_profile: str) -> str:
    """Re-thread ``row`` through a profile that has gained gap columns."""
    out = []
    k = 0
    for ch in gapped_profile:
        if ch == GAP:
            out.append(GAP)
        else:
            out.append(row[k] if k < len(row) else GAP)
            k += 1
    while k < len(row):
        out.append(row[k])
        k += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# trees
# ---------------------------------------------------------------------------

@dataclass
class PhyloTree:
    """A rooted tree with branch lengths, as newick plus a parsed structure."""

    newick: str
    method: str = "nj"
    tips: List[str] = field(default_factory=list)
    distances: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return f"PhyloTree({len(self.tips)} tips, method={self.method!r})"


def _corrected_distance(aln: Alignment, a: str, b: str) -> float:
    """Poisson-corrected distance from observed identity.

    ``d = -ln(p_identical)``, which stops distant pairs from saturating at 1 the
    way raw p-distance does.
    """
    import math
    p = aln.identity(a, b)
    p = min(max(p, 1e-6), 1.0)
    return float(-math.log(p))


@register_function(
    aliases=["protein_tree", "蛋白进化树", "build_protein_tree", "系统发育树",
             "phylogenetic_tree", "NJ树"],
    category="alignment",
    description="从比对结果建蛋白系统发育树。装了 FastTree 就用它(蛋白模型),否则用内置邻接法(NJ,基于泊松校正距离)—— 免依赖也能得到树。现有 build_phylogeny 是核酸/ASV 取向(nt=True、GTR 模型)。Build a protein phylogenetic tree from an alignment.",
    examples=[
        "tree = ov.alignment.protein_tree(aln)",
        "tree.newick, tree.tips",
    ],
    related=["alignment.msa", "alignment.plot_tree", "alignment.fasttree",
             "synbio.ancestral_reconstruction"],
)
def protein_tree(alignment: Alignment, method: str = "auto") -> PhyloTree:
    """Infer a tree. ``method`` is ``'auto'`` | ``'fasttree'`` | ``'nj'``."""
    _VALID = ("auto", "fasttree", "nj")
    if method not in _VALID:
        raise ValueError(f"method must be one of {list(_VALID)}, got {method!r}")

    have_ft = shutil.which("FastTree") or shutil.which("FastTreeMP")
    if method == "fasttree" and not have_ft:
        raise ImportError(
            "method='fasttree' 需要 FastTree 在 PATH 上"
            "(conda install -c bioconda fasttree)。默认 method='auto' 会退回内置 NJ。")
    if method == "auto" and have_ft:
        try:
            return _fasttree_tree(alignment)
        except Exception:
            pass
    elif method == "fasttree":
        return _fasttree_tree(alignment)
    return _nj_tree(alignment)


def _fasttree_tree(alignment: Alignment) -> PhyloTree:
    import tempfile
    from .fasttree import fasttree as _ft

    with tempfile.TemporaryDirectory(prefix="ov_tree_") as tmp:
        fa = os.path.join(tmp, "aln.faa")
        alignment.to_fasta(fa)
        res = _ft(fa, output_dir=os.path.join(tmp, "out"), model="lg",
                  nt=False, auto_install=False)
        with open(res["tree"], "r", encoding="utf-8") as fh:
            newick = fh.read().strip()
    return PhyloTree(newick=newick, method="fasttree", tips=list(alignment.names))


def _nj_tree(alignment: Alignment) -> PhyloTree:
    """Saitou-Nei neighbour joining on Poisson-corrected distances."""
    names = list(alignment.names)
    d: Dict[Tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = _corrected_distance(alignment, a, b)
            d[(a, b)] = d[(b, a)] = v
        d[(a, a)] = 0.0

    nodes = list(names)
    label = {n: n for n in names}
    dist = {(a, b): d[(a, b)] for a in names for b in names}
    counter = [0]

    while len(nodes) > 2:
        n = len(nodes)
        r = {x: sum(dist[(x, y)] for y in nodes if y != x) / (n - 2)
             for x in nodes}
        best, best_q = None, float("inf")
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                q = dist[(a, b)] - r[a] - r[b]
                if q < best_q:
                    best_q, best = q, (a, b)
        a, b = best
        dab = dist[(a, b)]
        la = 0.5 * dab + 0.5 * (r[a] - r[b])
        lb = dab - la
        la, lb = max(la, 0.0), max(lb, 0.0)
        counter[0] += 1
        new = f"_node{counter[0]}"
        label[new] = f"({label[a]}:{la:.5f},{label[b]}:{lb:.5f})"
        for x in nodes:
            if x in (a, b):
                continue
            dnew = 0.5 * (dist[(a, x)] + dist[(b, x)] - dab)
            dist[(new, x)] = dist[(x, new)] = max(dnew, 0.0)
        dist[(new, new)] = 0.0
        nodes = [x for x in nodes if x not in (a, b)] + [new]

    a, b = nodes
    newick = f"({label[a]}:{dist[(a, b)] / 2:.5f},{label[b]}:{dist[(a, b)] / 2:.5f});"
    return PhyloTree(newick=newick, method="nj", tips=names,
                     distances={k: v for k, v in d.items() if k[0] != k[1]})


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

@register_function(
    aliases=["plot_msa", "比对图", "MSA图", "alignment_plot", "保守性图"],
    category="alignment",
    description="画多序列比对:残基按理化性质着色,上方是逐列保守性曲线,一眼看出保守核心与可变区。Plot a multiple sequence alignment coloured by residue class, with a conservation track.",
    examples=["ov.alignment.plot_msa(aln)"],
    related=["alignment.msa", "alignment.plot_tree"],
)
def plot_msa(alignment: Alignment, start: int = 0, end: Optional[int] = None,
             axes=None):
    """Alignment heatmap with a conservation track above it."""
    import matplotlib.pyplot as plt
    import numpy as np

    end = alignment.length if end is None else min(end, alignment.length)
    cols = list(range(start, end))
    if not cols:
        raise ValueError("要画的区间是空的。")

    classes = {**{a: 0 for a in "AGSTP"}, **{a: 1 for a in "ILVMC"},
               **{a: 2 for a in "FWY"}, **{a: 3 for a in "KRH"},
               **{a: 4 for a in "DENQ"}, GAP: 5}
    colours = ["#B3DE69", "#FDB462", "#BC80BD", "#80B1D3", "#FB8072", "#FFFFFF"]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colours)

    mat = np.array([[classes.get(alignment.aligned[n][i], 5) for i in cols]
                    for n in alignment.names])

    if axes is None:
        fig, axes = plt.subplots(
            2, 1, figsize=(min(18, 0.13 * len(cols) + 3),
                           0.28 * alignment.n_sequences + 2.4),
            gridspec_kw={"height_ratios": [1, 4]}, sharex=True)
    else:
        fig = axes[0].figure
    axes = list(axes)

    cons = alignment.conservation()[start:end]
    axes[0].fill_between(range(len(cols)), cons, color="#377EB8", alpha=0.8)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("cons.", fontsize=8)
    axes[0].set_title(f"{alignment.n_sequences} sequences, "
                      f"columns {start + 1}-{end} ({alignment.method})")

    axes[1].imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=5,
                   interpolation="nearest")
    axes[1].set_yticks(range(alignment.n_sequences))
    axes[1].set_yticklabels(alignment.names, fontsize=7)
    axes[1].set_xlabel("alignment column")
    fig.tight_layout()
    return fig, axes


@register_function(
    aliases=["plot_tree", "进化树图", "phylogeny_plot", "树图", "cladogram"],
    category="alignment",
    description="画系统发育树(newick 或 PhyloTree):带枝长的矩形分支图,可高亮指定叶节点。Plot a phylogenetic tree from newick with branch lengths.",
    examples=[
        "ov.alignment.plot_tree(tree)",
        "ov.alignment.plot_tree(tree, highlight=['ancestor'])",
    ],
    related=["alignment.protein_tree", "alignment.msa"],
)
def plot_tree(tree, highlight: Optional[Sequence[str]] = None, ax=None):
    """Rectangular cladogram from a newick string or :class:`PhyloTree`."""
    import matplotlib.pyplot as plt

    newick = tree.newick if isinstance(tree, PhyloTree) else str(tree)
    root = _parse_newick(newick)
    highlight = set(highlight or [])

    leaves: List[dict] = []

    def collect(node):
        if not node["children"]:
            leaves.append(node)
        for c in node["children"]:
            collect(c)
    collect(root)

    for i, lf in enumerate(leaves):
        lf["y"] = float(i)

    def assign(node, depth=0.0):
        node["x"] = depth + node["length"]
        if node["children"]:
            for c in node["children"]:
                assign(c, node["x"])
            node["y"] = sum(c["y"] for c in node["children"]) / len(node["children"])
        return node
    assign(root, 0.0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 0.32 * max(4, len(leaves)) + 1.4))
    else:
        fig = ax.figure

    def draw(node):
        for c in node["children"]:
            ax.plot([node["x"], c["x"]], [c["y"], c["y"]], c="#333333", lw=1.2)
            draw(c)
        if node["children"]:
            ys = [c["y"] for c in node["children"]]
            ax.plot([node["x"], node["x"]], [min(ys), max(ys)],
                    c="#333333", lw=1.2)
    draw(root)

    for lf in leaves:
        hot = lf["name"] in highlight
        ax.text(lf["x"] + 0.01, lf["y"], " " + lf["name"], va="center",
                fontsize=8, color="#E41A1C" if hot else "#000000",
                fontweight="bold" if hot else "normal")
    ax.set_yticks([])
    ax.set_xlabel("substitutions per site")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.margins(x=0.25)
    fig.tight_layout()
    return fig, ax


def _parse_newick(text: str) -> dict:
    """Minimal newick parser → nested ``{'name', 'length', 'children'}``."""
    text = text.strip().rstrip(";")
    pos = [0]

    def parse_node() -> dict:
        node = {"name": "", "length": 0.0, "children": []}
        if pos[0] < len(text) and text[pos[0]] == "(":
            pos[0] += 1
            while True:
                node["children"].append(parse_node())
                if pos[0] < len(text) and text[pos[0]] == ",":
                    pos[0] += 1
                    continue
                if pos[0] < len(text) and text[pos[0]] == ")":
                    pos[0] += 1
                break
        start = pos[0]
        while pos[0] < len(text) and text[pos[0]] not in ",():":
            pos[0] += 1
        node["name"] = text[start:pos[0]]
        if pos[0] < len(text) and text[pos[0]] == ":":
            pos[0] += 1
            start = pos[0]
            while pos[0] < len(text) and text[pos[0]] not in ",()":
                pos[0] += 1
            try:
                node["length"] = float(text[start:pos[0]])
            except ValueError:
                node["length"] = 0.0
        return node

    return parse_node()


__all__ = ["msa", "Alignment", "protein_tree", "PhyloTree", "plot_msa",
           "plot_tree"]
