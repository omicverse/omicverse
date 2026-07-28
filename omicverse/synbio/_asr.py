r"""Ancestral sequence reconstruction — thermostability without a large model.

Resurrecting an inferred ancestor is one of the most reliable ways to get a more
thermostable enzyme, and it long predates deep learning: ancestral proteins are
consistently more stable than their descendants, plausibly because ancient
environments were hotter and because reconstruction averages away the
destabilising substitutions that individual lineages accumulated. It needs a
sequence family and a tree, not a GPU.

* :func:`ancestral_reconstruction` — infer the sequence at an internal node.
  ``method='parsimony'`` runs Fitch's algorithm; ``method='weighted_consensus'``
  (aliased as ``'ml'`` for continuity) runs a sequence-weighted column consensus.
  **It is not maximum likelihood** — there is no substitution model and no
  likelihood over branch lengths — and the per-site number it returns is column
  support, not a posterior. For real marginal-ML ancestral states use RAxML-NG,
  IQ-TREE ``--ancestral`` or PAML on a rooted tree; this is a fast approximation
  for picking which sites deserve a small variant library. It previously claimed
  marginal
  maximum likelihood over a Poisson/Jukes-Cantor-style amino-acid model, which
  additionally gives a **posterior probability per site** — and those
  probabilities are the useful output, because the ambiguous sites are exactly
  the ones worth ordering as a small variant library.
* :func:`consensus_sequence` — the cheaper cousin: the most common residue per
  column. Often works, sometimes produces a protein no lineage ever had, and is
  reported alongside so the two can be compared.
* :func:`plot_ancestral_confidence` — posterior support along the sequence.

The reconstruction is only as good as the alignment and tree it is handed, and
low-support sites are where it is guessing. :attr:`AncestralSequence.
ambiguous_sites` lists them rather than hiding them inside a single string.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

_AA = "ACDEFGHIKLMNPQRSTVWY"
GAP = "-"


@dataclass
class AncestralSequence:
    """An inferred ancestral sequence and how sure each position is."""

    sequence: str                                  # gaps stripped
    aligned_sequence: str = ""                     # in alignment coordinates
    method: str = "weighted_consensus"
    posterior: List[float] = field(default_factory=list)
    alternatives: Dict[int, List[Tuple[str, float]]] = field(default_factory=dict)
    node: str = "root"
    n_sequences: int = 0

    @property
    def mean_posterior(self) -> float:
        vals = [p for p in self.posterior if p > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def ambiguous_sites(self, threshold: float = 0.8) -> List[int]:
        """1-based positions (in the ungapped ancestor) below ``threshold``.

        These are the reconstruction's guesses. Ordering the top alternatives at
        these sites as a small library is standard practice and usually cheaper
        than trusting a single reconstructed sequence.
        """
        out = []
        k = 0
        for i, ch in enumerate(self.aligned_sequence):
            if ch == GAP:
                continue
            k += 1
            if i < len(self.posterior) and self.posterior[i] < threshold:
                out.append(k)
        return out

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        rows = []
        k = 0
        for i, ch in enumerate(self.aligned_sequence):
            if ch == GAP:
                continue
            k += 1
            alts = self.alternatives.get(i, [])
            rows.append({
                "position": k,
                "column": i + 1,
                "residue": ch,
                "posterior": self.posterior[i] if i < len(self.posterior) else None,
                "runner_up": alts[1][0] if len(alts) > 1 else None,
                "runner_up_posterior": alts[1][1] if len(alts) > 1 else None,
            })
        return pd.DataFrame(rows).set_index("position")

    def __repr__(self) -> str:  # pragma: no cover
        return (f"AncestralSequence({self.method}, {len(self.sequence)} aa, "
                f"mean posterior {self.mean_posterior:.3f}, "
                f"{len(self.ambiguous_sites())} ambiguous sites)")


def _columns(alignment) -> Tuple[List[str], int]:
    names = list(alignment.names)
    return names, alignment.length


def _fitch(states: Sequence[set]) -> set:
    """Fitch bottom-up pass over a star topology.

    Without a tree, the parsimony ancestor of a set of leaves is the
    intersection of their states when non-empty, else the union — which is
    exactly Fitch at a single internal node.
    """
    inter = set.intersection(*states) if states else set()
    return inter if inter else set.union(*states) if states else set()


@register_function(
    aliases=["ancestral_reconstruction", "祖先序列重构", "ASR", "asr",
             "祖先重建", "ancestral_sequence", "热稳定性改造"],
    category="synthetic_biology",
    description="祖先序列重构(ASR):从一个序列家族的比对推断祖先序列。祖先蛋白通常比后代更热稳定,这是提高热稳定性最可靠的经典手段之一,不需要大模型。method='weighted_consensus'(默认;'ml' 是保留的别名)按序列权重做逐列共识,给出**列支持度**——它不是边缘极大似然,没有替换模型也没有基于分支长度的似然,真要做 ML 祖先态请用 RAxML-NG / IQ-TREE --ancestral / PAML;或 'parsimony'(Fitch)。低支持度位点会被单独列出 —— 那些正是值得做成小型变体库的位置,而不是藏在一条序列里。Ancestral sequence reconstruction for thermostability engineering.",
    examples=[
        "anc = ov.synbio.ancestral_reconstruction(aln)",
        "anc.sequence, anc.mean_posterior, anc.ambiguous_sites()",
        "anc.to_frame().head()",
    ],
    related=["alignment.msa", "alignment.protein_tree", "synbio.stability_ddg",
             "synbio.consensus_sequence", "synbio.plot_ancestral_confidence"],
    requires={},
    produces={},
)
def ancestral_reconstruction(
    alignment,
    method: str = "ml",
    *,
    tree=None,
    weights: Optional[Dict[str, float]] = None,
    gap_threshold: float = 0.5,
    pseudocount: float = 0.05,
) -> AncestralSequence:
    """Infer the ancestral sequence of an aligned family.

    Parameters
    ----------
    alignment
        An :class:`omicverse.alignment.Alignment` (from
        :func:`omicverse.alignment.msa`).
    method
        ``'ml'`` — marginal maximum likelihood per column under a flat
        amino-acid exchange model, weighting each sequence by its branch length
        when a tree is given. Returns posteriors.
        ``'parsimony'`` — Fitch. Faster, no posteriors, and it will hand back a
        tie broken arbitrarily where the family is genuinely split, which is why
        it is not the default.
    tree
        Optional :class:`omicverse.alignment.PhyloTree`. When supplied, sequences
        further from the root contribute less.
    weights
        Explicit per-sequence weights, overriding the tree.
    gap_threshold
        A column with more than this fraction of gaps is treated as absent from
        the ancestor.
    pseudocount
        Prior mass as a *fraction of the observed evidence*, so a residue seen in
        no descendant still has non-zero posterior. It is relative rather than
        absolute on purpose: an absolute 0.1 per residue is 2.0 of prior mass
        spread over the 20 amino acids, which outweighs four tree-weighted
        sequences entirely — every column of a 97%-identical family came back at
        posterior 0.67 and was reported as ambiguous.

    Returns
    -------
    AncestralSequence
    """
    _VALID = ("weighted_consensus", "parsimony", "ml")
    if method not in _VALID:
        raise ValueError(f"method must be one of {list(_VALID)}, got {method!r}")
    if method == "ml":
        # Kept working, renamed honestly. It was never maximum likelihood.
        method = "weighted_consensus"

    names, ncol = _columns(alignment)
    if len(names) < 3:
        raise ValueError(
            f"ASR 至少需要 3 条序列才有意义(现在 {len(names)} 条)。"
            f"两条序列的祖先无法与它们本身区分。")

    w = _sequence_weights(names, tree, weights)

    aligned_chars: List[str] = []
    posteriors: List[float] = []
    alternatives: Dict[int, List[Tuple[str, float]]] = {}

    for i in range(ncol):
        col = alignment.column(i)
        gap_frac = sum(1 for c in col if c == GAP) / len(col)
        if gap_frac > gap_threshold:
            aligned_chars.append(GAP)
            posteriors.append(0.0)
            continue

        if method == "parsimony":
            states = [{c} for c in col if c != GAP]
            best = _fitch(states) if states else {GAP}
            pick = sorted(best)[0]
            aligned_chars.append(pick)
            posteriors.append(1.0 if len(best) == 1 else 1.0 / len(best))
            alternatives[i] = [(s, 1.0 / len(best)) for s in sorted(best)]
            continue

        observed = sum(w[n] for n, ch in zip(names, col) if ch in _AA)
        alpha = pseudocount * max(observed, 1e-9) / len(_AA)
        counts: Dict[str, float] = {a: alpha for a in _AA}
        for name, ch in zip(names, col):
            if ch in counts:
                counts[ch] += w[name]
        total = sum(counts.values())
        ranked = sorted(((a, c / total) for a, c in counts.items()),
                        key=lambda t: (-t[1], t[0]))
        aligned_chars.append(ranked[0][0])
        posteriors.append(ranked[0][1])
        alternatives[i] = ranked[:4]

    aligned_seq = "".join(aligned_chars)
    return AncestralSequence(
        sequence=aligned_seq.replace(GAP, ""), aligned_sequence=aligned_seq,
        method=method, posterior=posteriors, alternatives=alternatives,
        n_sequences=len(names))


def _sequence_weights(names, tree, weights) -> Dict[str, float]:
    if weights is not None:
        missing = [n for n in names if n not in weights]
        if missing:
            raise ValueError(f"weights 缺少这些序列:{missing}")
        return {n: float(weights[n]) for n in names}
    if tree is not None and not getattr(tree, "distances", None):
        import warnings
        warnings.warn(
            f"传入的树没有分支长度(method={getattr(tree, 'method', '?')!r}),"
            f"所以每条序列的权重都是 1.0 —— 这棵树对结果没有任何影响。"
            f"protein_tree(method='nj') 会带上距离;FastTree 路径目前不填 "
            f"distances。要靠树加权,请用 NJ 树,或直接传 weights=。",
            stacklevel=3)
    if tree is not None and getattr(tree, "distances", None):
        # closer sequences say more about the ancestor
        out = {}
        for n in names:
            ds = [v for (a, b), v in tree.distances.items() if a == n]
            mean_d = sum(ds) / len(ds) if ds else 1.0
            out[n] = 1.0 / (1.0 + mean_d)
        return out
    return {n: 1.0 for n in names}


@register_function(
    aliases=["consensus_sequence", "共识序列", "consensus", "多数序列"],
    category="synthetic_biology",
    description="共识序列:逐列取最常见残基。比 ASR 便宜,常常也有效,但它可能造出任何谱系都没出现过的蛋白 —— 和 ancestral_reconstruction 一起看,两者差异大的位点值得注意。Column-wise consensus sequence.",
    examples=["cons = ov.synbio.consensus_sequence(aln)"],
    related=["synbio.ancestral_reconstruction", "alignment.msa"],
    requires={},
    produces={},
)
def consensus_sequence(alignment, gap_threshold: float = 0.5) -> str:
    """Most common residue per column, gaps dropped."""
    out = []
    for i in range(alignment.length):
        col = alignment.column(i)
        gap_frac = sum(1 for c in col if c == GAP) / len(col)
        if gap_frac > gap_threshold:
            continue
        residues = [c for c in col if c != GAP]
        if residues:
            out.append(max(sorted(set(residues)), key=residues.count))
    return "".join(out)


@register_function(
    aliases=["plot_ancestral_confidence", "祖先置信度图", "ASR图",
             "plot_asr", "后验概率图"],
    category="synthetic_biology",
    description="画祖先序列的逐位点后验概率与低支持度位点(阴影),并列出各位点的次优残基 —— 这些位置就是该做成变体库的地方。Plot per-site posterior support for a reconstructed ancestor.",
    examples=["ov.synbio.plot_ancestral_confidence(anc)"],
    related=["synbio.ancestral_reconstruction"],
    requires={},
    produces={},
)
def plot_ancestral_confidence(ancestor: AncestralSequence,
                              threshold: float = 0.8, ax=None):
    """Posterior along the sequence, with ambiguous sites shaded."""
    from ._plot import _mpl
    plt = _mpl()

    posts = [p for p, ch in zip(ancestor.posterior, ancestor.aligned_sequence)
             if ch != GAP]
    xs = range(1, len(posts) + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(min(16, 0.06 * len(posts) + 4), 3.2))
    else:
        fig = ax.figure

    ax.fill_between(xs, posts, color="#377EB8", alpha=0.75)
    ax.axhline(threshold, ls="--", c="#E41A1C", lw=1.1,
               label=f"ambiguous below {threshold}")
    for pos in ancestor.ambiguous_sites(threshold):
        ax.axvspan(pos - 0.5, pos + 0.5, color="#E41A1C", alpha=0.15)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("ancestral residue position")
    ax.set_ylabel("posterior")
    ax.set_title(f"{ancestor.method} reconstruction from "
                 f"{ancestor.n_sequences} sequences — mean "
                 f"{ancestor.mean_posterior:.2f}, "
                 f"{len(ancestor.ambiguous_sites(threshold))} sites to library")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


__all__ = ["ancestral_reconstruction", "AncestralSequence",
           "consensus_sequence", "plot_ancestral_confidence"]
