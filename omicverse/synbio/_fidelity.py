r"""Golden Gate overhang fidelity, and orthogonal overhang set design.

``golden_gate`` orders fragments by complementary overhangs and reports whether
they close a circle. That is the right answer to "does this assembly have a
valid order" and no answer at all to "will it actually work". Ligase does not
only join perfectly matched 4-nt overhangs — it joins mismatched ones too, at a
rate that depends on the specific pair. Below about six fragments the wrong
products are rare enough to ignore; above that they compound, and a set chosen
for convenience rather than orthogonality assembles into a smear.

* :func:`overhang_fidelity` — for a chosen set, the predicted fraction of
  correct ligation events, plus the specific pairs that cross-react.
* :func:`design_overhang_set` — pick ``n`` mutually orthogonal overhangs,
  optionally extending a set you are already committed to.
* :func:`plot_overhang_fidelity` — the cross-reactivity matrix, which is what
  makes a bad set obvious.

**On the scoring.** NEB published ligation-fidelity data measured by
high-throughput sequencing (Potapov 2018; Pryor 2020), and the honest way to use
it is to load their table. This module therefore takes ``fidelity_table=`` (a
path to, or mapping of, measured pair frequencies) and *uses it when given*.
Without one it falls back to a transparent mismatch model — a pair's
cross-reactivity is driven by how many of the four positions match, weighted
toward the middle two, which is the qualitative behaviour the measured data
show. The fallback is documented arithmetic and labelled ``method='mismatch'``;
it ranks candidate sets sensibly but its absolute rates are not NEB's numbers,
and :attr:`FidelityReport.method` says which you got.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


_COMP = str.maketrans("ACGT", "TGCA")


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _clean_oh(oh: str) -> str:
    s = str(oh).strip().upper().replace("U", "T")
    if len(s) != 4 or set(s) - set("ACGT"):
        raise ValueError(f"overhang 必须是 4 个 ACGT 字符,得到 {oh!r}")
    return s


# Position weights: the two central bases of a 4-nt overhang contribute most to
# ligation discrimination, the outer ones least. This is the qualitative shape
# of the measured NEB data and the only thing the fallback model claims.
_POS_WEIGHT = (0.15, 0.35, 0.35, 0.15)

# Mismatch discrimination. Ligase rejects a mismatched overhang far more
# sharply than a linear count of matching bases suggests — a single central
# mismatch drops ligation by well over an order of magnitude. A linear score
# made a perfectly orthogonal four-overhang set look 40% cross-reactive.
_MISMATCH_SHARPNESS = 12.0


def _pair_score(a: str, b: str) -> float:
    """Relative ligation propensity of overhangs ``a`` and ``b``, in ``(0, 1]``.

    Two Golden Gate fragments join when the 4-nt overhang each presents is the
    *same* sequence read on the same strand — one carries it as a 5' extension,
    the other as the complementary recessed end. So a correct junction is
    ``a`` with ``a``, and this returns exactly 1.0 there.

    Comparing ``a`` against ``revcomp(b)`` instead — which is what this did
    first — makes the diagonal of the matrix the palindromicity of each
    overhang rather than its correct ligation, and a perfectly orthogonal
    four-overhang set scored 0.30.
    """
    import math

    penalty = sum(w for w, (x, y) in zip(_POS_WEIGHT, zip(a, b)) if x != y)
    return math.exp(-_MISMATCH_SHARPNESS * penalty)


@dataclass
class CrossReaction:
    left: str
    right: str
    score: float

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.left}/{self.right}={self.score:.2f}"


@dataclass
class FidelityReport:
    overhangs: List[str]
    fidelity: float                     # predicted fraction of correct ligations
    method: str = "mismatch"
    cross_reactions: List[CrossReaction] = field(default_factory=list)
    matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)
    palindromic: List[str] = field(default_factory=list)
    duplicates: List[str] = field(default_factory=list)
    #: Pairs in the set that are each other's reverse complement — mutually
    #: ligatable in the wrong orientation, and a structural error in the set.
    reverse_complement_pairs: List[List[str]] = field(default_factory=list)

    @property
    def n_fragments(self) -> int:
        return len(self.overhangs)

    @property
    def verdict(self) -> str:
        if self.duplicates or self.palindromic or self.reverse_complement_pairs:
            return "invalid set"
        if self.fidelity >= 0.95:
            return "high fidelity"
        if self.fidelity >= 0.85:
            return "usable"
        return "expect misassembly"

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        idx = self.overhangs
        return pd.DataFrame(
            [[self.matrix.get((a, b), 0.0) for b in idx] for a in idx],
            index=idx, columns=idx)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"FidelityReport({self.n_fragments} overhangs, "
                f"fidelity={self.fidelity:.3f} [{self.method}], {self.verdict})")


def _load_fidelity_table(table) -> Optional[Dict[Tuple[str, str], float]]:
    if table is None:
        return None
    if isinstance(table, Mapping):
        return {(_clean_oh(a), _clean_oh(b)): float(v)
                for (a, b), v in table.items()}
    path = os.path.expanduser(str(table))
    if not os.path.exists(path):
        raise FileNotFoundError(f"fidelity_table 文件不存在:{path}")
    out: Dict[Tuple[str, str], float] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split("\t")
            if len(parts) < 3:
                continue
            out[(_clean_oh(parts[0]), _clean_oh(parts[1]))] = float(parts[2])
    if not out:
        raise ValueError(f"fidelity_table {path!r} 里没有解析到任何数据行"
                         f"(期望三列:overhang1 overhang2 frequency)。")
    return out


@register_function(
    aliases=["overhang_fidelity", "悬挂端保真度", "overhang保真度", "fidelity",
             "golden_gate_fidelity", "组装保真度", "错配率"],
    category="synthetic_biology",
    description="Golden Gate 悬挂端保真度:预测正确连接的比例,并列出会互相错配的具体悬挂端对。现有 golden_gate 只按互补性排序、不评估 fidelity —— 片段数 >6 时错配会显著累积。给 fidelity_table= 可用 NEB 实测连接数据;不给则用透明的错配模型(中间两位权重更高),method 字段标明用的是哪一种。Predict Golden Gate ligation fidelity for an overhang set.",
    examples=[
        "rep = ov.synbio.overhang_fidelity(['AATG', 'AGGT', 'GCTT', 'CGCT'])",
        "rep.fidelity, rep.verdict, rep.cross_reactions",
        "rep = ov.synbio.overhang_fidelity(ohs, fidelity_table='/data/neb_t4_37c.tsv')",
    ],
    related=["synbio.design_overhang_set", "synbio.golden_gate",
             "synbio.plot_overhang_fidelity"],
    requires={},
    produces={},
)
def overhang_fidelity(
    overhangs: Sequence[str],
    *,
    fidelity_table=None,
    method: str = "auto",
    cross_threshold: float = 0.05,
) -> FidelityReport:
    """Predicted fraction of correct ligation events for an overhang set.

    Parameters
    ----------
    overhangs
        The 4-nt overhangs, one per junction.
    fidelity_table
        Measured pair frequencies — a ``{(oh1, oh2): frequency}`` mapping or a
        TSV/CSV with three columns. When supplied these are used directly and
        ``method='measured'``.
    method
        ``'auto'`` (measured if a table is given, else mismatch), ``'measured'``
        (requires a table), or ``'mismatch'`` to force the fallback.
    cross_threshold
        Report a pair in ``cross_reactions`` once its relative ligation reaches
        this. 5% is already meaningful across eight junctions.

    Returns
    -------
    FidelityReport
    """
    if method not in ("auto", "measured", "mismatch"):
        raise ValueError(
            f"method must be one of ['auto', 'measured', 'mismatch'], got {method!r}")

    ohs = [_clean_oh(o) for o in overhangs]
    if len(ohs) < 2:
        raise ValueError("至少需要 2 个悬挂端才能谈保真度。")

    table = _load_fidelity_table(fidelity_table)
    if method == "measured" and table is None:
        raise ValueError(
            "method='measured' 需要 fidelity_table=(NEB Potapov 2018 / Pryor 2020 "
            "的实测连接频率)。本模块不内置这些数据。")
    use_measured = table is not None and method in ("auto", "measured")

    duplicates = sorted({o for o in ohs if ohs.count(o) > 1})
    palindromic = sorted({o for o in ohs if o == _revcomp(o)})
    # Two overhangs that are each other's reverse complement will ligate to one
    # another in the wrong orientation. That is a structural error in the set, not
    # a lower probability of success, so it belongs with duplicates and
    # palindromes rather than in the fidelity number.
    reverse_pairs = sorted({tuple(sorted((o, _revcomp(o)))) for o in ohs
                            if _revcomp(o) in ohs and o != _revcomp(o)})

    matrix: Dict[Tuple[str, str], float] = {}
    for a in ohs:
        for b in ohs:
            if use_measured:
                v = table.get((a, b), table.get((b, a)))
                if v is None:
                    v = _pair_score(a, b)
            else:
                v = _pair_score(a, b)
            matrix[(a, b)] = float(v)

    # Fidelity is per *junction*, then compounded: at each junction the correct
    # partner competes only with the other overhangs present, and the assembly
    # is right only if every junction is. Normalising over the whole matrix at
    # once instead made fidelity fall with set size regardless of how orthogonal
    # the set was — which is the opposite of what a design tool should report.
    fidelity = 1.0
    for a in ohs:
        competitors = sum(matrix[(a, b)] for b in ohs)
        if competitors > 0:
            fidelity *= matrix[(a, a)] / competitors
    fidelity = float(fidelity)

    # 5% relative ligation is already worth knowing about at eight junctions —
    # the earlier 0.5 cut-off reported nothing at all once the mismatch penalty
    # became appropriately steep.
    cross: List[CrossReaction] = []
    for a, b in itertools.combinations(ohs, 2):
        s = matrix[(a, b)]
        if s >= cross_threshold:
            cross.append(CrossReaction(a, b, s))
    cross.sort(key=lambda c: -c.score)

    return FidelityReport(
        overhangs=ohs, fidelity=fidelity,
        method="measured" if use_measured else "mismatch",
        cross_reactions=cross, matrix=matrix,
        palindromic=palindromic, duplicates=duplicates,
        reverse_complement_pairs=[list(p) for p in reverse_pairs],
    )


@register_function(
    aliases=["design_overhang_set", "正交悬挂端", "overhang_set", "正交集",
             "orthogonal_overhangs", "悬挂端设计"],
    category="synthetic_biology",
    description="设计 n 个互相正交的 Golden Gate 悬挂端(贪心最大化最小间距,排除回文与已用端),可在已经定死的部分集合上扩展。多片段组装时这是能否一锅装成的决定因素。Design a set of mutually orthogonal Golden Gate overhangs.",
    examples=[
        "ohs = ov.synbio.design_overhang_set(8)",
        "ohs = ov.synbio.design_overhang_set(10, fixed=['AATG', 'GCTT'])",
    ],
    related=["synbio.overhang_fidelity", "synbio.golden_gate"],
    requires={},
    produces={},
)
def design_overhang_set(
    n: int,
    *,
    fixed: Optional[Sequence[str]] = None,
    forbidden: Optional[Sequence[str]] = None,
    max_cross: float = 0.5,
    fidelity_table=None,
) -> List[str]:
    """Choose ``n`` overhangs that do not cross-react.

    Greedy: from the pool of non-palindromic 4-mers, repeatedly take the
    candidate whose worst cross-reaction with the current set is lowest. Ties
    break lexicographically so the result is deterministic.

    Parameters
    ----------
    n
        How many overhangs are needed (one per junction).
    fixed
        Overhangs already committed to — a vector's existing ends, for instance.
        They are kept and the rest are chosen around them.
    forbidden
        Overhangs to exclude outright.
    max_cross
        Reject a candidate whose cross-reaction with any chosen overhang exceeds
        this.
    fidelity_table
        Measured data, as in :func:`overhang_fidelity`.
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")
    fixed = [_clean_oh(o) for o in (fixed or [])]
    forbidden = {_clean_oh(o) for o in (forbidden or [])}
    table = _load_fidelity_table(fidelity_table)

    if len(fixed) > n:
        raise ValueError(f"fixed 里已有 {len(fixed)} 个,超过要求的 n={n}")

    def cross(a: str, b: str) -> float:
        if table is not None:
            v = table.get((a, b), table.get((b, a)))
            if v is not None:
                return float(v)
        return _pair_score(a, b)

    pool = ["".join(p) for p in itertools.product("ACGT", repeat=4)]
    pool = [o for o in pool
            if o != _revcomp(o)            # palindromes self-ligate
            and len(set(o)) > 1            # homopolymers anneal and slip badly
            and o not in forbidden and o not in fixed]

    def complexity_penalty(o: str) -> float:
        """Prefer balanced, mixed 4-mers when cross-reactivity cannot separate.

        With an empty set every candidate has worst-cross 0.0, so the first pick —
        and by extension the shape of the whole set — used to be decided purely by
        lexicographic order. That returned AAAC / CCCA / GGGT / TTTG: 3+1
        low-complexity 4-mers, and by twelve overhangs it was reaching for AT-only
        ends with essentially no duplex stability at 37 C.
        """
        gc = sum(1 for b in o if b in "GC") / 4.0
        counts = sorted((o.count(b) for b in "ACGT"), reverse=True)
        return abs(gc - 0.5) + 0.5 * (counts[0] - 1) / 3.0

    chosen = list(fixed)
    while len(chosen) < n:
        best, best_key = None, None
        taken = set(chosen)
        for cand in pool:
            # An overhang and its own reverse complement are mutually ligatable in
            # the wrong orientation. Above 20 overhangs the greedy search used to
            # return exactly that — ('ACCC', 'GGGT') — and reported it as a merely
            # lower fidelity number rather than as an invalid set.
            if _revcomp(cand) in taken:
                continue
            worst = max((cross(cand, c) for c in chosen), default=0.0)
            worst = max(worst, max((cross(c, cand) for c in chosen), default=0.0))
            if worst > max_cross:
                continue
            key = (round(worst, 6), complexity_penalty(cand), cand)
            if best_key is None or key < best_key:
                best, best_key = cand, key
        if best is None:
            raise ValueError(
                f"在 max_cross={max_cross} 的约束下只能找到 {len(chosen)} 个正交悬挂端,"
                f"要不到 {n} 个。放宽 max_cross,或减少片段数 —— 4 nt 悬挂端"
                f"本身的正交容量有限,这正是超过 ~6 个片段就要认真选端的原因。")
        chosen.append(best)
        pool.remove(best)
    return chosen


@register_function(
    aliases=["plot_overhang_fidelity", "保真度矩阵图", "plot_fidelity",
             "悬挂端矩阵", "交叉反应图"],
    category="synthetic_biology",
    description="画悬挂端交叉反应矩阵:对角线是正确连接,非对角亮点就是会错配的对。一眼看出一组悬挂端好不好。Plot the overhang cross-reactivity matrix.",
    examples=["ov.synbio.plot_overhang_fidelity(rep)"],
    related=["synbio.overhang_fidelity", "synbio.design_overhang_set"],
    requires={},
    produces={},
)
def plot_overhang_fidelity(report: FidelityReport, ax=None):
    """Heatmap of pairwise ligation propensity."""
    from ._plot import _mpl
    plt = _mpl()
    import numpy as np

    ohs = report.overhangs
    m = np.array([[report.matrix.get((a, b), 0.0) for b in ohs] for a in ohs])

    if ax is None:
        fig, ax = plt.subplots(figsize=(0.55 * len(ohs) + 3.0,
                                        0.55 * len(ohs) + 2.4))
    else:
        fig = ax.figure
    im = ax.imshow(m, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(ohs)))
    ax.set_yticks(range(len(ohs)))
    ax.set_xticklabels(ohs, rotation=90, fontsize=8, family="monospace")
    ax.set_yticklabels(ohs, fontsize=8, family="monospace")
    ax.set_title(f"{report.verdict} — fidelity {report.fidelity:.3f} "
                 f"({report.method})")
    fig.colorbar(im, ax=ax, fraction=0.046, label="ligation propensity")
    fig.tight_layout()
    return fig, ax


__all__ = ["overhang_fidelity", "FidelityReport", "CrossReaction",
           "design_overhang_set", "plot_overhang_fidelity"]
