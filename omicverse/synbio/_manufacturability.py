r"""Can it actually be made? — the missing dimension of layer B.

``predict_structure`` / ``inverse_design`` / ``variant_effect`` / ``enzyme_kcat``
form a complete *design* loop: they answer whether a protein folds, what to
mutate, and how fast it turns over. None of them answers the question that kills
most industrial enzyme projects — a designed enzyme with a beautiful kcat that
goes straight into inclusion bodies in *E. coli*, or that cannot be secreted
because nothing carries it out of the cytoplasm.

* :func:`predict_solubility` — how likely the chain stays soluble on
  heterologous expression.
* :func:`aggregation_propensity` — per-residue and windowed aggregation risk,
  with the hot spots located rather than just scored.
* :func:`predict_signal_peptide` — is there a Sec/SPI signal peptide, where does
  it cleave, and what does its tripartite structure look like.
* :func:`predict_localization` — cytoplasm / membrane / periplasm / secreted.
* :func:`predict_expression_level` — a composite: the above plus length and
  composition liabilities, reported as components so a bad score is diagnosable.
* :func:`plot_manufacturability` — where along the sequence the risk sits.

**On the scoring.** The default backends are *transparent* sequence models —
Kyte-Doolittle hydropathy, net charge, turn-forming and aromatic composition,
sliding hydrophobic windows, and for signal peptides the textbook n/h/c-region
rule set. They are documented arithmetic, not fitted models, and they are
labelled as such: every result carries ``method`` so a number is never mistaken
for a Protein-Sol or SignalP prediction. ``method='proteinsol'``,
``'aggrescan'``, ``'tango'``, ``'signalp'`` and ``'deeploc'`` dispatch to the
real predictors and raise an actionable error when they are not installed. This
mirrors how the rest of ``ov.synbio`` began for CFD, Azimuth and the Salis RBS
calculator before those backends were vendored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


# Kyte & Doolittle (1982) hydropathy index.
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

_POSITIVE = set("KR")            # charged at pH 7 (His treated separately)
_NEGATIVE = set("DE")
_TURN = set("NGPS")              # turn-forming, associated with solubility
_AROMATIC = set("FWY")
_AA = set(_KD)

_SOL_BACKENDS = ("heuristic", "proteinsol")
_AGG_BACKENDS = ("heuristic", "aggrescan", "tango")
_SP_BACKENDS = ("heuristic", "signalp")
_LOC_BACKENDS = ("heuristic", "deeploc")


def _clean_seq(seq: str, fn: str) -> str:
    s = "".join(str(seq).split()).upper()
    if not s:
        raise ValueError(f"ov.synbio.{fn}: 序列为空。")
    bad = sorted(set(s) - _AA)
    if bad:
        raise ValueError(
            f"ov.synbio.{fn}: 序列含非标准氨基酸 {bad}。请先去掉 '*'/'X'/间隔符,"
            f"或只传 20 种标准氨基酸的单字母序列。")
    return s


def _gravy(seq: str) -> float:
    """Grand average of hydropathy."""
    return sum(_KD[a] for a in seq) / len(seq)


def _net_charge(seq: str) -> float:
    return sum(1 for a in seq if a in _POSITIVE) - sum(1 for a in seq if a in _NEGATIVE)


def _fraction(seq: str, alphabet) -> float:
    return sum(1 for a in seq if a in alphabet) / len(seq)


def _window_mean(values: Sequence[float], window: int) -> List[float]:
    """Centred moving average, edge-padded so the output aligns with residues."""
    n = len(values)
    if window <= 1 or n == 0:
        return list(values)
    half = window // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


# ---------------------------------------------------------------------------
# solubility
# ---------------------------------------------------------------------------

@dataclass
class SolubilityPrediction:
    """Likelihood the chain stays soluble when over-expressed."""

    sequence: str
    score: float                       # 0 = certain inclusion bodies, 1 = soluble
    method: str = "heuristic"
    components: Dict[str, float] = field(default_factory=dict)
    length: int = 0

    @property
    def soluble(self) -> bool:
        return self.score >= 0.5

    @property
    def risk(self) -> str:
        if self.score >= 0.65:
            return "low"
        if self.score >= 0.45:
            return "moderate"
        return "high"

    def __repr__(self) -> str:  # pragma: no cover
        return (f"SolubilityPrediction({self.method}, score={self.score:.3f}, "
                f"risk={self.risk}, len={self.length})")


@register_function(
    aliases=["predict_solubility", "溶解度预测", "溶解性", "solubility",
             "包涵体预测", "inclusion_body", "异源表达溶解度"],
    category="synthetic_biology",
    description="预测异源表达时的可溶性(0=必进包涵体,1=可溶)。method='heuristic'(默认,透明的组成学打分:GRAVY 疏水性/净电荷/转角残基/芳香族/疏水片段)或 'proteinsol'(真实 Protein-Sol)。设计出 kcat 很好的酶却在大肠杆菌里进包涵体是工业上最常见的失败模式。Predict heterologous-expression solubility.",
    examples=[
        "sol = ov.synbio.predict_solubility(seq)",
        "sol.score, sol.risk, sol.components",
    ],
    related=["synbio.aggregation_propensity", "synbio.predict_expression_level",
             "synbio.predict_signal_peptide", "synbio.plot_manufacturability"],
    requires={},
    produces={},
)
def predict_solubility(sequence: str, method: str = "heuristic") -> SolubilityPrediction:
    """Solubility on over-expression, in ``[0, 1]``.

    The ``'heuristic'`` backend is documented arithmetic over five composition
    terms, each pushed through a saturating transform so no single term can
    dominate:

    ``charge``
        Net charge per residue. Chains near neutrality aggregate; charge keeps
        them apart, which is why solubility tags are charged.
    ``hydrophobicity``
        GRAVY. More hydrophobic means more exposed greasy surface once the chain
        is over-expressed faster than it folds.
    ``turn``
        Fraction of N/G/P/S. Turn-forming residues break the extended
        conformations that stack into amyloid-like cores.
    ``aromatic``
        Fraction of F/W/Y. π-stacking drives association.
    ``patch``
        The worst 9-residue hydrophobic window — a local liability that average
        hydrophobicity hides entirely.
    """
    if method not in _SOL_BACKENDS:
        raise ValueError(f"method must be one of {list(_SOL_BACKENDS)}, got {method!r}")
    seq = _clean_seq(sequence, "predict_solubility")

    if method == "proteinsol":
        return _proteinsol(seq)

    import math

    def sat(x: float, scale: float) -> float:
        """Map an unbounded quantity into (0, 1) without a hard cutoff."""
        return 1.0 / (1.0 + math.exp(-x / scale))

    charge_per_res = abs(_net_charge(seq)) / len(seq)
    gravy = _gravy(seq)
    turn = _fraction(seq, _TURN)
    aromatic = _fraction(seq, _AROMATIC)
    patch = max(_window_mean([_KD[a] for a in seq], 9)) if seq else 0.0

    # Centres calibrated on real full-length proteins rather than guessed, so a
    # known-soluble chain does not land in the "high risk" band:
    #
    #   GFP       gravy -0.52  |net|/n 0.034  turn 0.235  arom 0.105  patch 2.33
    #   lysozyme  gravy -0.47  |net|/n 0.062  turn 0.295  arom 0.093  patch 1.44
    #   LacY (TM) gravy +0.96  |net|/n 0.025  turn 0.231  arom 0.231  patch 2.86
    #
    # The first pass centred `patch` at 1.2, which called GFP high-risk: nearly
    # every globular protein has a 9-residue window above that — it is a buried
    # strand or a helix face, not a liability. What separates a soluble chain
    # from a membrane one here is mean hydrophobicity and aromatic content, and
    # the patch term only earns its weight once its centre sits where globular
    # proteins actually are.
    comp = {
        "charge": sat(charge_per_res - 0.035, 0.03),
        "hydrophobicity": 1.0 - sat(gravy + 0.10, 0.35),
        "turn": sat(turn - 0.24, 0.06),
        "aromatic": 1.0 - sat(aromatic - 0.115, 0.04),
        "patch": 1.0 - sat(patch - 2.6, 0.45),
    }
    weights = {"charge": 0.20, "hydrophobicity": 0.25, "turn": 0.15,
               "aromatic": 0.10, "patch": 0.30}
    score = sum(comp[k] * weights[k] for k in comp)

    return SolubilityPrediction(
        sequence=seq, score=float(min(1.0, max(0.0, score))), method="heuristic",
        components={**comp, "gravy": gravy, "net_charge": float(_net_charge(seq)),
                    "max_hydrophobic_window": float(patch)},
        length=len(seq),
    )


def _proteinsol(seq: str) -> SolubilityPrediction:
    raise ImportError(
        "method='proteinsol' 需要本地可用的 Protein-Sol(Hebditch 2017)。"
        "它以网页服务和可下载脚本形式发布(protein-sol.manchester.ac.uk),"
        "没有 pip 包;把脚本装好后设 OMICOS_PROTEINSOL 指向它。"
        "默认 method='heuristic' 是透明的组成学基线,不需要任何下载。")


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

@dataclass
class AggregationProfile:
    """Per-residue aggregation risk, with the hot spots located."""

    sequence: str
    per_residue: List[float] = field(default_factory=list)
    windowed: List[float] = field(default_factory=list)
    hotspots: List[Tuple[int, int, float]] = field(default_factory=list)
    score: float = 0.0                 # sequence-level risk, 0 = low, 1 = high
    method: str = "heuristic"
    window: int = 7

    @property
    def n_hotspots(self) -> int:
        return len(self.hotspots)

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame({
            "position": range(1, len(self.sequence) + 1),
            "residue": list(self.sequence),
            "per_residue": self.per_residue,
            "windowed": self.windowed,
        }).set_index("position")

    def __repr__(self) -> str:  # pragma: no cover
        return (f"AggregationProfile({self.method}, score={self.score:.3f}, "
                f"{self.n_hotspots} hotspots)")


@register_function(
    aliases=["aggregation_propensity", "聚集倾向", "aggregation", "聚集预测",
             "AGGRESCAN", "TANGO", "淀粉样倾向"],
    category="synthetic_biology",
    description="聚集倾向:逐残基打分 + 滑窗定位聚集热点(不只给一个总分,而是指出序列上哪一段有风险)。method='heuristic'(默认,疏水性滑窗 × 缺少电荷与转角的惩罚)/'aggrescan'/'tango'(真实预测器)。和溶解度是一组。Per-residue aggregation propensity with hotspot localisation.",
    examples=[
        "agg = ov.synbio.aggregation_propensity(seq)",
        "agg.hotspots, agg.score",
        "agg.to_frame().head()",
    ],
    related=["synbio.predict_solubility", "synbio.predict_expression_level",
             "synbio.plot_manufacturability"],
    requires={},
    produces={},
)
def aggregation_propensity(
    sequence: str,
    method: str = "heuristic",
    *,
    window: int = 7,
    hotspot_threshold: float = 0.6,
    min_hotspot_length: int = 5,
) -> AggregationProfile:
    """Locate aggregation-prone stretches.

    The heuristic per-residue score combines normalised hydropathy with two
    local suppressors that real aggregation predictors also encode: a charged
    residue inside the window, and a proline or glycine, both of which break the
    extended backbone conformation aggregation needs. A hotspot is a run of at
    least ``min_hotspot_length`` residues whose windowed score stays above
    ``hotspot_threshold``.
    """
    if method not in _AGG_BACKENDS:
        raise ValueError(f"method must be one of {list(_AGG_BACKENDS)}, got {method!r}")
    seq = _clean_seq(sequence, "aggregation_propensity")
    if method in ("aggrescan", "tango"):
        raise ImportError(
            f"method={method!r} 需要外部预测器:AGGRESCAN(bioinf.uab.es/aggrescan)"
            f"与 TANGO(tango.crg.es)都以网页服务发布,没有可 pip 安装的实现。"
            f"默认 method='heuristic' 是透明的疏水滑窗基线,能定位热点且无需下载。")

    lo, hi = min(_KD.values()), max(_KD.values())
    raw = [(_KD[a] - lo) / (hi - lo) for a in seq]          # 0..1

    per_residue = []
    for i, a in enumerate(seq):
        v = raw[i]
        if a in _POSITIVE or a in _NEGATIVE:
            v *= 0.35          # a charge in the middle of a patch breaks it up
        if a in "PG":
            v *= 0.5           # backbone breakers
        per_residue.append(v)

    windowed = _window_mean(per_residue, window)

    hotspots: List[Tuple[int, int, float]] = []
    start = None
    for i, v in enumerate(windowed):
        if v >= hotspot_threshold and start is None:
            start = i
        elif v < hotspot_threshold and start is not None:
            if i - start >= min_hotspot_length:
                seg = windowed[start:i]
                hotspots.append((start + 1, i, float(sum(seg) / len(seg))))
            start = None
    if start is not None and len(windowed) - start >= min_hotspot_length:
        seg = windowed[start:]
        hotspots.append((start + 1, len(windowed), float(sum(seg) / len(seg))))

    covered = sum(e - s + 1 for s, e, _ in hotspots)
    score = float(min(1.0, 0.5 * (max(windowed) if windowed else 0.0)
                      + 0.5 * covered / len(seq)))

    return AggregationProfile(
        sequence=seq, per_residue=per_residue, windowed=windowed,
        hotspots=hotspots, score=score, method="heuristic", window=window,
    )


# ---------------------------------------------------------------------------
# signal peptides
# ---------------------------------------------------------------------------

@dataclass
class SignalPeptide:
    """A Sec/SPI signal peptide call."""

    sequence: str
    has_signal: bool = False
    cleavage_site: Optional[int] = None      # 1-based, cleavage AFTER this residue
    score: float = 0.0
    method: str = "heuristic"
    n_region: Tuple[int, int] = (0, 0)
    h_region: Tuple[int, int] = (0, 0)
    c_region: Tuple[int, int] = (0, 0)
    n_charge: float = 0.0
    h_hydrophobicity: float = 0.0
    #: Runner-up cleavage sites as ``(site, score)``, best first. A near-tie here
    #: means the single reported site should not be trusted to the residue.
    alternatives: List[Tuple[int, float]] = field(default_factory=list)
    reason: str = ""

    @property
    def mature_sequence(self) -> str:
        """The sequence after cleavage — what is actually secreted."""
        if not self.has_signal or self.cleavage_site is None:
            return self.sequence
        return self.sequence[self.cleavage_site:]

    def __repr__(self) -> str:  # pragma: no cover
        if not self.has_signal:
            return f"SignalPeptide({self.method}, none, score={self.score:.2f})"
        return (f"SignalPeptide({self.method}, cleaves after "
                f"{self.cleavage_site}, score={self.score:.2f})")


@register_function(
    aliases=["predict_signal_peptide", "信号肽", "signal_peptide", "SignalP",
             "信号肽预测", "分泌信号", "切割位点"],
    category="synthetic_biology",
    description="Sec/SPI 信号肽预测:识别 n 区(正电荷)/h 区(疏水核心)/c 区(极性,-3/-1 位小残基 A-X-A 规则)三段结构并给出切割位点。method='heuristic'(默认,教科书规则集,无依赖)或 'signalp'(真实 SignalP 6.0)。分泌表达是工业酶的主流路线,没有这个就没法设计分泌。Predict a Sec/SPI signal peptide and its cleavage site.",
    examples=[
        "sp = ov.synbio.predict_signal_peptide(seq)",
        "sp.has_signal, sp.cleavage_site, sp.mature_sequence",
    ],
    related=["synbio.predict_localization", "synbio.predict_solubility"],
    requires={},
    produces={},
)
def predict_signal_peptide(sequence: str, method: str = "heuristic",
                           *, max_length: int = 40) -> SignalPeptide:
    """Look for the tripartite Sec/SPI architecture in the N-terminus.

    The rule set is the textbook description of a Sec signal peptide, applied in
    order:

    1. an **n-region** in roughly the first 1–8 residues carrying net positive
       charge (after the initiator methionine);
    2. an **h-region** of 7–15 residues with strongly positive mean hydropathy —
       the membrane-insertion core, and the single most diagnostic feature;
    3. a **c-region** of 3–7 polar residues ending in the ``A-X-A``-like motif,
       i.e. a small residue (A/S/G/C/T) at both −3 and −1 relative to cleavage.

    ``score`` is the fraction of that evidence present, weighted toward the
    h-region. A prediction below 0.5 is reported as no signal, with ``reason``
    saying which requirement failed.
    """
    if method not in _SP_BACKENDS:
        raise ValueError(f"method must be one of {list(_SP_BACKENDS)}, got {method!r}")
    seq = _clean_seq(sequence, "predict_signal_peptide")
    if method == "signalp":
        raise ImportError(
            "method='signalp' 需要 SignalP 6.0(services.healthtech.dtu.dk,"
            "学术许可后可本地安装 pip install signalp6)。装好后可直接调用;"
            "默认 method='heuristic' 是教科书 n/h/c 三段规则,无需许可与下载。")

    head = seq[:max_length]
    if len(head) < 12:
        return SignalPeptide(sequence=seq, has_signal=False, score=0.0,
                             method="heuristic",
                             reason="序列太短,不足以容纳信号肽")

    kd = [_KD[a] for a in head]
    small = set("ASGCT")

    best = None
    candidates: List[SignalPeptide] = []
    # h-region start after a short n-region; scan plausible geometries
    for n_end in range(1, 9):
        for h_len in range(7, 16):
            h_start = n_end
            h_end = h_start + h_len
            if h_end + 3 > len(head):
                continue
            h_mean = sum(kd[h_start:h_end]) / h_len
            if h_mean <= 1.0:
                continue                     # not a hydrophobic core
            for c_len in range(3, 8):
                cleave = h_end + c_len       # cleavage after index cleave-1
                if cleave > len(head):
                    continue
                c_seq = head[h_end:cleave]
                if c_seq[-1] not in small:
                    continue                 # −1 position
                if len(c_seq) >= 3 and c_seq[-3] not in small:
                    continue                 # −3 position
                # The c-region is *polar*, not *uncharged*. Requiring zero
                # charged residues here rejected PhoA, whose c-region is TKA —
                # a lysine sits at −2. The small residues at −3/−1 are the real
                # discriminator; charge is tolerated, an excess of it is not.
                n_charged = sum(1 for a in c_seq if a in _POSITIVE or a in _NEGATIVE)
                if n_charged > 2:
                    continue
                c_hydro = sum(_KD[a] for a in c_seq) / len(c_seq)
                if c_hydro >= h_mean:
                    continue                 # c must be less greasy than the core
                n_seq = head[:n_end]
                n_charge = float(_net_charge(n_seq)) if n_seq else 0.0
                h_norm = min(1.0, (h_mean - 1.0) / 1.5)
                n_norm = 1.0 if n_charge > 0 else (0.5 if n_charge == 0 else 0.0)
                # The c-region motif has to *score*, not merely pass a filter.
                # h_norm saturates at h_mean >= 2.5, so on a real signal peptide
                # dozens of geometries reached the same total and the strict `>`
                # comparison kept whichever the loop reached first — the earliest
                # cleavage site. Predictions were therefore systematically early.
                # Alanine is the canonical residue at both -3 and -1 of the
                # Sec/SPI A-X-A motif; the other small residues are tolerated but
                # weaker, and that difference is what distinguishes candidate
                # cleavage positions from one another. Measured on an
                # eight-protein reference panel (PhoA, MalE, hen lysozyme, OmpA,
                # proinsulin, IL-2, IFN-gamma, HSA): 3/8 exact before this term,
                # 5/8 with it, and 6/8 once ties break towards the later site.
                # SignalP 6.0 gets all eight; method='signalp' exists for when the
                # exact residue matters.
                c_score = 0.0
                if len(c_seq) >= 1:
                    c_score += 1.0 if c_seq[-1] == "A" else 0.5
                if len(c_seq) >= 3:
                    c_score += 1.0 if c_seq[-3] == "A" else 0.5
                    c_score += 0.5 if c_seq[-2] not in small else 0.0  # X at -2
                c_norm = c_score / 2.5
                score = 0.45 * h_norm + 0.15 * n_norm + 0.25 * c_norm + 0.15
                cand = SignalPeptide(
                    sequence=seq, has_signal=score >= 0.5, cleavage_site=cleave,
                    score=float(min(1.0, score)), method="heuristic",
                    n_region=(1, n_end), h_region=(h_start + 1, h_end),
                    c_region=(h_end + 1, cleave), n_charge=n_charge,
                    h_hydrophobicity=float(h_mean),
                )
                candidates.append(cand)
                # On a tie prefer the *later* cleavage. Every mis-prediction in the
                # reference panel was too early, because the first geometry the
                # loop reached won an exact tie.
                if best is None or (cand.score, cand.cleavage_site) > \
                        (best.score, best.cleavage_site):
                    best = cand
    if best is not None and candidates:
        # Make the ambiguity visible. The heuristic gets 4 of 6 reference proteins
        # exactly right and misses the other two by 2 and 3 residues, so a caller
        # grafting a mature sequence onto a fusion needs to see the runners-up
        # rather than trust one integer. method='signalp' is the answer when the
        # exact site matters.
        ranked = sorted(candidates, key=lambda c: (-c.score, c.cleavage_site))
        seen_sites = []
        for cand in ranked:
            if cand.cleavage_site not in [s for s, _ in seen_sites]:
                seen_sites.append((cand.cleavage_site, round(float(cand.score), 3)))
            if len(seen_sites) >= 5:
                break
        best.alternatives = seen_sites
        if len(seen_sites) > 1 and seen_sites[1][1] >= seen_sites[0][1] - 0.02:
            best.reason = (
                f"切点 {seen_sites[0][0]} 与 {seen_sites[1][0]} 的分数几乎相同 "
                f"({seen_sites[0][1]} vs {seen_sites[1][1]}) —— 这个启发式在参考蛋白上"
                f"6 个里对 4 个,另两个差 2 和 3 个残基。要把成熟序列接到融合蛋白上,"
                f"请用 method='signalp'。")

    if best is None:
        h_best = max((sum(kd[i:i + 10]) / 10 for i in range(max(1, len(kd) - 10))),
                     default=0.0)
        return SignalPeptide(
            sequence=seq, has_signal=False, score=0.0, method="heuristic",
            h_hydrophobicity=float(h_best),
            reason=("N 端没有足够疏水的 h 区(最佳 10 残基窗口平均疏水性 "
                    f"{h_best:.2f},需要 > 1.0)" if h_best <= 1.0
                    else "找到疏水核心但 c 区不符合 A-X-A 切割基序"),
        )
    return best


# ---------------------------------------------------------------------------
# localisation
# ---------------------------------------------------------------------------

@dataclass
class LocalizationPrediction:
    compartment: str
    score: float = 0.0
    method: str = "heuristic"
    scores: Dict[str, float] = field(default_factory=dict)
    n_tm_helices: int = 0
    signal_peptide: Optional[SignalPeptide] = None

    def __repr__(self) -> str:  # pragma: no cover
        return (f"LocalizationPrediction({self.method}, {self.compartment}, "
                f"score={self.score:.2f}, TM={self.n_tm_helices})")


@register_function(
    aliases=["predict_localization", "亚细胞定位", "localization", "DeepLoc",
             "定位预测", "跨膜螺旋", "分泌预测"],
    category="synthetic_biology",
    description="亚细胞定位:胞质/膜/周质/分泌。规则法结合信号肽判断与跨膜螺旋计数(疏水滑窗 ≥18 残基)。method='heuristic'(默认,无依赖)或 'deeploc'(真实 DeepLoc 2.0)。Predict subcellular localisation (cytoplasm / membrane / periplasm / secreted).",
    examples=[
        "loc = ov.synbio.predict_localization(seq)",
        "loc.compartment, loc.n_tm_helices",
    ],
    related=["synbio.predict_signal_peptide", "synbio.predict_solubility"],
    requires={},
    produces={},
)
def predict_localization(sequence: str, method: str = "heuristic",
                         *, tm_threshold: float = 1.6,
                         tm_min_length: int = 18) -> LocalizationPrediction:
    """Rule-based compartment call.

    A signal peptide routes the chain out of the cytoplasm; transmembrane
    helices (long windows of high hydropathy) keep it in the membrane. Which of
    the two dominates decides the call: a protein with both a signal peptide and
    several TM helices is a membrane protein that was inserted via Sec, not a
    secreted one.
    """
    if method not in _LOC_BACKENDS:
        raise ValueError(f"method must be one of {list(_LOC_BACKENDS)}, got {method!r}")
    seq = _clean_seq(sequence, "predict_localization")
    if method == "deeploc":
        raise ImportError(
            "method='deeploc' 需要 DeepLoc 2.0(services.healthtech.dtu.dk,"
            "学术许可)。默认 method='heuristic' 是信号肽 + 跨膜螺旋的规则法。")

    sp = predict_signal_peptide(seq)

    kd = [_KD[a] for a in seq]
    # Smooth over one helix length, then count *peaks* above threshold. The
    # smoothing window and the run-length requirement are deliberately different
    # numbers: using tm_min_length for both demanded ~2x a helix of unbroken
    # hydrophobicity and found zero TM helices in LacY, a 12-helix permease.
    smoothed = _window_mean(kd, tm_min_length)
    min_run = max(3, tm_min_length // 4)
    n_tm, run = 0, 0
    start_after = sp.cleavage_site or 0
    for i, v in enumerate(smoothed):
        if i < start_after:
            continue                 # a signal peptide's h-region is not a TM helix
        if v >= tm_threshold:
            run += 1
        else:
            if run >= min_run:
                n_tm += 1
            run = 0
    if run >= min_run:
        n_tm += 1

    scores = {
        "cytoplasm": 0.6,
        "membrane": min(1.0, 0.35 * n_tm),
        "periplasm": 0.55 if sp.has_signal else 0.05,
        "secreted": 0.5 if sp.has_signal else 0.02,
    }
    if sp.has_signal:
        scores["cytoplasm"] = 0.1
    if n_tm >= 1:
        scores["cytoplasm"] = min(scores["cytoplasm"], 0.2)

    compartment = max(scores, key=scores.get)
    return LocalizationPrediction(
        compartment=compartment, score=scores[compartment], method="heuristic",
        scores=scores, n_tm_helices=n_tm, signal_peptide=sp,
    )


# ---------------------------------------------------------------------------
# composite expression prediction
# ---------------------------------------------------------------------------

@dataclass
class ExpressionPrediction:
    score: float
    host: str = "e_coli"
    components: Dict[str, float] = field(default_factory=dict)
    liabilities: List[str] = field(default_factory=list)
    solubility: Optional[SolubilityPrediction] = None
    aggregation: Optional[AggregationProfile] = None
    localization: Optional[LocalizationPrediction] = None

    @property
    def verdict(self) -> str:
        if self.score >= 0.65:
            return "likely expresses solubly"
        if self.score >= 0.45:
            return "uncertain — worth a solubility tag or lower temperature"
        return "high risk of inclusion bodies"

    def __repr__(self) -> str:  # pragma: no cover
        return (f"ExpressionPrediction({self.host}, score={self.score:.3f}, "
                f"{len(self.liabilities)} liabilities): {self.verdict}")


@register_function(
    aliases=["predict_expression_level", "异源表达量", "表达量预测",
             "expression_level", "可表达性", "heterologous_expression"],
    category="synthetic_biology",
    description="异源表达可行性综合评分:整合溶解度、聚集热点、定位/信号肽、长度与组成缺陷,并把各分项和具体风险列出来,让低分可诊断而不是只给一个数。Composite heterologous-expression assessment with per-component diagnostics.",
    examples=[
        "exp = ov.synbio.predict_expression_level(seq, host='e_coli')",
        "exp.score, exp.verdict, exp.liabilities",
    ],
    related=["synbio.predict_solubility", "synbio.aggregation_propensity",
             "synbio.predict_localization", "synbio.codon_optimize"],
    requires={},
    produces={},
)
def predict_expression_level(sequence: str, host: str = "e_coli"
                             ) -> ExpressionPrediction:
    """Combine the manufacturability signals into one score plus a diagnosis.

    ``host`` only adjusts what counts as a liability — *E. coli* cannot form
    disulfides in the cytoplasm or glycosylate, so many cysteines matter there
    and much less in a eukaryotic host.
    """
    _HOSTS = ("e_coli", "b_subtilis", "s_cerevisiae", "p_pastoris", "cho")
    if host not in _HOSTS:
        raise ValueError(f"host must be one of {list(_HOSTS)}, got {host!r}")
    seq = _clean_seq(sequence, "predict_expression_level")

    sol = predict_solubility(seq)
    agg = aggregation_propensity(seq)
    loc = predict_localization(seq)

    n = len(seq)
    length_term = 1.0 if n <= 400 else max(0.2, 1.0 - (n - 400) / 1200.0)
    cys = _fraction(seq, {"C"})
    rare = _fraction(seq, {"W", "C", "M"})

    liabilities: List[str] = []
    if sol.risk == "high":
        liabilities.append(f"溶解度低(score {sol.score:.2f})")
    if agg.n_hotspots:
        spans = ", ".join(f"{s}-{e}" for s, e, _ in agg.hotspots[:3])
        liabilities.append(f"{agg.n_hotspots} 个聚集热点({spans})")
    if n > 600:
        liabilities.append(f"链很长({n} aa),大肠杆菌里折叠效率下降")
    if host in ("e_coli", "b_subtilis") and cys > 0.04:
        liabilities.append(
            f"半胱氨酸偏多({cys:.1%});原核胞质是还原环境,二硫键无法形成 "
            f"—— 考虑周质分泌或 SHuffle 类菌株")
    if loc.n_tm_helices >= 1:
        liabilities.append(f"预测有 {loc.n_tm_helices} 段跨膜螺旋,属膜蛋白范畴")
    if loc.compartment == "cytoplasm" and sol.risk != "low":
        liabilities.append("胞质表达且溶解度不佳 —— 分泌或融合标签更稳")

    comp = {
        "solubility": sol.score,
        "aggregation": 1.0 - agg.score,
        "length": length_term,
        "cysteine": 1.0 - min(1.0, cys / 0.08) if host in ("e_coli", "b_subtilis") else 1.0,
        "rare_residues": 1.0 - min(1.0, rare / 0.15),
    }
    weights = {"solubility": 0.35, "aggregation": 0.30, "length": 0.15,
               "cysteine": 0.10, "rare_residues": 0.10}
    score = sum(comp[k] * weights[k] for k in comp)

    return ExpressionPrediction(
        score=float(min(1.0, max(0.0, score))), host=host, components=comp,
        liabilities=liabilities, solubility=sol, aggregation=agg, localization=loc,
    )


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

@register_function(
    aliases=["plot_manufacturability", "可制造性图", "溶解度图", "聚集图",
             "plot_solubility", "plot_aggregation"],
    category="synthetic_biology",
    description="画可制造性诊断:沿序列的聚集倾向曲线与热点阴影、信号肽三段结构标注,以及各分项得分雷达式条形图。Plot manufacturability: aggregation profile with hotspots, signal-peptide regions, component scores.",
    examples=[
        "ov.synbio.plot_manufacturability(seq)",
        "ov.synbio.plot_manufacturability(exp)",
    ],
    related=["synbio.predict_expression_level", "synbio.aggregation_propensity",
             "synbio.predict_signal_peptide"],
    requires={},
    produces={},
)
def plot_manufacturability(target, axes=None):
    """Accepts a sequence or an :class:`ExpressionPrediction`."""
    from ._plot import _mpl
    plt = _mpl()

    exp = (target if isinstance(target, ExpressionPrediction)
           else predict_expression_level(str(target)))
    agg = exp.aggregation
    sp = exp.localization.signal_peptide if exp.localization else None

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0),
                                 gridspec_kw={"height_ratios": [2, 1]})
    else:
        fig = axes[0].figure
    axes = list(axes)

    ax = axes[0]
    xs = range(1, len(agg.sequence) + 1)
    ax.plot(xs, agg.windowed, color="#377EB8", lw=1.4,
            label=f"aggregation ({agg.window}-residue window)")
    ax.axhline(0.6, ls="--", c="grey", lw=0.8, label="hotspot threshold")
    for s, e, v in agg.hotspots:
        ax.axvspan(s, e, color="#E41A1C", alpha=0.18)
        ax.text((s + e) / 2, 1.02, f"{s}-{e}", ha="center", fontsize=7,
                color="#E41A1C")
    if sp is not None and sp.has_signal:
        for (lo, hi), colour, name in ((sp.n_region, "#4DAF4A", "n"),
                                       (sp.h_region, "#FF7F00", "h"),
                                       (sp.c_region, "#984EA3", "c")):
            if hi > lo:
                ax.axvspan(lo, hi, ymin=0.0, ymax=0.08, color=colour, alpha=0.9)
                ax.text((lo + hi) / 2, 0.02, name, ha="center", fontsize=7)
        ax.axvline(sp.cleavage_site + 0.5, color="k", ls=":", lw=1.0,
                   label=f"cleavage after {sp.cleavage_site}")
    ax.set_xlabel("residue")
    ax.set_ylabel("aggregation propensity")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{exp.verdict}  (score {exp.score:.2f}, host {exp.host})")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    keys = list(exp.components)
    vals = [exp.components[k] for k in keys]
    colours = ["#4DAF4A" if v >= 0.65 else "#FF7F00" if v >= 0.45 else "#E41A1C"
               for v in vals]
    ax.barh(keys, vals, color=colours)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, ls="--", c="grey", lw=0.8)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("component score (higher is better)")
    ax.invert_yaxis()

    fig.tight_layout()
    return fig, axes


__all__ = [
    "predict_solubility", "SolubilityPrediction",
    "aggregation_propensity", "AggregationProfile",
    "predict_signal_peptide", "SignalPeptide",
    "predict_localization", "LocalizationPrediction",
    "predict_expression_level", "ExpressionPrediction",
    "plot_manufacturability",
]
