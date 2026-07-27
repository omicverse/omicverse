r"""Beyond k_cat — affinity, catalytic efficiency, and substrate scope.

``enzyme_kcat`` gives the turnover number, and a turnover number alone often
answers the wrong question. Inside a cell the substrate concentration is rarely
saturating, so the rate is set by ``k_cat/K_M`` and not by ``k_cat``; a variant
that doubles turnover while raising K_M fourfold is slower where it matters. And
when a heterologous pathway is being assembled, the question is frequently not
"how fast" but "will this enzyme accept a substrate it never evolved on".

* :func:`enzyme_km` — Michaelis constant for an enzyme/substrate pair.
* :func:`catalytic_efficiency` — ``k_cat/K_M`` with the diffusion limit made
  explicit, because an efficiency above ~10^8–10^9 M⁻¹s⁻¹ is not a better enzyme
  but a broken prediction.
* :func:`substrate_specificity` — rank a panel of candidate substrates for one
  enzyme, so a designed pathway step can be checked before it is built.
* :func:`plot_substrate_scope` — the specificity profile and where the native
  substrate sits in it.

Backends follow this module's existing convention: ``method='baseline'`` is a
deterministic, sequence-sensitive estimator meant for *relative* comparison and
wiring, and it says so; ``method='unikp'`` / ``'dlkcat'`` dispatch to the trained
predictors when they are installed. Every result carries ``method``, and
:attr:`KineticsPrediction.quantitative` is False for the baseline so a caller
can refuse to use it as a calibrated number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


# Diffusion-limited association is ~10^8–10^9 M^-1 s^-1. Anything above this is
# physically impossible, so it is a ceiling on predictions rather than a good
# score. Catalytically "perfect" enzymes (triosephosphate isomerase, catalase)
# sit right at it.
DIFFUSION_LIMIT = 1e9

_KM_BACKENDS = ("baseline", "unikp")
_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _clean_seq(seq: str, fn: str) -> str:
    s = "".join(str(seq).split()).upper()
    if not s:
        raise ValueError(f"ov.synbio.{fn}: 酶序列为空。")
    bad = sorted(set(s) - _AA)
    if bad:
        raise ValueError(
            f"ov.synbio.{fn}: 序列含非标准氨基酸 {bad}(只接受 20 种标准残基)。")
    return s


def _substrate_descriptor(smiles: str) -> Dict[str, float]:
    """Cheap composition descriptors — no rdkit required.

    Enough to separate small polar metabolites from large hydrophobic ones,
    which is the axis substrate affinity varies along most strongly.
    """
    s = str(smiles)
    heavy = sum(1 for c in s if c.isalpha() and c.upper() not in ("H",))
    polar = sum(1 for c in s if c in "ONPS")
    charged = s.count("+") + s.count("-")
    rings = sum(1 for c in s if c.isdigit())
    return {
        "size": float(heavy),
        "polarity": polar / max(1.0, heavy),
        "charge": float(charged),
        "aromaticity": rings / max(1.0, heavy),
    }


@dataclass
class KineticsPrediction:
    """A kinetic constant, with its provenance attached."""

    sequence: str
    substrate: str
    km: float                       # M
    method: str = "baseline"
    kcat: Optional[float] = None    # 1/s, when supplied or predicted
    quantitative: bool = False
    descriptors: Dict[str, float] = field(default_factory=dict)

    @property
    def kcat_over_km(self) -> Optional[float]:
        """Catalytic efficiency, M⁻¹s⁻¹."""
        if self.kcat is None or self.km <= 0:
            return None
        return self.kcat / self.km

    @property
    def diffusion_limited(self) -> bool:
        eff = self.kcat_over_km
        return eff is not None and eff >= DIFFUSION_LIMIT

    def __repr__(self) -> str:  # pragma: no cover
        eff = self.kcat_over_km
        tail = "" if eff is None else f", kcat/KM={eff:.3g} /M/s"
        return (f"KineticsPrediction(KM={self.km:.3g} M, method={self.method!r}, "
                f"quantitative={self.quantitative}{tail})")


def _baseline_km(seq: str, smiles: str) -> Tuple[float, Dict[str, float]]:
    """Deterministic, sequence-sensitive K_M estimate.

    Centred on the BRENDA median (~10⁻⁴ M) and modulated by two things that do
    carry real signal: the substrate's size and polarity, and the composition of
    the enzyme's putative binding environment (aromatic and charged residue
    content, which is what a pocket uses to hold a ligand).

    This responds to mutation — the point of a baseline in this module is that a
    variant gives a different number — but it is not calibrated. Treat it as an
    ordering, not a measurement.
    """
    import hashlib
    import math

    d = _substrate_descriptor(smiles)

    aromatic = sum(1 for a in seq if a in "FWY") / len(seq)
    charged = sum(1 for a in seq if a in "DEKR") / len(seq)

    # log10(K_M) around -4 (100 µM), the BRENDA median
    log_km = -4.0
    log_km += 0.6 * (d["size"] / 20.0 - 1.0)      # bigger substrates bind tighter
    log_km -= 1.2 * (d["polarity"] - 0.3)         # polar contacts tighten binding
    log_km -= 2.0 * (aromatic - 0.08)             # aromatic pockets bind tighter
    log_km -= 1.0 * (charged - 0.25) * d["charge"]

    # a small, deterministic sequence-specific term so distinct enzymes differ
    h = hashlib.sha256(seq.encode()).digest()
    jitter = (int.from_bytes(h[:4], "big") / 2 ** 32 - 0.5) * 0.6
    log_km += jitter

    log_km = max(-8.0, min(-1.0, log_km))
    return float(10 ** log_km), {**d, "aromatic_fraction": aromatic,
                                 "charged_fraction": charged}


@register_function(
    aliases=["enzyme_km", "米氏常数", "Km", "KM", "底物亲和力", "affinity",
             "michaelis_constant"],
    category="synthetic_biology",
    description="预测酶对底物的米氏常数 K_M(M)。细胞内底物浓度通常远未饱和,真正限速的是 k_cat/K_M 而不是 k_cat —— 一个把周转数翻倍但 K_M 涨四倍的变体,在实际条件下更慢。method='baseline'(默认,确定性、对突变敏感,用于相对比较;quantitative=False)或 'unikp'(真实训练模型)。Predict the Michaelis constant for an enzyme/substrate pair.",
    examples=[
        "km = ov.synbio.enzyme_km(enzyme_seq, 'OCC1OC(O)C(O)C(O)C1O')",
        "km.km, km.method, km.quantitative",
    ],
    related=["synbio.enzyme_kcat", "synbio.catalytic_efficiency",
             "synbio.substrate_specificity", "synbio.ec_model"],
    requires={},
    produces={},
)
def enzyme_km(sequence: str, substrate: str, method: str = "baseline",
              **kwargs) -> KineticsPrediction:
    """Michaelis constant in molar.

    Parameters
    ----------
    sequence
        Enzyme amino-acid sequence.
    substrate
        Substrate SMILES.
    method
        ``'baseline'`` (default) or ``'unikp'`` (UniKP, trained; requires the
        package and its weights).
    """
    if method not in _KM_BACKENDS:
        raise ValueError(f"method must be one of {list(_KM_BACKENDS)}, got {method!r}")
    seq = _clean_seq(sequence, "enzyme_km")

    if method == "unikp":
        raise ImportError(
            "method='unikp' 需要 UniKP(github.com/Luo-SynBioLab/UniKP,无 PyPI 包)"
            "及其训练权重。装好后设 OMICOS_UNIKP 指向 checkout。"
            "默认 method='baseline' 是确定性基线,适合相对比较,"
            "结果的 quantitative=False 明确标出它不是标定值。")

    km, desc = _baseline_km(seq, substrate)
    return KineticsPrediction(sequence=seq, substrate=str(substrate), km=km,
                              method="baseline", quantitative=False,
                              descriptors=desc)


@register_function(
    aliases=["catalytic_efficiency", "催化效率", "kcat_over_km", "kcat/Km",
             "特异性常数", "specificity_constant"],
    category="synthetic_biology",
    description="催化效率 k_cat/K_M(M⁻¹s⁻¹)—— 底物不饱和时真正决定速率的参数。同时给出扩散极限(~10⁹ M⁻¹s⁻¹):超过它的不是更好的酶,而是坏掉的预测。Catalytic efficiency with the diffusion limit made explicit.",
    examples=[
        "eff = ov.synbio.catalytic_efficiency(seq, smiles)",
        "eff.kcat_over_km, eff.diffusion_limited",
    ],
    related=["synbio.enzyme_km", "synbio.enzyme_kcat",
             "synbio.substrate_specificity"],
    requires={},
    produces={},
)
def catalytic_efficiency(
    sequence: str,
    substrate: str,
    *,
    kcat: Optional[float] = None,
    km: Optional[float] = None,
    method: str = "baseline",
) -> KineticsPrediction:
    """``k_cat / K_M``, predicting either constant that is not supplied.

    Pass measured ``kcat=`` / ``km=`` whenever you have them — a measured
    constant beats any prediction, and mixing one measured with one predicted
    value is a normal and useful thing to do.
    """
    seq = _clean_seq(sequence, "catalytic_efficiency")

    if km is None:
        km_pred = enzyme_km(seq, substrate, method=method)
        km_value, desc, quantitative = km_pred.km, km_pred.descriptors, False
    else:
        km_value, desc, quantitative = float(km), _substrate_descriptor(substrate), True

    if kcat is None:
        from ._kcat import enzyme_kcat
        kcat_value = float(enzyme_kcat(seq, substrate).kcat)
        quantitative = False
    else:
        kcat_value = float(kcat)

    res = KineticsPrediction(
        sequence=seq, substrate=str(substrate), km=km_value, kcat=kcat_value,
        method=("measured" if (kcat is not None and km is not None) else method),
        quantitative=quantitative and kcat is not None, descriptors=desc)
    return res


# ---------------------------------------------------------------------------
# substrate scope
# ---------------------------------------------------------------------------

@dataclass
class SubstrateScope:
    """A ranked panel of candidate substrates for one enzyme."""

    sequence: str
    substrates: List[str] = field(default_factory=list)
    efficiency: Dict[str, float] = field(default_factory=dict)
    km: Dict[str, float] = field(default_factory=dict)
    kcat: Dict[str, float] = field(default_factory=dict)
    native: Optional[str] = None
    method: str = "baseline"
    quantitative: bool = False

    @property
    def ranked(self) -> List[str]:
        return sorted(self.substrates, key=lambda s: -self.efficiency.get(s, 0.0))

    @property
    def best(self) -> Optional[str]:
        return self.ranked[0] if self.substrates else None

    @property
    def promiscuity(self) -> float:
        """Normalised Shannon entropy of the efficiency profile, in ``[0, 1]``.

        0 means the enzyme is specific for one substrate in the panel; 1 means
        it treats them all alike. Entropy rather than a best/second ratio,
        because promiscuity is about the shape of the whole profile — an enzyme
        with three equally good substrates is more promiscuous than one with two,
        and a ratio cannot see that.
        """
        import math

        vals = [self.efficiency.get(s, 0.0) for s in self.substrates]
        total = sum(vals)
        if total <= 0 or len(vals) < 2:
            return 0.0
        ps = [v / total for v in vals if v > 0]
        h = -sum(p * math.log(p) for p in ps)
        return float(h / math.log(len(vals)))

    def fold_over_native(self, substrate: str) -> Optional[float]:
        if self.native is None or self.native not in self.efficiency:
            return None
        base = self.efficiency[self.native]
        return self.efficiency.get(substrate, 0.0) / base if base > 0 else None

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        rows = []
        for s in self.ranked:
            rows.append({
                "substrate": s,
                "km": self.km.get(s),
                "kcat": self.kcat.get(s),
                "kcat_over_km": self.efficiency.get(s),
                "fold_over_native": self.fold_over_native(s),
                "is_native": s == self.native,
            })
        return pd.DataFrame(rows).set_index("substrate")

    def __repr__(self) -> str:  # pragma: no cover
        return (f"SubstrateScope({len(self.substrates)} substrates, "
                f"best={self.best!r}, promiscuity={self.promiscuity:.2f}, "
                f"quantitative={self.quantitative})")


@register_function(
    aliases=["substrate_specificity", "底物特异性", "底物谱", "promiscuity",
             "substrate_scope", "混杂性", "非天然底物"],
    category="synthetic_biology",
    description="对一个酶排一组候选底物:按 k_cat/K_M 排序,给出相对天然底物的倍数,并用效率谱的归一化熵量化混杂性(promiscuity)。设计异源通路时,某一步的酶会不会接受非天然底物,是能不能建成的前提。Rank candidate substrates for an enzyme and quantify promiscuity.",
    examples=[
        "scope = ov.synbio.substrate_specificity(seq, ['CC(=O)O', 'CCC(=O)O'], native='CC(=O)O')",
        "scope.ranked, scope.promiscuity, scope.to_frame()",
    ],
    related=["synbio.enzyme_km", "synbio.catalytic_efficiency",
             "synbio.enzyme_function", "synbio.retro_biosynthesis",
             "synbio.plot_substrate_scope"],
    requires={},
    produces={},
)
def substrate_specificity(
    sequence: str,
    substrates: Sequence[str],
    *,
    native: Optional[str] = None,
    method: str = "baseline",
    kcat: Optional[Dict[str, float]] = None,
    km: Optional[Dict[str, float]] = None,
) -> SubstrateScope:
    """Score a panel of substrates for one enzyme.

    Parameters
    ----------
    sequence
        The enzyme.
    substrates
        Candidate substrate SMILES.
    native
        The enzyme's natural substrate, if it is in the panel — enables
        ``fold_over_native``, which is usually the number a designer wants.
    kcat, km
        Measured constants keyed by substrate, used in place of predictions
        wherever supplied.
    """
    seq = _clean_seq(sequence, "substrate_specificity")
    subs = [str(s) for s in substrates]
    if not subs:
        raise ValueError("substrates 不能为空。")
    if native is not None and native not in subs:
        raise ValueError(
            f"native={native!r} 不在 substrates 里 —— 天然底物必须也在候选面板中,"
            f"否则 fold_over_native 没有基准。")

    kcat = dict(kcat or {})
    km = dict(km or {})
    eff: Dict[str, float] = {}
    km_out: Dict[str, float] = {}
    kcat_out: Dict[str, float] = {}
    all_measured = True

    for s in subs:
        res = catalytic_efficiency(seq, s, kcat=kcat.get(s), km=km.get(s),
                                   method=method)
        km_out[s] = res.km
        kcat_out[s] = res.kcat if res.kcat is not None else float("nan")
        e = res.kcat_over_km
        eff[s] = 0.0 if e is None else float(e)
        if not (s in kcat and s in km):
            all_measured = False

    return SubstrateScope(
        sequence=seq, substrates=subs, efficiency=eff, km=km_out,
        kcat=kcat_out, native=native,
        method="measured" if all_measured else method,
        quantitative=all_measured)


@register_function(
    aliases=["plot_substrate_scope", "底物谱图", "特异性图", "plot_specificity",
             "混杂性图"],
    category="synthetic_biology",
    description="画底物谱:各底物的 k_cat/K_M 条形图(天然底物高亮)与 K_M–k_cat 散点,右上角是又快又紧的理想区。Plot an enzyme's substrate scope and the KM/kcat trade-off.",
    examples=["ov.synbio.plot_substrate_scope(scope)"],
    related=["synbio.substrate_specificity"],
    requires={},
    produces={},
)
def plot_substrate_scope(scope: SubstrateScope, axes=None):
    """Two panels: ranked efficiency, and the K_M / k_cat plane."""
    from ._plot import _mpl
    plt = _mpl()

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    else:
        fig = axes[0].figure
    axes = list(axes)

    ranked = scope.ranked
    labels = [s if len(s) <= 18 else s[:15] + "…" for s in ranked]

    ax = axes[0]
    vals = [scope.efficiency[s] for s in ranked]
    colours = ["#E41A1C" if s == scope.native else "#377EB8" for s in ranked]
    ax.barh(range(len(ranked)), vals, color=colours)
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(labels, fontsize=7, family="monospace")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.axvline(DIFFUSION_LIMIT, ls="--", c="k", lw=1,
               label="diffusion limit")
    ax.set_xlabel("k$_{cat}$/K$_M$ (M⁻¹s⁻¹)")
    ax.set_title(f"promiscuity {scope.promiscuity:.2f}"
                 + ("  (red = native)" if scope.native else ""))
    ax.legend(fontsize=7)

    ax = axes[1]
    for s in ranked:
        ax.scatter(scope.km[s], scope.kcat.get(s, float("nan")),
                   s=45, color="#E41A1C" if s == scope.native else "#377EB8",
                   alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K$_M$ (M)   ← tighter")
    ax.set_ylabel("k$_{cat}$ (s⁻¹)   faster ↑")
    ax.set_title("top-left is the good corner")
    if not scope.quantitative:
        ax.text(0.02, 0.02, f"method={scope.method}, not calibrated",
                transform=ax.transAxes, fontsize=7, color="#666666")

    fig.tight_layout()
    return fig, axes


__all__ = ["enzyme_km", "catalytic_efficiency", "KineticsPrediction",
           "substrate_specificity", "SubstrateScope", "plot_substrate_scope",
           "DIFFUSION_LIMIT"]
