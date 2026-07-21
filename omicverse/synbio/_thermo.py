r"""Pathway thermodynamics: reaction ΔG and the Max-min Driving Force (MDF).

Thermodynamics decides whether a designed pathway can actually run and which
step is the bottleneck:

* :func:`reaction_dg` — transformed Gibbs energy ΔG'° of a reaction.  Uses the
  `eQuilibrator <https://equilibrator.weizmann.ac.il>`_ API when installed;
  otherwise a small built-in formation-energy table (central metabolism) as a
  transparent baseline.
* :func:`max_min_driving_force` — the MDF (Noor *et al.* 2014): the largest
  driving force achievable at the worst step, optimised over metabolite
  concentrations. A pathway with MDF ≤ 0 cannot proceed; the limiting reaction
  is the thermodynamic bottleneck. Solved exactly as a linear program (optlang).

MDF is fully implemented; ΔG values are the transparent part — supply your own
(from eQuilibrator) for quantitative work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence

from .._registry import register_function

_RT = 2.579  # kJ/mol at 298.15 K

# tiny ΔGf'° table (kJ/mol, pH7, I=0.1M) for a few central-metabolism species —
# a baseline so reaction_dg runs offline; NOT a substitute for eQuilibrator.
_DGF = {
    "atp": -2298.2, "adp": -1424.7, "amp": -550.9, "pi": -1059.2,
    "glc__D": -915.9, "g6p": -1763.9, "f6p": -1760.8, "fdp": -2601.4,
    "pyr": -472.3, "pep": -1263.7, "lac__D": -516.7, "accoa": -178.4,
    "co2": -386.0, "h2o": -237.2, "nad": 0.0, "nadh": 22.6, "h": 0.0,
    "succ": -690.4, "fum": -604.2, "mal__L": -843.1, "oaa": -794.4,
    "akg": -793.3, "cit": -1165.6, "icit": -1160.0, "glx": -468.6,
}


def _clean(mid: str) -> str:
    return mid.rsplit("_", 1)[0] if mid[-2:] in ("_c", "_e", "_p", "_m") else mid


@register_function(
    aliases=["reaction_dg", "反应吉布斯能", "反应deltaG", "reaction_gibbs",
             "deltaG", "反应能量", "gibbs"],
    category="synthetic_biology",
    description="反应吉布斯自由能 ΔG'°:由计量式估算(eQuilibrator 若装则用之,否则内置生成能表基线)。Reaction transformed Gibbs energy (eQuilibrator / built-in baseline).",
    examples=[
        "ov.synbio.reaction_dg({'g6p':-1,'f6p':1})",
        "ov.synbio.reaction_dg('g6p_c --> f6p_c')",
    ],
    related=["synbio.max_min_driving_force"],
    requires={},
    produces={},
)
def reaction_dg(reaction, method: str = "baseline") -> float:
    """Standard transformed Gibbs energy ΔG'° (kJ/mol) of a reaction.

    *reaction* is either a ``{metabolite_id: stoichiometry}`` dict (negative =
    substrate) or a string like ``'g6p_c --> f6p_c'``.

    ``method`` selects the backend: ``"baseline"`` (built-in ΔGf table) or
    ``"equilibrator"`` (component-contribution via equilibrator-api)."""
    _VALID = {"baseline", "equilibrator"}
    if method not in _VALID:
        raise ValueError(f"method must be one of {sorted(_VALID)}, got {method!r}")
    stoich = _parse_reaction(reaction)
    if method == "equilibrator":
        try:
            return _equilibrator_dg(stoich)
        except ImportError:
            import logging
            logging.getLogger("omicverse.synbio").warning(
                "equilibrator-api 未安装,回退内置 ΔGf 基线表。")
    dg = 0.0
    for mid, coeff in stoich.items():
        key = _clean(mid)
        if key not in _DGF:
            raise KeyError(
                f"代谢物 '{mid}' 不在内置 ΔGf 表中。请装 equilibrator-api 并传 "
                "method='equilibrator',或自行提供 ΔG'° 给 max_min_driving_force。")
        dg += coeff * _DGF[key]
    return float(dg)


def _parse_reaction(reaction) -> Dict[str, float]:
    if isinstance(reaction, dict):
        return dict(reaction)
    s = str(reaction)
    for arrow in ("<=>", "-->", "->", "=>", "="):
        if arrow in s:
            lhs, rhs = s.split(arrow, 1)
            break
    else:
        raise ValueError("反应字符串需含箭头,如 'a + b --> c'。")

    def side(part, sign):
        out = {}
        for tok in part.split("+"):
            tok = tok.strip()
            if not tok:
                continue
            bits = tok.split()
            coeff = float(bits[0]) if len(bits) > 1 else 1.0
            mid = bits[-1]
            out[mid] = out.get(mid, 0.0) + sign * coeff
        return out

    stoich = side(lhs, -1)
    for k, v in side(rhs, +1).items():
        stoich[k] = stoich.get(k, 0.0) + v
    return stoich


_CC = {}


def _get_cc():
    """Cache the (expensive) ComponentContribution instance."""
    if "cc" not in _CC:
        from equilibrator_api import ComponentContribution
        _CC["cc"] = ComponentContribution()
    return _CC["cc"]


def _equilibrator_dg(stoich) -> float:
    """Real ΔG'° via eQuilibrator component-contribution (Noor 2013; Beber 2022).

    Metabolite ids are resolved as BiGG identifiers (``bigg.metabolite:<id>``),
    matching COBRApy/GEM conventions."""
    cc = _get_cc()

    def _fmt(side):
        return " + ".join(f"{abs(c):g} bigg.metabolite:{_clean(m)}"
                          for m, c in side)
    subs = [(m, c) for m, c in stoich.items() if c < 0]
    prods = [(m, c) for m, c in stoich.items() if c > 0]
    formula = f"{_fmt(subs)} = {_fmt(prods)}"
    rxn = cc.parse_reaction_formula(formula)
    return float(cc.standard_dg_prime(rxn).value.to("kJ/mol").magnitude)


@dataclass
class MDFResult:
    mdf: float                        # kJ/mol; > 0 means feasible
    bottleneck: str                   # limiting reaction id
    concentrations: Dict[str, float]  # optimal metabolite concentrations (M)
    driving_forces: Dict[str, float]  # per-reaction -ΔG' (kJ/mol)

    @property
    def feasible(self) -> bool:
        return self.mdf > 0

    def __repr__(self) -> str:  # pragma: no cover
        return (f"MDFResult(MDF={self.mdf:.2f} kJ/mol, feasible={self.feasible}, "
                f"bottleneck={self.bottleneck!r})")


@register_function(
    aliases=["max_min_driving_force", "最大最小驱动力", "MDF", "mdf",
             "热力学瓶颈", "driving_force", "通路热力学"],
    category="synthetic_biology",
    description="最大最小驱动力 (MDF):优化代谢物浓度使通路中最弱一步的驱动力最大化,判断异源通路是否热力学可行并定位瓶颈反应(LP,Noor 2014)。Max-min Driving Force of a pathway.",
    examples=[
        "res = ov.synbio.max_min_driving_force(reactions, dg0)",
        "res.mdf, res.bottleneck, res.feasible",
    ],
    related=["synbio.reaction_dg", "synbio.pathway_search"],
    requires={},
    produces={},
)
def max_min_driving_force(
    reactions: Mapping[str, Mapping[str, float]],
    dg0: Mapping[str, float],
    c_min: float = 1e-6,
    c_max: float = 1e-2,
    fixed: Optional[Mapping[str, float]] = None,
) -> MDFResult:
    """Compute the Max-min Driving Force of a pathway.

    Parameters
    ----------
    reactions
        ``{reaction_id: {metabolite: stoichiometry}}`` (negative = substrate).
    dg0
        ``{reaction_id: ΔG'° (kJ/mol)}`` — from :func:`reaction_dg` / eQuilibrator.
    c_min, c_max
        Concentration bounds (M) applied to every non-fixed metabolite.
    fixed
        Metabolites pinned to a concentration (e.g. ``{'h2o':1, 'co2':1e-5}``).

    Returns
    -------
    MDFResult
    """
    import numpy as np
    try:
        import optlang
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.synbio.max_min_driving_force 需要 optlang(随 cobra 安装)。") from exc

    fixed = dict(fixed or {})
    mets = sorted({m for r in reactions.values() for m in r})
    Model, Var, Constr, Obj = (optlang.Model, optlang.Variable,
                               optlang.Constraint, optlang.Objective)
    model = Model(name="MDF")

    # log-concentration variables (natural log)
    import math
    lx = {}
    for m in mets:
        if m in fixed:
            v = math.log(fixed[m])
            lx[m] = Var(f"x_{m}", lb=v, ub=v)
        else:
            lx[m] = Var(f"x_{m}", lb=math.log(c_min), ub=math.log(c_max))
    B = Var("B")                         # the min driving force to maximise
    model.add(list(lx.values()) + [B])

    # for each reaction: driving force df_r = -(dg0_r + RT * sum S*x) >= B
    for rid, stoich in reactions.items():
        expr = dg0[rid] + _RT * sum(coeff * lx[m] for m, coeff in stoich.items())
        # -expr >= B  ->  -expr - B >= 0
        model.add(Constr(-expr - B, lb=0, name=f"df_{rid}"))
    model.objective = Obj(B, direction="max")
    model.optimize()

    conc = {m: float(np.exp(lx[m].primal)) for m in mets}
    dfs = {}
    for rid, stoich in reactions.items():
        val = -(dg0[rid] + _RT * sum(coeff * math.log(conc[m])
                                     for m, coeff in stoich.items()))
        dfs[rid] = float(val)
    mdf = float(B.primal)
    bottleneck = min(dfs, key=dfs.get) if dfs else ""
    return MDFResult(mdf=mdf, bottleneck=bottleneck, concentrations=conc,
                     driving_forces=dfs)


@register_function(
    aliases=["plot_driving_forces", "驱动力图", "MDF图", "plot_mdf",
             "驱动力柱状图", "热力学图"],
    category="synthetic_biology",
    description="画通路各反应的驱动力(-ΔG')柱状图,标出 MDF 水平线与瓶颈反应(红色)。Plot per-reaction driving forces with the MDF level and bottleneck.",
    examples=["ov.synbio.plot_driving_forces(mdf_result)"],
    related=["synbio.max_min_driving_force"],
    requires={},
    produces={},
)
def plot_driving_forces(mdf_result, ax=None):
    """Bar chart of per-reaction driving forces (from :func:`max_min_driving_force`).

    The MDF is drawn as a horizontal line; the bottleneck reaction (the step at
    the MDF) is highlighted in red."""
    from ._plot import _mpl
    plt = _mpl()

    rxns = list(mdf_result.driving_forces.keys())
    vals = [mdf_result.driving_forces[r] for r in rxns]
    colors = ["#E41A1C" if r == mdf_result.bottleneck else "#377EB8" for r in rxns]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(rxns)), 3.4))
    else:
        fig = ax.figure
    ax.bar(range(len(rxns)), vals, color=colors)
    ax.axhline(mdf_result.mdf, ls="--", c="k", lw=1,
               label=f"MDF = {mdf_result.mdf:.1f} kJ/mol")
    ax.axhline(0, c="grey", lw=0.8)
    ax.set_xticks(range(len(rxns)))
    ax.set_xticklabels(rxns, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("driving force  −ΔG'  (kJ/mol)")
    ax.set_title("Pathway thermodynamics (red = bottleneck)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


__all__ = ["reaction_dg", "max_min_driving_force", "MDFResult",
           "plot_driving_forces"]
