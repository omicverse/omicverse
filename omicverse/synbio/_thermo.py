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

# Transformed standard formation energies ΔfG'° (kJ/mol) at pH 7, I = 0.25 M —
# a baseline so reaction_dg runs offline; NOT a substitute for eQuilibrator.
#
# These MUST be *transformed* (ΔfG'°) values, and mixing in an untransformed
# ΔfG° for even one species corrupts every reaction that species appears in. An
# earlier version of this table did exactly that: it carried the untransformed
# ΔfG° of liquid water (-237.2 instead of -155.66) and untransformed values for
# glucose (-915.9 vs -426.71) and the sugar phosphates, so ATP hydrolysis came
# out at **+51.5 kJ/mol — endergonic**. Six of the eleven reactions in
# ``DGF_VALIDATION`` were wrong, four of them by more than 40 kJ/mol.
#
# ``nadh`` is +60.6 relative to ``nad`` = 0. That value is not a formation
# energy in isolation; it is fixed by the requirement that the NAD-linked
# reactions come out right, and three independent ones (LDH, MDH, fumarate
# reduction) agree on it to better than 1.5 kJ/mol. The previous +22.6 put every
# NAD-linked reaction ~38 kJ/mol out, with the sign depending on which side NADH
# sat — which is why malate dehydrogenase read +71.3 against a literature +29.7.
#
# Anything added here must keep ``check_dgf_table()`` passing.
_DGF = {
    "atp": -2292.50, "adp": -1424.70, "amp": -554.83, "pi": -1059.49,
    "glc__D": -426.71, "g6p": -1318.92, "f6p": -1315.74, "fdp": -2202.73,
    "pyr": -352.40, "pep": -1185.68, "lac__D": -316.90, "accoa": -178.4,
    "co2": -386.02, "h2o": -155.66, "nad": 0.0, "nadh": 60.6, "h": 0.0,
    "succ": -530.53, "fum": -521.97, "mal__L": -682.85, "oaa": -713.42,
    "akg": -633.55, "cit": -963.46, "icit": -955.70, "glx": -468.6,
}

#: Reactions with well-established ΔG'° (pH 7), used to validate :data:`_DGF`.
#: ``(label, stoichiometry, literature ΔG'° in kJ/mol)``.
DGF_VALIDATION = (
    ("ATP hydrolysis", {"atp": -1, "h2o": -1, "adp": 1, "pi": 1}, -30.5),
    ("phosphoglucose isomerase", {"g6p": -1, "f6p": 1}, 2.5),
    ("phosphofructokinase", {"f6p": -1, "atp": -1, "fdp": 1, "adp": 1}, -14.2),
    ("hexokinase", {"glc__D": -1, "atp": -1, "g6p": 1, "adp": 1}, -16.7),
    ("pyruvate kinase", {"pep": -1, "adp": -1, "pyr": 1, "atp": 1}, -31.4),
    ("lactate dehydrogenase",
     {"pyr": -1, "nadh": -1, "h": -1, "lac__D": 1, "nad": 1}, -25.1),
    ("fumarase", {"fum": -1, "h2o": -1, "mal__L": 1}, -3.6),
    ("malate dehydrogenase",
     {"mal__L": -1, "nad": -1, "oaa": 1, "nadh": 1, "h": 1}, 29.7),
    ("aconitase", {"cit": -1, "icit": 1}, 6.3),
    ("PEP carboxylase",
     {"pep": -1, "co2": -1, "h2o": -1, "oaa": 1, "pi": 1}, -40.3),
    ("fumarate reduction",
     {"fum": -1, "nadh": -1, "h": -1, "succ": 1, "nad": 1}, -67.7),
)


def check_dgf_table(tolerance: float = 8.0) -> Dict[str, float]:
    """Residuals of the built-in ΔfG'° table against :data:`DGF_VALIDATION`.

    Returns ``{label: table_value - literature_value}`` for every reaction whose
    residual exceeds ``tolerance`` kJ/mol — so an empty dict means the table is
    consistent with all of them. Exposed rather than kept in the test suite
    because a formation-energy table is exactly the kind of data that rots
    silently: every number it produces looks like a number.
    """
    bad = {}
    for label, stoich, literature in DGF_VALIDATION:
        value = sum(coeff * _DGF[m] for m, coeff in stoich.items())
        if abs(value - literature) > tolerance:
            bad[label] = float(value - literature)
    return bad


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
    cofactor_ratios: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.mdf > 0

    def __repr__(self) -> str:  # pragma: no cover
        warn = "  [see .notes]" if self.notes else ""
        return (f"MDFResult(MDF={self.mdf:.2f} kJ/mol, feasible={self.feasible}, "
                f"bottleneck={self.bottleneck!r}){warn}")


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

    import math
    import warnings

    fixed = dict(fixed or {})
    notes: List[str] = []

    # H+ and H2O must NOT enter the reaction quotient. dG0 here is the
    # *transformed* ΔG'° at pH 7, which already carries the proton term, and
    # water's activity is 1 in dilute solution. Including them double-counts:
    # with h pinned at 1e-7 a reaction with one H+ on the product side picks up
    # -RT*ln(1e-7) = +41.6 kJ/mol of driving force that does not exist.
    ignored = {m for m in {m for r in reactions.values() for m in r}
               if _clean(m) in ("h", "h2o")}
    pinned_ignored = sorted(m for m in ignored if m in fixed)
    if pinned_ignored:
        warnings.warn(
            f"fixed 里的 {pinned_ignored} 被忽略:ΔG'° 已经是 pH 7 转换过的值,"
            f"H+ 再进反应商会重复计一次(每个 H+ 约 41.6 kJ/mol),"
            f"水在稀溶液里活度为 1。", stacklevel=2)
    for m in ignored:
        fixed.pop(m, None)

    mets = sorted({m for r in reactions.values() for m in r} - ignored)
    Model, Var, Constr, Obj = (optlang.Model, optlang.Variable,
                               optlang.Constraint, optlang.Objective)
    model = Model(name="MDF")

    # log-concentration variables (natural log)
    lx = {}
    for m in mets:
        if m in fixed:
            v = math.log(fixed[m])
            lx[m] = Var(f"x_{m}", lb=v, ub=v)
        else:
            lx[m] = Var(f"x_{m}", lb=math.log(c_min), ub=math.log(c_max))
    B = Var("B")                         # the min driving force to maximise
    model.add(list(lx.values()) + [B])

    def quotient(stoich, value_of):
        return sum(coeff * value_of(m) for m, coeff in stoich.items()
                   if m not in ignored)

    # for each reaction: driving force df_r = -(dg0_r + RT * sum S*x) >= B
    for rid, stoich in reactions.items():
        expr = dg0[rid] + _RT * quotient(stoich, lambda m: lx[m])
        # -expr >= B  ->  -expr - B >= 0
        model.add(Constr(-expr - B, lb=0, name=f"df_{rid}"))
    model.objective = Obj(B, direction="max")
    model.optimize()

    conc = {m: float(np.exp(lx[m].primal)) for m in mets}
    dfs = {}
    for rid, stoich in reactions.items():
        dfs[rid] = float(-(dg0[rid]
                           + _RT * quotient(stoich, lambda m: math.log(conc[m]))))
    mdf = float(B.primal)
    bottleneck = min(dfs, key=dfs.get) if dfs else ""

    # The LP will drive any unpinned cofactor pair to the corners of
    # [c_min, c_max] because that is free driving force. A ratio of 1e4 is not a
    # cell; report it rather than let a "feasible" verdict rest on it.
    ratios: Dict[str, float] = {}
    for num, den, low, high in (("nadh", "nad", 0.01, 1.0),
                                ("nadph", "nadp", 0.5, 100.0),
                                ("atp", "adp", 1.0, 100.0),
                                ("accoa", "coa", 0.1, 10.0)):
        keys = {_clean(m): m for m in conc}
        if num in keys and den in keys and conc[keys[den]] > 0:
            ratio = conc[keys[num]] / conc[keys[den]]
            ratios[f"{num}/{den}"] = float(ratio)
            unpinned = keys[num] not in fixed or keys[den] not in fixed
            if unpinned and not (low <= ratio <= high):
                notes.append(
                    f"{num}/{den} = {ratio:.3g},生理范围约 {low}–{high}。"
                    f"这个比值是 LP 自己选的,不是细胞里的值 —— 把它 fixed= 到实测"
                    f"比值再看 MDF 是否还为正。")
    if ratios and notes:
        notes.append(
            "MDF 为正但依赖非生理的辅因子比值时,通路实际上不可行。")

    return MDFResult(mdf=mdf, bottleneck=bottleneck, concentrations=conc,
                     driving_forces=dfs, cofactor_ratios=ratios, notes=notes)


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


# ---------------------------------------------------------------------------
# thermodynamics *inside* FBA — TMFA
# ---------------------------------------------------------------------------

@dataclass
class ThermoFBAResult:
    """An FBA solution that also respects the second law."""

    objective_value: float
    fluxes: Dict[str, float] = field(default_factory=dict)
    unconstrained_objective: float = 0.0
    blocked_forward: List[str] = field(default_factory=list)
    already_irreversible: List[str] = field(default_factory=list)
    blocked_reverse: List[str] = field(default_factory=list)
    dg_range: Dict[str, tuple] = field(default_factory=dict)
    dg_prime: Dict[str, float] = field(default_factory=dict)
    concentrations: Dict[str, float] = field(default_factory=dict)
    method: str = "directionality"
    n_constrained: int = 0
    status: str = ""

    @property
    def cost(self) -> float:
        """How much objective the thermodynamic constraints removed."""
        if self.unconstrained_objective <= 1e-12:
            return 0.0
        return 1.0 - self.objective_value / self.unconstrained_objective

    def __repr__(self) -> str:  # pragma: no cover
        return (f"ThermoFBAResult({self.method}, objective="
                f"{self.objective_value:.4f} vs {self.unconstrained_objective:.4f} "
                f"unconstrained ({self.cost:.1%} cost), "
                f"{self.n_constrained} reactions constrained)")


def _dg_bounds_from_concentrations(stoich, dg0, c_min, c_max, fixed):
    """Range of ΔG' reachable over the allowed concentration box.

    ΔG' = ΔG'° + RT·Σ Sᵢ·ln cᵢ is monotone in each ln cᵢ, so the extremes sit at
    the corners: push products to ``c_min`` and substrates to ``c_max`` for the
    minimum, and the reverse for the maximum.
    """
    import math

    lo = hi = dg0
    for mid, coeff in stoich.items():
        key = _clean(mid)
        if key in fixed:
            lo += _RT * coeff * math.log(fixed[key])
            hi += _RT * coeff * math.log(fixed[key])
            continue
        a = _RT * coeff * math.log(c_min)
        b = _RT * coeff * math.log(c_max)
        lo += min(a, b)
        hi += max(a, b)
    return lo, hi


@register_function(
    aliases=["thermo_fba", "热力学约束FBA", "TMFA", "tmfa", "TFA",
             "热力学FBA", "thermodynamic_fba", "热力学可行通量"],
    category="synthetic_biology",
    description="把热力学作为约束放进 FBA。method='directionality'(纯 LP:用 ΔG'° 与浓度范围推每个反应的 ΔG' 区间,区间整体为正就禁止正向)或 'tmfa'(完整 Henry-2007 MILP:方向二元变量 + 对数浓度变量,消除热力学不可行的内循环)。原来 reaction_dg / MDF 是独立算的,不进 FBA。Thermodynamically constrained FBA.",
    examples=[
        "res = ov.synbio.thermo_fba(m)",
        "res = ov.synbio.thermo_fba(m, method='tmfa', dg0={'PGI': 2.5})",
        "res.objective_value, res.cost, res.blocked_forward",
    ],
    related=["synbio.reaction_dg", "synbio.max_min_driving_force", "synbio.fba",
             "synbio.plot_thermo_fba"],
    requires={},
    produces={},
)
def thermo_fba(
    model,
    dg0: Optional[Mapping[str, float]] = None,
    *,
    method: str = "directionality",
    c_min: float = 1e-6,
    c_max: float = 1e-2,
    fixed: Optional[Mapping[str, float]] = None,
    dg_method: str = "baseline",
    big_m: float = 1e5,
    epsilon: float = 1e-3,
    time_limit: Optional[float] = 60.0,
) -> ThermoFBAResult:
    """FBA that cannot use a thermodynamically impossible direction.

    Parameters
    ----------
    model
        A :class:`cobra.Model`. Never mutated.
    dg0
        ``{reaction_id: ΔG'° kJ/mol}``. Reactions absent from this mapping are
        left unconstrained — silence means "no data", never "ΔG = 0". When
        omitted, ΔG'° is estimated for every reaction whose metabolites are all
        in the built-in table (or via eQuilibrator with
        ``dg_method='equilibrator'``).
    method
        ``'directionality'`` — compute each reaction's ΔG' interval over the
        concentration box and close off any direction the interval forbids.
        Pure LP, and the constraints are interpretable one reaction at a time.
        ``'tmfa'`` — the full Henry 2007 formulation: a binary per constrained
        reaction plus log-concentration variables, so directions and
        concentrations are chosen *together*. This is what removes
        thermodynamically infeasible internal loops, which the per-reaction test
        cannot see. MILP.
    c_min, c_max
        Metabolite concentration bounds (M).
    fixed
        Metabolites pinned to a concentration, e.g. ``{'h2o': 1.0}``.
    dg_method
        Backend for estimating missing ΔG'° — ``'baseline'`` or
        ``'equilibrator'``.
    big_m, epsilon
        MILP big-M and the strictness of ``ΔG' < 0`` for a carrying direction.
    time_limit
        Seconds for the MILP.

    Returns
    -------
    ThermoFBAResult
    """
    _VALID = ("directionality", "tmfa")
    if method not in _VALID:
        raise ValueError(f"method must be one of {list(_VALID)}, got {method!r}")

    import math
    import numpy as np

    fixed = {_clean(k): float(v) for k, v in (fixed or {}).items()}

    unconstrained = model.slim_optimize()
    unconstrained = 0.0 if unconstrained is None or unconstrained != unconstrained \
        else float(unconstrained)

    # --- assemble ΔG'° per reaction --------------------------------------
    stoichs: Dict[str, Dict[str, float]] = {}
    dgs: Dict[str, float] = {}
    for rxn in model.reactions:
        if rxn.boundary or "BIOMASS" in rxn.id.upper():
            continue
        stoich = {met.id: coeff for met, coeff in rxn.metabolites.items()}
        stoichs[rxn.id] = stoich
        if dg0 is not None and rxn.id in dg0:
            dgs[rxn.id] = float(dg0[rxn.id])
            continue
        if dg0 is not None:
            continue          # explicit mapping given: absence means "no data"
        keys = [_clean(m) for m in stoich]
        if all(k in _DGF for k in keys):
            try:
                dgs[rxn.id] = reaction_dg(stoich, method=dg_method)
            except Exception:
                continue

    ranges = {rid: _dg_bounds_from_concentrations(stoichs[rid], dgs[rid],
                                                  c_min, c_max, fixed)
              for rid in dgs}

    work = model.copy()

    if method == "directionality":
        blocked_f, blocked_r, conflicts, already = [], [], [], []
        for rid, (lo, hi) in ranges.items():
            rxn = work.reactions.get_by_id(rid)
            # Clamp, never assign. Assigning `lower_bound = 0.0` is a tightening
            # only when the existing lower bound is negative; on a reaction with
            # a forced minimum forward flux it *widens* the bound. ATPM in
            # e_coli_core has bounds (8.39, 1000) — the non-growth-associated
            # maintenance demand — so `lower_bound = 0.0` deleted the ATP
            # maintenance requirement and the "thermodynamically constrained"
            # model then grew at 0.9166 /h against an unconstrained 0.8739. A
            # constraint that raises the objective is not a constraint.
            if lo > 0:
                # Even at the most favourable concentrations the forward
                # direction is uphill — it cannot carry flux.
                if rxn.lower_bound > 0:
                    conflicts.append(rid)
                    continue
                if rxn.upper_bound > 0:          # only report a real change
                    rxn.upper_bound = 0.0
                    blocked_f.append(rid)
                else:
                    already.append(rid)
            elif hi < 0:
                if rxn.upper_bound < 0:
                    conflicts.append(rid)
                    continue
                if rxn.lower_bound < 0:
                    rxn.lower_bound = 0.0
                    blocked_r.append(rid)
                else:
                    already.append(rid)
        if already:
            notes_dir = (f"{len(already)} 个反应的该方向本来就已经关着"
                         f"({already[:6]}),没有新增约束 —— 不计入 n_constrained。")
        if conflicts:
            warnings.warn(
                f"{len(conflicts)} 个反应的模型边界强制要求某个方向的通量,而热力学"
                f"判定该方向不可行:{conflicts[:6]}。已跳过而非改动它们的边界 —— "
                f"这类冲突要么是 ΔG'° 估计错了,要么是模型里的强制通量(如维持"
                f"需求 ATPM)本来就不该参与方向性筛查。", stacklevel=2)
        sol = work.optimize()
        obj = float(sol.objective_value or 0.0)
        return ThermoFBAResult(
            objective_value=0.0 if obj != obj else obj,
            fluxes={k: float(v) for k, v in sol.fluxes.items()},
            unconstrained_objective=unconstrained,
            blocked_forward=blocked_f, blocked_reverse=blocked_r,
            already_irreversible=already,
            dg_range=ranges, method="directionality",
            n_constrained=len(blocked_f) + len(blocked_r), status=str(sol.status),
        )

    # --- full TMFA MILP ---------------------------------------------------
    if time_limit is not None:
        try:
            work.solver.configuration.timeout = int(max(1, round(float(time_limit))))
        except Exception:  # pragma: no cover
            pass

    prob = work.problem
    mets = sorted({_clean(m) for rid in ranges for m in stoichs[rid]})
    logc = {}
    for mkey in mets:
        if mkey in fixed:
            v = math.log(fixed[mkey])
            logc[mkey] = prob.Variable(f"_tmfa_lc_{mkey}", lb=v, ub=v)
        else:
            logc[mkey] = prob.Variable(f"_tmfa_lc_{mkey}",
                                       lb=math.log(c_min), ub=math.log(c_max))
    work.add_cons_vars(list(logc.values()))

    zs = {}
    for rid, dg0v in dgs.items():
        rxn = work.reactions.get_by_id(rid)
        dg_expr = dg0v + _RT * sum(coeff * logc[_clean(m)]
                                   for m, coeff in stoichs[rid].items())
        zf = prob.Variable(f"_tmfa_zf_{rid}", type="binary")
        zr = prob.Variable(f"_tmfa_zr_{rid}", type="binary")
        work.add_cons_vars([zf, zr])
        zs[rid] = (zf, zr)
        ub = max(abs(rxn.upper_bound), 1.0)
        lb = max(abs(rxn.lower_bound), 1.0)
        work.add_cons_vars([
            # a direction may carry flux only if its binary is on
            prob.Constraint(rxn.forward_variable - ub * zf, ub=0,
                            name=f"_tmfa_vf_{rid}"),
            prob.Constraint(rxn.reverse_variable - lb * zr, ub=0,
                            name=f"_tmfa_vr_{rid}"),
            # and never both at once
            prob.Constraint(zf + zr, ub=1, name=f"_tmfa_dir_{rid}"),
            # forward flux requires ΔG' < 0; reverse requires ΔG' > 0
            prob.Constraint(dg_expr + epsilon - big_m * (1 - zf), ub=0,
                            name=f"_tmfa_gf_{rid}"),
            prob.Constraint(dg_expr - epsilon + big_m * (1 - zr), lb=0,
                            name=f"_tmfa_gr_{rid}"),
        ])

    solver = work.solver
    try:
        solver.optimize()
    except Exception as exc:
        raise RuntimeError(
            f"TMFA 的 MILP 求解失败({exc})。可以改用 method='directionality'"
            f"(纯 LP,逐反应可解释),或安装支持 MILP 的求解器。") from exc

    status = str(solver.status or "")
    try:
        primal = {k: float(v) for k, v in solver.primal_values.items()}
    except Exception:
        primal = {}
    fluxes = {}
    for rxn in work.reactions:
        fluxes[rxn.id] = (primal.get(rxn.forward_variable.name, 0.0)
                          - primal.get(rxn.reverse_variable.name, 0.0))
    conc = {m: float(np.exp(primal.get(logc[m].name, math.log(c_min))))
            for m in mets}
    dg_prime = {}
    for rid in dgs:
        dg_prime[rid] = float(
            dgs[rid] + _RT * sum(coeff * math.log(conc[_clean(m)])
                                 for m, coeff in stoichs[rid].items()))
    try:
        obj = float(solver.objective.value)
    except Exception:
        obj = 0.0

    return ThermoFBAResult(
        objective_value=0.0 if obj != obj else obj, fluxes=fluxes,
        unconstrained_objective=unconstrained, dg_range=ranges,
        dg_prime=dg_prime, concentrations=conc, method="tmfa",
        n_constrained=len(dgs), status=status,
    )


@register_function(
    aliases=["plot_thermo_fba", "热力学FBA图", "plot_tmfa", "TMFA图",
             "热力学约束图"],
    category="synthetic_biology",
    description="画热力学约束 FBA 的结果:目标值损失、被禁止方向的反应数,以及各反应 ΔG' 区间(跨过 0 的才是方向自由的)。Plot the cost of thermodynamic constraints and per-reaction ΔG' intervals.",
    examples=["ov.synbio.plot_thermo_fba(res)"],
    related=["synbio.thermo_fba"],
    requires={},
    produces={},
)
def plot_thermo_fba(result: "ThermoFBAResult", top_n: int = 20, axes=None):
    """Two panels: objective with vs without thermodynamics, and ΔG' intervals."""
    from ._plot import _mpl
    plt = _mpl()

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                 gridspec_kw={"width_ratios": [1, 2]})
    else:
        fig = axes[0].figure
    axes = list(axes)

    ax = axes[0]
    vals = [result.unconstrained_objective, result.objective_value]
    ax.bar(["FBA", f"+thermo\n({result.method})"], vals,
           color=["#B0B0B0", "#377EB8"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("objective")
    ax.set_title(f"{result.cost:.1%} of the objective was infeasible")

    ax = axes[1]
    blocked = set(result.blocked_forward) | set(result.blocked_reverse)
    items = sorted(result.dg_range.items(), key=lambda kv: kv[1][0])[:top_n]
    for i, (rid, (lo, hi)) in enumerate(items):
        ax.plot([lo, hi], [i, i], lw=5, solid_capstyle="butt",
                color="#E41A1C" if rid in blocked else "#377EB8", alpha=0.9)
    ax.axvline(0, c="k", lw=1, ls="--")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([rid for rid, _ in items], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("ΔG′ interval over the concentration box (kJ/mol)")
    ax.set_title("red = a direction was closed off")
    if not items:
        ax.text(0.5, 0.5, "no ΔG data for any reaction",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)

    fig.tight_layout()
    return fig, axes


__all__ = ["reaction_dg", "max_min_driving_force", "MDFResult",
           "plot_driving_forces", "thermo_fba", "ThermoFBAResult",
           "plot_thermo_fba"]
