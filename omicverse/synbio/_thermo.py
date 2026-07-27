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
        blocked_f, blocked_r = [], []
        for rid, (lo, hi) in ranges.items():
            rxn = work.reactions.get_by_id(rid)
            if lo > 0:
                # Even at the most favourable concentrations the forward
                # direction is uphill — it cannot carry flux.
                rxn.upper_bound = 0.0
                blocked_f.append(rid)
            elif hi < 0:
                rxn.lower_bound = 0.0
                blocked_r.append(rid)
        sol = work.optimize()
        obj = float(sol.objective_value or 0.0)
        return ThermoFBAResult(
            objective_value=0.0 if obj != obj else obj,
            fluxes={k: float(v) for k, v in sol.fluxes.items()},
            unconstrained_objective=unconstrained,
            blocked_forward=blocked_f, blocked_reverse=blocked_r,
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
