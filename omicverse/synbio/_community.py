r"""Resource allocation, and more than one organism in the flask.

* :func:`rba_model` / :func:`rba` — Resource Balance Analysis. ``ec_model``
  already charges reactions against a shared enzyme pool; RBA goes one step
  further and makes the *ribosome* a constrained resource too, so growth has to
  pay for the protein synthesis that produces the enzymes. That is what
  reproduces overflow metabolism — a cell excretes acetate at high glucose not
  because it cannot oxidise it, but because respiration costs more protein per
  ATP than fermentation does, and above a certain rate the proteome budget is
  the binding constraint rather than the carbon.
* :func:`community_model` — merge several GEMs into one compartmentalised model
  with a shared medium, so cross-feeding is a flux rather than an assumption.
* :func:`steadycom` — SteadyCom (Chan 2017): find the abundance profile at which
  every member grows at the *same* rate, which is the only self-consistent
  steady state for a co-culture. Fixed-abundance community FBA silently assumes
  an answer to exactly the question that matters.
* :func:`micom_grow` — dispatch to MICOM when it is installed.
* :func:`plot_community` / :func:`plot_rba` — abundance and exchange structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import cobra
    import pandas as pd


_MISSING = (
    "ov.synbio.{fn} 需要额外依赖 COBRApy。请 pip install 'omicverse[synbio]'。"
)


def _cobra(fn: str):
    try:
        import cobra
        return cobra
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_MISSING.format(fn=fn)) from exc


def _biomass(model):
    for r in model.reactions:
        if r.objective_coefficient != 0:
            return r
    for r in model.reactions:
        if "BIOMASS" in r.id.upper():
            return r
    return None


# ---------------------------------------------------------------------------
# Resource Balance Analysis
# ---------------------------------------------------------------------------

@dataclass
class RBAResult:
    """Growth under a proteome budget that includes the ribosome."""

    growth: float
    unconstrained_growth: float
    protein_fraction_used: float = 0.0
    ribosome_fraction: float = 0.0
    enzyme_fraction: float = 0.0
    fluxes: Dict[str, float] = field(default_factory=dict)
    total_protein: float = 0.0
    status: str = ""
    #: Mass actually charged, g protein / gDW. Compare with ``total_protein``:
    #: if it is far below, the budget is not doing anything.
    protein_used: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def binding(self) -> bool:
        """Whether the proteome budget actually limited growth."""
        return self.growth < self.unconstrained_growth - 1e-6

    @property
    def cost(self) -> float:
        if self.unconstrained_growth <= 1e-12:
            return 0.0
        return 1.0 - self.growth / self.unconstrained_growth

    def __repr__(self) -> str:  # pragma: no cover
        return (f"RBAResult(growth={self.growth:.4f} vs "
                f"{self.unconstrained_growth:.4f} unconstrained, ribosome "
                f"{self.ribosome_fraction:.1%} of proteome)")


@register_function(
    aliases=["rba", "资源分配", "RBA", "resource_balance",
             "核糖体约束", "蛋白质组预算", "overflow"],
    category="synthetic_biology",
    description="资源平衡分析 (RBA):在 ec_model 的酶池之上,再把**核糖体**也变成受限资源 —— 生长必须为合成这些酶所需的蛋白质付费。这是溢流代谢的机理:高糖下细胞排乙酸不是因为氧化不了,而是呼吸每产一个 ATP 要的蛋白比发酵多,超过某个速率后限制因素是蛋白质组预算而不是碳源。Resource Balance Analysis — growth under a proteome budget including the ribosome.",
    examples=[
        "res = ov.synbio.rba(m, total_protein=0.5)",
        "res.growth, res.ribosome_fraction, res.cost",
    ],
    related=["synbio.ec_model", "synbio.fba", "synbio.plot_rba"],
    requires={},
    produces={},
)
def rba(
    model: "cobra.Model",
    *,
    total_protein: float = 0.55,
    ribosome_efficiency: float = 8.4,
    enzyme_efficiency: float = 65.0,
    protein_per_biomass: float = 0.55,
    kcat_default: float = 65.0,
    kcats: Optional[Mapping[str, float]] = None,
    enzyme_mw: float = 40.0,
    enzyme_mws: Optional[Mapping[str, float]] = None,
) -> RBAResult:
    """Maximise growth subject to a proteome that must also build itself.

    Parameters
    ----------
    model
        A :class:`cobra.Model`. Never mutated.
    total_protein
        Proteome mass fraction available, g protein / gDW.
    ribosome_efficiency
        Peptide bonds per ribosome per second, translated into growth per unit
        ribosome mass. Lowering it makes translation more expensive and brings
        the overflow point down.
    enzyme_efficiency, kcat_default, kcats
        Turnover used to charge metabolic reactions against the pool; per
        reaction values override the default.
    protein_per_biomass
        Protein content of biomass, used to size the translation demand.

    Returns
    -------
    RBAResult

    Notes
    -----
    The ribosome constraint is what distinguishes this from
    :func:`ec_model`: growth ``mu`` requires ``mu * protein_per_biomass /
    ribosome_efficiency`` of ribosome mass, and that mass comes out of the same
    budget the enzymes are drawing on. The two demands compete, which is the
    whole point.
    """
    cobra = _cobra("rba")
    if total_protein <= 0:
        raise ValueError("total_protein 必须为正。")

    biomass = _biomass(model)
    if biomass is None:
        raise ValueError("模型没有 biomass 反应,RBA 需要它来定义生长。")

    unconstrained = model.slim_optimize()
    unconstrained = 0.0 if unconstrained is None or unconstrained != unconstrained \
        else float(unconstrained)

    work = model.copy()
    # Re-resolve the biomass reaction *on the copy*. `flux_expression` is built
    # from the owning model's solver variables, so using the original's
    # expression as the copy's objective tries to import those variables and
    # collides on their names.
    biomass = _biomass(work)
    if biomass is None:  # pragma: no cover - copy always keeps the objective
        raise ValueError("模型副本上找不到 biomass 反应。")
    prob = work.problem
    kcats = dict(kcats or {})

    enzyme_terms = []
    aux_vars = []
    for rxn in work.reactions:
        if rxn.boundary or rxn is None:
            continue
        if rxn.id == biomass.id:
            continue
        if not rxn.gene_reaction_rule:
            continue                      # spontaneous: no protein cost
        k = float(kcats.get(rxn.id, kcat_default))
        if k <= 0:
            continue
        a = prob.Variable(f"_rba_e_{rxn.id}", lb=0)
        work.add_cons_vars([a])
        work.add_cons_vars([
            prob.Constraint(a - rxn.flux_expression, lb=0, name=f"_rba_p_{rxn.id}"),
            prob.Constraint(a + rxn.flux_expression, lb=0, name=f"_rba_n_{rxn.id}"),
        ])
        aux_vars.append(a)
        # MW x v / (kcat x 3600) is a MASS in g/gDW. Without the molecular
        # weight the term is v/kcat in **mmol/gDW**, which was then added
        # directly to a ribosome term in g/gDW — undercharging every enzyme by
        # roughly its molecular weight (~40x). The proteome then read 97%
        # ribosome against a real ~20% at mu ~ 0.9 /h, the enzyme/ribosome split
        # was flat across a 40-fold budget sweep, and the budget did not bind at
        # all at the physiological 0.55 g/gDW.
        mw = float((enzyme_mws or {}).get(rxn.id, enzyme_mw))
        enzyme_terms.append(mw * a / (k * 3600.0))

    # ribosome mass required to sustain growth mu
    ribosome_term = biomass.flux_expression * (
        protein_per_biomass / max(ribosome_efficiency, 1e-9))

    budget = prob.Constraint(
        sum(enzyme_terms) + ribosome_term if enzyme_terms else ribosome_term,
        ub=total_protein, name="_rba_proteome")
    work.add_cons_vars([budget])

    work.objective = prob.Objective(biomass.flux_expression, direction="max")
    sol = work.optimize()
    fluxes = {k: float(v) for k, v in sol.fluxes.items()}
    growth = float(fluxes.get(biomass.id, 0.0))

    enzyme_used = sum(float(a.primal or 0.0) for a in aux_vars) if aux_vars else 0.0
    enzyme_mass = 0.0
    for a, rxn_id in zip(aux_vars, [v.name[len("_rba_e_"):] for v in aux_vars]):
        k = float(kcats.get(rxn_id, kcat_default))
        mw = float((enzyme_mws or {}).get(rxn_id, enzyme_mw))
        enzyme_mass += mw * float(a.primal or 0.0) / (k * 3600.0)
    ribosome_mass = growth * protein_per_biomass / max(ribosome_efficiency, 1e-9)
    used = enzyme_mass + ribosome_mass

    notes: List[str] = []
    if growth >= unconstrained - 1e-6:
        notes.append(
            f"蛋白预算没有起作用:只用掉 {used:.4g} g/gDW,预算是 {total_protein:.4g}。"
            f"结果与普通 FBA 相同。核心模型(如 e_coli_core 的 95 个反应)只覆盖蛋白组"
            f"的一小部分,所以在生理预算 0.55 下几乎不会 binding —— 要么换基因组尺度"
            f"模型,要么把 total_protein 调到实际用量附近再讨论。")
    if ribosome_mass > 0 and used > 0 and ribosome_mass / used > 0.8:
        notes.append(
            f"核糖体占了蛋白预算的 {100 * ribosome_mass / used:.0f}%,而真实大肠杆菌"
            f"在 µ≈0.9/h 时核糖体蛋白约占 20%。这个比例过高通常说明酶的分子量没有计入"
            f"(酶质量必须是 MW x v / (kcat x 3600),单位 g/gDW)。")

    return RBAResult(
        growth=growth, unconstrained_growth=unconstrained,
        protein_fraction_used=used / total_protein if total_protein else 0.0,
        ribosome_fraction=ribosome_mass / used if used > 0 else 0.0,
        enzyme_fraction=enzyme_mass / used if used > 0 else 0.0,
        fluxes=fluxes, total_protein=total_protein, status=str(sol.status),
        protein_used=float(used), notes=notes,
    )


@register_function(
    aliases=["me_model", "ME_model", "ME模型", "COBRAme", "cobrame",
             "转录翻译模型", "代谢表达模型"],
    category="synthetic_biology",
    description="ME 模型(代谢-表达模型,COBRAme):把转录、翻译、复合物组装本身作为**反应**显式建进模型,因此可以问『多合成这个酶要占多少核糖体时间』这类 RBA 答不了的问题。RBA 是同一想法的粗粒度版本 —— 一个总的蛋白质组预算约束,不展开合成机器;两者不能互相冒充。需要 COBRAme + 一份物种特异的 ME 重建。ME-model dispatch (COBRAme).",
    examples=[
        "me = ov.synbio.me_model('iJL1678b')",
        "res = ov.synbio.rba(m)   # the coarse-grained alternative, no extra deps",
    ],
    related=["synbio.rba", "synbio.ec_model", "synbio.fba"],
    requires={},
    produces={},
)
def me_model(model_id: str = "iJL1678b", *, solver_tolerance: float = 1e-6):
    """Load and solve a metabolism-and-expression model through COBRAme.

    Parameters
    ----------
    model_id
        A ME reconstruction, e.g. ``'iJL1678b'`` (*E. coli*).
    solver_tolerance
        ME models are numerically hard and need a high-precision solver
        (qMINOS/SoPlex); this is passed through.

    Notes
    -----
    This is deliberately a *separate* function from :func:`rba` rather than a
    method of it. A ME model represents transcription, translation and complex
    formation as explicit reactions with their own stoichiometry; RBA imposes a
    single aggregate proteome-budget constraint and never writes those reactions
    down. They answer different questions, and labelling one as the other —
    which an earlier alias on ``rba`` did — makes it look as though ov.synbio
    ships a ME model when it ships a proteome constraint.
    """
    try:
        import cobrame  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ME 模型需要 COBRAme(github.com/SBRG/cobrame,无 PyPI 包)以及一份"
            "物种特异的 ME 重建(如 ecolime 的 iJL1678b),还需要高精度求解器"
            "(qMINOS 或 SoPlex)——ME 模型的数值条件很差,普通 LP 求解器解不动。\n"
            "只想要蛋白质组资源约束的话,ov.synbio.rba 是粗粒度版本,"
            "只依赖 COBRApy,不需要任何额外安装。") from exc

    from cobrame.io.json import load_json_me_model  # type: ignore

    import os
    root = os.environ.get("OMICOS_ME_MODELS", "")
    path = os.path.join(root, f"{model_id}.json") if root else f"{model_id}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 ME 模型 {path!r}。把 ME 重建放到 OMICOS_ME_MODELS 指向的目录,"
            f"或传一个完整路径。")
    return load_json_me_model(path)


@register_function(
    aliases=["plot_rba", "RBA图", "蛋白质组预算图", "资源分配图", "overflow图"],
    category="synthetic_biology",
    description="画 RBA 结果:蛋白质组预算在核糖体与代谢酶之间的分配,以及预算收紧时生长如何下降(溢流的形状)。Plot the proteome budget split and the growth response to tightening it.",
    examples=["ov.synbio.plot_rba(m)"],
    related=["synbio.rba", "synbio.ec_model"],
    requires={},
    produces={},
)
def plot_rba(model, budgets: Optional[Sequence[float]] = None, axes=None):
    """Growth against proteome budget, plus how the budget is spent."""
    from ._plot import _mpl
    plt = _mpl()

    budgets = list(budgets or [0.1, 0.2, 0.3, 0.4, 0.55, 0.7, 0.9])
    results = [rba(model, total_protein=b) for b in budgets]

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    else:
        fig = axes[0].figure
    axes = list(axes)

    ax = axes[0]
    ax.plot(budgets, [r.growth for r in results], "o-", color="#377EB8")
    ax.axhline(results[0].unconstrained_growth, ls="--", c="grey", lw=1,
               label="unconstrained FBA")
    ax.set_xlabel("proteome budget (g protein / gDW)")
    ax.set_ylabel("growth (1/h)")
    ax.set_title("growth is proteome-limited below the plateau")
    ax.legend(fontsize=8)

    ax = axes[1]
    ribo = [r.ribosome_fraction for r in results]
    enz = [r.enzyme_fraction for r in results]
    ax.stackplot(budgets, ribo, enz, labels=["ribosome", "metabolic enzymes"],
                 colors=["#E41A1C", "#4DAF4A"], alpha=0.85)
    ax.set_xlabel("proteome budget")
    ax.set_ylabel("fraction of used protein")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("where the budget goes")

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# communities
# ---------------------------------------------------------------------------

@dataclass
class CommunityResult:
    """A co-culture solved for a self-consistent steady state."""

    growth: float
    abundances: Dict[str, float] = field(default_factory=dict)
    member_growth: Dict[str, float] = field(default_factory=dict)
    exchanges: Dict[str, float] = field(default_factory=dict)
    cross_feeding: List[Tuple[str, str, str, float]] = field(default_factory=list)
    method: str = "steadycom"
    status: str = ""
    abundance_is_unique: bool = True
    notes: List[str] = field(default_factory=list)

    @property
    def members(self) -> List[str]:
        return sorted(self.abundances)

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame({
            "abundance": pd.Series(self.abundances),
            "growth": pd.Series(self.member_growth),
        })

    def __repr__(self) -> str:  # pragma: no cover
        return (f"CommunityResult({self.method}, {len(self.abundances)} members, "
                f"community growth={self.growth:.4f}, "
                f"{len(self.cross_feeding)} cross-feeding links)")


@register_function(
    aliases=["community_model", "群落模型", "共培养", "co_culture",
             "community", "菌群模型"],
    category="synthetic_biology",
    description="把多个 GEM 合并成一个带共享培养基的隔室化群落模型:每个成员的反应加物种前缀,交换反应改为与共享外部池交换,于是互养(cross-feeding)是被算出来的通量,而不是事先假设。Merge several GEMs into one compartmentalised community model with a shared medium.",
    examples=[
        "com = ov.synbio.community_model({'ecoli': m1, 'bsub': m2})",
        "res = ov.synbio.steadycom(com)",
    ],
    related=["synbio.steadycom", "synbio.micom_grow", "synbio.load_gem"],
    requires={},
    produces={},
)
def community_model(
    models: "Mapping[str, cobra.Model]",
    *,
    shared_compartment: str = "u",
    model_id: str = "community",
) -> "cobra.Model":
    """Build a shared-medium community model from ``{name: model}``."""
    cobra = _cobra("community_model")
    if len(models) < 2:
        raise ValueError("群落至少需要 2 个成员模型。")

    com = cobra.Model(model_id)
    shared: Dict[str, "cobra.Metabolite"] = {}
    member_biomass: List[str] = []
    supply: Dict[str, float] = {}

    for name, model in models.items():
        if "_" in name:
            raise ValueError(
                f"成员名 {name!r} 含下划线,会和反应 id 前缀冲突;请用不含下划线的名字。")

        # Build fresh objects rather than renaming copies. Assigning to
        # `Reaction.id` does not rename the reaction's solver variables, so
        # copied reactions stayed bound to their original names and the second
        # member collided with the first on `Biomass_Ecoli_core` — the failure
        # surfaced far away, inside optlang's objective cloning. Constructing
        # new Metabolite/Reaction objects carries no solver state at all.
        met_map: Dict[str, "cobra.Metabolite"] = {}
        for met in model.metabolites:
            met_map[met.id] = cobra.Metabolite(
                f"{name}__{met.id}", formula=met.formula, name=met.name,
                charge=met.charge, compartment=met.compartment)

        bm = _biomass(model)
        if bm is not None:
            member_biomass.append(f"{name}__{bm.id}")

        new_reactions = []
        exchange_ids = []
        for rxn in model.reactions:
            nr = cobra.Reaction(f"{name}__{rxn.id}", name=rxn.name,
                                lower_bound=rxn.lower_bound,
                                upper_bound=rxn.upper_bound)
            nr.add_metabolites({met_map[m.id]: c
                                for m, c in rxn.metabolites.items()})
            new_reactions.append(nr)
            if rxn.boundary and len(rxn.metabolites) == 1:
                exchange_ids.append(nr.id)
        com.add_reactions(new_reactions)

        # each member's exchange becomes a transport to the shared medium
        for rid in exchange_ids:
            rxn = com.reactions.get_by_id(rid)
            met = list(rxn.metabolites)[0]
            base = met.id.split("__", 1)[1]
            if base not in shared:
                shared[base] = cobra.Metabolite(
                    f"{base}_{shared_compartment}",
                    compartment=shared_compartment)
            rxn.add_metabolites({shared[base]: -rxn.metabolites[met]})
            original = model.reactions.get_by_id(rid.split("__", 1)[1])
            supply[base] = min(supply.get(base, 0.0), float(original.lower_bound))

    # One medium exchange per shared metabolite, and its uptake bound is the
    # tightest a single member had — not -1000. With an unlimited medium the
    # members never compete: two identical E. coli scored a community growth of
    # exactly 2x the monoculture rate, and SteadyCom's abundance answer was
    # degenerate because there was nothing to divide.
    exchanges = []
    for base, met in shared.items():
        ex = cobra.Reaction(f"EX_{base}_{shared_compartment}")
        ex.lower_bound = supply.get(base, -1000.0)
        ex.upper_bound = 1000.0          # secretion into the medium is free
        ex.add_metabolites({met: -1.0})
        exchanges.append(ex)
    if exchanges:
        com.add_reactions(exchanges)

    # Community objective: total biomass across members. steadycom replaces this
    # with its own abundance objective, but a community model that cannot be
    # optimised at all is not a usable object.
    present = [rid for rid in member_biomass
               if rid in {r.id for r in com.reactions}]
    if present:
        com.objective = {com.reactions.get_by_id(rid): 1.0 for rid in present}

    # Record the membership explicitly. Recovering it from reaction-id prefixes
    # is ambiguous: the shared medium exchanges are named EX_glc__D_e_u, whose
    # own double underscore made "EX_glc" look like a member.
    com.notes = dict(getattr(com, "notes", None) or {})
    com.notes["community_members"] = list(models)
    com.notes["community_biomass"] = {n: b for n, b in
                                      zip(models, member_biomass)}
    return com


@register_function(
    aliases=["steadycom", "SteadyCom", "稳态群落", "群落生长",
             "community_growth", "丰度分布"],
    category="synthetic_biology",
    description="SteadyCom (Chan 2017):求解共培养的自洽稳态 —— 找到使**所有成员生长速率相同**的丰度分布。固定丰度的群落 FBA 等于预先假设了最该被求解的那个量。二分搜索群落生长率,每一步解一个 LP。SteadyCom: find the abundance profile at which every member grows at the same rate.",
    examples=[
        "res = ov.synbio.steadycom(com)",
        "res.growth, res.abundances, res.cross_feeding",
    ],
    related=["synbio.community_model", "synbio.micom_grow",
             "synbio.plot_community"],
    requires={},
    produces={},
)
def steadycom(
    community: "cobra.Model",
    *,
    members: Optional[Sequence[str]] = None,
    mu_max: float = 2.0,
    tolerance: float = 1e-4,
    max_iter: int = 40,
) -> CommunityResult:
    """Solve for the community growth rate every member can sustain together.

    The search is a bisection on ``mu``: at a candidate rate, each member's
    biomass reaction is tied to its abundance variable, abundances are required
    to sum to one, and the LP asks whether *any* feasible abundance profile
    exists. The largest feasible ``mu`` is the community growth rate, and the
    abundances that achieve it are the composition.
    """
    cobra = _cobra("steadycom")

    if members is None:
        recorded = (getattr(community, "notes", None) or {}).get("community_members")
        if recorded:
            members = list(recorded)
        else:
            # Fall back to prefixes, excluding the shared medium exchanges —
            # EX_glc__D_e_u would otherwise register "EX_glc" as a member.
            members = sorted({r.id.split("__", 1)[0] for r in community.reactions
                              if "__" in r.id and not r.id.startswith("EX_")})
    members = list(members)
    if len(members) < 2:
        raise ValueError(
            "没有识别出 2 个以上成员。community_model 会给反应加 '<name>__' 前缀,"
            "steadycom 依赖这个前缀来分组;请传 members= 明确指定。")

    biomass_ids = {}
    for name in members:
        cands = [r for r in community.reactions
                 if r.id.startswith(f"{name}__") and "BIOMASS" in r.id.upper()]
        if not cands:
            raise ValueError(f"成员 {name!r} 没有找到 biomass 反应。")
        biomass_ids[name] = max(cands, key=lambda r: len(r.metabolites)).id

    def feasible(mu: float):
        with community as m:
            prob = m.problem
            ab = {}
            for name in members:
                v = prob.Variable(f"_ab_{name}", lb=0, ub=1)
                m.add_cons_vars([v])
                ab[name] = v
                bm = m.reactions.get_by_id(biomass_ids[name])
                # biomass flux == mu * abundance
                m.add_cons_vars([prob.Constraint(
                    bm.flux_expression - mu * v, lb=0, ub=0,
                    name=f"_mu_{name}")])
            m.add_cons_vars([prob.Constraint(
                sum(ab.values()), lb=1.0, ub=1.0, name="_ab_sum")])
            m.objective = prob.Objective(sum(ab.values()), direction="max")
            sol = m.optimize(raise_error=False)
            if sol.status != "optimal":
                return None, None, None
            return ({n: float(ab[n].primal or 0.0) for n in members},
                    {k: float(v) for k, v in sol.fluxes.items()},
                    str(sol.status))

    lo, hi = 0.0, mu_max
    best = None
    ab0, fl0, st0 = feasible(0.0)
    if ab0 is None:
        return CommunityResult(growth=0.0, method="steadycom",
                               status="infeasible at mu=0")
    best = (0.0, ab0, fl0, st0)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        abs_, fl, st = feasible(mid)
        if abs_ is not None:
            lo = mid
            best = (mid, abs_, fl, st)
        else:
            hi = mid
        if hi - lo < tolerance:
            break

    mu, abundances, fluxes, status = best
    member_growth = {n: mu for n in members}

    exchanges = {r.id: fluxes.get(r.id, 0.0) for r in community.reactions
                 if r.id.startswith("EX_")}

    # cross-feeding: one member secretes what another consumes
    produced: Dict[str, List[Tuple[str, float]]] = {}
    consumed: Dict[str, List[Tuple[str, float]]] = {}
    for rid, v in fluxes.items():
        if "__EX_" not in rid:
            continue
        owner, ex = rid.split("__", 1)
        met = ex[len("EX_"):]
        if v > 1e-6:
            produced.setdefault(met, []).append((owner, v))
        elif v < -1e-6:
            consumed.setdefault(met, []).append((owner, -v))
    cross = []
    for met, prods in produced.items():
        for cons in consumed.get(met, []):
            for p in prods:
                if p[0] != cons[0]:
                    cross.append((p[0], cons[0], met, min(p[1], cons[1])))
    cross.sort(key=lambda t: -t[3])

    # The composition need not be unique: with sum(abundance) fixed at 1 the
    # feasibility LP has no preference between vertices, so two interchangeable
    # members can come back as 0.90/0.10 purely from solver tie-breaking.
    # Re-solve pushing the composition the other way and see whether it moves.
    unique = True
    alt = _alternate_composition(community, members, biomass_ids, mu)
    if alt is not None:
        drift = max(abs(alt.get(n, 0.0) - abundances.get(n, 0.0)) for n in members)
        unique = drift < 0.05
    notes = []
    if not unique:
        notes.append(
            "丰度分布不唯一:在同一群落生长率下存在多种可行组成"
            "(成员可互换或代谢冗余时必然如此)。这里给出的是其中一个顶点,"
            "不要当作预测的组成。")

    return CommunityResult(
        growth=mu, abundances=abundances, member_growth=member_growth,
        exchanges=exchanges, cross_feeding=cross, method="steadycom",
        status=status or "", abundance_is_unique=unique, notes=notes)


def _alternate_composition(community, members, biomass_ids, mu):
    """Solve the same feasibility problem with a different objective.

    If the composition is genuinely determined, both objectives land on it; if
    it drifts, the answer was an arbitrary vertex of a feasible set.
    """
    try:
        with community as m:
            prob = m.problem
            ab = {}
            for name in members:
                v = prob.Variable(f"_alt_{name}", lb=0, ub=1)
                m.add_cons_vars([v])
                ab[name] = v
                bm = m.reactions.get_by_id(biomass_ids[name])
                m.add_cons_vars([prob.Constraint(
                    bm.flux_expression - mu * v, lb=0, ub=0,
                    name=f"_altmu_{name}")])
            m.add_cons_vars([prob.Constraint(
                sum(ab.values()), lb=1.0, ub=1.0, name="_alt_sum")])
            first = members[0]
            m.objective = prob.Objective(ab[first], direction="min")
            sol = m.optimize(raise_error=False)
            if sol.status != "optimal":
                return None
            return {n: float(ab[n].primal or 0.0) for n in members}
    except Exception:
        return None


@register_function(
    aliases=["micom_grow", "MICOM", "micom", "群落丰度求解"],
    category="synthetic_biology",
    description="调用 MICOM 求解群落生长与丰度(需要 pip install micom)。MICOM 用协作权衡(cooperative tradeoff)而非严格等速率假设,与 steadycom 互为对照。Dispatch to MICOM for community growth.",
    examples=["res = ov.synbio.micom_grow({'ecoli': m1, 'bsub': m2})"],
    related=["synbio.steadycom", "synbio.community_model"],
    requires={},
    produces={},
)
def micom_grow(models, abundances=None, *, tradeoff: float = 0.5,
               medium=None) -> CommunityResult:
    """Solve a community with MICOM's cooperative-tradeoff objective."""
    try:
        import micom  # noqa: F401
        from micom import Community
    except ImportError as exc:
        raise ImportError(
            "method 需要 MICOM(pip install micom)。MICOM 用协作权衡目标,"
            "与内置的 steadycom(严格等速率)是不同的建模假设 —— "
            "两者都跑一遍再比较是有意义的。") from exc

    import pandas as pd

    names = list(models)
    ab = abundances or {n: 1.0 / len(names) for n in names}
    taxonomy = pd.DataFrame({
        "id": names,
        "genus": names,
        "abundance": [ab[n] for n in names],
        "model": [models[n] for n in names],
    })
    com = Community(taxonomy)
    if medium is not None:
        com.medium = medium
    sol = com.cooperative_tradeoff(fraction=tradeoff)
    member_growth = {str(k): float(v) for k, v in sol.members.growth_rate.items()
                     if str(k) != "medium"}
    return CommunityResult(
        growth=float(sol.growth_rate), abundances={n: float(ab[n]) for n in names},
        member_growth=member_growth, method="micom", status="optimal")


@register_function(
    aliases=["plot_community", "群落图", "丰度图", "互养图", "cross_feeding图"],
    category="synthetic_biology",
    description="画群落结果:成员丰度条形图,以及互养关系(谁分泌什么、谁消费)的流向图。Plot community abundances and the cross-feeding links.",
    examples=["ov.synbio.plot_community(res)"],
    related=["synbio.steadycom", "synbio.community_model"],
    requires={},
    produces={},
)
def plot_community(result: CommunityResult, top_n: int = 12, axes=None):
    """Abundance bars and the strongest cross-feeding links."""
    from ._plot import _mpl
    plt = _mpl()

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    else:
        fig = axes[0].figure
    axes = list(axes)

    ax = axes[0]
    members = result.members
    ax.bar(members, [result.abundances[m] for m in members], color="#377EB8")
    ax.set_ylabel("relative abundance")
    ax.set_ylim(0, 1)
    ax.set_title(f"community growth {result.growth:.4f} /h "
                 f"({result.method})")
    for i, m in enumerate(members):
        ax.text(i, result.abundances[m], f"{result.abundances[m]:.2f}",
                ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    links = result.cross_feeding[:top_n]
    if links:
        labels = [f"{src}→{dst}\n{met}" for src, dst, met, _ in links]
        ax.barh(range(len(links)), [f for *_, f in links], color="#4DAF4A")
        ax.set_yticks(range(len(links)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("exchanged flux")
    else:
        ax.text(0.5, 0.5, "no cross-feeding detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
    ax.set_title("cross-feeding")

    fig.tight_layout()
    return fig, axes


__all__ = ["rba", "RBAResult", "plot_rba", "community_model", "steadycom",
           "CommunityResult", "micom_grow", "plot_community"]
