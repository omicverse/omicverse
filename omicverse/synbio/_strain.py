r"""Layer A — strain design: which genes/reactions to knock out or amplify to
push a genome-scale model toward over-producing a target metabolite.

Two entry points:

* :func:`production_envelope` — the classic growth-vs-product trade-off curve
  (pure COBRApy).
* :func:`strain_design` — ranked amplification **and** knockout targets.  The
  default method is dependency-free: FSEOF (Flux Scanning based on Enforced
  Objective Flux, Choi *et al.* 2010) for amplification targets plus a
  single-reaction knockout scan scored by product yield for deletion targets.
  If `cameo <https://github.com/biosustain/cameo>`_ is installed you may switch
  ``method="optgene"`` / ``"optknock"`` for exact MILP designs.

All CPU, all on top of a :class:`cobra.Model`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from .._registry import register_function
from ._gem import _cobra

if TYPE_CHECKING:  # pragma: no cover
    import cobra
    import pandas as pd


@dataclass
class StrainDesignResult:
    """Ranked engineering targets for over-producing ``target``.

    Attributes
    ----------
    target : str
        Target reaction id (usually an exchange / demand reaction).
    amplify : pandas.DataFrame
        Over-expression targets (FSEOF slope > 0), most promising first.
    knockout : pandas.DataFrame
        Single-deletion targets ranked by predicted product yield gain while
        growth stays feasible.
    method : str
        Which method produced the design (``"fseof"`` or a cameo method).
    """

    target: str
    amplify: "pd.DataFrame"
    knockout: "pd.DataFrame"
    method: str = "fseof"
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"StrainDesignResult(target={self.target!r}, method={self.method!r}, "
            f"amplify={len(self.amplify)} targets, knockout={len(self.knockout)} targets)"
        )


def _resolve_target(model, target: str):
    """Return a reaction id to maximise for *target*.

    Accepts a reaction id directly, or a metabolite id — in which case a
    demand reaction is (temporarily, within the caller's ``with model``) the
    responsibility of the caller; here we just map metabolite→its exchange if
    one exists.
    """
    if target in [r.id for r in model.reactions]:
        return target
    # try to find an exchange reaction for a metabolite id
    if target in [m.id for m in model.metabolites]:
        met = model.metabolites.get_by_id(target)
        for rxn in met.reactions:
            if rxn.id.startswith("EX_") or rxn.id.startswith("DM_"):
                return rxn.id
    raise ValueError(
        f"target '{target}' 不是模型中的反应或可映射的交换反应。"
        f" 请传入产物的交换/需求反应 id (如 'EX_succ_e')。"
    )


@register_function(
    aliases=[
        "production_envelope", "生产包络", "得率生长权衡", "phenotypic_phase_plane",
        "生产曲线",
    ],
    category="synthetic_biology",
    description="生产包络:扫描生长速率并计算目标产物的可行得率区间,给出得率-生长权衡曲线。Production envelope (growth vs product-yield trade-off) → DataFrame.",
    examples=["df = ov.synbio.production_envelope(m, 'EX_ac_e')"],
    related=["synbio.strain_design", "synbio.fba"],
    requires={},
    produces={},
)
def production_envelope(
    model: "cobra.Model",
    target: str,
    carbon_source: Optional[str] = None,
    points: int = 20,
) -> "pd.DataFrame":
    """Growth-vs-production trade-off curve for *target*."""
    cobra = _cobra("production_envelope")
    tgt = _resolve_target(model, target)
    return cobra.flux_analysis.production_envelope(
        model, reactions=[tgt], carbon_sources=carbon_source, points=points,
    )


def _fseof(model, target_rxn: str, n_steps: int = 10, growth_frac: float = 0.1):
    """Flux Scanning based on Enforced Objective Flux.

    Enforce increasing lower bounds on product flux (from a low fraction to
    ~90% of its theoretical max) and record how each reaction's flux moves.
    Reactions whose |flux| rises monotonically with enforced production are
    amplification targets; the slope ranks them.
    """
    import numpy as np
    import pandas as pd

    with model as m:
        # capture the growth (biomass) objective *before* touching it.
        orig_obj = m.objective
        # theoretical max product flux (ignoring growth)
        with m as probe:
            probe.objective = target_rxn
            max_prod = probe.slim_optimize()
        if max_prod is None or not np.isfinite(max_prod) or max_prod <= 1e-9:
            return pd.DataFrame(columns=["reaction", "slope", "l0", "l1"])

        levels = np.linspace(0.1 * max_prod, 0.9 * max_prod, n_steps)
        rxn_ids = [r.id for r in m.reactions]
        flux_track = {rid: [] for rid in rxn_ids}
        prod_rxn = m.reactions.get_by_id(target_rxn)
        # enforce increasing product while maximising growth (biomass objective)
        m.objective = orig_obj
        for lvl in levels:
            prod_rxn.lower_bound = float(lvl)
            sol = m.optimize()
            if sol.status != "optimal":
                for rid in rxn_ids:
                    flux_track[rid].append(np.nan)
                continue
            for rid in rxn_ids:
                flux_track[rid].append(sol.fluxes[rid])
        prod_rxn.lower_bound = 0.0

    rows = []
    x = levels
    for rid, ys in flux_track.items():
        y = np.asarray(ys, dtype=float)
        if np.any(~np.isfinite(y)):
            continue
        a = np.abs(y)
        # amplification target: |flux| grows as production is enforced.
        # rank by slope; require a net increase and a positive trend, but
        # tolerate small non-monotone numerical wiggles.
        if a[-1] - a[0] > 1e-6:
            slope = float(np.polyfit(x, a, 1)[0])
            # positive trend (allow tiny dips) and target reaction excluded
            if slope > 1e-9 and rid != target_rxn:
                rows.append((rid, slope, float(a[0]), float(a[-1])))
    df = pd.DataFrame(rows, columns=["reaction", "slope", "l0", "l1"])
    if len(df):
        df = df.sort_values("slope", ascending=False).reset_index(drop=True)
    return df


def _biomass_reaction(model):
    for r in model.reactions:
        if r.objective_coefficient != 0:
            return r
    return None


def _guaranteed_product(m, biomass, target_rxn, keep_frac):
    """At (near-)max growth, the *minimum* product flux the cell must carry.

    This is the growth-coupling metric OptKnock optimises: maximise growth,
    fix it, then minimise product — a positive value means product is forced
    even when the cell only cares about growing.
    """
    import numpy as np

    g = m.slim_optimize()
    if g is None or not np.isfinite(g) or g <= 1e-9:
        return None, g  # lethal / no growth
    with m as inner:
        biomass.lower_bound = keep_frac * g
        inner.objective = target_rxn
        inner.objective.direction = "min"
        min_prod = inner.slim_optimize()
    return (float(min_prod) if min_prod is not None and np.isfinite(min_prod)
            else 0.0), float(g)


def _knockout_scan(model, target_rxn: str, growth_frac: float = 0.1,
                   max_targets: int = 30):
    """Rank single-reaction knockouts by **growth-coupled** product yield.

    OptKnock surrogate: a good knockout makes the product unavoidable at the
    cell's own growth optimum.  For each non-essential reaction we delete it,
    re-maximise growth (must stay ≥ ``growth_frac`` of wild-type), then measure
    the guaranteed (minimum) product flux at that growth.  Knockouts that force
    more product than wild-type are the designs.
    """
    import numpy as np
    import pandas as pd

    cols = ["reaction", "coupled_product", "growth"]
    with model as m:
        biomass = _biomass_reaction(m)
        if biomass is None:
            return pd.DataFrame(columns=cols)
        wt_growth = m.slim_optimize()
        if wt_growth is None or wt_growth <= 1e-9:
            return pd.DataFrame(columns=cols)
        wt_coupled, _ = _guaranteed_product(m, biomass, target_rxn, 0.99)
        wt_coupled = wt_coupled or 0.0

    candidates = [r for r in model.reactions
                  if not r.id.startswith("EX_") and not r.id.startswith("DM_")
                  and r.id != target_rxn and r.gene_reaction_rule]
    rows = []
    for r in candidates:
        with model as m:
            bm = _biomass_reaction(m)
            m.reactions.get_by_id(r.id).knock_out()
            coupled, g = _guaranteed_product(m, bm, target_rxn, 0.99)
            if coupled is None or g is None:
                continue  # lethal knockout
            if g < growth_frac * wt_growth:
                continue  # too costly for growth
            if coupled > wt_coupled + 1e-6:
                rows.append((r.id, coupled, g))
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df = df.sort_values("coupled_product", ascending=False).reset_index(drop=True)
        df = df.head(max_targets)
    return df


@register_function(
    aliases=[
        "strain_design", "菌株设计", "代谢工程", "敲除靶点", "过表达靶点",
        "metabolic_engineering", "optknock", "robustknock", "fseof",
    ],
    category="synthetic_biology",
    description="菌株设计:过表达 (FSEOF) + 敲除靶点。method='fseof'(免依赖,生长偶联启发式);method='optknock'/'robustknock' 走 StrainDesign 的真实双层 MILP。Rank amplification (FSEOF) & knockout targets; exact OptKnock/RobustKnock via StrainDesign.",
    examples=[
        "res = ov.synbio.strain_design(m, 'EX_succ_e')",
        "res.amplify.head(); res.knockout.head()",
    ],
    related=["synbio.production_envelope", "synbio.single_gene_deletion", "synbio.ec_model"],
    requires={},
    produces={},
)
def strain_design(
    model: "cobra.Model",
    target: str,
    method: str = "fseof",
    growth_frac: float = 0.1,
    max_knockouts: int = 30,
    time_limit: int = 300,
) -> StrainDesignResult:
    """Rank engineering targets for over-producing *target*.

    Parameters
    ----------
    model
        A :class:`cobra.Model`.
    target
        Reaction id of the product (usually an exchange, e.g. ``"EX_succ_e"``).
    method
        ``"fseof"`` (default, dependency-free): FSEOF amplification targets +
        a knockout yield-scan.  ``"optgene"`` / ``"optknock"`` route to cameo
        when it is installed.
    growth_frac
        Minimum growth (fraction of wild-type) that knockouts must preserve.
    max_knockouts
        Cap on the number of knockout targets returned.

    Returns
    -------
    StrainDesignResult
    """
    _cobra("strain_design")
    _VALID = ("fseof", "optknock", "robustknock")
    if method not in _VALID:
        raise ValueError(f"method must be one of {list(_VALID)}, got {method!r}")
    tgt = _resolve_target(model, target)

    if method in ("optknock", "robustknock"):
        # exact bilevel-MILP design via StrainDesign; FSEOF still gives the
        # complementary over-expression targets.
        knockout = _straindesign_milp(model, tgt, method, max_knockouts,
                                      time_limit=time_limit)
        amplify = _fseof(model, tgt)
        return StrainDesignResult(target=tgt, amplify=amplify, knockout=knockout,
                                  method=method)

    amplify = _fseof(model, tgt)
    knockout = _knockout_scan(model, tgt, growth_frac=growth_frac,
                              max_targets=max_knockouts)
    return StrainDesignResult(target=tgt, amplify=amplify, knockout=knockout,
                              method="fseof")


def _straindesign_milp(model, target_rxn: str, method: str, max_cost: int,
                       max_solutions: int = 5, time_limit: int = 300):
    """Exact OptKnock / RobustKnock via the StrainDesign package (real MILP).

    Returns a DataFrame with one row per computed design: the knockout set and
    its size."""
    import logging
    import pandas as pd
    try:
        import straindesign as sd
    except ImportError as exc:
        raise ImportError(
            "method='%s' 需要 StrainDesign(真实 MILP 菌株设计)。请 "
            "pip install straindesign(或 pip install 'omicverse[synbio]')。"
            % method) from exc
    for lg in ("straindesign", "straindesign.compression", "root"):
        logging.getLogger(lg).setLevel(logging.ERROR)

    biomass = _biomass_reaction(model)
    if biomass is None:
        raise ValueError("模型没有目标(biomass)反应,OptKnock 需要它作为内层目标。")
    name = sd.names.OPTKNOCK if method == "optknock" else sd.names.ROBUSTKNOCK
    module = sd.SDModule(model, name, inner_objective=biomass.id,
                         outer_objective=target_rxn)
    res = sd.compute_strain_designs(
        model, sd_modules=[module], max_cost=max_cost,
        max_solutions=max_solutions, solver="glpk", time_limit=time_limit)

    designs = getattr(res, "reaction_sd", None) or []
    rows = []
    for d in designs:
        kos = sorted(r for r, v in d.items() if v < 0)
        if kos:
            rows.append({"knockouts": kos, "n_knockouts": len(kos)})
    return pd.DataFrame(rows, columns=["knockouts", "n_knockouts"])


__all__ = ["strain_design", "production_envelope", "StrainDesignResult"]
