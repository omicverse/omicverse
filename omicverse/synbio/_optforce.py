r"""OptForce — which fluxes *must* change, and the smallest set of interventions
that forces them (Ranganathan, Suthers & Maranas 2010).

OptKnock and RobustKnock answer "which deletions couple product to growth".
OptForce asks a different and often more actionable question: comparing the flux
ranges the wild type can occupy against the ranges an over-producing strain must
occupy, **which reactions cannot stay where they are?** Reactions whose two
ranges do not overlap have to move — no amount of regulatory slack can keep them
put — and that set is derived, not guessed.

Two stages, and they differ in kind:

* the **MUST sets** are deterministic. Run FVA on the wild type and again on the
  model pinned to the target production level; any reaction whose intervals are
  disjoint belongs to MUST^U (must increase) or MUST^L (must decrease). Nothing
  heuristic happens here, which is why this module reports the MUST sets as a
  first-class result rather than only as an intermediate.
* the **FORCE set** is a search over subsets of MUST for the smallest
  combination that actually forces production. The published formulation is a
  bilevel MILP; ``straindesign`` ships OptKnock, RobustKnock and OptCouple but
  no OptForce, so the selection here is an explicit greedy search over MUST
  candidates, scored with the same guaranteed-product metric the rest of
  ``ov.synbio`` uses. It is honest about being a search: :attr:`OptForceResult.
  force_is_exact` is False, and the guaranteed product is reported so a design
  can be checked rather than trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

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


@dataclass
class Intervention:
    """One engineering action on one reaction."""

    reaction: str
    action: str           # 'up' | 'down' | 'knockout'
    wild_type: Tuple[float, float]
    target: Tuple[float, float]

    def __repr__(self) -> str:  # pragma: no cover
        return (f"{self.action}({self.reaction}: "
                f"[{self.wild_type[0]:.3g},{self.wild_type[1]:.3g}] -> "
                f"[{self.target[0]:.3g},{self.target[1]:.3g}])")


@dataclass
class OptForceResult:
    """MUST sets (derived) and a FORCE set (searched)."""

    target: str
    must_increase: List[Intervention] = field(default_factory=list)
    must_decrease: List[Intervention] = field(default_factory=list)
    force: List[Intervention] = field(default_factory=list)
    guaranteed_product: float = 0.0
    baseline_product: float = 0.0
    growth: float = 0.0
    force_is_exact: bool = False
    target_flux: float = 0.0

    @property
    def must(self) -> List[Intervention]:
        return self.must_increase + self.must_decrease

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        rows = []
        forced = {(i.reaction, i.action) for i in self.force}
        for iv in self.must:
            rows.append({
                "reaction": iv.reaction,
                "action": iv.action,
                "wt_min": iv.wild_type[0], "wt_max": iv.wild_type[1],
                "target_min": iv.target[0], "target_max": iv.target[1],
                "in_force_set": (iv.reaction, iv.action) in forced,
            })
        df = pd.DataFrame(rows)
        return df if df.empty else df.set_index("reaction")

    def __repr__(self) -> str:  # pragma: no cover
        return (f"OptForceResult(target={self.target!r}, "
                f"MUST↑{len(self.must_increase)} MUST↓{len(self.must_decrease)}, "
                f"FORCE={len(self.force)}, guaranteed={self.guaranteed_product:.4g})")


def _fva_ranges(model, reactions, fraction: float) -> Dict[str, Tuple[float, float]]:
    cobra = _cobra("optforce")
    df = cobra.flux_analysis.flux_variability_analysis(
        model, reaction_list=reactions, fraction_of_optimum=fraction)
    return {rid: (float(row["minimum"]), float(row["maximum"]))
            for rid, row in df.iterrows()}


def _biomass(model):
    for r in model.reactions:
        if r.objective_coefficient != 0:
            return r
    for r in model.reactions:
        if "BIOMASS" in r.id.upper():
            return r
    return None


def _guaranteed_product(m, biomass, target_rxn: str, keep_frac: float):
    """Minimum product flux the cell must carry at (near-)maximal growth."""
    import numpy as np

    g = m.slim_optimize()
    if g is None or not np.isfinite(g) or g <= 1e-9:
        return None, (0.0 if g is None or not np.isfinite(g) else float(g))
    with m as inner:
        biomass.lower_bound = keep_frac * g
        inner.objective = target_rxn
        inner.objective.direction = "min"
        mn = inner.slim_optimize()
    return (float(mn) if mn is not None and np.isfinite(mn) else 0.0), float(g)


@register_function(
    aliases=["optforce", "OptForce", "必须改变集", "must_sets", "强制集",
             "force_set", "通量必改集"],
    category="synthetic_biology",
    description="OptForce:比较野生型与目标过量生产态的通量区间,找出区间不重叠、因此**必须改变**的反应(MUST↑/MUST↓),再搜索能真正强制出产物的最小干预组合(FORCE)。MUST 集是确定性推导,FORCE 集是显式贪心搜索(straindesign 没有 OptForce),并报告可保证产物量以便核对。OptForce: derive MUST sets from disjoint flux ranges, then search a FORCE set.",
    examples=[
        "res = ov.synbio.optforce(m, 'EX_succ_e', target_fraction=0.5)",
        "res.must_increase, res.must_decrease, res.force",
        "res.to_frame()",
    ],
    related=["synbio.strain_design", "synbio.knockout_flux", "synbio.fva",
             "synbio.plot_optforce"],
    requires={},
    produces={},
)
def optforce(
    model: "cobra.Model",
    target: str,
    *,
    target_fraction: float = 0.5,
    wild_type_fraction: float = 0.9,
    growth_fraction: float = 0.1,
    max_interventions: int = 5,
    candidate_limit: int = 60,
    reactions: Optional[Sequence[str]] = None,
) -> OptForceResult:
    """Derive MUST sets for over-producing ``target``, then search a FORCE set.

    Parameters
    ----------
    model
        The model. Never mutated.
    target
        Reaction id to over-produce — usually an exchange such as
        ``'EX_succ_e'``.
    target_fraction
        The over-production level to aim for, as a fraction of the *theoretical
        maximum* target flux. 0.5 asks "what must change to reach half the
        theoretical yield" — high enough to force real rewiring, low enough to
        stay feasible.
    wild_type_fraction
        Growth fraction used for the wild-type FVA. Below 1.0 because a real
        culture is not exactly at the optimum, and a razor-thin wild-type
        interval would declare almost everything a MUST reaction.
    growth_fraction
        Growth that must be retained when scoring a candidate FORCE set.
    max_interventions
        Largest FORCE set to search for.
    candidate_limit
        Cap on MUST reactions considered during the FORCE search, taken in order
        of how far the intervals are apart. The search is O(n × set size) LPs.
    reactions
        Restrict the analysis to these reactions (default: all non-boundary).

    Returns
    -------
    OptForceResult
    """
    import numpy as np
    _cobra("optforce")

    try:
        target_rxn = model.reactions.get_by_id(target)
    except KeyError as exc:
        raise ValueError(
            f"目标反应 {target!r} 不在模型里。产物一般是交换反应(如 'EX_succ_e');"
            f"可以用 [r.id for r in model.reactions if r.boundary] 看候选。") from exc

    biomass = _biomass(model)
    if biomass is None:
        raise ValueError("模型没有 biomass/目标反应,OptForce 需要它来定义生长约束。")

    if reactions is None:
        reactions = [r.id for r in model.reactions
                     if not r.boundary and r.id != biomass.id]
    reactions = list(reactions)

    # 1) theoretical maximum target flux, and the level we aim for
    with model as m:
        m.objective = target
        m.objective.direction = "max"
        max_target = m.slim_optimize()
    max_target = 0.0 if max_target is None or not np.isfinite(max_target) else float(max_target)
    if max_target <= 1e-9:
        raise ValueError(
            f"模型在当前培养基下无法产生 {target!r}(理论最大通量 ≈ 0)。"
            f"先检查培养基/交换反应边界,或换一个产物。")
    aim = target_fraction * max_target

    # 2) wild-type ranges, and ranges under forced over-production
    wt = _fva_ranges(model, reactions, wild_type_fraction)
    with model as m:
        target_rxn_m = m.reactions.get_by_id(target)
        target_rxn_m.lower_bound = aim
        # The same growth fraction as the wild-type pass, deliberately. Running
        # this side with fraction_of_optimum=0 drops the growth requirement and
        # widens every interval to the network's full capability — two maximally
        # wide intervals almost always overlap, so the MUST sets came back
        # essentially empty. The target state has to be a *viable* overproducer
        # for "cannot stay where it is" to mean anything.
        tgt = _fva_ranges(m, reactions, wild_type_fraction)

    # 3) MUST sets — intervals that cannot overlap
    up: List[Intervention] = []
    down: List[Intervention] = []
    for rid in reactions:
        w = wt.get(rid)
        t = tgt.get(rid)
        if w is None or t is None:
            continue
        if t[0] > w[1] + 1e-6:
            up.append(Intervention(rid, "up", w, t))
        elif t[1] < w[0] - 1e-6:
            down.append(Intervention(rid, "down", w, t))

    # 4) FORCE search over MUST candidates
    def gap(iv: Intervention) -> float:
        return (iv.target[0] - iv.wild_type[1]) if iv.action == "up" \
            else (iv.wild_type[0] - iv.target[1])

    candidates = sorted(up + down, key=gap, reverse=True)[:candidate_limit]

    base_prod, _ = _guaranteed_product(model.copy(), _biomass(model.copy()),
                                       target, growth_fraction)
    base_prod = 0.0 if base_prod is None else base_prod

    chosen: List[Intervention] = []
    best_prod = base_prod
    best_growth = 0.0
    pool = list(candidates)

    for _ in range(max_interventions):
        improved = None
        improved_prod = best_prod
        improved_growth = best_growth
        for iv in pool:
            trial = model.copy()
            _apply(trial, chosen + [iv])
            bm = _biomass(trial)
            if bm is None:
                continue
            prod, growth = _guaranteed_product(trial, bm, target, growth_fraction)
            if prod is None:
                continue
            if prod > improved_prod + 1e-9:
                improved, improved_prod, improved_growth = iv, prod, growth
        if improved is None:
            break
        chosen.append(improved)
        best_prod, best_growth = improved_prod, improved_growth
        pool = [c for c in pool if c.reaction != improved.reaction]

    return OptForceResult(
        target=target, must_increase=up, must_decrease=down, force=chosen,
        guaranteed_product=best_prod, baseline_product=base_prod,
        growth=best_growth, force_is_exact=False, target_flux=aim,
    )


def _apply(m, interventions: Sequence[Intervention]) -> None:
    """Pin each intervened reaction into the range the target state requires."""
    for iv in interventions:
        rxn = m.reactions.get_by_id(iv.reaction)
        lo, hi = iv.target
        if iv.action == "up":
            rxn.lower_bound = max(rxn.lower_bound, lo)
        elif iv.action == "down":
            rxn.upper_bound = min(rxn.upper_bound, hi)
        else:
            rxn.knock_out()


@register_function(
    aliases=["plot_optforce", "OptForce图", "MUST集图", "plot_must_sets",
             "必改集图"],
    category="synthetic_biology",
    description="画 OptForce 结果:MUST↑/MUST↓ 反应的野生型区间与目标区间对比(区间不重叠即必须改变),以及 FORCE 集带来的可保证产物提升。Plot OptForce MUST intervals and the FORCE set's gain.",
    examples=["ov.synbio.plot_optforce(res)"],
    related=["synbio.optforce"],
    requires={},
    produces={},
)
def plot_optforce(result: OptForceResult, top_n: int = 15, axes=None):
    """Interval plot of the MUST reactions, plus the guaranteed-product gain."""
    from ._plot import _mpl
    plt = _mpl()

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                 gridspec_kw={"width_ratios": [2, 1]})
    else:
        fig = axes[0].figure
    axes = list(axes)

    ax = axes[0]
    ivs = (result.must_increase + result.must_decrease)[:top_n]
    forced = {(i.reaction, i.action) for i in result.force}
    for i, iv in enumerate(ivs):
        in_force = (iv.reaction, iv.action) in forced
        ax.plot(iv.wild_type, [i, i], lw=6, solid_capstyle="butt",
                color="#B0B0B0", alpha=0.9)
        ax.plot(iv.target, [i, i], lw=6, solid_capstyle="butt",
                color="#E41A1C" if in_force else "#377EB8", alpha=0.9)
    ax.set_yticks(range(len(ivs)))
    ax.set_yticklabels([f"{'↑' if iv.action == 'up' else '↓'} {iv.reaction}"
                        for iv in ivs], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("flux range")
    ax.set_title("grey = wild type, blue = required, red = in FORCE set")
    if not ivs:
        ax.text(0.5, 0.5, "no MUST reactions\n(target already reachable)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)

    ax = axes[1]
    vals = [result.baseline_product, result.guaranteed_product]
    ax.bar(["wild type", f"+{len(result.force)} interventions"], vals,
           color=["#B0B0B0", "#4DAF4A"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("guaranteed product flux")
    ax.set_title("growth-coupled yield")

    fig.tight_layout()
    return fig, axes


__all__ = ["optforce", "OptForceResult", "Intervention", "plot_optforce"]
