r"""Layer A — enzyme-constrained metabolic models (a lightweight GECKO).

This is the metabolic half of the A↔B hinge that distinguishes ov.synbio from
a metabolism-only or a protein-only toolkit: turn a per-reaction turnover
number :math:`k_{cat}` (which the protein layer can *predict* from an enzyme
sequence via :func:`omicverse.synbio.enzyme_kcat`) into a hard capacity
constraint on the corresponding metabolic flux, then let FBA recompute the
achievable yield.

Formulation (single protein-pool, MOMENT/GECKO-light)
-----------------------------------------------------
For each reaction *r* carrying an assigned turnover :math:`k_{cat,r}` we add a
non-negative enzyme-usage variable :math:`e_r \ge |v_r| / k_{cat,r}` and a
single shared budget

.. math::  \sum_r MW_r \, e_r \le P

where :math:`P` is the total enzyme mass fraction available.  Lower
:math:`k_{cat,r}` ⇒ a larger :math:`e_r` is needed to carry the same flux ⇒ the
budget binds sooner ⇒ lower attainable growth/product yield.  The absolute
value is encoded with two linear constraints so reversible reactions need no
splitting.

Everything runs through the model's existing optlang problem, so the returned
object is still an ordinary :class:`cobra.Model` you can hand straight to
:func:`omicverse.synbio.fba`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Mapping, Optional

from .._registry import register_function
from ._gem import _cobra

if TYPE_CHECKING:  # pragma: no cover
    import cobra

# default molecular weight for an enzyme when none is supplied (g/mmol ≈ kDa).
_DEFAULT_MW = 40.0
_SECONDS_PER_HOUR = 3600.0
_POOL_MET = "prot_pool_synbio"


def _add_pool_constraints(
    model: "cobra.Model",
    kcat_map: Mapping[str, float],
    mw_map: Optional[Mapping[str, float]],
    total_protein: float,
) -> Dict[str, object]:
    """Add enzyme-usage variables + the shared pool constraint in place.

    Returns a dict of the created optlang variables/constraints (for
    bookkeeping / later inspection).
    """
    _strip_ec(model)
    prob = model.problem
    pool_terms = []
    created = {"usage_vars": {}, "constraints": []}
    for rxn_id, kcat in kcat_map.items():
        if rxn_id not in [r.id for r in model.reactions]:
            raise ValueError(f"kcat_map 中的反应 '{rxn_id}' 不在模型里。")
        if kcat is None or kcat <= 0:
            raise ValueError(f"反应 '{rxn_id}' 的 kcat 必须为正 (得到 {kcat})。")
        kcat_h = float(kcat) * _SECONDS_PER_HOUR  # 1/s -> 1/h
        mw = float((mw_map or {}).get(rxn_id, _DEFAULT_MW))
        rxn = model.reactions.get_by_id(rxn_id)
        v = rxn.flux_expression

        e = prob.Variable(f"e_usage_{rxn_id}", lb=0)
        model.add_cons_vars([e])
        # e >= v / kcat_h  ->  e - v/kcat_h >= 0
        c_pos = prob.Constraint(e - v / kcat_h, lb=0,
                                name=f"ec_pos_{rxn_id}")
        # e >= -v / kcat_h ->  e + v/kcat_h >= 0
        c_neg = prob.Constraint(e + v / kcat_h, lb=0,
                                name=f"ec_neg_{rxn_id}")
        model.add_cons_vars([c_pos, c_neg])
        created["usage_vars"][rxn_id] = e
        created["constraints"].extend([c_pos, c_neg])
        pool_terms.append(mw * e)

    if pool_terms:
        pool_expr = sum(pool_terms)
        pool = prob.Constraint(pool_expr, ub=float(total_protein),
                               name="ec_protein_pool")
        model.add_cons_vars([pool])
        created["pool"] = pool
    model.solver.update()
    return created


#: A single enzyme above this mass fraction of dry weight is not believable.
_IMPLAUSIBLE_MASS_FRACTION = 0.25


def _warn_impossible_enzyme_mass(model, kcat_map, mw_map) -> None:
    """Flag a supplied kcat that would need an impossible amount of enzyme.

    At the wild-type flux, ``mass = MW x v / (kcat x 3600)`` g/gDW. A kcat low
    enough to demand more protein than the cell weighs falsifies itself, and this
    is the cheapest possible check on a predicted kcat: DLKcat's 0.040 /s for
    *E. coli* phosphofructokinase implies **2.08 g PfkA per gDW** — 208 % of dry
    weight — at a WT flux of 7.48 mmol/gDW/h.
    """
    import warnings

    if not kcat_map:
        return
    try:
        sol = model.optimize()
        if sol.status != "optimal":
            return
    except Exception:                                    # pragma: no cover
        return
    bad = {}
    for rxn_id, kcat in kcat_map.items():
        if rxn_id not in sol.fluxes.index or not kcat or float(kcat) <= 0:
            continue
        mw = float((mw_map or {}).get(rxn_id, _DEFAULT_MW))
        mass = mw * abs(float(sol.fluxes[rxn_id])) / (float(kcat) * _SECONDS_PER_HOUR)
        if mass > _IMPLAUSIBLE_MASS_FRACTION:
            bad[rxn_id] = mass
    if bad:
        detail = ", ".join(f"{r}: {v:.2f} g/gDW" for r, v in
                           sorted(bad.items(), key=lambda kv: -kv[1])[:4])
        warnings.warn(
            f"以下反应的 kcat 低到需要不可能多的酶({detail}) —— 按野生型通量算,"
            f"单个酶就要占干重的 25% 以上,超过 1.0 则直接不可能。"
            f"这通常说明 kcat 预测值错了(先去 BRENDA/SABIO-RK 核一下),"
            f"而不是这个酶真的成了瓶颈。", stacklevel=3)


def _strip_ec(model) -> int:
    """Remove any enzyme-usage variables and pool constraint already on *model*.

    Without this, applying a second kcat map to a model that already carries an
    enzyme budget raises ``ContainerAlreadyContains: 'e_usage_ACALD'`` — so
    :func:`apply_kcat`, whose entire purpose is to re-solve a *different* kcat
    under the *same* protein budget, could not be called at all. That is the only
    fair way to compare enzyme variants, and it was unreachable.

    Returns how many objects were removed.
    """
    doomed = [c for c in model.constraints
              if c.name.startswith(("ec_pos_", "ec_neg_", "ec_protein_pool"))]
    doomed += [v for v in model.variables if v.name.startswith("e_usage_")]
    if doomed:
        model.remove_cons_vars(doomed)
        model.solver.update()
    return len(doomed)


def _baseline_enzyme_demand(model, kcat_map, mw_map) -> float:
    """Enzyme mass the *wild-type* optimum would need for the mapped reactions
    — used to auto-scale the pool so the constraint is actually informative."""
    sol = model.optimize()
    if sol.status != "optimal":
        return 0.0
    demand = 0.0
    for rxn_id, kcat in kcat_map.items():
        kcat_h = float(kcat) * _SECONDS_PER_HOUR
        mw = float((mw_map or {}).get(rxn_id, _DEFAULT_MW))
        demand += mw * abs(sol.fluxes[rxn_id]) / kcat_h
    return demand


@register_function(
    aliases=[
        "ec_model", "酶约束模型", "酶约束代谢模型", "enzyme_constrained_model",
        "GECKO", "ecModel", "酶约束",
    ],
    category="synthetic_biology",
    description="酶约束代谢模型 (GECKO / sMOMENT):把 kcat + 酶分子量转成蛋白质量预算约束。method='gecko'(默认)约束全酶促反应组(改一个酶的 kcat 会在共享蛋白预算里重新分配,物理正确);method='pool' 只约束给定反应。Enzyme-constrained model (sMOMENT/GECKO): full-proteome or targeted.",
    examples=[
        "ecm = ov.synbio.ec_model(m, {'PFK': 12.5})            # full-proteome GECKO",
        "ecm = ov.synbio.ec_model(m, {'PFK': 12.5}, method='pool')  # only PFK",
        "sol = ov.synbio.fba(ecm)  # yield recomputed under the enzyme budget",
    ],
    related=["synbio.apply_kcat", "synbio.enzyme_kcat", "synbio.fba", "synbio.load_gem"],
    requires={},
    produces={},
)
def ec_model(
    model: "cobra.Model",
    kcat_map: Mapping[str, float],
    mw_map: Optional[Mapping[str, float]] = None,
    total_protein: Optional[float] = None,
    method: str = "gecko",
    default_kcat: float = 25.0,
    pool_tightness: float = 0.5,
    inplace: bool = False,
) -> "cobra.Model":
    """Return an enzyme-constrained copy of *model* (sMOMENT / GECKO).

    Each enzymatic reaction *r* gets a non-negative usage variable
    :math:`e_r \\ge |v_r| / k_{cat,r}`, and a shared protein-mass budget
    :math:`\\sum_r MW_r\\, e_r \\le P` couples them — exactly the sMOMENT/GECKO
    formulation (Bekiaris & Klamt 2020; Sánchez *et al.* 2017).

    Parameters
    ----------
    model
        Base :class:`cobra.Model`.
    kcat_map
        ``{reaction_id: kcat}`` with kcat in **1/s** (turnover number).
    mw_map
        Optional ``{reaction_id: molecular_weight_kDa}`` (defaults to 40 kDa) —
        pass real per-enzyme masses for quantitative work.
    total_protein
        Protein-mass budget :math:`P` (g protein / gDW).  ``None`` auto-sizes it
        to ``pool_tightness ×`` the wild-type enzyme demand.
    method
        ``"gecko"`` (default) — constrain the **whole enzymatic reaction set**:
        reactions in ``kcat_map`` use their kcat, the rest use ``default_kcat``.
        The shared budget makes changing one enzyme's kcat physiologically
        reallocate the proteome (the correct A↔B behaviour).
        ``"pool"`` — constrain only the reactions in ``kcat_map`` (targeted).
    default_kcat
        kcat (1/s) assigned to unmapped enzymatic reactions under ``"gecko"``
        (BRENDA median ≈ 25/s).
    pool_tightness
        Fraction used when auto-sizing ``total_protein``.
    inplace
        Mutate *model* instead of copying.

    Returns
    -------
    cobra.Model
        Enzyme-constrained model; ``model.synbio_ec`` holds the created
        variables/constraints, the pool size, and the effective kcat map.
    """
    _cobra("ec_model")
    if method not in ("gecko", "pool"):
        raise ValueError(f"method must be one of ['gecko', 'pool'], got {method!r}")
    m = model if inplace else model.copy()

    if method == "gecko":
        # every gene-associated metabolic reaction draws on the proteome
        eff_kcat = {}
        skip = ("EX_", "DM_", "SK_", "BIOMASS", "ATPM")
        for r in m.reactions:
            if r.gene_reaction_rule and not r.id.startswith(skip) and "BIOMASS" not in r.id:
                eff_kcat[r.id] = float(kcat_map.get(r.id, default_kcat))
        for rid, k in kcat_map.items():        # honour explicit entries regardless
            eff_kcat[rid] = float(k)
    else:
        eff_kcat = {k: float(v) for k, v in kcat_map.items()}

    if total_protein is None:
        # Size the budget from a *reference* kcat applied to every mapped
        # reaction, NOT from the kcat values under test. Enzyme mass demand goes
        # as 1/kcat, so auto-sizing off `eff_kcat` made the budget grow as the
        # tested enzyme got slower — the constraint loosened, and an 830x slower
        # enzyme predicted 2x *more* growth. The A<->B hinge ran backwards.
        reference = {rid: float(default_kcat) for rid in eff_kcat}
        demand = _baseline_enzyme_demand(m, reference, mw_map)
        total_protein = pool_tightness * demand if demand > 0 else 1.0
        import warnings
        warnings.warn(
            f"total_protein 未指定,已按参考 kcat={default_kcat} /s 自动定为 "
            f"{total_protein:.4g} g/gDW(= {pool_tightness} x 野生型需求)。"
            f"要比较酶变体,请显式传 total_protein=,或用 apply_kcat() —— "
            f"它复用已有预算,这才是同一预算下比较 kcat 的做法。", stacklevel=2)

    _warn_impossible_enzyme_mass(model, kcat_map, mw_map)
    created = _add_pool_constraints(m, eff_kcat, mw_map, total_protein)
    created["total_protein"] = float(total_protein)
    created["method"] = method
    created["kcat_map"] = dict(eff_kcat)
    created["user_kcat"] = dict(kcat_map)
    # stash metadata without breaking cobra (plain attribute).
    try:
        m.synbio_ec = created
    except Exception:  # pragma: no cover
        pass
    return m


@register_function(
    aliases=[
        "apply_kcat", "更新kcat", "应用kcat", "set_kcat", "update_kcat",
        "改酶动力学",
    ],
    category="synthetic_biology",
    description="在已有酶约束模型上更新某些反应的 kcat(等价于换一版酶),返回新模型供 FBA 重算得率。Update kcat(s) on a model and rebuild the enzyme constraints.",
    examples=[
        "ecm2 = ov.synbio.apply_kcat(m, {'PFK': 25.0})  # a faster enzyme variant",
    ],
    related=["synbio.ec_model", "synbio.enzyme_kcat", "synbio.fba"],
    requires={},
    produces={},
)
def apply_kcat(
    model: "cobra.Model",
    kcat_map: Mapping[str, float],
    mw_map: Optional[Mapping[str, float]] = None,
    total_protein: Optional[float] = None,
    pool_tightness: float = 0.5,
) -> "cobra.Model":
    """Convenience wrapper around :func:`ec_model` reading a fresh kcat map.

    If *model* already carries a ``synbio_ec`` budget and ``total_protein`` is
    not given, that same budget is reused so you compare enzyme variants under
    an identical protein pool (the fair way to see a kcat change move yield).
    """
    prior = getattr(model, "synbio_ec", None)
    base = model
    # strip any prior EC scaffolding by copying from a clean base if present.
    if prior is not None and total_protein is None:
        total_protein = prior.get("total_protein")
    # rebuild on a fresh copy of the *original-style* model
    return ec_model(base, kcat_map, mw_map=mw_map,
                    total_protein=total_protein, pool_tightness=pool_tightness)


__all__ = ["ec_model", "apply_kcat"]
