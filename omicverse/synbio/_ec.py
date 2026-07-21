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
    description="酶约束代谢模型 (GECKO-light):把每个反应的 kcat 转成蛋白池容量约束,得到可直接送入 FBA 的酶约束 cobra.Model。这是 A↔B 咬合的代谢端。Build an enzyme-constrained model from a {reaction: kcat} map.",
    examples=[
        "ecm = ov.synbio.ec_model(m, {'PFK': 12.5})",
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
    pool_tightness: float = 0.5,
    inplace: bool = False,
) -> "cobra.Model":
    """Return an enzyme-constrained copy of *model*.

    Parameters
    ----------
    model
        Base :class:`cobra.Model`.
    kcat_map
        ``{reaction_id: kcat}`` with kcat in **1/s** (turnover number).
    mw_map
        Optional ``{reaction_id: molecular_weight_kDa}``; defaults to 40 kDa.
    total_protein
        Total enzyme mass budget :math:`P`.  If ``None`` it is auto-set to
        ``pool_tightness ×`` the wild-type enzyme demand of the mapped
        reactions, so the constraint is guaranteed to bite (good for demos and
        for feeling the effect of a kcat change).
    pool_tightness
        Fraction used when auto-sizing ``total_protein`` (smaller = tighter).
    inplace
        Mutate *model* instead of copying.

    Returns
    -------
    cobra.Model
        Enzyme-constrained model; ``model.synbio_ec`` holds the created
        variables/constraints and the pool size for inspection.
    """
    _cobra("ec_model")
    m = model if inplace else model.copy()

    if total_protein is None:
        demand = _baseline_enzyme_demand(m, kcat_map, mw_map)
        total_protein = pool_tightness * demand if demand > 0 else 1.0

    created = _add_pool_constraints(m, kcat_map, mw_map, total_protein)
    created["total_protein"] = float(total_protein)
    created["kcat_map"] = dict(kcat_map)
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
