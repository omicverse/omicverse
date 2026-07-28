r"""Pathway search / retrosynthesis over a genome-scale reaction network.

Given a genome-scale model (the reaction database) and a target metabolite,
find routes that produce it — the design step of "which enzymes do I need to
express to make X?". A backward breadth-first search over the bipartite
metabolite–reaction graph enumerates the shortest pathways from available
precursors (medium / a chosen source) to the target.

This is the in-network baseline. For *de novo* heterologous routes from a
reaction-rule database, use RetroPath / RetroRules; :func:`pathway_search`
already covers "can my host, plus known reactions, reach this target, and how".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set

from .._registry import register_function
from ._gem import _cobra

if TYPE_CHECKING:  # pragma: no cover
    import cobra


@dataclass
class Pathway:
    """A candidate production route to a target metabolite."""
    target: str
    reactions: List[str]              # reaction ids, target-producing first
    length: int
    equations: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Pathway(target={self.target!r}, {self.length} steps: {self.reactions})"


_CURRENCY = {"h", "h2o", "atp", "adp", "amp", "pi", "ppi", "nad", "nadh",
             "nadp", "nadph", "co2", "coa", "o2", "nh4", "q8", "q8h2", "pqq",
             "fad", "fadh2", "accoa"}


def _base(mid: str) -> str:
    return mid.rsplit("_", 1)[0] if mid[-2:] in ("_c", "_e", "_p", "_m") else mid


@register_function(
    aliases=["pathway_search", "通路搜索", "逆合成", "retrosynthesis",
             "pathway_design", "异源通路", "retropath", "生产路线"],
    category="synthetic_biology",
    description="通路搜索/逆合成:在基因组尺度模型(反应库)上,从可用前体到目标代谢物做反向 BFS,枚举最短生产路线(要表达哪些酶)。Find shortest production pathways to a target metabolite in a GEM.",
    examples=[
        "m = ov.synbio.load_gem('e_coli_core')",
        "paths = ov.synbio.pathway_search(m, 'succ_c', source='glc__D_e')",
    ],
    related=["synbio.load_gem", "synbio.max_min_driving_force", "synbio.strain_design"],
    requires={},
    produces={},
)
def pathway_search(model: "cobra.Model", target: str,
                   source: Optional[str] = None, max_depth: int = 6,
                   max_paths: int = 5,
                   ignore_currency: bool = True,
                   exclude_transport: bool = True) -> List[Pathway]:
    """Enumerate shortest production pathways to *target*.

    Parameters
    ----------
    model
        A :class:`cobra.Model` used as the reaction database.
    target
        Target metabolite id (e.g. ``'succ_c'``). If not found, a ``*_c`` match
        is attempted.
    source
        Optional precursor metabolite id to root pathways at (e.g. a carbon
        source). If ``None``, any medium/exchange metabolite terminates a route.
    max_depth
        Maximum pathway length (reactions).
    max_paths
        Number of pathways to return (shortest first).
    ignore_currency
        Skip ubiquitous cofactors (ATP, NAD(H), H2O, …) when tracing precursors.
    exclude_transport
        Exclude reactions that only move a metabolite across a membrane, and
        exchange reactions. Without this the shortest route to ``succ_c`` is
        ``SUCCt2_2`` — import succinate — with *dctA* named as "the gene to
        express", and three of five routes to succinate were transporters. A
        transport step is real chemistry but it is not a way to *make* the
        molecule, which is what a production route is asked for.

    Returns
    -------
    list[Pathway]
    """
    _cobra("pathway_search")
    met_ids = {m.id for m in model.metabolites}
    if target not in met_ids:
        cand = [m for m in met_ids if _base(m) == _base(target) or m.startswith(target)]
        if not cand:
            raise ValueError(f"目标代谢物 '{target}' 不在模型中。")
        target = cand[0]

    # producer map: metabolite -> reactions that can produce it (both directions)
    def is_transport(rxn) -> bool:
        """True when a reaction only relocates a metabolite between compartments.

        Detected structurally: some base metabolite id appears on both the
        substrate and the product side. That catches symporters, antiporters and
        PTS-style transport without needing an annotation.
        """
        left = {_base(m.id) for m, c in rxn.metabolites.items() if c < 0}
        right = {_base(m.id) for m, c in rxn.metabolites.items() if c > 0}
        shared = {b for b in left & right if not (ignore_currency and b in _CURRENCY)}
        return bool(shared)

    producers: Dict[str, List] = {}
    for rxn in model.reactions:
        if exclude_transport and (rxn.boundary or is_transport(rxn)):
            continue
        lb, ub = rxn.lower_bound, rxn.upper_bound
        for met, coeff in rxn.metabolites.items():
            fwd = coeff > 0 and ub > 0
            rev = coeff < 0 and lb < 0
            if fwd or rev:
                producers.setdefault(met.id, []).append((rxn, 1 if fwd else -1))

    def substrates(rxn, direction) -> List[str]:
        subs = []
        for met, coeff in rxn.metabolites.items():
            consumed = (coeff < 0) if direction == 1 else (coeff > 0)
            if consumed:
                if ignore_currency and _base(met.id) in _CURRENCY:
                    continue
                subs.append(met.id)
        return subs

    # Which base metabolites the medium supplies, in any compartment. Availability
    # has to look *through* transport even when production routes may not use it:
    # glucose is an available precursor in the cytosol because glc__D_e is in the
    # medium and a transporter exists. Requiring the exact compartment-qualified id
    # to carry an EX_ reaction meant that, once transport was excluded from the
    # producer map, no cytosolic route could terminate at all and the search
    # returned nothing.
    medium_bases = {_base(m.id) for m in model.metabolites
                    if any(r.boundary for r in m.reactions)}

    def is_available(mid: str) -> bool:
        if source and mid == source:
            return True
        if ignore_currency and _base(mid) in _CURRENCY:
            return True
        if source is not None:
            return False
        # the target itself is never "available" — that is the question being asked
        if _base(mid) == _base(target):
            return False
        return _base(mid) in medium_bases

    # BFS over states (frontier metabolites still to be made, reactions used)
    from collections import deque
    start_frontier = frozenset([target])
    queue = deque([(start_frontier, [], {target})])
    found: List[Pathway] = []
    seen: Set = set()

    while queue and len(found) < max_paths:
        frontier, used, visited = queue.popleft()
        # pick an unmet metabolite to expand
        need = [mid for mid in frontier if not is_available(mid)]
        if not need:
            eqs = [model.reactions.get_by_id(r).build_reaction_string() for r in used]
            found.append(Pathway(target=target, reactions=list(used),
                                  length=len(used), equations=eqs))
            continue
        if len(used) >= max_depth:
            continue
        mid = need[0]
        for rxn, direction in producers.get(mid, [])[:8]:
            if rxn.id in used or rxn.id.startswith("EX_"):
                continue
            subs = substrates(rxn, direction)
            new_frontier = (frozenset((frozenset(frontier) - {mid}) | set(subs)))
            key = (new_frontier, rxn.id)
            if key in seen:
                continue
            seen.add(key)
            queue.append((new_frontier, used + [rxn.id],
                          visited | {mid} | set(subs)))

    found.sort(key=lambda p: p.length)
    return found[:max_paths]


__all__ = ["pathway_search", "Pathway"]
