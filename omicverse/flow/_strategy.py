r"""The gating strategy — a tree of gates that belongs to the ANALYSIS, not to
any one sample.

THIS IS THE WHOLE POINT OF THE MODULE. A gate on its own is a predicate; what a
cytometrist actually has is a *strategy*: time gate, then debris, then singlets,
then live, then lineage — each step evaluated only on the events its parent
kept. Modelling that tree as a first-class object, independent of the file it
runs on, is what makes the two things FlowJo cannot do possible: applying one
strategy to ninety samples, and diffing two strategies to see what changed.

PARENT-RESTRICTED EVALUATION — but only where it pays
-----------------------------------------------------
The obvious implementation gates every event at every node and ANDs with the
parent's mask. The obvious fix is to test only the parent's members instead.
Measured on this implementation, the fix is not free and not always a win:

    parent keeps    1e6 events    5e6 events
        50%            0.11x         0.94x      <- SLOWER
        10%            5.17x         5.27x
         1%           18.28x        22.21x

Gathering the subset costs a fancy-index copy, and near the root — where a
"Cells" gate keeps 95% of events — that copy costs more than the geometry it
saves. Deep in a hierarchy, where a population is a few percent of its parent,
it is worth an order of magnitude.

So the traversal restricts only when the parent is actually selective. The
threshold is measured, not guessed, and the pass-through case is not a
degradation: it is the same work the naive version would have done. Correctness
is identical either way and pinned by a test that compares the two directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._gates import BooleanGate, Gate, QuadrantGate, gate_from_dict

__all__ = ["GatingStrategy", "GatingResult"]

#: Prefix for the boolean columns written into ``adata.obs``. Namespaced so a
#: population called "Time" cannot collide with a real obs column.
OBS_PREFIX = "gate:"

ROOT = "root"

#: Restrict a child's evaluation to its parent's members only when the parent
#: keeps at most this fraction. Measured crossover sits between 10% and 50%;
#: 1/3 is inside the region where restriction was 5x faster and safely clear of
#: the region where it was slower.
RESTRICT_BELOW = 1.0 / 3.0


@dataclass
class GatingResult:
    """What a strategy produced on one sample."""

    #: population name -> boolean mask over all events
    masks: Dict[str, np.ndarray]
    #: population name -> its parent's name (``root`` for top-level)
    parents: Dict[str, str]
    n_events: int

    def stats(self, min_events: int = 100) -> pd.DataFrame:
        """Per-population counts and frequencies.

        ``freq_parent`` is the number people actually report — "12% of CD3+" —
        and it is meaningless without knowing the denominator, so the parent is
        named in its own column rather than left implicit.

        ``low_n`` flags populations small enough that the percentage is noise.
        A frequency quoted off 30 events has a 95% CI of roughly +/-18 points;
        reporting it to one decimal place, as every cytometry tool does, is
        false precision that this column exists to contradict.
        """
        rows = []
        for name, mask in self.masks.items():
            parent = self.parents.get(name, ROOT)
            n = int(mask.sum())
            n_parent = int(self.masks[parent].sum()) if parent in self.masks else self.n_events
            rows.append({
                "population": name,
                "parent": parent,
                "count": n,
                "parent_count": n_parent,
                "freq_parent": (n / n_parent) if n_parent else np.nan,
                "freq_total": (n / self.n_events) if self.n_events else np.nan,
                "low_n": n < min_events,
            })
        df = pd.DataFrame(rows)
        return df.sort_values("population").reset_index(drop=True)


class GatingStrategy:
    """An ordered tree of gates.

    >>> gs = GatingStrategy("T cells")
    >>> gs.add_gate(singlets)                    # doctest: +SKIP
    >>> gs.add_gate(live, parent="Singlets")     # doctest: +SKIP
    >>> res = gs.apply(adata)                    # doctest: +SKIP
    >>> res.stats()                              # doctest: +SKIP
    """

    def __init__(self, name: str = "strategy") -> None:
        self.name = name
        self._gates: Dict[str, Gate] = {}
        self._parent: Dict[str, str] = {}
        self._order: List[str] = []

    # ── construction ────────────────────────────────────────────────────────

    def add_gate(self, gate: Gate, parent: str = ROOT) -> "GatingStrategy":
        """Add ``gate`` under ``parent``. Returns self, so calls chain."""
        if gate.name in self._gates:
            raise ValueError(f"a gate named {gate.name!r} is already in this strategy")
        if gate.name == ROOT:
            raise ValueError(f"{ROOT!r} is reserved for the ungated population")
        if parent != ROOT and parent not in self._gates:
            # Checked here rather than at apply() time so the error names the
            # gate being added, which is the thing the caller can fix.
            known = [ROOT] + self._order
            raise KeyError(
                f"gate {gate.name!r} names parent {parent!r}, which is not in this "
                f"strategy. Known populations: {known}"
            )
        self._gates[gate.name] = gate
        self._parent[gate.name] = parent
        self._order.append(gate.name)
        if isinstance(gate, QuadrantGate):
            # Each quadrant is a population in its own right and must be
            # addressable as a parent; they share the gate object.
            for pop in gate.quadrant_names:
                if pop in self._parent:
                    raise ValueError(f"quadrant population {pop!r} collides with an existing name")
                self._parent[pop] = parent
        return self

    @property
    def gates(self) -> Mapping[str, Gate]:
        return dict(self._gates)

    def parent_of(self, population: str) -> str:
        return self._parent.get(population, ROOT)

    def children_of(self, population: str) -> List[str]:
        return [n for n, p in self._parent.items() if p == population]

    def __len__(self) -> int:
        return len(self._gates)

    def __repr__(self) -> str:
        return f"GatingStrategy({self.name!r}, {len(self._gates)} gates)"

    def tree(self) -> str:
        """The hierarchy as indented text — the thing you actually want when a
        strategy misbehaves."""
        lines: List[str] = [ROOT]

        def walk(node: str, depth: int) -> None:
            for child in self.children_of(node):
                lines.append("  " * (depth + 1) + "└ " + child)
                walk(child, depth + 1)

        walk(ROOT, 0)
        return "\n".join(lines)

    # ── evaluation ──────────────────────────────────────────────────────────

    def apply(
        self,
        adata: Any,
        *,
        layer: Optional[str] = None,
        write_obs: bool = True,
        min_events: int = 100,
    ) -> GatingResult:
        """Run the strategy over an AnnData and return the per-population masks.

        Dimensions are looked up in ``var`` first (by index, i.e. marker name
        where the file named one) and then in ``obs``, so a gate can be drawn on
        a measured channel or on anything derived — a UMAP axis, a cluster
        score — without a second code path.
        """
        frame = self._frame(adata, layer=layer)
        n = adata.n_obs
        masks: Dict[str, np.ndarray] = {}

        for name in self._order:
            gate = self._gates[name]
            parent = self._parent[name]
            parent_mask = masks.get(parent) if parent != ROOT else None

            if isinstance(gate, BooleanGate):
                # Booleans compose already-resolved populations, so they are not
                # restricted — restricting would silently intersect with the
                # parent and change the meaning of "NOT".
                masks[name] = gate.combine(masks)
                continue

            # Restrict to the parent's members only when the parent is
            # selective enough for the gather to pay for itself — see the
            # module docstring for the measurement behind this threshold.
            idx = None
            if parent_mask is not None:
                kept = float(parent_mask.mean())
                if kept <= RESTRICT_BELOW:
                    idx = np.flatnonzero(parent_mask)
            sub = {d: (frame[d][idx] if idx is not None else frame[d]) for d in gate.dims}

            if isinstance(gate, QuadrantGate):
                for pop, m in gate.masks(sub).items():
                    full = self._expand(m, idx, n)
                    if idx is None and parent_mask is not None:
                        full &= parent_mask
                    masks[pop] = full
                masks[name] = parent_mask.copy() if parent_mask is not None else np.ones(n, bool)
            else:
                m = self._expand(gate.mask(sub), idx, n)
                # When the gather was skipped the gate saw every event, so the
                # parent constraint has to be reapplied. Forgetting this is how
                # a child ends up containing events its parent excluded.
                if idx is None and parent_mask is not None:
                    m &= parent_mask
                masks[name] = m

        if write_obs:
            for name, m in masks.items():
                adata.obs[OBS_PREFIX + name] = pd.Categorical(m.astype(bool))

        return GatingResult(masks=masks, parents=dict(self._parent), n_events=n)

    @staticmethod
    def _expand(sub_mask: np.ndarray, idx: Optional[np.ndarray], n: int) -> np.ndarray:
        """Lift a mask computed on the parent's members back to all events."""
        if idx is None:
            return np.asarray(sub_mask, dtype=bool)
        full = np.zeros(n, dtype=bool)
        full[idx[np.asarray(sub_mask, dtype=bool)]] = True
        return full

    @staticmethod
    def _frame(adata: Any, layer: Optional[str] = None) -> Dict[str, np.ndarray]:
        X = adata.layers[layer] if layer else adata.X
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        frame: Dict[str, np.ndarray] = {
            str(name): X[:, i] for i, name in enumerate(adata.var.index)
        }
        # A detector name reaches the same column as its marker, so a strategy
        # written against 'FITC-A' still applies to a file indexed by 'CD3'.
        if "channel" in adata.var.columns:
            for i, chan in enumerate(adata.var["channel"]):
                if chan and str(chan) not in frame:
                    frame[str(chan)] = X[:, i]
        for col in adata.obs.columns:
            if col.startswith(OBS_PREFIX):
                continue
            values = adata.obs[col].to_numpy()
            if np.issubdtype(values.dtype, np.number):
                frame.setdefault(str(col), values)
        return frame

    # ── serialisation ───────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able. This is the seam the UI works against: a gate editor
        renders and edits these dicts and never needs the Python objects."""
        return {
            "name": self.name,
            "gates": [
                {**self._gates[n].to_dict(), "parent": self._parent[n]}
                for n in self._order
            ],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GatingStrategy":
        gs = cls(d.get("name", "strategy"))
        for entry in d.get("gates", []):
            entry = dict(entry)
            parent = entry.pop("parent", ROOT)
            gs.add_gate(gate_from_dict(entry), parent=parent)
        return gs
