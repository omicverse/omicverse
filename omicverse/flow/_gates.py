r"""Gate geometry for cytometry.

A gate is a named region in one or two measured dimensions plus the display
scaling its boundary was drawn on. That last part is not decoration: a polygon
drawn on a logicle axis and the same vertices read as linear values describe
completely different populations, so a gate that does not carry its transform
cannot be reapplied, saved or shared honestly.

Every gate is a pure predicate — ``mask(frame)`` returns a boolean array and
nothing else. No gate knows about AnnData, about its parent, or about which
sample it is being applied to; the tree lives in
:class:`~omicverse.flow._strategy.GatingStrategy`. That split is what makes one
strategy applicable to a hundred files.

Geometry conventions follow Gating-ML 2.0 (ISAC Recommendation), so gates
round-trip through the interchange format without a second representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._transforms import Transform, transform_from_dict

__all__ = [
    "Gate", "RectangleGate", "PolygonGate", "EllipsoidGate", "QuadrantGate",
    "BooleanGate", "gate_from_dict", "points_in_polygon",
]


def points_in_polygon(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test.

    Uses ``matplotlib.path.Path`` — already an omicverse dependency, measurably
    faster than the alternatives at 1e7 points, and the same even-odd rule
    Gating-ML specifies. Falls back to a vectorised ray cast if matplotlib is
    somehow unavailable, so a headless install cannot lose gating.

    Boundary points are INCLUDED. A vertex-exact event is vanishingly rare in
    real data but common in tests and in gates drawn by snapping, and silently
    dropping it would make a gate that visually contains a point report that it
    does not.
    """
    verts = np.asarray(vertices, dtype=float)
    pts = np.asarray(points, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError("polygon vertices must be an (n, 2) array")
    if verts.shape[0] < 3:
        raise ValueError("a polygon needs at least 3 vertices")
    if pts.size == 0:
        return np.zeros(pts.shape[0], dtype=bool)
    try:
        from matplotlib.path import Path
        # radius=0 excludes the boundary on some backends; a tiny positive
        # radius includes it consistently.
        return Path(verts, closed=False).contains_points(pts, radius=1e-12)
    except ImportError:                                       # pragma: no cover
        return _ray_cast(verts, pts)


def _ray_cast(verts: np.ndarray, pts: np.ndarray) -> np.ndarray:  # pragma: no cover
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(pts.shape[0], dtype=bool)
    n = verts.shape[0]
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= crosses & (x < xint)
    return inside


@dataclass
class Gate:
    """Base class. Subclasses implement :meth:`mask`."""

    name: str
    #: Channel or marker names, in the order the geometry uses them.
    dims: Tuple[str, ...]
    #: Per-dimension display scaling the boundary was defined on. A dimension
    #: with no entry is treated as linear (raw data units).
    transforms: Dict[str, Transform] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return type(self).__name__.replace("Gate", "").lower()

    def _scaled(self, frame: Mapping[str, np.ndarray]) -> np.ndarray:
        """Pull this gate's dimensions out of ``frame`` and put them on the
        scale its boundary was drawn on."""
        cols = []
        for d in self.dims:
            if d not in frame:
                raise KeyError(
                    f"gate {self.name!r} needs dimension {d!r}, which is not in "
                    f"the data (have: {sorted(frame)[:12]}...)"
                )
            v = np.asarray(frame[d], dtype=float)
            tr = self.transforms.get(d)
            cols.append(tr.apply(v) if tr is not None else v)
        return np.column_stack(cols)

    def mask(self, frame: Mapping[str, np.ndarray]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "dims": list(self.dims),
            "transforms": {k: v.to_dict() for k, v in self.transforms.items()},
            **self._payload(),
        }

    def _payload(self) -> Dict[str, Any]:                     # pragma: no cover
        raise NotImplementedError


@dataclass
class RectangleGate(Gate):
    """Axis-aligned interval in 1..n dimensions.

    Bounds are half-open ``[min, max)`` per Gating-ML, and ``None`` means
    unbounded on that side — which is how a one-sided threshold ("CD3 positive")
    is expressed without inventing an arbitrary ceiling.
    """

    bounds: Tuple[Tuple[Optional[float], Optional[float]], ...] = ()

    def __post_init__(self) -> None:
        if len(self.bounds) != len(self.dims):
            raise ValueError(
                f"gate {self.name!r}: {len(self.dims)} dims but {len(self.bounds)} bounds"
            )
        for d, (lo, hi) in zip(self.dims, self.bounds):
            if lo is not None and hi is not None and lo >= hi:
                raise ValueError(f"gate {self.name!r}, dim {d!r}: min {lo} >= max {hi}")

    def mask(self, frame: Mapping[str, np.ndarray]) -> np.ndarray:
        vals = self._scaled(frame)
        out = np.ones(vals.shape[0], dtype=bool)
        for i, (lo, hi) in enumerate(self.bounds):
            if lo is not None:
                out &= vals[:, i] >= lo
            if hi is not None:
                out &= vals[:, i] < hi
        return out

    def _payload(self) -> Dict[str, Any]:
        return {"bounds": [list(b) for b in self.bounds]}


@dataclass
class PolygonGate(Gate):
    """Arbitrary 2D region — the gate people actually draw."""

    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float)
        if len(self.dims) != 2:
            raise ValueError(f"gate {self.name!r}: a polygon needs exactly 2 dims")
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError(f"gate {self.name!r}: vertices must be (n, 2)")
        if self.vertices.shape[0] < 3:
            raise ValueError(f"gate {self.name!r}: a polygon needs at least 3 vertices")

    def mask(self, frame: Mapping[str, np.ndarray]) -> np.ndarray:
        return points_in_polygon(self.vertices, self._scaled(frame))

    def _payload(self) -> Dict[str, Any]:
        return {"vertices": self.vertices.tolist()}


@dataclass
class EllipsoidGate(Gate):
    """Mahalanobis-distance region: ``(x-mu)' S^-1 (x-mu) <= d2``.

    Gating-ML parameterises an ellipse by a covariance matrix and a squared
    distance rather than by semi-axes and an angle, so that is what is stored —
    converting on the way in would lose the interchange fidelity that is the
    only reason to have this gate type rather than a fine polygon.
    """

    mean: np.ndarray = field(default_factory=lambda: np.empty(0))
    covariance: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    distance_square: float = 1.0

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)
        k = len(self.dims)
        if self.mean.shape != (k,):
            raise ValueError(f"gate {self.name!r}: mean must have {k} entries")
        if self.covariance.shape != (k, k):
            raise ValueError(f"gate {self.name!r}: covariance must be {k}x{k}")
        if self.distance_square <= 0:
            raise ValueError(f"gate {self.name!r}: distance_square must be positive")

    def mask(self, frame: Mapping[str, np.ndarray]) -> np.ndarray:
        vals = self._scaled(frame) - self.mean
        inv = np.linalg.inv(self.covariance)
        d2 = np.einsum("ij,jk,ik->i", vals, inv, vals)
        return d2 <= self.distance_square

    def _payload(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "distance_square": float(self.distance_square),
        }


@dataclass
class QuadrantGate(Gate):
    """One divider per dimension, producing 2**n named populations.

    Modelled as ONE gate with several outputs rather than as four independent
    rectangles because that is what it is: the quadrants share their dividers,
    so moving a divider must move every boundary at once. Four separate gates
    drift apart the first time someone edits one.
    """

    #: Divider position per dimension, in the gate's own scale.
    dividers: Tuple[float, ...] = ()
    #: Population name per quadrant, indexed by the bit pattern of
    #: "above the divider" per dimension (dim 0 is the least significant bit).
    quadrant_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.dividers) != len(self.dims):
            raise ValueError(f"gate {self.name!r}: one divider per dimension")
        expected = 2 ** len(self.dims)
        if self.quadrant_names and len(self.quadrant_names) != expected:
            raise ValueError(
                f"gate {self.name!r}: {len(self.dims)} dims needs {expected} quadrant names"
            )
        if not self.quadrant_names:
            self.quadrant_names = tuple(
                self.name + "".join(
                    ("+" if (i >> b) & 1 else "-") for b in range(len(self.dims))
                )
                for i in range(expected)
            )

    def masks(self, frame: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """All quadrants at once, keyed by population name."""
        vals = self._scaled(frame)
        above = [vals[:, i] >= self.dividers[i] for i in range(len(self.dims))]
        out: Dict[str, np.ndarray] = {}
        for i, pop in enumerate(self.quadrant_names):
            m = np.ones(vals.shape[0], dtype=bool)
            for b in range(len(self.dims)):
                m &= above[b] if (i >> b) & 1 else ~above[b]
            out[pop] = m
        return out

    def mask(self, frame: Mapping[str, np.ndarray]) -> np.ndarray:
        """Everything the quadrants cover — which is every event. Present only
        so a QuadrantGate satisfies the Gate contract; use :meth:`masks`."""
        return np.ones(next(iter(frame.values())).shape[0], dtype=bool)

    def _payload(self) -> Dict[str, Any]:
        return {
            "dividers": list(self.dividers),
            "quadrant_names": list(self.quadrant_names),
        }


@dataclass
class BooleanGate(Gate):
    """AND / OR / NOT over populations that have already been computed.

    Takes no dimensions of its own; it is evaluated by the strategy, which is
    the only thing that knows what the referenced populations resolved to.
    """

    operator: str = "and"
    operands: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        op = self.operator.lower()
        if op not in ("and", "or", "not"):
            raise ValueError(f"gate {self.name!r}: operator must be and/or/not")
        self.operator = op
        if op == "not" and len(self.operands) != 1:
            raise ValueError(f"gate {self.name!r}: 'not' takes exactly one operand")
        if op != "not" and len(self.operands) < 2:
            raise ValueError(f"gate {self.name!r}: {op!r} needs at least two operands")

    def combine(self, resolved: Mapping[str, np.ndarray]) -> np.ndarray:
        missing = [o for o in self.operands if o not in resolved]
        if missing:
            raise KeyError(
                f"boolean gate {self.name!r} references population(s) {missing} "
                "that have not been computed — a boolean gate can only refer to "
                "populations defined before it."
            )
        parts = [np.asarray(resolved[o], dtype=bool) for o in self.operands]
        if self.operator == "not":
            return ~parts[0]
        out = parts[0].copy()
        for p in parts[1:]:
            out = (out & p) if self.operator == "and" else (out | p)
        return out

    def mask(self, frame: Mapping[str, np.ndarray]) -> np.ndarray:
        raise TypeError(
            f"boolean gate {self.name!r} is evaluated by the GatingStrategy, "
            "not against raw event data"
        )

    def _payload(self) -> Dict[str, Any]:
        return {"operator": self.operator, "operands": list(self.operands)}


_KINDS = {
    "rectangle": RectangleGate,
    "polygon": PolygonGate,
    "ellipsoid": EllipsoidGate,
    "quadrant": QuadrantGate,
    "boolean": BooleanGate,
}


def gate_from_dict(d: Mapping[str, Any]) -> Gate:
    """Rebuild a gate from :meth:`Gate.to_dict` — the half of round-tripping
    that makes a saved strategy usable again."""
    d = dict(d)
    kind = d.pop("kind")
    if kind not in _KINDS:
        raise ValueError(f"unknown gate kind {kind!r}; expected one of {sorted(_KINDS)}")
    d["dims"] = tuple(d.get("dims", ()))
    d["transforms"] = {k: transform_from_dict(v) for k, v in (d.get("transforms") or {}).items()}
    for key in ("bounds", "dividers", "quadrant_names", "operands"):
        if key in d and d[key] is not None:
            d[key] = tuple(tuple(x) if isinstance(x, list) else x for x in d[key])
    return _KINDS[kind](**d)
