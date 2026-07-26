r"""Display transforms for cytometry — logicle, hyperlog, asinh, log, linear.

WHY THESE ARE WRITTEN HERE RATHER THAN WRAPPED
----------------------------------------------
Fluorescence data cannot be shown on a linear axis: after compensation a real
population sits partly BELOW zero, and a log axis cannot represent that at all.
The field's answer is a family of biexponential functions that are logarithmic
in the bright decades and approximately linear through zero. Every serious
cytometry tool has them; omicverse had none.

The obvious move would be to call ``flowutils``. Three things argue against it,
and they are the reason this file exists:

1. **A hard version conflict.** ``flowutils`` >= 1.2 requires ``numpy>=2``,
   ``fcsparser`` requires ``numpy<2``, and omicverse's floor is ``numpy>=1.23``.
   Making the transform layer a hard dependency would force a numpy major on
   every omicverse user so that one submodule can import.

2. **It is a C extension.** ``flowutils`` ships ``logicle_c`` as a compiled
   object — logicle AND hyperlog are both in there; only asinh and log are
   Python. Vendoring it means shipping and building C, and the published source
   of its root finder is annotated as "based on RTSAFE from Numerical Recipes
   1st Edition", whose code is proprietary and non-redistributable. That is a
   problem for *vendoring*, never for depending on the wheel — and it is moot
   here, because the root finder below is an ordinary bisection written from
   scratch.

3. **The clean-room seam.** Gating-ML 2.0 is an ISAC *Recommendation* — a
   specification, not code — and the logicle parameterisation is published in
   Parks, Roederer & Moore, *Cytometry A* 2006;69(6):541-51. A spec-derived
   implementation can be reimplemented again in TypeScript for the browser with
   no notice obligation and no provenance question. A wrapper cannot.

So: derived from the specification, and verified against ``flowutils``' C
extension as a black-box ORACLE in ``tests/flow/test_transforms.py``. Testing
against a black box is not copying it; the parity test is what makes the
derivation trustworthy, and it is the module's stated exit criterion.

THE DEFINITIONS (GatingML 2.0 §6)
---------------------------------
``logicle(x, T, W, M, A) = root(B(y, T, W, M, A) - x)`` where
``B(y) = a*e^(b*y) - c*e^(-d*y) - f``

``hyperlog(x, T, W, M, A) = root(EH(y, T, W, M, A) - x)`` where
``EH(y) = a*e^(b*y) + c*y - f``

In both, ``T`` is the top of scale, ``M`` the number of decades the display
spans, ``W`` the number of those decades given over to the quasi-linear region
around zero, and ``A`` extra decades shown below zero. The scale output is in
``[0, 1]``; multiply by a channel count for pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .._registry import register_function

__all__ = [
    "Transform", "Linear", "Log", "Asinh", "Logicle", "Hyperlog",
    "make_transform", "transform_from_dict", "auto_transforms", "auto_width",
    "apply_transforms",
]

# float64 has ~2.2e-16 of relative precision; 100 bisection halvings of a unit
# interval reach ~8e-31, i.e. far past the point where it stops mattering. The
# loop exits on the interval width, so this is only a backstop against a
# pathological parameter set spinning forever.
_MAX_BISECT = 100


def _solve_d(b: float, w: float) -> float:
    """Solve ``2*ln(d/b) + w*(b + d) = 0`` for ``d`` in ``(0, b]``.

    This is the logicle parameter equation. The published reference solves it
    with a Numerical-Recipes-derived safeguarded Newton; this is a plain
    bisection, which is independent of it, needs no derivative, and cannot
    diverge — the equation is monotone decreasing in ``d`` on ``(0, b]``, so
    bracketing is trivial and convergence is guaranteed.
    """
    if w == 0:
        # W = 0 degenerates to a pure asinh-like form; d = b.
        return b

    def g(d: float) -> float:
        return 2.0 * math.log(d / b) + w * (b + d)

    # g is increasing in d: at d -> 0+ it goes to -inf, at d = b it is 2*w*b > 0.
    lo, hi = np.finfo(float).tiny, b
    if g(hi) <= 0:
        return b
    for _ in range(_MAX_BISECT):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 2.0 * np.finfo(float).eps * hi:
            break
    return 0.5 * (lo + hi)


def _bisect_root(fn, target: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Vectorised bisection of the increasing scalar function ``fn``.

    Both display transforms are defined as the ROOT of their own inverse, and
    both inverses are strictly increasing, so bisection is exact to float
    precision and — unlike Newton — has no starting-guess failure mode near the
    linearisation point, which is precisely where these functions are stiff.

    THE BRACKET IS EXPANDED, not clamped. [0, 1] is the *display* range, not the
    domain: an event dimmer than ``B(0)`` or brighter than ``T`` has a scale
    value outside [0, 1] and that is meaningful — it is off the axis, and
    callers gate and plot on it. Clamping instead made every value below
    ``B(0)`` collapse onto 0, which on a 262144/W=0.5 scale is everything under
    about -139 — i.e. exactly the deep-negative events that a badly compensated
    channel produces and that you most need to see.
    """
    t = np.asarray(target, dtype=float)
    a = np.full(t.shape, lo, dtype=float)
    b = np.full(t.shape, hi, dtype=float)
    # Grow outward geometrically until every target is bracketed. 40 doublings
    # covers ~1e12 scale units; anything past that is not a real measurement.
    for _ in range(40):
        need_lo = fn(a) > t
        need_hi = fn(b) < t
        if not (need_lo.any() or need_hi.any()):
            break
        span = np.maximum(b - a, 1.0)
        a = np.where(need_lo, a - span, a)
        b = np.where(need_hi, b + span, b)

    for _ in range(_MAX_BISECT):
        mid = 0.5 * (a + b)
        below = fn(mid) < t
        a = np.where(below, mid, a)
        b = np.where(below, b, mid)
        if np.all(b - a <= 1e-16 * np.maximum(1.0, np.abs(b))):
            break
    return 0.5 * (a + b)


@dataclass
class Transform:
    """Base class: a named, serialisable display scaling.

    Transforms are objects rather than functions because a gate is meaningless
    without the scaling its vertices were drawn on. Storing the transform beside
    the gate — and being able to round-trip it through ``to_dict`` — is what
    lets a gating strategy be saved, diffed, and re-applied to another sample.
    """

    #: Top of scale (the data value that maps to 1.0).
    t: float = 262144.0

    @property
    def name(self) -> str:
        return type(self).__name__.lower()

    def apply(self, x: np.ndarray) -> np.ndarray:            # pragma: no cover
        raise NotImplementedError

    def inverse(self, y: np.ndarray) -> np.ndarray:          # pragma: no cover
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        d = {"type": self.name}
        d.update({k: v for k, v in self.__dict__.items()})
        return d

    def ticks(self, n_decades: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Axis tick positions in SCALE space, with their data-space labels.

        A biexponential axis is unreadable with linear ticks — the whole point
        is that most of the range is logarithmic. Returns major ticks at powers
        of ten (plus their negatives, which is where these scales earn their
        keep) and unlabelled minor ticks at 2..9 within each decade.
        """
        top = float(self.t)
        decades = n_decades if n_decades is not None else int(math.ceil(math.log10(max(top, 10))))
        majors, labels, minors = [], [], []
        for e in range(0, decades + 1):
            v = 10.0 ** e
            for signed in ((v,) if e == 0 else (v, -v)):
                s = float(self.apply(np.array([signed]))[0])
                if 0.0 <= s <= 1.0:
                    majors.append(s)
                    labels.append(("-" if signed < 0 else "") + (f"10^{e}" if e else "0"))
            for k in range(2, 10):
                for signed in (k * v, -k * v):
                    s = float(self.apply(np.array([signed]))[0])
                    if 0.0 <= s <= 1.0:
                        minors.append(s)
        order = np.argsort(majors)
        return {
            "major": np.asarray(majors)[order],
            "labels": np.asarray(labels)[order],
            "minor": np.unique(np.asarray(minors)) if minors else np.asarray([]),
        }


@register_function(
    aliases=["Linear", "linear_transform", "线性变换", "flin"],
    category="flow",
    description=(
        "GatingML's flin. Right for scatter channels (FSC/SSC), wrong for fluorescence, which spans four or five decades."
    ),
    examples=[
        'ov.flow.Linear(t=262144, a=0.0)',
    ],
    related=["flow.auto_transforms", "flow.auto_width", "flow.make_transform"],
)
@dataclass
class Linear(Transform):
    """``y = (x - a) / (t - a)`` — GatingML's flin. Right for scatter, wrong for
    fluorescence."""

    a: float = 0.0

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.a) / (self.t - self.a)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) * (self.t - self.a) + self.a


@register_function(
    aliases=["Log", "log_transform", "对数变换", "flog"],
    category="flow",
    description=(
        "GatingML's flog. NaN at and below zero, which is the honest answer — and the reason compensated cytometry data needs a biexponential scale instead."
    ),
    examples=[
        'ov.flow.Log(t=262144, m=4.5)',
    ],
    related=["flow.auto_transforms", "flow.auto_width", "flow.make_transform"],
)
@dataclass
class Log(Transform):
    """``y = (1/m) * log10(x / t) + 1`` — GatingML's flog.

    Undefined at and below zero, which is exactly why compensated fluorescence
    needs one of the biexponentials instead. Non-positive input yields NaN
    rather than a silent clamp, so a mis-chosen scale is visible.
    """

    m: float = 4.5

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (1.0 / self.m) * np.log10(x / self.t) + 1.0
        return np.where(x > 0, out, np.nan)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return self.t * 10.0 ** ((np.asarray(y, dtype=float) - 1.0) * self.m)


@register_function(
    aliases=["Asinh", "asinh", "反双曲正弦变换", "cytof变换"],
    category="flow",
    description=(
        "Inverse hyperbolic sine scale — the CyTOF workhorse. Cofactor 5 for mass cytometry, around 150 for fluorescence."
    ),
    examples=[
        'ov.flow.Asinh(t=262144, m=4.5, a=0.0)',
    ],
    related=["flow.auto_transforms", "flow.auto_width", "flow.make_transform"],
)
@dataclass
class Asinh(Transform):
    """GatingML's fasinh. The CyTOF workhorse, and the simplest thing that
    handles negatives at all."""

    m: float = 4.0
    a: float = 1.0

    def apply(self, x: np.ndarray) -> np.ndarray:
        # GatingML 2.0 fasinh:
        #   (asinh(x * sinh(M*ln10) / T) + A*ln10) / ((M + A) * ln10)
        x = np.asarray(x, dtype=float)
        ln10 = math.log(10.0)
        return (np.arcsinh(x * math.sinh(self.m * ln10) / self.t) + self.a * ln10) / (
            (self.m + self.a) * ln10
        )

    def inverse(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        ln10 = math.log(10.0)
        return self.t * np.sinh((y * (self.m + self.a) - self.a) * ln10) / math.sinh(self.m * ln10)


@dataclass
class _Biexponential(Transform):
    """Shared parameter solve for logicle and hyperlog.

    Both take (T, W, M, A) and both are defined as the root of a strictly
    increasing function, so the only thing that differs is that function.
    """

    w: float = 0.5
    m: float = 4.5
    a: float = 0.0
    _p: Dict[str, float] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.t <= 0:
            raise ValueError("T must be positive")
        if self.m <= 0:
            raise ValueError("M must be positive")
        if self.w < 0:
            raise ValueError("W must be non-negative")
        if self.a < 0:
            raise ValueError("A must be non-negative")
        if 2 * self.w > self.m + self.a:
            raise ValueError(
                f"2*W ({2 * self.w}) must not exceed M+A ({self.m + self.a}) — "
                "the linear region cannot be wider than the scale."
            )
        self._p = self._solve()

    def _solve(self) -> Dict[str, float]:                     # pragma: no cover
        raise NotImplementedError

    def _fn(self, y: np.ndarray) -> np.ndarray:               # pragma: no cover
        raise NotImplementedError

    def apply(self, x: np.ndarray) -> np.ndarray:
        return _bisect_root(self._fn, np.asarray(x, dtype=float), 0.0, 1.0)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return self._fn(np.asarray(y, dtype=float))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.name, "t": self.t, "w": self.w, "m": self.m, "a": self.a}


@register_function(
    aliases=["Logicle", "logicle", "logicle变换", "双指数变换", "biexponential"],
    category="flow",
    description=(
        "The logicle display scale (Parks, Roederer & Moore 2006) — linear near zero and logarithmic away from it, so compensated data that goes BELOW zero can be displayed at all. The default for fluorescence. Its W parameter sets the width of the linear region and should be fitted to the data with ov.flow.auto_width, not chosen by hand."
    ),
    examples=[
        'ov.flow.Logicle(t=262144, m=4.5, w=1.0, a=0.0)',
        'ov.flow.auto_transforms(adata)',
    ],
    related=["flow.auto_transforms", "flow.auto_width", "flow.make_transform"],
)
@dataclass
class Logicle(_Biexponential):
    r"""Logicle / biexponential (Parks, Roederer & Moore 2006).

    ``B(y) = a*e^(b*y) - c*e^(-d*y) - f``, reflected below the linearisation
    point ``x1`` so the function is odd about it — that reflection is what makes
    the scale symmetric through zero and is the whole trick.
    """

    def _solve(self) -> Dict[str, float]:
        m, a_dec, w_dec, t = self.m, self.a, self.w, self.t
        w = w_dec / (m + a_dec)
        x2 = a_dec / (m + a_dec)
        x1 = x2 + w
        x0 = x2 + 2.0 * w
        b = (m + a_dec) * math.log(10.0)
        d = _solve_d(b, w)
        c_a = math.exp(x0 * (b + d))
        mf_a = math.exp(b * x1) - c_a / math.exp(d * x1)
        a = t / ((math.exp(b) - mf_a) - c_a / math.exp(d))
        c = c_a * a
        # f = +mf_a*a, not -mf_a*a. The defining constraint is B(x1) == 0 (x1 IS
        # the point the reflection is odd about, so it must map to data zero);
        # substituting gives B(x1) = a*(mf_a - f/a), which vanishes only for
        # this sign. With the sign flipped every value below ~1e4 was wrong —
        # data 0 mapped to 0.292 instead of x1 = 0.111 — while the top of scale
        # still landed exactly on T, so it looked plausible end-to-end.
        f = mf_a * a
        return {"a": a, "b": b, "c": c, "d": d, "f": f, "x1": x1}

    def _fn(self, y: np.ndarray) -> np.ndarray:
        p = self._p
        y = np.asarray(y, dtype=float)
        # Odd reflection about x1: below it, evaluate the mirrored point and
        # negate. Written branchlessly so the array stays one pass.
        below = y < p["x1"]
        yy = np.where(below, 2.0 * p["x1"] - y, y)
        val = p["a"] * np.exp(p["b"] * yy) - p["c"] * np.exp(-p["d"] * yy) - p["f"]
        return np.where(below, -val, val)


@register_function(
    aliases=["Hyperlog", "hyperlog", "超对数变换"],
    category="flow",
    description=(
        "The hyperlog display scale (Bagwell 2005) — biexponential like logicle but strictly increasing with no branch in the positive region."
    ),
    examples=[
        'ov.flow.Hyperlog(t=262144, m=4.5, w=1.0, a=0.0)',
    ],
    related=["flow.auto_transforms", "flow.auto_width", "flow.make_transform"],
)
@dataclass
class Hyperlog(_Biexponential):
    r"""Hyperlog (Bagwell 2005), GatingML's ``fhyperlog``.

    ``EH(y) = a*e^(b*y) + c*y - f``, reflected oddly about ``x1`` below it —
    exactly as logicle is.

    That reflection is NOT in the plain spec formula, and leaving it out is a
    silent, one-sided error: the parameters (a, c, f) come out identical, every
    value at or above ``x1`` agrees to 1e-16, and only the region below zero
    diverges — by 38 data units at scale 0, growing as you go down. The negative
    region is the entire reason to use a biexponential, so the bug would have
    hidden in the one place that matters. Established by comparing against
    ``flowutils``' C extension point by point (agreement 3.6e-15), not from the
    spec text.
    """

    def _solve(self) -> Dict[str, float]:
        m, a_dec, w_dec, t = self.m, self.a, self.w, self.t
        w = w_dec / (m + a_dec)
        x2 = a_dec / (m + a_dec)
        x1 = x2 + w
        x0 = x2 + 2.0 * w
        b = (m + a_dec) * math.log(10.0)
        e0 = math.exp(b * x0)
        ca = e0 / w if w > 0 else 0.0
        fa = math.exp(b * x1) + ca * x1
        a = t / (math.exp(b) + ca - fa)
        c = ca * a
        f = fa * a
        return {"a": a, "b": b, "c": c, "f": f, "x1": x1}

    def _fn(self, y: np.ndarray) -> np.ndarray:
        p = self._p
        y = np.asarray(y, dtype=float)
        below = y < p["x1"]
        yy = np.where(below, 2.0 * p["x1"] - y, y)
        val = p["a"] * np.exp(p["b"] * yy) + p["c"] * yy - p["f"]
        return np.where(below, -val, val)


_REGISTRY = {
    "linear": Linear, "flin": Linear,
    "log": Log, "flog": Log,
    "asinh": Asinh, "fasinh": Asinh,
    "logicle": Logicle,
    "hyperlog": Hyperlog, "fhyperlog": Hyperlog,
}


def make_transform(kind: str, **params: Any) -> Transform:
    """Build a transform by name. ``kind`` accepts the GatingML spelling too."""
    key = str(kind).lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"unknown transform {kind!r}; choose from {sorted(set(_REGISTRY))}"
        )
    return _REGISTRY[key](**params)


def transform_from_dict(d: Dict[str, Any]) -> Transform:
    """Rebuild a transform from :meth:`Transform.to_dict` — the other half of
    making a gating strategy round-trip through JSON."""
    d = dict(d)
    kind = d.pop("type")
    d.pop("_p", None)
    return make_transform(kind, **d)


# ── fitting a scale to the data ─────────────────────────────────────────────

#: Detector names that are scatter or timing rather than fluorescence. These get
#: a linear scale: they have no negative population to make room for, and a
#: biexponential axis on FSC is just a harder-to-read linear one.
_NON_FLUORESCENCE = ("FSC", "SSC", "TIME")


def auto_width(values: np.ndarray, t: float, m: float = 4.5) -> float:
    """The logicle linear-region width ``W`` implied by the data.

    ``W = (M - log10(T / |r|)) / 2`` with ``r`` the 5th percentile of the
    NEGATIVE values — make the linear region just wide enough to hold the noise
    the detector actually produced. Published in Parks, Roederer & Moore (2006),
    Cytometry A 69:541, the paper the logicle scale itself comes from.

    This is not a nicety. A ``W`` chosen by hand on one instrument is wrong on
    the next: real compensated negatives run to tens of thousands of units below
    zero, and a linear region sized for tens compresses the entire negative
    population into a spike against the axis, which is exactly where a threshold
    gate has to be drawn.

    Returns ``0.5`` when there is essentially nothing below zero — there is then
    no noise floor to make room for, and any estimate would be fitting one
    stray event.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    negatives = v[v < 0]
    if negatives.size < 10:
        return 0.5
    r = float(np.percentile(negatives, 5))
    w = (m - math.log10(float(t) / abs(r))) / 2.0
    # W must leave room for the log region: 0 < W < M/2.
    return float(np.clip(w, 0.1, m / 2 - 0.1))


def auto_transforms(
    adata: Any,
    *,
    markers: Optional[Any] = None,
    kind: str = "logicle",
    m: float = 4.5,
    layer: Optional[str] = None,
) -> Dict[str, "Transform"]:
    """Fit one display transform per channel, from the file and from the data.

    Two things that a hand-written transform dict gets wrong on real data:

    **The top of scale is in the file.** ``t`` comes from each channel's
    ``$PnR``, not from a constant. A BD instrument writes 262144; a Cytek Aurora
    writes 2293297 and 4194304 depending on the channel. Hard-coding one of them
    puts every event of the other on the wrong part of the axis.

    **The linear region has to be measured.** Fluorescence channels get
    :func:`auto_width` run against their own negative population; see there for
    why a hand-picked ``W`` does not travel between instruments.

    Scatter and time channels get a :class:`Linear` scale — they have no
    negative population, and a biexponential axis on FSC is a harder-to-read
    linear one.

    Arguments
    ---------
    adata
        Read by :func:`ov.io.read_fcs`, ideally **after** compensation, since
        compensation is what creates most of the negative values ``W`` is
        being fitted to.
    markers
        Restrict to these channels. Default is every channel in ``var``.
    kind
        ``'logicle'`` or ``'hyperlog'`` for the fluorescence channels.

    Returns ``{channel: Transform}``, ready to hand to a gate or a plot.

    >>> tr = ov.flow.auto_transforms(adata)          # doctest: +SKIP
    >>> ov.flow.biaxial(adata, 'CD4', 'CD8', transforms=tr)   # doctest: +SKIP
    """
    if kind.lower() not in ("logicle", "hyperlog"):
        raise ValueError("kind must be 'logicle' or 'hyperlog' — the other "
                         "scales take no fitted width")
    biexp = _REGISTRY[kind.lower()]

    names = list(adata.var.index) if markers is None else [str(x) for x in markers]
    missing = [n for n in names if n not in set(adata.var.index)]
    if missing:
        raise KeyError(f"not channels in this data: {missing}")

    X = adata.layers[layer] if layer else adata.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    index = {name: i for i, name in enumerate(adata.var.index)}

    out: Dict[str, Transform] = {}
    for name in names:
        col = X[:, index[name]]
        detector = str(adata.var["channel"].get(name, name)) if "channel" in adata.var else str(name)
        t = float(adata.var["PnR"].get(name, 0) or 0) if "PnR" in adata.var else 0.0
        if not t or not np.isfinite(t):
            # No $PnR — fall back to the observed maximum rather than to a
            # constant, and never to zero.
            t = float(max(np.nanmax(col), 1.0))
        if any(tag in detector.upper() or tag in str(name).upper()
               for tag in _NON_FLUORESCENCE):
            out[name] = Linear(t=t)
        else:
            out[name] = biexp(t=t, m=m, w=auto_width(col, t, m), a=0.0)
    return out


def apply_transforms(
    adata: Any,
    transforms: Optional[Dict[str, "Transform"]] = None,
    *,
    key_added: str = "transformed",
    layer: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Write the transformed values into ``adata.layers[key_added]``.

    Gating does not need this — a gate transforms its own two dimensions on the
    fly. Clustering does: a SOM measures distance across every marker at once,
    and on raw fluorescence that distance is dominated by whichever channel
    happens to have the largest numbers. The clusters that come out then describe
    the panel rather than the biology.

    ``transforms=None`` fits them with :func:`auto_transforms`, so the common
    case is one call:

    >>> ov.flow.apply_transforms(adata)                        # doctest: +SKIP
    >>> ov.flow.flowsom(adata, layer='transformed', markers=[...])  # doctest: +SKIP

    Channels with no transform are copied through unchanged rather than dropped,
    so the layer stays the same shape as ``X`` and a marker list keeps working.
    """
    if transforms is None:
        transforms = auto_transforms(adata, layer=layer, **kwargs)

    X = adata.layers[layer] if layer else adata.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X, dtype=float)
    out = np.array(X, dtype=float, copy=True)
    for i, name in enumerate(adata.var.index):
        tr = transforms.get(str(name))
        if tr is not None:
            out[:, i] = tr.apply(X[:, i])
    adata.layers[key_added] = out
    return adata
