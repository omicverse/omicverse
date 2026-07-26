r"""Plotting for ``ov.flow`` — because gating is a visual procedure.

Every other part of this module can be checked by reading a number. A gate
cannot. "CD3+ is 41.2% of singlets" is not evidence that the CD3 gate is drawn
in the right place; the only thing that settles it is seeing the boundary lying
on the population, on the scale the boundary was drawn on. A cytometry module
that returns a statistics table and no picture asks the reader to trust the
gate, which is precisely what a reviewer will not do.

WHAT MAKES THESE PLOTS RATHER THAN GENERIC SCATTERS
    The axis is the scale.  Events are drawn in SCALE space — the same
        ``Transform`` the gate carries — and the ticks are re-labelled back into
        data units via :meth:`Transform.ticks`. Draw a logicle gate over a
        linear axis and the boundary lands somewhere else entirely; here the
        plot cannot disagree with the gate, because it reads the scaling off the
        gate object instead of being told it a second time.
    The events are the gate's events.  Channel lookup goes through
        ``GatingStrategy._frame``, the same accessor :meth:`GatingStrategy.apply`
        uses, so a marker/detector alias resolves identically in the picture and
        in the mask.
    Density, not overplot.  At 1e5 events a plain scatter is a black blob and
        every population looks equally dense. Colour is per-event 2D-histogram
        density, drawn low-to-high so rare events survive.

The one thing to keep in mind: percentages shown on a plot come from the
``GatingResult`` when you pass one, not from re-running the gate on whatever is
being displayed. Those differ — a child gate is restricted by its parent — and
the number a reader should see is the one the strategy actually produced.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._registry import register_function
from ._gates import (
    BooleanGate,
    EllipsoidGate,
    Gate,
    PolygonGate,
    QuadrantGate,
    RectangleGate,
)
from ._strategy import ROOT, GatingResult, GatingStrategy
from ._transforms import Linear, Transform

__all__ = [
    "biaxial",
    "histogram",
    "backgate",
    "hierarchy",
    "plot_strategy",
    "plot_batch",
    "spillover_heatmap",
    "flowsom_heatmap",
]


# ── shared plumbing ─────────────────────────────────────────────────────────


def _frame(adata: Any, layer: Optional[str]) -> Dict[str, np.ndarray]:
    """Channel values by name — the accessor the gating strategy uses.

    Deliberately not reimplemented here. If marker/detector aliasing ever
    changes, the plot has to change with it or the picture starts describing a
    different column than the mask does.
    """
    return GatingStrategy._frame(adata, layer=layer)


def _resolve_transform(
    dim: str,
    transforms: Optional[Mapping[str, Transform]],
    strategy: Optional[GatingStrategy],
    gates: Sequence[Gate],
) -> Optional[Transform]:
    """Find the scaling for ``dim``: explicit, then from the gates being drawn,
    then from anywhere in the strategy.

    Searching the gates first matters when a strategy scales the same channel
    two ways (it happens — a wide logicle for the lineage plot, a tighter one
    downstream). The gate on THIS plot is the one whose axis the reader is
    looking at.
    """
    if transforms and dim in transforms:
        return transforms[dim]
    for g in gates:
        tr = g.transforms.get(dim)
        if tr is not None:
            return tr
    if strategy is not None:
        for g in strategy.gates.values():
            tr = g.transforms.get(dim)
            if tr is not None:
                return tr
    return None


def _scaled(values: np.ndarray, tr: Optional[Transform]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return tr.apply(v) if tr is not None else v


def _style_axis(ax, tr: Optional[Transform], which: str, label: str) -> None:
    """Put data-space labels on a scale-space axis.

    Without this the axis reads 0.0–1.0, which is the internal representation
    and means nothing to anyone. ``Transform.ticks`` already knows where the
    decades land — it existed before this module did and had no caller.

    Decade ticks are only right for an axis that IS logarithmic somewhere. On a
    linear scatter axis they pile 0, 10¹, 10², 10³ and 10⁴ into the leftmost 4%
    of the plot, where they overprint each other into an unreadable smudge —
    which is what this function did until the first FSC × SSC figure was
    actually looked at. Linear axes get evenly spaced ticks instead.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    setter = ax.set_xlabel if which == "x" else ax.set_ylabel
    setter(label)
    if tr is None:
        return
    if isinstance(tr, Linear):
        _style_linear_axis(ax, tr, which)
        return
    t = tr.ticks()
    major, labels, minor = t["major"], t["labels"], t["minor"]
    if len(major) == 0:
        return
    lo, hi = (ax.get_xlim() if which == "x" else ax.get_ylim())
    major, labels = _thin_ticks(major, labels, 0.055 * abs(hi - lo))
    pretty = [_tick_label(s) for s in labels]
    if which == "x":
        ax.set_xticks(major)
        ax.set_xticks(minor, minor=True)
        ax.set_xticklabels(pretty)
    else:
        ax.set_yticks(major)
        ax.set_yticks(minor, minor=True)
        ax.set_yticklabels(pretty)
    axis.set_tick_params(which="minor", length=2)
    axis.set_tick_params(which="major", length=4)


def _thin_ticks(
    positions: np.ndarray, labels: np.ndarray, min_sep: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop labelled decades that would overprint their neighbours.

    The linear region of a biexponential scale is where these transforms earn
    their keep, and it is also where every decade between -10¹ and 10¹ lands on
    top of the zero tick. Cytometry software thins the labels; the alternative
    is a black smudge where the most important part of the axis should be.

    Walks outward from zero so the zero tick — the one that says where the
    linear region is — always survives, and the decades kept on each side are
    the ones nearest it that still fit.
    """
    if positions.size == 0:
        return positions, labels
    at_zero = [i for i, l in enumerate(labels) if str(l) == "0"]
    zero = at_zero[0] if at_zero else int(np.argmin(positions))
    keep = [zero]
    last = positions[zero]
    for i in range(zero + 1, positions.size):        # outward, positive side
        if positions[i] - last >= min_sep:
            keep.append(i)
            last = positions[i]
    last = positions[zero]
    for i in range(zero - 1, -1, -1):                # outward, negative side
        if last - positions[i] >= min_sep:
            keep.append(i)
            last = positions[i]
    idx = np.array(sorted(keep))
    return positions[idx], labels[idx]


def _style_linear_axis(ax, tr: Transform, which: str) -> None:
    """Evenly spaced ticks for a linear axis, labelled in data units.

    Scatter channels are linear, and a linear axis wants a linear locator. The
    ticks are chosen in DATA space (so they land on round numbers like 100 000
    rather than on round fractions of full scale) and then mapped forward into
    the scale space the events are drawn in.
    """
    from matplotlib.ticker import MaxNLocator

    lo, hi = (ax.get_xlim() if which == "x" else ax.get_ylim())
    d_lo = float(tr.inverse(np.array([lo]))[0])
    d_hi = float(tr.inverse(np.array([hi]))[0])
    values = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(d_lo, d_hi)
    values = values[(values >= min(d_lo, d_hi)) & (values <= max(d_lo, d_hi))]
    if values.size == 0:
        return
    pos = tr.apply(values)
    labels = [_si(v) for v in values]
    if which == "x":
        ax.set_xticks(pos, labels=labels)
    else:
        ax.set_yticks(pos, labels=labels)


def _si(v: float) -> str:
    """``150000`` -> ``150K``. Scatter runs to 262 144 and five full-width
    numbers along an axis do not fit."""
    a = abs(v)
    if a >= 1000:
        s = f"{v / 1000:.10g}"
        return f"{s}K"
    return f"{v:.10g}"


def _tick_label(raw: str) -> str:
    """``10^3`` -> ``10³``. Superscripts because a biexponential axis has a lot
    of ticks and ``10^3`` is two characters wider at every one of them."""
    sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    s = str(raw)
    if "^" in s:
        base, exp = s.split("^", 1)
        return base + exp.translate(sup)
    return s


def _limits(
    tr: Optional[Transform], values: np.ndarray, explicit: Optional[Tuple[float, float]]
) -> Tuple[float, float]:
    """Full scale when there is a scale; padded data range when there is not.

    Cytometry convention is to show the whole axis rather than zooming to the
    data: a population sitting at the top of scale and a population sitting in
    the middle are different findings, and auto-zoom hides which one you have.
    """
    if explicit is not None:
        return float(explicit[0]), float(explicit[1])
    if tr is not None:
        return 0.0, 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return lo - 0.5, hi + 0.5
    pad = 0.02 * (hi - lo)
    return lo - pad, hi + pad


def _density(x: np.ndarray, y: np.ndarray, bins: int) -> np.ndarray:
    """Per-event 2D-histogram density.

    A KDE would be smoother and is what a statistics package would reach for;
    it is also O(n²) or needs a tree, and cytometry files run to millions of
    events. Binned density is what every cytometry tool actually draws.
    """
    if x.size == 0:
        return np.zeros(0)
    H, xe, ye = np.histogram2d(x, y, bins=bins)
    ix = np.clip(np.searchsorted(xe, x, side="right") - 1, 0, H.shape[0] - 1)
    iy = np.clip(np.searchsorted(ye, y, side="right") - 1, 0, H.shape[1] - 1)
    return H[ix, iy]


def _subsample(n: int, max_events: Optional[int], random_state: int) -> Optional[np.ndarray]:
    """Indices to draw, or None for all of them.

    Rendering 5e6 markers produces a 40 MB figure that no notebook should carry.
    Density is computed on the drawn subset so the colours stay self-consistent;
    the number in the corner is always the full count, which is the one that
    matters.
    """
    if max_events is None or n <= max_events:
        return None
    rng = np.random.default_rng(random_state)
    return rng.choice(n, size=max_events, replace=False)


def _gate_list(
    gates: Optional[Iterable[Any]], strategy: Optional[GatingStrategy]
) -> List[Gate]:
    """Accept gate objects or, when a strategy is at hand, gate names."""
    if gates is None:
        return []
    out: List[Gate] = []
    for g in gates:
        if isinstance(g, Gate):
            out.append(g)
        elif isinstance(g, str):
            if strategy is None:
                raise ValueError(
                    f"gate {g!r} was given by name, but no strategy= was passed to "
                    "resolve it against"
                )
            known = strategy.gates
            if g not in known:
                raise KeyError(f"gate {g!r} is not in strategy {strategy.name!r}")
            out.append(known[g])
        else:
            raise TypeError(f"gates must be Gate objects or names, got {type(g).__name__}")
    return out


def _dim_index(gate: Gate, dim: str) -> Optional[int]:
    for i, d in enumerate(gate.dims):
        if d == dim:
            return i
    return None


def _gate_artists(
    gate: Gate,
    xdim: str,
    ydim: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> List[Tuple[str, np.ndarray]]:
    """Boundary geometry for ``gate`` in (x, y) plot order.

    Returns ``[(kind, coords), ...]`` where coords is an (n, 2) polyline. The
    dimension lookup is by NAME, not by position: a polygon stored as
    ``dims=('CD8','CD4')`` drawn on a CD4 × CD8 plot has to be transposed, and
    getting that wrong draws a mirrored gate that still looks plausible.
    """
    xi, yi = _dim_index(gate, xdim), _dim_index(gate, ydim)
    if xi is None and yi is None:
        return []

    if isinstance(gate, BooleanGate):
        # Nothing to draw: a boolean has no geometry of its own, only operands.
        return []

    if isinstance(gate, RectangleGate):
        lo_x, hi_x = gate.bounds[xi] if xi is not None else (None, None)
        lo_y, hi_y = gate.bounds[yi] if yi is not None else (None, None)
        x0 = xlim[0] if lo_x is None else lo_x
        x1 = xlim[1] if hi_x is None else hi_x
        y0 = ylim[0] if lo_y is None else lo_y
        y1 = ylim[1] if hi_y is None else hi_y
        return [("closed", np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))]

    if isinstance(gate, PolygonGate):
        if xi is None or yi is None:
            return []
        verts = np.column_stack([gate.vertices[:, xi], gate.vertices[:, yi]])
        return [("closed", verts)]

    if isinstance(gate, EllipsoidGate):
        if xi is None or yi is None:
            return []
        mu = gate.mean[[xi, yi]]
        cov = gate.covariance[np.ix_([xi, yi], [xi, yi])]
        return [("closed", _ellipse_points(mu, cov, gate.distance_square))]

    if isinstance(gate, QuadrantGate):
        out = []
        if xi is not None:
            dx = gate.dividers[xi]
            out.append(("open", np.array([[dx, ylim[0]], [dx, ylim[1]]], float)))
        if yi is not None:
            dy = gate.dividers[yi]
            out.append(("open", np.array([[xlim[0], dy], [xlim[1], dy]], float)))
        return out

    return []


def _ellipse_points(mu: np.ndarray, cov: np.ndarray, d2: float, n: int = 200) -> np.ndarray:
    """The contour ``(p-mu)' S^-1 (p-mu) = d2``.

    Gating-ML stores the covariance and the squared Mahalanobis distance rather
    than semi-axes and an angle, so the drawing has to do the eigendecomposition
    the format declined to do.
    """
    w, V = np.linalg.eigh(np.asarray(cov, dtype=float))
    w = np.clip(w, 1e-300, None)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    axes = np.sqrt(w * float(d2))
    return np.asarray(mu, dtype=float) + circle * axes @ V.T


def _gate_frequency(
    gate: Gate,
    result: Optional[GatingResult],
    frame: Mapping[str, np.ndarray],
) -> Optional[Tuple[int, float]]:
    """``(count, freq_of_parent)`` for the label, preferring the strategy's own
    numbers.

    Recomputing ``gate.mask(frame)`` here would ignore the parent restriction
    and quote a percentage of ALL events, which for anything below the first
    node is simply a different — and larger — number than the one the analysis
    reported. Only fall back to it when there is no result to read.
    """
    if result is not None and gate.name in result.masks:
        n = int(result.masks[gate.name].sum())
        parent = result.parents.get(gate.name, ROOT)
        denom = (
            int(result.masks[parent].sum())
            if parent in result.masks
            else result.n_events
        )
        return n, (n / denom if denom else float("nan"))
    if isinstance(gate, (BooleanGate, QuadrantGate)):
        return None
    try:
        m = gate.mask(frame)
    except (KeyError, TypeError):
        return None
    n = int(m.sum())
    total = int(m.size)
    return n, (n / total if total else float("nan"))


def _label_quadrants(
    ax,
    gate: QuadrantGate,
    xdim: str,
    ydim: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    result: Optional[GatingResult],
    color: str,
) -> None:
    """A percentage in each corner — the entire point of a quadrant plot.

    The gate's own ``mask`` is every event by construction, so labelling a
    quadrant gate the way the other gates are labelled prints "100.0%" and
    tells the reader nothing. What they came for is the four numbers, and
    cytometry has put them in the corners since the 1980s.
    """
    xi, yi = _dim_index(gate, xdim), _dim_index(gate, ydim)
    if xi is None or yi is None:
        return
    padx = 0.02 * (xlim[1] - xlim[0])
    pady = 0.02 * (ylim[1] - ylim[0])
    for i, pop in enumerate(gate.quadrant_names):
        x_hi = bool((i >> xi) & 1)
        y_hi = bool((i >> yi) & 1)
        text = pop
        if result is not None and pop in result.masks:
            n = int(result.masks[pop].sum())
            parent = result.parents.get(pop, ROOT)
            denom = (
                int(result.masks[parent].sum())
                if parent in result.masks else result.n_events
            )
            text = f"{pop}\n{n:,} ({n / denom:.1%})" if denom else f"{pop}\n{n:,}"
        ax.annotate(
            text,
            xy=(xlim[1] - padx if x_hi else xlim[0] + padx,
                ylim[1] - pady if y_hi else ylim[0] + pady),
            ha="right" if x_hi else "left",
            va="top" if y_hi else "bottom",
            fontsize=7, color=color,
        )


# ── the plots ───────────────────────────────────────────────────────────────


@register_function(
    # No "dot_plot": in cytometry that means exactly this figure, but
    # ov.pl._dotplot already owns the alias for the gene-expression dot plot,
    # and a duplicated alias routes a natural-language query to whichever
    # module registered last.
    aliases=["flow_biaxial", "biaxial", "双参数图", "散点门图", "gate_plot",
             "flow_dot_plot"],
    category="visualization",
    description=(
        "Biaxial density plot for cytometry: two channels on their display "
        "scales, coloured by event density, with gate boundaries drawn on top "
        "and each gate labelled with its count and frequency-of-parent. Axes "
        "are rendered in the gate's own transform (logicle/hyperlog/asinh/log) "
        "and tick-labelled back into data units."
    ),
    examples=[
        "ov.flow.biaxial(adata, 'FSC-A', 'SSC-A')",
        "ov.flow.biaxial(adata, 'CD4', 'CD8', strategy=gs, gates=['CD4 T'], result=res)",
        "ov.flow.biaxial(adata, 'CD3', 'CD19', strategy=gs, population='Live')",
    ],
    related=["flow.GatingStrategy", "flow.backgate", "flow.hierarchy"],
)
def biaxial(
    adata: Any,
    x: str,
    y: str,
    *,
    layer: Optional[str] = None,
    transforms: Optional[Mapping[str, Transform]] = None,
    strategy: Optional[GatingStrategy] = None,
    gates: Optional[Iterable[Any]] = None,
    result: Optional[GatingResult] = None,
    population: Optional[str] = None,
    bins: int = 256,
    s: float = 2.0,
    cmap: str = "viridis",
    gate_color: str = "#d62728",
    gate_lw: float = 1.4,
    max_events: Optional[int] = 200_000,
    random_state: int = 0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_freq: bool = True,
    title: Optional[str] = None,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (4.4, 4.0),
):
    """Two channels, density-coloured, with gates drawn on their own scale.

    Arguments
    ---------
    x, y
        Channel or marker names. Resolved exactly as a gate resolves them, so
        a detector name works wherever the marker name does.
    transforms
        ``{dim: Transform}``. Usually unnecessary: the scaling is read off the
        gates being drawn, or off ``strategy``, which is what keeps the picture
        and the mask on the same axis.
    gates
        ``Gate`` objects, or names when ``strategy`` is given. Gates that touch
        neither ``x`` nor ``y`` are silently skipped — a strategy can be passed
        whole without filtering it by hand. A gate on ONE of the two axes is
        drawn as a band across the other, which is what a one-sided threshold
        is.
    population
        Draw only the events in this population, i.e. the parent gate's members.
        This is what makes a hierarchy readable: the CD4 × CD8 plot everyone
        publishes is drawn on CD3+ events, not on all of them. Requires
        ``result``.
    result
        The :class:`GatingResult` those masks came from. Also supplies the
        percentages, which is why they agree with ``result.stats()``.

    Returns the matplotlib axes.
    """
    import matplotlib.pyplot as plt

    gate_objs = _gate_list(gates, strategy)
    frame = _frame(adata, layer)
    for dim in (x, y):
        if dim not in frame:
            raise KeyError(
                f"{dim!r} is not a channel, marker or numeric obs column "
                f"(have: {sorted(frame)[:12]}...)"
            )

    keep = None
    if population is not None:
        if result is None:
            raise ValueError("population= needs the result= it was computed in")
        if population not in result.masks:
            raise KeyError(
                f"population {population!r} is not in this result "
                f"(have: {sorted(result.masks)[:12]}...)"
            )
        keep = np.flatnonzero(result.masks[population])

    trx = _resolve_transform(x, transforms, strategy, gate_objs)
    try_ = _resolve_transform(y, transforms, strategy, gate_objs)

    xs_all = _scaled(frame[x], trx)
    ys_all = _scaled(frame[y], try_)
    if keep is not None:
        xs_all, ys_all = xs_all[keep], ys_all[keep]
    n_total = xs_all.size

    idx = _subsample(n_total, max_events, random_state)
    xs = xs_all if idx is None else xs_all[idx]
    ys = ys_all if idx is None else ys_all[idx]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    xl = _limits(trx, xs_all, xlim)
    yl = _limits(try_, ys_all, ylim)

    finite = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[finite], ys[finite]
    if xs.size:
        d = _density(xs, ys, bins)
        order = np.argsort(d)          # rare events last, so they stay visible
        ax.scatter(
            xs[order], ys[order], c=d[order], s=s, cmap=cmap,
            edgecolor="none", rasterized=True,
        )

    for g in gate_objs:
        artists = _gate_artists(g, x, y, xl, yl)
        if not artists:
            continue
        for kind, coords in artists:
            if kind == "closed":
                loop = np.vstack([coords, coords[:1]])
                ax.plot(loop[:, 0], loop[:, 1], color=gate_color, lw=gate_lw)
            else:
                ax.plot(coords[:, 0], coords[:, 1], color=gate_color,
                        lw=gate_lw, ls="--")
        if not show_freq:
            continue
        if isinstance(g, QuadrantGate):
            _label_quadrants(ax, g, x, y, xl, yl, result, gate_color)
            continue
        freq = _gate_frequency(g, result, frame)
        label = g.name if freq is None else f"{g.name}\n{freq[0]:,} ({freq[1]:.1%})"
        anchor = artists[0][1]
        # Clamp inside the axes: a gate whose top edge sits at top-of-scale
        # would otherwise put its label on the title.
        top = float(anchor[:, 1].max())
        inside = top < yl[1] - 0.06 * (yl[1] - yl[0])
        ax.annotate(
            label,
            xy=(float(anchor[:, 0].mean()), top if inside else yl[1]),
            xytext=(0, 4 if inside else -4), textcoords="offset points",
            ha="center", va="bottom" if inside else "top",
            fontsize=7, color=gate_color,
        )

    ax.set_xlim(xl)
    ax.set_ylim(yl)
    _style_axis(ax, trx, "x", x)
    _style_axis(ax, try_, "y", y)
    shown = "" if idx is None else f" ({xs.size:,} drawn)"
    ax.set_title(
        title if title is not None
        else f"{population or 'all events'} — {n_total:,} events{shown}",
        fontsize=9,
    )
    return ax


@register_function(
    aliases=["flow_histogram", "flow_hist", "单参数直方图", "荧光直方图"],
    category="visualization",
    description=(
        "Single-channel histogram on its display scale, optionally split by a "
        "gate or overlaid per population — the one-parameter view used for "
        "threshold gates and for comparing a stain against its control."
    ),
    examples=[
        "ov.flow.histogram(adata, 'CD3', strategy=gs, gates=['CD3+'])",
        "ov.flow.histogram(adata, 'Viability', result=res, populations=['Cells','Live'])",
    ],
    related=["flow.biaxial", "flow.GatingStrategy"],
)
def histogram(
    adata: Any,
    x: str,
    *,
    layer: Optional[str] = None,
    transforms: Optional[Mapping[str, Transform]] = None,
    strategy: Optional[GatingStrategy] = None,
    gates: Optional[Iterable[Any]] = None,
    result: Optional[GatingResult] = None,
    populations: Optional[Sequence[str]] = None,
    bins: int = 256,
    fill: bool = True,
    palette: Optional[Sequence[str]] = None,
    gate_color: str = "#d62728",
    xlim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (4.4, 2.6),
):
    """One channel as a density curve, with thresholds marked.

    ``populations`` overlays several masks from ``result`` on one axis, each
    normalised to its own peak — the standard way to show that a marker
    separates in the gated subset but not in the parent. Normalising to area
    instead would let a population that is 2% of its parent vanish, which is the
    comparison the plot exists to make.
    """
    import matplotlib.pyplot as plt

    gate_objs = _gate_list(gates, strategy)
    frame = _frame(adata, layer)
    if x not in frame:
        raise KeyError(f"{x!r} is not a channel, marker or numeric obs column")

    tr = _resolve_transform(x, transforms, strategy, gate_objs)
    values = _scaled(frame[x], tr)
    xl = _limits(tr, values, xlim)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    series: List[Tuple[str, np.ndarray]] = []
    if populations:
        if result is None:
            raise ValueError("populations= needs the result= they were computed in")
        for p in populations:
            if p not in result.masks:
                raise KeyError(f"population {p!r} is not in this result")
            series.append((p, values[result.masks[p]]))
    else:
        series.append(("all events", values))

    colors = list(palette) if palette else ["#4c72b0", "#dd8452", "#55a868",
                                            "#c44e52", "#8172b3", "#937860"]
    edges = np.linspace(xl[0], xl[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for i, (label, vals) in enumerate(series):
        vals = vals[np.isfinite(vals)]
        counts, _ = np.histogram(vals, bins=edges)
        peak = counts.max()
        h = counts / peak if peak else counts.astype(float)
        c = colors[i % len(colors)]
        ax.plot(centers, h, color=c, lw=1.2, label=f"{label} ({vals.size:,})")
        if fill:
            ax.fill_between(centers, h, color=c, alpha=0.25)

    for g in gate_objs:
        xi = _dim_index(g, x)
        if xi is None:
            continue
        if isinstance(g, RectangleGate):
            for b in g.bounds[xi]:
                if b is not None:
                    ax.axvline(float(b), color=gate_color, lw=1.2, ls="--")
        elif isinstance(g, QuadrantGate):
            ax.axvline(float(g.dividers[xi]), color=gate_color, lw=1.2, ls="--")

    ax.set_xlim(xl)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("normalised to mode")
    _style_axis(ax, tr, "x", x)
    if len(series) > 1 or populations:
        ax.legend(fontsize=7, frameon=False)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


@register_function(
    aliases=["flow_backgate", "backgate", "回溯门", "反向门控"],
    category="visualization",
    description=(
        "Back-gating: draw a downstream population in colour on top of its "
        "ancestor's events in grey, to show WHERE a gated subset actually sits "
        "in a plot it was not gated on."
    ),
    examples=[
        "ov.flow.backgate(adata, 'FSC-A', 'SSC-A', result=res, population='CD4 T')",
    ],
    related=["flow.biaxial", "flow.hierarchy"],
)
def backgate(
    adata: Any,
    x: str,
    y: str,
    *,
    result: GatingResult,
    population: str,
    parent: Optional[str] = None,
    layer: Optional[str] = None,
    transforms: Optional[Mapping[str, Transform]] = None,
    strategy: Optional[GatingStrategy] = None,
    color: str = "#d62728",
    background_color: str = "#cccccc",
    s: float = 2.0,
    max_events: Optional[int] = 200_000,
    random_state: int = 0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (4.4, 4.0),
):
    """Where does this population live on axes it was never gated on?

    The question that catches a bad gate. A "CD4 T" population scattered across
    the debris corner of FSC × SSC is not a T-cell population, however clean the
    CD4 × CD8 plot looked, and no statistics table will tell you that.

    ``parent`` defaults to the population's own parent in the result; pass
    ``'root'`` to show it against every event.
    """
    import matplotlib.pyplot as plt

    if population not in result.masks:
        raise KeyError(
            f"population {population!r} is not in this result "
            f"(have: {sorted(result.masks)[:12]}...)"
        )
    parent = parent if parent is not None else result.parents.get(population, ROOT)
    if parent != ROOT and parent not in result.masks:
        raise KeyError(f"parent population {parent!r} is not in this result")

    frame = _frame(adata, layer)
    for dim in (x, y):
        if dim not in frame:
            raise KeyError(f"{dim!r} is not a channel, marker or numeric obs column")

    trx = _resolve_transform(x, transforms, strategy, [])
    try_ = _resolve_transform(y, transforms, strategy, [])
    xs = _scaled(frame[x], trx)
    ys = _scaled(frame[y], try_)

    bg = np.ones(xs.size, bool) if parent == ROOT else result.masks[parent]
    fg = result.masks[population]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    xl = _limits(trx, xs[bg], xlim)
    yl = _limits(try_, ys[bg], ylim)

    # The background layer excludes the highlighted events so they are not
    # drawn twice — but it is still labelled with the PARENT's own count.
    # Labelling it with the count of what was drawn would put "CD3+ (9,000)"
    # next to a population that has 23,001 events, which is a caption that
    # contradicts itself.
    n_parent = int(bg.sum())
    for mask, c, z, lab in ((bg & ~fg, background_color, 1, f"{parent} ({n_parent:,})"),
                            (fg, color, 2, f"{population} ({int(fg.sum()):,})")):
        i = np.flatnonzero(mask)
        sub = _subsample(i.size, max_events, random_state)
        if sub is not None:
            i = i[sub]
        ax.scatter(xs[i], ys[i], s=s, c=c, edgecolor="none", zorder=z,
                   rasterized=True, label=lab)

    ax.set_xlim(xl)
    ax.set_ylim(yl)
    _style_axis(ax, trx, "x", x)
    _style_axis(ax, try_, "y", y)
    ax.legend(fontsize=7, frameon=False, markerscale=4)
    frac = int(fg.sum()) / n_parent if n_parent else float("nan")
    ax.set_title(
        title if title is not None
        else f"{population} back-gated on {parent} ({frac:.1%})",
        fontsize=9,
    )
    return ax


@register_function(
    aliases=["flow_hierarchy", "gating_tree", "门控树", "门控层级图", "gating_hierarchy"],
    category="visualization",
    description=(
        "Draw the gating hierarchy as a tree, each node labelled with its "
        "count and frequency-of-parent, and populations below a size threshold "
        "flagged. The visual audit of a gating strategy."
    ),
    examples=[
        "ov.flow.hierarchy(gs, result=res)",
        "ov.flow.hierarchy(gs)",
    ],
    related=["flow.GatingStrategy", "flow.biaxial"],
)
def hierarchy(
    strategy: GatingStrategy,
    *,
    result: Optional[GatingResult] = None,
    min_events: int = 100,
    node_color: str = "#4c72b0",
    low_n_color: str = "#c44e52",
    fontsize: float = 8.0,
    title: Optional[str] = None,
    ax: Optional[Any] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """The strategy as a picture, with the numbers on it.

    :meth:`GatingStrategy.tree` prints the same shape as text and is better for
    a terminal. This exists because the thing a reader checks is where the
    events went, and a percentage next to a name in a tree answers "did the
    hierarchy collapse somewhere" in one glance — a population that keeps 3% of
    its parent when you expected 60% is visible here and invisible in a sorted
    statistics table.

    ``low_n`` nodes — fewer than ``min_events`` events — are drawn in a warning
    colour, matching :meth:`GatingResult.stats`, because a frequency computed
    off 30 events should not be read as a measurement.
    """
    import matplotlib.pyplot as plt

    # Leaf-order layout: y is assigned by DFS arrival, x by depth. Nothing here
    # needs a graph library, and adding one for a tree of twelve nodes would be
    # a dependency for the sake of a dependency.
    rows: List[Tuple[str, int, int]] = []      # (name, depth, y)

    def walk(node: str, depth: int) -> None:
        for child in strategy.children_of(node):
            rows.append((child, depth + 1, len(rows)))
            walk(child, depth + 1)

    rows.append((ROOT, 0, 0))
    walk(ROOT, 0)
    if len(rows) == 1:
        raise ValueError(f"strategy {strategy.name!r} has no gates to draw")

    y_of = {name: i for i, (name, _, i) in enumerate(rows)}
    depth_of = {name: d for name, d, _ in rows}
    n = len(rows)

    if ax is None:
        max_depth = max(depth_of.values())
        _, ax = plt.subplots(
            figsize=figsize or (3.2 + 1.9 * max_depth, 0.42 * n + 0.6)
        )

    stats = result.stats(min_events=min_events).set_index("population") if result else None

    for name, depth, y in rows:
        parent = ROOT if name == ROOT else strategy.parent_of(name)
        if name != ROOT and parent in y_of:
            # Elbow: down the parent's column, then across. Straight diagonals
            # cross each other as soon as a node has three children.
            ax.plot([depth - 1 + 0.08, depth - 1 + 0.08],
                    [-y_of[parent], -y], color="#999999", lw=0.9, zorder=1)
            ax.plot([depth - 1 + 0.08, depth], [-y, -y],
                    color="#999999", lw=0.9, zorder=1)

        if name == ROOT:
            label = f"root ({result.n_events:,})" if result else "root"
            color = "#444444"
        elif stats is not None and name in stats.index:
            row = stats.loc[name]
            label = f"{name}\n{int(row['count']):,}  ({row['freq_parent']:.1%} of parent)"
            color = low_n_color if bool(row["low_n"]) else node_color
        else:
            label = name
            color = node_color

        ax.annotate(
            label, xy=(depth, -y), fontsize=fontsize, color=color,
            va="center", ha="left", zorder=3,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=color, lw=0.9),
        )

    ax.set_xlim(-0.25, max(depth_of.values()) + 1.9)
    ax.set_ylim(-n + 0.4, 0.6)
    ax.set_axis_off()
    ax.set_title(title if title is not None else f"{strategy.name} — gating hierarchy",
                 fontsize=9, loc="left")
    return ax


@register_function(
    aliases=["flow_spillover_heatmap", "spillover_heatmap", "补偿矩阵热图", "溢出矩阵"],
    category="visualization",
    description=(
        "Heatmap of a spillover (or compensation) matrix, diagonal excluded "
        "from the colour scale so the off-diagonal spillover is actually "
        "visible. Reads uns['flow']['spillover'] or uns['fcs']['spillover'] "
        "when given an AnnData."
    ),
    examples=[
        "ov.flow.spillover_heatmap(adata)",
        "ov.flow.spillover_heatmap(spill_df, annotate=True)",
    ],
    related=["flow.compensate", "flow.spillover_spreading_matrix"],
)
def spillover_heatmap(
    source: Any,
    *,
    annotate: bool = True,
    cmap: str = "RdPu",
    vmax: Optional[float] = None,
    fmt: str = "{:.2f}",
    title: Optional[str] = None,
    ax: Optional[Any] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Which detector bleeds into which.

    The diagonal is 1.0 by construction and is left out of the colour
    normalisation: include it and every off-diagonal value — the entire content
    of the matrix — is squeezed into the bottom few percent of the colour map
    and the plot shows a white square with a dark diagonal.
    """
    import matplotlib.pyplot as plt

    if isinstance(source, pd.DataFrame):
        mat = source
    else:
        flow_uns = getattr(source, "uns", {}).get("flow", {}) or {}
        fcs_uns = getattr(source, "uns", {}).get("fcs", {}) or {}
        mat = flow_uns.get("spillover")
        if mat is None:
            mat = fcs_uns.get("spillover")
        if mat is None:
            raise KeyError(
                "no spillover matrix found — expected uns['flow']['spillover'] "
                "(written by ov.flow.compensate) or uns['fcs']['spillover'] "
                "(read from the file by ov.io.read_fcs)"
            )
        if not isinstance(mat, pd.DataFrame):
            mat = pd.DataFrame(np.asarray(mat, dtype=float))

    values = np.asarray(mat, dtype=float)
    k = values.shape[0]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (0.55 * k + 2.2, 0.5 * k + 1.8))

    off = values[~np.eye(k, dtype=bool)] if k > 1 else values.ravel()
    hi = vmax if vmax is not None else (float(np.abs(off).max()) if off.size else 1.0)
    im = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=max(hi, 1e-6))

    ax.set_xticks(range(values.shape[1]), labels=list(mat.columns),
                  rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(k), labels=list(mat.index), fontsize=7)
    ax.set_xlabel("into detector")
    ax.set_ylabel("from fluorochrome")
    if annotate:
        for i in range(k):
            for j in range(values.shape[1]):
                v = values[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6,
                        color="#888888" if i == j else
                        ("white" if v > 0.6 * hi else "#222222"))
    ax.figure.colorbar(im, ax=ax, fraction=0.046, shrink=0.8,
                       label="spillover (diagonal excluded from scale)")
    ax.set_title(title if title is not None else "Spillover matrix", fontsize=9)
    return ax


@register_function(
    aliases=["flow_flowsom_heatmap", "flowsom_heatmap", "元聚类热图", "flowsom热图"],
    category="visualization",
    description=(
        "Metacluster × marker heatmap of a FlowSOM result, with per-cluster "
        "event counts — the phenotype table that says what ov.flow.flowsom "
        "actually found."
    ),
    examples=[
        "ov.flow.flowsom_heatmap(adata, markers=['CD3','CD4','CD8','CD19'])",
    ],
    related=["flow.flowsom", "flow.biaxial"],
)
def flowsom_heatmap(
    adata: Any,
    *,
    key: str = "flowsom",
    markers: Optional[Sequence[str]] = None,
    layer: Optional[str] = None,
    scale: str = "marker",
    cmap: str = "viridis",
    show_counts: bool = True,
    title: Optional[str] = None,
    ax: Optional[Any] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """What each metacluster is, as a phenotype.

    ``scale='marker'`` z-scores each marker across clusters, which is what makes
    the plot readable: raw MFI is dominated by the brightest fluorochrome in the
    panel, and every cluster then looks like "high CD45, everything else zero".
    Pass ``scale='none'`` when the absolute level is the point.

    Cluster sizes are drawn beside the heatmap because a beautifully distinct
    metacluster holding 0.2% of events is a rounding error, and the heatmap
    alone gives it the same visual weight as a real population.
    """
    import matplotlib.pyplot as plt

    if key not in adata.obs:
        raise KeyError(
            f"obs[{key!r}] not found — run ov.flow.flowsom(adata, key_added={key!r}) first"
        )
    if markers is None:
        info = (adata.uns.get("flow", {}) or {}).get("flowsom", {}) or {}
        markers = info.get("markers")
    if markers is None:
        markers = [str(v) for v in adata.var.index]
    markers = [str(m) for m in markers]

    frame = _frame(adata, layer)
    missing = [m for m in markers if m not in frame]
    if missing:
        raise KeyError(f"markers not found in the data: {missing}")

    labels = pd.Series(adata.obs[key].astype(str).to_numpy(), name=key)
    order = sorted(labels.unique(), key=lambda v: (len(v), v))
    mat = np.column_stack([frame[m] for m in markers])
    df = pd.DataFrame(mat, columns=markers)
    means = df.groupby(labels.to_numpy()).mean().reindex(order)
    counts = labels.value_counts().reindex(order).fillna(0).astype(int)

    values = means.to_numpy(dtype=float)
    if scale == "marker":
        mu = np.nanmean(values, axis=0)
        sd = np.nanstd(values, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        values = (values - mu) / sd
        cbar_label = "z-score across metaclusters"
    elif scale == "none":
        cbar_label = "mean intensity"
    else:
        raise ValueError("scale must be 'marker' or 'none'")

    n_c, n_m = values.shape
    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (0.52 * n_m + 3.0, 0.36 * n_c + 1.8))
    im = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(n_m), labels=markers, rotation=45, ha="right", fontsize=7)
    ylabels = (
        [f"{c}  (n={counts[c]:,}, {counts[c]/len(labels):.1%})" for c in order]
        if show_counts else list(order)
    )
    ax.set_yticks(range(n_c), labels=ylabels, fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.030, shrink=0.8, label=cbar_label)
    ax.set_title(title if title is not None else f"FlowSOM metaclusters ({key})",
                 fontsize=9)
    return ax


@register_function(
    aliases=["flow_plot_strategy", "plot_strategy", "门控策略图", "分层门控图",
             "gating_sequence", "plot_gates"],
    category="visualization",
    description=(
        "Draw a whole gating strategy as a row of panels — one per gate, each "
        "on its own parent's events, in the transform that gate carries. The "
        "figure a cytometry paper puts in its supplement, from the strategy "
        "object rather than assembled by hand."
    ),
    examples=[
        "ov.flow.plot_strategy(adata, gs, res)",
        "ov.flow.plot_strategy(adata, gs, res, ncols=3)",
    ],
    related=["flow.GatingStrategy", "flow.biaxial", "flow.hierarchy"],
)
def plot_strategy(
    adata: Any,
    strategy: GatingStrategy,
    result: GatingResult,
    *,
    layer: Optional[str] = None,
    transforms: Optional[Mapping[str, Transform]] = None,
    ncols: int = 4,
    panel_size: Tuple[float, float] = (4.3, 4.0),
    max_events: Optional[int] = 40_000,
    against: Optional[str] = None,
    **kwargs: Any,
):
    """Every gate in the strategy, each drawn on the events its parent kept.

    This is the standard cytometry figure — "here is how I got to this
    population" — and building it by hand means keeping a row of subplots, a
    list of gates, and a list of parent names in sync. They drift. Here the
    strategy is the single source of all three.

    Panel choice per gate: a two-dimensional gate gets a biaxial plot of its own
    dimensions; a one-dimensional gate gets a histogram, unless ``against`` names
    a channel to spread it over. Boolean gates have no geometry and are skipped.

    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    gates = [(name, strategy.gates[name]) for name in strategy.gates]
    drawable = [(n, g) for n, g in gates if not isinstance(g, BooleanGate)]
    if not drawable:
        raise ValueError(
            f"strategy {strategy.name!r} has no gate with geometry to draw"
        )

    ncols = max(1, min(ncols, len(drawable)))
    nrows = math.ceil(len(drawable) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
    )
    flat = [ax for row in axes for ax in row]

    for ax, (name, gate) in zip(flat, drawable):
        parent = strategy.parent_of(name)
        population = None if parent == ROOT else parent
        if len(gate.dims) >= 2:
            biaxial(adata, gate.dims[0], gate.dims[1], layer=layer,
                    transforms=transforms, strategy=strategy, gates=[gate],
                    result=result, population=population, ax=ax,
                    max_events=max_events, **kwargs)
        elif against is not None:
            biaxial(adata, gate.dims[0], against, layer=layer,
                    transforms=transforms, strategy=strategy, gates=[gate],
                    result=result, population=population, ax=ax,
                    max_events=max_events, **kwargs)
        else:
            histogram(adata, gate.dims[0], layer=layer, transforms=transforms,
                      strategy=strategy, gates=[gate], result=result,
                      populations=[parent] if population else None, ax=ax)
            ax.set_title(f"{name} — of {parent}", fontsize=9)

    for ax in flat[len(drawable):]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig


@register_function(
    aliases=["flow_plot_batch", "plot_batch", "批次频率图", "样本频率对比",
             "batch_frequencies"],
    category="visualization",
    description=(
        "Grouped bar chart of population frequencies across samples, drawn "
        "from the tidy table ov.flow.batch_stats returns. Shows the SPREAD "
        "between samples, which a pooled frequency hides."
    ),
    examples=[
        "stats = ov.flow.batch_stats(adata, gs); ov.flow.plot_batch(stats)",
        "ov.flow.plot_batch(stats, populations=['CD4 T', 'CD8 T'])",
    ],
    related=["flow.batch_stats", "flow.GatingStrategy"],
)
def plot_batch(
    stats: "pd.DataFrame",
    *,
    populations: Optional[Sequence[str]] = None,
    groupby: str = "sample",
    value: str = "freq_parent",
    hue: Optional[str] = None,
    palette: Optional[Sequence[str]] = None,
    mark_low_n: bool = True,
    title: Optional[str] = None,
    ax: Optional[Any] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """The spread across samples, as bars.

    Takes the long-form table from :func:`~omicverse.flow.batch_stats` rather
    than an AnnData, so the numbers on the chart are provably the numbers in the
    table — there is no second computation to disagree with the first.

    ``mark_low_n`` hatches any bar whose population fell below the event
    threshold in that sample. A 90% frequency measured on 12 events and one
    measured on 12,000 are the same height otherwise, and only one of them means
    anything.
    """
    import matplotlib.pyplot as plt

    for column in (groupby, "population", value):
        if column not in stats.columns:
            raise KeyError(
                f"{column!r} is not in this table — expected the output of "
                f"ov.flow.batch_stats (have: {list(stats.columns)})"
            )
    df = stats if populations is None else stats[stats["population"].isin(populations)]
    if df.empty:
        raise ValueError(f"no rows left after filtering to {populations}")

    order = list(populations) if populations is not None else list(
        dict.fromkeys(df["population"])
    )
    samples = list(dict.fromkeys(df[groupby]))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (1.15 * len(order) + 2.5, 3.4))

    colors = list(palette) if palette else ["#4c72b0", "#dd8452", "#55a868",
                                            "#c44e52", "#8172b3", "#937860"]
    width = 0.8 / max(len(samples), 1)
    x = np.arange(len(order))
    for i, sample in enumerate(samples):
        sub = df[df[groupby] == sample].set_index("population")
        heights = [float(sub[value].get(p, np.nan)) for p in order]
        low = [bool(sub["low_n"].get(p, False)) if "low_n" in sub else False
               for p in order]
        bars = ax.bar(x + (i - (len(samples) - 1) / 2) * width, heights, width,
                      label=str(sample), color=colors[i % len(colors)])
        if mark_low_n:
            for bar, flag in zip(bars, low):
                if flag:
                    bar.set_hatch("///")
                    bar.set_edgecolor("white")

    ax.set_xticks(x, labels=order, rotation=25, ha="right")
    ax.set_ylabel(value.replace("_", " "))
    ax.legend(fontsize=7, frameon=False, ncol=min(len(samples), 3))
    hatched = mark_low_n and bool(df.get("low_n", pd.Series(dtype=bool)).any())
    ax.set_title(
        title if title is not None else
        ("population frequencies by sample"
         + ("  (hatched = too few events to trust)" if hatched else "")),
        fontsize=9,
    )
    return ax
