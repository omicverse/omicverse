"""Detect overlapping tick labels and separate them.

Tick labels are the one piece of text on a plot whose crowding depends entirely
on the physical size the axes ends up at. "Fri Sat Sun Thur" is comfortable on
a 90 mm panel and a smear on a 40 mm one, and nothing in the plotting call
knows which it will be — a layout engine may resize the axes afterwards.

:func:`declutter_ticks` measures the drawn label boxes and, when they collide,
works through a ladder of remedies: rotate, deal into two rows, or slide apart
along the axis with a leader line back to each tick — the same bargain
:func:`~omicverse.pl.adjust_text` strikes for point labels, where the leader is
what licenses moving the text away from what it names. Font sizes are never
changed.

Because the answer depends on the final geometry, the check is registered as a
draw-time callback by default: it re-evaluates on every draw and is idempotent,
so it stays correct after a figure is resized or its axes repositioned.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .._registry import register_function

_PAD_POINTS = 1.5
_ROTATIONS = (0.0, 30.0, 45.0, 60.0, 90.0)
_FLAG = "_ov_declutter_running"


def _all_labels(ax, which: str) -> list:
    """Every tick label, hidden ones included.

    ``get_xticklabels`` returns only the labels currently visible, so a pass
    that hid some could not find them again — a panel thinned while small would
    stay thinned after being enlarged. The tick objects keep them.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    return [tick.label1 for tick in axis.get_major_ticks()
            if tick.label1.get_text().strip()]


def _visible_labels(ax, which: str) -> list:
    return [t for t in _all_labels(ax, which) if t.get_visible()]


def _boxes(labels: Sequence[Any], renderer) -> list:
    out = []
    for text in labels:
        try:
            out.append(text.get_window_extent(renderer))
        except (RuntimeError, ValueError):
            pass
    return out


def _collides(labels, renderer, which: str) -> bool:
    """Do consecutive labels overlap along the axis they run on?"""
    boxes = _boxes(labels, renderer)
    if len(boxes) < 2:
        return False
    if which == "x":
        boxes = sorted(boxes, key=lambda b: b.x0)
        return any(boxes[i].x1 + _PAD_POINTS > boxes[i + 1].x0
                   for i in range(len(boxes) - 1))
    boxes = sorted(boxes, key=lambda b: b.y0)
    return any(boxes[i].y1 + _PAD_POINTS > boxes[i + 1].y0
               for i in range(len(boxes) - 1))


def _apply_rotation(labels, angle: float, which: str) -> None:
    for text in labels:
        text.set_rotation(angle)
        if which == "x" and angle:
            text.set_horizontalalignment("right")
            text.set_rotation_mode("anchor")
        elif which == "x":
            text.set_horizontalalignment("center")
            text.set_rotation_mode(None)


def _stagger(labels, which: str, amount_points: float) -> None:
    """Deal alternate labels into a second row, further from the axis."""
    import matplotlib.transforms as mtransforms

    for index, text in enumerate(labels):
        if index % 2 == 0:
            continue
        base = text.get_transform()
        # Strip a previous stagger so repeated draws do not accumulate.
        base = getattr(text, "_ov_declutter_base_transform", base)
        text._ov_declutter_base_transform = base
        dx, dy = (0.0, -amount_points) if which == "x" else (-amount_points, 0.0)
        offset = mtransforms.ScaledTranslation(
            dx / 72.0, dy / 72.0, text.figure.dpi_scale_trans)
        text.set_transform(base + offset)


def _unstagger(labels) -> None:
    for text in labels:
        base = getattr(text, "_ov_declutter_base_transform", None)
        if base is not None:
            text.set_transform(base)


def _separate_1d(wanted, extents, lo, hi, pad):
    """Slide items along a line so they stop overlapping, order preserved.

    Overlapping neighbours are merged into a block, the block is centred on the
    mean of the positions its members wanted, and its members are laid out
    touching inside it — repeated until no block overlaps its neighbour. The
    result is the closest arrangement to what was asked for that also fits, and
    it never reorders, which for tick labels would be a lie about the data.

    Blocks are clamped to ``[lo, hi]``; if the total width exceeds that span
    there is no overlap-free answer and the caller is told so.
    """
    need = [extents[i] + pad for i in range(len(wanted))]
    if sum(need) > (hi - lo):
        return None

    def size(block):
        return sum(need[i] for i in block)

    def target(block):
        return sum(wanted[i] for i in block) / len(block)

    def start_of(block):
        return min(max(target(block) - size(block) / 2.0, lo),
                   hi - size(block))

    blocks = [[i] for i in sorted(range(len(wanted)), key=lambda i: wanted[i])]
    while True:
        starts = [start_of(b) for b in blocks]
        for k in range(len(blocks) - 1):
            if starts[k] + size(blocks[k]) > starts[k + 1] + 1e-9:
                blocks[k:k + 2] = [blocks[k] + blocks[k + 1]]
                break
        else:
            break

    centres = {}
    for block, start in zip(blocks, starts):
        cursor = start
        for i in block:
            centres[i] = cursor + need[i] / 2.0
            cursor += need[i]
    return [centres[i] for i in range(len(wanted))]


def _spread(ax, labels, renderer, which: str, leader_color, leader_linewidth):
    """Slide the labels apart along the axis and draw a leader to each tick.

    This is the tick-label form of what :func:`~omicverse.pl.adjust_text` does.
    A displaced tick label would point at the wrong value on its own — the
    leader line is what makes moving it honest, exactly as it is for repelled
    point labels.

    Returns True if the collision was cleared.
    """
    import matplotlib.transforms as mtransforms
    from matplotlib.lines import Line2D

    boxes = _boxes(labels, renderer)
    if len(boxes) != len(labels):
        return False
    axes_box = ax.get_window_extent()
    figure_box = ax.figure.bbox
    # Tick labels may reach a little past the axes — the first and last always
    # do — so the room available is the axes span plus a margin of a quarter of
    # its length on each side, never outside the figure.
    if which == "x":
        wanted = [(b.x0 + b.x1) / 2 for b in boxes]
        extents = [b.width for b in boxes]
        slack = 0.25 * axes_box.width
        lo = max(figure_box.x0, axes_box.x0 - slack)
        hi = min(figure_box.x1, axes_box.x1 + slack)
    else:
        wanted = [(b.y0 + b.y1) / 2 for b in boxes]
        extents = [b.height for b in boxes]
        slack = 0.25 * axes_box.height
        lo = max(figure_box.y0, axes_box.y0 - slack)
        hi = min(figure_box.y1, axes_box.y1 + slack)

    # Twice the collision pad: each slot carries one pad, so the gap between
    # neighbours comes out at the slot padding — and `_collides` wants the gap
    # to *exceed* `_PAD_POINTS`, which an exactly-equal gap does not.
    placed = _separate_1d(wanted, extents, lo, hi, _PAD_POINTS * 2)
    if placed is None:                    # cannot fit at this size at all
        return False

    _clear_leaders(ax)
    leaders = []
    for text, box, want, got in zip(labels, boxes, wanted, placed):
        shift = got - want
        base = getattr(text, "_ov_declutter_base_transform", text.get_transform())
        text._ov_declutter_base_transform = base
        # `shift` is in display pixels, and the label's transform lands in
        # display space, so the offset composes directly — no unit conversion.
        dx, dy = (shift, 0.0) if which == "x" else (0.0, shift)
        text.set_transform(base + mtransforms.Affine2D().translate(dx, dy))

        # Leader: from the tick on the axis edge to the label's new position.
        if abs(shift) < 0.5:
            continue
        if which == "x":
            x_from, y_from = want, axes_box.y0
            x_to, y_to = got, box.y1
        else:
            x_from, y_from = axes_box.x0, want
            x_to, y_to = box.x1, got
        line = Line2D([x_from, x_to], [y_from, y_to],
                      transform=mtransforms.IdentityTransform(),
                      color=leader_color,
                      linewidth=leader_linewidth, zorder=1, clip_on=False)
        ax.figure.add_artist(line)
        leaders.append(line)
    ax._ov_declutter_leaders = leaders

    if _collides(labels, renderer, which):
        _clear_leaders(ax)
        _unstagger(labels)
        return False
    return True


def _clear_leaders(ax) -> None:
    for line in getattr(ax, "_ov_declutter_leaders", []):
        try:
            line.remove()
        except (ValueError, NotImplementedError):
            pass
    ax._ov_declutter_leaders = []


def _thin(labels, step: int) -> None:
    for index, text in enumerate(labels):
        text.set_visible(index % step == 0)


@register_function(
    aliases=["declutter_ticks", "刻度标签避让", "tick_declutter",
             "fix_tick_overlap", "刻度重叠"],
    category="pl",
    description=(
        "Detect overlapping tick labels at draw time and separate them by "
        "rotating, sliding them apart with leader lines, or showing fewer"
    ),
    examples=[
        "ax = ov.pl.boxplot(df, 'day', 'total_bill')",
        "ov.pl.declutter_ticks(ax)",
        "# x-axis only, and never rotate past 45 degrees",
        "ov.pl.declutter_ticks(ax, axis='x', max_rotation=45)",
        "# decide once, now, instead of on every draw",
        "ov.pl.declutter_ticks(ax, on_draw=False)",
    ],
    related=["pl.adjust_text", "pl.style_axes", "pl.multipanel"],
)
def declutter_ticks(ax,
                    *,
                    axis: str = "both",
                    max_rotation: float = 90.0,
                    rotate: bool = True,
                    stagger: bool = True,
                    spread: bool = True,
                    leader_color: str = '0.55',
                    leader_linewidth: float = 0.5,
                    thin: bool = True,
                    max_thin: int = 4,
                    on_draw: bool = True):
    r"""Separate tick labels that overlap.

    The remedies are tried in order and the first that clears the collision
    wins, so a plot is changed as little as the crowding requires:

    1. **rotate** — up to ``max_rotation`` degrees (x-axis only; rotating y
       labels does not buy vertical room).
    2. **stagger** — alternate labels are pushed into a second row, one label
       height further from the axis (x-axis only). This is the closest analogue
       to what :func:`~omicverse.pl.adjust_text` does for free-floating text: a
       tick label cannot move *along* its axis without pointing at the wrong
       value, so the space has to come from the perpendicular direction.
    3. **spread** — slide the labels apart *along* the axis, keeping their
       order, and draw a thin leader from each one back to its tick. This is
       the tick-label form of what :func:`~omicverse.pl.adjust_text` does: a
       displaced label would point at the wrong value on its own, and the
       leader is what makes moving it honest — exactly as it is for repelled
       point labels. Font sizes are never touched.
    4. **thin** — show every 2nd, then 3rd, ... label up to ``max_thin``. Only
       reached when the labels cannot be made to fit side by side at all.

    Arguments
    ---------
    ax
        Axes to check.
    axis
        ``'x'``, ``'y'`` or ``'both'``.
    max_rotation
        Ceiling for step 1, in degrees. ``0`` disables rotation.
    rotate, stagger, spread, thin
        Enable or disable individual steps.
    leader_color, leader_linewidth
        Style of the leader lines drawn by step 3.
    max_thin
        Largest stride step 4 may use.
    on_draw
        Register a draw-time callback instead of deciding immediately. This is
        the default because the answer depends on the axes' final physical
        size, which a caller building a multi-panel figure has not fixed yet.
        The callback is idempotent and re-runs on every draw.

    Returns
    -------
    ``ax``.

    Notes
    -----
    A no-op when nothing overlaps, so it is safe to call unconditionally — that
    is what :func:`~omicverse.pl.style_axes` does.
    """
    if axis not in {"x", "y", "both"}:
        raise ValueError(f"`axis` must be 'x', 'y' or 'both', got {axis!r}.")

    if on_draw:
        figure = ax.figure
        if figure is None:
            return ax
        registry = getattr(figure, "_ov_declutter_axes", None)
        if registry is None:
            registry = {}
            figure._ov_declutter_axes = registry

            def _on_draw(event, _figure=figure):
                if getattr(_figure, _FLAG, False):
                    return
                setattr(_figure, _FLAG, True)
                try:
                    for target, options in list(
                            getattr(_figure, "_ov_declutter_axes", {}).items()):
                        _declutter_now(target, **options)
                finally:
                    setattr(_figure, _FLAG, False)

            figure.canvas.mpl_connect("draw_event", _on_draw)
        registry[ax] = dict(axis=axis, max_rotation=max_rotation,
                            rotate=rotate, stagger=stagger, spread=spread,
                            leader_color=leader_color,
                            leader_linewidth=leader_linewidth, thin=thin,
                            max_thin=max_thin)
        return ax

    return _declutter_now(ax, axis=axis, max_rotation=max_rotation,
                          rotate=rotate, stagger=stagger, spread=spread,
                          leader_color=leader_color,
                          leader_linewidth=leader_linewidth, thin=thin,
                          max_thin=max_thin)


def _declutter_now(ax, *, axis: str, max_rotation: float, rotate: bool,
                   stagger: bool, spread: bool, leader_color, leader_linewidth,
                   thin: bool, max_thin: int):
    """Measure and fix, using the geometry as it stands right now."""
    figure = ax.figure
    if figure is None or figure.canvas is None:
        return ax
    try:
        renderer = figure.canvas.get_renderer()
    except AttributeError:                       # a canvas without a renderer
        return ax

    for which in (("x", "y") if axis == "both" else (axis,)):
        labels = _all_labels(ax, which)
        if len(labels) < 2:
            continue

        # Start from a clean slate so an earlier verdict, reached at a different
        # axes size, does not stick: un-hide, un-stagger, un-rotate.
        _unstagger(labels)
        _clear_leaders(ax)
        for text in labels:
            text.set_visible(True)
        if which == "x":
            _apply_rotation(labels, 0.0, which)
        if not _collides(labels, renderer, which):
            continue

        if rotate and which == "x" and max_rotation > 0:
            for angle in _ROTATIONS:
                if angle > max_rotation:
                    break
                _apply_rotation(labels, angle, which)
                if not _collides(labels, renderer, which):
                    break
            else:
                pass
            if not _collides(labels, renderer, which):
                continue

        if stagger and which == "x":
            # Only on x: pushing y labels sideways moves them away from the
            # axis they annotate without buying any vertical room.
            boxes = _boxes(labels, renderer)
            extent = max(b.height for b in boxes) if boxes else 10.0
            _stagger(labels, which, extent + 2.0)
            if not _collides(labels, renderer, which):
                continue
            _unstagger(labels)

        if spread and _spread(ax, labels, renderer, which, leader_color,
                              leader_linewidth):
            continue

        if thin:
            for step in range(2, max(max_thin, 2) + 1):
                _thin(labels, step)
                if not _collides(_visible_labels(ax, which), renderer, which):
                    break
    return ax
