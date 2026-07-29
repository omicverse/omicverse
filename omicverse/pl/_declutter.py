"""Detect overlapping tick labels and separate them.

Tick labels are the one piece of text on a plot whose crowding depends entirely
on the physical size the axes ends up at. "Fri Sat Sun Thur" is comfortable on
a 90 mm panel and a smear on a 40 mm one, and nothing in the plotting call
knows which it will be — a layout engine may resize the axes afterwards.

:func:`declutter_ticks` measures the drawn label boxes and, when they collide,
works through a ladder of remedies. Unlike :func:`~omicverse.pl.adjust_text` it
cannot simply push labels apart: a tick label is anchored to its tick, so
moving it along the axis would make it point at the wrong value. What it can do
is rotate them, deal them into two rows, or show fewer of them.

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


def _remember_size(text) -> float:
    """The label's size before any pass shrank it."""
    if not hasattr(text, "_ov_declutter_base_size"):
        text._ov_declutter_base_size = text.get_size()
    return text._ov_declutter_base_size


def _restore_sizes(labels) -> None:
    for text in labels:
        base = getattr(text, "_ov_declutter_base_size", None)
        if base is not None:
            text.set_size(base)


def _thin(labels, step: int) -> None:
    for index, text in enumerate(labels):
        text.set_visible(index % step == 0)


@register_function(
    aliases=["declutter_ticks", "刻度标签避让", "tick_declutter",
             "fix_tick_overlap", "刻度重叠"],
    category="pl",
    description=(
        "Detect overlapping tick labels at draw time and separate them by "
        "rotating, staggering into two rows, or showing fewer of them"
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
                    shrink: bool = True,
                    min_fontsize: float = 5.0,
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
    3. **shrink** — reduce the label size, down to ``min_fontsize``. On a
       category axis a point or two of font size costs less than losing half
       the category names, and it is the only remedy besides thinning that the
       y-axis can use at all.
    4. **thin** — show every 2nd, then 3rd, ... label up to ``max_thin``.

    Arguments
    ---------
    ax
        Axes to check.
    axis
        ``'x'``, ``'y'`` or ``'both'``.
    max_rotation
        Ceiling for step 1, in degrees. ``0`` disables rotation.
    rotate, stagger, shrink, thin
        Enable or disable individual steps.
    min_fontsize
        Floor for step 3, in points.
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
                            rotate=rotate, stagger=stagger, shrink=shrink,
                            min_fontsize=min_fontsize, thin=thin,
                            max_thin=max_thin)
        return ax

    return _declutter_now(ax, axis=axis, max_rotation=max_rotation,
                          rotate=rotate, stagger=stagger, shrink=shrink,
                          min_fontsize=min_fontsize, thin=thin,
                          max_thin=max_thin)


def _declutter_now(ax, *, axis: str, max_rotation: float, rotate: bool,
                   stagger: bool, shrink: bool, min_fontsize: float,
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
        _restore_sizes(labels)
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

        if shrink:
            # Before dropping labels, try making them smaller. On a category
            # axis losing half the category names is a worse outcome than a
            # point or two of font size, and this is the only remedy available
            # to the y axis besides thinning.
            base = [_remember_size(t) for t in labels]
            for factor in (0.9, 0.8, 0.7, 0.6):
                for text, size in zip(labels, base):
                    text.set_size(max(size * factor, min_fontsize))
                if not _collides(labels, renderer, which):
                    break
            if not _collides(labels, renderer, which):
                continue
            # Still colliding at the floor: keep the smaller size rather than
            # giving it back. Thinning now has less to remove, so fewer labels
            # are lost than if the pass had gone to full size and thinned.

        if thin:
            for step in range(2, max(max_thin, 2) + 1):
                _thin(labels, step)
                if not _collides(_visible_labels(ax, which), renderer, which):
                    break
    return ax
