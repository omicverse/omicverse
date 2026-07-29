r"""Measure-then-place panel layout: the axes box is the size you asked for.

:func:`~omicverse.pl.multipanel` is a gridspec. A gridspec divides the canvas
into cells and the *axes box* is whatever survives after matplotlib has fitted
the decorations into the cell — so a column shares one width, a row shares one
height, and a pie chart is given the same box as a survival curve. When a
journal says "each panel 45 mm wide", 45 mm of *cell* is not 45 mm of *axes*,
and the request cannot be honoured at all.

``panelflow`` inverts that. Every panel is declared at its finished axes size,
panels flow left to right and wrap, and everything drawn *outside* the axes box
— y tick labels, the y axis label, the panel tag, and any loose text a plotting
function put there, such as the estimate strings ``ov.pl.forest`` writes beside
its intervals — is measured from the rendered figure on a draw event and
reserved next to the box rather than taken out of it. The axes box stays
exactly ``width x height``; the canvas and the panel positions are what move.

Why measured and not a fixed number: the width of a y tick label is a property
of the *text* — "0.2" and "1.25e+06" and "Oligodendrocyte" are not the same
number of millimetres, and neither is the same string at a different font or
dpi. A static reserve is therefore either too small (the tick labels run into
the panel to its left) or too large (the whitespace this module exists to
remove). The only number that is right is the one the renderer produced, which
is why the reserve is refreshed on ``draw_event`` and the layout re-run when it
moves.

Design credit: the flow-packing and measured-reserve approach follows cnsplots
(Farid Rashidi, BSD-3-Clause, https://github.com/faridrashidi/cnsplots). This
is an independent implementation — it works in physical units rather than
72-dpi layout pixels, measures all four sides, and deliberately does *not* make
a new panel the current axes.

Examples
--------
>>> fl = ov.pl.panelflow(max_width=183, units='mm', title='Figure 1')
>>> a = fl.panel('A', 45, 100)
>>> ov.pl.violinplot(df, x='cell_type', y='score', ax=a)
>>> b = fl.panel('B', 60, 100, color_cycle=['#4C72B0', '#DD8452'])
>>> sns.scatterplot(data=df, x='x', y='y', ax=b)
>>> c = fl.panel('C', 45, 45, below='A')
>>> fl.savefig('figure1', formats=['pdf', 'svg'])
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .._registry import register_function
from ._layout import (
    _PANEL_LABEL_ALPHABET,
    _resolve_width,
    _to_inch_factor,
    savefig as _savefig,
    set_editable_text,
)

__all__ = ["PanelFlow", "panelflow"]


#: A measured reserve is only worth re-laying-out for if it moved by more than
#: this many rendered pixels. Below that the position shift is invisible and
#: chasing it risks two measurements alternating forever.
_RESERVE_TOLERANCE_PX = 0.5

#: Re-layout inside a draw is bounded. Reserves are font metrics and so do not
#: depend on where the panel ended up, which means one pass normally settles
#: it; the extra passes exist for artists (legends, offset text) that appear
#: only once something else has moved.
_MAX_RELAYOUT_PASSES = 4

#: What "the panel is empty" resolves to, so that a flow with no panels still
#: has a legal figure size.
_MIN_FIGURE_INCHES = 0.05

_POINTS_PER_INCH = 72.0


def _positive(name: str, value: Any) -> float:
    """Reject the dimensions matplotlib would silently turn into a broken figure."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a number, got {value!r}.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"`{name}` must be a positive finite number, got {value}.")
    return value


def _as_quad(value, name: str) -> Tuple[float, float, float, float]:
    """One number, or ``(left, right, bottom, top)`` — the order `_layout` uses."""
    if isinstance(value, (int, float)):
        return (float(value),) * 4
    quad = tuple(float(v) for v in value)
    if len(quad) != 4:
        raise ValueError(
            f"`{name}` takes one number or four (left, right, bottom, top), "
            f"got {value!r}."
        )
    return quad


@contextmanager
def _keep_current_axes():
    """Run a block without letting it steal ``plt.gca()``.

    ``Figure.add_axes`` makes the new axes current, and pyplot makes the new
    figure current, so building a panel would quietly redirect every later
    bare ``plt.plot`` / ``ov.pl.*`` call into it. That is the failure mode this
    module is trying to remove — drawing into the wrong panel has to be an
    error, not a silent success — so the previous (figure, axes) pair is put
    back afterwards and callers must pass ``ax=`` explicitly.
    """
    previous_fig = plt.gcf() if plt.get_fignums() else None
    previous_ax = previous_fig.gca() if (previous_fig is not None
                                         and previous_fig.axes) else None
    try:
        yield
    finally:
        if previous_fig is not None and plt.fignum_exists(previous_fig.number):
            plt.figure(previous_fig.number)
            if previous_ax is not None and previous_ax in previous_fig.axes:
                previous_fig.sca(previous_ax)


@dataclass
class _Panel:
    """One panel's declared geometry plus what the renderer said about it.

    Everything is in inches. Declared fields come from the caller; the
    ``decor_*`` / ``tag_*`` / ``over_*`` fields are zero until the first draw
    and are refreshed from rendered artists after every draw.
    """

    label: str
    width: float
    height: float
    pad_left: float
    pad_top: float
    margin_left: float
    margin_right: float
    margin_bottom: float
    margin_top: float
    below: Optional[str] = None
    tag: bool = True

    # Measured outside the axes box, on the side named.
    decor_left: float = 0.0
    decor_top: float = 0.0
    over_right: float = 0.0
    over_bottom: float = 0.0
    tag_width: float = 0.0
    tag_height: float = 0.0

    @property
    def left_reserve(self) -> float:
        """Space between the panel's outer left edge and the axes box.

        The tag sits at the far left of it, then ``pad_left`` of clear air,
        then the y-axis decorations, then the box.
        """
        reserve = self.decor_left
        if self.tag:
            reserve += self.tag_width + self.pad_left
        return reserve

    @property
    def top_reserve(self) -> float:
        """Space between the panel's outer top edge and the axes box."""
        reserve = self.decor_top
        if self.tag:
            reserve += self.tag_height + self.pad_top
        return reserve

    @property
    def total_width(self) -> float:
        return (self.margin_left + self.left_reserve + self.width
                + self.over_right + self.margin_right)

    @property
    def total_height(self) -> float:
        return (self.margin_top + self.top_reserve + self.height
                + self.over_bottom + self.margin_bottom)


@dataclass(frozen=True)
class _Layout:
    """The pure result of packing panels: no matplotlib state is touched here."""

    rows: Tuple[Tuple[int, ...], ...]
    row_heights: Tuple[float, ...]
    #: Top-left corner of each panel's *axes box*, in inches from the top-left
    #: of the canvas. Top-left because that is the order panels are read in;
    #: the flip to matplotlib's bottom-left origin happens once, at apply time.
    positions: Tuple[Tuple[float, float], ...]
    content_width: float


def _child_map(panels: Sequence[_Panel]) -> List[List[int]]:
    """Resolve ``below=`` into a forest, refusing anything that is not one."""
    index_of: Dict[str, int] = {}
    for idx, panel in enumerate(panels):
        if panel.label in index_of:
            raise ValueError(f"Duplicate panel label {panel.label!r}.")
        index_of[panel.label] = idx

    children: List[List[int]] = [[] for _ in panels]
    for idx, panel in enumerate(panels):
        if panel.below is None:
            continue
        parent = index_of.get(panel.below)
        if parent is None:
            raise ValueError(
                f"Panel {panel.label!r} is declared below {panel.below!r}, "
                f"which is not a panel. Known panels: "
                f"{sorted(index_of)}."
            )
        if parent == idx:
            raise ValueError(
                f"Panel {panel.label!r} cannot be below itself."
            )
        children[parent].append(idx)

    # Three-colour DFS. A cycle here would make the recursive size and place
    # passes below recurse until the interpreter gives up, so it is worth
    # naming the offending panel rather than letting that happen.
    UNVISITED, ON_STACK, DONE = 0, 1, 2
    state = [UNVISITED] * len(panels)

    def visit(idx: int) -> None:
        if state[idx] == ON_STACK:
            raise ValueError(
                f"`below` relationships form a cycle through panel "
                f"{panels[idx].label!r}; a panel cannot be stacked under "
                f"itself, directly or transitively."
            )
        if state[idx] == DONE:
            return
        state[idx] = ON_STACK
        for child in children[idx]:
            visit(child)
        state[idx] = DONE

    for idx in range(len(panels)):
        visit(idx)
    return children


def _pack(panels: Sequence[_Panel], max_width: float,
          title_height: float) -> _Layout:
    """Flow the panels left to right, wrapping at ``max_width`` (inches).

    A ``below=`` chain is packed as a single unit: the column is as wide as its
    widest member and as tall as the sum, so a stack wraps and is placed as one
    thing rather than letting its members drift into different rows.
    """
    if not panels:
        return _Layout(rows=(), row_heights=(), positions=(), content_width=0.0)

    children = _child_map(panels)
    subtree: List[Optional[Tuple[float, float]]] = [None] * len(panels)

    def size_of(idx: int) -> Tuple[float, float]:
        cached = subtree[idx]
        if cached is not None:
            return cached
        panel = panels[idx]
        width, height = panel.total_width, panel.total_height
        for child in children[idx]:
            child_w, child_h = size_of(child)
            width = max(width, child_w)
            height += child_h
        subtree[idx] = (width, height)
        return width, height

    rows: List[List[int]] = []
    row_heights: List[float] = []
    current: List[int] = []
    current_width = 0.0
    current_height = 0.0
    widest_row = 0.0

    for idx, panel in enumerate(panels):
        if panel.below is not None:
            continue  # placed by its parent
        width, height = size_of(idx)
        # A panel wider than the whole flow still gets a row of its own rather
        # than an empty row above it, hence the `current and` guard.
        if current and current_width + width > max_width:
            rows.append(current)
            row_heights.append(current_height)
            widest_row = max(widest_row, current_width)
            current, current_width, current_height = [idx], width, height
        else:
            current.append(idx)
            current_width += width
            current_height = max(current_height, height)
    if current:
        rows.append(current)
        row_heights.append(current_height)
        widest_row = max(widest_row, current_width)

    positions = [(0.0, 0.0)] * len(panels)

    def place(idx: int, column_x: float, outer_y: float) -> float:
        panel = panels[idx]
        positions[idx] = (column_x + panel.margin_left + panel.left_reserve,
                          outer_y + panel.margin_top + panel.top_reserve)
        next_y = outer_y + panel.total_height
        for child in children[idx]:
            next_y = place(child, column_x, next_y)
        return next_y

    row_y = title_height
    for row, height in zip(rows, row_heights):
        column_x = 0.0
        for idx in row:
            place(idx, column_x, row_y)
            column_x += size_of(idx)[0]
        row_y += height

    return _Layout(
        rows=tuple(tuple(row) for row in rows),
        row_heights=tuple(row_heights),
        positions=tuple(positions),
        content_width=widest_row,
    )


class PanelFlow:
    r"""A figure whose panels are declared at their finished axes size.

    Panels are added with :meth:`panel`, which **returns** the axes and does
    not make it current — pass it on with ``ax=``. They flow left to right and
    wrap when the row would exceed ``max_width``; ``below=`` stacks a panel
    under another in the same column.

    Arguments
    ---------
    max_width
        Width of the canvas in ``units``, or a journal preset name from
        :data:`~omicverse.pl.JOURNAL_WIDTH_MM` (``'nature-double'``, ...).
        Panels wrap to a new row rather than exceed it.
    units
        ``'mm'`` (default), ``'cm'``, ``'in'`` or ``'px'`` — the same set
        :func:`~omicverse.pl.figure` takes, resolved by the same code.
    dpi
        Figure resolution. Also the conversion for ``units='px'``.
    panel_size
        ``(width, height)`` used by :meth:`panel` when the call omits them.
    margins
        Space around each panel's outer box, in ``units`` — one number or
        ``(left, right, bottom, top)``. This is the gutter between panels; the
        room needed by tick labels is measured, not budgeted here.
    pad_left, pad_top
        Clear air between the panel tag and the nearest measured decoration,
        in ``units``.
    label
        ``'A'``/``'upper'`` (default) or ``'a'``/``'lower'`` — which alphabet
        automatic panel labels come from, matching ``multipanel``'s ``label``.
    tag
        Draw the panel tag. ``False`` still labels the panels for ``fl[...]``
        addressing, it just does not print them.
    title, title_kw, title_loc, title_pad
        A figure-level title drawn in a band reserved above the first row.
    editable_text
        Apply :data:`~omicverse.pl.EDITABLE_TEXT_RCPARAMS` so vector exports
        keep text as text.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
    axes : dict
        Panel label to axes, in declaration order.

    Notes
    -----
    Two consequences of measuring rather than guessing are worth knowing:

    * Row assignment depends on the reserves, so a flow can re-wrap on its
      first draw. Positions read back before any draw are provisional.
    * A colorbar made with ``fig.colorbar(mappable, ax=panel_axes)`` resizes
      its parent to make room, which this engine then undoes on the next
      layout pass. Give the colorbar its own panel and pass ``cax=``.

    And one thing matplotlib will not let any layout engine promise. An axes
    with a **fixed aspect ratio and** ``adjustable='box'`` — which is what
    ``ax.pie``, ``ax.imshow`` and ``set_aspect('equal')`` leave behind — has
    its drawn box shrunk to the largest one of the right shape that fits
    inside the rectangle it was given, and centred in it. ``panel(55, 45)``
    for a pie therefore draws 45x45. The two requests genuinely contradict
    each other, so the choice is which one to keep: ask for a square panel, or
    call ``ax.set_adjustable('datalim')`` to keep the box and let the data
    limits give way instead. ``ax.get_position(original=True)`` still reports
    the rectangle this engine assigned.
    """

    def __init__(self,
                 max_width: Union[float, str] = "nature-double",
                 *,
                 units: str = "mm",
                 dpi: float = 300,
                 panel_size: Tuple[float, float] = (45.0, 45.0),
                 margins: Union[float, Sequence[float]] = (0.0, 5.0, 5.0, 0.0),
                 pad_left: float = 1.5,
                 pad_top: float = 1.5,
                 label: str = "A",
                 tag: bool = True,
                 tag_kw: Optional[Mapping[str, Any]] = None,
                 title: Optional[str] = None,
                 title_kw: Optional[Mapping[str, Any]] = None,
                 title_loc: str = "left",
                 title_pad: float = 2.0,
                 editable_text: bool = True):
        self._units = str(units)
        self._dpi = float(dpi)
        self._to_inch = _to_inch_factor(self._units, self._dpi)

        resolved = _resolve_width(max_width, self._units)
        self._max_width_in = _positive("max_width", resolved) * self._to_inch

        self._default_size = (
            _positive("panel_size[0]", panel_size[0]),
            _positive("panel_size[1]", panel_size[1]),
        )
        self._default_margins = _as_quad(margins, "margins")
        self._default_pad_left = float(pad_left)
        self._default_pad_top = float(pad_top)
        self._default_tag = bool(tag)
        self._tag_kw = dict(tag_kw) if tag_kw else {}

        if label in {"A", "upper", "uppercase"}:
            self._alphabet = _PANEL_LABEL_ALPHABET.upper()
        elif label in {"a", "lower", "lowercase"}:
            self._alphabet = _PANEL_LABEL_ALPHABET
        else:
            raise ValueError(
                "`label` must be 'A'/'upper' or 'a'/'lower'; pass an explicit "
                "label to `panel()` for anything else."
            )

        if title_loc not in {"left", "center", "right"}:
            raise ValueError(
                f"`title_loc` must be 'left', 'center' or 'right', got "
                f"{title_loc!r}."
            )
        self._title = title
        self._title_kw = dict(title_kw) if title_kw else {}
        self._title_loc = title_loc
        self._title_pad_in = float(title_pad) * self._to_inch
        self._title_text = None
        # Before the first draw the band is a font-metric estimate; it is
        # replaced by the rendered height as soon as there is a renderer.
        self._title_height_in = 0.0 if title is None else self._estimated_title_in()

        if editable_text:
            set_editable_text(True)

        self._panels: List[_Panel] = []
        self._axes: Dict[str, Axes] = {}
        self._tags: Dict[str, Any] = {}
        self._layout: Optional[_Layout] = None
        self._relaying_out = False
        self._draw_cid: Optional[int] = None

        with _keep_current_axes():
            self._fig = plt.figure(
                figsize=(self._max_width_in, max(self._title_height_in,
                                                 _MIN_FIGURE_INCHES)),
                dpi=self._dpi, layout=None,
            )
            self._draw_cid = self._fig.canvas.mpl_connect("draw_event",
                                                          self._on_draw)
            # Lay out once while empty so a titled flow has its title before
            # the first panel, and so the band above row one is already there
            # when that panel is placed.
            self._apply_layout()

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------

    @property
    def fig(self):
        """The matplotlib figure."""
        return self._fig

    @property
    def figure(self):
        """Alias of :attr:`fig`, for symmetry with matplotlib artists."""
        return self._fig

    @property
    def axes(self) -> Dict[str, Axes]:
        """``{label: axes}`` in declaration order (a copy — mutate the flow)."""
        return dict(self._axes)

    @property
    def labels(self) -> List[str]:
        """Panel labels in declaration order."""
        return [panel.label for panel in self._panels]

    def __getitem__(self, label: str) -> Axes:
        try:
            return self._axes[label]
        except KeyError:
            raise KeyError(
                f"No panel {label!r} in this flow. Panels: "
                f"{self.labels or 'none yet'}."
            ) from None

    def __contains__(self, label: object) -> bool:
        return label in self._axes

    def __len__(self) -> int:
        return len(self._panels)

    def __iter__(self):
        return iter(self._axes.values())

    def __repr__(self) -> str:
        return (f"<PanelFlow {len(self._panels)} panel(s) "
                f"{self.labels} max_width="
                f"{self._max_width_in / self._to_inch:g} {self._units}>")

    def panel(self,
              label: Optional[str] = None,
              width: Optional[float] = None,
              height: Optional[float] = None,
              *,
              below: Optional[str] = None,
              pad_left: Optional[float] = None,
              pad_top: Optional[float] = None,
              margins: Union[float, Sequence[float], None] = None,
              color_cycle: Optional[Sequence[Any]] = None,
              tag: Optional[bool] = None,
              style: Optional[Mapping[str, Any]] = None) -> Axes:
        r"""Add a panel and return its axes.

        ``width`` and ``height`` are the size of the **axes box**, in the
        flow's units. That is what is drawn; tick labels and the tag are
        measured and placed outside it.

        Arguments
        ---------
        label
            Panel label. ``None`` takes the next letter of the flow's
            alphabet. The label is what :meth:`__getitem__` addresses and
            what ``below=`` names.
        width, height
            Axes box size in the flow's units. Default to the flow's
            ``panel_size``.
        below
            Label of an already-declared panel. This panel is stacked directly
            under it, in the same column, and the two travel together when the
            row wraps.
        pad_left, pad_top
            Override the flow's clear air between the tag and the measured
            decorations.
        margins
            Override the flow's outer margins for this panel.
        color_cycle
            Bound to the axes with ``set_prop_cycle``, so the plotting calls
            that follow do not each have to carry the colour.
        tag
            Draw this panel's tag. ``False`` keeps the label addressable.
        style
            Passed to ``Axes.set`` — ``title``, ``xlabel``, ``xlim`` and so on.

        Returns
        -------
        matplotlib.axes.Axes
            The panel's axes. It is deliberately **not** made current: pass it
            on as ``ax=``.
        """
        width = self._default_size[0] if width is None else width
        height = self._default_size[1] if height is None else height
        width_in = _positive("width", width) * self._to_inch
        height_in = _positive("height", height) * self._to_inch

        if label is None:
            if len(self._panels) >= len(self._alphabet):
                raise ValueError(
                    f"Automatic labels run out after {len(self._alphabet)} "
                    f"panels; pass an explicit `label`."
                )
            label = self._alphabet[len(self._panels)]
        if not isinstance(label, str):
            raise TypeError(f"`label` must be a string or None, got {label!r}.")
        if label in self._axes:
            raise ValueError(
                f"Panel {label!r} already exists in this flow. Panels: "
                f"{self.labels}."
            )

        if below is not None:
            if not isinstance(below, str):
                raise TypeError(f"`below` must be a string or None, got {below!r}.")
            if below not in self._axes:
                raise ValueError(
                    f"`below={below!r}` names a panel that does not exist yet; "
                    f"declare it first. Panels: {self.labels or 'none yet'}."
                )

        quad = (self._default_margins if margins is None
                else _as_quad(margins, "margins"))
        panel = _Panel(
            label=label,
            width=width_in,
            height=height_in,
            pad_left=(self._default_pad_left if pad_left is None
                      else float(pad_left)) * self._to_inch,
            pad_top=(self._default_pad_top if pad_top is None
                     else float(pad_top)) * self._to_inch,
            margin_left=quad[0] * self._to_inch,
            margin_right=quad[1] * self._to_inch,
            margin_bottom=quad[2] * self._to_inch,
            margin_top=quad[3] * self._to_inch,
            below=below,
            tag=self._default_tag if tag is None else bool(tag),
        )
        self._panels.append(panel)

        with _keep_current_axes():
            # A provisional unit position; _apply_layout overwrites it before
            # this method returns. add_axes needs *something* legal.
            ax = self._fig.add_axes((0.0, 0.0, 1e-3, 1e-3))
            self._axes[label] = ax
            if color_cycle is not None:
                ax.set_prop_cycle(color=list(color_cycle))
            if style:
                ax.set(**dict(style))
            self._apply_layout()
        return ax

    def draw(self) -> None:
        """Settle the layout: lay out, render, re-measure, lay out again.

        Reserves only exist once something has been rendered, so anything that
        wants final geometry — saving, or reading positions back in a test —
        has to force a draw first.
        """
        self._apply_layout()
        self._fig.canvas.draw()

    def savefig(self, path, *, formats: Optional[Iterable[str]] = None,
                bbox_inches: Optional[str] = None, **kwargs: Any) -> List[str]:
        r"""Write the figure, after settling the layout.

        ``bbox_inches`` defaults to ``None`` rather than ``'tight'``: this
        module exists to make the physical geometry exact, and a tight box
        re-crops the canvas. Pass ``'tight'`` explicitly to crop anyway.
        """
        # Saving renders, and rendering is what triggers re-layout — including
        # a possible change of figure size. Settle first, or the file is
        # written from the pre-relayout state.
        self.draw()
        return _savefig(self._fig, path, formats=formats,
                        bbox_inches=bbox_inches, **kwargs)

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------

    def _estimated_title_in(self) -> float:
        """A font-metric guess at the title band, used until it is measured."""
        from matplotlib.font_manager import FontProperties

        size = self._title_kw.get("fontsize")
        if size is None:
            size = FontProperties(
                size=plt.rcParams.get("figure.titlesize", "large")
            ).get_size_in_points()
        return float(size) * 1.6 / _POINTS_PER_INCH + 2 * self._title_pad_in

    def _apply_layout(self) -> None:
        """Pack the panels and push the result onto the figure."""
        layout = _pack(self._panels, self._max_width_in, self._title_height_in)
        self._layout = layout

        fig_w = self._max_width_in
        fig_h = max(self._title_height_in + sum(layout.row_heights),
                    _MIN_FIGURE_INCHES)
        if (abs(self._fig.get_figwidth() - fig_w) > 1e-9
                or abs(self._fig.get_figheight() - fig_h) > 1e-9):
            self._fig.set_size_inches(fig_w, fig_h, forward=False)

        for idx, panel in enumerate(self._panels):
            ax = self._axes.get(panel.label)
            if ax is None:
                continue
            x_in, y_in = layout.positions[idx]
            # matplotlib's origin is bottom-left and the pack works top-left,
            # so this is the one place the flip happens. The width/height
            # fractions are the whole point of the module: they are derived
            # from the requested size and the canvas, and from nothing that
            # was measured, so the drawn box is exact whether or not the
            # reserves have converged.
            ax.set_position((x_in / fig_w,
                             (fig_h - y_in - panel.height) / fig_h,
                             panel.width / fig_w,
                             panel.height / fig_h))
            self._sync_tag(panel, ax)

        self._sync_title(fig_w, fig_h)

    def _sync_tag(self, panel: _Panel, ax: Axes) -> None:
        """Park the tag just outside the measured decorations.

        The offset is in typographic points rather than axes fractions so the
        gap does not change with the panel size, and ``annotation_clip`` is
        off because the tag is by construction outside the axes.
        """
        existing = self._tags.get(panel.label)
        if not panel.tag:
            if existing is not None:
                existing.remove()
                self._tags.pop(panel.label, None)
            return

        dx = -(panel.decor_left + panel.pad_left) * _POINTS_PER_INCH
        dy = (panel.decor_top + panel.pad_top) * _POINTS_PER_INCH
        if existing is None or existing.axes is not ax:
            kw = dict(fontsize=plt.rcParams.get("axes.titlesize", "medium"),
                      fontweight="bold")
            kw.update(self._tag_kw)
            tag = ax.annotate(
                panel.label, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                ha="right", va="bottom", annotation_clip=False, **kw)
            # Keep the tag out of `ax.get_tightbbox`, which `_overhang` reads.
            # The tag's own offset is computed *from* that measurement, so
            # letting it contribute would make each pass push it a tag-width
            # further left. It is measured separately instead.
            tag.set_in_layout(False)
            self._tags[panel.label] = tag
        else:
            existing.xyann = (dx, dy)

    def _sync_title(self, fig_w: float, fig_h: float) -> None:
        if self._title is None:
            if self._title_text is not None:
                self._title_text.remove()
                self._title_text = None
            return
        x = {"left": self._title_pad_in / fig_w,
             "center": 0.5,
             "right": 1.0 - self._title_pad_in / fig_w}[self._title_loc]
        y = 1.0 - (self._title_height_in / 2.0) / fig_h
        if self._title_text is None:
            kw = dict(fontsize=plt.rcParams.get("figure.titlesize", "large"),
                      fontweight="bold")
            kw.update(self._title_kw)
            kw.setdefault("ha", self._title_loc)
            kw.setdefault("va", "center")
            self._title_text = self._fig.text(x, y, self._title, **kw)
        else:
            self._title_text.set_position((x, y))
            self._title_text.set_text(self._title)

    # ------------------------------------------------------------------
    # measurement
    # ------------------------------------------------------------------

    def _overhang(self, ax: Axes, renderer) -> Dict[str, float]:
        """How far the panel's rendered content spills past its axes box.

        ``ax.get_tightbbox`` unions *everything* the axes will draw — both
        axis' tick labels, labels and offset text, the title, the legend, child
        and inset axes, and any loose ``ax.text`` / ``ax.annotate``. Listing the
        artists we expect instead was a bug: several ``ov.pl`` functions write
        in axes coordinates outside the box on purpose (``ov.pl.forest`` puts
        "0.68 (0.52, 0.89)" at ``x=1.04``, some 30 mm to the right of a 50 mm
        panel), an enumeration silently measured none of it, and the numbers
        landed on top of the next panel in the row.

        Two properties of ``get_tightbbox`` this relies on, so that nobody
        "fixes" them later:

        * It honours clipping. An annotation with ``clip_on=True`` is drawn
          inside the box and correctly contributes nothing — only artists that
          actually escape the box are reserved for.
        * It skips artists with ``get_in_layout() == False``, which is how the
          panel tag is kept out of this measurement. The tag is *placed* from
          the reserve, so folding it back into the reserve would be a feedback
          loop that walks the tag further left on every pass.

        It also measures the axis with matplotlib's own ``for_layout_only``
        rule, which trims the overhang of the first and last tick label. That
        is what stops the leftmost x tick label from being charged to the left
        reserve and dragging the tag out with it.
        """
        box = ax.get_window_extent(renderer)
        full = ax.get_tightbbox(renderer)
        if full is None:
            return {"left": 0.0, "top": 0.0, "bottom": 0.0, "right": 0.0}
        return {
            "left": max(0.0, float(box.x0 - full.x0)),
            "right": max(0.0, float(full.x1 - box.x1)),
            "top": max(0.0, float(full.y1 - box.y1)),
            "bottom": max(0.0, float(box.y0 - full.y0)),
        }

    def _measure(self, renderer) -> bool:
        """Refresh every panel's reserves; report whether anything moved.

        Returns ``True`` when at least one measurement changed by more than
        :data:`_RESERVE_TOLERANCE_PX`, which is the signal to re-run the
        layout. Returning ``False`` on a settled figure is what stops the draw
        handler from looping.
        """
        dpi = self._fig.dpi
        tolerance_in = _RESERVE_TOLERANCE_PX / dpi
        changed = False

        for panel in self._panels:
            ax = self._axes.get(panel.label)
            if ax is None or ax.figure is not self._fig:
                continue
            measured = self._overhang(ax, renderer)
            tag = self._tags.get(panel.label)
            if tag is not None and panel.tag:
                extent = tag.get_window_extent(renderer=renderer)
                tag_w, tag_h = extent.width / dpi, extent.height / dpi
            else:
                tag_w = tag_h = 0.0

            for name, value in (("decor_left", measured["left"] / dpi),
                                ("decor_top", measured["top"] / dpi),
                                ("over_right", measured["right"] / dpi),
                                ("over_bottom", measured["bottom"] / dpi),
                                ("tag_width", tag_w),
                                ("tag_height", tag_h)):
                if abs(value - getattr(panel, name)) > tolerance_in:
                    setattr(panel, name, value)
                    changed = True

        if self._title_text is not None:
            height = self._title_text.get_window_extent(
                renderer=renderer).height / dpi + 2 * self._title_pad_in
            if abs(height - self._title_height_in) > tolerance_in:
                self._title_height_in = height
                changed = True
        return changed

    def _on_draw(self, event) -> None:
        """Re-measure after a draw and re-lay-out if the reserves moved.

        The re-layout has to render again to check its own work, and rendering
        re-enters this handler — so the pass is guarded by a flag and bounded
        by a pass count. Without the flag a single draw would recurse until the
        stack ran out; without the bound, two measurements that disagree by a
        hair would ping-pong forever. Bounding it is safe because the axes box
        is not derived from the measurement: an unconverged flow has slightly
        wrong *positions*, never a wrong panel size.
        """
        if self._relaying_out or self._fig is None:
            return
        canvas = getattr(event, "canvas", None)
        if canvas is not self._fig.canvas:
            return
        renderer = getattr(event, "renderer", None)
        if renderer is None or not self._measure(renderer):
            return

        self._relaying_out = True
        try:
            for _ in range(_MAX_RELAYOUT_PASSES):
                self._apply_layout()
                canvas.draw()
                get_renderer = getattr(canvas, "get_renderer", None)
                if not callable(get_renderer):
                    break
                renderer = get_renderer()
                if renderer is None or not self._measure(renderer):
                    break
        finally:
            self._relaying_out = False


@register_function(
    aliases=["面板流", "panelflow", "panel_flow", "流式面板", "定尺面板",
             "measured_panels", "flow_panels"],
    category="pl",
    description=(
        "Assemble a figure from panels declared at their finished axes size. "
        "Panels flow left to right and wrap; tick labels, axis labels and "
        "panel tags are measured from the rendered figure and reserved "
        "outside the axes box, so the drawn panel is exactly the requested "
        "size — unlike a gridspec, where the box is the leftover"
    ),
    examples=[
        "# Panels sized in mm on a Nature double-column canvas",
        "fl = ov.pl.panelflow(max_width=183, units='mm', title='Figure 1')",
        "a = fl.panel('A', 45, 40)          # returns the axes, does not set gca",
        "a.plot([0, 1], [0, 1])",
        "# A colour cycle bound to the panel, so later calls need not carry it",
        "b = fl.panel('B', 60, 40, color_cycle=['#4C72B0', '#DD8452'])",
        "b.bar(['x', 'y'], [3, 5])",
        "# Stack a panel under A, in the same column",
        "c = fl.panel('C', 45, 30, below='A')",
        "c.scatter([0, 1, 2], [1, 0, 2])",
        "# Panels stay addressable by their label",
        "fl['B'].set_ylabel('counts')",
        "fl.savefig('figure1', formats=['pdf', 'svg'])",
    ],
    related=["pl.multipanel", "pl.figure", "pl.savefig", "pl.add_panel_label"],
)
def panelflow(max_width: Union[float, str] = "nature-double",
              *,
              units: str = "mm",
              dpi: float = 300,
              panel_size: Tuple[float, float] = (45.0, 45.0),
              margins: Union[float, Sequence[float]] = (0.0, 5.0, 5.0, 0.0),
              pad_left: float = 1.5,
              pad_top: float = 1.5,
              label: str = "A",
              tag: bool = True,
              tag_kw: Optional[Mapping[str, Any]] = None,
              title: Optional[str] = None,
              title_kw: Optional[Mapping[str, Any]] = None,
              title_loc: str = "left",
              title_pad: float = 2.0,
              editable_text: bool = True) -> PanelFlow:
    r"""Start a measure-then-place panel layout.

    ``multipanel`` divides a canvas into cells and the axes box is the
    leftover, so panels in a column share a width and a request like "45 mm
    wide" cannot be met. ``panelflow`` declares the axes box instead and
    measures the decorations out of the rendered figure, reserving them beside
    the box rather than inside it.

    See :class:`PanelFlow` for the full argument list; this function only
    constructs one.

    Returns
    -------
    PanelFlow
        Add panels with ``.panel(label, width, height)``, which returns the
        axes for you to pass on as ``ax=``.
    """
    return PanelFlow(max_width, units=units, dpi=dpi, panel_size=panel_size,
                     margins=margins, pad_left=pad_left, pad_top=pad_top,
                     label=label, tag=tag, tag_kw=tag_kw, title=title,
                     title_kw=title_kw, title_loc=title_loc,
                     title_pad=title_pad, editable_text=editable_text)
