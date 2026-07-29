r"""Figure geometry, multi-panel layout and publication-grade export.

Journals specify figure *width* in millimetres and expect text in a vector
export to stay text — editable in Illustrator/Inkscape, not outlines. Neither
is what matplotlib does by default: ``figsize`` is in inches, and the SVG
backend converts every glyph to a path.

This module closes that gap:

``figure``            a canvas sized in mm / cm / inches / pixels, with the
                      column widths of common journals available by name
``multipanel``        a labelled panel grid (or mosaic) in one call
``add_panel_label``   the ``a`` / ``b`` / ``c`` tags, offset in typographic
                      points so they do not drift with the axes size
``savefig``           write one figure to several formats at once, keeping
                      text as text
``set_editable_text`` make that the default for plain ``plt.savefig`` too
``take_legend_out``   move a legend outside the axes without resizing games

Nothing here is specific to omics data — these are the plumbing functions the
rest of ``ov.pl`` and any user script can share.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .._registry import register_function

__all__ = [
    "JOURNAL_WIDTH_MM",
    "EDITABLE_TEXT_RCPARAMS",
    "figure",
    "multipanel",
    "add_panel_label",
    "savefig",
    "set_editable_text",
    "take_legend_out",
]


#: Column widths in millimetres, from the author guidelines of each journal.
#: Use the key as ``width=`` in :func:`figure` / :func:`multipanel`.
JOURNAL_WIDTH_MM: Dict[str, float] = {
    "nature-single": 89.0,
    "nature-medium": 120.0,
    "nature-double": 183.0,
    "science-single": 55.0,
    "science-double": 120.0,
    "science-triple": 183.0,
    "cell-single": 85.0,
    "cell-medium": 114.0,
    "cell-double": 174.0,
    "pnas-single": 87.0,
    "pnas-medium": 114.0,
    "pnas-double": 178.0,
    "elife-single": 85.0,
    "elife-double": 174.0,
}

#: rcParams that keep text as text in vector output.
#: ``svg.fonttype='none'`` stops the SVG backend from converting glyphs to
#: paths; ``42`` selects TrueType (rather than Type-3) font embedding in
#: PDF/PS, which Illustrator can edit.
EDITABLE_TEXT_RCPARAMS: Dict[str, Any] = {
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

_UNIT_TO_INCH = {
    "in": 1.0,
    "inch": 1.0,
    "inches": 1.0,
    "mm": 1.0 / 25.4,
    "cm": 1.0 / 2.54,
}

_PANEL_LABEL_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _resolve_width(value: Union[float, str, None], units: str) -> Optional[float]:
    """Turn a number or a journal preset name into a value in ``units``."""
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower().replace("_", "-").replace(" ", "-")
        if key not in JOURNAL_WIDTH_MM:
            raise KeyError(
                f"Unknown journal width preset {value!r}. Available: "
                + ", ".join(sorted(JOURNAL_WIDTH_MM))
            )
        mm = JOURNAL_WIDTH_MM[key]
        # The preset is defined in mm; convert into whatever the caller uses.
        return mm * _UNIT_TO_INCH["mm"] / _to_inch_factor(units)
    return float(value)


def _to_inch_factor(units: str, dpi: float = 100.0) -> float:
    u = str(units).strip().lower()
    if u in _UNIT_TO_INCH:
        return _UNIT_TO_INCH[u]
    if u in {"px", "pixel", "pixels"}:
        return 1.0 / float(dpi)
    raise ValueError(
        f"Unknown units {units!r}; use one of 'mm', 'cm', 'in' or 'px'."
    )


def _figsize_inches(width, height, units: str, dpi: float,
                    aspect: Optional[float]) -> Tuple[float, float]:
    factor = _to_inch_factor(units, dpi)
    w = _resolve_width(width, units)
    h = _resolve_width(height, units)
    if w is None and h is None:
        raise ValueError("Give at least one of `width` or `height`.")
    if aspect is not None and float(aspect) <= 0:
        raise ValueError("`aspect` must be positive (width / height).")
    if w is None:
        w = h * float(aspect if aspect is not None else 1.4)
    if h is None:
        h = w / float(aspect if aspect is not None else 1.4)
    w_in, h_in = w * factor, h * factor
    if w_in <= 0 or h_in <= 0:
        raise ValueError(f"Figure size must be positive, got {w_in}x{h_in} in.")
    return w_in, h_in


def _as_pair(value, name: str) -> Tuple[float, float]:
    if isinstance(value, (int, float)):
        return float(value), float(value)
    pair = tuple(float(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"`{name}` takes one number or two, got {value!r}.")
    return pair


def _as_margins(value) -> Tuple[float, float, float, float]:
    if isinstance(value, (int, float)):
        return (float(value),) * 4
    quad = tuple(float(v) for v in value)
    if len(quad) != 4:
        raise ValueError(
            f"`margins` takes one number or four (left, right, bottom, top), "
            f"got {value!r}."
        )
    return quad


def _panel_geometry(grid, panel_size, panel_widths, panel_heights,
                    margins, spacing, units, dpi, gridspec_kw):
    """Derive the canvas and gridspec margins from absolute panel sizes.

    Returns ``None`` when no panel size was asked for, which leaves the caller
    on the ordinary figure-sized path.

    A gridspec positions its panels in figure fractions, so fixing a panel in
    millimetres means solving for the fractions that reproduce it on a canvas
    that is itself derived from the panels. That is only well posed on a
    regular grid: on a mosaic a panel may span two columns, and "the panel
    width" is then not one number.
    """
    if panel_size is None and panel_widths is None and panel_heights is None:
        return None
    if grid is None:
        raise ValueError(
            "Panel sizes need a regular grid — pass `(nrows, ncols)` or a "
            "panel count. A mosaic panel can span several columns, so it has "
            "no single width to fix."
        )
    for ratio_key in ("width_ratios", "height_ratios"):
        if ratio_key in gridspec_kw:
            raise ValueError(
                f"`{ratio_key}` is relative and panel sizes are absolute — "
                f"give one or the other. Per-column sizes go in `panel_widths` "
                f"/ `panel_heights`."
            )

    nrows, ncols = grid
    if panel_size is not None:
        if panel_widths is not None or panel_heights is not None:
            raise ValueError(
                "`panel_size` sets every panel; use `panel_widths` / "
                "`panel_heights` instead to vary them per column or row."
            )
        pw, ph = _as_pair(panel_size, "panel_size")
        widths = [pw] * ncols
        heights = [ph] * nrows
    else:
        if panel_widths is None or panel_heights is None:
            raise ValueError(
                "Give both `panel_widths` and `panel_heights` (or use "
                "`panel_size`) — a panel needs both dimensions fixed for the "
                "canvas to be derivable."
            )
        widths = [float(v) for v in panel_widths]
        heights = [float(v) for v in panel_heights]
        if len(widths) != ncols or len(heights) != nrows:
            raise ValueError(
                f"`panel_widths` needs {ncols} values (one per column) and "
                f"`panel_heights` needs {nrows} (one per row); got "
                f"{len(widths)} and {len(heights)}."
            )
    if min(widths) <= 0 or min(heights) <= 0:
        raise ValueError("Panel sizes must be positive.")

    left_m, right_m, bottom_m, top_m = _as_margins(margins)
    h_gap, v_gap = _as_pair(spacing, "spacing")

    panels_w = sum(widths) + h_gap * (ncols - 1)
    panels_h = sum(heights) + v_gap * (nrows - 1)
    fig_w = panels_w + left_m + right_m
    fig_h = panels_h + bottom_m + top_m
    if fig_w <= 0 or fig_h <= 0:
        raise ValueError(
            f"Margins leave no room for the panels: figure would be "
            f"{fig_w}x{fig_h} {units}."
        )

    factor = _to_inch_factor(units, dpi)
    # gridspec's left/right/bottom/top are figure fractions; wspace/hspace are
    # gap-over-mean-panel. Converting here is what keeps the drawn panel equal
    # to the number the caller asked for.
    mean_w = sum(widths) / ncols
    mean_h = sum(heights) / nrows
    return {
        "figsize_in": (fig_w * factor, fig_h * factor),
        "gridspec_kw": {
            "left": left_m / fig_w,
            "right": 1.0 - right_m / fig_w,
            "bottom": bottom_m / fig_h,
            "top": 1.0 - top_m / fig_h,
            "wspace": (h_gap / mean_w) if ncols > 1 else 0.0,
            "hspace": (v_gap / mean_h) if nrows > 1 else 0.0,
            "width_ratios": widths,
            "height_ratios": heights,
        },
    }


@register_function(
    aliases=["画布", "figure", "figure_size", "出版尺寸", "期刊尺寸", "journal figure"],
    category="pl",
    description=(
        "Create a figure sized in mm/cm/inches/pixels, or by journal column "
        "preset (e.g. 'nature-single' = 89 mm), with publication defaults"
    ),
    examples=[
        "# A single-column Nature figure, 60 mm tall",
        "fig = ov.pl.figure('nature-single', 60)",
        "ax = fig.add_subplot(111)",
        "# Explicit units",
        "fig, ax = ov.pl.figure(89, 60, units='mm', dpi=300, axes=True)",
        "# Pixel-exact canvas for a slide",
        "fig = ov.pl.figure(1200, 800, units='px', dpi=150)",
    ],
    related=["pl.multipanel", "pl.savefig", "pl.add_panel_label", "pl.plot_set"],
)
def figure(width: Union[float, str, None] = None,
           height: Union[float, str, None] = None,
           *,
           units: str = "mm",
           dpi: float = 300,
           aspect: Optional[float] = None,
           axes: bool = False,
           editable_text: bool = True,
           constrained: bool = True,
           **kwargs: Any):
    r"""Create a figure with an exactly specified physical size.

    Arguments
    ---------
    width, height
        Size in ``units``. ``width`` may instead be a journal preset name —
        see :data:`JOURNAL_WIDTH_MM` (``'nature-single'``, ``'cell-double'``,
        ...). If only one of the two is given, the other follows ``aspect``.
    units
        ``'mm'`` (default), ``'cm'``, ``'in'`` or ``'px'``. Pixels are
        converted through ``dpi``, so ``figure(1200, 800, units='px',
        dpi=150)`` produces exactly a 1200x800 PNG.
    dpi
        Figure resolution. 300 is the usual journal minimum for raster
        panels; it does not affect vector output.
    aspect
        Width / height, used only when one dimension is omitted (default 1.4).
    axes
        If ``True`` return ``(fig, ax)`` with a single axes already added.
    editable_text
        Also set this figure's SVG/PDF font handling so exported text stays
        editable. Applied globally to rcParams — call
        :func:`set_editable_text` to control that separately.
    constrained
        Use matplotlib's constrained layout, which keeps labels inside the
        requested physical size instead of letting them overflow.

    Returns
    -------
    ``matplotlib.figure.Figure``, or ``(Figure, Axes)`` when ``axes=True``.
    """
    if editable_text:
        set_editable_text(True)
    w_in, h_in = _figsize_inches(width, height, units, dpi, aspect)
    kwargs.setdefault("layout", "constrained" if constrained else None)
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi, **kwargs)
    if axes:
        return fig, fig.add_subplot(111)
    return fig


def _panel_labels(n: int, label: Union[bool, str, Sequence[str]]) -> Optional[List[str]]:
    """Resolve the ``label=`` argument of :func:`multipanel` into a list."""
    if label is False or label is None:
        return None
    if isinstance(label, str):
        if label in {"a", "lower", "lowercase"}:
            base = _PANEL_LABEL_ALPHABET
        elif label in {"A", "upper", "uppercase"}:
            base = _PANEL_LABEL_ALPHABET.upper()
        else:
            raise ValueError(
                "`label` as a string must be 'a'/'lower' or 'A'/'upper'; "
                "pass a list for custom labels."
            )
        if n > len(base):
            raise ValueError(f"Cannot auto-label {n} panels with 26 letters.")
        return list(base[:n])
    if label is True:
        return list(_PANEL_LABEL_ALPHABET[:n])
    labels = list(label)
    if len(labels) < n:
        raise ValueError(f"Got {len(labels)} labels for {n} panels.")
    return labels[:n]


@register_function(
    aliases=["多面板", "multipanel", "panel_grid", "subplot_mosaic", "组合图", "拼图"],
    category="pl",
    description=(
        "Lay out a labelled multi-panel figure (grid or mosaic) at a physical "
        "size, with automatic a/b/c panel tags. Size the canvas with `width`, "
        "or size the panels themselves with `panel_size` when a journal "
        "specifies the panel rather than the figure"
    ),
    examples=[
        "# 2x2 grid, 183 mm wide (Nature double column)",
        "fig, ax = ov.pl.multipanel((2, 2), width='nature-double', height=120)",
        "ov.pl.volcano(deg, ax=ax['a'])",
        "# Panels fixed at 45x100 mm each; the canvas is derived from them",
        "fig, ax = ov.pl.multipanel((2, 3), panel_size=(45, 100))",
        "# Per-column widths, a figure number, and a colour cycle bound to a panel",
        "fig, ax = ov.pl.multipanel((1, 3), panel_widths=[45, 60, 30],",
        "                           panel_heights=[80], title='Figure 1',",
        "                           panel_styles={'a': {'color_cycle': ['#4C72B0', '#DD8452']}})",
        "# Mosaic: a wide panel on top of two narrow ones",
        "fig, ax = ov.pl.multipanel('AA\\nBC', width=183, height=110)",
        "ax['A'].plot(x, y)",
        "# Uppercase tags, custom ratios",
        "fig, ax = ov.pl.multipanel((1, 3), width=183, height=55, label='A',",
        "                           width_ratios=[2, 1, 1])",
    ],
    related=["pl.figure", "pl.add_panel_label", "pl.savefig"],
)
def multipanel(layout: Union[int, Tuple[int, int], str, Sequence[Sequence[Any]]],
               *,
               width: Union[float, str, None] = None,
               height: Union[float, str, None] = None,
               units: str = "mm",
               dpi: float = 300,
               aspect: Optional[float] = None,
               label: Union[bool, str, Sequence[str]] = True,
               label_kw: Optional[Mapping[str, Any]] = None,
               sharex: bool = False,
               sharey: bool = False,
               editable_text: bool = True,
               constrained: bool = True,
               panel_size: Optional[Tuple[float, float]] = None,
               panel_widths: Optional[Sequence[float]] = None,
               panel_heights: Optional[Sequence[float]] = None,
               margins: Union[float, Tuple[float, float, float, float]] = 14.0,
               spacing: Union[float, Tuple[float, float]] = 8.0,
               title: Optional[str] = None,
               title_kw: Optional[Mapping[str, Any]] = None,
               panel_styles: Optional[Mapping[Any, Mapping[str, Any]]] = None,
               **gridspec_kw: Any):
    r"""Build a multi-panel figure and label the panels.

    Arguments
    ---------
    layout
        One of

        * ``(nrows, ncols)`` — a regular grid;
        * ``n`` — that many panels in a near-square grid;
        * a mosaic string such as ``'AA\nBC'`` or a nested list, passed to
          ``Figure.subplot_mosaic``. ``'.'`` leaves a slot empty.
    width, height, units, dpi, aspect
        Figure geometry — identical to :func:`figure`, including the journal
        width presets.
    label
        ``True`` for lowercase ``a, b, c``, ``'A'`` for uppercase, a list for
        custom text, ``False`` to skip. Panels are labelled in reading order.
    label_kw
        Extra keyword arguments forwarded to :func:`add_panel_label`.
    sharex, sharey
        Share axes across panels.
    panel_size, panel_widths, panel_heights
        Size the **panels** instead of the figure, in ``units``. A journal that
        specifies a panel ("each panel 45 mm wide") cannot be satisfied by
        ``width`` plus ``width_ratios``, because those fix the canvas and let
        the panels fall where they may — the drawn panel then depends on how
        much room the tick labels happened to need.

        ``panel_size=(w, h)`` gives every panel that size; ``panel_widths`` /
        ``panel_heights`` give per-column and per-row sizes. The figure is
        sized to fit: ``sum(panels) + spacing + margins``.

        This turns constrained layout **off** — the two are contradictory, one
        reflows to fit content and the other refuses to. That makes ``margins``
        load-bearing: anything outside the axes (tick labels, y-labels, panel
        tags) has to fit in them, and will be clipped rather than accommodated.
        Grid layouts only; a mosaic's spans have no single panel width.
    margins
        Space outside the panels, in ``units`` — one number, or
        ``(left, right, bottom, top)``. Only consulted when panels are sized.
    spacing
        Gap between panels, in ``units`` — one number, or ``(horizontal,
        vertical)``. Only consulted when panels are sized.
    title
        Figure title, e.g. ``'Figure 1'``, drawn with ``fig.suptitle``.
    title_kw
        Extra keyword arguments for the title, e.g. ``{'x': 0.02, 'ha': 'left'}``
        to left-align it.
    panel_styles
        Per-panel settings, keyed the same way ``axes`` is. ``color_cycle``
        calls :meth:`~matplotlib.axes.Axes.set_prop_cycle`; every other key is
        passed to :meth:`~matplotlib.axes.Axes.set`, so ``title``, ``xlabel``,
        ``xlim`` and friends all work::

            panel_styles={'a': {'color_cycle': ['#4C72B0', '#DD8452'],
                                'title': 'Baseline'}}

        Binding the cycle to the panel means the plotting calls that follow do
        not each have to carry the colour.
    **gridspec_kw
        Passed to the gridspec — e.g. ``width_ratios``, ``height_ratios``,
        ``wspace``, ``hspace``. ``width_ratios`` / ``height_ratios`` are
        relative and are therefore rejected alongside ``panel_widths`` /
        ``panel_heights``, which are absolute.

    Returns
    -------
    ``(fig, axes)`` where ``axes`` is a dict.

    A panel is always reachable by **the tag printed on it** — ``axes['a']``,
    ``axes['b']``, ... For a mosaic the mosaic's own keys work too, so
    ``'AAB\nCDB'`` gives both ``axes['A']`` and ``axes['a']`` for the same
    axes. With ``label=False`` a grid is keyed by ``(row, col)`` tuples.
    """
    if isinstance(layout, int):
        import math

        ncols = int(math.ceil(math.sqrt(layout)))
        nrows = int(math.ceil(layout / ncols))
        grid: Optional[Tuple[int, int]] = (nrows, ncols)
        n_panels = layout
    elif isinstance(layout, tuple) and len(layout) == 2 and all(
        isinstance(v, int) for v in layout
    ):
        grid = (int(layout[0]), int(layout[1]))
        n_panels = grid[0] * grid[1]
    else:
        grid = None
        n_panels = 0

    if editable_text:
        set_editable_text(True)

    sized_panels = _panel_geometry(
        grid, panel_size, panel_widths, panel_heights,
        margins, spacing, units, dpi, gridspec_kw,
    )
    if sized_panels is None:
        w_in, h_in = _figsize_inches(width, height, units, dpi, aspect)
        fig = plt.figure(figsize=(w_in, h_in), dpi=dpi,
                         layout="constrained" if constrained else None)
    else:
        # Panels are absolute, so the canvas is derived and constrained layout
        # must stay off — it exists precisely to move panels around.
        if width is not None or height is not None:
            raise ValueError(
                "Give either the figure size (`width`/`height`) or the panel "
                "size (`panel_size`/`panel_widths`/`panel_heights`), not both "
                "— with panels sized, the figure size is derived from them."
            )
        fig = plt.figure(figsize=sized_panels["figsize_in"], dpi=dpi, layout=None)

    if grid is not None:
        nrows, ncols = grid
        labels = _panel_labels(n_panels, label)
        keys = labels if labels is not None else [
            (r, c) for r in range(nrows) for c in range(ncols)
        ]
        if sized_panels is not None:
            gridspec_kw = dict(gridspec_kw, **sized_panels["gridspec_kw"])
        gs = fig.add_gridspec(nrows, ncols, **gridspec_kw)
        axes: Dict[Any, Any] = {}
        first = None
        for idx in range(n_panels):
            r, c = divmod(idx, ncols)
            ax = fig.add_subplot(
                gs[r, c],
                sharex=first if sharex else None,
                sharey=first if sharey else None,
            )
            if first is None:
                first = ax
            axes[keys[idx]] = ax
        ordered = list(axes.values())
    else:
        mosaic_kw = dict(gridspec_kw)
        axes = fig.subplot_mosaic(layout, sharex=sharex, sharey=sharey,
                                  gridspec_kw=mosaic_kw or None)
        # Reading order: sort by the top-left corner of each panel.
        ordered_items = sorted(
            axes.items(),
            key=lambda kv: (-kv[1].get_position().y1, kv[1].get_position().x0),
        )
        axes = dict(ordered_items)
        ordered = [ax for _, ax in ordered_items]
        labels = _panel_labels(len(ordered), label)

    if labels is not None:
        for ax, text in zip(ordered, labels):
            add_panel_label(ax, text, **(dict(label_kw) if label_kw else {}))
        # A mosaic is keyed by its own letters ('A'), but the tag drawn on the
        # panel is 'a' — so `ax['a']` failed on a mosaic and worked on a grid.
        # Alias the printed tag onto the same axes unless the mosaic already
        # claims that key, so a panel is always reachable by what the reader
        # sees on it.
        for ax, text in zip(ordered, labels):
            if text not in axes:
                axes[text] = ax

    if title is not None:
        fig.suptitle(title, **(dict(title_kw) if title_kw else {}))

    if panel_styles:
        unknown = [k for k in panel_styles if k not in axes]
        if unknown:
            raise KeyError(
                f"panel_styles names panels that do not exist: {sorted(map(str, unknown))}. "
                f"Available: {sorted(map(str, axes))}."
            )
        # Iterate over the axes, not the mapping: a panel reachable under two
        # keys ('A' and 'a') would otherwise have its style applied twice, and
        # a prop cycle applied twice is silently reset to its start.
        applied: set = set()
        for key, style in panel_styles.items():
            ax = axes[key]
            if id(ax) in applied:
                continue
            applied.add(id(ax))
            settings = dict(style)
            cycle = settings.pop("color_cycle", None)
            if cycle is not None:
                ax.set_prop_cycle(color=list(cycle))
            if settings:
                ax.set(**settings)

    return fig, axes


@register_function(
    aliases=["面板标签", "add_panel_label", "panel_label", "子图标签", "abc标注"],
    category="pl",
    description=(
        "Add an a/b/c panel tag outside the top-left corner of an axes, "
        "offset in typographic points so it stays put at any figure size"
    ),
    examples=[
        "ov.pl.add_panel_label(ax, 'a')",
        "# Nudge further out, larger",
        "ov.pl.add_panel_label(ax, 'b', dx=-28, dy=8, fontsize=10)",
        "# Include the label inside the axes instead",
        "ov.pl.add_panel_label(ax, 'c', dx=4, dy=-4, ha='left', va='top')",
    ],
    related=["pl.multipanel", "pl.figure"],
)
def add_panel_label(ax,
                    label: str,
                    *,
                    dx: float = -20.0,
                    dy: float = 4.0,
                    fontsize: Optional[float] = None,
                    fontweight: str = "bold",
                    fontfamily: Optional[str] = None,
                    ha: str = "right",
                    va: str = "baseline",
                    color: str = "black",
                    **kwargs: Any):
    r"""Tag an axes with a panel letter.

    The tag is anchored to the axes' top-left corner and displaced by
    ``(dx, dy)`` **points**, not axes fractions — so the same offset looks the
    same on a 40 mm and a 180 mm panel, which is not true of the usual
    ``ax.text(-0.1, 1.05, ...)`` idiom.

    Arguments
    ---------
    ax
        Target axes.
    label
        Text of the tag, e.g. ``'a'``.
    dx, dy
        Offset from the top-left corner, in points. Negative ``dx`` moves left
        (into the margin), positive ``dy`` moves up.
    fontsize
        Defaults to ``rcParams['axes.titlesize']`` resolved to points.

    Returns
    -------
    The created ``matplotlib.text.Annotation``.
    """
    if fontsize is None:
        from matplotlib.font_manager import FontProperties

        fontsize = FontProperties(
            size=plt.rcParams.get("axes.titlesize", "medium")
        ).get_size_in_points()
    return ax.annotate(
        str(label),
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        fontweight=fontweight,
        fontfamily=fontfamily,
        color=color,
        ha=ha,
        va=va,
        annotation_clip=False,
        **kwargs,
    )


@register_function(
    aliases=["可编辑文字", "set_editable_text", "editable_svg", "矢量字体", "illustrator"],
    category="pl",
    description=(
        "Keep text as editable text (not outlines) in SVG/PDF/PS exports, "
        "so figures can be relabelled in Illustrator or Inkscape"
    ),
    examples=[
        "ov.pl.set_editable_text()        # affects every later plt.savefig",
        "old = ov.pl.set_editable_text(True)",
        "ov.pl.set_editable_text(False)   # back to outlined glyphs",
    ],
    related=["pl.savefig", "pl.figure", "pl.plot_set"],
)
def set_editable_text(enable: bool = True) -> Dict[str, Any]:
    r"""Switch vector text between *editable* and *outlined* globally.

    matplotlib's SVG backend converts glyphs to paths by default, which makes
    an exported figure impossible to relabel; PDF defaults to Type-3 fonts,
    which Illustrator refuses to edit. This sets
    ``svg.fonttype='none'`` and ``pdf/ps.fonttype=42`` (TrueType).

    Note that ``svg.fonttype='none'`` means the SVG *references* the font by
    name rather than embedding it — whoever opens the file needs that font
    installed, or the viewer substitutes one. That is the trade for editability.

    Returns the previous values, so they can be restored.
    """
    previous = {k: plt.rcParams.get(k) for k in EDITABLE_TEXT_RCPARAMS}
    if enable:
        plt.rcParams.update(EDITABLE_TEXT_RCPARAMS)
    else:
        plt.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 3,
                             "ps.fonttype": 3})
    return previous


@register_function(
    aliases=["保存图形", "savefig", "save_figure", "导出图形", "export_figure"],
    category="pl",
    description=(
        "Save a figure to one or several formats at once with publication "
        "defaults and text that stays editable in vector output"
    ),
    examples=[
        "ov.pl.savefig(fig, 'figure1.pdf')",
        "# One call, three files: figure1.pdf / .svg / .png",
        "ov.pl.savefig(fig, 'figure1', formats=['pdf', 'svg', 'png'], dpi=600)",
        "# Current figure, no explicit handle",
        "ov.pl.savefig('figure1.svg')",
    ],
    related=["pl.figure", "pl.multipanel", "pl.set_editable_text"],
)
def savefig(fig: Union[Figure, str, "os.PathLike[str]"],
            path: Union[str, "os.PathLike[str]", None] = None,
            *,
            dpi: Union[float, str] = 300,
            formats: Optional[Iterable[str]] = None,
            editable_text: bool = True,
            transparent: bool = False,
            bbox_inches: Optional[str] = "tight",
            pad_inches: float = 0.02,
            verbose: bool = True,
            **kwargs: Any) -> List[str]:
    r"""Write a figure to disk, keeping vector text editable.

    Arguments
    ---------
    fig
        The figure. May be omitted by passing the path as the first argument,
        in which case the current figure (``plt.gcf()``) is used.
    path
        Output path. With ``formats`` given, any extension on ``path`` is
        replaced by each requested format.
    dpi
        Raster resolution; ignored for pure-vector formats.
    formats
        Write several files from one figure, e.g. ``['pdf', 'svg', 'png']``.
    editable_text
        Apply :data:`EDITABLE_TEXT_RCPARAMS` for the duration of the save
        only, leaving the global rcParams untouched.
    bbox_inches
        ``'tight'`` crops to the drawn content. Pass ``None`` to preserve the
        exact physical size requested from :func:`figure` — which matters when
        a journal asks for a figure of a specific width.

    Returns
    -------
    List of the paths written.
    """
    if isinstance(fig, (str, os.PathLike)):
        if path is not None:
            raise TypeError(
                "Pass either savefig(fig, path) or savefig(path), not both."
            )
        path, fig = fig, plt.gcf()
    if path is None:
        raise TypeError("`path` is required.")

    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    if formats:
        stem = os.path.splitext(path)[0]
        targets = [f"{stem}.{str(f).lstrip('.').lower()}" for f in formats]
    else:
        targets = [path]

    context = EDITABLE_TEXT_RCPARAMS if editable_text else {}
    written: List[str] = []
    with plt.rc_context(context):
        for target in targets:
            fig.savefig(target, dpi=dpi, transparent=transparent,
                        bbox_inches=bbox_inches, pad_inches=pad_inches,
                        **kwargs)
            written.append(target)
    if verbose:
        print("Saved: " + ", ".join(written))
    return written


@register_function(
    aliases=["图例外移", "take_legend_out", "legend_outside", "移出图例"],
    category="pl",
    description=(
        "Move an axes legend outside the plotting area (right/left/top/bottom) "
        "without shrinking the data region by hand"
    ),
    examples=[
        "ov.pl.take_legend_out(ax)                  # to the right",
        "ov.pl.take_legend_out(ax, loc='bottom', ncol=4)",
        "ov.pl.take_legend_out(ax, loc='top', title='Cell type')",
    ],
    related=["pl.multipanel", "pl.figure"],
)
def take_legend_out(ax,
                    loc: str = "right",
                    *,
                    pad: float = 0.02,
                    ncol: Optional[int] = None,
                    title: Optional[str] = None,
                    frameon: bool = False,
                    fontsize: Optional[float] = None,
                    handles=None,
                    labels=None,
                    **kwargs: Any):
    r"""Re-place an axes legend outside the axes.

    Arguments
    ---------
    ax
        Axes whose legend should move. Its current handles/labels are reused
        unless ``handles``/``labels`` are given explicitly.
    loc
        ``'right'`` (default), ``'left'``, ``'top'`` or ``'bottom'``.
    pad
        Gap between axes and legend, in axes fractions.
    ncol
        Legend columns. Defaults to 1 for left/right and to the number of
        entries for top/bottom, which is what usually reads best.

    Returns
    -------
    The new ``matplotlib.legend.Legend``.
    """
    if handles is None or labels is None:
        cur_handles, cur_labels = ax.get_legend_handles_labels()
        handles = cur_handles if handles is None else handles
        labels = cur_labels if labels is None else labels
    if not handles:
        raise ValueError(
            "No legend entries on this axes — plot with `label=` first, or "
            "pass `handles=`/`labels=` explicitly."
        )

    anchors = {
        "right": ((1.0 + pad, 0.5), "center left"),
        "left": ((-pad, 0.5), "center right"),
        "top": ((0.5, 1.0 + pad), "lower center"),
        "bottom": ((0.5, -pad), "upper center"),
    }
    if loc not in anchors:
        raise ValueError(
            f"`loc` must be one of {sorted(anchors)}, got {loc!r}."
        )
    bbox, legend_loc = anchors[loc]
    if ncol is None:
        ncol = len(labels) if loc in {"top", "bottom"} else 1

    existing = ax.get_legend()
    if existing is not None:
        existing.remove()
    return ax.legend(handles, labels, loc=legend_loc, bbox_to_anchor=bbox,
                     ncol=ncol, title=title, frameon=frameon,
                     fontsize=fontsize, **kwargs)
