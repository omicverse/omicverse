import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from .._registry import register_function


def _build_upset_intersections(sets, min_size=1):
    names = list(sets.keys())
    normalized = {name: set(values) for name, values in sets.items()}
    records = []

    for mask in range(1, 2 ** len(names)):
        included = [names[i] for i in range(len(names)) if mask & (1 << i)]
        excluded = [name for name in names if name not in included]
        elements = set.intersection(*(normalized[name] for name in included))
        if excluded:
            elements = elements.difference(*(normalized[name] for name in excluded))
        if len(elements) >= min_size:
            records.append(
                {
                    "sets": tuple(included),
                    "degree": len(included),
                    "size": len(elements),
                    "elements": elements,
                }
            )

    return pd.DataFrame.from_records(records, columns=["sets", "degree", "size", "elements"])


def _is_adata_like(data):
    return all(hasattr(data, attr) for attr in ("obs", "var", "obs_names", "var_names"))


def _coerce_upset_sets(data, keys=None, axis="obs"):
    if isinstance(data, dict):
        if keys is None:
            return data
        missing = [key for key in keys if key not in data]
        if missing:
            raise KeyError(f"Keys not found in dictionary input: {', '.join(map(str, missing))}")
        return {key: data[key] for key in keys}

    if not _is_adata_like(data):
        raise ValueError("`sets` must be a dictionary or an AnnData object.")

    if axis not in {"obs", "var"}:
        raise ValueError("`axis` must be either 'obs' or 'var'.")

    frame = data.obs if axis == "obs" else data.var
    names = data.obs_names if axis == "obs" else data.var_names

    if keys is None:
        keys = [
            column
            for column in frame.columns
            if pd.api.types.is_bool_dtype(frame[column].dropna())
        ]
        if not keys:
            raise ValueError(
                "AnnData input requires `keys` unless `adata.obs` or `adata.var` "
                "contains boolean columns."
            )

    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise KeyError(f"Keys not found in adata.{axis}: {', '.join(map(str, missing))}")

    result = {}
    for key in keys:
        series = frame[key]
        if not pd.api.types.is_bool_dtype(series.dropna()):
            raise TypeError(
                f"`adata.{axis}[{key!r}]` must be boolean to define an UpSet set."
            )
        mask = series.fillna(False).astype(bool).to_numpy()
        result[str(key)] = set(pd.Index(names)[mask])
    return result


def _apply_ov_style(style):
    if style in (None, False):
        return
    if style is True:
        from ._plot_backend import style as ov_style

        ov_style(show_monitor=False)
        return
    if isinstance(style, dict):
        from ._plot_backend import style as ov_style

        kwargs = dict(style)
        kwargs.setdefault("show_monitor", False)
        ov_style(**kwargs)
        return
    if callable(style):
        try:
            style(show_monitor=False)
        except TypeError:
            style()
        return
    raise TypeError("`style` must be None, bool, dict, or a callable such as `ov.style`.")


def _upset_colors(empty_color, line_color):
    cycle = rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["#30343b", "#7aa6c2", "#b7c7d8"]
    return {
        "intersection_color": cycle[0],
        "set_size_color": cycle[min(1, len(cycle) - 1)],
        "matrix_color": cycle[0],
        "empty_color": empty_color or "#d9dde1",
        "line_color": line_color or "#9aa1a8",
    }


def _canonical_intersection_key(key):
    if isinstance(key, str):
        parts = key.split("&")
    else:
        parts = list(key)
    return tuple(sorted(map(str, parts)))


def _resolve_column_colors(color, intersections, default):
    if color is None:
        return default
    if isinstance(color, str):
        return color
    if isinstance(color, dict):
        color_map = {
            _canonical_intersection_key(key): value
            for key, value in color.items()
        }
        return [
            color_map.get(_canonical_intersection_key(names), default)
            for names in intersections["sets"]
        ]

    values = list(color)
    if not values:
        return default
    repeats = int(np.ceil(len(intersections) / len(values)))
    return (values * repeats)[: len(intersections)]


def _resolve_set_colors(color, set_names, default):
    if color is None:
        return [default] * len(set_names)
    if isinstance(color, str):
        return [color] * len(set_names)
    if isinstance(color, dict):
        return [color.get(name, default) for name in set_names]

    values = list(color)
    if not values:
        return [default] * len(set_names)
    repeats = int(np.ceil(len(set_names) / len(values)))
    return (values * repeats)[: len(set_names)]


def _normalize_intersection_key(key):
    return _canonical_intersection_key(key)


def _apply_intersection_order(intersections, intersection_order):
    if intersection_order is None:
        return intersections

    order = {
        _normalize_intersection_key(key): idx
        for idx, key in enumerate(intersection_order)
    }
    intersections = intersections.copy()
    intersections["_manual_order"] = intersections["sets"].map(
        lambda values: order.get(_canonical_intersection_key(values), len(order))
    )
    return (
        intersections.sort_values(
            ["_manual_order", "size", "degree", "_sort_key"],
            ascending=[True, False, False, True],
        )
        .drop(columns="_manual_order")
    )


@register_function(
    aliases=["upset", "upset_plot", "UpSet", "集合交集图", "多集合交集图"],
    category="pl",
    description="Create an UpSet plot to visualize intersections across many sets",
    examples=[
        "# Basic UpSet plot from a dictionary",
        "sets = {'A': {'g1','g2'}, 'B': {'g2','g3'}, 'C': {'g2','g4'}}",
        "fig, axes = ov.pl.upset(sets)",
        "# Cell-level UpSet plot from AnnData boolean obs columns",
        "fig, axes = ov.pl.upset(adata, keys=['high_CD3D', 'cycling'], axis='obs')",
        "# Apply OmicVerse style just for this plot",
        "fig, axes = ov.pl.upset(adata, keys=keys, style=ov.style)",
    ],
    related=["pl.venn"]
)
def upset(
    sets,
    keys=None,
    axis="obs",
    top_n=30,
    min_size=1,
    sort_by="size",
    figsize=None,
    style=None,
    intersection_color=None,
    set_size_color=None,
    matrix_color=None,
    empty_color=None,
    line_color=None,
    title="",
    show_counts=True,
    grid=True,
    bar_width=0.72,
    set_bar_height=0.58,
    dot_size=42,
    empty_dot_size=34,
    line_width=1.3,
    grid_color="#e7eaee",
    grid_linewidth=0.8,
    count_fontsize=None,
    count_offset=None,
    matrix_alpha=1.0,
    empty_alpha=1.0,
    set_order=None,
    intersection_order=None,
    height_ratios=None,
    max_sets=12,
):
    r"""
    Create an UpSet plot to visualize overlaps across multiple sets.

    Parameters
    ----------
    sets : dict or AnnData
        Dictionary mapping set names to iterables of elements, or an AnnData
        object with boolean columns in ``adata.obs`` or ``adata.var``.
    keys : list or None
        For AnnData input, boolean columns to use as sets. For dictionary input,
        optional subset/reordering of dictionary keys.
    axis : {"obs", "var"}
        AnnData axis used when ``sets`` is an AnnData object. ``"obs"`` plots
        cell-set intersections; ``"var"`` plots feature/gene-set intersections.
    top_n : int or None
        Maximum number of non-empty intersections to show. ``None`` shows all.
    min_size : int
        Minimum exclusive intersection size to retain.
    sort_by : {"size", "degree"}
        Sort intersections by size or by number of participating sets.
    figsize : tuple or None
        Figure size. If ``None``, size is chosen from the number of displayed
        intersections and input sets.
    style : None, bool, dict, or callable
        Optional OmicVerse style application. Use ``style=ov.style`` to apply
        the package plotting style before drawing, ``style=True`` to call
        ``ov.pl.style(show_monitor=False)``, or a dict of keyword arguments to
        pass to ``ov.pl.style``.
    intersection_color : str, sequence, or dict
        Color for the top intersection-size bars. A sequence colors bars by
        displayed column order; a dict can map ``("A", "B")`` or ``"A&B"`` to
        colors.
    set_size_color : str, sequence, or dict
        Color for the left set-size bars. A dict maps set names to colors and
        leaves unspecified rows with the default color.
    matrix_color : str, sequence, or dict
        Color for active dots in the intersection matrix. A dict maps set names
        to row colors and leaves unspecified rows with the default color.
    empty_color : str
        Color for inactive dots in the intersection matrix.
    line_color : str
        Color for vertical lines connecting active dots.
    title : str
        Optional figure title. Defaults to an empty title.
    show_counts : bool
        Whether to annotate bar counts.
    grid : bool
        Whether to show a light y-axis grid behind intersection bars.
    bar_width : float
        Width of the top intersection-size bars.
    set_bar_height : float
        Height of the left set-size bars.
    dot_size : float
        Size of active dots in the intersection matrix.
    empty_dot_size : float
        Size of inactive dots in the intersection matrix.
    line_width : float
        Width of vertical lines connecting active dots.
    grid_color : str
        Color for the intersection and set-size grids.
    grid_linewidth : float
        Line width for the intersection and set-size grids.
    count_fontsize : float or None
        Font size for bar count labels. Defaults to a value derived from
        ``matplotlib.rcParams``.
    count_offset : float or None
        Vertical offset for bar count labels. Defaults to a small fraction of
        the maximum intersection size.
    matrix_alpha : float
        Alpha value for active dots.
    empty_alpha : float
        Alpha value for inactive dots.
    set_order : sequence or None
        Optional set row order. Unlisted sets are appended in their original
        order.
    intersection_order : sequence or None
        Optional intersection column order. Entries can be ``"A&B"`` strings or
        tuples such as ``("A", "B")``. Unlisted intersections keep the default
        sorted order after the listed intersections.
    height_ratios : tuple or None
        Relative heights for the top intersection bars and lower matrix/set-size
        panels. If ``None``, the lower panel height is chosen from the number of
        sets.
    max_sets : int or None
        Maximum number of sets to enumerate. UpSet intersection enumeration is
        exponential in the number of sets. Use ``None`` to disable the guard.

    Returns
    -------
    tuple
        ``(fig, axes)`` where ``axes`` is a dictionary containing
        ``"intersections"``, ``"matrix"``, and ``"set_sizes"`` axes.
    """
    _apply_ov_style(style)

    if min_size < 1:
        raise ValueError("`min_size` must be at least 1.")
    if top_n is not None and int(top_n) < 1:
        raise ValueError("`top_n` must be at least 1 or None.")
    if max_sets is not None and int(max_sets) < 1:
        raise ValueError("`max_sets` must be at least 1 or None.")

    normalized = _coerce_upset_sets(sets, keys=keys, axis=axis)
    if not isinstance(normalized, dict) or len(normalized) == 0:
        raise ValueError("`sets` must resolve to a non-empty dictionary of set names to iterables.")
    if max_sets is not None and len(normalized) > int(max_sets):
        raise ValueError(
            f"UpSet plots enumerate 2^n intersections; received {len(normalized)} sets. "
            f"Pass fewer `keys`, increase `max_sets`, or set `max_sets=None`."
        )

    set_names = list(normalized.keys())
    if set_order is not None:
        requested = [name for name in set_order if name in normalized]
        set_names = requested + [name for name in set_names if name not in requested]
    normalized = {name: set(values) for name, values in normalized.items()}
    if any(len(name_set) == 0 for name_set in normalized.values()):
        empty = [str(name) for name, values in normalized.items() if len(values) == 0]
        raise ValueError(f"All input sets must be non-empty; empty sets: {', '.join(empty)}")

    intersections = _build_upset_intersections(normalized, min_size=min_size)
    if intersections.empty:
        raise ValueError("No intersections meet `min_size`; lower `min_size` or check the input sets.")

    intersections["_sort_key"] = intersections["sets"].map(lambda values: "||".join(map(str, values)))
    if sort_by == "size":
        intersections = intersections.sort_values(["size", "degree", "_sort_key"], ascending=[False, False, True])
    elif sort_by == "degree":
        intersections = intersections.sort_values(["degree", "size", "_sort_key"], ascending=[False, False, True])
    else:
        raise ValueError("`sort_by` must be either 'size' or 'degree'.")
    intersections = _apply_intersection_order(intersections, intersection_order)

    if top_n is not None:
        intersections = intersections.head(int(top_n))
    intersections = intersections.drop(columns="_sort_key").reset_index(drop=True)

    if figsize is None:
        figsize = (max(6, 0.42 * len(intersections) + 2.5), max(4, 0.35 * len(set_names) + 2.8))
    if height_ratios is None:
        height_ratios = (2.5, max(1.4, 0.35 * len(set_names)))
    colors = _upset_colors(
        empty_color=empty_color,
        line_color=line_color,
    )
    intersection_colors = _resolve_column_colors(
        intersection_color,
        intersections,
        colors["intersection_color"],
    )
    set_size_colors = _resolve_set_colors(
        set_size_color,
        set_names,
        colors["set_size_color"],
    )
    matrix_colors = _resolve_set_colors(
        matrix_color,
        set_names,
        colors["matrix_color"],
    )
    label_fontsize = count_fontsize or max(8, rcParams["font.size"] * 0.68)

    fig = plt.figure(figsize=figsize, facecolor=rcParams["figure.facecolor"])
    layout = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.5, max(3.5, 0.42 * len(intersections))),
        height_ratios=height_ratios,
        hspace=0.10,
        wspace=0.05,
    )
    ax_empty = fig.add_subplot(layout[0, 0])
    ax_intersections = fig.add_subplot(layout[0, 1])
    ax_set_sizes = fig.add_subplot(layout[1, 0])
    ax_matrix = fig.add_subplot(layout[1, 1], sharex=ax_intersections)
    ax_empty.axis("off")

    x = np.arange(len(intersections))
    sizes = intersections["size"].to_numpy()
    ax_intersections.bar(x, sizes, color=intersection_colors, width=bar_width)
    ax_intersections.set_ylabel("Intersection size")
    ax_intersections.set_xticks([])
    ax_intersections.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ("top", "right"):
        ax_intersections.spines[spine].set_visible(False)
    if title:
        ax_intersections.set_title(title)
    if show_counts:
        offset = count_offset
        if offset is None:
            offset = max(sizes) * 0.02 if max(sizes) > 0 else 0.2
        for xpos, value in zip(x, sizes):
            ax_intersections.text(
                xpos,
                value + offset,
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                color=rcParams["text.color"],
            )
        ax_intersections.set_ylim(0, max(sizes) * 1.18)
    if grid:
        ax_intersections.yaxis.grid(True, color=grid_color, linewidth=grid_linewidth)
        ax_intersections.set_axisbelow(True)
    else:
        ax_intersections.xaxis.grid(False)
        ax_intersections.yaxis.grid(False)
        for line in [*ax_intersections.get_xgridlines(), *ax_intersections.get_ygridlines()]:
            line.set_visible(False)

    y = np.arange(len(set_names))
    set_sizes = [len(normalized[name]) for name in set_names]
    ax_set_sizes.barh(y, set_sizes, color=set_size_colors, height=set_bar_height)
    ax_set_sizes.set_yticks(y, labels=set_names)
    ax_set_sizes.invert_xaxis()
    ax_set_sizes.invert_yaxis()
    ax_set_sizes.set_xlabel("Set size")
    for spine in ("top", "right"):
        ax_set_sizes.spines[spine].set_visible(False)
    if grid:
        ax_set_sizes.xaxis.grid(True, color=grid_color, linewidth=grid_linewidth)
        ax_set_sizes.set_axisbelow(True)
    else:
        ax_set_sizes.xaxis.grid(False)
        ax_set_sizes.yaxis.grid(False)
        for line in [*ax_set_sizes.get_xgridlines(), *ax_set_sizes.get_ygridlines()]:
            line.set_visible(False)

    membership = np.zeros((len(set_names), len(intersections)), dtype=bool)
    name_to_row = {name: idx for idx, name in enumerate(set_names)}
    for col, group_names in enumerate(intersections["sets"]):
        rows = [name_to_row[name] for name in group_names]
        membership[rows, col] = True
        if len(rows) > 1:
            ax_matrix.plot([col, col], [min(rows), max(rows)], color=colors["line_color"], lw=line_width, zorder=1)

    for row in range(len(set_names)):
        ax_matrix.scatter(x, np.full_like(x, row), s=empty_dot_size, color=colors["empty_color"], alpha=empty_alpha, zorder=2)
        active = membership[row]
        if active.any():
            ax_matrix.scatter(x[active], np.full(active.sum(), row), s=dot_size, color=matrix_colors[row], alpha=matrix_alpha, zorder=3)

    ax_matrix.set_yticks([])
    ax_matrix.tick_params(axis="y", left=False, labelleft=False)
    ax_matrix.set_xticks(x, labels=[])
    ax_matrix.set_ylim(len(set_names) - 0.5, -0.5)
    ax_matrix.set_xlim(-0.6, len(intersections) - 0.4)
    ax_matrix.set_xlabel("Intersections")
    for spine in ("top", "right", "left"):
        ax_matrix.spines[spine].set_visible(False)

    axes = {
        "intersections": ax_intersections,
        "matrix": ax_matrix,
        "set_sizes": ax_set_sizes,
    }
    return fig, axes
