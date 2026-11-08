from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path

from .._registry import register_function


def _polar(radius, angle):
    theta = np.deg2rad(angle)
    return radius * np.cos(theta), radius * np.sin(theta)


@register_function(
    aliases=["celltype_tree", "radial cell annotation tree", "环形细胞注释树"],
    category="pl",
    description="Draw a radial cell annotation tree with optional numeric tracks.",
    examples=["ov.pl.celltype_tree(celltypes, hierarchy=['lineage', 'family'], label='cell_type')"],
    related=["pl.plot1cell", "pl.cellproportion"],
)
def celltype_tree(
    data: pd.DataFrame,
    hierarchy: Sequence[str],
    label: str,
    *,
    proportion: str | None = None,
    tracks: Sequence[str] | None = None,
    track_markers: Mapping[str, str] | None = None,
    palette: Mapping | None = None,
    cmap: str = "RdBu_r",
    vmax: float | None = None,
    figsize=(10, 10),
    ax: Axes | None = None,
    show: bool = True,
) -> Axes:
    """Plot a radial annotation tree with one leaf per row.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per leaf. The full path through ``hierarchy`` and ``label``
        must be unique and contain no missing values.
    hierarchy : sequence of str
        Classification columns, from the broadest level to the finest.
        Sibling order follows first appearance in ``data``. Descendants of
        each node occupy consecutive leaf positions.
    label : str
        Column used for leaf labels. The same label can occur in different
        branches.
    proportion : str, optional
        Column of percentages in [0, 100] to append to leaf labels.
    tracks : sequence of str, optional
        Numeric columns to draw as rings, from inside to outside. Each ring
        uses its own column. Missing values are shown in gray.
    track_markers : mapping of str to str, optional
        Map a track name to a column of text markers, such as significance
        stars computed upstream. Missing and empty markers are omitted.
    palette : mapping, optional
        Colors keyed by values in the first hierarchy column.
    cmap : str, default 'RdBu_r'
        Matplotlib colormap shared by the tracks.
    vmax : float, optional
        Positive limit for the shared color scale from ``-vmax`` to ``vmax``.
        By default, use the largest finite absolute value across the tracks,
        or 1 if all values are missing or zero. Use comparable units when
        displaying multiple tracks together. Values outside the range use
        the endpoint colors.
    figsize : tuple, default (10, 10)
        Size of a new figure, in inches.
    ax : matplotlib.axes.Axes, optional
        Draw on an existing axes.
    show : bool, default True
        Show the figure after drawing.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the tree and tracks.

    References
    ----------
    Inspired by the cell annotation tree in Yin et al., Chinese Immune
    Multi-Omics Atlas, Science (2026), https://doi.org/10.1126/science.adt3130,
    and the visualization walkthrough at
    https://mp.weixin.qq.com/s/F7cgHFmpEAhwhrzAN0NvUA.
    """
    levels = [hierarchy] if isinstance(hierarchy, str) else list(hierarchy)
    columns = [*levels, label]
    rings = [tracks] if isinstance(tracks, str) else list(tracks or ())
    markers = dict(track_markers or {})
    if not levels or len(set(columns)) != len(columns):
        raise ValueError("Specify distinct hierarchy columns and a separate leaf label column.")
    if len(set(rings)) != len(rings) or not markers.keys() <= set(rings):
        raise ValueError("Tracks must be distinct and marker keys must name tracks.")
    if data.empty or not data.columns.is_unique:
        raise ValueError("data must contain at least one row and unique column names.")
    required = {*columns, *rings, *markers.values()}
    if proportion is not None:
        required.add(proportion)
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    taxonomy = data[columns]
    if taxonomy.isna().any().any() or taxonomy.duplicated().any():
        raise ValueError("Leaf paths must be unique and contain no missing values.")

    paths = list(taxonomy.itertuples(index=False, name=None))
    ranks = {}
    for path in paths:
        for depth in range(1, len(path) + 1):
            ranks.setdefault(path[:depth], len(ranks))
    order = sorted(range(len(paths)), key=lambda i: tuple(
        ranks[paths[i][:depth]] for depth in range(1, len(columns) + 1)
    ))
    data = data.iloc[order]
    paths = [paths[i] for i in order]
    values = data[rings].apply(pd.to_numeric).to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Track values must be finite or missing.")
    percentages = None
    if proportion is not None:
        percentages = pd.to_numeric(data[proportion]).to_numpy(dtype=float, na_value=np.nan)
        if not np.all((percentages >= 0) & (percentages <= 100)):
            raise ValueError("Proportions must be percentages in [0, 100].")
    if vmax is None:
        vmax = np.max(np.abs(values[np.isfinite(values)]), initial=0.0) or 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("vmax must be positive and finite.")
    norm = Normalize(-vmax, vmax, clip=True)
    color_map = plt.get_cmap(cmap)
    lineages = list(dict.fromkeys(path[0] for path in paths))
    if palette is None:
        default = plt.get_cmap("tab10" if len(lineages) <= 10 else "tab20", len(lineages))
        colors = dict(zip(lineages, (default(i) for i in range(len(lineages)))))
    else:
        colors = {lineage: to_rgba(palette[lineage]) for lineage in lineages}

    spans = {}
    for i, path in enumerate(paths):
        for depth in range(1, len(path) + 1):
            spans.setdefault(path[:depth], [i, i])[1] = i
    step = 326 / len(paths)
    angles = 73 - step * (np.arange(len(paths)) + 0.5)
    radii = np.linspace(0, 0.94, len(columns) + 1)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    nodes = {(): (0.0, 90.0)}
    for path, (first, last) in spans.items():
        depth = len(path)
        radius, angle = radii[depth], (angles[first] + angles[last]) / 2
        nodes[path] = (radius, angle)
        color = colors[path[0]]
        if depth == 1:
            ax.add_patch(Wedge((0, 0), 0.94, angles[last] - step / 2,
                               angles[first] + step / 2, facecolor=color, alpha=0.09, lw=0))
        parent_radius, parent_angle = nodes[path[:-1]]
        if depth == 1:
            parent_angle = angle
        mid = (parent_radius + radius) / 2
        vertices = [_polar(parent_radius, parent_angle), _polar(mid, parent_angle),
                    _polar(mid, angle), _polar(radius, angle)]
        ax.add_patch(PathPatch(Path(vertices, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                               facecolor="none", edgecolor=color, lw=max(0.7, 3.4 / depth),
                               capstyle="round", zorder=2))
        if depth == 1:
            ax.text(*_polar(radius * 0.84, angle), str(path[0]), color=color, weight="semibold",
                    ha="center", va="center", fontsize=9, zorder=3,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.3})

    leaf_half_width = step * 0.47
    label_radius = 1.035 + 0.105 * len(rings)
    for i, (path, angle) in enumerate(zip(paths, angles)):
        left, right = angle - leaf_half_width, angle + leaf_half_width
        ax.add_patch(Wedge((0, 0), 0.99, left, right, width=0.05,
                           facecolor=colors[path[0]], lw=0))
        for j, track in enumerate(rings):
            radius = 1.10 + 0.105 * j
            value = values[i, j]
            color = color_map(norm(value)) if np.isfinite(value) else to_rgba("#e0e0e0")
            ax.add_patch(Wedge((0, 0), radius, left, right, width=0.088, facecolor=color, lw=0))
            if track in markers:
                mark = data[markers[track]].iloc[i]
                if pd.notna(mark) and str(mark):
                    ink = "white" if np.dot(color[:3], [0.2126, 0.7152, 0.0722]) < 0.45 else "#222222"
                    ax.text(*_polar(radius - 0.044, angle), str(mark), color=ink,
                            fontsize=7, ha="center", va="center")
        name = str(path[-1])
        if percentages is not None:
            percent = f"{percentages[i]:.2f}".rstrip("0").rstrip(".")
            name = f"{name} ({percent}%)"
        rotation = (angle + 90) % 180 - 90
        ha = "left" if np.cos(np.deg2rad(angle)) >= 0 else "right"
        ax.text(*_polar(label_radius, angle), name, color=colors[path[0]], fontsize=8,
                rotation=rotation, rotation_mode="anchor", ha=ha, va="center")

    for j, track in enumerate(rings):
        ax.text(0, 1.056 + 0.105 * j, track.replace("_", " "), fontsize=7,
                ha="center", va="center", color="#333333")
    limit = label_radius + 0.57
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    ax.set_axis_off()
    if rings:
        bar_ax = ax.inset_axes((0.41, 0.915, 0.18, 0.012))
        bar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=color_map), cax=bar_ax,
                                 orientation="horizontal", ticks=[-vmax, 0, vmax])
        bar.ax.tick_params(labelsize=7, length=2)
        bar.ax.set_title("Track score", fontsize=7, pad=3)
        bar.outline.set_visible(False)
    if show:
        plt.show()
    return ax
