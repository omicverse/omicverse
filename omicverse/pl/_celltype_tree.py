from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path

from .._registry import register_function
from ._plotdata import get_values


def _polar(radius, angle):
    theta = np.deg2rad(angle)
    return radius * np.cos(theta), radius * np.sin(theta)


def _summarize_anndata(adata, columns, rings, markers, use_raw, layer):
    obs = adata.obs
    if adata.n_obs == 0 or not obs.columns.is_unique:
        raise ValueError("adata must contain at least one cell and unique obs column names.")
    missing = set(columns).union(markers.values()).difference(obs.columns)
    if missing:
        raise KeyError(f"Missing adata.obs columns: {sorted(missing)}")
    if obs[columns].isna().any().any():
        raise ValueError("Leaf paths in adata.obs must contain no missing values.")
    groups = [obs[column].astype(object) for column in columns]
    tracks = pd.DataFrame({
        ring: pd.to_numeric(get_values(adata, ring, use_raw=use_raw,
                                       layer=None if ring in obs.columns else layer))
        for ring in rings
    }, index=obs.index)
    if np.isinf(tracks.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Track values must be finite or missing.")
    grouped = tracks.groupby(groups, sort=False)
    counts = grouped.size()
    data = counts.index.to_frame(index=False)
    for ring, values in grouped.mean().items():
        if ring not in columns:
            data[ring] = values.to_numpy()
    for key in dict.fromkeys(markers.values()):
        if key in columns:
            continue
        grouped_markers = obs[key].groupby(groups, sort=False)
        if (grouped_markers.nunique(dropna=False) > 1).any():
            raise ValueError(f"Track marker column {key!r} must be constant within each leaf.")
        data[key] = grouped_markers.first().reindex(counts.index).to_numpy()
    return data, 100 * counts.to_numpy() / adata.n_obs


@register_function(
    aliases=["celltype_tree", "radial cell annotation tree", "环形细胞注释树"],
    category="pl",
    description="Draw a radial cell annotation tree with optional numeric tracks.",
    examples=["ov.pl.celltype_tree(adata, hierarchy=['lineage'], label='cell_type', tracks=['CD3D'])"],
    related=["pl.plot1cell", "pl.cellproportion"],
)
def celltype_tree(
    data: AnnData | pd.DataFrame,
    hierarchy: Sequence[str] | str | None = None,
    label: str = "cell_type",
    *,
    proportion: str | bool | None = None,
    tracks: Sequence[str] | str | None = None,
    track_markers: Mapping[str, str] | None = None,
    use_raw: bool = False,
    layer: str | None = None,
    standardize: bool = False,
    palette: Mapping | None = None,
    cmap: str = "RdBu_r",
    vmax: float | None = None,
    figsize=(10, 10),
    ax: Axes | None = None,
    show: bool = True,
) -> Axes:
    """Plot a radial cell annotation tree with optional numeric tracks.

    Parameters
    ----------
    data : anndata.AnnData or pandas.DataFrame
        For AnnData, group cells by the full path through ``hierarchy`` and
        ``label`` in ``adata.obs``. For a DataFrame, supply one row per unique
        leaf path. Path columns must not contain missing values.
    hierarchy : str or sequence of str, optional
        Classification columns, ordered from broad to fine. Omit for a flat
        tree. Siblings follow first appearance in the input, and descendants
        of each node occupy consecutive leaf positions.
    label : str, default 'cell_type'
        Column used for leaf labels. A label can occur in different branches.
    proportion : str or bool, optional
        For AnnData, append the percentage of all input cells in each leaf
        by default, or set to False to omit it. For a DataFrame, supply the
        name of a column containing percentages in [0, 100].
    tracks : str or sequence of str, optional
        Names of numeric tracks from inside to outside. For AnnData, take the
        mean of each requested numeric ``obs`` column or gene within each leaf.
        An ``obs`` column takes precedence over a gene of the same name. For a
        DataFrame, use the supplied column values directly. Missing values
        are shown in gray.
    track_markers : mapping of str to str, optional
        Map a track name to a column of text markers, such as significance
        stars computed upstream. AnnData marker columns must be constant
        within each leaf. Missing and empty markers are omitted.
    use_raw : bool, default False
        Read genes from ``adata.raw.X`` instead of ``adata.X``.
    layer : str, optional
        Read genes from this AnnData layer. Cannot be combined with
        ``use_raw=True``.
    standardize : bool, default False
        Z-score each track independently across leaves. Constant tracks
        become zero, and missing values remain missing.
    palette : mapping, optional
        Colors keyed by values in the first hierarchy column, or by leaf
        label when there is no hierarchy.
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
    levels = [hierarchy] if isinstance(hierarchy, str) else list(() if hierarchy is None else hierarchy)
    columns = [*levels, label]
    rings = [tracks] if isinstance(tracks, str) else list(() if tracks is None else tracks)
    markers = dict(track_markers or {})
    if len(set(columns)) != len(columns):
        raise ValueError("Specify distinct hierarchy columns and a separate leaf label column.")
    if len(set(rings)) != len(rings) or not markers.keys() <= set(rings):
        raise ValueError("Tracks must be distinct and marker keys must name tracks.")
    percentages = None
    if isinstance(data, AnnData):
        if proportion is not None and not isinstance(proportion, bool):
            raise ValueError("For AnnData, proportion must be True, False, or None.")
        if use_raw and layer is not None:
            raise ValueError("use_raw=True and layer cannot be combined.")
        data, percentages = _summarize_anndata(data, columns, rings, markers, use_raw, layer)
        if proportion is False:
            percentages = None
        proportion = None
    elif not isinstance(data, pd.DataFrame):
        raise TypeError("data must be an AnnData or a DataFrame.")
    elif use_raw or layer is not None:
        raise ValueError("use_raw and layer require AnnData input.")
    elif proportion is False:
        proportion = None
    elif proportion is not None and not isinstance(proportion, str):
        raise ValueError("For a DataFrame, proportion must name a column.")
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
    scores = data[rings].apply(pd.to_numeric).astype(float)
    values = scores.to_numpy()
    if np.isinf(values).any():
        raise ValueError("Track values must be finite or missing.")
    if standardize:
        scores = scores.mask(scores.eq(scores.min()) & scores.eq(scores.max()), 0)
        scores = (scores - scores.mean()) / scores.std(ddof=0).replace(0, 1)
        values = scores.to_numpy()
    if percentages is not None:
        percentages = percentages[order]
    elif proportion is not None:
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
        if depth == 1 and len(columns) > 1:
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
        bar.ax.set_title("Z-score" if standardize else "Track score", fontsize=7, pad=3)
        bar.outline.set_visible(False)
    if show:
        plt.show()
    return ax
