r"""Embeddable clustered heatmap built on marsilea.

Unlike :func:`omicverse.pl.complexheatmap` (which is PyComplexHeatmap-based and
always builds its own figure), this heatmap renders into a caller-supplied
``matplotlib`` Figure or SubFigure. That makes it usable as one panel of a
larger multi-panel figure: marsilea's own layout engine (dendrograms,
annotation strips, colour bars, legends) runs inside the region the parent
hands it.

marsilea must be installed (``pip install marsilea``). The import is guarded
with an actionable error the same way other optional-backend ov.pl functions
are.
"""

from collections.abc import Mapping
import types

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ._palette import palette_28, palette_56
from .._registry import register_function


def _import_marsilea():
    try:
        import marsilea as ma
        import marsilea.plotter as mp
    except ImportError as exc:
        raise ImportError(
            "marsilea is required for ov.pl.heatmap. "
            "Please install it with `pip install marsilea`."
        ) from exc
    return ma, mp


def _resolve_palette(n_colors):
    if n_colors <= len(palette_28):
        return list(palette_28[:n_colors])
    if n_colors <= len(palette_56):
        return list(palette_56[:n_colors])
    # Recycle when there are more categories than named colours.
    reps = int(np.ceil(n_colors / len(palette_56)))
    return list((palette_56 * reps)[:n_colors])


def _as_frame(data):
    """Normalise ndarray / DataFrame / AnnData to (DataFrame, adata_or_None)."""
    # AnnData without importing anndata at module top (keeps it optional-light).
    if hasattr(data, "obs") and hasattr(data, "var") and hasattr(data, "X"):
        adata = data
        matrix = adata.X
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix)
        frame = pd.DataFrame(
            matrix,
            index=adata.obs_names.astype(str),
            columns=adata.var_names.astype(str),
        )
        return frame, adata

    if isinstance(data, pd.DataFrame):
        return data.copy(), None

    matrix = np.asarray(data)
    if matrix.ndim != 2:
        raise ValueError(
            f"heatmap data must be 2-D, got array with {matrix.ndim} dimension(s)."
        )
    frame = pd.DataFrame(
        matrix,
        index=[str(i) for i in range(matrix.shape[0])],
        columns=[str(j) for j in range(matrix.shape[1])],
    )
    return frame, None


def _apply_zscore(frame, z_score):
    if z_score is None:
        return frame
    if z_score not in (0, 1):
        raise ValueError(
            f"z_score must be None, 0 (standardise rows) or 1 (standardise "
            f"columns), got {z_score!r}."
        )
    values = frame.values.astype(float)
    axis = 1 if z_score == 0 else 0  # z_score=0 standardises each row
    mean = values.mean(axis=axis, keepdims=True)
    std = values.std(axis=axis, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    standardised = (values - mean) / std
    return pd.DataFrame(standardised, index=frame.index, columns=frame.columns)


def _collect_annotations(annotation, adata, source, axis_len, axis_name):
    """Return a list of (name, pandas.Series) for one axis.

    ``annotation`` may be a single name (str), a dict {name: array-like}, a
    DataFrame, or a Series. When it is a name and an AnnData is available the
    column is pulled from ``adata.obs`` (rows) or ``adata.var`` (columns).
    """
    if annotation is None:
        return []

    columns = {}
    if isinstance(annotation, str):
        if source is None:
            raise ValueError(
                f"{axis_name}_annotation was given the name {annotation!r}, but "
                f"that only works with AnnData input. Pass an array-like or a "
                f"DataFrame instead."
            )
        if annotation not in source.columns:
            where = "adata.obs" if axis_name == "row" else "adata.var"
            raise ValueError(
                f"{axis_name}_annotation {annotation!r} was not found in {where}."
            )
        columns[annotation] = pd.Series(np.asarray(source[annotation]))
    elif isinstance(annotation, pd.DataFrame):
        for name in annotation.columns:
            columns[str(name)] = pd.Series(np.asarray(annotation[name]))
    elif isinstance(annotation, pd.Series):
        columns[str(annotation.name or axis_name)] = pd.Series(
            np.asarray(annotation)
        )
    elif isinstance(annotation, Mapping):
        for name, values in annotation.items():
            columns[str(name)] = pd.Series(np.asarray(values))
    else:
        # A bare array-like.
        columns[axis_name] = pd.Series(np.asarray(annotation))

    for name, series in columns.items():
        if len(series) != axis_len:
            raise ValueError(
                f"{axis_name}_annotation {name!r} has {len(series)} entries but "
                f"the data has {axis_len} {axis_name}s. They must match."
            )
    return list(columns.items())


def _palette_for(name, annotation_palette):
    if annotation_palette is None:
        return None
    if isinstance(annotation_palette, Mapping):
        return annotation_palette.get(name)
    return annotation_palette


def _annotation_plotter(mp, name, series, annotation_palette):
    """Build a marsilea plotter for one annotation strip.

    A categorical annotation gets discrete colours (``mp.Colors``); a numeric
    one gets a gradient (``mp.ColorMesh``).
    """
    user = _palette_for(name, annotation_palette)
    if is_numeric_dtype(series):
        cmap = user if isinstance(user, str) else "viridis"
        return mp.ColorMesh(series.values.astype(float), cmap=cmap, label=name)
    # Categorical: map each category to a discrete colour.
    categories = list(pd.unique(series.astype(str)))
    if isinstance(user, Mapping):
        palette = {cat: user[cat] for cat in categories if cat in user}
    elif isinstance(user, (list, tuple)):
        palette = dict(zip(categories, list(user)))
    else:
        palette = dict(zip(categories, _resolve_palette(len(categories))))
    return mp.Colors(series.astype(str).values, palette=palette, label=name)


def _place_in_rect(build, figure, rect, scale, trial):
    r"""Draw a marsilea block into ``rect`` of ``figure`` at true size.

    marsilea owns its own layout: :meth:`freeze` derives a figure size from the
    block's content, calls ``figure.set_size_inches`` with it, and then adds
    every axes at ``inches / that figure size``. Handing it a host figure
    therefore resizes the host and scatters the heatmap's axes across the whole
    canvas — the block cannot be asked to occupy a corner.

    Rescaling the axes rectangles afterwards is not a fix either: an axes
    rectangle scales but the text inside it does not, so a block squeezed to
    half width keeps full-size tick labels and legends, which then overflow.
    Only a *translation* is safe.

    So the size is arranged before the draw instead of corrected after it. The
    overhead around the heatmap cell — dendrograms, annotation strips, labels,
    legends, margins — is set in inches and does not depend on how large the
    cell is, so one throwaway render measures it:

    ``overhead = natural_figure_size - trial_cell_size``

    The cell that makes the block exactly fill ``rect`` is then
    ``rect_size - overhead``, and a block built at that size needs only moving
    into place. Text keeps its size because nothing is scaled.

    Arguments
    ---------
    build
        ``build(cell_w, cell_h)`` returning a fresh, unrendered marsilea object.
    figure
        Host figure. Its size is restored after marsilea has resized it.
    rect
        ``(x0, y0, width, height)`` in host-figure fractions.
    trial
        Cell size for the measuring pass. Any positive size works; a value near
        the answer just avoids relying on the overhead being perfectly constant.

    Returns
    -------
    The rendered marsilea object, with its axes inside ``rect``.
    """
    import matplotlib.pyplot as plt

    host_w, host_h = figure.get_size_inches()

    # Measuring pass on a throwaway figure: what does the overhead cost?
    probe_fig = plt.figure(figsize=(4, 4))
    _render_into(build(*trial), probe_fig, scale)
    natural = probe_fig.get_size_inches().copy()
    plt.close(probe_fig)
    overhead = (natural[0] - trial[0] * scale, natural[1] - trial[1] * scale)

    x0, y0, width, height = rect
    target = (width * host_w, height * host_h)
    cell = (max((target[0] - overhead[0]) / scale, 0.2),
            max((target[1] - overhead[1]) / scale, 0.2))

    before = set(figure.axes)
    heatmap = build(*cell)
    _render_into(heatmap, figure, scale)
    added = [ax for ax in figure.axes if ax not in before]

    # marsilea resized the host to its own figure size; the axes it added are
    # fractions of *that*. Restore the host and re-express them as fractions of
    # the host, offset to the rect's corner — a translation, not a rescale.
    block_w, block_h = figure.get_size_inches()
    figure.set_size_inches(host_w, host_h)
    for ax in added:
        pos = ax.get_position()
        ax.set_position([
            x0 + pos.x0 * block_w / host_w,
            y0 + pos.y0 * block_h / host_h,
            pos.width * block_w / host_w,
            pos.height * block_h / host_h,
        ])
    return heatmap


def _render_into(heatmap, figure, scale):
    """Render a marsilea object, tolerating a SubFigure target.

    marsilea 0.5.8's composite layout calls ``figure.set_size_inches`` during
    freeze, which a ``matplotlib`` SubFigure does not implement (its size is
    owned by the parent gridspec). We temporarily attach a no-op so the layout
    engine can run.

    Note that surviving the call is all the no-op buys: the rectangles marsilea
    then computes are fractions of the figure size *it* derived, not of the
    SubFigure, so the block still lands across the parent figure rather than
    inside the subfigure. Use ``rect=`` (see :func:`_place_in_rect`) to confine
    a heatmap to part of a figure.
    """
    from matplotlib.figure import SubFigure

    patched = False
    if isinstance(figure, SubFigure) and not hasattr(figure, "set_size_inches"):
        figure.set_size_inches = types.MethodType(
            lambda self, *args, **kwargs: None, figure
        )
        patched = True
    try:
        heatmap.render(figure=figure, scale=scale)
    finally:
        if patched:
            try:
                del figure.set_size_inches
            except AttributeError:
                pass
    return heatmap


def _reorder_labels(labels, index):
    if index is None:
        return list(labels)
    flat = np.asarray(index).ravel()
    return [labels[i] for i in flat]


@register_function(
    aliases=["热图", "聚类热图", "heatmap", "marsilea", "marsilea_heatmap"],
    category="pl",
    description=(
        "Embeddable clustered heatmap built on marsilea. Renders into a "
        "caller-supplied matplotlib Figure or SubFigure so it can be one panel "
        "of a larger multi-panel figure, with optional row/column dendrograms "
        "and categorical or numeric annotation strips."
    ),
    examples=[
        "ov.pl.heatmap(df, row_cluster=True, col_cluster=True)",
        "ov.pl.heatmap(adata, row_annotation='celltype', z_score=0)",
        "sub = fig.subfigures(1, 2)[0]; ov.pl.heatmap(data, figure=sub)",
    ],
    related=["pl.complexheatmap", "pl.group_heatmap", "pl.marker_heatmap"],
)
def heatmap(
    data,
    *,
    figure=None,
    z_score=None,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    center=None,
    row_cluster=True,
    col_cluster=True,
    row_dendrogram=True,
    col_dendrogram=False,
    row_annotation=None,
    col_annotation=None,
    show_rownames=False,
    show_colnames=False,
    xlabel=None,
    ylabel=None,
    label="value",
    annotation_palette=None,
    scale=1,
    width=None,
    height=None,
    rect=None,
):
    r"""Draw an embeddable clustered heatmap with marsilea.

    The distinguishing property versus :func:`ov.pl.complexheatmap` is that this
    function renders into a matplotlib Figure or SubFigure supplied by the
    caller, instead of building its own figure. Everything marsilea draws
    (dendrograms, annotation strips, colour bar, legends) lands inside that
    region, so the heatmap can be one panel of a multi-panel figure.

    Args:
        data: A 2-D array, a :class:`pandas.DataFrame`, or an
            :class:`anndata.AnnData`. For AnnData ``.X`` is used, with ``.obs``
            and ``.var`` available so annotations can be pulled by name.
        figure: Target :class:`~matplotlib.figure.Figure` or
            :class:`~matplotlib.figure.SubFigure`. If None, a new figure sized
            from the data is created and rendered into.
        z_score: Standardise before drawing. ``None`` leaves data untouched;
            ``0`` standardises each row (row means become ~0); ``1``
            standardises each column. This changes the colour scale.
        cmap: Colormap for the heatmap values.
        vmin, vmax, center: Colour-scale limits and diverging centre.
        row_cluster, col_cluster: Hierarchically cluster and reorder rows /
            columns.
        row_dendrogram, col_dendrogram: Draw the dendrogram for that axis. A
            dendrogram is only shown when the matching ``*_cluster`` is True.
        row_annotation, col_annotation: Annotation strips aligned to rows /
            columns. Accepts a dict ``{name: array-like}``, a DataFrame, a
            Series, a bare array-like, or -- for AnnData input -- a column name
            resolved against ``.obs`` (rows) or ``.var`` (columns). Categorical
            annotations get discrete colours, numeric ones a gradient, each with
            a legend.
        show_rownames, show_colnames: Draw row / column tick labels.
        xlabel, ylabel: Axis titles.
        label: Title of the heatmap colour bar.
        annotation_palette: Optional colours for annotations. Either a single
            palette/colormap applied to every annotation, or a dict keyed by
            annotation name (each value a colour list, a {category: colour} map,
            or a colormap name for numeric annotations).
        scale: Passed through to marsilea ``render`` to scale the layout.
        width, height: Size of the heatmap cell itself, in inches — the
            dendrograms, annotation strips, labels and legends are added
            around it. Left to marsilea when None.
        rect: Place the whole block inside this ``(x0, y0, width, height)``
            region of ``figure``, in figure fractions, and restore the
            figure's size afterwards. Requires ``figure``. Without it,
            marsilea resizes the figure to suit itself and spreads the
            heatmap's axes over the whole canvas, so a multi-panel figure can
            only host the heatmap by giving it a region. The cell is sized so
            the block fills the region at true scale and is then translated
            into place — nothing is rescaled, so tick labels, legends and the
            colour bar keep the size they were drawn at.

    Returns:
        The marsilea ``Heatmap`` object, already rendered. Reach ``.figure`` for
        the matplotlib figure, ``.row_order`` / ``.col_order`` for the label
        order after clustering, or add more marsilea components before saving.
    """
    ma, mp = _import_marsilea()

    frame, adata = _as_frame(data)
    row_source = adata.obs if adata is not None else None
    col_source = adata.var if adata is not None else None

    row_labels = list(frame.index)
    col_labels = list(frame.columns)
    n_rows, n_cols = frame.shape

    row_ann = _collect_annotations(
        row_annotation, adata, row_source, n_rows, "row"
    )
    col_ann = _collect_annotations(
        col_annotation, adata, col_source, n_cols, "col"
    )

    frame = _apply_zscore(frame, z_score)

    def _build(main_w=None, main_h=None):
        """Assemble the marsilea object. ``main_*`` size the heatmap cell only.

        A marsilea object is consumed by ``render``, so placing the block at an
        exact size needs a throwaway build to measure with and a fresh one to
        draw — hence a factory rather than a single object.
        """
        obj = ma.Heatmap(
            frame.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            label=label,
            width=main_w,
            height=main_h,
        )

        # Annotation strips (with legends). Rows go left, columns go top.
        for name, series in row_ann:
            obj.add_left(
                _annotation_plotter(mp, name, series, annotation_palette),
                size=0.2,
                pad=0.05,
                legend=True,
            )
        for name, series in col_ann:
            obj.add_top(
                _annotation_plotter(mp, name, series, annotation_palette),
                size=0.2,
                pad=0.05,
                legend=True,
            )

        # Tick labels.
        if show_rownames:
            obj.add_right(mp.Labels(row_labels), pad=0.05)
        if show_colnames:
            obj.add_bottom(mp.Labels(col_labels, rotation=90), pad=0.05)

        # Axis titles.
        title_kwargs = {}
        if xlabel is not None:
            title_kwargs["bottom"] = xlabel
        if ylabel is not None:
            title_kwargs["left"] = ylabel
        if title_kwargs:
            obj.add_title(**title_kwargs)

        # Clustering. add_dendrogram both reorders and (optionally) draws; a
        # dendrogram cannot be shown without clustering, so *_dendrogram is
        # honoured only when *_cluster is True.
        if row_cluster:
            obj.add_dendrogram("left", show=bool(row_dendrogram), pad=0.02)
        if col_cluster:
            obj.add_dendrogram("top", show=bool(col_dendrogram), pad=0.02)

        if row_ann or col_ann:
            obj.add_legends()
        return obj

    import matplotlib.pyplot as plt

    if rect is not None:
        if figure is None:
            raise TypeError("`rect` places the heatmap inside a figure you "
                            "supply — pass `figure=` as well.")
        heatmap_obj = _place_in_rect(
            _build, figure, rect, scale,
            trial=(width if width is not None else max(n_cols * 0.06, 2.0),
                   height if height is not None else max(n_rows * 0.16, 1.5)))
        deform = heatmap_obj.get_deform()
        heatmap_obj.row_order = _reorder_labels(
            row_labels,
            getattr(deform, "row_reorder_index", None) if row_cluster else None)
        heatmap_obj.col_order = _reorder_labels(
            col_labels,
            getattr(deform, "col_reorder_index", None) if col_cluster else None)
        return heatmap_obj

    heatmap_obj = _build(width, height)

    if figure is None:
        fig_w = min(max(n_cols * 0.35 + 3, 4), 20)
        fig_h = min(max(n_rows * 0.3 + 2, 3), 20)
        figure = plt.figure(figsize=(fig_w, fig_h))

    _render_into(heatmap_obj, figure, scale)

    # Expose the post-clustering order on the returned object.
    deform = heatmap_obj.get_deform()
    heatmap_obj.row_order = _reorder_labels(
        row_labels, getattr(deform, "row_reorder_index", None) if row_cluster else None
    )
    heatmap_obj.col_order = _reorder_labels(
        col_labels, getattr(deform, "col_reorder_index", None) if col_cluster else None
    )

    return heatmap_obj
