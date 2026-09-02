import collections.abc as cabc
from copy import copy
from numbers import Integral
from itertools import combinations, product
from pathlib import Path
import matplotlib
from typing import (
    Collection,
    Union,
    Optional,
    Sequence,
    Any,
    Mapping,
    List,
    Tuple,
    Literal,
)
from warnings import warn

import numpy as np
import pandas as pd
from anndata import AnnData
from cycler import Cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
# Removed deprecated is_categorical_dtype import
# Using isinstance(dtype, pd.CategoricalDtype) instead
from matplotlib import pyplot as pl, colors, colormaps
from matplotlib import rcParams
from matplotlib import patheffects
from matplotlib.colors import Colormap, Normalize
from functools import partial
from ._scanpy_compat import (
    _FontWeight,
    _FontSize,
    ColorLike,
    VBound,
    circles,
    check_projection,
    check_colornorm,
    doc_adata_color_etc,
    doc_edges_arrows,
    doc_scatter_embedding,
    doc_scatter_spatial,
    doc_show_save_ax,
    logg,
    settings,
    sanitize_anndata,
    _doc_params,
    Empty,
    _empty,
    savefig_or_show,
    plot_edges,
    plot_arrows,
    additional_colors,
)


def register_function(*args, **kwargs):
    """No-op decorator for plotting backends used only as implementation detail."""

    def decorator(func):
        return func

    return decorator

def _get_vector_friendly():
    """Get the vector_friendly setting from omicverse plot settings."""
    try:
        from ._plot_backend import _vector_friendly
        return _vector_friendly
    except ImportError:
        try:
            return settings._vector_friendly
        except AttributeError:
            return True  # Default fallback


@register_function(
    aliases=["嵌入可视化", "embedding", "scatter_plot", "嵌入绘图", "降维可视化"],
    category="utils",
    description="Scatter plot visualization of single-cell embeddings with flexible coloring options",
    examples=[
        "# Basic UMAP visualization",
        "ov.pl.embedding(adata, basis='X_umap', color='leiden')",
        "# Multiple color variables",
        "ov.pl.embedding(adata, basis='X_pca', color=['n_genes', 'leiden', 'CD14'])",
        "# Gene expression visualization",
        "ov.pl.embedding(adata, basis='X_tsne', color=['CD3D', 'CD79A'])",
        "# Custom visualization settings",
        "ov.pl.embedding(adata, basis='X_umap', color='celltype',",
        "                   frameon=False, legend_loc='right margin')"
    ],
    related=["pp.pca", "utils.mde", "pl.umap", "pl.tsne"]
)
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    edges_arrows=doc_edges_arrows,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def embedding(
    adata: AnnData,
    basis: str,
    *,
    color: Union[str, Sequence[str], None] = None,
    gene_symbols: Optional[str] = None,
    use_raw: Optional[bool] = None,
    sort_order: bool = True,
    edges: bool = False,
    edges_width: float = 0.1,
    edges_color: Union[str, Sequence[float], Sequence[str]] = 'grey',
    neighbors_key: Optional[str] = None,
    arrows: bool = False,
    arrows_kwds: Optional[Mapping[str, Any]] = None,
    groups: Optional[str] = None,
    components: Union[str, Sequence[str]] = None,
    dimensions: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    layer: Optional[str] = None,
    projection: Literal['2d', '3d'] = '2d',
    scale_factor: Optional[float] = None,
    color_map: Union[Colormap, str, None] = None,
    cmap: Union[Colormap, str, None] = None,
    palette: Union[str, Sequence[str], Cycler, None] = None,
    na_color: ColorLike = "lightgray",
    na_in_legend: bool = True,
    size: Union[float, Sequence[float], None] = None,
    frameon: Optional[bool] = None,
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = 'bold',
    legend_loc: str = 'right margin',
    legend_fontoutline: Optional[int] = None,
    colorbar_loc: Optional[str] = "right",
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    add_outline: Optional[bool] = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ('black', 'white'),
    ncols: int = 4,
    hspace: float = 0.25,
    wspace: Optional[float] = None,
    title: Union[str, Sequence[str], None] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    marker: Union[str, Sequence[str]] = '.',
    arrow_scale: float = 10,
    arrow_width: float = 0.005,
    layout: Literal['grid', 'flow'] = 'flow',
    **kwargs,
) -> Union[Figure, Axes, None]:
    r"""Scatter plot for user specified embedding basis (e.g. umap, pca, etc).

    Parameters
    ----------
        adata: Annotated data matrix
        basis: Name of the obsm basis to use
        color: Keys for annotations of observations/cells or variables/genes (None)
        gene_symbols: Column name in .var DataFrame for gene symbols (None)
        use_raw: Whether to use .raw attribute of adata (None)
        sort_order: Sort order for points by color values (True)
        edges: Whether to draw edges of graph (False)
        edges_width: Width of edges (0.1)
        edges_color: Color of edges ('grey')
        neighbors_key: Key to use for neighbors (None)
        arrows: Whether to draw arrows (False) 
        arrows_kwds: Keywords for arrow plotting (None)
        groups: Restrict to a subset of groups (None)
        components: Components to plot (None)
        dimensions: Dimensions to plot (None)
        layer: Layer to use for coloring (None)
        projection: Projection type - '2d' or '3d' ('2d')
        scale_factor: Scaling factor for spatial coordinates (None)
        color_map: Colormap for continuous variables (None)
        cmap: Alias for color_map (None)
        palette: Colors to use for categorical variables (None)
        na_color: Color for missing values ('lightgray')
        na_in_legend: Include missing values in legend (True)
        size: Point size (None)
        frameon: Draw frame around plot (None)
        legend_fontsize: Font size for legend (None)
        legend_fontweight: Font weight for legend ('bold')
        legend_loc: Location of legend ('right margin')
        legend_fontoutline: Font outline width for legend (None)
        colorbar_loc: Location of colorbar ('right')
        vmax: Maximum color scale value (None)
        vmin: Minimum color scale value (None)
        vcenter: Center color scale value (None)
        norm: Normalization for color scale (None)
        add_outline: Add outline to points (False)
        outline_width: Width of outline (0.3, 0.05)
        outline_color: Color of outline ('black', 'white')
        ncols: Number of columns for multi-panel plots (4)
        hspace: Height spacing between subplots (0.25)
        wspace: Width spacing between subplots (None)
        title: Plot title (None)
        show: Show the plot (None)
        save: Save the plot (None)
        ax: Matplotlib axes object (None)
        return_fig: Return figure object (None)
        marker: Marker style ('.') 
        **kwargs: Additional arguments passed to scatter
        
    Returns
    -------
        Matplotlib axes or figure object if show=False
    """
    #####################
    # Argument handling #
    #####################

    check_projection(projection)
    sanitize_anndata(adata)

    basis_values = _get_basis(adata, basis)
    dimensions = _components_to_dimensions(
        components, dimensions, projection=projection, total_dims=basis_values.shape[1]
    )
    args_3d = dict(projection='3d') if projection == '3d' else {}

    # Figure out if we're using raw
    if use_raw is None:
        # check if adata.raw is set
        use_raw = layer is None and adata.raw is not None
    if use_raw and layer is not None:
        raise ValueError(
            "Cannot use both a layer and the raw representation. Was passed:"
            f"use_raw={use_raw}, layer={layer}."
        )
    if use_raw and adata.raw is None:
        raise ValueError(
            "`use_raw` is set to True but AnnData object does not have raw. "
            "Please check."
        )

    if isinstance(groups, str):
        groups = [groups]

    # Color map
    if color_map is not None:
        if cmap is not None:
            raise ValueError("Cannot specify both `color_map` and `cmap`.")
        else:
            cmap = color_map
    if matplotlib.__version__ < "3.7.0":
        if cmap is not None:
            pass
        else: 
            cmap = 'RdBu_r'
        if type(cmap)==matplotlib.colors.LinearSegmentedColormap:
            pass
        else:
            cmap = copy(colormaps.get_cmap(cmap))
            cmap.set_bad(na_color)
    else:
        if cmap is not None:
            pass
        else: 
            cmap = 'RdBu_r'
        if type(cmap)==matplotlib.colors.LinearSegmentedColormap:
            pass
        else:
            cmap = copy(matplotlib.colormaps[cmap])
            cmap.set_bad(na_color)

    
    kwargs["cmap"] = cmap
    # Prevents warnings during legend creation
    na_color = colors.to_hex(na_color, keep_alpha=True)

    if 'edgecolor' not in kwargs:
        # by default turn off edge color. Otherwise, for
        # very small sizes the edge will not reduce its size
        # (https://github.com/scverse/scanpy/issues/293)
        kwargs['edgecolor'] = 'none'

    # Vectorized arguments

    # turn color into a python list
    color = [color] if isinstance(color, str) or color is None else list(color)

    # turn marker into a python list
    marker = [marker] if isinstance(marker, str) else list(marker)

    if title is not None:
        # turn title into a python list if not None
        title = [title] if isinstance(title, str) else list(title)

    # turn vmax and vmin into a sequence
    if isinstance(vmax, str) or not isinstance(vmax, cabc.Sequence):
        vmax = [vmax]
    if isinstance(vmin, str) or not isinstance(vmin, cabc.Sequence):
        vmin = [vmin]
    if isinstance(vcenter, str) or not isinstance(vcenter, cabc.Sequence):
        vcenter = [vcenter]
    if isinstance(norm, Normalize) or not isinstance(norm, cabc.Sequence):
        norm = [norm]

    # Size
    if 's' in kwargs and size is None:
        size = kwargs.pop('s')
    if size is not None:
        # check if size is any type of sequence, and if so
        # set as ndarray
        if (
            size is not None
            and isinstance(size, (cabc.Sequence, pd.Series, np.ndarray))
            and len(size) == adata.shape[0]
        ):
            size = np.array(size, dtype=float)
    else:
        size = 120000 / adata.shape[0]

    ##########
    # Layout #
    ##########
    # Most of the code is for the case when multiple plots are required

    if wspace is None:
        #  try to set a wspace that is not too large or too small given the
        #  current figure size
        wspace = 0.75 / rcParams['figure.figsize'][0] + 0.02

    if components is not None:
        color, dimensions = list(zip(*product(color, dimensions)))

    color, dimensions, marker = _broadcast_args(color, dimensions, marker)

    # 'color' is a list of names that want to be plotted.
    # Eg. ['Gene1', 'louvain', 'Gene2'].
    # component_list is a list of components [[0,1], [1,2]]
    if (
        not isinstance(color, str)
        and isinstance(color, cabc.Sequence)
        and len(color) > 1
    ) or len(dimensions) > 1:
        if ax is not None:
            raise ValueError(
                "Cannot specify `ax` when plotting multiple panels "
                "(each for a given value of 'color')."
            )

        # each plot needs to be its own panel
        fig, grid = _panel_grid(hspace, wspace, ncols, len(color))
    else:
        grid = None
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111, **args_3d)

    ############
    # Plotting #
    ############
    axs = []
    # Per-panel colorbar axes, keyed by id(panel_ax). Populated only for
    # continuous-color panels rendered with frameon ∈ {'small', False}.
    # `_flow_layout_panels` consults this dict to drag the colorbar
    # alongside the panel when it gets repositioned, otherwise the
    # colorbar stays pinned at its original figure coordinates while the
    # panel migrates left/right — the visual bug visible at issue review.
    panel_colorbars: dict = {}

    # use itertools.product to make a plot for each color and for each component
    # For example if color=[gene1, gene2] and components=['1,2, '2,3'].
    # The plots are: [
    #     color=gene1, components=[1,2], color=gene1, components=[2,3],
    #     color=gene2, components = [1, 2], color=gene2, components=[2,3],
    # ]
    for count, (value_to_plot, dims) in enumerate(zip(color, dimensions)):
        color_source_vector = _get_color_source_vector(
            adata,
            value_to_plot,
            layer=layer,
            use_raw=use_raw,
            gene_symbols=gene_symbols,
            groups=groups,
        )
        color_vector, categorical = _color_vector(
            adata,
            value_to_plot,
            color_source_vector,
            palette=palette,
            na_color=na_color,
        )
        def _is_numeric_array(x):
            arr = np.asarray(x)
            return np.issubdtype(arr.dtype, np.number)

        # Order points
        order = slice(None)
        if sort_order is True and value_to_plot is not None and (categorical is False) and _is_numeric_array(color_vector):
            # 连续数值：数值高的点盖在上面
            arr = np.asarray(color_vector)
            order = np.argsort(-arr, kind="stable")[::-1]
        elif sort_order and (categorical or not _is_numeric_array(color_vector)):
            # 分类型或非数值（字符串/颜色）：空值下沉
            order = np.argsort(~pd.isnull(color_source_vector), kind="stable")
        # Set orders
        if isinstance(size, np.ndarray):
            size = np.array(size)[order]
        color_source_vector = color_source_vector[order]
        color_vector = color_vector[order]
        coords = basis_values[:, dims][order, :]

        # if plotting multiple panels, get the ax from the grid spec
        # else use the ax value (either user given or created previously)
        if grid:
            ax = pl.subplot(grid[count], **args_3d)
            axs.append(ax)
        if frameon is False:
            # Match scanpy semantics: `frameon=False` means hide the
            # frame entirely (no axis, no arrow). The arrow is the
            # `frameon='small'` extension below.
            ax.axis('off')
        elif frameon == 'small':
            ax.axis('off')
            from ..pl._single import add_arrow
            add_arrow(ax,adata,basis,fontsize=legend_fontsize,arrow_scale=arrow_scale,arrow_width=arrow_width)
            '''
            #ax.axis('off')
            xmin=coords[:, 0].min()
            xmax=coords[:, 0].max()
            ymin=coords[:, 1].min()
            ymax=coords[:, 1].max()

            #ax.spines['left'].set_position(('outward', 10))
            #ax.spines['bottom'].set_position(('axes', 0))
            ax.spines['left'].set_position(('data', xmin))
            ax.spines['bottom'].set_position(('data', ymin))

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.spines['bottom'].set_bounds(xmin,xmin+(xmax-xmin)/6)
            #ax.spines['left'].set_bounds(ymin,ymin+(ymax-ymin)/6)
            '''


        
        if title is None:
            if value_to_plot is not None:
                ax.set_title(value_to_plot)
            else:
                ax.set_title('')
        else:
            try:
                ax.set_title(title[count])
            except IndexError:
                logg.warning(
                    "The title list is shorter than the number of panels. "
                    "Using 'color' value instead for some plots."
                )
                ax.set_title(value_to_plot)

        if not categorical:
            vmin_float, vmax_float, vcenter_float, norm_obj = _get_vboundnorm(
                vmin, vmax, vcenter, norm, count, color_vector
            )
            normalize = check_colornorm(
                vmin_float,
                vmax_float,
                vcenter_float,
                norm_obj,
            )
        else:
            normalize = None

        scatter_kwargs = dict(kwargs)
        if categorical or not _is_numeric_array(color_vector):
            scatter_kwargs.pop("cmap", None)

        # make the scatter plot
        if projection == '3d':
            cax = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=color_vector,
                rasterized=_get_vector_friendly(),
                norm=normalize,
                marker=marker[count],
                **scatter_kwargs,
            )
        else:
            scatter = (
                partial(ax.scatter, s=size, plotnonfinite=True)
                if scale_factor is None
                else partial(
                    circles, s=size, ax=ax, scale_factor=scale_factor
                )  # size in circles is radius
            )

            if add_outline:
                # the default outline is a black edge followed by a
                # thin white edged added around connected clusters.
                # To add an outline
                # three overlapping scatter plots are drawn:
                # First black dots with slightly larger size,
                # then, white dots a bit smaller, but still larger
                # than the final dots. Then the final dots are drawn
                # with some transparency.

                bg_width, gap_width = outline_width
                point = np.sqrt(size)
                gap_size = (point + (point * gap_width) * 2) ** 2
                bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2
                # the default black and white colors can be changes using
                # the contour_config parameter
                bg_color, gap_color = outline_color

                # remove edge from kwargs if present
                # because edge needs to be set to None
                scatter_kwargs['edgecolor'] = 'none'

                # remove alpha for outline
                alpha = scatter_kwargs.pop('alpha') if 'alpha' in scatter_kwargs else None

                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    s=bg_size,
                    c=bg_color,
                    rasterized=_get_vector_friendly(),
                    norm=normalize,
                    marker=marker[count],
                    **scatter_kwargs,
                )
                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    s=gap_size,
                    c=gap_color,
                    rasterized=_get_vector_friendly(),
                    norm=normalize,
                    marker=marker[count],
                    **scatter_kwargs,
                )
                # if user did not set alpha, set alpha to 0.7
                scatter_kwargs['alpha'] = 0.7 if alpha is None else alpha

            cax = scatter(
                coords[:, 0],
                coords[:, 1],
                c=color_vector,
                rasterized=_get_vector_friendly(),
                norm=normalize,
                marker=marker[count],
                **scatter_kwargs,
            )

        # remove y and x ticks
        ax.set_yticks([])
        ax.set_xticks([])
        if projection == '3d':
            ax.set_zticks([])

        # set default axis_labels
        name = _basis2name(basis)
        axis_labels = [name + str(d + 1) for d in dims]

        ax.set_xlabel(axis_labels[0],loc='left',fontsize=legend_fontsize)
        ax.set_ylabel(axis_labels[1],loc='bottom',fontsize=legend_fontsize)
        if projection == '3d':
            # shift the label closer to the axis
            ax.set_zlabel(axis_labels[2], labelpad=-7)
        ax.autoscale_view()

        # 画散点后

        if edges:
            plot_edges(ax, adata, basis, edges_width, edges_color, neighbors_key)
        if arrows:
            plot_arrows(ax, adata, basis, arrows_kwds)

        if value_to_plot is None:
            # if only dots were plotted without an associated value
            # there is not need to plot a legend or a colorbar
            continue

        if legend_fontoutline is not None:
            path_effect = [
                patheffects.withStroke(linewidth=legend_fontoutline, foreground='w')
            ]
        else:
            path_effect = None

        # Adding legends
        if categorical or color_vector.dtype == bool:
            _add_categorical_legend(
                ax,
                color_source_vector,
                palette=_get_palette(adata, value_to_plot),
                scatter_array=coords,
                legend_loc=legend_loc,
                legend_fontweight=legend_fontweight,
                legend_fontsize=legend_fontsize,
                legend_fontoutline=path_effect,
                na_color=na_color,
                na_in_legend=na_in_legend,
                multi_panel=bool(grid),
            )
        elif colorbar_loc is not None:
            # Native Matplotlib inset colorbar. Do not use axes_grid1 here:
            # its locator interacts badly with PDF tight-bbox export when a
            # colourbar contains rasterised artists.
            from matplotlib.ticker import MaxNLocator

            cax1 = ax.inset_axes(
                [0.98, 0.0, 0.02, 0.30], transform=ax.transAxes,
            )
            # Register the inset so _flow_layout_panels measures its
            # tightbbox in Pass 1 — ax.get_tightbbox excludes children, so
            # without this the colorbar's tick labels (which stick out to
            # the right of the inset bar) would still overlap the next
            # panel's X_umap label.
            panel_colorbars[id(ax)] = cax1
            cb = pl.colorbar(cax, cax=cax1, orientation="vertical")
            cb.locator = MaxNLocator(nbins=3, integer=False)
            cb.update_ticks()

    # Flow layout: re-position panels left-to-right (with wrap) so that
    # each panel + its external legend takes its actual footprint, never
    # overlapping a neighboring panel. Only meaningful when multiple
    # panels exist (i.e. grid is not None).
    if layout == 'flow' and grid is not None and axs:
        _flow_layout_panels(fig, axs, panel_colorbars=panel_colorbars)

    if return_fig is True:
        return fig
    axs = axs if grid else ax
    if save not in (None, False):
        save_path = Path(save) if isinstance(save, str) else Path(f"{basis}.png")
        bbox_inches = None if save_path.suffix.lower() == ".pdf" else "tight"
        fig.savefig(save_path, bbox_inches=bbox_inches, pad_inches=0.1)
    should_show = settings.autoshow if show is None else show
    if should_show:
        pl.show()
    if show is False:
        return axs


def _panel_grid(hspace, wspace, ncols, num_panels):
    from matplotlib import gridspec

    n_panels_x = min(ncols, num_panels)
    n_panels_y = np.ceil(num_panels / n_panels_x).astype(int)
    # each panel will have the size of rcParams['figure.figsize']
    fig = pl.figure(
        figsize=(
            n_panels_x * rcParams['figure.figsize'][0] * (1 + wspace),
            n_panels_y * rcParams['figure.figsize'][1],
        ),
    )
    left = 0.2 / n_panels_x
    bottom = 0.13 / n_panels_y
    gs = gridspec.GridSpec(
        nrows=n_panels_y,
        ncols=n_panels_x,
        left=left,
        right=1 - (n_panels_x - 1) * left - 0.01 / n_panels_x,
        bottom=bottom,
        top=1 - (n_panels_y - 1) * bottom - 0.1 / n_panels_y,
        hspace=hspace,
        wspace=wspace,
    )
    return fig, gs


def _flow_layout_panels(fig, axs, panel_colorbars=None, gap=0.3, margin=0.3):
    r"""Reposition multi-panel axes left-to-right with wrap.

    Each panel's "footprint" includes its external categorical legend
    (if any) *and* any per-panel colorbar axes registered in
    ``panel_colorbars``, so panels are guaranteed not to overlap
    regardless of legend label length / category count / colorbar
    presence. Wraps to the next row when a panel would exceed
    ``fig_width - margin``. Resizes the figure height if total layout
    exceeds the initial height.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing ``axs``.
    axs : list[matplotlib.axes.Axes]
        Panel axes in plotting order. Each axes may carry an external
        legend created via ``ax.legend(bbox_to_anchor=...,
        bbox_transform=ax.transAxes)``; the legend follows the axes
        when ``set_position`` is called, so this routine only needs to
        re-position the axes themselves.
    panel_colorbars : dict[int, matplotlib.axes.Axes] or None
        Mapping of ``id(panel_ax) -> colorbar_ax`` for panels that
        rendered a continuous-color colorbar via
        ``fig.add_axes([...])`` (the ``frameon='small' / False``
        branch). The colorbar's offset is captured *before* the panel
        moves, then re-applied relative to the new panel position so
        the colorbar travels with its panel. Unlike legends, colorbars
        live in figure coordinates and do not follow the panel
        automatically.
    """
    if not axs:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()
    fig_w_in = fig.get_figwidth()

    # ── Pass 1: measure each panel's full footprint in inches ───────
    items = []
    for ax in axs:
        # Use the *data* bbox for the panel's own position, but seed the
        # footprint with the *tight* bbox so axis labels and tick labels
        # (X_umap1 below, X_umap2 to the left) are included. Without this,
        # tight-packed multi-panel grids let labels poke into adjacent
        # panels' colorbars.
        ax_bbox = ax.get_window_extent(renderer).transformed(inv)
        try:
            tight = ax.get_tightbbox(renderer).transformed(inv)
            x0, x1 = tight.x0, tight.x1
            y0, y1 = tight.y0, tight.y1
        except Exception:
            x0, x1 = ax_bbox.x0, ax_bbox.x1
            y0, y1 = ax_bbox.y0, ax_bbox.y1

        lg = ax.get_legend()
        if lg is not None:
            lg_bbox = lg.get_window_extent(renderer).transformed(inv)
            x0 = min(x0, lg_bbox.x0); x1 = max(x1, lg_bbox.x1)
            y0 = min(y0, lg_bbox.y0); y1 = max(y1, lg_bbox.y1)

        # Include the title above the axes so multi-line / long titles
        # don't bleed into the row above. ax.title is always present (even
        # if empty); only fold it in when it has visible text.
        title = ax.title
        if title is not None and title.get_text():
            try:
                t_bbox = title.get_window_extent(renderer).transformed(inv)
                x0 = min(x0, t_bbox.x0); x1 = max(x1, t_bbox.x1)
                y0 = min(y0, t_bbox.y0); y1 = max(y1, t_bbox.y1)
            except Exception:
                pass

        # Include per-panel colorbar bbox so the panel reserves space for it.
        # Use tight bbox so the colorbar's *tick labels* (e.g. "0.30") are
        # also counted — otherwise neighbouring panels' axis labels poke
        # into the colorbar's number column.
        cb_ax = (panel_colorbars or {}).get(id(ax))
        cb_pos_pre = cb_ax.get_position() if cb_ax is not None else None
        ax_pos_pre = ax.get_position()  # for delta computation
        if cb_ax is not None:
            try:
                cb_bbox = cb_ax.get_tightbbox(renderer).transformed(inv)
            except Exception:
                cb_bbox = cb_ax.get_window_extent(renderer).transformed(inv)
            x0 = min(x0, cb_bbox.x0); x1 = max(x1, cb_bbox.x1)
            y0 = min(y0, cb_bbox.y0); y1 = max(y1, cb_bbox.y1)

        items.append({
            'ax': ax,
            'full_w': x1 - x0,
            'full_h': y1 - y0,
            'ax_dx': ax_bbox.x0 - x0,   # axes offset within full bbox
            'ax_dy': ax_bbox.y0 - y0,
            'ax_w':  ax_bbox.width,
            'ax_h':  ax_bbox.height,
            'cb_ax': cb_ax,
            'cb_pos_pre': cb_pos_pre,
            'ax_pos_pre': ax_pos_pre,
        })

    # ── Pass 2: assign positions left-to-right with wrap ─────────────
    placements = []
    cursor_x = margin
    cumulative_top = 0.0   # inches from top of current logical layout
    row_max_h = 0.0
    row_max_w = 0.0        # widest cursor reached per row (for width shrink)
    used_w_in = 0.0
    for it in items:
        if cursor_x + it['full_w'] > fig_w_in - margin and cursor_x > margin:
            # wrap to next row
            used_w_in = max(used_w_in, row_max_w)
            cursor_x = margin
            cumulative_top += row_max_h + gap
            row_max_h = 0.0
        placements.append((cursor_x, cumulative_top))
        cursor_x += it['full_w'] + gap
        row_max_h = max(row_max_h, it['full_h'])
        # cursor_x has trailing gap; the actual row width ends `gap` earlier
        row_max_w = max(row_max_w, cursor_x - gap + margin)
    used_w_in = max(used_w_in, row_max_w)

    total_h_in = cumulative_top + row_max_h + 2 * margin
    if total_h_in > fig.get_figheight():
        fig.set_figheight(total_h_in)
    # Reclaim unused right-side whitespace: the original figure width was
    # sized assuming the previous (wider) colorbar; flow-layout panels now
    # finish earlier, so shrink the figure to the actual used extent.
    if used_w_in > 0 and used_w_in < fig_w_in:
        fig.set_figwidth(used_w_in)
    fig_h_in = fig.get_figheight()
    fig_w_in = fig.get_figwidth()

    # ── Pass 3: apply positions (figure-relative) ────────────────────
    for it, (x_in, top_in) in zip(items, placements):
        # full bbox top in figure coords (measured from figure bottom)
        full_top_y = fig_h_in - margin - top_in
        full_bot_y = full_top_y - it['full_h']
        ax_x = x_in + it['ax_dx']
        ax_y = full_bot_y + it['ax_dy']
        new_pos_norm = [
            ax_x / fig_w_in,
            ax_y / fig_h_in,
            it['ax_w'] / fig_w_in,
            it['ax_h'] / fig_h_in,
        ]
        it['ax'].set_position(new_pos_norm)

        # Native inset colorbars follow their parent panel automatically.
        # Only legacy figure-coordinate axes need their position rewritten.
        cb_ax = it['cb_ax']
        if cb_ax is not None and cb_ax.get_axes_locator() is None:
            ap = it['ax_pos_pre']
            cp = it['cb_pos_pre']
            # offset of cb origin from panel origin, normalised by panel size
            dx_rel = (cp.x0 - ap.x0) / ap.width if ap.width else 0
            dy_rel = (cp.y0 - ap.y0) / ap.height if ap.height else 0
            cb_w_rel = cp.width / ap.width if ap.width else cp.width
            cb_h_rel = cp.height / ap.height if ap.height else cp.height
            new_panel_w = new_pos_norm[2]
            new_panel_h = new_pos_norm[3]
            cb_ax.set_position([
                new_pos_norm[0] + dx_rel * new_panel_w,
                new_pos_norm[1] + dy_rel * new_panel_h,
                cb_w_rel * new_panel_w,
                cb_h_rel * new_panel_h,
            ])


def _get_vboundnorm(
    vmin: Sequence[VBound],
    vmax: Sequence[VBound],
    vcenter: Sequence[VBound],
    norm: Sequence[Normalize],
    index: int,
    color_vector: Sequence[float],
) -> Tuple[Union[float, None], Union[float, None]]:
    """
    Evaluates the value of vmin, vmax and vcenter, which could be a
    str in which case is interpreted as a percentile and should
    be specified in the form 'pN' where N is the percentile.
    Eg. for a percentile of 85 the format would be 'p85'.
    Floats are accepted as p99.9

    Alternatively, vmin/vmax could be a function that is applied to
    the list of color values (`color_vector`).  E.g.

    def my_vmax(color_vector): np.percentile(color_vector, p=80)


    Parameters
    ----------
    index
        This index of the plot
    color_vector
        List or values for the plot

    Returns
    -------

    (vmin, vmax, vcenter, norm) containing None or float values for
    vmin, vmax, vcenter and matplotlib.colors.Normalize  or None for norm.

    """
    out = []
    for v_name, v in [('vmin', vmin), ('vmax', vmax), ('vcenter', vcenter)]:
        if len(v) == 1:
            # this case usually happens when the user sets eg vmax=0.9, which
            # is internally converted into list of len=1, but is expected that this
            # value applies to all plots.
            v_value = v[0]
        else:
            try:
                v_value = v[index]
            except IndexError:
                logg.error(
                    f"The parameter {v_name} is not valid. If setting multiple {v_name} values,"
                    f"check that the length of the {v_name} list is equal to the number "
                    "of plots. "
                )
                v_value = None

        if v_value is not None:
            if isinstance(v_value, str) and v_value.startswith('p'):
                try:
                    float(v_value[1:])
                except ValueError:
                    logg.error(
                        f"The parameter {v_name}={v_value} for plot number {index + 1} is not valid. "
                        f"Please check the correct format for percentiles."
                    )
                # interpret value of vmin/vmax as quantile with the following syntax 'p99.9'
                v_value = np.nanpercentile(color_vector, q=float(v_value[1:]))
            elif callable(v_value):
                # interpret vmin/vmax as function
                v_value = v_value(color_vector)
                if not isinstance(v_value, float):
                    logg.error(
                        f"The return of the function given for {v_name} is not valid. "
                        "Please check that the function returns a number."
                    )
                    v_value = None
            else:
                try:
                    float(v_value)
                except ValueError:
                    logg.error(
                        f"The given {v_name}={v_value} for plot number {index + 1} is not valid. "
                        f"Please check that the value given is a valid number, a string "
                        f"starting with 'p' for percentiles or a valid function."
                    )
                    v_value = None
        out.append(v_value)
    out.append(norm[0] if len(norm) == 1 else norm[index])
    return tuple(out)


def _wraps_plot_scatter(wrapper):
    import inspect

    params = inspect.signature(embedding).parameters.copy()
    wrapper_sig = inspect.signature(wrapper)
    wrapper_params = wrapper_sig.parameters.copy()

    params.pop("basis")
    params.pop("kwargs")
    wrapper_params.pop("adata")

    params.update(wrapper_params)
    annotations = {
        k: v.annotation
        for k, v in params.items()
        if v.annotation != inspect.Parameter.empty
    }
    if wrapper_sig.return_annotation is not inspect.Signature.empty:
        annotations["return"] = wrapper_sig.return_annotation

    wrapper.__signature__ = inspect.Signature(
        list(params.values()), return_annotation=wrapper_sig.return_annotation
    )
    wrapper.__annotations__ = annotations

    return wrapper


# API


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    edges_arrows=doc_edges_arrows,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def umap(adata, **kwargs) -> Union[Axes, List[Axes], None]:
    r"""Scatter plot in UMAP basis.

    Parameters
    ----------
        adata: Annotated data matrix
        **kwargs: Additional arguments passed to embedding function

    Returns
    -------
        Matplotlib axes or list of axes if show=False
    """
    return embedding(adata, 'umap', **kwargs)


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    edges_arrows=doc_edges_arrows,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def tsne(adata, **kwargs) -> Union[Axes, List[Axes], None]:
    r"""Scatter plot in tSNE basis.

    Parameters
    ----------
        adata: Annotated data matrix
        **kwargs: Additional arguments passed to embedding function

    Returns
    -------
        Matplotlib axes or list of axes if show=False
    """
    return embedding(adata, 'tsne', **kwargs)


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def diffmap(adata, **kwargs) -> Union[Axes, List[Axes], None]:
    """\
    Scatter plot in Diffusion Map basis.

    Parameters
    ----------
    {adata_color_etc}
    {scatter_bulk}
    {show_save_ax}

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Examples
    --------
    .. plot::
        :context: close-figs

        ov.pl.diffmap(adata, color='bulk_labels')

    See also
    --------
    ov.pl.embedding
    """
    return embedding(adata, 'diffmap', **kwargs)


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    edges_arrows=doc_edges_arrows,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def draw_graph(
    adata: AnnData, *, layout = None, **kwargs
) -> Union[Axes, List[Axes], None]:
    """\
    Scatter plot in graph-drawing basis.

    Parameters
    ----------
    {adata_color_etc}
    layout
        One of the :func:`~scanpy.tl.draw_graph` layouts.
        By default, the last computed layout is used.
    {edges_arrows}
    {scatter_bulk}
    {show_save_ax}

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Examples
    --------
    .. plot::
        :context: close-figs

        ov.pl.draw_graph(adata, color=['phase', 'bulk_labels'])

    See also
    --------
    ov.pl.embedding
    """
    if layout is None:
        layout = str(adata.uns['draw_graph']['params']['layout'])
    basis = 'draw_graph_' + layout
    if 'X_' + basis not in adata.obsm_keys():
        raise ValueError(
            'Did not find {} in adata.obs. Did you compute layout {}?'.format(
                'draw_graph_' + layout, layout
            )
        )

    return embedding(adata, basis, **kwargs)


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def pca(
    adata,
    *,
    annotate_var_explained: bool = False,
    show: Optional[bool] = None,
    return_fig: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
) -> Union[Axes, List[Axes], None]:
    r"""Scatter plot in PCA coordinates.

    Parameters
    ----------
        adata: Annotated data matrix
        annotate_var_explained: Annotate explained variance (False)
        show: Show the plot (None)
        return_fig: Return figure object (None)
        save: Save the plot (None)
        **kwargs: Additional arguments passed to embedding function

    Returns
    -------
        Matplotlib axes or list of axes if show=False
    """
    if not annotate_var_explained:
        return embedding(
            adata, 'pca', show=show, return_fig=return_fig, save=save, **kwargs
        )
    else:
        if 'pca' not in adata.obsm.keys() and 'X_pca' not in adata.obsm.keys():
            raise KeyError(
                f"Could not find entry in `obsm` for 'pca'.\n"
                f"Available keys are: {list(adata.obsm.keys())}."
            )

        label_dict = {
            'PC{}'.format(i + 1): 'PC{} ({}%)'.format(i + 1, round(v * 100, 2))
            for i, v in enumerate(adata.uns['pca']['variance_ratio'])
        }

        if return_fig is True:
            # edit axis labels in returned figure
            fig = embedding(adata, 'pca', return_fig=return_fig, **kwargs)
            for ax in fig.axes:
                ax.set_xlabel(label_dict[ax.xaxis.get_label().get_text()])
                ax.set_ylabel(label_dict[ax.yaxis.get_label().get_text()])
            return fig

        else:
            # get the axs, edit the labels and apply show and save from user
            axs = embedding(adata, 'pca', show=False, save=False, **kwargs)
            if isinstance(axs, list):
                for ax in axs:
                    ax.set_xlabel(label_dict[ax.xaxis.get_label().get_text()])
                    ax.set_ylabel(label_dict[ax.yaxis.get_label().get_text()])
            else:
                axs.set_xlabel(label_dict[axs.xaxis.get_label().get_text()])
                axs.set_ylabel(label_dict[axs.yaxis.get_label().get_text()])
            savefig_or_show('pca', show=show, save=save)
            if show is False:
                return axs


@_wraps_plot_scatter
@_doc_params(
    adata_color_etc=doc_adata_color_etc,
    scatter_spatial=doc_scatter_spatial,
    scatter_bulk=doc_scatter_embedding,
    show_save_ax=doc_show_save_ax,
)
def spatial(
    adata,
    *,
    basis: str = "spatial",
    img: Union[np.ndarray, None] = None,
    img_key: Union[str, None, Empty] = _empty,
    library_id: Union[str, None, Empty] = _empty,
    crop_coord: Tuple[int, int, int, int] = None,
    alpha_img: float = 1.0,
    bw: Optional[bool] = False,
    size: float = 1.0,
    scale_factor: Optional[float] = None,
    spot_size: Optional[float] = None,
    na_color: Optional[ColorLike] = None,
    show: Optional[bool] = None,
    return_fig: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
) -> Union[Axes, List[Axes], None]:
    """\
    Scatter plot in spatial coordinates.

    This function allows overlaying data on top of images.
    Use the parameter `img_key` to see the image in the background
    And the parameter `library_id` to select the image.
    By default, `'hires'` and `'lowres'` are attempted.

    Use `crop_coord`, `alpha_img`, and `bw` to control how it is displayed.
    Use `size` to scale the size of the Visium spots plotted on top.

    As this function is designed to for imaging data, there are two key assumptions
    about how coordinates are handled:

    1. The origin (e.g `(0, 0)`) is at the top left – as is common convention
    with image data.

    2. Coordinates are in the pixel space of the source image, so an equal
    aspect ratio is assumed.

    If your anndata object has a `"spatial"` entry in `.uns`, the `img_key`
    and `library_id` parameters to find values for `img`, `scale_factor`,
    and `spot_size` arguments. Alternatively, these values be passed directly.

    Parameters
    ----------
    {adata_color_etc}
    {scatter_spatial}
    {scatter_bulk}
    {show_save_ax}

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Examples
    --------
    This function behaves very similarly to other embedding plots like
    :func:`~omicverse.pl.umap`

    >>> ov.pl.spatial(adata, color="log1p_n_genes_by_counts")

    See Also
    --------
    :tutorial:`spatial/basic-analysis`
        Tutorial on spatial analysis.
    """
    from ._spatial import spatial as canonical_spatial

    return canonical_spatial(
        adata,
        basis=basis,
        img=img,
        img_key=img_key,
        library_id=library_id,
        crop_coord=crop_coord,
        alpha_img=alpha_img,
        bw=bw,
        size=size,
        scale_factor=scale_factor,
        spot_size=spot_size,
        na_color=na_color,
        show=show,
        return_fig=return_fig,
        save=save,
        **kwargs,
    )


# Helpers
def _components_to_dimensions(
    components: Optional[Union[str, Collection[str]]],
    dimensions: Optional[Union[Collection[int], Collection[Collection[int]]]],
    *,
    projection: Literal["2d", "3d"] = "2d",
    total_dims: int,
) -> List[Collection[int]]:
    """Normalize components/ dimensions args for embedding plots."""
    # TODO: Deprecate components kwarg
    ndims = {"2d": 2, "3d": 3}[projection]
    if components is None and dimensions is None:
        dimensions = [tuple(i for i in range(ndims))]
    elif components is not None and dimensions is not None:
        raise ValueError("Cannot provide both dimensions and components")

    # TODO: Consider deprecating this
    # If components is not None, parse them and set dimensions
    if components == "all":
        dimensions = list(combinations(range(total_dims), ndims))
    elif components is not None:
        if isinstance(components, str):
            components = [components]
        # Components use 1 based indexing
        dimensions = [[int(dim) - 1 for dim in c.split(",")] for c in components]

    if all(isinstance(el, Integral) for el in dimensions):
        dimensions = [dimensions]
    # if all(isinstance(el, Collection) for el in dimensions):
    for dims in dimensions:
        if len(dims) != ndims or not all(isinstance(d, Integral) for d in dims):
            raise ValueError()

    return dimensions


def _add_categorical_legend(
    ax,
    color_source_vector,
    palette: dict,
    legend_loc: str,
    legend_fontweight,
    legend_fontsize,
    legend_fontoutline,
    multi_panel,
    na_color,
    na_in_legend: bool,
    scatter_array=None,
):
    """Add a legend to the passed Axes."""
    if na_in_legend and pd.isnull(color_source_vector).any():
        if "NA" in color_source_vector:
            raise NotImplementedError(
                "No fallback for null labels has been defined if NA already in categories."
            )
        # Ensure color_source_vector is categorical before adding categories
        if not hasattr(color_source_vector, 'add_categories'):
            color_source_vector = pd.Categorical(color_source_vector)
        color_source_vector = color_source_vector.add_categories("NA").fillna("NA")
        palette = palette.copy()
        palette["NA"] = na_color
    if color_source_vector.dtype == bool:
        cats = pd.Categorical(color_source_vector.astype(str)).categories
    else:
        # Safely get categories - handle both Categorical and Series objects
        if hasattr(color_source_vector, 'categories'):
            cats = color_source_vector.categories
        else:
            # Convert to categorical if it's not already
            cats = pd.Categorical(color_source_vector).categories

    if multi_panel is True:
        # Shrink current axis by 10% to fit legend and match
        # size of plots that are not categorical
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.91, box.height])

    if legend_loc == 'right margin':
        for label in cats:
            # `_get_palette` normalises category keys via `str(x)` (see
            # `_obs_to_categorical`), so palette keys are strings even when
            # the underlying Categorical still carries ints/bools/etc.
            # Fall back to the stringified label, then na_color, before
            # raising so integer-keyed `adata.obs` columns don't blow up
            # the legend loop.
            color = palette.get(label)
            if color is None:
                color = palette.get(str(label), na_color)
            ax.scatter([], [], c=color, label=label)
        ax.legend(
            frameon=False,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            ncol=(1 if len(cats) <= 14 else 2 if len(cats) <= 30 else 3),
            fontsize=legend_fontsize,
        )
    elif legend_loc == 'on data':
        # identify centroids to put labels

        # Use a category array without index alignment to keep coords and labels in sync.
        if isinstance(color_source_vector, pd.Categorical):
            groupby_key = color_source_vector
        else:
            groupby_key = pd.Categorical(np.asarray(color_source_vector))

        all_pos = (
            pd.DataFrame(scatter_array, columns=["x", "y"])
            .groupby(groupby_key, observed=True)
            .median()
            # Have to sort_index since if observed=True and categorical is unordered
            # the order of values in .index is undefined. Related issue:
            # https://github.com/pandas-dev/pandas/issues/25167
            .sort_index()
        )

        # Convert legend_fontoutline to PathEffect list if needed
        if legend_fontoutline is not None:
            if isinstance(legend_fontoutline, (int, float)):
                # Create white stroke outline with specified width
                text_path_effects = [
                    patheffects.withStroke(linewidth=legend_fontoutline, foreground='white')
                ]
            elif isinstance(legend_fontoutline, list):
                # Already a list of PathEffects
                text_path_effects = legend_fontoutline
            else:
                # Single PathEffect object
                text_path_effects = [legend_fontoutline]
        else:
            text_path_effects = None

        for label, x_pos, y_pos in all_pos.itertuples():
            ax.text(
                x_pos,
                y_pos,
                label,
                weight=legend_fontweight,
                verticalalignment='center',
                horizontalalignment='center',
                fontsize=legend_fontsize,
                path_effects=text_path_effects,
            )


def _get_basis(adata: AnnData, basis: str) -> np.ndarray:
    """Get array for basis from anndata. Just tries to add 'X_'."""
    if basis in adata.obsm:
        return adata.obsm[basis]
    elif f"X_{basis}" in adata.obsm:
        return adata.obsm[f"X_{basis}"]
    else:
        raise KeyError(f"Could not find '{basis}' or 'X_{basis}' in .obsm")


def _safe_check_obs_columns(adata, key):
    """Safely check if a key exists in adata.obs, compatible with both pandas and Rust backends."""
    try:
        # For pandas DataFrame
        if hasattr(adata.obs, 'columns'):
            return key in adata.obs.columns
        # For Rust/Polars backends that don't have .columns attribute
        else:
            # Try to access the column directly - if it exists, this won't raise an error
            try:
                _ = adata.obs[key]
                return True
            except (KeyError, IndexError):
                return False
    except Exception:
        return False

def _safe_check_var_names(adata, key):
    """Safely check if a key exists in adata.var_names, compatible with both pandas and Rust backends."""
    try:
        return key in adata.var_names
    except Exception:
        # Fallback for Rust backends
        try:
            if hasattr(adata.var_names, '__contains__'):
                return key in adata.var_names
            else:
                # Convert to list and check
                return key in list(adata.var_names)
        except Exception:
            return False

def _get_color_source_vector(
    adata, value_to_plot, use_raw=False, gene_symbols=None, layer=None, groups=None
):
    """
    Get array from adata that colors will be based on.
    Compatible with both pandas and Rust/Polars anndata backends.
    """
    if value_to_plot is None:
        # Points will be plotted with `na_color`. Ideally this would work
        # with the "bad color" in a color map but that throws a warning. Instead
        # _color_vector handles this.
        # https://github.com/matplotlib/matplotlib/issues/18294
        return np.broadcast_to(np.nan, adata.n_obs)
    
    # Safe checks for obs and var
    in_obs = _safe_check_obs_columns(adata, value_to_plot)
    # When use_raw is True, check raw.var_names; otherwise check var_names
    if use_raw and adata.raw is not None:
        in_var = _safe_check_var_names(adata.raw, value_to_plot)
    else:
        in_var = _safe_check_var_names(adata, value_to_plot)

    # Handle gene symbols - convert to actual gene names if needed
    if (
        gene_symbols is not None
        and not in_obs
        and not in_var
    ):
        # We should probably just make an index for this, and share it over runs
        try:
            if use_raw and adata.raw is not None:
                value_to_plot = adata.raw.var.index[adata.raw.var[gene_symbols] == value_to_plot][0]
                in_var = _safe_check_var_names(adata.raw, value_to_plot)  # Update after conversion
            else:
                value_to_plot = adata.var.index[adata.var[gene_symbols] == value_to_plot][0]
                in_var = _safe_check_var_names(adata, value_to_plot)  # Update after conversion
        except (IndexError, KeyError):
            pass  # Will be handled in the error case below

    # Determine the source of the data
    if in_obs:
        # Data is in adata.obs (metadata)
        values = adata.obs[value_to_plot]
    elif use_raw and in_var:
        # Data is gene expression from raw
        values = adata.raw.obs_vector(value_to_plot)
    elif in_var:
        # Data is gene expression from processed data
        values = adata.obs_vector(value_to_plot, layer=layer)
    else:
        # Last resort - try obs_vector which might handle other cases
        try:
            values = adata.obs_vector(value_to_plot, layer=layer)
        except (KeyError, AttributeError):
            raise KeyError(f"Could not find '{value_to_plot}' in adata.obs or adata.var_names")

    # 🔧 修改：只对真正的字符串/对象数据转为分类，避免误转数值型基因表达数据
    if not isinstance(values.dtype, pd.CategoricalDtype):
        arr = np.asarray(values)
        # 只有当数据是字符串类型且有重复值时才转为分类
        if arr.dtype.kind in ("U", "S", "O"):                  # 字符串/对象
            # 只在确实是"类别"而非全唯一时才转
            if pd.unique(arr).size < arr.size:
                values = pd.Categorical(arr)
        # 对于数值型数据（基因表达），保持原样不转为分类
            
    if groups and isinstance(values.dtype, pd.CategoricalDtype):
        # `values` may be a categorical-dtyped Series (needs the `.cat`
        # accessor) or a bare pd.Categorical (methods live on the object).
        cat = values.cat if isinstance(values, pd.Series) else values
        values = cat.remove_categories(cat.categories.difference(groups))
    return values


def _get_palette(adata, values_key: str, palette=None):
    """
    返回 {category -> hex}。
    - Python anndata：优先读 uns['<key>_colors']，不足/缺失再读 '<key>_colors_rgba'。
    - Rust/Polars：只读 uns['<key>_colors_rgba']（避免读取字符串数组触发 PanicException）。
    - 若都没有或长度不足，按默认规则生成，并写回：
        * Rust/Polars：只写 '<key>_colors_rgba' (float32 RGBA)
        * Python anndata：同时写 '<key>_colors' (unicode) 和 '<key>_colors_rgba'
    """
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    from matplotlib import rcParams
    from matplotlib.colors import to_hex, to_rgba, is_color_like
    from cycler import Cycler

    color_key = f"{values_key}_colors"
    color_key_rgba = f"{values_key}_colors_rgba"

    # --------- 判断是否 Rust/Polars 后端（完全避开读取字符串数组） ---------
    def _is_rust_backend():
        try:
            if type(adata.obs).__name__.endswith("PyDataFrameElem"):  # 你前面贴过
                return True
        except Exception:
            pass
        try:
            if type(adata.uns).__name__.endswith("PyElemCollection"):
                return True
        except Exception:
            pass
        # 兜底：模块名里出现 snapatac2 / pyanndata 也算
        m = type(adata).__module__
        return ("snapatac2" in m) or ("pyanndata" in m)

    IS_RUST = _is_rust_backend()

    # --------- 把 obs 列转成 pandas.Categorical，并拿到有序类别 ---------
    def _obs_to_categorical(adata, key):
        s = adata.obs[key]
        try:
            import polars as pl
        except Exception:
            pl = None

        if s.__class__.__module__.startswith("pandas"):
            if isinstance(s.dtype, pd.CategoricalDtype):
                cats = [str(x) for x in s.cat.categories]
                return pd.Categorical(pd.Series(s).astype(str), categories=cats)
            if getattr(s, "dtype", None) == bool:
                return pd.Categorical(pd.Series(s).astype(str))
            return pd.Categorical(pd.Series(s, dtype="string"))

        if pl is not None and isinstance(s, pl.Series):
            if s.dtype == pl.Boolean:
                return pd.Categorical(pd.Series(s.to_list()).astype(str), categories=["False", "True"])
            if s.dtype == pl.Categorical and hasattr(s.cat, "get_categories"):
                cats = [str(x) for x in s.cat.get_categories().to_list()]
                return pd.Categorical(pd.Series(s.to_list()).astype(str), categories=cats)
            arr = [str(x) for x in s.cast(pl.Utf8).to_list()]
            try:
                from natsort import natsorted
                cats = natsorted(pd.unique(pd.Series(arr)).tolist())
            except Exception:
                cats = sorted(pd.unique(pd.Series(arr)).tolist(), key=str)
            return pd.Categorical(arr, categories=cats)

        # 兜底
        arr = np.asarray(s, dtype=object)
        if arr.size and isinstance(arr.flat[0], (np.bool_, bool)):
            return pd.Categorical(pd.Series(arr).astype(str))
        return pd.Categorical([str(x) for x in arr])

    values = _obs_to_categorical(adata, values_key)
    cats = list(values.categories)
    n_cat = len(cats)

    # --------- 写颜色（双轨：Rust 仅 RGBA；Python 字符串+RGBA） ---------
    def _write_colors(hex_list):
        rgba = np.asarray([to_rgba(h) for h in hex_list], dtype=np.float32)
        adata.uns[color_key_rgba] = rgba
        if not IS_RUST:
            adata.uns[color_key] = np.asarray(hex_list, dtype="U16")

    # --------- 处理用户传入的 palette ---------
    if palette is not None:
        if isinstance(palette, dict):
            hex_list = [to_hex(palette.get(cat, "#808080"), keep_alpha=True) for cat in cats]
            _write_colors(hex_list)
            return dict(zip(cats, hex_list))

        if isinstance(palette, str) and (palette in mpl.colormaps):
            cmap = mpl.colormaps[palette]
            denom = max(n_cat - 1, 1)
            hex_list = [to_hex(cmap(i/denom), keep_alpha=True) for i in range(n_cat)]
            _write_colors(hex_list)
            return dict(zip(cats, hex_list))

        if isinstance(palette, (list, tuple)):
            try:
                seq = [(c if is_color_like(c) else additional_colors[c]) for c in palette]
            except Exception as e:
                raise ValueError(f"Invalid color in palette: {e}") from None
            hex_list = [to_hex(seq[i % len(seq)], keep_alpha=True) for i in range(n_cat)]
            _write_colors(hex_list)
            return dict(zip(cats, hex_list))

        if isinstance(palette, Cycler):
            cc = palette()
            hex_list = [to_hex(next(cc)["color"], keep_alpha=True) for _ in range(n_cat)]
            _write_colors(hex_list)
            return dict(zip(cats, hex_list))

        raise ValueError(
            "palette must be a matplotlib colormap name, a sequence of colors, "
            "a dict {category: color}, or a cycler(color=...)."
        )

    # --------- 未传 palette：尝试读取现存颜色 ---------
    hex_list = None
    if IS_RUST:
        # Rust：只读 RGBA；不要碰 '<key>_colors'，哪怕加 try 也会冒泡
        try:
            v = adata.uns[color_key_rgba]
            arr = v.to_numpy() if hasattr(v, "to_numpy") else np.asarray(v)
            if arr.ndim == 2 and arr.shape[1] in (3,4):
                hex_list = [to_hex(tuple(row), keep_alpha=True) for row in arr]
        except BaseException:  # 注意：PanicException 可能不是 Exception
            hex_list = None
    else:
        # Python：优先读字符串数组
        try:
            v = adata.uns[color_key]
            if hasattr(v, "to_list"):
                v = v.to_list()
            arr = np.asarray(v)
            if arr.dtype.kind in ("U","S","O"):
                hex_list = [str(x) for x in (arr.tolist() if isinstance(arr, np.ndarray) else list(arr))]
        except Exception:
            hex_list = None
        # 再读 RGBA 兜底
        if hex_list is None:
            try:
                v = adata.uns[color_key_rgba]
                arr = v.to_numpy() if hasattr(v, "to_numpy") else np.asarray(v)
                if arr.ndim == 2 and arr.shape[1] in (3,4):
                    hex_list = [to_hex(tuple(row), keep_alpha=True) for row in arr]
            except Exception:
                hex_list = None

    # --------- 若仍无/长度不足：生成默认并写回 ---------
    if (hex_list is None) or (len(hex_list) < n_cat):
        base = rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(base) >= n_cat:
            cc = rcParams["axes.prop_cycle"]()
            hex_list = [to_hex(next(cc)["color"], keep_alpha=True) for _ in range(n_cat)]
        else:
            try:
                from ..pl._palette import sc_color, palette_56, palette_112
            except Exception:
                sc_color = palette_56 = palette_112 = None
            if sc_color is not None and n_cat <= len(sc_color):
                hex_list = [to_hex(c, keep_alpha=True) for c in sc_color[:n_cat]]
            elif palette_56 is not None and n_cat <= 56:
                hex_list = [to_hex(c, keep_alpha=True) for c in palette_56[:n_cat]]
            elif palette_112 is not None and n_cat <= 112:
                hex_list = [to_hex(c, keep_alpha=True) for c in palette_112[:n_cat]]
            else:
                hex_list = ["#808080"] * n_cat
                logg.info(f"the obs value {values_key!r} has many categories; using uniform grey.")
        _write_colors(hex_list)

    return dict(zip(cats, hex_list))



def _color_vector(
    adata, values_key: str, values, palette, na_color="lightgray"
) -> Tuple[np.ndarray, bool]:
    """
    Map array of values to array of hex (plus alpha) codes.

    For categorical data, the return value is list of colors taken
    from the category palette or from the given `palette` value.

    For continuous values, the input array is returned (may change in future).
    """
    ###
    # when plotting, the color of the dots is determined for each plot
    # the data is either categorical or continuous and the data could be in
    # 'obs' or in 'var'
    to_hex = partial(colors.to_hex, keep_alpha=True)
    if values_key is None:
        return np.broadcast_to(to_hex(na_color), adata.n_obs), False
    if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == bool:
        if values.dtype == bool:
            values = pd.Categorical(values.astype(str))
        color_map = {
            k: to_hex(v)
            for k, v in _get_palette(adata, values_key, palette=palette).items()
        }
        # If color_map does not have unique values, this can be slow as the
        # result is not categorical
        color_vector = pd.Categorical(values.map(color_map))

        # Set color to 'missing color' for all missing values
        if color_vector.isna().any():
            color_vector = color_vector.add_categories([to_hex(na_color)])
            color_vector = color_vector.fillna(to_hex(na_color))
        return color_vector, True
    elif not isinstance(values.dtype, pd.CategoricalDtype):
        # 将数值列强制转为 float array，避免 object dtype 传入 matplotlib 时报错
        # （例如 obs 列存为 Python int / object，circles() 无法识别）
        try:
            return np.asarray(pd.to_numeric(values, errors="raise"), dtype=float), False
        except (ValueError, TypeError):
            return values, False


def _basis2name(basis):
    """
    converts the 'basis' into the proper name.
    """

    component_name = (
        'DC'
        if basis == 'diffmap'
        else 'tSNE'
        if basis == 'tsne'
        else 'UMAP'
        if basis == 'umap'
        else 'PC'
        if basis == 'pca'
        else basis.replace('draw_graph_', '').upper()
        if 'draw_graph' in basis
        else basis
    )
    return component_name


def _check_spot_size(
    spatial_data: Optional[Mapping], spot_size: Optional[float]
) -> float:
    """
    Resolve spot_size value.

    This is a required argument for spatial plots.
    """
    if spatial_data is None and spot_size is None:
        raise ValueError(
            "When .uns['spatial'][library_id] does not exist, spot_size must be "
            "provided directly."
        )
    elif spot_size is None:
        return spatial_data['scalefactors']['spot_diameter_fullres']
    else:
        return spot_size


def _check_scale_factor(
    spatial_data: Optional[Mapping],
    img_key: Optional[str],
    scale_factor: Optional[float],
) -> float:
    """Resolve scale_factor, defaults to 1."""
    if scale_factor is not None:
        return scale_factor
    elif spatial_data is not None and img_key is not None:
        return spatial_data['scalefactors'][f"tissue_{img_key}_scalef"]
    else:
        return 1.0


def _check_spatial_data(
    uns: Mapping, library_id: Union[str, None, Empty]
) -> Tuple[Optional[str], Optional[Mapping]]:
    """
    Given a mapping, try and extract a library id/ mapping with spatial data.

    Assumes this is `.uns` from how we parse visium data.
    """
    spatial_mapping = uns.get("spatial", {})
    if library_id is _empty:
        if len(spatial_mapping) > 1:
            raise ValueError(
                "Found multiple possible libraries in `.uns['spatial']. Please specify."
                f" Options are:\n\t{list(spatial_mapping.keys())}"
            )
        elif len(spatial_mapping) == 1:
            library_id = list(spatial_mapping.keys())[0]
        else:
            library_id = None
    if library_id is not None:
        spatial_data = spatial_mapping[library_id]
    else:
        spatial_data = None
    return library_id, spatial_data


def _check_img(
    spatial_data: Optional[Mapping],
    img: Optional[np.ndarray],
    img_key: Union[None, str, Empty],
    bw: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Resolve image for spatial plots.
    """
    if img is None and spatial_data is not None and img_key is _empty:
        img_key = next(
            (k for k in ['hires', 'lowres'] if k in spatial_data['images']),
        )  # Throws StopIteration Error if keys not present
    if img is None and spatial_data is not None and img_key is not None:
        img = spatial_data["images"][img_key]
    if bw:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img, img_key


def _check_crop_coord(
    crop_coord: Optional[tuple],
    scale_factor: float,
) -> Tuple[float, float, float, float]:
    """Handle cropping with image or basis."""
    if crop_coord is None:
        return None
    if len(crop_coord) != 4:
        raise ValueError("Invalid crop_coord of length {len(crop_coord)}(!=4)")
    crop_coord = tuple(c * scale_factor for c in crop_coord)
    return crop_coord


def _check_na_color(
    na_color: Optional[ColorLike], *, img: Optional[np.ndarray] = None
) -> ColorLike:
    if na_color is None:
        if img is not None:
            na_color = (0.0, 0.0, 0.0, 0.0)
        else:
            na_color = "lightgray"
    return na_color


def _broadcast_args(*args):
    """Broadcasts arguments to a common length."""
    from itertools import repeat

    lens = [len(arg) for arg in args]
    longest = max(lens)
    if not (set(lens) == {1, longest} or set(lens) == {longest}):
        raise ValueError(f"Could not broadast together arguments with shapes: {lens}.")
    return list(
        [[arg[0] for _ in range(longest)] if len(arg) == 1 else arg for arg in args]
    )

def _embedding(
    adata: AnnData,
    basis: str,
    *,
    color: Union[str, Sequence[str], None] = None,
    gene_symbols: Optional[str] = None,
    use_raw: Optional[bool] = None,
    sort_order: bool = True,
    edges: bool = False,
    edges_width: float = 0.1,
    edges_color: Union[str, Sequence[float], Sequence[str]] = 'grey',
    neighbors_key: Optional[str] = None,
    arrows: bool = False,
    arrows_kwds: Optional[Mapping[str, Any]] = None,
    groups: Optional[str] = None,
    components: Union[str, Sequence[str]] = None,
    dimensions: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    layer: Optional[str] = None,
    projection: Literal['2d', '3d'] = '2d',
    scale_factor: Optional[float] = None,
    color_map: Union[Colormap, str, None] = None,
    cmap: Union[Colormap, str, None] = None,
    palette: Union[str, Sequence[str], Cycler, None] = None,
    na_color: ColorLike = "lightgray",
    na_in_legend: bool = True,
    size: Union[float, Sequence[float], None] = None,
    frameon: Optional[bool] = None,
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = 'bold',
    legend_loc: str = 'right margin',
    legend_fontoutline: Optional[int] = None,
    colorbar_loc: Optional[str] = "right",
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    add_outline: Optional[bool] = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ('black', 'white'),
    ncols: int = 4,
    hspace: float = 0.25,
    wspace: Optional[float] = None,
    title: Union[str, Sequence[str], None] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    marker: Union[str, Sequence[str]] = '.',
    **kwargs,
) -> Union[Figure, Axes, None]:
    """\
    Scatter plot for user specified embedding basis (e.g. umap, pca, etc)

    Parameters
    ----------
        adata: Annotated data matrix.
        basis: Name of the `obsm` basis to use.

    Returns
    -------
        If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.
    """

    return embedding(adata=adata, basis=basis, color=color, 
                     gene_symbols=gene_symbols, use_raw=use_raw, 
                     sort_order=sort_order, edges=edges, 
                     edges_width=edges_width, edges_color=edges_color, 
                     neighbors_key=neighbors_key, arrows=arrows, 
                     arrows_kwds=arrows_kwds, groups=groups, 
                     components=components, dimensions=dimensions, 
                     layer=layer, projection=projection, scale_factor=scale_factor,
                       color_map=color_map, cmap=cmap, palette=palette, 
                       na_color=na_color, na_in_legend=na_in_legend, 
                       size=size, frameon=frameon, legend_fontsize=legend_fontsize, 
                       legend_fontweight=legend_fontweight, legend_loc=legend_loc, 
                       legend_fontoutline=legend_fontoutline, colorbar_loc=colorbar_loc, 
                       vmax=vmax, vmin=vmin, vcenter=vcenter, norm=norm, 
                       add_outline=add_outline, outline_width=outline_width, 
                       outline_color=outline_color, ncols=ncols, hspace=hspace,
                         wspace=wspace, title=title, show=show, save=save, ax=ax,
                           return_fig=return_fig, marker=marker, **kwargs)


# === drop-in replacement: pandas / polars compatible ===


import numpy as np
import pandas as pd
from typing import Mapping, Sequence
from cycler import Cycler
import matplotlib as mpl
from matplotlib.colors import is_color_like, to_hex
from natsort import natsorted

def _obs_series(adata, key):
    """兼容 pandas/Polars 的 obs 列取值。"""
    try:
        return adata.obs[key]
    except Exception as e:
        raise KeyError(f"obs does not contain column {key!r}") from e

def _obs_categories_ordered(adata, key) -> list[str]:
    """获取分类类别（按已有顺序）；若不是分类，则取唯一值并自然排序。"""
    s = _obs_series(adata, key)

    # pandas.Series
    if s.__class__.__module__.startswith("pandas"):
        if isinstance(s.dtype, pd.CategoricalDtype):
            cats = list(s.cat.categories)
        else:
            cats = list(pd.unique(pd.Series(s, dtype="string")))
            cats = [str(x) for x in cats]
            cats = natsorted(cats)
        return [str(x) for x in cats]

    # Polars.Series
    import polars as pl
    if pl is not None and isinstance(s, pl.Series):
        if s.dtype == pl.Boolean:
            return ["False", "True"]  # 与 pandas.bool -> str 后一样的顺序
        if s.dtype == pl.Categorical and hasattr(s.cat, "get_categories"):
            cats = s.cat.get_categories().to_list()
        else:
            # 非分类列：取唯一字符串并自然排序
            cats = s.cast(pl.Utf8).unique().to_list()
            cats = natsorted([str(x) for x in cats])
        return [str(x) for x in cats]

    # 兜底：任何 array-like
    arr = np.asarray(s, dtype=object)
    arr = arr[~pd.isnull(arr)]
    return [str(x) for x in natsorted(np.unique(arr))]

def _set_colors_for_categorical_obs(
    adata, value_to_plot: str, palette: Union[str, Sequence[str], Cycler, Mapping[str, str]]
):
    """Set `adata.uns[f'{value_to_plot}_colors']` according to the given palette.
    兼容 pandas/Polars；若 palette 是 dict，会按类别键匹配。
    """
    cats = _obs_categories_ordered(adata, value_to_plot)
    n = len(cats)
    color_key = f"{value_to_plot}_colors"

    # 1) 处理不同类型的 palette，生成长度为 n 的颜色列表
    if isinstance(palette, Mapping):
        # dict: {category: color}
        # 缺失的类别用默认色补齐
        base_cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", None) or [
            to_hex(mpl.colormaps["tab20"](i / 19)) for i in range(20)
        ]
        colors_list = []
        for i, cat in enumerate(cats):
            c = palette.get(cat, base_cycle[i % len(base_cycle)])
            colors_list.append(c)

    elif isinstance(palette, str) and (palette in mpl.colormaps):
        cmap = mpl.colormaps[palette]
        denom = max(n - 1, 1)
        colors_list = [to_hex(cmap(i / denom), keep_alpha=True) for i in range(n)]

    else:
        # Sequence 或 Cycler
        if isinstance(palette, Sequence) and not isinstance(palette, str):
            # 校验颜色合法性，并转为 Cycler
            try:
                _color_list = [
                    (color if is_color_like(color) else additional_colors[color])
                    for color in palette
                ]
            except KeyError as e:
                raise ValueError(
                    f"The following color value of the given palette is not valid: {e.args[0]!r}"
                ) from None
            if len(_color_list) < n:
                logg.warning(
                    "Length of palette colors is smaller than the number of categories "
                    f"(palette length: {len(_color_list)}, categories length: {n}). "
                    "Some categories will have the same color."
                )
            from cycler import Cycler, cycler
            palette = cycler(color=_color_list)

        if not isinstance(palette, Cycler):
            raise ValueError(
                "Please check that 'palette' is a valid matplotlib colormap name, "
                "a list/tuple of colors, or a cycler with key='color'."
            )
        if "color" not in palette.keys:
            raise ValueError("Please set the palette key 'color'.")

        cc = palette()
        colors_list = [to_hex(next(cc)["color"], keep_alpha=True) for _ in range(n)]

    # 2) 统一为 hex 并写入 adata.uns
    #adata.uns[color_key] = [to_hex(c, keep_alpha=True) for c in colors_list]

    _uns_put_colors_dual(adata, color_key, [to_hex(c, keep_alpha=True) for c in colors_list])

import numpy as np
from matplotlib.colors import to_hex

def _uns_put_colors(adata, key, colors_list):
    """
    兼容 anndata(pandas) 与 SnapATAC2(Rust) 的 .uns 颜色写入：
    - 优先写入 NumPy Unicode 定型数组（<U…）
    - 若仍不行，再尝试写入 Polars Series[Utf8]
    - 最后兜底回 Python list（给纯 anndata 用）
    """
    # 统一成 hex 字符串
    colors_list = [
        (c if isinstance(c, str) and c.startswith("#") else to_hex(c, keep_alpha=True))
        for c in colors_list
    ]
    # ① Rust 端最友好：定长 Unicode 数组（长度给宽一点）
    arr = np.asarray(colors_list, dtype="U16")
    try:
        adata.uns[key] = arr
        return
    except TypeError:
        pass

    # ② 尝试 Polars Series[Utf8]
    try:
        import polars as pl
        adata.uns[key] = pl.Series(name=key, values=colors_list, dtype=pl.Utf8)
        return
    except Exception:
        pass

    # ③ 兜底：Python 列表（纯 Python anndata 没问题）
    adata.uns[key] = list(colors_list)


import numpy as np
from matplotlib.colors import to_hex, to_rgba
from .._anndata_compat import sanitize_key as _safe_key

def _uns_supports_str_array(uns) -> bool:
    """检测 .uns 是否能安全读回“字符串数组”。Rust 端通常 False。"""
    try:
        uns["_ov_probe"] = np.asarray(["#000000"], dtype="U9")
        _ = uns["_ov_probe"]        # 读一遍
        del uns["_ov_probe"]
        return True
    except Exception:
        try: del uns["_ov_probe"]
        except Exception: pass
        return False

def _uns_put_colors_dual(adata, name: str, colors_list):
    """
    双轨写入：
    - 一定写: {name}_colors_rgba -> (n,4) float32  RGBA  (Rust 端稳定可读)
    - 能写就写: {name}_colors -> <U… 字符串数组（Python anndata 友好）
    """
    # 统一成 hex
    hex_list = [
        c if (isinstance(c, str) and c.startswith("#")) else to_hex(c, keep_alpha=True)
        for c in colors_list
    ]
    # ① RGBA 始终写（Rust/pyanndata 最稳）
    rgba = np.asarray([to_rgba(h) for h in hex_list], dtype=np.float32)
    adata.uns[f"{_safe_key(name)}_colors_rgba"] = rgba

    # ② 若支持字符串数组读取，再额外写 …_colors
    adata.uns[f"{_safe_key(name)}_colors"] = np.asarray(hex_list, dtype="U16")

def _uns_read_colors_dual(adata, name: str):
    """
    读颜色：先试 …_colors（字符串），失败则用 …_colors_rgba（float32）并转回 hex。
    返回 List[str]（#RRGGBB(AA)）
    """
    # 先试字符串数组（Python anndata 情况）
    try:
        v = adata.uns[f"{_safe_key(name)}_colors"]
        if hasattr(v, "to_list"):      # Polars Series
            v = v.to_list()
        arr = np.asarray(v)
        if arr.dtype.kind in ("U", "S", "O"):
            return [str(x) for x in (arr.tolist() if isinstance(arr, np.ndarray) else list(arr))]
    except Exception:
        pass  # Rust 端可能抛 PanicException 或 TypeError

    # 降级到 RGBA
    v = adata.uns[f"{_safe_key(name)}_colors_rgba"]
    arr = v.to_numpy() if hasattr(v, "to_numpy") else np.asarray(v)
    return [to_hex(tuple(row), keep_alpha=True) for row in arr]
