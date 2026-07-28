# +
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from .._registry import register_function
from ._plotdata import get_values

def to_rgb_grayscale(img, p_low: float = 1.0, p_high: float = 99.0):
    """Convert a 2-D grayscale image to a percentile-clipped RGB stack.

    matplotlib's ``imshow`` applies a default *viridis* colormap to 2-D
    arrays. When the image is a morphology stain (DAPI, membrane, RNA) and
    we want it rendered as true grayscale alongside polygon overlays, this
    silently colour-maps the channel and washes structural detail into a
    uniform purple-to-yellow ramp. Passing a 3-channel array bypasses the
    colormap entirely so the intensities are rendered verbatim.

    The percentile clip handles long-tailed stain distributions: a few
    bright outliers would otherwise compress the dynamic range of the rest
    of the image to nearly black.

    Parameters
    ----------
    img : np.ndarray
        2-D image of any numeric dtype.
    p_low, p_high : float
        Percentile clip range over the image's *non-zero* pixels.
        Defaults: 1st / 99th percentiles.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` float32 array in the [0, 1] range. Suitable to drop
        directly into ``adata.uns['spatial'][lib]['images'][<key>]`` or
        passed to ``imshow`` without a ``cmap`` kwarg.
    """
    arr = np.asarray(img, dtype=np.float32)
    nz = arr[arr > 0]
    if nz.size:
        lo = float(np.percentile(nz, p_low))
        hi = float(np.percentile(nz, p_high))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.stack([arr, arr, arr], axis=-1)


def html_to_rgb(html_color):
    r"""
    Convert HTML hex color code to RGB tuple.
    
    Parameters
    ----------
    html_color : str
        HTML hex color string (for example ``'#FF0000'`` or ``'FF0000'``).
        
    Returns
    -------
    Tuple[int, int, int]
        RGB tuple with values in ``[0, 255]``.
    """
    # 去掉颜色代码前的 `#`
    html_color = html_color.lstrip('#')

    # 处理简写形式的颜色代码（如 `#RGB` 或 `#RGBA`）
    if len(html_color) == 3:
        html_color = ''.join([c*2 for c in html_color])
    elif len(html_color) == 4:
        html_color = ''.join([c*2 for c in html_color])

    # 只保留前六位颜色信息，忽略透明度
    if len(html_color) == 6 or len(html_color) == 8:
        html_color = html_color[:6]
    else:
        raise ValueError("Invalid HTML color code length")

    # 将十六进制颜色代码转换为 RGB 元组
    rgb_color = tuple(int(html_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb_color

def get_rgb_function(cmap, min_value, max_value):
    r"""
    Generate a function to map continuous values to RGB using colormap.
    
    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Colormap used for value-to-color mapping.
    min_value : float
        Lower bound of mapped value range.
    max_value : float
        Upper bound of mapped value range.
        
    Returns
    -------
    Callable
        Function mapping numeric arrays to RGBA values.
    """

    if min_value > max_value:
        raise ValueError("Max_value should be greater or than min_value.")

    if min_value == max_value:
        warnings.warn(
            "Max_color is equal to min_color. It might be because of the data or bad parameter choice. "
            "If you are using plot_contours function try increasing max_color_quantile parameter and"
            "removing cell types with all zero values."
        )

        def func_equal(x):
            factor = 0 if max_value == 0 else 0.5
            return cmap(np.ones_like(x) * factor)

        return func_equal

    def func(x):
        return cmap((np.clip(x, min_value, max_value) - min_value) / (max_value - min_value))

    return func


def rgb_to_ryb(rgb):
    r"""
    Convert colors from RGB colorspace to RYB (red-yellow-blue) colorspace.
    
    Parameters
    ----------
    rgb : array-like
        RGB color array with shape ``(N, 3)`` or ``(3,)``.
        
    Returns
    -------
    numpy.ndarray
        RYB color array with same shape as input.
    """
    rgb = np.array(rgb)
    if len(rgb.shape) == 1:
        rgb = rgb[np.newaxis, :]

    white = rgb.min(axis=1)
    black = (1 - rgb).min(axis=1)
    rgb = rgb - white[:, np.newaxis]

    yellow = rgb[:, :2].min(axis=1)
    ryb = np.zeros_like(rgb)
    ryb[:, 0] = rgb[:, 0] - yellow
    ryb[:, 1] = (yellow + rgb[:, 1]) / 2
    ryb[:, 2] = (rgb[:, 2] + rgb[:, 1] - yellow) / 2

    mask = ~(ryb == 0).all(axis=1)
    if mask.any():
        norm = ryb[mask].max(axis=1) / rgb[mask].max(axis=1)
        ryb[mask] = ryb[mask] / norm[:, np.newaxis]

    return ryb + black[:, np.newaxis]


def ryb_to_rgb(ryb):
    r"""
    Convert colors from RYB (red-yellow-blue) colorspace to RGB colorspace.
    
    Parameters
    ----------
    ryb : array-like
        RYB color array with shape ``(N, 3)`` or ``(3,)``.
        
    Returns
    -------
    numpy.ndarray
        RGB color array with same shape as input.
    """
    ryb = np.array(ryb)
    if len(ryb.shape) == 1:
        ryb = ryb[np.newaxis, :]

    black = ryb.min(axis=1)
    white = (1 - ryb).min(axis=1)
    ryb = ryb - black[:, np.newaxis]

    green = ryb[:, 1:].min(axis=1)
    rgb = np.zeros_like(ryb)
    rgb[:, 0] = ryb[:, 0] + ryb[:, 1] - green
    rgb[:, 1] = green + ryb[:, 1]
    rgb[:, 2] = (ryb[:, 2] - green) * 2

    mask = ~(ryb == 0).all(axis=1)
    if mask.any():
        norm = rgb[mask].max(axis=1) / ryb[mask].max(axis=1)
        rgb[mask] = rgb[mask] / norm[:, np.newaxis]

    return rgb + white[:, np.newaxis]




def plot_spatial_general(
        adata,
    value_df,
    coords,
    labels,
    text=None,
    circle_diameter=4.0,
    alpha_scaling=1.0,
    max_col=(np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
    max_color_quantile=0.98,
    show_img=True,
    img=None,
    img_alpha=1.0,
    adjust_text=False,
    plt_axis="off",
    axis_y_flipped=True,
    x_y_labels=("", ""),
    crop_x=None,
    crop_y=None,
    text_box_alpha=0.9,
    reorder_cmap=range(7),
    style="fast",
    colorbar_position="bottom",
    colorbar_label_kw={},
    colorbar_shape={},
    colorbar_tick_size=12,
    colorbar_grid=None,
    image_cmap="Greys_r",
    white_spacing=20,
    palette=None,
    legend_title_fontsize=12,
    return_ax=False,
    
):
    r"""
    Create spatial plot with color gradient and interpolation for cell type abundances.
    
    Supports up to 7 cell types with default colors: yellow, orange, blue, green, purple, grey, white.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object associated with plotted spatial values.
    value_df : pd.DataFrame
        Spot-by-feature matrix (max 7 columns) to render as blended colors.
    coords : array-like
        XY coordinates for spots.
    labels : list
        Feature labels used in legends/colorbars.
    text : pd.DataFrame or None
        Optional annotation table with coordinates and text labels.
    circle_diameter : float
        Diameter of rendered circles.
    alpha_scaling : float
        Scaling factor for transparency.
    max_col : tuple
        Per-channel cap values for color scaling.
    max_color_quantile : float
        Quantile for clipping extreme values before color mapping.
    show_img : bool
        Whether to show tissue background image.
    img : array-like or None
        Tissue image array.
    img_alpha : float
        Background image transparency.
    adjust_text : bool
        Whether to optimize text label positions.
    plt_axis : str
        Axis visibility mode.
    axis_y_flipped : bool
        Whether to invert y-axis for image-aligned coordinates.
    x_y_labels : tuple
        Axis labels as ``(x_label, y_label)``.
    crop_x : tuple or None
        Optional x-axis crop limits.
    crop_y : tuple or None
        Optional y-axis crop limits.
    text_box_alpha : float
        Alpha of text-box background.
    reorder_cmap : iterable
        Order indices used to reorder default colormaps.
    style : str
        Matplotlib style context.
    colorbar_position : str or None
        Colorbar location: ``'bottom'``, ``'right'`` or ``None``.
    colorbar_label_kw : dict
        Extra text kwargs for colorbar titles.
    colorbar_shape : dict
        Layout parameters for colorbar grid.
    colorbar_tick_size : int
        Tick label size of colorbar.
    colorbar_grid : tuple or None
        ``(rows, cols)`` of colorbar panel grid.
    image_cmap : str
        Colormap used for grayscale background image.
    white_spacing : int
        White-gap proportion in generated alpha colormaps.
    palette : list or dict or None
        Custom feature colors.
    legend_title_fontsize : int
        Font size of legend titles.
    return_ax : bool
        Whether to return axes along with figure.
        
    Returns
    -------
    matplotlib.figure.Figure or Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure, or figure-plus-axes if ``return_ax=True``.
    """

    if value_df.shape[1] > 7:
        raise ValueError("Maximum of 7 cell types / factors can be plotted at the moment")

    def create_colormap(R, G, B):
        spacing = int(white_spacing * 2.55)

        N = 255
        M = 3

        alphas = np.concatenate([[0] * spacing * M, np.linspace(0, 1.0, (N - spacing) * M)])

        vals = np.ones((N * M, 4))
        #         vals[:, 0] = np.linspace(1, R / 255, N * M)
        #         vals[:, 1] = np.linspace(1, G / 255, N * M)
        #         vals[:, 2] = np.linspace(1, B / 255, N * M)
        for i, color in enumerate([R, G, B]):
            vals[:, i] = color / 255
        vals[:, 3] = alphas

        return ListedColormap(vals)
    if palette==None:
        # Create linearly scaled colormaps
        YellowCM = create_colormap(240, 228, 66)  # #F0E442 ['#F0E442', '#D55E00', '#56B4E9',
        # '#009E73', '#5A14A5', '#C8C8C8', '#323232']
        RedCM = create_colormap(213, 94, 0)  # #D55E00
        BlueCM = create_colormap(86, 180, 233)  # #56B4E9
        GreenCM = create_colormap(0, 158, 115)  # #009E73
        GreyCM = create_colormap(200, 200, 200)  # #C8C8C8
        WhiteCM = create_colormap(50, 50, 50)  # #323232
        PurpleCM = create_colormap(90, 20, 165)  # #5A14A5

        cmaps = [YellowCM, RedCM, BlueCM, GreenCM, PurpleCM, GreyCM, WhiteCM]

        cmaps = [cmaps[i] for i in reorder_cmap]
    else:
        if isinstance(palette, list):
            cmaps = [html_to_rgb(i) for i in palette]
            cmaps = [create_colormap(*i) for i in cmaps]
        elif isinstance(palette, dict):
            #adata uns to color dict
            cmaps = [html_to_rgb(palette[i]) for i in value_df.columns]
            cmaps = [create_colormap(*i) for i in cmaps]
        else:
            raise ValueError("palette should be a list or dict or None")


    with mpl.style.context(style):
        fig = plt.figure()

        if colorbar_position == "right":
            if colorbar_grid is None:
                colorbar_grid = (len(labels), 1)

            shape = {"vertical_gaps": 1.5, "horizontal_gaps": 0, "width": 0.15, "height": 0.2}
            shape = {**shape, **colorbar_shape}

            gs = GridSpec(
                nrows=colorbar_grid[0] + 2,
                ncols=colorbar_grid[1] + 1,
                width_ratios=[1, *[shape["width"]] * colorbar_grid[1]],
                height_ratios=[1, *[shape["height"]] * colorbar_grid[0], 1],
                hspace=shape["vertical_gaps"],
                wspace=shape["horizontal_gaps"],
            )
            ax = fig.add_subplot(gs[:, 0], aspect="equal", rasterized=True)

        if colorbar_position == "bottom":
            if colorbar_grid is None:
                if len(labels) <= 3:
                    colorbar_grid = (1, len(labels))
                else:
                    n_rows = round(len(labels) / 3 + 0.5 - 1e-9)
                    colorbar_grid = (n_rows, 3)

            shape = {"vertical_gaps": 0.3, "horizontal_gaps": 0.6, "width": 0.2, "height": 0.035}
            shape = {**shape, **colorbar_shape}

            gs = GridSpec(
                nrows=colorbar_grid[0] + 1,
                ncols=colorbar_grid[1] + 2,
                width_ratios=[0.3, *[shape["width"]] * colorbar_grid[1], 0.3],
                height_ratios=[1, *[shape["height"]] * colorbar_grid[0]],
                hspace=shape["vertical_gaps"],
                wspace=shape["horizontal_gaps"],
            )

            ax = fig.add_subplot(gs[0, :], aspect="equal", rasterized=True)

        if colorbar_position is None:
            ax = fig.add_subplot(aspect="equal", rasterized=True)

        if colorbar_position is not None:
            cbar_axes = []
            for row in range(1, colorbar_grid[0] + 1):
                for column in range(1, colorbar_grid[1] + 1):
                    cbar_axes.append(fig.add_subplot(gs[row, column]))

            n_excess = colorbar_grid[0] * colorbar_grid[1] - len(labels)
            if n_excess > 0:
                for i in range(1, n_excess + 1):
                    cbar_axes[-i].set_visible(False)

        ax.set_xlabel(x_y_labels[0])
        ax.set_ylabel(x_y_labels[1])

        if img is not None and show_img:
            ax.imshow(img, aspect="equal", alpha=img_alpha, origin="lower", cmap=image_cmap)

        # crop images in needed
        if crop_x is not None:
            ax.set_xlim(crop_x[0], crop_x[1])
        if crop_y is not None:
            ax.set_ylim(crop_y[0], crop_y[1])

        if axis_y_flipped:
            ax.invert_yaxis()

        if plt_axis == "off":
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        counts = value_df.values.copy()

        # plot spots as circles
        c_ord = list(np.arange(0, counts.shape[1]))

        colors = np.zeros((*counts.shape, 4))
        weights = np.zeros(counts.shape)

        for c in c_ord:
            min_color_intensity = counts[:, c].min()
            max_color_intensity = np.min([np.quantile(counts[:, c], max_color_quantile), max_col[c]])

            rgb_function = get_rgb_function(cmap=cmaps[c], min_value=min_color_intensity, max_value=max_color_intensity)

            color = rgb_function(counts[:, c])
            color[:, 3] = color[:, 3] * alpha_scaling

            norm = mpl.colors.Normalize(vmin=min_color_intensity, vmax=max_color_intensity)

            if colorbar_position is not None:
                cbar_ticks = [
                    min_color_intensity,
                    np.mean([min_color_intensity, max_color_intensity]),
                    max_color_intensity,
                ]
                cbar_ticks = np.array(cbar_ticks)

                if max_color_intensity > 13:
                    cbar_ticks = cbar_ticks.astype(np.int32)
                else:
                    cbar_ticks = cbar_ticks.round(2)

                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmaps[c]),
                    cax=cbar_axes[c],
                    orientation="horizontal",
                    extend="both",
                    ticks=cbar_ticks,
                )

                cbar.ax.tick_params(labelsize=colorbar_tick_size)
                max_color = rgb_function(max_color_intensity / 1.5)
                cbar.ax.set_title(labels[c], **{**{"size": legend_title_fontsize, "color": max_color, "alpha": 1}, **colorbar_label_kw})

            colors[:, c] = color
            weights[:, c] = np.clip(counts[:, c] / (max_color_intensity + 1e-10), 0, 1)
            weights[:, c][counts[:, c] < min_color_intensity] = 0

        colors_ryb = np.zeros((*weights.shape, 3))

        for i in range(colors.shape[0]):
            colors_ryb[i] = rgb_to_ryb(colors[i, :, :3])

        def kernel(w):
            return w**2

        kernel_weights = kernel(weights[:, :, np.newaxis])
        weighted_colors_ryb = (colors_ryb * kernel_weights).sum(axis=1) / kernel_weights.sum(axis=1)

        weighted_colors = np.zeros((weights.shape[0], 4))

        weighted_colors[:, :3] = ryb_to_rgb(weighted_colors_ryb)

        weighted_colors[:, 3] = colors[:, :, 3].max(axis=1)

        ax.scatter(x=coords[:, 0], y=coords[:, 1], c=weighted_colors, s=circle_diameter**2)

        # add text
        if text is not None:
            bbox_props = dict(boxstyle="round", ec="0.5", alpha=text_box_alpha, fc="w")
            texts = []
            for x, y, s in zip(
                np.array(text.iloc[:, 0].values).flatten(),
                np.array(text.iloc[:, 1].values).flatten(),
                text.iloc[:, 2].tolist(),
            ):
                texts.append(ax.text(x, y, s, ha="center", va="bottom", bbox=bbox_props))

            if adjust_text:
                from adjustText import adjust_text

                adjust_text(texts, arrowprops=dict(arrowstyle="->", color="w", lw=0.5))
    if return_ax==True:
        return fig,ax
    else:
        return fig


@register_function(
    aliases=['空间表达绘图', 'plot_spatial', 'spatial abundance plot', 'deconvolution spatial plot'],
    category="pl",
    description="Render spatial abundance or expression on tissue coordinates with optional histology background for spot-level interpretation.",
    prerequisites={},
    requires={'obsm': ['spatial'], 'uns': ['spatial']},
    produces={},
    auto_fix='none',
    examples=['ov.pl.plot_spatial(adata=decov_obj.adata_cell2location, color=clust_labels, labels=clust_labels, show_img=True, style="fast")'],
    related=['pl.add_pie2spatial', 'space.crop_space_visium', 'space.map_spatial_auto']
)
def plot_spatial(adata, color, img_key="hires", show_img=True, **kwargs):
    r"""
    Create spatial plot from Visium data with color gradient and interpolation.
    
    Supports up to 7 cell types with default colors: yellow, orange, blue, green, purple, grey, white.
    
    Parameters
    ----------
    adata : AnnData
        Spatial AnnData containing coordinates and optional image metadata.
    color : list
        Feature names from ``adata.obs`` or genes from ``adata.var_names``.
    img_key : str
        Spatial image resolution key (for example ``'hires'`` or ``'lowres'``).
    show_img : bool
        Whether to draw tissue image as background.
    **kwargs
        Additional arguments forwarded to ``plot_spatial_general``.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure of spatial abundance/expression plot.
    """

    if show_img is True:
        kwargs["show_img"] = True
        kwargs["img"] = list(adata.uns["spatial"].values())[0]["images"][img_key]

    # location coordinates
    if "spatial" in adata.uns.keys():
        kwargs["coords"] = (
            adata.obsm["spatial"] * list(adata.uns["spatial"].values())[0]["scalefactors"][f"tissue_{img_key}_scalef"]
        )
    else:
        kwargs["coords"] = adata.obsm["spatial"]

    plot_data=pd.DataFrame()
    for ct in color:
        if ct in adata.obs.columns:
            plot_data[ct] = adata.obs[ct]
        elif ct in adata.var_names:
            plot_data[ct] = get_values(adata, ct)
    plot_data=plot_data.fillna(0)

    fig = plot_spatial_general(adata,value_df=plot_data, **kwargs)  # cell abundance values
    fig.axes[0].set_xlim(kwargs["coords"][:,0].min(),kwargs["coords"][:,0].max())
    fig.axes[0].set_ylim(kwargs["coords"][:,1].max(),kwargs["coords"][:,1].min())

    return fig

def create_colormap(R, G, B):
    r"""
    Create a matplotlib colormap from RGB values with alpha gradient.
    
    Parameters
    ----------
    R : int
        Red channel in ``[0, 255]``.
    G : int
        Green channel in ``[0, 255]``.
    B : int
        Blue channel in ``[0, 255]``.
        
    Returns
    -------
    matplotlib.colors.ListedColormap
        Colormap with alpha gradient from transparent to opaque.
    """
    spacing = int(20 * 2.55)

    N = 255
    M = 3

    alphas = np.concatenate([[0] * spacing * M, np.linspace(0, 1.0, (N - spacing) * M)])

    vals = np.ones((N * M, 4))
    #         vals[:, 0] = np.linspace(1, R / 255, N * M)
    #         vals[:, 1] = np.linspace(1, G / 255, N * M)
    #         vals[:, 2] = np.linspace(1, B / 255, N * M)
    for i, color in enumerate([R, G, B]):
        vals[:, i] = color / 255
    vals[:, 3] = alphas

    return ListedColormap(vals)

def spatial_value(adata,color,library_id,
            dot_size=4,white_spacing=20,
            colorbar_show=True,
            colorbar_tick_size=12,cmap=create_colormap(255, 0, 0),
            legend_title_fontsize=12,
            legend_title_color=None,
            colorbar_label_kw={},res='hires',
            img_show=True,ax=None,alpha_img=1):
    r"""
    Plot spatial expression values for a single gene or feature on tissue image.
    
    Parameters
    ----------
    adata : AnnData
        Spatial AnnData object.
    color : str
        Gene name or ``obs`` column to visualize.
    library_id : str
        Key in ``adata.uns['spatial']`` selecting target library.
    dot_size : float
        Spot size used in scatter overlay.
    white_spacing : int
        Reserved white-spacing parameter for colormap style.
    colorbar_show : bool
        Whether to display horizontal colorbar.
    colorbar_tick_size : int
        Tick label size for colorbar.
    cmap : matplotlib.colors.Colormap
        Colormap used for signal values.
    legend_title_fontsize : int
        Font size of colorbar title.
    legend_title_color : tuple or None
        Color of colorbar title text.
    colorbar_label_kw : dict
        Extra kwargs for colorbar title text.
    res : str
        Tissue image resolution key.
    img_show : bool
        Whether to render tissue image background.
    ax : matplotlib.axes.Axes or None
        Existing axes to plot into.
    alpha_img : float
        Background image transparency.
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes containing spatial value rendering.
    """

    #ax_input=False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))

    img=adata.uns["spatial"][library_id]["images"][res]
    if img_show:
        ax.imshow(img, aspect="equal", alpha=alpha_img, origin="lower", cmap=cmap)
        ax.invert_yaxis()
    ax.axis(False)
    coords=adata.obsm['spatial']*adata.uns['spatial'][library_id]['scalefactors'][f'tissue_{res}_scalef']
    if color in adata.obs.columns:
        weighted_colors=adata.obs[color].values.reshape(-1)
    elif color in adata.var_names:
        weighted_colors=adata[:,color].to_df().values.reshape(-1)
    elif adata.raw is not None and color in adata.raw.var_names:
        weighted_colors=adata.raw[:,color].to_adata().to_df().values.reshape(-1)
    ax.scatter(x=coords[:, 0], y=coords[:, 1], c=weighted_colors, s=dot_size**2,
              cmap=cmap)
    
    ax.set_xlim(coords[:,0].min(),coords[:,0].max())
    ax.set_ylim(coords[:,1].max(),coords[:,1].min())
    
    
    colorbar_grid=None
    labels=[color]
    colorbar_shape={}
    if colorbar_grid is None:
        if len(labels) <= 3:
            colorbar_grid = (1, len(labels))
        else:
            n_rows = round(len(labels) / 3 + 0.5 - 1e-9)
            colorbar_grid = (n_rows, 3)
    
    shape = {"vertical_gaps": 1, "horizontal_gaps": 0.6, "width": 0.5, "height": 0.05}
    shape = {**shape, **colorbar_shape}
    
    gs = GridSpec(
        nrows=colorbar_grid[0] + 1,
        ncols=colorbar_grid[1] + 2,
        width_ratios=[0.3, *[shape["width"]] * colorbar_grid[1], 0.3],
        height_ratios=[1, *[shape["height"]] * colorbar_grid[0]],
        hspace=shape["vertical_gaps"],
        wspace=shape["horizontal_gaps"],
    )
    
    #ax1 = fig.add_subplot(gs[0, :], aspect="equal", rasterized=True)
    if colorbar_show is True:
        fig=plt.gcf()
        cbar_axes = []
        for row in range(1, colorbar_grid[0] + 1):
            for column in range(1, colorbar_grid[1] + 1):
                cbar_axes.append(fig.add_subplot(gs[row, column]))
        
        n_excess = colorbar_grid[0] * colorbar_grid[1] - len(labels)
        if n_excess > 0:
            for i in range(1, n_excess + 1):
                cbar_axes[-i].set_visible(False)
        
        norm=mpl.colors.Normalize(vmin=min(weighted_colors), vmax=max(weighted_colors))
        rgb_function = get_rgb_function(cmap=cmap,
                                        min_value=min(weighted_colors), max_value=max(weighted_colors))

        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_axes[0],
            orientation="horizontal",
            extend="both",
            #ticks=cbar_ticks,
        )
        cbar.ax.tick_params(labelsize=colorbar_tick_size)
        if legend_title_color is None:
            max_color = rgb_function(max(weighted_colors) / 1.5)
        else:
            max_color=legend_title_color
        cbar.ax.set_title(labels[0], **{**{"size": legend_title_fontsize, 
                                        "color": max_color, "alpha": 1},
                                            **colorbar_label_kw})

    return ax


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Wedge

def _make_color_dict(cell_type_columns, colors):
    """
    Build color mapping dictionary for cell-type columns.

    Parameters
    ----------
    cell_type_columns : list
        Cell-type column names.
    colors : list, dict or None
        User-defined colors.

    Returns
    -------
    dict
        Mapping from cell type to RGBA/hex color.
    """
    if colors is None:
        pal = plt.cm.Set3(np.linspace(0, 1, len(cell_type_columns)))
        return {ct: pal[i] for i, ct in enumerate(cell_type_columns)}
    if isinstance(colors, list):
        assert len(colors) >= len(cell_type_columns), "colors 列表长度不足"
        return {ct: colors[i] for i, ct in enumerate(cell_type_columns)}
    # dict
    return colors

def _draw_pie(ax, x, y, fracs, cols, radius, start_angle=90, alpha=0.8, zorder=5):
    """
    Draw a pie chart at a given data coordinate.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x : float
        X coordinate.
    y : float
        Y coordinate.
    fracs : sequence
        Slice fractions.
    cols : sequence
        Slice colors.
    radius : float
        Pie radius in data units.
    start_angle : float
        Start angle in degrees.
    alpha : float
        Pie transparency.
    zorder : int
        Artist draw order.
    """
    if len(fracs) == 0:
        return
    theta = start_angle
    for f, c in zip(fracs, cols):
        if f <= 0:
            continue
        dtheta = 360.0 * float(f)
        w = Wedge((x, y), r=radius, theta1=theta, theta2=theta + dtheta,
                  facecolor=c, edgecolor="none", linewidth=0, alpha=alpha, zorder=zorder)
        ax.add_patch(w)
        theta += dtheta

def add_pie_charts_to_spatial(
    adata,
    cell_type_columns,
    ax,
    spatial_coords=None,
    pie_radius=15,
    pie_radius_px=None,      # 新增：按像素指定半径
    min_proportion=0.01,
    colors=None,
    alpha=0.8,
    add_gray_remainder=True,
    renormalize_if_sum_gt1=True,
    zorder=50,               # 提高层级，确保盖住散点
):
    """
    Overlay per-spot composition pie charts on an existing spatial axis.

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData object.
    cell_type_columns : list
        Columns in ``adata.obs`` holding component fractions.
    ax : matplotlib.axes.Axes
        Existing axes to draw on.
    spatial_coords : array-like or None
        Optional external coordinates.
    pie_radius : float
        Pie radius in data units.
    pie_radius_px : float or None
        Pie radius in pixels (converted to data units).
    min_proportion : float
        Minimum slice proportion retained for drawing.
    colors : list, dict or None
        Color definitions for slices.
    alpha : float
        Pie transparency.
    add_gray_remainder : bool
        Whether to add a gray remainder slice if fractions sum below 1.
    renormalize_if_sum_gt1 : bool
        Whether to renormalize rows when fractions sum above 1.
    zorder : int
        Artist draw order.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with overlaid pie charts.
    """
    if 'spatial' not in adata.obsm:
        raise ValueError("No spatial coordinates found in adata.obsm['spatial']")

    # 坐标与比例矩阵
    if spatial_coords is None:
        spatial_coords = np.asarray(adata.obsm['spatial'])
        if spatial_coords.shape[0] != adata.n_obs:
            raise ValueError("spatial 坐标与 obs 数量不一致")

    # 提前把比例列转数值，NaN->0
    miss = [c for c in cell_type_columns if c not in adata.obs.columns]
    if len(miss) > 0:
        raise KeyError(f"以下列在 adata.obs 中不存在：{miss}")

    plot_data=pd.DataFrame()
    for ct in cell_type_columns:
        if ct in adata.obs.columns:
            plot_data[ct] = adata.obs[ct]
        elif ct in adata.var_names:
            plot_data[ct] = get_values(adata, ct)
    plot_data=plot_data.fillna(0)
    props_df = (
        plot_data
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    props = props_df.to_numpy(copy=False)  # (n_obs, n_types)

    # 颜色映射
    color_dict = _make_color_dict(cell_type_columns, colors)

    #xlim = ax.get_xlim(); ylim = ax.get_ylim()
    #ax.set_autoscale_on(False)

    # 逐点绘制（Wedge 高效、不会触发轴自动重算）
    from tqdm import tqdm
    for i in tqdm(range(adata.n_obs)):
        x, y = spatial_coords[i, 0], spatial_coords[i, 1]
        p = props[i]

        # 过滤小于阈值的类别
        keep_mask = p >= float(min_proportion)
        kept_vals = p[keep_mask]
        kept_cols = [color_dict[cell_type_columns[j]] for j in np.nonzero(keep_mask)[0]]

        total = float(kept_vals.sum())

        # 没有任何类别超过阈值则跳过
        if total <= 0:
            continue

        fracs = kept_vals.copy()

        # 总和 > 1 时可选重标（更稳健）
        if total > 1.0 and renormalize_if_sum_gt1:
            fracs = fracs / total
            total = 1.0

        # 总和 < 1 时可选补灰色剩余
        cols_list = kept_cols
        if add_gray_remainder and total < 0.999:
            fracs = np.append(fracs, 1.0 - total)
            cols_list = cols_list + ['lightgray']

        radius_data = _data_radius_from_pixels(ax, pie_radius_px) if pie_radius_px is not None else pie_radius

        _draw_pie(ax, x, y, fracs, cols_list, radius=radius_data, alpha=alpha, zorder=zorder)


        #_draw_pie(ax, x, y, fracs, cols_list, radius=pie_radius, alpha=alpha, zorder=zorder)

    return ax

from matplotlib.patches import Wedge

def _data_radius_from_pixels(ax, r_px: float) -> float:
    """
    Convert pixel radius to data-coordinate radius.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    r_px : float
        Radius in pixels.

    Returns
    -------
    float
        Radius in data coordinates.
    """
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transData.inverted()
    p0 = inv.transform((0, 0))
    p1 = inv.transform((1, 1))
    dx = abs(p1[0] - p0[0]); dy = abs(p1[1] - p0[1])
    return r_px * min(dx, dy)

@register_function(
    aliases=['空间饼图叠加', 'add_pie2spatial', 'pie chart on spatial'],
    category="pl",
    description="Overlay per-spot cell-type proportion pie charts onto spatial coordinates to inspect local microenvironment composition.",
    prerequisites={},
    requires={'obsm': ['spatial'], 'obs': ['cell-type proportion columns']},
    produces={},
    auto_fix='none',
    examples=['ov.pl.add_pie2spatial(adata, cell_type_columns=clust_labels, pie_radius=0.45)'],
    related=['pl.plot_spatial', 'utils.plot_cellproportion']
)
def add_pie2spatial(
    adata,
    cell_type_columns,
    ax,
    pie_radius=15,
    img_key='hires',
    spatial_coords=None,
    pie_radius_px=None,          # 可选：按像素给半径
    min_proportion=0.01,
    colors=None,
    alpha=0.9,
    remainder='gap',             # ★ 新增：'gap' | 'gray' | 'outline'
    remainder_color='lightgray',
    remainder_alpha=0.5,
    renormalize_if_sum_gt1=True,
    zorder=80,
    legend_loc=(1.05, 1.0),
    ncols=3,
):
    """
    Overlay per-spot pie charts of cell-type composition on a spatial map.
    
    Parameters
    ----------
    adata : AnnData
        Spatial AnnData object. Requires ``adata.obsm['spatial']``.
    cell_type_columns : list of str
        Columns in ``adata.obs`` (or genes in ``adata.var_names``) used as pie components.
    ax : matplotlib.axes.Axes
        Target axes.
    pie_radius : float, optional
        Pie radius in data coordinates when ``pie_radius_px`` is not provided.
    img_key : str, optional
        Spatial image key used to rescale coordinates (for example ``'hires'``).
    spatial_coords : array-like, optional
        External coordinates overriding ``adata.obsm['spatial']``.
    pie_radius_px : float or None, optional
        Pie radius in pixels; converted to data units automatically.
    min_proportion : float, optional
        Minimum slice proportion to draw.
    colors : dict or sequence, optional
        Color mapping for cell types.
    alpha : float, optional
        Pie transparency.
    remainder : {'gap', 'gray', 'outline'}, optional
        How to render unassigned mass when proportions do not sum to 1.
    remainder_color : str, optional
        Color for remainder slice when ``remainder='gray'``.
    remainder_alpha : float, optional
        Alpha for remainder slice.
    renormalize_if_sum_gt1 : bool, optional
        If ``True``, renormalize rows whose sum exceeds 1.
    zorder : int, optional
        Drawing order for pie artists.
    legend_loc : tuple, optional
        Legend anchor location.
    ncols : int, optional
        Number of legend columns.
    
    Returns
    -------
    matplotlib.axes.Axes
        Axes with pie overlays.
    
    Examples
    --------
    >>> ax = ov.pl.plot_spatial(adata)
    >>> ov.pl.add_pie2spatial(adata, cell_type_columns=['T', 'B', 'Myeloid'], ax=ax)
    """
    if 'spatial' not in adata.obsm:
        raise ValueError("No spatial coordinates found in adata.obsm['spatial']")

    # 坐标与比例矩阵
    if spatial_coords is None:
        spatial_coords = np.asarray(adata.obsm['spatial'])
        if spatial_coords.shape[0] != adata.n_obs:
            raise ValueError("spatial 坐标与 obs 数量不一致")

        if img_key is not None and 'spatial' in adata.uns:
            spatial_key=list(adata.uns['spatial'].keys())[0]
            spatial_coords=spatial_coords*adata.uns['spatial'][spatial_key]['scalefactors'][f'tissue_{img_key}_scalef']


    plot_data=pd.DataFrame()
    for ct in cell_type_columns:
        if ct in adata.obs.columns:
            plot_data[ct] = adata.obs[ct]
        elif ct in adata.var_names:
            plot_data[ct] = get_values(adata, ct)
    plot_data=plot_data.fillna(0)
    props = (
        plot_data
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float, copy=False)
    )

    # 准备颜色
    if colors is None:
        pal = plt.cm.Set3(np.linspace(0, 1, len(cell_type_columns)))
        color_dict = {ct: pal[i] for i, ct in enumerate(cell_type_columns)}
    elif isinstance(colors, list):
        color_dict = {ct: colors[i] for i, ct in enumerate(cell_type_columns)}
    else:
        color_dict = colors

    # 冻结范围，计算一次数据半径
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_autoscale_on(False)
    radius_data = _data_radius_from_pixels(ax, pie_radius_px) if pie_radius_px is not None else pie_radius

    from tqdm import trange
    for i in trange(adata.n_obs):
        x, y = spatial_coords[i, 0], spatial_coords[i, 1]
        p = props[i]

        # 过滤阈值以下的类别
        keep = p >= float(min_proportion)
        vals = p[keep]
        cols = [color_dict[cell_type_columns[j]] for j in np.nonzero(keep)[0]]

        total = float(vals.sum())
        if total <= 0:
            continue

        fracs = vals.copy()
        if total > 1.0 and renormalize_if_sum_gt1:
            fracs = fracs / total
            total = 1.0

        # 画已选类别
        theta = 90.0
        for f, c in zip(fracs, cols):
            if f <= 0:
                continue
            dtheta = 360.0 * float(f)
            ax.add_patch(Wedge((x, y), radius_data, theta, theta + dtheta,
                               facecolor=c, edgecolor='none', linewidth=0,
                               alpha=alpha, zorder=zorder, clip_on=False))
            theta += dtheta

        # 处理“剩余比例” → 缺口/灰块/描边
        remain = max(0.0, 1.0 - total)
        if remain > 1e-6:
            if remainder == 'gray':
                dtheta = 360.0 * remain
                ax.add_patch(Wedge((x, y), radius_data, theta, theta + dtheta,
                                   facecolor=remainder_color, edgecolor='none',
                                   linewidth=0, alpha=remainder_alpha,
                                   zorder=zorder, clip_on=False))
                # theta += dtheta  # 不需要继续了
            elif remainder == 'outline':
                dtheta = 360.0 * remain
                ax.add_patch(Wedge((x, y), radius_data, theta, theta + dtheta,
                                   facecolor='none', edgecolor=remainder_color,
                                   linewidth=0.8, alpha=1.0,
                                   zorder=zorder, clip_on=False))
                # 'gap' 模式就是不画，让背景露出来

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    # 图例（与饼图颜色一致）
    #color_dict = _make_color_dict(cell_type_columns, pie_kwargs.get("colors", None) if pie_kwargs else None)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colors[ct], markersize=10, label=ct)
        for ct in cell_type_columns
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=legend_loc, loc='lower center',ncols=ncols,)

    return ax

from matplotlib.patches import Wedge

def _data_radius_from_pixels(ax, r_px: float) -> float:
    """
    Convert pixel radius to data-coordinate radius (aspect aware).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    r_px : float
        Radius in pixels.

    Returns
    -------
    float
        Radius in data coordinates.
    """
    # 轴的可视范围（数据单位）
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # 轴在屏幕上的宽高（像素）
    bbox = ax.get_window_extent()
    w_px = bbox.width
    h_px = bbox.height
    # 每像素对应的数据单位
    rx = abs(x1 - x0) / max(w_px, 1e-9)
    ry = abs(y1 - y0) / max(h_px, 1e-9)
    return r_px * min(rx, ry)

def _draw_pie(ax, x, y, fracs, cols, radius, start_angle=90, alpha=0.8, zorder=50):
    """
    Draw a pie chart at given coordinate using Wedge patches.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x : float
        X coordinate.
    y : float
        Y coordinate.
    fracs : sequence
        Slice fractions.
    cols : sequence
        Slice colors.
    radius : float
        Radius in data units.
    start_angle : float
        Starting angle in degrees.
    alpha : float
        Pie transparency.
    zorder : int
        Artist draw order.
    """
    theta = start_angle
    for f, c in zip(fracs, cols):
        if f <= 0:
            continue
        dtheta = 360.0 * float(f)
        w = Wedge((x, y), r=radius, theta1=theta, theta2=theta + dtheta,
                  facecolor=c, edgecolor="none", linewidth=0,
                  alpha=alpha, zorder=zorder, clip_on=False)  # 关键：clip_on=False
        ax.add_patch(w)
        theta += dtheta
