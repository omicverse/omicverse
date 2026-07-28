"""Nanostring spatial plotting utilities for OmicVerse.

Multi-FOV composite visualisation for Nanostring SMI (CosMx) data.
Cells are rendered in the global coordinate space (``obsm['spatial_fov']``)
so that all selected FOVs are stitched into a single figure with their
background images placed at the correct positions.
"""

from __future__ import annotations

from typing import List, Optional, Union
import warnings

import matplotlib.pyplot as plt

from ._plotdata import get_values
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib import colors as mcolors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_fov_column(adata) -> str:
    """Return the FOV column name in ``adata.obs``."""
    for col in ("fov", "FOV", "fov_id", "fovID"):
        if col in adata.obs.columns:
            return col
    raise ValueError(
        "No FOV column found in adata.obs. "
        "Expected one of: 'fov', 'FOV', 'fov_id', 'fovID'. "
        "Make sure the data was loaded with `ov.io.read_nanostring()`."
    )


def _compute_fov_offset(adata, fov_id: str, fov_col: str) -> tuple:
    """Compute ``(offset_x, offset_y)`` to translate local pixel coords → global.

    The offset is derived from the median difference between
    ``obsm['spatial_fov']`` and ``obsm['spatial']`` for cells in this FOV.
    """
    mask = adata.obs[fov_col].astype(str) == str(fov_id)
    if (
        not mask.any()
        or "spatial" not in adata.obsm
        or "spatial_fov" not in adata.obsm
    ):
        return (0.0, 0.0)
    local_xy = adata.obsm["spatial"][mask.to_numpy(), :2]
    global_xy = adata.obsm["spatial_fov"][mask.to_numpy(), :2]
    if len(local_xy) == 0:
        return (0.0, 0.0)
    diff = global_xy - local_xy
    return (float(np.median(diff[:, 0])), float(np.median(diff[:, 1])))


def _require_geopandas():
    try:
        import geopandas as gpd
        from shapely import wkt
    except ImportError as exc:
        raise ImportError(
            "`nanostringseg` requires `geopandas` and `shapely`. "
            "Install with: pip install geopandas shapely"
        ) from exc
    return gpd, wkt


def _normalize_fov_ids(fovs: Optional[Union[str, int, List[Union[str, int]]]]) -> Optional[List[str]]:
    """Normalize FOV selectors to a list of strings."""
    if fovs is None:
        return None
    if isinstance(fovs, (str, int)):
        return [str(fovs)]
    return [str(fov) for fov in fovs]


def _validate_basis(adata, basis: str) -> str:
    """Validate and normalize the plotting basis."""
    if basis not in ("spatial", "spatial_fov"):
        raise ValueError("`basis` must be either 'spatial' or 'spatial_fov'.")
    if basis not in adata.obsm:
        raise ValueError(f"`{basis}` not found in `adata.obsm`.")
    return basis


def _validate_img_extent_mode(img_extent_mode: str) -> str:
    """Validate how image extents are computed."""
    if img_extent_mode not in ("cells", "panel"):
        raise ValueError("`img_extent_mode` must be either 'cells' or 'panel'.")
    return img_extent_mode


def _metadata_numeric(metadata, *candidates) -> Optional[float]:
    """Return the first numeric metadata value matching candidate keys."""
    if not isinstance(metadata, dict):
        return None
    lowered = {str(key).lower(): value for key, value in metadata.items()}
    for candidate in candidates:
        if candidate.lower() not in lowered:
            continue
        value = lowered[candidate.lower()]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_fov_scale_factor(fov_info, img_key, scale_factor) -> float:
    """Resolve the effective scale factor for one FOV."""
    if scale_factor is not None:
        return float(scale_factor)

    sf_dict = fov_info.get("scalefactors", {})
    if img_key is not None:
        return float(
            sf_dict.get(
                f"tissue_{img_key}_scalef",
                sf_dict.get("tissue_hires_scalef", 1.0),
            )
        )
    return 1.0


def _compute_basis_bbox(adata, fov_id: str, fov_col: str, basis: str):
    """Return `(xmin, xmax, ymin, ymax)` for one FOV in the chosen basis."""
    mask = adata.obs[fov_col].astype(str) == str(fov_id)
    if not mask.any():
        return None
    xy = np.asarray(adata.obsm[basis][mask.to_numpy(), :2], dtype=float)
    if xy.shape[0] == 0:
        return None
    return (
        float(np.nanmin(xy[:, 0])),
        float(np.nanmax(xy[:, 0])),
        float(np.nanmin(xy[:, 1])),
        float(np.nanmax(xy[:, 1])),
    )


def _compute_fov_geometry_transform(adata, fov_id: str, fov_col: str, basis: str):
    """Return `(sx, sy, tx, ty)` mapping local geometry coords into the chosen basis."""
    if basis == "spatial":
        return 1.0, 1.0, 0.0, 0.0

    mask = adata.obs[fov_col].astype(str) == str(fov_id)
    if not mask.any() or "spatial" not in adata.obsm:
        return 1.0, 1.0, 0.0, 0.0

    local_xy = np.asarray(adata.obsm["spatial"][mask.to_numpy(), :2], dtype=float)
    target_xy = np.asarray(adata.obsm[basis][mask.to_numpy(), :2], dtype=float)
    if local_xy.shape[0] == 0 or target_xy.shape[0] == 0:
        return 1.0, 1.0, 0.0, 0.0

    local_min = np.nanmin(local_xy, axis=0)
    local_max = np.nanmax(local_xy, axis=0)
    target_min = np.nanmin(target_xy, axis=0)
    target_max = np.nanmax(target_xy, axis=0)
    local_span = local_max - local_min
    target_span = target_max - target_min

    sx = float(target_span[0] / local_span[0]) if local_span[0] > 0 else 1.0
    tx = float(target_min[0] - (local_min[0] * sx))

    # Segmentation polygons are stored in image-style local coordinates.
    # When projecting them into the stitched basis, flip Y within each FOV bbox
    # so the polygon orientation matches `imshow(origin="upper")`.
    sy = float(-target_span[1] / local_span[1]) if local_span[1] > 0 else -1.0
    ty = float(target_max[1] - (local_min[1] * sy))
    return sx, sy, tx, ty


def _compute_fov_image_extent(adata, fov_id: str, fov_col: str, basis: str, img_extent_mode: str):
    """Compute `[xmin, xmax, ymin, ymax]` for one FOV image in the chosen basis."""
    _validate_img_extent_mode(img_extent_mode)
    return _compute_basis_bbox(adata, fov_id, fov_col, basis)


def _rasterize_new_collections(ax, start_idx: int) -> None:
    """Rasterize collections added to an axis after `start_idx`."""
    for coll in ax.collections[start_idx:]:
        try:
            coll.set_rasterized(True)
        except Exception:
            continue


def _resolve_vbound(values: np.ndarray, bound) -> float:
    """Resolve numeric or percentile-string bounds such as `p99.2`."""
    if bound is None:
        return float(np.nanmin(values))
    if isinstance(bound, str):
        if bound.startswith("p"):
            try:
                return float(np.nanpercentile(values, q=float(bound[1:])))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid percentile bound `{bound}`. Use syntax like `p99` or `p99.2`."
                ) from exc
        try:
            return float(bound)
        except ValueError as exc:
            raise ValueError(
                f"Invalid bound `{bound}`. Use a float or percentile string like `p99.2`."
            ) from exc
    return float(bound)


def _resolve_cmap(cmap):
    """Resolve either a colormap name or a matplotlib colormap object."""
    if isinstance(cmap, mcolors.Colormap):
        return cmap
    return plt.get_cmap(cmap)


def _scale_rgba_alpha(colors, alpha: float):
    """Scale the alpha channel of RGBA colors without discarding existing transparency."""
    rgba = np.asarray(colors, dtype=float).copy()
    if rgba.ndim == 1:
        rgba = rgba.reshape(1, -1)
    if rgba.shape[1] == 4:
        rgba[:, 3] = np.clip(rgba[:, 3] * float(alpha), 0.0, 1.0)
    return rgba


def _place_fov_images(
    ax,
    adata,
    fovs_to_plot,
    fov_col,
    basis,
    img_key,
    alpha_img,
    bw,
    img_extent_mode,
    panel_extent=None,
):
    """Draw each FOV's background image at its position in the chosen basis."""
    for fov_id in fovs_to_plot:
        fov_info = adata.uns.get("spatial", {}).get(fov_id, {})
        img = fov_info.get("images", {}).get(img_key) if img_key else None
        if img is None:
            continue

        if img_extent_mode == "panel" and panel_extent is not None:
            extent = panel_extent
        else:
            extent = _compute_fov_image_extent(adata, fov_id, fov_col, basis, img_extent_mode)
        if extent is None:
            continue
        img = np.asarray(img)
        if bw and img.ndim == 3 and img.shape[2] >= 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

        # Match `ov.pl.spatial`: use `origin="upper"` and invert the axis later
        # via `set_ylim(y_max, y_min)`. Do not pre-flip the extent here.
        ax.imshow(
            img,
            cmap="gray" if bw and img.ndim == 2 else None,
            origin="upper",
            extent=extent,
            alpha=alpha_img,
            aspect="auto",
            zorder=0,
        )


def _global_bounding_box(adata, fovs_to_plot, fov_col, basis, img_extent_mode, pad_frac=0.02):
    """Return (xlim, ylim) covering all selected FOV cells in the chosen basis."""
    mask = adata.obs[fov_col].astype(str).isin(set(fovs_to_plot))
    if not mask.any() or basis not in adata.obsm:
        return (0.0, 1.0), (1.0, 0.0)

    xy = np.asarray(adata.obsm[basis][mask.to_numpy(), :2], dtype=float)
    if xy.shape[0] == 0 or not np.isfinite(xy).all():
        return (0.0, 1.0), (1.0, 0.0)

    x_min = float(np.nanmin(xy[:, 0]))
    y_min = float(np.nanmin(xy[:, 1]))
    x_max = float(np.nanmax(xy[:, 0]))
    y_max = float(np.nanmax(xy[:, 1]))

    dx = (x_max - x_min) * pad_frac or 1.0
    dy = (y_max - y_min) * pad_frac or 1.0
    xlim = (x_min - dx, x_max + dx)
    ylim = (y_max + dy, y_min - dy)  # inverted y
    return xlim, ylim


# ---------------------------------------------------------------------------
# Public function: nanostring  (scatter-based)
# ---------------------------------------------------------------------------

def nanostring(
    adata,
    fovs: Optional[List[Union[str, int]]] = None,
    color: Optional[Union[str, List[str]]] = None,
    basis: str = "spatial_fov",
    figsize: Optional[tuple] = None,
    img_key: Optional[str] = "hires",
    img_extent_mode: str = "cells",
    bw: bool = False,
    rasterized: bool = False,
    scale_factor: Optional[float] = None,
    alpha_img: float = 1.0,
    cmap: Union[str, mcolors.Colormap] = "viridis",
    palette: Optional[Union[dict, list, np.ndarray]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    size: float = 20.0,
    na_color: str = "lightgray",
    legend: bool = True,
    legend_loc: str = "right margin",
    legend_fontsize: Optional[Union[int, float]] = 12,
    legend_fontweight: Union[str, int] = "bold",
    legend_fontoutline: Optional[float] = None,
    na_in_legend: bool = True,
    colorbar_loc: Optional[str] = "right",
    show: bool = True,
    ax: Optional[plt.Axes] = None,
    xlabel: Optional[str] = "global x (px)",
    ylabel: Optional[str] = "global y (px)",
    show_ticks: bool = False,
    **kwargs,
):
    """Plot Nanostring data across multiple FOVs with stitched background images.

    Each FOV's background image is placed at its correct position in the
    global coordinate space (``obsm['spatial_fov']``), and cells are rendered
    as scatter points on top to produce a single composite figure.

    Parameters
    ----------
    adata : AnnData
        Object produced by :func:`~omicverse.io.read_nanostring`.
    fovs : list of str or int, optional
        FOV identifiers to include.  If *None*, all FOVs in the dataset are used.
    color : str or list of str, optional
        ``obs`` column name or gene name to colour cells by.
        Multiple keys produce side-by-side panels.
    figsize : tuple, optional
        Figure size ``(width, height)``.
    img_key : str, optional
        Key in ``uns['spatial'][fov]['images']`` for the background image
        (default ``'hires'``).
    img_extent_mode : str
        Strategy used to place each FOV image. ``'cells'`` fits image extent to
        the selected basis bbox of the FOV's cells. ``'panel'`` stretches a
        single selected FOV image to the full panel extent.
    bw : bool
        If ``True``, render the background image in grayscale.
    rasterized : bool
        If ``True``, rasterize the point layer while keeping the overall figure
        in vector format when saving to SVG/PDF.
    scale_factor : float, optional
        Override the scale factor stored per FOV.
    alpha_img : float
        Background image transparency (0–1, default 1.0).
    cmap : str
        Colormap for continuous features.
    palette : dict, list, or ndarray, optional
        Colour palette for categorical features.
    vmin, vmax : float, optional
        Value range clamp for continuous colormaps.
    size : float
        Scatter point area in points² (default 20).
    na_color : str
        Colour for missing / NA values.
    legend : bool
        Whether to draw a legend or colorbar.
    legend_loc : str
        Legend location (e.g. ``'right margin'``).
    legend_fontsize : int or float, optional
        Legend label size.
    legend_fontweight : str or int
        Legend label weight.
    legend_fontoutline : float, optional
        Stroke width for legend text outlines.
    na_in_legend : bool
        Whether to include NA category in the legend.
    colorbar_loc : str, optional
        Colorbar anchor (``'right'``, ``'bottom'``, etc.).
    show : bool
        Call :func:`plt.show` after plotting.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw on (only one panel supported).
    xlabel, ylabel : str, optional
        Axis labels.
    show_ticks : bool
        Show axis tick marks and labels.
    **kwargs
        Extra keyword arguments forwarded to :func:`plt.scatter`.

    Returns
    -------
    matplotlib.axes.Axes or list of matplotlib.axes.Axes
    """
    basis = _validate_basis(adata, basis)
    img_extent_mode = _validate_img_extent_mode(img_extent_mode)
    if "spatial" not in adata.uns or not adata.uns["spatial"]:
        raise ValueError("`nanostring` requires non-empty `adata.uns['spatial']`.")

    fov_col = _get_fov_column(adata)
    all_fovs = sorted(
        adata.obs[fov_col].astype(str).unique().tolist(),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    if fovs is None:
        fovs_to_plot = all_fovs
    else:
        fovs_to_plot = _normalize_fov_ids(fovs)
        missing = [f for f in fovs_to_plot if f not in all_fovs]
        if missing:
            raise ValueError(
                f"FOV(s) {missing} not found in adata.obs['{fov_col}']. "
                f"Available: {all_fovs}"
            )

    if color is None:
        colors_to_plot = [None]
    elif isinstance(color, str):
        colors_to_plot = [color]
    else:
        colors_to_plot = list(color)

    for ck in colors_to_plot:
        if ck is not None and ck not in adata.obs.columns and ck not in adata.var_names:
            raise ValueError(
                f"`color` key '{ck}' not found in `adata.obs.columns` or `adata.var_names`."
            )

    # --- Build figure / axes ---
    if ax is None:
        if figsize is None:
            figsize = (6 * len(colors_to_plot), 6)
        if len(colors_to_plot) > 1:
            fig, axes_arr = plt.subplots(1, len(colors_to_plot), figsize=figsize,
                                         sharex=True, sharey=True)
            axes = list(axes_arr)
        else:
            fig, ax_single = plt.subplots(1, 1, figsize=figsize)
            axes = [ax_single]
    else:
        fig = ax.figure
        axes = [ax]
        if len(colors_to_plot) > 1:
            warnings.warn(
                "Multiple colors specified but only one ax provided. "
                "Only the first color will be plotted."
            )
            colors_to_plot = [colors_to_plot[0]]

    xlim, ylim = _global_bounding_box(adata, fovs_to_plot, fov_col, basis, img_extent_mode)
    fov_mask = adata.obs[fov_col].astype(str).isin(set(fovs_to_plot))
    cells_in_fovs = adata.obs_names[fov_mask]
    cell_xy = np.asarray(adata.obsm[basis][fov_mask.to_numpy(), :2], dtype=float)

    axes_list = []
    for plot_idx, color_key in enumerate(colors_to_plot):
        current_ax = axes[plot_idx]

        # Background images
        img_alpha = alpha_img if color_key is None else alpha_img
        _place_fov_images(
            current_ax,
            adata,
            fovs_to_plot,
            fov_col,
            basis,
            img_key,
            img_alpha,
            bw,
            img_extent_mode,
            panel_extent=(xlim[0], xlim[1], ylim[1], ylim[0]),
        )

        if color_key is None:
            current_ax.set_xlim(*xlim)
            current_ax.set_ylim(*ylim)
            current_ax.set_aspect("equal")
            if xlabel:
                current_ax.set_xlabel(xlabel)
            if ylabel:
                current_ax.set_ylabel(ylabel)
            if not show_ticks:
                current_ax.set_xticks([])
                current_ax.set_yticks([])
            axes_list.append(current_ax)
            continue

        # Gather colour data
        if color_key in adata.obs.columns:
            color_data = adata.obs.loc[cells_in_fovs, color_key]
        else:
            vals = get_values(adata, color_key)[fov_mask.to_numpy()]
            color_data = pd.Series(vals, index=cells_in_fovs, name=color_key)

        is_categorical = (
            isinstance(color_data.dtype, pd.CategoricalDtype)
            or color_data.dtype == object
            or color_data.dtype == bool
        )

        safe_scatter_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("c", "color", "s", "cmap", "norm")
        }

        if is_categorical:
            from ._scatterplot_backend import _add_categorical_legend, _color_vector, _get_palette

            if not isinstance(color_data.dtype, pd.CategoricalDtype):
                color_data = pd.Series(
                    pd.Categorical(color_data), index=cells_in_fovs, name=color_key
                )
            color_data = color_data.cat.remove_unused_categories()

            color_vector, _ = _color_vector(
                adata, color_key, color_data, palette=palette, na_color=na_color
            )
            if isinstance(color_vector, pd.Categorical):
                face_colors = np.asarray(color_vector.astype(str), dtype=object)
            else:
                face_colors = np.asarray(color_vector, dtype=object)

            current_ax.scatter(
                cell_xy[:, 0], cell_xy[:, 1],
                c=face_colors, s=size, zorder=2, rasterized=rasterized, **safe_scatter_kwargs,
            )

            if legend:
                path_effect = (
                    [patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")]
                    if legend_fontoutline is not None else None
                )
                _add_categorical_legend(
                    current_ax,
                    color_data,
                    palette=_get_palette(adata, color_key, palette=palette),
                    scatter_array=cell_xy,
                    legend_loc=legend_loc,
                    legend_fontweight=legend_fontweight,
                    legend_fontsize=legend_fontsize,
                    legend_fontoutline=path_effect,
                    na_color=na_color,
                    na_in_legend=na_in_legend,
                    multi_panel=len(colors_to_plot) > 1,
                )
        else:
            vals = pd.to_numeric(color_data, errors="coerce").to_numpy()
            vmin_val = float(np.nanmin(vals)) if vmin is None else _resolve_vbound(vals, vmin)
            vmax_val = float(np.nanmax(vals)) if vmax is None else _resolve_vbound(vals, vmax)
            if not (np.isfinite(vmin_val) and np.isfinite(vmax_val)):
                vmin_val, vmax_val = 0.0, 1.0

            norm_obj = mcolors.Normalize(vmin=vmin_val, vmax=vmax_val)
            current_ax.scatter(
                cell_xy[:, 0], cell_xy[:, 1],
                c=vals, cmap=cmap, norm=norm_obj,
                s=size, zorder=2, rasterized=rasterized, **safe_scatter_kwargs,
            )

            if legend and colorbar_loc is not None:
                sm = plt.cm.ScalarMappable(norm=norm_obj, cmap=_resolve_cmap(cmap))
                sm.set_array([])
                try:
                    cb = plt.colorbar(sm, ax=current_ax, pad=0.01, fraction=0.08,
                                      aspect=30, location=colorbar_loc)
                except TypeError:
                    cb = plt.colorbar(sm, ax=current_ax, pad=0.01, fraction=0.08, aspect=30)
                if legend_fontsize is not None:
                    cb.ax.tick_params(labelsize=legend_fontsize)

        current_ax.set_xlim(*xlim)
        current_ax.set_ylim(*ylim)
        current_ax.set_aspect("equal")
        if xlabel:
            current_ax.set_xlabel(xlabel)
        if ylabel:
            current_ax.set_ylabel(ylabel)
        if not show_ticks:
            current_ax.set_xticks([])
            current_ax.set_yticks([])
        current_ax.set_title(color_key)

        axes_list.append(current_ax)

    if show:
        fig.tight_layout(rect=[0, 0, 0.95, 1] if ax is None else [0, 0, 1, 1])
        plt.show()

    return axes_list[0] if len(axes_list) == 1 else axes_list


# ---------------------------------------------------------------------------
# Public function: nanostringseg  (polygon-based)
# ---------------------------------------------------------------------------

def nanostringseg(
    adata,
    fovs: Optional[List[Union[str, int]]] = None,
    color: Optional[Union[str, List[str]]] = None,
    basis: str = "spatial_fov",
    groups: Optional[List[str]] = None,
    groupby: Optional[str] = None,
    figsize: Optional[tuple] = None,
    img_key: Optional[str] = "hires",
    img_extent_mode: str = "cells",
    bw: bool = False,
    rasterized: bool = False,
    scale_factor: Optional[float] = None,
    alpha_img: float = 0.5,
    cmap: Union[str, mcolors.Colormap] = "viridis",
    palette: Optional[Union[dict, list, np.ndarray]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    edges_width: float = 0.5,
    edges_color: str = "black",
    seg_contourpx: Optional[float] = None,
    seg_outline: bool = False,
    alpha: float = 0.8,
    na_color: str = "lightgray",
    legend: bool = True,
    legend_loc: str = "right margin",
    legend_fontsize: Optional[Union[int, float]] = 12,
    legend_fontweight: Union[str, int] = "bold",
    legend_fontoutline: Optional[float] = None,
    na_in_legend: bool = True,
    colorbar_loc: Optional[str] = "right",
    show: bool = True,
    ax: Optional[plt.Axes] = None,
    xlabel: Optional[str] = "global x (px)",
    ylabel: Optional[str] = "global y (px)",
    show_ticks: bool = False,
    **kwargs,
):
    """Plot Nanostring segmentation polygons across multiple FOVs.

    Cell segmentation boundaries (stored as WKT strings in ``adata.obs['geometry']``)
    are translated from per-FOV local pixel coordinates to the global coordinate
    space (``obsm['spatial_fov']``) and rendered as a single stitched figure.
    FOV background images (e.g. CellComposite) are placed at the correct
    global positions.

    Parameters
    ----------
    adata : AnnData
        Object produced by :func:`~omicverse.io.read_nanostring` with
        ``obs['geometry']`` containing WKT polygon strings.
    fovs : list of str or int, optional
        FOV identifiers to include.  If *None*, all FOVs are plotted.
    color : str or list of str, optional
        ``obs`` column or gene name for polygon fill colour.
        Multiple keys produce side-by-side panels.
    groups : list of str, optional
        Restrict plotting to these categories (requires a categorical ``color``
        or ``groupby``).
    groupby : str, optional
        ``obs`` column to use when ``groups`` is specified.
    figsize : tuple, optional
        Figure size ``(width, height)``.
    img_key : str, optional
        Key for the background image in ``uns['spatial'][fov]['images']``
        (default ``'hires'``).
    img_extent_mode : str
        Strategy used to place each FOV image. ``'cells'`` fits image extent to
        the selected basis bbox of the FOV's cells. ``'panel'`` stretches a
        single selected FOV image to the full panel extent.
    bw : bool
        If ``True``, render the background image in grayscale.
    rasterized : bool
        If ``True``, rasterize the segmentation polygon layer while keeping the
        overall figure in vector format when saving to SVG/PDF.
    scale_factor : float, optional
        Override the per-FOV scale factor.
    alpha_img : float
        Background image transparency (0–1, default 0.5).
    cmap : str
        Colormap for continuous features.
    palette : dict, list, or ndarray, optional
        Colour palette for categorical features.
    vmin, vmax : float, optional
        Value range for continuous colormap.
    edges_width : float
        Polygon edge line width.
    edges_color : str
        Polygon edge colour.
    seg_contourpx : float, optional
        If set, draw only an outline of this stroke width instead of filled polygons.
    seg_outline : bool
        If *True*, draw outline-only polygons (overrides ``seg_contourpx``).
    alpha : float
        Polygon fill transparency (0–1).
    na_color : str
        Colour for cells with missing values.
    legend : bool
        Whether to draw a legend or colorbar.
    legend_loc : str
        Legend position (e.g. ``'right margin'``).
    legend_fontsize : int or float, optional
        Legend label size.
    legend_fontweight : str or int
        Legend label weight.
    legend_fontoutline : float, optional
        Stroke width for legend text outlines.
    na_in_legend : bool
        Whether to include NA category in the legend.
    colorbar_loc : str, optional
        Colorbar anchor.
    show : bool
        Call :func:`plt.show` after plotting.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes (single panel only).
    xlabel, ylabel : str, optional
        Axis labels.
    show_ticks : bool
        Show axis tick marks and labels.
    **kwargs
        Extra keyword arguments forwarded to geopandas ``GeoDataFrame.plot``.

    Returns
    -------
    matplotlib.axes.Axes or list of matplotlib.axes.Axes
    """
    # --- Validate data ---
    if "geometry" not in adata.obs.columns:
        raise ValueError(
            "`nanostringseg` requires `adata.obs['geometry']` with WKT polygon strings. "
            "Make sure the data was loaded with `ov.io.read_nanostring()` and that "
            "segmentation images (CellLabels) or polygon columns are available."
        )
    if "spatial" not in adata.uns or not adata.uns["spatial"]:
        raise ValueError("`nanostringseg` requires non-empty `adata.uns['spatial']`.")
    basis = _validate_basis(adata, basis)
    img_extent_mode = _validate_img_extent_mode(img_extent_mode)

    gpd, wkt_mod = _require_geopandas()

    # pop segmentation args that should not reach geopandas
    kwargs.pop("seg_contourpx", None)
    kwargs.pop("seg_outline", None)
    kwargs.pop("crop_coord", None)
    if "color_dict" in kwargs:
        legacy = kwargs.pop("color_dict")
        if legacy is not None:
            if palette is not None:
                warnings.warn("Both `palette` and legacy `color_dict` provided. Using `color_dict`.")
            palette = legacy

    outline_only = bool(seg_outline or (seg_contourpx is not None))
    if seg_contourpx is not None:
        edges_width = float(seg_contourpx)

    fov_col = _get_fov_column(adata)
    all_fovs = sorted(
        adata.obs[fov_col].astype(str).unique().tolist(),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    if fovs is None:
        fovs_to_plot = all_fovs
    else:
        fovs_to_plot = _normalize_fov_ids(fovs)
        missing = [f for f in fovs_to_plot if f not in all_fovs]
        if missing:
            raise ValueError(
                f"FOV(s) {missing} not found in adata.obs['{fov_col}']. "
                f"Available: {all_fovs}"
            )

    if color is None:
        colors_to_plot = [None]
    elif isinstance(color, str):
        colors_to_plot = [color]
    else:
        colors_to_plot = list(color)

    for ck in colors_to_plot:
        if ck is None:
            continue
        if ck not in adata.obs.columns and ck not in adata.var_names:
            raise ValueError(
                f"`color` key '{ck}' not found in `adata.obs.columns` or `adata.var_names`."
            )

    # --- Build figure / axes ---
    if ax is None:
        if figsize is None:
            figsize = (6 * len(colors_to_plot), 6) if len(colors_to_plot) > 1 else (10, 10)
        if len(colors_to_plot) > 1:
            fig, axes_arr = plt.subplots(1, len(colors_to_plot), figsize=figsize,
                                         sharex=True, sharey=True)
            axes = list(axes_arr)
        else:
            fig, ax_single = plt.subplots(1, 1, figsize=figsize)
            axes = [ax_single]
    else:
        fig = ax.figure
        axes = [ax]
        if len(colors_to_plot) > 1:
            warnings.warn(
                "Multiple colors specified but only one ax provided. "
                "Only the first color will be plotted."
            )
            colors_to_plot = [colors_to_plot[0]]

    xlim, ylim = _global_bounding_box(adata, fovs_to_plot, fov_col, basis, img_extent_mode)

    # Maps fov_id -> `(sx, sy, tx, ty)` needed to place local polygons in the chosen basis.
    fov_affines: dict = {}
    for fov_id in fovs_to_plot:
        fov_affines[fov_id] = _compute_fov_geometry_transform(adata, fov_id, fov_col, basis)

    fov_mask = adata.obs[fov_col].astype(str).isin(set(fovs_to_plot))
    if groups is not None:
        if groupby is not None:
            if groupby not in adata.obs.columns:
                raise ValueError(f"`groupby` column '{groupby}' not found in `adata.obs.columns`.")
            grp_col = groupby
        elif colors_to_plot[0] is not None and colors_to_plot[0] in adata.obs.columns:
            grp_col = colors_to_plot[0]
        else:
            raise ValueError(
                "`groups` requires either a categorical `color` column in `adata.obs` "
                "or an explicit `groupby` parameter."
            )
        group_mask = adata.obs[grp_col].isin(groups)
        fov_mask = fov_mask & group_mask

    cells_to_plot = adata.obs_names[fov_mask]
    if len(cells_to_plot) == 0:
        warnings.warn("No cells to plot after applying FOV / group filters.")
        ax_out = axes[0]
        ax_out.set_xlim(*xlim)
        ax_out.set_ylim(*ylim)
        ax_out.set_aspect("equal")
        if show:
            plt.show()
        return ax_out

    # Translate WKT geometries to global coordinate space
    try:
        from shapely import affinity as shp_affinity
    except ImportError as exc:
        raise ImportError(
            "`nanostringseg` requires `shapely`. Install with: pip install shapely"
        ) from exc

    geom_list = []
    valid_cells = []
    for cell_id in cells_to_plot:
        wkt_str = adata.obs.at[cell_id, "geometry"]
        if not wkt_str or pd.isna(wkt_str):
            continue
        try:
            geom = wkt_mod.loads(wkt_str)
        except Exception:
            continue
        if geom is None or not hasattr(geom, "bounds"):
            continue
        if hasattr(geom, "is_valid") and not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue

        # Affine-map local geometry coordinates into the stitched global frame.
        fov_id = str(adata.obs.at[cell_id, fov_col])
        sx, sy, tx, ty = fov_affines.get(fov_id, (1.0, 1.0, 0.0, 0.0))
        if sx != 1.0 or sy != 1.0:
            geom = shp_affinity.scale(geom, xfact=sx, yfact=sy, origin=(0.0, 0.0))
        if tx != 0.0 or ty != 0.0:
            geom = shp_affinity.translate(geom, xoff=tx, yoff=ty)

        bounds = geom.bounds
        if (not all(np.isfinite(bounds))
                or bounds[2] <= bounds[0]
                or bounds[3] <= bounds[1]):
            continue

        geom_list.append(geom)
        valid_cells.append(cell_id)

    if len(valid_cells) == 0:
        warnings.warn("No valid geometries found after filtering.")
        ax_out = axes[0]
        ax_out.set_xlim(*xlim)
        ax_out.set_ylim(*ylim)
        ax_out.set_aspect("equal")
        if show:
            plt.show()
        return ax_out

    geom_series = gpd.GeoSeries(geom_list, index=valid_cells)

    axes_list = []
    for plot_idx, color_key in enumerate(colors_to_plot):
        current_ax = axes[plot_idx]

        # Background images
        _place_fov_images(
            current_ax,
            adata,
            fovs_to_plot,
            fov_col,
            basis,
            img_key,
            1.0 if color_key is None else alpha_img,
            bw,
            img_extent_mode,
            panel_extent=(xlim[0], xlim[1], ylim[1], ylim[0]),
        )

        if color_key is None:
            current_ax.set_xlim(*xlim)
            current_ax.set_ylim(*ylim)
            current_ax.set_aspect("equal")
            if xlabel:
                current_ax.set_xlabel(xlabel)
            if ylabel:
                current_ax.set_ylabel(ylabel)
            if not show_ticks:
                current_ax.set_xticks([])
                current_ax.set_yticks([])
            axes_list.append(current_ax)
            continue

        # Build GeoDataFrame with colour column
        if color_key in adata.obs.columns:
            color_data = adata.obs.loc[valid_cells, color_key]
        else:
            idx_pos = adata.obs_names.get_indexer(valid_cells)
            gene_vals = get_values(adata, color_key)[idx_pos]
            color_data = pd.Series(gene_vals, index=valid_cells, name=color_key)

        temp_gdf = gpd.GeoDataFrame({color_key: color_data}, geometry=geom_series,
                                     index=valid_cells)

        # Retry loop for invalid bounds (mirrors spatialseg logic)
        for _retry in range(3):
            try:
                tb = temp_gdf.total_bounds
                if all(np.isfinite(tb)) and tb[2] > tb[0] and tb[3] > tb[1]:
                    break
                valid_mask = pd.Series(True, index=temp_gdf.index)
                for idx2 in temp_gdf.index:
                    try:
                        g = temp_gdf.at[idx2, "geometry"]
                        if g is None or pd.isna(g):
                            valid_mask.at[idx2] = False
                            continue
                        gb = g.bounds
                        if not all(np.isfinite(gb)) or gb[2] <= gb[0] or gb[3] <= gb[1]:
                            valid_mask.at[idx2] = False
                    except Exception:
                        valid_mask.at[idx2] = False
                temp_gdf = temp_gdf[valid_mask]
                if len(temp_gdf) == 0:
                    warnings.warn("No valid geometries remaining. Skipping panel.")
                    break
            except Exception:
                break

        if len(temp_gdf) == 0:
            axes_list.append(current_ax)
            continue

        is_categorical = temp_gdf[color_key].dtype == bool or (
            not pd.api.types.is_numeric_dtype(temp_gdf[color_key])
        )

        plot_kw = {
            "ax": current_ax,
            "edgecolor": edges_color,
            "linewidth": edges_width,
            **{k: v for k, v in kwargs.items()
               if k not in ("column", "cmap", "color", "legend", "aspect")},
        }

        if is_categorical:
            from ._scatterplot_backend import _add_categorical_legend, _color_vector, _get_palette

            cd = temp_gdf[color_key]
            if not isinstance(cd.dtype, pd.CategoricalDtype):
                cd = pd.Series(pd.Categorical(cd), index=temp_gdf.index, name=color_key)
            cd = cd.cat.remove_unused_categories()

            color_vector, _ = _color_vector(
                adata, color_key, cd, palette=palette, na_color=na_color
            )
            if isinstance(color_vector, pd.Categorical):
                face_colors = np.asarray(color_vector.astype(str), dtype=object)
            else:
                face_colors = np.asarray(color_vector, dtype=object)

            if outline_only:
                plot_kw["facecolor"] = "none"
                plot_kw["edgecolor"] = face_colors
            else:
                plot_kw["color"] = face_colors
                plot_kw["alpha"] = alpha
            plot_kw["legend"] = False
            collection_start = len(current_ax.collections)

            try:
                temp_gdf.plot(**plot_kw)
            except ValueError as exc:
                if "aspect must be finite" in str(exc):
                    plot_kw["aspect"] = "equal"
                    temp_gdf.plot(**plot_kw)
                else:
                    raise
            if rasterized:
                _rasterize_new_collections(current_ax, collection_start)

            if legend:
                path_effect = (
                    [patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")]
                    if legend_fontoutline is not None else None
                )
                try:
                    centroids = np.column_stack((
                        temp_gdf.geometry.centroid.x.to_numpy(),
                        temp_gdf.geometry.centroid.y.to_numpy(),
                    ))
                except Exception:
                    bdf = temp_gdf.bounds
                    centroids = np.column_stack((
                        (bdf["minx"].to_numpy() + bdf["maxx"].to_numpy()) / 2,
                        (bdf["miny"].to_numpy() + bdf["maxy"].to_numpy()) / 2,
                    ))
                _add_categorical_legend(
                    current_ax,
                    cd,
                    palette=_get_palette(adata, color_key, palette=palette),
                    scatter_array=centroids,
                    legend_loc=legend_loc,
                    legend_fontweight=legend_fontweight,
                    legend_fontsize=legend_fontsize,
                    legend_fontoutline=path_effect,
                    na_color=na_color,
                    na_in_legend=na_in_legend,
                    multi_panel=len(colors_to_plot) > 1,
                )
        else:
            vals = pd.to_numeric(temp_gdf[color_key], errors="coerce").to_numpy()
            vmin_val = float(np.nanmin(vals)) if vmin is None else _resolve_vbound(vals, vmin)
            vmax_val = float(np.nanmax(vals)) if vmax is None else _resolve_vbound(vals, vmax)
            if not (np.isfinite(vmin_val) and np.isfinite(vmax_val)) or vmax_val <= vmin_val:
                vmin_val, vmax_val = 0.0, 1.0

            cmap_obj = _resolve_cmap(cmap)
            norm_obj = mcolors.Normalize(vmin=vmin_val, vmax=vmax_val)

            if outline_only:
                edge_colors = []
                for v in vals:
                    if pd.isna(v):
                        edge_colors.append(na_color)
                    else:
                        edge_colors.append(cmap_obj(norm_obj(float(v))))
                edge_colors = _scale_rgba_alpha(edge_colors, alpha)
                plot_kw["facecolor"] = "none"
                plot_kw["edgecolor"] = edge_colors
                plot_kw["legend"] = False
                collection_start = len(current_ax.collections)
                try:
                    temp_gdf.plot(**plot_kw)
                except ValueError as exc:
                    if "aspect must be finite" in str(exc):
                        plot_kw["aspect"] = "equal"
                        temp_gdf.plot(**plot_kw)
                    else:
                        raise
                if rasterized:
                    _rasterize_new_collections(current_ax, collection_start)
            else:
                face_colors = []
                for v in vals:
                    if pd.isna(v):
                        face_colors.append(na_color)
                    else:
                        face_colors.append(cmap_obj(norm_obj(float(v))))
                face_colors = _scale_rgba_alpha(face_colors, alpha)
                plot_kw["color"] = face_colors
                plot_kw["legend"] = False
                collection_start = len(current_ax.collections)
                try:
                    temp_gdf.plot(**plot_kw)
                except ValueError as exc:
                    if "aspect must be finite" in str(exc):
                        plot_kw["aspect"] = "equal"
                        temp_gdf.plot(**plot_kw)
                    else:
                        raise
                if rasterized:
                    _rasterize_new_collections(current_ax, collection_start)

            if legend and colorbar_loc is not None:
                sm = plt.cm.ScalarMappable(norm=norm_obj, cmap=cmap_obj)
                sm.set_array([])
                # Use last collection if available, else ScalarMappable
                mappable = (current_ax.collections[-1]
                            if current_ax.collections and not outline_only else sm)
                try:
                    cb = plt.colorbar(mappable, ax=current_ax, pad=0.01,
                                      fraction=0.08, aspect=30, location=colorbar_loc)
                except TypeError:
                    cb = plt.colorbar(mappable, ax=current_ax, pad=0.01,
                                      fraction=0.08, aspect=30)
                if legend_fontsize is not None:
                    cb.ax.tick_params(labelsize=legend_fontsize)

        current_ax.set_aspect("equal")
        current_ax.set_xlim(*xlim)
        current_ax.set_ylim(*ylim)
        if xlabel:
            current_ax.set_xlabel(xlabel)
        if ylabel:
            current_ax.set_ylabel(ylabel)
        if not show_ticks:
            current_ax.set_xticks([])
            current_ax.set_yticks([])
        current_ax.set_title(color_key)

        axes_list.append(current_ax)

    if show:
        fig.tight_layout(rect=[0, 0, 0.95, 1] if ax is None else [0, 0, 1, 1])
        plt.show()

    return axes_list[0] if len(axes_list) == 1 else axes_list


__all__ = ["nanostring", "nanostringseg"]
