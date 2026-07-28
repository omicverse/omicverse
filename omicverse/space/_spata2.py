from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .._registry import register_function


def _ensure_adata(adata: AnnData) -> AnnData:
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData object.")
    return adata


def _coords_array(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
    x: str = "x",
    y: str = "y",
) -> np.ndarray:
    _ensure_adata(adata)

    if spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key])
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"adata.obsm[{spatial_key!r}] must have shape n_obs x >=2.")
        coords = coords[:, :2]
    elif x in adata.obs and y in adata.obs:
        coords = adata.obs[[x, y]].to_numpy()
    else:
        raise KeyError(
            f"Coordinates require adata.obsm[{spatial_key!r}] or obs columns {x!r}/{y!r}."
        )

    coords = coords.astype(float, copy=False)
    if not np.isfinite(coords).all():
        raise ValueError("Coordinates must be finite numeric values.")
    if coords.shape[0] != adata.n_obs:
        raise ValueError("Coordinate rows must match adata.n_obs.")
    return coords


@register_function(
    aliases=["SPATA2坐标提取", "spatial coordinates", "获取空间坐标", "coords_df"],
    category="space",
    description="Return SPATA2-style spatial coordinates as a barcode-indexed table, optionally joined with obs metadata",
    requires={'obsm': ['spatial']},
    produces={},
    auto_fix='none',
    examples=[
        "coords = ov.space.spata2_get_coords(adata)",
        "coords = ov.space.spata2_get_coords(adata, include_obs=['cluster'])",
    ],
)
def spata2_get_coords(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
    x: str = "x",
    y: str = "y",
    include_obs: bool | Sequence[str] = False,
    barcode_col: str = "barcode",
) -> pd.DataFrame:
    """Return SPATA2-style spatial coordinates as an observation-indexed table.

    Parameters
    ----------
    adata
        Spatial AnnData object.
    spatial_key
        Key in ``adata.obsm`` containing x/y coordinates. If absent, ``x`` and
        ``y`` are read from ``adata.obs``.
    x, y
        Output coordinate column names, and fallback ``adata.obs`` column names.
    include_obs
        ``False`` for coordinates only, ``True`` for all observation metadata, or
        a list of observation columns to join.
    barcode_col
        Name of the column containing observation IDs.

    Returns
    -------
    pandas.DataFrame
        Columns ``barcode_col``, ``x``, ``y``, and optional observation metadata.
    """

    coords = _coords_array(adata, spatial_key=spatial_key, x=x, y=y)
    df = pd.DataFrame(coords, index=adata.obs_names.copy(), columns=[x, y])
    df.insert(0, barcode_col, adata.obs_names.to_numpy())

    if include_obs is True:
        obs = adata.obs.copy()
    elif include_obs:
        missing = [col for col in include_obs if col not in adata.obs]
        if missing:
            raise KeyError(f"Observation columns not found: {missing}")
        obs = adata.obs.loc[:, list(include_obs)].copy()
    else:
        obs = None

    if obs is not None:
        overlap = obs.columns.intersection(df.columns)
        if len(overlap) > 0:
            obs = obs.drop(columns=list(overlap))
        if obs.shape[1] > 0:
            df = df.join(obs)
    return df


def _matrix_for_variables(
    adata: AnnData,
    *,
    layer: str | None,
    use_raw: bool,
) -> tuple[Any, pd.Index]:
    if use_raw and layer is not None:
        raise ValueError("`layer` and `use_raw=True` cannot be used together.")
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True requires adata.raw.")
        return adata.raw.X, pd.Index(adata.raw.var_names)
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} is not present in adata.layers.")
        return adata.layers[layer], pd.Index(adata.var_names)
    return adata.X, pd.Index(adata.var_names)


def _to_1d(values: Any) -> np.ndarray:
    if sparse.issparse(values):
        values = values.toarray()
    values = np.asarray(values)
    if values.ndim == 2 and 1 in values.shape:
        values = values.reshape(-1)
    if values.ndim != 1:
        raise ValueError("Extracted variable must be one-dimensional.")
    return values


@register_function(
    aliases=["SPATA2变量提取", "extract variables", "提取基因表达", "取变量"],
    category="space",
    description="Extract obs columns and/or gene expression into one observation-indexed frame",
    requires={},
    produces={},
    auto_fix='none',
    examples=[
        "vals = ov.space.spata2_extract_variables(adata, ['METRN', 'total_counts'])",
    ],
)
def spata2_extract_variables(
    adata: AnnData,
    variables: str | Iterable[str],
    *,
    layer: str | None = None,
    use_raw: bool = False,
) -> pd.DataFrame:
    """Extract observation metadata and molecular variables from AnnData.

    Names found in ``adata.obs`` are returned as metadata. Other names are
    resolved against ``adata.var_names`` and extracted from ``adata.X``, a layer,
    or ``adata.raw.X``.
    """

    _ensure_adata(adata)
    if isinstance(variables, str):
        names = [variables]
    else:
        names = list(variables)
    if not names:
        raise ValueError("variables must contain at least one name.")

    matrix, var_names = _matrix_for_variables(adata, layer=layer, use_raw=use_raw)
    out: dict[str, Any] = {}

    for name in names:
        if name in adata.obs:
            out[name] = adata.obs[name].to_numpy()
            continue
        matches = np.flatnonzero(var_names == name)
        if matches.size == 0:
            raise KeyError(f"Variable {name!r} is neither in adata.obs nor var_names.")
        if matches.size > 1:
            raise ValueError(f"Variable {name!r} is duplicated in var_names.")
        out[name] = _to_1d(matrix[:, int(matches[0])])

    return pd.DataFrame(out, index=adata.obs_names.copy())


@register_function(
    aliases=["SPATA2坐标合并", "join variables", "坐标与变量拼接"],
    category="space",
    description="Join spatial coordinates with selected obs or gene variables in a single table",
    requires={'obsm': ['spatial']},
    produces={},
    auto_fix='none',
    examples=[
        "df = ov.space.spata2_join_variables(adata, ['METRN'])",
    ],
)
def spata2_join_variables(
    adata: AnnData,
    variables: str | Iterable[str],
    *,
    spatial_key: str = "spatial",
    x: str = "x",
    y: str = "y",
    layer: str | None = None,
    use_raw: bool = False,
) -> pd.DataFrame:
    """Join spatial coordinates with selected metadata or molecular variables."""

    coords = spata2_get_coords(adata, spatial_key=spatial_key, x=x, y=y)
    values = spata2_extract_variables(adata, variables, layer=layer, use_raw=use_raw)
    return coords.join(values)


@register_function(
    aliases=["SPATA2组织轮廓", "tissue outline", "组织边界", "convex hull"],
    category="space",
    description="Outline the observed tissue coordinates with a convex hull and store it in adata.uns",
    requires={'obsm': ['spatial']},
    produces={'uns': ['spata2_tissue_outline', 'spata2_tissue_outline_source_obs']},
    auto_fix='none',
    examples=[
        "outline = ov.space.spata2_tissue_outline(adata)",
    ],
)
def spata2_tissue_outline(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
    x: str = "x",
    y: str = "y",
    write_key: str | None = "spata2_tissue_outline",
) -> pd.DataFrame:
    """Identify a convex hull outlining the observed tissue coordinates.

    SPATA2's ``identifyTissueOutline(method = "obs")`` is not a convex hull: it
    runs ``concaveman(concavity = 2)`` over the spots, and separately assigns a
    ``tissue_section`` to every spot with DBSCAN (``eps = CCD * 1.25``,
    ``minPts = 3`` on Visium) so a slide carrying several fragments yields one
    outline per fragment.

    On real Visium data the two agree closely, because the spots fill the capture
    area and the point cloud is already near-convex. Measured against R
    ``concaveman`` 1.2.0 on the same coordinates:

    ==========================  ============  ==============  ========
    data                        convex        concaveman(2)   ratio
    ==========================  ============  ==============  ========
    SPATA2 ``example_data``       173,041.8       168,821.6     1.025
    DLPFC 151673              316,686,266     316,430,828      1.001
    ==========================  ============  ==============  ========

    DBSCAN found a single section and no noise spots in both, so the per-section
    decomposition made no difference either. Expect a wider gap on a slide whose
    tissue is genuinely concave, or that carries more than one fragment: a convex
    hull bridges the opening and merges the fragments into one polygon.
    """

    coords = _coords_array(adata, spatial_key=spatial_key, x=x, y=y)
    unique_coords, unique_idx = np.unique(coords, axis=0, return_index=True)

    if unique_coords.shape[0] < 3:
        hull_idx = np.lexsort((unique_coords[:, 1], unique_coords[:, 0]))
        hull_coords = unique_coords[hull_idx]
    else:
        try:
            hull = ConvexHull(unique_coords)
            hull_idx = hull.vertices
            hull_coords = unique_coords[hull_idx]
        except QhullError:
            hull_idx = np.lexsort((unique_coords[:, 1], unique_coords[:, 0]))
            hull_coords = unique_coords[hull_idx]

    outline = pd.DataFrame(hull_coords, columns=[x, y])
    outline.index.name = "vertex"

    if write_key is not None:
        adata.uns[write_key] = outline.copy()
        adata.uns[f"{write_key}_source_obs"] = adata.obs_names.to_numpy()[unique_idx[hull_idx]].tolist()

    return outline


def _robust_threshold(values: np.ndarray, scale: float) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        q75, q25 = np.percentile(values, [75, 25])
        mad = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    if mad == 0:
        return float(np.max(values))
    return median + scale * 1.4826 * mad


@register_function(
    aliases=["SPATA2离群检测", "spatial outliers", "空间离群点", "孤立点检测"],
    category="space",
    description="Flag spatially isolated observations from k-nearest-neighbour distances with a median/MAD threshold",
    requires={'obsm': ['spatial']},
    produces={'obs': ['spata2_spatial_outlier']},
    auto_fix='none',
    examples=[
        "flags = ov.space.spata2_identify_outliers(adata)",
        "flags = ov.space.spata2_identify_outliers(adata, radius=150, min_neighbors=3)",
    ],
)
def spata2_identify_outliers(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
    x: str = "x",
    y: str = "y",
    radius: float | None = None,
    min_neighbors: int = 3,
    mad_scale: float = 3.5,
    write_key: str | None = "spata2_spatial_outlier",
) -> pd.Series:
    """Flag observations that are spatially isolated from the main tissue."""

    if min_neighbors < 1:
        raise ValueError("min_neighbors must be >= 1.")
    if radius is not None and radius <= 0:
        raise ValueError("radius must be positive when provided.")

    coords = _coords_array(adata, spatial_key=spatial_key, x=x, y=y)
    n_obs = coords.shape[0]
    if n_obs == 0:
        outliers = np.zeros(0, dtype=bool)
    elif n_obs <= min_neighbors:
        outliers = np.zeros(n_obs, dtype=bool)
    else:
        tree = cKDTree(coords)
        if radius is not None:
            neighbors = tree.query_ball_point(coords, r=radius)
            counts = np.fromiter((len(items) - 1 for items in neighbors), dtype=int, count=n_obs)
            outliers = counts < min_neighbors
        else:
            k = min(min_neighbors + 1, n_obs)
            distances, _ = tree.query(coords, k=k)
            kth_dist = np.asarray(distances[:, -1], dtype=float)
            threshold = _robust_threshold(kth_dist, mad_scale)
            outliers = kth_dist > threshold

    series = pd.Series(outliers, index=adata.obs_names.copy(), name=write_key or "spatial_outlier")
    if write_key is not None:
        adata.obs[write_key] = series
    return series


@register_function(
    aliases=["SPATA2离群移除", "remove outliers", "剔除离群点"],
    category="space",
    description="Drop the observations flagged as spatially isolated, in place or as a copy",
    requires={'obsm': ['spatial']},
    produces={'obs': ['spata2_spatial_outlier']},
    auto_fix='none',
    examples=[
        "clean = ov.space.spata2_remove_outliers(adata, copy=True)",
    ],
)
def spata2_remove_outliers(
    adata: AnnData,
    *,
    outlier_key: str = "spata2_spatial_outlier",
    spatial_key: str = "spatial",
    x: str = "x",
    y: str = "y",
    radius: float | None = None,
    min_neighbors: int = 3,
    mad_scale: float = 3.5,
    copy: bool = True,
) -> AnnData | None:
    """Remove observations flagged as SPATA2-style spatial outliers."""

    _ensure_adata(adata)
    if outlier_key not in adata.obs:
        spata2_identify_outliers(
            adata,
            spatial_key=spatial_key,
            x=x,
            y=y,
            radius=radius,
            min_neighbors=min_neighbors,
            mad_scale=mad_scale,
            write_key=outlier_key,
        )

    keep = ~adata.obs[outlier_key].astype(bool).to_numpy()
    if copy:
        return adata[keep].copy()

    adata._inplace_subset_obs(keep)
    return None


@register_function(
    aliases=["SPATA2像素转换", "pixels to unit", "像素转物理单位"],
    category="space",
    description="Convert pixel distances to physical units given a pixels-per-unit scale factor",
    requires={},
    produces={},
    auto_fix='none',
    examples=[
        "microns = ov.space.spata2_pixels_to_unit(120.0, pixels_per_unit=2.5)",
    ],
)
def spata2_pixels_to_unit(pixels: Any, pixels_per_unit: float) -> Any:
    """Convert pixel distances to physical units with an explicit scale."""

    if pixels_per_unit <= 0:
        raise ValueError("pixels_per_unit must be positive.")
    return np.asarray(pixels) / pixels_per_unit


@register_function(
    aliases=["SPATA2单位转像素", "unit to pixels", "物理单位转像素"],
    category="space",
    description="Convert physical distances to pixels given a pixels-per-unit scale factor",
    requires={},
    produces={},
    auto_fix='none',
    examples=[
        "px = ov.space.spata2_unit_to_pixels(48.0, pixels_per_unit=2.5)",
    ],
)
def spata2_unit_to_pixels(units: Any, pixels_per_unit: float) -> Any:
    """Convert physical-unit distances to pixels with an explicit scale."""

    if pixels_per_unit <= 0:
        raise ValueError("pixels_per_unit must be positive.")
    return np.asarray(units) * pixels_per_unit
