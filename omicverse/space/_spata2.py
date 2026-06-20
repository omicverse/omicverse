from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.spatial import ConvexHull, QhullError, cKDTree


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
        df = df.join(obs)
    return df


def _matrix_for_variables(
    adata: AnnData,
    *,
    layer: str | None,
    use_raw: bool,
) -> tuple[Any, pd.Index]:
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


def spata2_tissue_outline(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
    x: str = "x",
    y: str = "y",
    write_key: str | None = "spata2_tissue_outline",
) -> pd.DataFrame:
    """Identify a convex hull outlining the observed tissue coordinates."""

    coords = _coords_array(adata, spatial_key=spatial_key, x=x, y=y)
    unique_coords, unique_idx = np.unique(coords, axis=0, return_index=True)

    if unique_coords.shape[0] < 3:
        hull_coords = unique_coords[np.lexsort((unique_coords[:, 1], unique_coords[:, 0]))]
    else:
        try:
            hull = ConvexHull(unique_coords)
            hull_coords = unique_coords[hull.vertices]
        except QhullError:
            hull_coords = unique_coords[np.lexsort((unique_coords[:, 1], unique_coords[:, 0]))]

    outline = pd.DataFrame(hull_coords, columns=[x, y])
    outline.index.name = "vertex"

    if write_key is not None:
        adata.uns[write_key] = outline.copy()
        adata.uns[f"{write_key}_source_obs"] = adata.obs_names.to_numpy()[unique_idx].tolist()

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


def spata2_pixels_to_unit(pixels: Any, pixels_per_unit: float) -> Any:
    """Convert pixel distances to physical units with an explicit scale."""

    if pixels_per_unit <= 0:
        raise ValueError("pixels_per_unit must be positive.")
    return np.asarray(pixels) / pixels_per_unit


def spata2_unit_to_pixels(units: Any, pixels_per_unit: float) -> Any:
    """Convert physical-unit distances to pixels with an explicit scale."""

    if pixels_per_unit <= 0:
        raise ValueError("pixels_per_unit must be positive.")
    return np.asarray(units) * pixels_per_unit
