"""Nanostring SMI reader for OmicVerse spatial I/O."""

from __future__ import annotations

import ast
import os
import re
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from PIL import Image
from scipy.sparse import csr_matrix

from ..._registry import register_function

try:
    from ..._settings import Colors
except Exception:
    class Colors:
        """Fallback ANSI color codes when omicverse._settings import is unavailable."""

        HEADER = "\033[95m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        WARNING = "\033[93m"
        FAIL = "\033[91m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"
        UNDERLINE = "\033[4m"


def _progress(message: str, level: str = "info") -> None:
    color = Colors.CYAN
    if level == "success":
        color = Colors.GREEN
    elif level == "warn":
        color = Colors.WARNING
    print(f"{color}[Nanostring] {message}{Colors.ENDC}")


def _read_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img)


def _find_first_column(df: pd.DataFrame, candidates: tuple[str, ...], context: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find required {context} column. Tried: {list(candidates)}")


def _find_xy_columns(df: pd.DataFrame, kind: str) -> Optional[tuple[str, str]]:
    pairs = (
        ("CenterX_local_px", "CenterY_local_px"),
        ("centerx_local_px", "centery_local_px"),
        ("center_x_local_px", "center_y_local_px"),
        ("CenterX", "CenterY"),
    ) if kind == "local" else (
        ("CenterX_global_px", "CenterY_global_px"),
        ("centerx_global_px", "centery_global_px"),
        ("center_x_global_px", "center_y_global_px"),
        ("x_global_px", "y_global_px"),
    )
    for x_col, y_col in pairs:
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col
    return None


def _maybe_parse_points(value) -> Optional[list[tuple[float, float]]]:
    parsed = value
    if parsed is None or (isinstance(parsed, float) and np.isnan(parsed)):
        return None

    if isinstance(parsed, str):
        text = parsed.strip()
        if text == "":
            return None
        if text.upper().startswith(("POLYGON", "MULTIPOLYGON")):
            return None
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return None

    if isinstance(parsed, dict):
        if "coordinates" in parsed:  # GeoJSON-like
            coords = parsed["coordinates"]
            if isinstance(coords, (list, tuple)) and len(coords) > 0:
                ring = coords[0]
                if isinstance(ring, (list, tuple)):
                    pts = []
                    for p in ring:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            pts.append((float(p[0]), float(p[1])))
                    return pts if len(pts) >= 3 else None
            return None
        if "x" in parsed and "y" in parsed:
            x = parsed["x"]
            y = parsed["y"]
            if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)) and len(x) == len(y):
                pts = [(float(px), float(py)) for px, py in zip(x, y)]
                return pts if len(pts) >= 3 else None
            return None

    if isinstance(parsed, (list, tuple)):
        pts = []
        for p in parsed:
            if isinstance(p, dict) and "x" in p and "y" in p:
                pts.append((float(p["x"]), float(p["y"])))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
        return pts if len(pts) >= 3 else None

    return None


def _points_to_wkt(points: list[tuple[float, float]]) -> str:
    try:
        from shapely.geometry import Polygon
    except Exception as exc:
        raise ImportError(
            "Converting Nanostring polygon points to WKT requires `shapely`. "
            "Install with: pip install shapely"
        ) from exc

    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return ""
    return poly.wkt


def _extract_geometry_wkt_from_obs(obs: pd.DataFrame) -> Optional[pd.Series]:
    geometry_like_columns = (
        "geometry",
        "Geometry",
        "wkt",
        "WKT",
        "polygon",
        "Polygon",
        "cell_polygon",
        "cell_polygons",
        "cell_boundary",
        "cell_boundaries",
        "boundary",
        "boundaries",
    )
    geom_col = next((col for col in geometry_like_columns if col in obs.columns), None)
    if geom_col is None:
        return None

    raw = obs[geom_col]
    result = pd.Series("", index=obs.index, dtype=object)
    converted = 0

    for idx, val in raw.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if isinstance(val, str) and val.strip().upper().startswith(("POLYGON", "MULTIPOLYGON")):
            result.at[idx] = val
            converted += 1
            continue
        points = _maybe_parse_points(val)
        if points is None:
            continue
        try:
            result.at[idx] = _points_to_wkt(points)
            if result.at[idx] != "":
                converted += 1
        except ImportError:
            raise
        except Exception:
            continue

    return result if converted > 0 else None


def _geometry_from_segmentation_images(
    adata: AnnData,
    *,
    fov_key: str,
    cell_id_key: str,
) -> Optional[pd.Series]:
    try:
        from shapely.geometry import Polygon, shape
        from shapely.ops import unary_union
    except ImportError:
        warnings.warn(
            "Cannot extract cell contours from segmentation images: `shapely` is not installed. "
            "Install with: pip install shapely"
        )
        return None

    try:
        from skimage.measure import find_contours, regionprops
    except ImportError:
        warnings.warn(
            "Cannot extract cell contours from segmentation images: `scikit-image` is not installed. "
            "Install with: pip install scikit-image"
        )
        return None

    if cell_id_key not in adata.obs.columns or fov_key not in adata.obs.columns:
        return None

    def _decode_label_image(seg: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(seg)
        if arr.ndim == 2:
            if np.issubdtype(arr.dtype, np.integer):
                return arr.astype(np.int64, copy=False)
            return np.rint(arr).astype(np.int64)

        if arr.ndim == 3:
            if arr.shape[2] == 1:
                return np.rint(arr[..., 0]).astype(np.int64)

            if arr.shape[2] >= 3:
                rgb = arr[..., :3]
                # If channels are identical, it's effectively grayscale.
                if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
                    return np.rint(rgb[..., 0]).astype(np.int64)

                # Decode packed integer labels from RGB channels.
                # Works for common segmentation exports where label = R + 256*G + 65536*B.
                rgb = rgb.astype(np.int64, copy=False)
                decoded = rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)
                return decoded

        return None

    out = pd.Series("", index=adata.obs.index, dtype=object)
    written = 0

    for fov in adata.obs[fov_key].astype("category").cat.categories.astype(str):
        fov_cells = adata.obs.index[adata.obs[fov_key].astype(str) == fov]
        if len(fov_cells) == 0:
            continue

        seg = adata.uns.get("spatial", {}).get(fov, {}).get("images", {}).get("segmentation")
        if seg is None:
            continue

        label_img = _decode_label_image(seg)
        if label_img is None:
            continue

        target_ids = pd.to_numeric(adata.obs.loc[fov_cells, cell_id_key], errors="coerce")
        target_ids = target_ids.dropna().astype(np.int64)
        if len(target_ids) == 0:
            continue
        target_set = set(target_ids.tolist())
        label_mask = np.isin(label_img, list(target_set))

        cid_to_wkt: dict[int, str] = {}

        # Prefer raster polygonization for pixel-accurate boundaries.
        try:
            from rasterio.features import shapes as rio_shapes

            parts: dict[int, list] = {}
            for geom, val in rio_shapes(label_img.astype(np.int32), mask=label_mask):
                cid = int(val)
                if cid == 0 or cid not in target_set:
                    continue
                try:
                    g = shape(geom)
                except Exception:
                    continue
                if g.is_empty:
                    continue
                parts.setdefault(cid, []).append(g)

            for cid, geoms in parts.items():
                if len(geoms) == 1:
                    g = geoms[0]
                else:
                    g = unary_union(geoms)
                if not g.is_valid:
                    g = g.buffer(0)
                if not g.is_empty:
                    cid_to_wkt[cid] = g.wkt
        except Exception:
            # Fallback: contour tracing from binary masks.
            for region in regionprops(label_img):
                cid = int(region.label)
                if cid == 0 or cid not in target_set:
                    continue

                minr, minc, maxr, maxc = region.bbox
                local_mask = (label_img[minr:maxr, minc:maxc] == cid).astype(np.uint8)
                contours = find_contours(local_mask, level=0.5)
                if len(contours) == 0:
                    continue
                contour = max(contours, key=lambda c: c.shape[0])
                if contour.shape[0] < 3:
                    continue

                xy = [(float(c[1] + minc), float(c[0] + minr)) for c in contour]
                poly = Polygon(xy)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
                cid_to_wkt[cid] = poly.wkt

        if not cid_to_wkt:
            continue

        # map label id -> obs rows in this fov
        fov_cell_ids = pd.to_numeric(adata.obs.loc[fov_cells, cell_id_key], errors="coerce")
        for cid, wkt_str in cid_to_wkt.items():
            hit = fov_cells[(fov_cell_ids == cid).to_numpy()]
            if len(hit) == 0:
                continue
            out.loc[hit] = wkt_str
            written += len(hit)

    if written == 0:
        return None
    return out


@register_function(
    aliases=["read_nanostring", "nanostring", "读取nanostring", "cosmx", "nanostring smi"],
    category="io",
    description="Read Nanostring Spatial Molecular Imager (SMI) output with cell-level coordinates and FOV images.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = ov.io.spatial.read_nanostring(",
        "    'sample_dir',",
        "    counts_file='sample_exprMat_file.csv',",
        "    meta_file='sample_metadata_file.csv',",
        "    fov_file='sample_fov_positions_file.csv',",
        ")",
    ],
    related=["io.spatial.read_visium", "io.spatial.read_visium_hd"],
)
def read_nanostring(
    path: Union[str, Path],
    *,
    counts_file: str,
    meta_file: str,
    fov_file: Optional[str] = None,
) -> AnnData:
    """Read Nanostring formatted dataset.

    This follows the Squidpy nanostring reader layout:

    - ``obsm['spatial']``: local cell center coordinates.
    - ``obsm['spatial_fov']``: global cell center coordinates.
    - ``uns['spatial'][fov]['images']``: hires and segmentation images from
      ``CellComposite`` and ``CellLabels``.
    - ``uns['spatial'][fov]['metadata']``: optional FOV-level metadata.
    - If geometry-like columns exist in metadata, writes ``obs['geometry']``
      as WKT strings to match ``read_visium_hd_seg`` format for
      ``ov.pl.spatialseg``.
    """
    root = Path(path).resolve()
    _progress(f"Reading Nanostring data from: {root}")

    counts_path = root / counts_file
    meta_path = root / meta_file
    if not counts_path.exists():
        raise FileNotFoundError(f"Counts file not found: {counts_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    counts = pd.read_csv(counts_path, header=0)
    counts_cell_id = _find_first_column(
        counts, ("cell_ID", "cell_id", "cellid", "CellID"), context="cell id (counts)"
    )
    counts_fov = _find_first_column(
        counts, ("fov", "FOV", "fov_id", "fovID"), context="fov (counts)"
    )
    counts = counts.set_index(counts_cell_id)
    counts.index = counts.index.astype(str).str.cat(counts.pop(counts_fov).astype(str).values, sep="_")

    obs = pd.read_csv(meta_path, header=0)
    obs_cell_id = _find_first_column(
        obs, ("cell_ID", "cell_id", "cellid", "CellID"), context="cell id (metadata)"
    )
    obs_fov = _find_first_column(
        obs, ("fov", "FOV", "fov_id", "fovID"), context="fov (metadata)"
    )
    obs = obs.set_index(obs_cell_id)
    obs[obs_fov] = pd.Categorical(obs[obs_fov].astype(str))
    # Keep original cell_ID column for segmentation label matching.
    obs["cell_ID"] = pd.to_numeric(obs.index, errors="coerce")
    obs.index = obs.index.astype(str).str.cat(obs[obs_fov].astype(str).values, sep="_")
    obs.rename_axis(None, inplace=True)

    common_index = obs.index.intersection(counts.index)
    if len(common_index) == 0:
        raise ValueError(
            "No overlapping cell IDs between counts and metadata after combining with FOV suffix."
        )

    _progress(f"Matched cells: {len(common_index)}")
    adata = AnnData(
        X=csr_matrix(counts.loc[common_index, :].values),
        obs=obs.loc[common_index, :].copy(),
        uns={"spatial": {}},
    )
    adata.var_names = counts.columns.astype(str)

    local_xy = _find_xy_columns(adata.obs, kind="local")
    if local_xy is None:
        raise ValueError(
            "Could not find local coordinate columns in metadata. "
            "Expected e.g. ['CenterX_local_px', 'CenterY_local_px']."
        )
    adata.obsm["spatial"] = adata.obs[list(local_xy)].to_numpy()
    adata.obs.drop(columns=list(local_xy), inplace=True)

    global_xy = _find_xy_columns(adata.obs, kind="global")
    if global_xy is not None:
        adata.obsm["spatial_fov"] = adata.obs[list(global_xy)].to_numpy()
    else:
        warnings.warn(
            "Global coordinate columns not found in metadata; `obsm['spatial_fov']` will not be created."
        )

    geometry_wkt = _extract_geometry_wkt_from_obs(adata.obs)
    has_geometry = geometry_wkt is not None

    fov_categories = adata.obs[obs_fov].astype("category").cat.categories.astype(str)
    for fov in fov_categories:
        adata.uns["spatial"][fov] = {
            "images": {},
            "scalefactors": {"tissue_hires_scalef": 1.0, "spot_diameter_fullres": 1.0},
        }

    _progress("Loading optional FOV images")
    from tqdm import tqdm
    file_extensions = (".jpg", ".png", ".jpeg", ".tif", ".tiff")
    fov_pattern = re.compile(r".*_F(\d+)")
    for subdir, kind in (("CellComposite", "hires"), ("CellLabels", "segmentation")):
        folder = root / subdir
        if not folder.is_dir():
            continue
        for fname in tqdm(os.listdir(folder)):
            if not fname.lower().endswith(file_extensions):
                continue
            match = fov_pattern.findall(fname)
            if len(match) == 0:
                continue
            fov = str(int(match[0]))
            if fov not in adata.uns["spatial"]:
                warnings.warn(f"FOV `{fov}` does not exist in `{subdir}`, skipping image `{fname}`.")
                continue
            adata.uns["spatial"][fov]["images"][kind] = _read_image(folder / fname)

    if has_geometry:
        adata.obs["geometry"] = geometry_wkt
        _progress(
            f"Detected cell contours in metadata (geometry WKT generated for {(geometry_wkt != '').sum()} cells)"
        )
    else:
        geometry_wkt = _geometry_from_segmentation_images(adata, fov_key=obs_fov, cell_id_key="cell_ID")
        if geometry_wkt is not None:
            adata.obs["geometry"] = geometry_wkt
            has_geometry = True
            _progress(
                f"Extracted cell contours from CellLabels images (geometry WKT generated for {(geometry_wkt != '').sum()} cells)"
            )
        else:
            _progress(
                "No geometry-like contour column found and failed to extract contours from CellLabels; "
                "skipping `obs['geometry']`.",
                level="warn",
            )

    if fov_file is not None:
        fov_path = root / fov_file
        if not fov_path.exists():
            warnings.warn(f"FOV metadata file not found: {fov_path}")
        else:
            _progress(f"Loading FOV metadata: {fov_path.name}")
            fov_positions = pd.read_csv(fov_path, header=0)
            fov_key = "fov" if "fov" in fov_positions.columns else ("FOV" if "FOV" in fov_positions.columns else None)
            if fov_key is None:
                warnings.warn("FOV metadata does not contain `fov`/`FOV` column; skipping FOV metadata merge.")
            else:
                fov_positions = fov_positions.set_index(fov_key)
                for fov, row in fov_positions.iterrows():
                    fov_str = str(fov)
                    if fov_str not in adata.uns["spatial"]:
                        warnings.warn(f"FOV `{fov_str}` does not exist in expression data, skipping.")
                        continue
                    adata.uns["spatial"][fov_str]["metadata"] = row.to_dict()

    if has_geometry:
        adata.uns["omicverse_io"] = {"type": "nanostring_seg"}
    else:
        adata.uns["omicverse_io"] = {"type": "nanostring"}
    _progress(f"Done (n_obs={adata.n_obs}, n_vars={adata.n_vars})", level="success")
    return adata


__all__ = ["read_nanostring"]
