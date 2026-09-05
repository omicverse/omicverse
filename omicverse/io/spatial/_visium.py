"""
Data reading functions for OmicVerse.

This module provides functions for reading Visium (standard) spatial
transcriptomics data from Space Ranger output directories.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from PIL import Image

from ..._registry import register_function
from ..single import read_10x_h5

if TYPE_CHECKING:
    from os import PathLike

try:
    from ..._settings import Colors
except Exception:
    class Colors:
        """Fallback ANSI color codes when omicverse._settings import is unavailable."""
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'


def _progress(message: str, level: str = "info") -> None:
    color = Colors.CYAN
    if level == "success":
        color = Colors.GREEN
    elif level == "warn":
        color = Colors.WARNING
    print(f"{color}[Visium] {message}{Colors.ENDC}")


def _read_table_with_auto_sep(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.name == "tissue_positions_list.csv":
        return pd.read_csv(
            path,
            header=None,
            names=[
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ],
        )
    try:
        return pd.read_csv(path, sep=",")
    except Exception:
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")


def _read_spatial_images(
    root: Path,
    hires_image_path: str,
    lowres_image_path: str,
):
    try:
        with Image.open(root / hires_image_path) as hires_img:
            hires = np.asarray(hires_img)
        with Image.open(root / lowres_image_path) as lowres_img:
            lowres = np.asarray(lowres_img)
        return hires, lowres
    except FileNotFoundError as exc:
        warnings.warn(f"Could not load tissue images: {exc}")
        return None, None


def _read_scalefactors(root: Path, scalefactors_path: str) -> dict:
    try:
        with open(root / scalefactors_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        warnings.warn(f"Could not load scalefactors: {exc}")
        return {}


def _init_spatial_slot(
    adata: AnnData,
    sample: str,
    hires_img,
    lowres_img,
    scalefactors: dict,
) -> None:
    adata.uns.setdefault("spatial", {})
    adata.uns["spatial"][sample] = {}
    if hires_img is not None and lowres_img is not None:
        adata.uns["spatial"][sample]["images"] = {
            "hires": hires_img,
            "lowres": lowres_img,
        }
    adata.uns["spatial"][sample]["scalefactors"] = scalefactors


def _resolve_tissue_positions(spatial_dir: Path) -> Path:
    """Return the first existing tissue positions file (parquet > csv > legacy csv)."""
    for candidate in (
        spatial_dir / "tissue_positions.parquet",
        spatial_dir / "tissue_positions.csv",
        spatial_dir / "tissue_positions_list.csv",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No tissue positions file found under {spatial_dir}. "
        "Expected tissue_positions.parquet, tissue_positions.csv, or "
        "tissue_positions_list.csv."
    )


@register_function(
    aliases=["read_visium", "visium reader", "读取visium", "10x visium", "spaceranger reader"],
    category="io",
    description="Read 10x Genomics Visium spatial transcriptomics dataset from Space Ranger output directory.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "adata = ov.io.spatial.read_visium('outs/')",
        "adata = ov.io.spatial.read_visium('outs/', count_file='raw_feature_bc_matrix.h5')",
        "adata = ov.io.spatial.read_visium('outs/', library_id='sample_A')",
        "adata = ov.io.spatial.read_visium('outs/', load_images=False)",
    ],
    related=["io.spatial.read_visium_hd", "io.single.read_10x_h5", "io.single.read_10x_mtx"],
)
def read_visium(
    path: Union[str, PathLike],
    genome: Optional[str] = None,
    *,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: Optional[str] = None,
    load_images: bool = True,
    source_image_path: Optional[Union[str, PathLike]] = None,
    hires_image_path: str = "spatial/tissue_hires_image.png",
    lowres_image_path: str = "spatial/tissue_lowres_image.png",
    scalefactors_path: str = "spatial/scalefactors_json.json",
) -> AnnData:
    r"""Read 10x-Genomics-formatted Visium dataset.

    In addition to reading regular 10x output, this looks for the ``spatial``
    folder and loads images, coordinates and scale factors. Based on the
    `Space Ranger output docs <https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview>`_.

    Arguments:
        path: Path to the Space Ranger output directory (typically ``outs/``).
        genome: Filter expression to genes within this genome.
        count_file: Count matrix filename inside *path*. Typically
            ``'filtered_feature_bc_matrix.h5'`` or ``'raw_feature_bc_matrix.h5'``.
        library_id: Identifier stored under ``adata.uns['spatial']``. Inferred
            from the HDF5 ``library_ids`` attribute when not provided.
        load_images: Whether to load tissue images, scale factors and spatial
            coordinates.
        source_image_path: Optional path to the full-resolution source image.
            Stored in ``adata.uns['spatial'][library_id]['metadata']['source_image_path']``.
        hires_image_path: Relative path to the hires tissue image inside *path*.
        lowres_image_path: Relative path to the lowres tissue image inside *path*.
        scalefactors_path: Relative path to the scalefactors JSON inside *path*.

    Returns:
        adata: Annotated data matrix where observations/cells are named by their
            barcode and variables/genes by gene name.

            - **X** – count matrix
            - **obs_names** – barcode names
            - **var_names** – gene/probe names
            - **var['gene_ids']** – gene IDs
            - **var['feature_types']** – feature types
            - **uns['spatial'][library_id]['images']** – ``{'hires': ndarray, 'lowres': ndarray}``
            - **uns['spatial'][library_id]['scalefactors']** – parsed scalefactors JSON
            - **uns['spatial'][library_id]['metadata']** – chemistry/version info
            - **obsm['spatial']** – spot pixel coordinates (x=column, y=row in full-res image)

    Examples:
        >>> import omicverse as ov
        >>> adata = ov.io.spatial.read_visium("outs/")
        >>> adata = ov.io.spatial.read_visium("outs/", count_file="raw_feature_bc_matrix.h5")
    """
    root = Path(path).resolve()
    _progress(f"Reading Visium data from: {root}")

    h5_path = root / count_file
    if not h5_path.exists():
        raise FileNotFoundError(f"Count file not found: {h5_path}")
    _progress(f"Loading count matrix: {count_file}")
    adata = read_10x_h5(h5_path, genome=genome)

    # Resolve library_id from HDF5 attributes
    if library_id is None:
        with h5py.File(h5_path, mode="r") as f:
            attrs = dict(f.attrs)
        raw = attrs.get("library_ids")
        if raw is not None:
            library_id = str(raw[0], "utf-8") if isinstance(raw[0], bytes) else str(raw[0])
        else:
            library_id = root.name
    _progress(f"Library ID: {library_id}")

    adata.uns["spatial"] = {}

    if load_images:
        spatial_dir = root / "spatial"

        # Tissue positions (parquet > csv > legacy csv)
        tissue_positions_file = _resolve_tissue_positions(spatial_dir)
        _progress(f"Loading tissue positions: {tissue_positions_file.name}")
        tissue_df = _read_table_with_auto_sep(tissue_positions_file)

        # Normalise index to barcode
        if "barcode" in tissue_df.columns:
            tissue_df = tissue_df.set_index("barcode")
        elif tissue_df.index.name != "barcode" and len(tissue_df.columns) > 0:
            tissue_df = tissue_df.set_index(tissue_df.columns[0])

        expected_position_columns = {
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        }
        missing_position_columns = expected_position_columns.difference(tissue_df.columns)
        if missing_position_columns:
            raise ValueError(
                f"Tissue positions file {tissue_positions_file} is missing required "
                f"columns: {sorted(missing_position_columns)}."
            )

        adata.obs = pd.merge(
            adata.obs, tissue_df, left_index=True, right_index=True, how="left"
        )

        adata.obsm["spatial"] = adata.obs[
            ["pxl_col_in_fullres", "pxl_row_in_fullres"]
        ].to_numpy()

        # Images and scale factors
        _progress("Loading images and scale factors")
        hires_img, lowres_img = _read_spatial_images(root, hires_image_path, lowres_image_path)
        scalefactors = _read_scalefactors(root, scalefactors_path)
        _init_spatial_slot(adata, library_id, hires_img, lowres_img, scalefactors)

        # Metadata from h5 attributes
        with h5py.File(h5_path, mode="r") as f:
            attrs = dict(f.attrs)
        adata.uns["spatial"][library_id]["metadata"] = {
            k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
            for k in ("chemistry_description", "software_version")
            if k in attrs
        }
        if source_image_path is not None:
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                Path(source_image_path).resolve()
            )

    _progress(f"Done (n_obs={adata.n_obs}, n_vars={adata.n_vars})", level="success")
    return adata
