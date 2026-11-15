r"""WSI IO and tiling — thin wrappers around :mod:`wsidata` / :mod:`lazyslide`.

All meaningful logic lives upstream; this module exists so users do not have
to learn the LazySlide API surface to use ``ov.space.histo``. When you need
the full LazySlide control plane, drop down to ``zs.pp.*`` / ``zs.tl.*``
directly on the returned ``WSIData``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..._registry import register_function

if TYPE_CHECKING:
    from anndata import AnnData
    from wsidata import WSIData


@register_function(
    aliases=["打开WSI", "open_wsi", "wsi_reader", "HE加载", "病理图加载"],
    category="space",
    description="Open a whole-slide H&E image as a wsidata.WSIData container (thin wrapper around wsidata.open_wsi). Returns a SpatialData subclass with WSI accessors that downstream ov.space.histo.* functions consume.",
    examples=[
        "wsi = ov.space.histo.open_wsi('slide.svs')",
        "wsi = ov.space.histo.open_wsi('slide.tif', reader='tiffslide')",
    ],
    related=["space.histo.tile", "space.histo.embed", "space.histo.read_visium_with_image"],
)
def open_wsi(
    path: str | Path,
    *,
    store: str = "auto",
    reader: Literal["openslide", "tiffslide", "bioformats"] | None = None,
    attach_thumbnail: bool = True,
    thumbnail_size: int = 2000,
    **kwargs,
) -> "WSIData":
    """Open a whole-slide image as a :class:`wsidata.WSIData` container.

    Parameters
    ----------
    path
        Path to a WSI (svs/tiff/ndpi/…). For convenience also accepts an
        existing :class:`spatialdata.SpatialData` zarr store path.
    store
        On-disk zarr store for cached tiles/features. ``"auto"`` derives the
        path from ``path``.
    reader
        Force a specific backend. Defaults to ``tiffslide`` if available, then
        ``openslide``.
    """
    from wsidata import open_wsi as _open_wsi

    return _open_wsi(
        path,
        store=store,
        reader=reader,
        attach_thumbnail=attach_thumbnail,
        thumbnail_size=thumbnail_size,
        **kwargs,
    )


@register_function(
    aliases=["切片", "tile", "wsi_tile", "WSI切块", "tile_wsi"],
    category="space",
    description="Tile a WSIData into a grid of patches (delegates to LazySlide find_tissues + tile_tissues). Tile geometries land in wsi.shapes[tile_key]; tile_spec in wsi.attrs['tile_spec'][tile_key].",
    examples=[
        "ov.space.histo.tile(wsi, tile_px=224, mpp=0.5)",
        "ov.space.histo.tile(wsi, tile_px=256, mpp=0.5, overlap=0.1)",
    ],
    related=["space.histo.open_wsi", "space.histo.embed", "space.histo.predict_expression"],
)
def tile(
    wsi: "WSIData",
    tile_px: int = 224,
    *,
    mpp: float | None = 0.5,
    stride_px: int | None = None,
    overlap: float | None = None,
    background_fraction: float = 0.3,
    find_tissue: bool = True,
    tissue_key: str = "tissues",
    tile_key: str = "tiles",
) -> "WSIData":
    """Tile a WSI into a grid of patches (delegates to LazySlide).

    Runs :func:`lazyslide.pp.find_tissues` followed by
    :func:`lazyslide.pp.tile_tissues`. Tile geometries land in
    ``wsi.shapes[tile_key]`` and the tile spec in
    ``wsi.attrs['tile_spec'][tile_key]``.

    Parameters
    ----------
    tile_px
        Tile size at the requested ``mpp``. ``224`` matches most pathology
        foundation models (UNI/CONCH/Virchow2/GigaPath).
    mpp
        Target microns-per-pixel. Visium spots are typically 55 µm; for
        spot-aligned prediction set ``mpp=0.5`` and ``tile_px=224`` which
        gives 112 µm × 112 µm crops centred on each spot.
    """
    import lazyslide as zs

    if find_tissue and tissue_key not in wsi.shapes:
        zs.pp.find_tissues(wsi, key_added=tissue_key)

    zs.pp.tile_tissues(
        wsi,
        tile_px=tile_px,
        stride_px=stride_px,
        overlap=overlap,
        mpp=mpp,
        background_fraction=background_fraction,
        tissue_key=tissue_key,
        key_added=tile_key,
    )
    return wsi


@register_function(
    aliases=["读取Visium带图像", "read_visium_with_image", "load_visium_he", "Visium配对加载"],
    category="space",
    description="Load a Space Ranger Visium output AND wrap its source H&E as a WSIData. Returns (adata, wsi). The two share the same physical coordinate system; downstream HE-zoo backends use adata as the paired reference and wsi for tile-level inference.",
    examples=[
        "adata, wsi = ov.space.histo.read_visium_with_image('/path/to/outs', image_path='HE.tif')",
        "adata, wsi = ov.space.histo.read_visium_with_image('outs', image_path='HE.tif', count_file='filtered_feature_bc_matrix.h5')",
    ],
    related=["space.histo.open_wsi", "space.histo.load_breast", "space.histo.tile"],
)
def read_visium_with_image(
    visium_path: str | Path,
    *,
    image_path: str | Path | None = None,
    library_id: str | None = None,
    count_file: str = "filtered_feature_bc_matrix.h5",
    source_image_path: str | Path | None = None,
) -> tuple["AnnData", "WSIData"]:
    """Load a 10x Visium output **and** wrap its source H&E as ``WSIData``.

    Returns ``(adata, wsi)`` — the AnnData carries gene counts with spot
    coordinates in ``adata.obsm['spatial']``, the WSIData wraps the
    full-resolution H&E for tiling and embedding. The two share the same
    physical coordinate system; downstream backends use ``adata`` as the
    paired reference and ``wsi`` for tile-level inference.

    Parameters
    ----------
    visium_path
        Path to a Space Ranger ``outs/`` directory.
    image_path
        Path to the full-resolution H&E (``*.tif``/``*.svs``). Required for
        any backend that needs whole-slide image access; if omitted, only
        the low-resolution ``tissue_hires_image.png`` shipped by Space
        Ranger is available.
    library_id
        Library id stored in ``adata.uns['spatial']``. Inferred when ``None``.
    count_file
        Count file inside ``outs/``.
    source_image_path
        Forwarded to :func:`omicverse.io.read_visium`.
    """
    from ...io import read_visium

    visium_path = Path(visium_path)
    adata = read_visium(
        visium_path,
        count_file=count_file,
        library_id=library_id,
        source_image_path=source_image_path,
    )
    adata.var_names_make_unique()

    wsi = None
    if image_path is not None:
        wsi = open_wsi(image_path)
        # Persist the link between AnnData and WSIData so downstream
        # backends can recover it without an extra argument.
        adata.uns.setdefault("histo", {})["wsi_path"] = str(Path(image_path))
        # Visium TIFFs almost never embed mpp metadata. Derive it from
        # the spot_diameter_fullres (a 55 µm spot in pixels) so LazySlide
        # plotting (scalebar) and the prediction backends work out of the
        # box.
        if wsi.properties.mpp is None and adata.uns.get("spatial"):
            lib = next(iter(adata.uns["spatial"]))
            sf = adata.uns["spatial"][lib].get("scalefactors", {})
            spot_px = sf.get("spot_diameter_fullres")
            if spot_px:
                wsi.set_mpp(55.0 / float(spot_px))
    return adata, wsi


def spot_geodataframe(adata: "AnnData", *, spot_diameter: float | None = None):
    """Convert Visium spot coordinates to a GeoPandas frame matching LazySlide.

    LazySlide stores tile geometries as a :class:`geopandas.GeoDataFrame`
    of square polygons in slide pixels. ``spot_geodataframe`` builds the
    equivalent frame from ``adata.obsm['spatial']`` so that Visium spots and
    HE→ST predictions live in the same spatial reference.
    """
    import geopandas as gpd
    from shapely.geometry import box

    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] missing — not a Visium AnnData.")

    library_id = next(iter(adata.uns["spatial"]))
    sf = adata.uns["spatial"][library_id]["scalefactors"]
    if spot_diameter is None:
        spot_diameter = sf.get("spot_diameter_fullres", 100.0)
    r = float(spot_diameter) / 2.0

    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    geoms = [box(x - r, y - r, x + r, y + r) for x, y in coords]
    return gpd.GeoDataFrame(
        {"barcode": adata.obs_names.values},
        geometry=geoms,
        crs=None,
    )
