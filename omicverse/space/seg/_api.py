r"""Public entry points of ``ov.space.seg`` for images."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ..._registry import register_function
from ._backends import BACKENDS, _sample
from ._tiling import segment_tiled

ImageMethod = Literal["watershed", "cellpose", "stardist", "sam", "sam2", "micro_sam", "cellsam"]

_METHOD_TABLE = [
    # name, input, needs, gpu, reference
    ("watershed", "image", "scikit-image (bundled)", "no",
     "Spateo-style marker watershed; classical baseline"),
    ("cellpose", "image", "cellpose>=2 (v3 restore, v4 Cellpose-SAM)", "recommended",
     "Stringer & Pachitariu, Nat. Methods 2025"),
    ("stardist", "image", "stardist + tensorflow", "optional",
     "Schmidt et al., MICCAI 2018"),
    ("sam", "image", "segment-anything", "recommended",
     "Kirillov et al., ICCV 2023"),
    ("sam2", "image", "sam2 + huggingface_hub (py>=3.10)", "recommended",
     "Ravi et al., ICLR 2025"),
    ("micro_sam", "image", "micro_sam (conda-forge)", "recommended",
     "Archit et al., Nat. Methods 2025"),
    ("cellsam", "image", "cellSAM (git) + DEEPCELL_ACCESS_TOKEN", "recommended",
     "Israel et al., bioRxiv 2023"),
    ("baysor", "transcripts (+ prior)", "Baysor CLI on PATH", "no",
     "Petukhov et al., Nat. Biotechnol. 2022"),
]


@register_function(
    aliases=["分割方法列表", "seg_methods", "segmentation methods", "细胞分割方法"],
    category="space",
    description="List the cell segmentation methods available through ov.space.seg, their input type, dependencies and whether each is installed.",
    examples=["ov.space.seg.methods()"],
    related=["space.seg.segment", "space.seg.segment_transcripts"],
)
def methods() -> pd.DataFrame:
    """Segmentation methods behind ``ov.space.seg``.

    Returns
    -------
    pandas.DataFrame
        One row per method: ``input`` (image or transcripts), ``needs``
        (what to install), ``gpu``, ``installed`` (importable / on PATH now)
        and a literature ``reference``.
    """
    import importlib.util
    import shutil

    rows = []
    for name, kind, needs, gpu, ref in _METHOD_TABLE:
        if name == "baysor":
            installed = shutil.which("baysor") is not None
        else:
            mods = BACKENDS[name].requires
            installed = all(importlib.util.find_spec(m.split(".")[0]) is not None for m in mods)
        rows.append({"method": name, "input": kind, "needs": needs, "gpu": gpu,
                     "installed": installed, "reference": ref})
    return pd.DataFrame(rows).set_index("method")


def load_image(path, *, level: int = 0) -> np.ndarray:
    """Read a TIFF/OME-TIFF (pyramid ``level``), PNG or JPEG into memory."""
    path = str(path)
    if path.lower().endswith((".tif", ".tiff", ".btf")):
        import tifffile
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            if level and len(series.levels) > level:
                return series.levels[level].asarray()
            return series.asarray()
    from skimage.io import imread
    return imread(path)


def _channels_last(image: np.ndarray) -> np.ndarray:
    """Move a leading channel axis ``(C, H, W)`` (C <= 4) to the end."""
    if image.ndim == 3 and image.shape[0] <= 4 and image.shape[-1] > 4:
        return np.moveaxis(image, 0, -1)
    return image


@register_function(
    aliases=["细胞分割", "segment", "cell segmentation", "核分割", "图像分割",
             "cellpose", "stardist", "SAM分割", "sam2", "micro_sam", "watershed"],
    category="space",
    description="Segment cells or nuclei in a DAPI/fluorescence or H&E image of any size with one of several backends (watershed, cellpose incl. Cellpose 3 restoration and Cellpose-SAM, stardist, SAM, SAM2, micro-SAM, CellSAM); tiled inference with seam-free stitching, returns an integer label image.",
    examples=[
        "labels = ov.space.seg.segment(dapi, method='cellpose', diameter=30, gpu=True)",
        "labels = ov.space.seg.segment(dapi, method='cellpose', model='cyto3', restore_type='denoise_cyto3')",
        "labels = ov.space.seg.segment(he_rgb, method='stardist', image_type='he')",
        "labels = ov.space.seg.segment('morphology_focus.ome.tif', method='sam2', tile_size=1024)",
        "labels = ov.space.seg.segment(dapi, method='watershed', diameter=20)",
    ],
    related=["space.seg.assign_transcripts", "space.seg.aggregate", "space.seg.evaluate",
             "space.seg.methods", "space.visium_10x_hd_cellpose_he"],
)
def segment(
    image,
    method: ImageMethod = "cellpose",
    *,
    image_type: Literal["auto", "fluorescence", "he"] = "auto",
    channel: int | None = None,
    diameter: float | None = None,
    gpu: bool | str = False,
    tile_size: int | None = None,
    context: int | None = None,
    min_size: int = 0,
    expand: float = 0,
    sparse: bool = False,
    level: int = 0,
    show_progress: bool = True,
    **method_kwargs,
):
    """Segment cells or nuclei in a 2-D image.

    Large images are segmented tile by tile: every tile is processed with a
    ``context`` margin and keeps the objects whose centroid it owns, so the
    result has no seams and no duplicated cells. Intensity scaling is fixed
    once from the whole image, so tiles agree with each other.

    Parameters
    ----------
    image
        ``(H, W)`` or ``(H, W, C)`` array, or a path to a TIFF/OME-TIFF,
        PNG or JPEG. A leading channel axis ``(C, H, W)`` is moved last.
    method
        ``'watershed'`` (no extra dependency), ``'cellpose'``,
        ``'stardist'``, ``'sam'``, ``'sam2'``, ``'micro_sam'`` or
        ``'cellsam'``. See :func:`ov.space.seg.methods`.
    image_type
        ``'fluorescence'`` (bright objects on dark), ``'he'`` (RGB H&E;
        the haematoxylin channel is the nuclear signal for single-channel
        models) or ``'auto'`` (3/4-channel images are H&E).
    channel
        Channel of a multi-channel fluorescence image to segment. ``None``
        averages channels.
    diameter
        Expected object diameter in pixels. Sets Cellpose's ``diameter``,
        the watershed scales, the SAM-family area filters and the default
        ``context``.
    gpu
        ``True`` for CUDA (or Apple MPS), ``False`` for CPU, or a torch
        device string.
    tile_size, context
        Tile size and margin in pixels. Defaults depend on the method
        (1024 for the SAM family, 2048 otherwise); ``context`` defaults to
        ``max(64, 2 * diameter)``.
    min_size
        Drop objects of at most this many pixels.
    expand
        Grow each object by this many pixels without merging neighbours
        (nucleus -> cell approximation, as in Xenium).
    sparse
        Return a ``scipy.sparse.csr_matrix`` (for whole-slide images).
    level
        Pyramid level when ``image`` is an OME-TIFF path.
    **method_kwargs
        Method options, e.g. ``model``, ``flow_threshold``,
        ``cellprob_threshold``, ``restore_type`` (cellpose);
        ``prob_thresh``, ``nms_thresh`` (stardist); ``points_per_side``,
        ``pred_iou_thresh``, ``stability_score_thresh``, ``checkpoint``,
        ``max_area`` (SAM family); ``otsu_classes``, ``otsu_index``,
        ``sigma``, ``min_distance`` (watershed).

    Returns
    -------
    numpy.ndarray or scipy.sparse.csr_matrix
        ``int32`` label image of the input's height and width, 0 = background.
    """
    if method not in BACKENDS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(BACKENDS)}. "
                         "Transcript-based methods are in ov.space.seg.segment_transcripts.")
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        image = load_image(image, level=level)
    image = _channels_last(np.asarray(image) if not hasattr(image, "shape") else image)
    if image.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D image (H, W) or (H, W, C), got shape {image.shape}")

    cls = BACKENDS[method]
    backend = cls(gpu=gpu, image_type=image_type, channel=channel, diameter=diameter,
                  **method_kwargs)
    backend.prepare(_sample(image))
    tile_size = int(tile_size or cls.default_tile_size)
    if context is None:
        context = int(max(64, 2 * diameter)) if diameter else 64
    labels = segment_tiled(image, backend, tile_size=tile_size, context=context,
                           min_size=min_size, sparse=sparse and not expand,
                           show_progress=show_progress, desc=f"seg[{method}]")
    if expand:
        labels = expand_labels(labels, expand)
        if sparse:
            from scipy import sparse as sp
            labels = sp.csr_matrix(labels)
    return labels


@register_function(
    aliases=["扩展标签", "expand_labels", "核扩展为细胞", "nucleus expansion"],
    category="space",
    description="Grow segmented nuclei by a fixed distance into cell approximations without merging neighbouring objects.",
    examples=["cells = ov.space.seg.expand_labels(nuclei, distance=10)"],
    related=["space.seg.segment"],
)
def expand_labels(labels, distance: float):
    """Grow every object by ``distance`` pixels; contested pixels go to the nearest.

    Parameters
    ----------
    labels
        Integer label image (dense or ``scipy.sparse``).
    distance
        Expansion distance in pixels.

    Returns
    -------
    numpy.ndarray
        Expanded ``int32`` label image.
    """
    from skimage.segmentation import expand_labels as _expand
    if hasattr(labels, "toarray"):
        labels = labels.toarray()
    return _expand(np.asarray(labels), distance=distance).astype(np.int32)
