r"""Image segmentation backends for ``ov.space.seg.segment``.

A backend is built once per call, sees a downsampled copy of the whole image in
:meth:`Backend.prepare` (so intensity scaling is global, not per tile), and is
then called on each window by the tiling engine. Heavy libraries are imported
in ``_build`` only, the first time a window is segmented.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ..._optional import build_optional_dependency_error


def _cache_dir() -> Path:
    root = os.environ.get("OV_SEG_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "omicverse", "seg"
    )
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _torch_device(gpu, *, mps: bool = True) -> str:
    """``'cuda'`` / ``'mps'`` / ``'cpu'`` from a bool or an explicit device string.

    ``mps=False`` skips Apple MPS when picking automatically, for models whose
    post-processing needs float64 tensors (unsupported on MPS).
    """
    if isinstance(gpu, str):
        return gpu
    if not gpu:
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if mps and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sample(image, max_pixels: int = 4_000_000) -> np.ndarray:
    """A strided view of ``image`` with at most ``max_pixels`` pixels."""
    H, W = int(image.shape[0]), int(image.shape[1])
    step = max(1, int(np.ceil(np.sqrt(H * W / max_pixels))))
    return np.asarray(image[::step, ::step])


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


class Backend:
    """Base class: intensity handling shared by every image backend.

    ``image_type`` decides what "signal" means. ``'fluorescence'`` -- bright
    objects on a dark background (DAPI, a stain channel, an expression image);
    ``'he'`` -- an RGB brightfield H&E image, whose nuclear signal is the
    haematoxylin channel; ``'auto'`` -- ``'he'`` for 3-/4-channel images,
    otherwise ``'fluorescence'``.
    """

    name = "base"
    #: modules imported by ``_build``; reported in the missing-dependency error
    requires: tuple = ()
    install = ""
    default_tile_size = 2048

    def __init__(self, *, gpu=False, image_type="auto", channel=None,
                 diameter=None, percentiles=(1.0, 99.8)):
        if image_type not in ("auto", "fluorescence", "he"):
            raise ValueError("image_type must be 'auto', 'fluorescence' or 'he'")
        self.gpu = gpu
        self.image_type = image_type
        self.channel = channel
        self.diameter = diameter
        self.percentiles = tuple(percentiles)
        self._model = None
        self._lo, self._hi = 0.0, 1.0            # nuclear signal
        self._gray_lo, self._gray_hi = 0.0, 1.0  # luminance of an RGB image
        self._ch_lo = self._ch_hi = None         # per RGB channel
        self._rgb_lo, self._rgb_hi = 0.0, 255.0  # whole RGB range

    # -- intensity ---------------------------------------------------------
    def _resolve_type(self, image) -> None:
        if self.image_type == "auto":
            is_rgb = image.ndim == 3 and image.shape[-1] in (3, 4) and self.channel is None
            self.image_type = "he" if is_rgb else "fluorescence"
        if self.image_type == "he" and not (image.ndim == 3 and image.shape[-1] >= 3):
            raise ValueError("image_type='he' expects an (H, W, 3) RGB image")

    def _raw_signal(self, tile: np.ndarray) -> np.ndarray:
        """Nuclear/foreground signal, unscaled, one channel."""
        if self.image_type == "he":
            from skimage.color import rgb2hed
            rgb = tile[..., :3]
            if rgb.dtype.kind == "f" and rgb.max() > 1.0:
                rgb = rgb / 255.0
            return rgb2hed(rgb)[..., 0]
        if tile.ndim == 3:
            if self.channel is not None:
                return tile[..., self.channel].astype(np.float32)
            return tile.mean(axis=-1, dtype=np.float32)
        return tile.astype(np.float32, copy=False)

    def prepare(self, sample: np.ndarray) -> None:
        """Fix the global intensity scale from a downsampled whole image."""
        self._resolve_type(sample)
        sig = self._raw_signal(sample)
        lo, hi = np.percentile(sig, self.percentiles)
        self._lo, self._hi = float(lo), float(max(hi, lo + 1e-12))
        if sample.ndim == 3 and sample.shape[-1] >= 3:
            rgb = sample[..., :3].astype(np.float32)
            lo, hi = np.percentile(rgb, (0.0, 100.0))
            self._rgb_lo, self._rgb_hi = float(lo), float(max(hi, lo + 1e-12))
            lo, hi = np.percentile(rgb.reshape(-1, 3), self.percentiles, axis=0)
            self._ch_lo, self._ch_hi = lo, np.maximum(hi, lo + 1e-12)
            lo, hi = np.percentile(self._luminance(rgb), self.percentiles)
            self._gray_lo, self._gray_hi = float(lo), float(max(hi, lo + 1e-12))

    @staticmethod
    def _luminance(rgb: np.ndarray) -> np.ndarray:
        return rgb[..., :3].astype(np.float32) @ np.array([0.2125, 0.7154, 0.0721], np.float32)

    # Percentile stretches map the global low/high percentiles to 0/1 and do not
    # clip, like csbdeep.normalize (StarDist) and normalize99 (Cellpose).
    def signal(self, tile: np.ndarray) -> np.ndarray:
        """Foreground signal stretched with the global percentiles (~``[0, 1]``)."""
        sig = (self._raw_signal(tile) - self._lo) / (self._hi - self._lo)
        return sig.astype(np.float32)

    def gray(self, tile: np.ndarray) -> np.ndarray:
        """Luminance of an RGB tile, stretched with the global percentiles."""
        g = (self._luminance(tile) - self._gray_lo) / (self._gray_hi - self._gray_lo)
        return g.astype(np.float32)

    def rgb_norm(self, tile: np.ndarray, *, joint: bool = False) -> np.ndarray:
        """RGB tile stretched per channel (or jointly) with the global percentiles."""
        rgb = tile[..., :3].astype(np.float32)
        if joint:
            lo, hi = np.min(self._ch_lo), np.max(self._ch_hi)
        else:
            lo, hi = self._ch_lo, self._ch_hi
        return ((rgb - lo) / (hi - lo)).astype(np.float32)

    def signal8(self, tile: np.ndarray) -> np.ndarray:
        """:meth:`signal` as ``uint8``."""
        return (np.clip(self.signal(tile), 0.0, 1.0) * 255).astype(np.uint8)

    def rgb8(self, tile: np.ndarray) -> np.ndarray:
        """``uint8`` RGB for the natural-image models (SAM family, CellSAM)."""
        if self.image_type == "he":
            rgb = tile[..., :3].astype(np.float32)
            if tile.dtype == np.uint8:
                return np.ascontiguousarray(tile[..., :3])
            rgb = (rgb - self._rgb_lo) / (self._rgb_hi - self._rgb_lo)
            return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return np.repeat(self.signal8(tile)[..., None], 3, axis=-1)

    # -- model -------------------------------------------------------------
    def _build(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def model(self):
        if self._model is None:
            try:
                self._model = self._build()
            except ImportError as exc:
                missing = getattr(exc, "name", None)
                if missing and missing.split(".")[0] not in {r.split(".")[0] for r in self.requires}:
                    raise
                raise build_optional_dependency_error(
                    f"ov.space.seg.segment(method={self.name!r})",
                    self.requires,
                    install_hint=self.install,
                ) from exc
        return self._model

    def __call__(self, tile: np.ndarray) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError


# ---------------------------------------------------------------------------
# classical
# ---------------------------------------------------------------------------
class WatershedBackend(Backend):
    """Marker-controlled watershed -- the classical baseline.

    Follows the Spateo pipeline benchmarked by Martinez-de-Morentin et al.
    (Nat. Methods 2026): a global multi-Otsu threshold plus a local adaptive
    threshold give the foreground, Gaussian-smoothed intensity maxima are the
    markers, and the watershed of the inverted smoothed image splits touching
    objects. Needs only scikit-image.
    """

    name = "watershed"
    requires = ("skimage",)

    def __init__(self, *, otsu_classes=4, otsu_index=1, sigma=None, min_distance=None,
                 local_block=None, local_offset=0.02, min_size=None, compactness=0.0,
                 **kw):
        super().__init__(**kw)
        self.otsu_classes = otsu_classes
        self.otsu_index = otsu_index
        d = self.diameter
        self.sigma = sigma if sigma is not None else (max(1.0, d / 6.0) if d else 2.0)
        self.min_distance = (min_distance if min_distance is not None
                             else (max(2, int(d / 3)) if d else 4))
        self.local_block = local_block if local_block is not None else (_odd(4 * d) if d else 55)
        self.local_offset = local_offset
        self.min_size = (min_size if min_size is not None
                         else (int(0.15 * np.pi * (d / 2) ** 2) if d else 16))
        self.compactness = compactness
        self._threshold = None

    def prepare(self, sample):
        super().prepare(sample)
        from skimage.filters import gaussian, threshold_multiotsu
        smooth = gaussian(np.clip(self.signal(sample), 0.0, 1.0), sigma=1.0)
        try:
            thresholds = threshold_multiotsu(smooth, classes=self.otsu_classes)
            self._threshold = float(thresholds[min(self.otsu_index, len(thresholds) - 1)])
        except ValueError:  # flat image: fewer grey levels than classes
            self._threshold = float(smooth.mean())

    def _build(self):
        import skimage  # noqa: F401
        return True

    def __call__(self, tile):
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.filters import gaussian, threshold_local
        from skimage.segmentation import watershed

        self.model()
        img = np.clip(self.signal(tile), 0.0, 1.0)
        smooth = gaussian(img, sigma=self.sigma)
        fg = smooth > self._threshold
        if self.local_block and min(img.shape) > self.local_block:
            fg &= img > threshold_local(img, self.local_block, method="gaussian",
                                        offset=self.local_offset)
        cross = ndi.generate_binary_structure(2, 1)
        fg = ndi.binary_opening(ndi.binary_closing(fg, cross), cross)
        fg = ndi.binary_fill_holes(fg)
        components, _ = ndi.label(fg)
        small = np.bincount(components.ravel()) < self.min_size
        small[0] = False
        fg[small[components]] = False
        components[~fg] = 0
        if not fg.any():
            return np.zeros(img.shape, dtype=np.int32)
        peaks = peak_local_max(smooth, min_distance=self.min_distance, labels=components,
                               exclude_border=False)
        markers = np.zeros(img.shape, dtype=np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        # components without a peak (flat tops) still become one object
        seeded = np.unique(components[markers > 0])
        orphan = np.setdiff1d(np.unique(components[components > 0]), seeded)
        if orphan.size:
            extra = np.isin(components, orphan)
            markers[extra] = components[extra] + len(peaks)
        labels = watershed(-smooth, markers, mask=fg, compactness=self.compactness)
        return labels.astype(np.int32)


# ---------------------------------------------------------------------------
# deep learning, image only
# ---------------------------------------------------------------------------
def _call_with_supported(fn, args: tuple, kwargs: dict):
    """Call ``fn`` with only the keyword arguments its signature accepts."""
    import inspect
    params = inspect.signature(fn).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


class CellposeBackend(Backend):
    """Cellpose (v2, v3 with image restoration, v4 Cellpose-SAM).

    ``model`` defaults to ``'cpsam'`` on Cellpose 4 and ``'cyto3'`` on 3
    (``'cyto'`` before). ``restore_type='denoise_cyto3'`` runs Cellpose 3's
    denoising model first, as in the Nature Methods 2026 benchmark. Images are
    normalised once globally and passed with ``normalize=False``.
    """

    name = "cellpose"
    requires = ("cellpose",)
    install = "Install with `pip install cellpose` (or `pip install 'omicverse[seg]'`)."

    _V3_ONLY = {"cyto", "cyto2", "cyto3", "nuclei", "tissuenet_cp3", "livecell_cp3",
                "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3",
                "deepbacs_cp3", "cyto2_cp3"}

    def __init__(self, *, model=None, restore_type=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, batch_size=None, **kw):
        eval_keys = ("niter", "min_size", "stitch_threshold", "augment", "tile_overlap",
                     "bsize", "max_size_fraction", "resample", "normalize", "invert")
        self.eval_kwargs = {k: kw.pop(k) for k in list(kw) if k in eval_keys}
        super().__init__(**kw)
        self.model_name = model
        self.restore_type = restore_type
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.batch_size = batch_size
        self._major = None

    def _build(self):
        from importlib.metadata import version

        from cellpose import models
        try:
            self._major = int(version("cellpose").split(".")[0])
        except Exception:  # pragma: no cover - editable installs without metadata
            self._major = 4 if not hasattr(models, "Cellpose") else 3
        device_gpu = _torch_device(self.gpu) != "cpu"
        if self._major >= 4:
            if self.restore_type:
                raise ValueError("restore_type needs Cellpose 3; Cellpose 4 (Cellpose-SAM) "
                                 "is trained to be robust to noise and has no restore models")
            name = self.model_name or "cpsam"
            if name in self._V3_ONLY:
                warnings.warn(f"Cellpose 4 has no {name!r} model; using Cellpose-SAM ('cpsam').",
                              stacklevel=3)
                name = "cpsam"
            return models.CellposeModel(gpu=device_gpu, pretrained_model=name)
        name = self.model_name or ("cyto3" if self._major == 3 else "cyto")
        if self.restore_type:
            from cellpose import denoise
            return denoise.CellposeDenoiseModel(gpu=device_gpu, model_type=name,
                                                restore_type=self.restore_type)
        if name in self._V3_ONLY:
            return models.CellposeModel(gpu=device_gpu, model_type=name)
        return models.CellposeModel(gpu=device_gpu, pretrained_model=name)

    def __call__(self, tile):
        model = self.model()
        if self._major >= 4 and self.image_type == "he":
            x = self.rgb_norm(tile)
            channel_kw = {"channel_axis": -1}
        else:
            # Cellpose <= 3 on H&E: grayscale, as the original bin2cell path did
            x = self.gray(tile) if self.image_type == "he" else self.signal(tile)
            channel_kw = {"channel_axis": None, "channels": [0, 0]}
        if min(x.shape[:2]) < 16:
            return np.zeros(x.shape[:2], dtype=np.int32)
        kwargs = {"diameter": self.diameter, "flow_threshold": self.flow_threshold,
                  "cellprob_threshold": self.cellprob_threshold, "normalize": False,
                  **channel_kw, **self.eval_kwargs}
        if self.batch_size is not None:
            kwargs["batch_size"] = self.batch_size
        out = _call_with_supported(model.eval, (x,), kwargs)
        masks = out[0] if isinstance(out, tuple) else out
        return np.asarray(masks, dtype=np.int32)


class StarDistBackend(Backend):
    """StarDist star-convex polygons (``2D_versatile_fluo`` / ``2D_versatile_he``).

    The model follows ``image_type`` unless given. StarDist runs on
    TensorFlow, which picks the GPU itself; ``gpu`` is ignored.
    """

    name = "stardist"
    requires = ("stardist", "tensorflow")
    install = "Install with `pip install stardist tensorflow`."
    default_tile_size = 2048

    def __init__(self, *, model=None, prob_thresh=None, nms_thresh=None, n_tiles=None, **kw):
        super().__init__(**kw)
        self.model_name = model
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.n_tiles = n_tiles

    def _build(self):
        from stardist.models import StarDist2D
        name = self.model_name or ("2D_versatile_he" if self.image_type == "he"
                                   else "2D_versatile_fluo")
        if os.path.isdir(str(name)):
            p = Path(name)
            return StarDist2D(None, name=p.name, basedir=str(p.parent))
        return StarDist2D.from_pretrained(name)

    def __call__(self, tile):
        model = self.model()
        if model.config.n_channel_in == 3:
            x = self.rgb_norm(tile, joint=True)  # StarDist's normalize(img, 1, 99.8, axis=(0,1,2))
        else:
            x = self.signal(tile)
        if min(x.shape[:2]) < 16:
            return np.zeros(x.shape[:2], dtype=np.int32)
        kwargs = {"prob_thresh": self.prob_thresh, "nms_thresh": self.nms_thresh,
                  "show_tile_progress": False, "verbose": False}
        if self.n_tiles is not None:
            kwargs["n_tiles"] = self.n_tiles
        labels, _ = _call_with_supported(model.predict_instances, (x,), kwargs)
        return np.asarray(labels, dtype=np.int32)


def _masks_to_labels(masks, *, min_area=0, max_area=None) -> np.ndarray | None:
    """Paint SAM-style mask records into one label image, small over large.

    Masks larger than ``max_area`` (tissue, background) are dropped; where
    masks overlap the smaller one wins, which keeps a nucleus that SAM also
    returned as part of a bigger blob.
    """
    if not masks:
        return None
    shape = np.asarray(masks[0]["segmentation"]).shape
    out = np.zeros(shape, dtype=np.int32)
    kept = [m for m in masks
            if m.get("area", int(np.count_nonzero(m["segmentation"]))) >= min_area
            and (max_area is None or m.get("area", 0) <= max_area)]
    kept.sort(key=lambda m: m.get("area", 0), reverse=True)
    for i, m in enumerate(kept, start=1):
        out[np.asarray(m["segmentation"], dtype=bool)] = i
    if not kept:
        return out
    # painting can leave a large mask as scattered fragments: renumber what survives
    ids, out_flat = np.unique(out, return_inverse=True)
    out = out_flat.reshape(shape).astype(np.int32)
    if ids[0] != 0:
        out += 1
    if min_area:
        area = np.bincount(out.ravel())
        out[area[out] < min_area] = 0
        _, out_flat = np.unique(out, return_inverse=True)
        out = out_flat.reshape(shape).astype(np.int32)
    return out


class _AMGBackend(Backend):
    """Shared logic for automatic-mask-generator models (SAM, SAM 2)."""

    default_tile_size = 1024

    def __init__(self, *, points_per_side=32, pred_iou_thresh=0.88,
                 stability_score_thresh=0.95, points_per_batch=64, crop_n_layers=0,
                 min_area=None, max_area=None, amg_kwargs=None, **kw):
        super().__init__(**kw)
        self.amg_kwargs = dict(points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh,
                               stability_score_thresh=stability_score_thresh,
                               points_per_batch=points_per_batch, crop_n_layers=crop_n_layers,
                               **(amg_kwargs or {}))
        d = self.diameter
        self.min_area = min_area if min_area is not None else (
            int(0.1 * np.pi * (d / 2) ** 2) if d else 10)
        self.max_area = max_area if max_area is not None else (
            int(4 * np.pi * (d / 2) ** 2) if d else None)

    def __call__(self, tile):
        amg = self.model()
        rgb = self.rgb8(tile)
        if min(rgb.shape[:2]) < 16:
            return np.zeros(rgb.shape[:2], dtype=np.int32)
        masks = amg.generate(rgb)
        max_area = self.max_area or int(0.02 * rgb.shape[0] * rgb.shape[1])
        out = _masks_to_labels(masks, min_area=self.min_area, max_area=max_area)
        return np.zeros(rgb.shape[:2], dtype=np.int32) if out is None else out


_SAM_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


class SAMBackend(_AMGBackend):
    """Segment Anything (Kirillov et al., ICCV 2023), automatic mask generation.

    Defaults are the benchmark's: ViT-H, 32 points per side, predicted-IoU
    threshold 0.88, stability threshold 0.95. The checkpoint is downloaded to
    ``$OV_SEG_CACHE`` (``~/.cache/omicverse/seg``) unless ``checkpoint`` is given.
    """

    name = "sam"
    requires = ("segment_anything", "torch")
    install = "Install with `pip install segment-anything` (or `pip install 'omicverse[seg]'`)."

    def __init__(self, *, model="vit_h", checkpoint=None, **kw):
        super().__init__(**kw)
        self.model_name = model
        self.checkpoint = checkpoint

    def _build(self):
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        ckpt = self.checkpoint
        if ckpt is None:
            if self.model_name not in _SAM_URLS:
                raise ValueError(f"model must be one of {sorted(_SAM_URLS)} or pass checkpoint=")
            url = _SAM_URLS[self.model_name]
            ckpt = _cache_dir() / Path(url).name
            if not ckpt.exists():
                torch.hub.download_url_to_file(url, str(ckpt))
        sam = sam_model_registry[self.model_name](checkpoint=str(ckpt))
        # the mask generator casts to float64, which MPS cannot hold
        sam.to(_torch_device(self.gpu, mps=False))
        return SamAutomaticMaskGenerator(sam, **self.amg_kwargs)


class SAM2Backend(_AMGBackend):
    """Segment Anything 2 (Ravi et al., ICLR 2025), automatic mask generation.

    ``model`` is a Hugging Face id (default ``facebook/sam2-hiera-large``, the
    benchmark's checkpoint) or a local config with ``checkpoint=``.
    """

    name = "sam2"
    requires = ("sam2", "torch", "huggingface_hub")
    install = ("Install with `pip install sam2 huggingface_hub` (Python >= 3.10, "
               "or `pip install 'omicverse[seg]'`).")

    def __init__(self, *, model="facebook/sam2-hiera-large", checkpoint=None,
                 pred_iou_thresh=0.8, **kw):
        super().__init__(pred_iou_thresh=pred_iou_thresh, **kw)
        self.model_name = model
        self.checkpoint = checkpoint

    def _build(self):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2, build_sam2_hf

        device = _torch_device(self.gpu, mps=False)
        if self.checkpoint is not None:
            net = build_sam2(self.model_name, self.checkpoint, device=device)
        else:
            net = build_sam2_hf(self.model_name, device=device)
        return SAM2AutomaticMaskGenerator(net, **self.amg_kwargs)


class MicroSAMBackend(Backend):
    """micro-SAM (Archit et al., Nat. Methods 2025), SAM fine-tuned for microscopy.

    ``model='vit_b_lm'`` (light microscopy, the benchmark's) by default.
    micro-SAM's own automatic instance segmentation decoder is used when the
    model has one; ``amg=True`` forces the plain mask generator instead, as in
    the benchmark.
    """

    name = "micro_sam"
    requires = ("micro_sam", "torch")
    install = ("micro-SAM is distributed on conda-forge: "
               "`conda install -c conda-forge micro_sam`.")
    default_tile_size = 1024

    def __init__(self, *, model="vit_b_lm", amg=None, generate_kwargs=None, **kw):
        super().__init__(**kw)
        self.model_name = model
        self.amg = amg
        self.generate_kwargs = generate_kwargs or {}

    def _build(self):
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter
        return get_predictor_and_segmenter(model_type=self.model_name,
                                           device=_torch_device(self.gpu), amg=self.amg,
                                           is_tiled=False)

    def __call__(self, tile):
        from micro_sam.automatic_segmentation import automatic_instance_segmentation
        predictor, segmenter = self.model()
        x = self.rgb8(tile) if self.image_type == "he" else self.signal8(tile)
        if min(x.shape[:2]) < 16:
            return np.zeros(x.shape[:2], dtype=np.int32)
        labels = automatic_instance_segmentation(predictor=predictor, segmenter=segmenter,
                                                 input_path=x, ndim=2, verbose=False,
                                                 **self.generate_kwargs)
        return np.asarray(labels, dtype=np.int32)


class CellSAMBackend(Backend):
    """CellSAM (Israel et al., 2023), a SAM-based cell foundation model.

    Needs ``DEEPCELL_ACCESS_TOKEN`` for the first weight download.
    """

    name = "cellsam"
    requires = ("cellSAM", "torch")
    install = ("Install with `pip install git+https://github.com/vanvalenlab/cellSAM.git` "
               "and set DEEPCELL_ACCESS_TOKEN.")
    default_tile_size = 1024

    def __init__(self, *, model="cellsam_general", bbox_threshold=0.4, **kw):
        super().__init__(**kw)
        self.model_name = model
        self.bbox_threshold = bbox_threshold

    def _build(self):
        from cellSAM import get_model
        return get_model(model=self.model_name)

    def __call__(self, tile):
        from cellSAM import segment_cellular_image
        model = self.model()
        try:
            out = segment_cellular_image(self.rgb8(tile), model, device=_torch_device(self.gpu),
                                         bbox_threshold=self.bbox_threshold)
        except Exception as exc:  # CellSAM raises on tiles with nothing to box
            warnings.warn(f"CellSAM failed on a tile ({exc}); leaving it empty.", stacklevel=2)
            return None
        masks = out[0] if isinstance(out, tuple) else out
        if masks is None:
            return None
        return np.asarray(masks, dtype=np.int32)


BACKENDS: dict[str, type[Backend]] = {
    cls.name: cls
    for cls in (WatershedBackend, CellposeBackend, StarDistBackend, SAMBackend,
                SAM2Backend, MicroSAMBackend, CellSAMBackend)
}
