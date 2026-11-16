r"""Reading time-lapse movies and segmenting every frame."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..._registry import register_function


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.shape[-1] == 1:
        return frame[..., 0]
    # video codecs store grey movies as three identical channels
    if np.array_equal(frame[..., 0], frame[..., 1]) and np.array_equal(frame[..., 1], frame[..., 2]):
        return frame[..., 0]
    return (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]).astype(frame.dtype)


@register_function(
    aliases=["读取延时视频", "read time-lapse movie", "read movie", "load video frames",
             "time-lapse microscopy reader", "mp4 avi tif stack to array"],
    category="space",
    description="Read a time-lapse microscopy movie (mp4/avi/mov via OpenCV, multi-page "
                "tif/ome-tif via tifffile, or a folder of images) into a (T, H, W) array, "
                "optionally into an on-disk memmap; frame i of the result is frame i of the "
                "decoded movie (origin top-left, x to the right, y downwards)",
    examples=["movie = ov.space.track.read_movie('movie.mp4')",
              "movie = ov.space.track.read_movie('movie.mp4', memmap='frames.npy')"],
)
def read_movie(path, frames: Optional[Sequence[int]] = None, *, gray: bool = True,
               memmap: Optional[str] = None) -> np.ndarray:
    r"""Read a movie into an array of shape ``(T, H, W)`` (``(T, H, W, C)`` if not grey).

    Parameters
    ----------
    path
        Video file (``.mp4``, ``.avi``, ``.mov``, ...), multi-page TIFF, ``.npy`` or a
        folder of single-frame images (sorted by name).
    frames
        Optional subset of frame indices to keep (in the decoded order).
    gray
        Collapse colour channels to one grey channel.
    memmap
        Write the frames into this ``.npy`` file (memory-mapped) instead of RAM; useful
        for long movies. Re-used as is if it already exists with the right length.
    """
    path = Path(path)
    keep = None if frames is None else set(int(f) for f in frames)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        return arr if frames is None else arr[sorted(keep)]

    def _iter():
        if path.is_dir():
            import imageio.v3 as iio
            files = sorted(p for p in path.iterdir() if p.suffix.lower() in
                           (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"))
            for p in files:
                yield np.asarray(iio.imread(p))
        elif suffix in (".tif", ".tiff"):
            import tifffile
            arr = tifffile.imread(str(path))
            for fr in arr:
                yield fr
        else:
            import cv2
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise IOError(f"cannot open {path}")
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if fr.ndim == 3:
                    fr = fr[..., ::-1]  # BGR -> RGB
                yield fr
            cap.release()

    out = []
    for i, fr in enumerate(_iter()):
        if keep is not None and i not in keep:
            continue
        out.append(_to_gray(fr) if gray else fr)
    if not out:
        raise ValueError(f"no frames read from {path}")
    if memmap is None:
        return np.stack(out)
    shape = (len(out),) + out[0].shape
    if os.path.exists(memmap):
        mm = np.load(memmap, mmap_mode="r")
        if mm.shape == shape:
            return mm
    mm = np.lib.format.open_memmap(memmap, mode="w+", dtype=out[0].dtype, shape=shape)
    for i, fr in enumerate(out):
        mm[i] = fr
    mm.flush()
    return np.load(memmap, mmap_mode="r")


def _cellpose_models(model: str, gpu: bool):
    from cellpose import models
    try:
        net = models.CellposeModel(model_type=model, gpu=gpu)
    except TypeError:  # cellpose 4: pretrained_model instead of model_type
        net = models.CellposeModel(pretrained_model=model, gpu=gpu)
    size = None
    if hasattr(models, "SizeModel"):
        try:
            size = models.Cellpose(model_type=model, gpu=gpu).sz
        except Exception:
            size = None
    return net, size


def _resolve_gpu(gpu) -> bool:
    if gpu in (True, False):
        return bool(gpu)
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@register_function(
    aliases=["逐帧细胞分割", "segment movie", "time-lapse segmentation", "cellpose every frame",
             "segment all frames", "相差显微镜分割"],
    category="space",
    description="Segment every frame of a time-lapse movie with a pretrained generalist "
                "model (Cellpose, default 'cyto3', works on phase contrast / brightfield / "
                "fluorescence). The cell diameter is estimated once from frames sampled across "
                "the movie (median) so that all frames share one scale; GPU used when available",
    examples=["labels = ov.space.track.segment_movie(movie)",
              "labels = ov.space.track.segment_movie(movie, diameter=30, gpu=True)"],
    related=["space.track.track_cells", "space.seg.segment"],
)
def segment_movie(movie: np.ndarray, *, method: str = "cellpose", model: str = "cyto3",
                  diameter=None, gpu="auto", scale: float = 1.0,
                  n_diameter_frames: int = 7, diameter_every: int = 50, batch_size: int = 8,
                  flow_threshold: float = 0.4, cellprob_threshold: float = 0.0,
                  min_area: Optional[int] = None, out: Optional[str] = None,
                  verbose: bool = True) -> np.ndarray:
    r"""Label image per frame, ``(T, H, W)`` int32, 0 = background.

    Parameters
    ----------
    movie
        ``(T, H, W)`` grey movie (see :func:`read_movie`).
    method
        ``'cellpose'`` (pretrained generalist). Other image methods of
        :func:`ov.space.seg.segment` are accepted and run frame by frame.
    model
        Cellpose model name (``'cyto3'`` cytoplasm generalist; ``'nuclei'``; ``'cpsam'``
        for Cellpose 4).
    diameter
        Cell diameter in pixels of the *input* movie. ``None`` estimates it with the
        model's size network on ``n_diameter_frames`` frames spread over the movie and
        uses the median for every frame. ``'adaptive'`` estimates it every
        ``diameter_every`` frames and follows it (median-smoothed, interpolated), for
        movies in which cells change size (spreading, crowding).
    gpu
        ``'auto'`` uses CUDA when torch sees a GPU.
    scale
        Down-sampling factor applied before segmentation (e.g. ``0.5`` to halve run time
        on CPU); labels are returned at the input resolution.
    min_area
        Drop objects smaller than this (pixels at input resolution). Default: an eighth
        of the area of a disc of the working diameter.
    out
        Optional ``.npy`` path; labels are written there as a memmap and the call
        resumes from the last finished frame if interrupted (progress kept in
        ``<out>.progress``).
    """
    import cv2

    T, H, W = movie.shape[:3]
    use_gpu = _resolve_gpu(gpu)
    if method != "cellpose":
        from ..seg import segment
        labs = np.zeros((T, H, W), np.int32)
        for t in range(T):
            labs[t] = segment(np.asarray(movie[t]), method=method, diameter=diameter, gpu=use_gpu)
        return labs

    net, size_model = _cellpose_models(model, use_gpu)
    h, w = (int(round(H * scale)), int(round(W * scale))) if scale != 1.0 else (H, W)

    def _frame(t):
        fr = np.asarray(movie[t])
        if scale != 1.0:
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        return fr

    def _estimate(t):
        if size_model is not None:
            try:
                d, _ = size_model.eval(_frame(t), channels=[0, 0])
                return float(np.atleast_1d(d)[0])
            except Exception:
                pass
        m = net.eval(_frame(t), diameter=30.0 * scale, channels=[0, 0])[0]
        from cellpose.utils import diameters
        return float(diameters(m)[0])

    per_frame = None
    if diameter is None or diameter == "adaptive":
        n = max(1, n_diameter_frames if diameter is None else int(np.ceil(T / diameter_every)) + 1)
        idx = np.unique(np.linspace(0, T - 1, n).round().astype(int))
        ests = np.array([_estimate(t) for t in idx])
        ok = ests > 0
        if not ok.any():
            ests, ok = np.array([30.0 * scale]), np.array([True])
            idx = np.array([0])
        diam_w = float(np.median(ests[ok]))
        if diameter == "adaptive":
            # cells change size over a movie (spreading, crowding): follow it, smoothed
            from scipy.ndimage import median_filter
            e = median_filter(ests[ok], size=min(5, int(ok.sum())), mode="nearest")
            per_frame = np.interp(np.arange(T), idx[ok], e)
    else:
        diam_w = float(diameter) * scale
    if verbose:
        print(f"[segment_movie] {T} frames {H}x{W}, working diameter {diam_w:.1f}px "
              f"(scale {scale}), gpu={use_gpu}")
    if verbose and per_frame is not None:
        print(f"[segment_movie] adaptive diameter {per_frame.min():.1f}-{per_frame.max():.1f}px")
    if min_area is None:
        dmin = diam_w if per_frame is None else float(per_frame.min())
        min_area = int(np.pi * (dmin / scale / 2) ** 2 / 8)

    if out is not None:
        mode = "r+" if os.path.exists(out) else "w+"
        labs = np.lib.format.open_memmap(out, mode=mode, dtype=np.int32, shape=(T, H, W))
        prog = out + ".progress"
        start = int(open(prog).read()) if (mode == "r+" and os.path.exists(prog)) else 0
    else:
        labs = np.zeros((T, H, W), np.int32)
        start = 0

    t = start
    while t < T:
        ts = list(range(t, min(T, t + batch_size)))
        imgs = [_frame(k) for k in ts]
        if per_frame is None:
            res = net.eval(imgs, diameter=diam_w, channels=[0, 0], flow_threshold=flow_threshold,
                           cellprob_threshold=cellprob_threshold)
            masks = res[0]
        else:
            masks = [net.eval(im, diameter=float(per_frame[k]), channels=[0, 0],
                              flow_threshold=flow_threshold,
                              cellprob_threshold=cellprob_threshold)[0]
                     for k, im in zip(ts, imgs)]
        for k, m in zip(ts, masks):
            m = np.asarray(m)
            if scale != 1.0:
                m = cv2.resize(m.astype(np.int32).astype(np.float32), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(np.int32)
            if min_area:
                ids, cnt = np.unique(m, return_counts=True)
                small = ids[(cnt < min_area) & (ids > 0)]
                if small.size:
                    m[np.isin(m, small)] = 0
            labs[k] = m
        t = ts[-1] + 1
        if out is not None:
            labs.flush()
            with open(prog, "w") as fh:
                fh.write(str(t))
        if verbose and (t // batch_size) % 10 == 0:
            print(f"[segment_movie] {t}/{T}")
    return labs
