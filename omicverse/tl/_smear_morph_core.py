r"""Bright-field stained-object morphometry and group-held-out field classification.

Self-contained core (numpy, scipy, scikit-image, Pillow, scikit-learn only; no
omicverse import), so that a stand-alone predictor can embed it verbatim:
``ov.tl.smear_morphometry_source(path)`` writes this file next to a
``predict.py``.

Scope: transmitted-light images of small stained objects (bacteria, yeasts,
parasites, cells) stained with one or two absorbing dyes, where each object of
interest is marked by an axis-aligned box.  Three layers:

1. **Measurement** (``measure_objects``): per box, optical density against the
   image's own white reference, an Otsu object mask on the padded crop clipped
   back to the box, then

   * size and outline: ellipse long / short axes, area, solidity, fill;
   * *per-cell* shape that does not depend on how many cells a box holds:
     medial-axis width, autocorrelation repeat length of the absorbance profile
     along the long axis (a chain / cluster of round cells repeats every cell
     diameter, a rod does not repeat), constriction counting along the same
     profile (cell count -> cell length), distance-transform cell splitting
     (round-piece fraction);
   * two-dye colour: each pixel's optical density unmixed onto two dye
     directions (estimated from the cohort itself with
     ``estimate_stain_vectors`` when not given), giving the primary-dye
     fraction, amounts, share of primary-dominated pixels, and the same
     contrast relative to a ring of local background.

   Stain and shape features are kept in separate name spaces (``stain_*`` vs
   everything else) so a classifier for morphology can be fitted without any
   colour input.

2. **Field summaries** (``field_features``): robust quantiles of every object
   feature over the objects of one field / image.

3. **Group-held-out model selection** (``balanced_weights``,
   ``group_holdout_predict``, ``group_holdout_select``,
   ``species_balanced_recall``): leave-one-group-out prediction with training
   weights that make every group and every class count equally, scored by the
   recall averaged over held-out groups within each class (the estimand when
   evaluation groups are new species / sites / slides); ``fit_hierarchy``
   fits a fixed decision tree of binary stages (e.g. large-vs-small, then
   round-vs-elongated on stain-free features, then dye retention) and
   ``predict_hierarchy`` applies it.

Nothing here knows any class name, organism or threshold of a particular data
set; decision stages, feature subsets and labels are supplied by the caller.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage as ndi

try:  # Pillow is the only image reader needed
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except Exception:  # pragma: no cover
    Image = None

__all__ = [
    "white_reference", "optical_density", "otsu_threshold", "object_mask",
    "estimate_stain_vectors", "measure_objects", "measure_image", "measure_cohort",
    "field_features", "SHAPE_PREFIXES", "feature_columns",
    "balanced_weights", "group_holdout_predict", "species_balanced_recall",
    "group_holdout_select", "fit_hierarchy", "predict_hierarchy",
    "group_laplace_prior", "fit_two_axis_classifier", "predict_two_axis",
    "group_holdout_two_axis", "resolve_image_path", "measure_box_tables",
    "save_predictor", "load_predictor",
]

_EPS = 1e-9


# --------------------------------------------------------------- image basics

def load_rgb(path):
    """RGB float64 array (0-255) of an image file."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float64)


def white_reference(image, q=98.0, step=6):
    """Per-channel bright reference (glass) of a transmitted-light image."""
    sample = np.asarray(image, dtype=np.float64)[::step, ::step].reshape(-1, 3)
    return np.maximum(np.percentile(sample, q, axis=0), 1.0)


def optical_density(rgb, white):
    """Beer-Lambert optical density per channel, clipped at 0."""
    return np.clip(-np.log10(np.clip(rgb, 1.0, None) / white), 0.0, None)


def otsu_threshold(values, bins=128):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size < 2 or values.max() <= values.min():
        return float(values.mean()) if values.size else 0.0
    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = np.cumsum(hist * centers) / w1
        m2 = (np.cumsum((hist * centers)[::-1]) / w2[::-1])[::-1]
    var = np.nan_to_num(w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2)
    return float(centers[int(np.argmax(var))])


def _crop(image, box, pad):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    cx0 = int(max(0, np.floor(x0 - pad)))
    cy0 = int(max(0, np.floor(y0 - pad)))
    cx1 = int(min(w, np.ceil(x1 + pad)))
    cy1 = int(min(h, np.ceil(y1 + pad)))
    return image[cy0:cy1, cx0:cx1], cx0, cy0


def object_mask(od_total, box_local, floor=0.10, margin=2):
    """Otsu mask of total optical density, clipped to the box (+margin).

    Returns (mask of the object = all components inside the box, mask of the
    heaviest component, number of components)."""
    thr = max(otsu_threshold(od_total), floor)
    mask = od_total >= thr
    x0, y0, x1, y1 = box_local
    inside = np.zeros(mask.shape, bool)
    inside[int(max(0, np.floor(y0 - margin))):int(np.ceil(y1 + margin)),
           int(max(0, np.floor(x0 - margin))):int(np.ceil(x1 + margin))] = True
    mask &= inside
    mask = ndi.binary_fill_holes(ndi.binary_closing(mask, np.ones((2, 2), bool)) | mask)
    mask &= inside
    labels, n = ndi.label(mask)
    if n == 0:
        return mask, mask, 0
    weights = ndi.sum(od_total, labels, range(1, n + 1))
    sizes = ndi.sum(mask, labels, range(1, n + 1))
    keep = np.flatnonzero(sizes >= 6) + 1
    heavy = labels == (int(np.argmax(weights)) + 1)
    union = np.isin(labels, keep) if keep.size else heavy
    return union, heavy, int(max(1, keep.size))


# ---------------------------------------------------------------- shape tools

def _pca_axes(mask):
    ys, xs = np.nonzero(mask)
    x = xs - xs.mean()
    y = ys - ys.mean()
    cov = np.array([[np.mean(x * x), np.mean(x * y)], [np.mean(x * y), np.mean(y * y)]])
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    return xs, ys, x, y, vals, vecs


def _profile(mask, density):
    xs, ys, x, y, vals, vecs = _pca_axes(mask)
    axis = vecs[:, 1]
    t = x * axis[0] + y * axis[1]
    lo, hi = float(t.min()), float(t.max())
    bins = max(6, int(np.ceil(hi - lo)))
    prof, _ = np.histogram(t, bins=bins, range=(lo, hi), weights=density[ys, xs])
    return prof, (hi - lo), (hi - lo) / bins, vals


def repeat_length_px(profile, step):
    """First autocorrelation peak of a 1-D profile (in px); nan if none."""
    if profile.size < 8 or profile.std() < 1e-9:
        return float("nan")
    c = profile - profile.mean()
    acf = np.correlate(c, c, "full")[c.size - 1:]
    acf = acf / (acf[0] + 1e-12)
    if not (acf < 0).any():
        return float("nan")
    zero = int(np.argmax(acf < 0))
    tail = acf[zero:]
    if tail.size < 3:
        return float("nan")
    peak = int(np.argmax(tail))
    if tail[peak] < 0.12:
        return float("nan")
    return float((zero + peak) * step)


def constriction_count(profile, rel_depth=0.18):
    """1 + number of prominent local minima of a smoothed profile."""
    if profile.size < 6:
        return 1
    win = max(1, int(round(profile.size * 0.08)))
    k = np.ones(2 * win + 1) / (2 * win + 1)
    s = np.convolve(profile, k, mode="same")
    necks = 0
    for i in range(1, s.size - 1):
        lo = max(0, i - win)
        hi = min(s.size, i + win + 1)
        if s[i] != s[lo:hi].min() or s[i] == s[i - 1]:
            continue
        left = s[:i].max()
        right = s[i + 1:].max()
        shoulder = min(left, right)
        if shoulder > 0 and (shoulder - s[i]) / shoulder > rel_depth:
            necks += 1
    return 1 + necks


def _split_cells(mask):
    """Distance-transform cell splitting: (n pieces, round fraction, median piece aspect)."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    dist = ndi.distance_transform_edt(mask)
    r = float(dist.max())
    if r <= 0:
        return 0, float("nan"), float("nan")
    peaks = peak_local_max(dist, min_distance=max(2, int(round(1.3 * r))),
                           threshold_abs=0.55 * r, labels=mask.astype(int),
                           exclude_border=False)
    markers = np.zeros(mask.shape, int)
    for i, (rr, cc) in enumerate(peaks, 1):
        markers[rr, cc] = i
    if markers.max() == 0:
        markers[np.unravel_index(np.argmax(dist), dist.shape)] = 1
    lab = watershed(-dist, markers, mask=mask)
    aspects = []
    for i in range(1, lab.max() + 1):
        piece = lab == i
        if piece.sum() < 6:
            continue
        *_, vals, _ = _pca_axes(piece)
        aspects.append(np.sqrt((vals[1] + 0.25) / (vals[0] + 0.25)))
    if not aspects:
        return 0, float("nan"), float("nan")
    aspects = np.asarray(aspects)
    return len(aspects), float(np.mean(aspects < 1.35)), float(np.median(aspects))


def _skeleton(mask):
    from skimage.morphology import skeletonize
    return skeletonize(mask)


# ------------------------------------------------------------------ dye model

def estimate_stain_vectors(od_pixels, n_dyes=2, max_pixels=200000, seed=0):
    """Dye optical-density directions from pooled object pixels (label-free).

    Non-negative factorisation of the (pixels x RGB) optical-density matrix
    into ``n_dyes`` unit-norm directions. Rows are ordered so that the first
    direction is the one absorbing most in the red channel relative to green
    (the blue / violet dye of a two-dye stain), the second the other one.
    """
    from sklearn.decomposition import NMF
    od = np.asarray(od_pixels, dtype=float).reshape(-1, 3)
    od = od[np.isfinite(od).all(1) & (od.sum(1) > 0.15)]
    if od.shape[0] < 50:
        raise ValueError("too few stained pixels to estimate dye vectors")
    rng = np.random.default_rng(seed)
    if od.shape[0] > max_pixels:
        od = od[rng.choice(od.shape[0], max_pixels, replace=False)]
    model = NMF(n_components=n_dyes, init="nndsvda", max_iter=800, random_state=seed)
    model.fit(od)
    v = model.components_
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + _EPS)
    order = np.argsort(-(v[:, 0] / (v[:, 1] + _EPS)))
    return v[order]


def _unmix(od_pixels, stain_vectors):
    pinv = np.linalg.pinv(np.asarray(stain_vectors, dtype=float))
    return np.clip(np.asarray(od_pixels).reshape(-1, 3) @ pinv, 0.0, None)


def _hue_turns(rgb):
    """Circular mean hue (turns) of 0-255 RGB pixels with some saturation."""
    rgb = np.asarray(rgb, dtype=float) / 255.0
    mx, mn = rgb.max(1), rgb.min(1)
    d = mx - mn
    keep = d > 0.04
    if keep.sum() < 5:
        return float("nan")
    rgb, mx, d = rgb[keep], mx[keep], d[keep]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    h = np.zeros(len(rgb))
    ir = mx == r
    ig = (mx == g) & ~ir
    ib = ~ir & ~ig
    h[ir] = ((g[ir] - b[ir]) / d[ir]) % 6
    h[ig] = (b[ig] - r[ig]) / d[ig] + 2
    h[ib] = (r[ib] - g[ib]) / d[ib] + 4
    a = 2 * np.pi * h / 6.0
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / (2 * np.pi) % 1.0)


# ---------------------------------------------------------------- measurement

SHAPE_PREFIXES = ("long_", "short_", "area_", "solidity", "fill", "medial_", "skel_",
                  "repeat_", "cells_", "cell_", "split_", "ncomp", "box_", "heavy_",
                  "circular")


def _fallback(box, um):
    w = abs(box[2] - box[0]) * um
    h = abs(box[3] - box[1]) * um
    return {"long_axis_um": max(max(w, h), 0.25), "short_axis_um": max(min(w, h), 0.25),
            "segmented": 0}


def _colour_features(rec, od, crop, union, total, stain_vectors, bg_ring_px):
    """Colour of the object pixels (primary-dye retention), in place on rec."""
    pix_od = od[union]
    pix_rgb = crop[union]
    w_od = pix_od.sum(1)
    rec["stain_od_mean"] = float(w_od.mean())
    ring = ndi.binary_dilation(union, iterations=bg_ring_px) & ~ndi.binary_dilation(union, iterations=2)
    ring &= total < max(otsu_threshold(total), 0.10)
    bg_od = od[ring] if ring.sum() >= 10 else np.zeros((1, 3))
    logbr = np.log(np.clip(pix_rgb[:, 2], 1, None) / np.clip(pix_rgb[:, 0], 1, None))
    rec["stain_log_blue_red"] = float(np.average(logbr, weights=np.maximum(w_od, 0.01)))
    # OD channel balance: red-channel absorbance share (high for violet)
    rec["stain_red_od_share"] = float(pix_od[:, 0].sum() / (pix_od[:, :2].sum() + _EPS))
    rec["stain_red_od_share_bgsub"] = float(
        (pix_od[:, 0].mean() - bg_od[:, 0].mean())
        / (pix_od[:, :2].mean(0).sum() - bg_od[:, :2].mean(0).sum() + _EPS))
    rec["stain_hue"] = _hue_turns(pix_rgb)
    if stain_vectors is not None:
        amt = _unmix(pix_od, stain_vectors)
        tot = amt.sum(1)
        rec["stain_primary_fraction"] = float(amt[:, 0].sum() / (tot.sum() + _EPS))
        rec["stain_primary_amount"] = float(amt[:, 0].mean())
        rec["stain_counter_amount"] = float(amt[:, 1].mean())
        frac = amt[:, 0] / (tot + _EPS)
        rec["stain_primary_pixel_share"] = float((frac > 0.5).mean())
        rec["stain_primary_q90"] = float(np.quantile(frac, 0.9))
        bga = _unmix(bg_od, stain_vectors).mean(0)
        rec["stain_primary_amount_bgsub"] = float(amt[:, 0].mean() - bga[0])
        rec["stain_counter_amount_bgsub"] = float(amt[:, 1].mean() - bga[1])
        d0 = max(amt[:, 0].mean() - bga[0], 0.0)
        d1 = max(amt[:, 1].mean() - bga[1], 0.0)
        rec["stain_primary_fraction_bgsub"] = float(d0 / (d0 + d1 + _EPS))
    # one scalar colour index: primary-dye fraction (larger = more primary dye retained)
    rec["stain_index"] = rec.get("stain_primary_fraction", rec["stain_red_od_share"])


def measure_objects(image, boxes, um_per_px, stain_vectors=None, white=None,
                    pad_frac=0.25, min_pad=4.0, bg_ring_px=6):
    """Per-object measurements for the boxes of one image.

    Parameters
    ----------
    image : (H, W, 3) array, 0-255 RGB.
    boxes : sequence of (x_min, y_min, x_max, y_max) in pixels (origin top-left).
    um_per_px : pixel size in micrometres.
    stain_vectors : optional (2, 3) dye OD directions (primary dye first);
        without them only dye-independent colour features are reported.

    Returns a list of dicts (one per box, same order). Keys ``long_axis_um``,
    ``short_axis_um`` are always finite and ordered; ``stain_*`` keys carry
    colour, every other key is colour-free shape.
    """
    image = np.asarray(image, dtype=np.float64)
    if white is None:
        white = white_reference(image)
    um = float(um_per_px)
    out = []
    for box in boxes:
        box = tuple(float(b) for b in box)
        bw = max(box[2] - box[0], 1.0)
        bh = max(box[3] - box[1], 1.0)
        pad = max(min_pad, pad_frac * max(bw, bh)) + bg_ring_px
        crop, cx0, cy0 = _crop(image, box, pad)
        rec = {"box_long_um": max(bw, bh) * um, "box_short_um": min(bw, bh) * um,
               "box_aspect": max(bw, bh) / min(bw, bh)}
        if crop.size == 0:
            rec.update(_fallback(box, um))
            out.append(rec)
            continue
        od = optical_density(crop, white)
        total = od.sum(axis=2)
        local = (box[0] - cx0, box[1] - cy0, box[2] - cx0, box[3] - cy0)
        union, heavy, ncomp = object_mask(total, local)
        if heavy.sum() < 6:
            rec.update(_fallback(box, um))
            x0, y0, x1, y1 = (int(max(0, np.floor(v))) for v in local)
            inbox = np.zeros(total.shape, bool)
            inbox[y0:max(y0 + 1, int(np.ceil(local[3]))), x0:max(x0 + 1, int(np.ceil(local[2])))] = True
            vals = total[inbox]
            if vals.size >= 4:
                dark = inbox & (total >= np.quantile(vals, 0.75))
                _colour_features(rec, od, crop, dark, total, stain_vectors, bg_ring_px)
            out.append(rec)
            continue
        rec["segmented"] = 1
        # ---- outline of the whole marked object (all components in the box)
        *_, vals_u, _ = _pca_axes(union)
        major = 4.0 * np.sqrt(vals_u[1])
        minor = 4.0 * np.sqrt(vals_u[0])
        diag = np.hypot(bw, bh)
        major = min(max(major, minor), 2.0 * diag)
        rec["long_axis_um"] = max(major * um, 0.2)
        rec["short_axis_um"] = max(min(minor, major) * um, 0.2)
        rec["long_short_ratio"] = rec["long_axis_um"] / rec["short_axis_um"]
        area = float(union.sum())
        rec["area_um2"] = area * um * um
        rec["fill"] = area / (bw * bh)
        rec["ncomp"] = ncomp
        try:
            from skimage.morphology import convex_hull_image
            hull = convex_hull_image(union)
            rec["solidity"] = area / max(float(hull.sum()), 1.0)
        except Exception:
            rec["solidity"] = float("nan")
        # ---- per-cell shape on the heaviest component
        dist = ndi.distance_transform_edt(heavy)
        skel = _skeleton(heavy)
        prof_w = dist[skel] if skel.sum() >= 3 else dist[heavy]
        width = 2.0 * float(np.median(prof_w))
        rec["medial_width_um"] = width * um
        rec["medial_width_max_um"] = 2.0 * float(dist.max()) * um
        rec["medial_width_cv"] = float(np.std(prof_w) / (np.mean(prof_w) + _EPS))
        rec["skel_length_um"] = float(skel.sum()) * um
        rec["skel_elongation"] = float(skel.sum()) / max(width, 1.0)
        prof, extent, step, vals_h = _profile(heavy, total)
        hminor = 4.0 * np.sqrt(vals_h[0])
        rec["heavy_long_um"] = 4.0 * np.sqrt(vals_h[1]) * um
        rec["heavy_short_um"] = hminor * um
        rep = repeat_length_px(prof, step)
        rec["repeat_um"] = rep * um if rep == rep else float("nan")
        rec["repeat_aspect"] = rep / max(hminor, 1.0) if rep == rep else float("nan")
        rec["repeat_found"] = float(rep == rep)
        ncell = constriction_count(prof)
        rec["cells_n"] = ncell
        rec["cell_length_um"] = extent * um / ncell
        rec["cell_aspect"] = (extent / ncell) / max(hminor, 1.0)
        n_split, round_frac, piece_aspect = _split_cells(heavy)
        rec["split_n"] = n_split
        rec["split_round_fraction"] = round_frac
        rec["split_piece_aspect"] = piece_aspect
        per = 0.0
        er = ndi.binary_erosion(heavy)
        per = float((heavy & ~er).sum())
        rec["circularity"] = 4 * np.pi * float(heavy.sum()) / max(per * per, 1.0)
        _colour_features(rec, od, crop, union, total, stain_vectors, bg_ring_px)
        out.append(rec)
    return out


def measure_image(path, boxes, um_per_px, stain_vectors=None):
    """Read one image file and measure its boxes."""
    return measure_objects(load_rgb(path), boxes, um_per_px, stain_vectors=stain_vectors)


def _pixels_for_vectors(job):
    path, boxes, max_per_image = job
    image = load_rgb(path)
    white = white_reference(image)
    rng = np.random.default_rng(len(boxes))
    chunks = []
    for box in boxes:
        crop, cx0, cy0 = _crop(image, box, 4)
        if crop.size == 0:
            continue
        od = optical_density(crop, white)
        local = (box[0] - cx0, box[1] - cy0, box[2] - cx0, box[3] - cy0)
        union, _, _ = object_mask(od.sum(2), local)
        if union.sum():
            chunks.append(od[union])
    if not chunks:
        return np.zeros((0, 3))
    px = np.concatenate(chunks)
    if px.shape[0] > max_per_image:
        px = px[rng.choice(px.shape[0], max_per_image, replace=False)]
    return px


def _measure_job(job):
    path, boxes, um, vectors = job
    return measure_image(path, boxes, um, stain_vectors=vectors)


def measure_cohort(items, stain_vectors="estimate", n_jobs=1, max_pixels_per_image=3000):
    """Measure many images.

    items : list of (image_path, boxes, um_per_px).
    stain_vectors : (2, 3) array, ``"estimate"`` (pooled from these images'
        objects, label-free) or None.
    Returns (list of per-image lists of dicts, stain_vectors used).
    """
    n_jobs = max(1, int(n_jobs))
    pool = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
    try:
        mapper = pool.map if pool is not None else map
        if isinstance(stain_vectors, str) and stain_vectors == "estimate":
            px = list(mapper(_pixels_for_vectors,
                             [(p, b, max_pixels_per_image) for p, b, _ in items]))
            stain_vectors = estimate_stain_vectors(np.concatenate(px))
        jobs = [(p, b, u, stain_vectors) for p, b, u in items]
        results = list(mapper(_measure_job, jobs))
    finally:
        if pool is not None:
            pool.shutdown()
    return results, stain_vectors


# ------------------------------------------------------------ field summaries

def feature_columns(columns, kind="shape"):
    """Split field-feature names into colour-free shape (``kind='shape'``) or stain."""
    cols = [c for c in columns if c not in ("field_id",)]
    if kind == "stain":
        return [c for c in cols if c.startswith("stain_")]
    if kind == "shape":
        return [c for c in cols if not c.startswith("stain_")]
    return cols


def field_features(records, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """Robust per-field summaries of per-object measurements.

    For each numeric key: quantiles (``<key>_q50`` ...), mean and std; plus the
    object count. Missing / non-finite values are ignored; a key absent from
    every object gives nan (impute downstream)."""
    keys = sorted({k for r in records for k, v in r.items()
                   if isinstance(v, (int, float, np.floating, np.integer))})
    out = {"n_objects": float(len(records)), "log_n_objects": float(np.log1p(len(records)))}
    for k in keys:
        v = np.array([r.get(k, np.nan) for r in records], dtype=float)
        v = v[np.isfinite(v)]
        for q in quantiles:
            out[f"{k}_q{int(round(q * 100))}"] = float(np.quantile(v, q)) if v.size else float("nan")
        out[f"{k}_mean"] = float(v.mean()) if v.size else float("nan")
        out[f"{k}_std"] = float(v.std()) if v.size > 1 else 0.0 if v.size else float("nan")
    return out


# --------------------------------------------------- group-held-out selection

def balanced_weights(y, groups):
    """Sample weights so every group counts once and every class equally."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    w = np.zeros(len(y), dtype=float)
    for g in np.unique(groups):
        m = groups == g
        w[m] = 1.0 / m.sum()
    for c in np.unique(y):
        m = y == c
        w[m] /= w[m].sum()
    return w * len(y) / w.sum()


def _impute(X, med=None):
    X = np.asarray(X, dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)
    if med is None:
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isnan(X), med, X), med


def make_estimator(kind="extra_trees", seed=0, **kw):
    """A small library of robust classifiers for few-group data."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        p = dict(n_estimators=300, min_samples_leaf=3, max_features=0.4, n_jobs=1, random_state=seed)
        p.update(kw)
        return ExtraTreesClassifier(**p)
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        p = dict(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=1, random_state=seed)
        p.update(kw)
        return RandomForestClassifier(**p)
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        p = dict(C=0.3, max_iter=3000)
        p.update(kw)
        return make_pipeline(StandardScaler(), LogisticRegression(**p))
    if kind == "shallow_tree":
        from sklearn.tree import DecisionTreeClassifier
        p = dict(max_depth=2, min_samples_leaf=5, random_state=seed)
        p.update(kw)
        return DecisionTreeClassifier(**p)
    raise ValueError(kind)


def _fit(est, X, y, w):
    try:
        est.fit(X, y, sample_weight=w)
    except (TypeError, ValueError):
        est.fit(X, y, **{est.steps[-1][0] + "__sample_weight": w})
    return est


def group_holdout_predict(X, y, groups, estimator="extra_trees", seed=0, **kw):
    """Leave-one-group-out class probabilities (rows of held-out groups only
    ever predicted by models that never saw that group).

    Returns (proba (n, n_classes), classes)."""
    from sklearn.base import clone
    X, _ = _impute(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))
    base = make_estimator(estimator, seed=seed, **kw) if isinstance(estimator, str) else estimator
    for g in np.unique(groups):
        test = groups == g
        train = ~test
        if len(np.unique(y[train])) < 2:
            proba[test] = np.mean(y[train][:, None] == classes[None, :], axis=0)
            continue
        Xtr, med = _impute(X[train])
        est = _fit(clone(base), Xtr, y[train], balanced_weights(y[train], groups[train]))
        Xte, _ = _impute(X[test], med)
        p = est.predict_proba(Xte)
        cols = [list(classes).index(c) for c in est.classes_]
        proba[np.ix_(np.flatnonzero(test), cols)] = p
    return proba, classes


def species_balanced_recall(y_true, y_pred, groups):
    """Per class: recall computed within each group of that class, averaged
    over groups. Returns (dict class -> balanced recall, macro over classes)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    res = {}
    for c in np.unique(y_true):
        vals = []
        for g in np.unique(groups[y_true == c]):
            m = (groups == g) & (y_true == c)
            vals.append(float(np.mean(y_pred[m] == c)))
        res[c] = float(np.mean(vals))
    return res, float(np.mean(list(res.values()))) if res else float("nan")


def group_holdout_select(X, y, groups, candidates, feature_names=None, score="min",
                         seed=0):
    """Choose among candidate (name, columns, estimator kind / params) by
    leave-one-group-out species-balanced recall.

    candidates : list of dicts {"name", "columns" (indices or names), "estimator", "params"}.
    score : ``"min"`` (worst class, the default for a per-class floor),
        ``"macro"`` or ``"min+macro"``.
    Returns (best candidate dict with fitted ``model`` on all rows, table of
    every candidate's held-out scores as a list of dicts)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    names = list(feature_names) if feature_names is not None else None
    table = []
    best, best_score = None, -np.inf
    for cand in candidates:
        cols = cand.get("columns")
        if cols is None:
            idx = np.arange(X.shape[1])
        elif names is not None and len(cols) and isinstance(cols[0], str):
            idx = np.array([names.index(c) for c in cols])
        else:
            idx = np.asarray(cols, dtype=int)
        proba, classes = group_holdout_predict(X[:, idx], y, groups, cand.get("estimator", "extra_trees"),
                                               seed=seed, **cand.get("params", {}))
        pred = classes[proba.argmax(1)]
        per, macro = species_balanced_recall(y, pred, groups)
        s = {"min": min(per.values()), "macro": macro,
             "min+macro": min(per.values()) + macro}[score]
        row = {"name": cand.get("name"), "score": s, "macro": macro, **{f"recall_{k}": v for k, v in per.items()}}
        table.append(row)
        if s > best_score:
            best_score, best = s, dict(cand, index=idx, oof_proba=proba, classes=classes)
    Xb, med = _impute(X[:, best["index"]])
    est = make_estimator(best.get("estimator", "extra_trees"), seed=seed, **best.get("params", {})) \
        if isinstance(best.get("estimator", "extra_trees"), str) else best["estimator"]
    best["model"] = _fit(est, Xb, y, balanced_weights(y, groups))
    best["impute"] = med
    return best, table


# ------------------------------------------------------ hierarchical decisions

def fit_hierarchy(X, labels, groups, stages, feature_names, estimator="extra_trees",
                  seed=0, **kw):
    """Fit a fixed tree of binary stages.

    stages : list of dicts, applied in order to the rows still undecided:
        {"name", "positive": [labels sent to the 'yes' leaf],
         "eligible": [labels that train this stage] (default: all remaining),
         "columns": list of feature names (e.g. shape-only for morphology),
         "yes": label or None (None = keep going), "no": label or None}
    Each stage is trained with group- and class-balanced weights on its eligible
    rows only. Returns a dict to pass to ``predict_hierarchy``."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    names = list(feature_names)
    fitted = []
    for st in stages:
        elig = np.isin(labels, st["eligible"]) if st.get("eligible") else np.ones(len(labels), bool)
        idx = np.array([names.index(c) for c in st["columns"]])
        target = np.isin(labels[elig], st["positive"]).astype(int)
        Xs, med = _impute(X[elig][:, idx])
        est = make_estimator(st.get("estimator", estimator), seed=seed, **{**kw, **st.get("params", {})})
        _fit(est, Xs, target, balanced_weights(target, groups[elig]))
        fitted.append({**st, "index": idx, "impute": med, "model": est})
    return {"stages": fitted, "feature_names": names}


def predict_hierarchy(model, X, return_proba=False):
    """Apply ``fit_hierarchy`` output; stage 'yes' if P(yes) >= stage['cut'] (default 0.5)."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    out = np.array([None] * n, dtype=object)
    probs = {}
    for st in model["stages"]:
        Xs, _ = _impute(X[:, st["index"]], st["impute"])
        p = st["model"].predict_proba(Xs)
        p1 = p[:, list(st["model"].classes_).index(1)] if 1 in st["model"].classes_ else np.zeros(n)
        probs[st["name"]] = p1
        undecided = np.array([o is None for o in out])
        yes = p1 >= st.get("cut", 0.5)
        if st.get("yes") is not None:
            out[undecided & yes] = st["yes"]
        if st.get("no") is not None:
            out[undecided & ~yes] = st["no"]
    return (out, probs) if return_proba else out


# ----------------------------------------------- two-axis (shape x stain) calls

def group_laplace_prior(context, outcome, groups, outcomes=None, alpha=1.0):
    """P(outcome | context) counted over *groups*, not rows, with Laplace smoothing.

    Every group votes once for its (context, outcome) combination (its majority
    context), so a context seen only with one outcome in the training groups
    still keeps probability alpha / (n_groups + alpha * n_outcomes) for the
    others -- the right prior when new groups (species, sites) are expected.
    Returns {context: {outcome: p}}."""
    context = np.asarray(context, dtype=object)
    outcome = np.asarray(outcome, dtype=object)
    groups = np.asarray(groups, dtype=object)
    outcomes = sorted(set(outcome.tolist())) if outcomes is None else list(outcomes)
    votes = {}
    for g in np.unique(groups):
        m = groups == g
        ctx_vals, ctx_n = np.unique(context[m], return_counts=True)
        out_vals, out_n = np.unique(outcome[m], return_counts=True)
        votes.setdefault(ctx_vals[np.argmax(ctx_n)], []).append(out_vals[np.argmax(out_n)])
    prior = {}
    for ctx in sorted(set(context.tolist())):
        v = votes.get(ctx, [])
        n = len(v) + alpha * len(outcomes)
        prior[ctx] = {o: (v.count(o) + alpha) / n for o in outcomes}
    return prior


def fit_two_axis_classifier(X, labels, groups, label_axes, feature_names=None,
                            shape_columns=None, stain_columns=None, positive_stain=None,
                            shape_estimator="random_forest", stain_C=0.1,
                            prior_alpha=1.0, seed=0):
    """Field classifier that keeps morphology and stain as separate evidence.

    label_axes : {label: (shape_value, stain_value)}; ``stain_value`` None for
        classes defined by shape alone (e.g. a large-cell class).
    shape stage : multi-class (shape values) on stain-free columns, group- and
        class-balanced.
    stain stage : binary stain value on stain columns, group- and
        class-balanced logistic regression -- with balanced training its logit
        is a prior-free log likelihood ratio of the two stain outcomes.
    prior : P(stain | shape) by ``group_laplace_prior`` over training groups,
        so stain disagreeing with shape needs proportionally stronger colour
        evidence, while an unseen shape x stain pairing keeps nonzero prior.
    Prediction (``predict_two_axis``) scores every *seen* label by
    P(shape | x) * P(stain | shape, x) and takes the argmax; labels never seen in
    training are not emitted (their stain outcome falls to the seen label of
    the same stain value with the most probable shape)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if hasattr(X, "columns"):
        if feature_names is None:
            feature_names = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
        X = X[list(feature_names)]
    if feature_names is None:
        raise ValueError("feature_names is required for an array X")
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=object)
    groups = np.asarray(groups, dtype=object)
    names = list(feature_names)
    if shape_columns is None:
        shape_columns = feature_columns(names, "shape")
    if stain_columns is None:
        stain_columns = feature_columns(names, "stain")
    sidx = np.array([names.index(c) for c in shape_columns])
    cidx = np.array([names.index(c) for c in stain_columns])
    shape_y = np.array([label_axes[l][0] for l in labels], dtype=object)
    stain_y = np.array([label_axes[l][1] for l in labels], dtype=object)
    Xs, smed = _impute(X[:, sidx])
    shape_model = make_estimator(shape_estimator, seed=seed)
    _fit(shape_model, Xs, shape_y.astype(str), balanced_weights(shape_y.astype(str), groups))
    has = np.array([v is not None for v in stain_y])
    stain_values = sorted(set(stain_y[has].tolist()))
    if len(stain_values) != 2:
        raise ValueError("the stain axis needs exactly two observed values")
    pos = positive_stain if positive_stain is not None else stain_values[1]
    t = (stain_y[has] == pos).astype(int)
    Xc, cmed = _impute(X[has][:, cidx])
    scaler = StandardScaler().fit(Xc)
    lr = LogisticRegression(C=stain_C, max_iter=5000)
    lr.fit(scaler.transform(Xc), t, sample_weight=balanced_weights(t, groups[has]))
    prior = group_laplace_prior(shape_y[has], stain_y[has], groups[has], outcomes=stain_values,
                                alpha=prior_alpha)
    seen = sorted(set(labels.tolist()))
    return {"label_axes": {k: tuple(v) for k, v in label_axes.items()}, "seen_labels": seen,
            "shape_index": sidx, "shape_impute": smed, "shape_model": shape_model,
            "stain_index": cidx, "stain_impute": cmed, "stain_scaler": scaler, "stain_model": lr,
            "positive_stain": pos, "stain_values": stain_values, "prior": prior,
            "feature_names": names}


def predict_two_axis(model, X, return_scores=False):
    """Labels (and optionally the per-label score table) for ``fit_two_axis_classifier``.

    X may be a DataFrame (columns are aligned to the training feature names;
    missing ones are imputed) or an array in training column order."""
    if hasattr(X, "columns"):
        X = X.reindex(columns=model["feature_names"])
    X = np.asarray(X, dtype=float)
    Xs, _ = _impute(X[:, model["shape_index"]], model["shape_impute"])
    p_shape = model["shape_model"].predict_proba(Xs)
    shape_classes = [str(c) for c in model["shape_model"].classes_]
    Xc, _ = _impute(X[:, model["stain_index"]], model["stain_impute"])
    llr = model["stain_model"].decision_function(model["stain_scaler"].transform(Xc))
    pos = model["positive_stain"]
    neg = [v for v in model["stain_values"] if v != pos][0]
    seen = model["seen_labels"]
    scores = np.zeros((X.shape[0], len(seen)))
    for j, lab in enumerate(seen):
        shp, stn = model["label_axes"][lab]
        ps = p_shape[:, shape_classes.index(str(shp))]
        if stn is None:
            scores[:, j] = ps
            continue
        pr = model["prior"].get(shp, {pos: 0.5, neg: 0.5})[pos]
        post = 1.0 / (1.0 + np.exp(-(llr + np.log(pr / (1 - pr)))))
        scores[:, j] = ps * (post if stn == pos else 1 - post)
    # a stain outcome whose seen labels miss this shape keeps its mass on the seen
    # label of the same stain value: add the unseen combinations' mass there
    for shp in shape_classes:
        for stn in model["stain_values"]:
            if any(model["label_axes"][l] == (shp, stn) for l in seen):
                continue
            if not any(model["label_axes"][l][1] is not None and model["label_axes"][l][0] == shp
                       for l in seen):
                continue
            same = [j for j, l in enumerate(seen) if model["label_axes"][l][1] == stn]
            if not same:
                continue
            ps = p_shape[:, shape_classes.index(shp)]
            pr = model["prior"].get(shp, {pos: 0.5, neg: 0.5})[pos]
            post = 1.0 / (1.0 + np.exp(-(llr + np.log(pr / (1 - pr)))))
            mass = ps * (post if stn == pos else 1 - post)
            sub = scores[:, same]
            best = np.asarray(same)[sub.argmax(1)]
            scores[np.arange(X.shape[0]), best] += mass
    labels = np.array(seen, dtype=object)[scores.argmax(1)]
    if return_scores:
        return labels, {"scores": scores, "labels": seen, "p_shape": p_shape,
                        "shape_classes": shape_classes, "stain_llr": llr}
    return labels


def group_holdout_two_axis(X, labels, groups, **kw):
    """Leave-one-group-out predictions of ``fit_two_axis_classifier``."""
    labels = np.asarray(labels, dtype=object)
    groups = np.asarray(groups, dtype=object)
    pred = np.empty(len(labels), dtype=object)
    if hasattr(X, "columns"):
        if kw.get("feature_names") is None:
            kw["feature_names"] = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
        X = X[list(kw["feature_names"])]
    X = np.asarray(X, dtype=float)
    for g in np.unique(groups):
        te = groups == g
        m = fit_two_axis_classifier(X[~te], labels[~te], groups[~te], **kw)
        pred[te] = predict_two_axis(m, X[te])
    return pred


# ------------------------------------------------------------ table-level IO

def resolve_image_path(path, root):
    """Find an image listed in a table: as given, relative to ``root``, or by its
    trailing path components / file name under ``root``."""
    path = str(path)
    cands = []
    if os.path.isabs(path):
        cands.append(path)
    if root is not None:
        cands.append(os.path.join(root, path.lstrip("/")))
        parts = path.replace("\\", "/").strip("/").split("/")
        for k in range(len(parts) - 1, 0, -1):
            cands.append(os.path.join(root, *parts[-k:]))
    cands.append(path)
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(path)


def measure_box_tables(fields, objects, root=None, field_id="field_id", image_col="image_path",
                       um_col="um_per_px", object_id="organism_id",
                       box_cols=("x_min", "y_min", "x_max", "y_max"),
                       stain_vectors="estimate", n_jobs=1):
    """Measure every boxed object of every image listed in two tables.

    fields : DataFrame (one row per image: id, image path, pixel size).
    objects : DataFrame (one row per box: object id, image id, box corners).
    Returns (per-object DataFrame with ``object_id``, ``field_id`` and all
    measurements; per-field DataFrame of ``field_features``; stain vectors).
    Fields without boxes get a row of NaN features (impute downstream)."""
    import pandas as pd
    fields = fields.reset_index(drop=True)
    by_field = {k: v for k, v in objects.groupby(field_id, sort=False)}
    items, ids = [], []
    for row in fields.itertuples(index=False):
        r = row._asdict()
        fid = r[field_id]
        obj = by_field.get(fid)
        boxes = obj[list(box_cols)].to_numpy(float).tolist() if obj is not None else []
        items.append((resolve_image_path(r[image_col], root), boxes, float(r[um_col])))
        ids.append((fid, obj[object_id].tolist() if obj is not None else []))
    results, vectors = measure_cohort(items, stain_vectors=stain_vectors, n_jobs=n_jobs)
    orows, frows = [], []
    for (fid, oids), recs in zip(ids, results):
        for oid, rec in zip(oids, recs):
            orows.append({object_id: oid, field_id: fid, **rec})
        frows.append({field_id: fid, **field_features(recs)})
    return pd.DataFrame(orows), pd.DataFrame(frows), vectors


def save_predictor(obj, path):
    """Pickle a fitted model bundle (dicts, arrays, scikit-learn estimators)."""
    import pickle
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_predictor(path):
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)
