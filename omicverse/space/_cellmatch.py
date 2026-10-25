"""Cell-to-cell correspondence between two segmentations of the same tissue.

The same physical section imaged twice -- DAPI in an in-situ run, H&E in a
later staining pass, or two rounds of any nuclear stain -- gives two lists of
cell centroids in two coordinate frames. They differ by a global similarity
(shift, rotation, possibly a flip and a small scale change), a smooth
non-rigid deformation from handling the slide, partial overlap of the imaged
fields, and independent segmentation (cells missing or split on either side).

:func:`match_cells` recovers a **one-to-one** matching in four stages:

1. **coarse search** -- both point sets are rendered as Gaussian density maps
   and every flip x rotation x scale on a grid is scored by FFT
   cross-correlation, which also yields the best translation. Correlation of
   density maps tolerates partial overlap and unmatched cells, which defeats
   ICP-from-scratch;
2. **robust affine** -- mutual nearest neighbours within a shrinking radius,
   least-squares affine on them, residual trimming;
3. **non-rigid** -- Coherent Point Drift (soft correspondences + smooth
   Gaussian motion field) for deformations up to about the cell spacing, then
   a smoothed thin-plate spline on mutual pairs that agree with their
   neighbours' displacement;
4. **assignment** -- the Hungarian algorithm on warped distances, with a
   distance cap so cells without a partner stay unmatched; then the matching
   is grown outward, each unmatched cell borrowing the local deformation of
   its matched neighbours.

Everything is NumPy/SciPy; nothing is fitted to a particular dataset. The
distance scales default to multiples of the median nearest-neighbour spacing
of the fixed cells, so the same call works in pixels or microns.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from .._registry import register_function

__all__ = ["match_cells", "hematoxylin_channel", "nn_spacing"]


def nn_spacing(xy) -> float:
    """Median distance from each point to its nearest neighbour."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 2:
        return 1.0
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


def hematoxylin_channel(rgb) -> np.ndarray:
    """Nuclear (haematoxylin) signal of an RGB H&E image, scaled to [0, 1].

    Colour deconvolution (Ruifrok & Johnston) separates haematoxylin, which
    stains nuclei, from eosin. The result is directly comparable to a DAPI or
    other nuclear-fluorescence image -- bright where nuclei are -- so it is
    the H&E image to register against DAPI, or to segment nuclei on.
    """
    from skimage.color import rgb2hed

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError("hematoxylin_channel expects an (H, W, 3) RGB image")
    h = rgb2hed(rgb[..., :3])[..., 0]
    lo, hi = np.percentile(h, [1, 99.5])
    return np.clip((h - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


# ---------------------------------------------------------------------------
# stage 1: coarse similarity by density cross-correlation
# ---------------------------------------------------------------------------
def _density(xy, origin, shape, px, sigma_px):
    from scipy.ndimage import gaussian_filter

    img = np.zeros(shape, dtype=np.float64)
    ij = np.floor((xy - origin) / px).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
    np.add.at(img, (ij[ok, 1], ij[ok, 0]), 1.0)
    img = gaussian_filter(img, sigma_px)
    img -= img.mean()
    n = np.linalg.norm(img)
    return img / n if n > 0 else img


def _similarity_matrix(theta_deg, scale, flip):
    t = np.deg2rad(theta_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) * scale
    if flip:
        R = R @ np.array([[1.0, 0.0], [0.0, -1.0]])
    return R


def _coarse_search(fixed, moving, *, rotations, scales, allow_flip, sigma):
    """Best (A, b) with moving @ A.T + b ~ fixed, by density cross-correlation."""
    px = sigma / 2.0
    f_lo = fixed.min(0)
    f_shape = tuple((np.ceil((fixed.max(0) - f_lo) / px).astype(int) + 1)[::-1])
    F = _density(fixed, f_lo, f_shape, px, 2.0)
    mc = moving.mean(0)

    best = (-np.inf, None, None)
    flips = (False, True) if allow_flip else (False,)
    for flip in flips:
        for s in scales:
            for th in rotations:
                A = _similarity_matrix(th, s, flip)
                m = (moving - mc) @ A.T
                m_lo = m.min(0)
                m_shape = tuple((np.ceil((m.max(0) - m_lo) / px).astype(int) + 1)[::-1])
                M = _density(m, m_lo, m_shape, px, 2.0)
                cc = fftconvolve(F, M[::-1, ::-1], mode="full")
                k = int(np.argmax(cc))
                if cc.flat[k] > best[0]:
                    r, c = np.unravel_index(k, cc.shape)
                    # pixel (r, c) of the full correlation = offset of M's origin in F
                    shift = np.array([c - (M.shape[1] - 1), r - (M.shape[0] - 1)]) * px
                    b = f_lo + shift - m_lo - mc @ A.T
                    best = (float(cc.flat[k]), A, b)
    return best


# ---------------------------------------------------------------------------
# stage 2-3: robust affine, then thin-plate spline
# ---------------------------------------------------------------------------
def _mutual_pairs(fixed, warped, radius):
    tf, tm = cKDTree(fixed), cKDTree(warped)
    d_m, i_f = tf.query(warped, k=1)
    _, i_m = tm.query(fixed, k=1)
    j = np.arange(len(warped))
    ok = (i_m[i_f] == j) & (d_m <= radius)
    return j[ok], i_f[ok], d_m[ok]


def _fit_affine(src, dst):
    X = np.c_[src, np.ones(len(src))]
    P, *_ = np.linalg.lstsq(X, dst, rcond=None)
    return P[:2].T, P[2]


def _robust_affine(fixed, moving, A, b, spacing, n_iter=12):
    for radius in np.geomspace(1.5 * spacing, 0.4 * spacing, n_iter):
        jm, jf, d = _mutual_pairs(fixed, moving @ A.T + b, radius)
        if len(jm) < 6:
            continue
        keep = d <= np.median(d) + 2.5 * 1.4826 * np.median(np.abs(d - np.median(d)))
        A, b = _fit_affine(moving[jm[keep]], fixed[jf[keep]])
    return A, b


def _tps_warp(src, dst, smoothing):
    from scipy.interpolate import RBFInterpolator

    return RBFInterpolator(src, dst - src, kernel="thin_plate_spline",
                           smoothing=smoothing, degree=1)


def _cpd_nonrigid(fixed, moving, *, beta, lam=2.0, w=0.2, n_iter=60, tol=1e-6):
    """Non-rigid Coherent Point Drift (Myronenko & Song, 2010).

    Soft correspondences plus a Gaussian-kernel motion field regularised
    toward smoothness: the matching and the deformation are estimated
    together, so it survives deformations comparable to the cell spacing,
    where "pair nearest neighbours, then fit a spline" starts from wrong pairs.
    ``w`` is the outlier weight -- cells with no partner on the other side.
    """
    # CPD's uniform outlier density assumes unit-scale coordinates: standardise
    # both sets with the fixed set's centre and spread, solve, then map back.
    # (On raw pixel coordinates the outlier term swamps every match and the
    # motion field stays at zero.)
    mu = fixed.mean(0)
    sc = float(np.sqrt(((fixed - mu) ** 2).sum(1).mean())) or 1.0
    X, Y = (fixed - mu) / sc, (moving - mu) / sc
    beta = beta / sc
    N, D = X.shape
    M = Y.shape[0]
    G = np.exp(-((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1) / (2.0 * beta ** 2))
    W = np.zeros((M, D))
    T = Y.copy()
    sigma2 = ((X[None, :, :] - Y[:, None, :]) ** 2).sum() / (D * M * N)
    for _ in range(n_iter):
        d2 = ((X[None, :, :] - T[:, None, :]) ** 2).sum(-1)          # M x N
        P = np.exp(-d2 / (2.0 * sigma2))
        c = (2.0 * np.pi * sigma2) ** (D / 2.0) * w / (1.0 - w) * M / N
        P /= P.sum(0, keepdims=True) + c
        P1, Pt1 = P.sum(1), P.sum(0)
        Np = P1.sum()
        if Np < 1e-9:
            break
        A = G + lam * sigma2 * np.diag(1.0 / np.maximum(P1, 1e-12))
        B = (P @ X) / np.maximum(P1, 1e-12)[:, None] - Y
        W = np.linalg.solve(A, B)
        T = Y + G @ W
        new = ((Pt1[:, None] * X ** 2).sum() - 2.0 * np.trace((P @ X).T @ T)
               + (P1[:, None] * T ** 2).sum()) / (Np * D)
        new = max(new, 1e-8)
        if abs(sigma2 - new) < tol * sigma2:
            sigma2 = new
            break
        sigma2 = new
    return T * sc + mu


def _coherent(src, dst, spacing, k=8, tol=0.3):
    """Keep pairs whose displacement agrees with their neighbours' median.

    A smooth deformation moves nearby cells alike; a wrong pair points
    somewhere else. This filters wrong pairs before they bend the spline.
    """
    if len(src) <= k:
        return np.ones(len(src), dtype=bool)
    v = dst - src
    _, nb = cKDTree(src).query(src, k=k + 1)
    local = np.median(v[nb[:, 1:]], axis=1)
    return np.linalg.norm(v - local, axis=1) <= tol * spacing


def _nonrigid(fixed, affine_moved, spacing, n_iter, smoothing):
    """Coarse-to-fine spline: wide radius + stiff spline first, then tighten."""
    warped = affine_moved
    n_iter = max(int(n_iter), 1)
    radii = np.linspace(1.0 * spacing, 0.4 * spacing, n_iter)
    stiff = np.geomspace(10.0, 1.0, n_iter) * smoothing
    for radius, sm in zip(radii, stiff):
        jm, jf, _ = _mutual_pairs(fixed, warped, radius)
        if len(jm) < 10:
            break
        ok = _coherent(warped[jm], fixed[jf], spacing)
        jm, jf = jm[ok], jf[ok]
        if len(jm) < 10:
            break
        field = _tps_warp(affine_moved[jm], fixed[jf], sm * len(jm))
        warped = affine_moved + field(affine_moved)
    return warped


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------
@register_function(
    aliases=["细胞配对", "细胞对应", "match cells", "DAPI H&E cell matching",
             "cell correspondence", "跨模态细胞匹配"],
    category="space",
    description="One-to-one correspondence between two cell-centroid sets of the "
                "same tissue (e.g. DAPI vs H&E): coarse FFT search, robust affine, "
                "thin-plate-spline refinement, Hungarian assignment",
    requires={},
    produces={},
    auto_fix='none',
    examples=[
        "pairs = ov.space.geom.match_cells(dapi_xy, he_xy)",
        "pairs, info = ov.space.geom.match_cells(dapi_df[['x','y']], he_df[['x','y']],"
        " fixed_ids=dapi_df.id, moving_ids=he_df.id, return_info=True)",
    ],
)
def match_cells(
    fixed_xy,
    moving_xy,
    *,
    fixed_ids: Optional[Sequence] = None,
    moving_ids: Optional[Sequence] = None,
    fixed_image=None,
    moving_image=None,
    fixed_rgb: bool = False,
    moving_rgb: bool = False,
    image_kwargs: Optional[dict] = None,
    rotations: Optional[Sequence[float]] = None,
    scales: Optional[Sequence[float]] = None,
    allow_flip: bool = True,
    coarse_sigma: float = 1.5,
    nonrigid: bool = True,
    cpd_beta: float = 4.0,
    cpd_lambda: float = 2.0,
    cpd_outlier: float = 0.2,
    tps_iter: int = 4,
    tps_smoothing: float = 1.0,
    max_distance: Optional[float] = None,
    grow: bool = True,
    return_info: bool = False,
):
    """One-to-one matching between two cell-centroid sets of the same tissue.

    Parameters
    ----------
    fixed_xy, moving_xy
        ``(n, 2)`` centroids. ``moving`` is mapped onto ``fixed``; which side is
        which does not change the matching, only the frame of the warp.
    fixed_ids, moving_ids
        Identifiers returned in the result (default: row positions).
    fixed_image, moving_image
        The images the centroids were segmented from, in the same pixel frames.
        **Pass them whenever you have them**: cell positions are close to
        random and carry little to register on, while the images carry dense
        tissue texture. With images, the global pose and the deformation come
        from :func:`register_images` (patch voting + block matching), and the
        point-only search below is skipped. Use ``*_rgb=True`` for an RGB H&E
        image (converted to its haematoxylin channel). ``image_kwargs`` goes to
        :func:`register_images`.
    rotations, scales
        Coarse-search grid. Default: every 2 degrees over the full circle, and
        nine scales from 0.8 to 1.25. Narrow them when the frames are known to
        be close -- it is faster, not more accurate.
    allow_flip
        Also search mirrored orientations (slides are sometimes imaged flipped).
    coarse_sigma
        Blur of the density maps in the global search, in units of the fixed
        cells' median spacing. It must exceed the typical local displacement
        left after the global transform, or the search locks onto noise.
    nonrigid
        Refine the affine non-rigidly: Coherent Point Drift, then a smoothed
        thin-plate spline on coherent pairs. Turn off only for data known to
        be rigidly related.
    cpd_beta, cpd_lambda, cpd_outlier
        CPD motion-field width (in units of the fixed cells' median spacing),
        smoothness weight, and expected fraction of cells without a partner.
    tps_iter, tps_smoothing
        Spline refinement rounds and smoothing per pair; larger smoothing
        follows the deformation less closely but resists wrong pairs.
    max_distance
        Largest accepted distance after warping. Default: ``0.5 x`` the median
        nearest-neighbour spacing of the fixed cells -- beyond that a partner
        is more likely a neighbour than the same cell. Raise it for recall,
        lower it for precision.
    grow
        After the first assignment, give each unmatched cell the local
        deformation of its matched neighbours and assign again, repeatedly.
        Recovers cells where the warp is still off, typically at the field
        borders; pairs are still capped by ``max_distance``.
    return_info
        Also return a dict with the transform, warped coordinates and scores.

    Returns
    -------
    pandas.DataFrame
        Columns ``moving_id, fixed_id, distance``; each id appears at most once.
    """
    fixed = np.asarray(fixed_xy, dtype=np.float64)
    moving = np.asarray(moving_xy, dtype=np.float64)
    if fixed.ndim != 2 or fixed.shape[1] != 2 or moving.ndim != 2 or moving.shape[1] != 2:
        raise ValueError("fixed_xy and moving_xy must be (n, 2) coordinate arrays")
    if len(fixed) < 3 or len(moving) < 3:
        raise ValueError("need at least 3 cells on each side")
    fixed_ids = np.arange(len(fixed)) if fixed_ids is None else np.asarray(fixed_ids)
    moving_ids = np.arange(len(moving)) if moving_ids is None else np.asarray(moving_ids)

    spacing = nn_spacing(fixed)
    if fixed_image is not None and moving_image is not None:
        return _match_with_images(
            fixed, moving, fixed_ids, moving_ids, fixed_image, moving_image,
            fixed_rgb=fixed_rgb, moving_rgb=moving_rgb,
            image_kwargs=image_kwargs or {}, spacing=spacing, nonrigid=nonrigid,
            tps_iter=tps_iter, tps_smoothing=tps_smoothing,
            max_distance=max_distance, grow=grow, return_info=return_info)
    rotations = np.arange(0.0, 360.0, 2.0) if rotations is None else np.asarray(rotations)
    scales = np.geomspace(0.8, 1.25, 9) if scales is None else np.asarray(scales)

    # Broad blur for the global search: it must tolerate local displacements of
    # about a cell spacing, which a sharp density map cannot.
    score, A, b = _coarse_search(fixed, moving, rotations=rotations, scales=scales,
                                 allow_flip=allow_flip, sigma=coarse_sigma * spacing)
    # Fine pass around the coarse optimum. A = s R F with F the optional mirror,
    # so A F = s R gives the rotation; search the mirrored points without flip.
    F = np.diag([1.0, -1.0]) if np.linalg.det(A) < 0 else np.eye(2)
    sR = A @ F
    th0 = np.degrees(np.arctan2(sR[1, 0], sR[0, 0]))
    s0 = float(np.sqrt(abs(np.linalg.det(A))))
    step_r = float(np.median(np.diff(np.sort(rotations)))) if len(rotations) > 1 else 2.0
    step_s = float(np.median(np.diff(np.log(np.sort(scales))))) if len(scales) > 1 else 0.05
    score_f, A_f, b_f = _coarse_search(
        fixed, moving @ F.T,
        rotations=th0 + np.linspace(-step_r, step_r, 9),
        scales=s0 * np.exp(np.linspace(-step_s, step_s, 5)),
        allow_flip=False, sigma=coarse_sigma * spacing)
    if A_f is not None and score_f >= score:
        score, A, b = score_f, A_f @ F, b_f

    A, b = _robust_affine(fixed, moving, A, b, spacing)
    moved = moving @ A.T + b
    if nonrigid:
        # CPD carries the bulk of the deformation; the coherent spline polishes it.
        # Only the moving cells near the fixed field enter CPD (cost is M x N).
        near = cKDTree(fixed).query(moved, k=1)[0] <= 3.0 * spacing
        warped = moved.copy()
        if near.sum() >= 10:
            warped[near] = _cpd_nonrigid(fixed, moved[near], beta=cpd_beta * spacing,
                                         lam=cpd_lambda, w=cpd_outlier)
        warped = _nonrigid(fixed, warped, spacing, tps_iter, tps_smoothing)
    else:
        warped = moved

    cap = 0.5 * spacing if max_distance is None else float(max_distance)
    mi, fj, _ = _assign(fixed, warped, cap)
    if grow:
        mi, fj = _grow(fixed, warped, mi, fj, cap)
    dist = np.linalg.norm(warped[mi] - fixed[fj], axis=1)
    out = pd.DataFrame({"moving_id": moving_ids[mi], "fixed_id": fixed_ids[fj],
                        "distance": dist}).sort_values("distance").reset_index(drop=True)
    if not return_info:
        return out
    info = {
        "affine_matrix": A, "affine_offset": b, "coarse_score": score,
        "spacing": spacing, "max_distance": cap,
        "warped_moving": warped, "n_moving": len(moving), "n_fixed": len(fixed),
    }
    return out, info


def _assign(fixed, warped, cap):
    """Hungarian one-to-one assignment, pairs farther than ``cap`` refused."""
    tree = cKDTree(fixed)
    cand = tree.query_ball_point(warped, r=cap)
    rows = sorted({i for i, c in enumerate(cand) if c})
    cols = sorted({j for c in cand for j in c})
    if not rows:
        return np.array([], int), np.array([], int), np.array([])
    ri = {r: k for k, r in enumerate(rows)}
    ci = {c: k for k, c in enumerate(cols)}
    C = np.full((len(rows), len(cols)), 1e6 * cap)
    for i in rows:
        for j in cand[i]:
            C[ri[i], ci[j]] = np.linalg.norm(warped[i] - fixed[j])
    r_idx, c_idx = linear_sum_assignment(C)
    keep = C[r_idx, c_idx] <= cap
    return (np.asarray(rows)[r_idx[keep]], np.asarray(cols)[c_idx[keep]],
            C[r_idx[keep], c_idx[keep]])


def _match_with_images(fixed, moving, fixed_ids, moving_ids, fixed_image, moving_image,
                       *, fixed_rgb, moving_rgb, image_kwargs, spacing, nonrigid,
                       tps_iter, tps_smoothing, max_distance, grow, return_info):
    from scipy.interpolate import RBFInterpolator
    from ._imreg import register_images

    mapping, reg = register_images(fixed_image, moving_image, fixed_rgb=fixed_rgb,
                                   moving_rgb=moving_rgb, return_info=True, **image_kwargs)
    # invert the fixed->moving mapping on a grid to carry moving cells home
    H, W = np.asarray(fixed_image).shape[:2]
    step = max(int(spacing / 2), 2)
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    g = np.c_[xx.ravel(), yy.ravel()].astype(float)
    inv = RBFInterpolator(mapping(g), g, kernel="thin_plate_spline", neighbors=64,
                          smoothing=0.0, degree=1)
    moved = inv(moving)
    warped = _nonrigid(fixed, moved, spacing, tps_iter, tps_smoothing) if nonrigid else moved
    cap = 0.5 * spacing if max_distance is None else float(max_distance)
    mi, fj, _ = _assign(fixed, warped, cap)
    n_first = len(mi)
    if grow:
        mi, fj = _grow(fixed, warped, mi, fj, cap)
    dist = np.linalg.norm(warped[mi] - fixed[fj], axis=1)
    out = pd.DataFrame({"moving_id": moving_ids[mi], "fixed_id": fixed_ids[fj],
                        "distance": dist}).sort_values("distance").reset_index(drop=True)
    if not return_info:
        return out
    return out, {"registration": reg, "n_direct": n_first, "n_grown": len(mi) - n_first, "spacing": spacing, "max_distance": cap,
                 "warped_moving": warped, "n_moving": len(moving), "n_fixed": len(fixed)}


def _local_pred(anchor_xy, anchor_res, query_xy, k, exclude_self=False):
    """Residual at each query point from a local linear fit of nearby anchors."""
    tree = cKDTree(anchor_xy)
    kk = min(k + int(exclude_self), len(anchor_xy))
    _, nb = tree.query(query_xy, k=kk)
    nb = nb.reshape(len(query_xy), kk)
    if exclude_self:
        nb = nb[:, 1:]
    out = np.empty((len(query_xy), 2))
    for r in range(len(query_xy)):
        j = nb[r]
        X = np.c_[np.ones(len(j)), anchor_xy[j] - query_xy[r]]
        if len(j) >= 4:
            coef, *_ = np.linalg.lstsq(X, anchor_res[j], rcond=None)
            out[r] = coef[0]
        else:
            out[r] = np.median(anchor_res[j], axis=0)
    return out


def _grow(fixed, warped, mi, fj, cap, k=8, n_iter=10):
    """Prune and grow the matching with the local deformation of neighbours.

    Matched pairs give the residual ``fixed - warped`` where the registration
    is right. Each round (1) drops pairs whose residual disagrees with the
    local linear fit of their neighbours' residuals -- a wrong pair occupies
    two cells and blocks their true partners -- and (2) gives every unmatched
    cell its neighbours' local residual and assigns the still-unmatched cells
    one-to-one within ``cap``. This follows the deformation where the global
    warp is still off, typically at the borders of the field.
    """
    mi, fj = np.asarray(mi, int), np.asarray(fj, int)
    for _ in range(n_iter):
        if len(mi) < 6:
            break
        res = fixed[fj] - warped[mi]
        pred = _local_pred(warped[mi], res, warped[mi], k, exclude_self=True)
        keep = np.linalg.norm(res - pred, axis=1) <= cap
        mi_k, fj_k = mi[keep], fj[keep]
        if len(mi_k) < 6:
            break
        um = np.setdiff1d(np.arange(len(warped)), mi_k)
        uf = np.setdiff1d(np.arange(len(fixed)), fj_k)
        if len(um) == 0 or len(uf) == 0:
            mi, fj = mi_k, fj_k
            break
        res_k = fixed[fj_k] - warped[mi_k]
        p = warped[um] + _local_pred(warped[mi_k], res_k, warped[um], k)
        a, b, _ = _assign(fixed[uf], p, cap)
        new_mi = np.r_[mi_k, um[a]].astype(int)
        new_fj = np.r_[fj_k, uf[b]].astype(int)
        changed = len(new_mi) != len(mi) or set(zip(new_mi, new_fj)) != set(zip(mi, fj))
        mi, fj = new_mi, new_fj
        if not changed:
            break
    return mi, fj
