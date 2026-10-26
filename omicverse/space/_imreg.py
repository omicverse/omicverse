"""Deformable registration of two images of the same tissue section.

Built for the common multi-round case -- a nuclear fluorescence image (DAPI)
against a later H&E of the same slide, or two rounds of any stain -- where the
frames differ by a global similarity, partial overlap, and a smooth non-rigid
deformation that can exceed a cell diameter. Cell centroids alone are nearly
Poisson-distributed and carry little to register on; the images carry dense
tissue texture, so registration is done on images and the result is then
applied to the cells.

:func:`register_images` returns a mapping ``fixed (x, y) -> moving (x, y)``:

1. **global** -- local patch voting over a flip x rotation x scale grid: each
   small fixed patch is template-matched against the transformed moving image
   and votes for a translation. Patches stay nearly rigid under a deformation
   that ruins any whole-image correlation, so the right pose is the one on
   which the most patches agree. The agreeing votes also give a first coarse
   displacement field;
2. **deformable** -- coarse-to-fine block matching: control points on a grid
   in the fixed frame, each fixed patch template-matched against the moving
   image warped by the current mapping; displacements that disagree with
   their neighbours or score poorly are dropped, the rest are interpolated by
   a smoothed thin-plate spline and composed onto the mapping.

Pass an H&E image through :func:`hematoxylin_channel` (or give the RGB image
and ``moving_rgb=True``) so both images show nuclei bright.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

__all__ = ["register_images"]


def _prep(img, blur):
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(-1)
    lo, hi = np.percentile(img, [1, 99.5])
    img = np.clip((img - lo) / max(hi - lo, 1e-12), 0, 1)
    return ndimage.gaussian_filter(img, blur) if blur else img


def _similarity(theta_deg, scale, flip):
    t = np.deg2rad(theta_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) * scale
    if flip:
        R = R @ np.diag([1.0, -1.0])
    return R


def _render(moving, A):
    """Moving image under x' = A x (about its centre), on a tight canvas.

    Returns the canvas, its validity mask, and the canvas origin (x, y) in the
    rotated frame, so a canvas pixel (col, row) is the rotated-frame point
    origin + (col, row).
    """
    h, w = moving.shape
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float) - c
    rc = corners @ A.T
    lo, hi = np.floor(rc.min(0)), np.ceil(rc.max(0))
    W, H = int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1
    Ai = np.linalg.inv(A)
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.c_[xx.ravel() + lo[0], yy.ravel() + lo[1]] @ Ai.T + c
    img = ndimage.map_coordinates(moving, [pts[:, 1], pts[:, 0]], order=1, cval=0.0)
    ok = (pts[:, 0] >= 0) & (pts[:, 0] <= w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= h - 1)
    return img.reshape(H, W), ok.reshape(H, W).astype(float), lo, c


def _vote_patches(fixed, half, grid, max_patches):
    """Centres of the most textured patches on a grid (blank ones carry no pose)."""
    H, W = fixed.shape
    cand = []
    for y in range(half, H - half, grid):
        for x in range(half, W - half, grid):
            sd = fixed[y - half:y + half + 1, x - half:x + half + 1].std()
            if sd > 1e-6:
                cand.append((sd, x, y))
    cand.sort(reverse=True)
    return [(x, y) for _, x, y in cand[:max_patches]]


def _patch_votes(fixed, canvas, half, centres):
    """Each fixed patch's best translation into ``canvas`` (template matching)."""
    from skimage.feature import match_template

    pts, offs, peak = [], [], []
    for x, y in centres:
            tpl = fixed[y - half:y + half + 1, x - half:x + half + 1]
            res = match_template(canvas, tpl)
            r, c = np.unravel_index(int(np.argmax(res)), res.shape)
            pts.append((x, y))
            offs.append((c + half - x, r + half - y))
            peak.append(res[r, c])
    return np.array(pts, float), np.array(offs, float), np.array(peak, float)


def _global_vote(fixed, moving, rotations, scales, flips, half, centres, radius):
    """Similarity found by local patch voting, robust to large deformation.

    Small patches stay nearly rigid even when the whole field is deformed by
    more than a cell diameter. Under the right rotation and scale many patches
    agree on (nearly) one translation; under a wrong one their votes scatter.
    The score is the largest fraction of patches whose translations lie within
    ``radius`` of each other. A global correlation of the whole image, by
    contrast, is degraded by the deformation and can prefer a wrong pose.
    """
    best = None
    for flip in flips:
        for s in scales:
            for th in rotations:
                A = _similarity(th, s, flip)
                canvas, _, lo, c = _render(moving, A)
                pts, offs, peak = _patch_votes(fixed, canvas, half, centres)
                if len(pts) < 4:
                    continue
                cnt = np.array([len(v) for v in
                                cKDTree(offs).query_ball_point(offs, radius)])
                k = int(np.argmax(cnt))
                frac = cnt[k] / len(pts)
                if best is None or frac > best[0]:
                    best = (frac, th, s, flip, A, lo, c, pts, offs, offs[k], peak)
    return best


def _block_displacements(fixed, warped, valid, grid, half, radius, min_score,
                         local_maps=(), retry_below=0.5, retry_margin=0.1):
    """Per-block shift of ``warped`` that best matches ``fixed`` (template NCC).

    Only the template must lie inside the fixed image; the search window is
    clipped at the border, so blocks close to the edge still contribute.

    ``local_maps`` are 2x2 linear maps (local scale/rotation) tried for blocks
    whose translation-only match scores below ``retry_below``; a map replaces
    the translation-only answer only if it scores ``retry_margin`` higher, so
    look-alike texture under some distortion does not win by a hair. A thin-plate
    deformation stretches and rotates tissue locally -- by 30% and tens of
    degrees where it bends most -- and a translation-only template cannot
    match there. The window is resampled under each map about the block
    centre, matched, and the shift mapped back.
    """
    from skimage.feature import match_template

    H, W = fixed.shape
    pts, disp, score = [], [], []

    def sub(v, i):
        if 0 < i < len(v) - 1:
            d = v[i - 1] - 2 * v[i] + v[i + 1]
            return i + (0.5 * (v[i - 1] - v[i + 1]) / d if abs(d) > 1e-12 else 0.0)
        return float(i)

    for y in range(half, H - half, grid):
        for x in range(half, W - half, grid):
            tpl = fixed[y - half:y + half + 1, x - half:x + half + 1]
            y0, y1 = max(y - half - radius, 0), min(y + half + radius + 1, H)
            x0, x1 = max(x - half - radius, 0), min(x + half + radius + 1, W)
            win = warped[y0:y1, x0:x1]
            if valid[y0:y1, x0:x1].mean() < 0.5 or tpl.std() < 1e-6 or win.std() < 1e-6:
                continue
            res = match_template(win, tpl)
            r, c = np.unravel_index(int(np.argmax(res)), res.shape)
            best = (res[r, c],
                    x0 + sub(res[r, :], c) + half - x,
                    y0 + sub(res[:, c], r) + half - y)
            plain = best[0]
            if best[0] < retry_below and local_maps:
                # local window centred on the block, resampled under L
                ry = np.arange(-half - radius, half + radius + 1)
                gy, gx = np.meshgrid(ry, ry, indexing="ij")
                for L in local_maps:
                    qx = x + L[0, 0] * gx + L[0, 1] * gy
                    qy = y + L[1, 0] * gx + L[1, 1] * gy
                    rw = ndimage.map_coordinates(warped, [qy, qx], order=1, cval=0.0)
                    if rw.std() < 1e-6:
                        continue
                    rr = match_template(rw, tpl)
                    r2, c2 = np.unravel_index(int(np.argmax(rr)), rr.shape)
                    if rr[r2, c2] > best[0] and rr[r2, c2] >= plain + retry_margin:
                        dq = np.array([sub(rr[r2, :], c2) - radius, sub(rr[:, c2], r2) - radius])
                        dd = L @ dq            # shift in the warped frame
                        best = (rr[r2, c2], dd[0], dd[1])
            if best[0] < min_score:
                continue
            pts.append((x, y)); disp.append((best[1], best[2])); score.append(best[0])
    return np.array(pts, float), np.array(disp, float), np.array(score, float)


def _coherent(pts, disp, tol, k=8):
    """Keep displacements predicted by a local linear fit to their neighbours.

    Comparing with the neighbours' *median* fails exactly where the field is
    deformed most: on a strong gradient, and at the border where all the
    neighbours lie on one side, the median is biased and good blocks are
    thrown out. A local affine fit of displacement against position follows
    the gradient and extrapolates correctly to the border.
    """
    n = len(pts)
    if n <= 4:
        return np.ones(n, bool)
    kk = min(k, n - 1)
    _, nb = cKDTree(pts).query(pts, k=kk + 1)
    ok = np.ones(n, bool)
    for i in range(n):
        j = nb[i, 1:]
        X = np.c_[np.ones(len(j)), pts[j] - pts[i]]
        if len(j) >= 4:
            coef, *_ = np.linalg.lstsq(X, disp[j], rcond=None)
            pred = coef[0]                      # value of the local fit at pts[i]
        else:
            pred = np.median(disp[j], axis=0)
        ok[i] = np.linalg.norm(disp[i] - pred) <= tol
    return ok


def register_images(
    fixed_image,
    moving_image,
    *,
    moving_rgb: bool = False,
    fixed_rgb: bool = False,
    rotations: Optional[Sequence[float]] = None,
    scales: Optional[Sequence[float]] = None,
    allow_flip: bool = True,
    downsample: int = 2,
    vote_patch: int = 32,
    vote_radius: float = 12.0,
    max_vote_patches: int = 64,
    levels: Sequence[tuple] = ((4, 8, 8, 6, 6), (2, 10, 8, 6, 4), (1, 18, 10, 6, 2), (1, 14, 8, 3, 1)),
    min_score: float = 0.3,
    smoothing: float = 2.0,
    local_scales: Sequence[float] = (0.75, 0.87, 1.0, 1.15, 1.33),
    local_rotations: Sequence[float] = (-20.0, -10.0, 0.0, 10.0, 20.0),
    local_affine: bool = False,
    local_maps: Optional[Sequence] = None,
    return_info: bool = False,
):
    """Map fixed-image pixel coordinates to moving-image coordinates.

    Parameters
    ----------
    fixed_image, moving_image
        2-D images (or RGB with ``*_rgb=True``, converted with
        :func:`hematoxylin_channel` -- use it for H&E). Both should show the
        same structures bright, e.g. DAPI and haematoxylin.
    rotations, scales, allow_flip
        Global pose grid (default: every 8 deg, scales 0.92/1.0/1.08, with and
        without a mirror flip), refined around the winner.
    downsample
        Factor for the global vote only.
    vote_patch, vote_radius
        Patch size and translation-agreement radius of the global vote, in
        full-resolution pixels. The patch should span a few cells; the radius
        should exceed the deformation *across one patch spacing*, not the total.
    max_vote_patches
        Most-textured patches used for the coarse vote (four times as many for
        the refinement); blank background patches carry no pose information.
    levels
        ``(scale, grid, half_patch, search_radius[, max_passes])`` per
        refinement level, coarse to fine; ``scale`` is a downsampling factor
        and the other three are pixels *at that scale*. Coarse levels see
        large patches of tissue context, so they lock onto large displacements
        that a full-resolution search with the same radius cannot reach. A
        level repeats until a pass adds no blocks (or ``max_passes``): blocks
        matched in one pass pull the field right around them, so their
        neighbours fall inside the search radius in the next pass.
    min_score
        Template-matching NCC below which a refinement block is ignored.
    smoothing
        Spline smoothing per control point; larger = stiffer field.
    local_scales, local_rotations
        Local scale/rotation hypotheses tried for blocks that match poorly by
        translation alone. Where a smooth deformation bends most, tissue is
        locally stretched and turned (by tens of percent and degrees), which a
        translation-only block cannot follow. Used when ``local_affine``.
    local_affine
        Retry poorly matching blocks (translation-only NCC < 0.6) under the
        local scale/rotation hypotheses inside every pyramid level. It follows
        strongly stretched tissue but can also lock onto look-alike texture, so
        :func:`match_cells` runs both settings and keeps the one that yields
        more one-to-one pairs.

    Returns
    -------
    callable
        ``mapping(xy) -> xy`` taking ``(n, 2)`` fixed coordinates (x, y) to
        moving coordinates. With ``return_info=True`` also a dict of scores.
    """
    from scipy.interpolate import RBFInterpolator
    from ._cellmatch import hematoxylin_channel

    fixed = hematoxylin_channel(fixed_image) if fixed_rgb else np.asarray(fixed_image, float)
    moving = hematoxylin_channel(moving_image) if moving_rgb else np.asarray(moving_image, float)
    if fixed.ndim != 2 or moving.ndim != 2:
        raise ValueError("register_images needs 2-D images (pass *_rgb=True for RGB)")
    F, M = _prep(fixed, 1.0), _prep(moving, 1.0)
    rotations = np.arange(0.0, 360.0, 8.0) if rotations is None else np.asarray(rotations)
    scales = np.array([0.92, 1.0, 1.08]) if scales is None else np.asarray(scales)
    flips = (False, True) if allow_flip else (False,)

    # --- global pose by patch voting, on downsampled images ----------------
    ds = max(int(downsample), 1)
    Fd = ndimage.zoom(ndimage.gaussian_filter(F, 0.75 * ds), 1.0 / ds, order=1)
    Md = ndimage.zoom(ndimage.gaussian_filter(M, 0.75 * ds), 1.0 / ds, order=1)
    half = max(int(vote_patch / (2 * ds)), 4)
    rad = max(vote_radius / ds, 2.0)
    coarse_c = _vote_patches(Fd, half, 2 * half, max_vote_patches)
    best = _global_vote(Fd, Md, rotations, scales, flips, half, coarse_c, rad)
    if best is None:
        raise RuntimeError("register_images: no usable patches in the fixed image")
    frac, th, s, flip, A, lo, c, pts, offs, centre, peak = best
    # (1) refine rotation/scale around the winner on the SAME patch set, so
    #     scores stay comparable; (2) at the chosen pose, re-vote with a denser
    #     patch set whose agreeing votes seed the deformation field.
    step = float(np.median(np.diff(np.sort(rotations)))) if len(rotations) > 1 else 8.0
    cand = _global_vote(Fd, Md, th + np.linspace(-step / 2, step / 2, 5),
                        s * np.array([0.97, 1.0, 1.03]), (flip,), half, coarse_c, rad)
    if cand is not None and cand[0] >= frac:
        frac, th, s = cand[0], cand[1], cand[2]
    fine_c = _vote_patches(Fd, half, half, 4 * max_vote_patches)
    _, th, s, flip, A, lo, c, pts, offs, centre, peak = _global_vote(
        Fd, Md, [th], [s], (flip,), half, fine_c, rad)

    # Inlier votes seed the field conservatively: confident, near the consensus
    # translation, and agreeing with their neighbours. Strongly deformed regions
    # (often the borders) are left to the refinement below, which grows the
    # field outward from where it is already right.
    near = np.linalg.norm(offs - centre, axis=1) <= 4.0 * rad
    ok = near & (peak >= min_score) & _coherent(pts, offs, tol=rad)
    if ok.sum() < 4:
        ok = np.linalg.norm(offs - centre, axis=1) <= rad
    pts_f, offs_f = pts[ok] * ds, offs[ok] * ds          # full resolution
    Ai = np.linalg.inv(A)
    lo_f, c_f = lo * ds, c * ds
    if len(pts_f) >= 6:
        u0 = RBFInterpolator(pts_f, offs_f, kernel="thin_plate_spline",
                             smoothing=smoothing * len(pts_f), degree=1)
    else:
        const = np.median(offs_f, axis=0)
        u0 = lambda q: np.broadcast_to(const, q.shape)  # noqa: E731

    def base(p):
        p = np.asarray(p, float)
        return (lo_f + p + u0(p)) @ Ai.T + c_f

    # --- deformable refinement by block matching ---------------------------
    # The accumulated fixed-frame displacement D lives on the pixel grid, so a
    # new correction u composes in one step, D'(p) = u(p) + D(p + u(p)),
    # instead of growing a chain of splines that every evaluation must walk.
    H, W = F.shape
    yy, xx = np.mgrid[0:H, 0:W]
    grid_xy = np.c_[xx.ravel(), yy.ravel()].astype(float)
    Dx = np.zeros((H, W))
    Dy = np.zeros((H, W))

    def _disp_at(q):
        q = np.asarray(q, float)
        dx = ndimage.map_coordinates(Dx, [q[:, 1], q[:, 0]], order=1, mode="nearest")
        dy = ndimage.map_coordinates(Dy, [q[:, 1], q[:, 0]], order=1, mode="nearest")
        return np.c_[dx, dy]

    def mapping(p):
        q = np.asarray(p, float)
        return base(q + _disp_at(q))

    level_info = []
    if local_maps is not None:
        maps = [np.asarray(L, float) for L in local_maps]
    else:
        maps = []
        for sc_ in local_scales:
            for rt in local_rotations:
                if abs(sc_ - 1.0) < 1e-9 and abs(rt) < 1e-9:
                    continue
                maps.append(_similarity(rt, sc_, False))

    def _shrink(img, k):
        return img if k == 1 else ndimage.zoom(ndimage.gaussian_filter(img, 0.5 * k), 1.0 / k, order=1)

    def _one_pass(k, g, hp, radius, coherence):
        q = mapping(grid_xy)
        warped = ndimage.map_coordinates(M, [q[:, 1], q[:, 0]], order=1, cval=0.0).reshape(H, W)
        valid = ((q[:, 0] >= 0) & (q[:, 0] <= M.shape[1] - 1) &
                 (q[:, 1] >= 0) & (q[:, 1] <= M.shape[0] - 1)).reshape(H, W).astype(float)
        Fk, Wk, Vk = _shrink(F, k), _shrink(warped, k), _shrink(valid, k)
        bp, disp, _ = _block_displacements(Fk, Wk, Vk > 0.5, int(g), int(hp),
                                           int(radius), min_score,
                                           local_maps=maps if local_affine else (),
                                           retry_below=0.6, retry_margin=0.0)
        if len(bp) >= 8:
            keep = coherence(bp, disp, tol=max(1.5, 0.35 * radius))
            bp, disp = bp[keep], disp[keep]
        if len(bp) < 8:
            return 0
        bp, disp = bp * k, disp * k                         # back to full resolution
        u = RBFInterpolator(bp, disp, kernel="thin_plate_spline",
                            smoothing=smoothing * len(bp), degree=1)(grid_xy)
        D_new = u + _disp_at(grid_xy + u)
        Dx[:] = D_new[:, 0].reshape(H, W)
        Dy[:] = D_new[:, 1].reshape(H, W)
        return int(len(bp))

    for lv in levels:
        k, g, hp, radius = int(lv[0]), lv[1], lv[2], lv[3]
        max_rep = int(lv[4]) if len(lv) > 4 else 1
        counts = []
        for _ in range(max_rep):
            counts.append(_one_pass(k, g, hp, radius, _coherent))
            # stop once a pass no longer adds blocks: the field has grown as
            # far as this level's search radius can carry it
            if len(counts) > 1 and counts[-1] <= counts[-2]:
                break
        level_info.append({"scale": k, "grid": g, "radius": radius, "passes": len(counts),
                           "n_blocks": counts[-1], "n_blocks_by_pass": counts})

    if not return_info:
        return mapping
    q = mapping(grid_xy)
    warped = ndimage.map_coordinates(M, [q[:, 1], q[:, 0]], order=1, cval=np.nan).reshape(H, W)
    good = ~np.isnan(warped)
    corr = float(np.corrcoef(F[good], warped[good])[0, 1]) if good.sum() > 10 else float("nan")
    return mapping, {"vote_fraction": float(frac), "rotation": float(th), "scale": float(s),
                     "flip": bool(flip), "n_vote_inliers": int(ok.sum()),
                     "levels": level_info, "local_affine": bool(local_affine),
                     "final_correlation": corr,
                     "overlap_fraction": float(good.mean())}
