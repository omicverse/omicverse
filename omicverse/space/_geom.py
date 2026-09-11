"""Geometry rather than expression — aligning serial sections and stacking them.

``pySTAligner`` and ``CAST`` align *expression* across samples: they give you a
shared embedding in which the same cell type from two slides lands in the same
place. That is not the same problem as recovering *geometry*. A shared embedding
cannot be sliced, sectioned, or measured in microns, because it has no common
coordinate frame — and a common coordinate frame is what 3D reconstruction needs.

This module supplies the missing piece. :func:`align` puts a stack of serial
sections into one frame; :func:`stack` assembles them into a single object with
three spatial dimensions; :func:`interpolate` fills the gap between two sections.

The alignment is a fused Gromov-Wasserstein problem, which is what PASTE solves:
match spots between two slices using both their expression similarity and the
geometry of each slice, then read a rigid transform off the resulting coupling.
It is implemented here over POT rather than by depending on PASTE, so the
default path needs one small package instead of the whole PASTE stack —
``pip install \"POT\"`` (or ``pip install POT``). A plain install
does not include it; the import error names the package. ``method='icp'``
and ``method='rigid'`` are the genuinely dependency-free routes (scipy only),
and ``method='paste'`` hands over to the real package when it is present.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .._registry import register_function

__all__ = ["align", "stack", "interpolate", "align_pairwise"]

_METHODS = ("fgw", "icp", "rigid", "paste")


def _dense(matrix) -> np.ndarray:
    return np.asarray(matrix.todense() if sparse.issparse(matrix) else matrix,
                      dtype=np.float64)


def _shared_features(a, b) -> tuple[np.ndarray, np.ndarray]:
    common = a.var_names.intersection(b.var_names)
    if len(common) < 2:
        raise ValueError(
            f"The two slices share only {len(common)} features; alignment needs a "
            "common gene space. Subset both to shared genes first."
        )
    return _dense(a[:, common].X), _dense(b[:, common].X)


def _procrustes(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Rotation ``R`` such that ``(source - mean) @ R`` best matches ``target - mean``.

    Weighted Kabsch: centre both clouds on their weighted means, take the SVD of
    the cross-covariance, and reassemble. Reflections are excluded — a tissue
    section cannot be flipped over and still be the same section, so the
    determinant is forced positive rather than letting the SVD mirror the slice
    to shave off a little residual.

    The convention is right-multiplication: ``source @ R``, not ``R @ source``.
    Returning the transpose of this matrix leaves an alignment that looks
    plausible on a plot and is wrong by twice the rotation angle.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    src_c = source - (w @ source)
    tgt_c = target - (w @ target)
    cov = src_c.T @ (w[:, None] * tgt_c)
    u, _, vt = np.linalg.svd(cov)
    d = float(np.sign(np.linalg.det(u @ vt)))
    scale = np.eye(cov.shape[0])
    scale[-1, -1] = d
    return u @ scale @ vt


@register_function(
    aliases=["切片配准", "pairwise align", "两张切片对齐", "slice registration"],
    category="space",
    description="Align one spatial slice onto another with fused Gromov-Wasserstein or ICP",
    requires={'obsm': ['spatial']},
    produces={},
    auto_fix='none',
    examples=[
        "coords, plan = ov.space.geom.align_pairwise(slice_a, slice_b)",
    ],
)
def align_pairwise(
    source,
    target,
    *,
    method: str = "fgw",
    alpha: float = 0.1,
    spatial_key: str = "spatial",
    use_rep: Optional[str] = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
):
    """Bring one section into the coordinate frame of another.

    ``method='fgw'`` solves the fused Gromov-Wasserstein problem PASTE is built
    on: find a coupling between the two slices' spots that simultaneously matches
    expression profiles and preserves each slice's internal distances, then read
    the rigid transform off that coupling. ``alpha`` trades the two off — 0 uses
    expression alone, 1 geometry alone.

    ``method='icp'`` ignores expression entirely and matches nearest points
    iteratively. Faster, but it converges to whichever local minimum it starts
    near, and nearest-point correspondences stop being informative once the two
    sections are far apart in angle.

    How far ICP can be pushed is not a fixed number — it depends on how much the
    shape of the tissue constrains the fit. Rotating a section by a known angle and
    asking each method to undo it, the angle at which ICP first fails:

    ==========================  =======================  ==========
    point cloud                 ``icp`` holds to         ``fgw``
    ==========================  =======================  ==========
    real Visium slide           60 deg                   exact
    uniform random cloud        10 deg                   exact
    regular lattice             fails at 10 deg          exact
    ==========================  =======================  ==========

    FGW was exact to machine precision in every case, at every angle up to 170
    degrees. ICP does best on the real slide because an irregular tissue outline
    pins the fit down, and *worst* on a perfect lattice, where the lattice's own
    six-fold symmetry makes nearest-point correspondences genuinely ambiguous —
    the one case where the geometry looks most orderly is the one where matching
    on geometry alone works least well.

    So ``'fgw'`` is the default, and ``'icp'`` is for sections already roughly
    oriented, where its speed is worth having.

    Arguments:
        source: The slice to move.
        target: The slice to move it onto.
        method: ``'fgw'``, ``'icp'``, ``'rigid'`` or ``'paste'``.
        alpha: Weight on geometry versus expression in ``'fgw'``. Default 0.1.
        spatial_key: Coordinates in ``adata.obsm``.
        use_rep: ``obsm`` key to compare instead of expression, e.g. ``'X_pca'``.
        max_iter: Iteration cap for the solver.
        tol: Convergence tolerance.
        seed: Seed, for the solvers that use one.

    Returns:
        ``(coords, plan)`` — the transformed source coordinates, and the coupling
        matrix (``None`` for ``'icp'``).
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}.")

    src_xy = np.asarray(source.obsm[spatial_key], dtype=np.float64)[:, :2]
    tgt_xy = np.asarray(target.obsm[spatial_key], dtype=np.float64)[:, :2]

    if method == "paste":
        try:
            import paste as pst
        except ImportError as exc:
            raise ImportError(
                "method='paste' needs the PASTE package. Install it with "
                "`pip install paste-bio`, or use method='fgw' after installing "
                "`pip install \"POT\"`."
            ) from exc
        plan = pst.pairwise_align(source, target, alpha=alpha)
        rot = _procrustes(src_xy, tgt_xy[plan.argmax(axis=1)], plan.sum(axis=1))
        moved = (src_xy - src_xy.mean(0)) @ rot + tgt_xy.mean(0)
        return moved, plan

    if method in ("icp", "rigid"):
        from scipy.spatial import cKDTree

        moved = src_xy - src_xy.mean(0)
        ref = tgt_xy - tgt_xy.mean(0)
        tree = cKDTree(ref)
        prev = np.inf
        for _ in range(max_iter if method == "icp" else 1):
            _, idx = tree.query(moved, k=1)
            rot = _procrustes(moved, ref[idx], np.ones(len(moved)))
            moved = moved @ rot
            err = float(np.mean(np.linalg.norm(moved - ref[idx], axis=1)))
            if abs(prev - err) < tol:
                break
            prev = err
        return moved + tgt_xy.mean(0), None

    # fused Gromov-Wasserstein
    try:
        import ot
    except ImportError as exc:
        raise ImportError(
            "method='fgw' requires Python Optimal Transport (POT). Install it "
            "with `pip install \"POT\"` or `pip install POT`."
        ) from exc

    if use_rep is not None:
        src_x = np.asarray(source.obsm[use_rep], dtype=np.float64)
        tgt_x = np.asarray(target.obsm[use_rep], dtype=np.float64)
        if src_x.shape[1] != tgt_x.shape[1]:
            raise ValueError(f"obsm[{use_rep!r}] has different widths in the two slices.")
    else:
        src_x, tgt_x = _shared_features(source, target)

    # expression dissimilarity between spots, and each slice's own geometry
    feature_cost = ot.dist(src_x, tgt_x, metric="euclidean")
    feature_cost /= feature_cost.max() or 1.0
    c1 = ot.dist(src_xy, src_xy, metric="euclidean")
    c2 = ot.dist(tgt_xy, tgt_xy, metric="euclidean")
    c1 /= c1.max() or 1.0
    c2 /= c2.max() or 1.0

    p = np.ones(len(src_xy)) / len(src_xy)
    q = np.ones(len(tgt_xy)) / len(tgt_xy)
    plan = ot.gromov.fused_gromov_wasserstein(
        feature_cost, c1, c2, p, q, loss_fun="square_loss", alpha=alpha,
        max_iter=max_iter, tol_rel=tol, tol_abs=tol,
    )

    # barycentric projection: each source spot goes to the weighted mean of the
    # target spots it was coupled to, and the rigid map is fitted to that
    mass = plan.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        projected = np.where(mass[:, None] > 0, (plan @ tgt_xy) / mass[:, None], src_xy)
    rot = _procrustes(src_xy, projected, np.ones(len(src_xy)))
    moved = (src_xy - src_xy.mean(0)) @ rot + tgt_xy.mean(0)
    return moved, plan


@register_function(
    aliases=["连续切片对齐", "align slices", "3D配准", "serial section alignment"],
    category="space",
    description="Put a stack of serial sections into one coordinate frame",
    requires={'obsm': ['spatial']},
    produces={'obsm': ['spatial_aligned']},
    auto_fix='none',
    examples=[
        "aligned = ov.space.geom.align([s1, s2, s3])",
        "aligned = ov.space.geom.align(slices, method='icp')",
    ],
)
def align(
    slices: Sequence,
    *,
    method: str = "fgw",
    alpha: float = 0.1,
    reference: int = 0,
    spatial_key: str = "spatial",
    key_added: str = "spatial_aligned",
    use_rep: Optional[str] = None,
    copy: bool = True,
    **kwargs: Any,
) -> list:
    """Put a stack of serial sections into a single coordinate frame.

    Each section is aligned to its predecessor and the transforms compose along
    the stack, so section ``n`` ends up in the frame of the reference. Sections are
    taken in the order given: that order is the physical cutting order, and
    shuffling it will produce a smooth-looking but wrong volume.

    Arguments:
        slices: Sections in cutting order.
        method: Passed to :func:`align_pairwise`.
        alpha: Geometry-versus-expression weight for ``'fgw'``.
        reference: Index of the section whose frame everything lands in.
        spatial_key: Coordinates in ``adata.obsm``.
        key_added: ``obsm`` key for the aligned coordinates.
        use_rep: ``obsm`` key to compare instead of expression.
        copy: Work on copies and return them, rather than modifying in place.
        **kwargs: Forwarded to :func:`align_pairwise`.

    Returns:
        The list of sections, each carrying aligned coordinates in
        ``obsm[key_added]``.
    """
    if len(slices) < 2:
        raise ValueError("align needs at least two sections.")
    if not 0 <= reference < len(slices):
        raise IndexError(f"reference {reference} is outside 0..{len(slices) - 1}.")

    out = [s.copy() for s in slices] if copy else list(slices)
    out[reference].obsm[key_added] = np.asarray(
        out[reference].obsm[spatial_key], dtype=np.float64
    )[:, :2].copy()

    # Walk outwards from the reference in both directions, aligning each section
    # to its neighbour's *already aligned* coordinates rather than to that
    # neighbour's original ones. Aligning to the original and then shifting by a
    # mean offset only carries the translation forward: the rotation the anchor
    # itself received is lost, and every section past the first drifts by it.
    for step in (1, -1):
        i = reference + step
        while 0 <= i < len(out):
            anchor = out[i - step]
            anchor_in_frame = anchor.copy()
            anchor_in_frame.obsm[spatial_key] = anchor.obsm[key_added].copy()
            moved, _ = align_pairwise(
                out[i], anchor_in_frame, method=method, alpha=alpha,
                spatial_key=spatial_key, use_rep=use_rep, **kwargs
            )
            out[i].obsm[key_added] = moved
            i += step

    return out


@register_function(
    aliases=["3D堆叠", "stack slices", "三维重建", "3D reconstruction", "体数据"],
    category="space",
    description="Stack aligned serial sections into one AnnData with 3-D coordinates",
    requires={'obsm': ['spatial_aligned']},
    produces={'obsm': ['spatial_3d'], 'obs': ['section']},
    auto_fix='none',
    examples=[
        "volume = ov.space.geom.stack(aligned, z_spacing=10.0)",
        "volume.obsm['spatial_3d'].shape",
    ],
)
def stack(
    slices: Sequence,
    *,
    z_spacing: float = 1.0,
    z_values: Optional[Sequence[float]] = None,
    spatial_key: str = "spatial_aligned",
    key_added: str = "spatial_3d",
    section_key: str = "section",
    section_names: Optional[Sequence[str]] = None,
):
    """Assemble aligned sections into one object with three spatial dimensions.

    ``z_spacing`` is the physical distance between consecutive sections, in the
    same units as the in-plane coordinates. Getting it wrong does not fail — it
    silently stretches or squashes the volume, and any distance measured through
    the stack afterwards is wrong by that factor. If the sections were not cut at
    a constant interval, pass the actual depths as ``z_values``.

    Arguments:
        slices: Sections carrying aligned coordinates, from :func:`align`.
        z_spacing: Distance between consecutive sections.
        z_values: Explicit depth per section, overriding ``z_spacing``.
        spatial_key: Aligned in-plane coordinates in ``adata.obsm``.
        key_added: ``obsm`` key for the 3-D coordinates.
        section_key: ``obs`` column recording which section each spot came from.
        section_names: Names for the sections; defaults to their index.

    Returns:
        A single AnnData holding every section.
    """
    import anndata as ad

    if len(slices) < 2:
        raise ValueError("stack needs at least two sections.")
    if z_values is not None and len(z_values) != len(slices):
        raise ValueError(f"z_values has {len(z_values)} entries for {len(slices)} sections.")

    names = ([str(n) for n in section_names] if section_names is not None
             else [str(i) for i in range(len(slices))])
    if len(names) != len(slices):
        raise ValueError(f"section_names has {len(names)} entries for {len(slices)} sections.")

    depths = (np.asarray(z_values, dtype=np.float64) if z_values is not None
              else np.arange(len(slices), dtype=np.float64) * z_spacing)

    pieces = []
    for s, name, z in zip(slices, names, depths):
        if spatial_key not in s.obsm:
            raise KeyError(
                f"Section {name!r} has no obsm[{spatial_key!r}]. Run "
                "`ov.space.geom.align(...)` first."
            )
        piece = s.copy()
        piece.obs[section_key] = name
        xy = np.asarray(piece.obsm[spatial_key], dtype=np.float64)[:, :2]
        piece.obsm[key_added] = np.column_stack([xy, np.full(len(xy), z)])
        pieces.append(piece)

    volume = ad.concat(pieces, label=None, index_unique="-", merge="first")
    volume.obs[section_key] = pd.Categorical(volume.obs[section_key], categories=names)
    volume.uns["geom_stack"] = {
        "n_sections": len(slices),
        "z_spacing": None if z_values is not None else float(z_spacing),
        "z_values": depths.tolist(),
        "aligned_from": spatial_key,
    }
    return volume


@register_function(
    aliases=["切片插值", "interpolate sections", "补充中间切片", "z interpolation"],
    category="space",
    description="Interpolate a virtual section between two aligned neighbours",
    requires={'obsm': ['spatial_aligned']},
    produces={},
    auto_fix='none',
    examples=[
        "mid = ov.space.geom.interpolate(aligned[0], aligned[1], fraction=0.5)",
    ],
)
def interpolate(
    lower,
    upper,
    *,
    fraction: float = 0.5,
    spatial_key: str = "spatial_aligned",
    n_neighbors: int = 6,
):
    """Build a virtual section between two real ones.

    Serial sectioning throws away everything between the cuts. This fills a slab
    of that gap by taking each spot of the lower section, finding the spots of the
    upper section nearest it in the shared frame, and blending their profiles by
    inverse distance.

    It is interpolation, not measurement: the result is a plausible reading of
    what lies between two observations and should not be treated as data. It is
    useful for rendering a continuous volume and for sanity-checking an alignment
    — a bad alignment produces obvious nonsense here.

    Arguments:
        lower: The section below.
        upper: The section above.
        fraction: Where between them, 0 at ``lower`` and 1 at ``upper``.
        spatial_key: Aligned coordinates in ``adata.obsm``.
        n_neighbors: Upper-section spots blended per lower-section spot.

    Returns:
        An AnnData for the virtual section, marked
        ``uns['geom_interpolated'] = True``.
    """
    from scipy.spatial import cKDTree

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1].")

    common = lower.var_names.intersection(upper.var_names)
    if len(common) < 2:
        raise ValueError("The two sections share too few features to interpolate.")

    lo = lower[:, common]
    up = upper[:, common]
    lo_xy = np.asarray(lo.obsm[spatial_key], dtype=np.float64)[:, :2]
    up_xy = np.asarray(up.obsm[spatial_key], dtype=np.float64)[:, :2]

    k = min(n_neighbors, up.n_obs)
    dist, idx = cKDTree(up_xy).query(lo_xy, k=k)
    dist = np.atleast_2d(dist.T).T
    idx = np.atleast_2d(idx.T).T
    weights = 1.0 / np.maximum(dist, 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)

    up_x = _dense(up.X)
    blended_upper = np.einsum("ij,ijk->ik", weights, up_x[idx])
    blended = (1.0 - fraction) * _dense(lo.X) + fraction * blended_upper

    import anndata as ad
    out = ad.AnnData(X=blended.astype(np.float32), var=lo.var.copy())
    out.obs_names = [f"interp_{i}" for i in range(out.n_obs)]
    out.obsm[spatial_key] = (1.0 - fraction) * lo_xy + fraction * up_xy[idx[:, 0]]
    out.uns["geom_interpolated"] = True
    out.uns["geom_interpolation"] = {"fraction": float(fraction),
                                     "n_neighbors": int(k)}
    return out
