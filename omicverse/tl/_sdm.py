r"""Species-distribution set prediction (community / multi-species SDMs).

Predict, for every survey (site, plot, sampling event), the **set** of species
present, from presence-absence (PA) surveys plus optional presence-only (PO)
occurrences and site covariates (climate tables, time-series cubes, satellite
embeddings).  The evaluation target is usually a per-survey set metric
(sample-averaged F1, as in GeoLifeCLEF / GeoPlant), so the method has two
parts that both matter:

1. **probabilities** ``P[survey, species]`` — a multi-label network on the
   covariates (with a PO-neighbourhood embedding and an optional per-species
   PO occurrence offset), fused with analog-neighbour community composition
   (nearest training surveys in an embedding space or in geography);
2. **set decoding** — per survey, the number of species is chosen from the
   probabilities (plug-in expected F1), not one fixed top-k for everyone.

Every decision (training length, fusion weights, set-size calibration) is
made on **spatially blocked** validation that imitates the deployment
regime: ``near`` = held-out small geographic cells inside the sampled area,
``far`` = held-out whole groups (countries / regions / programmes) or large
blocks with an exclusion buffer.  Random splits leak near-duplicate plots and
over-state accuracy.

Nothing here is specific to one dataset: coordinates, labels, covariate
blocks, groups and the regime of each query survey are supplied by the
caller.

Functions
---------
``sdm_sample_f1``               per-survey F1 of ranked sets
``sdm_decode_sets``             per-survey set size by plug-in expected F1
``sdm_sets_to_strings``         sorted, space-separated label strings
``sdm_spatial_holdout``         blocked validation splits (cells / groups / blocks)
``sdm_neighbor_composition``    analog community composition (geo or embedding kNN)
``sdm_presence_only_histogram`` distance-weighted PO species histogram per site
``sdm_cube_summary``            summary features of per-site time-series cubes
``sdm_read_tensor_dir``         read per-site tensors (``<id>.pt`` / ``.npy``) into an array
``sdm_standardize``             covariate blocks -> standardised matrix + missing flags
``sdm_fit_mlp``                 multi-label MLP (BCE, optional PO offset), checkpoints
``sdm_fuse``                    geometric fusion of probability sources at fixed mass
``sdm_community_pipeline``      all of the above, decisions on blocked validation
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from .._registry import register_function

__all__ = ["sdm_sample_f1", "sdm_decode_sets", "sdm_sets_to_strings", "sdm_spatial_holdout",
           "sdm_neighbor_composition", "sdm_presence_only_histogram", "sdm_cube_summary",
           "sdm_read_tensor_dir", "sdm_standardize", "sdm_fit_mlp", "sdm_fuse",
           "sdm_community_pipeline"]

EARTH_RADIUS_KM = 6371.0088


# --------------------------------------------------------------------------- helpers
def _xyz(lat, lon) -> np.ndarray:
    """Unit-free 3-D coordinates in km (chord distance ~ great-circle for small d)."""
    la = np.radians(np.asarray(lat, float))
    lo = np.radians(np.asarray(lon, float))
    return EARTH_RADIUS_KM * np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def _as_csr(y) -> sparse.csr_matrix:
    y = sparse.csr_matrix(y)
    y.data = (y.data != 0).astype(np.float32)
    y.eliminate_zeros()
    return y


def _cells(lat, lon, km: float) -> np.ndarray:
    """Integer id of an approximately ``km`` x ``km`` grid cell."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    dy = km / 111.32
    iy = np.floor(lat / dy)
    dx = km / (111.32 * np.maximum(np.cos(np.radians((iy + .5) * dy)), 1e-3))
    ix = np.floor(lon / dx)
    return pd.factorize(pd.Series(iy.astype(np.int64)).astype(str) + "_" +
                        pd.Series(ix.astype(np.int64)).astype(str))[0]


def _print(msg):
    print(msg, flush=True)


def _log(log, msg):
    if log is not None:
        log(msg)


def _topk(P: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = int(min(k, P.shape[1]))
    ix = np.argpartition(-P, k - 1, axis=1)[:, :k]
    v = np.take_along_axis(P, ix, 1)
    o = np.argsort(-v, axis=1, kind="stable")
    return np.take_along_axis(ix, o, 1), np.take_along_axis(v, o, 1)


# --------------------------------------------------------------------------- metric + decoding
@register_function(
    aliases=["样本平均F1", "sample F1", "samples-averaged F1", "set F1 per survey"],
    category="tl",
    description="Per-survey F1 of predicted species sets given ranked candidates and set sizes "
                "(or boolean prediction matrix) against a sparse truth matrix; richness may "
                "include species outside the candidate columns",
    examples=["f = ov.tl.sdm_sample_f1(ranking, sizes, Y_val); f.mean()"],
    related=["tl.sdm_decode_sets"],
)
def sdm_sample_f1(ranking: np.ndarray, sizes: np.ndarray, truth, richness=None) -> np.ndarray:
    """F1 of each row: the first ``sizes[i]`` columns of ``ranking[i]`` vs ``truth[i]``.

    ``truth`` is an (n, S) 0/1 matrix (sparse or dense).  ``richness`` (n,) overrides the
    true set size (e.g. when some true species are outside the S columns).
    """
    truth = _as_csr(truth)
    ranking = np.asarray(ranking)
    sizes = np.clip(np.asarray(sizes, int), 0, ranking.shape[1])
    n = len(ranking)
    hit = np.asarray(truth[np.arange(n)[:, None], ranking].todense()).reshape(ranking.shape)
    cs = np.cumsum(hit, axis=1)
    h = np.where(sizes > 0, cs[np.arange(n), np.maximum(sizes, 1) - 1], 0)
    rich = np.asarray(truth.sum(1)).ravel() if richness is None else np.asarray(richness, float)
    return 2 * h / np.maximum(sizes + rich, 1)


@register_function(
    aliases=["物种集合解码", "set-size decoding", "expected F1 top-k", "per-survey top-k",
             "plug-in F1 decoder"],
    category="tl",
    description="Turn per-survey species probabilities into prediction sets: rank species and "
                "choose each survey's own set size maximising the plug-in expected F1 "
                "2*sum_top_k(p)/(k + expected richness); expected richness = probability mass "
                "(optionally rescaled or supplied)",
    examples=["ranking, sizes = ov.tl.sdm_decode_sets(P)",
              "ranking, sizes = ov.tl.sdm_decode_sets(P, mass_scale=0.9, kmax=150)"],
    related=["tl.sdm_sample_f1", "tl.sdm_sets_to_strings"],
)
def sdm_decode_sets(prob: np.ndarray, mass: Optional[np.ndarray] = None, mass_scale: float = 1.0,
                    kmax: int = 128, min_size: int = 1, size_scale: float = 1.0
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Ranked candidate columns (n, kmax) and per-row set sizes.

    The utility of predicting the top ``k`` is approximated by
    ``2 * cumsum(p)[k] / (k + m)`` where ``m`` is the expected number of true species
    (``mass`` or ``mass_scale * prob.sum(1)``).  ``size_scale`` multiplies the chosen
    size (a calibration knob that validation may set; 1 = plug-in optimum).
    """
    prob = np.asarray(prob, np.float32)
    ranking, vals = _topk(prob, kmax)
    m = prob.sum(1) if mass is None else np.asarray(mass, float)
    m = mass_scale * m
    ks = np.arange(1, vals.shape[1] + 1, dtype=np.float32)
    util = 2 * np.cumsum(vals, 1) / (ks[None] + m[:, None])
    sizes = util.argmax(1) + 1
    if size_scale != 1.0:
        sizes = np.rint(sizes * size_scale).astype(int)
    sizes = np.clip(sizes, min_size, vals.shape[1])
    return ranking, sizes


@register_function(
    aliases=["预测集合转字符串", "prediction strings", "space separated species list"],
    category="tl",
    description="Convert ranked column indices + set sizes into ascending, de-duplicated, "
                "space-separated label strings (one per survey) for a submission table",
    examples=["preds = ov.tl.sdm_sets_to_strings(ranking, sizes, species_ids)"],
    related=["tl.sdm_decode_sets"],
)
def sdm_sets_to_strings(ranking: np.ndarray, sizes: np.ndarray, labels: Sequence) -> List[str]:
    """One string per row: ``labels`` of the first ``sizes[i]`` ranked columns, unique and in
    ascending order, separated by single spaces (empty string for size 0)."""
    labels = np.asarray(labels)
    out = []
    for r, s in zip(ranking, sizes):
        chosen = np.unique(labels[r[:int(s)]])
        out.append(" ".join(str(v) for v in np.sort(chosen)))
    return out


# --------------------------------------------------------------------------- validation design
@register_function(
    aliases=["空间分块验证", "spatial block cross-validation", "spatial holdout",
             "blocked validation", "group holdout"],
    category="tl",
    description="Spatially blocked validation split for SDMs: 'cells' holds out a fraction of "
                "small grid cells inside every group (within-range / IID-like); 'groups' holds "
                "out whole groups such as countries (transfer to new regions); 'blocks' holds "
                "out large grid blocks; an exclusion buffer removes training rows near "
                "validation rows. Returns fit / validation row indices and per-row weights",
    examples=["fit, val, w = ov.tl.sdm_spatial_holdout(lat, lon, groups=country, scheme='cells', cell_km=10)",
              "fit, val, w = ov.tl.sdm_spatial_holdout(lat, lon, groups=country, scheme='groups', buffer_km=10)"],
    related=["tl.sdm_community_pipeline"],
)
def sdm_spatial_holdout(lat, lon, *, groups=None, scheme: str = "cells", cell_km: float = 10.0,
                        block_km: float = 100.0, fraction: float = 0.08,
                        buffer_km: Optional[float] = None, min_group: int = 100,
                        max_val: Optional[int] = None, seed: int = 0
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blocked split.

    scheme='cells'  : in each group hold out ``fraction`` of its ``cell_km`` cells
                      (buffer default 0.5 km) — validation sites have training sites nearby
                      but never the same plot.
    scheme='groups' : hold out whole groups (not the largest one), drawn at random among
                      groups with at least ``min_group`` rows until ``fraction`` of the rows is
                      held out (buffer default ``cell_km``).  Weights ~ group_size**-0.5 so
                      one big held-out group does not decide everything.
    scheme='blocks' : hold out ``fraction`` of ``block_km`` blocks (buffer default ``cell_km``).
    """
    rng = np.random.default_rng(seed)
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    n = len(lat)
    g = np.zeros(n, int) if groups is None else pd.factorize(pd.Series(groups).astype(str))[0]
    if scheme == "groups" and len(np.unique(g)) < 3:
        scheme = "blocks"
    if scheme == "cells":
        cells = _cells(lat, lon, cell_km)
        held = np.zeros(n, bool)
        for gg in np.unique(g):
            uc = np.unique(cells[g == gg])
            if len(uc) < 3:
                continue
            pick = rng.choice(uc, max(1, int(round(fraction * len(uc)))), replace=False)
            held |= (g == gg) & np.isin(cells, pick)
        buf = 0.5 if buffer_km is None else buffer_km
    elif scheme == "groups":
        sizes = pd.Series(g).value_counts()
        cand = [k for k in sizes.index[1:] if sizes[k] >= min_group]
        if not cand:
            cand = list(sizes.index[1:])
        order = rng.permutation(cand)
        held = np.zeros(n, bool)
        for k in order:
            if held.sum() >= fraction * n:
                break
            if held.sum() + sizes[k] > 2.5 * fraction * n and held.any():
                continue
            held |= g == k
        buf = cell_km if buffer_km is None else buffer_km
    elif scheme == "blocks":
        blocks = _cells(lat, lon, block_km)
        ub = np.unique(blocks)
        pick = rng.choice(ub, max(1, int(np.ceil(fraction * len(ub)))), replace=False)
        held = np.isin(blocks, pick)
        buf = cell_km if buffer_km is None else buffer_km
    else:
        raise ValueError("scheme must be 'cells', 'groups' or 'blocks'")
    val = np.flatnonzero(held)
    fit = np.flatnonzero(~held)
    if len(val) == 0 or len(fit) == 0:
        raise ValueError("empty fit or validation set; change fraction / scheme")
    xyz = _xyz(lat, lon)
    if buf and buf > 0:
        d = cKDTree(xyz[val]).query(xyz[fit], k=1)[0]
        fit = fit[d > buf]
    if max_val is not None and len(val) > max_val:
        # keep every group represented
        keep = []
        per = max(1, max_val // max(1, len(np.unique(g[val]))))
        for gg in np.unique(g[val]):
            rr = val[g[val] == gg]
            keep.extend(rng.choice(rr, min(len(rr), per), replace=False))
        keep = np.asarray(keep)
        if len(keep) < max_val:
            rest = np.setdiff1d(val, keep)
            keep = np.r_[keep, rng.choice(rest, min(len(rest), max_val - len(keep)), replace=False)]
        val = np.sort(keep[:max_val])
    cnt = pd.Series(g[val]).value_counts()
    if scheme == "cells":
        w = np.ones(len(val))
    else:
        w = np.array([cnt[x] ** -0.5 for x in g[val]], float)
        w /= w.mean()
    return np.sort(fit), val, w


# --------------------------------------------------------------------------- neighbour sources
@register_function(
    aliases=["近邻群落组成", "analog community", "kNN species composition",
             "habitat analog neighbours", "embedding nearest neighbours"],
    category="tl",
    description="Community composition of each query site as the similarity-weighted mean of "
                "the species sets of its nearest reference surveys, either in geography "
                "(great-circle, inverse distance) or in a covariate / satellite embedding "
                "(cosine similarity, exp kernel)",
    examples=["P = ov.tl.sdm_neighbor_composition(Y_fit, emb[fit], emb[q], metric='cosine', k=64)",
              "P = ov.tl.sdm_neighbor_composition(Y_fit, (lat[fit], lon[fit]), (lat[q], lon[q]), metric='geographic', k=30)"],
    related=["tl.sdm_community_pipeline"],
)
def sdm_neighbor_composition(labels, reference, query, *, metric: str = "cosine", k: int = 64,
                             tau: float = 0.15, min_km: float = 0.5, batch: int = 2048,
                             n_threads: int = 4) -> np.ndarray:
    """(n_query, S) dense composition.

    metric='cosine': ``reference`` / ``query`` are (n, d) embeddings; weight
    ``exp(-(2-2cos)/tau)``.  metric='geographic': (lat, lon) tuples; weight
    ``1/max(d_km, min_km)``.
    """
    Y = _as_csr(labels)
    if metric == "geographic":
        rx = _xyz(*reference)
        qx = _xyz(*query)
        k = min(k, len(rx))
        d, ix = cKDTree(rx).query(qx, k=k, workers=n_threads)
        d, ix = d.reshape(len(qx), k), ix.reshape(len(qx), k)
        w = 1.0 / np.maximum(d, min_km)
    elif metric == "cosine":
        R = np.nan_to_num(np.asarray(reference, np.float32))
        Q = np.nan_to_num(np.asarray(query, np.float32))
        R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-8)
        Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        k = min(k, len(R))
        ix = np.empty((len(Q), k), np.int64)
        w = np.empty((len(Q), k), np.float32)
        try:
            import torch
            Rt = torch.from_numpy(R)
            for s in range(0, len(Q), batch):
                v, i = (torch.from_numpy(Q[s:s + batch]) @ Rt.T).topk(k, 1)
                ix[s:s + batch] = i.numpy()
                w[s:s + batch] = np.exp(-(2 - 2 * v.numpy()) / tau)
        except ImportError:
            for s in range(0, len(Q), batch):
                sim = Q[s:s + batch] @ R.T
                i = np.argpartition(-sim, k - 1, 1)[:, :k]
                ix[s:s + batch] = i
                w[s:s + batch] = np.exp(-(2 - 2 * np.take_along_axis(sim, i, 1)) / tau)
    else:
        raise ValueError("metric must be 'cosine' or 'geographic'")
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    S = sparse.csr_matrix((w.ravel().astype(np.float32),
                           (np.repeat(np.arange(len(ix)), ix.shape[1]), ix.ravel())),
                          shape=(len(ix), Y.shape[0]))
    return (S @ Y).toarray().astype(np.float32)


@register_function(
    aliases=["仅出现记录邻域", "presence-only neighbourhood", "PO histogram",
             "occurrence density around site"],
    category="tl",
    description="For each site, the inverse-distance-weighted species histogram of its k "
                "nearest presence-only occurrence records (sparse, rows sum to 1) and the "
                "distances to the 1st / k-th record (record density); PO silence is not an "
                "absence, use as covariate / offset, never as negative labels",
    examples=["H, dist = ov.tl.sdm_presence_only_histogram(lat, lon, po.lat, po.lon, po_col, n_species, k=512)"],
    related=["tl.sdm_community_pipeline"],
)
def sdm_presence_only_histogram(lat, lon, po_lat, po_lon, po_species, n_species: int, *,
                                k: int = 512, min_km: float = 0.5, batch: int = 4096,
                                n_threads: int = 4) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """``po_species`` = integer column (0..n_species-1) of every PO record; records with a
    negative column (species outside the modelled set) are dropped."""
    po_species = np.asarray(po_species)
    ok = (po_species >= 0) & np.isfinite(np.asarray(po_lat, float)) & np.isfinite(np.asarray(po_lon, float))
    tree = cKDTree(_xyz(np.asarray(po_lat)[ok], np.asarray(po_lon)[ok]))
    sp = po_species[ok].astype(np.int64)
    q = _xyz(lat, lon)
    k = int(min(k, ok.sum()))
    blocks, dist = [], []
    for s in range(0, len(q), batch):
        d, ix = tree.query(q[s:s + batch], k=k, workers=n_threads)
        d, ix = d.reshape(-1, k), ix.reshape(-1, k)
        dist.append(np.c_[d[:, 0], d[:, k // 4], d[:, -1]])
        w = 1.0 / np.maximum(d, min_km)
        w /= w.sum(1, keepdims=True)
        blocks.append(sparse.csr_matrix((w.ravel().astype(np.float32),
                                         (np.repeat(np.arange(len(ix)), k), sp[ix].ravel())),
                                        shape=(len(ix), n_species)))
    H = sparse.vstack(blocks).tocsr()
    H.sum_duplicates()
    return H, np.concatenate(dist).astype(np.float32)


# --------------------------------------------------------------------------- covariates
@register_function(
    aliases=["时间序列立方体特征", "time-series cube features", "satellite time series summary",
             "monthly climate cube summary"],
    category="tl",
    description="Summary features of per-site time-series cubes (n, ..., T): mean, standard "
                "deviation and recent-period mean over the time axis, plus the mean over any "
                "extra cyclic axis (e.g. seasons/months) kept as a climatology; missing cubes "
                "become NaN rows",
    examples=["F = ov.tl.sdm_cube_summary(landsat, time_axis=-1, missing=landsat_missing)",
              "F = ov.tl.sdm_cube_summary(bioclim, time_axis=2, missing=bio_missing)"],
    related=["tl.sdm_read_tensor_dir", "tl.sdm_standardize"],
)
def sdm_cube_summary(cube: np.ndarray, time_axis: int = -1, recent: Optional[int] = None,
                     missing: Optional[np.ndarray] = None) -> np.ndarray:
    """(n, f) features: mean, std and mean of the last ``recent`` steps (default T // 4) over
    ``time_axis``, each flattened over the remaining axes (bands x seasons, variables x
    months ...).  Rows flagged in ``missing`` are NaN."""
    cube = np.asarray(cube, np.float32)
    n = cube.shape[0]
    ax = time_axis % cube.ndim
    if ax == 0:
        raise ValueError("time_axis cannot be the site axis")
    T = cube.shape[ax]
    recent = max(1, T // 4) if recent is None else recent
    c = np.moveaxis(cube, ax, -1)
    feats = [c.mean(-1).reshape(n, -1), c.std(-1).reshape(n, -1),
             c[..., -recent:].mean(-1).reshape(n, -1)]
    out = np.concatenate(feats, 1)
    if missing is not None:
        out[np.asarray(missing, bool)] = np.nan
    return out


@register_function(
    aliases=["读取逐站点张量", "read per-site tensors", "load time series cubes"],
    category="tl",
    description="Read a directory tree of per-site arrays named '<id>.pt', '<id>_cube.pt' or "
                "'<id>.npy' into one (n_ids, *shape) float32 array in the order of the given "
                "ids, with a missing mask; NaN/inf -> 0",
    examples=["cube, miss = ov.tl.sdm_read_tensor_dir('/data/cubes', survey_ids)"],
    related=["tl.sdm_cube_summary"],
)
def sdm_read_tensor_dir(directory, ids, shape: Optional[tuple] = None,
                        log: Optional[Callable[[str], None]] = _print) -> Tuple[np.ndarray, np.ndarray]:
    """Returns ``(array (len(ids), *shape), missing (len(ids),))``.  The site id is the integer
    file stem (an optional ``_cube`` suffix is allowed); files whose id is not in ``ids`` or
    whose shape differs from the first array are skipped."""
    import re
    from pathlib import Path
    rows = {int(v): i for i, v in enumerate(np.asarray(ids).astype(np.int64))}
    paths = [p for p in Path(directory).rglob("*") if p.suffix in (".pt", ".npy")]
    out = None
    miss = np.ones(len(rows), bool)
    t0 = time.time()
    for p in paths:
        m = re.fullmatch(r"(\d+)(?:_cube)?", p.stem)
        if m is None or int(m[1]) not in rows:
            continue
        if p.suffix == ".npy":
            v = np.load(p)
        else:
            import torch
            v = torch.load(p, weights_only=True, map_location="cpu")
            if isinstance(v, dict):
                v = [t for t in v.values() if hasattr(t, "shape")][0]
            v = torch.as_tensor(v, dtype=torch.float32).numpy()
        v = np.nan_to_num(np.asarray(v, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if out is None:
            shape = tuple(v.shape) if shape is None else tuple(shape)
            out = np.zeros((len(rows), *shape), np.float32)
        if v.shape != out.shape[1:]:
            continue
        r = rows[int(m[1])]
        out[r] = v
        miss[r] = False
    if out is None:
        out = np.zeros((len(rows), *(shape or (1,))), np.float32)
    _log(log, f"sdm_read_tensor_dir: {int((~miss).sum())}/{len(rows)} arrays from {directory} "
              f"in {time.time() - t0:.0f}s")
    return out, miss


@register_function(
    aliases=["协变量标准化", "standardize covariates", "feature matrix with missing flags"],
    category="tl",
    description="Stack covariate blocks into one float32 matrix: per-column standardisation "
                "fitted on the fit rows only, NaN -> 0 plus one missing-indicator column per "
                "block that has missing values, values clipped to +-clip",
    examples=["X = ov.tl.sdm_standardize([env, alpha, cube_feats], fit_rows=train_idx)"],
    related=["tl.sdm_fit_mlp"],
)
def sdm_standardize(blocks: Sequence[np.ndarray], fit_rows=None, clip: float = 6.0) -> np.ndarray:
    """Concatenate z-scored blocks (mean / sd over ``fit_rows``); NaN -> 0 and a 0/1 column per
    block flagging rows with any missing value in that block."""
    out = []
    for b in blocks:
        b = np.asarray(b, np.float32)
        if b.ndim == 1:
            b = b[:, None]
        b = np.where(np.isfinite(b), b, np.nan)
        ref = b if fit_rows is None else b[fit_rows]
        with np.errstate(all="ignore"):
            mu = np.nanmean(ref, 0)
            sd = np.nanstd(ref, 0)
        mu = np.nan_to_num(mu)
        sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
        m = np.isnan(b)
        z = np.clip(np.nan_to_num((b - mu) / sd), -clip, clip)
        out.append(z)
        if m.any():
            out.append(m.any(1, keepdims=True).astype(np.float32))
    return np.concatenate(out, 1).astype(np.float32)


# --------------------------------------------------------------------------- neural model
class _Offset:
    """Per-species log PO intensity relative to the fit-row background (sparse-cached)."""

    def __init__(self, H, fit_rows):
        H = sparse.csr_matrix(H, dtype=np.float32)
        bg = np.asarray(H[fit_rows].mean(0)).ravel().astype(np.float32) + 1e-6
        self.bg = bg / bg.sum()
        self.H = H.copy()
        self.H.sum_duplicates()
        self.base = np.clip(np.log(np.full_like(self.bg, .1 / 1.1)), -5, 7)
        cols = self.H.indices
        self.H.data = np.clip(np.log((self.H.data + .1 * self.bg[cols]) / (1.1 * self.bg[cols])), -5, 7)

    def dense(self, rows):
        sub = self.H[rows]
        v = np.broadcast_to(self.base, sub.shape).astype(np.float32, copy=True)
        r = np.repeat(np.arange(sub.shape[0]), np.diff(sub.indptr))
        v[r, sub.indices] = sub.data
        return v


class SDMNetwork:
    """Fitted multi-label MLP; ``predict(rows)`` returns probabilities."""

    def __init__(self, nets, X, offset, n_threads):
        self.nets, self.X, self.offset, self.n_threads = nets, X, offset, n_threads

    def predict(self, rows, batch: int = 2048) -> np.ndarray:
        import torch
        torch.set_num_threads(self.n_threads)
        rows = np.asarray(rows)
        out = None
        for net in self.nets:
            net.eval()
            parts = []
            with torch.no_grad():
                for s in range(0, len(rows), batch):
                    r = rows[s:s + batch]
                    o = None if self.offset is None else torch.from_numpy(self.offset.dense(r))
                    parts.append(torch.sigmoid(net(torch.from_numpy(self.X[r]), o)).numpy())
            p = np.concatenate(parts)
            out = p if out is None else out + p
        return (out / len(self.nets)).astype(np.float32)


def _make_net(d, S, hidden, dropout, use_offset):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.SiLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                      nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(dropout))
            self.head = nn.Linear(hidden, S)
            self.off_w = nn.Parameter(torch.zeros(S)) if use_offset else None

        def forward(self, x, off=None):
            z = self.head(self.body(x))
            if self.off_w is not None and off is not None:
                z = z + self.off_w * off
            return z

    return Net()


@register_function(
    aliases=["多标签物种分布网络", "multi-label SDM", "community MLP", "joint species distribution MLP"],
    category="tl",
    description="Fit a multi-label MLP (2 hidden layers, BCE over all species) on standardised "
                "site covariates, optionally with a learned per-species presence-only "
                "occurrence offset; can evaluate probabilities on validation rows at chosen "
                "epochs (checkpoint selection) and average several seeds",
    examples=["net, hist = ov.tl.sdm_fit_mlp(X, Y, fit_rows, epochs=12, checkpoints=[6, 9, 12], eval_rows=val)",
              "P = net.predict(query_rows)"],
    related=["tl.sdm_community_pipeline", "tl.sdm_standardize"],
)
def sdm_fit_mlp(X: np.ndarray, Y, fit_rows, *, epochs: int = 12, hidden: int = 2048,
                dropout: float = 0.2, lr: float = 2e-3, weight_decay: float = 1e-4,
                batch_size: int = 512, seeds: Sequence[int] = (0,), offset=None,
                checkpoints: Optional[Sequence[int]] = None, eval_rows=None,
                n_threads: int = 4, log: Optional[Callable[[str], None]] = _print):
    """Returns ``(SDMNetwork, {epoch: probabilities on eval_rows})``.

    ``Y`` is (n_rows_of_X, S) — rows not in ``fit_rows`` are never read.  ``offset`` is a
    sparse (n_rows_of_X, S) PO histogram (see ``sdm_presence_only_histogram``); its
    background is computed on ``fit_rows`` only.  One-cycle learning rate over ``epochs``; a
    checkpoint epoch is evaluated in-flight, so early checkpoints share one training run.
    """
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(n_threads)
    X = np.ascontiguousarray(X, np.float32)
    Y = _as_csr(Y)
    fit_rows = np.asarray(fit_rows)
    off = None if offset is None else _Offset(offset, fit_rows)
    checkpoints = sorted(set(checkpoints or [])) if eval_rows is not None else []
    hist: Dict[int, np.ndarray] = {}
    nets = []
    Xt = torch.from_numpy(X)
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = _make_net(X.shape[1], Y.shape[1], hidden, dropout, off is not None)
        opt = torch.optim.AdamW(net.parameters(), lr / 2, weight_decay=weight_decay)
        steps = epochs * int(np.ceil(len(fit_rows) / batch_size))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
        t0 = time.time()
        for ep in range(1, epochs + 1):
            net.train()
            perm = rng.permutation(fit_rows)
            tot = 0.0
            for s in range(0, len(perm), batch_size):
                b = perm[s:s + batch_size]
                if len(b) < 2:
                    continue
                yb = torch.from_numpy(Y[b].toarray())
                o = None if off is None else torch.from_numpy(off.dense(b))
                loss = F.binary_cross_entropy_with_logits(net(Xt[b], o), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
                tot += float(loss) * len(b)
            _log(log, f"sdm_fit_mlp seed {seed} epoch {ep}/{epochs} loss {tot / len(perm):.5f} "
                      f"({time.time() - t0:.0f}s)")
            if ep in checkpoints:
                p = SDMNetwork([net], X, off, n_threads).predict(eval_rows)
                hist[ep] = p if ep not in hist else hist[ep] + p
        nets.append(net)
    for ep in hist:
        hist[ep] /= len(seeds)
    return SDMNetwork(nets, X, off, n_threads), hist


# --------------------------------------------------------------------------- fusion
@register_function(
    aliases=["概率融合", "probability fusion", "geometric ensemble of SDM sources"],
    category="tl",
    description="Fuse a base probability matrix with auxiliary sources (e.g. analog-neighbour "
                "compositions) by a weighted geometric mean, rescaled to the base source's "
                "per-row probability mass (so set-size decoding keeps its calibration)",
    examples=["P = ov.tl.sdm_fuse(P_mlp, [P_knn], [0.3])"],
    related=["tl.sdm_community_pipeline", "tl.sdm_decode_sets"],
)
def sdm_fuse(base: np.ndarray, sources: Sequence[np.ndarray] = (), weights: Sequence[float] = (),
             eps: float = 1e-4) -> np.ndarray:
    """``exp((1 - sum w) log base + sum_j w_j log(source_j + eps))`` per row, rescaled so each
    row keeps the probability mass of ``base``; weights 0 return ``base`` unchanged."""
    base = np.asarray(base, np.float32)
    wsum = float(sum(weights))
    if wsum <= 0:
        return base
    logp = (1 - wsum) * np.log(base + 1e-6)
    for s, w in zip(sources, weights):
        if w:
            logp = logp + w * np.log(np.asarray(s, np.float32) + eps)
    G = np.exp(logp - logp.max(1, keepdims=True))
    G *= (base.sum(1) / np.maximum(G.sum(1), 1e-12))[:, None]
    return np.clip(G, 0, 1).astype(np.float32)


def _tune(base_by_ckpt, sources, truth, weights, rich=None, grid=(0, .1, .2, .3, .4, .5),
          mass_grid=(0.8, 0.9, 1.0, 1.1, 1.25), kmax=128):
    """Coordinate search of checkpoint, source weights and mass scale on validation."""
    w8 = np.asarray(weights, float)

    def score(P, ms=1.0):
        r, s = sdm_decode_sets(P, mass_scale=ms, kmax=kmax)
        return float(np.average(sdm_sample_f1(r, s, truth, rich), weights=w8))

    best = None
    for ep, P in base_by_ckpt.items():
        sc = score(P)
        if best is None or sc > best["score"]:
            best = dict(score=sc, epoch=ep, weights=[0.0] * len(sources), mass_scale=1.0)
    P0 = base_by_ckpt[best["epoch"]]
    ws = [0.0] * len(sources)
    for _ in range(2):
        for j in range(len(sources)):
            for g in grid:
                trial = list(ws)
                trial[j] = g
                if sum(trial) >= 0.9:
                    continue
                sc = score(sdm_fuse(P0, sources, trial))
                if sc > best["score"] + 1e-5:
                    best.update(score=sc, weights=list(trial))
                    ws = list(trial)
    Pf = sdm_fuse(P0, sources, ws)
    for ms in mass_grid:
        sc = score(Pf, ms)
        if sc > best["score"] + 1e-5:
            best.update(score=sc, mass_scale=ms)
    best["plugin_only_score"] = score(P0)
    return best


# --------------------------------------------------------------------------- pipeline
@register_function(
    aliases=["物种群落集合预测流程", "species set prediction pipeline", "GeoLifeCLEF pipeline",
             "plant community prediction", "presence-absence presence-only SDM pipeline",
             "multi-species distribution model"],
    category="tl",
    description="End-to-end per-survey species-set prediction from presence-absence surveys "
                "(+ optional presence-only records, covariate tables, time-series cube "
                "features, satellite embeddings): PO-neighbourhood embedding + per-species PO "
                "offset, multi-label MLP, analog-neighbour compositions (embedding and "
                "geographic), training length / fusion weights / set-size calibration chosen "
                "separately for 'near' (held-out cells) and 'far' (held-out groups) regimes on "
                "spatially blocked validation, final refit on all surveys, plug-in expected-F1 "
                "set sizes per survey; returns prediction strings, probabilities and a report",
    examples=[
        "res = ov.tl.sdm_community_pipeline(train_lat, train_lon, Y, query_lat, query_lon, "
        "covariates=[env_all, cube_feats_all], embedding=alpha_all, groups=train_country, "
        "query_regime=np.where(is_ood, 'far', 'near'), po=(po.lat, po.lon, po_col), "
        "species_ids=species_ids, n_threads=4)",
        "sub = pd.DataFrame({'surveyId': query_ids, 'predictions': res['predictions']})",
        "print(res['report'])",
    ],
    related=["tl.sdm_spatial_holdout", "tl.sdm_fit_mlp", "tl.sdm_decode_sets",
             "tl.sdm_neighbor_composition", "tl.sdm_presence_only_histogram",
             "tl.sdm_cube_summary"],
)
def sdm_community_pipeline(train_lat, train_lon, Y, query_lat, query_lon, *,
                           covariates: Sequence[np.ndarray] = (),
                           embedding: Optional[np.ndarray] = None,
                           groups=None, query_regime=None,
                           po: Optional[Tuple] = None, species_ids=None,
                           epochs: int = 12, hidden: int = 2048, final_seeds: Sequence[int] = (0, 1),
                           po_svd_dim: int = 128, po_k: int = 512, use_po_offset: bool = True,
                           near_cell_km: float = 10.0, near_fraction: float = 0.08,
                           far_fraction: float = 0.1, far_buffer_km: float = 10.0,
                           max_val: int = 6000, validate: Sequence[str] = ("near", "far"),
                           kmax: int = 128, n_threads: int = 4, seed: int = 0,
                           log: Optional[Callable[[str], None]] = _print) -> Dict:
    """Predict a species set for every query survey.

    Parameters
    ----------
    train_lat, train_lon, Y : coordinates of the n training PA surveys and their (n, S) 0/1
        species matrix (sparse ok).  Columns are the modelled species (``species_ids``).
    query_lat, query_lon : coordinates of the m query surveys.
    covariates : list of (n + m, d_i) arrays, rows = training surveys then query surveys
        (climate / soil / land-cover tables, ``sdm_cube_summary`` outputs, survey effort such
        as plot area, year ...).  NaN = missing.  Coordinates are added automatically.
    embedding : (n + m, d) array (e.g. satellite embedding) used for analog-neighbour
        composition (cosine) and as covariates.
    groups : (n,) labels of training surveys (country, region, data programme) — the unit
        held out by the 'far' validation.
    query_regime : (m,) 'near' / 'far' per query survey.  Default: 'far' when the nearest
        training survey is farther than ``far_buffer_km``.
    po : (lat, lon, column) of presence-only records; column = index into the S species
        (negative = not modelled).  Used for a PO-neighbourhood SVD embedding (covariates) and
        a per-species occurrence offset in the network.
    final_seeds : seeds averaged in the final refit on all training surveys.

    Returns dict: ``predictions`` (m strings, ascending ids; needs ``species_ids``),
    ``ranking``, ``sizes``, ``prob`` (m, S float16), ``regime``, ``report`` (text),
    ``decisions`` (per regime: epoch, fusion weights, mass scale, validation F1).
    """
    t_start = time.time()
    Y = _as_csr(Y)
    n, S = Y.shape
    kmax = int(min(kmax, S))
    tl_lat = np.r_[np.asarray(train_lat, float), np.asarray(query_lat, float)]
    tl_lon = np.r_[np.asarray(train_lon, float), np.asarray(query_lon, float)]
    m = len(tl_lat) - n
    train = np.arange(n)
    query = np.arange(n, n + m)
    xyz = _xyz(tl_lat, tl_lon)
    report = [f"sdm_community_pipeline: {n} training surveys, {S} species, {m} queries"]

    # regimes of the queries
    d_near = cKDTree(xyz[train]).query(xyz[query], k=1, workers=n_threads)[0]
    if query_regime is None:
        regime = np.where(d_near > far_buffer_km, "far", "near")
    else:
        regime = np.asarray(query_regime).astype(str)
        regime = np.where(np.isin(regime, ["far", "ood", "OOD"]), "far", "near")
    report.append(f"queries: near={int((regime == 'near').sum())} far={int((regime == 'far').sum())}; "
                  f"nearest training survey km q50 near={np.median(d_near[regime == 'near']) if (regime == 'near').any() else float('nan'):.1f} "
                  f"far={np.median(d_near[regime == 'far']) if (regime == 'far').any() else float('nan'):.1f}")

    # covariates
    blocks = [np.asarray(c, np.float32) for c in covariates]
    for b in blocks:
        if len(b) != n + m:
            raise ValueError("every covariate block must have n_train + n_query rows")
    if embedding is not None:
        embedding = np.asarray(embedding, np.float32)
        blocks.append(embedding)
    blocks.append(xyz / EARTH_RADIUS_KM)
    H = None
    if po is not None:
        H, pdist = sdm_presence_only_histogram(tl_lat, tl_lon, po[0], po[1], po[2], S, k=po_k,
                                               n_threads=n_threads)
        from sklearn.decomposition import TruncatedSVD
        Hs = H.copy()
        Hs.data = np.sqrt(Hs.data)
        dim = int(min(po_svd_dim, S - 1, n - 1))
        E = TruncatedSVD(dim, n_iter=4, random_state=seed).fit(Hs[train]).transform(Hs).astype(np.float32)
        blocks += [E, np.log1p(pdist)]
        report.append(f"presence-only: {len(np.asarray(po[2]))} records, k={po_k} neighbourhood, "
                      f"SVD {dim}, offset={'on' if use_po_offset else 'off'}")
    X = sdm_standardize(blocks, fit_rows=train)
    report.append(f"covariate matrix: {X.shape[1]} columns")
    offset = H if (use_po_offset and H is not None) else None

    def sources_for(ref, q):
        src, names = [], []
        if embedding is not None:
            src.append(sdm_neighbor_composition(Y[ref], embedding[ref], embedding[q], metric="cosine",
                                                k=64, n_threads=n_threads))
            names.append("embedding_knn")
        src.append(sdm_neighbor_composition(Y[ref], (tl_lat[ref], tl_lon[ref]), (tl_lat[q], tl_lon[q]),
                                            metric="geographic", k=30, n_threads=n_threads))
        names.append("geo_knn")
        return src, names

    ckpts = sorted({max(1, int(round(epochs * f))) for f in (0.5, 0.75, 1.0)})
    decisions = {}
    g_train = None if groups is None else np.asarray(groups)
    for reg in ("near", "far"):
        default = dict(epoch=epochs, weights=None, mass_scale=1.0, score=float("nan"), validated=False)
        if reg not in validate or not (regime == reg).any():
            decisions[reg] = default
            continue
        scheme = "cells" if reg == "near" else ("groups" if g_train is not None else "blocks")
        try:
            fit, val, w = sdm_spatial_holdout(tl_lat[train], tl_lon[train], groups=g_train, scheme=scheme,
                                              cell_km=near_cell_km if reg == "near" else far_buffer_km,
                                              fraction=near_fraction if reg == "near" else far_fraction,
                                              buffer_km=None if reg == "near" else far_buffer_km,
                                              max_val=max_val, seed=seed)
        except ValueError as e:
            report.append(f"[{reg}] validation skipped: {e}")
            decisions[reg] = default
            continue
        t0 = time.time()
        _log(log, f"[{reg}] validation: scheme={scheme} fit={len(fit)} val={len(val)}")
        Xv = sdm_standardize(blocks, fit_rows=fit)   # scaling from the fit rows only
        net, hist = sdm_fit_mlp(Xv, Y, fit, epochs=epochs, hidden=hidden, seeds=(seed,), offset=offset,
                                checkpoints=ckpts, eval_rows=val, n_threads=n_threads, log=log)
        src, names = sources_for(fit, val)
        best = _tune(hist, src, Y[val], w, kmax=kmax)
        best["sources"] = names
        best["validated"] = True
        best["scheme"] = scheme
        best["n_fit"], best["n_val"] = int(len(fit)), int(len(val))
        decisions[reg] = best
        report.append(f"[{reg}] {scheme} holdout fit={len(fit)} val={len(val)}: best epoch {best['epoch']}, "
                      f"fusion {dict(zip(names, best['weights']))}, mass x{best['mass_scale']}, "
                      f"validation F1 {best['score']:.4f} (network+plug-in only {best['plugin_only_score']:.4f}) "
                      f"[{time.time() - t0:.0f}s]")
        del net

    # final refit on all training surveys
    used = [decisions[r]["epoch"] for r in ("near", "far") if (regime == r).any()]
    final_epochs = int(max(used)) if used else epochs
    t0 = time.time()
    eps_needed = sorted(set(used)) or [final_epochs]
    net, hist = sdm_fit_mlp(X, Y, train, epochs=final_epochs, hidden=hidden, seeds=tuple(final_seeds),
                            offset=offset, checkpoints=eps_needed, eval_rows=query,
                            n_threads=n_threads, log=log)
    report.append(f"final refit: {len(final_seeds)} seed(s) x {final_epochs} epochs on {n} surveys "
                  f"[{time.time() - t0:.0f}s]")
    src, names = sources_for(train, query)
    P = np.zeros((m, S), np.float32)
    ranking = np.zeros((m, kmax), np.int64)
    sizes = np.zeros(m, np.int64)
    for reg in ("near", "far"):
        rows = np.flatnonzero(regime == reg)
        if not len(rows):
            continue
        dec = decisions[reg]
        base = hist[dec["epoch"]] if dec["epoch"] in hist else hist[max(hist)]
        wts = dec["weights"] if dec["weights"] is not None else [0.0] * len(src)
        Pr = sdm_fuse(base[rows], [s_[rows] for s_ in src], wts)
        r, s = sdm_decode_sets(Pr, mass_scale=dec["mass_scale"], kmax=kmax)
        P[rows], ranking[rows], sizes[rows] = Pr, r, s
        report.append(f"[{reg}] {len(rows)} queries: mean set size {s.mean():.1f}, "
                      f"mean expected richness {Pr.sum(1).mean():.1f}")
    preds = None
    if species_ids is not None:
        preds = sdm_sets_to_strings(ranking, sizes, species_ids)
    report.append(f"total {time.time() - t_start:.0f}s")
    rep = "\n".join(report)
    _log(log, rep)
    return dict(predictions=preds, ranking=ranking, sizes=sizes, prob=P.astype(np.float16),
                regime=regime, decisions=decisions, report=rep)
