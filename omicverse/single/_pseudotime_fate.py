"""Pseudotime → terminal-state → lineage-fate unification.

Implements the 4-step pipeline shared by MIRA (`mira.pseudotime.*`) and
CellRank (`cellrank.kernels.PseudotimeKernel` + `GPCCA`), in pure
numpy/scipy with no heavyweight dependencies (no pyGPCCA, no jax):

    1. bias kNN graph        — Palantir-style hard threshold OR
                                VIA-style soft generalized-logistic
    2. transition matrix     — adaptive Gaussian kernel + row-normalise
    3. macrostates           — top-K Schur vectors (ARPACK on P^T) +
                                lightweight PCCA+ approximation
                                (k-means simplex projection)
    4. fate probabilities    — sparse linear solve (I − Q) X = R
                                (no dense fundamental-matrix inverse)

The module is **backend-agnostic**: it consumes any pseudotime vector
written to ``adata.obs[pseudotime_key]``. After running
``ov.single.TrajInfer(method='slingshot' | 'palantir' | …).inference()``,
just call::

    fate = ov.single.PseudotimeFate(adata, pseudotime_key='slingshot_pseudotime')
    fate.fit()

Speed targets (laptop, single core):

    n=3.7k cells  →   ~0.3 s
    n=10k cells   →   ~1.5 s
    n=50k cells   →   ~10 s

References
----------
- Setty, M. et al. *Characterization of cell fate probabilities in
  single-cell data with Palantir.* Nat Biotechnol 37, 451–460 (2019).
- Stassen, S. V. et al. *Generalizing RNA velocity to transient cell
  states through dynamical modeling.* Nat Biotechnol 39, 1582–1590 (2021).
  (VIA — soft threshold scheme.)
- Lange, M. et al. *CellRank for directed single-cell fate mapping.*
  Nat Methods 19, 159–170 (2022).
- Reuter, B. et al. *Generalized Markov modeling of nonreversible
  dynamics.* Multiscale Model. Simul. 17, 1245–1268 (2019).
  (GPCCA / PCCA+ theory.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# -------------------------------------------------------------------- step 1
def _hard_threshold_bias(
    knn: sp.csr_matrix, pt: np.ndarray, frac_to_keep: float = 0.3,
) -> sp.csr_matrix:
    """Palantir-style edge pruning: keep the ``frac_to_keep`` closest
    neighbours of every cell (regardless of pt direction); among the rest
    keep only those that point into the pseudotime *future*.
    """
    knn = knn.tocsr()
    n = knn.shape[0]
    out = knn.copy()
    indptr, indices, data = out.indptr, out.indices, out.data
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        if s == e:
            continue
        nbrs = indices[s:e]
        conn = data[s:e]
        order = np.argsort(-conn)  # descending connectivity
        k = max(1, min(len(conn), int(np.floor(len(conn) * frac_to_keep))))
        close = order[:k]
        far = order[k:]
        mask = np.ones_like(conn, dtype=bool)
        if len(far):
            past = pt[nbrs[far]] < pt[i]
            mask[far[past]] = False
        data[s:e] = conn * mask
    out.eliminate_zeros()
    return out


def _soft_threshold_bias(
    knn: sp.csr_matrix, pt: np.ndarray, b: float = 10.0, nu: float = 0.5,
) -> sp.csr_matrix:
    """VIA-style downweighting via generalised logistic function.

        weight = 2 / (1 + exp(b · Δpt))^(1/ν)    for past neighbours
        weight = 1                                for future neighbours
    """
    knn = knn.tocsr().copy()
    coo = knn.tocoo()
    i, j, d = coo.row, coo.col, coo.data
    dt = pt[i] - pt[j]                                  # >0 means j is in the past
    past = dt > 0
    w = np.ones_like(d, dtype=np.float64)
    w[past] = 2.0 / np.power(1.0 + np.exp(b * dt[past]), 1.0 / nu)
    out = sp.coo_matrix((d * w, (i, j)), shape=knn.shape).tocsr()
    out.eliminate_zeros()
    return out


# -------------------------------------------------------------------- step 2
def _adaptive_affinity(
    knn: sp.csr_matrix, distances: sp.csr_matrix | None, ka: int,
) -> sp.csr_matrix:
    """Adaptive Gaussian kernel σ_i = distance to the ka-th NN. Returns
    an affinity matrix on the (already-biased) kNN edges.
    """
    if distances is None:
        # fall back: treat connectivity values directly as affinities
        return knn
    n = distances.shape[0]
    d = distances.tocsr()
    sigma = np.empty(n)
    indptr, dvals = d.indptr, d.data
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        row = dvals[s:e]
        if row.size:
            # 0-based: the ka-th NN distance is at index ka-1 in the
            # sorted row. (np.sort(row)[ka] would be the (ka+1)-th NN.)
            k_idx = max(0, min(int(ka) - 1, row.size - 1))
            sigma[i] = float(np.sort(row)[k_idx])
        else:
            sigma[i] = 1.0
    sigma = np.maximum(sigma, 1e-12)
    coo = knn.tocoo()
    di = d[coo.row, coo.col].A1 if d.shape == knn.shape else np.zeros_like(coo.data)
    aff = np.exp(-0.5 * (di ** 2) / (sigma[coo.row] ** 2)
                 - 0.5 * (di ** 2) / (sigma[coo.col] ** 2))
    return sp.coo_matrix((aff * coo.data, (coo.row, coo.col)),
                         shape=knn.shape).tocsr()


def _row_normalise(M: sp.csr_matrix) -> sp.csr_matrix:
    rsum = np.asarray(M.sum(axis=1)).ravel()
    rsum = np.where(rsum > 0, rsum, 1.0)
    D = sp.diags(1.0 / rsum)
    return (D @ M).tocsr()


# -------------------------------------------------------------------- step 3
def _top_schur_vectors(P: sp.csr_matrix, K: int) -> np.ndarray:
    """Return ``K`` leading real-valued left Schur vectors of ``P``.

    For nearly-block-stochastic matrices the Schur basis is real and
    spans the metastable subspace identified by PCCA+. We approximate it
    by the leading right eigenvectors of P^T via ARPACK, then
    orthonormalise (Gram-Schmidt) for numerical safety. This avoids the
    dense O(n³) ``scipy.linalg.schur`` while keeping the macrostate-
    detection signal.
    """
    K = max(2, min(K, P.shape[0] - 2))
    vals, vecs = spla.eigs(P.T.astype(np.float64), k=K, which='LM',
                            sigma=None, maxiter=1000, tol=1e-6)
    # ARPACK can return complex pairs — split into real / imag. The
    # conjugate partner at column i+1 holds the same information as i
    # (v[i+1] ≈ conj(v[i])), so we mark it used to avoid appending a
    # near-duplicate (real, imag) pair that would later evict a real
    # Schur direction during the [:, :K] truncation.
    real_basis = []
    used = np.zeros(K, dtype=bool)
    for i in range(K):
        if used[i]:
            continue
        used[i] = True
        v = vecs[:, i]
        if abs(v.imag).max() < 1e-9:
            real_basis.append(v.real)
        else:
            real_basis.append(v.real)
            real_basis.append(v.imag)
            if i + 1 < K:
                used[i + 1] = True
    B = np.column_stack(real_basis)[:, :K]
    # Modified Gram-Schmidt
    Q, _ = np.linalg.qr(B)
    return Q


class _Beeswarm:
    """Verbatim port of MIRA's ``mira.plots.swarmplot.Beeswarm`` (which is
    in turn lifted from seaborn ≥ 0.12). Packs scatter points adjacent to
    one another perpendicular to ``orient`` so they form the dense
    rectangular strip MIRA's swarm plots are recognised by.
    """

    def __init__(self, orient="h", width=0.8):
        self.orient = orient
        self.width = width

    def __call__(self, points, center):
        ax = points.axes
        dpi = ax.figure.dpi
        orig_xy_data = points.get_offsets()
        cat_idx = 1 if self.orient == "h" else 0
        orig_xy_data[:, cat_idx] = center
        orig_x_data, orig_y_data = orig_xy_data.T
        orig_xy = ax.transData.transform(orig_xy_data)
        if self.orient == "h":
            orig_xy = orig_xy[:, [1, 0]]
        sizes = points.get_sizes()
        if sizes.size == 1:
            sizes = np.repeat(sizes, orig_xy.shape[0])
        edge = points.get_linewidth()
        edge = edge if np.isscalar(edge) else float(np.asarray(edge).item())
        radii = (np.sqrt(sizes) + edge) / 2 * (dpi / 72)
        orig_xyr = np.c_[orig_xy, radii]
        sorter = np.argsort(orig_xyr[:, 1])
        orig_xyr = orig_xyr[sorter]
        new_xyr = np.empty_like(orig_xyr)
        new_xyr[sorter] = self._beeswarm(orig_xyr)
        if self.orient == "h":
            new_xy = new_xyr[:, [1, 0]]
        else:
            new_xy = new_xyr[:, :2]
        new_x_data, new_y_data = ax.transData.inverted().transform(new_xy).T
        if self.orient == "h":
            self._gutter(new_y_data, center)
            points.set_offsets(np.c_[orig_x_data, new_y_data])
        else:
            self._gutter(new_x_data, center)
            points.set_offsets(np.c_[new_x_data, orig_y_data])

    def _beeswarm(self, orig_xyr):
        midline = orig_xyr[0, 0]
        swarm = np.atleast_2d(orig_xyr[0])
        for xyr_i in orig_xyr[1:]:
            neighbors = self._could_overlap(xyr_i, swarm)
            candidates = self._candidates(xyr_i, neighbors)
            offsets = np.abs(candidates[:, 0] - midline)
            candidates = candidates[np.argsort(offsets)]
            new_xyr_i = self._first_ok(candidates, neighbors)
            swarm = np.vstack([swarm, new_xyr_i])
        return swarm

    @staticmethod
    def _could_overlap(xyr_i, swarm):
        _, y_i, r_i = xyr_i
        neighbors = []
        for xyr_j in reversed(swarm):
            _, y_j, r_j = xyr_j
            if (y_i - y_j) < (r_i + r_j):
                neighbors.append(xyr_j)
            else:
                break
        return np.array(neighbors)[::-1]

    @staticmethod
    def _candidates(xyr_i, neighbors):
        candidates = [xyr_i]
        x_i, y_i, r_i = xyr_i
        left_first = True
        for x_j, y_j, r_j in neighbors:
            dy = y_i - y_j
            dx = np.sqrt(max((r_i + r_j) ** 2 - dy ** 2, 0)) * 1.05
            cl, cr = (x_j - dx, y_i, r_i), (x_j + dx, y_i, r_i)
            new_candidates = [cl, cr] if left_first else [cr, cl]
            candidates.extend(new_candidates)
            left_first = not left_first
        return np.array(candidates)

    @staticmethod
    def _first_ok(candidates, neighbors):
        if len(neighbors) == 0:
            return candidates[0]
        nx = neighbors[:, 0]; ny = neighbors[:, 1]; nr = neighbors[:, 2]
        for xyr_i in candidates:
            x_i, y_i, r_i = xyr_i
            dx = nx - x_i; dy = ny - y_i
            sq = dx * dx + dy * dy
            need = (nr + r_i) ** 2
            if np.all(sq >= need):
                return xyr_i
        return candidates[-1]

    def _gutter(self, points, center):
        # Index-assign matches MIRA exactly — `np.clip(out=)` on a
        # transposed-view target can silently no-op, leaving outliers
        # outside the band.
        half = self.width / 2
        low = center - half
        high = center + half
        off_low = points < low
        if off_low.any():
            points[off_low] = low
        off_high = points > high
        if off_high.any():
            points[off_high] = high


def _indexsearch(X: np.ndarray, K: int) -> np.ndarray:
    """Vertex selection from a Schur basis — the PCCA+ inner simplex
    algorithm (Roeblitz & Weber 2013, Multiscale Model. Simul.).

    Greedily picks K rows of ``X`` that form a maximally spread simplex:

    - vertex 1 = row with the largest L2 norm
    - vertex k > 1 = row maximising distance from the affine hull of the
      already-selected k-1 vertices (computed via running orthogonal
      projection, O(n·K) per step → O(n·K²) total).

    Returns indices into the rows of ``X``.
    """
    n, _ = X.shape
    indices = np.empty(K, dtype=np.int64)
    indices[0] = int(np.argmax(np.sum(X ** 2, axis=1)))
    Y = X - X[indices[0]]
    for k in range(1, K):
        norms = np.einsum('ij,ij->i', Y, Y)
        norms[indices[:k]] = -1.0
        indices[k] = int(np.argmax(norms))
        v = Y[indices[k]]
        vn = float(np.linalg.norm(v))
        if vn < 1e-12:
            indices[k:] = indices[k - 1]
            break
        v = v / vn
        Y = Y - np.outer(Y @ v, v)
    return indices


def _pcca_plus_approx(schur_basis: np.ndarray, K: int,
                       seed: int = 0) -> np.ndarray:
    """PCCA+-style soft memberships via the inner-simplex algorithm.

    Far better than k-means on the Schur basis for capturing **small**
    macrostates (single-vertex assignment), and deterministic so no
    multi-restart cost. Implementation follows pyGPCCA's ``indexsearch``
    + affine map step. We skip the crispness optimisation (an O(K³)
    rotation that pyGPCCA does on top to maximise diagonal mass) for
    speed — empirically the bare ISA already gives terminal-state
    agreement comparable to GPCCA on the pancreas fixture.
    """
    K = max(2, min(K, schur_basis.shape[1]))
    n = schur_basis.shape[0]
    verts = _indexsearch(schur_basis, K)
    M = schur_basis[verts]                              # (K, K)
    # Build affine map A such that schur_basis[verts] · A = I_K, then
    # chi = schur_basis · A is the soft membership.
    try:
        A = np.linalg.solve(M, np.eye(K))
    except np.linalg.LinAlgError:
        A = np.linalg.pinv(M)
    chi = schur_basis @ A                               # (n, K)
    # Clip tiny negatives, renormalise rows.
    chi = np.clip(chi, 0.0, None)
    rs = chi.sum(axis=1, keepdims=True)
    chi = np.divide(chi, rs, where=rs > 0)
    chi[~np.isfinite(chi)] = 1.0 / K
    return chi.astype(np.float32)


def _coarse_grain(P: sp.csr_matrix, Z: np.ndarray) -> np.ndarray:
    """Compute the K × K coarse-grained (Galerkin) transition matrix from
    soft memberships ``Z`` (n × K, rows sum to 1):

        P_coarse_ij = (Z^T · P · Z)_ij / (sum_n Z_ni)

    This is the standard PCCA+ coarse-graining and matches pyGPCCA's
    ``coarse_T`` output. The diagonal gives self-residency.
    """
    Zsp = sp.csr_matrix(Z.astype(np.float64))
    num = (Zsp.T @ P @ Zsp).toarray()
    den = Z.sum(axis=0).astype(np.float64)
    den = np.where(den > 0, den, 1.0)
    return num / den[:, None]


# -------------------------------------------------------------------- step 4
def _solve_fate_probabilities(
    P: sp.csr_matrix,
    terminal_groups: list[np.ndarray],
    *,
    tol: float = 1e-6, max_iter: int = 5000,
) -> np.ndarray:
    """Compute absorption probabilities into each terminal **group** of
    cells via Neumann-series power iteration.

    Each terminal group is a *set* of cells (the macrostate assigned to
    that lineage), not a single representative. Aggregating absorption
    over all cells in the macrostate is what CellRank's GPCCA and the
    original Palantir do, and it is what gives a smooth fate-
    probability landscape rather than a delta function at one cell.

    Solves ``(I − Q) X = R`` where ``Q`` = transitions among non-
    terminal cells, and ``R`` is **column-aggregated** so that each
    column is the sum of transitions from transient cells into every
    member of that terminal group.

    Returns: ``(n, K_groups)`` matrix. Rows sum to ≤ 1 (= 1 for cells
    inside any terminal group, ≤ 1 for transient cells that may bleed
    into unfilled mass when groups don't span the full Markov chain).
    """
    n = P.shape[0]
    if len(terminal_groups) == 0:
        return np.zeros((n, 0))

    # Union of all terminal cells across groups (a cell may *not* sit
    # in two groups; we enforce this by deduping greedily by group order).
    in_any: set = set()
    cleaned_groups = []
    for g in terminal_groups:
        g_arr = np.asarray(g, dtype=np.int64)
        g_arr = np.array([c for c in g_arr if c not in in_any], dtype=np.int64)
        in_any.update(g_arr.tolist())
        cleaned_groups.append(g_arr)
    terminal_groups = cleaned_groups

    absorb = np.zeros(n, dtype=bool)
    for g in terminal_groups:
        absorb[g] = True
    trans_idx = np.where(~absorb)[0]
    if trans_idx.size == 0:
        out = np.zeros((n, len(terminal_groups)))
        for j, g in enumerate(terminal_groups):
            out[g, j] = 1.0
        return out

    Pcsr = P.tocsr()
    Q = Pcsr[trans_idx][:, trans_idx]

    # Aggregated R: column j = sum of transitions from each transient cell
    # into every member of terminal group j.
    R_full = Pcsr[trans_idx]                             # (n_trans, n)
    R = np.empty((R_full.shape[0], len(terminal_groups)))
    for j, g in enumerate(terminal_groups):
        R[:, j] = np.asarray(R_full[:, g].sum(axis=1)).ravel()

    # Neumann series.
    X = np.zeros_like(R)
    delta = R.copy()
    for _it in range(max_iter):
        X += delta
        delta = Q @ delta
        if np.max(np.abs(delta)) < tol:
            break
    else:
        import warnings
        warnings.warn(
            f"Neumann series did not converge in {max_iter} iterations "
            f"(max |Δ| = {np.max(np.abs(delta)):.2e}). Fate "
            "probabilities may be inaccurate — try increasing max_iter "
            "or check that the biased kNN is well-connected.",
            RuntimeWarning, stacklevel=3,
        )

    out = np.zeros((n, len(terminal_groups)), dtype=np.float64)
    out[trans_idx] = X
    for j, g in enumerate(terminal_groups):
        out[g, j] = 1.0
    out = np.clip(out, 0.0, None)
    rs = out.sum(axis=1, keepdims=True)
    out = np.divide(out, rs, where=rs > 0)
    return out


# -------------------------------------------------------------------- public
@dataclass
class PseudotimeFateResult:
    """Container for the four pipeline outputs."""

    transition_matrix: sp.csr_matrix
    macrostate_assignment: np.ndarray              # (n,) int macrostate id
    macrostate_residency: np.ndarray               # (K,) self-residency
    terminal_macrostates: np.ndarray               # (K_term,) indices into macrostates
    terminal_cells: np.ndarray                     # representative cell per term ms
    fate_probabilities: np.ndarray | None = None   # (n, K_term)
    lineage_entropy: np.ndarray | None = None      # (n,) Shannon entropy of fate
    params: dict = field(default_factory=dict)


class PseudotimeFate:
    """Unified terminal-state and fate-probability estimator on top of
    any pseudotime computed by :class:`~omicverse.single.TrajInfer`.

    Parameters
    ----------
    adata
        Single-cell AnnData. Must contain a precomputed kNN graph at
        ``adata.obsp['connectivities']`` (e.g. from ``ov.pp.neighbors``).
    pseudotime_key
        Column in ``adata.obs`` holding the pseudotime to use.
    scheme
        ``'hard'`` (Palantir-style) or ``'soft'`` (VIA-style).
    n_macrostates
        Number of metastable macrostates to extract via the Schur+PCCA+
        approximation. Pick larger than your expected number of lineages.
    residency_threshold
        Macrostates whose diagonal of the coarse-grained P exceeds this
        are flagged as terminal.

    Examples
    --------
    >>> Traj = ov.single.TrajInfer(adata, basis='X_umap', groupby='clusters',
    ...                            use_rep='scaled|original|X_pca', n_comps=50)
    >>> Traj.set_origin_cells('Ductal')
    >>> Traj.inference(method='slingshot', num_epochs=1)
    >>>
    >>> fate = ov.single.PseudotimeFate(adata,
    ...                                  pseudotime_key='slingshot_pseudotime')
    >>> fate.fit()
    >>> adata.obs['lineage_entropy']    # uncertainty per cell
    >>> adata.obsm['fate_probabilities']  # (n_cells, n_terminal_macrostates)
    """

    def __init__(
        self,
        adata,
        pseudotime_key: str,
        *,
        groupby: str | None = None,
        scheme: Literal['hard', 'soft'] = 'hard',
        n_macrostates: int = 10,
        residency_threshold: float = 0.60,
        late_pt_quantile: float = 0.70,
        ka: int = 5,
        frac_to_keep: float = 0.3,
        soft_b: float = 10.0,
        soft_nu: float = 0.5,
        connectivity_key: str = 'connectivities',
        distance_key: str = 'distances',
        seed: int = 0,
    ):
        if pseudotime_key not in adata.obs.columns:
            raise KeyError(
                f"pseudotime_key {pseudotime_key!r} not in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )
        if connectivity_key not in adata.obsp:
            raise KeyError(
                f"Expected a kNN graph at adata.obsp[{connectivity_key!r}]. "
                f"Run `ov.pp.neighbors(adata, ...)` first."
            )
        self.adata = adata
        self.pseudotime_key = pseudotime_key
        self.groupby = groupby
        self.scheme = scheme
        self.n_macrostates = n_macrostates
        self.residency_threshold = residency_threshold
        self.late_pt_quantile = late_pt_quantile
        self.ka = ka
        self.frac_to_keep = frac_to_keep
        self.soft_b = soft_b
        self.soft_nu = soft_nu
        self.connectivity_key = connectivity_key
        self.distance_key = distance_key
        self.seed = seed

        pt = np.asarray(adata.obs[pseudotime_key].values, dtype=np.float64)
        if np.any(~np.isfinite(pt)):
            raise ValueError("pseudotime contains NaN/inf values")
        self._pt = pt
        self._knn = adata.obsp[connectivity_key].tocsr()
        self._dist = (adata.obsp[distance_key].tocsr()
                      if distance_key in adata.obsp else None)
        self.result: PseudotimeFateResult | None = None

    def _prefix_from_pseudotime_key(self) -> str:
        """Return the backend prefix used to name written adata keys.

        Strips a *trailing* ``_pseudotime`` suffix only — using
        ``str.replace`` here would corrupt keys that contain the
        substring elsewhere (e.g. ``my_pseudotime_score`` →
        ``my_score``) or leave non-conventional keys unchanged
        (e.g. ``velocity_pt`` → full string as prefix).
        """
        suffix = "_pseudotime"
        if self.pseudotime_key.endswith(suffix):
            return self.pseudotime_key[:-len(suffix)]
        return self.pseudotime_key

    # ------------------------------------------------------------------ public
    def fit(self, *, compute_fates: bool = True) -> PseudotimeFateResult:
        """Run all four stages and return a :class:`PseudotimeFateResult`."""
        biased = self._bias_knn()
        affinity = _adaptive_affinity(biased, self._dist, self.ka)
        P = _row_normalise(affinity)

        Q_schur = _top_schur_vectors(P, self.n_macrostates)
        Z_soft = _pcca_plus_approx(Q_schur, self.n_macrostates, seed=self.seed)
        ms = Z_soft.argmax(axis=1)

        # Hard one-hot Z for sharp coarse-graining (soft Z gives diluted
        # off-diagonal mass and washes out the residency signal).
        K = Z_soft.shape[1]
        n = self.adata.n_obs
        Z_hard = np.zeros((n, K), dtype=np.float32)
        Z_hard[np.arange(n), ms] = 1.0
        P_coarse = _coarse_grain(P, Z_hard)
        residency = np.diag(P_coarse)

        ms_mean_pt = np.array([
            self._pt[ms == k].mean() if (ms == k).any() else np.inf
            for k in range(K)
        ])
        pt_min, pt_max = float(np.nanmin(self._pt)), float(np.nanmax(self._pt))
        pt_span = max(pt_max - pt_min, 1e-12)
        ms_pt_rank = (ms_mean_pt - pt_min) / pt_span

        # Forward flow diagnostic (mass leaving for strictly-later
        # macrostates); kept on the result dataclass for inspection but
        # not used as a hard filter — sibling terminals (e.g. α vs β in
        # pancreas) can have small forward_flow into one another simply
        # because their mean pseudotimes differ by epsilon.
        later_mask = ms_mean_pt[None, :] > ms_mean_pt[:, None]
        np.fill_diagonal(later_mask, False)
        forward_flow = np.where(later_mask, P_coarse, 0.0).sum(axis=1)

        initial_ms = int(np.argmin(ms_mean_pt))
        candidate = np.where(
            (ms_pt_rank >= self.late_pt_quantile)
            & (residency >= self.residency_threshold)
        )[0]
        candidate = candidate[candidate != initial_ms]
        # Rank by residency descending so the dedup-by-cluster step
        # picks the sharpest representative per terminal cell type.
        candidate = candidate[np.argsort(-residency[candidate])]
        terminal_ms = candidate
        self._forward_flow = forward_flow

        terminal_cells, kept_terminal_ms = self._pick_terminal_cells(ms, terminal_ms)
        # Per-lineage absorbing SETS — restricted to the *dominant cluster*
        # of each macrostate when a ``groupby`` is provided. Including
        # off-cluster "contaminants" (e.g. 4 Alpha cells that wandered
        # into an Epsilon-dominated macrostate) leaks fate mass across
        # lineages because those off-cluster cells are kNN hubs to their
        # actual cluster; restricting the absorbing set to the dominant
        # cluster removes that bridge. Without ``groupby`` we fall back
        # to the full macrostate.
        terminal_groups = self._build_terminal_groups(ms, kept_terminal_ms,
                                                       terminal_cells)
        terminal_ms = kept_terminal_ms

        fates = entropy = None
        if compute_fates and len(terminal_cells) > 0:
            fates = _solve_fate_probabilities(P, terminal_groups)
            with np.errstate(divide='ignore', invalid='ignore'):
                logp = np.where(fates > 0, np.log(fates), 0.0)
                entropy = -np.sum(fates * logp, axis=1)

        self.result = PseudotimeFateResult(
            transition_matrix=P,
            macrostate_assignment=ms,
            macrostate_residency=residency,
            terminal_macrostates=terminal_ms,
            terminal_cells=terminal_cells,
            fate_probabilities=fates,
            lineage_entropy=entropy,
            params=dict(
                scheme=self.scheme, n_macrostates=self.n_macrostates,
                residency_threshold=self.residency_threshold,
                ka=self.ka, frac_to_keep=self.frac_to_keep,
                pseudotime_key=self.pseudotime_key,
            ),
        )
        self._write_adata(self.result)
        return self.result

    # --------------------------------------------------------------- internals
    def _bias_knn(self) -> sp.csr_matrix:
        if self.scheme == 'hard':
            return _hard_threshold_bias(self._knn, self._pt, self.frac_to_keep)
        elif self.scheme == 'soft':
            return _soft_threshold_bias(self._knn, self._pt,
                                         self.soft_b, self.soft_nu)
        raise ValueError(f"unknown scheme {self.scheme!r}")

    def _pick_terminal_cells(self, ms: np.ndarray,
                              terminal_ms: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray]:
        """For each terminal macrostate, pick (rep_cell, kept_ms_id).

        Returns
        -------
        rep_cells : (K_kept,) int — one representative cell per kept macrostate.
        kept_ms_ids : (K_kept,) int — the subset of ``terminal_ms`` that
            survived cluster-purity deduplication. The two arrays are
            index-aligned, so downstream code can call
            ``[np.where(ms == k)[0] for k in kept_ms_ids]`` and trust
            that the j-th column of the fate matrix corresponds to the
            j-th entry of ``rep_cells``.
        """
        groupby = getattr(self, 'groupby', None)
        if not groupby or groupby not in self.adata.obs.columns:
            reps, kept = [], []
            for k in terminal_ms:
                in_ms = np.where(ms == k)[0]
                if in_ms.size:
                    reps.append(int(in_ms[np.argmax(self._pt[in_ms])]))
                    kept.append(int(k))
            return np.asarray(reps, dtype=np.int64), np.asarray(kept, dtype=np.int64)

        clusters = self.adata.obs[groupby].astype(str).values
        from collections import Counter
        total_per_cluster = Counter(clusters)

        reps, kept = [], []
        seen_clusters: set = set()
        for k in terminal_ms:
            in_ms = np.where(ms == k)[0]
            if not in_ms.size:
                continue
            ms_clusters = clusters[in_ms]
            in_ms_counts = Counter(ms_clusters)
            purity_ranked = sorted(
                in_ms_counts.items(),
                key=lambda kv: kv[1] / max(1, total_per_cluster[kv[0]]),
                reverse=True,
            )
            chosen_cluster = None
            for cluster, _ in purity_ranked:
                if cluster not in seen_clusters:
                    chosen_cluster = cluster
                    break
            if chosen_cluster is None:
                continue
            seen_clusters.add(chosen_cluster)
            in_cluster_in_ms = in_ms[ms_clusters == chosen_cluster]
            rep = int(in_cluster_in_ms[np.argmax(self._pt[in_cluster_in_ms])])
            reps.append(rep)
            kept.append(int(k))
        return (np.asarray(reps, dtype=np.int64),
                np.asarray(kept, dtype=np.int64))

    def _build_terminal_groups(self, ms: np.ndarray,
                                kept_terminal_ms: np.ndarray,
                                terminal_cells: np.ndarray) -> list:
        """Per-lineage absorbing-set indices, restricted to the
        dominant cluster of each macrostate (the cluster of the rep cell).
        """
        groupby = getattr(self, 'groupby', None)
        if not groupby or groupby not in self.adata.obs.columns:
            return [np.where(ms == k)[0] for k in kept_terminal_ms]

        clusters = self.adata.obs[groupby].astype(str).values
        groups = []
        for k, rep in zip(kept_terminal_ms, terminal_cells):
            in_ms = np.where(ms == k)[0]
            rep_cluster = clusters[int(rep)]
            keep = in_ms[clusters[in_ms] == rep_cluster]
            # If filtering wipes the set (shouldn't happen because rep
            # is in the set by construction), fall back to full macrostate.
            groups.append(keep if keep.size else in_ms)
        return groups

    # =============================================================== drivers
    def compute_lineage_drivers(
        self,
        lineages: str | list | None = None,
        *,
        layer: str | None = None,
        use_raw: bool = False,
        cluster_key: str | None = None,
        clusters: list | None = None,
        method: Literal["fisher", "spearman"] = "fisher",
        confidence_level: float = 0.95,
        n_top: int | None = None,
    ) -> "pd.DataFrame":
        """Genes whose expression best correlates with each lineage's
        fate probability — the CellRank ``compute_lineage_drivers``
        analogue (Lange 2022, Reuter 2019 :cite:t:`reuter:19`).

        Per-gene Pearson correlation between gene expression and the
        fate-probability column (Fisher z-transform CI, BH-adjusted
        q-values across all genes), restricted to a subset of clusters
        via ``cluster_key`` / ``clusters`` (recommended — driver genes
        are most meaningful within the lineage's basin of attraction).

        Returns a long-form :class:`pandas.DataFrame` with columns
        ``[gene, lineage, corr, pval, qval, ci_low, ci_high]`` sorted by
        ``corr`` within each lineage. Also writes the wide-form result
        to ``adata.varm['<prefix>_lineage_drivers']``.
        """
        import pandas as pd
        from scipy.stats import norm

        if self.result is None or self.result.fate_probabilities is None:
            raise RuntimeError(
                "Call fit() first (with compute_fates=True). Driver "
                "genes need fate probabilities."
            )
        adata = self.adata
        F = self.result.fate_probabilities
        names = [str(adata.obs[self.groupby].iloc[c])
                 if self.groupby else f"L{j}"
                 for j, c in enumerate(self.result.terminal_cells)]

        # Resolve which lineages the user wants
        if lineages is None:
            sel = list(range(F.shape[1]))
        elif isinstance(lineages, str):
            sel = [names.index(lineages)]
        else:
            sel = [names.index(n) for n in lineages]

        # Resolve the expression matrix
        if use_raw and adata.raw is not None:
            X = adata.raw.X
            var_names = list(adata.raw.var_names)
        elif layer:
            X = adata.layers[layer]
            var_names = list(adata.var_names)
        else:
            X = adata.X
            var_names = list(adata.var_names)
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        # Optional cluster restriction
        if cluster_key and clusters is not None:
            mask = adata.obs[cluster_key].astype(str).isin([str(c) for c in clusters]).values
            X = X[mask]
            F_use = F[mask]
        else:
            F_use = F

        n_cells = X.shape[0]
        if n_cells < 4:
            raise ValueError(
                f"too few cells ({n_cells}) for correlation — relax the "
                "cluster restriction or pick a smaller cluster set."
            )

        # Pearson correlation (vectorised across genes for one lineage at a time)
        Xc = X - X.mean(axis=0, keepdims=True)
        x_norm = np.sqrt((Xc * Xc).sum(axis=0))
        rows = []
        for j in sel:
            yc = F_use[:, j] - F_use[:, j].mean()
            y_norm = float(np.sqrt(np.sum(yc * yc)))
            denom = x_norm * y_norm
            corr = np.where(denom > 0, (Xc.T @ yc) / np.maximum(denom, 1e-12), 0.0)
            # Fisher z-transform for CI / p-value
            r = np.clip(corr, -0.999999, 0.999999)
            z = np.arctanh(r)
            se = 1.0 / np.sqrt(n_cells - 3)
            zcrit = norm.ppf(0.5 + confidence_level / 2)
            lo = np.tanh(z - zcrit * se)
            hi = np.tanh(z + zcrit * se)
            # Two-sided p from Fisher z
            pval = 2 * (1 - norm.cdf(np.abs(z / se)))
            # BH q-value
            order = np.argsort(pval)
            ranked = pval[order]
            qval_sorted = ranked * len(pval) / (np.arange(len(pval)) + 1)
            qval_sorted = np.minimum.accumulate(qval_sorted[::-1])[::-1]
            qval = np.empty_like(pval)
            qval[order] = qval_sorted
            for g_idx, gene in enumerate(var_names):
                rows.append((gene, names[j], float(corr[g_idx]),
                             float(pval[g_idx]), float(qval[g_idx]),
                             float(lo[g_idx]), float(hi[g_idx])))

        df = pd.DataFrame(rows, columns=["gene", "lineage", "corr",
                                          "pval", "qval", "ci_low", "ci_high"])
        df = df.sort_values(["lineage", "corr"], ascending=[True, False])
        if n_top:
            df = df.groupby("lineage", group_keys=False).head(n_top)

        # Wide form into adata.varm for downstream tooling.
        wide = df.pivot(index="gene", columns="lineage", values="corr").reindex(var_names)
        prefix = self._prefix_from_pseudotime_key()
        self.adata.varm[f"{prefix}_lineage_drivers"] = wide.values
        self.adata.uns[f"{prefix}_lineage_driver_columns"] = list(wide.columns)
        return df.reset_index(drop=True)

    # ============================================================ projection
    def compute_circular_projection(
        self,
        normalize_by_mean: bool = True,
        lineage_order: list | None = None,
    ) -> "np.ndarray":
        """Velten-2017-style circular embedding: place terminal lineages
        evenly on the unit circle, then position each cell at the
        weighted barycentre of those vertices using its fate
        probabilities. Cells with one dominant fate sit near that
        vertex; uncommitted cells sit in the middle.

        Writes the (n, 2) projection to ``adata.obsm['<prefix>_X_fate_simplex']``
        and returns it. ``lineage_order`` (list of terminal-cluster
        names) lets you control the angular order; the default uses the
        ``terminal_cells`` order produced by :meth:`fit`.
        """
        if self.result is None or self.result.fate_probabilities is None:
            raise RuntimeError("Call fit() first (with compute_fates=True)")
        F = self.result.fate_probabilities
        if normalize_by_mean:
            mean = F.mean(axis=0, keepdims=True)
            F = F / np.maximum(mean, 1e-12)
            F = F / F.sum(axis=1, keepdims=True)
        names = [str(self.adata.obs[self.groupby].iloc[c])
                 if self.groupby else f"L{j}"
                 for j, c in enumerate(self.result.terminal_cells)]
        if lineage_order is not None:
            idx = [names.index(n) for n in lineage_order]
            F = F[:, idx]
            names = lineage_order
        K = F.shape[1]
        angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
        verts = np.column_stack([np.cos(angles), np.sin(angles)])    # (K, 2)
        proj = F @ verts                                              # (n, 2)
        prefix = self._prefix_from_pseudotime_key()
        self.adata.obsm[f"{prefix}_X_fate_simplex"] = proj
        self.adata.uns[f"{prefix}_fate_simplex_names"] = names
        return proj

    def plot_circular_projection(
        self,
        color: str | None = None,
        figsize=(5, 5),
        ax=None,
        s: float = 4,
        cmap: str = "viridis",
        label_distance: float = 1.20,
    ):
        """Render the circular embedding from
        :meth:`compute_circular_projection`. Cells inside the unit
        circle, lineage labels arranged on its perimeter."""
        if self.result is None or self.result.fate_probabilities is None:
            raise RuntimeError("Call fit() first")
        import matplotlib.pyplot as plt
        prefix = self._prefix_from_pseudotime_key()
        key = f"{prefix}_X_fate_simplex"
        if key not in self.adata.obsm:
            self.compute_circular_projection()
        coords = self.adata.obsm[key]
        names = self.adata.uns[f"{prefix}_fate_simplex_names"]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        cat_legend_labels = None  # set when categorical with named legend
        if color is None or color == 'lineage_entropy':
            c = self.result.lineage_entropy
            title = 'lineage entropy'
        elif color in self.adata.obs.columns:
            v = self.adata.obs[color]
            if v.dtype.name == 'category':
                cats = list(v.cat.categories)
                # Reuse ``adata.uns[f'{color}_colors']`` if available
                # (set when ``color == self.groupby``), otherwise fall
                # back to tab20.
                src_key = f"{color}_colors"
                if src_key in self.adata.uns:
                    import matplotlib.colors as mc
                    cat_colors = [mc.to_hex(x) for x in self.adata.uns[src_key]]
                    cat_colors = cat_colors[:len(cats)] + [
                        '#dddddd'
                    ] * max(0, len(cats) - len(cat_colors))
                else:
                    cmap_ = plt.get_cmap('tab20', max(3, len(cats)))
                    cat_colors = [cmap_(i % cmap_.N) for i in range(len(cats))]
                c = [cat_colors[i] for i in v.cat.codes.values]
                cat_legend_labels = list(zip(cats, cat_colors))
                title = color
                cmap = None
            else:
                c = v.values
                title = color
        else:
            c = self.adata.obs_vector(color) if color in self.adata.var_names else None
            title = color
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=c, s=s, cmap=cmap)
        # Unit circle outline
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color='#444', lw=0.8)
        # Lineage labels around the perimeter — colour each label using
        # the cluster palette (perturbed for duplicates), matching the
        # tree-leaf colour scheme in ``plot_stream``.
        label_colors = self._resolve_cluster_colors(names)
        K = len(names)
        for j, name in enumerate(names):
            ang = 2 * np.pi * j / K
            ax.text(label_distance * np.cos(ang),
                    label_distance * np.sin(ang),
                    name, ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color=label_colors[j])
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title)
        for s_ in ax.spines.values(): s_.set_visible(False)
        if cmap is not None and color != 'clusters':
            try:
                plt.colorbar(sc, ax=ax, shrink=0.6)
            except Exception:
                pass
        elif cat_legend_labels is not None:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], marker='o', linestyle='none',
                        markerfacecolor=col, markeredgecolor='none',
                        markersize=7, label=lbl)
                for lbl, col in cat_legend_labels
            ]
            ax.legend(handles=handles, loc='center left',
                       bbox_to_anchor=(1.02, 0.5), frameon=False,
                       fontsize=8, title=color)
        return ax

    # =========================================================== differentiation traces
    def trace_differentiation(
        self,
        start_cells: list | int | None = None,
        start_lineage: str | None = None,
        num_start_cells: int = 50,
        num_steps: int = 1500,
        steps_per_frame: int = 1,
        direction: Literal["forward", "backward"] = "forward",
        log_prob: bool = False,
        sqrt_time: bool = False,
        ka: int | None = None,
        save_name: str | None = None,
        figsize: tuple = (10, 7),
        fps: int = 24,
        palette: str = "BuPu",
        vmax_quantile: float = 0.99,
        num_preview_frames: int = 0,
        basis: str = "X_umap",
    ) -> "np.ndarray":
        """Deterministic forward probability diffusion through the
        biased Markov chain — the **exact** equivalent of MIRA's
        ``mira.time.trace_differentiation`` (Lopez Garcia, mira/
        pseudotime/backtrace.py ``_trace``).

        Starts from an initial probability distribution concentrated on
        ``start_cells`` and iteratively applies ``P^T``::

            p_{t+1} = P^T · p_t

        so ``p_t[i]`` is the probability that a walker starting from the
        initial distribution is at cell ``i`` after ``t`` Markov steps.
        Unlike a Monte-Carlo random walk this is **deterministic and
        smooth** — the same input always gives the same trace, and
        every cell carries a non-negative mass at every frame.

        Returns
        -------
        states : (num_steps, n_cells) ndarray
            Probability distribution over cells at each time step. Use
            :meth:`plot_trace_density` or :meth:`plot_trace_animation`
            to render. ``sqrt_time=True`` returns frames spaced
            quadratically in time — useful for processes that evolve
            fastest near the start.
        """
        if self.result is None:
            raise RuntimeError("Call fit() first")
        n = self.adata.n_obs
        pt = self._pt
        # MIRA's backward direction rebuilds the entire transport map
        # with reversed pseudotime (prune_edges + adaptive affinity +
        # row-normalise from scratch) instead of just transposing the
        # forward chain. The two differ because the pt-biased kNN graph
        # is not reversible — taking P^T of the forward map is not the
        # same Markov chain a user would get with pt = max - pt.
        # ``ka`` overrides the saved kernel-width neighbour rank, also
        # matching MIRA's per-call ``ka`` argument.
        ka_use = self.ka if ka is None else int(ka)
        if direction == "backward":
            pt_rev = float(pt.max()) - pt
            biased = (
                _hard_threshold_bias(self._knn, pt_rev, self.frac_to_keep)
                if self.scheme == "hard"
                else _soft_threshold_bias(self._knn, pt_rev, self.soft_b, self.soft_nu)
            )
            affinity = _adaptive_affinity(biased, self._dist, ka_use)
            P = _row_normalise(affinity)
        elif ka is not None:
            biased = (
                _hard_threshold_bias(self._knn, pt, self.frac_to_keep)
                if self.scheme == "hard"
                else _soft_threshold_bias(self._knn, pt, self.soft_b, self.soft_nu)
            )
            affinity = _adaptive_affinity(biased, self._dist, ka_use)
            P = _row_normalise(affinity)
        else:
            P = self.result.transition_matrix.tocsr()
        # MIRA `start_lineage='X'` semantics — top ``num_start_cells`` cells
        # by branch probability for that lineage.
        if start_lineage is not None:
            if self.result is None or self.result.fate_probabilities is None:
                raise RuntimeError("fit() with compute_fates=True required for start_lineage")
            names = [str(self.adata.obs[self.groupby].iloc[c])
                     if self.groupby else f"L{j}"
                     for j, c in enumerate(self.result.terminal_cells)]
            if start_lineage not in names:
                raise KeyError(
                    f"start_lineage {start_lineage!r} not in {names}"
                )
            j = names.index(start_lineage)
            probs = self.result.fate_probabilities[:, j]
            order = np.argsort(-probs)
            starts = order[: int(num_start_cells)].astype(np.int64)
        elif start_cells is None:
            starts = np.array([int(np.argmin(pt) if direction == "forward"
                                    else np.argmax(pt))], dtype=np.int64)
        elif isinstance(start_cells, int):
            starts = np.array([start_cells], dtype=np.int64)
        else:
            # MIRA accepts either integer indices, barcode strings, or a
            # boolean mask the size of adata. Coerce all three to indices.
            sc_arr = np.asarray(start_cells)
            if sc_arr.dtype == bool:
                if sc_arr.shape[0] != n:
                    raise ValueError(
                        f"boolean start_cells mask has length {sc_arr.shape[0]}, "
                        f"expected {n}")
                starts = np.where(sc_arr)[0].astype(np.int64)
            elif sc_arr.dtype.kind == "U":
                names_map = {n_: i for i, n_ in enumerate(self.adata.obs_names)}
                missing = [s for s in sc_arr if s not in names_map]
                if missing:
                    raise KeyError(f"barcodes not in adata: {missing[:5]}")
                starts = np.array([names_map[s] for s in sc_arr], dtype=np.int64)
            else:
                starts = sc_arr.astype(np.int64)
        if direction not in ("forward", "backward"):
            raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")

        # Initial probability distribution — uniform over the start set
        # (matches MIRA's `start_cells/start_cells.sum()` line).
        p = np.zeros(n, dtype=np.float64)
        p[starts] = 1.0
        if p.sum() <= 0:
            raise ValueError("start set is empty")
        p = p / p.sum()

        # Diffusion via ``P^T`` for both directions — for the backward
        # case the *map itself* was rebuilt with pt = max - pt above,
        # so the same ``P^T`` propagator that diffuses forward through
        # the forward map diffuses backward through the backward map.
        operator = P.T.tocsr()

        # Compute the FULL ``backtrace`` of shape (num_steps, n_cells)
        # exactly as MIRA's ``_trace``:
        #   for i in range(num_steps):
        #       current = operator @ current
        #       steps.append(current)
        # i.e. ``backtrace[k]`` is the state AFTER ``k+1`` applications,
        # NOT the raw initial distribution.
        states = np.empty((num_steps, n), dtype=np.float64)
        cur = p
        for k in range(num_steps):
            cur = operator @ cur
            states[k] = cur

        # log_prob comes BEFORE frame subsampling (MIRA order).
        if log_prob:
            states = np.log(states + 1e-4)

        # MIRA's frame_slices selection.
        if sqrt_time:
            frame_slices = np.square(
                np.linspace(0, np.sqrt(num_steps - 1), num_steps // steps_per_frame)
            ).astype(int)
        else:
            frame_slices = np.arange(0, num_steps, steps_per_frame)
        sub = states[frame_slices, :]
        num_frames = len(sub)

        # MIRA's preview-frames selection:
        #   test_frames = [1] + range(1, num_partitions)*preview_interval
        #                     + [num_frames - 1]
        # so num_preview_frames=4 on num_frames=65 → Frame 1, 21, 42, 64.
        if num_preview_frames > 0 and num_frames > 1:
            import matplotlib.pyplot as plt
            num_partitions = max(1, num_preview_frames - 1)
            preview_interval = max(1, num_frames // num_partitions)
            test_frames = ([1]
                            + list(np.arange(1, num_partitions) * preview_interval)
                            + [num_frames - 1])
            test_frames = sorted(set(int(f) for f in test_frames if 0 <= f < num_frames))
            coords = self.adata.obsm[basis]
            fig, axes = plt.subplots(
                1, len(test_frames),
                figsize=(figsize[0] * len(test_frames) / 3, figsize[1] / 2),
                squeeze=False,
            )
            for ax, k in zip(axes.flat, test_frames):
                probs = sub[k]
                vmax = float(np.quantile(probs, vmax_quantile))
                order = probs.argsort()
                ax.scatter(coords[order, 0], coords[order, 1],
                            c=probs[order], cmap=palette, s=2,
                            vmin=probs.min(), vmax=max(vmax, probs.min() + 1e-12))
                ax.set_title(f"Frame {k}")
                ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout(); plt.show()

        if save_name:
            self.plot_trace_animation(
                states=sub, basis=basis, figsize=figsize, cmap=palette,
                fps=fps, num_frames=num_frames, save_path=save_name,
                vmax_quantile=vmax_quantile,
            )

        return states

    def plot_trace_density(
        self,
        states: "np.ndarray | None" = None,
        basis: str = "X_umap",
        ax=None,
        figsize=(5, 4),
        cmap: str = "BuPu",
        num_steps: int = 2000,
        frame: int | None = None,
        vmax_quantile: float = 0.99,
        time_average: bool = True,
    ):
        """Per-cell probability mass from
        :meth:`trace_differentiation`, plotted on the embedding. Either
        a single frame (``frame=...``) or the time-averaged mass across
        all frames (``time_average=True``, default).

        Equivalent to a single frame of MIRA's ``show_gif`` rendering.
        """
        import matplotlib.pyplot as plt
        if states is None:
            states = self.trace_differentiation(num_steps=num_steps)
        if frame is not None:
            density = states[frame]
        elif time_average:
            density = states.mean(axis=0)
        else:
            density = states[-1]
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        coords = self.adata.obsm[basis]
        order = density.argsort()
        vmax = float(np.quantile(density, vmax_quantile))
        sc = ax.scatter(coords[order, 0], coords[order, 1],
                         c=density[order], s=4, cmap=cmap,
                         vmin=0.0, vmax=max(vmax, 1e-12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            "Diffusion trace — time-averaged density"
            if frame is None and time_average else f"Diffusion trace — frame {frame}"
        )
        try:
            ax.get_figure().colorbar(sc, ax=ax, shrink=0.6)
        except Exception:
            pass
        return ax

    def plot_trace_animation(
        self,
        states: "np.ndarray | None" = None,
        basis: str = "X_umap",
        figsize=(7, 5),
        cmap: str = "BuPu",
        fps: int = 24,
        num_frames: int = 80,
        save_path: str | None = None,
        num_steps: int = 2000,
        vmax_quantile: float = 0.99,
    ):
        """Build a MIRA-style ``trace_differentiation`` GIF animation.

        Returns the :class:`matplotlib.animation.FuncAnimation` so the
        caller can display it inline (`HTML(anim.to_jshtml())`) or save
        it with ``save_path=...``.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        if states is None:
            states = self.trace_differentiation(num_steps=num_steps)
        frame_idx = np.linspace(0, len(states) - 1, num_frames).astype(int)
        coords = self.adata.obsm[basis]
        fig, ax = plt.subplots(figsize=figsize)
        # Frameless layout: no spines, no ticks, equal aspect so the
        # embedding isn't squashed, axes fill the full figure so the
        # GIF has no white border.
        fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0)
        ax.set_aspect("equal")
        ax.axis("off")
        # Lock data limits to the UMAP extent so the camera doesn't pan
        # as different cells are highlighted across frames.
        pad = 0.04 * np.ptp(coords, axis=0)
        ax.set_xlim(coords[:, 0].min() - pad[0], coords[:, 0].max() + pad[0])
        ax.set_ylim(coords[:, 1].min() - pad[1], coords[:, 1].max() + pad[1])
        density0 = states[frame_idx[0]]
        order = density0.argsort()
        vmin0 = float(density0.min())
        vmax0 = float(np.quantile(density0, vmax_quantile))
        scat = ax.scatter(coords[order, 0], coords[order, 1],
                           c=density0[order], cmap=cmap, s=4,
                           vmin=vmin0, vmax=max(vmax0, vmin0 + 1e-12))

        def update(i):
            d = states[frame_idx[i]]
            order_i = d.argsort()
            scat.set_offsets(coords[order_i])
            scat.set_array(d[order_i])
            vmin_i = float(d.min())
            vmax_i = float(np.quantile(d, vmax_quantile))
            # Per-frame autoscale — required so log_prob traces (which
            # are negative) actually render, and so the colormap tracks
            # the diffusion's growing/shrinking dynamic range.
            scat.set_clim(vmin_i, max(vmax_i, vmin_i + 1e-12))
            ax.set_title(f"Frame {frame_idx[i]}")
            return (scat,)

        anim = FuncAnimation(fig, update, frames=num_frames,
                              interval=1000 / fps, blit=False)
        if save_path:
            anim.save(save_path, writer=PillowWriter(fps=fps))
        return anim

    # ============================================================ tree
    def compute_tree_structure(self, threshold: float = 0.1) -> dict:
        """Port of MIRA ``mira.time.get_tree_structure`` (mira/pseudotime/
        pseudotime.py:get_tree_structure).

        Iteratively merges the two lineages whose **branch time** is the
        latest into a parent supercluster, where the branch time is the
        first pseudotime at which one lineage's log-fold-change versus
        the other exceeds ``threshold``. Each merge writes a
        ``tree_state`` label onto every cell whose pseudotime is past
        that branch and lies in one of the two merged lineages.

        Writes ``adata.obs['tree_states']``, ``adata.uns['tree_state_names']``
        and ``adata.uns['connectivities_tree']`` exactly as MIRA does
        (so :func:`mira.pl.plot_stream` would accept them too).
        """
        if self.result is None or self.result.fate_probabilities is None:
            raise RuntimeError("Call fit() first (with compute_fates=True)")
        import networkx as nx
        F = self.result.fate_probabilities.astype(np.float64)
        names_init = [str(self.adata.obs[self.groupby].iloc[c])
                      if self.groupby else f"L{j}"
                      for j, c in enumerate(self.result.terminal_cells)]
        pt = self._pt
        # Start cell = the global min-pseudotime cell (MIRA's convention).
        start_cell = int(np.argmin(pt))

        def _prob_fc(branch_probs):
            ep = 0.01
            return (np.log2(branch_probs + ep)
                    - np.log2(branch_probs[start_cell:start_cell + 1] + ep))

        def _branch_time(i, j, pt_, fc_, th):
            mask = (fc_[:, i] > 0) | (fc_[:, j] > 0)
            div = fc_[mask, i] - fc_[mask, j]
            s1 = pt_[mask][div > th]
            s2 = pt_[mask][div < -th]
            if len(s1) == 0 and len(s2) == 0:
                return pt_.max()
            if len(s1) == 0:
                return float(s2.min())
            if len(s2) == 0:
                return float(s1.min())
            return float(max(s1.min(), s2.min()))

        def _merge_rows(x, c1, c2):
            return np.hstack([(x[:, c1] + x[:, c2])[:, None],
                              x[:, ~np.isin(np.arange(x.shape[-1]),
                                            [c1, c2])]])

        branch_probs = F.copy()
        lineages = (_prob_fc(branch_probs) >= 0)
        names = list(names_init)
        n_cells, n_lin = branch_probs.shape
        tree_states = np.zeros(n_cells, dtype=np.int64)
        tree = nx.DiGraph()
        states_assigned = 1

        while n_lin > 1:
            fc = _prob_fc(branch_probs)
            split = np.full((n_lin, n_lin), -1.0)
            for i in range(n_lin - 1):
                for j in range(i + 1, n_lin):
                    split[i, j] = _branch_time(i, j, pt, fc, threshold)
            t_branch = float(split.max())
            i, j = np.where(split == t_branch)
            m1, m2 = int(i[0]), int(j[0])
            parent = (names[m1], names[m2])

            assign = (pt >= t_branch) & (lineages[:, m1] | lineages[:, m2])
            assign = assign & ~tree_states.astype(bool)
            div = fc[assign, m1] - fc[assign, m2]
            idx_assign = np.where(assign)[0]

            tree_states[idx_assign[div > 0]] = states_assigned
            tree.add_edge(parent, names[m1], branch_time=t_branch,
                           state=states_assigned)
            states_assigned += 1
            tree_states[idx_assign[div < 0]] = states_assigned
            tree.add_edge(parent, names[m2], branch_time=t_branch,
                           state=states_assigned)
            states_assigned += 1

            lineages = _merge_rows(lineages.astype(int), m1, m2).astype(bool)
            branch_probs = _merge_rows(branch_probs, m1, m2)
            names = [parent] + [n for k, n in enumerate(names) if k not in (m1, m2)]
            n_lin = branch_probs.shape[1]

        tree.add_edge('Root', names[0], branch_time=-1.0, state=0)

        def _leaves(node):
            if isinstance(node, tuple):
                return [*_leaves(node[0]), *_leaves(node[1])]
            return [node]

        def _node_name(node):
            return node if node == 'Root' else ', '.join(sorted(set(_leaves(node))))

        state_names = {e[2]['state']: _node_name(e[1]) for e in tree.edges(data=True)}
        node_list = list(tree.nodes)
        adj = nx.to_numpy_array(tree, nodelist=node_list, weight='branch_time')

        tree_state_names = [_node_name(n) for n in node_list]
        tree_state_labels = [state_names[s] for s in tree_states]

        prefix = self._prefix_from_pseudotime_key()
        self.adata.obs[f'{prefix}_tree_states'] = pd.Categorical(tree_state_labels)
        self.adata.uns[f'{prefix}_tree_state_names'] = tree_state_names
        self.adata.uns[f'{prefix}_connectivities_tree'] = adj
        self._tree_state_names = tree_state_names
        self._tree_adj = adj
        return {
            'tree_states': tree_state_labels,
            'tree_state_names': tree_state_names,
            'connectivities_tree': adj,
        }

    # ============================================================ streamplot
    def _dendrogram_levels(self, tree_adj: np.ndarray) -> np.ndarray:
        """Vertical layout of tree nodes — verbatim port of MIRA's
        ``get_dendogram_levels``.

        Leaves get integer positions 0, 1, 2, … in DFS order; internal
        nodes get the **mean** of their children. We do **not** rescale
        to [0, 1] — adjacent leaves are spaced one unit apart so a
        ``max_bar_height ≤ 1`` strip fits within ±0.5 around each
        centerline with no overlap into the neighbouring branch.
        """
        import networkx as nx
        G = nx.from_numpy_array(tree_adj, create_using=nx.DiGraph)
        root = next(n for n, d in G.in_degree() if d == 0)
        positions: dict = {}
        counter = [0]

        def _set(node):
            if node in positions:
                return positions[node]
            children = list(G.successors(node))
            if not children:
                positions[node] = float(counter[0])
                counter[0] += 1
            else:
                positions[node] = float(np.mean([_set(c) for c in children]))
            return positions[node]

        # DFS-tree traversal mirrors MIRA's ``dfs_predecessors[::-1]``
        # order so leaf indexing matches a typical tree visualisation.
        dfs_tree = list(nx.dfs_predecessors(G, root))[::-1] + [root]
        for node in dfs_tree:
            _set(node)
        for node in G.nodes:
            if node not in positions:
                _set(node)
        return np.array([positions[k] for k in G.nodes])

    # ------------------------------------------------------------------ colors
    def _cluster_color_map(self) -> dict:
        """Return ``{category_name: hex}`` from ``adata.uns[f'{groupby}_colors']``.

        Empty dict if ``groupby`` is unset or scanpy hasn't written the
        palette yet (it writes it the first time you call
        ``sc.pl.embedding(color=groupby)``).
        """
        import matplotlib.colors as mc
        if self.groupby is None:
            return {}
        key = f"{self.groupby}_colors"
        if key not in self.adata.uns:
            return {}
        series = self.adata.obs[self.groupby]
        cats = (list(series.cat.categories)
                if hasattr(series, "cat") else list(series.unique()))
        colors = list(self.adata.uns[key])
        return {str(c): mc.to_hex(colors[i])
                for i, c in enumerate(cats) if i < len(colors)}

    @staticmethod
    def _cellrank_create_colors(
        base_color,
        n: int,
        hue_range=(-0.1, 0.1),
        saturation_range=(-0.3, 0.3),
        value_range=(-0.3, 0.3),
    ) -> list:
        """Verbatim port of ``cellrank._utils._colors._create_colors``.

        Returns ``n`` hex colour variations of ``base_color`` by
        perturbing HSV channels within the given ranges. Doubles the
        sample count and takes every-other to avoid neighbour colours
        becoming visually indistinguishable. For ``n == 1`` returns the
        base unchanged.

        The duplicate-cluster macrostate path uses
        ``saturation_range=None`` (CellRank ``_map_names_and_colors``)
        so we shift hue + value only.
        """
        import matplotlib.colors as mc
        base_hsv = mc.rgb_to_hsv(mc.to_rgb(base_color))
        if n == 1:
            return [mc.to_hex(mc.hsv_to_rgb(base_hsv))]
        n2 = n * 2
        res = np.repeat(base_hsv[..., np.newaxis], n2, axis=1).T
        for i, r in enumerate((hue_range, saturation_range, value_range)):
            if r is None:
                continue
            r_low, r_high = sorted(r)
            c = base_hsv[i]
            res[:, i] = np.linspace(max(c + r_low, 0), min(c + r_high, 1), n2)
        res_rgb = [mc.hsv_to_rgb(c) for c in res]
        res_hex = [mc.to_hex(c) for c in res_rgb]
        return res_hex[::2]

    def _resolve_cluster_colors(self, categories,
                                  fallback_palette: str = "Set3") -> list:
        """List-aligned hex colors for ``categories`` (may contain duplicates).

        Reuses ``adata.uns[f'{groupby}_colors']`` when a category name
        matches a cluster. For categories that occur more than once
        (e.g. two macrostates that collapse to the same cluster name),
        CellRank's ``_map_names_and_colors`` is invoked with
        ``saturation_range=None`` so each occurrence is visually
        distinguishable (hue + value variants of the base colour).
        """
        import matplotlib.colors as mc
        import matplotlib.pyplot as plt
        from collections import Counter, defaultdict
        cmap_base = self._cluster_color_map()
        cmap_fallback = plt.get_cmap(fallback_palette)
        counts = Counter(map(str, categories))
        # Pre-generate the variant pool per duplicated cluster name.
        variants: dict = {}
        for cat, total in counts.items():
            if total <= 1:
                continue
            base = cmap_base.get(cat)
            if base is None:
                continue
            variants[cat] = self._cellrank_create_colors(
                base, total, saturation_range=None
            )
        seen: dict = defaultdict(int)
        out: list = []
        n_unknown = 0
        for cat in categories:
            cat_s = str(cat)
            if cat_s in cmap_base:
                base = cmap_base[cat_s]
            else:
                base = mc.to_hex(cmap_fallback(n_unknown % cmap_fallback.N))
                n_unknown += 1
            total = counts[cat_s]
            idx = seen[cat_s]
            seen[cat_s] += 1
            if total > 1 and cat_s in variants:
                out.append(variants[cat_s][idx])
            else:
                out.append(mc.to_hex(base))
        return out

    def plot_stream(
        self,
        data,                                                # str | list[str]
        *,
        style: Literal['stream', 'swarm'] = 'stream',
        log_pseudotime: bool = False,
        max_bar_height: float = 0.6,
        window_size: int = 101,
        clip: float | None = 3.0,
        scale_features: bool = False,
        split: bool = False,
        hide_feature_threshold: float = 0.0,
        legend_cols: int = 5,
        palette: str | list = 'Set3',
        color: str = 'black',
        size: float = 5,
        max_swarm_density: float = 2000,
        title: str | None = None,
        figsize=(10, 5),
        plots_per_row: int = 4,
        scaffold_linecolor: str = 'lightgrey',
        scaffold_linewidth: float = 1.0,
        linecolor: str = 'black',
        linewidth: float | None = None,
        ax=None,
    ):
        """Port of ``mira.pl.plot_stream`` (mira/plots/streamplot.py).

        Renders the lineage tree as a scaffold of horizontal segments
        (one per ``tree_state``), then for each segment plots
        feature(s) along pseudotime as either:

        * ``style='stream'`` — savgol-smoothed, ``cumsum``-stacked
          filled area; multiple features stack on top of one another
          inside each segment.
        * ``style='swarm'`` — categorical feature rendered as a
          density-capped swarm (one dot per cell, ``y`` jittered
          inside the segment, colour by feature value).

        Accepts the same shape of API call as MIRA::

            fate.plot_stream(data='true_cell', style='swarm', palette='Set3')
            fate.plot_stream(data=['LGR5', 'LEF1', 'DSG4'], style='stream',
                              clip=3, window_size=301, scale_features=True,
                              split=True)
        """
        import matplotlib.pyplot as plt
        from scipy.signal import savgol_filter
        if self.result is None:
            raise RuntimeError("Call fit() first")
        prefix = self._prefix_from_pseudotime_key()
        if f'{prefix}_tree_states' not in self.adata.obs.columns:
            self.compute_tree_structure()

        # Resolve features
        if isinstance(data, str):
            data_list = [data]
        else:
            data_list = list(data)

        # Feature matrix
        cols = []
        for name in data_list:
            v = np.asarray(self.adata.obs_vector(name))
            cols.append(v[:, None])
        features = np.hstack(cols)
        numeric = np.issubdtype(features.dtype, np.number)
        if not numeric and style != 'swarm':
            raise ValueError("Non-numeric features must be plotted with style='swarm'")

        pt = self._pt.copy()
        if log_pseudotime:
            pt = np.log1p(pt - pt.min())

        tree_state = self.adata.obs[f'{prefix}_tree_states'].astype(str).values
        tree_state_names = self._tree_state_names
        tree_adj = self._tree_adj
        centerlines = self._dendrogram_levels(tree_adj)

        # ----- single-axis layout: tree scaffold + per-segment fill -----
        if split and numeric and features.shape[1] > 1:
            n = features.shape[1]
            nrow = (n + plots_per_row - 1) // plots_per_row
            fig, axes = plt.subplots(nrow, min(n, plots_per_row),
                                      figsize=(4 * min(n, plots_per_row), 3 * nrow),
                                      squeeze=False)
            for k, name in enumerate(data_list):
                self.plot_stream(
                    name, style=style, log_pseudotime=False, max_bar_height=max_bar_height,
                    window_size=window_size, clip=clip, scale_features=scale_features,
                    split=False, palette=palette, ax=axes.flat[k],
                    scaffold_linecolor=scaffold_linecolor, scaffold_linewidth=scaffold_linewidth,
                    linecolor=linecolor, linewidth=linewidth,
                )
                axes.flat[k].set_title(name)
            for k in range(n, nrow * plots_per_row):
                axes.flat[k].axis('off')
            if title:
                fig.suptitle(title, fontsize=14, y=1.02)
            fig.tight_layout()
            return fig

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Normalise features (clip + scale) — port of MIRA
        # `_normalize_numerical_features`. Bounds the stream to
        # ``[centerline ± max_bar_height/2]``.
        if numeric and clip is not None:
            mu = features.mean(0, keepdims=True)
            sd = features.std(0, keepdims=True)
            features = np.clip(features, mu - clip * sd, mu + clip * sd)
        if numeric:
            f_min = features.min(0, keepdims=True)
            f_max = features.max(0, keepdims=True)
            if scale_features:
                rng = np.maximum(f_max - f_min, 1e-12)
                features = (features - f_min) / rng
            else:
                features = features - f_min
            features = np.maximum(features, 0)
            if style == 'stream' and not split and features.shape[1] > 1:
                # Stacked multi-feature stream — bound the *total* height
                # so cumsum tops out at ``max_bar_height``.
                normaliser = max(float(features.sum(-1).max()), 1e-12)
                features = features / normaliser * max_bar_height
            else:
                # Single-feature stream (or split-panel) — per-feature
                # max so each panel fills the band regardless of absolute
                # expression scale.
                normaliser = np.maximum(features.max(axis=0), 1e-12)
                features = features / normaliser * max_bar_height

        # Walk the tree (BFS from root) and plot each segment.
        import networkx as nx
        G = nx.from_numpy_array(tree_adj, create_using=nx.DiGraph)
        root_idx = next(n for n, d in G.in_degree() if d == 0)
        bfs = [(root_idx, root_idx), *list(nx.bfs_edges(G, root_idx))]

        # Pre-compute one colour per leaf, reusing the cluster palette
        # (``adata.uns[f'{groupby}_colors']``) with HLS perturbation when
        # the same cluster name appears on multiple leaves.
        leaf_nodes = [n for n in G.nodes if G.out_degree(n) == 0]
        leaf_names_list = [tree_state_names[n] for n in leaf_nodes]
        leaf_colors_list = self._resolve_cluster_colors(leaf_names_list)
        leaf_color_by_node = dict(zip(leaf_nodes, leaf_colors_list))

        # Segment positions cache so we can draw scaffold connectors.
        seg_pt_min, seg_pt_max = {}, {}
        seg_pt_min[root_idx] = float(pt.min())
        seg_pt_max[root_idx] = float(pt.min())

        for parent, child in bfs:
            name = tree_state_names[child]
            mask = tree_state == name
            if not mask.any():
                seg_pt_max[child] = seg_pt_max.get(parent, pt.min())
                seg_pt_min[child] = seg_pt_max[child]
                continue
            seg_pt_full = pt[mask]
            order = np.argsort(seg_pt_full)
            seg_pt_full = seg_pt_full[order]
            seg_feats_full = features[mask][order]

            # MIRA-style clipping: a child segment starts where the
            # parent segment ends. Any cells that ended up tagged with a
            # leaf/internal state but sit before the parent's split time
            # (a tree-build artifact for very-low-fate-probability cells)
            # are pushed back into the parent's range so the scaffold
            # stays clean and the savgol smoothing of post-split fates
            # isn't dragged by early-time outliers.
            if parent != child:
                t_split = seg_pt_max.get(parent, float(seg_pt_full.min()))
                clip_mask = seg_pt_full >= t_split
                if clip_mask.sum() >= 3:
                    seg_pt = seg_pt_full[clip_mask]
                    seg_feats = seg_feats_full[clip_mask] if numeric else seg_feats_full[clip_mask]
                else:
                    seg_pt = seg_pt_full
                    seg_feats = seg_feats_full
            else:
                seg_pt = seg_pt_full
                seg_feats = seg_feats_full

            seg_pt_min[child] = float(seg_pt.min())
            seg_pt_max[child] = float(seg_pt.max())
            cl = float(centerlines[child])

            # Scaffold connector: L-shaped — short vertical at the split,
            # then horizontal only up to the child segment's start. The
            # child segment's stream/swarm draws the rest of the line.
            if parent != child:
                cl_p = float(centerlines[parent])
                t_split = seg_pt_max.get(parent, seg_pt_min[child])
                ax.vlines(t_split, ymin=min(cl, cl_p), ymax=max(cl, cl_p),
                          color=scaffold_linecolor, linewidth=scaffold_linewidth)
                ax.hlines(cl, xmin=t_split, xmax=seg_pt_min[child],
                          color=scaffold_linecolor, linewidth=scaffold_linewidth)

            # ---- segment rendering ----
            if style == 'stream' and numeric:
                # Savgol smoothing (matches MIRA — window auto-clipped to
                # the segment length, must be odd).
                max_ws = max(3, len(seg_pt) - 1)
                if max_ws % 2 == 0:
                    max_ws -= 1
                ws = min(window_size, max_ws)
                if ws % 2 == 0:
                    ws -= 1
                ws = max(3, ws)
                # Hide-feature threshold: zero out values below threshold
                # so very-weakly-expressed features don't clutter the
                # plot (MIRA `hide_feature_threshold`).
                seg_feats_use = np.where(seg_feats < hide_feature_threshold,
                                          0.0, seg_feats)
                smooth = savgol_filter(seg_feats_use, ws, 1, axis=0)
                cum = np.cumsum(smooth, axis=-1)
                # MIRA `center_baseline=True`: baseline = features[:, -1] / 2
                # *regardless* of how many features there are. This makes
                # the stream symmetric around the centerline (expands both
                # up AND down) for single- and multi-feature alike.
                base = cum[:, -1] / 2
                bottom = cl - base
                top_cum = cum - base[:, None] + cl
                colors = plt.get_cmap(palette).colors if isinstance(palette, str) else palette
                prev = bottom
                # Match MIRA `_plot_stream_segment`: single feature → use
                # the ``color`` parameter (default 'black'); multi-feature
                # → cycle through ``palette``.
                use_palette = len(data_list) > 1
                for k in range(top_cum.shape[1]):
                    if use_palette:
                        c = colors[k % len(colors)]
                    else:
                        c = color
                    ax.fill_between(seg_pt, prev, top_cum[:, k], color=c,
                                     alpha=0.9, edgecolor=linecolor,
                                     linewidth=linewidth or 0.1)
                    prev = top_cum[:, k]

            elif style == 'swarm':
                # Categorical-feature swarm — port of MIRA
                # ``mira.plots.swarmplot._plot_swarm_segment``: downsample
                # by *density* (cells per unit pseudotime) instead of by
                # absolute count, jitter on the centerline, draw without
                # spines, and use the same colour map per segment so the
                # legend below stays consistent.
                col = features[mask][order, 0]
                col_str = col.astype(str)
                # Build / reuse the global colour map (computed once across
                # the whole tree so legend categories don't shift between
                # segments).
                if not hasattr(self, "_swarm_color_cache") or self._swarm_color_cache.get("ms_key") is not id(palette):
                    all_cats = sorted(set(features[:, 0].astype(str)))
                    # Prefer the cluster palette when every label is a
                    # known cluster name; fall back to ``palette`` cmap
                    # otherwise. Duplicates get HLS-perturbed copies.
                    cluster_map = self._cluster_color_map()
                    if cluster_map and all(c in cluster_map for c in all_cats):
                        color_list = self._resolve_cluster_colors(all_cats)
                        cmap_used = {c: color_list[i] for i, c in enumerate(all_cats)}
                    else:
                        cm_name = palette if isinstance(palette, str) else "Set3"
                        cm = plt.get_cmap(cm_name, max(3, len(all_cats)))
                        cmap_used = {c: cm(i % cm.N) for i, c in enumerate(all_cats)}
                    self._swarm_color_cache = {
                        "ms_key": id(palette),
                        "map": cmap_used,
                    }
                color_map = self._swarm_color_cache["map"]

                rng = np.random.default_rng(0)
                pt_range = float(np.ptp(seg_pt)) if len(seg_pt) > 1 else 1.0
                density = len(seg_pt) / max(pt_range, 1e-9)
                if density > max_swarm_density:
                    keep_rate = max_swarm_density / density
                    mask_keep = rng.random(len(seg_pt)) < keep_rate
                else:
                    mask_keep = np.ones(len(seg_pt), dtype=bool)
                kept_pt = seg_pt[mask_keep]
                kept_lbl = col_str[mask_keep]
                cs = [color_map[c] for c in kept_lbl]
                # Initial scatter on the centerline; Beeswarm then displaces
                # the dots perpendicular to pseudotime so they tile into a
                # rectangular strip without overlap — exactly MIRA's look.
                pts = ax.scatter(
                    kept_pt, np.full(len(kept_pt), cl),
                    c=cs, s=size, edgecolors='none', linewidths=0,
                )
                _Beeswarm(orient="h", width=max_bar_height)(pts, cl)

            # Leaf label — coloured by cluster palette (perturbed for
            # duplicates) so a stream plot reads like a coloured tree.
            if G.out_degree(child) == 0:
                ax.text(seg_pt_max[child] * 1.005, cl, name,
                        fontsize=10, va='center', ha='left', fontweight='bold',
                        color=leaf_color_by_node.get(child, 'black'))

        # Explicit ylim with padding for Beeswarm dot radii, mirroring
        # MIRA `_build_tree`'s `ax.set(ylim=(plot_bottom, max_centerline
        # + max_bar_height/2 + 0.15))`. Without this matplotlib's auto-
        # ylim leaves the top lineage's cell dots clipped at the figure
        # edge.
        min_cl = float(np.min(centerlines))
        max_cl = float(np.max(centerlines))
        plot_bottom = min_cl - max_bar_height / 2 - 0.3
        plot_top = max_cl + max_bar_height / 2 + 0.15
        ax.set_ylim(plot_bottom, plot_top)

        # MIRA-style pseudotime triangle at the bottom of the plot.
        bar_h = 0.04 * (plot_top - plot_bottom)
        base = plot_bottom + 0.05
        ax.fill_between([float(pt.min()), float(pt.max())],
                         [base, base + bar_h], [base, base],
                         color='lightgrey', linewidth=0)
        ax.text(float(pt.max()) * 1.005 + 0.005 * float(np.ptp(pt)), base,
                "Time", fontsize=11, ha='left', va='bottom')

        # Multi-feature stream legend: one Patch handle per feature,
        # MIRA-style colour swatches placed below the plot.
        if style == 'stream' and numeric and len(data_list) > 1:
            from matplotlib.patches import Patch
            cmap_colors = (plt.get_cmap(palette).colors
                            if isinstance(palette, str) else palette)
            handles = [
                Patch(facecolor=cmap_colors[k % len(cmap_colors)],
                      edgecolor=linecolor, linewidth=linewidth or 0.1,
                      label=data_list[k])
                for k in range(len(data_list))
            ]
            ax.legend(handles=handles, loc='upper center',
                       bbox_to_anchor=(0.5, -0.06),
                       ncol=min(legend_cols, len(data_list)),
                       frameon=False, fontsize=10)

        # Categorical swarm legend (only for swarm style with a finite set
        # of labels — matches MIRA's behaviour on `style='swarm'`).
        if style == 'swarm' and hasattr(self, "_swarm_color_cache"):
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], marker='o', linestyle='none', markersize=8,
                        markerfacecolor=c, markeredgecolor='none', label=lbl)
                for lbl, c in self._swarm_color_cache["map"].items()
            ]
            ax.legend(handles=handles, loc='center left',
                       bbox_to_anchor=(1.02, 0.5), frameon=False,
                       fontsize=9, title=data_list[0] if data_list else None)

        ax.set_xlabel('pseudotime')
        ax.set_xticks(np.linspace(float(pt.min()), float(pt.max()), 6))
        ax.set_yticks([])
        ax.set_title(title or ('Lineage stream' if style == 'stream' else 'Lineage swarm'))
        for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
        return ax

    # =========================================================== streamplot
    def write_branch_keys(
        self,
        lineage_key: str | None = None,
        pseudotime_key: str | None = None,
    ) -> tuple:
        """Write the lineage assignment + lineage-resolved pseudotime
        keys that :func:`ov.pl.branch_streamplot` expects. Each cell is
        assigned to its argmax fate-lineage. Returns the
        ``(lineage_key, pseudotime_key)`` actually written.
        """
        if self.result is None or self.result.fate_probabilities is None:
            raise RuntimeError("Call fit() first")
        import pandas as pd
        prefix = self._prefix_from_pseudotime_key()
        lk = lineage_key or f"{prefix}_lineage"
        pk = pseudotime_key or self.pseudotime_key
        F = self.result.fate_probabilities
        names = [str(self.adata.obs[self.groupby].iloc[c])
                 if self.groupby else f"L{j}"
                 for j, c in enumerate(self.result.terminal_cells)]
        # Preserve all terminal lineage names as categories — even when a
        # lineage receives zero argmax votes (e.g. small terminals like
        # Delta/Epsilon in pancreas) — so downstream tools that iterate
        # ``cat.categories`` still see every lineage.
        unique_names = list(dict.fromkeys(names))
        cat = pd.Categorical(
            [names[i] for i in F.argmax(axis=1)],
            categories=unique_names,
        )
        self.adata.obs[lk] = cat
        # Propagate the cluster palette to the lineage column so scanpy/
        # PseudotimeFate downstream tools (dynamic_trends, etc.) reuse
        # ``adata.uns[f'{groupby}_colors']`` instead of falling back to a
        # generic colormap. Order follows ``cat.categories``; duplicate
        # cluster names (two macrostates of the same cluster) get
        # HLS-perturbed copies (CellRank style).
        cluster_map = self._cluster_color_map()
        if cluster_map:
            base_for_cat = list(cat.categories)
            self.adata.uns[f"{lk}_colors"] = self._resolve_cluster_colors(base_for_cat)
        return lk, pk

    # ------------------------------------------------------------------ plots
    def plot_eigengap(self, n_macrostates: int | None = None,
                       ax=None, figsize=(5, 3)):
        """Plot the leading eigenvalues of the transition matrix and the
        consecutive gap between them. Inspired by :func:`mira.plots.
        plot_eigengap`. Large gaps indicate natural cut-points for
        ``n_macrostates`` — pick K just above a gap.

        Must be called after :meth:`fit`. If ``n_macrostates`` is given,
        the corresponding bar is highlighted.
        """
        if self.result is None:
            raise RuntimeError("Call fit() first")
        import matplotlib.pyplot as plt
        K = n_macrostates or self.n_macrostates
        P = self.result.transition_matrix
        vals, _ = spla.eigs(P.T.astype(np.float64),
                            k=min(K + 5, P.shape[0] - 2),
                            which="LM", maxiter=1000, tol=1e-6)
        vals = np.sort(np.abs(vals))[::-1]
        gaps = -np.diff(vals)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x = np.arange(1, len(gaps) + 1)
        bars = ax.bar(x, gaps, color="#888888")
        if K - 1 < len(gaps):
            bars[K - 1].set_color("#d62728")
            ax.axvline(K, color="#d62728", ls="--", lw=0.8,
                       label=f"K = {K}")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("between eigenvalues k and k+1")
        ax.set_ylabel("|λ_k| − |λ_{k+1}|")
        ax.set_title("Macrostate eigengap")
        return ax

    def plot_fate(self, basis: str = "X_umap", cmap: str = "magma",
                   ncols: int = 3, figsize: tuple | None = None,
                   show_terminals: bool = True):
        """One UMAP panel per inferred lineage, coloured by fate
        probability. The terminal cell of each lineage is overlaid as a
        yellow star (``show_terminals=True``). Equivalent to MIRA's
        per-branch UMAP overlay (``sc.pl.umap(color='lineage_prob')``)
        but plots all lineages in a single grid.
        """
        if self.result is None or self.result.fate_probabilities is None:
            raise RuntimeError("Call fit() first (with compute_fates=True)")
        import matplotlib.pyplot as plt
        F = self.result.fate_probabilities
        n_lin = F.shape[1]
        nrows = (n_lin + ncols - 1) // ncols
        figsize = figsize or (3.5 * ncols, 3.2 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                  squeeze=False)
        names = [str(self.adata.obs[self.groupby].iloc[c])
                 if self.groupby else f"L{j}"
                 for j, c in enumerate(self.result.terminal_cells)]
        prefix = self._prefix_from_pseudotime_key()
        coords = self.adata.obsm[basis]
        for j in range(n_lin):
            ax = axes.flat[j]
            sc_obj = ax.scatter(coords[:, 0], coords[:, 1], c=F[:, j],
                                 cmap=cmap, s=2, vmin=0, vmax=1)
            if show_terminals:
                c = self.result.terminal_cells[j]
                ax.scatter(coords[c, 0], coords[c, 1], s=200, marker="*",
                            edgecolor="black", facecolor="yellow",
                            linewidth=1.2, zorder=10)
            ax.set_title(f"{prefix} → {names[j]}")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(sc_obj, ax=ax, shrink=0.7)
        for k in range(n_lin, nrows * ncols):
            axes.flat[k].axis("off")
        fig.tight_layout()
        return fig

    def plot_macrostates(
        self,
        which: str = "terminal",
        basis: str = "X_umap",
        ax=None,
        size: float = 30.0,
        background_color: str = "lightgrey",
        background_size: float = 2.0,
        legend: bool = True,
        legend_loc: str = "right",
    ):
        """Colour cells assigned to a macrostate by that macrostate; the
        rest of the cells are shown in ``background_color``. Mirrors
        CellRank's ``g.plot_macrostates(which='all' | 'terminal')`` —
        each macrostate is named ``<cluster>`` (unique) or
        ``<cluster>_<i>`` (duplicates), and the duplicate copies are
        coloured with HLS-perturbed variants of the cluster's base
        colour from ``adata.uns[f'{groupby}_colors']``.

        Parameters
        ----------
        which
            ``'terminal'`` (default) shows only the inferred terminal
            macrostates — the rest of the macrostates are folded into
            the grey background. ``'all'`` colours every macrostate.
        """
        if self.result is None:
            raise RuntimeError("Call fit() first")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 4))
        coords = self.adata.obsm[basis]
        ms = self.result.macrostate_assignment
        if which == "all":
            ms_ids = np.arange(int(ms.max()) + 1)
        elif which == "terminal":
            ms_ids = np.asarray(self.result.terminal_macrostates, dtype=int)
        else:
            raise ValueError("`which` must be 'terminal' or 'all'.")

        # Build per-macrostate label: cluster name of its representative
        # cell (the same convention CellRank uses), suffixed with _i
        # when several macrostates share a cluster.
        if self.groupby is not None:
            terminal_cells = list(self.result.terminal_cells)
            terminal_ms = list(self.result.terminal_macrostates)
            ms_to_repcell = {int(m): int(c) for m, c
                             in zip(terminal_ms, terminal_cells)}
            base_names = []
            for m in ms_ids:
                if int(m) in ms_to_repcell:
                    base_names.append(
                        str(self.adata.obs[self.groupby].iloc[ms_to_repcell[int(m)]])
                    )
                else:
                    cells = np.where(ms == int(m))[0]
                    if len(cells):
                        # vote majority cluster
                        labels = self.adata.obs[self.groupby].iloc[cells].astype(str)
                        base_names.append(labels.value_counts().idxmax())
                    else:
                        base_names.append(f"M{int(m)}")
        else:
            base_names = [f"M{int(m)}" for m in ms_ids]

        from collections import Counter
        counts = Counter(base_names)
        seen: dict = {}
        ms_labels = []
        for nm in base_names:
            if counts[nm] > 1:
                seen[nm] = seen.get(nm, 0) + 1
                ms_labels.append(f"{nm}_{seen[nm]}")
            else:
                ms_labels.append(nm)
        ms_colors = self._resolve_cluster_colors(base_names)

        # Grey background — every cell not in `ms_ids`.
        in_set = np.isin(ms, ms_ids)
        ax.scatter(coords[~in_set, 0], coords[~in_set, 1],
                    s=background_size, c=background_color,
                    linewidths=0, edgecolors='none')

        # Each macrostate on top.
        for m, lbl, col in zip(ms_ids, ms_labels, ms_colors):
            sel = ms == int(m)
            if not sel.any():
                continue
            ax.scatter(coords[sel, 0], coords[sel, 1],
                        s=size, c=col, linewidths=0,
                        edgecolors='none', label=lbl)

        if legend:
            handles = [
                Line2D([0], [0], marker='o', linestyle='none',
                        markerfacecolor=col, markeredgecolor='none',
                        markersize=8, label=lbl)
                for lbl, col in zip(ms_labels, ms_colors)
            ]
            if legend_loc == "right":
                ax.legend(handles=handles, loc='center left',
                           bbox_to_anchor=(1.02, 0.5), frameon=False,
                           fontsize=9)
            else:
                ax.legend(handles=handles, loc=legend_loc,
                           frameon=False, fontsize=9)

        ax.set_xticks([]); ax.set_yticks([])
        title = ("terminal states" if which == "terminal"
                  else "all macrostates")
        ax.set_title(title)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        return ax

    # ------------------------------------------------------------------ adata
    def _write_adata(self, res: PseudotimeFateResult) -> None:
        prefix = self._prefix_from_pseudotime_key()
        self.adata.obs[f'{prefix}_macrostate'] = pd.Categorical(
            res.macrostate_assignment.astype(str)
        )
        self.adata.uns[f'{prefix}_macrostate_residency'] = res.macrostate_residency
        self.adata.uns[f'{prefix}_terminal_macrostates'] = res.terminal_macrostates
        if res.fate_probabilities is not None:
            self.adata.obsm[f'{prefix}_fate_probabilities'] = res.fate_probabilities
            self.adata.obs[f'{prefix}_lineage_entropy'] = res.lineage_entropy
