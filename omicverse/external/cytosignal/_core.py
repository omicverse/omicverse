"""NumPy/SciPy re-implementation of the R CytoSignal pipeline (Liu et al., Nat Genet 2026).

Mirrors the default workflow of the R package (v0.5.1)::

    removeLowQuality -> changeUniprot -> inferEpsParams -> findNN
    -> imputeLR -> inferIntrScore -> inferSignif

Orientation: matrices are **cells x genes** here (R uses genes x cells), and a
neighbour graph ``W`` is stored as in R -- column ``i`` holds the weights of
the neighbours ``j`` of centre cell ``i`` -- so an imputation is ``W.T @ X``.

Every random draw goes through :class:`~._rrng.RRandom`, which reproduces R's
``set.seed`` / ``sample``; with the same seed the permutation nulls (and hence
p-values) are identical to the R package.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ._db import CytoSignalDB, to_uniprot
from ._rrng import RRandom

NORM_METHODS = ("default", "cpm", "none", "scanpy")


# --------------------------------------------------------------------------- #
# neighbour graphs                                                            #
# --------------------------------------------------------------------------- #
def infer_eps_params(scale_factor: float, r_eps_real: float = 200.0, thresh: float = 0.001):
    """R ``inferEpsParams``: diffusion radius and Gaussian sigma in coordinate units."""
    r_eps = r_eps_real / scale_factor
    sigma = r_eps / math.sqrt(-2 * math.log(thresh))
    return r_eps, sigma


def _gauss(x, sigma: float):
    return np.exp(-np.power(x, 2) / (2 * sigma * sigma)) / (sigma * math.sqrt(2 * math.pi))


def _colsum(m: sp.csc_matrix) -> np.ndarray:
    out = np.zeros(m.shape[1])
    nz = np.diff(m.indptr) > 0
    out[nz] = np.add.reduceat(m.data, m.indptr[:-1][nz])
    return out


def _scale_columns(m: sp.csc_matrix, fac: np.ndarray) -> sp.csc_matrix:
    m = m.copy()
    m.data = m.data / np.repeat(fac, np.diff(m.indptr))
    return m


def _set_diag(m: sp.csc_matrix, mask: np.ndarray, value) -> sp.csc_matrix:
    """``Matrix::diag(m)[mask] <- value`` on a graph without self-loops
    (no structural zeros are stored for the unmasked cells)."""
    if m.diagonal().any():
        raise ValueError("graph already has self-weights")
    idx = np.where(mask)[0]
    m = sp.csc_matrix(m + sp.csc_matrix((np.full(idx.size, value, dtype=np.float64), (idx, idx)), shape=m.shape))
    m.sort_indices()
    return m


def find_nn_gaueps(loc: np.ndarray, eps: float, sigma: float, self_weight="auto"):
    """R ``findNNGauEB``: Gaussian-weighted epsilon-ball neighbours (column = centre)."""
    from scipy.spatial import cKDTree

    loc = np.asarray(loc, dtype=np.float64)
    n = loc.shape[0]
    x, y = loc[:, 0], loc[:, 1]
    pairs = cKDTree(loc).query_pairs(eps * (1 + 1e-9), output_type="ndarray")
    a, b = pairs[:, 0], pairs[:, 1]
    # select_EB_rcpp2: strict box test, then dist < eps; a zero distance is not stored.
    rows = np.concatenate([b, a])  # neighbour j
    cols = np.concatenate([a, b])  # centre i
    xi, yi, xj, yj = x[cols], y[cols], x[rows], y[rows]
    box = (xj < xi + eps) & (xj > xi - eps) & (yj < yi + eps) & (yj > yi - eps)
    d = np.sqrt(np.power(xj - xi, 2) + np.power(yj - yi, 2))
    keep = box & (d < eps) & (d != 0)
    dist = sp.csc_matrix((d[keep], (rows[keep], cols[keep])), shape=(n, n))
    dist.sort_indices()
    has = np.diff(dist.indptr) > 0
    w = dist.copy()
    w.data = _gauss(w.data, sigma)
    if isinstance(self_weight, (int, float)) and not isinstance(self_weight, bool):
        if not 0 <= self_weight <= 1:
            raise ValueError("self_weight must be within [0, 1].")
        w = _scale_columns(w, _colsum(w))
        w = _set_diag(w, has, float(self_weight))
    elif self_weight == "auto":
        w = _set_diag(w, has, float(_gauss(np.array([1e-9]), sigma)[0] * 5))
        w = _scale_columns(w, _colsum(w))
    elif self_weight == "sum_1":
        w = _set_diag(w, has, float(_gauss(np.array([1e-9]), sigma)[0]))
        w = _scale_columns(w, _colsum(w))
    else:
        raise ValueError("self_weight must be a number in [0, 1], 'auto' or 'sum_1'.")
    return w.tocsc(), np.diff(dist.indptr)


def delaunay_edges(loc: np.ndarray) -> np.ndarray:
    """Unique Delaunay edges ``(k, 2)`` of a 2-D point set (duplicate points dropped)."""
    from scipy.spatial import Delaunay

    tri = Delaunay(np.asarray(loc, dtype=np.float64))
    s = tri.simplices
    e = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def find_nn_dt(loc: np.ndarray, max_r: float, mode: str = "weight_sum_2", edges=None):
    """R ``findNNDT``: Delaunay neighbours within ``max_r``, mean-weighted."""
    loc = np.asarray(loc, dtype=np.float64)
    n = loc.shape[0]
    e = delaunay_edges(loc) if edges is None else np.asarray(edges)
    d = np.sqrt(np.sum(np.power(loc[e[:, 0]] - loc[e[:, 1]], 2), axis=1))
    e = e[d <= max_r]
    g = sp.csc_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    g.sort_indices()
    deg = _colsum(g)
    has = deg > 0
    if mode == "weight_sum_2":
        g = _scale_columns(g, deg)
        g = _set_diag(g, has, 1.0)
    elif mode == "weight_sum_1":
        g = _set_diag(g, has, 1.0)
        g = _scale_columns(g, deg + 1)
    else:
        raise ValueError("mode must be 'weight_sum_2' or 'weight_sum_1'.")
    return g.tocsc(), e


def to_mean(g: sp.csc_matrix) -> sp.csc_matrix:
    """R ``to_mean``: every stored entry of a column becomes ``1 / nnz(column)``."""
    g = g.copy()
    nnz = np.diff(g.indptr)
    g.data = np.repeat(1.0 / np.where(nnz > 0, nnz, 1), nnz)
    return g


# --------------------------------------------------------------------------- #
# normalisation / scores                                                      #
# --------------------------------------------------------------------------- #
def norm_counts(x: sp.csr_matrix, scale: np.ndarray, method: str = "default") -> sp.csr_matrix:
    """R ``normCounts`` on a cells x genes matrix (nonzeros only)."""
    if method == "none":
        return x
    x = sp.csr_matrix(x, copy=True)
    x.data = x.data / np.repeat(scale, np.diff(x.indptr))
    if method == "default":
        x.data = np.log1p(x.data * 1e4)
    elif method == "cpm":
        x.data = np.log1p(x.data * 1e6)
    elif method == "scanpy":
        x.data = x.data * float(np.median(scale))
    else:
        raise ValueError(f"Unknown norm method {method!r}; use one of {NORM_METHODS}.")
    return x


@dataclass
class _Intr:
    ids: list[str]
    lig: list[list[int]]   # uniprot row indices per interaction (DB order)
    rec: list[list[int]]


def check_intr(symbols: list[str], comp: dict) -> _Intr:
    """R ``checkIntr``: interactions whose every component is measured."""
    pos = {u: i for i, u in enumerate(symbols)}
    ids, lig, rec = [], [], []
    for intr, c in comp.items():
        if all(u in pos for u in c["combined"]):
            ids.append(intr)
            lig.append([pos[u] for u in c["ligands"]])
            rec.append([pos[u] for u in c["receptors"]])
    return _Intr(ids, lig, rec)


def _subunit_sum(x: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Column sums over subunits in database order (``arma::sum(rows, 0)``)."""
    out = np.zeros((x.shape[0], len(groups)))
    width = max((len(g) for g in groups), default=0)
    for t in range(width):
        k = np.array([i for i, g in enumerate(groups) if len(g) > t], dtype=np.int64)
        idx = np.array([groups[i][t] for i in k], dtype=np.int64)
        out[:, k] += x[:, idx]
    return out


def score_lr(lig: np.ndarray, rec: np.ndarray, intr: _Intr) -> np.ndarray:
    """``inferScoreLR_cpp``: (sum of ligand subunits) * (sum of receptor subunits)."""
    return _subunit_sum(lig, intr.lig) * _subunit_sum(rec, intr.rec)


def _dense(m) -> np.ndarray:
    return m.toarray() if sp.issparse(m) else np.asarray(m)


# --------------------------------------------------------------------------- #
# pipeline                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class CytoSignalState:
    counts: sp.csr_matrix            # cells x uniprot raw counts
    lib_size: np.ndarray
    uniprot: list[str]
    genes: list[str]
    loc: np.ndarray
    eps: float
    sigma: float
    graphs: dict = field(default_factory=dict)   # 'diffusion' (GauEps) / 'contact' (DT) / 'raw'
    imputed: dict = field(default_factory=dict)  # name -> (cells x uniprot, scale)


def preprocess(counts, var_names, *, db: CytoSignalDB, counts_thresh: float = 300):
    """R ``removeLowQuality`` (cells) + ``changeUniprot``.

    ``counts`` is cells x genes raw counts. Returns ``(keep_cells, sub_counts,
    lib_size, uniprot, gene_symbols)``; ``lib_size`` uses all genes, as in R.
    (``removeLowQuality``'s gene filter never touches database genes and so has
    no effect on the result.)
    """
    counts = sp.csr_matrix(counts, dtype=np.float64)
    lib = np.asarray(counts.sum(axis=1)).ravel()
    keep = lib > counts_thresh
    rows, uni = to_uniprot(var_names, db)
    sub = counts[keep][:, rows].tocsr()
    sub.sort_indices()
    genes = [str(var_names[i]) for i in rows]
    return keep, sub, lib[keep], uni, genes


def build_state(counts, lib, uniprot, genes, loc, *, scale_factor, r_eps_real=200.0, thresh=0.001,
                self_weight="auto", dt_mode="weight_sum_2", dt_edges=None) -> CytoSignalState:
    """``inferEpsParams`` + ``findNN`` + ``imputeLR``."""
    eps, sigma = infer_eps_params(scale_factor, r_eps_real, thresh)
    st = CytoSignalState(counts=counts, lib_size=np.asarray(lib, dtype=np.float64), uniprot=list(uniprot),
                         genes=list(genes), loc=np.asarray(loc, dtype=np.float64), eps=eps, sigma=sigma)
    w_gau, _ = find_nn_gaueps(st.loc, eps, sigma, self_weight=self_weight)
    w_dt, _ = find_nn_dt(st.loc, eps, mode=dt_mode, edges=dt_edges)
    n = counts.shape[0]
    st.graphs = {"diffusion": w_gau, "contact": w_dt, "raw": sp.identity(n, format="csc")}
    for name in ("diffusion", "contact"):
        w = st.graphs[name]
        st.imputed[name] = (sp.csr_matrix(w.T @ counts), np.asarray(w.T @ st.lib_size).ravel())
    st.imputed["raw"] = (counts, st.lib_size)
    return st


def _norm_avg(st: CytoSignalState, x, scale, method, dt_mean_t):
    x = norm_counts(x, scale, method)
    if method != "none":
        x = dt_mean_t @ x
    return _dense(x)


@dataclass
class SlotResult:
    name: str
    lig_slot: str
    rec_slot: str
    db: str
    interactions: list[str]
    score: np.ndarray            # cells x I, smoothed LR-score
    score_raw: np.ndarray        # cells x I, before smoothing
    n_null: int
    pval: np.ndarray | None = None   # has-neighbour cells x I
    padj: np.ndarray | None = None
    has_neighbor: np.ndarray | None = None
    result: dict | None = None
    result_hq: dict | None = None
    intr: _Intr | None = None


def infer_slot(st: CytoSignalState, db: CytoSignalDB, rng: RRandom, *, lig_slot: str, rec_slot: str,
               intr_db: str, perm_size: int = 100000, norm_method: str = "default", name: str | None = None,
               progress: bool = False) -> SlotResult:
    """``inferScoreLR`` -> ``permuteLR`` -> ``inferNullScoreLR`` -> ``smoothScoreLR``
    -> ``getPvalues`` for one ligand/receptor imputation pair.

    The null scores are streamed one permutation round at a time; only the
    count of null scores below each observed score is kept (``findInterval``),
    so memory stays O(cells x interactions) even for ``perm_size = 1e5``.
    """
    n = st.counts.shape[0]
    intr = check_intr(st.uniprot, db.components(intr_db))
    dt_mean = to_mean(st.graphs["contact"])
    dt_mean_t = sp.csr_matrix(dt_mean.T)
    lig_x, lig_s = st.imputed[lig_slot]
    rec_x, rec_s = st.imputed[rec_slot]
    lig = _norm_avg(st, lig_x, lig_s, norm_method, dt_mean_t)
    rec = _norm_avg(st, rec_x, rec_s, norm_method, dt_mean_t)
    score = score_lr(lig, rec, intr)

    # ---- permuteLR: R draws every shuffle up front, then smoothScoreLR draws its own.
    perm_size = int(perm_size)
    if perm_size < n:
        perm_size = n
    times = math.ceil(perm_size / n)
    each = math.ceil(perm_size / times)
    perms = [rng.sample(n) for _ in range(times)]
    sample_idx = rng.sample(n, each)
    smooth_idx = [(rng.sample(each), rng.sample(each)) for _ in range(times)]

    w_gau = st.graphs["diffusion"]
    dt_mean_tc = sp.csr_matrix(dt_mean.T)[sample_idx]  # to_mean(DT)[, sample] transposed
    graphs = {"diffusion": st.graphs["diffusion"], "contact": st.graphs["contact"], "raw": None}

    def null_block(slot, perm, pinv):
        # permuteLR.sparse: dge %*% W[perm, ] -> normCounts -> %*% to_mean(DT)[perm, sample]
        # (the DT average is applied whatever the normalisation, as in R)
        x, s = st.imputed["raw"]
        x, s = x[pinv], s[pinv]
        g = graphs[slot]
        if g is not None:
            x, s = sp.csr_matrix(g.T @ x), np.asarray(g.T @ s).ravel()
        x = norm_counts(x, s, norm_method)
        return _dense(dt_mean_tc[:, perm] @ x)

    score_smooth = np.asarray(w_gau.T @ score)  # smoothScoreLR: t(t(score) %*% nn.graph)
    has = np.diff(st.graphs["contact"].indptr) > 0  # receptor slot 'Raw' carries the DT graph
    x = score_smooth[has]
    below = np.zeros(x.shape, dtype=np.int64)
    nonzero = np.zeros(len(intr.ids), dtype=bool)
    it = range(times)
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc=f"CytoSignal null ({name or lig_slot + '-' + rec_slot})")
        except ImportError:  # pragma: no cover
            pass
    for r in it:
        perm = perms[r]
        pinv = np.empty(n, dtype=np.int64)
        pinv[perm] = np.arange(n)
        b = score_lr(null_block(lig_slot, perm, pinv), null_block(rec_slot, perm, pinv), intr)
        nonzero |= (b != 0).any(axis=0)
        a_idx, b_idx = smooth_idx[r]
        gsub = sp.csr_matrix(w_gau[a_idx][:, b_idx])
        b = np.sort(np.asarray(gsub.T @ b), axis=0)
        for k in range(b.shape[1]):
            below[:, k] += np.searchsorted(b[:, k], x[:, k], side="left")
    # inferNullScoreLR: drop interactions whose (unsmoothed) null scores are all zero
    keep = np.where(nonzero)[0]
    intr = _Intr([intr.ids[k] for k in keep], [intr.lig[k] for k in keep], [intr.rec[k] for k in keep])
    n_null = times * each
    pval = 1 - below[:, keep] / n_null
    score, score_smooth = score[:, keep], score_smooth[:, keep]
    return SlotResult(name=name or f"{lig_slot}-{rec_slot}", lig_slot=lig_slot, rec_slot=rec_slot, db=intr_db,
                      interactions=intr.ids, score=score_smooth, score_raw=score, n_null=n_null,
                      pval=pval, has_neighbor=has, intr=intr)


def graph_spatial_fdr(nn_graph: sp.csc_matrix, pval: np.ndarray) -> np.ndarray:
    """R ``graphSpatialFDRNew``: Cydar-style density-weighted BH per interaction.

    Weights are ``1 / max(neighbour weight)`` of each cell in the DT graph.
    Sums and cumulative sums use extended precision like R's ``sum``/``cumsum``.
    """
    g = sp.csc_matrix(nn_graph)
    n = g.shape[1]
    t = np.full(n, np.nan)
    for j in range(n):
        lo, hi = g.indptr[j], g.indptr[j + 1]
        rows, vals = g.indices[lo:hi], g.data[lo:hi]
        if vals.sum() == 0:
            continue
        rest = vals[rows != j]
        # R: max(weights[-j]) over the dense column, i.e. zeros count too
        t[j] = max(rest.max(), 0.0) if rest.size else 0.0
    t = t[~np.isnan(t)]
    w = 1 / t
    w[np.isinf(w)] = 1
    padj = np.empty_like(pval)
    for k in range(pval.shape[1]):
        p = pval[:, k]
        o = np.argsort(p, kind="stable")
        po, wo = p[o], w[o]
        cs = np.cumsum(wo.astype(np.longdouble))
        tot = cs[-1]
        v = np.asarray((tot * po) / cs, dtype=np.float64)
        adj = np.empty_like(v)
        adj[o] = np.minimum.accumulate(v[::-1])[::-1]
        padj[:, k] = np.minimum(adj, 1)
    return padj


def bh_fdr(pval: np.ndarray) -> np.ndarray:
    """Per-cell (row-wise) BH, R ``inferSignif(fdr.method = 'fdr')``."""
    from scipy.stats import false_discovery_control

    return np.vstack([false_discovery_control(row) for row in pval])


def infer_signif(st: CytoSignalState, res: SlotResult, db: CytoSignalDB, *, p_thresh=0.05,
                 reads_thresh=100, sig_thresh=100, fdr_method="spatialFDR") -> SlotResult:
    """R ``inferSignif``: adjusted p-values, significant cells, high-quality filter."""
    has = res.has_neighbor
    g = st.graphs["contact"][has][:, has].tocsc()
    if fdr_method == "spatialFDR":
        padj = graph_spatial_fdr(g, res.pval)
    elif fdr_method == "fdr":
        padj = bh_fdr(res.pval)
    else:
        raise ValueError("fdr_method must be 'spatialFDR' or 'fdr'.")
    res.padj = padj
    cells = np.where(has)[0]
    sig = {}
    for k, intr in enumerate(res.interactions):
        hit = cells[padj[:, k] < p_thresh]
        if hit.size:
            sig[intr] = hit
    order = sorted(sig, key=lambda k: -len(sig[k]))  # stable: R order(decreasing=TRUE)
    res.result = {k: sig[k] for k in order}
    # filterRes: drop interactions with any component detected in < reads_thresh cells
    det = np.diff(sp.csc_matrix(st.counts).indptr)
    low = {u for u, c in zip(st.uniprot, det) if c < reads_thresh}
    comp = db.components(res.db)
    low_intr = {i for i, c in comp.items() if any(u in low for u in c["combined"])}
    hq = {k: v for k, v in res.result.items() if k not in low_intr and len(v) >= sig_thresh}
    res.result_hq = {k: hq[k] for k in sorted(hq, key=lambda k: -len(hq[k]))}
    return res


DEFAULT_SLOTS = {
    # name: (ligand imputation, receptor imputation, database)
    "diffusion": ("diffusion", "raw", "diffusion"),
    "contact": ("contact", "raw", "contact"),
    "raw": ("raw", "raw", "contact"),
}


def run_pipeline(counts, var_names, loc, *, db: CytoSignalDB, scale_factor: float = 1.0,
                 r_eps_real: float = 200.0, thresh: float = 0.001, counts_thresh: float = 300,
                 perm_size: int = 100000, seed: int = 42, slots=("diffusion", "contact", "raw"),
                 recep_smooth: bool = False, self_weight="auto", dt_mode: str = "weight_sum_2",
                 norm_method: str = "default", p_thresh: float = 0.05, reads_thresh: float = 100,
                 sig_thresh: float = 100, fdr_method: str = "spatialFDR", dt_edges=None, progress=False):
    """Full CytoSignal run (vignette defaults). Returns ``(keep_cells, state, {slot: SlotResult})``."""
    keep, sub, lib, uni, genes = preprocess(counts, var_names, db=db, counts_thresh=counts_thresh)
    if sub.shape[1] == 0:
        raise ValueError("No CytoSignal database genes found in var_names.")
    loc = np.asarray(loc, dtype=np.float64)[keep]
    st = build_state(sub, lib, uni, genes, loc, scale_factor=scale_factor, r_eps_real=r_eps_real, thresh=thresh,
                     self_weight=self_weight, dt_mode=dt_mode, dt_edges=dt_edges)
    rng = RRandom(seed)
    out = {}
    order = [s for s in ("diffusion", "contact", "raw") if s in slots]
    if recep_smooth and "raw" in order:
        order.remove("raw")  # R skips Raw-Raw when receptors are smoothed
    for s in order:
        lig_slot, rec_slot, intr_db = DEFAULT_SLOTS[s]
        if recep_smooth:
            rec_slot = "contact"
        r = infer_slot(st, db, rng, lig_slot=lig_slot, rec_slot=rec_slot, intr_db=intr_db, perm_size=perm_size,
                       norm_method=norm_method, name=s, progress=progress)
        out[s] = infer_signif(st, r, db, p_thresh=p_thresh, reads_thresh=reads_thresh,
                              sig_thresh=sig_thresh, fdr_method=fdr_method)
    return keep, st, out
