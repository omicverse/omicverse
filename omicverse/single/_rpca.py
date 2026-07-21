r"""Seurat-style RPCA (reciprocal PCA) integration — a faithful Python port.

This module reproduces Seurat v5's
``IntegrateLayers(method = RPCAIntegration, ...)`` /
``FindIntegrationAnchors(reduction = "rpca")`` + ``IntegrateEmbeddings``
pipeline in pure Python (numpy / scipy / scikit-learn), so users can run RPCA
batch correction directly on an :class:`~anndata.AnnData` without switching to
R. Requested in omicverse issue #883.

The algorithm, mirroring the Seurat source function-for-function:

1. **Per-batch scale + PCA** on a shared set of integration features
   (``ScaleData`` with ``scale.max = 10`` → ``RunPCA``). ``Loadings`` are the
   right singular vectors of the (cells × genes) scaled matrix; ``Embeddings``
   are ``scaled.T @ loadings``.
2. **Reciprocal projection** (:func:`ReciprocalProject`): for a pair (i, j),
   project each batch's scaled data onto the *other* batch's loadings, giving
   two reciprocal spaces — batch-i's PCA (native i + projected j) and batch-j's
   PCA (projected i + native j). Each is column-scaled by its per-dim SD then
   L2-normalized per cell.
3. **Anchor finding** (:func:`FindNN` → :func:`FindAnchorPairs`): mutual nearest
   neighbors between the two batches across the reciprocal spaces.
4. **Anchor scoring** (:func:`ScoreAnchors`): shared-neighborhood overlap
   (SNN-style), rescaled by the 1st/90th score percentiles.
5. **Weighting + correction** (:func:`FindWeights` → ``IntegrateDataC``): each
   query cell is corrected by a Gaussian-kernel-weighted average of anchor
   batch vectors, applied to the *original* joint PCA embedding, mirroring
   ``IntegrateEmbeddings(reductions = orig)``.
6. **Sample tree** (:func:`_build_sample_tree`) for >2 batches: batches are
   merged in ``hclust`` order of pairwise anchor counts, each query corrected
   onto the running reference.

Validated numerically against R Seurat 5.4.0 (see
``tests/test_single_rpca.py``): given identical per-batch loadings/embeddings,
integration features and original joint PCA, the Python ``integrated.dr``
matches Seurat's to within floating-point / neighbor-tie tolerance.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Low-level building blocks (each mirrors a Seurat internal function)
# ---------------------------------------------------------------------------
def _as_dense(x) -> np.ndarray:
    """Return a dense float64 ndarray from dense or sparse input."""
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def _scale_data(data_gc: np.ndarray, scale_max: float = 10.0) -> np.ndarray:
    """Seurat ``ScaleData``: per-gene z-score (ddof=1), clipped to ``scale_max``.

    ``data_gc`` is a genes × cells matrix. Genes with zero variance become all
    zeros (matching Seurat, which leaves constant features at 0 after
    centering).
    """
    mean = data_gc.mean(axis=1, keepdims=True)
    # Seurat's ScaleData uses the sample SD (1/(n-1)).
    sd = data_gc.std(axis=1, ddof=1, keepdims=True)
    scaled = data_gc - mean
    nz = sd.ravel() > 0
    scaled[nz] = scaled[nz] / sd[nz]
    scaled[~nz] = 0.0
    if scale_max is not None:
        np.clip(scaled, -scale_max, scale_max, out=scaled)
    return scaled


def _run_pca(scaled_gc: np.ndarray, npcs: int):
    """Seurat ``RunPCA`` on scaled data (genes × cells).

    Returns ``(loadings, embeddings)`` where ``loadings`` is genes × npcs (the
    right singular vectors of the cells × genes matrix) and ``embeddings`` is
    cells × npcs (``scaled.T @ loadings``), exactly matching how Seurat derives
    ``cell.embeddings = u %*% diag(d)`` from ``irlba(t(scale.data))``.
    """
    a = scaled_gc.T  # cells × genes
    npcs = int(min(npcs, min(a.shape) - 1))
    # Full SVD then truncate — deterministic (no randomized-SVD variance),
    # which matters for the numerical parity test.
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    loadings = vt[:npcs].T                    # genes × npcs
    embeddings = u[:, :npcs] * s[:npcs]       # cells × npcs  (== a @ loadings)
    return loadings, embeddings


def _project(scaled_gc: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Seurat ``ProjectSVD`` (mode='pca', no center/scale): ``scaled.T @ V``."""
    return scaled_gc.T @ loadings


def _l2_reduction(emb: np.ndarray) -> np.ndarray:
    """Seurat reciprocal-space L2: column /sd (ddof=1), then per-row L2 norm.

    Non-finite entries (from zero-norm rows / zero-sd columns) are zeroed, as in
    Seurat's ``L2Norm``.
    """
    sd = emb.std(axis=0, ddof=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = emb / sd
    out[~np.isfinite(out)] = 0.0
    norm = np.sqrt((out ** 2).sum(axis=1, keepdims=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = out / norm
    out[~np.isfinite(out)] = 0.0
    return out


def _knn(data: np.ndarray, query: np.ndarray, k: int):
    """Exact Euclidean kNN. Returns ``(indices, distances)`` (query × k, 0-based).

    Exact (brute) search removes the annoy approximation so results are
    reproducible and comparable to R run with ``nn.method = "rann"``.
    """
    from sklearn.neighbors import NearestNeighbors

    k = int(min(k, data.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
    nn.fit(data)
    dist, idx = nn.kneighbors(query, return_distance=True)
    return idx, dist


def _find_anchor_pairs(idx_ab: np.ndarray, idx_ba: np.ndarray, k_anchor: int):
    """Seurat ``FindAnchorPairs``: mutual nearest neighbors.

    ``idx_ab[c1]`` = cell1 c1's nearest cells2; ``idx_ba[c2]`` = cell2 c2's
    nearest cells1. An anchor (c1, c2) is kept iff c2 ∈ idx_ab[c1][:k] **and**
    c1 ∈ idx_ba[c2][:k].
    """
    ka = int(min(k_anchor, idx_ab.shape[1], idx_ba.shape[1]))
    pairs = []
    ba_sets = [set(idx_ba[c2, :ka]) for c2 in range(idx_ba.shape[0])]
    for c1 in range(idx_ab.shape[0]):
        for c2 in idx_ab[c1, :ka]:
            if c1 in ba_sets[c2]:
                pairs.append((c1, int(c2)))
    return pairs


def _score_anchors(pairs, idx_aa, idx_ab, idx_ba, idx_bb, n1, k_score):
    """Seurat ``ScoreAnchors``: shared-neighborhood overlap, percentile-rescaled.

    Combined index space: dataset-1 cells are ``[0, n1)``; dataset-2 cells are
    offset by ``n1``. Returns an array of per-anchor scores in ``[0, 1]``.
    """
    ks = int(k_score)
    ks_aa = min(ks, idx_aa.shape[1])
    ks_ab = min(ks, idx_ab.shape[1])
    ks_ba = min(ks, idx_ba.shape[1])
    ks_bb = min(ks, idx_bb.shape[1])
    setA = [set(idx_aa[c1, :ks_aa]).union(idx_ab[c1, :ks_ab] + n1)
            for c1 in range(idx_aa.shape[0])]
    setB = [set(idx_ba[c2, :ks_ba]).union(idx_bb[c2, :ks_bb] + n1)
            for c2 in range(idx_bb.shape[0])]
    raw = np.array([len(setA[c1] & setB[c2]) for (c1, c2) in pairs],
                   dtype=np.float64)
    if raw.size == 0:
        return raw
    max_score = np.quantile(raw, 0.9)
    min_score = np.quantile(raw, 0.01)
    denom = max_score - min_score
    if denom <= 0:
        denom = 1.0
    score = (raw - min_score) / denom
    return np.clip(score, 0.0, 1.0)


def _find_pair_anchors(
    emb_i, emb_j, scaled_i, scaled_j, load_i, load_j,
    nn_ii, nn_jj, dims, k_anchor, k_score,
):
    """Full anchor-finding for a pair of batches (i, j).

    Mirrors the ``pca`` branch of Seurat ``FindIntegrationAnchors`` +
    ``FindAnchors_v3``: reciprocal projection → L2 → cross kNN (nnab / nnba) →
    MNN pairs → SNN scoring. ``k.filter`` is ``NA`` for rpca, so no filtering.

    Returns a list of ``(cell1_local, cell2_local, score)`` tuples.
    """
    d = np.asarray(dims)
    # Reciprocal projections onto the *other* batch's loadings.
    proj_i_in_j = _project(scaled_i, load_j)      # cells_i in batch-j PCA
    proj_j_in_i = _project(scaled_j, load_i)      # cells_j in batch-i PCA

    # reduction (ref):   batch-i PCA space  = [native i ; projected j]
    ref = np.vstack([emb_i[:, d], proj_j_in_i[:, d]])
    # reduction.2 (query): batch-j PCA space = [projected i ; native j]
    query = np.vstack([proj_i_in_j[:, d], emb_j[:, d]])
    ref = _l2_reduction(ref)
    query = _l2_reduction(query)

    n1 = emb_i.shape[0]
    ref_c1, ref_c2 = ref[:n1], ref[n1:]
    query_c1, query_c2 = query[:n1], query[n1:]

    k_neighbor = max(k_anchor, k_score)
    # nnab: cell1's neighbors among cells2, in reduction.2 (batch-j space)
    idx_ab, _ = _knn(data=query_c2, query=query_c1, k=k_neighbor)
    # nnba: cell2's neighbors among cells1, in reduction (batch-i space)
    idx_ba, _ = _knn(data=ref_c1, query=ref_c2, k=k_neighbor)

    pairs = _find_anchor_pairs(idx_ab, idx_ba, k_anchor)
    if not pairs:
        return []
    scores = _score_anchors(pairs, nn_ii, idx_ab, idx_ba, nn_jj, n1, k_score)
    return [(c1, c2, float(s)) for (c1, c2), s in zip(pairs, scores)]


# ---------------------------------------------------------------------------
# Weighting + correction (IntegrateEmbeddings core)
# ---------------------------------------------------------------------------
def _find_weights(query_orig, ref_anchor_cells, anchor_query_cells,
                  anchor_scores, k_weight, sd_weight):
    """Seurat ``FindWeights`` + ``FindWeightsC`` (min_dist = 0 path).

    ``query_orig`` : (n_query × dims) original-reduction coords of query cells.
    ``anchor_query_cells`` : per-anchor query-cell local index (may repeat).
    ``ref_anchor_cells`` : the unique query cells that are anchors.

    Returns a dense weight matrix ``W`` of shape (n_anchors × n_query), column
    (per query cell) normalized to sum 1.
    """
    n_query = query_orig.shape[0]
    n_anchor = len(anchor_query_cells)
    uniq = list(ref_anchor_cells)
    k = int(min(k_weight, len(uniq)))

    # kNN of every query cell among the *unique anchor* query cells.
    anchor_coords = query_orig[uniq]
    idx, dist = _knn(data=anchor_coords, query=query_orig, k=k)
    # Seurat: dist' = 1 - d / d[:, last]  (nearest→~1, k-th→0)
    last = dist[:, -1][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        distt = 1.0 - dist / last
    distt[~np.isfinite(distt)] = 0.0

    # Map each unique anchor cell → the integration-matrix rows sharing it.
    cell_to_rows = {}
    for row, qc in enumerate(anchor_query_cells):
        cell_to_rows.setdefault(qc, []).append(row)
    uniq_rows = [cell_to_rows[uc] for uc in uniq]

    W = np.zeros((n_anchor, n_query), dtype=np.float64)
    coef = (sd_weight ** 2) / 4.0   # 1 / (2/sd)^2
    for c in range(n_query):
        for jj in range(k):
            u = int(idx[c, jj])          # index into uniq
            dval = distt[c, jj]
            for row in uniq_rows[u]:
                # 1 - exp(-dist' * score / (2/sd)^2)
                W[row, c] = 1.0 - np.exp(-dval * anchor_scores[row] * coef)
    colsums = W.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = W / colsums
    W[~np.isfinite(W)] = 0.0
    return W


def _integrate_pair(ref_orig, query_orig, anchors_local, k_weight, sd_weight):
    """Correct ``query_orig`` onto ``ref_orig`` given ref→query anchors.

    ``anchors_local`` : list of ``(ref_cell, query_cell, score)`` with indices
    local to ref_orig / query_orig respectively.

    Returns the corrected query embedding (n_query × dims). Reference is left
    unchanged, exactly as ``PairwiseIntegrateReference`` keeps object.1 fixed.
    """
    if len(anchors_local) == 0:
        return query_orig.copy()

    k_weight = int(min(k_weight, len(anchors_local)))
    ref_cells = np.array([a[0] for a in anchors_local], dtype=int)
    query_cells = np.array([a[1] for a in anchors_local], dtype=int)
    scores = np.array([a[2] for a in anchors_local], dtype=np.float64)

    # FindIntegrationMatrix: batch vector = query - ref at each anchor.
    integration_matrix = query_orig[query_cells] - ref_orig[ref_cells]

    uniq_query_anchor = list(dict.fromkeys(query_cells.tolist()))
    W = _find_weights(query_orig, uniq_query_anchor, query_cells.tolist(),
                      scores, k_weight, sd_weight)
    # IntegrateDataC: corrected = query - W.T @ integration_matrix
    corrected = query_orig - W.T @ integration_matrix
    return corrected


# ---------------------------------------------------------------------------
# Sample tree (BuildSampleTree for >2 batches)
# ---------------------------------------------------------------------------
def _count_anchors(anchor_df, n_batches, obj_lengths):
    """Seurat ``CountAnchors``: lower-triangular anchor-count similarity."""
    sim = np.zeros((n_batches, n_batches))
    for i in range(n_batches):
        for j in range(n_batches):
            if i <= j:
                continue
            mask = (np.isin(anchor_df["dataset1"], [i, j]) &
                    np.isin(anchor_df["dataset2"], [i, j]))
            score = int(mask.sum())
            ncell = min(obj_lengths[i], obj_lengths[j])
            sim[i, j] = score / ncell if ncell else 0.0
    return sim


def _build_sample_tree(sim):
    """``BuildSampleTree``: hclust (complete linkage) on 1/similarity.

    Returns the R ``hclust$merge`` matrix: each row merges two items, negative
    entries are leaves (batch index, 1-based negated), positive entries refer to
    a previously formed cluster (1-based row index).
    """
    from scipy.cluster.hierarchy import linkage

    n = sim.shape[0]
    # condensed distance (upper triangle) with d = 1 / similarity
    d = []
    for i in range(n):
        for j in range(i + 1, n):
            s = sim[j, i] if i > j else sim[i, j]
            s = max(sim[i, j], sim[j, i])
            d.append(np.inf if s == 0 else 1.0 / s)
    d = np.array(d, dtype=np.float64)
    finite_max = d[np.isfinite(d)].max() if np.isfinite(d).any() else 1.0
    d[~np.isfinite(d)] = finite_max * 1e6 + 1.0
    Z = linkage(d, method="complete")

    # Convert scipy linkage → R hclust$merge convention.
    merge = np.zeros((n - 1, 2), dtype=int)
    for r in range(n - 1):
        for col in (0, 1):
            node = int(Z[r, col])
            if node < n:
                merge[r, col] = -(node + 1)      # leaf → -(1-based batch)
            else:
                merge[r, col] = node - n + 1      # cluster → 1-based row
    return merge


def _leaves_of(merge, row):
    """Return the list of batch indices (0-based) under a merge-tree node."""
    out = []
    for col in (0, 1):
        node = merge[row, col]
        if node < 0:
            out.append(-node - 1)
        else:
            out.extend(_leaves_of(merge, node - 1))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def rpca_integrate(
    adata,
    batch_key: str,
    *,
    n_pcs: int = 30,
    dims: Optional[Sequence[int]] = None,
    features: Optional[Sequence[str]] = None,
    n_features: int = 2000,
    layer: Optional[str] = None,
    reference: Optional[str] = None,
    orig_rep: Optional[str] = None,
    k_anchor: int = 5,
    k_score: int = 30,
    k_weight: int = 100,
    sd_weight: float = 1.0,
    scale_max: float = 10.0,
    key_added: str = "X_rpca",
    verbose: bool = True,
    _precomputed: Optional[dict] = None,
):
    r"""Seurat-style RPCA (reciprocal PCA) batch integration on an AnnData.

    Faithful Python port of Seurat v5
    ``IntegrateLayers(method = RPCAIntegration)`` — see the module docstring for
    the algorithm. Writes the integrated embedding to
    ``adata.obsm[key_added]`` (default ``'X_rpca'``) and returns ``adata``.

    Parameters
    ----------
    adata
        Log-normalized :class:`~anndata.AnnData` (``adata.X`` or ``layer``).
    batch_key
        ``adata.obs`` column with the batch label per cell.
    n_pcs
        Number of per-batch principal components used for anchor finding
        (Seurat ``dims = 1:n_pcs``, default 30).
    dims
        Explicit 0-based PC indices to use for anchors. Overrides ``n_pcs``.
    features
        Integration features (genes) shared across batches. If ``None``, the
        top ``n_features`` are selected by cross-batch HVG frequency
        (Seurat ``SelectIntegrationFeatures``).
    n_features
        Number of integration features to auto-select when ``features`` is
        ``None``.
    layer
        Layer holding log-normalized data (``None`` → ``adata.X``).
    reference
        Batch label to hold fixed as the integration reference. ``None`` builds
        an integrated reference across all batches via the anchor sample tree
        (Seurat default).
    orig_rep
        ``adata.obsm`` key of the *original joint* embedding to correct
        (Seurat ``orig.reduction``). ``None`` uses ``'scaled|original|X_pca'``
        or ``'X_pca'`` if present, else computes a joint PCA over ``features``.
    k_anchor, k_score, k_weight, sd_weight
        Seurat anchor/scoring/weighting parameters (defaults 5 / 30 / 100 / 1).
    scale_max
        ``ScaleData`` clipping value (Seurat default 10).
    key_added
        Destination ``adata.obsm`` key.

    Returns
    -------
    AnnData
        ``adata`` with ``adata.obsm[key_added]`` populated.
    """
    from .._settings import add_reference

    batches = adata.obs[batch_key].astype(str).to_numpy()
    uniq = list(dict.fromkeys(batches))
    if len(uniq) < 2:
        raise ValueError(
            f"{batch_key!r} has {len(uniq)} unique value(s); RPCA integration "
            f"requires >= 2 batches."
        )
    dims = np.arange(n_pcs) if dims is None else np.asarray(dims)

    if _precomputed is not None:
        # Test/advanced hook: inject R-computed features / per-batch
        # loadings+embeddings / orig joint PCA to isolate the algorithm.
        features = list(_precomputed["features"])
        per_batch = _precomputed["per_batch"]           # {batch: (load, emb)}
        orig_emb = np.asarray(_precomputed["orig"], dtype=np.float64)
        scaled_by_batch = _precomputed["scaled"]        # {batch: genes×cells}
    else:
        if features is None:
            features = _select_integration_features(
                adata, batch_key, n_features=n_features, layer=layer)
        features = [f for f in features if f in set(adata.var_names)]
        fidx = adata.var_names.get_indexer(features)

        per_batch = {}
        scaled_by_batch = {}
        for b in uniq:
            cells = np.where(batches == b)[0]
            sub = adata[cells]
            X = sub.layers[layer] if layer is not None else sub.X
            data_gc = _as_dense(X).T[fidx, :]           # genes × cells
            scaled = _scale_data(data_gc, scale_max=scale_max)
            load, emb = _run_pca(scaled, n_pcs)
            per_batch[b] = (load, emb)
            scaled_by_batch[b] = scaled

        orig_emb = _resolve_orig(adata, orig_rep, features, fidx, layer,
                                 n_pcs, scale_max)

    if verbose:
        print(f"   RPCA integration: {len(uniq)} batches, "
              f"{len(features)} features, {len(dims)} PCs")

    # ------- anchor finding for every batch pair (i < j) -------------------
    batch_cells = {b: np.where(batches == b)[0] for b in uniq}
    obj_lengths = [len(batch_cells[b]) for b in uniq]

    # within-batch neighbors (internal.neighbors) on native PCA embeddings
    k_internal = max(k_anchor, k_score) + 1
    nn_self = {}
    for b in uniq:
        emb = per_batch[b][1][:, dims]
        idx, _ = _knn(data=emb, query=emb, k=k_internal)
        nn_self[b] = idx

    all_anchors = []   # rows: (cell1_local, dataset1, cell2_local, dataset2, score)
    ref_set = None if reference is None else {uniq.index(reference)}
    for a in range(len(uniq)):
        for c in range(a + 1, len(uniq)):
            if ref_set is not None and a not in ref_set and c not in ref_set:
                continue
            bi, bj = uniq[a], uniq[c]
            pair = _find_pair_anchors(
                per_batch[bi][1], per_batch[bj][1],
                scaled_by_batch[bi], scaled_by_batch[bj],
                per_batch[bi][0], per_batch[bj][0],
                nn_self[bi], nn_self[bj],
                dims, k_anchor, k_score,
            )
            for (c1, c2, s) in pair:
                all_anchors.append((c1, a, c2, c, s))
                all_anchors.append((c2, c, c1, a, s))   # mirror
    if verbose:
        print(f"   Found {len(all_anchors) // 2} anchor pairs")

    anchor_df = {
        "cell1": np.array([r[0] for r in all_anchors], dtype=int),
        "dataset1": np.array([r[1] for r in all_anchors], dtype=int),
        "cell2": np.array([r[2] for r in all_anchors], dtype=int),
        "dataset2": np.array([r[3] for r in all_anchors], dtype=int),
        "score": np.array([r[4] for r in all_anchors], dtype=np.float64),
    }

    # ------- integrate along the sample tree -------------------------------
    corrected = orig_emb.copy()
    n_batch = len(uniq)
    if n_batch == 2 or reference is not None:
        _integrate_flat(corrected, orig_emb, anchor_df, uniq, batch_cells,
                        obj_lengths, reference, k_weight, sd_weight)
    else:
        sim = _count_anchors(anchor_df, n_batch, obj_lengths)
        merge = _build_sample_tree(sim)
        _integrate_tree(corrected, orig_emb, merge, anchor_df, uniq,
                        batch_cells, k_weight, sd_weight, verbose)

    adata.obsm[key_added] = corrected
    adata.uns.setdefault("rpca", {})[key_added] = {
        "n_pcs": int(n_pcs),
        "n_batches": n_batch,
        "batches": uniq,
        "n_features": len(features),
        "reference": reference,
        "n_anchors": len(all_anchors) // 2,
    }
    add_reference(adata, "RPCA",
                  "batch correction with Seurat-style reciprocal PCA (rpca)")
    return adata


def _integrate_flat(corrected, orig_emb, anchor_df, uniq, batch_cells,
                    obj_lengths, reference, k_weight, sd_weight):
    """2-batch (or fixed-reference) correction: query batches → reference."""
    if reference is not None:
        ref_idx = uniq.index(reference)
    else:
        # larger batch is the reference (Seurat: object.1 = larger)
        ref_idx = int(np.argmax(obj_lengths))
    ref_global = batch_cells[uniq[ref_idx]]
    for q in range(len(uniq)):
        if q == ref_idx:
            continue
        _apply_pair_correction(corrected, orig_emb, anchor_df,
                               ref_idx, q, batch_cells, uniq,
                               k_weight, sd_weight)


def _apply_pair_correction(corrected, orig_emb, anchor_df, ref_ds, query_ds,
                           batch_cells, uniq, k_weight, sd_weight,
                           ref_cells_global=None, query_cells_global=None):
    """Correct one query batch onto one reference (group) using their anchors."""
    if ref_cells_global is None:
        ref_cells_global = batch_cells[uniq[ref_ds]]
    if query_cells_global is None:
        query_cells_global = batch_cells[uniq[query_ds]]

    ref_pos = {g: i for i, g in enumerate(ref_cells_global)}
    query_pos = {g: i for i, g in enumerate(query_cells_global)}

    mask = ((anchor_df["dataset1"] == ref_ds) &
            (anchor_df["dataset2"] == query_ds))
    # local anchor cells → positions within the ref / query groups
    ref_local = batch_cells[uniq[ref_ds]][anchor_df["cell1"][mask]]
    query_local = batch_cells[uniq[query_ds]][anchor_df["cell2"][mask]]
    scores = anchor_df["score"][mask]
    anchors_local = [
        (ref_pos[r], query_pos[qq], s)
        for r, qq, s in zip(ref_local, query_local, scores)
        if r in ref_pos and qq in query_pos
    ]
    ref_orig = orig_emb[ref_cells_global]
    query_orig = orig_emb[query_cells_global]
    new_q = _integrate_pair(ref_orig, query_orig, anchors_local,
                            k_weight, sd_weight)
    corrected[query_cells_global] = new_q


def _integrate_tree(corrected, orig_emb, merge, anchor_df, uniq, batch_cells,
                    k_weight, sd_weight, verbose):
    """>2-batch correction following the hclust sample tree.

    Each merge corrects the smaller group (query) onto the larger group (ref).
    Anchors between the groups are pooled across constituent batches.
    """
    # cluster global-cell membership; start with singleton leaves
    def batch_globals(bset):
        return np.concatenate([batch_cells[uniq[b]] for b in sorted(bset)])

    for r in range(merge.shape[0]):
        left = set(_leaves_of(merge, r)) if False else None
        # resolve the two child node memberships
        members = []
        for col in (0, 1):
            node = merge[r, col]
            if node < 0:
                members.append({-node - 1})
            else:
                members.append(set(_leaves_of(merge, node - 1)))
        g1, g2 = members
        cells1 = batch_globals(g1)
        cells2 = batch_globals(g2)
        # larger group is the reference
        if len(cells2) > len(cells1):
            g1, g2 = g2, g1
            cells1, cells2 = cells2, cells1
        if verbose:
            print(f"   Merging batches {sorted(g2)} into {sorted(g1)}")
        # pool anchors with dataset1 in ref group and dataset2 in query group
        mask = (np.isin(anchor_df["dataset1"], list(g1)) &
                np.isin(anchor_df["dataset2"], list(g2)))
        ref_pos = {g: i for i, g in enumerate(cells1)}
        query_pos = {g: i for i, g in enumerate(cells2)}
        ds1 = anchor_df["dataset1"][mask]
        ds2 = anchor_df["dataset2"][mask]
        a_c1 = anchor_df["cell1"][mask]
        a_c2 = anchor_df["cell2"][mask]
        sc = anchor_df["score"][mask]
        anchors_local = []
        for d1, d2, c1, c2, s in zip(ds1, ds2, a_c1, a_c2, sc):
            rg = batch_cells[uniq[d1]][c1]
            qg = batch_cells[uniq[d2]][c2]
            if rg in ref_pos and qg in query_pos:
                anchors_local.append((ref_pos[rg], query_pos[qg], s))
        ref_orig = corrected[cells1]      # use running-corrected reference
        query_orig = corrected[cells2]
        new_q = _integrate_pair(ref_orig, query_orig, anchors_local,
                                k_weight, sd_weight)
        corrected[cells2] = new_q


# ---------------------------------------------------------------------------
# Feature selection + original-reduction resolution
# ---------------------------------------------------------------------------
def _select_integration_features(adata, batch_key, n_features=2000, layer=None):
    """Seurat ``SelectIntegrationFeatures``: rank genes by cross-batch HVG rank.

    Per batch, computes HVGs (scanpy ``seurat`` flavor) and ranks; genes are
    then ordered by how often they are variable across batches, breaking ties by
    summed rank — matching Seurat's selection semantics.
    """
    import scanpy as sc
    from collections import defaultdict

    batches = adata.obs[batch_key].astype(str).to_numpy()
    uniq = list(dict.fromkeys(batches))
    freq = defaultdict(int)
    rank_sum = defaultdict(float)
    for b in uniq:
        sub = adata[batches == b].copy()
        if layer is not None:
            sub.X = sub.layers[layer]
        try:
            hvg = sc.pp.highly_variable_genes(
                sub, n_top_genes=min(n_features, sub.n_vars),
                flavor="seurat", inplace=False)
        except Exception:
            hvg = sc.pp.highly_variable_genes(
                sub, n_top_genes=min(n_features, sub.n_vars), inplace=False)
        sel = np.where(hvg["highly_variable"].to_numpy())[0]
        # rank by dispersion (higher = better = lower rank number)
        disp = hvg["dispersions_norm"].to_numpy()
        order = np.argsort(-disp[sel])
        for rank, gi in enumerate(sel[order]):
            g = adata.var_names[gi]
            freq[g] += 1
            rank_sum[g] += rank
    # sort: more batches first, then lower summed rank
    genes = sorted(freq.keys(), key=lambda g: (-freq[g], rank_sum[g]))
    return genes[:n_features]


def _resolve_orig(adata, orig_rep, features, fidx, layer, n_pcs, scale_max):
    """Return the original joint embedding to be corrected (n_cells × dims)."""
    if orig_rep is not None:
        return np.asarray(adata.obsm[orig_rep], dtype=np.float64)
    for key in ("scaled|original|X_pca", "X_pca"):
        if key in adata.obsm:
            return np.asarray(adata.obsm[key], dtype=np.float64)
    # compute a joint PCA over the integration features
    X = adata.layers[layer] if layer is not None else adata.X
    data_gc = _as_dense(X).T[fidx, :]
    scaled = _scale_data(data_gc, scale_max=scale_max)
    _, emb = _run_pca(scaled, n_pcs)
    return emb
