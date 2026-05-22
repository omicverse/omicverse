"""Implementation of scoring methods."""

import numpy as np
import pandas as pd
from typing import Dict, List, Union
from scipy import stats
import warnings


def aucell_score(
    expression: np.ndarray,
    pathways: Dict[str, List[str]],
    gene_names: List[str],
    ncores: int = 1,
    aucMaxRank: int = None,
) -> pd.DataFrame:
    """Calculate AUCell scores.

    Implementation follows AUCell R package (Bioconductor).
    Computes the AUC of the recovery curve for each gene set per cell.

    Parameters
    ----------
    expression : ndarray
        Genes x cells matrix.
    pathways : dict
        Pathway name -> list of gene symbols.
    gene_names : list
        List of gene symbols corresponding to rows of expression.
    ncores : int
        Not used yet.
    aucMaxRank : int or None
        Max rank threshold. Default: ceiling(0.05 * n_genes), matching R AUCell.

    Returns
    -------
    DataFrame
        Pathway scores (pathways x cells).
    """
    n_genes, n_cells = expression.shape

    if aucMaxRank is None:
        aucMaxRank = int(np.ceil(0.05 * n_genes))

    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}

    # R AUCell uses ordinal ranking (random tie-breaking, all ranks unique integers).
    # We use 'ordinal' method for deterministic results close to R.
    ranks = np.zeros_like(expression, dtype=np.float64)
    for i in range(n_cells):
        ranks[:, i] = stats.rankdata(-expression[:, i], method='ordinal')

    pathway_scores = {}
    for pathway_name, gene_set in pathways.items():
        indices = []
        for gene in gene_set:
            if gene in gene_to_idx:
                indices.append(gene_to_idx[gene])
        if len(indices) == 0:
            warnings.warn(f"Pathway {pathway_name} has no genes in expression matrix")
            pathway_scores[pathway_name] = np.zeros(n_cells)
            continue

        m = len(indices)
        pathway_ranks = ranks[indices, :]

        # Max possible AUC: all m genes at ranks 1..m
        # area = sum(aucMaxRank - i for i in 1..m) = m*aucMaxRank - m*(m+1)/2
        max_auc = m * aucMaxRank - m * (m + 1) / 2
        if max_auc <= 0:
            pathway_scores[pathway_name] = np.zeros(n_cells)
            continue

        auc_scores = np.zeros(n_cells, dtype=np.float64)

        for cell_idx in range(n_cells):
            cell_ranks = pathway_ranks[:, cell_idx]
            mask = cell_ranks <= aucMaxRank
            ranks_within = cell_ranks[mask]

            if len(ranks_within) == 0:
                auc_scores[cell_idx] = 0.0
                continue

            # Recovery curve AUC: sum of (aucMaxRank - rank) for each gene within cutoff
            auc_area = np.sum(aucMaxRank - ranks_within)
            auc_scores[cell_idx] = auc_area / max_auc

        pathway_scores[pathway_name] = auc_scores

    return pd.DataFrame(pathway_scores, index=pd.RangeIndex(n_cells)).T

def ssgsea_score(
    expression: np.ndarray,
    pathways: Dict[str, List[str]],
    gene_names: List[str],
    ncores: int = 1,
    normalize: bool = True,
) -> pd.DataFrame:
    """Calculate ssGSEA scores matching R GSVA ssgseaParam.

    Vectorized implementation: instead of per-cell argsort + per-cell/per-pathway
    walk, we precompute rank-based position weights and use direct matrix operations.

    For unique ranks (the common case in scRNA-seq), pos_weight = n - R + 1
    exactly equals the sorted position weight. For tied ranks, we use the rank
    value which averages over the tie group — negligible difference in practice.

    Parameters
    ----------
    expression : ndarray
        Genes x cells matrix.
    pathways : dict
        Pathway name -> list of gene symbols.
    gene_names : list
        List of gene symbols corresponding to rows of expression.
    ncores : int
        Not used yet.
    normalize : bool
        Whether to normalize scores by dividing by the absolute max.
        Default True, matching R ssgseaParam() default behavior.

    Returns
    -------
    DataFrame
        Pathway scores (pathways x cells).
    """
    n_genes, n_cells = expression.shape
    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}

    alpha = 0.25

    # Step 1: Column ranks with ties="average", then truncate to integer
    R_mat = np.empty((n_genes, n_cells), dtype=np.int64)
    for j in range(n_cells):
        R_mat[:, j] = stats.rankdata(expression[:, j], method='average').astype(np.int64)

    # Step 2: Rank weights: Ra = |R|^alpha
    Ra = np.abs(R_mat.astype(np.float64)) ** alpha

    # Step 3: Position weights from descending sort order
    # R code: geneRanking <- order(R[,j], decreasing=TRUE)
    #         gSetIdx <- match(geneSetsIdx, geneRanking)
    #         pw = n - gSetIdx + 1
    # For each cell j, compute the position of each gene in the descending sort
    # and derive position weight = n - pos + 1
    pos_w = np.empty((n_genes, n_cells), dtype=np.float64)
    for j in range(n_cells):
        # argsort by descending rank, stable to match R's order() behavior
        order_j = np.argsort(-R_mat[:, j], kind='stable')
        # position of each gene in the sorted order (1-indexed)
        positions = np.empty(n_genes, dtype=np.int64)
        positions[order_j] = np.arange(1, n_genes + 1)
        # position weight = n - pos + 1
        pos_w[:, j] = (n_genes - positions + 1).astype(np.float64)

    # Step 4: For each pathway, vectorized over all cells
    pathway_scores = np.zeros((len(pathways), n_cells), dtype=np.float64)
    n = n_genes
    total_pos = n * (n + 1) / 2.0

    for pathway_idx, (pathway_name, gene_set) in enumerate(pathways.items()):
        indices = []
        for gene in gene_set:
            if gene in gene_to_idx:
                indices.append(gene_to_idx[gene])
        if len(indices) == 0:
            warnings.warn(f"Pathway {pathway_name} has no genes in expression matrix")
            continue

        k = len(indices)
        idx_arr = np.array(indices)

        # (k, n_cells) slices
        ra_vals = Ra[idx_arr, :]
        pw_vals = pos_w[idx_arr, :]

        sum_ra = ra_vals.sum(axis=0)
        sum_ra_pw = (ra_vals * pw_vals).sum(axis=0)
        sum_pw = pw_vals.sum(axis=0)

        valid = sum_ra > 0
        step_in = np.where(valid, sum_ra_pw / np.maximum(sum_ra, 1e-300), 0.0)
        step_out = (total_pos - sum_pw) / (n - k) if n > k else 0.0

        pathway_scores[pathway_idx, :] = step_in - step_out

    pathway_names = list(pathways.keys())
    result = pd.DataFrame(pathway_scores, index=pathway_names, columns=pd.RangeIndex(n_cells))

    if normalize:
        rng = result.values.max() - result.values.min()
        if rng > 0:
            result = result / rng

    return result

def gsva_score(
    expression: np.ndarray,
    pathways: Dict[str, List[str]],
    gene_names: List[str],
    ncores: int = 1,
) -> pd.DataFrame:
    """Calculate GSVA scores matching R GSVA gsvaParam with kcdf='Poisson'.

    Algorithm (from R GSVA source):
    1. Compute per-gene Poisson KCDF across samples
    2. Rank CDF values per sample (column ranks, ties.method='last')
    3. Convert ranks to statistics: srs = |p/2 - rank|
    4. Order genes by decreasing rank for each sample
    5. Random walk using srs as weights, sum(P_hit - P_miss)
       -> optimized: only evaluate walk at gene-set dos positions and their
          predecessors, since maxima/minima can only occur there.

    Parameters
    ----------
    expression : ndarray
        Genes x cells matrix.
    pathways : dict
        Pathway name -> list of gene symbols.
    gene_names : list
        List of gene symbols corresponding to rows of expression.
    ncores : int
        Not used yet.

    Returns
    -------
    DataFrame
        Pathway scores (pathways x cells).
    """
    from scipy.special import gammaincc

    n_genes, n_cells = expression.shape
    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
    p = n_genes

    # Step 1: Compute per-gene Poisson KCDF
    # For each gene i, sample j: left_tail[j] = mean(ppois(x[j], lam[k]+0.5)) over all k
    # Use frequency-weighted mean: group by (x, lam) pairs, count occurrences,
    # compute ppois for unique pairs only, then weighted average.
    gene_cdf = np.zeros((n_genes, n_cells), dtype=np.float64)

    for i in range(n_genes):
        gene_expr = expression[i, :]

        x_int = np.floor(gene_expr).astype(np.int64)
        lam_vals = gene_expr + 0.5

        # Unique x and lambda values for LUT
        unique_x = np.unique(x_int)
        unique_lam = np.unique(lam_vals)

        ppois_lut = gammaincc(
            np.clip(unique_x[:, None].astype(np.int64) + 1, 1, 10001),
            np.maximum(unique_lam[None, :], 1e-300)
        )  # (n_unique_x, n_unique_lam)

        # Frequency-weighted mean: for each cell j, left_tail[j] = mean(ppois_lut[x_idx[j], lam_idx[k]] for all k)
        # = sum over lam_groups: (count_lam / n_cells) * ppois_lut[x_idx[j], lam_group]
        # Vectorize: for each cell j, compute weighted sum over lambda groups
        lam_idx = np.searchsorted(unique_lam, lam_vals)
        lam_counts = np.bincount(lam_idx, minlength=len(unique_lam))
        lam_weights = lam_counts / n_cells  # (n_unique_lam,)

        # ppois_at_x[j, l] = ppois_lut[x_idx[j], l] for each unique lambda l
        x_idx = np.searchsorted(unique_x, x_int)
        ppois_at_x = ppois_lut[x_idx, :]  # (n_cells, n_unique_lam) — small

        left_tail = ppois_at_x @ lam_weights  # (n_cells,) = frequency-weighted mean
        left_tail = np.clip(left_tail, 1e-15, 1 - 1e-15)
        gene_cdf[i, :] = -np.log((1.0 - left_tail) / left_tail)

    # Step 2: Rank CDF values per column (ties.method="last")
    col_ranks = np.empty((n_genes, n_cells), dtype=np.int64)
    for j in range(n_cells):
        col = gene_cdf[:, j]
        order = np.lexsort((-np.arange(n_genes), col))
        ranks = np.empty(n_genes, dtype=np.int64)
        ranks[order] = np.arange(1, n_genes + 1)
        col_ranks[:, j] = ranks

    # Step 3: Convert ranks to statistics
    srs = np.abs(p / 2.0 - col_ranks.astype(np.float64))
    dos = p - col_ranks + 1

    # Step 4: Sparse random walk — evaluate only at gene-set dos positions
    pathway_scores = np.zeros((len(pathways), n_cells), dtype=np.float64)

    for pathway_idx, (pathway_name, gene_set) in enumerate(pathways.items()):
        indices = []
        for gene in gene_set:
            if gene in gene_to_idx:
                indices.append(gene_to_idx[gene])
        if len(indices) == 0:
            warnings.warn(f"Pathway {pathway_name} has no genes in expression matrix")
            continue

        k = len(indices)
        idx_arr = np.array(indices)
        n_out = p - k
        if n_out == 0:
            continue

        # Gene-set dos and srs: (k, n_cells)
        set_dos = dos[idx_arr, :]
        set_srs = srs[idx_arr, :]

        sum_in = set_srs.sum(axis=0)  # (n_cells,)
        valid = (sum_in > 0) & (k < p)
        if not np.any(valid):
            continue

        # Sort gene-set genes by dos position per cell
        sort_idx = np.argsort(set_dos, axis=0)
        sorted_dos = np.take_along_axis(set_dos, sort_idx, axis=0)
        sorted_srs = np.take_along_axis(set_srs, sort_idx, axis=0)

        # Cumulative srs: cum_srs[i] = sum of srs for genes 0..i
        cum_srs = np.cumsum(sorted_srs, axis=0)
        cum_count = np.arange(1, k + 1)[:, None]  # (k, 1)

        # Walk at gene-set positions:
        #   cum_in = cum_srs[i], cum_out = sorted_dos[i] - cum_count[i]
        #   walk = cum_in / sum_in - cum_out / n_out
        inv_sum_in = np.where(valid, 1.0 / np.maximum(sum_in, 1e-300), 0.0)
        inv_n_out = 1.0 / n_out
        walk_at = cum_srs * inv_sum_in[None, :] - (sorted_dos - cum_count) * inv_n_out

        # Walk at position before gene-set positions (captures local minima):
        #   cum_in = cum_srs[i-1] (or 0), cum_out = sorted_dos[i] - cum_count[i]
        cum_srs_prev = np.zeros_like(cum_srs)
        cum_srs_prev[1:, :] = cum_srs[:-1, :]
        walk_before = cum_srs_prev * inv_sum_in[None, :] - (sorted_dos - cum_count) * inv_n_out

        # Combine candidates: stack (2k, n_cells)
        walk_stack = np.vstack([walk_at, walk_before])

        # Also add boundary walk at dos=1 (if first gene-set gene is at dos>1):
        # walk(dos=1) = 0 - 1/n_out  (only if d_1 > 1)
        walk_start = np.full(n_cells, -inv_n_out)
        # And walk at dos=p: always 0
        walk_end = np.zeros(n_cells)

        walk_stack = np.vstack([walk_stack, walk_start[None, :], walk_end[None, :]])

        walk_max = walk_stack.max(axis=0)
        walk_min = walk_stack.min(axis=0)

        pathway_scores[pathway_idx, valid] = (walk_max[valid] + walk_min[valid])

    pathway_names = list(pathways.keys())
    return pd.DataFrame(pathway_scores, index=pathway_names, columns=pd.RangeIndex(n_cells))

def vision_score(
    expression: np.ndarray,
    pathways: Dict[str, List[str]],
    gene_names: List[str],
    ncores: int = 1,
) -> pd.DataFrame:
    """Calculate VISION scores.

    Implementation matching R scMetabolism + VISION package behavior:
    1. Library size normalization (divide by colSums, multiply by median)
    2. Z-norm columns (z-score normalize each cell/column) — R VISION default sig_norm_method="znorm_columns"
    3. Average z-score of genes in each pathway

    Parameters
    ----------
    expression : ndarray
        Genes x cells matrix.
    pathways : dict
        Pathway name -> list of gene symbols.
    gene_names : list
        List of gene symbols corresponding to rows of expression.
    ncores : int
        Not used yet.

    Returns
    -------
    DataFrame
        Pathway scores (pathways x cells).
    """
    n_genes, n_cells = expression.shape
    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}

    # Step 1: Library size normalization (as in R scMetabolism code)
    # n.umi <- colSums(countexp2)
    # scaled_counts <- t(t(countexp2) / n.umi) * median(n.umi)
    col_sums = np.sum(expression, axis=0)
    median_col_sum = np.median(col_sums)
    scale_factors = median_col_sum / col_sums
    scaled = expression * scale_factors[None, :]

    # Step 2: Log2 transform (R VISION matLog2: log2(x + 1) for non-zero values)
    logged = np.where(scaled > 0, np.log2(scaled + 1), 0.0)

    # Step 3: Z-norm columns (R VISION default: sig_norm_method="znorm_columns")
    # R code: colOffsets = -colMeans(data), colScaleFactors = colVars ** -0.5
    # colSds uses ddof=1 in R (sample standard deviation)
    col_means = np.mean(logged, axis=0)
    col_vars = np.var(logged, axis=0, ddof=1)
    col_sds = np.where(col_vars > 0, np.sqrt(col_vars), 1.0)
    z_normed = (logged - col_means[None, :]) / col_sds[None, :]

    # Step 4: For each pathway, compute average z-score of genes in set
    # R VISION: sigScores = C / rowSums(abs(sigSparseMatrix))
    # Since all sig weights = 1 (from GMT), denom = number of genes in set
    pathway_scores = np.zeros((len(pathways), n_cells), dtype=np.float64)

    for pathway_idx, (pathway_name, gene_set) in enumerate(pathways.items()):
        indices = []
        for gene in gene_set:
            if gene in gene_to_idx:
                indices.append(gene_to_idx[gene])
        if len(indices) == 0:
            warnings.warn(f"Pathway {pathway_name} has no genes in expression matrix")
            pathway_scores[pathway_idx, :] = 0
            continue

        pathway_scores[pathway_idx, :] = np.mean(z_normed[indices, :], axis=0)

    pathway_names = list(pathways.keys())
    return pd.DataFrame(pathway_scores, index=pathway_names, columns=pd.RangeIndex(n_cells))