"""Core computation functions for scMetabolism."""

import numpy as np
import pandas as pd
import logging
from typing import Union, Optional, Dict, List
import anndata
import pathlib
from . import methods

logger = logging.getLogger(__name__)

def read_gmt(filepath: Union[str, pathlib.Path]) -> Dict[str, List[str]]:
    """Read a GMT file into a dictionary of pathway name -> gene list.

    GMT format: each line: pathway_name\tdescription\tgene1\tgene2...
    """
    pathways = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            pathway = parts[0]
            # Some GMT files have description in second column, we skip it
            genes = [g for g in parts[2:] if g]
            pathways[pathway] = genes
    return pathways

def load_metabolism_gmt(metabolism_type: str = "KEGG") -> Dict[str, List[str]]:
    """Load the built-in GMT file for metabolism pathways.

    Parameters
    ----------
    metabolism_type : str
        Either "KEGG" or "REACTOME".

    Returns
    -------
    dict
        Pathway name -> list of gene symbols.
    """
    data_dir = pathlib.Path(__file__).parent / "data"
    if metabolism_type == "KEGG":
        gmt_file = data_dir / "KEGG_metabolism_nc.gmt"
    elif metabolism_type == "REACTOME":
        gmt_file = data_dir / "REACTOME_metabolism.gmt"
    else:
        raise ValueError('metabolism.type must be "KEGG" or "REACTOME"')

    if not gmt_file.exists():
        raise FileNotFoundError(f"GMT file not found: {gmt_file}")

    logger.info(f"Your choice is: {metabolism_type}")
    return read_gmt(gmt_file)

def alra_imputation(count_matrix: np.ndarray, **kwargs):
    """Perform ALRA imputation on count matrix.

    Implementation of ALRA (Zero-preserving imputation of scRNA-seq data using low-rank approximation).
    Based on the R code from KlugerLab.

    Parameters
    ----------
    count_matrix : ndarray
        Genes x cells matrix.

    Returns
    -------
    ndarray
        Imputed matrix (genes x cells).
    """
    import warnings
    # Transpose to cells x genes for ALRA (as in R implementation)
    A = count_matrix.T  # cells x genes

    # Step 1: Normalize data (library size normalization and log transform)
    def normalize_data(A):
        # Remove cells with zero total UMI
        totalUMIPerCell = np.sum(A, axis=1)
        zero_cells = np.where(totalUMIPerCell == 0)[0]
        if len(zero_cells) > 0:
            A = np.delete(A, zero_cells, axis=0)
            totalUMIPerCell = np.delete(totalUMIPerCell, zero_cells)
            warnings.warn(f"Removed {len(zero_cells)} cells which did not express any genes")

        # Normalize
        A_norm = A / totalUMIPerCell[:, np.newaxis] * 10000.0  # 10E3
        A_norm = np.log1p(A_norm)  # log(x + 1)
        return A_norm

    # Step 2: Choose k (rank) automatically
    def choose_k(A_norm, K=100, thresh=6, noise_start=80, q=2):
        # Heuristic for choosing rank k based on singular value spacings
        m, n = A_norm.shape
        if K > min(m, n):
            K = min(m, n) - 1
        if noise_start > K - 5:
            noise_start = K - 5

        # Compute randomized SVD
        # For simplicity, use full SVD (inefficient but works for testing)
        U, s, Vt = np.linalg.svd(A_norm, full_matrices=False)
        # Take first K singular values
        s = s[:K]

        # Compute differences between consecutive singular values
        diffs = s[:-1] - s[1:]

        # Noise singular values (tail)
        noise_svals = range(noise_start - 1, K - 1)  # -1 for 0-indexing
        if len(noise_svals) > 0:
            mu = np.mean(diffs[noise_svals])
            sigma = np.std(diffs[noise_svals])
        else:
            mu = np.mean(diffs[-10:])  # last 10
            sigma = np.std(diffs[-10:])

        num_of_sds = (diffs - mu) / sigma if sigma > 0 else 0
        k = np.max(np.where(num_of_sds > thresh)[0]) if np.any(num_of_sds > thresh) else 0
        return k + 1  # convert to 1-indexed rank

    # Step 3: Randomized SVD (simplified)
    def randomized_svd(A, k, q=2):
        # Simplified: use full SVD then truncate
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        return U[:, :k], s[:k], Vt[:k, :]

    # Step 4: ALRA main function
    def alra(A_norm, k=0, q=10, quantile_prob=0.001):
        m, n = A_norm.shape

        if k == 0:
            k = choose_k(A_norm)
            warnings.warn(f"Chose k={k}")

        # Original non-zero positions
        originally_nonzero = A_norm > 0

        # Randomized SVD
        U, s, Vt = randomized_svd(A_norm, k, q)
        # Rank-k approximation
        A_norm_rank_k = U @ np.diag(s) @ Vt

        # Adaptive thresholding: find quantile for each gene
        # In R: apply(A_norm_rank_k, 2, FUN=function(x) quantile(x, quantile.prob))
        quantiles = np.percentile(A_norm_rank_k, quantile_prob * 100, axis=0)
        # Take absolute value as in R code
        quantiles_abs = np.abs(quantiles)

        # Threshold: set values below quantile to 0
        # In R: replace(A_norm_rank_k, A_norm_rank_k <= A_norm_rank_k_mins[col(A_norm_rank_k)], 0)
        # where A_norm_rank_k_mins = abs(quantile(...))
        A_norm_rank_k_cor = A_norm_rank_k.copy()
        # For each column (gene), threshold values below absolute quantile
        for j in range(n):
            col = A_norm_rank_k_cor[:, j]
            col[col <= quantiles_abs[j]] = 0

        # Scale to match moments of original data
        def sd_nonzero(x):
            nonzero = x[x != 0]
            return np.std(nonzero) if len(nonzero) > 1 else 0

        sigma_1 = np.array([sd_nonzero(A_norm_rank_k_cor[:, j]) for j in range(n)])
        sigma_2 = np.array([sd_nonzero(A_norm[:, j]) for j in range(n)])

        mu_1 = np.array([np.sum(A_norm_rank_k_cor[:, j]) / np.sum(A_norm_rank_k_cor[:, j] != 0)
                         if np.sum(A_norm_rank_k_cor[:, j] != 0) > 0 else 0
                         for j in range(n)])
        mu_2 = np.array([np.sum(A_norm[:, j]) / np.sum(A_norm[:, j] != 0)
                         if np.sum(A_norm[:, j] != 0) > 0 else 0
                         for j in range(n)])

        # Columns to scale
        toscale = ~np.isnan(sigma_1) & ~np.isnan(sigma_2) & ~((sigma_1 == 0) & (sigma_2 == 0)) & ~(sigma_1 == 0)

        sigma_ratio = sigma_2 / sigma_1
        toadd = -mu_1 * sigma_ratio + mu_2

        # Apply scaling
        A_norm_rank_k_cor_sc = A_norm_rank_k_cor.copy()
        for j in np.where(toscale)[0]:
            if sigma_ratio[j] > 0:
                A_norm_rank_k_cor_sc[:, j] = A_norm_rank_k_cor[:, j] * sigma_ratio[j] + toadd[j]

        # Ensure zeros in A_norm_rank_k_cor remain zeros in scaled version
        # In R: A_norm_rank_k_cor_sc[A_norm_rank_k_cor==0] = 0
        A_norm_rank_k_cor_sc[A_norm_rank_k_cor == 0] = 0

        # Ensure non-negative
        A_norm_rank_k_cor_sc[A_norm_rank_k_cor_sc < 0] = 0

        # Restore original non-zero values that became zero
        # (where originally nonzero but imputed is zero)
        mask = originally_nonzero & (A_norm_rank_k_cor_sc == 0)
        A_norm_rank_k_cor_sc[mask] = A_norm[mask]

        return A_norm_rank_k_cor_sc

    # Apply ALRA
    A_norm = normalize_data(A)
    A_norm_imputed = alra(A_norm, k=0)

    # Reverse log transform and scaling
    # Note: R code works with log-normalized data, returns log-normalized imputed data
    # We need to reverse log1p and scaling
    # Actually, the imputation is done on log-normalized data, but the R function returns the imputed log-normalized matrix
    # The calling code expects counts, so we need to reverse transformation
    # However, R scMetabolism uses the imputed log-normalized matrix directly for scoring
    # So we'll return the imputed log-normalized matrix (genes x cells)
    # But note: A_norm_imputed is cells x genes, log-normalized

    # Reverse normalization: expm1 and multiply by cell total / 10000
    # For simplicity, we'll return the imputed log-normalized matrix transposed back to genes x cells
    # The scoring methods expect raw counts? In R, they use imputed counts (log-normalized)
    # Actually, sc.metabolism does: result.completed <- alra(as.matrix(countexp)); countexp2 <- result.completed[[3]]
    # result.completed[[3]] is the completed matrix (log-normalized)
    # Then they use countexp2 for scoring (which is log-normalized)

    # So we return imputed log-normalized matrix, transposed to genes x cells
    imputed_log_norm = A_norm_imputed.T  # genes x cells

    return imputed_log_norm

def vision_score(count_matrix: np.ndarray, pathways: Dict[str, List[str]], gene_names: List[str], ncores: int = 1):
    """Calculate pathway scores using VISION method."""
    return methods.vision_score(count_matrix, pathways, gene_names, ncores)

def aucell_score(count_matrix: np.ndarray, pathways: Dict[str, List[str]], gene_names: List[str], ncores: int = 1, aucMaxRank: int = None):
    """Calculate pathway scores using AUCell method."""
    return methods.aucell_score(count_matrix, pathways, gene_names, ncores, aucMaxRank=aucMaxRank)

def ssgsea_score(count_matrix: np.ndarray, pathways: Dict[str, List[str]], gene_names: List[str], ncores: int = 1, normalize: bool = False):
    """Calculate pathway scores using ssGSEA method."""
    return methods.ssgsea_score(count_matrix, pathways, gene_names, ncores, normalize=normalize)

def gsva_score(count_matrix: np.ndarray, pathways: Dict[str, List[str]], gene_names: List[str], ncores: int = 1):
    """Calculate pathway scores using GSVA method."""
    return methods.gsva_score(count_matrix, pathways, gene_names, ncores)

def sc_metabolism(
    countexp: Union[pd.DataFrame, np.ndarray, anndata.AnnData],
    method: str = "VISION",
    imputation: bool = False,
    ncores: int = 2,
    metabolism_type: str = "KEGG",
    aucMaxRank: int = None,
) -> pd.DataFrame:
    """Quantify metabolism activity from count matrix.

    Parameters
    ----------
    countexp : DataFrame, ndarray, or AnnData
        UMI count matrix (cells x genes) or AnnData object with counts in .X.
    method : str
        Scoring method: "VISION", "AUCell", "ssGSEA", "GSVA". Default "VISION".
    imputation : bool
        Whether to perform ALRA imputation before scoring.
    ncores : int
        Number of cores for parallel computation.
    metabolism_type : str
        Pathway database: "KEGG" or "REACTOME".
    aucMaxRank : int or None
        Max rank threshold for AUCell. Default: ceiling(0.05 * n_genes).

    Returns
    -------
    DataFrame
        Pathway scores (pathways x cells).
    """
    # Convert input to numpy array and extract gene names
    gene_names = None
    if isinstance(countexp, anndata.AnnData):
        count_matrix = countexp.X
        if hasattr(count_matrix, 'toarray'):
            count_matrix = count_matrix.toarray()
        gene_names = list(countexp.var_names)
        cell_names = list(countexp.obs_names)
    elif isinstance(countexp, pd.DataFrame):
        # DataFrame: assume cells x genes (rows=cells, cols=genes)
        count_matrix = countexp.values
        gene_names = list(countexp.columns)  # column names are genes
        cell_names = list(countexp.index)
    else:
        count_matrix = np.asarray(countexp)
        # No gene names available, create placeholder
        n_genes = count_matrix.shape[0] if count_matrix.shape[0] < count_matrix.shape[1] else count_matrix.shape[1]
        gene_names = [f"Gene{i+1}" for i in range(n_genes)]
        cell_names = None

    # R package expects genes as rows, cells as columns
    # For DataFrame/AnnData: we know orientation is cells x genes, always transpose
    # For ndarray: use heuristic (fewer rows than cols -> likely cells x genes)
    if isinstance(countexp, (pd.DataFrame, anndata.AnnData)):
        count_matrix = count_matrix.T
    elif count_matrix.shape[0] < count_matrix.shape[1]:
        count_matrix = count_matrix.T
    else:
        if gene_names is not None and len(gene_names) != count_matrix.shape[0]:
            gene_names = [f"Gene{i+1}" for i in range(count_matrix.shape[0])]

    logger.info(f"Input matrix: {count_matrix.shape[0]} genes, {count_matrix.shape[1]} cells")

    # Imputation
    if imputation:
        logger.info("Start imputation...")
        count_matrix = alra_imputation(count_matrix)

    # Load pathways
    pathways = load_metabolism_gmt(metabolism_type)

    logger.info("Start quantify the metabolism activity...")

    # Dispatch to method
    if method == "VISION":
        signature_exp = vision_score(count_matrix, pathways, gene_names, ncores=ncores)
    elif method == "AUCell":
        signature_exp = aucell_score(count_matrix, pathways, gene_names, ncores=ncores, aucMaxRank=aucMaxRank)
    elif method == "ssGSEA":
        signature_exp = ssgsea_score(count_matrix, pathways, gene_names, ncores=ncores)
    elif method == "GSVA":
        signature_exp = gsva_score(count_matrix, pathways, gene_names, ncores=ncores)
    else:
        raise ValueError(f'Unknown method: {method}')

    logger.info("\nPlease Cite: \nYingcheng Wu, Qiang Gao, et al. Cancer Discovery. 2021. \nhttps://pubmed.ncbi.nlm.nih.gov/34417225/   \n")

    # Assign cell names if available
    if cell_names is not None:
        signature_exp.columns = cell_names

    return signature_exp

def sc_metabolism_anndata(
    adata: anndata.AnnData,
    method: str = "VISION",
    imputation: bool = False,
    ncores: int = 2,
    metabolism_type: str = "KEGG",
    layer: Optional[str] = None,
    key_added: str = "metabolism",
) -> anndata.AnnData:
    """Quantify metabolism activity and store in AnnData.

    Stores result in adata.obsm[f'X_{key_added}'] and pathway names in adata.uns[f'{key_added}_pathways'].

    Parameters
    ----------
    adata : AnnData
        AnnData object with counts in .X or specified layer.
    method : str
        Scoring method.
    imputation : bool
        Whether to perform ALRA imputation.
    ncores : int
        Number of cores.
    metabolism_type : str
        Pathway database.
    layer : str, optional
        Layer to use instead of .X.
    key_added : str
        Key suffix for storing results.

    Returns
    -------
    AnnData
        Modified AnnData object.
    """
    if layer is not None:
        count_matrix = adata.layers[layer]
    else:
        count_matrix = adata.X

    # Convert to dense array if sparse
    if hasattr(count_matrix, 'toarray'):
        count_matrix = count_matrix.toarray()

    # Create DataFrame with gene names as columns (cells x genes)
    # This matches sc_metabolism's expectation for DataFrame input
    count_df = pd.DataFrame(
        count_matrix,
        index=adata.obs_names,
        columns=adata.var_names
    )

    signature_exp = sc_metabolism(
        count_df,
        method=method,
        imputation=imputation,
        ncores=ncores,
        metabolism_type=metabolism_type,
    )

    n_cells = len(adata.obs_names)
    if signature_exp.shape[1] != n_cells:
        if signature_exp.shape[1] == len(adata.var_names):
            logger.warning(
                "signature_exp has %d columns matching genes, not cells (%d). "
                "Transposing to pathways x cells.",
                signature_exp.shape[1], n_cells,
            )
            signature_exp = signature_exp.T
        else:
            raise ValueError(
                f"signature_exp shape mismatch: {signature_exp.shape} "
                f"(expected pathways x {n_cells} cells)"
            )

    signature_exp.columns = adata.obs_names

    adata.obsm[f"X_{key_added}"] = signature_exp.T.values
    adata.uns[f"{key_added}_pathways"] = list(signature_exp.index)

    return adata