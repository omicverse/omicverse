from typing import Optional
import gc
import sys
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import cm
import matplotlib.pyplot as plt
import plotly
from scipy import sparse
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import normalize
from tqdm import tqdm,trange

from .._optimal_transport import cot_sparse
from .._optimal_transport import cot_combine_sparse
from .._optimal_transport import uot
from .._optimal_transport import usot

def kernel_function(x, eta, nu, kernel, normalization=None):
    if kernel == "exp":
        phi = np.exp(-np.power(x/eta, nu))
    elif kernel == "lorentz":
        phi = 1 / ( 1 + np.power(x/eta, nu) )
    if normalization == "unit_row_sum":
        phi = (phi.T / np.sum(phi.T, axis=0)).T
    elif normalization == "unit_col_sum":
        phi = phi / np.sum(phi, axis=0)
    return phi

def coo_from_dense_submat(row, col, x, shape):
    col_ind, row_ind = np.meshgrid(col, row)
    sparse_mat = sparse.coo_matrix((x.reshape(-1), (row_ind.reshape(-1), col_ind.reshape(-1))), shape=shape)
    sparse_mat.eliminate_zeros()
    return(sparse_mat)

class CellCommunication(object):

    def __init__(self,
        adata,
        df_ligrec,
        dmat,
        dis_thr,
        cost_scale,
        cost_type,
        heteromeric = None,
        heteromeric_rule = 'min',
        heteromeric_delimiter = '_'
    ):
        data_genes = set(adata.var_names)
        if not heteromeric:
            self.ligs = list(set(df_ligrec.iloc[:,0]).intersection(data_genes))
            self.recs = list(set(df_ligrec.iloc[:,1]).intersection(data_genes))
            A = np.inf * np.ones([len(self.ligs), len(self.recs)], float)
            for i in tqdm(range(len(df_ligrec))):
                tmp_lig = df_ligrec.iloc[i][0]
                tmp_rec = df_ligrec.iloc[i][1]
                if tmp_lig in self.ligs and tmp_rec in self.recs:
                    if cost_scale is None:
                        A[self.ligs.index(tmp_lig), self.recs.index(tmp_rec)] = 1.0
                    else:
                        A[self.ligs.index(tmp_lig), self.recs.index(tmp_rec)] = cost_scale[(tmp_lig, tmp_rec)]
            self.A = A
            self.S = adata[:,self.ligs].X.toarray()
            self.D = adata[:,self.recs].X.toarray()

        elif heteromeric:
            tmp_ligs = list(set(df_ligrec.iloc[:,0]))
            tmp_recs = list(set(df_ligrec.iloc[:,1]))
            avail_ligs = []
            avail_recs = []
            for tmp_lig in tmp_ligs:
                lig_genes = set(tmp_lig.split(heteromeric_delimiter))
                if lig_genes.issubset(data_genes):
                    avail_ligs.append(tmp_lig)
            for tmp_rec in tmp_recs:
                rec_genes = set(tmp_rec.split(heteromeric_delimiter))
                if rec_genes.issubset(data_genes):
                    avail_recs.append(tmp_rec)
            self.ligs = avail_ligs
            self.recs = avail_recs
            A = np.inf * np.ones([len(self.ligs), len(self.recs)], float)
            for i in tqdm(range(len(df_ligrec))):
                tmp_lig = df_ligrec.iloc[i,0]
                tmp_rec = df_ligrec.iloc[i,1]
                if tmp_lig in self.ligs and tmp_rec in self.recs:
                    if cost_scale is None:
                        A[self.ligs.index(tmp_lig), self.recs.index(tmp_rec)] = 1.0
                    else:
                        A[self.ligs.index(tmp_lig), self.recs.index(tmp_rec)] = cost_scale[(tmp_lig, tmp_rec)]
            self.A = A
            ncell = adata.shape[0]
            S = np.zeros([ncell, A.shape[0]], float)
            D = np.zeros([ncell, A.shape[1]], float)
            for i in tqdm(range(len(self.ligs))):
                tmp_lig = self.ligs[i]
                lig_genes = tmp_lig.split(heteromeric_delimiter)
                if heteromeric_rule == 'min':
                    S[:,i] = adata[:, lig_genes].X.toarray().min(axis=1)[:]
                elif heteromeric_rule == 'ave':
                    S[:,i] = adata[:, lig_genes].X.toarray().mean(axis=1)[:]
            for i in tqdm(range(len(self.recs))):
                tmp_rec = self.recs[i]
                rec_genes = tmp_rec.split(heteromeric_delimiter)
                if heteromeric_rule == 'min':
                    D[:,i] = adata[:, rec_genes].X.toarray().min(axis=1)[:]
                elif heteromeric_rule == 'ave':
                    D[:,i] = adata[:, rec_genes].X.toarray().mean(axis=1)[:]
            self.S = S
            self.D = D
        
        if cost_type == 'euc':
            self.M = dmat
        elif cost_type == 'euc_square':
            self.M = dmat ** 2
        if np.isscalar(dis_thr):
            if cost_type == 'euc_square':
                dis_thr = dis_thr ** 2
            self.cutoff = float(dis_thr) * np.ones_like(A)
        elif type(dis_thr) is dict:
            self.cutoff = np.zeros_like(A)
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if A[i,j] > 0:
                        if cost_type == 'euc_square':
                            self.cutoff[i,j] = dis_thr[(self.ligs[i], self.recs[j])] ** 2
                        else:
                            self.cutoff[i,j] = dis_thr[(self.ligs[i], self.recs[j])]
        self.nlig = self.S.shape[1]; self.nrec = self.D.shape[1]
        self.npts = adata.shape[0]

    def run_cot_signaling(self,
        cot_eps_p=1e-1, 
        cot_eps_mu=None, 
        cot_eps_nu=None, 
        cot_rho=1e1, 
        cot_nitermax=1e4, 
        cot_weights=(0.25,0.25,0.25,0.25), 
        smooth=False, 
        smth_eta=None, 
        smth_nu=None, 
        smth_kernel=None
    ):
        if not smooth:
            self.comm_network = cot_combine_sparse(self.S, self.D, self.A, self.M, self.cutoff, \
                eps_p=cot_eps_p, eps_mu=cot_eps_mu, eps_nu=cot_eps_nu, rho=cot_rho, weights=cot_weights, nitermax=cot_nitermax)
        if smooth:
            # Smooth the expression
            S_smth = np.zeros_like(self.S)
            D_smth = np.zeros_like(self.D)
            for i in tqdm(range(self.nlig)):
                nzind = np.where(self.S[:,i] > 0)[0]
                phi = kernel_function(self.M[nzind,:][:,nzind], smth_eta, smth_nu, smth_kernel, normalization='unit_col_sum')
                S_smth[nzind,i] = np.matmul( phi, self.S[nzind,i].reshape(-1,1) )[:,0]
            for i in tqdm(range(self.nrec)):
                nzind = np.where(self.D[:,i] > 0)[0]
                phi = kernel_function(self.M[nzind,:][:,nzind], smth_eta, smth_nu, smth_kernel, normalization='unit_col_sum')
                D_smth[nzind,i] = np.matmul( phi, self.D[nzind,i].reshape(-1,1) )[:,0]
            # Get the transport plans for the smoothed distributions
            P_smth = cot_combine_sparse(S_smth, D_smth, self.A, self.M, self.cutoff, \
                eps_p=cot_eps_p, eps_mu=cot_eps_mu, eps_nu=cot_eps_nu, rho=cot_rho, weights=cot_weights, nitermax=cot_nitermax)
            # Deconvolute back to original cells
            self.comm_network = {}
            for i in tqdm(range(self.nlig)):
                S = self.S[:,i]
                nzind_S = np.where(S > 0)[0]
                phi_S = kernel_function(self.M[nzind_S,:][:,nzind_S], smth_eta, smth_nu, smth_kernel, normalization='unit_col_sum')
                S_contrib = phi_S * S[nzind_S]; S_contrib = S_contrib / np.sum(S_contrib, axis=1, keepdims=True)
                for j in range(self.nrec):
                    D = self.D[:,j]
                    nzind_D = np.where(D > 0)[0]
                    if np.isinf(self.A[i,j]): continue
                    P_dense = P_smth[(i,j)].toarray()
                    P_sub = P_dense[nzind_S,:][:,nzind_D]
                    phi_D = kernel_function(self.M[nzind_D,:][:,nzind_D], smth_eta, smth_nu, smth_kernel, normalization='unit_col_sum')
                    D_contrib = phi_D * D[nzind_D]; D_contrib = D_contrib / np.sum(D_contrib, axis=1, keepdims=True)
                    P_deconv = np.matmul(S_contrib.T, np.matmul(P_sub, D_contrib))
                    for k in range(len(nzind_D)):
                        P_dense[nzind_S, nzind_D[k]] = P_deconv[:,k]
                    P_deconv_sparse = coo_from_dense_submat(nzind_S, nzind_D, P_deconv, shape=(self.npts, self.npts))
                    self.comm_network[(i,j)] = P_deconv_sparse

    def smooth(self, eta, nu, kernel):
        S_smth = np.zeros_like(self.S)
        D_smth = np.zeros_like(self.D)
        for i in range(self.nlig):
            nzind = np.where(self.S[:,i] > 0)[0]
            phi = kernel_function(self.M[nzind,:][:,nzind], eta, nu, kernel, normalization='unit_col_sum')
            S_smth[nzind,i] = np.matmul( phi, self.S[nzind,i].reshape(-1,1) )[:,0]
        for i in range(self.nrec):
            nzind = np.where(self.D[:,i] > 0)[0]
            phi = kernel_function(self.M[nzind,:][:,nzind], eta, nu, kernel, normalization='unit_col_sum')
            D_smth[nzind,i] = np.matmul( phi, self.D[nzind,i].reshape(-1,1) )[:,0]
        self.S_smth = S_smth
        self.D_smth = D_smth
        self.kernel_eta = eta
        self.kernel_nu = nu
        self.kernel = kernel

def assign_distance(adata, dmat=None):
    if dmat is None:
        adata.obsp["spatial_distance"] = distance_matrix(adata.obsm["spatial"], adata.obsm["spatial"])
    else:
        adata.obsp["spatial_distance"] = dmat

def summarize_cluster(X, clusterid, clusternames, n_permutations=500):
    # Input a sparse matrix of cell signaling and output a pandas dataframe
    # for cluster-cluster signaling
    n = len(clusternames)
    X_cluster = np.empty([n,n], float)
    p_cluster = np.zeros([n,n], float)
    for i in range(n):
        tmp_idx_i = np.where(clusterid==clusternames[i])[0]
        for j in range(n):
            tmp_idx_j = np.where(clusterid==clusternames[j])[0]
            X_cluster[i,j] = X[tmp_idx_i,:][:,tmp_idx_j].mean()
    for i in range(n_permutations):
        clusterid_perm = np.random.permutation(clusterid)
        X_cluster_perm = np.empty([n,n], float)
        for j in range(n):
            tmp_idx_j = np.where(clusterid_perm==clusternames[j])[0]
            for k in range(n):
                tmp_idx_k = np.where(clusterid_perm==clusternames[k])[0]
                X_cluster_perm[j,k] = X[tmp_idx_j,:][:,tmp_idx_k].mean()
        p_cluster[X_cluster_perm >= X_cluster] += 1.0
    p_cluster = p_cluster / n_permutations
    df_cluster = pd.DataFrame(data=X_cluster, index=clusternames, columns=clusternames)
    df_p_value = pd.DataFrame(data=p_cluster, index=clusternames, columns=clusternames)
    return df_cluster, df_p_value

def summarize_cluster_optimized(X, clusterid, clusternames, n_permutations=500):
    """
    Optimized version of summarize_cluster with significantly improved performance.
    
    Key optimizations:
    1. Pre-compute cluster indices to avoid repeated np.where calls
    2. Vectorize matrix operations using advanced indexing
    3. Optimize permutation computations using broadcasting
    4. Handle sparse matrices efficiently
    5. Reduce memory allocations in permutation loop
    
    Parameters
    ----------
    X : array-like or sparse matrix
        Cell signaling matrix of shape (n_cells, n_cells)
    clusterid : array-like
        Cluster assignments for each cell
    clusternames : list
        List of unique cluster names
    n_permutations : int, default 500
        Number of permutations for p-value calculation
    
    Returns
    -------
    df_cluster : pd.DataFrame
        Cluster-cluster signaling values
    df_p_value : pd.DataFrame
        P-values from permutation test
    """
    n = len(clusternames)
    
    # Pre-compute cluster indices (major optimization)
    cluster_indices = {}
    for i, name in enumerate(clusternames):
        cluster_indices[name] = np.where(clusterid == name)[0]
    
    # Convert to array if sparse for faster indexing in this context
    if sparse.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = X
    
    # Vectorized computation of cluster-cluster signaling
    X_cluster = np.zeros((n, n), dtype=float)
    
    for i, name_i in enumerate(clusternames):
        idx_i = cluster_indices[name_i]
        if len(idx_i) == 0:
            continue
            
        # Extract submatrix for sender cluster i (vectorized)
        X_i = X_dense[idx_i, :]
        
        for j, name_j in enumerate(clusternames):
            idx_j = cluster_indices[name_j] 
            if len(idx_j) == 0:
                continue
                
            # Vectorized mean computation
            X_cluster[i, j] = X_i[:, idx_j].mean()
    
    # Optimized permutation test
    p_cluster = np.zeros((n, n), dtype=float)
    
    # Pre-allocate arrays for permutation
    cluster_sizes = np.array([len(cluster_indices[name]) for name in clusternames])
    total_cells = len(clusterid)
    
    for perm in range(n_permutations):
        # Generate permuted cluster assignments
        perm_indices = np.random.permutation(total_cells)
        
        # Create permuted cluster indices more efficiently
        start_idx = 0
        perm_cluster_indices = {}
        for i, name in enumerate(clusternames):
            size = cluster_sizes[i]
            if size > 0:
                perm_cluster_indices[name] = perm_indices[start_idx:start_idx + size]
                start_idx += size
            else:
                perm_cluster_indices[name] = np.array([], dtype=int)
        
        # Compute permuted cluster-cluster matrix (vectorized)
        X_cluster_perm = np.zeros((n, n), dtype=float)
        
        for i, name_i in enumerate(clusternames):
            perm_idx_i = perm_cluster_indices[name_i]
            if len(perm_idx_i) == 0:
                continue
                
            X_perm_i = X_dense[perm_idx_i, :]
            
            for j, name_j in enumerate(clusternames):
                perm_idx_j = perm_cluster_indices[name_j]
                if len(perm_idx_j) == 0:
                    continue
                    
                X_cluster_perm[i, j] = X_perm_i[:, perm_idx_j].mean()
        
        # Update p-values (vectorized comparison)
        p_cluster += (X_cluster_perm >= X_cluster).astype(float)
    
    # Normalize p-values
    p_cluster = p_cluster / n_permutations
    
    # Create output DataFrames
    df_cluster = pd.DataFrame(data=X_cluster, index=clusternames, columns=clusternames)
    df_p_value = pd.DataFrame(data=p_cluster, index=clusternames, columns=clusternames)
    
    return df_cluster, df_p_value

def _apply_scaling(X_cluster, cluster_indices, clusternames, scale_factor):
    """
    Apply scaling to cluster communication matrix for better visualization.
    
    Parameters
    ----------
    X_cluster : np.ndarray
        Original cluster communication matrix
    cluster_indices : dict
        Mapping of cluster names to cell indices
    clusternames : list
        List of cluster names
    scale_factor : str or float
        Scaling method or factor
        
    Returns
    -------
    np.ndarray
        Scaled cluster communication matrix
    """
    if scale_factor is None or scale_factor == 1:
        return X_cluster
    
    if scale_factor == 'auto':
        # Auto-scale based on median cluster size to improve visualization
        cluster_sizes = np.array([len(cluster_indices[name]) for name in clusternames])
        median_size = np.median(cluster_sizes[cluster_sizes > 0])
        # Scale factor to bring values to a more reasonable range for visualization
        scale_factor = max(10.0, median_size)  # At least 10x scaling
    elif scale_factor == 'sum':
        # Convert from mean to sum by multiplying by cluster sizes
        X_scaled = X_cluster.copy()
        for i, name_i in enumerate(clusternames):
            for j, name_j in enumerate(clusternames):
                size_i = len(cluster_indices[name_i])
                size_j = len(cluster_indices[name_j])
                if size_i > 0 and size_j > 0:
                    X_scaled[i, j] *= size_i * size_j
        return X_scaled
    
    # Apply numeric scaling factor
    if isinstance(scale_factor, (int, float)) and scale_factor > 0:
        return X_cluster * scale_factor
    else:
        print(f"Warning: Invalid scale_factor '{scale_factor}', using original values")
        return X_cluster

def summarize_cluster_gpu(X, clusterid, clusternames, n_permutations=500, use_gpu=True, scale_factor='auto'):
    """
    GPU-accelerated version of summarize_cluster using CuPy for CUDA computation.
    
    This function automatically falls back to CPU computation if GPU is not available.
    
    Parameters
    ----------
    X : array-like or sparse matrix
        Cell signaling matrix of shape (n_cells, n_cells)
    clusterid : array-like
        Cluster assignments for each cell
    clusternames : list
        List of unique cluster names
    n_permutations : int, default 500
        Number of permutations for p-value calculation
    use_gpu : bool, default True
        Whether to attempt GPU computation
    scale_factor : str or float, default 'auto'
        Scaling factor for visualization compatibility:
        - 'auto': automatically scale based on cluster sizes
        - 'sum': use sum instead of mean (multiply by cluster sizes)
        - float: multiply results by this factor
        - None or 1: no scaling (original behavior)
    
    Returns
    -------
    df_cluster : pd.DataFrame
        Cluster-cluster signaling values (potentially scaled)
    df_p_value : pd.DataFrame
        P-values from permutation test
    """
    # Try to import CuPy for GPU computation
    try:
        if use_gpu:
            import cupy as cp
            gpu_available = True
            print("🚀 GPU acceleration enabled with CuPy")
        else:
            raise ImportError("GPU disabled by user")
    except ImportError:
        print("⚠️  CuPy not available, falling back to CPU computation")
        df_cluster, df_p_value = summarize_cluster_optimized(
            X, clusterid, clusternames, n_permutations
        )
        cluster_indices = {
            name: np.flatnonzero(np.asarray(clusterid) == name)
            for name in clusternames
        }
        scaled = _apply_scaling(
            df_cluster.to_numpy(), cluster_indices, clusternames, scale_factor
        )
        return pd.DataFrame(scaled, index=df_cluster.index, columns=df_cluster.columns), df_p_value
    
    n = len(clusternames)
    
    # Pre-compute cluster indices and transfer to GPU
    cluster_indices_cpu = {}
    cluster_indices_gpu = {}
    for i, name in enumerate(clusternames):
        indices = np.where(clusterid == name)[0]
        cluster_indices_cpu[name] = indices
        if len(indices) > 0:
            cluster_indices_gpu[name] = cp.asarray(indices)
        else:
            cluster_indices_gpu[name] = cp.array([], dtype=cp.int32)
    
    # Convert data to GPU arrays
    if sparse.issparse(X):
        # For sparse matrices, convert to dense on GPU
        X_gpu = cp.asarray(X.toarray())
    else:
        X_gpu = cp.asarray(X)
    
    # Compute cluster-cluster signaling on GPU
    X_cluster = cp.zeros((n, n), dtype=cp.float32)
    
    for i, name_i in enumerate(clusternames):
        idx_i_gpu = cluster_indices_gpu[name_i]
        if len(idx_i_gpu) == 0:
            # Leave zeros for empty clusters - this is mathematically correct
            continue
        
        X_i = X_gpu[idx_i_gpu, :]
        
        for j, name_j in enumerate(clusternames):
            idx_j_gpu = cluster_indices_gpu[name_j]
            if len(idx_j_gpu) == 0:
                # Leave zeros for empty clusters - this is mathematically correct
                continue
            
            # GPU-accelerated mean computation
            X_cluster[i, j] = cp.mean(X_i[:, idx_j_gpu])
    
    # GPU-accelerated permutation test
    p_cluster = cp.zeros((n, n), dtype=cp.float32)
    total_cells = len(clusterid)
    
    for perm in range(n_permutations):
        # Generate permuted cluster labels (correct approach matching original implementation)
        clusterid_perm = np.random.permutation(clusterid)
        
        # Pre-compute permuted cluster indices on CPU, then transfer to GPU
        perm_cluster_indices = {}
        for name in clusternames:
            perm_indices = np.where(clusterid_perm == name)[0]
            if len(perm_indices) > 0:
                perm_cluster_indices[name] = cp.asarray(perm_indices)
            else:
                perm_cluster_indices[name] = cp.array([], dtype=cp.int32)
        
        # Compute permuted cluster-cluster matrix on GPU
        X_cluster_perm = cp.zeros((n, n), dtype=cp.float32)
        
        for i, name_i in enumerate(clusternames):
            perm_idx_i = perm_cluster_indices[name_i]
            if len(perm_idx_i) == 0:
                continue
            
            X_perm_i = X_gpu[perm_idx_i, :]
            
            for j, name_j in enumerate(clusternames):
                perm_idx_j = perm_cluster_indices[name_j]
                if len(perm_idx_j) == 0:
                    continue
                
                X_cluster_perm[i, j] = cp.mean(X_perm_i[:, perm_idx_j])
        
        # Update p-values (vectorized comparison on GPU)
        p_cluster += (X_cluster_perm >= X_cluster).astype(cp.float32)
    
    # Normalize p-values
    p_cluster = p_cluster / n_permutations
    
    # Transfer results back to CPU
    X_cluster_cpu = cp.asnumpy(X_cluster)
    p_cluster_cpu = cp.asnumpy(p_cluster)
    
    # Apply scaling for visualization compatibility
    X_cluster_scaled = _apply_scaling(X_cluster_cpu, cluster_indices_cpu, clusternames, scale_factor)
    
    # Create output DataFrames
    df_cluster = pd.DataFrame(data=X_cluster_scaled, index=clusternames, columns=clusternames)
    df_p_value = pd.DataFrame(data=p_cluster_cpu, index=clusternames, columns=clusternames)
    
    # Clear GPU memory
    del X_gpu, X_cluster, p_cluster
    for name in clusternames:
        del cluster_indices_gpu[name]
    cp.get_default_memory_pool().free_all_blocks()
    
    return df_cluster, df_p_value

def summarize_cluster_gpu_optimized(X, clusterid, clusternames, n_permutations=500, batch_size=100, use_gpu=True):
    """
    Advanced GPU-optimized version with batch processing and memory management.
    
    This version processes permutations in batches on GPU for better memory efficiency
    and includes advanced GPU optimization techniques.
    
    Parameters
    ----------
    X : array-like or sparse matrix
        Cell signaling matrix of shape (n_cells, n_cells)
    clusterid : array-like
        Cluster assignments for each cell
    clusternames : list
        List of unique cluster names
    n_permutations : int, default 500
        Number of permutations for p-value calculation
    batch_size : int, default 100
        Number of permutations to process in each GPU batch
    use_gpu : bool, default True
        Whether to attempt GPU computation
    
    Returns
    -------
    df_cluster : pd.DataFrame
        Cluster-cluster signaling values
    df_p_value : pd.DataFrame
        P-values from permutation test
    """
    # Try to import CuPy for GPU computation
    try:
        if use_gpu:
            import cupy as cp
            import cupyx.scipy.sparse as cp_sparse
            gpu_available = True
            print(f"🚀 Advanced GPU acceleration enabled (batch_size={batch_size})")
        else:
            raise ImportError("GPU disabled by user")
    except ImportError:
        print("⚠️  CuPy not available, falling back to optimized CPU computation")
        return summarize_cluster_optimized(X, clusterid, clusternames, n_permutations)
    
    n = len(clusternames)
    
    # Pre-compute cluster indices and sizes
    cluster_indices = {}
    cluster_sizes = []
    for i, name in enumerate(clusternames):
        indices = np.where(clusterid == name)[0]
        cluster_indices[name] = indices
        cluster_sizes.append(len(indices))
    
    cluster_sizes = cp.asarray(cluster_sizes)
    total_cells = len(clusterid)
    
    # Handle sparse vs dense matrices efficiently
    if sparse.issparse(X):
        # Keep sparse on GPU if possible
        try:
            X_gpu = cp_sparse.csr_matrix(X)
            use_sparse_gpu = True
        except:
            # Fall back to dense if sparse GPU operations fail
            X_gpu = cp.asarray(X.toarray())
            use_sparse_gpu = False
    else:
        X_gpu = cp.asarray(X)
        use_sparse_gpu = False
    
    # Compute original cluster-cluster signaling on GPU
    X_cluster = cp.zeros((n, n), dtype=cp.float32)
    
    # Pre-transfer all cluster indices to GPU
    cluster_indices_gpu = {}
    for name in clusternames:
        if len(cluster_indices[name]) > 0:
            cluster_indices_gpu[name] = cp.asarray(cluster_indices[name])
        else:
            cluster_indices_gpu[name] = cp.array([], dtype=cp.int32)
    
    # Optimized cluster-cluster computation
    for i, name_i in enumerate(clusternames):
        idx_i_gpu = cluster_indices_gpu[name_i]
        if len(idx_i_gpu) == 0:
            continue
        
        if use_sparse_gpu:
            X_i = X_gpu[idx_i_gpu, :]
        else:
            X_i = X_gpu[idx_i_gpu, :]
        
        for j, name_j in enumerate(clusternames):
            idx_j_gpu = cluster_indices_gpu[name_j]
            if len(idx_j_gpu) == 0:
                continue
            
            if use_sparse_gpu:
                submatrix = X_i[:, idx_j_gpu]
                X_cluster[i, j] = submatrix.mean()
            else:
                X_cluster[i, j] = cp.mean(X_i[:, idx_j_gpu])
    
    # Batch-processed permutation test on GPU
    p_cluster = cp.zeros((n, n), dtype=cp.float32)
    n_batches = (n_permutations + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_perm = batch_idx * batch_size
        end_perm = min((batch_idx + 1) * batch_size, n_permutations)
        current_batch_size = end_perm - start_perm
        
        # Generate batch of permutations on CPU, then transfer
        batch_permutations = np.array([
            np.random.permutation(total_cells) for _ in range(current_batch_size)
        ])
        batch_permutations_gpu = cp.asarray(batch_permutations)
        
        # Process each permutation in the current batch
        for perm_idx in range(current_batch_size):
            perm_indices = batch_permutations_gpu[perm_idx]
            
            # Create permuted cluster indices efficiently on GPU
            start_idx = 0
            perm_cluster_indices_gpu = {}
            for i, name in enumerate(clusternames):
                size = int(cluster_sizes[i])
                if size > 0:
                    perm_cluster_indices_gpu[name] = perm_indices[start_idx:start_idx + size]
                    start_idx += size
                else:
                    perm_cluster_indices_gpu[name] = cp.array([], dtype=cp.int32)
            
            # Compute permuted cluster-cluster matrix
            X_cluster_perm = cp.zeros((n, n), dtype=cp.float32)
            
            for i, name_i in enumerate(clusternames):
                perm_idx_i = perm_cluster_indices_gpu[name_i]
                if len(perm_idx_i) == 0:
                    continue
                
                if use_sparse_gpu:
                    X_perm_i = X_gpu[perm_idx_i, :]
                else:
                    X_perm_i = X_gpu[perm_idx_i, :]
                
                for j, name_j in enumerate(clusternames):
                    perm_idx_j = perm_cluster_indices_gpu[name_j]
                    if len(perm_idx_j) == 0:
                        continue
                    
                    if use_sparse_gpu:
                        submatrix = X_perm_i[:, perm_idx_j]
                        X_cluster_perm[i, j] = submatrix.mean()
                    else:
                        X_cluster_perm[i, j] = cp.mean(X_perm_i[:, perm_idx_j])
            
            # Update p-values (vectorized comparison)
            p_cluster += (X_cluster_perm >= X_cluster).astype(cp.float32)
        
        # Clear batch memory
        del batch_permutations_gpu
    
    # Normalize p-values
    p_cluster = p_cluster / n_permutations
    
    # Transfer final results back to CPU
    X_cluster_cpu = cp.asnumpy(X_cluster)
    p_cluster_cpu = cp.asnumpy(p_cluster)
    
    # Create output DataFrames
    df_cluster = pd.DataFrame(data=X_cluster_cpu, index=clusternames, columns=clusternames)
    df_p_value = pd.DataFrame(data=p_cluster_cpu, index=clusternames, columns=clusternames)
    
    # Comprehensive GPU memory cleanup
    del X_gpu, X_cluster, p_cluster
    for name in clusternames:
        del cluster_indices_gpu[name]
    cp.get_default_memory_pool().free_all_blocks()
    
    return df_cluster, df_p_value

def check_gpu_availability():
    """
    Check if GPU computation is available and return system information.
    
    Returns
    -------
    dict
        Dictionary containing GPU availability and system information
    """
    gpu_info = {
        'gpu_available': False,
        'cupy_available': False,
        'gpu_name': None,
        'gpu_memory': None,
        'cuda_version': None
    }
    
    try:
        import cupy as cp
        gpu_info['cupy_available'] = True
        
        # Check if GPU is actually available
        try:
            # Simple test operation
            test_array = cp.array([1, 2, 3])
            test_result = cp.mean(test_array)
            gpu_info['gpu_available'] = True
            
            # Get GPU information
            gpu_info['gpu_name'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            gpu_info['gpu_memory'] = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] // (1024**3)  # GB
            gpu_info['cuda_version'] = cp.cuda.runtime.runtimeGetVersion()
            
            # Clean up
            del test_array, test_result
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            gpu_info['error'] = str(e)
            
    except ImportError:
        pass
    
    return gpu_info

def cluster_center(adata, clustering, method="geometric_mean"):
    X = adata.obsm['spatial']
    cluster_pos = {}
    clusternames = list( adata.obs[clustering].unique() )
    clusternames.sort()
    clusterid = np.array( adata.obs[clustering], str )
    for name in clusternames:
        tmp_idx = np.where(clusterid==name)[0]
        tmp_X = X[tmp_idx]
        if method == "geometric_mean":
            X_mean = np.mean(tmp_X, axis=0)
        elif method == "representative_point":
            tmp_D = distance_matrix(tmp_X, tmp_X)
            tmp_D = tmp_D ** 2
            X_mean = tmp_X[np.argmin(tmp_D.sum(axis=1)),:]
        cluster_pos[name] = X_mean
    return cluster_pos

def spatial_communication(
    adata: anndata.AnnData, 
    database_name: str = None, 
    df_ligrec: pd.DataFrame = None,
    pathway_sum: bool = False,
    heteromeric: bool = False,
    heteromeric_rule: str = 'min',
    heteromeric_delimiter: str = '_', 
    dis_thr: Optional[float] = None, 
    cost_scale: Optional[dict] = None, 
    cost_type: str = 'euc',
    cot_eps_p: float = 1e-1, 
    cot_eps_mu: Optional[float] = None, 
    cot_eps_nu: Optional[float] = None, 
    cot_rho: float =1e1, 
    cot_nitermax: int = 10000, 
    cot_weights: tuple = (0.25,0.25,0.25,0.25), 
    smooth: bool = False, 
    smth_eta: float = None, 
    smth_nu: float = None, 
    smth_kernel: str = 'exp',
    copy: bool = False  
):
    """
    Infer spatial communication.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or spots and columns to genes.
        If the spatial distance is absent in ``.obsp['spatial_distance']``, Euclidean distance determined from ``.obsm['spatial']`` will be used.
    database_name
        Name of the ligand-receptor interaction database. Will be included in the keywords for anndata slots.
    df_ligrec
        A data frame where each row corresponds to a ligand-receptor pair with ligands, receptors, and the associated signaling pathways in the three columns, respectively.
    pathway_sum
        Whether to sum over all ligand-receptor pairs of each pathway.
    heteromeric
        Whether the ligands or receptors are made of heteromeric complexes.
    heteromeric_rule
        Use either 'min' (minimum) or 'ave' (average) expression of the components as the level for the heteromeric complex.
    heteromeric_delimiter
        The character in ligand and receptor names separating individual components.
    dis_thr
        The threshold of spatial distance of signaling.
    cost_scale
        Weight coefficients of the cost matrix for each ligand-receptor pair, e.g. cost_scale[('ligA','recA')] specifies weight for the pair ligA and recA.
        If None, all pairs have the same weight. 
    cost_type
        If 'euc', the original Euclidean distance will be used as cost matrix. If 'euc_square', the square of the Euclidean distance will be used.
    cot_eps_p
        The coefficient of entropy regularization for transport plan.
    cot_eps_mu
        The coefficient of entropy regularization for untransported source (ligand). Set to equal to cot_eps_p for fast algorithm.
    cot_eps_nu
        The coefficient of entropy regularization for unfulfilled target (receptor). Set to equal to cot_eps_p for fast algorithm.
    cot_rho
        The coefficient of penalty for unmatched mass.
    cot_nitermax
        Maximum iteration for collective optimal transport algorithm.
    cot_weights
        A tuple of four weights that add up to one. The weights corresponds to four setups of collective optimal transport: 
        1) all ligands-all receptors, 2) each ligand-all receptors, 3) all ligands-each receptor, 4) each ligand-each receptor.
    smooth
        Whether to (spatially) smooth the gene expression for identifying more global signaling trend.
    smth_eta
        Kernel bandwidth for smoothing
    smth_nu
        Kernel sharpness for smoothing
    smth_kernel
        'exp' exponential kernel. 'lorentz' Lorentz kernel.
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.

    Returns
    -------
    adata : anndata.AnnData
        Signaling matrices are added to ``.obsp``, e.g., for a LR interaction database named "databaseX", 
        ``.obsp['commot-databaseX-ligA-recA']``
        is a ``n_obs`` × ``n_obs`` matrix with the *ij* th entry being the "score" of 
        cell *i* sending signal to cell *j* through ligA and recA. If ``pathway_sum==True``, the cell-by-cell signaling matrix for a pathway "pathwayX" 
        will be stored in ``.obsp['commot-databaseX-pathwayX']``.
        The marginal sums (sender and receiver) of the signaling matrices are stored in ``.obsm['commot-databaseX-sum-sender']`` and ``.obsm['commot-databaseX-sum-receiver']``.
        Metadata of the analysis is added to ``.uns['commot-databaseX-info']``.
        If copy=True, return the AnnData object and return None otherwise.
        
    Examples
    --------
    >>> import anndata
    >>> import commot as ct
    >>> import pandas as pd
    >>> import numpy as np
    >>> # create a toy data with four cells and four genes
    >>> X = np.array([[1,0,0,0],[2,0,0,0],[0,1,1,4],[0,3,4,1]], float)
    >>> adata = anndata.AnnData(X=X, var=pd.DataFrame(index=['LigA','RecA','RecB','RecC']))
    >>> adata.obsm['spatial'] = np.array([[0,0],[1,0],[0,1],[1,2]], float)
    >>> # create a toy ligand-receptor database with two pairs sharing the same ligand
    >>> df_ligrec = pd.DataFrame([['LigA','RecA_RecB'],['LigA','RecC']])
    >>> df_ligrec = pd.DataFrame([['LigA','RecA_RecB','PathwayA'],['LigA','RecC','PathwayA']])
    >>> # infer CCC with a distance threshold of 1.1. Due the threshold, the last cell won't be communicating with other cells.
    >>> ct.tl.spatial_communication(adata, database_name='toyDB',df_ligrec=df_ligrec, pathway_sum=True, heteromeric=True, dis_thr=1.1)
    >>> adata
    AnnData object with n_obs × n_vars = 4 × 4
        uns: 'commot-toyDB-info'
        obsm: 'spatial', 'commot-toyDB-sum-sender', 'commot-toyDB-sum-receiver'
        obsp: 'commot-toyDB-LigA-RecA_RecB', 'commot-toyDB-LigA-RecC', 'commot-toyDB-PathwayA', 'commot-toyDB-total-total'
    >>> print(adata.obsp['commot-toyDB-LigA-RecA_RecB'])
    (0, 2)	0.600000000000005
    >>> print(adata.obsp['commot-toyDB-LigA-RecC'])
    (0, 2)	0.9000000000039632
    >>> print(adata.obsm['commot-toyDB-sum-receiver'])
    r-LigA-RecA_RecB  r-LigA-RecC  r-total-total  r-PathwayA
    0               0.0          0.0            0.0         0.0
    1               0.0          0.0            0.0         0.0
    2               0.6          0.9            1.5         1.5
    3               0.0          0.0            0.0         0.0
    >>> print(adata.obsm['commot-toyDB-sum-sender'])
    s-LigA-RecA_RecB  s-LigA-RecC  s-total-total  s-PathwayA
    0               0.6          0.9            1.5         1.5
    1               0.0          0.0            0.0         0.0
    2               0.0          0.0            0.0         0.0
    3               0.0          0.0            0.0         0.0
    >>> print(adata.obsp['commot-toyDB-PathwayA'])
    (0, 2)	1.5000000000039682
    >>> print(adata.obsp['commot-toyDB-total-total'])
    (0, 2)	1.5000000000039682
    >>> adata.uns['commot-toyDB-info']
    {'df_ligrec':   ligand   receptor   pathway
    0   LigA  RecA_RecB  PathwayA
    1   LigA       RecC  PathwayA, 'distance_threshold': 1.1}
    
    
    """

    assert database_name is not None, "Please give a database_name"
    assert df_ligrec is not None, "Please give a ligand-receptor database"

    if not 'spatial_distance' in adata.obsp.keys():
        dis_mat = distance_matrix(adata.obsm["spatial"], adata.obsm["spatial"])
        # assign_distance(adata, dmat=None)
    else:
        dis_mat = adata.obsp['spatial_distance']

    # remove unavailable genes from df_ligrec
    if not heteromeric:
        data_genes = list(adata.var_names)
        tmp_ligrec = []
        for i in range(df_ligrec.shape[0]):
            if df_ligrec.iloc[i][0] in data_genes and df_ligrec.iloc[i][1] in data_genes:
                tmp_ligrec.append([df_ligrec.iloc[i][0], df_ligrec.iloc[i][1], df_ligrec.iloc[i][2]])
        tmp_ligrec = np.array(tmp_ligrec, str)
        df_ligrec = pd.DataFrame(data=tmp_ligrec)
    elif heteromeric:
        data_genes = set(list(adata.var_names))
        tmp_ligrec = []
        for i in range(df_ligrec.shape[0]):
            tmp_lig = df_ligrec.iloc[i][0].split(heteromeric_delimiter)
            tmp_rec = df_ligrec.iloc[i][1].split(heteromeric_delimiter)
            if set(tmp_lig).issubset(data_genes) and set(tmp_rec).issubset(data_genes):
                tmp_ligrec.append([df_ligrec.iloc[i][0], df_ligrec.iloc[i][1], df_ligrec.iloc[i][2]])
        tmp_ligrec = np.array(tmp_ligrec, str)
        df_ligrec = pd.DataFrame(data=tmp_ligrec)
    # Drop duplicate pairs
    df_ligrec = df_ligrec.drop_duplicates()


    model = CellCommunication(adata, 
        df_ligrec, 
        dis_mat, 
        dis_thr, 
        cost_scale, 
        cost_type,
        heteromeric = heteromeric,
        heteromeric_rule = heteromeric_rule,
        heteromeric_delimiter = heteromeric_delimiter
    )
    model.run_cot_signaling(cot_eps_p=cot_eps_p, 
        cot_eps_mu = cot_eps_mu, 
        cot_eps_nu = cot_eps_nu, 
        cot_rho = cot_rho, 
        cot_nitermax = cot_nitermax, 
        cot_weights = cot_weights, 
        smooth = smooth, 
        smth_eta = smth_eta, 
        smth_nu = smth_nu, 
        smth_kernel = smth_kernel
    )

    
    adata.uns['commot-'+database_name+'-info'] = {}
    df_ligrec_write = pd.DataFrame(data=df_ligrec.values, columns=['ligand','receptor','pathway'])
    adata.uns['commot-'+database_name+'-info']['df_ligrec'] = df_ligrec_write
    adata.uns['commot-'+database_name+'-info']['distance_threshold'] = dis_thr

    ncell = adata.shape[0]
    X_sender = np.empty([ncell,0], float)
    X_receiver = np.empty([ncell,0], float)
    col_names_sender = []
    col_names_receiver = []
    tmp_ligs = model.ligs
    tmp_recs = model.recs
    S_total = sparse.csr_matrix((ncell, ncell), dtype=float)
    if pathway_sum:
        pathways = list(np.sort(list(set(df_ligrec.iloc[:,2].values))))
        S_pathway = [sparse.csr_matrix((ncell, ncell), dtype=float) for i in range(len(pathways))]
        X_sender_pathway = [np.zeros([ncell,1], float) for i in range(len(pathways))]
        X_receiver_pathway = [np.zeros([ncell,1], float) for i in range(len(pathways))]
    for (i,j) in model.comm_network.keys():
        S = model.comm_network[(i,j)]
        adata.obsp['commot-'+database_name+'-'+tmp_ligs[i]+'-'+tmp_recs[j]] = S
        S_total = S_total + S
        lig_sum = np.array(S.sum(axis=1))
        rec_sum = np.array(S.sum(axis=0).T)
        X_sender = np.concatenate((X_sender, lig_sum), axis=1)
        X_receiver = np.concatenate((X_receiver, rec_sum), axis=1)
        col_names_sender.append("s-%s-%s" % (tmp_ligs[i], tmp_recs[j]))
        col_names_receiver.append("r-%s-%s" % (tmp_ligs[i], tmp_recs[j]))
        if pathway_sum:
            mask = (df_ligrec.iloc[:,0] == tmp_ligs[i]) & (df_ligrec.iloc[:,1] == tmp_recs[j])
            pathway = df_ligrec[mask].iloc[:,2].values[0]
            pathway_idx = pathways.index(pathway)
            S_pathway[pathway_idx] = S_pathway[pathway_idx] + S
            X_sender_pathway[pathway_idx] = X_sender_pathway[pathway_idx] + np.array(S.sum(axis=1))
            X_receiver_pathway[pathway_idx] = X_receiver_pathway[pathway_idx] + np.array(S.sum(axis=0).T)
    if pathway_sum:
        for pathway_idx in range(len(pathways)):
            pathway = pathways[pathway_idx]
            adata.obsp['commot-'+database_name+'-'+pathway] = S_pathway[pathway_idx]

    X_sender = np.concatenate((X_sender, X_sender.sum(axis=1).reshape(-1,1)), axis=1)
    X_receiver = np.concatenate((X_receiver, X_receiver.sum(axis=1).reshape(-1,1)), axis=1)
    col_names_sender.append("s-total-total")
    col_names_receiver.append("r-total-total")
    
    if pathway_sum:
        for pathway_idx in range(len(pathways)):
            pathway = pathways[pathway_idx]
            X_sender = np.concatenate((X_sender, X_sender_pathway[pathway_idx]), axis=1)
            X_receiver = np.concatenate((X_receiver, X_receiver_pathway[pathway_idx]), axis=1)
            col_names_sender.append("s-"+pathway)
            col_names_receiver.append("r-"+pathway)

    adata.obsp['commot-'+database_name+'-total-total'] = S_total

    df_sender = pd.DataFrame(data=X_sender, columns=col_names_sender, index=adata.obs_names)
    df_receiver = pd.DataFrame(data=X_receiver, columns=col_names_receiver, index=adata.obs_names)
    adata.obsm['commot-'+database_name+'-sum-sender'] = df_sender
    adata.obsm['commot-'+database_name+'-sum-receiver'] = df_receiver

    del model
    gc.collect()

    return adata if copy else None


def communication_direction(
    adata: anndata.AnnData,
    database_name: str = None,
    pathway_name: str = None,
    lr_pair = None,
    k: int = 5,
    pos_idx: Optional[np.ndarray] = None,
    copy: bool = False
):
    """
    Construct spatial vector fields for inferred communication.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or spots and columns to genes.
    database_name
        Name of the ligand-receptor database.
        If both pathway_name and lr_pair are None, the signaling direction of all ligand-receptor pairs is computed.
    pathway_name
        Name of the signaling pathway.
        If given, only the signaling direction of this signaling pathway is computed.
    lr_pair
        A tuple of ligand-receptor pair. 
        If given, only the signaling direction of this pair is computed.
    k
        Top k senders or receivers to consider when determining the direction.
    pos_idx
        The columns in ``.obsm['spatial']`` to use. If None, all columns are used.
        For example, to use just the first and third columns, set pos_idx to ``numpy.array([0,2],int)``.
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.
    
    Returns
    -------
    adata : anndata.AnnData
        Vector fields describing signaling directions are added to ``.obsm``, 
        e.g., for a database named "databaseX", 
        ``.obsm['commot_sender_vf-databaseX-ligA-recA']`` and ``.obsm['commot_receiver_vf-databaseX-ligA-recA']``
        describe the signaling directions of the cells as, respectively, senders and receivers through the 
        ligand-receptor pair ligA and recA.
        If copy=True, return the AnnData object and return None otherwise.

    """

    obsp_names = []
    if not lr_pair is None:
        obsp_names.append(database_name+'-'+lr_pair[0]+'-'+lr_pair[1])
    elif not pathway_name is None:
        obsp_names.append(database_name+'-'+pathway_name)
    else:
        obsp_names.append(database_name+'-total-total')

    ncell = adata.shape[0]
    pts = np.array( adata.obsm['spatial'], float )
    if not pos_idx is None:
        pts = pts[:,pos_idx]
    storage = 'sparse'

    if storage == 'dense':
        for i in range(len(obsp_names)):
            # lig = name_mat[i,0]
            # rec = name_mat[i,1]
            S_np = adata.obsp['commot-'+obsp_names[i]].toarray()
            sender_vf = np.zeros_like(pts)
            receiver_vf = np.zeros_like(pts)

            tmp_idx = np.argsort(-S_np,axis=1)[:,:k]
            avg_v = np.zeros_like(pts)
            for ik in range(k):
                tmp_v = pts[tmp_idx[:,ik]] - pts[np.arange(ncell,dtype=int)]
                tmp_v = normalize(tmp_v, norm='l2')
                avg_v = avg_v + tmp_v * S_np[np.arange(ncell,dtype=int),tmp_idx[:,ik]].reshape(-1,1)
            avg_v = normalize(avg_v)
            sender_vf = avg_v * np.sum(S_np,axis=1).reshape(-1,1)

            S_np = S_np.T
            tmp_idx = np.argsort(-S_np,axis=1)[:,:k]
            avg_v = np.zeros_like(pts)
            for ik in range(k):
                tmp_v = -pts[tmp_idx[:,ik]] + pts[np.arange(ncell,dtype=int)]
                tmp_v = normalize(tmp_v, norm='l2')
                avg_v = avg_v + tmp_v * S_np[np.arange(ncell,dtype=int),tmp_idx[:,ik]].reshape(-1,1)
            avg_v = normalize(avg_v)
            receiver_vf = avg_v * np.sum(S_np,axis=1).reshape(-1,1)

            del S_np

    elif storage == 'sparse':
        for i in range(len(obsp_names)):
            # lig = name_mat[i,0]
            # rec = name_mat[i,1]
            S = adata.obsp['commot-'+obsp_names[i]]
            S_sum_sender = np.array( S.sum(axis=1) ).reshape(-1)
            S_sum_receiver = np.array( S.sum(axis=0) ).reshape(-1)
            sender_vf = np.zeros_like(pts)
            receiver_vf = np.zeros_like(pts)

            S_lil = S.tolil()
            for j in range(S.shape[0]):
                if len(S_lil.rows[j]) <= k:
                    tmp_idx = np.array( S_lil.rows[j], int )
                    tmp_data = np.array( S_lil.data[j], float )
                else:
                    row_np = np.array( S_lil.rows[j], int )
                    data_np = np.array( S_lil.data[j], float )
                    sorted_idx = np.argsort( -data_np )[:k]
                    tmp_idx = row_np[ sorted_idx ]
                    tmp_data = data_np[ sorted_idx ]
                if len(tmp_idx) == 0:
                    continue
                elif len(tmp_idx) == 1:
                    avg_v = pts[tmp_idx[0],:] - pts[j,:]
                else:
                    tmp_v = pts[tmp_idx,:] - pts[j,:]
                    tmp_v = normalize(tmp_v, norm='l2')
                    avg_v = tmp_v * tmp_data.reshape(-1,1)
                    avg_v = np.sum( avg_v, axis=0 )
                avg_v = normalize( avg_v.reshape(1,-1) )
                sender_vf[j,:] = avg_v[0,:] * S_sum_sender[j]
            
            S_lil = S.T.tolil()
            for j in range(S.shape[0]):
                if len(S_lil.rows[j]) <= k:
                    tmp_idx = np.array( S_lil.rows[j], int )
                    tmp_data = np.array( S_lil.data[j], float )
                else:
                    row_np = np.array( S_lil.rows[j], int )
                    data_np = np.array( S_lil.data[j], float )
                    sorted_idx = np.argsort( -data_np )[:k]
                    tmp_idx = row_np[ sorted_idx ]
                    tmp_data = data_np[ sorted_idx ]
                if len(tmp_idx) == 0:
                    continue
                elif len(tmp_idx) == 1:
                    avg_v = -pts[tmp_idx,:] + pts[j,:]
                else:
                    tmp_v = -pts[tmp_idx,:] + pts[j,:]
                    tmp_v = normalize(tmp_v, norm='l2')
                    avg_v = tmp_v * tmp_data.reshape(-1,1)
                    avg_v = np.sum( avg_v, axis=0 )
                avg_v = normalize( avg_v.reshape(1,-1) )
                receiver_vf[j,:] = avg_v[0,:] * S_sum_receiver[j]



            adata.obsm["commot_sender_vf-"+obsp_names[i]] = sender_vf
            adata.obsm["commot_receiver_vf-"+obsp_names[i]] = receiver_vf

    return adata if copy else None

def cluster_communication(
    adata: anndata.AnnData,
    database_name: str = None,
    pathway_name: str = None,
    lr_pair = None,
    clustering: str = None,
    n_permutations: int = 100,
    random_seed: int = 1,
    scale_factor: str = 'auto',
    copy: bool = False
):
    """
    Summarize cell-cell communication to cluster-cluster communication and compute p-values by permutating cell/spot labels.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or positions and columns to genes.
    database_name
        Name of the ligand-receptor database.
        If both pathway_name and lr_pair are None, the cluster signaling through all ligand-receptor pairs is summarized.
    pathway_name
        Name of the signaling pathway.
        If given, the signaling through all ligand-receptor pairs of the given pathway is summarized.
    lr_pair
        A tuple of ligand-receptor pair. 
        If given, only the cluster signaling through this pair is computed.
    clustering
        Name of clustering with the labels stored in ``.obs[clustering]``.
    n_permutations
        Number of label permutations for computing the p-value.
    random_seed
        The numpy random_seed for reproducible random permutations.
    scale_factor : str or float, default 'auto'
        Scaling factor for better visualization:
        - 'auto': automatically scale based on cluster sizes
        - 'sum': use sum instead of mean (multiply by cluster sizes)  
        - float: multiply results by this factor
        - None or 1: no scaling (original behavior)
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.
    
    Returns
    -------
    adata : anndata.AnnData
        Add cluster-cluster communication matrix to 
        ``.uns['commot_cluster-databaseX-clustering-ligA-recA']``
        for the ligand-receptor database named 'databaseX' and the cell clustering 
        named 'clustering' through the ligand-receptor pair 'ligA' and 'recA'.
        The first object is the communication score matrix and the second object contains
        the corresponding p-values.
        If copy=True, return the AnnData object and return None otherwise.

    """
    np.random.seed(random_seed)

    assert database_name is not None, "Please at least specify database_name."

    celltypes = list( adata.obs[clustering].unique() )
    celltypes.sort()
    for i in range(len(celltypes)):
        celltypes[i] = str(celltypes[i])
    clusterid = np.array(adata.obs[clustering], str)
    obsp_names = []
    if not lr_pair is None:
        obsp_names.append(database_name+'-'+lr_pair[0]+'-'+lr_pair[1])
    elif not pathway_name is None:
        obsp_names.append(database_name+'-'+pathway_name)
    else:
        obsp_names.append(database_name+'-total-total')
    # name_mat = adata.uns['commot-'+pathway_name+'-info']['df_ligrec'].values
    # name_mat = np.concatenate((name_mat, np.array([['total','total']],str)), axis=0)
    
    for i in range(len(obsp_names)):
        S = adata.obsp['commot-'+obsp_names[i]]
        tmp_df, tmp_p_value = summarize_cluster_gpu(S,
            clusterid, celltypes, n_permutations=n_permutations, scale_factor=scale_factor)
        adata.uns['commot_cluster-'+clustering+'-'+obsp_names[i]] = {'communication_matrix': tmp_df, 'communication_pvalue': tmp_p_value}
    
    return adata if copy else None


def cluster_communication_spatial_permutation(
    adata: anndata.AnnData,
    df_ligrec: pd.DataFrame = None,
    database_name: str = None,
    heteromeric: bool = False,
    heteromeric_rule: str = 'min',
    heteromeric_delimiter: str = '_',
    dis_thr: Optional[float] = None, 
    cost_scale: Optional[dict] = None, 
    cost_type: str = 'euc',
    cot_eps_p: float = 1e-1, 
    cot_eps_mu: Optional[float] = None, 
    cot_eps_nu: Optional[float] = None, 
    cot_rho: float =1e1, 
    cot_nitermax: int = 100, 
    cot_weights: tuple = (0.25,0.25,0.25,0.25), 
    smooth: bool = False, 
    smth_eta: float = None, 
    smth_nu: float = None, 
    smth_kernel: str = 'exp',
    clustering: str = None,
    perm_type: str = 'within_cluster',
    n_permutations: int = 100,
    random_seed: int = 1,
    verbose: bool = True,
    copy: bool = False
):
    """
    Infer cluster-cluster communication and compute p-value by permutating cell/spot locations.

    The cluster_communication function using label permutations is computationally efficient but may overestimate the communications of neighboring cell clusters.
    This function compute the p-value by permutating the locations of cell or spots and requires more computation time since the communication matrix is recomputed for each permutation.
    To avoid repeated calculation of the cell-level CCC matrices, the cluster-level CCC is summarized for all LR pairs and signaling pathways.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or spots and columns to genes.
        If the spatial distance is absent in ``.obsp['spatial_distance']``, Euclidean distance determined from ``.obsm['spatial']`` will be used.
    df_ligrec
        A data frame where each row corresponds to a ligand-receptor pair with ligands, receptors, and the associated signaling pathways in the three columns, respectively.
    database_name
        Name of the ligand-receptor interaction database. Will be included in the keywords for anndata slots.
    heteromeric
        Whether the ligands or receptors are made of heteromeric complexes.
    heteromeric_rule
        Use either 'min' (minimum) or 'ave' (average) expression of the components as the complex level.
    heteromeric_delimiter
        The character in ligand and receptor names separating individual components.
    dis_thr
        The threshold of spatial distance of signaling.
    cost_scale
        Weight coefficients of the cost matrix for each ligand-receptor pair, e.g. cost_scale[('ligA','recA')] specifies weight for the pair ligA and recA.
        If None, all pairs have the same weight. 
    cost_type
        If 'euc', the original Euclidean distance will be used as cost matrix. If 'euc_square', the square of the Euclidean distance will be used.
    cot_eps_p
        The coefficient of entropy regularization for transport plan.
    cot_eps_mu
        The coefficient of entropy regularization for untransported source (ligand). Set to equal to cot_eps_p for fast algorithm.
    cot_eps_nu
        The coefficient of entropy regularization for unfulfilled target (receptor). Set to equal to cot_eps_p for fast algorithm.
    cot_rho
        The coefficient of penalty for unmatched mass.
    cot_nitermax
        Maximum iteration for collective optimal transport algorithm.
        The default of this parameter is set to a much smaller one (100) compared to the default in ``spatial_communication`` to speed up the repeated OT calculation.
        The resulting communication matrices will be slightly different from the using 10000 for cot_nitermax but very similar.
    cot_weights
        A tuple of four weights that add up to one. The weights corresponds to four setups of collective optimal transport: 
        1) all ligands-all receptors, 2) each ligand-all receptors, 3) all ligands-each receptor, 4) each ligand-each receptor.
    smooth
        Whether to (spatially) smooth the gene expression for identifying more global signaling trend.
    smth_eta
        Kernel bandwidth for smoothing
    smth_nu
        Kernel sharpness for smoothing
    smth_kernel
        'exp' exponential kernel. 'lorentz' Lorentz kernel.
    clustering
        Name of clustering with the labels stored in ``.obs[clustering]``.
    perm_type
        The type of permutation to perform. 
        If "within_cluster", the cells/spots are permutated within each cluster.
        If "all_cell", the cells/spots are permutated all together.
    n_permutations
        Number of location permutations for computing the p-value.
    random_seed
        The numpy random_seed for reproducible random permutations.
    verbose
        Whether to print the permutation iterations.
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.

    Returns
    -------
    adata : anndata.AnnData
        For example, the cluster-level communication summary by this location permutation method for the LR pair 'LigA' and 'RecA' from the database 'databaseX' is stored in 
        ``adata.uns['commot_cluster_spatial_permutation-databaseX-clustering-ligA-recA']``
        If copy=True, return the AnnData object and return None otherwise.

    """

    assert perm_type in ['all_cell','within_cluster'], "Please specify 'all_cell' or 'within_cluster' for perm_type."

    # Assign a spatial distance matrix among spots
    dis_mats = []
    if not 'spatial_distance' in adata.obsp.keys():
        dis_mat = distance_matrix(adata.obsm["spatial"], adata.obsm["spatial"])
    else:
        dis_mat = adata.obsp['spatial_distance']

    # Remove unavailable genes from df_ligrec
    print(df_ligrec.shape)
    if not heteromeric:
        data_genes = list(adata.var_names)
        tmp_ligrec = []
        for i in range(df_ligrec.shape[0]):
            if df_ligrec.iloc[i][0] in data_genes and df_ligrec.iloc[i][1] in data_genes:
                tmp_ligrec.append([df_ligrec.iloc[i][0], df_ligrec.iloc[i][1], df_ligrec.iloc[i][2]])
        tmp_ligrec = np.array(tmp_ligrec, str)
        df_ligrec = pd.DataFrame(data=tmp_ligrec, columns=['ligand','receptor','pathway'])
    elif heteromeric:
        data_genes = set(list(adata.var_names))
        tmp_ligrec = []
        for i in range(df_ligrec.shape[0]):
            tmp_lig = df_ligrec.iloc[i][0].split(heteromeric_delimiter)
            tmp_rec = df_ligrec.iloc[i][1].split(heteromeric_delimiter)
            if set(tmp_lig).issubset(data_genes) and set(tmp_rec).issubset(data_genes):
                tmp_ligrec.append([df_ligrec.iloc[i][0], df_ligrec.iloc[i][1], df_ligrec.iloc[i][2]])
        tmp_ligrec = np.array(tmp_ligrec, str)
        df_ligrec = pd.DataFrame(data=tmp_ligrec, columns=['ligand','receptor','pathway'])
    # Drop duplicate pairs
    df_ligrec = df_ligrec.drop_duplicates()
    print(df_ligrec.shape)

    # Generate permutation indices
    perm_idx = []
    ncell = adata.shape[0]
    np.random.seed(random_seed)
    perm_idx.append(np.arange(ncell))
    for i in range(n_permutations):
        if perm_type == 'all_cell':
            perm_idx.append(np.random.permutation(ncell))
        elif perm_type == 'within_cluster':
            celltypes = np.sort( list( adata.obs[clustering].unique() ) )
            clusterid = np.array(adata.obs[clustering], str)
            tmp_perm_idx = np.arange(ncell)
            for celltype in celltypes:
                cell_idx = np.where(clusterid==celltype)[0]
                tmp_perm_idx[cell_idx] = np.random.permutation(cell_idx)
            perm_idx.append(tmp_perm_idx)
                
                
    
    # Compute the cluster-level commutation score for all permutations
    # and all lr_pairs, and pathways
    celltypes = list( adata.obs[clustering].unique() )
    celltypes.sort()
    clusterid = np.array(adata.obs[clustering], str)

    pathways = list(set(df_ligrec['pathway']))
    n_uns_names = len(df_ligrec) + len(pathways) + 1

    S_cl = np.empty([len(perm_idx), len(celltypes), len(celltypes), n_uns_names], float)

    uns_names = []


    for i_perm in range(len(perm_idx)):
        if verbose:
            print("Permutation: ", i_perm)
        adata_tmp = anndata.AnnData(adata[perm_idx[i_perm],:].X, 
            obs=pd.DataFrame(index=adata[perm_idx[i_perm],:].obs_names),
            var=pd.DataFrame(index=adata.var_names))
        model = CellCommunication(adata_tmp,
            df_ligrec, 
            dis_mat, 
            dis_thr, 
            cost_scale, 
            cost_type,
            heteromeric = heteromeric,
            heteromeric_rule = heteromeric_rule,
            heteromeric_delimiter = heteromeric_delimiter
        )
        model.run_cot_signaling(cot_eps_p=cot_eps_p, 
            cot_eps_mu = cot_eps_mu, 
            cot_eps_nu = cot_eps_nu, 
            cot_rho = cot_rho, 
            cot_nitermax = cot_nitermax, 
            cot_weights = cot_weights, 
            smooth = smooth, 
            smth_eta = smth_eta, 
            smth_nu = smth_nu, 
            smth_kernel = smth_kernel
        )
        for i_pair in range(len(df_ligrec)):
            lig = df_ligrec.iloc[i_pair]['ligand']
            rec = df_ligrec.iloc[i_pair]['receptor']
            i = model.ligs.index(lig)
            j = model.recs.index(rec)
            S_tmp = model.comm_network[(i,j)]
            uns_names.append(database_name+'-'+lig+'-'+rec)
            df_cluster_tmp,_ = summarize_cluster(S_tmp, clusterid, celltypes, n_permutations=1)
            S_cl[i_perm, :, :, i_pair] = df_cluster_tmp.values[:,:]
        for i_pathway in range(len(pathways)):
            pathway = pathways[i_pathway]
            S_tmp = sparse.csr_matrix((ncell, ncell), dtype=float)
            for i_pair in range(df_ligrec.shape[0]):
                if df_ligrec.iloc[i_pair,2] == pathway:
                    i = model.ligs.index(df_ligrec.iloc[i_pair]['ligand'])
                    j = model.recs.index(df_ligrec.iloc[i_pair]['receptor'])
                    S_tmp = S_tmp + model.comm_network[(i,j)]
            uns_names.append(database_name+'-'+pathway)
            df_cluster_tmp,_ = summarize_cluster(S_tmp, clusterid, celltypes, n_permutations=1)
            S_cl[i_perm, :, :, len(df_ligrec)+i_pathway] = df_cluster_tmp.values[:,:]
        S_tmp = sparse.csr_matrix((ncell, ncell), dtype=float)
        for (i,j) in model.comm_network.keys():
            S_tmp = S_tmp + model.comm_network[(i,j)]
        uns_names.append(database_name+'-total-total')
        df_cluster_tmp,_ = summarize_cluster(S_tmp, clusterid, celltypes, n_permutations=1)
        S_cl[i_perm, :, :, -1] = df_cluster_tmp.values[:,:]
        
        del model
        gc.collect()
    
    # Compute p-value
    for i_uns in range(S_cl.shape[-1]):
        p_cluster = np.zeros([len(celltypes), len(celltypes)], float)
        for i_perm in range(n_permutations):
            p_cluster[S_cl[i_perm+1,:,:, i_uns] >= S_cl[0,:,:, i_uns]] += 1.0
        p_cluster = p_cluster / n_permutations
        df_cluster = pd.DataFrame(data=S_cl[0,:,:,i_uns], index=celltypes, columns=celltypes)
        df_p_value = pd.DataFrame(data=p_cluster, index=celltypes, columns=celltypes)
        adata.uns['commot_cluster_spatial_permutation-'+clustering+'-'+uns_names[i_uns]] = {'communication_matrix': df_cluster, 'communication_pvalue': df_p_value}

    return adata if copy else None

def cluster_position(
    adata: anndata.AnnData,
    clustering: str = None,
    method: str = 'geometric_mean',
    copy: bool = False
):
    """
    Assign spatial positions to clusters.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or positions and columns to genes.
    clustering
        Name of clustering with the labels stored in ``.obs[clustering]``.
    method
        'geometric_mean' geometric mean of the positions. 
        'representative_point' the position in the cluster with 
        minimum total distance to other points.
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.

    Returns
    -------
    adata : anndata.AnnData
        Add cluster positions to ``.uns['cluster_pos-clustering_name']`` for the clustering named
        'clustering_name'.
        If copy=True, return the AnnData object and return None otherwise.

    """
    cluster_pos = cluster_center(adata, clustering, method=method)
    adata.uns['cluster_pos-'+clustering] = cluster_pos

    return adata if copy else None
