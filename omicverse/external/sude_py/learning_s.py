from sklearn.neighbors import NearestNeighbors
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist
from .init_pca import init_pca
from .pca import pca
from .mds import mds
import numpy as np
from tqdm import tqdm

from scipy.sparse.linalg import eigsh, LinearOperator


def learning_s(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, use_gpu=True, verbose=True):
    """
    This function returns representation of the landmarks in the lower-dimensional space and the number of nearest
    neighbors of landmarks. It computes the gradient using the entire probability matrix P and Q.
    Now supports GPU acceleration via MLX (Apple Silicon) or Torch (CUDA).

    Parameters
    ----------
    X_samp : array-like
        Landmark samples data matrix.
    k1 : int
        Number of nearest neighbors parameter.
    get_knn : array-like
        K-nearest neighbors indices.
    rnn : array-like
        Reverse nearest neighbors matrix.
    id_samp : array-like
        Sample indices.
    no_dims : int
        Number of dimensions for the output embedding.
    initialize : str
        Initialization method ('le', 'pca', or 'mds').
    agg_coef : float
        Aggregation coefficient.
    T_epoch : int
        Number of training epochs.
    use_gpu : bool, optional
        Whether to use GPU acceleration when available. Default: True.
    verbose : bool, optional
        Whether to print device selection information. Default: False.

    Returns
    -------
    tuple
        (Y, k2) where Y is the embedding and k2 is the number of nearest neighbors.
    """
    # If GPU is explicitly disabled, use CPU implementation
    if not use_gpu:
        return _learning_s_cpu(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, verbose)

    from ..._settings import settings
    omicverse_mode = getattr(settings, 'mode', 'cpu')
    
    # Detect optimal device and backend
    device_info = _detect_optimal_learning_s_backend(use_gpu, verbose)
    
    if device_info['backend'] == 'mlx' and device_info['device'] == 'mps' and omicverse_mode != 'cpu':
        return _learning_s_mlx(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, verbose)
    elif device_info['backend'] == 'torch' and device_info['device'] in ['cuda', 'cpu'] and omicverse_mode != 'cpu':
        return _learning_s_torch(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, device_info['device'], verbose)
    else:
        # Fallback to CPU implementation
        return _learning_s_cpu(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, verbose)


def _detect_optimal_learning_s_backend(use_gpu=True, verbose=False):
    """
    Detect the optimal learning_s backend based on available hardware and omicverse settings.
    
    Parameters
    ----------
    use_gpu : bool
        Whether to prefer GPU acceleration
    verbose : bool
        Whether to print detection information
        
    Returns
    -------
    dict
        Dictionary containing backend and device information
    """
    # Import omicverse settings
    try:
        from ..._settings import get_optimal_device, settings
        device = get_optimal_device(prefer_gpu=use_gpu, verbose=verbose)
        
        # Check omicverse mode
        omicverse_mode = getattr(settings, 'mode', 'cpu')
        
        if verbose:
            print(f"   Omicverse mode: {omicverse_mode}")
            print(f"   Detected device: {device}")
        
        # Determine backend based on device and omicverse mode
        if hasattr(device, 'type'):
            device_type = device.type
        else:
            device_type = str(device)
            
        if device_type == 'mps' and use_gpu and omicverse_mode != 'cpu':
            # Try MLX for Apple Silicon
            try:
                import mlx.core as mx
                if mx.metal.is_available():
                    return {'backend': 'mlx', 'device': 'mps', 'available': True}
            except ImportError:
                pass
                
        elif device_type == 'cuda' and use_gpu and omicverse_mode != 'cpu':
            # Try Torch for CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    return {'backend': 'torch', 'device': 'cuda', 'available': True}
            except ImportError:
                pass
        
        # Try Torch CPU as fallback
        try:
            import torch
            return {'backend': 'torch', 'device': 'cpu', 'available': True}
        except ImportError:
            pass
        
        # Fallback to pure CPU
        return {'backend': 'cpu', 'device': 'cpu', 'available': True}
        
    except ImportError:
        # Fallback if omicverse settings not available
        try:
            import torch
            return {'backend': 'torch', 'device': 'cpu', 'available': True}
        except ImportError:
            return {'backend': 'cpu', 'device': 'cpu', 'available': True}


def _learning_s_cpu(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, verbose=False):
    """
    Original CPU-based learning_s implementation.
    """
    if verbose:
        print(f"   🖥️ Using CPU learning_s implementation")
    
    # Obtain size and dimension of landmarks
    N, dim = X_samp.shape

    # Compute the number of nearest neighbors of landmarks adaptively
    if N < 9:
        k2 = N
    else:
        if N > 1000:
            k2 = int(np.ceil(np.log2(N)) + 18)
        elif N > 50:
            k2 = int(np.ceil(0.02 * N)) + 8
        else:
            k2 = 9

    # Compute high-dimensional probability matrix P
    if k1 > 0:
        # Compute SNN matrix of landmarks
        SNN = np.zeros((N, N))
        knn_rnn_mat = rnn[get_knn[id_samp]]
        for i in range(N):
            snn_id = np.isin(get_knn[id_samp], get_knn[id_samp[i]]).astype(int)
            nn_id = np.where(np.max(snn_id, axis=1) == 1)[0]
            SNN[i, nn_id] = np.sum(knn_rnn_mat[nn_id] * snn_id[nn_id], axis=1)
            SNN[i] = SNN[i] / max(np.max(SNN[i]), np.finfo(float).tiny)
        # Compute the modified distance matrix
        Dis = (1 - SNN) ** agg_coef * cdist(X_samp, X_samp)
        P = np.zeros((N, N))
        sort_dis = np.sort(Dis, axis=1)
        idx = np.argsort(Dis, axis=1)
        for i in range(N):
            P[i, idx[i, :k2]] = np.exp(
                -0.5 * np.square(sort_dis[i, :k2]) / np.maximum(np.square(np.mean(sort_dis[i, :k2])),
                                                                np.finfo(float).tiny))
    else:
        if N > 5000 and dim > 50:
            xx = init_pca(X_samp, no_dims, 0.8)
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(xx).kneighbors(xx)
        else:
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(X_samp).kneighbors(X_samp)
        mean_samp_dis_squared = np.square(np.mean(samp_dis, axis=1))
        Pval = np.exp(-0.5 * np.square(samp_dis) / np.maximum(mean_samp_dis_squared[:, np.newaxis], np.finfo(float).tiny))
        P = csr_matrix((Pval.flatten(), ([i for i in range(N) for _ in range(k2)], samp_knn.flatten())), shape=(N, N)).toarray()
    # Symmetrize matrix P
    P = (P + P.transpose()) / 2

    # Initialize embedding Y of landmarks
    if initialize == 'le':
        # 归一化图拉普拉斯：L = I - D^{-1/2} P D^{-1/2}
        # 注：若 P 对称，axis=0/1 求和等价；此处沿用你原实现的 axis=0。
        deg = np.asarray(P.sum(axis=0)).ravel()

        # 避免度为0导致的除零
        d_inv_sqrt = np.zeros_like(deg, dtype=float)
        nz = deg > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])

        n = P.shape[0]

        # 通过 LinearOperator 隐式实现 L 的 matvec，避免显式构造大矩阵
        def matvec(x):
            # y = D^{-1/2} P D^{-1/2} x
            y = P @ (d_inv_sqrt * x)
            y = d_inv_sqrt * y
            return x - y  # (I - D^{-1/2} P D^{-1/2}) x

        Lop = LinearOperator((n, n), matvec=matvec, dtype=float)

        # 取最小的 k 个特征对（含零特征值），丢掉首个零特征向量
        k = min(no_dims + 1, max(2, n - 1))
        vals, vecs = eigsh(Lop, k=k, which='SM')

        # 保险起见按特征值从小到大排序后丢第一列
        order = np.argsort(vals)
        Y = np.ascontiguousarray(vecs[:, order[1:no_dims+1]])

    elif initialize == 'pca':
        Y = pca(X_samp, no_dims)

    elif initialize == 'mds':
        Y = mds(X_samp, no_dims, use_gpu=False)  # Disable GPU for CPU mode

    # Normalize matrix P
    P = P / (np.sum(P) - N)

    # Initialization
    max_alpha = 2.5 * N
    min_alpha = 2 * N
    warm_step = 10
    preGrad = np.zeros((N, no_dims))
    
    # Use tqdm for epoch progress tracking
    for epoch in tqdm(range(1, T_epoch + 1), desc="Training epochs", unit="epoch"):
        # Update learning rate
        if epoch <= warm_step:
            alpha = max_alpha
        else:
            alpha = min_alpha + 0.5 * (max_alpha - min_alpha) * (
                        1 + np.cos(np.pi * ((epoch - warm_step) / (T_epoch - warm_step))))
        # Update matrix Q
        D = cdist(Y, Y) ** 2
        Q1 = 1 / (1 + np.log(1 + D))
        QQ1 = 1 / (1 + D)
        Q = Q1 / (np.sum(Q1) - N)
        # Compute gradient
        ProMatY = 4 * (P - Q) * Q1 * QQ1
        grad = (np.diag(np.sum(ProMatY, axis=0)) - ProMatY) @ Y
        # Update embedding Y
        Y = Y - alpha * (grad + (epoch - 1) / (epoch + 2) * preGrad)
        preGrad = grad

    if verbose:
        print(f"   ✅ CPU learning_s completed: {X_samp.shape} -> {Y.shape}")
    return Y, k2


def _compute_full_pairwise_distances_mlx(Y):
    """
    Compute full pairwise squared distances using MLX.
    
    Parameters
    ----------
    Y : mlx.array
        Data matrix (N x d)
        
    Returns
    -------
    mlx.array
        Squared distance matrix (N x N)
    """
    import mlx.core as mx
    
    # Compute squared distance matrix using efficient broadcasting
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a@b.T
    norms = mx.sum(Y ** 2, axis=1, keepdims=True)  # (N, 1)
    D_squared = norms + norms.T - 2 * Y @ Y.T      # (N, N)
    
    return mx.maximum(D_squared, 0.0)  # Ensure non-negative


def _compute_full_pairwise_distances_torch(Y, device):
    """
    Compute full pairwise squared distances using Torch.
    
    Parameters
    ----------
    Y : torch.Tensor
        Data matrix (N x d)
    device : str
        Device to use
        
    Returns
    -------
    torch.Tensor
        Squared distance matrix (N x N)
    """
    import torch
    
    # Compute squared distance matrix using efficient broadcasting
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a@b.T
    norms = torch.sum(Y ** 2, dim=1, keepdim=True)  # (N, 1)
    D_squared = norms + norms.T - 2 * Y @ Y.T        # (N, N)
    
    return torch.clamp(D_squared, min=0.0)  # Ensure non-negative


def _learning_s_mlx(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, verbose=False):
    """
    MLX-based learning_s implementation for Apple Silicon MPS devices.
    """
    if verbose:
        print(f"   🚀 Using MLX learning_s for Apple Silicon MPS acceleration")
    
    import mlx.core as mx
    
    # Convert to numpy first for initial computations
    X_samp = np.asarray(X_samp, dtype=np.float32)
    N, dim = X_samp.shape

    # Compute k2 (same as CPU)
    if N < 9:
        k2 = N
    else:
        if N > 1000:
            k2 = int(np.ceil(np.log2(N)) + 18)
        elif N > 50:
            k2 = int(np.ceil(0.02 * N)) + 8
        else:
            k2 = 9

    # Initial P matrix computation (CPU for complexity)
    if k1 > 0:
        # Compute SNN matrix of landmarks
        SNN = np.zeros((N, N))
        knn_rnn_mat = rnn[get_knn[id_samp]]
        for i in range(N):
            snn_id = np.isin(get_knn[id_samp], get_knn[id_samp[i]]).astype(int)
            nn_id = np.where(np.max(snn_id, axis=1) == 1)[0]
            SNN[i, nn_id] = np.sum(knn_rnn_mat[nn_id] * snn_id[nn_id], axis=1)
            SNN[i] = SNN[i] / max(np.max(SNN[i]), np.finfo(float).tiny)
        # Compute the modified distance matrix
        Dis = (1 - SNN) ** agg_coef * cdist(X_samp, X_samp)
        P = np.zeros((N, N))
        sort_dis = np.sort(Dis, axis=1)
        idx = np.argsort(Dis, axis=1)
        for i in range(N):
            P[i, idx[i, :k2]] = np.exp(
                -0.5 * np.square(sort_dis[i, :k2]) / np.maximum(np.square(np.mean(sort_dis[i, :k2])),
                                                                np.finfo(float).tiny))
    else:
        if N > 5000 and dim > 50:
            xx = init_pca(X_samp, no_dims, 0.8)
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(xx).kneighbors(xx)
        else:
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(X_samp).kneighbors(X_samp)
        mean_samp_dis_squared = np.square(np.mean(samp_dis, axis=1))
        Pval = np.exp(-0.5 * np.square(samp_dis) / np.maximum(mean_samp_dis_squared[:, np.newaxis], np.finfo(float).tiny))
        P = csr_matrix((Pval.flatten(), ([i for i in range(N) for _ in range(k2)], samp_knn.flatten())), shape=(N, N)).toarray()
    
    P = (P + P.transpose()) / 2

    # Initialize embedding Y
    if initialize == 'le':
        deg = np.asarray(P.sum(axis=0)).ravel()
        d_inv_sqrt = np.zeros_like(deg, dtype=float)
        nz = deg > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
        n = P.shape[0]
        
        def matvec(x):
            y = P @ (d_inv_sqrt * x)
            y = d_inv_sqrt * y
            return x - y
        
        Lop = LinearOperator((n, n), matvec=matvec, dtype=float)
        k = min(no_dims + 1, max(2, n - 1))
        vals, vecs = eigsh(Lop, k=k, which='SM')
        order = np.argsort(vals)
        Y = np.ascontiguousarray(vecs[:, order[1:no_dims+1]], dtype=np.float32)
    elif initialize == 'pca':
        Y = pca(X_samp, no_dims).astype(np.float32)
    elif initialize == 'mds':
        Y = mds(X_samp, no_dims, use_gpu=True, verbose=False).astype(np.float32)

    # Convert to MLX arrays for GPU computation
    Y_mx = mx.array(Y)
    P_mx = mx.array(P.astype(np.float32))
    P = P / (np.sum(P) - N)
    P_mx = mx.array(P.astype(np.float32))

    # Training loop with GPU acceleration
    max_alpha = 2.5 * N
    min_alpha = 2 * N
    warm_step = 10
    preGrad_mx = mx.zeros((N, no_dims))
    
    for epoch in tqdm(range(1, T_epoch + 1), desc="Training epochs (MLX)", unit="epoch"):
        if epoch <= warm_step:
            alpha = max_alpha
        else:
            alpha = min_alpha + 0.5 * (max_alpha - min_alpha) * (
                        1 + np.cos(np.pi * ((epoch - warm_step) / (T_epoch - warm_step))))
        
        # GPU-accelerated distance computation
        D_squared = _compute_full_pairwise_distances_mlx(Y_mx)
        
        # GPU-accelerated element-wise operations
        Q1 = 1 / (1 + mx.log(1 + D_squared))
        QQ1 = 1 / (1 + D_squared)
        Q = Q1 / (mx.sum(Q1) - N)
        
        # GPU-accelerated matrix operations
        ProMatY = 4 * (P_mx - Q) * Q1 * QQ1
        diag_sum = mx.diag(mx.sum(ProMatY, axis=0))
        grad = (diag_sum - ProMatY) @ Y_mx
        
        # Update embedding
        Y_mx = Y_mx - alpha * (grad + (epoch - 1) / (epoch + 2) * preGrad_mx)
        preGrad_mx = grad
    
    # Convert back to numpy
    Y_result = np.array(Y_mx)
    
    if verbose:
        print(f"   ✅ MLX learning_s completed: {X_samp.shape} -> {Y_result.shape}")
    
    return Y_result, k2
        
    
def _learning_s_torch(X_samp, k1, get_knn, rnn, id_samp, no_dims, initialize, agg_coef, T_epoch, device='cpu', verbose=False):
    """
    Torch-based learning_s implementation for CUDA/CPU devices.
    """
    if verbose:
        print(f"   🚀 Using Torch learning_s on {device.upper()}")
    
    import torch
    
    # Convert to tensor
    X_samp = torch.tensor(np.asarray(X_samp, dtype=np.float32), device=device)
    N, dim = X_samp.shape

    # Compute k2 (same as CPU)
    if N < 9:
        k2 = N
    else:
        if N > 1000:
            k2 = int(np.ceil(np.log2(N)) + 18)
        elif N > 50:
            k2 = int(np.ceil(0.02 * N)) + 8
        else:
            k2 = 9

    # Initial P matrix computation (CPU for complexity, then transfer to GPU)
    X_np = X_samp.cpu().numpy()
    if k1 > 0:
        # Compute SNN matrix of landmarks
        SNN = np.zeros((N, N))
        knn_rnn_mat = rnn[get_knn[id_samp]]
        for i in range(N):
            snn_id = np.isin(get_knn[id_samp], get_knn[id_samp[i]]).astype(int)
            nn_id = np.where(np.max(snn_id, axis=1) == 1)[0]
            SNN[i, nn_id] = np.sum(knn_rnn_mat[nn_id] * snn_id[nn_id], axis=1)
            SNN[i] = SNN[i] / max(np.max(SNN[i]), np.finfo(float).tiny)
        # Compute the modified distance matrix
        Dis = (1 - SNN) ** agg_coef * cdist(X_np, X_np)
        P = np.zeros((N, N))
        sort_dis = np.sort(Dis, axis=1)
        idx = np.argsort(Dis, axis=1)
        for i in range(N):
            P[i, idx[i, :k2]] = np.exp(
                -0.5 * np.square(sort_dis[i, :k2]) / np.maximum(np.square(np.mean(sort_dis[i, :k2])),
                                                                np.finfo(float).tiny))
    else:
        if N > 5000 and dim > 50:
            xx = init_pca(X_np, no_dims, 0.8)
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(xx).kneighbors(xx)
        else:
            samp_dis, samp_knn = NearestNeighbors(n_neighbors=k2).fit(X_np).kneighbors(X_np)
        mean_samp_dis_squared = np.square(np.mean(samp_dis, axis=1))
        Pval = np.exp(-0.5 * np.square(samp_dis) / np.maximum(mean_samp_dis_squared[:, np.newaxis], np.finfo(float).tiny))
        P = csr_matrix((Pval.flatten(), ([i for i in range(N) for _ in range(k2)], samp_knn.flatten())), shape=(N, N)).toarray()
    
    P = (P + P.transpose()) / 2

    # Initialize embedding Y
    if initialize == 'le':
        deg = np.asarray(P.sum(axis=0)).ravel()
        d_inv_sqrt = np.zeros_like(deg, dtype=float)
        nz = deg > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
        n = P.shape[0]
        
        def matvec(x):
            y = P @ (d_inv_sqrt * x)
            y = d_inv_sqrt * y
            return x - y
        
        Lop = LinearOperator((n, n), matvec=matvec, dtype=float)
        k = min(no_dims + 1, max(2, n - 1))
        vals, vecs = eigsh(Lop, k=k, which='SM')
        order = np.argsort(vals)
        Y = np.ascontiguousarray(vecs[:, order[1:no_dims+1]], dtype=np.float32)
    elif initialize == 'pca':
        Y = pca(X_np, no_dims).astype(np.float32)
    elif initialize == 'mds':
        Y = mds(X_np, no_dims, use_gpu=(device!='cpu'), verbose=False).astype(np.float32)

    # Convert to torch tensors for GPU computation
    Y_torch = torch.tensor(Y, device=device)
    P = P / (np.sum(P) - N)
    P_torch = torch.tensor(P.astype(np.float32), device=device)

    # Training loop with GPU acceleration
    max_alpha = 2.5 * N
    min_alpha = 2 * N
    warm_step = 10
    preGrad_torch = torch.zeros((N, no_dims), device=device)
    
    for epoch in tqdm(range(1, T_epoch + 1), desc=f"Training epochs (Torch-{device.upper()})", unit="epoch"):
        if epoch <= warm_step:
            alpha = max_alpha
        else:
            alpha = min_alpha + 0.5 * (max_alpha - min_alpha) * (
                        1 + np.cos(np.pi * ((epoch - warm_step) / (T_epoch - warm_step))))
        
        # GPU-accelerated distance computation
        D_squared = _compute_full_pairwise_distances_torch(Y_torch, device)
        
        # GPU-accelerated element-wise operations
        Q1 = 1 / (1 + torch.log(1 + D_squared))
        QQ1 = 1 / (1 + D_squared)
        Q = Q1 / (torch.sum(Q1) - N)
        
        # GPU-accelerated matrix operations
        ProMatY = 4 * (P_torch - Q) * Q1 * QQ1
        diag_sum = torch.sum(ProMatY, dim=0)
        grad = (torch.diag(diag_sum) - ProMatY) @ Y_torch
        
        # Update embedding
        Y_torch = Y_torch - alpha * (grad + (epoch - 1) / (epoch + 2) * preGrad_torch)
        preGrad_torch = grad
    
    # Convert back to numpy
    Y_result = Y_torch.cpu().numpy()
    
    if verbose:
        print(f"   ✅ Torch learning_s completed: {X_samp.shape} -> {Y_result.shape}")
    
    return Y_result, k2
