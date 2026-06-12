# harmonypy - A data alignment algorithm.
# Copyright (C) 2018  Ilya Korsunsky
#               2019  Kamil Slowikowski <kslowikowski@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
import logging
from datetime import datetime
from tqdm import tqdm
from ..._optional import normalize_torch_device

logger = logging.getLogger('harmonypy')
logger.addHandler(logging.NullHandler())

try:
    from ..._settings import EMOJI, Colors
except ImportError:
    EMOJI = {'start': '🚀', 'done': '✅', 'warning': '⚠️', 'cpu': '🖥️', 'gpu': '🚀'}
    Colors = type('Colors', (), {
        'CYAN': '\033[96m', 'BOLD': '\033[1m', 'ENDC': '\033[0m',
        'GREEN': '\033[92m', 'WARNING': '\033[93m',
    })()


def _mlx_available() -> bool:
    """Check if MLX is installed and Metal GPU is accessible."""
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except (ImportError, AttributeError):
        return False


def get_device(device=None):
    """Get the appropriate device for PyTorch operations.

    On Apple Silicon, returns ``'mlx'`` when MLX is available (preferred
    over PyTorch MPS which is still in beta).  All other platforms use
    the standard CUDA > CPU fallback.
    """
    if device is not None:
        if device == "mlx":
            return device
        return normalize_torch_device(device)

    if torch.cuda.is_available():
        return normalize_torch_device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if _mlx_available():
            return "mlx"
        return torch.device('mps')
    else:
        return torch.device('cpu')


def run_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    vars_use,
    theta=None,
    lamb=None,
    sigma=0.1, 
    nclust=None,
    tau=0,
    block_size=0.05, 
    max_iter_harmony=10,
    max_iter_kmeans=20,
    epsilon_cluster=1e-5,
    epsilon_harmony=1e-4, 
    alpha=0.2,
    verbose=True,
    random_state=0,
    device=None,
    **kwargs
):
    """Run Harmony batch effect correction.

    This is a PyTorch implementation matching the R package formulas.
    Supports CPU and GPU (CUDA, MPS) acceleration.
    
    Parameters
    ----------
    data_mat : np.ndarray
        PCA embedding matrix (cells x PCs or PCs x cells)
    meta_data : pd.DataFrame
        Metadata with batch variables (cells x variables)
    vars_use : str or list
        Column name(s) in meta_data to use for batch correction
    theta : float or list, optional
        Diversity penalty parameter(s). Default is 2 for each batch.
    lamb : float or list, optional
        Ridge regression penalty. Default is 1 for each batch.
        If -1, lambda is estimated automatically (matches R package).
    sigma : float, optional
        Kernel bandwidth for soft clustering. Default is 0.1.
    nclust : int, optional
        Number of clusters. Default is min(N/30, 100).
    tau : float, optional
        Protection against overcorrection. Default is 0.
    block_size : float, optional
        Proportion of cells to update in each block. Default is 0.05.
    max_iter_harmony : int, optional
        Maximum Harmony iterations. Default is 10.
    max_iter_kmeans : int, optional
        Maximum k-means iterations per Harmony iteration. Default is 20.
    epsilon_cluster : float, optional
        K-means convergence threshold. Default is 1e-5.
    epsilon_harmony : float, optional
        Harmony convergence threshold. Default is 1e-4.
    alpha : float, optional
        Alpha parameter for lambda estimation (when lamb=-1). Default is 0.2.
    verbose : bool, optional
        Print progress messages. Default is True.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    device : str, optional
        Device to use ('cpu', 'cuda', 'mps'). Default is auto-detect.
        
    Returns
    -------
    Harmony
        Harmony object with corrected data in Z_corr attribute.
    """
    import warnings as _warnings
    # Handle deprecated/removed kwargs from old API
    _removed = {'use_gpu', 'plot_convergence', 'reference_values',
                'cluster_prior', 'cluster_fn'}
    for k in list(kwargs):
        if k in _removed:
            _warnings.warn(
                f"run_harmony parameter '{k}' is no longer supported and will be ignored.",
                FutureWarning, stacklevel=2,
            )
            val = kwargs.pop(k)
            if k == 'use_gpu' and device is None:
                device = None if val else 'cpu'
    if kwargs:
        raise TypeError(
            f"run_harmony got unexpected keyword arguments: {list(kwargs)}"
        )

    N = meta_data.shape[0]
    if data_mat.shape[1] != N:
        data_mat = data_mat.T

    assert data_mat.shape[1] == N, \
       "data_mat and meta_data do not have the same number of cells" 

    if nclust is None:
        nclust = int(min(round(N / 30.0), 100))

    if not isinstance(sigma, np.ndarray):
        sigma = np.repeat(float(sigma), nclust)

    if isinstance(vars_use, str):
        vars_use = [vars_use]

    # Create batch indicator matrix (one-hot encoded)
    phi = pd.get_dummies(meta_data[vars_use]).to_numpy().T.astype(np.float32)
    phi_n = meta_data[vars_use].describe().loc['unique'].to_numpy().astype(int)

    # Theta handling - default is 2 (matches R package)
    if theta is None:
        theta = np.repeat([2] * len(phi_n), phi_n).astype(np.float32)
    elif isinstance(theta, (float, int)):
        theta = np.repeat([theta] * len(phi_n), phi_n).astype(np.float32)
    elif len(theta) == len(phi_n):
        theta = np.repeat([theta], phi_n).astype(np.float32)
    else:
        theta = np.asarray(theta, dtype=np.float32)

    assert len(theta) == np.sum(phi_n), \
        "each batch variable must have a theta"

    # Lambda handling (matches R package)
    lambda_estimation = False
    if lamb is None:
        lamb = np.repeat([1] * len(phi_n), phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    elif np.isscalar(lamb) and lamb == -1:
        lambda_estimation = True
        lamb = np.zeros(1, dtype=np.float32)
    elif isinstance(lamb, (float, int)):
        lamb = np.repeat([lamb] * len(phi_n), phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    elif len(lamb) == len(phi_n):
        lamb = np.repeat([lamb], phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    else:
        lamb = np.asarray(lamb, dtype=np.float32)
        if len(lamb) == np.sum(phi_n):
            lamb = np.insert(lamb, 0, 0).astype(np.float32)

    # Number of items in each category
    N_b = phi.sum(axis=1)
    Pr_b = (N_b / N).astype(np.float32)

    if tau > 0:
        theta = theta * (1 - np.exp(-(N_b / (nclust * tau)) ** 2))

    # Get device
    device_obj = get_device(device)
    use_mlx = (device_obj == "mlx")

    if verbose:
        if use_mlx:
            print(f"{EMOJI['gpu']} Using MLX Apple Silicon acceleration for Harmony")
        elif isinstance(device_obj, torch.device) and device_obj.type == "cuda":
            print(f"{EMOJI['gpu']} Using PyTorch CUDA acceleration for Harmony")
        elif isinstance(device_obj, torch.device) and device_obj.type == "cpu":
            print(f"{EMOJI['cpu']} Using NumPy CPU acceleration for Harmony")
        else:
            print(f"{EMOJI['gpu']} Using PyTorch {device_obj} acceleration for Harmony")
        print(f"{Colors.CYAN}    Data: {data_mat.shape[0]} PCs × {N} cells{Colors.ENDC}")
        print(f"{Colors.CYAN}    Batch variables: {vars_use}{Colors.ENDC}")
        print(f"{Colors.CYAN}    Max iterations: {Colors.BOLD}{max_iter_harmony}{Colors.ENDC}")
        print(f"{Colors.CYAN}    Convergence threshold: {Colors.BOLD}{epsilon_harmony}{Colors.ENDC}")

    # Set random seeds
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Ensure data_mat is a proper numpy array
    if hasattr(data_mat, 'values'):
        data_mat = data_mat.values
    data_mat = np.asarray(data_mat, dtype=np.float32)

    _common_args = (
        data_mat, phi, Pr_b, sigma.astype(np.float32),
        theta, lamb, alpha, lambda_estimation,
        max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, verbose,
        random_state,
    )

    if use_mlx:
        from ._harmony_mlx import HarmonyMLX
        ho = HarmonyMLX(*_common_args)
    elif isinstance(device_obj, torch.device) and device_obj.type == "cpu":
        from ._harmony_cpu import HarmonyCPU
        ho = HarmonyCPU(*_common_args)
    else:
        ho = Harmony(*_common_args, device_obj)

    return ho


class Harmony:
    """Harmony class for batch effect correction using PyTorch.
    
    Supports CPU and GPU acceleration.
    """
    
    def __init__(
            self, Z, Phi, Pr_b, sigma, theta, lamb, alpha, lambda_estimation,
            max_iter_harmony, max_iter_kmeans, 
            epsilon_kmeans, epsilon_harmony, K, block_size, verbose,
            random_state, device
    ):
        self.device = device
        
        # Convert to PyTorch tensors on device
        # Store with underscore prefix internally, expose as properties returning NumPy arrays
        self._Z_corr = torch.tensor(Z, dtype=torch.float32, device=device)
        self._Z_orig = torch.tensor(Z, dtype=torch.float32, device=device)
        
        # Simple L2 normalization
        self._Z_cos = self._Z_orig / torch.linalg.norm(self._Z_orig, ord=2, dim=0)
        
        # Batch indicators
        self._Phi = torch.tensor(Phi, dtype=torch.float32, device=device)
        self._Pr_b = torch.tensor(Pr_b, dtype=torch.float32, device=device)
        
        self.N = self._Z_corr.shape[1]
        self.B = Phi.shape[0]
        self.d = self._Z_corr.shape[0]
        
        # Build batch index for fast ridge correction
        self._batch_index = []
        for b in range(self.B):
            idx = torch.where(self._Phi[b, :] > 0)[0]
            self._batch_index.append(idx)
        
        # Create Phi_moe with intercept
        ones = torch.ones(1, self.N, dtype=torch.float32, device=device)
        self._Phi_moe = torch.cat([ones, self._Phi], dim=0)
        
        self.window_size = 3
        self.epsilon_kmeans = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self._lamb = torch.tensor(lamb, dtype=torch.float32, device=device)
        self.alpha = alpha
        self.lambda_estimation = lambda_estimation
        self._sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        self.block_size = block_size
        self.K = K
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose = verbose
        self._theta = torch.tensor(theta, dtype=torch.float32, device=device)

        # Generator for reproducible block shuffling in update_R
        self._rng = torch.Generator(device=device)
        self._rng.manual_seed(random_state)

        self.objective_harmony = []
        self.objective_kmeans = []
        self.objective_kmeans_dist = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross = []
        self.kmeans_rounds = []

        self.allocate_buffers()
        self.init_cluster(random_state)
        self.harmonize(self.max_iter_harmony, self.verbose)

    # =========================================================================
    # Properties - Return NumPy arrays for inspection and tutorials
    # =========================================================================
    
    @property
    def Z_corr(self):
        """Corrected embedding matrix (N x d). Batch effects removed."""
        return self._Z_corr.cpu().numpy().T
    
    @property
    def Z_orig(self):
        """Original embedding matrix (N x d). Input data before correction."""
        return self._Z_orig.cpu().numpy().T
    
    @property
    def Z_cos(self):
        """L2-normalized embedding matrix (N x d). Used for clustering."""
        return self._Z_cos.cpu().numpy().T
    
    @property
    def R(self):
        """Soft cluster assignment matrix (N x K). R[i,k] = P(cell i in cluster k)."""
        return self._R.cpu().numpy().T
    
    @property
    def Y(self):
        """Cluster centroids matrix (d x K). Columns are cluster centers."""
        return self._Y.cpu().numpy()
    
    @property
    def O(self):
        """Observed batch-cluster counts (K x B). O[k,b] = sum of R[k,:] for batch b."""
        return self._O.cpu().numpy()
    
    @property
    def E(self):
        """Expected batch-cluster counts (K x B). E[k,b] = cluster_size[k] * batch_proportion[b]."""
        return self._E.cpu().numpy()
    
    @property
    def Phi(self):
        """Batch indicator matrix (N x B). One-hot encoding of batch membership."""
        return self._Phi.cpu().numpy().T
    
    @property
    def Phi_moe(self):
        """Batch indicator with intercept (N x (B+1)). First column is all ones."""
        return self._Phi_moe.cpu().numpy().T
    
    @property
    def Pr_b(self):
        """Batch proportions (B,). Pr_b[b] = cells in batch b / total cells."""
        return self._Pr_b.cpu().numpy()
    
    @property
    def theta(self):
        """Diversity penalty parameters (B,). Higher = more mixing encouraged."""
        return self._theta.cpu().numpy()
    
    @property
    def sigma(self):
        """Clustering bandwidth parameters (K,). Soft assignment kernel width."""
        return self._sigma.cpu().numpy()
    
    @property 
    def lamb(self):
        """Ridge regression penalty ((B+1),). Regularization for batch correction."""
        return self._lamb.cpu().numpy()

    def result(self):
        """Return corrected data as NumPy array."""
        return self._Z_corr.cpu().numpy().T

    def allocate_buffers(self):
        self._scale_dist = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._dist_mat = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._O = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._E = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._R = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._Y = torch.zeros((self.d, self.K), dtype=torch.float32, device=self.device)

    def init_cluster(self, random_state):
        if self.verbose:
            print(f"{Colors.CYAN}    Initializing centroids (K={self.K}) ...{Colors.ENDC}", end=" ", flush=True)
        Z_cos_np = self._Z_cos.cpu().numpy()
        model = KMeans(n_clusters=self.K, init='k-means++',
                       n_init=1, max_iter=25, random_state=random_state)
        model.fit(Z_cos_np.T)
        self._Y = torch.tensor(model.cluster_centers_.T, dtype=torch.float32, device=self.device)
        if self.verbose:
            print("done")
        
        # Normalize centroids
        self._Y = self._Y / torch.linalg.norm(self._Y, ord=2, dim=0)
        
        # Compute distance matrix: dist = 2 * (1 - Y.T @ Z_cos)
        self._dist_mat = 2 * (1 - self._Y.T @ self._Z_cos)
        
        # Compute R
        self._R = -self._dist_mat / self._sigma[:, None]
        self._R = torch.exp(self._R)
        self._R = self._R / self._R.sum(dim=0)
        
        # Batch diversity statistics
        self._E = torch.outer(self._R.sum(dim=1), self._Pr_b)
        self._O = self._R @ self._Phi.T
        
        self.compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self):
        kmeans_error = torch.sum(self._R * self._dist_mat).item()
        _entropy = torch.sum(safe_entropy_torch(self._R) * self._sigma[:, None]).item()
        # Cross-entropy: R package formula with +1 smoothing to avoid log(0)
        R_sigma = self._R * self._sigma[:, None]
        theta_log = self._theta.unsqueeze(0).expand(self.K, -1) * torch.log((self._O + 1) / (self._E + 1))
        _cross_entropy = torch.sum(R_sigma * (theta_log @ self._Phi)).item()
        self.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(_entropy)
        self.objective_kmeans_cross.append(_cross_entropy)

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        if verbose:
            print(f"{EMOJI['start']} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running Harmony integration...")

        pbar = tqdm(range(1, iter_harmony + 1), desc="Harmony iterations", disable=not verbose)
        for i in pbar:
            self.cluster()
            self.moe_correct_ridge()

            converged = self.check_convergence(1)
            if verbose:
                pbar.set_description(f"Harmony iteration {i}/{iter_harmony}")
            if converged:
                if verbose:
                    pbar.set_description(f"Harmony converged after {i} iterations")
                    print(f"\n{EMOJI['done']} Harmony converged after {i} iteration{'s' if i > 1 else ''}")
                break
        pbar.close()

        if verbose and not converged:
            print(f"{EMOJI['warning']} Harmony stopped before convergence after {iter_harmony} iterations")

    def cluster(self):
        self._dist_mat = 2 * (1 - self._Y.T @ self._Z_cos)
        
        rounds = 0
        for i in range(self.max_iter_kmeans):
            # Update Y
            self._Y = self._Z_cos @ self._R.T
            self._Y = self._Y / torch.linalg.norm(self._Y, ord=2, dim=0)
            
            # Update distance matrix
            self._dist_mat = 2 * (1 - self._Y.T @ self._Z_cos)
            
            # Update R
            self.update_R()
            
            # Compute objective and check convergence
            self.compute_objective()
            
            if i > self.window_size:
                if self.check_convergence(0):
                    rounds = i + 1
                    break
            rounds = i + 1
            
        self.kmeans_rounds.append(rounds)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def update_R(self):
        # Compute scaled distances
        self._scale_dist = -self._dist_mat / self._sigma[:, None]
        self._scale_dist = torch.exp(self._scale_dist)
        self._scale_dist = self._scale_dist / self._scale_dist.sum(dim=0)
        
        # Reproducible shuffled update order
        update_order = torch.randperm(self.N, device=self.device, generator=self._rng)
        
        # Process in blocks
        n_blocks = int(np.ceil(1.0 / self.block_size))
        cells_per_block = int(self.N * self.block_size)
        
        # Permute matrices
        R_perm = self._R[:, update_order]
        scale_perm = self._scale_dist[:, update_order]
        Phi_perm = self._Phi[:, update_order]
        
        for blk in range(n_blocks):
            idx_min = blk * cells_per_block
            idx_max = self.N if blk == n_blocks - 1 else (blk + 1) * cells_per_block
            
            R_block = R_perm[:, idx_min:idx_max]
            scale_block = scale_perm[:, idx_min:idx_max]
            Phi_block = Phi_perm[:, idx_min:idx_max]
            
            # Remove cells from statistics
            self._E -= torch.outer(R_block.sum(dim=1), self._Pr_b)
            self._O -= R_block @ Phi_block.T
            
            # Recompute R for this block (R package formula) with numerical stability
            O_E_sum = self._O + self._E
            O_E_sum = torch.clamp(O_E_sum, min=1e-8)
            ratio = self._E / O_E_sum
            ratio = torch.clamp(ratio, min=1e-8, max=1.0)
            ratio_powered = harmony_pow_torch(ratio, self._theta)
            R_block_new = scale_block * (ratio_powered @ Phi_block)
            R_block_sum = R_block_new.sum(dim=0)
            R_block_sum = torch.clamp(R_block_sum, min=1e-8)
            R_block_new = R_block_new / R_block_sum
            
            # Put cells back
            self._E += torch.outer(R_block_new.sum(dim=1), self._Pr_b)
            self._O += R_block_new @ Phi_block.T
            
            R_perm[:, idx_min:idx_max] = R_block_new
        
        # Restore original order
        inverse_order = torch.argsort(update_order)
        self._R = R_perm[:, inverse_order]

    def check_convergence(self, i_type):
        if i_type == 0:
            if len(self.objective_kmeans) <= self.window_size + 1:
                return False
            
            w = self.window_size
            obj_old = sum(self.objective_kmeans[-w-1:-1])
            obj_new = sum(self.objective_kmeans[-w:])
            if abs(obj_old) < 1e-10:
                return True
            return abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans
        
        if i_type == 1:
            if len(self.objective_harmony) < 2:
                return False
            
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            if abs(obj_old) < 1e-10:
                return True
            return (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony
        
        return True

    def moe_correct_ridge(self):
        """Ridge regression correction for batch effects."""
        self._Z_corr = self._Z_orig.clone()
        
        for k in range(self.K):
            # Compute lambda if estimating
            if self.lambda_estimation:
                lamb_vec = find_lambda_torch(self.alpha, self._E[k, :], self.device)
            else:
                lamb_vec = self._lamb
            
            # Phi_Rk = Phi_moe scaled by R[k,:]
            Phi_Rk = self._Phi_moe * self._R[k, :]
            
            # Compute covariance
            cov_mat = Phi_Rk @ self._Phi_moe.T + torch.diag(lamb_vec)
            
            # Invert (with fallback for near-singular matrices)
            try:
                inv_cov = torch.linalg.inv(cov_mat)
            except torch.linalg.LinAlgError:
                inv_cov = torch.linalg.pinv(cov_mat)
            
            # W = inv(Phi_Rk @ Phi_moe.T + diag(lamb)) @ Phi_Rk @ Z_orig.T
            W = inv_cov @ Phi_Rk @ self._Z_orig.T
            
            W[0, :] = 0  # Do not remove intercept
            self._Z_corr = self._Z_corr - W.T @ Phi_Rk
        
        # Update Z_cos
        self._Z_cos = self._Z_corr / torch.linalg.norm(self._Z_corr, ord=2, dim=0)


def safe_entropy_torch(x):
    """Compute x * log(x), returning 0 where x is 0 or negative."""
    result = x * torch.log(x)
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    return result


def harmony_pow_torch(A, T):
    """Element-wise power with per-column exponents: A^T broadcast."""
    return torch.pow(A, T.unsqueeze(0))


def find_lambda_torch(alpha, cluster_E, device):
    """Compute dynamic lambda based on cluster expected counts."""
    lamb = torch.zeros(len(cluster_E) + 1, dtype=torch.float32, device=device)
    lamb[1:] = cluster_E * alpha
    return lamb
