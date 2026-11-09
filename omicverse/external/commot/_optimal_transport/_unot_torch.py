import torch
import numpy as np
from scipy import sparse
from scipy.spatial import distance_matrix
from scipy.special import wrightomega
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress bar class
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.total = total
            self.desc = desc
            self.n = 0
            
        def __enter__(self):
            if self.desc:
                print(f"{self.desc}: 0/{self.total}")
            return self
            
        def __exit__(self, *args):
            if self.desc:
                print(f"{self.desc}: {self.total}/{self.total} - Complete!")
                
        def update(self, n=1):
            self.n += n
            if self.desc and self.n % max(1, self.total // 10) == 0:
                print(f"{self.desc}: {self.n}/{self.total}")
                
        def set_postfix(self, **kwargs):
            pass


def _solver_dtype(device) -> torch.dtype:
    """float64 like the original NumPy COMMOT; MPS has no float64.

    In float32 the Sinkhorn error never reaches ``stopthr=1e-8`` (float32
    epsilon is 1.2e-7), so every problem ran the full ``nitermax`` iterations
    and stopped ~1e-5 away from COMMOT's solution. In float64 it converges in
    tens to a few thousand iterations and matches COMMOT to ~1e-14.
    """
    return torch.float32 if str(device).startswith("mps") else torch.float64


def unot_torch(a,
               b,
               C,
               eps_p,
               rho,
               eps_mu=None,
               eps_nu=None,
               sparse_mtx=False,
               solver="sinkhorn",
               nitermax=10000,
               stopthr=1e-8,
               verbose=False,
               show_progress=False,
               momentum_dt=0.1,
               momentum_beta=0.0,
               device='cuda' if torch.cuda.is_available() else 'cpu'):
    """ The main function calling different algorithms using PyTorch for acceleration.

    Parameters
    ----------
    a : (ns,) numpy.ndarray or torch.Tensor
        Source distribution. The summation should be less than or equal to 1.
    b : (nt,) numpy.ndarray or torch.Tensor
        Target distribution. The summation should be less than or equal to 1.
    C : (ns,nt) numpy.ndarray or torch.Tensor
        The cost matrix possibly with infinity entries.
    eps_p :  float
        The coefficient of entropy regularization for P.
    rho : float
        The coefficient of penalty for unmatched mass.
    eps_mu : float, defaults to eps_p
        The coefficient of entropy regularization for mu.
    eps_nu : float, defaults to eps_p
        The coefficient of entropy regularization for nu.
    sparse_mtx : boolean, defaults to False
        Whether using sparse matrix format. If True, C should be in coo_sparse format.
    solver : str, defaults to 'sinkhorn'
        The solver to use. Choose from 'sinkhorn' and 'momentum'.
    nitermax : int, defaults to 10000
        The maximum number of iterations.
    stopthr : float, defaults to 1e-8
        The relative error threshold for stopping.
    verbose : boolean, defaults to False
        Whether to print algorithm logs.
    show_progress : boolean, defaults to True
        Whether to show progress bar during iterations.
    momentum_dt : float, defaults to 1e-1
        Step size if momentum method is used.
    momentum_beta : float, defautls to 0
        The coefficient for the momentum term if momemtum method is used.
    device : str, defaults to 'cuda' if available else 'cpu'
        Device to run computation on.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    
    # Convert numpy arrays to torch tensors
    dtype = _solver_dtype(device)
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).to(device=device, dtype=dtype)
    else:
        a = a.to(device=device, dtype=dtype)
    
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b).to(device=device, dtype=dtype)
    else:
        b = b.to(device=device, dtype=dtype)
    
    if isinstance(C, np.ndarray):
        C = torch.from_numpy(C).to(device=device, dtype=dtype)
    elif isinstance(C, torch.Tensor):
        C = C.to(device=device, dtype=dtype)
    elif not sparse_mtx:
        C = torch.tensor(C).to(device=device, dtype=dtype)
    
    # Return a zero matrix if either a or b is all zero
    nzind_a = torch.where(a > 0)[0]
    nzind_b = torch.where(b > 0)[0]
    if len(nzind_a) == 0 or len(nzind_b) == 0:
        if sparse_mtx:
            # Convert back to scipy sparse for consistency
            P = sparse.coo_matrix(([], ([], [])), shape=(len(a), len(b)))
        else:
            P = torch.zeros([len(a), len(b)], dtype=dtype, device=device)
        return P
        
    if solver == "sinkhorn" and max(abs(eps_p-eps_mu), abs(eps_p-eps_nu)) > 1e-8:
        print("To use Sinkhorn algorithm, set eps_p=eps_mu=eps_nu")
        return None
        
    if solver == "sinkhorn" and not sparse_mtx:
        P = unot_sinkhorn_l1_dense_torch(a, b, C, eps_p, rho, 
                                         nitermax=nitermax, stopthr=stopthr, 
                                         verbose=verbose, show_progress=show_progress, device=device)
    elif solver == "sinkhorn" and sparse_mtx:
        P = unot_sinkhorn_l1_sparse_torch(a, b, C, eps_p, rho, 
                                          nitermax=nitermax, stopthr=stopthr, 
                                          verbose=verbose, show_progress=show_progress, device=device)
    elif solver == "momentum" and not sparse_mtx: 
        P = unot_momentum_l1_dense_torch(a, b, C, eps_p, eps_mu, eps_nu, rho, 
                                         nitermax=nitermax, stopthr=stopthr, dt=momentum_dt, 
                                         beta=momentum_beta, precondition=True, 
                                         verbose=verbose, show_progress=show_progress, device=device)
    elif solver == "momentum" and sparse_mtx:
        print("Sparse momentum solver under construction")
        return None
    
    return P


def wrightomega_torch(z, device='cpu'):
    """PyTorch implementation of Wright Omega function using scipy fallback"""
    # Convert to numpy for scipy computation, then back to torch
    if isinstance(z, torch.Tensor):
        z_np = z.detach().cpu().numpy()
    else:
        z_np = z
    
    result_np = wrightomega(z_np).real
    return torch.from_numpy(result_np).to(device=device, dtype=_solver_dtype(device))


def unot_sinkhorn_l1_dense_torch(a, b, C, eps, m, nitermax=10000, stopthr=1e-8, 
                                 verbose=False, output_fg=False, show_progress=True, device='cpu'):
    """ Solve the unnormalized optimal transport with l1 penalty using PyTorch.

    Parameters
    ----------
    a : torch.Tensor
        Source distribution. The summation should be less than or equal to 1.
    b : torch.Tensor
        Target distribution. The summation should be less than or equal to 1.
    C : torch.Tensor
        The cost matrix possibly with infinity entries.
    eps : float
        The coefficient for entropy regularization.
    m : float
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The max number of iterations. Defaults to 10000.
    stopthr : float, optional
        The threshold for terminating the iteration. Defaults to 1e-8.
    verbose : bool, optional
        Whether to print verbose output.
    show_progress : bool, optional
        Whether to show progress bar.
    device : str
        Device to run computation on.

    Returns
    -------
    torch.Tensor
        The optimal transport matrix.
    """
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    niter = 0
    err = 100
    
    # Setup progress bar
    pbar = None
    if show_progress:
        pbar = tqdm(total=nitermax, desc="Sinkhorn L1 Dense", 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] err={postfix}')
    
    try:
        while niter <= nitermax and err > stopthr:
            fprev = f.clone()
            gprev = g.clone()
            
            # Iteration
            # f update
            exp_term = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C) / eps)
            sum_exp = torch.sum(exp_term, dim=1)
            exp_f_term = torch.exp((-m + f) / eps)
            
            f = eps * torch.log(a) - eps * torch.log(sum_exp + exp_f_term) + f
            
            # g update
            exp_term = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C) / eps)
            sum_exp = torch.sum(exp_term, dim=0)
            exp_g_term = torch.exp((-m + g) / eps)
            
            g = eps * torch.log(b) - eps * torch.log(sum_exp + exp_g_term) + g
            
            # Check relative error
            if niter % 10 == 0:
                err_f = torch.abs(f - fprev).max() / max(torch.abs(f).max(), torch.abs(fprev).max(), 1.)
                err_g = torch.abs(g - gprev).max() / max(torch.abs(g).max(), torch.abs(gprev).max(), 1.)
                err = 0.5 * (err_f + err_g)
                
                # Update progress bar
                if pbar is not None:
                    pbar.set_postfix_str(f"{err:.2e}")
                    
            niter = niter + 1
            
            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                
    finally:
        if pbar is not None:
            pbar.close()
    
    if verbose:
        print('Number of iterations in unot:', niter)
    
    P = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C) / eps)
    
    if output_fg:
        return f, g
    else:
        return P


def unot_sinkhorn_l1_sparse_torch(a, b, C, eps, m, nitermax=10000, stopthr=1e-8, 
                                  verbose=False, show_progress=True, device='cpu'):
    """ Solve the unnormalized optimal transport with l1 penalty using PyTorch for sparse matrices.

    Parameters
    ----------
    a : torch.Tensor
        Source distribution. The summation should be less than or equal to 1.
    b : torch.Tensor
        Target distribution. The summation should be less than or equal to 1.
    C : scipy.sparse.coo_matrix
        The cost matrix in coo sparse format.
    eps : float
        The coefficient for entropy regularization.
    m : float
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The max number of iterations. Defaults to 10000.
    stopthr : float, optional
        The threshold for terminating the iteration. Defaults to 1e-8.
    verbose : bool, optional
        Whether to print verbose output.
    show_progress : bool, optional
        Whether to show progress bar.
    device : str
        Device to run computation on.

    Returns
    -------
    scipy.sparse.coo_matrix
        The optimal transport matrix.
    """
    if str(device) == 'cpu':
        from . import _sinkhorn_numba
        if _sinkhorn_numba.NUMBA:
            # Fused, row/column-parallel CPU kernel; same updates in float64.
            return _sinkhorn_numba.sinkhorn_l1_sparse(
                a.detach().cpu().numpy(), b.detach().cpu().numpy(), C, eps, m,
                nitermax=nitermax, stopthr=stopthr, verbose=verbose)

    # Convert sparse matrix indices and data to torch tensors
    row_idx = torch.from_numpy(C.row).long().to(device)
    col_idx = torch.from_numpy(C.col).long().to(device)
    C_data = torch.from_numpy(C.data).to(device=device, dtype=a.dtype)
    
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    niter = 0
    err = 100
    
    # Setup progress bar
    pbar = None
    if show_progress:
        pbar = tqdm(total=nitermax, desc="Sinkhorn L1 Sparse", 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] err={postfix}')
    
    try:
        while niter <= nitermax and err > stopthr:
            fprev = f.clone()
            gprev = g.clone()
            
            # Compute sparse exponential terms
            exp_data = torch.exp((-C_data + f[row_idx] + g[col_idx]) / eps)
            
            # f update
            sum_f = torch.zeros_like(a)
            sum_f.scatter_add_(0, row_idx, exp_data)
            exp_f_term = torch.exp((-m + f) / eps)
            f = eps * torch.log(a) - eps * torch.log(sum_f + exp_f_term) + f
            
            # Recompute exp_data for g update
            exp_data = torch.exp((-C_data + f[row_idx] + g[col_idx]) / eps)
            
            # g update
            sum_g = torch.zeros_like(b)
            sum_g.scatter_add_(0, col_idx, exp_data)
            exp_g_term = torch.exp((-m + g) / eps)
            g = eps * torch.log(b) - eps * torch.log(sum_g + exp_g_term) + g
            
            # Check relative error
            if niter % 10 == 0:
                err_f = torch.abs(f - fprev).max() / max(torch.abs(f).max(), torch.abs(fprev).max(), 1.)
                err_g = torch.abs(g - gprev).max() / max(torch.abs(g).max(), torch.abs(gprev).max(), 1.)
                err = 0.5 * (err_f + err_g)
                
                # Update progress bar
                if pbar is not None:
                    pbar.set_postfix_str(f"{err:.2e}")
                    
            niter = niter + 1
            
            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                
    finally:
        if pbar is not None:
            pbar.close()

    if verbose:
        print('Number of iterations in unot:', niter)
    
    # Create final result as sparse matrix
    final_exp_data = torch.exp((-C_data + f[row_idx] + g[col_idx]) / eps)
    
    # Convert back to scipy sparse matrix
    result_data = final_exp_data.detach().cpu().numpy()
    result_row = row_idx.detach().cpu().numpy()
    result_col = col_idx.detach().cpu().numpy()
    
    result_sparse = sparse.coo_matrix((result_data, (result_row, result_col)), 
                                      shape=(len(a), len(b)))
    
    return result_sparse


def unot_momentum_l1_dense_torch(a, b, C, eps_p, eps_mu, eps_nu, m, nitermax=1e4, 
                                 stopthr=1e-8, dt=0.01, beta=0.8, precondition=False, 
                                 verbose=False, show_progress=True, device='cpu'):
    """ Solve unnormalized optimal transport using momentum method with PyTorch.

    Parameters
    ----------
    a : torch.Tensor
        Source distribution.
    b : torch.Tensor
        Target distribution.
    C : torch.Tensor
        Cost matrix.
    eps_p : float
        Entropy regularization coefficient for P.
    eps_mu : float
        Entropy regularization coefficient for mu.
    eps_nu : float
        Entropy regularization coefficient for nu.
    m : float
        Penalty coefficient for unmatched mass.
    nitermax : float, optional
        Maximum number of iterations.
    stopthr : float, optional
        Stopping threshold.
    dt : float, optional
        Step size.
    beta : float, optional
        Momentum coefficient.
    precondition : bool, optional
        Whether to use preconditioning.
    verbose : bool, optional
        Whether to print verbose output.
    show_progress : bool, optional
        Whether to show progress bar.
    device : str
        Device to run computation on.

    Returns
    -------
    torch.Tensor
        The optimal transport matrix.
    """
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    
    if precondition:
        f, g = unot_sinkhorn_l1_dense_torch(a, b, C, eps_p, m, output_fg=True, 
                                           show_progress=show_progress, device=device)
    
    f_old = f.clone()
    g_old = g.clone()
    F_old = f.clone()
    G_old = g.clone()
    niter = 0
    err = 100
    
    def Qf(ff, gg, ee_p, ee_mu, ee_nu, mm, aa, bb, CC):
        exp_term = torch.exp(ff / ee_p) * torch.sum(torch.exp((gg.unsqueeze(0) - CC) / ee_p), dim=1)
        penalty_term = torch.exp((ff - mm) / ee_mu)
        return exp_term + penalty_term - aa
    
    def Qg(ff, gg, ee_p, ee_mu, ee_nu, mm, aa, bb, CC):
        exp_term = torch.exp(gg / ee_p) * torch.sum(torch.exp((ff.unsqueeze(1) - CC) / ee_p), dim=0)
        penalty_term = torch.exp((gg - mm) / ee_nu)
        return exp_term + penalty_term - bb
    
    nitermax = int(nitermax)
    
    # Setup progress bar
    pbar = None
    if show_progress:
        pbar = tqdm(total=nitermax, desc="Momentum L1 Dense", 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] err={postfix}')
    
    try:
        while niter <= nitermax and err > stopthr:
            F = beta * F_old + Qf(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
            f = f_old - dt * F
            G = beta * G_old + Qg(f_old, g_old, eps_p, eps_mu, eps_nu, m, a, b, C)
            g = g_old - dt * G
            
            if niter % 10 == 0:
                err_f = torch.abs(f - f_old).max() / max(torch.abs(f).max(), torch.abs(f_old).max(), 1.)
                err_g = torch.abs(g - g_old).max() / max(torch.abs(g).max(), torch.abs(g_old).max(), 1.)
                err = 0.5 * (err_f + err_g)
                
                # Update progress bar
                if pbar is not None:
                    pbar.set_postfix_str(f"{err:.2e}")
                    
            f_old = f.clone()
            F_old = F.clone()
            g_old = g.clone()
            G_old = G.clone()
            niter += 1
            
            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                
    finally:
        if pbar is not None:
            pbar.close()
    
    P = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C) / eps_p)
    
    if verbose:
        print(f'Number of iterations: {niter}')
    
    return P


def unot_sinkhorn_l2_dense_torch(a, b, C, eps, m, nitermax=10000, stopthr=1e-8, 
                                 verbose=False, show_progress=True, device='cpu'):
    """ Solve the unnormalized optimal transport with l2 penalty using PyTorch.

    Parameters
    ----------
    a : torch.Tensor
        Source distribution.
    b : torch.Tensor
        Target distribution.
    C : torch.Tensor
        The cost matrix.
    eps : float
        The coefficient for entropy regularization.
    m : float
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The max number of iterations.
    stopthr : float, optional
        The threshold for terminating the iteration.
    verbose : bool, optional
        Whether to print verbose output.
    show_progress : bool, optional
        Whether to show progress bar.
    device : str
        Device to run computation on.

    Returns
    -------
    torch.Tensor
        The optimal transport matrix.
    """
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    r = torch.zeros_like(a)
    s = torch.zeros_like(b)
    niter = 0
    err = 100
    
    # Setup progress bar
    pbar = None
    if show_progress:
        pbar = tqdm(total=nitermax, desc="Sinkhorn L2 Dense", 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] err={postfix}')
    
    try:
        while niter <= nitermax and err > stopthr:
            fprev = f.clone()
            gprev = g.clone()
            
            # Iteration
            exp_term = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C) / eps)
            sum_exp_f = torch.sum(exp_term, dim=1)
            exp_r_term = torch.exp((r + f) / eps)
            
            f = eps * torch.log(a) - eps * torch.log(sum_exp_f + exp_r_term) + f
            
            exp_term = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C) / eps)
            sum_exp_g = torch.sum(exp_term, dim=0)
            exp_s_term = torch.exp((s + g) / eps)
            
            g = eps * torch.log(b) - eps * torch.log(sum_exp_g + exp_s_term) + g
            
            # Update r and s using wright omega (fallback to cpu for scipy)
            r = -eps * wrightomega_torch(f / eps - torch.log(torch.tensor(eps / m)), device)
            s = -eps * wrightomega_torch(g / eps - torch.log(torch.tensor(eps / m)), device)
            
            # Check relative error
            if niter % 10 == 0:
                err_f = torch.abs(f - fprev).max() / max(torch.abs(f).max(), torch.abs(fprev).max(), 1.)
                err_g = torch.abs(g - gprev).max() / max(torch.abs(g).max(), torch.abs(gprev).max(), 1.)
                err = 0.5 * (err_f + err_g)
                
                # Update progress bar
                if pbar is not None:
                    pbar.set_postfix_str(f"{err:.2e}")
                    
            niter = niter + 1
            
            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                
    finally:
        if pbar is not None:
            pbar.close()
    
    if verbose:
        print('Number of iterations in unot:', niter)
    
    P = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C) / eps)
    return P


# Utility function to convert between numpy and torch
def to_numpy(tensor):
    """Convert torch tensor to numpy array"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def to_torch(array, device='cpu'):
    """Convert numpy array to torch tensor"""
    dtype = _solver_dtype(device)
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to(device=device, dtype=dtype)
    elif isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)
    else:
        return torch.tensor(array).to(device=device, dtype=dtype) 