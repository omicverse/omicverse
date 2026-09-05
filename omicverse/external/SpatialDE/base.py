''' Main underlying functions for SpatialDE functionality.

Optimizations over the original:
  - null_fits / const_fits fully vectorized (no Python gene loops)
  - lengthscale_fits: batch UTy = exp_tab.T @ U, then joblib-parallel lbfgsb
  - dyn_de / run accept n_jobs kwarg (default 1 = sequential)
'''
import sys
import logging
from time import time
import warnings

import numpy as np
from scipy import optimize
from scipy import linalg
from scipy import stats
from .._finite_difference import derivative
from scipy.special import logsumexp

import pandas as pd

from .util import qvalue

try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


def _print_progress(current, total, desc='Genes'):
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = '█' * filled + '░' * (bar_len - filled)
    print(f'\r  {desc} [{bar}] {current}/{total}', end='', flush=True)
    if current == total:
        print()


def _progress_iter(iterable, total, desc, use_tqdm=True, leave=False):
    if use_tqdm and _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, leave=leave)
    return iterable


def get_l_limits(X):
    X = np.asarray(X)
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 0, np.inf)
    R_vals = np.unique(R2.flatten())
    R_vals = R_vals[R_vals > 1e-8]

    l_min = np.sqrt(R_vals.min()) / 2.
    l_max = np.sqrt(R_vals.max()) * 2.

    return l_min, l_max

## Kernels ##

def _pairwise_sqdist(X, Y=None):
    X = np.asarray(X)
    if Y is None:
        Xsq = np.sum(np.square(X), 1)
        R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    else:
        Y = np.asarray(Y)
        Xsq = np.sum(np.square(X), 1)[:, None]
        Ysq = np.sum(np.square(Y), 1)[None, :]
        R2 = -2. * np.dot(X, Y.T) + Xsq + Ysq
    return np.clip(R2, 1e-12, np.inf)


def SE_kernel(X, l):
    R2 = _pairwise_sqdist(X)
    return np.exp(-R2 / (2 * l**2))


def SE_kernel_cross(X, Y, l):
    R2 = _pairwise_sqdist(X, Y)
    return np.exp(-R2 / (2 * l**2))


def linear_kernel(X):
    K = np.dot(X, X.T)
    return K / K.max()


def cosine_kernel(X, p):
    R2 = _pairwise_sqdist(X)
    return np.cos(2 * np.pi * np.sqrt(R2) / p)


def cosine_kernel_cross(X, Y, p):
    R2 = _pairwise_sqdist(X, Y)
    return np.cos(2 * np.pi * np.sqrt(R2) / p)


def gower_scaling_factor(K):
    # tr(PKP) = tr(K) - sum(K)/n  where P = I - 11^T/n
    # Avoids allocating two O(n²) temporary matrices.
    n = K.shape[0]
    trPKP = np.trace(K) - K.sum() / n
    return trPKP / (n - 1)


def gower_scaling_factor_lowrank(B):
    """Gower factor for K ≈ B B^T without materializing full K."""
    n = B.shape[0]
    trK = np.square(B).sum()
    oneKBone = np.square(B.sum(0)).sum()
    trPKP = trK - oneKBone / n
    return trPKP / (n - 1)


def factor(K):
    S, U = np.linalg.eigh(K)
    return U, np.clip(S, 1e-8, None)


def factor_nystrom(X, kernel_cross_fn, rank=256, seed=0, jitter=1e-8):
    """Nyström low-rank eigendecomposition approximation for PSD kernels."""
    X = np.asarray(X)
    n = X.shape[0]
    m = min(int(rank), n)
    if m >= n:
        K = kernel_cross_fn(X, X)
        U, S = factor(K)
        return U, S, gower_scaling_factor(K)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    Xl = X[idx]

    C = kernel_cross_fn(X, Xl)      # n x m
    W = kernel_cross_fn(Xl, Xl)     # m x m
    W = 0.5 * (W + W.T) + jitter * np.eye(m)

    ew, Uw = np.linalg.eigh(W)
    ew = np.clip(ew, 1e-10, None)

    B = C.dot(Uw / np.sqrt(ew))     # n x m
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    S = np.clip(np.square(s), 1e-8, None)
    return U, S, gower_scaling_factor_lowrank(B)


def get_UT1(U):
    return U.sum(0)


def get_UTy(U, y):
    return y.dot(U)


def mu_hat(delta, UTy, UT1, S, n, Yvar=None, y_sum=None):
    if Yvar is None:
        Yvar = np.ones_like(S)
    denom = (S + delta * Yvar)
    UT1_scaled = UT1 / denom
    num = UT1_scaled.dot(UTy)
    den = UT1_scaled.dot(UT1)

    # Low-rank correction for omitted eigenspace (S=0 there).
    if y_sum is not None and UTy.shape[0] < n:
        num += (y_sum - UT1.dot(UTy)) / delta
        den += (n - UT1.dot(UT1)) / delta
    return num / den


def s2_t_hat(delta, UTy, S, n, Yvar=None, y_sq=None):
    if Yvar is None:
        Yvar = np.ones_like(S)
    UTy_scaled = UTy / (S + delta * Yvar)
    val = UTy_scaled.dot(UTy)

    # Low-rank correction for omitted eigenspace (S=0 there).
    if y_sq is not None and UTy.shape[0] < n:
        val += max(y_sq - UTy.dot(UTy), 0.0) / delta
    return val / n


def LL(delta, UTy, UT1, S, n, Yvar=None, y_sum=None, y_sq=None):
    mu_h = mu_hat(delta, UTy, UT1, S, n, Yvar, y_sum=y_sum)
    if Yvar is None:
        Yvar = np.ones_like(S)
    denom = (S + delta * Yvar)
    centered = UTy - UT1 * mu_h
    sum_1 = (np.square(centered) / denom).sum()
    sum_2 = np.log(denom).sum()

    # Low-rank correction for omitted eigenspace (S=0 there).
    if y_sum is not None and y_sq is not None and UTy.shape[0] < n:
        residual_total = y_sq - 2 * mu_h * y_sum + (mu_h ** 2) * n
        residual_captured = np.square(centered).sum()
        residual_perp = max(residual_total - residual_captured, 0.0)
        sum_1 += residual_perp / delta
        sum_2 += (n - UTy.shape[0]) * np.log(delta)

    with np.errstate(divide='ignore'):
        return -0.5 * (n * np.log(2 * np.pi) + n * np.log(sum_1 / n) + sum_2 + n)


def logdelta_prior_lpdf(log_delta):
    s2p = 100.
    return -np.log(np.sqrt(2 * np.pi * s2p)) - np.square(log_delta - 20.) / (2 * s2p)


def make_objective(UTy, UT1, S, n, Yvar=None, y_sum=None, y_sq=None):
    def LL_obj(log_delta):
        return -LL(np.exp(log_delta), UTy, UT1, S, n, Yvar, y_sum=y_sum, y_sq=y_sq)
    return LL_obj


def brent_max_LL(UTy, UT1, S, n, y_sum=None, y_sq=None):
    LL_obj = make_objective(UTy, UT1, S, n, y_sum=y_sum, y_sq=y_sq)
    o = optimize.minimize_scalar(LL_obj, bounds=[-10, 10], method='bounded',
                                 options={'maxiter': 32})
    max_ll = -o.fun
    max_delta = np.exp(o.x)
    return (
        max_ll,
        max_delta,
        mu_hat(max_delta, UTy, UT1, S, n, y_sum=y_sum),
        s2_t_hat(max_delta, UTy, S, n, y_sq=y_sq),
    )


def lbfgsb_max_LL(UTy, UT1, S, n, Yvar=None, y_sum=None, y_sq=None):
    LL_obj = make_objective(UTy, UT1, S, n, Yvar, y_sum=y_sum, y_sq=y_sq)
    min_boundary = -10
    max_boundary = 20.
    x, f, d = optimize.fmin_l_bfgs_b(LL_obj, 0., approx_grad=True,
                                      bounds=[(min_boundary, max_boundary)],
                                      maxfun=64, factr=1e12, epsilon=1e-4)
    max_ll = -f
    max_delta = np.exp(x[0])

    for bd in (max_boundary, min_boundary):
        bll = -LL_obj(bd)
        if bll > max_ll:
            max_ll = bll
            max_delta = np.exp(bd)

    max_mu_hat = mu_hat(max_delta, UTy, UT1, S, n, Yvar, y_sum=y_sum)
    max_s2_t_hat = s2_t_hat(max_delta, UTy, S, n, Yvar, y_sq=y_sq)
    s2_logdelta = 1. / (derivative(LL_obj, np.log(max_delta), n=2) ** 2)

    return max_ll, max_delta, max_mu_hat, max_s2_t_hat, s2_logdelta


def search_max_LL(UTy, UT1, S, n, num=32, y_sum=None, y_sq=None):
    min_obj = np.inf
    max_log_delta = np.nan
    LL_obj = make_objective(UTy, UT1, S, n, y_sum=y_sum, y_sq=y_sq)
    for log_delta in np.linspace(start=-10, stop=20, num=num):
        cur_obj = LL_obj(log_delta)
        if cur_obj < min_obj:
            min_obj = cur_obj
            max_log_delta = log_delta
    max_delta = np.exp(max_log_delta)
    return (-min_obj, max_delta,
            mu_hat(max_delta, UTy, UT1, S, n, y_sum=y_sum),
            s2_t_hat(max_delta, UTy, S, n, y_sq=y_sq))


def make_FSV(UTy, S, n, Gower, y_sq=None):
    def FSV(log_delta):
        s2_t = s2_t_hat(np.exp(log_delta), UTy, S, n, y_sq=y_sq)
        s2_t_g = s2_t * Gower
        return s2_t_g / (s2_t_g + np.exp(log_delta) * s2_t)
    return FSV


def _fit_one_gene(g, gene_name, UTy, UT1, S, n, Gower, y_sum=None, y_sq=None):
    """Fit a single gene's GP model; used by both serial and parallel paths."""
    t0 = time()
    max_reg_ll, max_delta, max_mu_hat, max_s2_t_hat, s2_logdelta = lbfgsb_max_LL(
        UTy, UT1, S, n, y_sum=y_sum, y_sq=y_sq
    )
    t = time() - t0

    FSV = make_FSV(UTy, S, n, Gower, y_sq=y_sq)
    s2_FSV = derivative(FSV, np.log(max_delta), n=1) ** 2 * s2_logdelta

    return {
        'g': gene_name,
        'max_ll': max_reg_ll,
        'max_delta': max_delta,
        'max_mu_hat': max_mu_hat,
        'max_s2_t_hat': max_s2_t_hat,
        'time': t,
        'n': n,
        'FSV': FSV(np.log(max_delta)),
        's2_FSV': s2_FSV,
        's2_logdelta': s2_logdelta,
    }


def lengthscale_fits(exp_tab, U, UT1, S, Gower, n_jobs=1, num=64, use_tqdm=True):
    '''Fit GPs after pre-processing for a particular lengthscale.

    Optimizations vs original:
      - Batch UTy computation: one matrix multiply instead of G dot products
      - n_jobs > 1: joblib thread-pool over genes (lbfgsb releases the GIL)
    '''
    n, G = exp_tab.shape
    vals = exp_tab.values  # avoid repeated pandas overhead
    is_low_rank = U.shape[1] < n

    # Batch: compute UTy for ALL genes at once → G × n
    UTY = vals.T.dot(U)   # equivalent to: [y_g.dot(U) for each gene]
    if is_low_rank:
        Y_sum = vals.sum(0)
        Y_sq = np.square(vals).sum(0)
    else:
        Y_sum = None
        Y_sq = None

    gene_names = list(exp_tab.columns)

    if n_jobs == 1:
        results = []
        for g in _progress_iter(range(G), total=G, desc='Genes', use_tqdm=use_tqdm, leave=False):
            results.append(
                _fit_one_gene(
                    g,
                    gene_names[g],
                    UTY[g],
                    UT1,
                    S,
                    n,
                    Gower,
                    y_sum=None if Y_sum is None else Y_sum[g],
                    y_sq=None if Y_sq is None else Y_sq[g],
                )
            )
            if not use_tqdm:
                _print_progress(g + 1, G)
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_fit_one_gene)(
                g,
                gene_names[g],
                UTY[g],
                UT1,
                S,
                n,
                Gower,
                y_sum=None if Y_sum is None else Y_sum[g],
                y_sq=None if Y_sq is None else Y_sq[g],
            )
            for g in _progress_iter(range(G), total=G, desc='Genes', use_tqdm=use_tqdm, leave=False)
        )
        if not use_tqdm:
            _print_progress(G, G)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Vectorized baseline models (no Python gene loop)
# ---------------------------------------------------------------------------

def null_fits(exp_tab):
    '''Get maximum LL for null model (vectorized over all genes).'''
    n, G = exp_tab.shape
    vals = exp_tab.values
    max_s2_e = np.square(vals).sum(0) / n          # shape (G,)
    max_ll = -0.5 * (n * np.log(2 * np.pi) + n + n * np.log(max_s2_e))
    return pd.DataFrame({
        'g': exp_tab.columns,
        'max_ll': max_ll,
        'max_delta': np.inf,
        'max_mu_hat': 0.,
        'max_s2_t_hat': 0.,
        'time': 0,
        'n': n,
    })


def const_fits(exp_tab):
    '''Get maximum LL for const model (vectorized over all genes).'''
    n, G = exp_tab.shape
    vals = exp_tab.values
    means = vals.mean(0)                            # shape (G,)
    vars_ = vals.var(0)
    vars_ = np.where(vars_ == 0, 1e-10, vars_)     # guard against zero variance
    sum1 = np.square(vals - means).sum(0)
    max_ll = -0.5 * (n * np.log(vars_) + sum1 / vars_ + n * np.log(2 * np.pi))
    return pd.DataFrame({
        'g': exp_tab.columns,
        'max_ll': max_ll,
        'max_delta': np.inf,
        'max_mu_hat': means,
        'max_s2_t_hat': 0.,
        'time': 0,
        'n': n,
    })


def simulate_const_model(MLL_params, N):
    dfm = np.zeros((N, MLL_params.shape[0]))
    for i, params in enumerate(MLL_params.iterrows()):
        params = params[1]
        s2_e = params.max_s2_t_hat * params.max_delta
        dfm[:, i] = np.random.normal(params.max_mu_hat, s2_e, N)
    return pd.DataFrame(dfm)


def get_mll_results(results, null_model='const'):
    null_lls = results.query('model == "{}"'.format(null_model))[['g', 'max_ll']]
    model_results = results.query('model != "{}"'.format(null_model))
    model_results = model_results[
        model_results.groupby(['g'])['max_ll'].transform(max) == model_results['max_ll']
    ]
    mll_results = model_results.merge(null_lls, on='g', suffixes=('', '_null'))
    mll_results['LLR'] = mll_results['max_ll'] - mll_results['max_ll_null']
    return mll_results


def dyn_de(X, exp_tab, kernel_space=None, n_jobs=1,
           approx_rank=None, approx_seed=0, approx_models=('SE',), use_tqdm=True):
    if kernel_space is None:
        kernel_space = {'SE': [5., 25., 50.]}

    results = []

    if 'null' in kernel_space:
        result = null_fits(exp_tab)
        result['l'] = np.nan
        result['M'] = 1
        result['model'] = 'null'
        results.append(result)

    if 'const' in kernel_space:
        result = const_fits(exp_tab)
        result['l'] = np.nan
        result['M'] = 2
        result['model'] = 'const'
        results.append(result)

    logging.info("Pre-calculating USU^T = K's ...")
    US_mats = []
    t0 = time()
    X = np.asarray(X)
    n = X.shape[0]
    approx_model_set = {str(m).upper() for m in approx_models}

    def _should_approx(model_name):
        if approx_rank is None:
            return False
        try:
            rank = int(approx_rank)
        except Exception:
            return False
        if rank <= 0 or rank >= n:
            return False
        return str(model_name).upper() in approx_model_set

    if 'linear' in kernel_space:
        K = linear_kernel(X)
        U, S = factor(K)
        US_mats.append({'model': 'linear', 'M': 3, 'l': np.nan,
                        'U': U, 'S': S, 'UT1': get_UT1(U), 'Gower': gower_scaling_factor(K)})

    if 'SE' in kernel_space:
        for i, lengthscale in enumerate(kernel_space['SE']):
            if _should_approx('SE'):
                U, S, gower = factor_nystrom(
                    X,
                    lambda A, B: SE_kernel_cross(A, B, lengthscale),
                    rank=int(approx_rank),
                    seed=int(approx_seed) + i,
                )
            else:
                K = SE_kernel(X, lengthscale)
                U, S = factor(K)
                gower = gower_scaling_factor(K)

            US_mats.append({'model': 'SE', 'M': 4, 'l': lengthscale,
                            'U': U, 'S': S, 'UT1': get_UT1(U), 'Gower': gower})

    if 'PER' in kernel_space:
        se_offset = len(kernel_space.get('SE', []))
        for i, period in enumerate(kernel_space['PER']):
            if _should_approx('PER'):
                U, S, gower = factor_nystrom(
                    X,
                    lambda A, B: cosine_kernel_cross(A, B, period),
                    rank=int(approx_rank),
                    seed=int(approx_seed) + se_offset + i,
                )
            else:
                K = cosine_kernel(X, period)
                U, S = factor(K)
                gower = gower_scaling_factor(K)

            US_mats.append({'model': 'PER', 'M': 4, 'l': period,
                            'U': U, 'S': S, 'UT1': get_UT1(U), 'Gower': gower})

    logging.info('Done: {0:.2f}s'.format(time() - t0))
    logging.info('Fitting gene models')

    for cov in _progress_iter(US_mats, total=len(US_mats), desc='Models', use_tqdm=use_tqdm, leave=False):
        result = lengthscale_fits(exp_tab, cov['U'], cov['UT1'], cov['S'], cov['Gower'],
                                  n_jobs=n_jobs, use_tqdm=use_tqdm)
        result['l'] = cov['l']
        result['M'] = cov['M']
        result['model'] = cov['model']
        results.append(result)

    logging.info('Finished fitting {} models to {} genes'.format(len(US_mats), exp_tab.shape[1]))

    results = pd.concat(results, sort=True).reset_index(drop=True)
    results['BIC'] = -2 * results['max_ll'] + results['M'] * np.log(results['n'])
    return results


def run(X, exp_tab, kernel_space=None, n_jobs=1,
        approx_rank=None, approx_seed=0, approx_models=('SE',), use_tqdm=True):
    '''Perform SpatialDE test.

    Parameters
    ----------
    X : array-like, shape (n, 2)
        Spatial coordinates.
    exp_tab : DataFrame, shape (n, G)
        Normalised expression (cells × genes).
    kernel_space : dict, optional
        Covariance matrices to search. Default: 10 SE lengthscales + const.
    n_jobs : int, default 1
        Number of parallel threads for gene-level GP fitting.
        -1 = use all available cores.
    approx_rank : int or None, default None
        If set (e.g. 128/256), use Nyström low-rank approximation for selected kernels.
        Smaller rank is faster with larger approximation error.
    approx_seed : int, default 0
        Random seed for Nyström landmark sampling.
    approx_models : tuple[str], default ('SE',)
        Which kernels use Nyström approximation. Example: ('SE', 'PER').
    use_tqdm : bool, default True
        Show tqdm progress bars for models and genes.
    '''
    X = np.asarray(X)
    if kernel_space is None:
        l_min, l_max = get_l_limits(X)
        kernel_space = {
            'SE': np.logspace(np.log10(l_min), np.log10(l_max), 10),
            'const': 0,
        }

    logging.info('Performing DE test')
    results = dyn_de(
        X,
        exp_tab,
        kernel_space,
        n_jobs=n_jobs,
        approx_rank=approx_rank,
        approx_seed=approx_seed,
        approx_models=approx_models,
        use_tqdm=use_tqdm,
    )
    mll_results = get_mll_results(results)

    mll_results['pval'] = 1 - stats.chi2.cdf(mll_results['LLR'], df=1)
    mll_results['qval'] = qvalue(mll_results['pval'])

    return mll_results


def model_search(X, exp_tab, DE_mll_results, kernel_space=None, n_jobs=1):
    if kernel_space is None:
        P_min, P_max = get_l_limits(X)
        kernel_space = {
            'PER': np.logspace(np.log10(P_min), np.log10(P_max), 10),
            'linear': 0,
        }

    de_exp_tab = exp_tab[DE_mll_results['g']]
    logging.info('Performing model search')
    results = dyn_de(X, de_exp_tab, kernel_space, n_jobs=n_jobs)
    new_and_old_results = pd.concat((results, DE_mll_results), sort=True)

    mask = (new_and_old_results.groupby(['g', 'model'])['BIC'].transform(min)
            == new_and_old_results['BIC'])
    log_p_data_Hi = -new_and_old_results[mask].pivot_table(
        values='BIC', index='g', columns='model')
    log_Z = logsumexp(log_p_data_Hi, 1)
    log_p_Hi_data = (log_p_data_Hi.T - log_Z).T
    p_Hi_data = np.exp(log_p_Hi_data).add_suffix('_prob')

    mask = new_and_old_results.groupby('g')['BIC'].transform(min) == new_and_old_results['BIC']
    ms_results = new_and_old_results[mask]
    ms_results = ms_results.join(p_Hi_data, on='g')

    transfer_columns = ['pval', 'qval', 'max_ll_null']
    ms_results = (ms_results.drop(transfer_columns, 1)
                  .merge(DE_mll_results[transfer_columns + ['g']], on='g'))

    return ms_results
