import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import interpolate

import logging
from time import time
import warnings
from scipy import optimize
from scipy import linalg
from scipy import stats
from .._finite_difference import derivative
from scipy.special import logsumexp

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


def plotgene(X,mtx,draw_list,result,sp=10,lw=0.2,N=5,plotsize=5):
    n = len(draw_list)
    rownum = n//N + 1
    plt.figure(figsize=(N*(plotsize+2),plotsize*rownum))
    cmap = LinearSegmentedColormap.from_list('mycmap', ['blue','white','red'])
    for i in range(n):
        if draw_list[i] in list(mtx.T):
            plt.subplot(rownum,N,i+1)
            plt.scatter(X[:,0], X[:,1], c=mtx.T[draw_list[i]],cmap=cmap,s=sp,linewidths=lw,edgecolors='black')
            plt.colorbar()
            if hasattr(result, 'g'):
                plt.title(draw_list[i]+' qval:'+str(round(result[result.g==draw_list[i]].qval.values[0],2)))
            else:
                plt.title(draw_list[i])
        else:
            print('not contain '+str(draw_list[i]))
        
def draw_agree(intersection,r1,r2,verbose=False,N=5):
    r1br2=[]
    r2br1=[]
    all_g=[]
    m1=0
    m2=0
    for i in intersection:
        x1 = r1.index(i)
        x2 = r2.index(i)
        if (abs(x1-x2)>100)&verbose:
            continue
        plt.scatter(x1,x2)
        m1 = max(m1,x1)
        m2 = max(m2,x2)
        if (x1-x2)>N:
            r2br1.append(i)
        elif (x2-x1)>N:
            r1br2.append(i)
        else:
            all_g.append(i)
        plt.annotate("(%s,%s) " %(x1,x2)+str(i), xy=(x1,x2), xytext=(-20, 10), textcoords='offset points')
    plt.plot([0,m2],[N,m2+N],linestyle='-.',color='r')
    plt.plot([N,m2+N],[0,m2],linestyle='-.',color='r')
    plt.plot([0,m2],[10,m2+10],linestyle='-.',color='b')
    plt.plot([10,m2+10],[0,m2],linestyle='-.',color='b')
    plt.xlabel('original')
    plt.xlim(0, m1+10)
    plt.ylim(0, m2+10)
    plt.ylabel('SOM')
    plt.title('Rank 50'+' left_top:'+str(len(r1br2))+' right_down:'+str(len(r2br1))+' all:'+str(len(intersection)))
    return r1br2,r2br1,all_g
    
def draw_agree_log(intersection,r1,r2,label,verbose=False,N=5,al=1000):
    r1br2=[]
    r2br1=[]
    all_g=[]
    m1=0
    m2=0
    x_list=[]
    y_list=[]
    diff=[]
    plt.yscale('log')
    plt.xscale('log')
    plt.axis([1, al, 1, al])
    for i in intersection:
        x1 = r1.index(i)+1
        x2 = r2.index(i)+1
        x_list.append(x1)
        y_list.append(x2)
        diff.append(abs(x1-x2))
        m1 = max(m1,x1)
        m2 = max(m2,x2)
        if (x1-x2)>N:
            r2br1.append(i)
        elif (x2-x1)>N:
            r1br2.append(i)
        else:
            all_g.append(i)
            if x1<10 and x2<10:
                plt.annotate("(%s,%s) " %(x1,x2)+str(i), xy=(x1,x2), xytext=(-20, 10), textcoords='offset points')
            
    plt.scatter(x_list,y_list,c=diff,alpha=0.5,vmin=0,vmax=400)
    print(min(diff),max(diff))

    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.colorbar()
    plt.title(label[0]+' VS '+label[1]+' all:'+str(len(intersection)))
    return r1br2,r2br1,all_g

def qvalue(pv, pi0=None):
    assert(pv.min() >= 0 and pv.max() <= 1), "p-values should be between 0 and 1"

    original_shape = pv.shape
    pv = pv.ravel()  # flattens the array in place, more efficient than flatten()

    m = float(len(pv))

    # if the number of hypotheses is small, just set pi0 to 1
    if len(pv) < 100 and pi0 is None:
        pi0 = 1.0
    elif pi0 is not None:
        pi0 = pi0
    else:
        # evaluate pi0 for different lambdas
        pi0 = []
        lam = np.arange(0, 0.90, 0.01)
        counts = np.array([(pv > i).sum() for i in np.arange(0, 0.9, 0.01)])
        for l in range(len(lam)):
            pi0.append(counts[l]/(m*(1-lam[l])))

        pi0 = np.array(pi0)

        # fit natural cubic spline
        tck = interpolate.splrep(lam, pi0, k=3)
        pi0 = interpolate.splev(lam[-1], tck)

        if pi0 > 1:
            pi0 = 1.0

    assert(pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

    p_ordered = np.argsort(pv)
    pv = pv[p_ordered]
    qv = pi0 * m/len(pv) * pv
    qv[-1] = min(qv[-1], 1.0)

    for i in range(len(pv)-2, -1, -1):
        qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])

    # reorder qvalues
    qv_temp = qv.copy()
    qv = np.zeros_like(qv)
    qv[p_ordered] = qv_temp

    # reshape qvalues
    qv = qv.reshape(original_shape)

    return qv

def get_l_limits(X):
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 0, np.inf)
    R_vals = np.unique(R2.flatten())
    R_vals = R_vals[R_vals > 1e-8]
    
    l_min = np.sqrt(R_vals.min()) / 2.
    l_max = np.sqrt(R_vals.max()) * 2.
    
    return l_min, l_max

## Kernels ##

def SE_kernel(X, l):
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 1e-12, np.inf)
    return np.exp(-R2 / (2 * l ** 2))


def linear_kernel(X):
    K = np.dot(X, X.T)
    return K / K.max()


def cosine_kernel(X, p):
    ''' Periodic kernel as l -> oo in [Lloyd et al 2014]

    Easier interpretable composability with SE?
    '''
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 1e-12, np.inf)
    return np.cos(2 * np.pi * np.sqrt(R2) / p)


def gower_scaling_factor(K):
    ''' Gower normalization factor for covariance matric K

    Based on https://github.com/PMBio/limix/blob/master/limix/utils/preprocess.py
    tr(PKP) = tr(K) - sum(K)/n  where P = I - 11^T/n
    '''
    n = K.shape[0]
    trPKP = np.trace(K) - K.sum() / n
    return trPKP / (n - 1)


def factor(K):
    S, U = np.linalg.eigh(K)
    # .clip removes negative eigenvalues
    return U, np.clip(S, 1e-8, None)


def get_UT1(U):
    return U.sum(0)


def get_UTy(U, y):
    return y.dot(U)


def mu_hat(delta, UTy, UT1, S, n, Yvar=None):
    ''' ML Estimate of bias mu, function of delta.
    '''
    if Yvar is None:
        Yvar = np.ones_like(S)

    UT1_scaled = UT1 / (S + delta * Yvar)
    sum_1 = UT1_scaled.dot(UTy)
    sum_2 = UT1_scaled.dot(UT1)

    return sum_1 / sum_2


def s2_t_hat(delta, UTy, S, n, Yvar=None):
    ''' ML Estimate of structured noise, function of delta
    '''
    if Yvar is None:
        Yvar = np.ones_like(S)

    UTy_scaled = UTy / (S + delta * Yvar)
    return UTy_scaled.dot(UTy) / n


def LL(delta, UTy, UT1, S, n, Yvar=None):
    ''' Log-likelihood of GP model as a function of delta.

    The parameter delta is the ratio s2_e / s2_t, where s2_e is the
    observation noise and s2_t is the noise explained by covariance
    in time or space.
    '''

    mu_h = mu_hat(delta, UTy, UT1, S, n, Yvar)
    
    if Yvar is None:
        Yvar = np.ones_like(S)

    sum_1 = (np.square(UTy - UT1 * mu_h) / (S + delta * Yvar)).sum()
    sum_2 = np.log(S + delta * Yvar).sum()

    with np.errstate(divide='ignore'):
        return -0.5 * (n * np.log(2 * np.pi) + n * np.log(sum_1 / n) + sum_2 + n)


def logdelta_prior_lpdf(log_delta):
    s2p = 100.
    return -np.log(np.sqrt(2 * np.pi * s2p)) - np.square(log_delta - 20.) / (2 * s2p)


def make_objective(UTy, UT1, S, n, Yvar=None):
    def LL_obj(log_delta):
        return -LL(np.exp(log_delta), UTy, UT1, S, n, Yvar)

    return LL_obj


def brent_max_LL(UTy, UT1, S, n):
    LL_obj = make_objective(UTy, UT1, S, n)
    o = optimize.minimize_scalar(LL_obj, bounds=[-10, 10], method='bounded', options={'maxiter': 32})
    max_ll = -o.fun
    max_delta = np.exp(o.x)
    max_mu_hat = mu_hat(max_delta, UTy, UT1, S, n)
    max_s2_t_hat = s2_t_hat(max_delta, UTy, S, n)

    return max_ll, max_delta, max_mu_hat, max_s2_t_hat


def lbfgsb_max_LL(UTy, UT1, S, n, Yvar=None):
    LL_obj = make_objective(UTy, UT1, S, n, Yvar)
    min_boundary = -10
    max_boundary = 20.
    x, f, d = optimize.fmin_l_bfgs_b(LL_obj, 0., approx_grad=True,
                                                 bounds=[(min_boundary, max_boundary)],
                                                 maxfun=64, factr=1e12, epsilon=1e-4)
    max_ll = -f
    max_delta = np.exp(x[0])

    boundary_ll = -LL_obj(max_boundary)
    if boundary_ll > max_ll:
        max_ll = boundary_ll
        max_delta = np.exp(max_boundary)

    boundary_ll = -LL_obj(min_boundary)
    if boundary_ll > max_ll:
        max_ll = boundary_ll
        max_delta = np.exp(min_boundary)


    max_mu_hat = mu_hat(max_delta, UTy, UT1, S, n, Yvar)
    max_s2_t_hat = s2_t_hat(max_delta, UTy, S, n, Yvar)

    s2_logdelta = 1. / (derivative(LL_obj, np.log(max_delta), n=2) ** 2)

    return max_ll, max_delta, max_mu_hat, max_s2_t_hat, s2_logdelta


def search_max_LL(UTy, UT1, S, n, num=32):
    ''' Search for delta which maximizes log likelihood.
    '''
    min_obj = np.inf
    max_log_delta = np.nan
    LL_obj = make_objective(UTy, UT1, S, n)
    for log_delta in np.linspace(start=-10, stop=20, num=num):
        cur_obj = LL_obj(log_delta)
        if cur_obj < min_obj:
            min_obj = cur_obj
            max_log_delta = log_delta

    max_delta = np.exp(max_log_delta)
    max_mu_hat = mu_hat(max_delta, UTy, UT1, S, n)
    max_s2_t_hat = s2_t_hat(max_delta, UTy, S, n)
    max_ll = -min_obj

    return max_ll, max_delta, max_mu_hat, max_s2_t_hat


def make_FSV(UTy, S, n, Gower):
    def FSV(log_delta):
        s2_t = s2_t_hat(np.exp(log_delta), UTy, S, n)
        s2_t_g = s2_t * Gower

        return s2_t_g / (s2_t_g + np.exp(log_delta) * s2_t)

    return FSV


def _fit_one_gene(g, gene_name, UTy, UT1, S, n, Gower):
    """Fit a single gene; used by both serial and parallel paths."""
    t0 = time()
    max_reg_ll, max_delta, max_mu_hat, max_s2_t_hat, s2_logdelta = lbfgsb_max_LL(UTy, UT1, S, n)
    t = time() - t0
    FSV = make_FSV(UTy, S, n, Gower)
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
    '''Fit GPs after pre-processing for particular lengthscale.

    Optimizations: batch UTy via matrix multiply; optional joblib parallelism.
    '''
    n, G = exp_tab.shape
    vals = exp_tab.values
    UTY = vals.T.dot(U)          # G × n — one matrix mult instead of G dot products
    gene_names = list(exp_tab.columns)

    if n_jobs == 1:
        results = []
        for g in _progress_iter(range(G), total=G, desc='Genes', use_tqdm=use_tqdm, leave=False):
            results.append(_fit_one_gene(g, gene_names[g], UTY[g], UT1, S, n, Gower))
            if not use_tqdm:
                _print_progress(g + 1, G)
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_fit_one_gene)(g, gene_names[g], UTY[g], UT1, S, n, Gower)
            for g in _progress_iter(range(G), total=G, desc='Genes', use_tqdm=use_tqdm, leave=False)
        )
        if not use_tqdm:
            _print_progress(G, G)
    return pd.DataFrame(results)


def null_fits(exp_tab):
    '''Get maximum LL for null model (vectorized).'''
    n, G = exp_tab.shape
    vals = exp_tab.values
    max_s2_e = np.square(vals).sum(0) / n
    max_ll = -0.5 * (n * np.log(2 * np.pi) + n + n * np.log(max_s2_e))
    return pd.DataFrame({
        'g': exp_tab.columns, 'max_ll': max_ll,
        'max_delta': np.inf, 'max_mu_hat': 0.,
        'max_s2_t_hat': 0., 'time': 0, 'n': n,
    })


def const_fits(exp_tab):
    '''Get maximum LL for const model (vectorized).'''
    n, G = exp_tab.shape
    vals = exp_tab.values
    means = vals.mean(0)
    vars_ = np.where(vals.var(0) == 0, 1e-10, vals.var(0))
    sum1 = np.square(vals - means).sum(0)
    max_ll = -0.5 * (n * np.log(vars_) + sum1 / vars_ + n * np.log(2 * np.pi))
    return pd.DataFrame({
        'g': exp_tab.columns, 'max_ll': max_ll,
        'max_delta': np.inf, 'max_mu_hat': means,
        'max_s2_t_hat': 0., 'time': 0, 'n': n,
    })


def simulate_const_model(MLL_params, N):
    dfm = np.zeros((N, MLL_params.shape[0]))
    for i, params in enumerate(MLL_params.iterrows()):
        params = params[1]
        s2_e = params.max_s2_t_hat * params.max_delta
        dfm[:, i] = np.random.normal(params.max_mu_hat, s2_e, N)
        
    dfm = pd.DataFrame(dfm)
    return dfm


def get_mll_results(results, null_model='const'):
    null_lls = results.query('model == "{}"'.format(null_model))[['g', 'max_ll']]
    model_results = results.query('model != "{}"'.format(null_model))
    model_results = model_results[model_results.groupby(['g'])['max_ll'].transform(max) == model_results['max_ll']]
    mll_results = model_results.merge(null_lls, on='g', suffixes=('', '_null'))
    mll_results['LLR'] = mll_results['max_ll'] - mll_results['max_ll_null']

    return mll_results

def dyn_de(X, exp_tab, kernel_space=None, n_jobs=1, use_tqdm=True):
    if kernel_space is None:
        kernel_space = {'SE': [5., 25., 50.]}

    results = []

    if 'null' in kernel_space:
        result = null_fits(exp_tab)
        result['l'] = np.nan; result['M'] = 1; result['model'] = 'null'
        results.append(result)

    if 'const' in kernel_space:
        result = const_fits(exp_tab)
        result['l'] = np.nan; result['M'] = 2; result['model'] = 'const'
        results.append(result)

    logging.info("Pre-calculating USU^T = K's ...")
    US_mats = []
    t0 = time()
    if 'SE' in kernel_space:
        for lengthscale in kernel_space['SE']:
            K = SE_kernel(X, lengthscale)
            U, S = factor(K)
            US_mats.append({'model': 'SE', 'M': 4, 'l': lengthscale,
                            'U': U, 'S': S, 'UT1': get_UT1(U),
                            'Gower': gower_scaling_factor(K)})

    logging.info('Done: {0:.2f}s'.format(time() - t0))
    logging.info('Fitting gene models')
    for cov in _progress_iter(US_mats, total=len(US_mats), desc='Models', use_tqdm=use_tqdm, leave=False):
        result = lengthscale_fits(exp_tab, cov['U'], cov['UT1'], cov['S'], cov['Gower'],
                                  n_jobs=n_jobs, use_tqdm=use_tqdm)
        result['l'] = cov['l']; result['M'] = cov['M']; result['model'] = cov['model']
        results.append(result)

    results = pd.concat(results, sort=True).reset_index(drop=True)
    results['BIC'] = -2 * results['max_ll'] + results['M'] * np.log(results['n'])
    return results


def get_mll_results(results, null_model='const'):
    null_lls = results.query('model == "{}"'.format(null_model))[['g', 'max_ll']]
    model_results = results.query('model != "{}"'.format(null_model))
    model_results = model_results[model_results.groupby(['g'])['max_ll'].transform(max) == model_results['max_ll']]
    mll_results = model_results.merge(null_lls, on='g', suffixes=('', '_null'))
    mll_results['LLR'] = mll_results['max_ll'] - mll_results['max_ll_null']
    return mll_results

def stabilize(expression_matrix):
    from scipy import optimize
    phi_hat, _ = optimize.curve_fit(lambda mu, phi: mu + phi * mu ** 2, expression_matrix.mean(1), expression_matrix.var(1))

    return np.log(expression_matrix + 1. / (2 * phi_hat[0]))

def regress_out(sample_info, expression_matrix, covariate_formula, design_formula='1', rcond=-1):
    import patsy
    # Ensure intercept is not part of covariates
    covariate_formula += ' - 1'

    covariate_matrix = patsy.dmatrix(covariate_formula, sample_info)
    design_matrix = patsy.dmatrix(design_formula, sample_info)

    design_batch = np.hstack((design_matrix, covariate_matrix))

    coefficients, res, rank, s = np.linalg.lstsq(design_batch, expression_matrix.T, rcond=rcond)
    beta = coefficients[design_matrix.shape[1]:]
    regressed = expression_matrix - covariate_matrix.dot(beta).T

    return regressed
