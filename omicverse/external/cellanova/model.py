import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools as ft
import seaborn as sea
import anndata as ad
import scanpy as sc
import scanpy.external as sce
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.io import mmread
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import svds
import gc
import sys
from tqdm import tqdm
from ..._anndata_compat import as_dtype as _as_dtype





def calc_ME(adata, integrate_key, k_harmony=70, dim_ME = 70, return_harmony = False):
    ''' Compute cell states and main effects

    Parameters
    ----------
    adata : anndata object
        Preprocessed data. Should be normalized log-transformed and scaled.
    integrate_key : str
        Key for the integration unit for harmony. Should be a column name of adata.obs.
    dim_ME : int, optional
        Dimension of main effects. 
    n_pcs : int, optional
        Number of principle components used in harmony integration.
    return_harmony : boolean, optional
        Whether or not return harmony integrated data.

    Returns
    ----------
    adata: anndata object
        Three new attributes added. 
        adata.obsm['Cmat'] : np.array, the estimated cell embedding. 
        adata.varm['Mmat'] : np.array, the estimated main effect matrix. 
        adata.layers['main_effect'] : np.array, the estimated main effects.
    adata_harmony: anndata object
        Returned only when return_harmony = True. Harmony integrated data.
    '''

    # harmony integration
    rna_temp = ad.AnnData(_as_dtype(adata[:, adata.var.highly_variable].X, np.float32))
    rna_temp.var_names = adata[:, adata.var.highly_variable].var_names
    rna_temp.obs = adata.obs.copy()
    sc.tl.pca(rna_temp, n_comps = k_harmony)
    sce.pp.harmony_integrate(rna_temp, integrate_key, max_iter_harmony = 30)
    adata_harmony = ad.AnnData(_as_dtype(rna_temp.obsm['X_pca_harmony'] @ rna_temp.varm['PCs'].T, np.float32))
    adata_harmony.var_names = rna_temp.var_names
    adata_harmony.obs = rna_temp.obs.copy()

    # estimate C, M
    Cmat, _, _ = svds(adata_harmony.X, k=dim_ME)
    dataidx = np.unique(adata.obs[integrate_key])
    bloc_coef = ()
    counter = 0
    for x in tqdm(dataidx):
        bloc_temp = linear_model.LinearRegression(fit_intercept = False)
        bloc_temp.fit(Cmat[adata.obs[integrate_key] == x,:], adata.layers['scaled'][adata.obs[integrate_key] == x,:])
        bloc_coef = bloc_coef + (bloc_temp.coef_.T,)
        counter = counter + 1
    Mmat = ft.reduce(np.add, bloc_coef) / len(dataidx) 
    
    adata.obsm['Cmat'] = Cmat
    adata.varm['Mmat'] = Mmat.T
    adata.layers['main_effect'] = Cmat @ Mmat
    
    if return_harmony:
        return adata, adata_harmony
    else:
        return adata


def fit_M(adata, integrate_key):

    # input: 
    # adata is after preprocessing step.
    # adata including all control and treatment samples
    # adata should have cellstates stored in adata.obsm['Cmat']
    # integrate_key indicates sample ID, the smallest unit

    Cmat = adata.obsm['Cmat']
    dataidx = np.unique(adata.obs[integrate_key])
    bloc_coef = ()
    counter = 0
    for x in dataidx:
        bloc_temp = linear_model.LinearRegression(fit_intercept = False)
        bloc_temp.fit(Cmat[adata.obs[integrate_key] == x,:], adata.layers['scaled'][adata.obs[integrate_key] == x,:])
        bloc_coef = bloc_coef + (bloc_temp.coef_.T,)
        counter = counter + 1
    Mmat = ft.reduce(np.add, bloc_coef) / len(dataidx) 
    
    return Mmat

    


def calc_BE(adata, integrate_key, control_dict, var_cutoff = 0.9, k_max = 1500, verbose = False, k_select = None):
    ''' Compute basis of batch effect. Perform batch correction.

    Parameters
    ----------
    adata : anndata object
        Preprocessed data. Should be normalized log-transformed and scaled.
    integrate_key : str
        Key for the integration unit in the control_dict. Should be a column name of adata.obs.
    control_dict : dict
        A dictionary, in which each key is a control group, and the corresponding values are the integration units
        in this group.
    var_cutoff : float, optional
        Fraction of explained variance to determine the optimal value of k in truncated SVD.
    k_max: int, optional
        Max of singular values and vectors to compute.
    verbose: boolean, optional
        Whether to plot sigular values or not.
    k_select: int, optional
        Pre-determined number of singular values and vectors to compute.

    Returns
    ----------
    adata: anndata object
        Three new attributes added. 
        adata.varm['V_BE_basis'] : np.array, batch effect basis matrix. 
        adata.uns['S_BE_basis'] : np.array, singular values of batch basis. 
        adata.layers['corrected'] : np.array, the batch corrected object.
    '''

    # regression
    control_groups = list(control_dict.keys())
    LL = adata.obsm['Cmat']
    res = ()
    for g in control_groups:
        control_batch = control_dict[g]
        bloc_coef = ()
        counter = 0

        for x in control_batch:
            bloc_temp = linear_model.LinearRegression(fit_intercept = False)
            bloc_temp.fit(LL[adata.obs[integrate_key] == x,:], adata.layers['scaled'][adata.obs[integrate_key] == x,:])
            bloc_coef = bloc_coef + (bloc_temp.coef_.T,)
            counter = counter + 1

        M = ft.reduce(np.add, bloc_coef) / len(control_batch)
        res_temp = np.concatenate(bloc_coef, axis = 0) - np.tile(M, (len(control_batch),1))
        res = res + (res_temp,)

    res_combined = np.concatenate(res, axis = 0)

    # perform svd
    k_max = np.min([k_max, res_combined.shape[0]-1, res_combined.shape[1]-1])
    _, DD1, _ = svds(res_combined, k = k_max)
    if verbose:
        plt.plot(sorted(np.sqrt(DD1[DD1 > 0]),reverse = True),'bo-')
    if k_select is not None:
        k = k_select
    else:
        DD1_rev = np.array(sorted(np.sqrt(DD1[DD1 > 0]),reverse = True))
        variance = np.cumsum(DD1_rev ** 2) / np.sum(DD1_rev ** 2)
        k = np.argmax(variance >= var_cutoff)
    _, DD1, VV1T = svds(res_combined, k = k)

    adata.varm['V_BE_basis'] = VV1T.T
    adata.uns['S_BE_basis'] = DD1
    be = (adata.layers['scaled'] - adata.obsm['Cmat'] @ adata.varm['Mmat'].T) @ VV1T.T @ VV1T
    adata.layers['corrected'] = adata.layers['scaled'] - be

    return adata





    


def calc_TE(adata, integrate_key, #control_dict, 
            var_cutoff = 0.7, k_max = 1500, verbose = False, k_select = None):
    ''' Compute basis of treatment effect. Perform final integration.

    Parameters
    ----------
    adata : anndata object
        Preprocessed data. Should be normalized log-transformed and scaled.
    integrate_key : str
        Key for the integration unit in the control_dict. Should be a column name of adata.obs.
    control_dict : dict
        A dictionary, in which each key is a control group, and the corresponding values are the integration units
        in this group.
    var_cutoff : float, optional
        Fraction of explained variance to determine the optimal value of k in truncated SVD.
    k_max: int, optional
        Max of singular values and vectors to compute.
    verbose: boolean, optional
        Whether to plot sigular values or not.
    k_select: int, optional
        Pre-determined number of singular values and vectors to compute.

    Returns
    ----------
    adata: anndata object
        Four new attributes added. 
        adata.varm['W_TE_basis'] : np.array, treatment effect basis matrix. 
        adata.uns['S_TE_basis'] : np.array, singular values of treatment basis. 
        adata.layers['trt_effect'] : np.array, the estimated treatment effects.
        adata.layers['denoised'] : np.array, the integrated data.
    '''

    # regression
    trt_batch = list(set(adata.obs[integrate_key]))
    trt_bloc_coef = ()
    counter = 0
    for x in tqdm(trt_batch):
        bloc_temp = linear_model.LinearRegression(fit_intercept = False)
        bloc_temp.fit(adata.obsm['Cmat'][adata.obs[integrate_key] == x,:], 
                      adata.layers['corrected'][adata.obs[integrate_key] == x,:])
        trt_bloc_coef = trt_bloc_coef + (bloc_temp.coef_.T,)
        counter = counter + 1

    res_combined = np.concatenate(trt_bloc_coef, axis = 0) - np.tile(adata.varm['Mmat'].T, (len(trt_bloc_coef),1))

    # perform svd
    k_max = np.min([k_max, res_combined.shape[0]-1, res_combined.shape[1]-1])
    _, DD2, _ = svds(res_combined, k = k_max)
    if verbose:
        plt.plot(sorted(np.sqrt(DD2[DD2 > 0]),reverse = True),'bo-')
    if k_select is not None:
        k = k_select
    else:
        DD2_rev = np.array(sorted(np.sqrt(DD2[DD2 > 0]),reverse = True))
        variance = np.cumsum(DD2_rev ** 2) / np.sum(DD2_rev ** 2)
        k = np.argmax(variance >= var_cutoff)
    _, DD2, WW2T = svds(res_combined, k = k)

    adata.varm['W_TE_basis'] = WW2T.T
    adata.uns['S_TE_basis'] = DD2
    te = (adata.layers['corrected'] - adata.obsm['Cmat'] @ adata.varm['Mmat'].T) @ WW2T.T @ WW2T
    adata.layers['trt_effect'] = te
    adata.layers['denoised'] = adata.layers['trt_effect'] + adata.layers['main_effect']

    return adata



def calc_BT_coef(adata, integrate_key):
    
    Cmat = adata.obsm['Cmat']
    dataidx = np.unique(adata.obs[integrate_key])
    bloc_coef = ()
    counter = 0
    for x in tqdm(dataidx):
        bloc_temp = linear_model.LinearRegression(fit_intercept = False)
        bloc_temp.fit(Cmat[adata.obs[integrate_key] == x,:], adata.layers['scaled'][adata.obs[integrate_key] == x,:])
        bloc_coef = bloc_coef + (bloc_temp.coef_.T,)
        counter = counter + 1
    Mmat = ft.reduce(np.add, bloc_coef) / len(dataidx) 



def preprocess_data(adata, integrate_key, n_hvgs=3000, hvg_only = True, copy_raw = False, copy_lognorm = True):
    ''' Perform CellANOVA preprocessing: library size normalization + log1p + hvg selection + standardization 

    Parameters
    ----------
    adata : anndata object
        Raw data, containing gene expression counts.
    integrate_key : str
        Key indicating smallest batch unit. Should be a column name of adata.obs.
    n_hvgs : int, optional
        Number of highly variable genes.
    hvg_only : boolean, optional
        Whether or not return only highly variable genes or all genes.
    copy_raw : boolean, optional
        Whether or not save a copy of raw counts.

    Returns
    ----------
    adata: anndata object
        Data after preprocessing, which can be passed to cell state estimation.
    '''

    ## Preprocess: reorder, normalize per cell (sum to 1e5), log, select 3000 hvgs batch-wise, scale
    batch = np.unique(adata.obs[integrate_key])
    list_batch = []
    for x in batch:
        adata_iter = adata[adata.obs[integrate_key] == x]
        if copy_raw:
            adata_iter.layers['counts'] = adata_iter.X
        sc.pp.normalize_total(adata_iter, target_sum=1e5)
        list_batch.append(adata_iter)

    adata_prep = ad.concat(list_batch)
    adata_prep = sc.pp.log1p(adata_prep, copy = True)
    if copy_lognorm:
        adata_prep.layers['lognorm'] = adata_prep.X
    sc.pp.highly_variable_genes(adata_prep, n_top_genes=n_hvgs, batch_key = integrate_key)
    if hvg_only:
        adata_prep = adata_prep[:, adata_prep.var.highly_variable].copy()

    sc.pp.scale(adata_prep)
    adata_prep.layers['scaled'] = adata_prep.X

    return adata_prep



