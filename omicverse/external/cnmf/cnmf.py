#!/usr/bin/env python

import numpy as np
import pandas as pd
import os, errno, sys
import datetime

import itertools
import yaml
import subprocess
import scipy.sparse as sp

from scipy.spatial.distance import squareform
from sklearn.decomposition import non_negative_factorization
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import sparsefuncs


from scipy.cluster.hierarchy import leaves_list

import matplotlib.pyplot as plt

import scanpy as sc

from multiprocessing import Pool




def save_df_to_npz(obj, filename):
    np.savez_compressed(filename, data=obj.values, index=obj.index.values, columns=obj.columns.values)

def save_df_to_text(obj, filename):
    obj.to_csv(filename, sep='\t')

def load_df_from_npz(filename):
    with np.load(filename, allow_pickle=True) as f:
        obj = pd.DataFrame(**f)
    return obj

def check_dir_exists(path):
    """
    Checks if directory already exists or not and creates it if it doesn't
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def worker_filter(iterable, worker_index, total_workers):
    return (p for i,p in enumerate(iterable) if (i-worker_index)%total_workers==0)

def fast_ols_all_cols(X, Y):
    pinv = np.linalg.pinv(X)
    beta = np.dot(pinv, Y)
    return(beta)

def fast_ols_all_cols_df(X,Y):
    beta = fast_ols_all_cols(X, Y)
    beta = pd.DataFrame(beta, index=X.columns, columns=Y.columns)
    return(beta)

def var_sparse_matrix(X):
    mean = np.array(X.mean(axis=0)).reshape(-1)
    Xcopy = X.copy()
    Xcopy.data **= 2
    var = np.array(Xcopy.mean(axis=0)).reshape(-1) - (mean**2)
    return(var)


def get_highvar_genes_sparse(expression, expected_fano_threshold=None,
                       minimal_mean=0.5, numgenes=None):
    # Find high variance genes within those cells
    gene_mean = np.array(expression.mean(axis=0)).astype(float).reshape(-1)
    E2 = expression.copy(); E2.data **= 2; gene2_mean = np.array(E2.mean(axis=0)).reshape(-1)
    gene_var = pd.Series(gene2_mean - (gene_mean**2))
    del(E2)
    gene_mean = pd.Series(gene_mean)
    gene_fano = gene_var / gene_mean

    # Find parameters for expected fano line
    top_genes = gene_mean.sort_values(ascending=False)[:20].index
    A = (np.sqrt(gene_var)/gene_mean)[top_genes].min()
    
    w_mean_low, w_mean_high = gene_mean.quantile([0.10, 0.90])
    w_fano_low, w_fano_high = gene_fano.quantile([0.10, 0.90])
    winsor_box = ((gene_fano > w_fano_low) &
                    (gene_fano < w_fano_high) &
                    (gene_mean > w_mean_low) &
                    (gene_mean < w_mean_high))
    fano_median = gene_fano[winsor_box].median()
    B = np.sqrt(fano_median)

    gene_expected_fano = (A**2)*gene_mean + (B**2)
    fano_ratio = (gene_fano/gene_expected_fano)

    # Identify high var genes
    if numgenes is not None:
        highvargenes = fano_ratio.sort_values(ascending=False).index[:numgenes]
        high_var_genes_ind = fano_ratio.index.isin(highvargenes)
        T=None


    else:
        if not expected_fano_threshold:
            T = (1. + gene_fano[winsor_box].std())
        else:
            T = expected_fano_threshold

        high_var_genes_ind = (fano_ratio > T) & (gene_mean > minimal_mean)

    gene_counts_stats = pd.DataFrame({
        'mean': gene_mean,
        'var': gene_var,
        'fano': gene_fano,
        'expected_fano': gene_expected_fano,
        'high_var': high_var_genes_ind,
        'fano_ratio': fano_ratio
        })
    gene_fano_parameters = {
            'A': A, 'B': B, 'T':T, 'minimal_mean': minimal_mean,
        }
    return(gene_counts_stats, gene_fano_parameters)



def get_highvar_genes(input_counts, expected_fano_threshold=None,
                       minimal_mean=0.5, numgenes=None):
    # Find high variance genes within those cells
    gene_counts_mean = pd.Series(input_counts.mean(axis=0).astype(float))
    gene_counts_var = pd.Series(input_counts.var(ddof=0, axis=0).astype(float))
    gene_counts_fano = pd.Series(gene_counts_var/gene_counts_mean)

    # Find parameters for expected fano line
    top_genes = gene_counts_mean.sort_values(ascending=False)[:20].index
    A = (np.sqrt(gene_counts_var)/gene_counts_mean)[top_genes].min()

    w_mean_low, w_mean_high = gene_counts_mean.quantile([0.10, 0.90])
    w_fano_low, w_fano_high = gene_counts_fano.quantile([0.10, 0.90])
    winsor_box = ((gene_counts_fano > w_fano_low) &
                    (gene_counts_fano < w_fano_high) &
                    (gene_counts_mean > w_mean_low) &
                    (gene_counts_mean < w_mean_high))
    fano_median = gene_counts_fano[winsor_box].median()
    B = np.sqrt(fano_median)

    gene_expected_fano = (A**2)*gene_counts_mean + (B**2)

    fano_ratio = (gene_counts_fano/gene_expected_fano)

    # Identify high var genes
    if numgenes is not None:
        highvargenes = fano_ratio.sort_values(ascending=False).index[:numgenes]
        high_var_genes_ind = fano_ratio.index.isin(highvargenes)
        T=None


    else:
        if not expected_fano_threshold:
            T = (1. + gene_counts_fano[winsor_box].std())
        else:
            T = expected_fano_threshold

        high_var_genes_ind = (fano_ratio > T) & (gene_counts_mean > minimal_mean)

    gene_counts_stats = pd.DataFrame({
        'mean': gene_counts_mean,
        'var': gene_counts_var,
        'fano': gene_counts_fano,
        'expected_fano': gene_expected_fano,
        'high_var': high_var_genes_ind,
        'fano_ratio': fano_ratio
        })
    gene_fano_parameters = {
            'A': A, 'B': B, 'T':T, 'minimal_mean': minimal_mean,
        }
    return(gene_counts_stats, gene_fano_parameters)


def compute_tpm(input_counts):
    """
    Default TPM normalization
    """
    tpm = input_counts.copy()
    sc.pp.normalize_total(tpm, target_sum=1e6)
    return(tpm)


def factorize_mp_signature(args):
    """
    wrapper around factorize to be able to use mp pool.
    args is a list:
    worker-i: int
    total_workers: int
    pointer to nmf object.
    """
    args[2].factorize(worker_i=args[0],  total_workers=args[1])


from ..._registry import register_function

@register_function(
    aliases=["共识非负矩阵分解", "cNMF", "consensus_nmf", "cNMF聚类", "共识NMF"],
    category="single",
    description="Consensus Non-negative Matrix Factorization (cNMF) for robust topic modeling and clustering",
    examples=[
        "# Basic cNMF analysis",
        "cnmf_obj = ov.single.cNMF(adata, components=np.arange(5,11),",
        "                          n_iter=20, seed=14, num_highvar_genes=2000,",
        "                          output_dir='example_dg1/cNMF', name='dg_cNMF')",
        "# Run factorization",
        "cnmf_obj.factorize(worker_i=0, total_workers=4)",
        "cnmf_obj.combine(skip_missing_files=True)",
        "# K selection and consensus",
        "selected_K = 7",
        "density_threshold = 2.00",
        "cnmf_obj.consensus(k=selected_K, density_threshold=density_threshold)",
        "result_dict = cnmf_obj.load_results(K=selected_K, density_threshold=density_threshold)",
        "cnmf_obj.get_results(adata, result_dict)",
        "# Advanced classification with Random Forest",
        "cnmf_obj.get_results_rfc(adata, result_dict, cNMF_threshold=0.5)"
    ],
    related=["utils.LDA_topic", "utils.cluster", "pl.embedding"]
)
class cNMF():
    """Consensus NMF workflow wrapper for robust gene-program discovery.

    Parameters
    ----------
    adata : AnnData
        Input single-cell expression AnnData.
    components : array-like
        Candidate rank values (number of programs/topics) to evaluate.
    n_iter : int, default=100
        Number of NMF restarts per rank.
    output_dir : str, optional
        Directory used to store temporary and result files.
    name : str, optional
        Analysis name prefix for output artifacts.
    use_gpu : bool, default=True
        Whether to use GPU-accelerated factorization when available.
    gpu_id : int, default=0
        CUDA device index.
    """


    def __init__(self, adata, components, n_iter = 100, densify=False, tpm_fn=None, seed=None,
                        beta_loss='frobenius',num_highvar_genes=2000, genes_file=None,
                        alpha_usage=0.0, alpha_spectra=0.0, init='random',output_dir=None, name=None,
                        use_gpu=True, gpu_id=0):
        """
        Arguments:
            output_dir: path, optional (default=None). Output directory for analysis files. If None, all analysis is done in memory.
            name: string, optional (default=None). A name for this analysis. Will be prefixed to all output files. If set to None, will be automatically generated from date (and random string).
            use_gpu: bool, optional (default=True). If True and GPU is available, use GPU acceleration for NMF factorization.
            gpu_id: int, optional (default=0). GPU device ID to use when multiple GPUs are available.
        """

        self.output_dir = output_dir
        if name is None:
            import uuid
            now = datetime.datetime.now()
            rand_hash =  uuid.uuid4().hex[:6]
            name = '%s_%s' % (now.strftime("%Y_%m_%d"), rand_hash)
        self.name = name
        self.paths = None

        # GPU configuration
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        if use_gpu:
            import torch
            self.device = torch.device(f'cuda:{gpu_id}' if self.use_gpu else 'cpu')

        # In-memory storage
        self.tpm = None
        self.tpm_stats = None
        self.norm_counts = None
        self.replicate_params = None
        self.run_params = None
        self.highvar_genes = None
        self.iter_spectra = {}  # {(k, iter): spectra_df}
        self.merged_spectra_dict = {}  # {k: merged_spectra_df}
        self.consensus_results = {}  # {(k, density_threshold): result_dict}

        if self.output_dir is not None:
            self._initialize_dirs()
        self.prepare(adata, components, n_iter, densify, tpm_fn, seed,
                        beta_loss, num_highvar_genes, genes_file,
                        alpha_usage, alpha_spectra, init)



    def _initialize_dirs(self):
        if self.paths is None:
            # Check that output directory exists, create it if needed.
            check_dir_exists(self.output_dir)
            check_dir_exists(os.path.join(self.output_dir, self.name))
            check_dir_exists(os.path.join(self.output_dir, self.name, 'cnmf_tmp'))

            self.paths = {
                'normalized_counts' : os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.norm_counts.h5ad'),
                'nmf_replicate_parameters' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.nmf_params.df.npz'),
                'nmf_run_parameters' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.nmf_idvrun_params.yaml'),
                'nmf_genes_list' :  os.path.join(self.output_dir, self.name, self.name+'.overdispersed_genes.txt'),

                'tpm' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.tpm.h5ad'),
                'tpm_stats' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.tpm_stats.df.npz'),

                'iter_spectra' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.spectra.k_%d.iter_%d.df.npz'),
                'iter_usages' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.usages.k_%d.iter_%d.df.npz'),
                'merged_spectra': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.spectra.k_%d.merged.df.npz'),

                'local_density_cache': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.local_density_cache.k_%d.merged.df.npz'),
                'consensus_spectra': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.spectra.k_%d.dt_%s.consensus.df.npz'),
                'consensus_spectra__txt': os.path.join(self.output_dir, self.name, self.name+'.spectra.k_%d.dt_%s.consensus.txt'),
                'consensus_usages': os.path.join(self.output_dir, self.name, 'cnmf_tmp',self.name+'.usages.k_%d.dt_%s.consensus.df.npz'),
                'consensus_usages__txt': os.path.join(self.output_dir, self.name, self.name+'.usages.k_%d.dt_%s.consensus.txt'),

                'consensus_stats': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.stats.k_%d.dt_%s.df.npz'),

                'clustering_plot': os.path.join(self.output_dir, self.name, self.name+'.clustering.k_%d.dt_%s.png'),
                'gene_spectra_score': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.gene_spectra_score.k_%d.dt_%s.df.npz'),
                'gene_spectra_score__txt': os.path.join(self.output_dir, self.name, self.name+'.gene_spectra_score.k_%d.dt_%s.txt'),
                'gene_spectra_tpm': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.gene_spectra_tpm.k_%d.dt_%s.df.npz'),
                'gene_spectra_tpm__txt': os.path.join(self.output_dir, self.name, self.name+'.gene_spectra_tpm.k_%d.dt_%s.txt'),

                'k_selection_plot' :  os.path.join(self.output_dir, self.name, self.name+'.k_selection.png'),
                'k_selection_stats' :  os.path.join(self.output_dir, self.name, self.name+'.k_selection_stats.df.npz'),
            }


    def prepare(self, adata, components, n_iter = 100, densify=False, tpm_fn=None, seed=None,
                        beta_loss='frobenius',num_highvar_genes=2000, genes_file=None,
                        alpha_usage=0.0, alpha_spectra=0.0, init='random'):
        """
        Load input counts, reduce to high-variance genes, and variance normalize genes.
        Prepare file for distributing jobs over workers.

    Arguments:
        counts_fn: Path to input counts matrix
        components: Values of K to run NMF for
        n_iter: Number of iterations for factorization. If several "k" are specified, this many iterations will be run for each value of "k".
        densify: Convert sparse data to dense
        tpm_fn: If provided, load tpm data from file. Otherwise will compute it from the counts file
        seed: Seed for sklearn random state.
        beta_loss: 
        num_highvar_genes: If provided and genes_file is None, will compute this many highvar genes to use for factorization
        genes_file: If provided will load high-variance genes from a list of these genes
        alpha_usage: Regularization parameter for NMF corresponding to alpha_W in scikit-learn
        alpha_spectra: Regularization parameter for NMF corresponding to alpha_H in scikit-learn
        """
        
          
        if 'counts' not in adata.layers:
            raise Exception('Error: No counts data found in adata.layers. Quitting!')
        input_counts=adata.copy()
        input_counts.X = input_counts.layers['counts']

                
        if sp.issparse(input_counts.X) & densify:
            input_counts.X = np.array(input_counts.X.todense())
 
        if tpm_fn is None:
            tpm = compute_tpm(input_counts)
        elif tpm_fn.endswith('.h5ad'):
            tpm = sc.read(tpm_fn)
        else:
            if tpm_fn.endswith('.npz'):
                tpm = load_df_from_npz(tpm_fn)
            else:
                tpm = pd.read_csv(tpm_fn, sep='\t', index_col=0)

            if densify:
                tpm = sc.AnnData(X=tpm.values,
                            obs=pd.DataFrame(index=tpm.index),
                            var=pd.DataFrame(index=tpm.columns))
            else:
                tpm = sc.AnnData(X=sp.csr_matrix(tpm.values),
                            obs=pd.DataFrame(index=tpm.index),
                            var=pd.DataFrame(index=tpm.columns))

        # Store in memory
        self.tpm = tpm
        # Optionally save to disk
        if self.output_dir is not None:
            sc.write(self.paths['tpm'], tpm)
        
        if sp.issparse(tpm.X):
            gene_tpm_mean = np.array(tpm.X.mean(axis=0)).reshape(-1)
            gene_tpm_stddev = var_sparse_matrix(tpm.X)**.5
        else:
            gene_tpm_mean = np.array(tpm.X.mean(axis=0)).reshape(-1)
            gene_tpm_stddev = np.array(tpm.X.std(axis=0, ddof=0)).reshape(-1)
            
            
        input_tpm_stats = pd.DataFrame([gene_tpm_mean, gene_tpm_stddev],
             index = ['__mean', '__std'], columns = tpm.var.index).T

        # Store in memory
        self.tpm_stats = input_tpm_stats
        # Optionally save to disk
        if self.output_dir is not None:
            save_df_to_npz(input_tpm_stats, self.paths['tpm_stats'])

        if genes_file is not None:
            highvargenes = open(genes_file).read().rstrip().split('\n')
        else:
            highvargenes = None

        norm_counts = self.get_norm_counts(input_counts, tpm, num_highvar_genes=num_highvar_genes,
                                               high_variance_genes_filter=highvargenes)

        # Store in memory
        self.norm_counts = norm_counts
        # Optionally save to disk
        if self.output_dir is not None:
            self.save_norm_counts(norm_counts)

        (replicate_params, run_params) = self.get_nmf_iter_params(ks=components, n_iter=n_iter, random_state_seed=seed,
                                                                  beta_loss=beta_loss, alpha_usage=alpha_usage,
                                                                  alpha_spectra=alpha_spectra, init=init)

        # Store in memory
        self.replicate_params = replicate_params
        self.run_params = run_params
        # Optionally save to disk
        if self.output_dir is not None:
            self.save_nmf_iter_params(replicate_params, run_params)
        
    
    def combine(self, components=None, skip_missing_files=False):
        """
        Combine NMF iterations for the same value of K
        Parameters
        ----------
        components : list or None
            Values of K to combine iterations for. Defaults to all.

        skip_missing_files : boolean
            If True, ignore iteration files that aren't found rather than crashing. Default: False
        """

        if type(components) is int:
            ks = [components]
        elif components is None:
            # Use in-memory data
            run_params = self.replicate_params
            ks = sorted(set(run_params.n_components))
        else:
            ks = components

        for k in ks:
            self.combine_nmf(k, skip_missing_files=skip_missing_files)    
    
    
    
    def get_norm_counts(self, counts, tpm,
                         high_variance_genes_filter = None,
                         num_highvar_genes = None
                         ):
        """
        Parameters
        ----------

        counts : anndata.AnnData
            Scanpy AnnData object (cells x genes) containing raw counts. Filtered such that
            no genes or cells with 0 counts
        
        tpm : anndata.AnnData
            Scanpy AnnData object (cells x genes) containing tpm normalized data matching
            counts

        high_variance_genes_filter : np.array, optional (default=None)
            A pre-specified list of genes considered to be high-variance.
            Only these genes will be used during factorization of the counts matrix.
            Must match the .var index of counts and tpm.
            If set to None, high-variance genes will be automatically computed, using the
            parameters below.

        num_highvar_genes : int, optional (default=None)
            Instead of providing an array of high-variance genes, identify this many most overdispersed genes
            for filtering

        Returns
        -------

        normcounts : anndata.AnnData, shape (cells, num_highvar_genes)
            A counts matrix containing only the high variance genes and with columns (genes)normalized to unit
            variance

        """

        if high_variance_genes_filter is None:
            ## Get list of high-var genes if one wasn't provided
            if sp.issparse(tpm.X):
                (gene_counts_stats, gene_fano_params) = get_highvar_genes_sparse(tpm.X, numgenes=num_highvar_genes)  
            else:
                (gene_counts_stats, gene_fano_params) = get_highvar_genes(np.array(tpm.X), numgenes=num_highvar_genes)
                
            high_variance_genes_filter = list(tpm.var.index[gene_counts_stats.high_var.values])
                
        ## Subset out high-variance genes
        norm_counts = counts[:, high_variance_genes_filter]

        ## Scale genes to unit variance
        if sp.issparse(tpm.X):
            sc.pp.scale(norm_counts, zero_center=False)
            if np.isnan(norm_counts.X.data).sum() > 0:
                print('Warning NaNs in normalized counts matrix')                       
        else:
            norm_counts.X /= norm_counts.X.std(axis=0, ddof=1)
            if np.isnan(norm_counts.X).sum().sum() > 0:
                print('Warning NaNs in normalized counts matrix')                    
        
        ## Save high-variance genes list in memory
        self.highvar_genes = high_variance_genes_filter
        ## Optionally save to disk
        if self.output_dir is not None:
            open(self.paths['nmf_genes_list'], 'w').write('\n'.join(high_variance_genes_filter))

        ## Check for any cells that have 0 counts of the overdispersed genes
        zerocells = np.array(norm_counts.X.sum(axis=1)==0).reshape(-1)
        if zerocells.sum()>0:
            examples = norm_counts.obs.index[np.ravel(zerocells)]
            raise Exception('Error: %d cells have zero counts of overdispersed genes. E.g. %s. Filter those cells and re-run or adjust the number of overdispersed genes. Quitting!' % (zerocells.sum(), ', '.join(examples[:4])))
        
        return(norm_counts)

    
    def save_norm_counts(self, norm_counts):
        self._initialize_dirs()
        sc.write(self.paths['normalized_counts'], norm_counts)

        
    def get_nmf_iter_params(self, ks, n_iter = 100,
                               random_state_seed = None,
                               beta_loss = 'kullback-leibler',
                               alpha_usage=0.0, alpha_spectra=0.0,
                               init='random', max_iter=1000):
        """
        Create a DataFrame with parameters for NMF iterations.


        Parameters
        ----------
        ks : integer, or list-like.
            Number of topics (components) for factorization.
            Several values can be specified at the same time, which will be run independently.

        n_iter : integer, optional (defailt=100)
            Number of iterations for factorization. If several ``k`` are specified, this many
            iterations will be run for each value of ``k``.

        random_state_seed : int or None, optional (default=None)
            Seed for sklearn random state.
            
        alpha_usage : float, optional (default=0.0)
            Regularization parameter for NMF corresponding to alpha_W in scikit-learn

        alpha_spectra : float, optional (default=0.0)
            Regularization parameter for NMF corresponding to alpha_H in scikit-learn
        """

        if type(ks) is int:
            ks = [ks]

        # Remove any repeated k values, and order.
        k_list = sorted(set(list(ks)))

        n_runs = len(ks)* n_iter

        np.random.seed(seed=random_state_seed)
        nmf_seeds = np.random.randint(low=1, high=(2**31)-1, size=n_runs)

        replicate_params = []
        for i, (k, r) in enumerate(itertools.product(k_list, range(n_iter))):
            replicate_params.append([k, r, nmf_seeds[i]])
        replicate_params = pd.DataFrame(replicate_params, columns = ['n_components', 'iter', 'nmf_seed'])

        _nmf_kwargs = dict(
                        alpha_W=alpha_usage,
                        alpha_H=alpha_spectra,
                        l1_ratio=0.0,
                        beta_loss=beta_loss,
                        solver='mu',
                        tol=1e-4,
                        max_iter=max_iter,
                        init=init
                        )
        
        ## Coordinate descent is faster than multiplicative update but only works for frobenius
        if beta_loss == 'frobenius':
            _nmf_kwargs['solver'] = 'cd'

        return(replicate_params, _nmf_kwargs)


    def save_nmf_iter_params(self, replicate_params, run_params):
        self._initialize_dirs()
        save_df_to_npz(replicate_params, self.paths['nmf_replicate_parameters'])
        with open(self.paths['nmf_run_parameters'], 'w') as F:
            yaml.dump(run_params, F)


    def _nmf(self, X, nmf_kwargs):
        """
        GPU-accelerated NMF with automatic fallback to CPU.

        Parameters
        ----------
        X : pandas.DataFrame or scipy.sparse matrix,
            Normalized counts dataFrame to be factorized.

        nmf_kwargs : dict,
            Arguments to be passed to NMF

        Returns
        -------
        spectra : numpy.ndarray
            Gene expression programs (components x genes)
        usages : numpy.ndarray
            Usage matrix (cells x components)
        """
        if self.use_gpu:
            try:
                return self._nmf_torch(X, nmf_kwargs)
            except Exception as e:
                print(f"⚠️  GPU NMF failed: {e}")
                print("⚠️  Falling back to CPU mode")
                # Fall through to CPU mode

        # CPU mode (original sklearn implementation)
        (usages, spectra, niter) = non_negative_factorization(X, **nmf_kwargs)
        return(spectra, usages)

    def _nmf_torch(self, X, nmf_kwargs):
        """
        PyTorch GPU-accelerated NMF implementation.

        Parameters
        ----------
        X : scipy.sparse or numpy.ndarray
            Input matrix to factorize
        nmf_kwargs : dict
            NMF parameters from sklearn

        Returns
        -------
        spectra : numpy.ndarray
        usages : numpy.ndarray
        """
        # Convert sparse matrix to dense if needed
        # GPU acceleration support
        try:
            import torch
            from torchnmf.nmf import NMF as TorchNMF
            TORCH_AVAILABLE = True
            CUDA_AVAILABLE = torch.cuda.is_available()
        except ImportError:
            TORCH_AVAILABLE = False
            CUDA_AVAILABLE = False
            print("⚠ torchnmf not available. Install with: pip install torchnmf") 
        if sp.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = np.array(X)

        # Convert to PyTorch tensor and move to GPU
        X_tensor = torch.FloatTensor(X_dense).to(self.device)

        # Extract parameters
        n_components = nmf_kwargs['n_components']
        max_iter = nmf_kwargs.get('max_iter', 200)
        tol = nmf_kwargs.get('tol', 1e-4)
        beta_loss = nmf_kwargs.get('beta_loss', 'frobenius')
        random_state = nmf_kwargs.get('random_state', None)

        # Map beta_loss to torchnmf beta parameter
        # sklearn: 'frobenius' (beta=2), 'kullback-leibler' (beta=1)
        # torchnmf: beta parameter directly
        if beta_loss == 'frobenius':
            beta = 2.0
        elif beta_loss == 'kullback-leibler':
            beta = 1.0
        else:
            beta = 2.0  # default to Frobenius

        # Set random seed if provided
        if random_state is not None:
            torch.manual_seed(random_state)
            if self.use_gpu:
                torch.cuda.manual_seed(random_state)

        # Initialize torchnmf model
        # Note: Only shape and rank are passed to __init__()
        # Beta, tol, max_iter are passed to fit() method
        model = TorchNMF(
            X_tensor.shape,
            rank=n_components
        ).to(self.device)

        # Fit the model with all training parameters
        model.fit(
            X_tensor,
            beta=beta,
            tol=tol,
            max_iter=max_iter,
            verbose=False
        )

        # Extract results and move back to CPU
        # IMPORTANT: torchnmf uses V ≈ H @ W^T (not W @ H like sklearn)
        # - H: (cells, components) - this is usages
        # - W: (genes, components) - need to transpose for spectra
        usages = model.H.detach().cpu().numpy()  # (cells, components)
        spectra = model.W.T.detach().cpu().numpy()  # (components, genes) - transposed!

        # Clear GPU memory
        del X_tensor, model
        if self.use_gpu:
            torch.cuda.empty_cache()

        return (spectra, usages)

    def factorize_multi_process(self, total_workers):
        """
        multiproces wrapper for nmf.factorize()
        factorize_multi_process() is direct wrapper around factorize to be able to launch it form mp.
        total_workers: int; number of workers to use.
        """
        list_args = [(x, total_workers, self) for x in range(total_workers)]
        
        with Pool(total_workers) as p:
            
            p.map(factorize_mp_signature, list_args)
            p.close()
            p.join()    
    
    def factorize(self,
                worker_i=0, total_workers=1,
                ):
        """
        Iteratively run NMF with prespecified parameters.

        Use the `worker_i` and `total_workers` parameters for parallelization.

        Generic kwargs for NMF are loaded from self.run_params, defaults below::

            ``non_negative_factorization`` default arguments:
                alpha=0.0
                l1_ratio=0.0
                beta_loss='kullback-leibler'
                solver='mu'
                tol=1e-4,
                max_iter=200
                regularization=None
                init='random'
                random_state, n_components are both set by the prespecified self.replicate_params.


        Parameters
        ----------
        worker_i : int, optional (default=0)
            Worker index for parallelization. When using multiple workers, each worker
            should have a unique index from 0 to total_workers-1.

        total_workers : int, optional (default=1)
            Total number of workers for parallelization.

            IMPORTANT: If you set total_workers > 1, you MUST run factorize() for each
            worker separately:
                cnmf_obj.factorize(worker_i=0, total_workers=2)
                cnmf_obj.factorize(worker_i=1, total_workers=2)

            For in-memory mode (output_dir=None), it's recommended to use total_workers=1
            and run all iterations in a single call.

        """
        # Use in-memory data
        run_params = self.replicate_params
        norm_counts = self.norm_counts
        _nmf_kwargs = self.run_params.copy()

        # Display compute mode
        if self.use_gpu:
            print(f'🚀 Running NMF on GPU (device {self.gpu_id})')
        else:
            print(f'💻 Running NMF on CPU')

        # Warning for multi-worker setup in memory mode
        if total_workers > 1 and self.output_dir is None:
            print(f'⚠️  Using total_workers={total_workers} in memory mode.')
            print(f'⚠️  You MUST run factorize() for ALL workers (0 to {total_workers-1}), otherwise iterations will be missing!')
            print(f'⚠️  Currently running worker {worker_i}/{total_workers-1}')

        jobs_for_this_worker = worker_filter(range(len(run_params)), worker_i, total_workers)
        from tqdm import tqdm
        jobs_list = list(jobs_for_this_worker)

        if len(jobs_list) == 0:
            print(f'⚠️  No jobs assigned to worker {worker_i}. Check your worker_i and total_workers settings.')
            return

        print(f'Running {len(jobs_list)} factorization iterations for worker {worker_i}...')

        for idx in tqdm(jobs_list):

            p = run_params.iloc[idx, :]
            #print('[Worker %d]. Starting task %d.' % (worker_i, idx))
            _nmf_kwargs['random_state'] = p['nmf_seed']
            _nmf_kwargs['n_components'] = p['n_components']

            (spectra, usages) = self._nmf(norm_counts.X, _nmf_kwargs)
            spectra = pd.DataFrame(spectra,
                                   index=np.arange(1, _nmf_kwargs['n_components']+1),
                                   columns=norm_counts.var.index)

            # Store in memory
            self.iter_spectra[(p['n_components'], p['iter'])] = spectra
            # Optionally save to disk
            if self.output_dir is not None:
                save_df_to_npz(spectra, self.paths['iter_spectra'] % (p['n_components'], p['iter']))

        print(f'✓ Worker {worker_i} completed {len(jobs_list)} iterations. Total in memory: {len(self.iter_spectra)}')


    def combine_nmf(self, k, skip_missing_files=False, remove_individual_iterations=False):
        # Use in-memory data
        run_params = self.replicate_params
        print('Combining factorizations for k=%d.'%k)

        run_params_subset = run_params[run_params.n_components==k].sort_values('iter')
        combined_spectra = []
        missing_count = 0
        total_expected = len(run_params_subset)

        for i,p in run_params_subset.iterrows():
            spectra_key = (p['n_components'], p['iter'])
            # First try to get from memory
            if spectra_key in self.iter_spectra:
                spectra = self.iter_spectra[spectra_key]
                spectra.index = ['iter%d_topic%d' % (p['iter'], t+1) for t in range(k)]
                combined_spectra.append(spectra)
            # Fall back to disk if output_dir is set and file exists
            elif self.output_dir is not None:
                current_file = self.paths['iter_spectra'] % (p['n_components'], p['iter'])
                if not os.path.exists(current_file):
                    missing_count += 1
                    if not skip_missing_files:
                        print('Missing spectra: %s, run with skip_missing=True to override' % str(spectra_key))
                        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(spectra_key))
                else:
                    spectra = load_df_from_npz(current_file)
                    spectra.index = ['iter%d_topic%d' % (p['iter'], t+1) for t in range(k)]
                    combined_spectra.append(spectra)
            else:
                missing_count += 1
                if not skip_missing_files:
                    print('Missing spectra: %s, run with skip_missing=True to override' % str(spectra_key))
                    raise KeyError(f"Spectra not found for {spectra_key}")

        # Print summary
        found_count = len(combined_spectra)
        if missing_count > 0:
            print(f'  Found {found_count}/{total_expected} iterations for k={k} ({missing_count} missing)')
            if missing_count == total_expected // 2:
                print(f'  ⚠️  WARNING: Exactly half the iterations are missing!')
                print(f'  ⚠️  This usually means you used total_workers>1 but only ran one worker.')
                print(f'  ⚠️  Solution: Either run factorize() with total_workers=1, or run all workers.')
        else:
            print(f'  Found all {found_count} iterations for k={k}')

        if len(combined_spectra)>0:
            combined_spectra = pd.concat(combined_spectra, axis=0)
            # Store in memory
            self.merged_spectra_dict[k] = combined_spectra
            # Optionally save to disk
            if self.output_dir is not None:
                save_df_to_npz(combined_spectra, self.paths['merged_spectra']%k)
        else:
            print('No spectra found for k=%d' % k)
        return combined_spectra
    
    
    def refit_usage(self, X, spectra):
        """
        Takes an input data matrix and a fixed spectra and uses NNLS to find the optimal
        usage matrix. Generic kwargs for NMF are loaded from self.run_params.
        If input data are pandas.DataFrame, returns a DataFrame with row index matching X and
        columns index matching index of spectra

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, cells X genes
            Non-negative expression data to fit spectra to

        spectra : pandas.DataFrame or numpy.ndarray, programs X genes
            Non-negative spectra of expression programs
        """

        # Use in-memory run_params
        refit_nmf_kwargs = self.run_params.copy()
        if type(spectra) is pd.DataFrame:
            refit_nmf_kwargs.update(dict(n_components = spectra.shape[0], H = spectra.values, update_H = False))
        else:
            refit_nmf_kwargs.update(dict(n_components = spectra.shape[0], H = spectra, update_H = False))

        _, rf_usages = self._nmf(X, nmf_kwargs=refit_nmf_kwargs)
        if (type(X) is pd.DataFrame) and (type(spectra) is pd.DataFrame):
            rf_usages = pd.DataFrame(rf_usages, index=X.index, columns=spectra.index)

        return(rf_usages)
    
    
    def refit_spectra(self, X, usage):
        """
        Takes an input data matrix and a fixed usage matrix and uses NNLS to find the optimal
        spectra matrix. Generic kwargs for NMF are loaded from self.paths['nmf_run_parameters'].
        If input data are pandas.DataFrame, returns a DataFrame with row index matching X and
        columns index matching index of spectra

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, cells X genes
            Non-negative expression data to fit spectra to

        usage : pandas.DataFrame or numpy.ndarray, cells X programs
            Non-negative spectra of expression programs
        """
        return(self.refit_usage(X.T, usage.T).T)


    def consensus(self, k, density_threshold=0.5, local_neighborhood_size = 0.30,show_clustering = True,
                  skip_density_and_return_after_stats = False, close_clustergram_fig=False,
                  refit_usage=True, normalize_tpm_spectra=False, norm_counts=None):
        """
        Obtain consensus estimates of spectra and usages from a cNMF run and output a clustergram of
        the consensus matrix. Assumes prepare, factorize, and combine steps have already been run.


        Parameters
        ----------
        k : int
            Number of programs (must be within the k values specified in previous steps)

        density_threshold : float (default: 0.5)
            Threshold for filtering outlier spectra. 2.0 or greater applies no filter.
            
        local_neighborhood_size : float (default: 0.3)
            Determines number of neighbors to use for calculating KNN distance as local_neighborhood_size X n_iters

        show_clustering : boolean (default=False)
            If True, generates the consensus clustergram filter

        skip_density_and_return_after_stats : boolean (default=False)
            True when running k_selection_plot to compute stability and error for input parameters without computing
            consensus spectra and usages
            
        close_clustergram_fig : boolean (default=False)
            If True, closes the clustergram figure from output after saving the image to a file
            
        refit_usage : boolean (default=True)
            If True, refit the usage matrix one final time after finalizing the spectra_tpm matrix. Done by regressing 
            the tpm matrix against the tpm_spectra including only high-variance genes and with both the tpm matrix
            and tpm_spectra normalized by the standard deviations of the genes in tpm scale.
            
        normalize_tpm_spectra : boolean (default=False)
            If True, renormalizes the tpm_spectra to sum to 1e6 for each program. This is done before refitting usages.
            If not used, the tpm_spectra are exactly as calcuated when refitting the usage matrix against the tpm matrix
            and typically will not sum to the same value for each program.
            
        norm_counts : AnnData (default=None)
            Speed up calculation of k_selection_plot by avoiding reloading norm_counts for each K. Should not be used by
            most users
        """


        # Use in-memory data
        if k not in self.merged_spectra_dict:
            # Try loading from disk if output_dir is set
            if self.output_dir is not None and os.path.exists(self.paths['merged_spectra']%k):
                merged_spectra = load_df_from_npz(self.paths['merged_spectra']%k)
                self.merged_spectra_dict[k] = merged_spectra
            else:
                raise ValueError(f"Merged spectra for k={k} not found. Run combine() first.")
        else:
            merged_spectra = self.merged_spectra_dict[k]

        if norm_counts is None:
            norm_counts = self.norm_counts

        density_threshold_str = str(density_threshold)
        if skip_density_and_return_after_stats:
            density_threshold_str = '2'
        density_threshold_repl = density_threshold_str.replace('.', '_')
        n_neighbors = int(local_neighborhood_size * merged_spectra.shape[0]/k)

        # Rescale topics such to length of 1.
        l2_spectra = (merged_spectra.T/np.sqrt((merged_spectra**2).sum(axis=1))).T

        if not skip_density_and_return_after_stats:
            # Compute the local density matrix (if not previously cached)
            topics_dist = None
            # Check in-memory cache first
            cache_key = f'local_density_{k}'
            if not hasattr(self, '_local_density_cache'):
                self._local_density_cache = {}

            if cache_key in self._local_density_cache:
                local_density = self._local_density_cache[cache_key]
            # Then check disk cache if output_dir is set
            elif self.output_dir is not None and os.path.isfile(self.paths['local_density_cache'] % k):
                local_density = load_df_from_npz(self.paths['local_density_cache'] % k)
                self._local_density_cache[cache_key] = local_density
            else:
                #   first find the full distance matrix
                topics_dist = euclidean_distances(l2_spectra.values)
                #   partition based on the first n neighbors
                partitioning_order  = np.argpartition(topics_dist, n_neighbors+1)[:, :n_neighbors+1]
                #   find the mean over those n_neighbors (excluding self, which has a distance of 0)
                distance_to_nearest_neighbors = topics_dist[np.arange(topics_dist.shape[0])[:, None], partitioning_order]
                local_density = pd.DataFrame(distance_to_nearest_neighbors.sum(1)/(n_neighbors),
                                             columns=['local_density'],
                                             index=l2_spectra.index)
                # Cache in memory
                self._local_density_cache[cache_key] = local_density
                # Optionally save to disk
                if self.output_dir is not None:
                    save_df_to_npz(local_density, self.paths['local_density_cache'] % k)
                del(partitioning_order)
                del(distance_to_nearest_neighbors)

            # Apply density filtering only if threshold is reasonable
            if density_threshold >= 2.0:
                # Documentation states "2.0 or greater applies no filter"
                density_filter = pd.Series([True] * len(l2_spectra), index=l2_spectra.index)
            else:
                density_filter = local_density.iloc[:, 0] < density_threshold
            
            l2_spectra = l2_spectra.loc[density_filter, :]
            if l2_spectra.shape[0] == 0:
                local_density_stats = local_density.iloc[:, 0].describe()
                raise RuntimeError(
                    f"Zero components remain after density filtering with threshold {density_threshold}.\n"
                    f"Local density statistics:\n{local_density_stats}\n"
                    f"Consider:\n"
                    f"- Increasing density_threshold (try {local_density_stats['75%']:.3f} or higher)\n"
                    f"- Using density_threshold=2.0 or higher to disable filtering\n"
                    f"- Using a smaller local_neighborhood_size parameter"
                )

        kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
        kmeans_model.fit(l2_spectra)
        kmeans_cluster_labels = pd.Series(kmeans_model.labels_+1, index=l2_spectra.index)

        # Find median usage for each gene across cluster
        median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median(numeric_only=True)

        # Normalize median spectra to probability distributions.
        median_spectra = (median_spectra.T/median_spectra.sum(1)).T

        # Obtain reconstructed count matrix by re-fitting usage and computing dot product: usage.dot(spectra)
        rf_usages = self.refit_usage(norm_counts.X, median_spectra)
        rf_usages = pd.DataFrame(rf_usages, index=norm_counts.obs.index, columns=median_spectra.index)        
        
        if skip_density_and_return_after_stats:
            silhouette = silhouette_score(l2_spectra.values, kmeans_cluster_labels, metric='euclidean')
            
            # Compute prediction error as a frobenius norm
            rf_pred_norm_counts = rf_usages.dot(median_spectra)        
            if sp.issparse(norm_counts.X):
                prediction_error = ((norm_counts.X.todense() - rf_pred_norm_counts)**2).sum().sum()
            else:
                prediction_error = ((norm_counts.X - rf_pred_norm_counts)**2).sum().sum()    
                
            consensus_stats = pd.DataFrame([k, density_threshold, silhouette,  prediction_error],
                    index = ['k', 'local_density_threshold', 'silhouette', 'prediction_error'],
                    columns = ['stats'])

            return(consensus_stats)           
        
        # Re-order usage by total contribution
        norm_usages = rf_usages.div(rf_usages.sum(axis=1), axis=0)      
        reorder = norm_usages.sum(axis=0).sort_values(ascending=False)
        rf_usages = rf_usages.loc[:, reorder.index]
        norm_usages = norm_usages.loc[:, reorder.index]
        median_spectra = median_spectra.loc[reorder.index, :]
        rf_usages.columns = np.arange(1, rf_usages.shape[1]+1)
        norm_usages.columns = rf_usages.columns
        median_spectra.index = rf_usages.columns
        
        # Convert spectra to TPM units, and obtain results for all genes by running last step of NMF
        # with usages fixed and TPM as the input matrix
        tpm = self.tpm
        tpm_stats = self.tpm_stats
        spectra_tpm = self.refit_spectra(tpm.X, norm_usages.astype(tpm.X.dtype))
        spectra_tpm = pd.DataFrame(spectra_tpm, index=rf_usages.columns, columns=tpm.var.index)
        if normalize_tpm_spectra:
            spectra_tpm = spectra_tpm.div(spectra_tpm.sum(axis=1), axis=0) * 1e6
                    
        # Convert spectra to Z-score units, and obtain results for all genes by running last step of NMF
        # with usages fixed and Z-scored TPM as the input matrix
        if sp.issparse(tpm.X):
            norm_tpm = (np.array(tpm.X.todense()) - tpm_stats['__mean'].values) / tpm_stats['__std'].values
        else:
            norm_tpm = (tpm.X - tpm_stats['__mean'].values) / tpm_stats['__std'].values
        
        usage_coef = fast_ols_all_cols(rf_usages.values, norm_tpm)
        usage_coef = pd.DataFrame(usage_coef, index=rf_usages.columns, columns=tpm.var.index)
        
        if refit_usage:
            ## Re-fitting usage a final time on std-scaled HVG TPM seems to
            ## increase accuracy on simulated data
            hvgs = self.highvar_genes
            norm_tpm = tpm[:, hvgs]
            if sp.issparse(norm_tpm.X):
                sc.pp.scale(norm_tpm, zero_center=False)
            else:
                norm_tpm.X /= norm_tpm.X.std(axis=0, ddof=1)

            spectra_tpm_rf = spectra_tpm.loc[:,hvgs]

            spectra_tpm_rf = spectra_tpm_rf.div(tpm_stats.loc[hvgs, '__std'], axis=1)
            rf_usages = self.refit_usage(norm_tpm.X, spectra_tpm_rf.astype(norm_tpm.X.dtype))
            rf_usages = pd.DataFrame(rf_usages, index=norm_counts.obs.index, columns=spectra_tpm_rf.index)

        # Store consensus results in memory
        consensus_key = (k, density_threshold)
        self.consensus_results[consensus_key] = {
            'median_spectra': median_spectra,
            'rf_usages': rf_usages,
            'spectra_tpm': spectra_tpm,
            'usage_coef': usage_coef
        }

        # Optionally save to disk
        if self.output_dir is not None:
            save_df_to_npz(median_spectra, self.paths['consensus_spectra']%(k, density_threshold_repl))
            save_df_to_npz(rf_usages, self.paths['consensus_usages']%(k, density_threshold_repl))
            #save_df_to_npz(consensus_stats, self.paths['consensus_stats']%(k, density_threshold_repl))
            save_df_to_text(median_spectra, self.paths['consensus_spectra__txt']%(k, density_threshold_repl))
            save_df_to_text(rf_usages, self.paths['consensus_usages__txt']%(k, density_threshold_repl))
            save_df_to_npz(spectra_tpm, self.paths['gene_spectra_tpm']%(k, density_threshold_repl))
            save_df_to_text(spectra_tpm, self.paths['gene_spectra_tpm__txt']%(k, density_threshold_repl))
            save_df_to_npz(usage_coef, self.paths['gene_spectra_score']%(k, density_threshold_repl))
            save_df_to_text(usage_coef, self.paths['gene_spectra_score__txt']%(k, density_threshold_repl))
        if show_clustering:
            if topics_dist is None:
                topics_dist = euclidean_distances(l2_spectra.values)
                # (l2_spectra was already filtered using the density filter)
            else:
                # (but the previously computed topics_dist was not!)
                topics_dist = topics_dist[density_filter.values, :][:, density_filter.values]


            spectra_order = []
            for cl in sorted(set(kmeans_cluster_labels)):

                cl_filter = kmeans_cluster_labels==cl

                if cl_filter.sum() > 1:
                    cl_dist = squareform(topics_dist[cl_filter, :][:, cl_filter], checks=False)
                    cl_dist[cl_dist < 0] = 0 #Rarely get floating point arithmetic issues
                    from fastcluster import linkage
                    cl_link = linkage(cl_dist, 'average')
                    cl_leaves_order = leaves_list(cl_link)

                    spectra_order += list(np.where(cl_filter)[0][cl_leaves_order])
                else:
                    ## Corner case where a component only has one element
                    spectra_order += list(np.where(cl_filter)[0])


            from matplotlib import gridspec
            import matplotlib.pyplot as plt

            width_ratios = [0.5, 9, 0.5, 4, 1]
            height_ratios = [0.5, 9]
            fig = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios), fig,
                                    0.01, 0.01, 0.98, 0.98,
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios,
                                   wspace=0, hspace=0)

            dist_ax = fig.add_subplot(gs[1,1], xscale='linear', yscale='linear',
                                      xticks=[], yticks=[],xlabel='', ylabel='',
                                      frameon=True)

            D = topics_dist[spectra_order, :][:, spectra_order]
            dist_im = dist_ax.imshow(D, interpolation='none', cmap='viridis',
                                     aspect='auto', rasterized=True)

            left_ax = fig.add_subplot(gs[1,0], xscale='linear', yscale='linear', xticks=[], yticks=[],
                xlabel='', ylabel='', frameon=True)
            left_ax.imshow(kmeans_cluster_labels.values[spectra_order].reshape(-1, 1),
                            interpolation='none', cmap='Spectral', aspect='auto',
                            rasterized=True)


            top_ax = fig.add_subplot(gs[0,1], xscale='linear', yscale='linear', xticks=[], yticks=[],
                xlabel='', ylabel='', frameon=True)
            top_ax.imshow(kmeans_cluster_labels.values[spectra_order].reshape(1, -1),
                              interpolation='none', cmap='Spectral', aspect='auto',
                                rasterized=True)


            hist_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 3],
                                   wspace=0, hspace=0)

            hist_ax = fig.add_subplot(hist_gs[0,0], xscale='linear', yscale='linear',
                xlabel='', ylabel='', frameon=True, title='Local density histogram')
            hist_ax.hist(local_density.values, bins=np.linspace(0, 1, 50))
            hist_ax.yaxis.tick_right()

            xlim = hist_ax.get_xlim()
            ylim = hist_ax.get_ylim()
            if density_threshold < xlim[1]:
                hist_ax.axvline(density_threshold, linestyle='--', color='k')
                hist_ax.text(density_threshold  + 0.02, ylim[1] * 0.95, 'filtering\nthreshold\n\n', va='top')
            hist_ax.set_xlim(xlim)
            hist_ax.set_xlabel('Mean distance to k nearest neighbors\n\n%d/%d (%.0f%%) spectra above threshold\nwere removed prior to clustering'%(sum(~density_filter), len(density_filter), 100*(~density_filter).mean()))
            
            ## Add colorbar
            cbar_gs = gridspec.GridSpecFromSubplotSpec(8, 1, subplot_spec=hist_gs[1, 0],
                                   wspace=0, hspace=0)
            cbar_ax = fig.add_subplot(cbar_gs[4,0], xscale='linear', yscale='linear',
                xlabel='', ylabel='', frameon=True, title='Euclidean Distance')
            vmin = D.min().min()
            vmax = D.max().max()
            fig.colorbar(dist_im, cax=cbar_ax,
            ticks=np.linspace(vmin, vmax, 3),
            orientation='horizontal')
            
            
            #hist_ax.hist(local_density.values, bins=np.linspace(0, 1, 50))
            #hist_ax.yaxis.tick_right()            

            if self.output_dir is not None:
                fig.savefig(self.paths['clustering_plot']%(k, density_threshold_repl), dpi=250)
            if close_clustergram_fig:
                plt.close(fig)

            # Store clustering results
            self.topic_dist = topics_dist
            self.spectra_order = spectra_order
            self.local_density = local_density
            self.kmeans_cluster_labels = kmeans_cluster_labels
        else:
            # When show_clustering=False, still store what we have
            self.local_density = local_density if not skip_density_and_return_after_stats else None
            self.kmeans_cluster_labels = kmeans_cluster_labels


    def calculate_silhouette_k(self, k, density_threshold=2.0, use_l2_spectra=True):
        """
        Calculate silhouette scores for a given k value.

        Parameters
        ----------
        k : int
            Number of programs

        density_threshold : float, optional (default=2.0)
            Density threshold for filtering spectra

        use_l2_spectra : bool, optional (default=True)
            If True, use L2-normalized spectra for distance calculation.
            If False, use consensus usages from cells.

        Returns
        -------
        dict
            Dictionary containing:
            - 'avg_silhouette': Average silhouette score
            - 'silhouette_values': Per-sample silhouette scores
            - 'cluster_labels': Cluster labels
            - 'n_clusters': Number of clusters
        """
        from sklearn.metrics import silhouette_score, silhouette_samples

        # Get merged spectra
        if k not in self.merged_spectra_dict:
            if self.output_dir is not None and os.path.exists(self.paths['merged_spectra']%k):
                merged_spectra = load_df_from_npz(self.paths['merged_spectra']%k)
                self.merged_spectra_dict[k] = merged_spectra
            else:
                raise ValueError(f"Merged spectra for k={k} not found. Run combine() first.")
        else:
            merged_spectra = self.merged_spectra_dict[k]

        # L2 normalize spectra
        l2_spectra = (merged_spectra.T/np.sqrt((merged_spectra**2).sum(axis=1))).T

        # Apply density filtering if needed
        if density_threshold < 2.0:
            cache_key = f'local_density_{k}'
            if not hasattr(self, '_local_density_cache'):
                self._local_density_cache = {}

            if cache_key in self._local_density_cache:
                local_density = self._local_density_cache[cache_key]
            elif self.output_dir is not None and os.path.isfile(self.paths['local_density_cache'] % k):
                local_density = load_df_from_npz(self.paths['local_density_cache'] % k)
                self._local_density_cache[cache_key] = local_density
            else:
                # Calculate local density
                from sklearn.metrics.pairwise import euclidean_distances
                local_neighborhood_size = 0.30
                n_neighbors = int(local_neighborhood_size * merged_spectra.shape[0]/k)
                topics_dist = euclidean_distances(l2_spectra.values)
                partitioning_order = np.argpartition(topics_dist, n_neighbors+1)[:, :n_neighbors+1]
                distance_to_nearest_neighbors = topics_dist[np.arange(topics_dist.shape[0])[:, None], partitioning_order]
                local_density = pd.DataFrame(distance_to_nearest_neighbors.sum(1)/(n_neighbors),
                                             columns=['local_density'],
                                             index=l2_spectra.index)
                self._local_density_cache[cache_key] = local_density

            density_filter = local_density.iloc[:, 0] < density_threshold
            l2_spectra = l2_spectra.loc[density_filter, :]

        # Perform K-means clustering
        from sklearn.cluster import KMeans
        kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
        cluster_labels = kmeans_model.fit_predict(l2_spectra.values)

        # Calculate silhouette scores
        avg_silhouette = silhouette_score(l2_spectra.values, cluster_labels, metric='euclidean')
        silhouette_values = silhouette_samples(l2_spectra.values, cluster_labels, metric='euclidean')

        return {
            'avg_silhouette': avg_silhouette,
            'silhouette_values': silhouette_values,
            'cluster_labels': cluster_labels,
            'n_clusters': k,
            'l2_spectra': l2_spectra
        }


    def plot_silhouette_for_k(self, k, density_threshold=2.0, ax=None, show_avg=True,
                              cmap='Spectral', figsize=(8, 6)):
        """
        Plot detailed silhouette plot for a single k value.

        Parameters
        ----------
        k : int
            Number of programs

        density_threshold : float, optional (default=2.0)
            Density threshold for filtering spectra

        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        show_avg : bool, optional (default=True)
            If True, shows average silhouette score line

        cmap : str, optional (default='Spectral')
            Colormap for clusters

        figsize : tuple, optional (default=(8, 6))
            Figure size if creating new figure

        Returns
        -------
        fig, ax
            Figure and axes objects
        """
        # Calculate silhouette scores
        sil_data = self.calculate_silhouette_k(k, density_threshold)

        silhouette_values = sil_data['silhouette_values']
        cluster_labels = sil_data['cluster_labels']
        avg_silhouette = sil_data['avg_silhouette']

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Set up y-axis
        y_lower = 10

        # Get colormap - compatible with different matplotlib versions
        import matplotlib.pyplot as plt
        try:
            # Try matplotlib.pyplot.get_cmap (works in most versions)
            cmap_obj = plt.get_cmap(cmap)
        except AttributeError:
            try:
                # Try new API (matplotlib >= 3.7)
                import matplotlib as mpl
                cmap_obj = mpl.colormaps[cmap]
            except (AttributeError, KeyError):
                # Last resort: direct import
                from matplotlib.colors import get_cmap
                cmap_obj = get_cmap(cmap)

        colors = cmap_obj(np.linspace(0, 1, k))

        for i in range(k):
            # Aggregate silhouette scores for samples in cluster i
            ith_cluster_silhouette_values = silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=colors[i], edgecolor=colors[i], alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title(f'Silhouette Plot for k={k}')
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster Label')

        # The vertical line for average silhouette score of all the values
        if show_avg:
            ax.axvline(x=avg_silhouette, color="red", linestyle="--", linewidth=2,
                      label=f'Avg: {avg_silhouette:.3f}')
            ax.legend(loc='best')

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks(np.arange(-0.2, 1.0, 0.2))
        ax.set_xlim([-0.2, 1])

        return fig, ax


    def plot_silhouette_survey(self, k_range=None, density_threshold=2.0,
                               ncols=3, figsize=None, cmap='Spectral',
                               show_avg=True, title=None, save_path=None):
        """
        Generate a grid of silhouette plots for multiple k values for easy comparison.

        This automated visualization method generates detailed silhouette plots for each
        k value in the range and arranges them in a grid layout, allowing easy visual
        comparison of clustering quality across different factorization ranks.

        Parameters
        ----------
        k_range : list or range, optional
            Range of k values to plot. If None, uses all k values from run_params.

        density_threshold : float, optional (default=2.0)
            Density threshold for filtering spectra

        ncols : int, optional (default=3)
            Number of columns in the grid

        figsize : tuple, optional
            Figure size. If None, automatically calculated based on grid size.

        cmap : str, optional (default='Spectral')
            Colormap for clusters

        show_avg : bool, optional (default=True)
            If True, shows average silhouette score line on each subplot

        title : str, optional
            Overall figure title. If None, uses default.

        save_path : str, optional
            If provided, saves the figure to this path

        Returns
        -------
        fig, axes
            Figure and axes array

        Examples
        --------
        >>> # Compare silhouette plots for k=3 to k=10
        >>> fig, axes = cnmf_obj.plot_silhouette_survey(k_range=range(3, 11))
        >>>
        >>> # Custom layout with 2 columns
        >>> fig, axes = cnmf_obj.plot_silhouette_survey(k_range=[5, 6, 7, 8], ncols=2)
        """
        # Determine k values to plot
        if k_range is None:
            run_params = self.replicate_params
            k_range = sorted(set(run_params.n_components))
        else:
            k_range = list(k_range)

        n_plots = len(k_range)
        nrows = int(np.ceil(n_plots / ncols))

        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (6 * ncols, 5 * nrows)

        # Create figure and axes
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Plot each k
        for idx, k in enumerate(k_range):
            try:
                self.plot_silhouette_for_k(k, density_threshold=density_threshold,
                                          ax=axes[idx], show_avg=show_avg, cmap=cmap)
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error for k={k}:\n{str(e)}',
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'k={k} (Error)')

        # Hide extra subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        # Set overall title
        if title is None:
            title = f'Silhouette Analysis Survey (density_threshold={density_threshold})'
        fig.suptitle(title, fontsize=16, y=0.995)

        plt.tight_layout()

        # Save if requested
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Silhouette survey plot saved to: {save_path}')

        return fig, axes


    def k_selection_plot(self, close_fig=False, include_silhouette=True,
                        density_threshold=2.0):
        '''
        Plot stability, reconstruction error, and optionally silhouette scores for K selection.

        Borrowed from Alexandrov Et Al. 2013 Deciphering Mutational Signatures
        publication in Cell Reports, with additional silhouette analysis option.

        Parameters
        ----------
        close_fig : bool, optional (default=False)
            If True, closes the figure after saving

        include_silhouette : bool, optional (default=True)
            If True, adds average silhouette score as a third metric on the plot

        density_threshold : float, optional (default=2.0)
            Density threshold for filtering spectra when calculating silhouette scores

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        '''
        # Use in-memory data
        run_params = self.replicate_params
        stats = []
        norm_counts = self.norm_counts
        k_values = sorted(set(run_params.n_components))

        for k in k_values:
            stats.append(self.consensus(k, skip_density_and_return_after_stats=True,
                                        show_clustering=False, close_clustergram_fig=True,
                                        norm_counts=norm_counts).stats)

        stats = pd.DataFrame(stats)
        stats.reset_index(drop = True, inplace = True)

        # Calculate silhouette scores if requested
        if include_silhouette:
            print("Calculating silhouette scores for K selection...")
            silhouette_scores = []
            for k in k_values:
                try:
                    sil_data = self.calculate_silhouette_k(k, density_threshold=density_threshold)
                    silhouette_scores.append(sil_data['avg_silhouette'])
                except Exception as e:
                    print(f"Warning: Could not calculate silhouette for k={k}: {e}")
                    silhouette_scores.append(np.nan)
            stats['avg_silhouette'] = silhouette_scores

        # Optionally save to disk
        if self.output_dir is not None:
            save_df_to_npz(stats, self.paths['k_selection_stats'])

        # Create figure with appropriate layout
        if include_silhouette and not stats['avg_silhouette'].isna().all():
            # Three metrics: use larger figure with three y-axes
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            ax3 = ax1.twinx()

            # Offset the third axis
            ax3.spines['right'].set_position(('outward', 60))

            # Plot stability (silhouette from consensus - kept as "Stability")
            p1 = ax1.plot(stats.k, stats.silhouette, 'o-', color='b', linewidth=2,
                         markersize=8, label='Stability')
            ax1.set_ylabel('Stability', color='b', fontsize=13)
            ax1.tick_params(axis='y', labelcolor='b')
            for tl in ax1.get_yticklabels():
                tl.set_color('b')

            # Plot reconstruction error
            p2 = ax2.plot(stats.k, stats.prediction_error, 'o-', color='r', linewidth=2,
                         markersize=8, label='Reconstruction Error')
            ax2.set_ylabel('Reconstruction Error', color='r', fontsize=13)
            ax2.tick_params(axis='y', labelcolor='r')
            for tl in ax2.get_yticklabels():
                tl.set_color('r')

            # Plot average silhouette score
            p3 = ax3.plot(stats.k, stats.avg_silhouette, 's-', color='green', linewidth=2,
                         markersize=8, label='Avg Silhouette')
            ax3.set_ylabel('Average Silhouette Score', color='green', fontsize=13)
            ax3.tick_params(axis='y', labelcolor='green')
            ax3.set_ylim([max(0, stats.avg_silhouette.min() - 0.1),
                         min(1, stats.avg_silhouette.max() + 0.1)])
            for tl in ax3.get_yticklabels():
                tl.set_color('green')

            ax1.set_xlabel('Number of Components (K)', fontsize=13)
            ax1.grid(True, alpha=0.3)

            # Add legend
            lines = p1 + p2 + p3
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best', fontsize=10)

            fig.suptitle('cNMF K Selection Metrics', fontsize=14, fontweight='bold')
        else:
            # Two metrics only: original layout
            fig = plt.figure(figsize=(8, 5))
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()

            ax1.plot(stats.k, stats.silhouette, 'o-', color='b', linewidth=2, markersize=8)
            ax1.set_ylabel('Stability', color='b', fontsize=13)
            for tl in ax1.get_yticklabels():
                tl.set_color('b')

            ax2.plot(stats.k, stats.prediction_error, 'o-', color='r', linewidth=2, markersize=8)
            ax2.set_ylabel('Reconstruction Error', color='r', fontsize=13)
            for tl in ax2.get_yticklabels():
                tl.set_color('r')

            ax1.set_xlabel('Number of Components (K)', fontsize=13)
            ax1.grid(True, alpha=0.3)
            fig.suptitle('cNMF K Selection Metrics', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if self.output_dir is not None:
            fig.savefig(self.paths['k_selection_plot'], dpi=250, bbox_inches='tight')
        if close_fig:
            plt.close(fig)

        return fig
            
            
    def load_results(self, K, density_threshold, n_top_genes=100, norm_usage = True):
        """
        Loads normalized usages and gene_spectra_scores for a given choice of K and
        local_density_threshold for the cNMF run. Additionally returns a DataFrame of
        the top genes linked to each program

        Parameters
        ----------
        K : int
            Number of programs (must be within the k values specified in previous steps)

        density_threshold : float
            Threshold for filtering outlier spectra (must be within the values specified in consensus step)

        n_top_genes : integer, optional (default=100)
            Number of top genes per program to return

        norm_usage : boolean, optional (default=True)
            If True, normalize cNMF usages to sum to 1

        Returns
        ----------
        usage - cNMF usages (cells X K)
        spectra_scores - Z-score coeffecients for each program (K x genes) with high values cooresponding
                    to better marker genes
        spectra_tpm - Coeffecients for contribution of each gene to each program (K x genes) in TPM units
        top_genes - ranked list of marker genes per GEP (n_top_genes X K)
        """
        # First try to load from memory
        consensus_key = (K, density_threshold)
        if consensus_key in self.consensus_results:
            consensus_data = self.consensus_results[consensus_key]
            spectra_scores = consensus_data['usage_coef'].T
            spectra_tpm = consensus_data['spectra_tpm'].T
            usage = consensus_data['rf_usages']
        # Fall back to disk if output_dir is set
        elif self.output_dir is not None:
            scorefn = self.paths['gene_spectra_score__txt'] % (K, str(density_threshold).replace('.', '_'))
            tpmfn = self.paths['gene_spectra_tpm__txt'] % (K, str(density_threshold).replace('.', '_'))
            usagefn = self.paths['consensus_usages__txt'] % (K, str(density_threshold).replace('.', '_'))
            spectra_scores = pd.read_csv(scorefn, sep='\t', index_col=0).T
            spectra_tpm = pd.read_csv(tpmfn, sep='\t', index_col=0).T
            usage = pd.read_csv(usagefn, sep='\t', index_col=0)
        else:
            raise ValueError(f"Results for K={K}, density_threshold={density_threshold} not found. Run consensus() first.")
        
        if norm_usage:
            usage = usage.div(usage.sum(axis=1), axis=0)
        
        try:
            usage.columns = [int(x) for x in usage.columns]
        except:
            print('Usage matrix columns include non integer values')
    
        top_genes = []
        for gep in spectra_scores.columns:
            top_genes.append(list(spectra_scores.sort_values(by=gep, ascending=False).index[:n_top_genes]))
        
        top_genes = pd.DataFrame(top_genes, index=spectra_scores.columns).T
        usage.columns = ['cNMF_%d' % i for i in usage.columns]
        result_dict = {}
        result_dict['usage_norm'] = usage
        result_dict['gep_scores'] = spectra_scores
        result_dict['gep_tpm'] = spectra_tpm
        result_dict['top_genes'] = top_genes
        return result_dict

    def _align_results_to_adata(self, adata, result_dict):
        usage = result_dict['usage_norm']
        gep_scores = result_dict['gep_scores']

        if not adata.obs_names.is_unique:
            raise ValueError("adata.obs_names must be unique before mapping cNMF results.")
        if not usage.index.is_unique:
            raise ValueError("result_dict['usage_norm'].index must be unique before mapping cNMF results.")
        if not adata.var_names.is_unique:
            raise ValueError("adata.var_names must be unique before mapping cNMF gene scores.")
        if not gep_scores.index.is_unique:
            raise ValueError("result_dict['gep_scores'].index must be unique before mapping cNMF gene scores.")

        missing_cells = adata.obs_names.difference(usage.index)
        if len(missing_cells) > 0:
            n_missing = len(missing_cells)
            noun = "cell" if n_missing == 1 else "cells"
            examples = ", ".join(map(str, missing_cells[:5]))
            raise ValueError(
                "adata.obs_names must all be present in result_dict['usage_norm'].index "
                f"before mapping cNMF results. Missing {n_missing} {noun}; "
                f"examples: {examples}"
            )

        missing_genes = adata.var_names.difference(gep_scores.index)
        if len(missing_genes) > 0:
            n_missing = len(missing_genes)
            noun = "gene" if n_missing == 1 else "genes"
            examples = ", ".join(map(str, missing_genes[:5]))
            raise ValueError(
                "adata.var_names must all be present in result_dict['gep_scores'].index "
                f"before mapping cNMF gene scores. Missing {n_missing} {noun}; "
                f"examples: {examples}"
            )

        generated_obs_columns = {'cNMF_cluster', 'cNMF_cluster_rfc', 'cNMF_cluster_clf'}
        cNMF_columns = adata.obs.columns.astype(str).str.match(r'^cNMF_\d+$')
        stale_columns = adata.obs.columns[cNMF_columns | adata.obs.columns.isin(generated_obs_columns)]
        if len(stale_columns) > 0:
            adata.obs = adata.obs.drop(columns=stale_columns)

        usage = usage.reindex(adata.obs_names)
        gep_scores = gep_scores.reindex(adata.var_names)
        stale_gene_score_columns = [
            column
            for column in adata.var.columns
            if isinstance(column, (int, np.integer)) and column >= 1
        ]
        adata.var = adata.var.drop(columns=stale_gene_score_columns)
        for column in usage.columns:
            adata.obs[column] = usage[column].to_numpy()
        for column in gep_scores.columns:
            adata.var[column] = gep_scores[column].to_numpy()
        return usage

    def get_results(self,adata,result_dict):
        usage = self._align_results_to_adata(adata, result_dict)
        print('gene scores are added to adata.var')
        df=adata.obs[usage.columns].copy()
        max_topic = df.idxmax(axis=1)
        # 将结果添加到DataFrame中
        adata.obs['cNMF_cluster'] = max_topic
        print('cNMF_cluster is added to adata.obs')

    def get_results_rfc(self,adata,result_dict,use_rep='STAGATE',cNMF_threshold=0.5):
        """
        Map cNMF results to ``adata`` and train RFC/decision-tree cluster labels.

        This replaces existing cNMF usage and generated cNMF cluster columns,
        including ``cNMF_cluster`` from ``get_results``.
        """
        self._align_results_to_adata(adata, result_dict)

        import numpy as np
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        new_array = []
        class_array = []
        for i in range(1, result_dict['usage_norm'].shape[1] + 1):
            data = adata[adata.obs[f'cNMF_{i}'] > cNMF_threshold].obsm[use_rep]
            data = data.toarray() if hasattr(data, "toarray") else np.asarray(data)
            if data.shape[0] == 0:
                continue
            new_array.append(data)
            class_array.append(np.full(data.shape[0], i))

        if len(new_array) < 2:
            raise ValueError(
                "At least two cNMF factors must have cells above cNMF_threshold "
                "to train cNMF random forest classifiers."
            )

        new_array = np.concatenate(new_array, axis=0)
        class_array = np.concatenate(class_array)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(new_array,class_array,test_size=0.3)
        clf = DecisionTreeClassifier(random_state=0)
        rfc = RandomForestClassifier(random_state=0)
        clf = clf.fit(Xtrain,Ytrain)
        rfc = rfc.fit(Xtrain,Ytrain)
        #查看模型效果
        score_c = clf.score(Xtest,Ytest)
        score_r = rfc.score(Xtest,Ytest)
        #打印最后结果
        print("Single Tree:",score_c)
        print("Random Forest:",score_r)

        prediction_data = adata.obsm[use_rep]
        prediction_data = prediction_data.toarray() if hasattr(prediction_data, "toarray") else np.asarray(prediction_data)
        adata.obs['cNMF_cluster_rfc']=[str(i) for i in rfc.predict(prediction_data)]
        adata.obs['cNMF_cluster_clf']=[str(i) for i in clf.predict(prediction_data)]
        print('cNMF_cluster_rfc is added to adata.obs')
        print('cNMF_cluster_clf is added to adata.obs')

    def save(self, filename):
        """
        Save the cNMF object to a file for later use.

        This saves all in-memory data, parameters, and analysis results,
        allowing you to resume analysis without re-running factorization.

        Parameters
        ----------
        filename : str
            Path to save the cNMF object. Recommended extension: .cnmf or .pkl

        Examples
        --------
        >>> # Save after factorization
        >>> cnmf_obj.factorize()
        >>> cnmf_obj.combine()
        >>> cnmf_obj.save('my_cnmf_analysis.cnmf')
        >>>
        >>> # Later, load and continue
        >>> cnmf_obj = ov.single.cNMF.load('my_cnmf_analysis.cnmf')
        >>> cnmf_obj.consensus(k=7, density_threshold=2.0)
        """
        import pickle

        # Prepare state dictionary
        state = {
            # Basic parameters
            'output_dir': self.output_dir,
            'name': self.name,
            'paths': self.paths,

            # In-memory data
            'tpm': self.tpm,
            'tpm_stats': self.tpm_stats,
            'norm_counts': self.norm_counts,
            'replicate_params': self.replicate_params,
            'run_params': self.run_params,
            'highvar_genes': self.highvar_genes,

            # Analysis results
            'iter_spectra': self.iter_spectra,
            'merged_spectra_dict': self.merged_spectra_dict,
            'consensus_results': self.consensus_results,

            # Optional results
            '_local_density_cache': getattr(self, '_local_density_cache', None),
            'topic_dist': getattr(self, 'topic_dist', None),
            'spectra_order': getattr(self, 'spectra_order', None),
            'local_density': getattr(self, 'local_density', None),
            'kmeans_cluster_labels': getattr(self, 'kmeans_cluster_labels', None),
        }

        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✓ cNMF object saved to: {filename}")
        print(f"  - {len(self.iter_spectra)} factorization iterations")
        print(f"  - {len(self.merged_spectra_dict)} merged spectra (K values)")
        print(f"  - {len(self.consensus_results)} consensus results")


    @classmethod
    def load(cls, filename):
        """
        Load a saved cNMF object from file.

        Parameters
        ----------
        filename : str
            Path to the saved cNMF object file

        Returns
        -------
        cnmf_obj : cNMF
            Loaded cNMF object with all data and results restored

        Examples
        --------
        >>> # Load saved analysis
        >>> cnmf_obj = ov.single.cNMF.load('my_cnmf_analysis.cnmf')
        >>>
        >>> # Continue analysis
        >>> cnmf_obj.consensus(k=7, density_threshold=2.0)
        >>> result_dict = cnmf_obj.load_results(K=7, density_threshold=2.0)
        """
        import pickle

        # Load state from file
        with open(filename, 'rb') as f:
            state = pickle.load(f)

        # Create a minimal cNMF object (skip __init__ to avoid re-initialization)
        obj = cls.__new__(cls)

        # Restore basic parameters
        obj.output_dir = state['output_dir']
        obj.name = state['name']
        obj.paths = state['paths']

        # Restore in-memory data
        obj.tpm = state['tpm']
        obj.tpm_stats = state['tpm_stats']
        obj.norm_counts = state['norm_counts']
        obj.replicate_params = state['replicate_params']
        obj.run_params = state['run_params']
        obj.highvar_genes = state['highvar_genes']

        # Restore analysis results
        obj.iter_spectra = state['iter_spectra']
        obj.merged_spectra_dict = state['merged_spectra_dict']
        obj.consensus_results = state['consensus_results']

        # Restore optional results
        if state.get('_local_density_cache'):
            obj._local_density_cache = state['_local_density_cache']
        if state.get('topic_dist') is not None:
            obj.topic_dist = state['topic_dist']
        if state.get('spectra_order') is not None:
            obj.spectra_order = state['spectra_order']
        if state.get('local_density') is not None:
            obj.local_density = state['local_density']
        if state.get('kmeans_cluster_labels') is not None:
            obj.kmeans_cluster_labels = state['kmeans_cluster_labels']

        print(f"✓ cNMF object loaded from: {filename}")
        print(f"  - {len(obj.iter_spectra)} factorization iterations")
        print(f"  - {len(obj.merged_spectra_dict)} merged spectra (K values)")
        print(f"  - {len(obj.consensus_results)} consensus results")

        return obj
