"""
Copy from pegasus and cellual

"""

from typing import Union, Tuple, Optional, Sequence, List, Dict
import anndata
import numpy as np
import pandas as pd
import skmisc.loess as sl
import scanpy as sc
import time 

from scipy.sparse import issparse, csr_matrix

from ..utils import load_signatures_from_file,predefined_signatures
from .._registry import register_function
from .._settings import settings,print_gpu_usage_color,EMOJI,Colors,add_reference
from .._oom_compat import oom_guard as _oom_guard


from ._normalization import normalize_total,log1p
from datetime import datetime

from .._monitor import monitor
from ..report._provenance import tracked, note, pick_color_key


# Helper functions for Rust anndata compatibility
def _safe_copy(arr):
    """Safely copy array data, compatible with both numpy and Rust backends."""
    try:
        # Try the standard copy method first
        return arr.copy()
    except AttributeError:
        # For Rust backends that don't have copy method
        return np.array(arr, copy=True)

def _safe_to_df_copy(arr):
    """Safely convert to DataFrame and copy, compatible with both numpy and Rust backends."""
    try:
        # Try the standard to_df().copy() first
        return arr.to_df().copy()
    except AttributeError:
        # For Rust backends, convert to numpy first, then to DataFrame
        if hasattr(arr, 'to_numpy'):
            return pd.DataFrame(arr.to_numpy()).copy()
        else:
            return pd.DataFrame(np.array(arr)).copy()



# Emoji map for UMAP status reporting

@monitor
@register_function(
    aliases=["稳健基因", "robust_genes", "identify_robust_genes", "基因过滤", "expressed_genes"],
    category="preprocessing",
    description="Identify robust genes expressed in enough cells and initialize HVG candidates",
    prerequisites={
        "optional_functions": ["qc", "filter_cells", "filter_genes"]
    },
    requires={},
    produces={
        "var": ["n_cells", "percent_cells", "robust", "highly_variable_features"]
    },
    auto_fix="none",
    examples=[
        "ov.pp.identify_robust_genes(adata, percent_cells=0.05)",
        "adata.var['robust'].value_counts()",
    ],
    related=["highly_variable_features", "select_hvf_pegasus", "qc"],
)
def identify_robust_genes(data: anndata.AnnData, percent_cells: float = 0.05) -> None:
    r"""Identify robust genes for downstream HVG selection.

    Parameters
    ----------
    data : anndata.AnnData
        AnnData object containing a gene expression matrix in ``.X``.
        Genes with zero counts across all cells are removed.
    percent_cells : float, default=0.05
        Minimum percentage of cells in which a gene must be detected to be
        marked as ``robust``.

    Returns
    -------
    None
        Updates ``data.var`` with ``n_cells``, ``percent_cells``, ``robust``,
        and initializes ``highly_variable_features`` from ``robust``.
    """

    prior_n = data.shape[1]

    # OOM goes through chunked_identify_robust_genes in preprocess(); this
    # function only runs for regular AnnData.
    if issparse(data.X):
        data.var["n_cells"] = data.X.getnnz(axis=0)
        data._inplace_subset_var(data.var["n_cells"] > 0)
        data.var["percent_cells"] = (data.var["n_cells"] / data.shape[0]) * 100
        data.var["robust"] = data.var["percent_cells"] >= percent_cells
    else:
        data.var["robust"] = True

    data.var["highly_variable_features"] = data.var["robust"]  
    # default all robust genes are "highly" variable
    print(f"{Colors.BLUE}    After filtration, {data.shape[1]}/{prior_n} genes are kept.{Colors.ENDC}")
    print(f"{Colors.BLUE}    Among {data.shape[1]} genes, {data.var['robust'].sum()} genes are robust.{Colors.ENDC}")

def calc_mean_and_var(X: Union[csr_matrix, np.ndarray], axis: int) -> Tuple[np.ndarray, np.ndarray]:
    if issparse(X):
        #from ..cylib.fast_utils import calc_mean_and_var_sparse
        return calc_mean_and_var_sparse(X.shape[0], X.shape[1], X.data, X.indices, X.indptr, axis)
    else:
        #from ..cylib.fast_utils import calc_mean_and_var_dense
        return calc_mean_and_var_dense(X.shape[0], X.shape[1], X, axis)
def calc_stat_per_batch(X: Union[csr_matrix, np.ndarray], batch: \
    Union[pd.Categorical, np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from pandas.api.types import is_categorical_dtype
    if is_categorical_dtype(batch):
        nbatch = batch.categories.size
        codes = batch.codes.astype(np.int32)
    else:
        codes = np.array(batch, dtype = np.int32)
        nbatch = codes.max() + 1 # assume cluster label starts from 0

    if issparse(X):
        #from ..cylib.fast_utils import calc_stat_per_batch_sparse
        return calc_stat_per_batch_sparse(X.shape[0], X.shape[1], X.data, X.indices, X.indptr, nbatch, codes)
    else:
        #from ..cylib.fast_utils import calc_stat_per_batch_dense
        return calc_stat_per_batch_dense(X.shape[0], X.shape[1], X, nbatch, codes)

def estimate_feature_statistics(data: anndata.AnnData, batch: str) -> None:
    r"""Estimate feature (gene) statistics per channel, such as mean, var etc.
    
    Arguments:
        data: AnnData object
        batch: Batch column name in data.obs
    
    Returns:
        None: Updates data.var with mean and variance statistics
    """
    if batch is None:
        data.var["mean"], data.var["var"] = calc_mean_and_var(data.X, axis=0)
    else:
        ncells, means, partial_sum = calc_stat_per_batch(data.X, data.obs[batch].values)
        partial_sum[partial_sum < 1e-6] = 0.0

        data.uns["ncells"] = ncells
        data.varm["means"] = means
        data.varm["partial_sum"] = partial_sum

        data.var["mean"] = np.dot(means, ncells) / data.shape[0]
        data.var["var"] = partial_sum.sum(axis=1) / (data.shape[0] - 1.0)



@register_function(
    aliases=["HVG选择", "pegasus_hvg", "select_hvf_pegasus", "高变基因选择", "高变特征"],
    category="preprocessing",
    description="Select highly variable features using Pegasus loess-based variance modeling",
    prerequisites={
        "functions": ["identify_robust_genes"],
        "optional_functions": ["log1p"],
    },
    requires={
        "var": ["robust"]
    },
    produces={
        "var": ["hvf_loess", "hvf_rank", "highly_variable_features"]
    },
    auto_fix="none",
    examples=[
        "ov.pp.identify_robust_genes(adata, percent_cells=0.05)",
        "ov.pp.select_hvf_pegasus(adata, batch='sample', n_top=2000, span=0.02)",
    ],
    related=["highly_variable_features", "select_hvf_seurat", "remove_cc_genes"],
)
def select_hvf_pegasus(
    data: anndata.AnnData, batch: str, n_top: int = 2000, span: float = 0.02
) -> None:
    r"""Select highly variable features with the Pegasus strategy.

    Parameters
    ----------
    data : anndata.AnnData
        AnnData object with robust-gene annotations in ``data.var['robust']``.
    batch : str
        Batch column in ``data.obs`` used for cross-batch statistics. Set to
        ``None`` to ignore batch effects.
    n_top : int, default=2000
        Number of top-ranked highly variable features to retain.
    span : float, default=0.02
        Loess smoothing span for fitting the mean-variance trend.

    Returns
    -------
    None
        Writes ``hvf_loess``, ``hvf_rank`` and ``highly_variable_features`` to
        ``data.var``.
    """
    if "robust" not in data.var:
        raise ValueError("Please run `identify_robust_genes` to identify robust genes")

    estimate_feature_statistics(data, batch)

    robust_idx = data.var["robust"].values
    hvf_index = np.zeros(robust_idx.sum(), dtype=bool)

    mean = data.var.loc[robust_idx, "mean"]
    var = data.var.loc[robust_idx, "var"]

    span_value = span
    while True:
        lobj = fit_loess(mean, var, span = span_value, degree = 2)
        if lobj is not None:
            break
        span_value += 0.01
    if span_value > span:
        print("Leoss span is adjusted from {:.2f} to {:.2f} to avoid fitting errors.".format(span, span_value))

    rank1 = np.zeros(hvf_index.size, dtype=int)
    rank2 = np.zeros(hvf_index.size, dtype=int)

    delta = var - lobj.outputs.fitted_values
    fc = var / lobj.outputs.fitted_values

    rank1[np.argsort(delta)[::-1]] = range(hvf_index.size)
    rank2[np.argsort(fc)[::-1]] = range(hvf_index.size)
    hvf_rank = rank1 + rank2

    hvf_index[np.argsort(hvf_rank)[:n_top]] = True

    data.var["hvf_loess"] = 0.0
    data.var.loc[robust_idx, "hvf_loess"] = lobj.outputs.fitted_values

    data.var["hvf_rank"] = -1
    data.var.loc[robust_idx, "hvf_rank"] = hvf_rank
    data.var["highly_variable_features"] = False
    data.var.loc[robust_idx, "highly_variable_features"] = hvf_index

def calc_expm1(X: Union[csr_matrix, np.ndarray]) -> np.ndarray:
    
    '''
    exponential minus one
    
    '''
    if not issparse(X):
        return np.expm1(X)
    res = X.copy()
    np.expm1(res.data, out = res.data)
    return res

def select_hvf_seurat_single(
    X: Union[csr_matrix, np.ndarray],
    n_top: int,
    min_disp: float,
    max_disp: float,
    min_mean: float,
    max_mean: float,
) -> List[int]:
    """ HVF selection for one channel using Seurat method
    """
    X = calc_expm1(X)

    mean, var = calc_mean_and_var(X, axis=0)

    dispersion = np.full(X.shape[1], np.nan)
    idx_valid = (mean > 0.0) & (var > 0.0)
    dispersion[idx_valid] = var[idx_valid] / mean[idx_valid]

    mean = np.log1p(mean)
    dispersion = np.log(dispersion)

    df = pd.DataFrame({"log_dispersion": dispersion, "bin": pd.cut(mean, bins=20)})
    log_disp_groups = df.groupby("bin")["log_dispersion"]
    log_disp_mean = log_disp_groups.mean()
    log_disp_std = log_disp_groups.std(ddof=1)
    log_disp_zscore = (
        df["log_dispersion"].values - log_disp_mean.loc[df["bin"]].values
    ) / log_disp_std.loc[df["bin"]].values
    log_disp_zscore[np.isnan(log_disp_zscore)] = 0.0

    hvf_rank = np.full(X.shape[1], -1, dtype=int)
    ords = np.argsort(log_disp_zscore)[::-1]

    if n_top is None:
        hvf_rank[ords] = range(X.shape[1])
        idx = np.logical_and.reduce(
            (
                mean > min_mean,
                mean < max_mean,
                log_disp_zscore > min_disp,
                log_disp_zscore < max_disp,
            )
        )
        hvf_rank[~idx] = -1
    else:
        hvf_rank[ords[:n_top]] = range(n_top)

    return hvf_rank



def select_hvf_seurat(
    data: anndata.AnnData,
    batch: str,
    n_top: int,
    min_disp: float,
    max_disp: float,
    min_mean: float,
    max_mean: float,
    n_jobs: int,
) -> None:
    """ Select highly variable features using Seurat method.
    """

    robust_idx = data.var["robust"].values
    X = data.X[:, robust_idx]

    hvf_rank = (
        select_hvf_seurat_single(
            X,
            n_top=n_top,
            min_disp=min_disp,
            max_disp=max_disp,
            min_mean=min_mean,
            max_mean=max_mean,
        )
    )

    hvf_index = hvf_rank >= 0

    data.var["hvf_rank"] = -1
    data.var.loc[robust_idx, "hvf_rank"] = hvf_rank
    data.var["highly_variable_features"] = False
    data.var.loc[robust_idx, "highly_variable_features"] = hvf_index

@register_function(
    aliases=["高变基因", "highly_variable_features", "hvg", "hvf", "高变特征筛选"],
    category="preprocessing",
    description="Compute highly variable features using Pegasus or Seurat flavor",
    prerequisites={
        "functions": ["identify_robust_genes"],
        "optional_functions": ["log1p", "qc"],
    },
    requires={
        "var": ["robust"]
    },
    produces={
        "var": ["hvf_rank", "highly_variable_features"]
    },
    auto_fix="none",
    examples=[
        "ov.pp.identify_robust_genes(adata)",
        "ov.pp.highly_variable_features(adata, flavor='pegasus', n_top=2000)",
        "ov.pp.highly_variable_features(adata, flavor='Seurat', n_top=3000)",
    ],
    related=["select_hvf_pegasus", "highly_variable_genes", "remove_cc_genes"],
)
def highly_variable_features(
    data: anndata.AnnData,
    batch: str = None,
    flavor: str = "pegasus",
    n_top: int = 2000,
    span: float = 0.02,
    min_disp: float = 0.5,
    max_disp: float = np.inf,
    min_mean: float = 0.0125,
    max_mean: float = 7,
    n_jobs: int = -1,
) -> None:
    """Select highly variable features (HVF/HVG) for downstream modeling.

    Parameters
    ----------
    data : anndata.AnnData
        AnnData matrix with genes in columns.
    batch : str, optional
        Column name in ``data.obs`` for batch-aware HVF selection.
    flavor : {"pegasus", "Seurat"}, default="pegasus"
        Algorithm used to rank variable genes.
    n_top : int, default=2000
        Number of top genes to keep.
    span : float, default=0.02
        Loess span used by Pegasus flavor.
    min_disp : float, default=0.5
        Lower bound of normalized dispersion for Seurat flavor.
    max_disp : float, default=np.inf
        Upper bound of normalized dispersion for Seurat flavor.
    min_mean : float, default=0.0125
        Lower bound of mean expression for Seurat flavor.
    max_mean : float, default=7
        Upper bound of mean expression for Seurat flavor.
    n_jobs : int, default=-1
        Number of worker threads for computations where applicable.

    Returns
    -------
    None
        Updates ``data.var['highly_variable_features']`` and ranking fields.
    """

    if flavor == "pegasus":
        select_hvf_pegasus(data, batch, n_top=n_top, span=span)
    else:
        assert flavor == "Seurat"
        select_hvf_seurat(
            data,
            batch,
            n_top=n_top,
            min_disp=min_disp,
            max_disp=max_disp,
            min_mean=min_mean,
            max_mean=max_mean,
            n_jobs=n_jobs,
        )

    data.uns.pop("_tmp_fmat_highly_variable_features", None) # Pop up cached feature matrix

    print(f"{data.var['highly_variable_features'].sum()} \
        highly variable features have been selected.")

def fit_loess(x: List[float], y: List[float], span: float, degree: int) -> object:
    '''
    A LOESS (Locally Weighted Regression) model is used to fit a given data set
    '''
    try:
        lobj = sl.loess(x, y, span=span, degree=degree)
        lobj.fit()
        return lobj
    except ValueError:
        return None
def corr2_coeff(a, b):
    """
    Calculate Pearson correlation between matrix a and b
    a and b are allowed to have different shapes. Taken from Cospar, Wang et al., 2023.
    """
    resol = 10 ** (-15)

    # Convert sparse matrices to dense arrays if necessary
    if issparse(a):
        a = a.toarray()
    if issparse(b):
        b = b.toarray()

    a_ma = a - a.mean(1)[:, None]
    b_mb = b - b.mean(1)[:, None]
    ssa = (a_ma ** 2).sum(1)
    ssb = (b_mb ** 2).sum(1)

    corr = np.dot(a_ma, b_mb.T) / (np.sqrt(np.dot(ssa[:, None], ssb[None])) + resol)

    return corr

@register_function(
    aliases=["去除细胞周期基因", "remove_cc_genes", "cell_cycle_filter", "去除CC相关基因", "cc_genes"],
    category="preprocessing",
    description="Remove cell-cycle-correlated genes from the selected HVG set",
    prerequisites={
        "functions": ["highly_variable_features"],
        "optional_functions": ["score_genes_cell_cycle"],
    },
    requires={
        "var": ["highly_variable_features"]
    },
    produces={
        "var": ["highly_variable_features"]
    },
    auto_fix="none",
    examples=[
        "ov.pp.highly_variable_features(adata, n_top=3000)",
        "ov.pp.remove_cc_genes(adata, organism='human', corr_threshold=0.1)",
    ],
    related=["highly_variable_features", "score_genes_cell_cycle", "pca"],
)
def remove_cc_genes(adata:anndata.AnnData, organism:str='human', corr_threshold:float=0.1):
    """Remove cell-cycle-correlated genes from ``highly_variable_features``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing HVG annotations.
    organism : {"human", "mouse"}, default="human"
        Species used to load canonical cell-cycle signatures.
    corr_threshold : float, default=0.1
        Absolute correlation cutoff. HVGs with maximal correlation to
        cell-cycle genes above this threshold are removed.

    Returns
    -------
    None
        Rewrites ``adata.var['highly_variable_features']`` after filtering.
    """
    # Get cc genes
    cycling_genes = load_signatures_from_file(predefined_signatures[f'cell_cycle_{organism}'])
    cc_genes = list(set(cycling_genes['G1/S']) | set(cycling_genes['G2/M']))
    cc_genes = [ x for x in cc_genes if x in adata.var_names ]
    # Compute corr
    cc_expression = adata[:, cc_genes].X.toarray().T if issparse(adata[:, cc_genes].X) else adata[:, cc_genes].X.T
    hvgs = adata.var_names[adata.var['highly_variable_features']]
    hvgs_expression = adata[:, hvgs].X.toarray().T if issparse(adata[:, hvgs].X) else adata[:, hvgs].X.T
    cc_corr = corr2_coeff(hvgs_expression, cc_expression)

    # Discard genes having the maximum correlation with one of the cc > corr_threshold
    max_corr = np.max(abs(cc_corr), 1)
    hvgs_no_cc = hvgs[max_corr < corr_threshold]
    print(
        f'Number of selected non-cycling highly variable genes: {hvgs_no_cc.size}\n'
        f'{np.sum(max_corr > corr_threshold)} cell cycle correlated genes will be removed...'
    )
    # Update
    adata.var['highly_variable_features'] = adata.var_names.isin(hvgs_no_cc)

from sklearn.cluster import KMeans  

@monitor
@register_function(
    aliases=["数据转GPU", "anndata_to_GPU", "to_gpu", "GPU转换", "move_to_gpu"],
    category="preprocessing",
    description="Migrate AnnData objects to GPU memory for accelerated processing with RAPIDS",
    examples=[
        "# Move data to GPU for processing",
        "ov.pp.anndata_to_GPU(adata)",
        "# After analysis, move back to CPU",
        "ov.pp.anndata_to_CPU(adata)",
        "# Use with GPU preprocessing pipeline",
        "ov.settings.gpu_init()",
        "ov.pp.anndata_to_GPU(adata)",
        "adata = ov.pp.qc(adata)  # GPU-accelerated QC"
    ],
    related=["pp.anndata_to_CPU", "settings.gpu_init", "pp.qc", "pp.preprocess"]
)
def anndata_to_GPU(
    adata,
    *,
    convert_all: bool = True,
    copy: bool = False,
    **kwargs,
):
    """Migrate AnnData objects to GPU memory for accelerated processing.

    Arguments:
        adata: AnnData object containing single-cell data.
        **kwargs: Additional arguments passed to rapids_singlecell.get.anndata_to_GPU.

    Returns:
        None: The function modifies adata in place by moving data to GPU memory.

    Examples:
        >>> import omicverse as ov
        >>> # Initialize GPU mode
        >>> ov.settings.gpu_init()
        >>> # Move data to GPU
        >>> ov.pp.anndata_to_GPU(adata)
        >>> # Perform GPU-accelerated analysis
        >>> adata = ov.pp.qc(adata)
        >>> # Move back to CPU when done
        >>> ov.pp.anndata_to_CPU(adata)
    """
    import rapids_singlecell as rsc
    kwargs.setdefault("convert_all", convert_all)
    kwargs.setdefault("copy", copy)
    rsc.get.anndata_to_GPU(adata, **kwargs)
    print('Data has been moved to GPU')
    print('Don`t forget to move it back to CPU after analysis is done')
    print('Use `ov.pp.anndata_to_CPU(adata)`')

@monitor
@register_function(
    aliases=["数据转CPU", "anndata_to_CPU", "to_cpu", "CPU转换", "move_to_cpu"],
    category="preprocessing",
    description="Migrate AnnData objects from GPU back to CPU memory after analysis",
    examples=[
        "# Move data back to CPU after GPU processing",
        "ov.pp.anndata_to_CPU(adata)",
        "# Convert specific layer only",
        "ov.pp.anndata_to_CPU(adata, layer='scaled')",
        "# Convert with copy",
        "ov.pp.anndata_to_CPU(adata, convert_all=True, copy=True)",
        "# Complete GPU-CPU workflow",
        "ov.settings.gpu_init()",
        "ov.pp.anndata_to_GPU(adata)",
        "adata = ov.pp.qc(adata)  # GPU processing",
        "ov.pp.anndata_to_CPU(adata)  # Back to CPU"
    ],
    related=["pp.anndata_to_GPU", "settings.gpu_init", "pp.qc", "pp.preprocess"]
)
def anndata_to_CPU(adata,layer=None, convert_all=True, copy=False):
    """Migrate AnnData objects from GPU back to CPU memory after analysis.

    Arguments:
        adata: AnnData object containing single-cell data on GPU.
        layer: Specific layer to convert back to CPU. Default: None (all layers).
        convert_all: Whether to convert all arrays to CPU. Default: True.
        copy: Whether to create a copy during conversion. Default: False.

    Returns:
        None: The function modifies adata in place by moving data from GPU to CPU memory.

    Examples:
        >>> import omicverse as ov
        >>> # After GPU processing, move back to CPU
        >>> ov.pp.anndata_to_CPU(adata)
        >>> # Convert only specific layer
        >>> ov.pp.anndata_to_CPU(adata, layer='scaled', convert_all=False)
    """
    import rapids_singlecell as rsc
    rsc.get.anndata_to_CPU(adata,layer=layer, convert_all=convert_all, copy=copy)

@monitor
@register_function(
    aliases=["预处理", "preprocess", "preprocessing", "数据预处理","HVG","HVGs","高变基因","高变基因选择"],
    category="preprocessing",
    description="Complete preprocessing pipeline including normalization, HVG selection, scaling, and PCA",
    prerequisites={
        'optional_functions': ['qc']
    },
    requires={},
    produces={
        'layers': ['counts'],
        'var': ['highly_variable_features', 'means', 'variances', 'residual_variances']
    },
    auto_fix='none',
    examples=[
        "ov.pp.preprocess(adata, mode='shiftlog|pearson', n_HVGs=2000)",
        "ov.pp.preprocess(adata, mode='pearson|pearson', target_sum=50e4)"
    ],
    related=["qc", "normalize", "scale", "pca", "highly_variable_genes"]
)
@tracked("preprocess", "ov.pp.preprocess")
def preprocess(
    adata, mode='shiftlog|pearson', 
    target_sum=50*1e4, n_HVGs=2000,
    organism='human', no_cc=False,batch_key=None,
    identify_robust=True):
    """
    Preprocesses the AnnData object adata using either a scanpy or 
    a pearson residuals workflow for size normalization
    and highly variable genes (HVGs) selection, and calculates signature scores if necessary. 

    Arguments:
        adata: The data matrix.
        mode: The mode for size normalization and HVGs selection. 
        It can be either 'scanpy' or 'pearson'. If 'scanpy', 
        performs size normalization using scanpy's normalize_total() function and selects HVGs 
        using pegasus' highly_variable_features() function with batch correction. 
        If 'pearson', selects HVGs sing scanpy's experimental.pp.highly_variable_genes() function
        with pearson residuals method and performs 
        size normalization using scanpy's experimental.pp.normalize_pearson_residuals() function. 
        target_sum: The target total count after normalization.
        n_HVGs: the number of HVGs to select.
        organism: The organism of the data. It can be either 'human' or 'mouse'. 
        no_cc: Whether to remove cc-correlated genes from HVGs.

    Returns:
        adata: The preprocessed data matrix.
    """

    # Track the original object so we can propagate changes even if callers
    # forget to use the returned value.
    original_adata = adata

    # Log-normalization, HVGs identification
    from ._qc import _is_oom
    is_oom = _is_oom(adata)

    # Validate unsupported combinations BEFORE mutating adata (otherwise the
    # caller gets back a half-processed object with partial lazy transforms).
    method_list_preview = mode.split('|')
    if is_oom and len(method_list_preview) > 1 and method_list_preview[1] == 'seurat':
        raise NotImplementedError(
            "Seurat v3 HVG selection (Loess-smoothed dispersion) is not "
            "available for the OOM backend. Use 'shiftlog|pearson' instead, "
            "or switch to mode='cpu' to materialise the data."
        )
    if is_oom and method_list_preview[0] == 'pearson':
        # Pearson residual normalisation needs a materialised X. Rather than
        # leave adata in a hybrid state (in-memory data wrapped in
        # BackedArray, but _is_oom still True), refuse the combination and
        # let the user opt into mode='cpu' explicitly.
        raise NotImplementedError(
            "Pearson residual normalisation is not available for the OOM "
            "backend (it requires the full X in memory, which would leave "
            "adata in a hybrid state). Use 'shiftlog|pearson' instead, or "
            "switch to mode='cpu'."
        )
    if is_oom and no_cc:
        # remove_cc_genes materialises cc_genes × n_obs and hvgs × n_obs
        # slices via `.X.toarray()`, which doesn't stream on BackedArray.
        # Refuse rather than silently defeat OOM.
        raise NotImplementedError(
            "no_cc=True is not available for the OOM backend "
            "(remove_cc_genes requires a materialised X). Run preprocess "
            "without no_cc, or switch to mode='cpu'."
        )

    if is_oom:
        # OOM: save a lazy reference to raw counts (zero memory cost)
        adata.layers['counts'] = adata.X
    else:
        adata.layers['counts'] = adata.X.copy()
    
    
    print(f"{EMOJI['start']} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running preprocessing in '{settings.mode}' mode...")
    print(f"{Colors.CYAN}Begin robust gene identification{Colors.ENDC}")
    if identify_robust:
        if is_oom:
            from anndataoom import chunked_identify_robust_genes
            chunked_identify_robust_genes(adata, percent_cells=0.05)
        else:
            identify_robust_genes(adata, percent_cells=0.05)
        if is_oom:
            adata._inplace_subset_var(adata.var['robust'].values)
        else:
            adata = adata[:, adata.var['robust']]
        print(f"{EMOJI['done']} Robust gene identification completed successfully.")
    method_list = mode.split('|')
    print(f"{Colors.CYAN}Begin size normalization: {method_list[0]} and HVGs selection {method_list[1]}{Colors.ENDC}")
    if is_oom:
        # Out-of-memory path — lazy transforms, no full materialisation
        from anndataoom import (
            chunked_normalize_total, chunked_log1p,
        )
        data_load_start = time.time()
        # Only 'shiftlog' normalisation is supported for OOM; 'pearson' is
        # pre-rejected at the top of this function.
        chunked_normalize_total(
            adata,
            target_sum=target_sum,
            exclude_highly_expressed=True,
            max_fraction=0.2,
        )
        chunked_log1p(adata)

        if method_list[1] == 'pearson':
            from anndataoom import chunked_highly_variable_genes_pearson
            chunked_highly_variable_genes_pearson(
                adata,
                n_top_genes=n_HVGs,
                layer='counts',
                batch_key=batch_key,
            )
            if no_cc:
                remove_cc_genes(adata, organism=organism, corr_threshold=0.1)
        else:
            # 'seurat' is pre-rejected at the top; any other unknown HVG
            # method should fail loudly here rather than silently skip HVG
            # selection and leave the user with no 'highly_variable' column.
            raise NotImplementedError(
                f"HVG method {method_list[1]!r} is not supported for the OOM "
                "backend. Use 'pearson', or switch to mode='cpu'."
            )
        data_load_end = time.time()
        print(f"{Colors.BLUE}    Time to analyze data (out-of-memory): {data_load_end - data_load_start:.2f} seconds.{Colors.ENDC}")
    elif settings.mode == 'cpu' or settings.mode == 'cpu-gpu-mixed':
        data_load_start = time.time()
        if method_list[0] == 'shiftlog': # Size normalization + scanpy batch aware HVGs selection
            normalize_total(
                adata,
                target_sum=target_sum,
                exclude_highly_expressed=True,
                max_fraction=0.2,
            )
            log1p(adata)

        elif method_list[0] == 'pearson':
            # Perason residuals workflow
            from .experimental import normalize_pearson_residuals
            normalize_pearson_residuals(adata)

        if method_list[1] == 'pearson': # Size normalization + scanpy batch aware HVGs selection
            from .experimental import highly_variable_genes as _hvg_pearson
            _hvg_pearson(
                adata,
                flavor="pearson_residuals",
                layer='counts',
                n_top_genes=n_HVGs,
                batch_key=batch_key,
            )
            if no_cc:
                remove_cc_genes(adata, organism=organism, corr_threshold=0.1)
        elif method_list[1] == 'seurat':
            # Use the in-package implementation in
            # `omicverse.pp._highly_variable_genes`, which natively
            # supports `flavor='seurat_v3'`. The experimental variant in
            # `.experimental` only supports `flavor='pearson_residuals'`,
            # which is why the bare-name call here used to crash.
            from ._highly_variable_genes import highly_variable_genes as _hvg_seurat
            _hvg_seurat(
                adata,
                flavor="seurat_v3",
                layer='counts',
                n_top_genes=n_HVGs,
                batch_key=batch_key,
            )
            if no_cc:
                remove_cc_genes(adata, organism=organism, corr_threshold=0.1)
        data_load_end = time.time()
        print(f"{Colors.BLUE}    Time to analyze data in cpu: {data_load_end - data_load_start:.2f} seconds.{Colors.ENDC}")
    else:
        import rapids_singlecell as rsc
        data_load_start = time.time()
        if method_list[0] == 'shiftlog': # Size normalization + scanpy batch aware HVGs selection
            rsc.pp.normalize_total(adata, target_sum=target_sum)
            rsc.pp.log1p(adata)
        elif method_list[0] == 'pearson':
            # Perason residuals workflow
            rsc.pp.normalize_pearson_residuals(adata)
        if method_list[1] == 'pearson': # Size normalization + scanpy batch aware HVGs selection
            rsc.pp.highly_variable_genes(
                adata, 
                flavor="pearson_residuals",
                layer='counts',
                n_top_genes=n_HVGs,
                batch_key=batch_key,
            )
        elif method_list[1] == 'seurat':
            rsc.pp.highly_variable_genes(
                adata,
                flavor="seurat_v3",
                layer='counts',
                n_top_genes=n_HVGs,
                batch_key=batch_key,
            )
        data_load_end = time.time()
        print(f"{Colors.BLUE}    Time to analyze data in gpu: {data_load_end - data_load_start:.2f} seconds.{Colors.ENDC}")

    # Normalize HVG column naming across backends and keep both aliases available
    hv = adata.var['highly_variable'] if 'highly_variable' in adata.var.columns else None
    hv_features = adata.var['highly_variable_features'] if 'highly_variable_features' in adata.var.columns else None
    if hv is not None and hv_features is None:
        adata.var['highly_variable_features'] = hv
    elif hv_features is not None and hv is None:
        adata.var['highly_variable'] = hv_features
    elif hv is None and hv_features is None:
        # Fallback: create a False vector if nothing is present
        adata.var['highly_variable'] = False
        adata.var['highly_variable_features'] = adata.var['highly_variable']
    else:
        adata.var['highly_variable_features'] = adata.var['highly_variable']

    # Ensure PCA is available for downstream steps that expect it
    '''
    try:
        if 'X_pca' not in adata.obsm:
            from ._pca import pca as _pca
            _pca(adata, n_comps=min(50, adata.n_vars - 1), layer=None)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"{Colors.WARNING}⚠️  PCA computation skipped: {exc}{Colors.ENDC}")
    '''

    print(f"{EMOJI['done']} Preprocessing completed successfully.")
    print(f"{Colors.GREEN}    Added:{Colors.ENDC}")
    print(f"{Colors.CYAN}        'highly_variable_features', boolean vector (adata.var){Colors.ENDC}")
    print(f"{Colors.CYAN}        'means', float vector (adata.var){Colors.ENDC}")
    print(f"{Colors.CYAN}        'variances', float vector (adata.var){Colors.ENDC}")
    print(f"{Colors.CYAN}        'residual_variances', float vector (adata.var){Colors.ENDC}")
    print(f"{Colors.CYAN}        'counts', raw counts layer (adata.layers){Colors.ENDC}")
    print(f"{Colors.BLUE}    End of size normalization: {method_list[0]} and HVGs selection {method_list[1]}{Colors.ENDC}")

    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    if 'status_args' not in adata.uns.keys():
        adata.uns['status_args'] = {}

    adata.uns['status']['preprocess']=True
    adata.uns['status_args']['preprocess']={
        'mode':mode, 'target_sum':target_sum, 'n_HVGs':n_HVGs,
        'organism':organism,
    }
    add_reference(adata,'scanpy','size normalization with scanpy')

    # If we created a sliced copy above, mirror the updated state back onto
    # the original object so callers get the processed data even when they
    # forget to assign the returned adata.
    if adata is not original_adata:
        original_adata.__dict__.update(adata.__dict__)
        adata = original_adata

    note(backend=f"omicverse({settings.mode}) · mode={mode}")
    if "highly_variable" in adata.var.columns:
        note(viz=[{"function": "ov.pl.highly_variable_genes_scatter", "kwargs": {}}])
    return adata

@register_function(
    aliases=["pearson残差归一化", "normalize_pearson_residuals", "pearson_normalization", "皮尔逊残差", "sctransform_like"],
    category="preprocessing",
    description="Normalize counts using analytic Pearson residuals for variance stabilization",
    prerequisites={
        "optional_functions": ["qc", "filter_cells", "filter_genes"]
    },
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "ov.pp.normalize_pearson_residuals(adata, theta=100)",
        "ov.pp.normalize_pearson_residuals(adata, clip=30)",
    ],
    related=["highly_variable_genes", "scale", "pca"],
)
def normalize_pearson_residuals(
    adata,
    *,
    theta: float = 100,
    clip: float | None = None,
    check_values: bool = True,
    layer: str | None = None,
    inplace: bool = True,
    copy: bool = False,
    **kwargs,
):
    """Normalize a count matrix using analytic Pearson residuals (Lause 2021).

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing counts in ``adata.X`` (or ``layer``).
    theta
        Negative binomial overdispersion parameter. ``100`` matches
        Lause 2021 and is a reasonable default for scRNA.
    clip
        Clip residuals to ``[-clip, clip]``. ``None`` uses the
        scanpy default (``sqrt(n_obs)``).
    check_values
        Validate that the input looks like counts.
    layer
        Layer to normalise. ``None`` uses ``adata.X``.
    inplace
        Write back into ``adata`` (``True``) or return the matrix
        (``False``).
    copy
        Return a new AnnData instead of modifying in place (only when
        ``inplace=True``).
    **kwargs
        Forwarded to ``scanpy.experimental.pp.normalize_pearson_residuals``.

    Returns
    -------
    None
        Updates ``adata.X`` in place with Pearson residuals.
    """
    sc.experimental.pp.normalize_pearson_residuals(
        adata, theta=theta, clip=clip, check_values=check_values,
        layer=layer, inplace=inplace, copy=copy, **kwargs,
    )

@monitor
@register_function(
    aliases=["高变基因识别", "highly_variable_genes", "hvg_scanpy", "HVG", "高变基因检测"],
    category="preprocessing",
    description="Identify highly variable genes using Scanpy-compatible methods",
    prerequisites={
        "optional_functions": ["normalize_pearson_residuals", "log1p", "qc"]
    },
    requires={},
    produces={
        "var": ["highly_variable", "means", "dispersions", "dispersions_norm"]
    },
    auto_fix="none",
    examples=[
        "ov.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)",
        "ov.pp.highly_variable_genes(adata, layer='log1p', flavor='seurat')",
    ],
    related=["highly_variable_features", "normalize_pearson_residuals", "pca"],
)
@tracked("highly_variable_genes", "ov.pp.highly_variable_genes")
def highly_variable_genes(
    adata,
    *,
    layer: str | None = None,
    n_top_genes: int | None = None,
    min_disp: float = 0.5,
    max_disp=None,
    min_mean: float = 0.0125,
    max_mean: float = 3,
    span: float = 0.3,
    n_bins: int = 20,
    flavor: str = "seurat",
    subset: bool = False,
    inplace: bool = True,
    batch_key: str | None = None,
    filter_unexpressed_genes: bool | None = None,
    check_values: bool = True,
    **kwargs,
):
    """Annotate highly variable genes (Satija 2015 / Zheng 2017 / Stuart 2019).

    Parameters
    ----------
    adata : anndata.AnnData
        Expression matrix. Expects log-normalised data except for
        ``flavor='seurat_v3'`` / ``'seurat_v3_paper'`` which expect counts.
    layer
        Layer to use. ``None`` uses ``adata.X``.
    n_top_genes
        Number of HVGs to keep. Required for ``flavor='seurat_v3'``.
    min_disp, max_disp, min_mean, max_mean
        Dispersion / mean cutoffs for ``flavor='seurat'`` and
        ``flavor='cell_ranger'``.
    span
        Loess smoothing span for ``flavor='seurat_v3'``.
    n_bins
        Dispersion normalisation bins.
    flavor
        ``'seurat'`` (default), ``'cell_ranger'``, ``'seurat_v3'``, or
        ``'seurat_v3_paper'``.
    subset
        If True, subset the AnnData to the HVGs in place.
    inplace
        Write statistics into ``adata.var`` rather than returning a
        DataFrame.
    batch_key
        If set, HVG detection is done per batch.
    filter_unexpressed_genes
        Drop genes with zero total expression before selection.
    check_values
        Input validation (log-normalised vs counts).
    **kwargs
        Extra options forwarded to
        ``omicverse.pp._highly_variable_genes.highly_variable_genes``.

    Returns
    -------
    pandas.DataFrame | None
        Mutates ``adata.var``; also returns the stats DataFrame unless
        ``inplace=True``.
    """
    from ._highly_variable_genes import highly_variable_genes as _hvg
    import numpy as np
    if max_disp is None:
        max_disp = np.inf
    result = _hvg(
        adata, layer=layer, n_top_genes=n_top_genes,
        min_disp=min_disp, max_disp=max_disp, min_mean=min_mean,
        max_mean=max_mean, span=span, n_bins=n_bins, flavor=flavor,
        subset=subset, inplace=inplace, batch_key=batch_key,
        filter_unexpressed_genes=filter_unexpressed_genes,
        check_values=check_values, **kwargs,
    )
    note(backend=f"omicverse · flavor={flavor}",
         viz=[{"function": "ov.pl.highly_variable_genes_scatter", "kwargs": {}}])
    return result

@monitor
@register_function(
    aliases=["标准化", "scale", "scaling", "标准化处理"],
    category="preprocessing",
    description="Scale data to unit variance and zero mean",
    prerequisites={
        'optional_functions': ['normalize', 'qc']
    },
    requires={},
    produces={
        'layers': ['scaled']
    },
    auto_fix='none',
    examples=["ov.pp.scale(adata, max_value=10)", "ov.pp.scale(adata, max_value=10, to_sparse=True)"],
    related=["normalize", "regress"]
)
@tracked("scale", "ov.pp.scale")
def scale(adata, max_value=10, layers_add='scaled', to_sparse=False,
          use_implicit_centering=False, **kwargs):
    """
    Scale the input AnnData object.

    Arguments:
        adata: Annotated data matrix with n_obs x n_vars shape.
        max_value: Maximum value after scaling. Default: 10.
        layers_add: Name of the layer to store the scaled data. Default: 'scaled'.
            OOM backend only supports layers_add='scaled' (the underlying
            chunked_scale writes a lazy ScaledBackedArray to that fixed key);
            passing anything else raises ValueError.
        to_sparse: If True, convert the result to csr_matrix format. Default: False.
            Scaled data is 100% dense, so sparse storage only adds overhead.
        use_implicit_centering: If True and the in-memory adata.X is sparse,
            store the scaled matrix as a lazy ``CenteredSparseArray`` (anndataoom)
            in ``adata.uns['_scaled_implicit']`` instead of materialising a dense
            ``adata.layers['scaled']``. Trades a small dispatch hop in downstream
            consumers (only ``ov.pp.pca`` knows how to read this layout) for
            ``n_obs * n_vars * 4 bytes`` of avoided allocation -- at 1M cells x
            60606 genes that is ~240 GB, the difference between OOM-killed and
            completing under a typical 256 GB per-process RSS cap. ``ov.pp.pca``
            densifies only the HVG-subset slice the SVD actually needs.
            Default: False (preserves existing behavior). Ignored on the OOM
            backend (which already uses a lazy ``ScaledBackedArray``) and on
            GPU mode.
        **kwargs: Additional arguments passed to scaling functions.

    Returns:
        adata: Annotated data matrix with n_obs x n_vars shape.
        Adds a new layer called 'scaled' that stores the expression matrix
        that has been scaled to unit variance and zero mean.

    Examples:
        >>> import omicverse as ov
        >>> # Scale data with default sparse output
        >>> ov.pp.scale(adata, max_value=10)
        >>> # Scale data keeping dense format
        >>> ov.pp.scale(adata, max_value=10, to_sparse=False)
        >>> # Lazy implicit centering for million-cell in-memory pipelines
        >>> ov.pp.scale(adata, max_value=10, use_implicit_centering=True)
    """
    from ._qc import _is_oom
    if _is_oom(adata):
        # chunked_scale hard-codes the layer name to 'scaled' and installs a
        # lazy ScaledBackedArray. Re-wrapping/materialising it defeats the
        # point, so we short-circuit the rest of this function.
        if layers_add != "scaled":
            raise ValueError(
                "OOM scale only stores to adata.layers['scaled']; "
                f"layers_add={layers_add!r} is not supported."
            )
        from anndataoom import chunked_scale
        chunked_scale(adata, max_value=max_value)
        if "status" not in adata.uns.keys():
            adata.uns["status"] = {}
        add_reference(adata, "scanpy", "scaling with scanpy")
        adata.uns["status"]["scaled"] = True
        note(backend=f"omicverse({settings.mode}) · oom")
        return
    # Opt-in implicit-centered path: produce a CenteredSparseArray and
    # store it in adata.uns['_scaled_implicit'] rather than densifying.
    # Only runs when (a) caller asked for it, (b) we are on CPU mode, and
    # (c) adata.X is sparse (the densification this avoids is exactly the
    # sparse zero-centering step). zero_center defaults True in kwargs.
    _zero_center = kwargs.get('zero_center', True)
    if (use_implicit_centering
            and (settings.mode == 'cpu' or settings.mode == 'cpu-gpu-mixed')
            and _zero_center
            and issparse(adata.X)):
        try:
            from anndataoom import CenteredSparseArray
        except ImportError as exc:
            raise ImportError(
                "use_implicit_centering=True requires the anndataoom package "
                "(>=0.1.7); install with `pip install anndataoom`."
            ) from exc
        X = adata.X.tocsr().astype(np.float32)
        # Sparse-native E[X^2] − E[X]^2; matches scanpy / sklearn convention.
        mean = np.asarray(X.mean(axis=0)).ravel().astype(np.float32)
        ex2 = np.asarray(X.multiply(X).mean(axis=0)).ravel().astype(np.float32)
        var = np.maximum(ex2 - mean * mean, 0.0)
        std = np.sqrt(var).astype(np.float32)
        wrapper = CenteredSparseArray(X, mean, std, max_value=max_value)
        adata.uns['_scaled_implicit'] = wrapper
        # Expose mean/std on adata.var (matches scale_anndata's behavior).
        adata.var['mean'] = mean
        adata.var['std'] = std
        if 'status' not in adata.uns.keys():
            adata.uns['status'] = {}
        if 'status_args' not in adata.uns.keys():
            adata.uns['status_args'] = {}
        adata.uns['status']['scaled'] = True
        adata.uns['status_args']['scaled'] = {'implicit': True,
                                              'max_value': max_value}
        add_reference(adata, 'scanpy', 'scaling with scanpy')
        note(backend=f"omicverse({settings.mode}) · implicit")
        return

    if settings.mode == 'cpu' or settings.mode == 'cpu-gpu-mixed':
        from ._scale import scale_anndata as scale
        adata_mock = scale(adata, copy=True, max_value=max_value, **kwargs)
        scaled_data = adata_mock.X.copy()
        del adata_mock
    else:
        import rapids_singlecell as rsc
        scaled_data = rsc.pp.scale(adata, max_value=max_value, inplace=False)

    # Convert to sparse format if requested
    if to_sparse and not issparse(scaled_data):
        print(f"{Colors.BLUE}    Converting scaled data to csr_matrix format...{Colors.ENDC}")
        scaled_data = csr_matrix(scaled_data)
    elif to_sparse and issparse(scaled_data):
        # Ensure it's csr_matrix format
        if not isinstance(scaled_data, csr_matrix):
            print(f"{Colors.BLUE}    Converting scaled data to csr_matrix format...{Colors.ENDC}")
            scaled_data = csr_matrix(scaled_data)

    adata.layers[layers_add] = scaled_data

    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    if 'status_args' not in adata.uns.keys():
        adata.uns['status_args'] = {}
    add_reference(adata,'scanpy','scaling with scanpy')
    adata.uns['status']['scaled'] = True
    note(backend=f"omicverse({settings.mode})")

@monitor
@register_function(
    aliases=["回归校正", "regress", "regress_out", "去除技术效应", "协变量回归"],
    category="preprocessing",
    description="Regress out technical covariates (mitochondrial ratio and UMI counts)",
    prerequisites={
        "optional_functions": ["qc", "normalize_pearson_residuals"]
    },
    requires={
        "obs": ["mito_perc", "nUMIs"]
    },
    produces={
        "layers": ["regressed"]
    },
    auto_fix="none",
    examples=[
        "ov.pp.regress(adata)",
        "ov.pp.regress(adata, n_jobs=4)",
    ],
    related=["scale", "regress_and_scale", "pca"],
)
def regress(adata, *, layer=None, n_jobs=8, **kwargs):
    """Regress out technical covariates (``mito_perc``, ``nUMIs``) from
    each gene.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with ``adata.obs['mito_perc']`` and
        ``adata.obs['nUMIs']`` already computed (by ``ov.pp.qc``).
    layer
        Expression layer to regress. ``None`` uses ``adata.X``.
    n_jobs
        Parallel jobs for the CPU regressor. Default ``8``.
    **kwargs
        Extra options forwarded to the backend
        (``scanpy.pp.regress_out`` or ``rapids_singlecell.pp.regress_out``).

    Returns
    -------
    None
        Writes the residualised expression to ``adata.layers['regressed']``.
    """
    if layer is not None:
        kwargs.setdefault("layer", layer)
    kwargs.setdefault("n_jobs", n_jobs)
    if settings.mode == 'cpu' or settings.mode == 'cpu-gpu-mixed':
        kwargs.setdefault("copy", True)
        adata_mock = sc.pp.regress_out(adata, ['mito_perc', 'nUMIs'], **kwargs)
        adata.layers['regressed'] = adata_mock.X.copy()
        del adata_mock
    else:
        import rapids_singlecell as rsc
        adata.layers['regressed']=rsc.pp.regress_out(adata, ['mito_perc', 'nUMIs'], inplace=False)
    add_reference(adata,'scanpy','regressing out covariates with scanpy')

@monitor
@register_function(
    aliases=["回归并标准化", "regress_and_scale", "regress_scale", "校正后标准化", "regressed_scaled"],
    category="preprocessing",
    description="Apply covariate regression and z-score scaling in one convenience step",
    prerequisites={
        "functions": ["regress"],
        "optional_functions": ["scale"],
    },
    requires={
        "layers": ["regressed"]
    },
    produces={
        "layers": ["regressed_and_scaled"]
    },
    auto_fix="none",
    examples=[
        "ov.pp.regress(adata)",
        "ov.pp.regress_and_scale(adata)",
    ],
    related=["regress", "scale", "pca"],
)
def regress_and_scale(adata):
    """Scale the regressed layer and store it as a new analysis layer.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing ``adata.layers['regressed']``.

    Returns
    -------
    anndata.AnnData
        The same object with ``adata.layers['regressed_and_scaled']`` added.
    """
    if 'regressed' not in adata.layers:
        raise KeyError('Regress out covariates first!')
    adata_mock= adata.copy()
    adata_mock.X = adata_mock.layers['regressed']
    scale(adata_mock)
    adata.layers['regressed_and_scaled'] = adata_mock.layers['scaled']
    add_reference(adata,'scanpy','regressing out covariates with scanpy')
    return adata

from sklearn.decomposition import PCA 
class my_PCA:
    """
    A class to store the results of a sklearn PCA (i.e., embeddings, loadings and 
    explained variance ratios).
    """
    def __init__(self):
        self.n_pcs = None
        self.embs = None
        self.loads = None
        self.var_ratios = None

    def calculate_PCA(self, M, n_components=50):
        '''
        Perform PCA decomposition of some input obs x genes matrix.
        '''
        self.n_pcs = n_components
        # Convert to dense np.array if necessary)
        if isinstance(M, np.ndarray) is False:
            M = M.toarray()

        # Perform PCA
        model = PCA(n_components=n_components, random_state=1234)
        # Store results accordingly
        self.embs = np.round(model.fit_transform(M), 2) # Round for reproducibility
        self.loads = model.components_.T
        self.var_ratios = model.explained_variance_ratio_
        self.cum_sum_eigenvalues = np.cumsum(self.var_ratios)

        return self

@monitor
@register_function(
    aliases=["主成分分析", "pca", "PCA", "降维"],
    category="preprocessing",
    description="Perform Principal Component Analysis for dimensionality reduction",
    prerequisites={
        'functions': ['scale'],
        'optional_functions': ['qc', 'preprocess']
    },
    requires={
        'layers': ['scaled']
    },
    produces={
        'obsm': ['X_pca'],
        'varm': ['PCs'],
        'uns': ['pca']
    },
    auto_fix='escalate',
    examples=["ov.pp.pca(adata, n_pcs=50)"],
    related=["umap", "tsne", "mde"]
)
@tracked("pca", "ov.pp.pca")
def pca(adata, n_pcs=50, layer='scaled',inplace=True,**kwargs):
    """
    Performs Principal Component Analysis (PCA) on the data stored in a scanpy AnnData object.

    Arguments:
        adata : Annotated data matrix with rows representing cells 
        and columns representing features.
        n_pcs : Number of principal components to calculate.
        layer : The name of the layer in `adata` where the data to be analyzed is stored. 
        Defaults to the 'scaled' layer,
            and falls back to 'lognorm' if that layer does not exist. 
        Raises a KeyError if the specified layer is not present.

    Returns:
        adata : The original AnnData object with the calculated PCA embeddings 
        and other information stored in its `obsm`, `varm`,
            and `uns` fields.
    """
    #if 'lognorm' not in adata.layers:
    #    adata.layers['lognorm'] = adata.X
    # Implicit-centered scale: ov.pp.scale(use_implicit_centering=True) stores
    # the lazy CenteredSparseArray in adata.uns['_scaled_implicit'] rather
    # than a dense adata.layers['scaled'] (anndata's layer registry rejects
    # custom array types). When the caller asks for layer='scaled' and the
    # implicit wrapper is present, that counts as 'computed' — the deeper
    # _pca.pca() picks it up via the same uns key and densifies only the
    # HVG subset the SVD actually needs.
    _has_implicit_scaled = (
        layer == 'scaled' and '_scaled_implicit' in adata.uns
    )
    if layer in adata.layers:
        X = adata.layers[layer]
        key = f'{layer}|original'
    elif _has_implicit_scaled:
        X = adata.uns['_scaled_implicit']  # CenteredSparseArray
        key = f'{layer}|original'
    else:
        raise KeyError(f'Selected layer {layer} is not present. Compute it first!')
    if settings.mode == 'cpu':
        if sc.__version__ <'1.10':
            adata_mock=sc.AnnData(adata.layers[layer],obs=adata.obs,var=adata.var)
            sc.pp.pca(adata_mock, n_comps=n_pcs)
            adata.obsm[key + '|X_pca'] = adata_mock.obsm['X_pca']
            adata.varm[key + '|pca_loadings'] = adata_mock.varm['PCs']
            adata.uns[key + '|pca_var_ratios'] = adata_mock.uns['pca']['variance_ratio']
            adata.uns[key + '|cum_sum_eigenvalues'] = adata_mock.uns['pca']['variance']
        else:
            from ._pca import pca as _pca
            _pca(adata, layer=layer,n_comps=n_pcs,use_gpu=False,**kwargs)
            #print(res)
            adata.obsm[key + '|X_pca'] = adata.obsm['X_pca']
            adata.varm[key + '|pca_loadings'] = adata.varm['PCs']
            adata.uns[key + '|pca_var_ratios'] = adata.uns['pca']['variance_ratio']
            adata.uns[key + '|cum_sum_eigenvalues'] = adata.uns['pca']['variance']
    elif settings.mode == 'cpu-gpu-mixed':
        print(f"{EMOJI['gpu']} Using GPU to calculate PCA...")
        print_gpu_usage_color()
        from ._pca import pca as _pca
        if kwargs.get("svd_solver", "auto") in (None, "auto"):
            kwargs["svd_solver"] = "covariance_eigh"
        _pca(adata, layer=layer,n_comps=n_pcs,use_gpu=True,**kwargs)
        adata.obsm[key + '|X_pca'] = adata.obsm['X_pca']
        adata.varm[key + '|pca_loadings'] = adata.varm['PCs']
        adata.uns[key + '|pca_var_ratios'] = adata.uns['pca']['variance_ratio']
        adata.uns[key + '|cum_sum_eigenvalues'] = adata.uns['pca']['variance']
    else:
        import rapids_singlecell as rsc
        rsc.pp.pca(adata, layer=layer,n_comps=n_pcs,**kwargs)
        adata.obsm[key + '|X_pca'] = adata.obsm['X_pca']
        adata.varm[key + '|pca_loadings'] = adata.varm['PCs']
        adata.uns[key + '|pca_var_ratios'] = adata.uns['pca']['variance_ratio']
        adata.uns[key + '|cum_sum_eigenvalues'] = adata.uns['pca']['variance']
    
    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    if 'status_args' not in adata.uns.keys():
        adata.uns['status_args'] = {}

    adata.uns['status']['pca'] = True
    adata.uns['status_args']['pca']={
        'layer':layer,
        'n_pcs':n_pcs,
    }
    add_reference(adata,'scanpy','PCA with scanpy')
    note(backend=f"omicverse({settings.mode})",
         viz=[{"function": "ov.pl.plot_pca_variance_ratio",
                "kwargs": {"n_pcs": min(30, int(n_pcs or 50))}}])
    if inplace:
        return None
    else:
        return adata

@monitor
def red(adata):
    """
    Reduce the input AnnData object to highly variable features 
    and store the resulting expression matrices.

    Arguments:
        adata : Annotated data matrix with n_obs x n_vars shape. 
        Should contain a variable 'highly_variable_features'that 
        indicates which features are considered to be highly variable.

    Returns:
        adata : Annotated data matrix with n_obs x n_vars shape. 
        Adds new layers called 'lognorm' and 'raw' that store
            the logarithmized normalized expression matrix and 
            the unnormalized expression matrix, respectively.
            The matrix is reduced to the highly variable features only.

    """
    adata = adata[:, adata.var['highly_variable_features']].copy()
    adata.layers['lognorm'] = adata.X
    adata.layers['raw'] = adata.raw.to_adata()[:, adata.var_names].X
    return adata

def counts_store(adata,layers):
    '''
    counts store
    '''
    adata.uns[layers] = _safe_to_df_copy(adata.X)

def counts_retrieve(adata,layers):
    '''
    counts retrieve
    '''
    cell_idx=adata.obs.index
    adata.uns['raw_store'] = _safe_to_df_copy(adata.X)
    adata.X=adata.uns[layers].loc[cell_idx,:].values

from scipy.stats import median_abs_deviation
def is_outlier(adata, metric: str, nmads: int):
    '''
    Identify outliers based on specific metrics in a given AnnData object
    '''
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier


from types import MappingProxyType
from typing import (
    Union,
    Optional,
    Any,
    Mapping,
    Callable,
    NamedTuple,
    Generator,
    Tuple,
    Literal,
    )
N_DCS = 15  # default number of diffusion components

_Method = Literal['umap', 'gauss', 'rapids']
_MetricFn = Callable[[np.ndarray, np.ndarray], float]
# from sklearn.metrics.pairwise_distances.__doc__:
_MetricSparseCapable = Literal[
'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
_MetricScipySpatial = Literal[
    'braycurtis',
    'canberra',
    'chebyshev',
    'correlation',
    'dice',
    'hamming',
    'jaccard',
    'kulsinski',
    'mahalanobis',
    'minkowski',
    'rogerstanimoto',
    'russellrao',
    'seuclidean',
    'sokalmichener',
    'sokalsneath',
    'sqeuclidean',
    'yule',
    ]
_Metric = Union[_MetricSparseCapable, _MetricScipySpatial]
from types import MappingProxyType

@monitor
@register_function(
    aliases=["计算邻居", "neighbors", "knn", "邻居图"],
    category="preprocessing",
    description="Compute neighborhood graph of cells",
    prerequisites={
        'optional_functions': ['pca']
    },
    requires={
        'obsm': ['X_pca']
    },
    produces={
        'obsp': ['distances', 'connectivities'],
        'uns': ['neighbors']
    },
    auto_fix='auto',
    examples=["ov.pp.neighbors(adata, n_neighbors=15)"],
    related=["umap", "leiden", "louvain"]
)
@tracked("neighbors", "ov.pp.neighbors")
def neighbors(
    adata: anndata.AnnData,
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    knn: bool = True,
    random_state: int= 0,
    n_jobs: Optional[int] = None,
    method: Optional[_Method] = 'umap',
    transformer: Optional[str] = None,
    metric: Union[_Metric, _MetricFn] = 'euclidean',
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    key_added: Optional[str] = None,
    copy: bool = False,
    **kwargs,
) -> Optional[anndata.AnnData]:
    """
    Compute a neighborhood graph of observations [McInnes18]_.

    The neighbor search efficiency of this heavily relies on UMAP [McInnes18]_,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold (`method=='umap'`). If `method=='gauss'`,
    connectivities are computed according to [Coifman05]_, in the adaption of
    [Haghverdi16]_.

    Arguments:
        adata: Annotated data matrix.
        n_neighbors: The size of local neighborhood (in terms of number of neighboring data
            points) used for manifold approximation. Larger values result in more
            global views of the manifold, while smaller values result in more local
            data being preserved. In general values should be in the range 2 to 100.
            If `knn` is `True`, number of nearest neighbors to be searched. If `knn`
            is `False`, a Gaussian kernel width is set to the distance of the
            `n_neighbors` neighbor.
        knn: If `True`, use a hard threshold to restrict the number of neighbors to
            `n_neighbors`, that is, consider a knn graph. Otherwise, use a Gaussian
            Kernel to assign low weights to neighbors more distant than the
            `n_neighbors` nearest neighbor.
        random_state: A numpy random seed.
        n_jobs: Number of parallel jobs used by kNN backend. Defaults to scanpy-compatible behavior.
        method: Use 'umap' [McInnes18]_ or 'gauss' (Gauss kernel following [Coifman05]_
            with adaptive width [Haghverdi16]_) for computing connectivities.
            Use 'rapids' for the RAPIDS implementation of UMAP (experimental, GPU
            only). Use 'torch' for GPU-accelerated connectivity computation.
        transformer: KNN search implementation. Options: None (auto), 'pyg' (PyTorch Geometric,
            recommended for GPU), 'pynndescent', 'sklearn', or 'rapids'.
            'pyg' provides 20-100× speedup over other methods.
            In `cpu-gpu-mixed` mode, `None` defaults to `'pyg'`.
        metric: A known metric's name or a callable that returns a distance.
        metric_kwds: Options for the metric.
        key_added: If not specified, the neighbors data is stored in .uns['neighbors'],
            distances and connectivities are stored in .obsp['distances'] and
            .obsp['connectivities'] respectively.
            If specified, the neighbors data is added to .uns[key_added],
            distances are stored in .obsp[key_added+'_distances'] and
            connectivities in .obsp[key_added+'_connectivities'].
        copy: Return a copy instead of writing to adata.

    Returns: Depending on `copy`, updates or returns `adata` with the following:

    See `key_added` parameter description for the storage path of
    connectivities and distances.

    **connectivities** : sparse matrix of dtype `float32`.
        Weighted adjacency matrix of the neighborhood graph of data
        points. Weights should be interpreted as connectivities.
    **distances** : sparse matrix of dtype `float32`.
        Instead of decaying weights, this stores distances for each pair of
        neighbors.

    Notes
    -----
    If `method='umap'`, it's highly recommended to install pynndescent ``pip install pynndescent``.
    Installing `pynndescent` can significantly increase performance,
    and in later versions it will become a hard dependency.
    
    """
    # Ensure PCA exists; compute a default if missing so downstream code can proceed
    if settings.mode =='cpu':
        print(f"{EMOJI['cpu']} Using Scanpy CPU to calculate neighbors...")
        from ._neighbors import neighbors as _neighbors
        _neighbors(adata,use_rep=use_rep,n_neighbors=n_neighbors, n_pcs=n_pcs,
                         random_state=random_state,n_jobs=n_jobs,method=method,transformer=transformer,
                         metric=metric,metric_kwds=metric_kwds,
                         key_added=key_added,copy=copy,**kwargs)
    elif settings.mode == 'cpu-gpu-mixed':
        print(f"{EMOJI['gpu']} Using torch CPU/GPU mixed mode to calculate neighbors...")
        print_gpu_usage_color()
        effective_transformer = transformer if transformer is not None else "pyg"
        if transformer is None:
            print(f"{EMOJI['gpu']} Mixed mode default transformer: pyg")
        from ._neighbors import neighbors as _neighbors
        _neighbors(adata,use_rep=use_rep,n_neighbors=n_neighbors, n_pcs=n_pcs,
                         random_state=random_state,n_jobs=n_jobs,method='torch',
                         transformer=effective_transformer,
                         metric=metric,metric_kwds=metric_kwds,
                         key_added=key_added,copy=copy,**kwargs)
    else:
        print(f"{EMOJI['gpu']} Using RAPIDS GPU to calculate neighbors...")
        import rapids_singlecell as rsc
        rsc.pp.neighbors(adata,use_rep=use_rep,n_neighbors=n_neighbors, n_pcs=n_pcs,
                         random_state=random_state,algorithm=method,metric=metric,
                         metric_kwds=metric_kwds,
                         key_added=key_added,copy=copy,**kwargs)
    add_reference(adata,'scanpy','neighbors with scanpy')
    note(backend=f"omicverse({settings.mode}) · method={method}",
         viz=[{"function": "ov.pl.neighbor_degree_histogram", "kwargs": {}}])

@monitor
@register_function(
    aliases=["umap", "UMAP", "非线性降维"],
    category="preprocessing",
    description="Compute UMAP embedding for visualization",
    prerequisites={
        'functions': ['neighbors'],
        'optional_functions': ['pca']
    },
    requires={
        'uns': ['neighbors'],
        'obsp': ['connectivities', 'distances']
    },
    produces={
        'obsm': ['X_umap']
    },
    auto_fix='auto',
    examples=["ov.pp.umap(adata)"],
    related=["tsne", "pca", "mde", "neighbors"]
)
@tracked("umap", "ov.pp.umap")
def umap(
    adata,
    *,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos=None,
    random_state=0,
    a: float | None = None,
    b: float | None = None,
    method: str | None = None,
    key_added: str | None = None,
    neighbors_key: str = "neighbors",
    copy: bool = False,
    **kwargs,
):
    """Compute UMAP embedding, dispatching to the best backend for
    ``ov.settings.mode``.

    Parameters
    ----------
    adata
        Annotated data matrix; neighbors must already be computed.
    min_dist
        Minimum distance between embedded points. Default ``0.5`` matches
        scanpy; smaller gives more densely packed clusters.
    spread
        Scale at which embedded points are spread out.
    n_components
        Output dimensions (2 for typical visualisation).
    maxiter
        Number of edge-SGD passes. Defaults are backend-specific — scanpy
        uses 200 for N>10k, 500 otherwise.
    alpha
        Initial learning rate of the edge-SGD optimiser. Only used by
        non-parametric backends.
    gamma
        Weighting for negative samples. Default ``1.0`` matches scanpy.
    negative_sample_rate
        Number of negative samples per positive edge. Default ``5``.
    init_pos
        Embedding initialisation: ``'paga'``, ``'spectral'``, ``'random'``,
        or an ``ndarray``. ``None`` lets the backend pick.
    random_state
        Seed for reproducibility.
    a, b
        UMAP curve parameters. ``None`` derives them from ``spread`` +
        ``min_dist`` via ``umap.umap_.find_ab_params``.
    method
        Explicit backend override, e.g. ``'umap'`` (scanpy/umap-learn),
        ``'pumap'`` (omicverse's parametric GPU UMAP), ``'torchdr'``,
        ``'mde'``, ``'rapids'``. ``None`` picks the right default for the
        current ``ov.settings.mode``.
    key_added
        Key under which to store results; default ``'X_umap'`` /
        ``'umap'``.
    neighbors_key
        ``.uns`` key holding the precomputed neighbor graph.
    copy
        Return a new AnnData instead of modifying in place.
    **kwargs
        Forwarded to the underlying backend (``_umap.umap``,
        ``rapids_singlecell.tl.umap`` etc.).

    Notes
    -----
    Behaviour by ``ov.settings.mode``:

    * ``'cpu'``                    — scanpy / umap-learn (CPU).
    * ``'cpu-gpu-mixed'``          — parametric UMAP on GPU (pumap).
    * anything else (``'gpu'``)    — RAPIDS if available, falls back to
                                     pumap on import error.
    """
    # Build the kwargs dict that downstream dispatchers expect. Keep
    # `method` and `init_pos` optional so each branch can pick its default.
    forward = dict(
        min_dist=min_dist, spread=spread, n_components=n_components,
        maxiter=maxiter, alpha=alpha, gamma=gamma,
        negative_sample_rate=negative_sample_rate, random_state=random_state,
        a=a, b=b, neighbors_key=neighbors_key, copy=copy,
    )
    if init_pos is not None:
        forward["init_pos"] = init_pos
    if method is not None:
        forward["method"] = method
    if key_added is not None:
        forward["key_added"] = key_added
    forward.update(kwargs)
    kwargs = forward
    print(f"{EMOJI['start']} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running UMAP in '{settings.mode}' mode...")
    try:
        if settings.mode == 'cpu':
            print(f"{EMOJI['cpu']} Using Scanpy CPU UMAP...")
            from ._umap import umap as _umap
            _umap(adata, **kwargs)
            add_reference(adata, 'umap', 'UMAP with scanpy')
            note(backend=f"omicverse({settings.mode}) · scanpy")
        elif settings.mode == 'cpu-gpu-mixed':
            print(f"{EMOJI['gpu']} Using torch GPU to calculate UMAP...")
            print_gpu_usage_color()
            from ._umap import umap as _umap
            _umap(adata, method='pumap', **kwargs)
            add_reference(adata, 'umap', 'UMAP with pumap')
            note(backend=f"omicverse({settings.mode}) · pumap")
        else:
            try:
                print(f"{EMOJI['gpu']} Using RAPIDS GPU UMAP...")
                import rapids_singlecell as rsc
                rsc.tl.umap(adata, **kwargs)
                add_reference(adata, 'umap', 'UMAP with RAPIDS')
                note(backend=f"omicverse({settings.mode}) · rapids")
            except Exception as e:
                print(f"{EMOJI['error']} RAPIDS GPU UMAP failed: {e}")
                print(f"{EMOJI['error']} Using pumap instead...")
                from ._umap import umap as _umap
                _umap(adata, method='pumap', **kwargs)
                note(backend=f"omicverse({settings.mode}) · pumap")
        note(viz=[{"function": "ov.pl.embedding",
                    "kwargs": {"basis": "X_umap",
                               "color": pick_color_key(adata),
                               "frameon": "small"}}])
        print(f"{EMOJI['done']} UMAP completed successfully.")
    except Exception as e:
        print(f"{EMOJI['error']} UMAP failed: {e}")
        raise

@monitor
@register_function(
    aliases=["Louvain聚类", "louvain", "community_detection", "图聚类", "louvain_clustering"],
    category="preprocessing",
    description="Cluster cells by Louvain community detection on the neighborhood graph",
    prerequisites={
        "functions": ["neighbors"],
        "optional_functions": ["pca", "umap"],
    },
    requires={
        "uns": ["neighbors"],
        "obsp": ["connectivities"],
    },
    produces={
        "obs": ["louvain"]
    },
    auto_fix="auto",
    examples=[
        "ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)",
        "ov.pp.louvain(adata, resolution=1.0)",
    ],
    related=["leiden", "neighbors", "umap"],
)
@tracked("louvain", "ov.pp.louvain")
def louvain(
    adata,
    resolution=None,
    *,
    random_state=0,
    key_added="louvain",
    neighbors_key=None,
    directed=True,
    use_weights=False,
    copy=False,
    **kwargs,
):
    """Run Louvain clustering on the precomputed kNN graph.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with a neighborhood graph already computed.
    resolution
        Resolution γ; higher values yield more, smaller clusters.
    random_state
        Seed for reproducibility.
    key_added
        Column in ``adata.obs`` to store labels. Default ``'louvain'``.
    neighbors_key
        Key in ``adata.uns`` for the neighbor dict. ``None`` uses the
        standard ``'neighbors'`` slot.
    directed, use_weights
        Forwarded to the igraph backend; rarely need tuning.
    copy
        Return a new AnnData instead of modifying in place.
    **kwargs
        Forwarded to the underlying backend
        (``scanpy.tl.louvain`` or ``rapids_singlecell.tl.louvain``).

    Notes
    -----
    In ``ov.settings.mode='cpu-gpu-mixed'`` this is auto-routed to
    ``ov.pp.leiden`` (Louvain has no GPU backend and Leiden is its strict
    improvement). Labels are still written to ``adata.obs[key_added]`` so
    existing notebooks keep working.
    """
    if resolution is not None:
        kwargs["resolution"] = resolution
    kwargs.setdefault("random_state", random_state)
    kwargs.setdefault("key_added", key_added)
    if neighbors_key is not None:
        kwargs["neighbors_key"] = neighbors_key
    kwargs.setdefault("directed", directed)
    kwargs.setdefault("use_weights", use_weights)
    kwargs.setdefault("copy", copy)

    if settings.mode == 'cpu-gpu-mixed':
        # Louvain has no GPU implementation in mixed mode, but Leiden is its
        # strict improvement (Traag et al. 2019) and we have a fast GPU
        # version. Auto-redirect with a deprecation warning, writing results
        # under the requested key so existing notebooks keep working.
        import warnings
        key_added = kwargs.pop('key_added', 'louvain')
        # Only forward kwargs that ov.pp.leiden actually accepts. Anything
        # else (Louvain/igraph-specific like partition_type / node_sizes) is
        # dropped with a visible warning so users aren't surprised.
        _leiden_accepted = {
            "resolution", "random_state", "local_iterations", "max_levels",
            "device", "symmetrize",
        }
        dropped = {k: v for k, v in kwargs.items() if k not in _leiden_accepted}
        safe_kwargs = {k: v for k, v in kwargs.items() if k in _leiden_accepted}
        if dropped:
            warnings.warn(
                "Dropped louvain-specific kwargs that ov.pp.leiden does not "
                f"accept: {sorted(dropped)}. Either remove them or switch "
                "to ov.pp.leiden directly.",
                UserWarning,
                stacklevel=2,
            )
        warnings.warn(
            "ov.pp.louvain in cpu-gpu-mixed mode is deprecated: there is no "
            "GPU Louvain backend, and Leiden is a strict improvement on "
            "modularity-quality grounds (Traag et al. 2019). Auto-routing to "
            "ov.pp.leiden; switch your call to ov.pp.leiden to silence this "
            f"warning. Result is still stored at adata.obs[{key_added!r}].",
            DeprecationWarning,
            stacklevel=2,
        )
        print(f"{EMOJI['mixed']} cpu-gpu-mixed: louvain auto-routed to GPU Leiden (deprecation)")
        leiden(adata, key_added=key_added, **safe_kwargs)
        return
    if settings.mode == 'cpu':
        print(f"{EMOJI['cpu']} Using Scanpy CPU Louvain...")
        sc.tl.louvain(adata, **kwargs)
        add_reference(adata, 'louvain', 'Louvain clustering with scanpy')
        note(backend=f"omicverse({settings.mode}) · scanpy")
    else:
        print(f"{EMOJI['gpu']} Using RAPIDS GPU to calculate Louvain...")
        import rapids_singlecell as rsc
        rsc.tl.louvain(adata, **kwargs)
        add_reference(adata, 'louvain', 'Louvain clustering with RAPIDS')
        note(backend=f"omicverse({settings.mode}) · rapids")
    note(viz=[
        {"function": "ov.pl.cluster_sizes_bar",
          "kwargs": {"groupby": key_added}},
        *([{"function": "ov.pl.embedding",
             "kwargs": {"basis": "X_umap", "color": key_added,
                        "frameon": "small"}}] if "X_umap" in adata.obsm else []),
    ])

@monitor
@register_function(
    aliases=["莱顿聚类", "leiden", "clustering", "聚类"],
    category="preprocessing",
    description="Perform Leiden community detection clustering",
    prerequisites={
        'functions': ['neighbors'],
        'optional_functions': ['pca', 'umap']
    },
    requires={
        'uns': ['neighbors'],
        'obsp': ['connectivities']
    },
    produces={
        'obs': ['leiden']
    },
    auto_fix='auto',
    examples=["ov.pp.leiden(adata, resolution=1.0)"],
    related=["louvain", "neighbors"]
)
@tracked("leiden", "ov.pp.leiden")
def leiden(
    adata, resolution=1.0, random_state=0,
    key_added='leiden', local_iterations=100, max_levels=10, device=None, symmetrize=None, **kwargs):
    '''
    leiden clustering
    '''
    if settings.mode == 'cpu':
        print(f"{EMOJI['cpu']} Using Scanpy CPU Leiden...")
        from ._leiden import leiden as _leiden
        _leiden(adata, resolution=resolution, random_state=random_state,
            key_added=key_added, **kwargs)
        add_reference(adata, 'leiden', 'Leiden clustering with scanpy')
        note(backend=f"omicverse({settings.mode}) · scanpy")
    elif settings.mode == 'cpu-gpu-mixed':
        print(f"{EMOJI['mixed']} Using torch CPU/GPU mixed mode to calculate Leiden...")
        print_gpu_usage_color()
        from ._leiden_gpu import leiden_gpu_sparse_multilevel as _leiden
        _leiden(
            adata,
            resolution=resolution,
            random_state=random_state,
            key_added=key_added,
            local_iterations=local_iterations,
            max_levels=max_levels,
            device=device,
            symmetrize=symmetrize,
            **kwargs,
        )
        add_reference(adata, 'leiden', 'Leiden clustering with omicverse')
        note(backend=f"omicverse({settings.mode}) · torch")
    else:
        print(f"{EMOJI['gpu']} Using RAPIDS GPU to calculate Leiden...")
        import rapids_singlecell as rsc
        rsc.tl.leiden(adata, resolution=resolution, random_state=random_state,
            key_added=key_added, **kwargs)
        add_reference(adata, 'leiden', 'Leiden clustering with RAPIDS')
        note(backend=f"omicverse({settings.mode}) · rapids")
    note(viz=[
        {"function": "ov.pl.cluster_sizes_bar",
          "kwargs": {"groupby": key_added}},
        *([{"function": "ov.pl.embedding",
             "kwargs": {"basis": "X_umap", "color": key_added,
                        "frameon": "small"}}] if "X_umap" in adata.obsm else []),
    ])

@monitor
@register_function(
    aliases=["细胞周期评分", "score_genes_cell_cycle", "cell_cycle", "细胞周期", "cc_score"],
    category="preprocessing",
    description="Score cell cycle phases (S and G2M) using predefined gene sets",
    prerequisites={
        'optional_functions': ['qc', 'preprocess']
    },
    produces={
        'obs': ['S_score', 'G2M_score', 'phase']
    },
    auto_fix='none',
    examples=[
        "# Basic cell cycle scoring for human data",
        "ov.pp.score_genes_cell_cycle(adata, species='human')",
        "# Mouse cell cycle scoring",
        "ov.pp.score_genes_cell_cycle(adata, species='mouse')",
        "# Custom gene lists",
        "s_genes = ['MCM5', 'PCNA', 'TYMS']",
        "g2m_genes = ['HMGB2', 'CDK1', 'NUSAP1']",
        "ov.pp.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)",
        "# Visualize cell cycle phases",
        "ov.pl.embedding(adata, basis='X_umap', color='phase')"
    ],
    related=["pp.preprocess", "pl.embedding", "utils.embedding"]
)
@_oom_guard(
    materialize=True,
    result_keys_obs=['S_score', 'G2M_score', 'phase'],
    result_keys_uns=['*'],
)
@tracked("score_genes_cell_cycle", "ov.pp.score_genes_cell_cycle")
def score_genes_cell_cycle(adata,species='human',s_genes=None, g2m_genes=None):
    """Score cell cycle phases using predefined or custom gene sets.

    Arguments:
        adata: Annotated data matrix with rows for cells and columns for genes.
        species: The species of the data ('human' or 'mouse'). Default: 'human'.
        s_genes: Custom list of S phase genes. Default: None (uses predefined).
        g2m_genes: Custom list of G2M phase genes. Default: None (uses predefined).

    Returns:
        None: Updates adata.obs with 'S_score', 'G2M_score', and 'phase' columns.

    Examples:
        >>> import omicverse as ov
        >>> # Score cell cycle for human data
        >>> ov.pp.score_genes_cell_cycle(adata, species='human')
        >>> # Check results
        >>> print(adata.obs[['S_score', 'G2M_score', 'phase']].head())
    """
    if s_genes is None:
        if species=='human':
            s_genes=['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 
            'RRM1', 'UNG', 'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 
            'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', 
            'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 
            'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM',
             'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']
        elif species=='mouse':
            s_genes=['Cdca7', 'Mcm4', 'Mcm7', 'Rfc2', 'Ung', 'Mcm6', 
            'Rrm1', 'Slbp', 'Pcna', 'Atad2', 'Tipin', 'Mcm5', 'Uhrf1', 
            'Polr1b', 'Dtl', 'Prim1', 'Fen1', 'Hells', 'Gmnn', 'Pold3', 
            'Nasp', 'Chaf1b', 'Gins2', 'Pola1', 'Msh2', 'Casp8ap2', 'Cdc6',
             'Ubr7', 'Ccne2', 'Wdr76', 'Tyms', 'Cdc45', 'Clspn', 'Rrm2', 
             'Dscc1', 'Rad51', 'Usp1', 'Exo1', 'Blm', 'Rad51ap1', 'Cenpu', 'E2f8', 'Mrpl36']       
    if g2m_genes is None:
        if species=='human':
            g2m_genes=['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 
            'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 
            'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 
            'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1',
             'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C',
              'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR',
               'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 
               'G2E3', 'GAS2L3', 'CBX5', 'CENPA']
        elif species=='mouse':
            g2m_genes=['Cbx5', 'Aurkb', 'Cks1b', 'Cks2', 'Jpt1', 'Hmgb2', 
            'Anp32e', 'Lbr', 'Tmpo', 'Top2a', 'Tacc3', 'Tubb4b', 'Ncapd2', 
            'Rangap1', 'Cdk1', 'Smc4', 'Kif20b', 'Cdca8', 'Ckap2', 'Ndc80',
             'Dlgap5', 'Hjurp', 'Ckap5', 'Bub1', 'Ckap2l', 'Ect2', 'Kif11',
              'Birc5', 'Cdca2', 'Nuf2', 'Cdca3', 'Nusap1', 'Ttk', 'Aurka', 
              'Mki67', 'Pimreg', 'Ccnb2', 'Tpx2', 'Hjurp', 'Anln', 'Kif2c',
               'Cenpe', 'Gtse1', 'Kif23', 'Cdc20', 'Ube2c', 'Cenpf', 'Cenpa',
                'Hmmr', 'Ctcf', 'Psrc1', 'Cdc25c', 'Nek2', 'Gas2l3', 'G2e3']
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    if 'status_args' not in adata.uns.keys():
        adata.uns['status_args'] = {}
    adata.uns['status']['cell_cycle'] = True
    adata.uns['status_args']['cell_cycle'] = {'s_genes': s_genes,
                                               'g2m_genes': g2m_genes}
    note(viz=[{"function": "ov.pl.cluster_sizes_bar",
                "kwargs": {"groupby": "phase",
                            "title": "Cell cycle phase",
                            "xlabel": "phase"}}])

@monitor
@register_function(
    aliases=["tSNE", "tsne", "t-sne", "非线性降维", "细胞可视化嵌入"],
    category="preprocessing",
    description="Compute t-SNE embedding for nonlinear visualization of cell states",
    prerequisites={
        "functions": ["pca"],
        "optional_functions": ["neighbors"],
    },
    requires={
        "obsm": ["X_pca"]
    },
    produces={
        "obsm": ["X_tsne"]
    },
    auto_fix="auto",
    examples=[
        "ov.pp.tsne(adata, n_pcs=50)",
        "ov.pp.tsne(adata, perplexity=30, random_state=0)",
    ],
    related=["umap", "mde", "pca"],
)
@tracked("tsne", "ov.pp.tsne")
def tsne(
    adata,
    n_pcs=None,
    *,
    n_components: int = 2,
    use_rep: str | None = None,
    perplexity: float = 30,
    metric: str = "euclidean",
    early_exaggeration: float = 12,
    learning_rate: float = 1000,
    random_state=0,
    key_added: str | None = None,
    copy: bool = False,
    n_iter: int | None = None,
    **kwargs,
):
    """Compute t-SNE coordinates for cells, dispatching by
    ``ov.settings.mode``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with PCA (or another ``use_rep``) available.
    n_pcs
        Number of PCs to use from ``adata.obsm['X_pca']``.
    n_components
        Output dimensions (typically 2).
    use_rep
        ``.obsm`` key to compute t-SNE on; defaults to ``X_pca``.
    perplexity
        Controls the effective number of nearest neighbors used to
        construct the high-D affinities. Typical range 5–50.
    metric
        Distance metric for neighbor search. ``'euclidean'`` by default.
    early_exaggeration
        Scaling applied to P during the early-exaggeration phase
        (first ~250 iterations). Larger makes clusters tighter in the
        output.
    learning_rate
        Learning rate of the optimiser. ``1000`` is the sklearn default;
        the GPU backend auto-rescales to a much smaller value internally.
    random_state
        Seed for reproducibility.
    key_added
        Target ``.obsm`` / ``.uns`` key. Defaults to ``'X_tsne'`` / ``'tsne'``.
    copy
        Return a new AnnData instead of modifying in place.
    n_iter
        Total optimisation iterations. ``None`` uses the backend default.
    **kwargs
        Forwarded to the backend (``scanpy.tl.tsne``,
        ``omicverse.pp._tsne.tsne``, or ``rapids_singlecell.tl.tsne``).

    Notes
    -----
    ``cpu-gpu-mixed`` mode uses our KL-loss native torch t-SNE
    (see ``omicverse/external/torch_tsne.py``) — no ``torchdr`` dependency.
    """
    kwargs.setdefault("n_pcs", n_pcs)
    kwargs.setdefault("n_components", n_components)
    if use_rep is not None:
        kwargs.setdefault("use_rep", use_rep)
    kwargs.setdefault("perplexity", perplexity)
    kwargs.setdefault("metric", metric)
    kwargs.setdefault("early_exaggeration", early_exaggeration)
    kwargs.setdefault("learning_rate", learning_rate)
    kwargs.setdefault("random_state", random_state)
    if key_added is not None:
        kwargs.setdefault("key_added", key_added)
    kwargs.setdefault("copy", copy)
    if n_iter is not None:
        kwargs.setdefault("n_iter", n_iter)
    if settings.mode == 'cpu':
        # CPU t-SNE goes through our own sklearn-backed wrapper rather
        # than ``scanpy.tl.tsne`` (which does not accept n_components —
        # issue #683). Going direct to sklearn means n_components /
        # n_iter / metric all flow through unchanged regardless of the
        # installed scanpy version, and CPU users finally get k-D t-SNE.
        print(f"{EMOJI['cpu']} Using sklearn CPU t-SNE...")
        from ._tsne_cpu import tsne_cpu
        tsne_cpu(adata, **kwargs)
        add_reference(adata, 'tsne', 't-SNE with sklearn')
        note(backend=f"omicverse({settings.mode}) · sklearn")
    elif settings.mode == 'cpu-gpu-mixed':
        print(f"{EMOJI['mixed']} Using torch CPU/GPU mixed mode to calculate t-SNE...")
        print_gpu_usage_color()
        from ._tsne import tsne as _tsne
        _tsne(adata, **kwargs)
        add_reference(adata, 'tsne', 't-SNE with omicverse')
        note(backend=f"omicverse({settings.mode}) · torch")
    else:
        print(f"{EMOJI['gpu']} Using RAPIDS GPU to calculate t-SNE...")
        import rapids_singlecell as rsc
        rsc.tl.tsne(adata, **kwargs)
        add_reference(adata, 'tsne', 't-SNE with RAPIDS')
        note(backend=f"omicverse({settings.mode}) · rapids")
    note(viz=[{"function": "ov.pl.embedding",
                "kwargs": {"basis": "X_tsne",
                            "color": pick_color_key(adata),
                            "frameon": "small"}}])


@register_function(
    aliases=["MDE降维", "mde", "minimum_distortion_embedding", "嵌入优化", "低维嵌入"],
    category="preprocessing",
    description="Compute PyMDE embedding to preserve neighborhood geometry in low dimensions",
    prerequisites={
        "functions": ["pca"],
        "optional_functions": ["neighbors", "umap"],
    },
    requires={
        "obsm": ["X_pca"]
    },
    produces={
        "obsm": ["X_mde"]
    },
    auto_fix="auto",
    examples=[
        "ov.pp.mde(adata, embedding_dim=2, n_neighbors=15, use_rep='X_pca')",
        "ov.pp.mde(adata, repulsive_fraction=0.7, basis='X_mde')",
    ],
    related=["umap", "tsne", "pca"],
)
def mde(adata,embedding_dim=2,n_neighbors=15, basis='X_mde',n_pcs=None, use_rep=None, knn=True, 
        transformer=None, metric='euclidean',verbose=False,
        key_added=None,random_state=0,repulsive_fraction=0.7,constraint=None):
    """Run MDE (Minimum Distortion Embedding) from a latent representation.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing a latent representation in ``adata.obsm``.
    embedding_dim : int, default=2
        Number of dimensions in the output embedding.
    n_neighbors : int, default=15
        Number of neighbors used for graph construction in MDE.
    basis : str, default="X_mde"
        Key in ``adata.obsm`` where the result is stored.
    n_pcs : int, optional
        Number of principal components used from ``use_rep``.
    use_rep : str, optional
        Input representation key in ``adata.obsm``. Defaults to ``'X_pca'``.
    knn : bool, default=True
        Whether to use k-nearest-neighbor graph initialization.
    transformer : object, optional
        Optional neighbor transformer object.
    metric : str, default="euclidean"
        Distance metric for neighborhood search.
    verbose : bool, default=False
        Whether to print detailed optimization logs.
    key_added : str, optional
        Optional key alias for metadata outputs.
    random_state : int, default=0
        Random seed for reproducibility.
    repulsive_fraction : float, default=0.7
        Repulsion weight controlling global separation.
    constraint : object, optional
        PyMDE constraint object. Uses ``pymde.Standardized()`` when ``None``.

    Returns
    -------
    None
        Stores the low-dimensional embedding in ``adata.obsm[basis]``.
    """
    import pymde
    import logging
    import time

    # 配置日志
    #logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    #logger = logging.getLogger()
    # 记录开始时间
    start_time = time.time()

    print(f"{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} MDE Dimensionality Reduction:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Mode: {Colors.BOLD}{settings.mode}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Embedding dimensions: {Colors.BOLD}{embedding_dim}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Neighbors: {Colors.BOLD}{n_neighbors}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Repulsive fraction: {Colors.BOLD}{repulsive_fraction}{Colors.ENDC}")
    
    if use_rep is None:
        use_rep='X_pca'
    print(f"   {Colors.CYAN}Using representation: {Colors.BOLD}{use_rep}{Colors.ENDC}")
    data=adata.obsm[use_rep]
    if n_pcs is None:
        n_pcs=50
    print(f"   {Colors.CYAN}Principal components: {Colors.BOLD}{n_pcs}{Colors.ENDC}")
    data=data[:,:n_pcs]

    # Determine device based on available accelerators
    import torch
    from .._settings import get_optimal_device, prepare_data_for_device
    device = get_optimal_device(prefer_gpu=True, verbose=True)
    
    # Convert device to string for PyMDE compatibility
    if hasattr(device, 'type'):
        device_str = device.type
    else:
        device_str = str(device)
    
    # Prepare data for MPS compatibility (float32 requirement)
    data = prepare_data_for_device(data, device, verbose=True)
    
    print(f"   {Colors.GREEN}{EMOJI['start']} Computing k-nearest neighbors graph...{Colors.ENDC}")
    
    if constraint is None:
        _kwargs = {
        "embedding_dim": embedding_dim,
        "constraint": pymde.Standardized(),
        "repulsive_fraction": repulsive_fraction,
        "verbose": verbose,
        "device": device_str,
        "n_neighbors": n_neighbors,
    }
    else:
        _kwargs = {
        "embedding_dim": embedding_dim,
        "constraint": constraint,
        "repulsive_fraction": repulsive_fraction,
        "verbose": verbose,
        "device": device_str,
        "n_neighbors": n_neighbors,
    }
    #_kwargs.update(kwargs)
    
    gr=pymde.preprocess.k_nearest_neighbors(data,k=n_neighbors)
    
    print(f"   {Colors.GREEN}{EMOJI['start']} Creating MDE embedding...{Colors.ENDC}")
    mde = pymde.preserve_neighbors(data, **_kwargs)
    print(f"   {Colors.GREEN}{EMOJI['start']} Optimizing embedding...{Colors.ENDC}")
    emb=mde.embed(verbose=_kwargs["verbose"])

    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
    from scipy.spatial import KDTree
    # 使用KNN算法找到最近邻居
    n_neighbors = n_neighbors  # 设置KNN的邻居数
    kdtree = KDTree(emb)
    distances, indices = kdtree.query(emb, k=n_neighbors)

    # 构建稀疏的距离矩阵
    n_items = emb.shape[0]
    row_indices = np.repeat(np.arange(n_items), n_neighbors)
    col_indices = indices.flatten()
    distances = distances.flatten()
    sparse_distance_matrix = csr_matrix((distances, (row_indices, col_indices)), \
    shape=(n_items, n_items))

    # 构建连接矩阵
    # 这里我们简单地将连接矩阵设置为距离矩阵的二值化形式
    connectivities = (sparse_distance_matrix > 0).astype(float)

    if key_added is None:
        key_added = "neighbors"
        conns_key = "connectivities"
        dists_key = "distances"
    else:
        conns_key = key_added + "_connectivities"
        dists_key = key_added + "_distances"

    adata.uns[key_added] = {}

    neighbors_dict = adata.uns[key_added]

    neighbors_dict["connectivities_key"] = conns_key
    neighbors_dict["distances_key"] = dists_key

    neighbors_dict["params"] = dict(
        n_neighbors=n_neighbors,
        method='mde',
        random_state=random_state,
        metric=metric,
    )
    if use_rep is not None:
        neighbors_dict["params"]["use_rep"] = use_rep
    if n_pcs is not None:
        neighbors_dict["params"]["n_pcs"] = n_pcs


    # 创建或更新AnnData对象
    #adata = anndata.AnnData(X=data)
    adata.obsp[dists_key] = sparse_distance_matrix
    adata.obsp[conns_key] = gr.adjacency_matrix
    adata.obsm[basis]=emb
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n")
    print(f"{Colors.GREEN}{EMOJI['done']} MDE Dimensionality Reduction Completed Successfully!{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Embedding shape: {Colors.BOLD}{emb.shape[0]:,}{Colors.ENDC}{Colors.GREEN} cells × {Colors.BOLD}{emb.shape[1]}{Colors.ENDC}{Colors.GREEN} dimensions{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Runtime: {Colors.BOLD}{elapsed_time:.2f}s{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Results added to AnnData object:{Colors.ENDC}")
    print(f"     {Colors.CYAN}• '{basis}': {Colors.BOLD}MDE coordinates{Colors.ENDC}{Colors.CYAN} (adata.obsm){Colors.ENDC}")
    print(f"     {Colors.CYAN}• '{key_added}': {Colors.BOLD}Neighbors metadata{Colors.ENDC}{Colors.CYAN} (adata.uns){Colors.ENDC}")
    print(f"     {Colors.CYAN}• '{dists_key}': {Colors.BOLD}Distance matrix{Colors.ENDC}{Colors.CYAN} (adata.obsp){Colors.ENDC}")
    print(f"     {Colors.CYAN}• '{conns_key}': {Colors.BOLD}Connectivity matrix{Colors.ENDC}{Colors.CYAN} (adata.obsp){Colors.ENDC}")

    add_reference(adata,'pymde','MDE with pymde')
    #return emb


import numpy as np

def calc_mean_and_var_sparse(M, N, data, indices, indptr, axis):
    i, j=0,0

    size=0
    value=0.0

    size = N if axis == 0 else M
    mean = np.zeros(size, dtype = np.float64)
    var = np.zeros(size, dtype = np.float64)

    mean_view = mean
    var_view = var

    for i in range(M):
        for j in range(indptr[i], indptr[i + 1]):
            value = data[j]
            if axis == 0:
                mean_view[indices[j]] += value
                var_view[indices[j]] += value * value
            else:
                mean_view[i] += value
                var_view[i] += value * value

    size = M if axis == 0 else N
    for i in range(mean_view.size):
        mean_view[i] /= size
        var_view[i] = (var_view[i] - size * mean_view[i] * mean_view[i]) / (size - 1)

    return mean, var

def calc_stat_per_batch_sparse(M, N, data, indices, indptr,nbatch, codes):
    i, j=0,0

    col=0
    code=0
    value=0.0

    ncells = np.zeros(nbatch, dtype = np.int32)
    means = np.zeros((N, nbatch), dtype = np.float64)
    partial_sum = np.zeros((N, nbatch), dtype = np.float64)

    ncells_view = ncells
    means_view = means
    ps_view = partial_sum

    for i in range(M):
        code = codes[i]
        ncells_view[code] += 1
        for j in range(indptr[i], indptr[i + 1]):
            col = indices[j]
            value = data[j]
            means_view[col, code] += value
            ps_view[col, code] += value * value

    for j in range(nbatch):
        if ncells_view[j] > 1:
            for i in range(N):
                means_view[i, j] /= ncells_view[j]
                ps_view[i, j] = ps_view[i, j] - ncells_view[j] * means_view[i, j] * means_view[i, j]

    return ncells, means, partial_sum

def calc_mean_and_var_dense(M, N, X, axis):
    i, j=0,0

    size=0
    value=0.0

    size = N if axis == 0 else M
    mean = np.zeros(size, dtype = np.float64)
    var = np.zeros(size, dtype = np.float64)

    mean_view = mean
    var_view = var

    for i in range(M):
        for j in range(N):
            value = X[i, j]
            if axis == 0:
                mean_view[j] += value
                var_view[j] += value * value
            else:
                mean_view[i] += value
                var_view[i] += value * value

    size = M if axis == 0 else N
    for i in range(mean_view.size):
        mean_view[i] /= size
        var_view[i] = (var_view[i] - size * mean_view[i] * mean_view[i]) / (size - 1)

    return mean, var


def calc_stat_per_batch_dense(M, N, X, nbatch, codes):
    i, j=0,0

    code, col=0,0
    value=0.0

    ncells = np.zeros(nbatch, dtype = np.int32)
    means = np.zeros((N, nbatch), dtype = np.float64)
    partial_sum = np.zeros((N, nbatch), dtype = np.float64)

    ncells_view = ncells
    means_view = means
    ps_view = partial_sum

    for i in range(M):
        code = codes[i]
        ncells_view[code] += 1
        for j in range(N):
            value = X[i, j]
            means_view[j, code] += value
            ps_view[j, code] += value * value

    for j in range(nbatch):
        if ncells_view[j] > 1:
            for i in range(N):
                means_view[i, j] /= ncells_view[j]
                ps_view[i, j] = ps_view[i, j] - ncells_view[j] * means_view[i, j] * means_view[i, j]

    return ncells, means, partial_sum
