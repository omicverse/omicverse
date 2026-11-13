from anndata import AnnData
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Iterable, Union, Optional
from .._settings import add_reference
from .._registry import register_function


### Refer to: https://github.com/theislab/scanpy/blob/5533b644e796379fd146bf8e659fd49f92f718cd/scanpy/_compat.py
try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):
            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values,)
                return type('Literal_', (Literal,), dict(__args__=values))

        class Literal(metaclass=LiteralMeta):
            pass

### Refer to Scanpy       
def _select_top_n(scores, n_top):
    reference_indices = np.arange(scores.shape[0], dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]
    return global_indices


### Import from Scanpy
from scipy.sparse import issparse

### Import from Scanpy
def select_groups(adata, groups_order_subset='all', key='groups'):
    r"""Select groups and generate per-group cell masks.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with categorical grouping labels.
    groups_order_subset : str or sequence, default='all'
        Group subset to keep. Use ``'all'`` to include all categories.
    key : str, default='groups'
        Obs column used for grouping.

    Returns
    -------
    tuple
        ``(groups_order_subset, groups_masks)`` where ``groups_masks`` is a
        boolean matrix of shape ``(n_groups, n_cells)``.
    """
    groups_order = adata.obs[key].cat.categories
    if key + '_masks' in adata.uns:
        groups_masks = adata.uns[key + '_masks']
    else:
        groups_masks = np.zeros(
            (len(adata.obs[key].cat.categories), adata.obs[key].values.size), dtype=bool
        )
        for iname, name in enumerate(adata.obs[key].cat.categories):
            # if the name is not found, fallback to index retrieval
            if adata.obs[key].cat.categories[iname] in adata.obs[key].values:
                mask = adata.obs[key].cat.categories[iname] == adata.obs[key].values
            else:
                mask = str(iname) == adata.obs[key].values
            groups_masks[iname] = mask
    groups_ids = list(range(len(groups_order)))
    if groups_order_subset != 'all':
        groups_ids = []
        for name in groups_order_subset:
            groups_ids.append(
                np.where(adata.obs[key].cat.categories.values == name)[0][0]
            )
        if len(groups_ids) == 0:
            # fallback to index retrieval
            groups_ids = np.where(
                np.isin(
                    np.arange(len(adata.obs[key].cat.categories)).astype(str),
                    np.array(groups_order_subset),
                )
            )[0]
        if len(groups_ids) == 0:
            logg.debug(
                f'{np.array(groups_order_subset)} invalid! specify valid '
                f'groups_order (or indices) from {adata.obs[key].cat.categories}',
            )
            from sys import exit

            exit(0)
        groups_masks = groups_masks[groups_ids]
        groups_order_subset = adata.obs[key].cat.categories[groups_ids].values
    else:
        groups_order_subset = groups_order.values
    return groups_order_subset, groups_masks


### Import from Scanpy   
import numpy as np
from scipy import sparse
import numba




def _get_mean_var(X, *, axis=0):
    if sparse.issparse(X):
        mean, var = sparse_mean_variance_axis(X, axis=axis)
    else:
        mean = np.mean(X, axis=axis, dtype=np.float64)
        mean_sq = np.multiply(X, X).mean(axis=axis, dtype=np.float64)
        var = mean_sq - mean ** 2
    # enforce R convention (unbiased estimator) for variance
    var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var


def sparse_mean_variance_axis(mtx: sparse.spmatrix, axis: int):
    r"""Calculate mean and variance along specified axis for sparse matrix.
    
    This code and internal functions are based on sklearns `sparsefuncs.mean_variance_axis`.
    Modifications:
    * allow deciding on the output type, which can increase accuracy when calculating the mean and variance of 32bit floats.
    * This doesn't currently implement support for null values, but could.
    * Uses numba not cython
    
    Parameters
    ----------
    mtx : scipy.sparse.spmatrix
        Sparse matrix in CSR or CSC format.
    axis : int
        Axis over which to compute statistics (0 or 1).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Mean and variance vectors.
    """
    assert axis in (0, 1)
    if isinstance(mtx, sparse.csr_matrix):
        ax_minor = 1
        shape = mtx.shape
    elif isinstance(mtx, sparse.csc_matrix):
        ax_minor = 0
        shape = mtx.shape[::-1]
    else:
        raise ValueError("This function only works on sparse csr and csc matrices")
    if axis == ax_minor:
        return sparse_mean_var_major_axis(
            mtx.data, mtx.indices, mtx.indptr, *shape, np.float64
        )
    else:
        return sparse_mean_var_minor_axis(mtx.data, mtx.indices, *shape, np.float64)


@numba.njit(cache=True)
def sparse_mean_var_minor_axis(data, indices, major_len, minor_len, dtype):
    """
    Computes mean and variance for a sparse matrix for the minor axis.
    Given arrays for a csr matrix, returns the means and variances for each
    column back.
    """
    non_zero = indices.shape[0]

    means = np.zeros(minor_len, dtype=dtype)
    variances = np.zeros_like(means, dtype=dtype)

    counts = np.zeros(minor_len, dtype=np.int64)

    for i in range(non_zero):
        col_ind = indices[i]
        means[col_ind] += data[i]

    for i in range(minor_len):
        means[i] /= major_len

    for i in range(non_zero):
        col_ind = indices[i]
        diff = data[i] - means[col_ind]
        variances[col_ind] += diff * diff
        counts[col_ind] += 1

    for i in range(minor_len):
        variances[i] += (major_len - counts[i]) * means[i] ** 2
        variances[i] /= major_len

    return means, variances


@numba.njit(cache=True)
def sparse_mean_var_major_axis(data, indices, indptr, major_len, minor_len, dtype):
    """
    Computes mean and variance for a sparse array for the major axis.
    Given arrays for a csr matrix, returns the means and variances for each
    row back.
    """
    means = np.zeros(major_len, dtype=dtype)
    variances = np.zeros_like(means, dtype=dtype)

    for i in range(major_len):
        startptr = indptr[i]
        endptr = indptr[i + 1]
        counts = endptr - startptr

        for j in range(startptr, endptr):
            means[i] += data[j]
        means[i] /= minor_len

        for j in range(startptr, endptr):
            diff = data[j] - means[i]
            variances[i] += diff * diff

        variances[i] += (minor_len - counts) * means[i] ** 2
        variances[i] /= minor_len

    return means, variances  

### Import from Scanpy
class _RankGenes:
    def __init__(
        self,
        adata,
        groups,
        groupby,
        reference='rest',
        use_raw=True,
        layer=None,
        comp_pts=False,
    ):

        if 'log1p' in adata.uns_keys() and adata.uns['log1p']['base'] is not None:
            self.expm1_func = lambda x: np.expm1(x * np.log(adata.uns['log1p']['base']))
        else:
            self.expm1_func = np.expm1

        self.groups_order, self.groups_masks = select_groups(
            adata, groups, groupby
        )

        # Singlet groups cause division by zero errors
        invalid_groups_selected = set(self.groups_order) & set(
            adata.obs[groupby].value_counts().loc[lambda x: x < 2].index
        )

        if len(invalid_groups_selected) > 0:
            raise ValueError(
                "Could not calculate statistics for groups {} since they only "
                "contain one sample.".format(', '.join(invalid_groups_selected))
            )

        adata_comp = adata
        if layer is not None:
            if use_raw:
                raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
            X = adata_comp.layers[layer]
        else:
            if use_raw and adata.raw is not None:
                adata_comp = adata.raw
            X = adata_comp.X

        # for correct getnnz calculation
        if issparse(X):
            X.eliminate_zeros()

        self.X = X
        self.var_names = adata_comp.var_names

        self.ireference = None
        if reference != 'rest':
            self.ireference = np.where(self.groups_order == reference)[0][0]

        self.means = None
        self.vars = None

        self.means_rest = None
        self.vars_rest = None

        self.comp_pts = comp_pts
        self.pts = None
        self.pts_rest = None

        self.stats = None

        # for logreg only
        self.grouping_mask = adata.obs[groupby].isin(self.groups_order)
        self.grouping = adata.obs.loc[self.grouping_mask, groupby]

    def _basic_stats(self):
        n_genes = self.X.shape[1]
        n_groups = self.groups_masks.shape[0]

        self.means = np.zeros((n_groups, n_genes))
        self.vars = np.zeros((n_groups, n_genes))
        self.pts = np.zeros((n_groups, n_genes)) if self.comp_pts else None

        if self.ireference is None:
            self.means_rest = np.zeros((n_groups, n_genes))
            self.vars_rest = np.zeros((n_groups, n_genes))
            self.pts_rest = np.zeros((n_groups, n_genes)) if self.comp_pts else None
        else:
            mask_rest = self.groups_masks[self.ireference]
            X_rest = self.X[mask_rest]
            self.means[self.ireference], self.vars[self.ireference] = _get_mean_var(
                X_rest
            )
            # deleting the next line causes a memory leak for some reason
            del X_rest

        if issparse(self.X):
            get_nonzeros = lambda X: X.getnnz(axis=0)
        else:
            get_nonzeros = lambda X: np.count_nonzero(X, axis=0)

        for imask, mask in enumerate(self.groups_masks):
            X_mask = self.X[mask]

            if self.comp_pts:
                self.pts[imask] = get_nonzeros(X_mask) / X_mask.shape[0]

            if self.ireference is not None and imask == self.ireference:
                continue

            self.means[imask], self.vars[imask] = _get_mean_var(X_mask)

            if self.ireference is None:
                mask_rest = ~mask
                X_rest = self.X[mask_rest]
                self.means_rest[imask], self.vars_rest[imask] = _get_mean_var(X_rest)
                # this can be costly for sparse data
                if self.comp_pts:
                    self.pts_rest[imask] = get_nonzeros(X_rest) / X_rest.shape[0]
                # deleting the next line causes a memory leak for some reason
                del X_rest

        
@register_function(
    aliases=["COSG分析", "cosg", "marker_genes", "标记基因", "cluster_markers"],
    category="single",
    description="Identify cluster-specific marker genes using COSG. IMPORTANT: Results are stored in adata.uns['rank_genes_groups'], NOT adata.obs! COSG finds markers but does NOT automatically assign cell types.",
    prerequisites={
        'functions': ['leiden']
    },
    requires={
        'obs': []  # Dynamic: user-specified groupby column
    },
    produces={
        'uns': ['rank_genes_groups', 'cosg_logfoldchanges']
    },
    auto_fix='escalate',
    examples=[
        "# IMPORTANT: COSG stores results in adata.uns, NOT adata.obs!",
        "",
        "# Step 1: Run COSG marker gene identification",
        "ov.single.cosg(adata, groupby='leiden', n_genes_user=50)",
        "",
        "# Step 2: Access results from adata.uns (NOT adata.obs!)",
        "marker_names = adata.uns['rank_genes_groups']['names']  # DataFrame",
        "marker_scores = adata.uns['rank_genes_groups']['scores']",
        "",
        "# Step 3: Get top markers for specific cluster",
        "cluster_0_markers = adata.uns['rank_genes_groups']['names']['0'][:10].tolist()",
        "",
        "# Step 4: To create celltype column, MANUALLY map clusters based on markers",
        "cluster_to_celltype = {'0': 'T cells', '1': 'B cells', '2': 'Monocytes'}",
        "adata.obs['celltype'] = adata.obs['leiden'].map(cluster_to_celltype)",
        "",
        "# WRONG - DO NOT USE:",
        "# adata.obs['cosg_celltype']  # ERROR! This key does NOT exist!",
        "# COSG does NOT create adata.obs columns - it only finds marker genes!",
        "",
        "# With logfoldchange calculation",
        "ov.single.cosg(adata, groupby='leiden', calculate_logfoldchanges=True)",
        "logfc_df = adata.uns['cosg_logfoldchanges']"
    ],
    related=["pp.highly_variable_genes", "pl.marker_gene_overlap", "bulk.get_deg"]
)
def cosg(
    adata,
    groupby='CellTypes',
    groups: Union[Literal['all'], Iterable[str]] = 'all',

    mu=1,
    remove_lowly_expressed:bool=False,
    expressed_pct:Optional[float] = 0.1,

    n_genes_user:int =50,
    key_added: Optional[str] = None,
    calculate_logfoldchanges: bool = True,
    use_raw: bool = True,
    layer: Optional[str] = None,
    reference: str = 'rest',    

    copy:bool=False,
    backend: Optional[str] = None,
):
    r"""Identify cluster-specific marker genes with COSG.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix for marker detection.
    groupby : str, default='CellTypes'
        Obs column containing cluster/group labels.
    groups : {'all'} or Iterable[str], default='all'
        Group subset to analyze.
    mu : float, default=1
        Specificity penalty against expression in non-target groups.
    remove_lowly_expressed : bool, default=False
        Whether to remove genes with low detection rates in target groups.
    expressed_pct : float, default=0.1
        Minimum fraction of expressing cells when low-expression filtering is enabled.
    n_genes_user : int, default=50
        Number of top marker genes retained per group.
    key_added : str or None, default=None
        Key in ``adata.uns`` used to store COSG results.
    calculate_logfoldchanges : bool, default=True
        Whether to compute log-fold changes.
    use_raw : bool, default=True
        Whether to use ``adata.raw`` expression values when available.
    layer : str or None, default=None
        Layer key used as expression matrix.
    reference : str, default='rest'
        Reference strategy for differential comparison.
    copy : bool, default=False
        If ``True``, run on a copy and return it.

    Returns
    -------
    anndata.AnnData or None
        Returns modified copy when ``copy=True``; otherwise updates input
        object in place.

    """
    
    from .._backend import resolve_backend
    if resolve_backend(backend) == 'cpp' and groups == 'all' and reference == 'rest':
        # One multithreaded pass of per-group sums replaces the transpose,
        # normalisation and sparse product; scores are identical.
        from anndataoom_fast._omicverse import ov_cosg
        return ov_cosg(adata, groupby=groupby, groups=groups, mu=mu,
                       remove_lowly_expressed=remove_lowly_expressed,
                       expressed_pct=expressed_pct, n_genes_user=n_genes_user,
                       key_added=key_added, calculate_logfoldchanges=calculate_logfoldchanges,
                       use_raw=use_raw, layer=layer, reference=reference, copy=copy)
    adata = adata.copy() if copy else adata
        
    if layer is not None:
        if use_raw:
            raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
        cellxgene = adata.layers[layer]
    else:
        if use_raw and adata.raw is not None:
             cellxgene = adata.raw.X
        cellxgene = adata.X
    
    
    ### Refer to scanpy's framework
    # https://github.com/theislab/scanpy/blob/5533b644e796379fd146bf8e659fd49f92f718cd/scanpy/tools/_rank_genes_groups.py#L559
    if key_added is None:
        key_added = 'rank_genes_groups'
    adata.uns[key_added] = {}
    adata.uns[key_added]['params'] = dict(
    groupby=groupby,
    reference=reference,
    groups=groups,
    method='cosg',
    use_raw=use_raw,
    layer=layer,
    )
    
    ### Refer to: https://github.com/theislab/scanpy/blob/5533b644e796379fd146bf8e659fd49f92f718cd/scanpy/tools/_rank_genes_groups.py#L543
    if groups == 'all':
        ### group lable for each cell
        group_info=adata.obs[groupby]
    elif isinstance(groups, (str, int)):
        raise ValueError('Specify a sequence of groups')
    else:
        cells_selected=adata.obs[groupby].isin(groups)
        cells_selected=cells_selected.values
        if sparse.issparse(cellxgene):
            cellxgene=cellxgene[cells_selected]
        else:
            cellxgene=cellxgene[cells_selected,:]
            
            

        ### group lable for each cell
        group_info=adata.obs[groupby].copy()
        group_info=group_info[cells_selected]
        

    
    groups_order=np.unique(group_info)
    n_cluster=len(groups_order)
    
    n_cell=cellxgene.shape[0]
    cluster_mat=np.zeros(shape=(n_cluster,n_cell))

    ### To further restrict expression in other clusters, can think about a better way, such as taking the cluster similarities into consideration
    order_i=0
    for group_i in groups_order:    
        idx_i=group_info==group_i 
        cluster_mat[order_i,:][idx_i]=1
        order_i=order_i+1
    
    if sparse.issparse(cellxgene):
        ### Convert to sparse matrix
        from scipy.sparse import csr_matrix
        cluster_mat=csr_matrix(cluster_mat)

        from sklearn.metrics.pairwise import cosine_similarity

        ### the dimension is: Gene x lambda
        cosine_sim=cosine_similarity(X=cellxgene.T,Y=cluster_mat,dense_output=False) 

        pos_nonzero=cosine_sim.nonzero()
        genexlambda=cosine_sim.multiply(cosine_sim)

        e_power2_sum=genexlambda.sum(axis=1)


        if mu==1:
            genexlambda[pos_nonzero]=genexlambda[pos_nonzero]/(np.repeat(e_power2_sum,genexlambda.shape[1],axis=1)[pos_nonzero])
                                                    
        else:
            genexlambda[pos_nonzero]=genexlambda[pos_nonzero]/((1-mu)*genexlambda[pos_nonzero]+
                                                     mu*(
                                                        np.repeat(e_power2_sum,genexlambda.shape[1],axis=1)[pos_nonzero])
                                                    )
        genexlambda[pos_nonzero]=np.multiply(genexlambda[pos_nonzero],cosine_sim[pos_nonzero])
    
    ### If the cellxgene is not a sparse matrix
    else:
         ## Not using sparse matrix
        from sklearn.metrics.pairwise import cosine_similarity    
        cosine_sim=cosine_similarity(X=cellxgene.T,Y=cluster_mat,dense_output=True) 

        pos_nonzero=cosine_sim!=0
        e_power2=np.multiply(cosine_sim,cosine_sim)
        e_power2_sum=np.sum(e_power2,axis=1)
        e_power2[pos_nonzero]=np.true_divide(e_power2[pos_nonzero],(1-mu)*e_power2[pos_nonzero]+mu*(np.dot(e_power2_sum.reshape(e_power2_sum.shape[0],1),np.repeat(1,e_power2.shape[1]).reshape(1,e_power2.shape[1]))[pos_nonzero]))
        e_power2[pos_nonzero]=np.multiply(e_power2[pos_nonzero],cosine_sim[pos_nonzero])
        genexlambda=e_power2

    ### Refer to scanpy
    rank_stats=None
    
    ### Whether to calculate logfoldchanges, because this is required in scanpy 1.8
    if calculate_logfoldchanges:
        ### Calculate basic stats
        ### Refer to Scanpy
        # for clarity, rename variable
        if groups == 'all':
            groups_order2 = 'all'
        elif isinstance(groups, (str, int)):
            raise ValueError('Specify a sequence of groups')
        else:
            groups_order2 = list(groups)
            if isinstance(groups_order2[0], int):
                groups_order2 = [str(n) for n in groups_order2]
            if reference != 'rest' and reference not in set(groups_order2):
                groups_order2 += [reference]
        if reference != 'rest' and reference not in adata.obs[groupby].cat.categories:
            cats = adata.obs[groupby].cat.categories.tolist()
            raise ValueError(
                f'reference = {reference} needs to be one of groupby = {cats}.'
            )
        pts=False
        anndata_obj = _RankGenes(adata, groups_order2, groupby, reference, use_raw, layer, pts)
        anndata_obj._basic_stats()
    
    
    ### Refer to Scanpy
    # for correct getnnz calculation
    ### get non-zeros for columns
    if sparse.issparse(cellxgene):
        cellxgene.eliminate_zeros()
    if sparse.issparse(cellxgene):
        get_nonzeros = lambda X: X.getnnz(axis=0)
    else:
        get_nonzeros = lambda X: np.count_nonzero(X, axis=0)
    
    order_i=0
    for group_i in groups_order:    
        idx_i=group_info==group_i 

        ### Convert to numpy array
        idx_i=idx_i.values

        ## Compare the most ideal case to the worst case
        if sparse.issparse(cellxgene):
            scores=genexlambda[:,order_i].toarray()[:,0]
        else:
            scores=genexlambda[:,order_i]

        
        ### Mask these genes expressed in less than 3 cells in the cluster of interest
        if remove_lowly_expressed:
            n_cells_expressed=get_nonzeros(cellxgene[idx_i])
            n_cells_i=np.sum(idx_i)
            scores[n_cells_expressed<n_cells_i*expressed_pct]= -1

        if n_genes_user > len(scores):
            print(f"The length of scores is {len(scores)}, n_genes_user {n_genes_user} shouldn't be greater than {len(scores)}.")
            print(f"So n_genes_user was auto set to {len(scores)}.")
            n_genes_user = len(scores)
        global_indices = _select_top_n(scores, n_genes_user)

        if rank_stats is None:
            idx = pd.MultiIndex.from_tuples([(group_i,'names')])
            rank_stats = pd.DataFrame(columns=idx)
        rank_stats[group_i, 'names'] = adata.var_names[global_indices]
        rank_stats[group_i, 'scores'] = scores[global_indices]
        
        if calculate_logfoldchanges:
            group_index=np.where(anndata_obj.groups_order==group_i)[0][0]
            if anndata_obj.means is not None:
                mean_group = anndata_obj.means[group_index]
                if anndata_obj.ireference is None:
                    mean_rest = anndata_obj.means_rest[group_index]
                else:
                    mean_rest = anndata_obj.means[anndata_obj.ireference]
                foldchanges = (anndata_obj.expm1_func(mean_group) + 1e-9) / (
                    anndata_obj.expm1_func(mean_rest) + 1e-9
                )  # add small value to remove 0's
                rank_stats[group_i, 'logfoldchanges'] = np.log2(
                    foldchanges[global_indices]
                )
    
        order_i=order_i+1
            
    ## Refer to scanpy
    if calculate_logfoldchanges:
        dtypes = {
        'names': 'O',
        'scores': 'float32',
        'logfoldchanges': 'float32',
        }
    else:
        dtypes = {
        'names': 'O',
        'scores': 'float32',
        }
    ###
    rank_stats.columns = rank_stats.columns.swaplevel()
    for col in rank_stats.columns.levels[0]:
        adata.uns[key_added][col]=rank_stats[col].to_records(
    index=False, column_dtypes=dtypes[col]
    )
    # Convert structured arrays to DataFrames for easier downstream use
    for cluster_key, cluster_val in list(adata.uns[key_added].items()):
        try:
            if isinstance(cluster_val, np.ndarray) and getattr(cluster_val, "dtype", None) is not None and cluster_val.dtype.names:
                adata.uns[key_added][cluster_key] = pd.DataFrame(cluster_val)
        except Exception:
            continue
    # Safeguard pvals storage: some pipelines may not populate rank_genes_groups; fall back to zeros
    try:
        adata.uns[key_added]['pvals']=adata.uns['rank_genes_groups']['pvals']
        adata.uns[key_added]['pvals_adj']=adata.uns['rank_genes_groups']['pvals_adj']
    except Exception:
        # Build zero arrays matching the shape of scores; keep a local numpy handle
        import numpy as _np
        p_shape = adata.uns[key_added]['scores'].shape if 'scores' in adata.uns[key_added] else None
        if p_shape is not None:
            adata.uns[key_added]['pvals'] = _np.zeros(p_shape, dtype='float32')
            adata.uns[key_added]['pvals_adj'] = _np.zeros(p_shape, dtype='float32')
        else:
            adata.uns[key_added]['pvals'] = None
            adata.uns[key_added]['pvals_adj'] = None

    # Ensure a consistent key for downstream consumers expecting 'cosg' or 'cosg_markers'
    if key_added != 'cosg':
        adata.uns['cosg'] = adata.uns.get(key_added, adata.uns.get('cosg', {}))
    adata.uns.setdefault('cosg_markers', adata.uns.get(key_added, {}))
        
    
        
    print('**finished identifying marker genes by COSG**')
    add_reference(adata,'COSG','marker gene identification with COSG')
        
    ### Return the result
    return adata if copy else None
