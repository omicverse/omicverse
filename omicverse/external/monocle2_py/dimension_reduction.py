"""
Dimension reduction for monocle2_py.

Implements reduceDimension() with DDRTree, ICA, and tSNE methods.
"""

import numpy as np
from patsy import PatsyError, dmatrix
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import igraph as ig

from .core import _init_monocle_uns, estimate_size_factors
from .ddrtree import DDRTree

# Show the 'fast is default' hint once per Python session.
_FAST_HINT_SHOWN = False


def _residualize_model_effects(FM, obs, residual_model_formula):
    """Regress phenotype effects from a genes-by-cells expression matrix.

    This mirrors the ``residualModelFormulaStr`` stage in R Monocle 2's
    ``reduceDimension``: build a model matrix from cell metadata, fit every
    gene jointly by least squares, retain the intercept, and subtract the
    fitted contribution of all other terms.

    Parameters
    ----------
    FM : np.ndarray
        Normalized expression matrix with shape ``(genes, cells)``.
    obs : pandas.DataFrame
        Cell metadata. Its row order must match the columns of ``FM``.
    residual_model_formula : str
        Patsy/R-style formula such as ``"~ batch"`` or
        ``"~ batch + percent_mito"``. An intercept is required; use the
        conventional ``~ batch`` form rather than ``~ 0 + batch``.

    Returns
    -------
    residualized : np.ndarray
        Expression with non-intercept model effects removed.
    metadata : dict
        JSON-serializable audit metadata for ``adata.uns['monocle']``.
    """
    if not isinstance(residual_model_formula, str) or not residual_model_formula.strip():
        raise ValueError("residualModelFormulaStr must be a non-empty formula string")

    try:
        design_df = dmatrix(
            residual_model_formula,
            data=obs,
            return_type="dataframe",
            NA_action="raise",
        )
    except (PatsyError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid residualModelFormulaStr {residual_model_formula!r}: {exc}"
        ) from exc

    if len(design_df) != FM.shape[1]:
        raise ValueError(
            "residualModelFormulaStr changed the number of cells; missing metadata "
            "must be resolved before trajectory inference"
        )

    columns = list(design_df.columns)
    if "Intercept" not in columns:
        raise ValueError(
            "residualModelFormulaStr must include an intercept; use '~ batch' "
            "instead of '~ 0 + batch' or '~ batch - 1'"
        )

    design = np.asarray(design_df, dtype=np.float64)
    if not np.all(np.isfinite(design)):
        raise ValueError("residualModelFormulaStr produced non-finite design values")

    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        raise ValueError(
            "residualModelFormulaStr produced a rank-deficient design matrix "
            f"(rank {rank} < {design.shape[1]} columns). Remove nested or "
            "redundant covariates before trajectory inference."
        )

    effect_mask = np.asarray([name != "Intercept" for name in columns])
    if not effect_mask.any():
        adjusted = np.array(FM, dtype=np.float64, copy=True)
    else:
        # Solve all genes at once: design is cells x terms and FM.T is
        # cells x genes. Retain the intercept and subtract only covariate
        # contributions, matching R Monocle 2's beta[, -1] behavior for
        # ordinary formulas with an intercept.
        coefficients = np.linalg.lstsq(design, FM.T, rcond=None)[0]
        fitted_effects = design[:, effect_mask] @ coefficients[effect_mask, :]
        adjusted = (FM.T - fitted_effects).T

    metadata = {
        "formula": residual_model_formula,
        "design_columns": columns,
        "design_rank": rank,
        "n_cells": int(design.shape[0]),
        "n_terms": int(design.shape[1]),
    }
    return adjusted, metadata


def _cal_ncenter(ncells, ncells_limit=100, auto_scale=True):
    """Calculate number of centers for DDRTree.

    R Monocle2's formula saturates around ~130 for large datasets, which is
    often too few to capture small branches (e.g., rare cell types). We
    extend it with auto-scaling for datasets > 1000 cells.

    Formula:
      - n <= 1000: R's formula (matches Monocle2 exactly)
          2 * ncells_limit * log(n) / (log(n) + log(ncells_limit))
      - n > 1000: max(R_formula, n/15), capped at min(500, n/5)

    Parameters
    ----------
    ncells : int
        Number of cells.
    ncells_limit : int
        Base parameter (default 100, matches R).
    auto_scale : bool
        If True (default), use extended scaling for large datasets.
        If False, use R's exact formula (may be too few centers).

    Returns
    -------
    int
        Recommended number of DDRTree centers.
    """
    r_formula = int(round(2 * ncells_limit * np.log(ncells) /
                           (np.log(ncells) + np.log(ncells_limit))))
    if not auto_scale or ncells <= 1000:
        return r_formula
    # For larger datasets: ensure at least n/12 centers (≈8.3% coverage),
    # but cap at 500 to keep DDRTree fast. Empirically this matches
    # R Monocle2's branch discovery behavior on typical scRNA-seq trajectories.
    scaled = max(r_formula, ncells // 12)
    return min(scaled, 500)


def _normalize_expr_data(adata, norm_method='log', pseudo_expr=1):
    """
    Normalize expression data for dimension reduction.

    Parameters
    ----------
    adata : AnnData
    norm_method : str, one of 'log', 'none'
    pseudo_expr : float

    Returns
    -------
    FM : np.ndarray, (genes x cells) matrix of normalized expression
    gene_mask : bool array indicating which genes were selected
    """
    X = adata.X
    # Subset BEFORE densifying — a full float64 copy of a
    # ``cells × genes`` matrix with ~28k genes is ~800 MB and was
    # previously the single biggest cost of ``reduce_dimension``.
    if 'use_for_ordering' in adata.var.columns:
        use_mask = adata.var['use_for_ordering'].values.astype(bool)
        if use_mask.sum() > 0:
            X_sel = X[:, use_mask]
            gene_names = adata.var_names[use_mask]
        else:
            X_sel = X
            gene_names = adata.var_names
            use_mask = np.ones(adata.n_vars, dtype=bool)
    else:
        X_sel = X
        gene_names = adata.var_names
        use_mask = np.ones(adata.n_vars, dtype=bool)

    if sparse.issparse(X_sel):
        X_dense = X_sel.toarray().astype(np.float64, copy=False)
    else:
        X_dense = np.ascontiguousarray(X_sel, dtype=np.float64)

    # Normalize: cells x genes -> transpose to genes x cells for processing
    FM = X_dense.T  # genes x cells

    if norm_method == 'log':
        if 'Size_Factor' in adata.obs.columns:
            sf = adata.obs['Size_Factor'].values
            FM = FM / sf[None, :]
        FM = FM + pseudo_expr
        FM = np.log2(FM)
    elif norm_method == 'none':
        if 'Size_Factor' in adata.obs.columns:
            sf = adata.obs['Size_Factor'].values
            FM = FM / sf[None, :]
        FM = FM + pseudo_expr

    return FM, use_mask, gene_names


def reduce_dimension(adata, max_components=2, reduction_method='DDRTree',
                     norm_method='log', pseudo_expr=1, auto_param_selection=True,
                     verbose=False, scaling=True, random_state=2016,
                     residualModelFormulaStr=None, **kwargs):
    """
    Reduce dimensionality of the data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells x genes).
    max_components : int
        Number of dimensions to reduce to.
    reduction_method : str
        One of 'DDRTree', 'tSNE', 'ICA'.
    norm_method : str
        Normalization method: 'log', 'none'.
    pseudo_expr : float
        Pseudocount for log normalization.
    auto_param_selection : bool
        Automatically select DDRTree parameters.
    verbose : bool
    scaling : bool
        Whether to scale (z-score) genes.
    random_state : int or None
        Seed used for stochastic initialisation (tSNE, K-means, ICA).
        Does NOT mutate numpy's global RNG — pass it through to
        scikit-learn estimators directly.
    residualModelFormulaStr : str or None
        Patsy/R-style model formula specifying cell-level effects to remove
        after normalization and before scaling and dimension reduction, for
        example ``"~ batch"`` or ``"~ batch + percent_mito"``. The formula
        must include an intercept and reference columns in ``adata.obs``.
    **kwargs : dict
        Additional arguments passed to DDRTree or other methods.

    Returns
    -------
    adata with updated .obsm, .uns['monocle'] fields
    """
    _init_monocle_uns(adata)
    # This field describes the current reduction only. Avoid retaining stale
    # correction metadata if the same object is reduced again without a model.
    adata.uns['monocle'].pop('residual_model', None)

    # Normalize expression
    FM, use_mask, gene_names = _normalize_expr_data(adata, norm_method, pseudo_expr)

    # Filter genes with zero variance
    xm = FM.mean(axis=1)
    xsd = np.sqrt(np.mean((FM - xm[:, None]) ** 2, axis=1))
    nonzero_var = xsd > 0
    FM = FM[nonzero_var, :]

    if residualModelFormulaStr is not None:
        FM, residual_model = _residualize_model_effects(
            FM, adata.obs, residualModelFormulaStr,
        )
        # A gene explained entirely by the residual model has no remaining
        # trajectory information. Drop it before z-scoring and DDRTree.
        residual_sd = FM.std(axis=1, ddof=1)
        # Exact model effects normally leave floating-point residuals rather
        # than bitwise zeros. Treat values at least-squares precision as zero
        # so the following z-score cannot amplify numerical noise.
        residual_scale = np.maximum(1.0, np.max(np.abs(FM), axis=1))
        residual_tol = np.sqrt(np.finfo(np.float64).eps) * residual_scale
        residual_nonzero = (
            np.isfinite(residual_sd) & (residual_sd > residual_tol)
        )
        residual_model["n_genes_before"] = int(FM.shape[0])
        residual_model["n_genes_removed_zero_variance"] = int(
            (~residual_nonzero).sum()
        )
        FM = FM[residual_nonzero, :]
        if FM.shape[0] == 0:
            raise ValueError(
                "All ordering genes have zero variance after applying "
                "residualModelFormulaStr"
            )
        adata.uns['monocle']['residual_model'] = residual_model
        if verbose:
            print(
                "Removed model effects before dimension reduction: "
                f"{residualModelFormulaStr}"
            )

    # Scale genes (z-score across cells)
    if scaling:
        row_means = FM.mean(axis=1, keepdims=True)
        row_stds = FM.std(axis=1, keepdims=True, ddof=1)  # R's scale() uses ddof=1
        row_stds[row_stds == 0] = 1.0
        FM = (FM - row_means) / row_stds
        # Remove rows with NaN
        valid_rows = np.all(np.isfinite(FM), axis=1)
        FM = FM[valid_rows, :]

    if FM.shape[0] == 0:
        raise ValueError("All rows have standard deviation zero after filtering")

    # Remove non-finite rows
    valid_rows = np.all(np.isfinite(FM), axis=1)
    FM = FM[valid_rows, :]

    N_cells = FM.shape[1]

    if reduction_method == 'DDRTree':
        if verbose:
            print("Learning principal graph with DDRTree")

        ddr_kwargs = {'random_state': random_state}
        for key in ['initial_method', 'maxIter', 'sigma', 'lambda_param',
                    'param_gamma', 'tol', 'pca_method', 'method']:
            if key in kwargs:
                ddr_kwargs[key] = kwargs[key]

        # Notify the user — once per Python session — that the default
        # ``method`` is the fast (approximate) path.  Users who need
        # bitwise agreement with R Monocle 2 can opt into exact mode.
        global _FAST_HINT_SHOWN
        _ddr_method = ddr_kwargs.get('method', 'fast')
        if _ddr_method == 'fast' and not _FAST_HINT_SHOWN:
            print("[monocle2_py] Using fast DDRTree (≈3× speed-up, pseudotime "
                  "correlation with R ≥ 0.99). Pass method='exact' for bitwise "
                  "R Monocle 2 parity.")
            _FAST_HINT_SHOWN = True

        if auto_param_selection and N_cells >= 100:
            if 'ncenter' in kwargs:
                ncenter = kwargs['ncenter']
                if verbose:
                    print(f"  Using user-specified ncenter={ncenter} "
                          f"(for {N_cells} cells, ratio={ncenter/N_cells:.1%})")
            else:
                ncenter = _cal_ncenter(N_cells)
                if verbose:
                    r_ncenter = _cal_ncenter(N_cells, auto_scale=False)
                    if ncenter != r_ncenter:
                        print(f"  Auto-scaled ncenter={ncenter} for {N_cells} cells "
                              f"(R Monocle2 formula would give {r_ncenter}; "
                              f"larger ncenter helps capture fine branches)")
                    else:
                        print(f"  ncenter={ncenter} for {N_cells} cells")
            ddr_kwargs['ncenter'] = ncenter

        ddrtree_res = DDRTree(FM, dimensions=max_components, verbose=verbose, **ddr_kwargs)

        W = ddrtree_res['W']
        Z = ddrtree_res['Z']
        Y = ddrtree_res['Y']

        # Store in adata
        adata.uns['monocle']['dim_reduce_type'] = 'DDRTree'
        adata.uns['monocle']['W'] = W              # (D, dim) projection matrix
        adata.uns['monocle']['reducedDimS'] = Z     # (dim, N) cell coords in reduced space
        adata.uns['monocle']['reducedDimK'] = Y     # (dim, K) center coords
        adata.uns['monocle']['objective_vals'] = ddrtree_res['objective_vals']

        # Also store in obsm for scanpy compatibility
        adata.obsm['X_DDRTree'] = Z.T  # cells x dim

        # Build MST on Y centers
        dp = squareform(pdist(Y.T))
        adata.uns['monocle']['cellPairwiseDistances'] = dp

        # Build MST directly from scipy (O(K^2 log K)) — avoids
        # materializing a full K*(K-1)/2 edge list. K is usually <=500.
        K = Y.shape[1]
        mst_sp = minimum_spanning_tree(csr_matrix(dp)).tocoo()
        mst = ig.Graph(n=K, directed=False)
        mst.vs['name'] = [f'Y_{i}' for i in range(K)]
        mst.add_edges(list(zip(mst_sp.row.tolist(), mst_sp.col.tolist())))
        mst.es['weight'] = mst_sp.data.tolist()
        adata.uns['monocle']['mst'] = mst
        adata.uns['monocle']['mst_adj'] = np.array(mst.get_adjacency(attribute='weight').data)

        # Find nearest point on MST for each cell
        _find_nearest_point_on_mst(adata)

    elif reduction_method == 'tSNE':
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        num_dim = kwargs.get('num_dim', 50)
        n_components_pca = min(num_dim, min(FM.shape) - 1)

        pca = PCA(n_components=n_components_pca)
        pca_res = pca.fit_transform(FM.T)

        tsne = TSNE(n_components=max_components,
                    perplexity=kwargs.get('perplexity', 30),
                    random_state=random_state)
        tsne_res = tsne.fit_transform(pca_res)

        adata.obsm['X_tSNE'] = tsne_res
        adata.uns['monocle']['dim_reduce_type'] = 'tSNE'
        adata.uns['monocle']['reducedDimA'] = tsne_res.T

    elif reduction_method == 'ICA':
        from sklearn.decomposition import FastICA

        ica = FastICA(n_components=max_components, random_state=random_state)
        S = ica.fit_transform(FM.T)  # cells x components
        W_ica = ica.mixing_  # genes x components

        adata.obsm['X_ICA'] = S
        adata.uns['monocle']['dim_reduce_type'] = 'ICA'
        adata.uns['monocle']['reducedDimS'] = S.T  # (dim, N)
        adata.uns['monocle']['reducedDimW'] = W_ica

        # Build MST on ICA space
        dp = squareform(pdist(S))
        adata.uns['monocle']['cellPairwiseDistances'] = dp

        N = S.shape[0]
        mst_sp = minimum_spanning_tree(csr_matrix(dp)).tocoo()
        mst = ig.Graph(n=N, directed=False)
        mst.vs['name'] = list(adata.obs_names)
        mst.add_edges(list(zip(mst_sp.row.tolist(), mst_sp.col.tolist())))
        mst.es['weight'] = mst_sp.data.tolist()
        adata.uns['monocle']['mst'] = mst
    else:
        raise ValueError(f"Unrecognized reduction method: {reduction_method}")

    return adata


def _find_nearest_point_on_mst(adata):
    """Find nearest center (Y point) on MST for each cell."""
    Z = adata.uns['monocle']['reducedDimS']  # (dim, N)
    Y = adata.uns['monocle']['reducedDimK']  # (dim, K)

    # Distance from each cell to each center
    distances = np.sqrt(((Z[:, :, None] - Y[:, None, :]) ** 2).sum(axis=0))  # (N, K)
    closest_vertex = np.argmin(distances, axis=1)  # (N,)

    adata.uns['monocle']['pr_graph_cell_proj_closest_vertex'] = closest_vertex
    return adata
