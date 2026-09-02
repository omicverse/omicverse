import copy

import numpy as np
import pandas as pd
from tqdm import tqdm


from .evaluate import compute_celltype_purity
from ..._anndata_compat import as_dtype as _as_dtype


def SEACells(
    ad,
    build_kernel_on: str,
    n_SEACells: int,
    use_gpu: bool = False,
    verbose: bool = True,
    n_waypoint_eigs: int = 10,
    n_neighbors: int = 15,
    convergence_epsilon: float = 1e-3,
    l2_penalty: float = 0,
    max_franke_wolfe_iters: int = 50,
    use_sparse: bool = False,
):
    """Core SEACells class.

    :param ad: (AnnData) annotated data matrix
    :param build_kernel_on: (str) key corresponding to matrix in ad.obsm which is used to compute kernel for metacells
                            Typically 'X_pca' for scRNA or 'X_svd' for scATAC
    :param n_SEACells: (int) number of SEACells to compute
    :param use_gpu: (bool) whether to use GPU for computation
    :param verbose: (bool) whether to suppress verbose program logging
    :param n_waypoint_eigs: (int) number of eigenvectors to use for waypoint initialization
    :param n_neighbors: (int) number of nearest neighbors to use for graph construction
    :param convergence_epsilon: (float) convergence threshold for Franke-Wolfe algorithm
    :param l2_penalty: (float) L2 penalty for Franke-Wolfe algorithm
    :param max_franke_wolfe_iters: (int) maximum number of iterations for Franke-Wolfe algorithm
    :param use_sparse: (bool) whether to use sparse matrix operations. Currently only supported for CPU implementation.

    See cpu.py or gpu.py for descriptions of model attributes and methods.
    """
    if use_sparse:
        assert (
            not use_gpu
        ), "Sparse matrix operations are only supported for CPU implementation."
        try:
            from . import cpu
        except ImportError:
            import cpu
        model = cpu.SEACellsCPU(
            ad,
            build_kernel_on,
            n_SEACells,
            verbose,
            n_waypoint_eigs,
            n_neighbors,
            convergence_epsilon,
            l2_penalty,
            max_franke_wolfe_iters,
        )

        return model

    if use_gpu:
        try:
            from . import gpu
        except ImportError:
            import gpu

        model = gpu.SEACellsGPU(
            ad,
            build_kernel_on,
            n_SEACells,
            verbose,
            n_waypoint_eigs,
            n_neighbors,
            convergence_epsilon,
            l2_penalty,
            max_franke_wolfe_iters,
        )

    else:
        try:
            from . import cpu_dense
        except ImportError:
            import cpu_dense
        model = cpu_dense.SEACellsCPUDense(
            ad,
            build_kernel_on,
            n_SEACells,
            verbose,
            n_waypoint_eigs,
            n_neighbors,
            convergence_epsilon,
            l2_penalty,
            max_franke_wolfe_iters,
        )

    return model


def sparsify_assignments(A, thresh, keep_above_percentile=95):
    """
    :param: A is a cell x SEACells assignment matrix
    """
    A = copy.deepcopy(A)
    A[A == 0] = np.nan
    mins = np.nanpercentile(A, keep_above_percentile, axis=1).reshape(-1, 1)
    A = np.nan_to_num(A)
    mins[mins>thresh] = thresh
    A[A < mins] = 0

    # Renormalize
    A = A / A.sum(1, keepdims=True)

    return A


def summarize_by_soft_SEACell(
    ad, A, celltype_label=None, summarize_layer="raw", minimum_weight: float = 0.05
):
    """Summary of soft SEACell assignment.

    Aggregates cells within each SEACell, summing over all raw data x assignment weight for all cells belonging to a
    SEACell. Data is un-normalized and pseudo-raw aggregated counts are stored in .layers['raw'].
    Attributes associated with variables (.var) are copied over, but relevant per SEACell attributes must be
    manually copied, since certain attributes may need to be summed, or averaged etc, depending on the attribute.
    The output of this function is an anndata object of shape n_metacells x original_data_dimension.

    @param ad: (sc.AnnData) containing raw counts for single-cell data
    @param A: (np.array) of shape n_SEACells x n_cells containing assignment weights of cells to SEACells
    @param celltype_label: (str) optionally provide the celltype label to compute modal celltype per SEACell
    @param summarize_layer: (str) key for ad.layers to find raw data. Use 'raw' to search for ad.raw.X
    @param minimum_weight: (float) minimum value below which assignment weights are zero-ed out. If all cell assignment
                            weights are smaller than minimum_weight, the 95th percentile weight is used.
    @return: aggregated anndata containing weighted expression for aggregated SEACells
    """
    import scanpy as sc
    from scipy.sparse import csr_matrix

    compute_seacell_celltypes = False
    if celltype_label is not None:
        if celltype_label not in ad.obs.columns:
            raise ValueError(f"Celltype label {celltype_label} not present in ad.obs")
        compute_seacell_celltypes = True

    if summarize_layer == "raw" and ad.raw is not None:
        data = ad.raw.X
    else:
        data = ad.layers[summarize_layer]

    A = sparsify_assignments(A.T, thresh=minimum_weight)

    seacell_expressions = []
    seacell_celltypes = []
    seacell_purities = []
    for ix in tqdm(range(A.shape[1])):
        cell_weights = A[:, ix]
        # Construct the SEACell expression using the
        seacell_exp = (
            data.multiply(cell_weights[:, np.newaxis]).toarray().sum(0)
            / cell_weights.sum()
        )
        seacell_expressions.append(seacell_exp)

        if compute_seacell_celltypes:
            # Compute the consensus celltype and the celltype purity
            cell_weights = pd.DataFrame(cell_weights)
            cell_weights.index = ad.obs_names
            purity = (
                cell_weights.join(ad.obs[celltype_label])
                .groupby(celltype_label)
                .sum()
                .sort_values(by=0, ascending=False)
            )
            purity = purity / purity.sum()
            celltype = purity.iloc[0]
            seacell_celltypes.append(celltype.name)
            seacell_purities.append(celltype.values[0])

    seacell_expressions = csr_matrix(np.array(seacell_expressions))
    seacell_ad = sc.AnnData(_as_dtype(seacell_expressions, seacell_expressions.dtype))
    seacell_ad.var_names = ad.var_names
    seacell_ad.obs["Pseudo-sizes"] = A.sum(0)
    if compute_seacell_celltypes:
        seacell_ad.obs["celltype"] = seacell_celltypes
        seacell_ad.obs["celltype_purity"] = seacell_purities
    seacell_ad.var_names = ad.var_names
    return seacell_ad


def summarize_by_SEACell(
    ad, SEACells_label="SEACell", celltype_label=None, summarize_layer="raw"
):
    """Summary of SEACell assignment.

    Aggregates cells within each SEACell, summing over all raw data for all cells belonging to a SEACell.
    Data is unnormalized and raw aggregated counts are stored .layers['raw'].
    Attributes associated with variables (.var) are copied over, but relevant per SEACell attributes must be
    manually copied, since certain attributes may need to be summed, or averaged etc, depending on the attribute.
    The output of this function is an anndata object of shape n_metacells x original_data_dimension.
    :return: anndata.AnnData containing aggregated counts.

    """
    import scanpy as sc
    from scipy.sparse import csr_matrix

    # Set of metacells
    metacells = ad.obs[SEACells_label].unique()

    # Summary matrix
    summ_matrix = pd.DataFrame(0.0, index=metacells, columns=ad.var_names)

    for m in tqdm(summ_matrix.index):
        cells = ad.obs_names[ad.obs[SEACells_label] == m]
        if summarize_layer == "X":
            summ_matrix.loc[m, :] = np.ravel(ad[cells, :].X.sum(axis=0))
        elif summarize_layer == "raw" and ad.raw is not None:
            summ_matrix.loc[m, :] = np.ravel(ad[cells, :].raw.X.sum(axis=0))
        else:
            summ_matrix.loc[m, :] = np.ravel(
                ad[cells, :].layers[summarize_layer].sum(axis=0)
            )

    # Ann data

    # Counts
    _summ = csr_matrix(summ_matrix)
    meta_ad = sc.AnnData(_as_dtype(_summ, _summ.dtype))
    meta_ad.obs_names, meta_ad.var_names = summ_matrix.index.astype(str), ad.var_names
    meta_ad.layers["raw"] = csr_matrix(summ_matrix)

    # Also compute cell type purity
    if celltype_label is not None:
        # TODO: Catch specific exception
        try:
            purity_df = compute_celltype_purity(ad, celltype_label)
            meta_ad.obs = meta_ad.obs.join(purity_df)
        except Exception as e:  # noqa: BLE001
            print(f"Cell type purity failed with Exception {e}")

    return meta_ad
