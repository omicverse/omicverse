from functools import partial
from warnings import warn
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.linalg
from scipy import stats
from scipy.sparse import coo_matrix, issparse
import scipy.sparse.linalg
from sklearn.utils.extmath import randomized_svd

from omicverse.external.scTenifold.core._utils import cal_fdr, timer
from omicverse.external.scTenifold.core._types import Backend, ExpressionData, LayerName


__all__ = ["make_networks", "cal_pcNet", "cal_pc_coefs", "manifold_alignment", "d_regulation", "strict_direction"]

NETWORK_BACKENDS = {"serial", "joblib-loky", "joblib-threading", "ray"}


def anndata_to_dataframe(data: ExpressionData, layer: LayerName = None) -> pd.DataFrame:
    """Convert a pandas or AnnData-like object to genes x cells DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    if not all(hasattr(data, attr) for attr in ("X", "var_names", "obs_names")):
        raise TypeError("data must be a pandas DataFrame or an AnnData-like object")
    matrix = data.X if layer is None else data.layers[layer]
    if issparse(matrix):
        matrix = matrix.toarray()
    return pd.DataFrame(np.asarray(matrix).T,
                        index=pd.Index(data.var_names),
                        columns=pd.Index(data.obs_names))


def _resolve_backend(backend: Backend, n_jobs: int, n_cpus: Optional[int]) -> tuple:
    if n_cpus is not None:
        warn("n_cpus is deprecated and will be removed in a future 0.2.x release; use n_jobs instead.",
             DeprecationWarning,
             stacklevel=3)
        if n_jobs == 1:
            n_jobs = n_cpus
    if backend not in NETWORK_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(NETWORK_BACKENDS)}")
    return backend, n_jobs


def cal_pc_coefs(k: int,
                 X: np.ndarray,
                 n_comp: int,
                 method: str = "sklearn",
                 random_state: int = 42) -> np.ndarray:
    """Regress gene ``k`` on the remaining genes via low-rank SVD.

    Parameters
    ----------
    k
        Index of the response gene.
    X
        Cells-by-genes design matrix (standardized).
    n_comp
        Number of SVD components.
    method
        SVD backend: ``"sklearn"`` (randomized) or ``"scipy"``.
    random_state
        Seed used by the sklearn randomized SVD.

    Returns
    -------
    Column vector of regression coefficients (``(genes - 1, 1)``).
    """
    y = X[:, k]
    Xi = np.delete(X, k, 1)  # cells x (genes - 1)

    if method == "sklearn":
        U, Sigma, VT = randomized_svd(Xi,
                                      n_components=n_comp,
                                      flip_sign=True,  # to yield deterministic outputs
                                      n_iter=20,
                                      random_state=random_state)
    # elif method == "dask":
        # U, Sigma, VT = da.linalg.svd_compressed(da.from_array(Xi), k=n_comp)
        # VT = VT.compute()
    elif method == "scipy":
        U, Sigma, VT = scipy.linalg.svd(Xi, False, lapack_driver="gesvd")
    else:
        raise ValueError("Invalid method")
    coef = VT[:n_comp, :].T  # (genes - 1) x n_comp
    score = Xi @ coef  # cells x n_comp
    score = score / np.expand_dims((score ** 2).sum(axis=0), 0)
    betas = coef @ (score.T @ y)  # (genes - 1),
    return np.expand_dims(betas, 1)


def _check_pcNet_inp(data, selected_samples):
    Z = data.iloc[:, selected_samples]
    sel_genes = (Z.sum(axis=1) > 0)
    assert not any(sel_genes.index.duplicated()), "some genes are duplicated"
    Z = Z.loc[sel_genes, :]
    assert all(Z.sum(axis=1) > 0), "All genes must be expressed in at least one cell"
    return Z.values


def pc_net_calc(data: pd.DataFrame,  # genes x cells
                selected_samples: Union[List[int], np.ndarray],
                n_comp: int = 3,
                scale_scores: bool = True,
                symmetric: bool = False,
                q: float = 0.,
                random_state: int = 42) -> np.ndarray:
    """Compute a single principal-component (PC) network.

    Parameters
    ----------
    data
        Genes-by-cells expression DataFrame.
    selected_samples
        Cell column indices used to build the network.
    n_comp
        Number of principal components per gene regression (>2).
    scale_scores
        If True, divide by the maximum absolute weight.
    symmetric
        If True, return ``(A + A.T) / 2``.
    q
        Quantile cutoff in ``[0, 1]`` below which weights are zeroed.
    random_state
        Seed for the randomized SVD.

    Returns
    -------
    Dense genes-by-genes adjacency matrix.
    """
    assert 2 < n_comp <= data.shape[0]
    assert 0 <= q <= 1
    X = _check_pcNet_inp(data, selected_samples)
    Xt = X.T  # cells x genes
    Xt = (Xt - Xt.mean(axis=0)) / Xt.std(axis=0)
    A = 1 - np.eye(Xt.shape[1])  # genes x genes
    p_ = partial(cal_pc_coefs, X=Xt, n_comp=n_comp, random_state=random_state)
    bs = [p_(i) for i in range(Xt.shape[1])]
    B = np.concatenate(bs, axis=1).T  # beta matrix ((genes - 1), genes)

    A[A > 0] = np.ravel(B)
    if symmetric:
        A = (A + A.T) / 2
    abs_A = abs(A)
    if scale_scores:
        A = A / np.max(abs_A)
    A[abs_A < np.quantile(abs_A, q)] = 0
    np.fill_diagonal(A, 0)
    return A


@timer
def make_networks(data: ExpressionData,
                  n_nets: int = 10,
                  n_samp_cells: Optional[int] = 500,
                  n_comp: int = 3,
                  scale_scores: bool = True,
                  symmetric: bool = False,
                  q: float = 0.95,
                  random_state: int = 42,
                  backend: Backend = "serial",
                  n_jobs: int = 1,
                  n_cpus: Optional[int] = None,
                  replace: bool = True,
                  layer: LayerName = None,
                  **kwargs: object
                  ) -> List[coo_matrix]:
    """
    Make PCNets from a data frame by subsampling the cells

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe
    n_nets: int, default = 10
        Number of subsampling times
    n_samp_cells: int, None, default = 500
        Number of sampled cells, if None than select all cells
    n_comp: int, default = 3
        Number of PCNets composition
    scale_scores: bool, default = True
        To scale the final PCNets scores or not
    symmetric: bool, default = False
        To make the final PCNets symmetric or not
    q: float, default = 0.95
        The quantile value used to determine PCNet's threshold
    random_state: int, default = 42
        Random seed of constructing PCNets
    backend: str, default = "serial"
        Parallel backend: "serial", "joblib-loky", "joblib-threading", or "ray"
    n_jobs: int, default = 1
        Number of workers for parallel backends. -1 uses the backend default.
    n_cpus: int, optional
        Deprecated alias for n_jobs.

    kwargs
        Keyword arguments

    Returns
    -------
    networks: List[coo_matrix]
        A list contains PCNets (in coo sparse matrix format)
    """
    data = anndata_to_dataframe(data, layer=layer)
    backend, n_jobs = _resolve_backend(backend=backend, n_jobs=n_jobs, n_cpus=n_cpus)
    gene_names = data.index.to_numpy()
    n_genes, n_cells = data.shape
    assert not np.array_equal(gene_names, np.array([i for i in range(n_genes)])), 'Gene names are required'
    rng = np.random.default_rng(random_state)
    networks = []
    sel_samples = []
    for net in range(n_nets):
        sample = rng.choice(n_cells, n_samp_cells, replace=replace) if n_samp_cells is not None else np.arange(n_cells)
        sel_samples.append(sample)

    if backend == "serial":
        results = []
        for sample in sel_samples:
            results.append(pc_net_calc(data,
                                         selected_samples=sample,
                                         n_comp=n_comp,
                                         scale_scores=scale_scores,
                                         symmetric=symmetric,
                                         q=q,
                                         random_state=random_state))
    elif backend in {"joblib-loky", "joblib-threading"}:
        from joblib import Parallel, delayed
        prefer = "processes" if backend == "joblib-loky" else "threads"
        results = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(pc_net_calc)(data,
                                   selected_samples=sample,
                                   n_comp=n_comp,
                                   scale_scores=scale_scores,
                                   symmetric=symmetric,
                                   q=q,
                                   random_state=random_state)
            for sample in sel_samples
        )
    else:
        try:
            from importlib import import_module
            ray = import_module("ray")
        except ImportError as exc:
            raise ImportError("Install scTenifoldpy[parallel-ray] to use backend='ray'.") from exc

        @ray.remote
        def _pc_net_ray(data, selected_samples):
            return pc_net_calc(data=data,
                               selected_samples=selected_samples,
                               n_comp=n_comp,
                               scale_scores=scale_scores,
                               symmetric=symmetric,
                               q=q,
                               random_state=random_state)

        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=None if n_jobs == -1 else n_jobs)
        z_data = ray.put(data)
        tasks = [_pc_net_ray.remote(z_data, sample) for sample in sel_samples]
        results = ray.get(tasks)
        del z_data
        if ray.is_initialized():
            ray.shutdown()
    for i, pc_net in enumerate(results):
        Z = data.iloc[:, sel_samples[i]]
        sel_genes = (Z.sum(axis=1) > 0)
        Z = Z.loc[sel_genes, :]
        temp_df = pd.DataFrame(0.0, columns=gene_names, index=gene_names)
        temp_df.loc[sel_genes, sel_genes] = pd.DataFrame(pd.DataFrame(pc_net,
                                                                      index=Z.index,
                                                                      columns=Z.index),
                                                         index=sel_genes.index,
                                                         columns=sel_genes.index)
        networks.append(coo_matrix(temp_df.values.astype(float)))
    del results
    return networks


@timer
def cal_pcNet(data: ExpressionData,
              n_comp: int = 3,
              scale_scores: bool = True,
              symmetric: bool = False,
              q: float = 0.95,
              random_state: int = 42,
              **kwargs: object
              ) -> coo_matrix:
    """
    Calculate one pcNet without sampling. An API for getting one PCNet instead of many.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe
    n_comp: int, default = 3
        Number of PCNets composition
    scale_scores: bool, default = True
        To scale the final PCNets scores or not
    symmetric: bool, default = False
        To make the final PCNets symmetric or not
    q: float, default = 0.95
        The quantile value used to determine PCNet's threshold
    random_state: int, default = 42
        Random seed of constructing PCNets
    kwargs
        Keyword arguments

    Returns
    -------
    pcNet: coo_matrix
    Result network

    See Also
    --------
    make_networks

    """
    return make_networks(data,
                         n_nets=1,
                         n_samp_cells=None,
                         n_comp=n_comp,
                         scale_scores=scale_scores,
                         symmetric=symmetric, q=q,
                         random_state=random_state, **kwargs)[0]


@timer
def manifold_alignment(X: pd.DataFrame,
                       Y: pd.DataFrame,
                       d: int = 30,
                       tol: float = 1e-8,
                       **kwargs: object
                       ) -> pd.DataFrame:
    """
    Performing manifold alignment on two dataframes

    Parameters
    ----------
    X: pd.DataFrame
        A gene regulatory network X, expected shape = (n_genes, n_genes)
    Y: pd.DataFrame
        A gene regulatory network Y, expected shape = (n_genes, n_genes)
    d: int, default = 30
        The dimension of the low-dimensional feature space
    tol: float, default = 1e-8
        The tolerance of eigen values
    Returns
    -------
    ma_df: pd.DataFrame
        A dataframe contains manifold alignment result, expected shape = (n_genes * 2, d)
    """
    y_genes = set(Y.index)
    shared_genes = [gene for gene in X.index if gene in y_genes]
    if len(shared_genes) == 0:
        raise ValueError("X and Y do not share any genes")
    X = X.loc[shared_genes, shared_genes]
    Y = Y.loc[shared_genes, shared_genes]
    L = np.eye(len(shared_genes))
    w_X, w_Y = X.values + 1, Y.values + 1
    w_XY = L * (0.9 * (np.sum(w_X) + np.sum(w_Y)) / (2 * len(shared_genes)))
    W = -np.concatenate((np.concatenate((w_X, w_XY), axis=1),
                         np.concatenate((w_XY.T, w_Y), axis=1)), axis=0)
    np.fill_diagonal(W, 0)
    np.fill_diagonal(W, -W.sum(axis=0))
    k = d * 2
    if k >= W.shape[0] - 1:
        raise ValueError(f"d={d} is too large for {len(shared_genes)} shared genes; choose d < {(W.shape[0] - 1) / 2}.")
    eg_vals, eg_vecs = scipy.sparse.linalg.eigs(W, k=k, which="SR", tol=1e-14)
    eg_vals = eg_vals.real
    eg_vecs = eg_vecs.real
    eg_vecs = eg_vecs[:, eg_vals >= tol]
    eg_vecs = eg_vecs[:, np.argsort(eg_vals[eg_vals >= tol], )]
    return pd.DataFrame(eg_vecs[:, :d],
                        index=["X_{g}".format(g=g) for g in shared_genes]+["Y_{g}".format(g=g) for g in shared_genes],
                        columns=["NLMA_{i}".format(i=i+1) for i in range(min(d, eg_vecs.shape[1]))])


@timer
def d_regulation(data: pd.DataFrame,
                 sorted_by: Union[str, list] = "p-value",
                 ascending: Union[bool, list] = True,
                 **kwargs: object) -> pd.DataFrame:
    """
    Evaluates the difference in regulation

    Parameters
    ----------
    data: pd.DataFrame
        A dataframe contains manifold alignment results, expected shape = (n_genes * 2, d)
    sorted_by: str or list of str, default = "p-value"
        Name or list of names to sort by
    ascending: bool or list of bool, default = True
        Sorted ascending (otherwise descending)
    **kwargs
        Keyword arguments for statistic analyses, and n_ko_genes (if any)
            boxcox_kws - kwargs for boxcox test
            chi2_kws - kwargs for chi-square test
            n_ko_genes - int, indicating the number of KO genes

    Examples
    ---------
    d_reg_df = d_regulation(ma_df)

    d_reg_df = d_regulation(ma_df, boxcox_kws={"lmbda": 0}, chi2_kws={"df": 1})

    Returns
    -------
    d_reg_df: pd.DataFrame
        A dataFrame contains difference in regulation result sorted by p-value
        columns: ["Gene", "Distance", "boxcox-transformed distance", "Z", "FC", "p-value", "adjusted p-value"]

    """
    all_gene_names = data.index.to_list()
    gene_names = [g[2:] for g in all_gene_names if "X_" == g[:2]]
    assert len(gene_names) * 2 == len(all_gene_names), 'Number of identified and expected genes are not the same'
    assert all(["Y_" + g == y for g, y in zip(gene_names, all_gene_names[len(gene_names):])]), \
        'Genes are not ordered as expected. X_ genes should be followed by Y_ genes in the same order'
    d_metrics = np.array([np.linalg.norm((data.iloc[x, :] - data.iloc[y, :]).values)
                          for x, y in zip(range(len(gene_names)),
                                          range(len(gene_names), len(all_gene_names)))])
    boxcox_kws = kwargs.get("boxcox_kws") if "boxcox_kws" in kwargs else {}
    chi2_kws = kwargs.get("chi2_kws") if "chi2_kws" in kwargs else {}
    if "df" not in chi2_kws:
        chi2_kws["df"] = 1
    t_d_metrics = d_metrics.astype(float).copy()
    positive = d_metrics > 0
    try:
        if not positive.any():
            raise ValueError("boxcox requires at least one positive distance")
        t, max_log = stats.boxcox(d_metrics[positive], **boxcox_kws)
        t = np.array(t)
        if max_log < 0:
            t = 1 / t
        t_d_metrics[positive] = t
    except Exception:
        warn("cannot find the box-cox transformed values")
        t_d_metrics = d_metrics

    t_std = t_d_metrics.std()
    z_scores = np.zeros_like(t_d_metrics, dtype=float) if t_std == 0 else (t_d_metrics - t_d_metrics.mean()) / t_std
    n_ko_genes = kwargs.get("n_ko_genes") if "n_ko_genes" in kwargs else 0
    expected_val = np.mean(np.power(d_metrics[np.argsort(d_metrics)[::-1][n_ko_genes:]], 2))
    FC = np.zeros_like(d_metrics, dtype=float) if expected_val == 0 else np.power(d_metrics, 2) / expected_val
    p_values = stats.chi2.sf(FC, **chi2_kws)
    p_adj = cal_fdr(p_values)
    df = pd.DataFrame({
        "Gene": gene_names,
        "Distance": d_metrics,
        "boxcox-transformed distance": t_d_metrics,
        "Z": z_scores,
        "FC": FC,
        "p-value": p_values,
        "adjusted p-value": p_adj
    })
    return df.sort_values(sorted_by, ascending=ascending)


def strict_direction(data: np.ndarray, lambd: float = 1) -> np.ndarray:
    """Enforce edge directionality by zeroing the weaker of each ``(i, j)`` / ``(j, i)`` pair.

    Parameters
    ----------
    data
        Square adjacency matrix.
    lambd
        Interpolation weight between the original and strict matrix
        (0 = original, 1 = strict).

    Returns
    -------
    Adjacency with directionality applied.
    """
    if lambd == 0:
        return data
    s_data = data.copy()
    s_data[abs(s_data) < abs(s_data.T)] = 0
    return (1-lambd) * data + lambd * s_data
