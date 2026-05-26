from typing import Sequence

import numpy as np
import pandas as pd
from omicverse.external.scTenifold.core._utils import timer
from tensorly import decomposition
import tensorly as tl

__all__ = ["tensor_decomp"]


@timer
def tensor_decomp(networks: np.ndarray,
                  gene_names: Sequence[str],
                  method: str = "parafac",
                  n_decimal: int = 1,
                  K: int = 5,
                  tol: float = 1e-6,
                  max_iter: int = 1000,
                  random_state: int = 42,
                  **kwargs) -> pd.DataFrame:
    """
    Perform tensor decomposition on pc networks

    Parameters
    ----------
    networks: np.ndarray
        Concatenated network, expected shape = (n_genes, n_genes, n_pcnets)
    gene_names: sequence of str
        The name of each gene in the network (order matters)
    method: str, default = 'parafac'
        Tensor decomposition method, tensorly's decomposition method was used:
        http://tensorly.org/stable/modules/api.html#module-tensorly.decomposition
    n_decimal: int
        Number of decimal in the final df
    K: int
        Rank in parafac function
    tol: float
        Tolerance in the iteration
    max_iter: int
        Number of interation
    random_state: int
        Random seed used to reproduce the same result
    **kwargs:
        Keyword arguments used in the decomposition function

    Returns
    -------
    tensor_decomp_df: pd.DataFrame
        The result of tensor decomposition, expected shape = (n_genes, n_genes)

    References
    ----------
    http://tensorly.org/stable/modules/api.html#module-tensorly.decomposition

    """
    # Us, est, res_hist = cp_als(networks, n_components=K, max_iter=max_iter, tol=tol)
    # print(est.shape, len(Us), res_hist)
    print("Using tensorly")
    factors = getattr(decomposition, method)(networks, rank=K, n_iter_max=max_iter, tol=tol,
                                             random_state=random_state, **kwargs)
    estimate = tl.cp_to_tensor(factors)
    print(estimate.shape)
    out = np.sum(estimate, axis=-1) / networks.shape[-1]
    out = np.round(out / np.max(abs(out)), n_decimal)
    return pd.DataFrame(out, index=gene_names, columns=gene_names)
