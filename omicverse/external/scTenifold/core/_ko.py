from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from omicverse.external.scTenifold.core._networks import make_networks


def ko_propagation(B: np.ndarray,
                   x: np.ndarray,
                   ko_gene_id: Sequence[int],
                   degree: int = 1) -> np.ndarray:
    """Propagate a gene knockout through an adjacency matrix.

    Parameters
    ----------
    B
        Adjacency matrix (genes x genes); diagonal is zeroed in-place on a copy.
    x
        Expression matrix (genes x cells).
    ko_gene_id
        Index of the gene to knock out.
    degree
        Maximum propagation depth.

    Returns
    -------
    Knocked-out expression matrix (non-negative).
    """
    adj_mat = B.copy()
    np.fill_diagonal(adj_mat, 0)
    x_ko = x.copy()
    perturb = np.zeros(shape=x.shape)
    perturb[ko_gene_id, :] = x[ko_gene_id, :]
    is_visited = np.zeros(x_ko.shape[0], dtype=bool)
    x_ko = x_ko - perturb
    for d in range(degree):
        if not is_visited.all():
            perturb = adj_mat @ perturb
            new_visited = (perturb != 0).any(axis=1)
            adj_mat[is_visited, :] = 0
            adj_mat[:, is_visited] = 0
            is_visited = is_visited | new_visited
            x_ko = x_ko - perturb
    return np.where(x_ko >= 0, x_ko, 0)


def reconstruct_pcnets(nets: List[coo_matrix],
                       X_df: pd.DataFrame,
                       ko_gene_id: Sequence[int],
                       degree: int = 1,
                       **kwargs) -> List[np.ndarray]:
    """Rebuild PC networks from knocked-out expression for each input net.

    Parameters
    ----------
    nets
        PC networks (sparse) used to seed propagation.
    X_df
        Expression DataFrame (genes x cells).
    ko_gene_id
        Index of the gene to knock out.
    degree
        Propagation depth passed to :func:`ko_propagation`.
    **kwargs
        Forwarded to :func:`scTenifold.core._networks.make_networks`.

    Returns
    -------
    List of post-knockout PC networks.
    """
    ko_nets = []
    network_kws = dict(kwargs)
    network_kws["n_nets"] = 1
    for net in nets:
        data = ko_propagation(net.toarray(), X_df.values, ko_gene_id, degree)
        data = pd.DataFrame(data, index=X_df.index, columns=X_df.columns)
        ko_net = make_networks(data, **network_kws)[0]
        ko_nets.append(ko_net)
    return ko_nets
