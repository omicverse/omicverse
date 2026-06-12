from __future__ import annotations

import warnings
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from .._optional import normalize_torch_device

SpBase = sparse.spmatrix | sparse.sparray  # noqa: TID251
"""Only use when you directly convert it to a known subclass."""

_CSArray = sparse.csr_array | sparse.csc_array  # noqa: TID251
"""Only use if you want to specially handle arrays as opposed to matrices."""

_CSMatrix = sparse.csr_matrix | sparse.csc_matrix  # noqa: TID251
"""Only use if you want to specially handle matrices as opposed to arrays."""

CSRBase = sparse.csr_matrix | sparse.csr_array  # noqa: TID251
CSCBase = sparse.csc_matrix | sparse.csc_array  # noqa: TID251
CSBase = _CSArray | _CSMatrix

D = TypeVar("D", NDArray[np.float32], CSRBase)


# ================================================================================
# Helper functions from scanpy.neighbors._common (to remove scanpy dependency)
# ================================================================================

def _has_self_column(
    indices: NDArray[np.int32 | np.int64],
    distances: NDArray[np.float32 | np.float64],
) -> bool:
    """Check if the first column corresponds to self-neighbors."""
    return (indices[:, 0] == np.arange(indices.shape[0])).any()


def _remove_self_column(
    indices: NDArray[np.int32 | np.int64],
    distances: NDArray[np.float32 | np.float64],
) -> tuple[NDArray[np.int32 | np.int64], NDArray[np.float32 | np.float64]]:
    """Remove self-column from indices and distances."""
    if not _has_self_column(indices, distances):
        msg = "The first neighbor should be the cell itself."
        raise AssertionError(msg)
    return indices[:, 1:], distances[:, 1:]


def _get_sparse_matrix_from_indices_distances(
    indices: NDArray[np.int32 | np.int64],
    distances: NDArray[np.float32 | np.float64],
    *,
    keep_self: bool,
) -> CSRBase:
    """Create a sparse matrix from a pair of indices and distances.

    If keep_self=False, it verifies that the first column is the cell itself,
    then removes it from the explicitly stored zeroes.

    Duplicates in the data are kept as explicitly stored zeroes.
    """
    if not keep_self:
        indices, distances = _remove_self_column(indices, distances)
    indptr = np.arange(0, np.prod(indices.shape) + 1, indices.shape[1])
    return sparse.csr_matrix(
        (
            distances.copy().ravel(),
            indices.copy().ravel(),
            indptr,
        ),
        shape=(indices.shape[0],) * 2,
    )


def _get_indices_distances_from_dense_matrix(
    d: NDArray[np.float32 | np.float64], /, n_neighbors: int
):
    """Get indices and distances from a dense distance matrix."""
    sample_range = np.arange(d.shape[0])[:, None]
    indices = np.argpartition(d, n_neighbors - 1, axis=1)[:, :n_neighbors]
    indices = indices[sample_range, np.argsort(d[sample_range, indices])]
    distances = d[sample_range, indices]
    return indices, distances


def _ind_dist_shortcut(
    d: CSRBase, /
) -> tuple[NDArray[np.int32 | np.int64], NDArray[np.float32 | np.float64]] | None:
    """Shortcut for scipy or RAPIDS style distance matrices."""
    try:
        from fast_array_utils.stats import is_constant
    except ImportError:
        # Fallback implementation
        def is_constant(arr):
            return len(np.unique(arr)) == 1

    nnzs = d.getnnz(axis=1)
    if not is_constant(nnzs):
        msg = (
            "Sparse matrix has no constant number of neighbors per row. "
            "Cannot efficiently get indices and distances."
        )
        warnings.warn(msg, RuntimeWarning)
        return None
    n_obs, n_neighbors = d.shape[0], int(nnzs[0])
    return (
        d.indices.reshape(n_obs, n_neighbors),
        d.data.reshape(n_obs, n_neighbors),
    )


def _ind_dist_slow(
    d: CSRBase, /, n_neighbors: int
) -> tuple[NDArray[np.int32 | np.int64], NDArray[np.float32 | np.float64]]:
    """Slow path for getting indices and distances from sparse matrix."""
    indices = np.zeros((d.shape[0], n_neighbors), dtype=int)
    distances = np.zeros((d.shape[0], n_neighbors), dtype=d.dtype)
    n_neighbors_m1 = n_neighbors - 1
    for i in range(indices.shape[0]):
        neighbors = d[i].nonzero()
        indices[i, 0] = i
        distances[i, 0] = 0
        if len(neighbors[1]) > n_neighbors_m1:
            sorted_indices = np.argsort(d[i][neighbors].A1)[:n_neighbors_m1]
            indices[i, 1:] = neighbors[1][sorted_indices]
            distances[i, 1:] = d[i][
                neighbors[0][sorted_indices], neighbors[1][sorted_indices]
            ]
        else:
            indices[i, 1:] = neighbors[1]
            distances[i, 1:] = d[i][neighbors]
    return indices, distances


def _get_indices_distances_from_sparse_matrix(
    d: CSRBase, /, n_neighbors: int
) -> tuple[NDArray[np.int32 | np.int64], NDArray[np.float32 | np.float64]]:
    """Get indices and distances from a sparse matrix.

    Makes sure that for both of the returned matrices:
    1. the first column corresponds to the cell itself as nearest neighbor.
    2. the number of neighbors (`.shape[1]`) is restricted to `n_neighbors`.
    """
    if (shortcut := _ind_dist_shortcut(d)) is not None:
        indices, distances = shortcut
    else:
        indices, distances = _ind_dist_slow(d, n_neighbors)

    if not _has_self_column(indices, distances):
        indices = np.hstack([np.arange(indices.shape[0])[:, None], indices])
        distances = np.hstack([np.zeros(distances.shape[0])[:, None], distances])

    if indices.shape[1] > n_neighbors:
        indices, distances = indices[:, :n_neighbors], distances[:, :n_neighbors]

    return indices, distances


def gauss(distances: D, n_neighbors: int, *, knn: bool) -> D:  # noqa: PLR0912
    """Derive gaussian connectivities between data points from their distances.

    Parameters
    ----------
    distances
        The input matrix of distances between data points.
    n_neighbors
        The number of nearest neighbors to consider.
    knn
        Specify if the distances have been restricted to k nearest neighbors.

    """
    # init distances
    if isinstance(distances, CSRBase):
        Dsq = distances.power(2)
        indices, distances_sq = _get_indices_distances_from_sparse_matrix(
            Dsq, n_neighbors
        )
    else:
        assert isinstance(distances, np.ndarray)
        Dsq = np.power(distances, 2)
        indices, distances_sq = _get_indices_distances_from_dense_matrix(
            Dsq, n_neighbors
        )

    # exclude the first point, the 0th neighbor
    indices = indices[:, 1:]
    distances_sq = distances_sq[:, 1:]

    # choose sigma, the heuristic here doesn't seem to make much of a difference,
    # but is used to reproduce the figures of Haghverdi et al. (2016)
    if isinstance(distances, CSRBase):
        # as the distances are not sorted
        # we have decay within the n_neighbors first neighbors
        sigmas_sq = np.median(distances_sq, axis=1)
    else:
        # the last item is already in its sorted position through argpartition
        # we have decay beyond the n_neighbors neighbors
        sigmas_sq = distances_sq[:, -1] / 4
    sigmas = np.sqrt(sigmas_sq)

    # compute the symmetric weight matrix
    if not isinstance(distances, CSRBase):
        Num = 2 * np.multiply.outer(sigmas, sigmas)
        Den = np.add.outer(sigmas_sq, sigmas_sq)
        W = np.sqrt(Num / Den) * np.exp(-Dsq / Den)
        # make the weight matrix sparse
        if not knn:
            mask = W > 1e-14
            W[~mask] = 0
        else:
            # restrict number of neighbors to ~k
            # build a symmetric mask
            mask = np.zeros(Dsq.shape, dtype=bool)
            for i, row in enumerate(indices):
                mask[i, row] = True
                for j in row:
                    if i not in set(indices[j]):
                        W[j, i] = W[i, j]
                        mask[j, i] = True
            # set all entries that are not nearest neighbors to zero
            W[~mask] = 0
    else:
        assert isinstance(Dsq, CSRBase)
        W = Dsq.copy()  # need to copy the distance matrix here; what follows is inplace
        for i in range(len(Dsq.indptr[:-1])):
            row = Dsq.indices[Dsq.indptr[i] : Dsq.indptr[i + 1]]
            num = 2 * sigmas[i] * sigmas[row]
            den = sigmas_sq[i] + sigmas_sq[row]
            W.data[Dsq.indptr[i] : Dsq.indptr[i + 1]] = np.sqrt(num / den) * np.exp(
                -Dsq.data[Dsq.indptr[i] : Dsq.indptr[i + 1]] / den
            )
        W = W.tolil()
        for i, row in enumerate(indices):
            for j in row:
                if i not in set(indices[j]):
                    W[j, i] = W[i, j]
        W = W.tocsr()

    return W


def umap(
    knn_indices: NDArray[np.int32 | np.int64],
    knn_dists: NDArray[np.float32 | np.float64],
    *,
    n_obs: int,
    n_neighbors: int,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: float = 1.0,
) -> CSRBase:
    """Wrap for `umap.fuzzy_simplicial_set` :cite:p:`McInnes2018`.

    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """
    with warnings.catch_warnings():
        # umap 0.5.0
        warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
        from umap.umap_ import fuzzy_simplicial_set

    X = sparse.coo_matrix((n_obs, 1))
    connectivities, _sigmas, _rhos = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    return connectivities.tocsr()

from scipy.sparse import csr_matrix


def umap_gpu_optimized(
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    *,
    n_obs: int,
    n_neighbors: int,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: float = 1.0,
    device: str | None = None,
) -> csr_matrix:
    """GPU fuzzy simplicial set wrapping ``umap_pytorch.fuzzy_gpu``.

    Returns a scipy CSR matrix with the same semantics as
    ``umap.umap_.fuzzy_simplicial_set`` (smooth_knn_dist + membership +
    probabilistic set union ``A + Aᵀ − A⊙Aᵀ``).

    The previous implementation in this file diverged from umap-learn:
    ``target_log`` was hardcoded to ``log2(15)`` regardless of
    ``n_neighbors``, ``sigma`` was capped to 1000, the symmetrization only
    summed forward and reverse edges (instead of the fuzzy union), and
    self-loops were not zeroed. This wrapper delegates to the verified
    torch port shipped in ``omicverse.external.umap_pytorch.fuzzy_gpu``.
    """
    import torch
    from ..external.umap_pytorch.fuzzy_gpu import fuzzy_simplicial_set_gpu

    device = normalize_torch_device(device)
    knn_i = torch.as_tensor(knn_indices, dtype=torch.long, device=device)
    knn_d = torch.as_tensor(knn_dists, dtype=torch.float32, device=device)

    rows, cols, vals = fuzzy_simplicial_set_gpu(
        knn_i,
        knn_d,
        n_neighbors=int(n_neighbors),
        set_op_mix_ratio=float(set_op_mix_ratio),
        local_connectivity=float(local_connectivity),
    )

    rows_np = rows.detach().cpu().numpy()
    cols_np = cols.detach().cpu().numpy()
    vals_np = vals.detach().cpu().numpy()
    return sparse.coo_matrix((vals_np, (rows_np, cols_np)), shape=(n_obs, n_obs)).tocsr()
