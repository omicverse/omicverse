#!/usr/bin/env python
"""
GPU-accelerated KNN search for OmicVerse.

The previous implementation called ``torch_geometric.nn.knn(X, X, k)`` (which
under the hood is ``torch_cluster.knn``) on the full N x D matrix at once,
plus a ``torch.cdist(X, X)`` fallback. Both materialize an ``N x N`` distance
tensor at some point and OOM well before 1M cells (4 TB at fp32).

This rewrite replaces both paths with a **chunked matmul KNN**:

    dist^2_ij = ||X_i||^2 + ||Y_j||^2 - 2 X_i @ Y_j^T

For each query chunk ``Q`` we materialize only a ``(Q, N)`` distance tensor and
take ``topk`` along axis 1. Memory is ``N*D + Q*N + Q*k`` floats — independent
of total queries when chunked across the query axis. Chunk size is autotuned
from free VRAM with conservative headroom.

Only Euclidean distance is supported (the only metric the parametric-UMAP path
calls into here). For other metrics callers should use pynndescent.
"""

import math

import numpy as np
import torch
from scipy import sparse

from .._optional import normalize_torch_device


_CHUNK_CAP = 4096  # caps the (Q, N) distance buffer; 4096*1M*4B = 16 GB


def _autotune_chunk(n_samples, n_features, device, headroom_gb=1.0):
    """Pick a query-chunk size that keeps the (Q,N) distance buffer well under
    free VRAM. Headroom covers the squared-norm vectors, the topk output, the
    full X tensor itself, and the autograd / cuBLAS scratch space."""
    if device.type != "cuda":
        return min(2048, n_samples)

    free_bytes, _ = torch.cuda.mem_get_info(device)
    # X itself + Y normsq + headroom
    base = n_samples * n_features * 4 + n_samples * 4 + int(headroom_gb * (1 << 30))
    avail = max(free_bytes - base, 256 * (1 << 20))  # at least 256 MB for the chunk
    # each chunk row needs (N + n_features) floats for dist+gather, then topk is k floats per row
    per_row = (n_samples + n_features + 32) * 4
    q = max(64, min(_CHUNK_CAP, avail // per_row))
    return int(q)


def _chunked_knn_l2(X, k, device, chunk_size=None, include_self=True, return_tensor=False):
    """Chunked matmul-based exact L2 KNN. Returns (indices, distances).

    If ``return_tensor=True``, results stay on the input device as torch
    tensors (lets downstream consumers run a fully-GPU UMAP graph build).
    Otherwise returns numpy arrays (legacy behavior).

    Parameters
    ----------
    X : torch.Tensor, shape (N, D)
        Query == reference data (we only call self-KNN here).
    k : int
        Number of neighbors per row.
    device : torch.device
    chunk_size : int or None
        Query chunk size. None to autotune.
    include_self : bool
        If False, the row's own index (distance ~0) is excluded — emulates
        umap-learn's convention where the first neighbor is the point itself.

    Returns
    -------
    indices : np.ndarray (N, k) int64
    distances : np.ndarray (N, k) float32
    """
    device = normalize_torch_device(device, fallback_to_cpu=True)
    n_samples, n_features = X.shape
    Y = X if X.device == device else X.to(device, non_blocking=True)
    y_normsq = (Y * Y).sum(dim=1)  # (N,)

    if chunk_size is None:
        chunk_size = _autotune_chunk(n_samples, n_features, device)
    chunk_size = max(1, min(chunk_size, n_samples))

    # If we need to drop self, request k+1 internally and slice.
    k_query = k + 1 if not include_self else k
    k_query = min(k_query, n_samples)

    if return_tensor:
        out_idx = torch.empty((n_samples, k), device=device, dtype=torch.long)
        out_dist = torch.empty((n_samples, k), device=device, dtype=torch.float32)
    else:
        out_idx = torch.empty((n_samples, k), dtype=torch.long)
        out_dist = torch.empty((n_samples, k), dtype=torch.float32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        Xc = Y[start:end]  # (Q, D), already on device
        x_normsq = (Xc * Xc).sum(dim=1, keepdim=True)  # (Q, 1)

        # Compute ||x||^2 + ||y||^2 - 2 X @ Y.T as a SINGLE (Q, N) tensor.
        # Doing `torch.addmm(x_normsq + y_normsq, ...)` would materialize the
        # bias tensor first AND a separate addmm output, doubling peak VRAM.
        # Build the matmul output, then add the norms in-place.
        dist2 = torch.mm(Xc, Y.T)              # (Q, N), one alloc
        dist2.mul_(-2.0)
        dist2.add_(y_normsq.unsqueeze(0))      # broadcast (1, N) -> (Q, N)
        dist2.add_(x_normsq)                   # broadcast (Q, 1) -> (Q, N)
        dist2.clamp_(min=0.0)                   # cancellation noise -> 0

        d_top, i_top = torch.topk(dist2, k_query, dim=1, largest=False, sorted=True)

        if not include_self and k_query > k:
            # Drop the self-match. It is *almost always* the first column, but
            # in the rare case of exact duplicates it might not be — guard for
            # that explicitly.
            row_arange = torch.arange(start, end, device=device).unsqueeze(1)
            self_mask = i_top.eq(row_arange)
            # Position of the self entry per row (or k_query if absent).
            first_self = torch.where(
                self_mask.any(dim=1),
                self_mask.float().argmax(dim=1),
                torch.full((end - start,), k_query, device=device, dtype=torch.long),
            )
            # Build a boolean keep-mask, then gather k columns per row.
            col_idx = torch.arange(k_query, device=device).unsqueeze(0).expand(end - start, -1)
            keep = col_idx.ne(first_self.unsqueeze(1))
            # Shifted gather: for each row, take the first k columns where keep is True.
            keep_pos = keep.cumsum(dim=1) - 1  # 0-based position among kept columns
            sel = torch.zeros((end - start, k), device=device, dtype=torch.long)
            sel_d = torch.zeros((end - start, k), device=device, dtype=dist2.dtype)
            row_take = (keep_pos < k) & keep
            r, c = row_take.nonzero(as_tuple=True)
            sel[r, keep_pos[r, c]] = i_top[r, c]
            sel_d[r, keep_pos[r, c]] = d_top[r, c]
            i_top, d_top = sel, sel_d

        # Convert squared distance back to Euclidean.
        d_top = d_top.clamp_(min=0.0).sqrt_()

        if return_tensor:
            out_idx[start:end] = i_top[:, :k]
            out_dist[start:end] = d_top[:, :k]
        else:
            out_idx[start:end] = i_top[:, :k].cpu()
            out_dist[start:end] = d_top[:, :k].cpu()

        # Free the chunk tensors before the next iteration.
        del dist2, x_normsq, d_top, i_top, Xc

    if return_tensor:
        return out_idx, out_dist
    return out_idx.numpy(), out_dist.numpy()


def pyg_knn_search(X, k=15, device="cuda", chunk_size=None, include_self=True, batch_size=None, return_tensor=False):
    """GPU-accelerated KNN search.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor of shape (n_samples, n_features)
    k : int
        Number of nearest neighbors.
    device : str or torch.device
        'cuda', 'cpu', or a specific torch.device.
    chunk_size : int or None
        Query-axis chunk size. None autotunes from free VRAM.
    include_self : bool
        Keep the row's own index in the neighbor list (umap-learn convention
        treats this as the first neighbor with distance 0).
    batch_size : int, optional
        Deprecated alias for ``chunk_size`` (kept so old call sites do not
        break).

    Returns
    -------
    indices : np.ndarray (n_samples, k) int64
    distances : np.ndarray (n_samples, k) float32
    """
    # Cast scalar args to Python int — torch.empty refuses numpy.int64 in size
    # tuples, and upstream callers (scanpy obsp metadata, np.prod outputs)
    # often pass numpy scalars.
    k = int(k)
    if chunk_size is not None:
        chunk_size = int(chunk_size)
    if batch_size is not None:
        batch_size = int(batch_size)
    device = normalize_torch_device(device, fallback_to_cpu=True)

    if isinstance(X, np.ndarray):
        X_torch = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    else:
        X_torch = X.float().contiguous()

    if X_torch.device != device:
        X_torch = X_torch.to(device, non_blocking=True)

    if chunk_size is None and batch_size is not None:
        chunk_size = batch_size

    return _chunked_knn_l2(
        X_torch, k=k, device=device, chunk_size=chunk_size,
        include_self=include_self, return_tensor=return_tensor,
    )


# Kept for backwards compatibility with imports elsewhere in the codebase.
def torch_knn_fallback(X_torch, k=15, batch_size=None):
    if isinstance(X_torch, np.ndarray):
        X_torch = torch.from_numpy(np.ascontiguousarray(X_torch, dtype=np.float32))
    return _chunked_knn_l2(
        X_torch.float(),
        k=k,
        device=X_torch.device if X_torch.device.type != "cpu" else torch.device("cpu"),
        chunk_size=batch_size,
    )


def torch_knn_transformer(n_neighbors=15, metric="euclidean", device="auto"):
    return TorchKNNTransformer(n_neighbors=n_neighbors, metric=metric, device=device)


class TorchKNNTransformer:
    """sklearn-compatible KNN transformer backed by chunked matmul KNN."""

    def __init__(self, n_neighbors=15, metric="euclidean", device="auto"):
        if metric != "euclidean":
            raise ValueError(
                f"TorchKNNTransformer only supports metric='euclidean', got {metric!r}. "
                "Use pynndescent for other metrics."
            )
        self.n_neighbors = n_neighbors
        self.metric = metric
        normalized_device = None if device == "auto" else device
        self.device = normalize_torch_device(normalized_device, fallback_to_cpu=True)

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        knn_indices, knn_distances = pyg_knn_search(
            X, k=self.n_neighbors, device=self.device
        )
        n_samples = X.shape[0]
        row_indices = np.repeat(np.arange(n_samples), self.n_neighbors)
        col_indices = knn_indices.ravel()
        distances = knn_distances.ravel()
        return sparse.csr_matrix(
            (distances, (row_indices, col_indices)),
            shape=(n_samples, n_samples),
        )

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        indices, distances = pyg_knn_search(X, k=n_neighbors, device=self.device)
        if return_distance:
            return distances, indices
        return indices

    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors, "metric": self.metric, "device": self.device}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
