"""GPU non-parametric UMAP — a torch twin of umap-learn / scanpy CPU UMAP.

This package re-implements the *non-parametric* UMAP embedding optimisation
(umap-learn 0.5.7 ``simplicial_set_embedding`` + ``optimize_layout_euclidean``)
in PyTorch with GPU acceleration. It exists because omicverse's previous
GPU/"mixed" UMAP used *parametric* UMAP (an MLP), which produces visibly
different embeddings from the CPU path; users reported the inconsistency.

The entry point ``simplicial_set_embedding_torch`` is argument-compatible
with ``umap.umap_.simplicial_set_embedding`` and consumes the same fuzzy
``connectivities`` graph, so it is a drop-in backend for the CPU UMAP path
and yields structurally-equivalent output (same fuzzy graph, same a/b curve,
same spectral init, same edge-SGD update rule).

Device support (``backend='auto'`` picks the best available):
- **CUDA** (torch) — fully accelerated: lobpcg spectral init + edge-SGD.
- **Apple Silicon** (MLX/metal) — edge-SGD on metal via ``_sgd_mlx`` (the
  same MLX path omicverse uses for PCA/Harmony, not torch-MPS, which lacks
  the float64/sparse coverage here); spectral init runs on CPU.
- **CPU** (torch) — everything on host.

``backend`` may be ``'auto'`` | ``'torch'`` | ``'mlx'``.

Example
-------
>>> from omicverse.utils.gpuex.umap import (
...     simplicial_set_embedding_torch, find_ab_params)
>>> a, b = find_ab_params(spread=1.0, min_dist=0.5)
>>> emb, _ = simplicial_set_embedding_torch(
...     adata.X, adata.obsp['connectivities'], n_components=2,
...     initial_alpha=1.0, a=a, b=b, gamma=1.0, negative_sample_rate=5,
...     n_epochs=None, init='spectral', random_state=0,
...     metric='euclidean', metric_kwds={})
"""
from ._ab import find_ab_params
from ._embedding import simplicial_set_embedding_torch
from ._init import make_epochs_per_sample, noisy_scale_coords
from ._sgd import optimize_layout_torch
from ._sgd_mlx import mlx_available, optimize_layout_mlx

__all__ = [
    "simplicial_set_embedding_torch",
    "find_ab_params",
    "make_epochs_per_sample",
    "noisy_scale_coords",
    "optimize_layout_torch",
    "optimize_layout_mlx",
    "mlx_available",
]
