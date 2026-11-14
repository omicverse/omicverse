"""Minimum-distortion embedding via pymde (port of scvi-tools' ``scvi.model.utils.mde``;
no scvi-tools dependency)."""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
try:
    import torch  # Optional, used for GPU acceleration
except ImportError:  # pragma: no cover - optional dependency
    torch = None
from scipy.sparse import spmatrix
from .._registry import register_function

@register_function(
    aliases=["MDE降维", "mde", "minimum_distortion_embedding", "MDE嵌入", "最小失真嵌入"],
    category="utils",
    description="Minimum Distortion Embedding (MDE) for 2D visualization with GPU acceleration",
    prerequisites={
        'functions': ['pca'],
        'optional_functions': []
    },
    requires={
        'obsm': ['X_pca']
    },
    produces={
        'obsm': ['X_mde']
    },
    auto_fix='none',
    examples=[
        "# Basic MDE embedding",
        "X_mde = ov.utils.mde(adata.obsm['X_pca'])",
        "adata.obsm['X_mde'] = X_mde",
        "# GPU-accelerated MDE",
        "X_mde = ov.utils.mde(adata.obsm['X_pca'], device='cuda')",
        "# Custom embedding dimension",
        "X_mde = ov.utils.mde(adata.obsm['X_pca'], embedding_dim=3)",
        "# Use with visualization",
        "ov.pl.embedding(adata, basis='X_mde', color='leiden')"
    ],
    related=["pp.pca", "utils.embedding", "pl.umap", "pl.tsne"]
)
def mde(
    data: Union[np.ndarray, pd.DataFrame, spmatrix, torch.Tensor],
    device: Optional[Literal["cpu", "cuda"]] = None,
    **kwargs,
) -> np.ndarray:
    """Util to run :func:`pymde.preserve_neighbors` for visualization of scvi-tools embeddings.

    Parameters
    ----------
    data
        The data of shape (n_obs, k), where k is typically defined by one of the models
        in scvi-tools that produces an embedding (e.g., :class:`~scvi.model.SCVI`.)
    device
        Whether to run on cpu or gpu ("cuda"). If None, tries to run on gpu if available.
    kwargs
        Keyword args to :func:`pymde.preserve_neighbors`

    Returns
    -------
    The pymde embedding, defaults to two dimensions.

    Notes
    -----
    This function is included in scvi-tools to provide an alternative to UMAP/TSNE that is GPU-
    accelerated. The appropriateness of use of visualization of high-dimensional spaces in single-
    cell omics remains an open research questions. See:

    Chari, Tara, Joeyta Banerjee, and Lior Pachter. "The specious art of single-cell genomics." bioRxiv (2021).

    If you use this function in your research please cite:

    Agrawal, Akshay, Alnur Ali, and Stephen Boyd. "Minimum-distortion embedding." arXiv preprint arXiv:2103.02559 (2021).

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.SCVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_mde"] = scvi.model.utils.mde(adata.obsm["X_scVI"])
    """
    try:
        import pymde
    except ImportError as err:
        raise ImportError(
            "Please install pymde package via `pip install pymde`"
        ) from err

    if isinstance(data, pd.DataFrame):
        data = data.values

    if device is None:  # auto: CUDA when available (an explicit device is honoured)
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    _kwargs = {
        "embedding_dim": 2,
        "constraint": pymde.Standardized(),
        "repulsive_fraction": 0.7,
        "verbose": False,
        "device": device,
        "n_neighbors": 15,
    }
    _kwargs.update(kwargs)

    emb = pymde.preserve_neighbors(data, **_kwargs).embed(verbose=_kwargs["verbose"])

    if torch is not None and isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()

    return emb