"""Sklearn-backed CPU t-SNE for ``ov.pp.tsne``.

A direct ``sklearn.manifold.TSNE`` wrapper that bypasses
``scanpy.tools._tsne``. Two reasons we don't just call scanpy:

1. ``scanpy.tl.tsne`` does not expose ``n_components`` (issue #683) —
   we need it to be honoured end-to-end so ``ov.pp.tsne`` can produce
   3-D / k-D t-SNE embeddings on a CPU box.
2. Reduces the API-version-skew surface: we own the call site, so
   future scanpy refactors of their tsne signature can't break us.

The implementation follows scanpy's own thin-wrapper pattern:
- pick a data matrix from ``adata`` via ``_choose_representation``
- build a ``sklearn.manifold.TSNE`` estimator
- ``fit_transform`` and write back ``adata.obsm[key_added]`` /
  ``adata.uns[key_added]`` keeping scanpy's storage layout so
  downstream plots (``ov.pl.embedding(basis='X_tsne')``)
  keep working unchanged.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def _choose_X(adata: AnnData, *, use_rep: Optional[str], n_pcs: Optional[int]):
    """Reproduce ``scanpy.tools._utils._choose_representation`` semantics
    locally so ``ov.pp.tsne`` doesn't have to import scanpy internals.

    Order of resolution:

    - ``use_rep`` given and present in ``obsm`` / ``layers`` → use it.
    - ``use_rep`` given but missing → KeyError with the available keys.
    - ``use_rep`` is None and ``n_pcs`` is given → take ``obsm['X_pca']``
      truncated to ``n_pcs`` (compute PCA first if missing — same
      behaviour as scanpy / scipy stack).
    - Both None → use raw ``adata.X``.
    """
    if use_rep is not None:
        if use_rep in adata.obsm:
            X = np.asarray(adata.obsm[use_rep])
        elif use_rep in adata.layers:
            X = np.asarray(adata.layers[use_rep])
            if sp.issparse(X):
                X = X.toarray()
        elif use_rep == "X":
            X = adata.X
            if sp.issparse(X):
                X = X.toarray()
            X = np.asarray(X)
        else:
            available = sorted(set(adata.obsm.keys()) | set(adata.layers.keys()))
            raise KeyError(
                f"use_rep={use_rep!r} not found in adata.obsm / .layers. "
                f"Available: {available}"
            )
        if n_pcs is not None:
            if n_pcs > X.shape[1]:
                raise ValueError(
                    f"n_pcs={n_pcs} exceeds rep width "
                    f"adata.obsm[{use_rep!r}].shape[1]={X.shape[1]}"
                )
            X = X[:, :n_pcs]
        return X

    # use_rep is None — fall back to PCA, computing it on demand.
    if "X_pca" in adata.obsm:
        X = np.asarray(adata.obsm["X_pca"])
        if n_pcs is not None:
            X = X[:, :n_pcs]
        return X

    if n_pcs is not None:
        raise ValueError(
            "n_pcs requested but adata.obsm['X_pca'] is missing. "
            "Run ov.pp.pca(...) first or pass use_rep="
            "'<your obsm key>'."
        )

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X)


def tsne_cpu(
    adata: AnnData,
    n_pcs: Optional[int] = None,
    *,
    n_components: int = 2,
    use_rep: Optional[str] = None,
    perplexity: float = 30,
    metric: str = "euclidean",
    early_exaggeration: float = 12,
    learning_rate=1000,
    random_state: int = 0,
    n_jobs: Optional[int] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
    n_iter: Optional[int] = None,
    **_ignored,
) -> Optional[AnnData]:
    """Direct sklearn-backed CPU t-SNE; full ``n_components`` support.

    Drop-in replacement for ``scanpy.tl.tsne`` inside ``ov.pp.tsne``'s
    cpu code path. Produces the same ``adata.obsm['X_tsne']`` /
    ``adata.uns['tsne']`` storage layout so existing plot helpers keep
    working unchanged.

    Parameters
    ----------
    adata, n_pcs, use_rep, perplexity, metric, early_exaggeration,
    learning_rate, random_state, n_jobs, key_added, copy, n_iter
        Mirror the scanpy / sklearn parameter conventions.
    n_components
        Output dimensions. Issue #683's headline win — supported here
        because we go straight to sklearn.

    Returns
    -------
    The mutated AnnData when ``copy=True``; otherwise ``None``
    (writes ``X_tsne`` / ``tsne`` keys in place).
    """
    from sklearn.manifold import TSNE

    if copy:
        adata = adata.copy()

    X = _choose_X(adata, use_rep=use_rep, n_pcs=n_pcs)

    params_sklearn: dict = dict(
        n_components=int(n_components),
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=n_jobs,
        metric=metric,
    )
    if n_iter is not None:
        # sklearn renamed n_iter → max_iter at version 1.5; support both.
        try:
            estimator = TSNE(max_iter=int(n_iter), **params_sklearn)
        except TypeError:
            estimator = TSNE(n_iter=int(n_iter), **params_sklearn)
    else:
        estimator = TSNE(**params_sklearn)

    X_tsne = estimator.fit_transform(X)

    key_uns, key_obsm = ("tsne", "X_tsne") if key_added is None else (key_added, key_added)
    adata.obsm[key_obsm] = X_tsne
    adata.uns[key_uns] = {
        "params": {k: v for k, v in params_sklearn.items() if v is not None}
        | (dict(use_rep=use_rep) if use_rep is not None else {})
        | (dict(n_iter=n_iter) if n_iter is not None else {})
    }
    return adata if copy else None
