"""PaCMAP and LocalMAP embeddings via anndataoom-fast (C++ port of brisc).

``ov.pp.pacmap`` / ``ov.pp.localmap`` run anndataoom-fast's port of brisc's
PaCMAP / LocalMAP (brisc: Wainberg & Arbabi, MIT License).  Given the same
principal components and nearest neighbours the result is bit-identical to
``brisc.SingleCell.pacmap()`` / ``.localmap()`` and does not depend on the
number of threads.
"""
from __future__ import annotations

import inspect
from datetime import datetime

import numpy as np

from .._registry import register_function
from .._settings import EMOJI, add_reference
from ..report._provenance import note, pick_color_key, tracked

_INSTALL_MSG = (
    "ov.pp.{name} needs the C++ backend: install anndataoom-fast "
    "(pip install anndataoom-fast), which provides the PaCMAP / LocalMAP port of brisc.")

_REFERENCES = {
    "pacmap": ("Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021). Understanding how "
               "dimension reduction tools work: an empirical approach to deciphering t-SNE, "
               "UMAP, TriMAP, and PaCMAP for data visualization. JMLR, 22(201), 1-73."),
    "localmap": ("Wang, Y., Sun, Y., Huang, H., & Rudin, C. (2024). Dimension reduction with "
                 "locally adjusted graphs. arXiv:2412.15426."),
}


def _backend(name):
    try:
        from anndataoom_fast import _embed
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(_INSTALL_MSG.format(name=name)) from e
    return _embed


def _reuse_knn(adata, use_rep, n_total):
    """Per-cell ``n_total`` nearest neighbours (self excluded, ascending, squared
    Euclidean) from ``adata.obsp['distances']`` when it was computed on the same
    representation with enough neighbours; None otherwise."""
    import scipy.sparse as sp
    D = adata.obsp.get("distances") if hasattr(adata, "obsp") else None
    if D is None or not sp.issparse(D) or D.shape[0] != adata.n_obs:
        return None
    nb = adata.uns.get("neighbors")
    params = nb.get("params") if isinstance(nb, dict) else None
    if not isinstance(params, dict) or params.get("use_rep") != use_rep:
        return None
    if str(params.get("metric", "euclidean")).lower() not in ("euclidean", "l2"):
        return None
    n_pcs = params.get("n_pcs")
    if n_pcs and int(n_pcs) != np.shape(adata.obsm[use_rep])[1]:
        return None
    D = D.tocsr()
    n = D.shape[0]
    counts = np.diff(D.indptr)
    rows = np.repeat(np.arange(n, dtype=np.int64), counts)
    keep = D.indices != rows  # drop explicit self entries
    cols, vals = D.indices[keep], D.data[keep]
    if not keep.all():
        counts = np.bincount(rows[keep], minlength=n)
    if counts.min() < n_total:
        return None
    ptr = np.concatenate([[0], np.cumsum(counts)])
    if (counts == counts[0]).all():  # uniform: sort each row as a block
        m = int(counts[0])
        c2, v2 = cols.reshape(n, m), vals.reshape(n, m)
        order = np.argsort(v2, axis=1, kind="stable")[:, :n_total]
        idx = np.take_along_axis(c2, order, 1)
        dist = np.take_along_axis(v2, order, 1)
    else:  # ragged: global (row, distance) sort, first n_total of each row
        r = rows[keep]
        order = np.lexsort((vals, r))
        pos = np.arange(order.size) - ptr[r[order]]
        sel = order[pos < n_total]
        idx, dist = cols[sel].reshape(n, n_total), vals[sel].reshape(n, n_total)
    return idx.astype(np.uint32), np.square(dist.astype(np.float32))


def _run(name, adata, use_rep, key_added, random_state, n_threads, kwargs,
         reuse_neighbors=True):
    _embed = _backend(name)
    if use_rep not in adata.obsm:
        raise KeyError(f"use_rep {use_rep!r} is not in adata.obsm; run ov.pp.pca first "
                       f"(or pass use_rep=...)")
    X = np.asarray(adata.obsm[use_rep])
    print(f"{EMOJI['start']} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Computing "
          f"{name} on obsm[{use_rep!r}] ({X.shape[0]} x {X.shape[1]}) with anndataoom-fast...")
    reused = False
    if reuse_neighbors and kwargs.get("knn_indices") is None and kwargs.get("knn_dists") is None:
        sig = inspect.signature(getattr(_embed, name)).parameters
        k = int(kwargs.get("n_neighbors", sig["n_neighbors"].default)) + \
            int(kwargs.get("n_extra_neighbors", sig["n_extra_neighbors"].default))
        got = _reuse_knn(adata, use_rep, k)
        if got is not None:
            kwargs = dict(kwargs, knn_indices=got[0], knn_dists=got[1])
            reused = True
            print(f"   reusing adata.obsp['distances'] ({k} nearest neighbours per cell)")
    from .._backend import default_threads
    emb = getattr(_embed, name)(X, random_state=random_state,
                                n_threads=default_threads(n_threads), **kwargs)
    adata.obsm[key_added] = emb
    params = {k: v for k, v in kwargs.items() if k not in ("knn_indices", "knn_dists")}
    params.update(use_rep=use_rep, random_state=random_state,
                  knn=("reused obsp['distances']" if reused else "given"
                       if kwargs.get("knn_indices") is not None else "anndataoom_fast.knn"),
                  backend="anndataoom_fast (brisc port)")
    adata.uns[key_added[2:] if key_added.startswith("X_") else key_added] = {"params": params}
    add_reference(adata, name, _REFERENCES[name])
    note(backend=f"omicverse · anndataoom-fast ({name}, brisc port)")
    note(viz=[{"function": "ov.pl.embedding",
               "kwargs": {"basis": key_added, "color": pick_color_key(adata),
                          "frameon": "small"}}])
    print(f"{EMOJI['done']} {name} finished: adata.obsm[{key_added!r}]")


@register_function(
    aliases=["localmap", "LocalMAP", "local_map", "LocalMAP降维", "局部调整图降维"],
    category="preprocessing",
    description="LocalMAP 2-D embedding (PaCMAP with locally adjusted graph) via the "
                "anndataoom-fast C++ port of brisc; deterministic for any thread count",
    prerequisites={"functions": ["pca"]},
    requires={"obsm": ["X_pca"]},
    produces={"obsm": ["X_localmap"]},
    auto_fix="auto",
    examples=["ov.pp.localmap(adata)",
              "ov.pp.localmap(adata, use_rep='X_harmony', key_added='X_localmap_harmony')",
              "ov.pl.embedding(adata, basis='X_localmap', color='leiden')"],
    related=["pp.pacmap", "pp.umap", "pp.tsne", "pp.mde", "pp.pca"],
)
@tracked("localmap", "ov.pp.localmap")
def localmap(adata, use_rep: str = "X_pca", key_added: str = "X_localmap",
             random_state: int = 0, n_threads: int | None = None,
             reuse_neighbors: bool = True, **kwargs) -> None:
    """LocalMAP embedding (Wang et al. 2024), computed by anndataoom-fast's C++ port of brisc.

    Bit-identical to ``brisc.SingleCell.localmap()`` given the same PCs and
    nearest neighbours, and independent of the number of threads.

    Parameters
    ----------
    adata
        AnnData with a representation in ``adata.obsm[use_rep]``.
    use_rep
        Input coordinates, usually principal components (default ``'X_pca'``).
    key_added
        ``adata.obsm`` key for the ``(n_obs, 2)`` float32 embedding; the
        parameters go to ``adata.uns[key_added without 'X_']['params']``.
    random_state
        Seed (brisc's ``seed``).
    n_threads
        OpenMP threads; ``None`` uses the effective CPU count (affinity, cgroup
        quota, Slurm; ``OMP_NUM_THREADS`` wins).  The result does not depend on it.
    reuse_neighbors
        Reuse ``adata.obsp['distances']`` (from ``ov.pp.neighbors``) as the kNN
        when it was computed on the same ``use_rep`` with euclidean metric and
        has at least ``n_neighbors + n_extra_neighbors`` neighbours per cell;
        otherwise the kNN is recomputed.  Note ``ov.pp.neighbors(n_neighbors=k)``
        stores ``k - 1`` neighbours per cell (self excluded).
    **kwargs
        Passed to ``anndataoom_fast.localmap``: ``n_neighbors`` (10),
        ``n_extra_neighbors`` (10), ``n_mid_near_pairs`` (5),
        ``n_further_pairs`` (20), ``num_iterations`` ((100, 100, 250)),
        ``learning_rate`` (1.0), ``max_distance`` (10.0), and optionally precomputed
        ``knn_indices`` / ``knn_dists`` (nearest neighbours excluding the cell
        itself, squared Euclidean distances).  Without them the
        ``n_neighbors + n_extra_neighbors`` nearest neighbours are computed
        with ``anndataoom_fast.knn``.

    Returns
    -------
    None; sets ``adata.obsm[key_added]`` and ``adata.uns[...]['params']``.
    """
    _run("localmap", adata, use_rep, key_added, random_state, n_threads, kwargs, reuse_neighbors)


@register_function(
    aliases=["pacmap", "PaCMAP", "pac_map", "PaCMAP降维"],
    category="preprocessing",
    description="PaCMAP 2-D embedding via the anndataoom-fast C++ port of brisc; "
                "deterministic for any thread count",
    prerequisites={"functions": ["pca"]},
    requires={"obsm": ["X_pca"]},
    produces={"obsm": ["X_pacmap"]},
    auto_fix="auto",
    examples=["ov.pp.pacmap(adata)",
              "ov.pl.embedding(adata, basis='X_pacmap', color='leiden')"],
    related=["pp.localmap", "pp.umap", "pp.tsne", "pp.mde", "pp.pca"],
)
@tracked("pacmap", "ov.pp.pacmap")
def pacmap(adata, use_rep: str = "X_pca", key_added: str = "X_pacmap",
           random_state: int = 0, n_threads: int | None = None,
           reuse_neighbors: bool = True, **kwargs) -> None:
    """PaCMAP embedding (Wang et al. 2021), computed by anndataoom-fast's C++ port of brisc.

    Bit-identical to ``brisc.SingleCell.pacmap()`` given the same PCs and
    nearest neighbours, and independent of the number of threads.

    Parameters
    ----------
    adata
        AnnData with a representation in ``adata.obsm[use_rep]``.
    use_rep
        Input coordinates, usually principal components (default ``'X_pca'``).
    key_added
        ``adata.obsm`` key for the ``(n_obs, 2)`` float32 embedding; the
        parameters go to ``adata.uns[key_added without 'X_']['params']``.
    random_state
        Seed (brisc's ``seed``).
    n_threads
        OpenMP threads; ``None`` uses the effective CPU count (affinity, cgroup
        quota, Slurm; ``OMP_NUM_THREADS`` wins).  The result does not depend on it.
    reuse_neighbors
        Reuse ``adata.obsp['distances']`` (from ``ov.pp.neighbors``) as the kNN
        when it was computed on the same ``use_rep`` with euclidean metric and
        has at least ``n_neighbors + n_extra_neighbors`` neighbours per cell;
        otherwise the kNN is recomputed.  Note ``ov.pp.neighbors(n_neighbors=k)``
        stores ``k - 1`` neighbours per cell (self excluded).
    **kwargs
        Passed to ``anndataoom_fast.pacmap``: ``n_neighbors`` (10),
        ``n_extra_neighbors`` (10), ``n_mid_near_pairs`` (5),
        ``n_further_pairs`` (20), ``num_iterations`` ((100, 100, 250)),
        ``learning_rate`` (1.0), and optionally precomputed
        ``knn_indices`` / ``knn_dists`` (nearest neighbours excluding the cell
        itself, squared Euclidean distances).  Without them the
        ``n_neighbors + n_extra_neighbors`` nearest neighbours are computed
        with ``anndataoom_fast.knn``.

    Returns
    -------
    None; sets ``adata.obsm[key_added]`` and ``adata.uns[...]['params']``.
    """
    _run("pacmap", adata, use_rep, key_added, random_state, n_threads, kwargs, reuse_neighbors)

