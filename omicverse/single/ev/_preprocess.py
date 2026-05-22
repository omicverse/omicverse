"""Preprocessing for single-extracellular-vesicle (single-EV) proteomics.

Single-EV proteomics produces an EV x protein matrix (vesicle ~ cell, protein
marker ~ gene). The standard single-cell stack transfers, but the
normalization step must branch on the *value type* of the assay:

* ``count``     — PBA / DBS-Pro sequencing reads. Centered-log-ratio (CLR),
  the CITE-seq antibody-derived-tag (ADT) convention, is the right transform.
* ``intensity`` — NanoFCM / ExoView flow-imaging fluorescence. ``arcsinh``
  (the CyTOF convention, configurable cofactor) or ``log2``.
* ``binary``    — droplet digital presence/absence. Passes through unchanged.

The value type is read from ``adata.uns['ev']['value_type']`` so that
``normalize(adata, method='auto')`` does the right thing automatically.

Functions
---------
* :func:`normalize`  — value-type-aware normalization (CLR / arcsinh / log2),
  with an optional per-EV size-factor correction for vesicle size.
* :func:`scale`      — per-protein z-scoring with max-clip, stored in a layer.
* :func:`pca`        — PCA on the normalized/scaled matrix.
* :func:`neighbors`  — kNN graph for downstream clustering / UMAP.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..._registry import register_function


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require(modname: str, role: str):
    """Lazy-import a backend with an actionable error message."""
    import importlib

    try:
        return importlib.import_module(modname)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"{role} needs the '{modname}' package. Install with: "
            f"pip install {modname}."
        ) from exc


def _dense(x):
    """Return a dense float64 ndarray from a (possibly sparse) matrix."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def _value_type(adata) -> str:
    """Read the EV value type from ``uns['ev']``, defaulting to ``'count'``."""
    ev = adata.uns.get("ev", {}) if hasattr(adata, "uns") else {}
    return str(ev.get("value_type", "count")).lower()


def _clr(mat: np.ndarray, axis: int = 1, pseudocount: float = 1.0) -> np.ndarray:
    """Centered-log-ratio transform along ``axis`` (per-EV when ``axis=1``).

    ``clr(x) = log(x + pc) - mean_axis(log(x + pc))`` — the CITE-seq ADT
    normalization. The geometric-mean centering removes the per-EV
    composition / depth effect.
    """
    logm = np.log1p(mat) if pseudocount == 1.0 else np.log(mat + pseudocount)
    return logm - logm.mean(axis=axis, keepdims=True)


def _size_factors(mat: np.ndarray) -> np.ndarray:
    """Per-EV size factor = (total signal) / (median total signal).

    Dividing each EV by its size factor controls for vesicle size /
    total surface-protein abundance before the value transform.
    """
    totals = mat.sum(axis=1)
    med = np.median(totals[totals > 0]) if np.any(totals > 0) else 1.0
    sf = totals / med
    sf[sf <= 0] = 1.0
    return sf


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------
@register_function(
    aliases=["ev_normalize", "normalize_ev", "EV标准化", "单囊泡标准化"],
    category="ev",
    description=(
        "Value-type-aware normalization of a single-EV proteomics AnnData. "
        "method='auto' reads uns['ev']['value_type']: CLR (centered-log-"
        "ratio) for count data, arcsinh or log2 for intensity data, and a "
        "pass-through for binary data. An optional per-EV size-factor "
        "correction controls for vesicle size / total surface protein."
    ),
    examples=[
        "ov.single.ev.normalize(adata)",
        "ov.single.ev.normalize(adata, method='clr')",
        "ov.single.ev.normalize(adata, method='arcsinh', cofactor=150)",
        "ov.single.ev.normalize(adata, method='log2', size_factor=True)",
    ],
    related=["single.ev.scale", "single.ev.pca"],
)
def normalize(
    adata,
    *,
    method: str = "auto",
    cofactor: float = 5.0,
    pseudocount: float = 1.0,
    size_factor: bool = False,
    layer: Optional[str] = None,
    key_added: Optional[str] = None,
):
    """Value-type-aware normalization for single-EV proteomics.

    Parameters
    ----------
    adata
        EV x protein AnnData. Raw values are taken from ``layers['counts']``
        if present, otherwise from ``adata.X``.
    method
        ``'auto'`` (default) picks from ``uns['ev']['value_type']``:
        ``count`` -> ``'clr'``, ``intensity`` -> ``'arcsinh'``,
        ``binary`` -> ``'none'``. May also be set explicitly to ``'clr'``,
        ``'arcsinh'``, ``'log2'`` or ``'none'``.
    cofactor
        Cofactor for the ``arcsinh`` transform, ``arcsinh(x / cofactor)``
        (the CyTOF convention; 5 for mass cytometry, ~150 for fluorescence).
    pseudocount
        Pseudocount added before the ``clr`` / ``log2`` logarithm.
    size_factor
        If ``True``, divide each EV by its size factor (total signal over
        the median total signal) *before* the value transform, to control
        for vesicle size / total surface-protein abundance.
    layer
        Input layer; ``None`` uses ``layers['counts']`` or ``X``.
    key_added
        If given, the normalized matrix is written to ``layers[key_added]``
        and ``X`` is left untouched; otherwise it overwrites ``adata.X``.

    Returns
    -------
    :class:`anndata.AnnData`
        The same object, with the normalized matrix in ``X`` (or in
        ``layers[key_added]``) and a record under ``uns['ev']['normalize']``.

    Notes
    -----
    The raw input is preserved in ``layers['counts']`` (created on first
    call) so the transform is always reproducible / re-runnable.
    """
    valid = {"auto", "clr", "arcsinh", "log2", "none"}
    if method not in valid:
        raise ValueError(f"method must be one of {sorted(valid)}, got {method!r}")

    # resolve raw input matrix
    if layer is not None:
        mat = _dense(adata.layers[layer])
    elif "counts" in adata.layers:
        mat = _dense(adata.layers["counts"])
    else:
        mat = _dense(adata.X)
        adata.layers["counts"] = mat.copy()

    vtype = _value_type(adata)
    if method == "auto":
        method = {
            "count": "clr",
            "intensity": "arcsinh",
            "binary": "none",
        }.get(vtype, "clr")

    if size_factor and method != "none":
        sf = _size_factors(mat)
        mat = mat / sf[:, None]
    else:
        sf = None

    if method == "clr":
        out = _clr(mat, axis=1, pseudocount=pseudocount)
    elif method == "arcsinh":
        out = np.arcsinh(mat / float(cofactor))
    elif method == "log2":
        out = np.log2(mat + pseudocount)
    else:  # none — binary / pass-through
        out = mat.astype(np.float64)

    if key_added is not None:
        adata.layers[key_added] = out
    else:
        adata.X = out

    ev = adata.uns.setdefault("ev", {})
    ev["normalize"] = {
        "method": method,
        "value_type": vtype,
        "cofactor": float(cofactor),
        "pseudocount": float(pseudocount),
        "size_factor": bool(size_factor),
        "layer_out": key_added,
    }
    return adata


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------
@register_function(
    aliases=["ev_scale", "scale_ev", "EV缩放", "蛋白标准化"],
    category="ev",
    description=(
        "Per-protein z-scoring of a single-EV proteomics matrix with an "
        "optional max-clip on the scaled values. The scaled matrix is stored "
        "in a layer (default 'scaled') so the normalized X is preserved."
    ),
    examples=[
        "ov.single.ev.scale(adata)",
        "ov.single.ev.scale(adata, max_value=10, zero_center=True)",
    ],
    related=["single.ev.normalize", "single.ev.pca"],
)
def scale(
    adata,
    *,
    max_value: Optional[float] = 10.0,
    zero_center: bool = True,
    layer: Optional[str] = None,
    key_added: str = "scaled",
):
    """Per-protein z-scoring with max-clip, stored in a layer.

    Parameters
    ----------
    adata
        EV x protein AnnData (normalized).
    max_value
        Clip scaled values to ``[-max_value, max_value]``; ``None`` disables
        clipping.
    zero_center
        Subtract the per-protein mean (``True``) or only divide by the
        per-protein standard deviation (``False``).
    layer
        Input layer; ``None`` uses ``adata.X``.
    key_added
        Layer the scaled matrix is written to (default ``'scaled'``).

    Returns
    -------
    :class:`anndata.AnnData`
        The same object, with the scaled matrix in ``layers[key_added]``.
    """
    mat = _dense(adata.X) if layer is None else _dense(adata.layers[layer])
    mean = mat.mean(axis=0, keepdims=True)
    std = mat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    out = (mat - mean) / std if zero_center else mat / std
    if max_value is not None:
        out = np.clip(out, -float(max_value), float(max_value))
    adata.layers[key_added] = out
    ev = adata.uns.setdefault("ev", {})
    ev["scale"] = {
        "max_value": None if max_value is None else float(max_value),
        "zero_center": bool(zero_center),
        "layer_out": key_added,
    }
    return adata


# ---------------------------------------------------------------------------
# pca
# ---------------------------------------------------------------------------
@register_function(
    aliases=["ev_pca", "pca_ev", "EV主成分分析", "单囊泡PCA"],
    category="ev",
    description=(
        "Principal-component analysis of a single-EV proteomics matrix. "
        "Protein panels are small (8-250 markers), so there is no HVG step "
        "— every protein is kept. Writes obsm['X_pca']."
    ),
    examples=[
        "ov.single.ev.pca(adata)",
        "ov.single.ev.pca(adata, n_comps=20, layer='scaled')",
    ],
    related=["single.ev.scale", "single.ev.neighbors"],
)
def pca(
    adata,
    *,
    n_comps: Optional[int] = None,
    layer: Optional[str] = "scaled",
    use_highly_variable: bool = False,
    svd_solver: str = "auto",
    random_state: int = 0,
):
    """PCA on the normalized/scaled single-EV matrix.

    Parameters
    ----------
    adata
        EV x protein AnnData.
    n_comps
        Number of principal components. ``None`` keeps
        ``min(n_proteins, n_evs) - 1`` (small panels keep all proteins).
    layer
        Layer to run PCA on (default ``'scaled'`` if present, else ``X``).
    use_highly_variable
        Kept for API compatibility — single-EV panels are small, so HVG
        selection is skipped and every protein is used.
    svd_solver
        SVD solver passed to :class:`sklearn.decomposition.PCA`.
    random_state
        Random seed for reproducibility.

    Returns
    -------
    :class:`anndata.AnnData`
        The same object, with ``obsm['X_pca']``, the loadings in
        ``varm['PCs']`` and the variance ratio in
        ``uns['pca']['variance_ratio']``.
    """
    from sklearn.decomposition import PCA as _PCA

    if layer is not None and layer in adata.layers:
        mat = _dense(adata.layers[layer])
    else:
        mat = _dense(adata.X)

    max_comps = max(1, min(mat.shape) - 1)
    if n_comps is None:
        n_comps = max_comps
    n_comps = int(min(n_comps, max_comps))

    model = _PCA(
        n_components=n_comps, svd_solver=svd_solver, random_state=random_state
    )
    coords = model.fit_transform(mat - mat.mean(axis=0, keepdims=True))
    adata.obsm["X_pca"] = coords
    adata.varm["PCs"] = model.components_.T
    adata.uns["pca"] = {
        "variance_ratio": model.explained_variance_ratio_,
        "variance": model.explained_variance_,
        "n_comps": n_comps,
        "layer": layer,
    }
    return adata


# ---------------------------------------------------------------------------
# neighbors
# ---------------------------------------------------------------------------
@register_function(
    aliases=["ev_neighbors", "neighbors_ev", "EV近邻图", "单囊泡近邻"],
    category="ev",
    description=(
        "Build a k-nearest-neighbor graph over EVs (on the PCA embedding, "
        "or the protein matrix directly) for downstream Leiden clustering "
        "and UMAP. Wraps scanpy.pp.neighbors when scanpy is available, with "
        "a pure scikit-learn fallback."
    ),
    examples=[
        "ov.single.ev.neighbors(adata)",
        "ov.single.ev.neighbors(adata, n_neighbors=30, use_rep='X_pca')",
    ],
    related=["single.ev.pca", "single.ev.leiden"],
)
def neighbors(
    adata,
    *,
    n_neighbors: int = 15,
    use_rep: Optional[str] = None,
    n_pcs: Optional[int] = None,
    metric: str = "euclidean",
    random_state: int = 0,
):
    """kNN graph over EVs for clustering / UMAP.

    Parameters
    ----------
    adata
        EV x protein AnnData.
    n_neighbors
        Number of neighbors per EV.
    use_rep
        Representation to compute neighbors on. ``None`` uses
        ``obsm['X_pca']`` if present, else ``adata.X``.
    n_pcs
        Number of PCs to use when ``use_rep`` is a PCA embedding.
    metric
        Distance metric.
    random_state
        Random seed.

    Returns
    -------
    :class:`anndata.AnnData`
        The same object, with the connectivity / distance graphs in
        ``obsp`` and the neighbor parameters in ``uns['neighbors']``.
    """
    try:
        import scanpy as sc

        kw = dict(
            n_neighbors=n_neighbors, metric=metric, random_state=random_state
        )
        if use_rep is not None:
            kw["use_rep"] = use_rep
        elif "X_pca" in adata.obsm:
            kw["use_rep"] = "X_pca"
        else:
            kw["use_rep"] = "X"
        if n_pcs is not None:
            kw["n_pcs"] = n_pcs
        sc.pp.neighbors(adata, **kw)
        return adata
    except ImportError:  # pragma: no cover - fallback path
        pass

    # pure scikit-learn fallback
    from sklearn.neighbors import kneighbors_graph

    if use_rep is not None and use_rep in adata.obsm:
        rep = _dense(adata.obsm[use_rep])
    elif "X_pca" in adata.obsm:
        rep = _dense(adata.obsm["X_pca"])
    else:
        rep = _dense(adata.X)
    if n_pcs is not None:
        rep = rep[:, :n_pcs]

    n_obs = rep.shape[0]
    k = int(min(n_neighbors, n_obs - 1))
    dist = kneighbors_graph(rep, n_neighbors=k, mode="distance", metric=metric)
    conn = kneighbors_graph(rep, n_neighbors=k, mode="connectivity", metric=metric)
    conn = conn.maximum(conn.T)  # symmetrize
    adata.obsp["distances"] = dist
    adata.obsp["connectivities"] = conn
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {
            "n_neighbors": k,
            "metric": metric,
            "use_rep": use_rep or ("X_pca" if "X_pca" in adata.obsm else "X"),
        },
    }
    return adata
