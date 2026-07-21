"""Regression tests for issue #877 — a list/tuple ``resolution`` crashed the
torch GPU Leiden path with an opaque
``TypeError: only integer tensors of a single element can be converted to an
index`` (``resolution * tensor`` became Python list-repetition inside the
local-move). ``resolution`` must be coerced to a scalar float.
"""
import numpy as np
import pytest


def test_coerce_resolution_unwraps_scalars_and_containers():
    from omicverse.pp._leiden_gpu import _coerce_resolution

    assert _coerce_resolution(1.0) == 1.0
    assert _coerce_resolution(2) == 2.0
    assert _coerce_resolution([0.5]) == 0.5          # length-1 list
    assert _coerce_resolution((0.8,)) == 0.8         # length-1 tuple
    assert _coerce_resolution(np.float64(1.5)) == 1.5
    assert _coerce_resolution(np.array([0.3])) == pytest.approx(0.3)


def test_coerce_resolution_rejects_multi_element_and_junk():
    from omicverse.pp._leiden_gpu import _coerce_resolution

    for bad in ([1.0, 2.0], (0.5, 1.0), np.array([1.0, 2.0]), True, "x"):
        with pytest.raises((ValueError, TypeError)):
            _coerce_resolution(bad)


def _small_adata_with_graph(n=200, seed=0):
    import scanpy as sc
    from anndata import AnnData

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 30)).astype(np.float32)
    adata = AnnData(X)
    sc.pp.pca(adata, n_comps=10)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
    return adata


def test_leiden_gpu_multilevel_accepts_list_resolution():
    """The exact #877 trigger: resolution=[1.0]. On a CPU-only box this
    falls back to igraph Leiden, but the coercion runs first — the point is
    it must NOT raise the index TypeError."""
    from omicverse.pp._leiden_gpu import leiden_gpu_sparse_multilevel

    adata = _small_adata_with_graph()
    # would previously raise "only integer tensors ... converted to an index"
    leiden_gpu_sparse_multilevel(adata, resolution=[1.0], key_added="leiden")
    assert "leiden" in adata.obs
    assert adata.obs["leiden"].nunique() >= 1
