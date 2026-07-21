"""Tests for Seurat-style RPCA integration (``ov.single.rpca_integrate``).

Issue #883. These are lightweight, dependency-guarded tests that check the
port runs end to end and actually removes a batch shift. The heavy numerical
parity check against R Seurat 5.4.0 lives outside CI (documented in the PR):
given identical inputs it reproduces Seurat's ``integrated.dr`` to
Pearson r > 0.999 and finds the identical anchor set.
"""
import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scanpy")

import anndata as ad


def _two_batch_adata(seed=0, n_per=150, n_genes=200, shift=3.0):
    """Two batches sharing 3 cell types, with an additive per-gene batch shift.

    Returns log-normalized-like data so RPCA (which expects ``data``) is happy.
    """
    rng = np.random.default_rng(seed)
    n_types = 3
    centers = rng.normal(0, 2, size=(n_types, n_genes))
    batch_shift = rng.normal(0, shift, size=n_genes)

    def make(offset):
        labels = rng.integers(0, n_types, size=n_per)
        X = np.vstack([centers[l] + rng.normal(0, 1, n_genes) for l in labels])
        X = X + offset
        return np.clip(X, 0, None), labels

    x1, l1 = make(0.0)
    x2, l2 = make(batch_shift)
    X = np.vstack([x1, x2]).astype(np.float32)
    obs_batch = np.array(["A"] * n_per + ["B"] * n_per)
    obs_label = np.concatenate([l1, l2]).astype(str)
    adata = ad.AnnData(X=X)
    adata.obs["batch"] = obs_batch
    adata.obs["celltype"] = obs_label
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs_names = [f"c{i}" for i in range(2 * n_per)]
    return adata


def _cross_batch_frac(emb, batch, k=15):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb)
    b = (batch == batch[0]).astype(int)
    return float(np.mean([(b[idx[i, 1:]] != b[i]).mean() for i in range(len(b))]))


def test_rpca_runs_and_writes_obsm():
    from omicverse.single import rpca_integrate

    adata = _two_batch_adata()
    out = rpca_integrate(adata, "batch", n_pcs=20, n_features=150, verbose=False)
    assert out is adata
    assert "X_rpca" in adata.obsm
    assert adata.obsm["X_rpca"].shape[0] == adata.n_obs
    assert adata.obsm["X_rpca"].shape[1] <= 20
    assert np.isfinite(adata.obsm["X_rpca"]).all()
    info = adata.uns["rpca"]["X_rpca"]
    assert info["n_batches"] == 2 and info["n_anchors"] > 0


def test_rpca_removes_batch_shift():
    """After RPCA, batches should mix substantially better than raw PCA."""
    from omicverse.single._rpca import (
        rpca_integrate, _resolve_orig, _scale_data, _run_pca, _as_dense)

    adata = _two_batch_adata()
    batch = adata.obs["batch"].to_numpy()

    # baseline: joint PCA with no correction
    data_gc = _as_dense(adata.X).T
    _, orig = _run_pca(_scale_data(data_gc), 20)
    base = _cross_batch_frac(orig, batch)

    rpca_integrate(adata, "batch", n_pcs=20, n_features=150, verbose=False)
    after = _cross_batch_frac(adata.obsm["X_rpca"], batch)

    # perfectly mixed → ~0.5; raw batch-shifted data mixes poorly
    assert after > base
    assert after > 0.3


def test_rpca_via_batch_correction_dispatch():
    from omicverse.single import batch_correction

    adata = _two_batch_adata()
    batch_correction(adata, batch_key="batch", methods="rpca",
                     n_pcs=20, n_features=150)
    assert "X_rpca" in adata.obsm


def test_rpca_deterministic():
    from omicverse.single import rpca_integrate

    a1 = _two_batch_adata(seed=1)
    a2 = _two_batch_adata(seed=1)
    rpca_integrate(a1, "batch", n_pcs=20, n_features=150, verbose=False)
    rpca_integrate(a2, "batch", n_pcs=20, n_features=150, verbose=False)
    np.testing.assert_allclose(a1.obsm["X_rpca"], a2.obsm["X_rpca"], rtol=1e-6)


def test_rpca_three_batches_sample_tree():
    """>2 batches exercise the hclust sample-tree merge path."""
    from omicverse.single import rpca_integrate

    a = _two_batch_adata(n_per=100)
    b = _two_batch_adata(seed=5, n_per=100)
    b.obs["batch"] = "C"
    b.obs_names = [f"d{i}" for i in range(b.n_obs)]
    merged = ad.concat([a, b])
    merged.var_names = a.var_names
    rpca_integrate(merged, "batch", n_pcs=15, n_features=150, verbose=False)
    assert merged.obsm["X_rpca"].shape[0] == merged.n_obs
    assert merged.uns["rpca"]["X_rpca"]["n_batches"] == 3
