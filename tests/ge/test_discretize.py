"""``ov.ge.discretize`` vs. SIMBA's own formula
(``simba.tools._general.discretize``), re-derived independently in this
test (histogram -> weighted KMeans -> padded bin edges -> digitize)."""
from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.sparse as sp

anndata = pytest.importorskip("anndata")
pytest.importorskip("sklearn")

from omicverse.ge._simba import discretize


def _reference_discretize(X, n_bins=5, max_bins=100, seed=2021):
    from sklearn.cluster import KMeans

    X = sp.csr_matrix(X)
    nonzero = X.data
    hist_count, hist_edges = np.histogram(nonzero, bins=max_bins, density=False)
    hist_centroids = (hist_edges[:-1] + hist_edges[1:]) / 2
    km = KMeans(n_clusters=n_bins, random_state=seed, n_init=10).fit(
        hist_centroids.reshape(-1, 1), sample_weight=hist_count)
    cluster_centers = np.sort(km.cluster_centers_.flatten())
    padding = (hist_edges[-1] - hist_edges[0]) / (max_bins * 10)
    bin_edges = np.array([hist_edges[0] - padding]
                         + list((cluster_centers[:-1] + cluster_centers[1:]) / 2)
                         + [hist_edges[-1] + padding])
    nonzero_disc = np.digitize(nonzero, bin_edges).reshape(-1)
    return bin_edges, nonzero_disc


@pytest.fixture(scope="module")
def small_real_adata():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(here, "sample", "rna_test.h5ad")
    if not os.path.exists(path):
        pytest.skip(f"bundled sample not found: {path}")
    return anndata.read_h5ad(path)[:100, :80].copy()


def test_discretize_matches_reference_formula(small_real_adata):
    adata = small_real_adata.copy()
    discretize(adata, n_bins=5, max_bins=100, seed=2021)

    ref_edges, ref_disc = _reference_discretize(adata.X, n_bins=5, max_bins=100, seed=2021)
    np.testing.assert_allclose(adata.uns["disc"]["bin_edges"], ref_edges)
    np.testing.assert_array_equal(adata.layers["disc"].data, ref_disc)


def test_discretize_levels_bounded_by_n_bins(small_real_adata):
    adata = small_real_adata.copy()
    discretize(adata, n_bins=4)
    levels = np.unique(adata.layers["disc"].data)
    assert len(levels) <= 4
    assert levels.min() >= 1


def test_discretize_deterministic_with_fixed_seed(small_real_adata):
    a1 = small_real_adata.copy()
    a2 = small_real_adata.copy()
    discretize(a1, n_bins=5, seed=2021)
    discretize(a2, n_bins=5, seed=2021)
    np.testing.assert_array_equal(a1.layers["disc"].data, a2.layers["disc"].data)
