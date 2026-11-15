"""Native Moran's I / Geary's C kernels and chunked row sums in ``ov.space._svg``."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from omicverse.space import _svg


def _graph(n, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), 4)
    cols = rng.integers(0, n, size=rows.size)
    g = sp.csr_matrix((rng.random(rows.size), (rows, cols)), shape=(n, n))
    g.setdiag(0)
    g.eliminate_zeros()
    return g


def _reference(g, vals):
    """Loop form of the statistics (as in scanpy.metrics)."""
    g = sp.csr_matrix(g)
    n = vals.shape[0]
    W = g.sum()
    moran, geary = [], []
    for x in vals.T:
        z = x - x.mean()
        inum = sum((g.data[g.indptr[i]:g.indptr[i + 1]]
                    * z[g.indices[g.indptr[i]:g.indptr[i + 1]]]).sum() * z[i] for i in range(n))
        tot = sum((g.data[g.indptr[i]:g.indptr[i + 1]]
                   * (x[i] - x[g.indices[g.indptr[i]:g.indptr[i + 1]]]) ** 2).sum() for i in range(n))
        moran.append(n / W * inum / (z * z).sum())
        geary.append((n - 1) * tot / (2 * W * (z * z).sum()))
    return np.array(moran), np.array(geary)


@pytest.mark.parametrize("sparse_vals", [False, True])
def test_moran_geary_match_loop_reference(sparse_vals):
    rng = np.random.default_rng(1)
    g = _graph(60)
    vals = rng.poisson(3, size=(60, 7)).astype(float)
    ref_i, ref_c = _reference(g, vals)
    v = sp.csr_matrix(vals) if sparse_vals else vals
    np.testing.assert_allclose(_svg._moran_i_scores(g, v), ref_i, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(_svg._geary_c_scores(g, v), ref_c, rtol=1e-10, atol=1e-12)


def test_constant_gene_is_nan():
    g = _graph(30)
    vals = np.ones((30, 2))
    vals[:, 1] = np.arange(30)
    assert np.isnan(_svg._moran_i_scores(g, vals)[0])
    assert np.isnan(_svg._geary_c_scores(g, vals)[0])
    assert np.isfinite(_svg._moran_i_scores(g, vals)[1])


class _RowOnly:
    """Stand-in for a disk-backed matrix: only row slices are cheap."""

    def __init__(self, X):
        self._X = sp.csr_matrix(X)
        self.shape = self._X.shape

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise AssertionError("element-wise access")
        return self._X[key]


def test_row_sums_reads_row_blocks():
    X = np.random.default_rng(2).poisson(1, size=(1000, 50)).astype(float)
    np.testing.assert_allclose(_svg._row_sums(_RowOnly(X), chunk_size=128), X.sum(axis=1))
    np.testing.assert_allclose(_svg._row_sums(sp.csr_matrix(X)), X.sum(axis=1))
    np.testing.assert_allclose(_svg._row_sums(X), X.sum(axis=1))
