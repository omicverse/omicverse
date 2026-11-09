"""Spectral init must avoid ARPACK's int32 overflow on ~1.66M-2M cells (#990)."""
import numpy as np
import pytest
import scipy.sparse as sp

umap_spectral = pytest.importorskip("umap.spectral")

from omicverse.pp import _umap


def test_overflow_threshold_matches_arpack_workspace():
    # umap-learn uses ncv = int(sqrt(n)); n * ncv crosses 2**31 - 1 near 1.66M.
    assert not _umap._arpack_overflows(1_600_000, 2)
    assert _umap._arpack_overflows(1_700_000, 2)
    assert _umap._arpack_overflows(1_829_054, 2)  # dataset in #990


class _FakeGraph:
    def __init__(self, n):
        self.shape = (n, n)


@pytest.mark.parametrize("n, expected", [(1_000, None), (1_829_054, "lobpcg")])
def test_context_forces_lobpcg_only_when_needed(monkeypatch, n, expected):
    seen = {}

    def fake(data, graph, dim, random_state, metric="euclidean", metric_kwds={},
             init="random", method=None, tol=0.0, maxiter=0):
        seen["method"] = method
        return np.zeros((graph.shape[0], dim))

    monkeypatch.setattr(umap_spectral, "_spectral_layout", fake)
    with _umap._lobpcg_for_large_spectral_init():
        umap_spectral._spectral_layout(data=None, graph=_FakeGraph(n), dim=2,
                                       random_state=np.random.RandomState(0))
    assert seen["method"] == expected
    assert umap_spectral._spectral_layout is fake


def test_explicit_method_is_respected(monkeypatch):
    seen = {}

    def fake(data, graph, dim, random_state, metric="euclidean", metric_kwds={},
             init="random", method=None, tol=0.0, maxiter=0):
        seen["method"] = method

    monkeypatch.setattr(umap_spectral, "_spectral_layout", fake)
    with _umap._lobpcg_for_large_spectral_init():
        umap_spectral._spectral_layout(None, _FakeGraph(1_829_054), 2, None, method="eigsh")
    assert seen["method"] == "eigsh"


def test_small_graph_embedding_unchanged():
    # Below the threshold the wrapper must be a no-op on the real solver.
    rng = np.random.default_rng(0)
    n = 300
    rows = np.repeat(np.arange(n), 5)
    g = sp.csr_matrix((rng.uniform(.1, 1, n * 5), (rows, (rows + rng.integers(1, n, n * 5)) % n)),
                      shape=(n, n))
    g = g.maximum(g.T).tocoo()
    ref = umap_spectral.spectral_layout(None, g, 2, np.random.RandomState(0))
    with _umap._lobpcg_for_large_spectral_init():
        got = umap_spectral.spectral_layout(None, g, 2, np.random.RandomState(0))
    np.testing.assert_allclose(got, ref)
