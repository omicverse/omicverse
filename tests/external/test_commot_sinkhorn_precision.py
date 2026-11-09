"""COMMOT's torch sparse Sinkhorn must reproduce the original NumPy solver.

In float32 the relative error never drops below ``stopthr=1e-8``, so the torch
port ran all ``nitermax`` iterations and stopped ~1e-5 from COMMOT's answer.
"""
import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from omicverse.external.commot._optimal_transport import _unot as N
from omicverse.external.commot._optimal_transport import _unot_torch as U


def _problem(seed=0, n_s=400, n_t=800, cutoff=0.3):
    """Shaped like COMMOT's sub-problems: positive senders/receivers, squared
    normalised distances within the cutoff, receivers summing to 1."""
    rng = np.random.default_rng(seed)
    xs, xt = rng.random((n_s, 2)), rng.random((n_t, 2))
    d = np.sqrt(((xs[:, None] - xt[None]) ** 2).sum(-1))
    i, j = np.nonzero(d < cutoff)
    C = sp.coo_matrix(((d[i, j] / cutoff) ** 2, (i, j)), shape=(n_s, n_t))
    a = rng.gamma(0.5, size=n_s) + 1e-3
    b = rng.gamma(0.5, size=n_t) + 1e-3
    return 0.25 * a / a.sum(), b / b.sum(), C


@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("seed", [0, 1])
def test_sparse_sinkhorn_matches_numpy_commot(seed, use_numba, monkeypatch):
    from omicverse.external.commot._optimal_transport import _sinkhorn_numba

    if use_numba and not _sinkhorn_numba.NUMBA:
        pytest.skip("numba not installed")
    monkeypatch.setattr(_sinkhorn_numba, "NUMBA", use_numba)  # False -> torch float64 path
    a, b, C = _problem(seed)
    ref = sp.coo_matrix(N.unot_sinkhorn_l1_sparse(a, b, C, 0.1, 10.0)).tocsr()
    out = U.unot_torch(a, b, C, 0.1, 10.0, sparse_mtx=True, show_progress=False, device="cpu")
    out = sp.coo_matrix(out).tocsr()
    assert out.dtype == np.float64
    diff = abs(out - ref).max()
    assert diff <= 1e-10 * max(abs(ref).max(), 1e-300)


def test_sparse_sinkhorn_converges_before_nitermax(capsys):
    a, b, C = _problem(3)
    U.unot_torch(a, b, C, 0.1, 10.0, sparse_mtx=True, show_progress=False, device="cpu",
                 nitermax=10000, verbose=True)
    n_iter = int(capsys.readouterr().out.split("Number of iterations in unot:")[1].split()[0])
    assert n_iter < 10000  # float32 stalls at nitermax
