"""Fused CPU kernel for COMMOT's sparse l1-unbalanced Sinkhorn (``unot_sinkhorn_l1_sparse``).

acceleration: playbook §4.4 (numba-jitted inner loop) + §4.1 (fusion), class (E).
Each half-step evaluates exactly COMMOT's update

    f <- eps*log(a) - eps*log(sum_j exp((-C_ij + f_i + g_j)/eps) + exp((-m + f_i)/eps)) + f
    g <- eps*log(b) - eps*log(sum_i exp((-C_ij + f_i + g_j)/eps) + exp((-m + g_j)/eps)) + g

but walks the cost matrix row by row (CSR) for ``f`` and column by column (CSC)
for ``g``: no gathered/exponentiated temporaries of size nnz and no scatter-add,
and rows / columns run in parallel. Only the order of the per-row/column sums
differs from the torch/NumPy versions (results agree to ~1e-14 at convergence).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

try:
    from numba import njit, prange
    NUMBA = True
except ImportError:  # pragma: no cover - numba is an omicverse dependency
    NUMBA = False


if NUMBA:
    @njit(parallel=True, cache=True)
    def _half_step(indptr, indices, data, x, y, logw, eps, m, out):
        # x: potential being updated (rows of this layout), y: the other potential.
        for r in prange(indptr.size - 1):
            s = 0.0
            xr = x[r]
            for k in range(indptr[r], indptr[r + 1]):
                s += np.exp((-data[k] + xr + y[indices[k]]) / eps)
            out[r] = eps * logw[r] - eps * np.log(s + np.exp((-m + xr) / eps)) + xr

    @njit(cache=True)
    def _rel_change(new, old):
        num, den = 0.0, 1.0
        for i in range(new.size):
            d = abs(new[i] - old[i])
            if d > num:
                num = d
            if abs(new[i]) > den:
                den = abs(new[i])
            if abs(old[i]) > den:
                den = abs(old[i])
        return num / den


def sinkhorn_l1_sparse(a, b, C, eps, m, nitermax=10000, stopthr=1e-8, verbose=False):
    """COMMOT ``unot_sinkhorn_l1_sparse`` (float64); returns the plan as COO."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    C = sparse.coo_matrix(C)
    csr = sparse.csr_matrix((C.data.astype(np.float64), (C.row, C.col)), shape=C.shape)
    csc = csr.tocsc()
    csr.sort_indices()
    csc.sort_indices()
    # Same layout as the input COO when it has duplicates summed (COMMOT's cost matrices have none).
    loga, logb = np.log(a), np.log(b)
    f, g = np.zeros_like(a), np.zeros_like(b)
    f_new, g_new = np.empty_like(a), np.empty_like(b)
    niter, err = 0, 100.0
    while niter <= nitermax and err > stopthr:
        _half_step(csr.indptr, csr.indices, csr.data, f, g, loga, float(eps), float(m), f_new)
        _half_step(csc.indptr, csc.indices, csc.data, g, f_new, logb, float(eps), float(m), g_new)
        if niter % 10 == 0:
            err = 0.5 * (_rel_change(f_new, f) + _rel_change(g_new, g))
        f, f_new = f_new, f
        g, g_new = g_new, g
        niter += 1
    if verbose:
        print('Number of iterations in unot:', niter)
    rows = np.repeat(np.arange(csr.shape[0]), np.diff(csr.indptr))
    vals = np.exp((-csr.data + f[rows] + g[csr.indices]) / eps)
    return sparse.coo_matrix((vals, (rows, csr.indices)), shape=C.shape)
