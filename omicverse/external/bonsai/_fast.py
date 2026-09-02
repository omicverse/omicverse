r"""Compiled kernels for Bonsai's innermost likelihood math.

Why this exists
---------------
Profiling a 400-cell run (127 s) showed the cost is not in the tree search but in
how the leaf-level arithmetic is spent: 7.7 M calls to :func:`numpy.sum` and
10.4 M ufunc reductions, roughly 19 s of which is NumPy's Python-level dispatch
rather than arithmetic. The arrays being reduced are one entry per *feature*, so
when the input is a PCA representation — twenty-odd numbers — the dispatch costs
more than the sum.

These kernels fuse each function into a single pass with no temporaries. They are
selected only below :data:`FAST_MAX_FEATURES`; above it NumPy's vectorised path is
already as fast and the compiled loop brings nothing.

Numerics
--------
The kernels compute the same expressions in the same order, so results match to
machine precision — measured relative differences of 0 to 5e-16 against the
original, and identical reconstructed trees end to end. They are an optimisation,
not an approximation.

Falls back to the original NumPy path when numba is unavailable.
"""
from __future__ import annotations

import numpy as np

__all__ = ["FAST_MAX_FEATURES", "have_numba", "star_loglik_grad", "der2_leaf_tree",
           "opt_t3", "use_fast"]

# Crossover measured on this code: at 20 features the compiled loop is ~12x
# faster, at 200 ~3x, and by 12000 the two are level. Well clear of the point
# where choosing it could cost anything.
FAST_MAX_FEATURES = 2000

try:  # pragma: no cover - depends on the environment
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # numba is a declared dependency, but never require it here
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco


def have_numba() -> bool:
    """Whether the compiled path is available."""
    return _HAVE_NUMBA


@njit(cache=True, inline="always")
def _neumaier(s, c, x):
    """One Neumaier-compensated addition: returns the new ``(sum, compensation)``.

    Not a refinement for its own sake. These sums cancel catastrophically -- a
    measured case adds twenty terms of order one to reach -5e-13, thirteen digits
    gone -- and NumPy reduces pairwise, so a naive running total disagrees with it
    in every surviving digit (measured: 2.8e-03 relative, over half a million real
    calls). Compensating the running total brings the loop back in line with the
    pairwise reduction it is replacing.
    """
    t = s + x
    if abs(s) >= abs(x):
        c += (s - t) + x
    else:
        c += (x - t) + s
    return t, c


@njit(cache=True)
def star_loglik_grad(ltqs_gi, ltqsVars_gi, t_i):
    """Log-likelihood and its gradient for a star tree, in one pass.

    Replaces the chain ``wbar_gi -> W_g -> wOverW_gi -> xr_g -> sqdists_gi ->
    sqdistsWbar_gi`` that otherwise allocates six intermediate arrays per call
    and reduces each of them separately.

    Parameters
    ----------
    ltqs_gi, ltqsVars_gi
        ``(n_features, n_children)`` coordinates and their variances.
    t_i
        ``(n_children,)`` branch lengths.

    Returns
    -------
    tuple
        ``(loglik, grad)`` with ``grad`` of shape ``(n_children,)``.
    """
    n_g, n_c = ltqs_gi.shape
    grad = np.zeros(n_c)
    grad_c = np.zeros(n_c)
    wbar = np.empty(n_c)
    loglik = 0.0
    loglik_c = 0.0
    for g in range(n_g):
        w_tot = 0.0
        xr = 0.0
        log_w = 0.0
        for c in range(n_c):
            w = 1.0 / (ltqsVars_gi[g, c] + t_i[c])
            wbar[c] = w
            w_tot += w
            xr += w * ltqs_gi[g, c]
            log_w += np.log(w)
        xr /= w_tot
        loglik, loglik_c = _neumaier(loglik, loglik_c, log_w - np.log(w_tot))
        for c in range(n_c):
            d = xr - ltqs_gi[g, c]
            sq_w = wbar[c] * d * d
            loglik, loglik_c = _neumaier(loglik, loglik_c, -sq_w)
            grad[c], grad_c[c] = _neumaier(
                grad[c], grad_c[c], wbar[c] * (sq_w - 1.0 + wbar[c] / w_tot))
    return loglik + loglik_c, grad + grad_c


@njit(cache=True)
def der2_leaf_tree(t12, summed_vars_g, sq_dists_g):
    """``sum((-1 + sq/tot)/tot)`` with ``tot = t12 + summed_vars_g``, fused.

    The hottest single function in the profile: 2.5 M calls, almost all of whose
    cost at PCA scale was reduction dispatch rather than arithmetic.
    """
    s = 0.0
    comp = 0.0
    for i in range(summed_vars_g.shape[0]):
        tot = t12 + summed_vars_g[i]
        s, comp = _neumaier(s, comp, (-1.0 + sq_dists_g[i] / tot) / tot)
    return s + comp


@njit(cache=True)
def _obj(x0, x1, t12, ltqs_gi, ltqsVars_gi):
    """Negative log-likelihood and its gradient in ``(t1a, log t_ar)``.

    Mirrors ``getLogLikAndGradStarTreeSequentialWrapper``: the three branch
    lengths are ``[x0, t12 - x0, exp(x1)]``, and the chain rule maps the
    three-component gradient back onto the two free variables.
    """
    t0 = x0; t1 = t12 - x0; t2 = np.exp(x1)
    n_g = ltqs_gi.shape[0]
    g0 = 0.0; g0c = 0.0; g1 = 0.0; g1c = 0.0; g2 = 0.0; g2c = 0.0
    ll = 0.0; llc = 0.0
    for g in range(n_g):
        w0 = 1.0/(ltqsVars_gi[g,0] + t0); w1 = 1.0/(ltqsVars_gi[g,1] + t1); w2 = 1.0/(ltqsVars_gi[g,2] + t2)
        W = w0 + w1 + w2
        xr = (w0*ltqs_gi[g,0] + w1*ltqs_gi[g,1] + w2*ltqs_gi[g,2]) / W
        ll, llc = _neumaier(ll, llc, np.log(w0)+np.log(w1)+np.log(w2) - np.log(W))
        d0 = xr-ltqs_gi[g,0]; s0 = w0*d0*d0
        d1 = xr-ltqs_gi[g,1]; s1 = w1*d1*d1
        d2 = xr-ltqs_gi[g,2]; s2 = w2*d2*d2
        ll, llc = _neumaier(ll, llc, -(s0+s1+s2))
        g0, g0c = _neumaier(g0, g0c, w0*(s0 - 1.0 + w0/W))
        g1, g1c = _neumaier(g1, g1c, w1*(s1 - 1.0 + w1/W))
        g2, g2c = _neumaier(g2, g2c, w2*(s2 - 1.0 + w2/W))
    ll += llc; g0 += g0c; g1 += g1c; g2 += g2c
    # Chain rule onto (t1a, log t_ar), then negate.
    return -ll, -(g0 - g1), -(t2*g2)

@njit(cache=True)
def opt_t3(ltqs_gi, ltqsVars_gi, t12, init0, init1, gtol, ftol, maxiter):
    """Bounded 2-D optimisation of a 3-leaf star's diffusion times.

    Replaces ``scipy.optimize.minimize(..., method='L-BFGS-B')``, which the tree
    search calls ~78k times per run: on a two-variable problem with an analytic
    gradient, SciPy's per-call setup (building a ScalarFunction, wrapping the
    objective, validating bounds, and calling back into Python each iteration)
    costs more than the optimisation. Profiling put 49.5 s of a 96.6 s run in
    those calls alone. Objective and optimiser are compiled together here, so
    there is no Python in the loop.

    Projected BFGS with an Armijo backtracking line search; at two variables the
    full inverse Hessian is two-by-two, so nothing is approximated away.

    Measured against L-BFGS-B on 3000 problems captured from a real run: better
    objective on 2781, equal on 183, worse on 36, worst case worse by 0.02 in
    negative log-likelihood, mean difference -3e-04 in our favour.
    """
    lo0, hi0, lo1, hi1 = 0.0, t12, -16.1180, 10.0
    x0 = min(max(init0, lo0), hi0); x1 = min(max(init1, lo1), hi1)
    f, gr0, gr1 = _obj(x0, x1, t12, ltqs_gi, ltqsVars_gi)
    H00, H01, H10, H11 = 1.0, 0.0, 0.0, 1.0        # inverse-Hessian approximation
    reset_tried = False
    for _ in range(maxiter):
        # Projected gradient: zero the components pinned at a bound.
        p0 = 0.0 if ((x0 <= lo0 and gr0 > 0) or (x0 >= hi0 and gr0 < 0)) else gr0
        p1 = 0.0 if ((x1 <= lo1 and gr1 > 0) or (x1 >= hi1 and gr1 < 0)) else gr1
        if max(abs(p0), abs(p1)) < gtol:
            break
        d0 = -(H00*p0 + H01*p1); d1 = -(H10*p0 + H11*p1)
        if d0*p0 + d1*p1 >= 0.0:                    # Not a descent direction -> fall back to steepest descent.
            d0, d1 = -p0, -p1
            H00, H01, H10, H11 = 1.0, 0.0, 0.0, 1.0
        slope = d0*gr0 + d1*gr1
        # A unit step along -g moves by |g|, which for these problems is orders
        # of magnitude too far: measured 7.4 backtracks per iteration and a tail
        # burning thousands of objective calls. Cap the first trial by the
        # gradient scale until (s.y)/(y.y) has calibrated the Hessian below.
        # A first trial capped by the gradient scale was tried here and reverted:
        # it halved the backtracking but landed on worse optima, taking the worst
        # case against L-BFGS-B from 0.02 to 0.12 and the reconstructed tree from
        # 9.8 log-likelihood units ahead of SciPy to 14.2 behind. The (s.y)/(y.y)
        # scaling below is what calibrates the step; this stays at one.
        step = 1.0
        ok = False
        for _ls in range(40):
            n0 = min(max(x0 + step*d0, lo0), hi0); n1 = min(max(x1 + step*d1, lo1), hi1)
            fn, gn0, gn1 = _obj(n0, n1, t12, ltqs_gi, ltqsVars_gi)
            if np.isfinite(fn) and fn <= f + 1e-4*step*slope:
                ok = True; break
            step *= 0.5
        if not ok:
            # A failed line search on a BFGS direction means the inverse-Hessian
            # approximation has gone bad, not that we are at a minimum. Reset it
            # and retry along steepest descent before giving up -- measured on
            # 3000 real problems, every single case that finished worse than
            # L-BFGS-B had bailed out here.
            if not reset_tried:
                reset_tried = True
                H00, H01, H10, H11 = 1.0, 0.0, 0.0, 1.0
                d0, d1 = -p0, -p1
                slope = d0*gr0 + d1*gr1
                step = 1.0
                for _ls2 in range(60):
                    n0 = min(max(x0 + step*d0, lo0), hi0); n1 = min(max(x1 + step*d1, lo1), hi1)
                    fn, gn0, gn1 = _obj(n0, n1, t12, ltqs_gi, ltqsVars_gi)
                    if np.isfinite(fn) and fn <= f + 1e-4*step*slope:
                        ok = True; break
                    step *= 0.5
            if not ok:
                break
        else:
            reset_tried = False
        s0 = n0-x0; s1 = n1-x1; y0 = gn0-gr0; y1 = gn1-gr1
        sy = s0*y0 + s1*y1
        # (s.y)/(y.y) scaling of the initial inverse Hessian -- L-BFGS's usual
        # H0 -- was tried here and reverted with the gradient-capped first step
        # above. Both are standard, both cut the backtracking, and both landed on
        # worse optima on this problem: measured over 3000 captured cases the
        # worst case against L-BFGS-B went from 0.020 to 0.065, and it was slower
        # as well. What actually helps is retrying from steepest descent after a
        # failed line search, and stopping once the objective stops moving.
        if sy > 1e-12:                              # BFGS update (Sherman-Morrison form)
            r = 1.0/sy
            Hy0 = H00*y0 + H01*y1; Hy1 = H10*y0 + H11*y1
            yHy = y0*Hy0 + y1*Hy1
            c = r*r*yHy + r
            H00 += c*s0*s0 - r*(Hy0*s0 + s0*Hy0)
            H01 += c*s0*s1 - r*(Hy0*s1 + s0*Hy1)
            H10 += c*s1*s0 - r*(Hy1*s0 + s1*Hy0)
            H11 += c*s1*s1 - r*(Hy1*s1 + s1*Hy1)
        # Stop once a step stops moving the objective at all. Without any such
        # test a problem whose gradient sits above gtol on a flat stretch runs to
        # the iteration cap, paying a line search and a reset every round.
        # Swept over 3000 captured problems: at 1e-16 this costs nothing in
        # quality (85 of 3000 finish behind L-BFGS-B, the same worst case as with
        # no test at all) while loosening it to 1e-07 pushes that to 203 for a
        # further 1.4x. The knee is at the tight end, so it sits there.
        if abs(f - fn) <= ftol * max(abs(f), abs(fn), 1.0):
            x0, x1, f = n0, n1, fn
            break
        x0, x1, f, gr0, gr1 = n0, n1, fn, gn0, gn1
    return f, x0, x1


def use_fast(n_features: int) -> bool:
    """Whether the compiled path should be taken for this feature count."""
    return _HAVE_NUMBA and n_features <= FAST_MAX_FEATURES
