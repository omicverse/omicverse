"""Deterministic random streams that reproduce SecAct's permutation tables.

Three generators feed one Fisher-Yates kernel, so a permutation table here is
bit-identical to the one the reference implementation builds:

* ``"mt19937"`` -- GSL ``gsl_rng_mt19937`` with seed 0 (GSL maps 0 to 4357).
  This is what R SecAct (``.gsl_mt19937_perm_table``, RidgeFast, RidgeCuda)
  uses today.
* ``"srand"`` -- glibc ``srand(seed)`` / ``rand()`` (TYPE_3 additive feedback
  generator), the default of SecActpy. Implemented here in pure arithmetic, so
  it is platform independent rather than tied to the host libc.
* ``"numpy"`` -- NumPy permutations; fast, but matches neither reference.

The shuffle is SecAct's C ``shuffle()``::

    for (i = 0; i < n - 1; i++) {
        j = i + rand() / (RAND_MAX / (n - i) + 1);
        swap(a[i], a[j]);
    }

applied *cumulatively*: permutation ``k`` shuffles the result of ``k - 1``.

``r_sample_replace`` reproduces R's ``set.seed(seed); sample(x, size, TRUE)``
(Mersenne-Twister, ``sample.kind = "Rejection"``), which SecAct uses for the
spatial background in ``SecAct.CCC.scST``.
"""
from __future__ import annotations

import numpy as np

try:  # numba is an omicverse dependency; keep a pure-Python fallback anyway.
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

_N, _M = 624, 397
_UPPER, _LOWER, _MATRIX_A = 0x80000000, 0x7FFFFFFF, 0x9908B0DF
_U32 = 0xFFFFFFFF
_GLIBC_RAND_MAX = 2147483647


@njit(cache=True)
def _mt_seed_gsl(seed):
    mt = np.empty(_N, dtype=np.int64)
    if seed == 0:
        seed = 4357
    mt[0] = seed & _U32
    for i in range(1, _N):
        prev = mt[i - 1]
        mt[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & _U32
    return mt


@njit(cache=True)
def _mt_twist(mt):
    for kk in range(_N - _M):
        y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
        mt[kk] = mt[kk + _M] ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)
    for kk in range(_N - _M, _N - 1):
        y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
        mt[kk] = mt[kk + _M - _N] ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)
    y = (mt[_N - 1] & _UPPER) | (mt[0] & _LOWER)
    mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)


@njit(cache=True)
def _mt_temper(y):
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    return y & _U32


@njit(cache=True)
def _perm_table_mt19937(n, nrand, seed):
    mt = _mt_seed_gsl(seed)
    mti = _N
    table = np.empty((nrand, n), dtype=np.int64)
    arr = np.arange(n, dtype=np.int64)
    for p in range(nrand):
        for i in range(n - 1):
            if mti >= _N:
                _mt_twist(mt)
                mti = 0
            r = _mt_temper(mt[mti])
            mti += 1
            j = i + r // (_U32 // (n - i) + 1)
            tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp
        table[p, :] = arr
    return table


@njit(cache=True)
def _perm_table_glibc(n, nrand, seed):
    # glibc random_r TYPE_3: r[i] = r[i-3] + r[i-31]; output r >> 1; 310 discards.
    seed = seed & _U32
    if seed == 0:
        seed = 1
    r = np.empty(34, dtype=np.int64)
    r[0] = seed
    for i in range(1, 31):
        # 16807 * r[i-1] mod (2^31 - 1), Schrage's method on a signed 32-bit word.
        word = r[i - 1]
        if word >= 2147483648:
            word -= 4294967296
        hi = word // 127773
        lo = word % 127773
        if word < 0 and lo != 0:  # C truncating division
            hi += 1
            lo -= 127773
        word = 16807 * lo - 2836 * hi
        if word < 0:
            word += 2147483647
        r[i] = word & _U32
    state = np.empty(31, dtype=np.int64)
    for i in range(31):
        state[i] = r[i]
    # r[31..33] = r[0..2]; then generate, discarding the first 310 outputs.
    f, b = 3, 0  # front/rear pointers of the additive generator
    table = np.empty((nrand, n), dtype=np.int64)
    arr = np.arange(n, dtype=np.int64)
    for _ in range(310):
        state[f] = (state[f] + state[b]) & _U32
        f = (f + 1) % 31
        b = (b + 1) % 31
    for p in range(nrand):
        for i in range(n - 1):
            state[f] = (state[f] + state[b]) & _U32
            out = state[f] >> 1
            f = (f + 1) % 31
            b = (b + 1) % 31
            j = i + out // (_GLIBC_RAND_MAX // (n - i) + 1)
            tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp
        table[p, :] = arr
    return table


def permutation_table(n: int, nrand: int, rng_method: str = "mt19937", seed: int = 0) -> np.ndarray:
    """``(nrand, n)`` table of cumulative Fisher-Yates permutations (0-based).

    Row ``k`` is the permutation applied to the response rows in permutation
    ``k``: ``Y[table[k]]`` (R: ``Y[perm_table[i, ] + 1L, ]``).
    """
    n, nrand = int(n), int(nrand)
    if n < 1 or nrand < 0:
        raise ValueError("n must be >= 1 and nrand >= 0.")
    method = str(rng_method).lower()
    if method in ("mt19937", "gsl"):
        return _perm_table_mt19937(n, nrand, int(seed))
    if method == "srand":
        return _perm_table_glibc(n, nrand, int(seed))
    if method == "numpy":
        rng = np.random.default_rng(seed)
        return np.stack([rng.permutation(n) for _ in range(nrand)]) if nrand else np.empty((0, n), np.int64)
    raise ValueError(f"Unknown rng_method {rng_method!r}; use 'mt19937', 'srand' or 'numpy'.")


# ---------------------------------------------------------------------------
# R: set.seed(seed); sample(x, size, replace = TRUE)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _r_mt_state(seed):
    s = seed & _U32
    for _ in range(50):
        s = (69069 * s + 1) & _U32
    seeds = np.empty(625, dtype=np.int64)
    for j in range(625):
        s = (69069 * s + 1) & _U32
        seeds[j] = s
    mt = np.empty(_N, dtype=np.int64)
    for j in range(_N):
        mt[j] = seeds[j + 1]  # dummy[0] is mti (fixed up to 624)
    return mt


@njit(cache=True)
def _r_sample_draws(seed, ns, size):
    """Two consecutive ``sample.int(ns[k], size, replace=TRUE)`` calls (0-based)."""
    mt = _r_mt_state(seed)
    mti = _N
    i2_32m1 = 2.328306437080797e-10
    out = np.empty((len(ns), size), dtype=np.int64)
    for c in range(len(ns)):
        dn = ns[c]
        bits = int(np.ceil(np.log2(dn))) if dn > 1 else 0
        mask = (1 << bits) - 1
        for k in range(size):
            while True:
                v = 0
                nb = 0
                while nb <= bits:
                    if mti >= _N:
                        _mt_twist(mt)
                        mti = 0
                    u = _mt_temper(mt[mti]) * 2.3283064365386963e-10
                    mti += 1
                    if u <= 0.0:
                        u = 0.5 * i2_32m1
                    elif 1.0 - u <= 0.0:
                        u = 1.0 - 0.5 * i2_32m1
                    v = 65536 * v + int(np.floor(u * 65536))
                    nb += 16
                dv = v & mask
                if dv < dn:
                    break
            out[c, k] = dv
    return out


def r_sample_replace(populations, size: int, seed: int = 123) -> list[np.ndarray]:
    """R ``set.seed(seed); lapply(populations, sample, size, replace = TRUE)``.

    ``populations`` is a sequence of 1-D integer arrays, sampled in order from
    one RNG stream exactly like consecutive ``sample()`` calls in R >= 3.6
    (Mersenne-Twister, rejection sampling). Returns the sampled *elements*.
    A length-1 population returns that element (R's ``sample`` would instead
    sample ``1:x`` -- a documented R quirk that is deliberately not copied).
    """
    pops = [np.asarray(p) for p in populations]
    ns = np.asarray([len(p) for p in pops], dtype=np.int64)
    if (ns < 1).any():
        raise ValueError("Every population needs at least one element.")
    idx = _r_sample_draws(int(seed), ns, int(size))
    return [p[idx[c]] for c, p in enumerate(pops)]
