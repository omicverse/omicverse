"""R-compatible ``set.seed()`` / ``sample()`` (Mersenne-Twister, Rejection sampling).

CytoSignal's permutation test draws its shuffles with R's ``sample``. Emulating
R's default RNG (``RNGkind("Mersenne-Twister", "Inversion", "Rejection")``)
lets ``seed=42`` reproduce ``set.seed(42)`` in R bit-for-bit, so the null
distributions -- and therefore the p-values -- match the R package exactly.
"""
from __future__ import annotations

import math

import numpy as np

_I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1)


class RRandom:
    """Stream of R ``unif_rand()`` draws after ``set.seed(seed)``."""

    def __init__(self, seed: int):
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):  # RNG_Init initial scrambling
            s = (69069 * s + 1) & 0xFFFFFFFF
        key = np.empty(625, dtype=np.uint32)
        for j in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            key[j] = s
        # dummy[0] (mti) is reset to 624 by FixupSeeds; mt = dummy[1:].
        self._bg = np.random.MT19937()
        self._bg.state = {"bit_generator": "MT19937", "state": {"key": key[1:].copy(), "pos": 624}}
        self._buf = np.empty(0, dtype=np.float64)
        self._pos = 0

    def _refill(self, n: int) -> None:
        raw = self._bg.random_raw(max(n, 1 << 16)).astype(np.float64) * 2.3283064365386963e-10
        raw[raw <= 0.0] = 0.5 * _I2_32M1
        raw[(1.0 - raw) <= 0.0] = 1.0 - 0.5 * _I2_32M1
        self._buf = np.concatenate([self._buf[self._pos:], raw])
        self._pos = 0

    def unif_rand(self) -> float:
        if self._pos >= self._buf.size:
            self._refill(1)
        v = self._buf[self._pos]
        self._pos += 1
        return float(v)

    def _rbits(self, bits: int) -> int:
        v = 0
        for _ in range(0, bits + 1, 16):
            v = 65536 * v + int(math.floor(self.unif_rand() * 65536))
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        """R ``R_unif_index`` (rejection sampling below the next power of two)."""
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample(self, n: int, size: int | None = None) -> np.ndarray:
        """R ``sample(n, size)`` without replacement, returned **0-based**."""
        size = n if size is None else int(size)
        x = np.arange(n, dtype=np.int64)
        out = np.empty(size, dtype=np.int64)
        m = n
        for i in range(size):
            j = self.unif_index(m)
            out[i] = x[j]
            m -= 1
            x[j] = x[m]
        return out
