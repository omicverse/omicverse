r"""Version shims for anndata and zarr.

omicverse supports a range of anndata versions rather than tracking the newest,
so the differences between them are collected here instead of being scattered
through the codebase.

What actually differs
---------------------
Measured on real installs (anndata 0.11.4 + zarr 2.18.7, 0.12.19 + zarr 3.3.0,
0.13.3 + zarr 3.3.0):

================================  ========  =============  =============
behaviour                         0.11      0.12           0.13
================================  ========  =============  =============
``AnnData(..., dtype=)``          warns     warns          **TypeError**
``anndata.read``                  present   **removed**    removed
``/`` in a key when writing       allowed   warns          **ValueError**
``anndata.__version__``           fine      fine           **deprecated**
zarr 3 支持                        no        yes            required
================================  ========  =============  =============

Most of the surface needs no branching at all: casting before construction
satisfies every version, so :func:`as_dtype` has no version check in it. Only
the version lookup and the key sanitiser genuinely have to know.

zarr
----
omicverse touches zarr directly in two places only (an ``isinstance`` check and
one ``zarr.open`` on a tifffile store); everything else goes through anndata,
which absorbs the 2-to-3 difference itself. The floor is therefore anndata's:
``zarr>=2.18.7,!=3.0.*``, with zarr 3 required once anndata reaches 0.13.
"""
from __future__ import annotations

from typing import Any

__all__ = ["anndata_version", "as_dtype", "sanitize_key", "ANNDATA_LT_0_12",
           "ANNDATA_LT_0_13"]


def anndata_version() -> tuple[int, ...]:
    """anndata's version as a tuple.

    Read through :mod:`importlib.metadata`: ``anndata.__version__`` is
    deprecated in 0.13 and warns on access.
    """
    from importlib.metadata import version

    parts = []
    for chunk in version("anndata").split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


_V = anndata_version()
ANNDATA_LT_0_12 = _V < (0, 12)
ANNDATA_LT_0_13 = _V < (0, 13)


def as_dtype(X: Any, dtype: Any) -> Any:
    """Return ``X`` cast to ``dtype``, for anything AnnData accepts as ``X``.

    Replaces ``AnnData(X, dtype=...)``, whose ``dtype`` argument warns from 0.11
    and is gone in 0.13. Casting first is not a workaround for the removal --
    it is what the argument did, and it works on every version, so nothing here
    has to ask which one is installed.

    Handles the three things that reach it: dense arrays, scipy sparse matrices
    (whose ``astype`` is the only route -- ``numpy.asarray`` would densify them)
    and pandas frames.
    """
    if X is None or dtype is None:
        return X
    astype = getattr(X, "astype", None)
    if astype is not None:
        try:
            return astype(dtype)
        except (TypeError, ValueError):
            pass
    import numpy as np

    return np.asarray(X, dtype=dtype)


def sanitize_key(key: str, replacement: str = "_") -> str:
    """Make ``key`` safe to write into an h5ad or zarr store.

    Forward slashes are a path separator in both formats. anndata 0.12 warns on
    them and 0.13 raises ``ValueError``, so any key built from user data -- a
    gene name, a sample id, a file path used as a label -- has to be cleaned
    before it lands in ``uns``/``obsm``/``layers``.
    """
    return key.replace("/", replacement) if isinstance(key, str) else key
