"""GPU-accelerated drop-ins for a few ``scipy.stats`` primitives that
omicverse hits in tight loops.

Public surface:

* :func:`rankdata` — batched, descending / ascending ranks with
  average-tie handling, signature-compatible with
  ``scipy.stats.rankdata(..., axis=..., method='average')``. Bit-for-bit
  matches scipy on fp64.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# torch helpers (lazy-imported so the module loads without CUDA installed)
# ---------------------------------------------------------------------------


def _torch_or_raise():
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.utils.gpuex.scipy needs torch installed (CPU torch is fine; "
            "CUDA is auto-detected). `pip install torch` to get it."
        ) from exc


def _pick_device(prefer: Optional[str] = None):
    """Resolve the torch device for the call.

    ``prefer`` can be ``'cuda'``, ``'cpu'``, an explicit ``torch.device``,
    a CUDA index string like ``'cuda:1'``, or ``None`` to auto-detect.
    """
    torch = _torch_or_raise()
    if prefer is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(prefer, torch.device):
        return prefer
    return torch.device(prefer)


# ---------------------------------------------------------------------------
# rankdata
# ---------------------------------------------------------------------------


def rankdata(
    a,
    *,
    axis: int = -1,
    method: str = "average",
    nan_policy: str = "propagate",
    device: Optional[str] = None,
    dtype: Optional[str] = "float64",
) -> "np.ndarray | torch.Tensor":
    """GPU-accelerated, batched ``scipy.stats.rankdata``.

    Returns the *ascending* ranks of every element along ``axis``: rank 1
    is the smallest value, ties get the **average** of their ordinal
    ranks (the upstream default).

    The implementation is searchsorted-based — a single sort + two
    parallel binary searches per row — so it's O(B · N · log N) on GPU
    with no Python-side loop and no branching on per-row tie structure.

    Parameters
    ----------
    a : array-like
        1-D, 2-D, or higher-rank array. Computation is vectorised along
        ``axis``; all other axes are batched independently.
    axis : int, optional
        Axis along which to compute the ranks. Default ``-1``.
    method : {'average'}, optional
        Tie-handling strategy. Only ``'average'`` is supported today —
        for the other modes (``'min'`` / ``'max'`` / ``'dense'`` /
        ``'ordinal'``) just fall back to :func:`scipy.stats.rankdata`.
        Default ``'average'``.
    nan_policy : {'propagate', 'raise'}, optional
        How to handle NaNs in the input. ``'propagate'`` (default)
        produces NaN ranks for NaN inputs — matching scipy. ``'raise'``
        raises a ``ValueError`` if any NaN is present.
    device : str or torch.device, optional
        Where to place the intermediate tensors. ``None`` (default) auto-
        picks CUDA when available, otherwise CPU.
    dtype : {'float64', 'float32', None}, optional
        Output dtype. ``None`` returns a torch.Tensor instead of a numpy
        array (so callers that want to keep the result on the GPU for
        further work avoid a host round-trip). Default ``'float64'`` to
        match scipy exactly.

    Returns
    -------
    ranks : numpy.ndarray (shape == input) when ``dtype`` is a string,
    otherwise a ``torch.Tensor`` on ``device``.

    Notes
    -----
    Bit-for-bit matches ``scipy.stats.rankdata(..., method='average')``
    on fp64 inputs. We verify this in
    ``tests/utils/test_gpuex_scipy_rankdata.py``.

    The ranking convention is **ascending** (smallest value = rank 1),
    the same as scipy. For descending ranks just pass ``-a``.

    Examples
    --------
    >>> import numpy as np
    >>> from omicverse.utils.gpuex.scipy import rankdata
    >>> a = np.array([[5.0, 3.0, 5.0, 3.0, 3.0]])
    >>> rankdata(a, axis=1)
    array([[4.5, 2. , 4.5, 2. , 2. ]])

    Drop-in for AUCell-style per-cell ranking:

    >>> # CPU scipy:
    >>> # ranks = sts.rankdata(-mat, axis=1, method='average')
    >>> # GPU:
    >>> from omicverse.utils.gpuex.scipy import rankdata
    >>> ranks = rankdata(-mat, axis=1)  # numpy out by default
    """
    if method != "average":
        raise NotImplementedError(
            f"method={method!r} not supported on the GPU path yet — fall "
            "back to scipy.stats.rankdata for non-average tie handling."
        )
    if nan_policy not in ("propagate", "raise"):
        raise ValueError(
            f"nan_policy must be 'propagate' or 'raise', got {nan_policy!r}"
        )

    torch = _torch_or_raise()
    a_np = np.asarray(a)
    if a_np.ndim == 0:
        # Scalar fast-path (matches scipy).
        return np.array(1.0)

    if axis < 0:
        axis += a_np.ndim
    # Move the active axis to last, then flatten the leading batch dims to
    # a single dimension so we have a clean (B, N) layout for torch.
    permuted = np.moveaxis(a_np, axis, -1)
    out_shape = permuted.shape
    flat = permuted.reshape(-1, out_shape[-1])

    if nan_policy == "raise" and not np.all(np.isfinite(flat)):
        raise ValueError("rankdata received NaN/inf with nan_policy='raise'")
    has_nan = bool(np.isnan(flat).any()) if nan_policy == "propagate" else False

    dev = _pick_device(device)
    torch_dtype = torch.float64 if dtype != "float32" else torch.float32

    # Negate so descending-rank becomes ascending after sort — we want
    # smaller value = lower rank to match scipy's convention.
    t = torch.from_numpy(flat).to(device=dev, dtype=torch_dtype)
    sorted_vals, _ = torch.sort(t, dim=-1)

    # For each value v in row b:
    #   n_strictly_less(v)   = count of elements < v   = searchsorted(left)
    #   n_less_or_equal(v)   = count of elements <= v  = searchsorted(right)
    # Ordinal ranks of equal elements run from
    #   (n_strictly_less + 1) .. n_less_or_equal
    # so average rank = (n_strictly_less + 1 + n_less_or_equal) / 2.
    left = torch.searchsorted(sorted_vals, t, right=False)
    right_ = torch.searchsorted(sorted_vals, t, right=True)
    ranks = (left.to(torch_dtype) + 1.0 + right_.to(torch_dtype)) * 0.5

    if has_nan:
        # scipy sorts NaN to the end with method='average' and gives them
        # rank N for every NaN (the rank of the LAST tie-group). Match
        # that: any NaN input gets rank N.
        n = ranks.shape[-1]
        nan_mask = torch.isnan(t)
        # NaNs in `t` cause `searchsorted` to return N (insertion at end)
        # — but the formula then gives (N + 1 + N) / 2 = N + 0.5. Fix.
        ranks = torch.where(nan_mask, torch.full_like(ranks, float("nan")), ranks)

    if dtype is None:
        # Caller wants the torch tensor — return without host copy.
        return ranks.reshape(out_shape) if axis == permuted.ndim - 1 else \
            torch.from_numpy(np.moveaxis(ranks.cpu().numpy().reshape(out_shape), -1, axis)).to(dev)

    out_np = ranks.cpu().numpy().reshape(out_shape)
    return np.moveaxis(out_np, -1, axis)
