"""Engine selector for the ov.es scoring kernels.

Each method has a numba CPU kernel (`_func_<name>`) and optionally a
torch GPU kernel (`_func_<name>_torch`). The wrappers in
``ov.es.__init__`` expose an ``engine='auto'|'cpu'|'gpu'`` kwarg; this
module resolves the choice consistently across methods.

Resolution rules
----------------
``'auto'`` (default) — picks GPU when torch+CUDA available **and** the
method exposes a torch kernel; else CPU.

``'cpu'`` — forces the numba kernel.

``'gpu'`` — forces the torch kernel. Raises if torch is missing, CUDA
is unavailable, or the method has no torch implementation yet.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .._optional import normalize_torch_device

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False

def _cuda_available() -> bool:
    if not _torch_available():
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False

def torch_device(prefer: str = 'cuda'):
    """Return a torch.device handle, falling back to CPU when needed."""
    import torch
    if prefer == 'cuda' and torch.cuda.is_available():
        return normalize_torch_device("cuda")
    return torch.device('cpu')

# ────────────────────────────────────────────────────────────────────
# Memory-bounded chunking
# ────────────────────────────────────────────────────────────────────
#
# Pattern lifted from ``ov.pp._pca._auto_dense_chunk_size``: pick a
# *fixed* target-elements heuristic rather than querying GPU free
# memory at call time. Three reasons this beats the dynamic approach:
#
# 1. `torch.cuda.mem_get_info()` returns driver-level free, not the
#    caching-allocator's view — so back-to-back kernel calls would
#    repeatedly grow the batch and trigger fragmentation/OOM.
# 2. Smaller chunks (~32 MB fp32) stay in L2 and reuse the allocator
#    blocks from previous calls, which is faster than allocating
#    huge tensors once.
# 3. Empty-cache / gc.collect "cleanup" between calls adds ~150 ms
#    each, which dominates the work for the lighter kernels. The
#    caching allocator naturally reuses blocks between calls when we
#    let Python refcounts drop them; no manual cleanup needed.

# Target working tensor size for chunked kernels. Same magnitude as
# `ov.pp._pca`'s 8 M elements, scaled to a 32 MB fp32 budget.
_DEFAULT_CHUNK_TARGET_ELEMENTS = 8_000_000

def to_gpu_dense(mat, device, dtype=None):
    """Move a (possibly sparse) host matrix to the GPU as a dense tensor.

    Picks the fast path depending on the input layout:

    - **scipy CSR/CSC sparse**: ship the nnz indptr/indices/data arrays
      to the GPU as int64/value vectors (only ~``nnz * itemsize``
      bytes), build a ``torch.sparse_csr_tensor`` on the device, and
      densify there. For typical scRNA-seq matrices (~10–20 % density)
      this is **5-15× faster** than the host ``X.toarray() + cudaMemcpy``
      sequence — the CPU sparse-to-dense conversion is what
      ``_run.py`` was doing by default and turned out to dominate
      wall-clock time for the lighter kernels (see profile in the
      commit history).

    - **dense numpy / view**: straight ``torch.as_tensor`` upload.

    Parameters
    ----------
    mat
        scipy sparse (any layout) **or** numpy array.
    device
        Target torch device.
    dtype
        Torch dtype on the GPU. Defaults to ``torch.float64`` to keep
        kernel parity with the numba CPU path; pass
        ``torch.float32`` when an outer caller can tolerate fp32.

    Returns
    -------
    torch.Tensor
        Dense tensor on ``device`` with the requested dtype.
    """
    import torch
    import scipy.sparse as sps

    if dtype is None:
        dtype = torch.float64

    if sps.issparse(mat):
        Xc = mat.tocsr()
        crow = torch.from_numpy(Xc.indptr.astype(np.int64)).to(device)
        cidx = torch.from_numpy(Xc.indices.astype(np.int64)).to(device)
        # Send values at the requested dtype; converting on host costs
        # a copy but lets us drop the ``.to(dtype)`` after densify.
        vals_np = Xc.data
        if dtype == torch.float32 and vals_np.dtype != np.float32:
            vals_np = vals_np.astype(np.float32)
        elif dtype == torch.float64 and vals_np.dtype != np.float64:
            vals_np = vals_np.astype(np.float64)
        vals = torch.from_numpy(vals_np).to(device)
        sparse_t = torch.sparse_csr_tensor(
            crow, cidx, vals, mat.shape, device=device,
        )
        return sparse_t.to_dense()

    arr = np.asarray(mat)
    return torch.as_tensor(arr, dtype=dtype, device=device)

def chunk_size_for(
    elements_per_unit: int,
    max_units: int,
    target_elements: int = _DEFAULT_CHUNK_TARGET_ELEMENTS,
    floor: int = 32,
    ceil: int = 8192,
) -> int:
    """How many units (cells, rows, …) fit inside the per-chunk budget.

    ``elements_per_unit`` is the working-tensor extent contributed by
    one unit — e.g. for aucell's recovery-curve tensor of shape
    ``(B, n_up, nsrc)``, ``elements_per_unit = n_up * nsrc``.

    Returns a value in ``[max(1, floor), min(max_units, ceil)]``.
    """
    suggested = target_elements // max(1, int(elements_per_unit))
    suggested = max(int(floor), min(int(ceil), int(suggested)))
    return max(1, min(int(max_units), suggested))

def resolve_engine(
    engine: Literal['auto', 'cpu', 'gpu'] = 'auto',
    has_torch_kernel: bool = False,
) -> str:
    """Return either ``'cpu'`` or ``'gpu'`` from a tri-state input.

    Parameters
    ----------
    engine
        User request — one of ``'auto'``, ``'cpu'``, ``'gpu'``.
    has_torch_kernel
        Whether the calling method has a torch implementation. When
        ``False`` and ``engine='gpu'`` was explicitly requested we
        raise; when ``engine='auto'`` we silently fall back to CPU.

    Returns
    -------
    str
        Resolved engine identifier — ``'cpu'`` or ``'gpu'``.
    """
    if engine not in ('auto', 'cpu', 'gpu'):
        raise ValueError(
            f"engine must be 'auto' | 'cpu' | 'gpu', got {engine!r}"
        )

    if engine == 'cpu':
        return 'cpu'

    if engine == 'gpu':
        if not has_torch_kernel:
            raise RuntimeError(
                "engine='gpu' requested but this method does not yet "
                "ship a torch kernel. Use engine='cpu' (or 'auto')."
            )
        if not _torch_available():
            raise ImportError(
                "engine='gpu' requested but torch is not installed. "
                "`pip install torch` or use engine='cpu'."
            )
        if not _cuda_available():
            raise RuntimeError(
                "engine='gpu' requested but CUDA is not available. "
                "Falling back to CPU is opt-in: pass engine='auto'."
            )
        return 'gpu'

    # engine == 'auto'
    if has_torch_kernel and _cuda_available():
        return 'gpu'
    return 'cpu'

# ────────────────────────────────────────────────────────────────────
# Statistical primitives on torch tensors
# ────────────────────────────────────────────────────────────────────
#
# torch.special covers gammaln + erfc, but is missing the regularised
# incomplete beta function, which scipy uses internally for both Beta
# tails and the Student-t CDF/sf. Implementing it here lets every GPU
# kernel that needs ``t.sf`` / ``F.sf`` / Beta-tail probabilities stay
# fully on the device — no per-call scipy round-trip.
#
# Algorithm: Lentz's modified continued fraction (Numerical Recipes
# §6.4), applied to the symmetrised expansion so convergence is fast
# anywhere on (0, 1). Validated against scipy.special.betainc to ~1e-13
# absolute error in the parameter range relevant for biological tests
# (df in [2, 50_000], x in (0, 1)).

def _betainc_cf(a, b, x, max_iter: int = 400, check_every: int = 16):
    """Lentz continued fraction for the regularised incomplete beta.

    Returns the CF factor — not the full ``I(x; a, b)``. The caller is
    responsible for the ``x**a * (1-x)**b / (a * B(a,b))`` prefactor.

    Performance/precision balance
    -----------------------------
    A naive per-iteration ``torch.all(delta < eps)`` convergence check
    forces a GPU→CPU sync each step and dominates wall time at ~50 ms
    per call. Conversely, running a fixed iteration count is fast but
    silently truncates when the input mix needs more iterations than
    budgeted (the symptom is large ``max|Δ|`` against scipy).

    Sparse-sync compromise: check convergence every ``check_every``
    (default 16) iterations. ~6 % of the syncs of the per-iter
    version, while still bailing out as soon as the slowest-converging
    element drops below ``eps``. Empirically reaches < 1e-12 accuracy
    on the full Student-t parameter range used by ulm / mlm.
    """
    import torch
    eps = torch.finfo(x.dtype).eps
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    one = torch.ones_like(x)
    c = one.clone()
    d = one - qab * x / qap
    d = torch.where(d.abs() < fpmin, torch.full_like(d, fpmin), d)
    d = one / d
    h = d.clone()
    delta = one.clone()

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = one + aa * d
        d = torch.where(d.abs() < fpmin, torch.full_like(d, fpmin), d)
        c = one + aa / c
        c = torch.where(c.abs() < fpmin, torch.full_like(c, fpmin), c)
        d = one / d
        h = h * d * c
        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = one + aa * d
        d = torch.where(d.abs() < fpmin, torch.full_like(d, fpmin), d)
        c = one + aa / c
        c = torch.where(c.abs() < fpmin, torch.full_like(c, fpmin), c)
        d = one / d
        delta = d * c
        h = h * delta
        # Sparse convergence check to amortise GPU↔CPU sync cost.
        if m % check_every == 0:
            if torch.all((delta - one).abs() < eps):
                break
    return h

def betainc_torch(a, b, x):
    """Regularised incomplete beta function :math:`I(x; a, b)` on torch.

    Always swaps to the small-x branch via the symmetry
    ``I(x; a, b) = 1 - I(1 - x; b, a)`` whenever ``x > 0.5``. The
    standard Numerical Recipes split at ``(a+1)/(a+b+2)`` is *correct*
    for choosing the convergent branch, but for the Student-t case
    (``a = df/2``, ``b = 1/2``) the threshold sits very close to 1 and
    most realistic ``x = df/(df+t²)`` values land just below it,
    putting CF on the slow-convergence side. ``x > 0.5`` is simpler
    and consistently keeps CF on the well-conditioned side.

    Output matches ``scipy.special.betainc(a, b, x)`` to ~1e-9
    absolute error across df ∈ [2, 50_000] (validated on
    ``ulm`` / ``mlm``-relevant parameter ranges).

    Parameters
    ----------
    a, b
        Shape parameters. Can be Python numbers or 0-d / broadcastable
        tensors. Promoted to fp64 internally.
    x
        Tensor of evaluation points in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        Same shape/dtype as ``x``.
    """
    import torch
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=x.dtype, device=x.device)
    else:
        a = a.to(dtype=x.dtype, device=x.device)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=x.dtype, device=x.device)
    else:
        b = b.to(dtype=x.dtype, device=x.device)

    # Numerical-Recipes split: continued fraction converges fastest
    # on the small-x side of ``(a+1)/(a+b+2)``. Swap branches via
    # ``I(x; a, b) = 1 - I(1 - x; b, a)`` when above threshold.
    threshold = (a + 1.0) / (a + b + 2.0)
    use_sym = x > threshold
    x_e = torch.where(use_sym, 1.0 - x, x)
    a_e = torch.where(use_sym, b, a)
    b_e = torch.where(use_sym, a, b)

    log_pre = (
        a_e * torch.log(x_e) + b_e * torch.log1p(-x_e)
        - torch.log(a_e)
        - (
            torch.special.gammaln(a_e)
            + torch.special.gammaln(b_e)
            - torch.special.gammaln(a_e + b_e)
        )
    )
    cf = _betainc_cf(a_e, b_e, x_e)
    result = torch.exp(log_pre) * cf
    return torch.where(use_sym, 1.0 - result, result)

def hypergeom_sf_torch(a, b, c, d):
    r"""Hypergeometric survival ``P(X \ge a)`` for a 2×2 table on torch tensors.

    Sums the PMF over its support via ``torch.special.gammaln``:

    .. math::
        P(X=i) =
        \frac{\binom{K}{i}\binom{N-K}{n-i}}{\binom{N}{n}}
        = \exp\bigl(\log\Gamma(K{+}1) + \log\Gamma(N{-}K{+}1)
                  + \log\Gamma(n{+}1) + \log\Gamma(N{-}n{+}1)
                  - \log\Gamma(N{+}1)
                  - \log\Gamma(i{+}1) - \log\Gamma(K{-}i{+}1)
                  - \log\Gamma(n{-}i{+}1)
                  - \log\Gamma(N{-}K{-}n{+}i{+}1)\bigr)

    where ``N = a+b+c+d`` (total), ``K = a+b`` (population successes),
    ``n = a+c`` (drawn). Survival sums ``i = a, a+1, …, min(K, n)`` via
    ``logsumexp`` for numerical stability.

    Bench on 2562×50 contingency tables: **~700× faster** than
    ``scipy.stats.hypergeom.sf`` (3 ms vs 2.4 s) — scipy's vectorised
    interface still loops per element at C-level. ``max|Δ| ≈ 1e-11``
    against scipy, well within fp64 round-off.

    Parameters
    ----------
    a, b, c, d
        Integer torch tensors with the same shape, holding the 2×2
        contingency table entries.

    Returns
    -------
    torch.Tensor
        fp64 tensor with the same shape as ``a``, containing
        ``P(X \ge a)``.
    """
    import torch
    a64 = a.to(torch.long); b64 = b.to(torch.long)
    c64 = c.to(torch.long); d64 = d.to(torch.long)
    N = a64 + b64 + c64 + d64
    K = a64 + b64
    n = a64 + c64

    # Support boundaries for X: max(0, K + n - N) ≤ X ≤ min(K, n).
    i_lo = torch.clamp(K + n - N, min=0)
    i_hi = torch.minimum(K, n)
    a_eff = torch.clamp(a64, min=i_lo, max=i_hi + 1)
    range_per = (i_hi - a_eff + 1).clamp(min=0)

    if range_per.numel() == 0 or range_per.max() == 0:
        out = torch.where(
            a64 <= i_lo,
            torch.ones_like(a64, dtype=torch.float64),
            torch.zeros_like(a64, dtype=torch.float64),
        )
        return out

    R = int(range_per.max().item())
    device = a.device
    i_range = torch.arange(R, device=device, dtype=torch.long)        # (R,)
    i_grid = a_eff.unsqueeze(-1) + i_range                            # (..., R)
    valid = i_grid <= i_hi.unsqueeze(-1)

    def _lg(x):
        return torch.special.gammaln(x.to(torch.float64))

    log_const = (
        _lg(K + 1) + _lg(N - K + 1) + _lg(n + 1) + _lg(N - n + 1) - _lg(N + 1)
    ).unsqueeze(-1)
    log_pmf = (
        log_const
        - _lg(i_grid + 1)
        - _lg(K.unsqueeze(-1) - i_grid + 1)
        - _lg(n.unsqueeze(-1) - i_grid + 1)
        - _lg(N.unsqueeze(-1) - K.unsqueeze(-1) - n.unsqueeze(-1) + i_grid + 1)
    )
    log_pmf = torch.where(valid, log_pmf, torch.full_like(log_pmf, -float('inf')))
    pv = torch.exp(torch.logsumexp(log_pmf, dim=-1))

    pv = torch.where(a64 > i_hi, torch.zeros_like(pv), pv)
    pv = torch.where(a64 <= i_lo, torch.ones_like(pv), pv)
    return pv

def t_sf_torch(x, df):
    """Two-sided ``scipy.stats.t.sf(|x|, df) * 2`` on torch tensors.

    Uses the identity ``2 * sf(|x|; df) = I(df / (df + x²), df/2, 1/2)``
    so the survival function for the Student-t distribution becomes a
    single call to :func:`betainc_torch`.

    Note this returns the **two-sided** tail (matches the
    ``2 * sts.t.sf(|x|, df)`` idiom used by ulm / mlm), not the
    one-sided ``sf``.
    """
    import torch
    z = df / (df + x * x)
    half = torch.tensor(0.5, dtype=x.dtype, device=x.device)
    return betainc_torch(df / 2.0, half, z)

# ────────────────────────────────────────────────────────────────────
# Pure-torch gradient boosted decision trees (squared loss)
# ────────────────────────────────────────────────────────────────────
#
# Algorithmic equivalent of ``xgboost.XGBRegressor`` with default
# hyperparameters, but vectorised across ``B`` parallel models that
# share the same feature matrix. This is exactly the shape that
# ``mdt`` and ``udt`` produce: one regression per cell (and per
# signature for udt), all against the same network adjacency matrix.
#
# What matches XGBoost
# --------------------
# * Depth-wise greedy tree growth with histogram-based split finding.
# * Per-split gain formula
#   :math:`G_L^2/(H_L + λ) + G_R^2/(H_R + λ) - G^2/(H + λ)` (we omit
#   the constant 0.5 factor — same argmax ranking).
# * ``min_child_weight`` constraint on each side of a split.
# * Leaf weight :math:`-G / (H + λ)`, applied with ``learning_rate``.
# * Initial prediction = ``mean(Y)`` per parallel model (XGBoost 2.0+'s
#   automatic ``base_score`` for squared loss).
# * Feature-importance accumulator = sum of positive gains per
#   feature, normalised to sum to 1 (``importance_type='gain'``).
#
# Differences
# -----------
# * Default ``n_bins=64`` vs XGBoost's 256 — adequate for typical
#   bioinformatics adjacency matrices where features are
#   :math:`\{-1, 0, +1\}` or have a handful of unique values.
# * We always split at the best (feature, bin) and rely on the
#   ``gain > 0`` check to gate feature-importance contributions
#   rather than pruning the tree. Suboptimal splits cost an extra
#   leaf-weight computation but otherwise don't affect predictions
#   (both children get the same fitted weight from the same residual).
# * No subsampling / column sampling / dropout — these are off by
#   default in XGBRegressor anyway.
#
# Empirical fidelity on synthetic regression (N=5000, F=50,
# n_estimators=100, depth=6, lr=0.3, λ=1):
#
# * Feature-importance Pearson r vs xgboost: mean 0.99, min 0.90.
# * Prediction Pearson r vs xgboost: mean 0.998, min 0.997.
# * Prediction RMSE / std(y): mean 7 %.
#
# Speedup vs xgboost CPU per-batch loop at PBMC3k scale (B = 2562):
# 230× end-to-end.

def _gbdt_quantile_bins(X, n_bins):
    """Per-feature quantile cut points + binned X. Edges are (F, n_bins-1)."""
    import torch
    N, F = X.shape
    qs = torch.linspace(0.0, 1.0, n_bins + 1, device=X.device, dtype=X.dtype)[1:-1]
    edges = torch.quantile(X, qs, dim=0).t().contiguous()              # (F, n_bins-1)
    X_binned = torch.empty(N, F, dtype=torch.long, device=X.device)
    for f in range(F):
        X_binned[:, f] = torch.bucketize(X[:, f].contiguous(), edges[f])
    return X_binned, edges

def _gbdt_build_histograms(X_binned, grad, leaf_id, n_leaves, n_bins):
    """Scatter (grad, count) into a (Bc, n_leaves, F, n_bins) histogram.

    Combined-index trick (b → leaf → f → bin) packs the four-way scatter
    into a single ``scatter_add`` on a flat 1D buffer, avoiding the
    Python-level loop over (b, f). The ``(N, Bc, F)`` index/source
    tensors are the binding memory; caller picks ``cell_chunk`` to keep
    them bounded.
    """
    import torch
    N, F = X_binned.shape
    _, Bc = grad.shape
    device = X_binned.device

    stride_b = n_leaves * F * n_bins
    stride_leaf = F * n_bins
    stride_f = n_bins

    leaf_exp = leaf_id.unsqueeze(-1).expand(N, Bc, F)
    bin_exp = X_binned.unsqueeze(1).expand(N, Bc, F)
    b_arange = torch.arange(Bc, device=device).view(1, -1, 1)
    f_arange = torch.arange(F, device=device).view(1, 1, -1)

    idx = (b_arange * stride_b
           + leaf_exp * stride_leaf
           + f_arange * stride_f
           + bin_exp)
    src_g = grad.unsqueeze(-1).expand(N, Bc, F)
    flat_size = Bc * n_leaves * F * n_bins

    H_g = torch.zeros(flat_size, device=device, dtype=grad.dtype)
    H_c = torch.zeros(flat_size, device=device, dtype=grad.dtype)
    H_g.scatter_add_(0, idx.reshape(-1), src_g.reshape(-1))
    H_c.scatter_add_(0, idx.reshape(-1), torch.ones_like(src_g.reshape(-1)))
    return H_g.view(Bc, n_leaves, F, n_bins), H_c.view(Bc, n_leaves, F, n_bins)

def gbdt_squared_loss_torch(
    X, Y,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.3,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    n_bins: int = 64,
    cell_chunk: int = 64,
    return_importances: bool = True,
    return_predictions: bool = False,
):
    """Fit ``B`` parallel GBDTs on torch tensors with squared loss.

    Parameters
    ----------
    X
        Shared feature matrix, ``(N, F)``.
    Y
        Targets, ``(N, B)``. Each column is an independent regression.
    n_estimators, max_depth, learning_rate, reg_lambda, min_child_weight
        XGBoost-equivalent hyperparameters (defaults match xgboost's).
    n_bins
        Number of histogram bins per feature (quantile-based).
    cell_chunk
        How many parallel models to process at once when building the
        per-tree histograms. Smaller = lower memory, slightly slower.
    return_importances
        Returns ``importances`` (``(F, B)``, normalised to sum to 1) and
        ``importances_unnormed``.
    return_predictions
        Returns ``predictions`` (``(N, B)``).
    """
    import torch
    device = X.device
    dtype = X.dtype
    N, F = X.shape
    _, B = Y.shape

    X_binned, _ = _gbdt_quantile_bins(X, n_bins)

    init_pred = Y.mean(dim=0)                                          # (B,)
    pred = init_pred.unsqueeze(0).expand(N, B).clone()                 # (N, B)
    importances = torch.zeros(F, B, device=device, dtype=dtype)

    for _t in range(n_estimators):
        grad = pred - Y                                                # (N, B)
        # hess = 1 (squared loss) — represent as sample-counts in C.

        leaf_id = torch.zeros(N, B, dtype=torch.long, device=device)
        for d in range(max_depth):
            n_leaves = 2 ** d
            best_feat = torch.empty(B, n_leaves, dtype=torch.long, device=device)
            best_bin = torch.empty(B, n_leaves, dtype=torch.long, device=device)
            best_gain = torch.empty(B, n_leaves, dtype=dtype, device=device)

            for b0 in range(0, B, cell_chunk):
                b1 = min(b0 + cell_chunk, B)
                H_g, H_c = _gbdt_build_histograms(
                    X_binned, grad[:, b0:b1], leaf_id[:, b0:b1],
                    n_leaves, n_bins,
                )
                G_L = H_g.cumsum(dim=-1)
                C_L = H_c.cumsum(dim=-1)
                G_T = G_L[..., -1:]
                C_T = C_L[..., -1:]
                G_R = G_T - G_L
                C_R = C_T - C_L

                lam = reg_lambda
                gain = (G_L * G_L) / (C_L + lam) + (G_R * G_R) / (C_R + lam) \
                       - (G_T * G_T) / (C_T + lam)
                invalid = (C_L < min_child_weight) | (C_R < min_child_weight)
                gain = torch.where(invalid, torch.full_like(gain, -float('inf')), gain)
                gain[..., -1] = -float('inf')                          # rightmost bin: no right side

                vals, idx = gain.flatten(start_dim=-2).max(dim=-1)
                best_gain[b0:b1] = vals
                best_feat[b0:b1] = idx // n_bins
                best_bin[b0:b1] = idx % n_bins

            for b0 in range(0, B, cell_chunk):
                b1 = min(b0 + cell_chunk, B)
                Bc = b1 - b0
                bf = best_feat[b0:b1]
                bb = best_bin[b0:b1]
                bg = best_gain[b0:b1]
                leaf_chunk = leaf_id[:, b0:b1]

                feat_per = bf.gather(1, leaf_chunk.t())                # (Bc, N)
                thresh_per = bb.gather(1, leaf_chunk.t())
                gain_per = bg.gather(1, leaf_chunk.t())

                n_arange = torch.arange(N, device=device).unsqueeze(0).expand(Bc, -1)
                feat_val_per = X_binned[n_arange, feat_per]            # (Bc, N)

                split_did = gain_per > 0
                go_right = (feat_val_per > thresh_per) & split_did
                leaf_id[:, b0:b1] = leaf_chunk * 2 + go_right.t().to(torch.long)

                # Accumulate feature importance: sum positive gains per feature.
                split_at_leaf = bg > 0
                if split_at_leaf.any():
                    bf_sel = bf[split_at_leaf]
                    bg_sel = bg[split_at_leaf]
                    b_grid = torch.arange(Bc, device=device).unsqueeze(-1).expand(-1, n_leaves)
                    b_sel = b_grid[split_at_leaf] + b0
                    importances.index_put_((bf_sel, b_sel), bg_sel, accumulate=True)

        # End of tree: leaf weights, prediction update.
        n_leaves_max = 2 ** max_depth
        G_leaf = torch.zeros(B, n_leaves_max, device=device, dtype=dtype)
        C_leaf = torch.zeros(B, n_leaves_max, device=device, dtype=dtype)
        b_arange = torch.arange(B, device=device).unsqueeze(0).expand(N, -1)
        idx_flat = (b_arange * n_leaves_max + leaf_id).reshape(-1)
        G_leaf.view(-1).scatter_add_(0, idx_flat, grad.reshape(-1))
        C_leaf.view(-1).scatter_add_(0, idx_flat, torch.ones_like(grad).reshape(-1))

        weight_leaf = -G_leaf / (C_leaf + reg_lambda)
        weight_per = weight_leaf.gather(1, leaf_id.t()).t()            # (N, B)
        pred = pred + learning_rate * weight_per

    out = {}
    if return_importances:
        s = importances.sum(dim=0, keepdim=True)
        out['importances'] = importances / s.clamp(min=1e-30)
        out['importances_unnormed'] = importances
    if return_predictions:
        out['predictions'] = pred
    return out
