"""Ridge regression with a row-permutation null (R SecAct ``.ridge_pureR``).

``beta = (X'X + lambda I)^-1 X' Y``; for each of ``nrand`` permutations the
rows (genes) of ``Y`` are shuffled with the SecAct permutation table, giving
``se = sd(beta_rand)``, ``zscore = (beta - mean(beta_rand)) / se`` and
``pvalue = (#{|beta_rand| >= |beta|} + 1) / (nrand + 1)``.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from ._rng import permutation_table


def _projection(X: np.ndarray, lam: float) -> np.ndarray:
    A = X.T @ X + lam * np.eye(X.shape[1])
    L = linalg.cholesky(A, lower=True)
    return linalg.cho_solve((L, True), X.T)  # p x n, R: backsolve(R, forwardsolve(t(R), t(X)))


def ridge_permutation(X, Y, lam: float = 5e5, nrand: int = 1000, *, rng_method: str = "mt19937",
                      seed: int = 0, use_gpu: bool = False, batch: int | None = None,
                      progress: bool = False) -> dict[str, np.ndarray]:
    """Permutation ridge on column-scaled ``X`` (n x p) and ``Y`` (n x m)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, m = Y.shape
    T = _projection(X, float(lam))
    beta = T @ Y
    table = permutation_table(n, nrand, rng_method, seed)

    if use_gpu:
        res = _ridge_torch(T, Y, beta, table, progress)
        if res is not None:
            return res

    aver = np.zeros_like(beta)
    aver_sq = np.zeros_like(beta)
    count = np.zeros_like(beta)
    abs_beta = np.abs(beta)
    p = T.shape[0]
    # acceleration: T @ Y[perm] == T[:, argsort(perm)] @ Y (exact identity, playbook §1/§4.1).
    # Permuting the columns of the small projection T (p x n) instead of the rows of
    # Y (n x m, m = cells) avoids copying Y every round, and several permutations
    # share one GEMM: [T[:, inv_1]; T[:, inv_2]; ...] @ Y. Only the floating-point
    # summation order of the GEMM changes (zscore/pvalue agree with R to ~1e-13).
    step = batch or int(max(1, min(nrand, (512 * 2 ** 20) // max(p * m * 8, 1))))
    iterator = range(0, nrand, step)
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, total=-(-nrand // step), desc="SecAct permutations")
    for start in iterator:
        inv = np.argsort(table[start:start + step], axis=1, kind="stable")
        stacked_T = np.concatenate([T[:, row] for row in inv], axis=0)  # (k*p) x n
        rand = (stacked_T @ Y).reshape(len(inv), p, m)
        for k in range(len(inv)):
            b = rand[k]
            aver += b
            aver_sq += b * b
            count += np.abs(b) >= abs_beta
    return _finalize(beta, aver, aver_sq, count, nrand)


def _finalize(beta, aver, aver_sq, count, nrand):
    aver = aver / nrand
    aver_sq = aver_sq / nrand
    se = np.sqrt(np.maximum(aver_sq - aver ** 2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = np.where(se > 0, (beta - aver) / se, 0.0)
    pvalue = (count + 1.0) / (nrand + 1.0)
    return {"beta": beta, "se": se, "zscore": zscore, "pvalue": pvalue}


def _ridge_torch(T, Y, beta, table, progress):  # pragma: no cover - needs a GPU
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        device, dtype = torch.device("cuda"), torch.float64
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device, dtype = torch.device("mps"), torch.float32
    else:
        return None
    Tt = torch.as_tensor(T, device=device, dtype=dtype)
    Yt = torch.as_tensor(Y, device=device, dtype=dtype)
    bt = torch.as_tensor(beta, device=device, dtype=dtype)
    abs_b = bt.abs()
    aver = torch.zeros_like(bt)
    aver_sq = torch.zeros_like(bt)
    count = torch.zeros_like(bt)
    perms = torch.as_tensor(table, device=device)
    iterator = range(perms.shape[0])
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, desc="SecAct permutations (GPU)")
    for k in iterator:
        b = Tt @ Yt[perms[k]]
        aver += b
        aver_sq += b * b
        count += (b.abs() >= abs_b).to(dtype)
    to_np = lambda t: t.detach().cpu().numpy().astype(np.float64)
    return _finalize(beta, to_np(aver), to_np(aver_sq), to_np(count), perms.shape[0])
