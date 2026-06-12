"""Pure-torch GPU t-SNE — no ``torchdr`` dependency.

Implements Van der Maaten & Hinton's t-SNE KL objective directly:

    Q_{ij} = (1 + ‖y_i − y_j‖²)^-1 / Z,      Z = Σ_{k≠l} W_{kl}
    C      = Σ_{ij} P_{ij} · log(P_{ij} / Q_{ij})

What makes this fit on a GPU in a few seconds at 94k cells is:

  * ``Z`` is estimated once per iteration from a large sample of random
    pairs (mean W over sampled pairs is an unbiased estimator of
    Σ W / (N · (N-1))). This replaces the Barnes-Hut tree used by
    sklearn and scanpy.
  * Positive-edge gradients come from sampling ``batch_size`` edges from
    the fuzzy graph by their P weights (cumsum + searchsorted).
  * The full loss is written against ``Y`` so autograd handles the
    gradients; we just use Adam.

Compared to the earlier LargeVis draft this version tracks Z explicitly,
so the attractive/repulsive balance matches standard t-SNE and the
output looks like sklearn / openTSNE / scanpy rather than LargeVis.

High-D affinities P_{ij}:
  * k-NN graph with k = ⌊3·perplexity⌋, via the chunked matmul KNN in
    ``omicverse.pp.pyg_knn_implementation``.
  * Per-row Gaussian bandwidth σ_i tuned by vectorised binary search on
    β_i = 1/(2σ_i²) so the Shannon entropy of P_{j|i} equals
    log(perplexity).
  * Symmetrise P_{j|i} + P_{i|j} and divide by 2N so Σ P = 1.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from .._optional import normalize_torch_device


# ---------------------------------------------------------------------------
# Perplexity-tuned Gaussian affinities
# ---------------------------------------------------------------------------


def _tune_betas(
    knn_d2: torch.Tensor,
    perplexity: float,
    *,
    n_iter: int = 64,
    tol: float = 1e-5,
) -> torch.Tensor:
    """Vectorised binary search on β_i = 1/(2σ_i²) so the Shannon entropy of
    P_{j|i} matches log(perplexity). ``knn_d2`` is the (N, k) matrix of
    *squared* distances to k-NN of each row (self excluded).
    """
    device = knn_d2.device
    N, k = knn_d2.shape
    target_H = math.log(perplexity)

    beta = torch.ones(N, device=device, dtype=torch.float32)
    lo = torch.zeros(N, device=device, dtype=torch.float32)
    hi = torch.full((N,), float("inf"), device=device, dtype=torch.float32)

    for _ in range(n_iter):
        # logits_{ij} = -beta_i * d²_ij ; softmax along k gives P_{j|i}
        logits = -knn_d2 * beta.unsqueeze(1)
        # Numerically-stable log-softmax
        logits_max = logits.max(dim=1, keepdim=True).values
        shifted = logits - logits_max
        expd = shifted.exp()
        Z = expd.sum(dim=1, keepdim=True)
        P = expd / Z
        logP = shifted - Z.log()
        H = -(P * logP).sum(dim=1)

        diff = H - target_H
        if diff.abs().max().item() < tol:
            break

        # If entropy is too high, σ is too large / β too small → raise β.
        # If entropy is too low, σ too small / β too large → lower β.
        too_low_beta = H > target_H  # need MORE β
        new_lo = torch.where(too_low_beta, beta, lo)
        new_hi = torch.where(too_low_beta, hi, beta)
        hi_finite = torch.isfinite(new_hi)
        new_beta = torch.where(
            hi_finite,
            (new_lo + new_hi) * 0.5,
            torch.where(too_low_beta, beta * 2.0, beta * 0.5),
        )
        converged = diff.abs() < tol
        beta = torch.where(converged, beta, new_beta)
        lo = torch.where(converged, lo, new_lo)
        hi = torch.where(converged, hi, new_hi)

    return beta  # (N,)


def _gaussian_affinities(
    knn_indices: torch.Tensor,
    knn_d2: torch.Tensor,
    perplexity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the symmetric P as a COO sparse graph on device.

    Returns (rows, cols, vals) where ``vals`` is already normalised so
    Σ P_{ij} = 1 (i.e. divided by 2N after symmetrisation).
    """
    device = knn_indices.device
    N, k = knn_indices.shape
    beta = _tune_betas(knn_d2, perplexity)

    # P_{j|i}: row-normalised stable softmax with the tuned β.
    logits = -knn_d2 * beta.unsqueeze(1)
    logits = logits - logits.max(dim=1, keepdim=True).values
    expd = logits.exp()
    P = expd / expd.sum(dim=1, keepdim=True)  # (N, k)

    rows = torch.arange(N, device=device).unsqueeze(1).expand(N, k).reshape(-1)
    cols = knn_indices.reshape(-1)
    vals = P.reshape(-1)

    # Symmetrise: P^sym_{ij} = (P_{j|i} + P_{i|j}) / (2N) and coalesce via
    # the same hash-sort trick we use for fuzzy_simplicial_set.
    all_rows = torch.cat([rows, cols])
    all_cols = torch.cat([cols, rows])
    all_vals = torch.cat([vals, vals])
    key = all_rows.to(torch.long) * N + all_cols.to(torch.long)
    sorted_key, perm = torch.sort(key)
    sorted_vals = all_vals[perm]
    unique_keys, inverse = torch.unique_consecutive(sorted_key, return_inverse=True)
    out_vals = torch.zeros(unique_keys.numel(), device=device, dtype=vals.dtype)
    out_vals.scatter_add_(0, inverse, sorted_vals)
    out_rows = unique_keys // N
    out_cols = unique_keys % N
    # Σ_{j} P_{j|i} = 1 per row, so Σ all_vals = 2N. Dividing by 2N makes
    # out_vals a proper probability distribution over undirected edges.
    out_vals = out_vals / (2.0 * N)
    # Drop numerically-tiny entries.
    keep = out_vals > 1e-12
    return out_rows[keep], out_cols[keep], out_vals[keep]


# ---------------------------------------------------------------------------
# LargeVis-style optimisation
# ---------------------------------------------------------------------------


def _pca_init(X: torch.Tensor, n_components: int) -> torch.Tensor:
    """Tiny PCA-style init: project onto top-``n_components`` SVD directions,
    then scale to a small magnitude. Much better than random — follows
    the convention of openTSNE / sklearn v1.2+."""
    Xc = X - X.mean(dim=0, keepdim=True)
    # truncated SVD via torch.svd_lowrank — cheap at 2 components even at 1M
    U, S, V = torch.svd_lowrank(Xc, q=n_components + 2)
    Y = U[:, :n_components] * S[:n_components]
    # Standardise to ~1e-4 magnitude (sklearn convention)
    Y = Y / (Y.std(dim=0, keepdim=True) + 1e-12) * 1e-4
    return Y.contiguous()


def tsne_gpu(
    X,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1500,
    learning_rate: float = 0.5,
    early_exaggeration: float = 12.0,
    early_exaggeration_iter: int = 300,
    batch_size: int = 8192,
    z_sample_size: int = 32768,
    init: str = "pca",
    random_state: int = 0,
    device=None,
    verbose: bool = False,
) -> np.ndarray:
    """GPU t-SNE with KL(P||Q) loss — returns ``(N, n_components)`` numpy.

    Matches the sklearn / scanpy / openTSNE output style (compact clusters,
    proper Student-t long tails) at GPU speed. Z normaliser is estimated
    each iteration from ``z_sample_size`` random pairs.
    """
    device = normalize_torch_device(device)

    if isinstance(X, np.ndarray):
        X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
    else:
        X_t = X.float().contiguous().to(device)
    N, D = X_t.shape

    # 1) k-NN via the chunked matmul KNN we already ship
    from omicverse.pp.pyg_knn_implementation import pyg_knn_search

    k = max(int(3 * perplexity), 10)
    k = min(k, N - 1)
    # pyg_knn_search with include_self=False already strips the self-match
    # internally, so requesting ``k`` here returns exactly k non-self
    # neighbors — don't add a spurious +1.
    knn_idx, knn_d = pyg_knn_search(
        X_t, k=k, device=device, include_self=False, return_tensor=True,
    )
    knn_d2 = (knn_d * knn_d).contiguous()
    knn_idx = knn_idx.contiguous()

    # 2) Perplexity-tuned P (symmetric, normalised so Σ P = 1)
    rows, cols, vals = _gaussian_affinities(knn_idx, knn_d2, perplexity=perplexity)

    # 3) Initialise Y. PCA + std=1e-4 matches sklearn's init='pca' default.
    rng = torch.Generator(device=device)
    rng.manual_seed(int(random_state))
    if init == "pca":
        Y = _pca_init(X_t, n_components).to(device).clone().detach()
    else:
        Y = torch.randn(N, n_components, device=device, generator=rng) * 1e-4
    Y.requires_grad_(True)

    # Adam works better than SGD here because the loss-vs-Y landscape is
    # noisy under MC sampling — Adam's per-parameter scale adaptation
    # absorbs that noise without us having to hand-tune SGD lr.
    optimizer = torch.optim.Adam([Y], lr=learning_rate)

    total_weight = float(vals.sum().item())
    cum = torch.cumsum(vals.double(), dim=0)
    n_edges_m1 = vals.numel() - 1

    # Total number of undirected pairs used as the Z scale (the MC mean
    # estimator's denominator). N*(N-1)/2 for undirected; t-SNE's
    # formulation uses Σ over i≠j (ordered) so we use N*(N-1).
    z_scale = float(N) * (N - 1)

    if verbose:
        print(f"t-SNE on {device}: N={N} edges={rows.numel()} k={k} iters={n_iter}")

    for it in range(n_iter):
        exag = early_exaggeration if it < early_exaggeration_iter else 1.0

        # --- Attractive term: sample positive edges proportional to P ---
        u = torch.rand(batch_size, device=device, generator=rng, dtype=torch.float64) * total_weight
        idx = torch.searchsorted(cum, u).clamp_(max=n_edges_m1)
        h = rows[idx]
        t = cols[idx]
        y_h = Y[h]
        y_t = Y[t]
        d_pos_sq = ((y_h - y_t) ** 2).sum(dim=1)
        W_pos = 1.0 / (1.0 + d_pos_sq)

        # --- Z estimate: sample z_sample_size random (i, j) pairs ---
        zi = torch.randint(0, N, (z_sample_size,), device=device, generator=rng)
        zj = torch.randint(0, N, (z_sample_size,), device=device, generator=rng)
        y_zi = Y[zi]
        y_zj = Y[zj]
        d_z_sq = ((y_zi - y_zj) ** 2).sum(dim=1)
        W_z = 1.0 / (1.0 + d_z_sq)
        Z_est = W_z.mean() * z_scale

        # --- True t-SNE KL loss: C = -Σ_pos P log W + log Z ---
        # The positive-edge sampling is P-weighted, so mean(log W_pos) is
        # an unbiased estimate of Σ_pos P log W_pos. Multiply by total_weight
        # (= 1 after normalisation) for scale; early_exaggeration scales it up.
        loss_attract = -torch.log(W_pos + 1e-12).mean() * exag
        loss_repel = torch.log(Z_est + 1e-12)
        loss = loss_attract + loss_repel

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if verbose and (it + 1) % 100 == 0:
            print(f"  iter {it+1}/{n_iter}  loss={loss.item():.4f}  Z={Z_est.item():.2e}")

    return Y.detach().cpu().numpy()
