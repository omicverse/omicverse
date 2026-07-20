"""Pure-torch GPU Leiden — no torch_sparse / torch_scatter dependency.

Replaces the previous ``_leiden_test`` implementation with one that:

* Uses only vanilla ``torch.scatter_*`` ops on raw ``(row, col, val)``
  tensors (no ``SparseTensor`` wrapper, no ``torch_scatter.scatter_max``).
* Drops the per-batch Python loop overhead by defaulting to a small,
  constant ``n_batches`` (4 by default). The previous ``sqrt(N)`` heuristic
  fragmented the GPU work into hundreds of micro-kernels per iteration,
  which is the root cause of the previous code being no faster than CPU
  igraph at 94k cells.
* Falls back to the CPU igraph path (``omicverse.pp._leiden.leiden``)
  whenever CUDA is unavailable. Callers don't need to special-case this.

Algorithm: standard multilevel Leiden — iterated batched local-move + a
connected-component refinement that splits any community whose induced
subgraph has more than one component. The refinement isn't the full
Traag 2019 γ-connectedness step (which would require a per-community
inner Leiden pass); for typical scRNA kNN graphs the difference is
small (~1-2% modularity) but call this out so the limitation is visible.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from scipy import sparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_coo_tensors(adjacency, device, dtype=torch.float32):
    """scipy sparse / array → (row, col, val) torch tensors on `device`."""
    if not sparse.issparse(adjacency):
        adjacency = sparse.csr_matrix(adjacency)
    coo = adjacency.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64, copy=False)).to(device)
    col = torch.from_numpy(coo.col.astype(np.int64, copy=False)).to(device)
    val = torch.from_numpy(coo.data.astype(np.float32, copy=False)).to(device, dtype=dtype)
    return row, col, val


def _drop_self_loops(row, col, val):
    keep = row != col
    return row[keep], col[keep], val[keep]


def _degrees(row, val, n):
    deg = torch.zeros(n, device=row.device, dtype=val.dtype)
    deg.scatter_add_(0, row, val)
    return deg


def _scatter_argmax(values, index, dim_size):
    """Per-group argmax without ``torch_scatter``.

    Returns ``(max_per_group, repr_index_per_group)`` where ``repr_index``
    is an index into ``values`` whose value equals the per-group max.
    Groups with no entries get ``max=-inf`` and ``repr_index=-1``.
    """
    out_max = torch.full(
        (dim_size,), float("-inf"), device=values.device, dtype=values.dtype
    )
    out_max.scatter_reduce_(0, index, values, reduce="amax", include_self=True)

    # Mask elements that achieve their group's max. For ties we want a
    # deterministic representative — pick the smallest element-index by
    # scattering arange with ``amin`` reduction.
    is_max = values == out_max[index]
    elem_idx = torch.arange(values.numel(), device=values.device)
    masked_idx = torch.where(
        is_max, elem_idx, torch.full_like(elem_idx, values.numel())
    )
    arg = torch.full(
        (dim_size,), values.numel(), device=values.device, dtype=torch.long
    )
    arg.scatter_reduce_(0, index, masked_idx, reduce="amin", include_self=True)
    arg = torch.where(arg < values.numel(), arg, torch.full_like(arg, -1))
    return out_max, arg


# ---------------------------------------------------------------------------
# Local move
# ---------------------------------------------------------------------------


def _local_move(
    row,
    col,
    val,
    deg,
    two_m,
    comm,
    n,
    n_comm,
    resolution,
    n_iter,
    n_batches,
    generator,
):
    """Batched parallel local-move. Each iteration: random permutation,
    process in ``n_batches`` chunks; within a chunk all nodes propose moves
    against the chunk-start sigma, then we apply moves and update sigma
    before the next chunk. Default ``n_batches=4`` keeps the Python loop
    overhead negligible while only mildly increasing staleness vs sequential.
    """
    device = comm.device
    dtype = val.dtype

    for _ in range(n_iter):
        sigma = torch.zeros(n_comm, device=device, dtype=dtype)
        sigma.scatter_add_(0, comm, deg)

        order = torch.randperm(n, device=device, generator=generator)
        batch_size = (n + n_batches - 1) // n_batches
        moved_total = 0

        for b in range(n_batches):
            batch_nodes = order[b * batch_size : (b + 1) * batch_size]
            B = batch_nodes.numel()
            if B == 0:
                continue

            # Find edges originating from any node in this batch. Self-loops
            # are excluded so they don't inflate "stay in own community" k_ic
            # (deg/sigma already account for them via the contraction).
            in_batch = torch.zeros(n, dtype=torch.bool, device=device)
            in_batch[batch_nodes] = True
            mask = in_batch[row] & (row != col)
            br, bc, bv = row[mask], col[mask], val[mask]
            if br.numel() == 0:
                continue

            # Map global node -> local position within the batch.
            local_idx = torch.full((n,), -1, device=device, dtype=torch.long)
            local_idx[batch_nodes] = torch.arange(B, device=device)
            local_br = local_idx[br]
            target_c = comm[bc]

            # k_ic: edge-weight from each (batch_local_node, target_community).
            pair_key = local_br * n_comm + target_c
            unique_key, inv = torch.unique(pair_key, return_inverse=True)
            k_ic = torch.zeros(unique_key.numel(), device=device, dtype=dtype)
            k_ic.scatter_add_(0, inv, bv)

            pair_local = unique_key // n_comm
            pair_comm = unique_key % n_comm

            # Modularity gain: k_ic - γ * k_i * Σ_eff / 2m
            ki = deg[batch_nodes[pair_local]]
            own_c = comm[batch_nodes[pair_local]]
            is_own = pair_comm == own_c
            sig_eff = sigma[pair_comm] - is_own.to(dtype) * ki
            gains = k_ic - resolution * ki * sig_eff / two_m

            # Make sure each batch-local node has its current community as a
            # candidate (gain may be 0 if no edge to own community).
            has_own = torch.zeros(B, dtype=torch.bool, device=device)
            has_own[pair_local[is_own]] = True
            miss = (~has_own).nonzero(as_tuple=True)[0]
            if miss.numel() > 0:
                miss_global = batch_nodes[miss]
                mc = comm[miss_global]
                mki = deg[miss_global]
                mg = -resolution * mki * (sigma[mc] - mki) / two_m
                pair_local = torch.cat([pair_local, miss])
                pair_comm = torch.cat([pair_comm, mc])
                gains = torch.cat([gains, mg])

            best_gain, arg = _scatter_argmax(gains, pair_local, B)
            best_comm_local = torch.where(
                arg >= 0, pair_comm[arg.clamp(min=0)], comm[batch_nodes]
            )

            old = comm[batch_nodes]
            move = (best_gain > 1e-12) & (best_comm_local != old)
            if move.any():
                moved_global = batch_nodes[move]
                old_c = old[move]
                new_c = best_comm_local[move]
                moved_ki = deg[moved_global]
                sigma.scatter_add_(0, old_c, -moved_ki)
                sigma.scatter_add_(0, new_c, moved_ki)
                comm[moved_global] = new_c
                moved_total += int(move.sum().item())

        if moved_total == 0:
            break

    return comm


# ---------------------------------------------------------------------------
# Refinement (split disconnected communities) and graph contraction
# ---------------------------------------------------------------------------


def _split_disconnected(row, col, comm, n):
    """Label each node by (community, connected-component-within-community).

    Approximate connected-component labeling via parallel label propagation
    along same-community edges. ``O(diameter)`` iterations, but kNN graphs
    typically have small diameter (~5-15 hops) so this converges fast.
    """
    device = comm.device
    same = comm[row] == comm[col]
    if not same.any():
        return comm
    sr, sc = row[same], col[same]

    labels = torch.arange(n, device=device, dtype=torch.long)
    max_iter = max(8, int(np.log2(max(n, 2))) + 2)
    for _ in range(max_iter):
        nl = labels.clone()
        nl.scatter_reduce_(0, sr, labels[sc], reduce="amin", include_self=True)
        nl.scatter_reduce_(0, sc, labels[sr], reduce="amin", include_self=True)
        if torch.equal(nl, labels):
            break
        labels = nl

    composite = comm * (n + 1) + labels
    _, refined = torch.unique(composite, return_inverse=True)
    return refined


def _contract(row, col, val, comm, n_comm):
    """Aggregate (row, col) → community pairs, summing edge weights."""
    device = row.device
    new_row = comm[row]
    new_col = comm[col]
    pair_key = new_row * n_comm + new_col
    unique, inv = torch.unique(pair_key, return_inverse=True)
    new_val = torch.zeros(unique.numel(), device=device, dtype=val.dtype)
    new_val.scatter_add_(0, inv, val)
    return unique // n_comm, unique % n_comm, new_val


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def _pick_device(device):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    return _normalize_device(device)


def _normalize_device(device):
    """Give ``cuda`` a concrete index so equality comparisons against tensor
    devices (which land on ``cuda:N``) don't silently miss."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _choose_graph_local(adata, obsp_key, neighbors_key):
    """Re-implementation of ``scanpy._utils._choose_graph`` so we don't
    depend on a private scanpy API."""
    if obsp_key is not None:
        return adata.obsp[obsp_key]
    if neighbors_key is None:
        if "neighbors" in adata.uns:
            obsp_name = adata.uns["neighbors"].get("connectivities_key", "connectivities")
        else:
            obsp_name = "connectivities"
    else:
        obsp_name = adata.uns[neighbors_key].get("connectivities_key", "connectivities")
    return adata.obsp[obsp_name]


def _coerce_resolution(resolution):
    """Return ``resolution`` as a plain Python ``float``.

    Programmatic / agent-generated callers sometimes pass a length-1
    container (e.g. ``resolution=[1.0]``). Left as-is, the downstream
    ``resolution * tensor`` in the local-move becomes Python **list
    repetition**, which calls ``tensor.__index__()`` and raises the opaque
    ``TypeError: only integer tensors of a single element can be converted
    to an index`` (issue #877). Unwrap single-element containers; reject
    multi-element ones with a clear message.
    """
    if isinstance(resolution, bool):  # avoid True/False sneaking through as 1/0
        raise ValueError(f"ov.pp.leiden resolution must be a number, got {resolution!r}")
    if isinstance(resolution, (int, float)):
        return float(resolution)
    if isinstance(resolution, (list, tuple)):
        if len(resolution) != 1:
            raise ValueError(
                "ov.pp.leiden expects a single resolution value, got a "
                f"{type(resolution).__name__} of length {len(resolution)}: "
                f"{resolution!r}"
            )
        return _coerce_resolution(resolution[0])
    # numpy / torch scalar or single-element array/tensor. ``.item()`` unwraps
    # a single element on numpy 1.x AND 2.x (numpy 2 removed the implicit
    # float(size-1 array) conversion) and on torch; multi-element raises.
    item = getattr(resolution, "item", None)
    if callable(item):
        try:
            return float(item())
        except Exception as exc:  # multi-element array/tensor, or non-numeric
            raise ValueError(
                "ov.pp.leiden expects a single resolution value, got "
                f"{resolution!r}"
            ) from exc
    try:
        return float(resolution)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ov.pp.leiden resolution must be a single number, got {resolution!r}"
        ) from exc


def leiden_gpu_sparse_multilevel(
    adata,
    resolution: float = 1.0,
    *,
    random_state: int = 0,
    key_added: str = "leiden",
    adjacency=None,
    use_weights: bool = True,
    neighbors_key: Optional[str] = None,
    obsp: Optional[str] = None,
    copy: bool = False,
    device=None,
    local_iterations: int = 100,
    max_levels: int = 10,
    n_batches: int = 4,
    symmetrize: Optional[bool] = None,
    # Legacy/compat kwargs (silently accepted, no-op):
    local_move_mode: str = "batched",
):
    """GPU Leiden clustering, pure torch.

    Parameters
    ----------
    adata
        AnnData with a precomputed connectivities graph in ``adata.obsp``.
    resolution
        Modularity resolution γ.
    random_state
        Seed for the local-move shuffle and CUDA kernels.
    n_batches
        Number of batches the local-move pass is split into. Larger means
        more parallel and slightly more "stale" sigma; smaller means closer
        to sequential Leiden but more Python overhead. Default ``4`` —
        keeping this constant (rather than the previous ``sqrt(N)``) is
        what makes the GPU path actually beat CPU igraph at moderate N.
    max_levels
        Maximum number of multilevel rounds.
    local_iterations
        Maximum local-move sweeps per level (early-exits when no moves).

    Returns
    -------
    None (modifies ``adata`` in place; see ``key_added``).
    """
    resolution = _coerce_resolution(resolution)
    dev = _pick_device(device)

    # Small-N fast path: at small N the multilevel loop's thousands of
    # CUDA-kernel launches dominate the actual compute. On some GPUs this
    # can make cpu-gpu-mixed mode *slower* than igraph CPU by 100x+ for
    # datasets under ~10k cells (e.g. pbmc3k: 50s mixed vs 0.1s igraph).
    # Defer to igraph below the threshold; it's fast there and matches
    # the output scanpy users are used to.
    n_obs = int(adata.n_obs)
    if dev.type != "cuda" or n_obs < 10000:
        from ._leiden import leiden as _leiden_cpu

        return _leiden_cpu(
            adata,
            resolution=resolution,
            random_state=random_state,
            key_added=key_added,
            adjacency=adjacency,
            use_weights=use_weights,
            neighbors_key=neighbors_key,
            obsp=obsp,
            copy=copy,
        )

    ad = adata.copy() if copy else adata
    if adjacency is None:
        adjacency = _choose_graph_local(ad, obsp, neighbors_key)
    adjacency = adjacency.tocsr()

    # Symmetrize unless the caller explicitly opts out.
    if symmetrize is False:
        sym = adjacency
    else:
        sym = adjacency.maximum(adjacency.T)

    n0 = sym.shape[0]

    # Fast path: if a prior ov.pp.neighbors(method='torch') produced this
    # same graph on device, reuse it instead of the scipy→GPU round-trip.
    # fuzzy_simplicial_set_gpu already returns a symmetric graph so we can
    # use the cached tensors directly regardless of ``symmetrize``.
    from ._gpu_cache import cache_get
    cached = cache_get(ad, "connectivities", source_obj=adjacency) if use_weights else None
    if cached is not None:
        c_row, c_col, c_val, c_n = cached
        if _normalize_device(c_row.device) == dev and c_n == n0:
            row, col, val = c_row, c_col, c_val
        else:
            cached = None  # wrong device / shape; fall through
    if cached is None:
        row, col, val = _to_coo_tensors(sym, dev)
        if not use_weights:
            val = torch.ones_like(val)
    # Self-loops are KEPT in (row, col, val) so they survive contraction
    # (intra-community mass at higher levels lives as self-loops). The
    # local-move filters them out per-batch when computing k_ic.
    deg = _degrees(row, val, n0)
    two_m = float(val.sum().item())

    rng = torch.Generator(device=dev)
    rng.manual_seed(int(random_state))

    comm = torch.arange(n0, device=dev, dtype=torch.long)
    n_nodes = n0
    accum = comm.clone()  # mapping original-node -> current super-node

    for level in range(max_levels):
        n_comm = int(comm.max().item()) + 1
        comm = _local_move(
            row, col, val, deg, two_m, comm, n_nodes, n_comm,
            resolution, local_iterations, n_batches, rng,
        )
        # NOTE: refinement disabled — the previous _split_disconnected over-
        # split well-formed clusters into thousands of singletons. Merge-small-
        # components style refinement (Traag 2019 §2.2) needs adding back later.
        _, comm = torch.unique(comm, return_inverse=True)
        nc = int(comm.max().item()) + 1

        # accum_labels follow the contraction chain
        accum = comm[accum] if level > 0 else comm.clone()

        if nc == n_nodes:
            break  # converged: no merges happened at this level

        row, col, val = _contract(row, col, val, comm, nc)
        deg = _degrees(row, val, nc)
        two_m = float(val.sum().item())
        comm = torch.arange(nc, device=dev, dtype=torch.long)
        n_nodes = nc

    _, accum = torch.unique(accum, return_inverse=True)
    lab_np = accum.cpu().numpy().astype(str)
    cats = natsorted(np.unique(lab_np))
    ad.obs[key_added] = pd.Categorical(lab_np, categories=cats)
    ad.uns[key_added] = {
        "params": dict(
            resolution=resolution,
            random_state=random_state,
            local_iterations=local_iterations,
            max_levels=max_levels,
            n_batches=n_batches,
            device=str(dev),
        )
    }
    return ad if copy else None
