"""Epoch loop over edge batches for :class:`~omicverse.ge._kge.KGEModule`.

Mirrors ``torchbiggraph/train_cpu.py``'s ``Trainer`` / ``IterationManager``
/ eval-before-and-after bookkeeping, but as one vectorised single-process
loop (no Hogwild workers, no bucket/partition scheduling -- SIMBA always
has one partition per entity type, so none of that machinery is needed):

* each batch is all-one-relation (``batch_edges_group_by_relation_type``);
  batches are built by grouping an epoch's shuffled edges by relation, then
  shuffling the *order* the per-relation batches are visited in;
* the stochastic weight decay trick (``train_cpu.py``: with probability
  ``1/wd_interval`` add ``wd * wd_interval * l2_norm()`` to the loss) is
  applied per training step, exactly as in PBG;
* held-out-edge evaluation (loss + MRR, ``torchbiggraph/eval.py``'s
  ``RankingEvaluator``) runs once per epoch.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ._kge import LOSS_FUNCTIONS
from ._optim import RowAdagrad


def _epoch_batches(edges_rel: np.ndarray, idx: np.ndarray, batch_size: int,
                    rng: np.random.Generator) -> list:
    """Group ``idx`` by relation, shuffle within each group, cut into
    ``batch_size`` chunks, then shuffle the resulting batch order --
    vectorised equivalent of PBG's ``batch_edges_group_by_relation_type``
    (every batch is single-relation; relation visiting order is randomised,
    weighted by how many edges of that relation remain)."""
    if len(idx) == 0:
        return []
    rels = edges_rel[idx]
    order = np.argsort(rels, kind="stable")
    sorted_idx = idx[order]
    sorted_rel = rels[order]
    uniq, starts = np.unique(sorted_rel, return_index=True)
    bounds = list(starts) + [len(sorted_idx)]
    batches = []
    for g in range(len(uniq)):
        s, e = bounds[g], bounds[g + 1]
        group = sorted_idx[s:e]
        group = group[rng.permutation(len(group))]
        rel_id = int(uniq[g])
        for bs in range(0, len(group), batch_size):
            batches.append((rel_id, group[bs:bs + batch_size]))
    perm = rng.permutation(len(batches))
    return [batches[i] for i in perm]


def evaluate(graph, kge, eval_idx: np.ndarray, *, batch_size: int, rel_weights: torch.Tensor,
             loss_fn: str, num_batch_negs: int, num_uniform_negs: int, device) -> dict:
    """Loss + MRR over ``eval_idx`` (held-out edges), ``RankingEvaluator``'s
    metrics (reciprocal rank averaged over both corruption directions)."""
    loss_function = LOSS_FUNCTIONS[loss_fn]
    src_all = torch.as_tensor(graph.edges_src, device=device)
    dst_all = torch.as_tensor(graph.edges_dst, device=device)
    was_training = kge.training
    kge.eval()
    total_loss, total_count = 0.0, 0
    rank_parts = []
    rels = graph.edges_rel[eval_idx]
    with torch.no_grad():
        for rel_id in np.unique(rels):
            rel_id = int(rel_id)
            sub = eval_idx[rels == rel_id]
            for bs in range(0, len(sub), max(batch_size, 1)):
                chunk = sub[bs:bs + batch_size]
                idx_t = torch.as_tensor(chunk, device=device)
                src, dst = src_all[idx_t], dst_all[idx_t]
                pos, lneg, rneg = kge(rel_id, src, dst, num_batch_negs=num_batch_negs,
                                      num_uniform_negs=num_uniform_negs)
                lhs_loss = loss_function(pos, lneg)
                rhs_loss = loss_function(pos, rneg)
                loss = rel_weights[rel_id] * (lhs_loss + rhs_loss)
                total_loss += float(loss)
                total_count += len(chunk)
                if lneg.numel():
                    rank_parts.append(((lneg >= pos.unsqueeze(1)).sum(1) + 1).float())
                if rneg.numel():
                    rank_parts.append(((rneg >= pos.unsqueeze(1)).sum(1) + 1).float())
    kge.train(was_training)
    mrr = float((1.0 / torch.cat(rank_parts)).mean()) if rank_parts else float("nan")
    return {"loss": total_loss / max(total_count, 1), "mrr": mrr}


def fit(graph, kge, *, num_epochs: int = 10, batch_size: int = 1000, lr: float = 0.1,
        wd: float = 0.0, wd_interval: int = 50, loss_fn: str = "softmax",
        num_batch_negs: int = 50, num_uniform_negs: int = 50, eval_fraction: float = 0.05,
        seed: int = 0, device=None, progress: bool = True, early_stopping: bool = False,
        patience: int = 5, split_method: str = "random", holdout_idx=None) -> dict:
    """Train ``kge`` on ``graph`` for ``num_epochs`` passes. Returns a history
    dict with per-epoch ``train_loss``, ``eval_loss``, ``eval_mrr``.

    ``split_method``: passed to :meth:`HeteroGraph.split` (``"random"``
    disjoint-per-seed, or ``"pbg"`` for PyTorch-BigGraph's exact,
    training-seed-independent holdout -- see its docstring). Ignored if
    ``holdout_idx=(train_idx, eval_idx)`` is given directly (e.g. to share
    one fixed holdout across several training seeds, as PBG's own 3-seed
    runs do).
    """
    graph.build()
    device = device if device is not None else next(iter(kge.embeddings.values())).device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if holdout_idx is not None:
        train_idx, eval_idx = holdout_idx
    else:
        train_idx, eval_idx = graph.split(eval_fraction, seed, method=split_method)
    rel_weights = torch.as_tensor(graph.relation_weights(), dtype=torch.float32, device=device)
    loss_function = LOSS_FUNCTIONS[loss_fn]

    emb_optimizer = RowAdagrad(list(kge.embeddings.parameters()), lr=lr)
    op_params = [p for op in kge.rhs_operators for p in op.parameters()]
    op_optimizer = torch.optim.Adagrad(op_params, lr=lr) if op_params else None

    src_all = torch.as_tensor(graph.edges_src, device=device)
    dst_all = torch.as_tensor(graph.edges_dst, device=device)

    history = {"epoch": [], "train_loss": [], "eval_loss": [], "eval_mrr": []}
    t0 = time.time()
    rng_epochs = range(num_epochs)
    pbar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(rng_epochs, desc="ov.ge fit")
            rng_epochs = pbar
        except ImportError:
            pass

    best_eval, bad_epochs = float("inf"), 0
    for epoch in rng_epochs:
        batches = _epoch_batches(graph.edges_rel, train_idx, batch_size, rng)
        running_loss, running_count = 0.0, 0
        for rel_id, idx in batches:
            if len(idx) == 0:
                continue
            idx_t = torch.as_tensor(idx, device=device)
            src, dst = src_all[idx_t], dst_all[idx_t]
            emb_optimizer.zero_grad()
            if op_optimizer is not None:
                op_optimizer.zero_grad()
            pos, lneg, rneg = kge(rel_id, src, dst, num_batch_negs=num_batch_negs,
                                  num_uniform_negs=num_uniform_negs)
            lhs_loss = loss_function(pos, lneg)
            rhs_loss = loss_function(pos, rneg)
            loss = rel_weights[rel_id] * (lhs_loss + rhs_loss)
            if wd and wd > 0 and rng.random() < 1.0 / wd_interval:
                loss = loss + wd * wd_interval * kge.l2_norm()
            loss.backward()
            emb_optimizer.step()
            if op_optimizer is not None:
                op_optimizer.step()
            running_loss += float(loss.detach())
            running_count += len(idx)
        train_loss = running_loss / max(running_count, 1)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        if len(eval_idx):
            ev = evaluate(graph, kge, eval_idx, batch_size=batch_size, rel_weights=rel_weights,
                          loss_fn=loss_fn, num_batch_negs=num_batch_negs,
                          num_uniform_negs=num_uniform_negs, device=device)
            history["eval_loss"].append(ev["loss"])
            history["eval_mrr"].append(ev["mrr"])
            cur = ev["loss"]
        else:
            history["eval_loss"].append(float("nan"))
            history["eval_mrr"].append(float("nan"))
            cur = train_loss
        if pbar is not None:
            pbar.set_postfix(loss=f"{train_loss:.4f}")
        if early_stopping:
            if cur < best_eval - 1e-6:
                best_eval, bad_epochs = cur, 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
    history["_time"] = time.time() - t0
    return history


__all__ = ["fit", "evaluate"]
