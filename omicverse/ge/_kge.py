"""``KGEModule``: per-entity-type embedding tables + per-relation operators
+ comparator + contrastive (batch + uniform negatives) scoring.

A from-scratch, vectorised re-implementation of the scoring half of
PyTorch-BigGraph's ``MultiRelationEmbedder`` (``torchbiggraph/model.py``),
restricted to the single-machine / single-partition case SIMBA always runs
in (``num_partitions=1`` for every entity type). What's preserved from PBG,
exactly:

* embeddings initialised ``N(0, init_scale**2)`` per coordinate
  (``config.init_scale``, default ``1e-3``);
* the relation operator (``none`` / ``translation`` / ``diagonal``) is
  applied to the **right-hand side only** (SIMBA always uses ``none``, so
  this never actually fires in the SIMBA parity path, but is implemented
  for completeness per the spec);
* the **dot** comparator;
* for one batch of positive edges *all from the same relation* (PBG groups
  batches by relation; see ``_train.py``), negatives are formed by
  chunking the batch into sub-batches of size ``num_batch_negs`` and, within
  each chunk, using the *other* positives of that chunk as "batch"
  negatives (self-comparisons masked out) plus ``num_uniform_negs`` edges
  sampled uniformly at random from the *entire* entity table (independently
  per chunk) -- ``torchbiggraph/model.py:prepare_negatives``'s
  ``BATCH_UNIFORM`` mode, matching SIMBA's defaults
  (``num_batch_negs=num_uniform_negs=50``).

What's dropped (by design, see ``SPEC_ge_simba.md``): Hogwild multi-process
training, entity/relation partitioning, the ``ALL``/``UNIFORM``-only
negative-sampling modes (only used when ``num_batch_negs==0``, never the
case for SIMBA), and ``half_precision``/``bias``/``global_emb`` (SIMBA's own
``_settings.py`` sets ``global_emb=False``, ``comparator='dot'``, and never
enables ``bias``).
"""
from __future__ import annotations

import math

import torch
from torch import nn


class IdentityOperator(nn.Module):
    def forward(self, x):
        return x


class TranslationOperator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.translation = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x + self.translation


class DiagonalOperator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.diagonal = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.diagonal


_OPERATORS = {"none": IdentityOperator, "translation": TranslationOperator,
              "diagonal": DiagonalOperator}


def make_operator(name: str, dim: int) -> nn.Module:
    try:
        cls = _OPERATORS[name]
    except KeyError:
        raise ValueError(f"unknown operator {name!r}; choose from {sorted(_OPERATORS)}") from None
    return cls(dim) if cls is not IdentityOperator else cls()


class KGEModule(nn.Module):
    """One embedding table per entity type + one rhs operator per relation.

    Parameters
    ----------
    entity_counts : dict[str, int]
        Number of entities per entity type (table rows).
    relations : list[dict]
        ``{"lhs", "rhs", "operator", ...}`` per relation (as in
        ``HeteroGraph.relations``); only ``lhs``/``rhs``/``operator`` are read.
    dimension : int
        Embedding dimension (SIMBA/PBG default 50).
    init_scale : float
        Per-coordinate init std (PBG default ``1e-3``).
    max_norm : float, optional
        If set, embeddings are renormalised to this L2 norm after lookup
        (PBG's ``config.max_norm``; unused by SIMBA's own defaults).
    """

    def __init__(self, entity_counts: dict, relations: list, dimension: int = 50,
                 init_scale: float = 1e-3, max_norm: float | None = None):
        super().__init__()
        self.dimension = dimension
        self.max_norm = max_norm
        self.relations = [dict(r) for r in relations]
        self.entity_names = list(entity_counts.keys())
        self.embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.randn(max(int(n), 1), dimension) * init_scale)
            for name, n in entity_counts.items()
        })
        self.rhs_operators = nn.ModuleList([
            make_operator(r.get("operator", "none"), dimension) for r in self.relations
        ])

    # ------------------------------------------------------------------
    def _maybe_renorm(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is None:
            return x
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        factor = (self.max_norm / norm).clamp_max(1.0)
        return x * factor

    def l2_norm(self) -> torch.Tensor:
        """Sum of squares of *every* entity embedding table (all rows, not
        just the ones touched by the current batch) -- PBG's
        ``model.l2_norm()``, used by the stochastic weight-decay term."""
        total = None
        for p in self.embeddings.values():
            s = p.pow(2).sum()
            total = s if total is None else total + s
        return total if total is not None else torch.zeros(())

    # ------------------------------------------------------------------
    def forward(self, rel_id: int, src_idx: torch.Tensor, dst_idx: torch.Tensor, *,
                num_batch_negs: int = 50, num_uniform_negs: int = 50):
        """Score one batch of positive edges, all of relation ``rel_id``.

        Returns ``(pos_scores, lhs_neg_scores, rhs_neg_scores)``:
        ``pos_scores`` is ``(P,)``; ``lhs_neg_scores``/``rhs_neg_scores`` are
        ``(P, K)`` where ``K`` is the (possibly ragged-looking but here
        uniform, since masked rather than truncated) number of candidate
        negatives -- "corrupt the source, keep the target fixed" and
        "corrupt the target, keep the source fixed" respectively, as in
        PBG's bidirectional contrastive scoring.
        """
        rel = self.relations[rel_id]
        lhs_table = self.embeddings[rel["lhs"]]
        rhs_table = self.embeddings[rel["rhs"]]
        operator = self.rhs_operators[rel_id]

        num_pos = src_idx.numel()
        device = lhs_table.device
        if num_pos == 0:
            z = lhs_table.new_zeros((0,))
            return z, z.unsqueeze(1).expand(0, 0), z.unsqueeze(1).expand(0, 0)

        lhs_pos = self._maybe_renorm(lhs_table.index_select(0, src_idx))
        rhs_pos = self._maybe_renorm(operator(rhs_table.index_select(0, dst_idx)))

        if num_batch_negs > 0:
            chunk_size = min(num_batch_negs, num_pos)
        else:
            chunk_size = min(max(num_uniform_negs, 1), num_pos)
        chunk_size = max(chunk_size, 1)
        num_chunks = math.ceil(num_pos / chunk_size)
        pad = num_chunks * chunk_size - num_pos

        if pad:
            lhs_pos_p = torch.cat([lhs_pos, lhs_pos.new_zeros(pad, lhs_pos.size(1))], 0)
            rhs_pos_p = torch.cat([rhs_pos, rhs_pos.new_zeros(pad, rhs_pos.size(1))], 0)
        else:
            lhs_pos_p, rhs_pos_p = lhs_pos, rhs_pos
        lhs_pos_c = lhs_pos_p.view(num_chunks, chunk_size, -1)
        rhs_pos_c = rhs_pos_p.view(num_chunks, chunk_size, -1)

        if num_uniform_negs > 0:
            n_lhs, n_rhs = lhs_table.size(0), rhs_table.size(0)
            lhs_uni_idx = torch.randint(0, n_lhs, (num_chunks, num_uniform_negs), device=device)
            rhs_uni_idx = torch.randint(0, n_rhs, (num_chunks, num_uniform_negs), device=device)
            lhs_uni = self._maybe_renorm(lhs_table[lhs_uni_idx])
            rhs_uni = self._maybe_renorm(operator(rhs_table[rhs_uni_idx]))

        if num_batch_negs > 0:
            lhs_neg_c = torch.cat([lhs_pos_c, lhs_uni], dim=1) if num_uniform_negs > 0 else lhs_pos_c
            rhs_neg_c = torch.cat([rhs_pos_c, rhs_uni], dim=1) if num_uniform_negs > 0 else rhs_pos_c
            has_batch_negs = True
        elif num_uniform_negs > 0:
            lhs_neg_c, rhs_neg_c = lhs_uni, rhs_uni
            has_batch_negs = False
        else:  # no negatives at all (degenerate; loss will just be ~0)
            lhs_neg_c = lhs_pos_c.new_zeros((num_chunks, 0, lhs_pos_c.size(-1)))
            rhs_neg_c = rhs_pos_c.new_zeros((num_chunks, 0, rhs_pos_c.size(-1)))
            has_batch_negs = False

        pos_scores = (lhs_pos_c * rhs_pos_c).sum(-1)
        lhs_neg_scores = torch.bmm(rhs_pos_c, lhs_neg_c.transpose(-1, -2))
        rhs_neg_scores = torch.bmm(lhs_pos_c, rhs_neg_c.transpose(-1, -2))

        if has_batch_negs:
            diag = torch.arange(chunk_size, device=device)
            lhs_neg_scores[:, diag, diag] = -1e9
            rhs_neg_scores[:, diag, diag] = -1e9
            if pad:
                last_valid = chunk_size - pad
                lhs_neg_scores[-1, :last_valid, last_valid:chunk_size] = -1e9
                rhs_neg_scores[-1, :last_valid, last_valid:chunk_size] = -1e9

        pos_scores = pos_scores.reshape(-1)[:num_pos]
        lhs_neg_scores = lhs_neg_scores.reshape(num_chunks * chunk_size, -1)[:num_pos]
        rhs_neg_scores = rhs_neg_scores.reshape(num_chunks * chunk_size, -1)[:num_pos]
        return pos_scores, lhs_neg_scores, rhs_neg_scores


# ----------------------------------------------------------------------
# losses (torchbiggraph/losses.py)
# ----------------------------------------------------------------------
def softmax_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor,
                  weight: torch.Tensor | None = None) -> torch.Tensor:
    """``torchbiggraph.losses.SoftmaxLossFunction``: cross-entropy between
    ``[pos_score, logsumexp(neg_scores)]`` and the "positive is correct"
    label; sum (not mean) over the batch, optionally weighted per-edge."""
    num_pos = pos_scores.shape[0]
    if num_pos == 0 or neg_scores.shape[1] == 0:
        return pos_scores.new_zeros(())
    scores = torch.cat([pos_scores.unsqueeze(1), neg_scores.logsumexp(dim=1, keepdim=True)], dim=1)
    target = pos_scores.new_zeros((), dtype=torch.long).expand(num_pos)
    loss_per_sample = torch.nn.functional.cross_entropy(scores, target, reduction="none")
    if weight is not None:
        loss_per_sample = loss_per_sample * weight
    return loss_per_sample.sum()


def ranking_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 0.1,
                  weight: torch.Tensor | None = None) -> torch.Tensor:
    """``torchbiggraph.losses.RankingLossFunction`` (margin ranking)."""
    num_pos, num_neg = pos_scores.shape[0], neg_scores.shape[1]
    if num_pos == 0 or num_neg == 0:
        return pos_scores.new_zeros(())
    loss_per_sample = torch.nn.functional.margin_ranking_loss(
        neg_scores, pos_scores.unsqueeze(1),
        target=pos_scores.new_full((1, 1), -1.0), margin=margin, reduction="none")
    if weight is not None:
        loss_per_sample = loss_per_sample * weight.unsqueeze(-1)
    return loss_per_sample.sum()


def logistic_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor,
                   weight: torch.Tensor | None = None) -> torch.Tensor:
    """``torchbiggraph.losses.LogisticLossFunction``."""
    num_pos, num_neg = pos_scores.shape[0], neg_scores.shape[1]
    neg_weight = 1.0 / num_neg if num_neg > 0 else 0.0
    pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pos_scores, pos_scores.new_ones(()).expand(num_pos), reduction="sum", weight=weight)
    neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        neg_scores, neg_scores.new_zeros(()).expand(num_pos, num_neg), reduction="sum",
        weight=weight.unsqueeze(-1) if weight is not None else None)
    return pos_loss + neg_weight * neg_loss


LOSS_FUNCTIONS = {"softmax": softmax_loss, "ranking": ranking_loss, "logistic": logistic_loss}


__all__ = ["KGEModule", "softmax_loss", "ranking_loss", "logistic_loss", "LOSS_FUNCTIONS",
           "make_operator"]
