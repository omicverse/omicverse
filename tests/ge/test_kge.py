"""``KGEModule`` negative sampling + softmax loss vs. hand computation."""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omicverse.ge._kge import KGEModule, logistic_loss, ranking_loss, softmax_loss


def test_softmax_loss_matches_manual_cross_entropy():
    torch.manual_seed(0)
    pos = torch.randn(5)
    neg = torch.randn(5, 7)
    got = softmax_loss(pos, neg)

    # manual: cross entropy between [pos, logsumexp(neg)] and label 0, summed
    scores = torch.stack([pos, neg.logsumexp(dim=1)], dim=1)
    log_softmax = scores - scores.logsumexp(dim=1, keepdim=True)
    manual = -log_softmax[:, 0].sum()
    assert torch.allclose(got, manual, atol=1e-5)


def test_softmax_loss_zero_negatives_is_zero():
    pos = torch.randn(3)
    neg = torch.zeros(3, 0)
    assert float(softmax_loss(pos, neg)) == 0.0


def test_ranking_and_logistic_losses_match_torch_functional():
    torch.manual_seed(1)
    pos = torch.randn(4)
    neg = torch.randn(4, 3)
    got = ranking_loss(pos, neg, margin=0.1)
    manual = torch.nn.functional.margin_ranking_loss(
        neg, pos.unsqueeze(1), target=pos.new_full((1, 1), -1.0), margin=0.1, reduction="sum")
    assert torch.allclose(got, manual)

    got2 = logistic_loss(pos, neg)
    pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pos, torch.ones_like(pos), reduction="sum")
    neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        neg, torch.zeros_like(neg), reduction="sum")
    manual2 = pos_loss + neg_loss / neg.shape[1]
    assert torch.allclose(got2, manual2)


def _manual_forward(C, G, src, dst, num_batch_negs, num_uniform_negs=0, uniform_idx=None):
    """Hand-rolled re-derivation of KGEModule.forward for the (operator='none',
    dot comparator) case, independent of the implementation under test."""
    num_pos = len(src)
    chunk_size = min(num_batch_negs, num_pos) if num_batch_negs > 0 else num_pos
    num_chunks = math.ceil(num_pos / chunk_size)
    pad = num_chunks * chunk_size - num_pos
    lhs = C[src]
    rhs = G[dst]
    if pad:
        lhs = torch.cat([lhs, lhs.new_zeros(pad, lhs.size(1))], 0)
        rhs = torch.cat([rhs, rhs.new_zeros(pad, rhs.size(1))], 0)
    lhs_c = lhs.view(num_chunks, chunk_size, -1)
    rhs_c = rhs.view(num_chunks, chunk_size, -1)
    pos_scores = (lhs_c * rhs_c).sum(-1).reshape(-1)[:num_pos]

    lhs_neg = lhs_c
    rhs_neg = rhs_c
    if num_uniform_negs > 0:
        lhs_uni = C[uniform_idx[0]]
        rhs_uni = G[uniform_idx[1]]
        lhs_neg = torch.cat([lhs_c, lhs_uni], dim=1)
        rhs_neg = torch.cat([rhs_c, rhs_uni], dim=1)

    lhs_neg_scores = torch.bmm(rhs_c, lhs_neg.transpose(-1, -2))
    rhs_neg_scores = torch.bmm(lhs_c, rhs_neg.transpose(-1, -2))
    diag = torch.arange(chunk_size)
    lhs_neg_scores[:, diag, diag] = -1e9
    rhs_neg_scores[:, diag, diag] = -1e9
    if pad:
        last_valid = chunk_size - pad
        lhs_neg_scores[-1, :last_valid, last_valid:chunk_size] = -1e9
        rhs_neg_scores[-1, :last_valid, last_valid:chunk_size] = -1e9
    lhs_neg_scores = lhs_neg_scores.reshape(num_chunks * chunk_size, -1)[:num_pos]
    rhs_neg_scores = rhs_neg_scores.reshape(num_chunks * chunk_size, -1)[:num_pos]
    return pos_scores, lhs_neg_scores, rhs_neg_scores


def test_forward_batch_negatives_only_matches_hand_derivation():
    torch.manual_seed(0)
    kge = KGEModule({"C": 4, "G": 4}, [{"lhs": "C", "rhs": "G", "operator": "none"}], dimension=3)
    C = kge.embeddings["C"].detach().clone()
    G = kge.embeddings["G"].detach().clone()
    src = torch.tensor([0, 1, 2, 3])
    dst = torch.tensor([0, 1, 2, 3])

    pos, lneg, rneg = kge(0, src, dst, num_batch_negs=2, num_uniform_negs=0)
    mpos, mlneg, mrneg = _manual_forward(C, G, src, dst, num_batch_negs=2, num_uniform_negs=0)
    assert torch.allclose(pos, mpos)
    assert torch.allclose(lneg, mlneg)
    assert torch.allclose(rneg, mrneg)
    # self-comparisons must be masked
    assert (lneg == -1e9).sum() == 4  # one per positive (2 chunks x 2 diag entries)
    assert (rneg == -1e9).sum() == 4


def test_forward_ragged_last_chunk_pads_and_masks_correctly():
    torch.manual_seed(2)
    kge = KGEModule({"C": 5, "G": 5}, [{"lhs": "C", "rhs": "G", "operator": "none"}], dimension=4)
    src = torch.tensor([0, 1, 2, 3, 4])
    dst = torch.tensor([0, 1, 2, 3, 4])
    pos, lneg, rneg = kge(0, src, dst, num_batch_negs=3, num_uniform_negs=0)
    assert pos.shape == (5,)
    assert lneg.shape == (5, 3)
    # last chunk has 2 valid rows (indices 3,4) + 1 padding slot;
    # the padding column must be masked for the valid rows of that chunk.
    assert float(lneg[3, 2]) == -1e9
    assert float(lneg[4, 2]) == -1e9


def test_forward_with_uniform_negatives_matches_hand_derivation_given_same_indices(monkeypatch):
    torch.manual_seed(3)
    kge = KGEModule({"C": 10, "G": 10}, [{"lhs": "C", "rhs": "G", "operator": "none"}], dimension=4)
    C = kge.embeddings["C"].detach().clone()
    G = kge.embeddings["G"].detach().clone()
    src = torch.tensor([0, 1, 2, 3])
    dst = torch.tensor([0, 1, 2, 3])

    captured = {}
    real_randint = torch.randint

    def fake_randint(low, high, size, **kwargs):
        out = real_randint(low, high, size, **kwargs)
        captured.setdefault("calls", []).append(out)
        return out

    monkeypatch.setattr(torch, "randint", fake_randint)
    pos, lneg, rneg = kge(0, src, dst, num_batch_negs=2, num_uniform_negs=2)
    lhs_uni_idx, rhs_uni_idx = captured["calls"][0], captured["calls"][1]

    mpos, mlneg, mrneg = _manual_forward(C, G, src, dst, num_batch_negs=2, num_uniform_negs=2,
                                         uniform_idx=(lhs_uni_idx, rhs_uni_idx))
    assert torch.allclose(pos, mpos)
    assert torch.allclose(lneg, mlneg)
    assert torch.allclose(rneg, mrneg)


def test_l2_norm_sums_all_entity_tables():
    kge = KGEModule({"C": 3, "G": 2}, [{"lhs": "C", "rhs": "G", "operator": "none"}], dimension=2)
    manual = kge.embeddings["C"].pow(2).sum() + kge.embeddings["G"].pow(2).sum()
    assert torch.allclose(kge.l2_norm(), manual)
