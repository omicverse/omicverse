"""A tiny ``fit()`` run improves held-out-edge MRR.

Uses a controlled block-structured bipartite graph (clusters of cells each
connected to their own cluster of genes) rather than a real dataset: this
is a mechanism test of the optimiser/training-loop/eval-metric triangle
(does the loss landscape the model is actually optimising correspond to
what ``evaluate()`` measures), not a biology demonstration -- those belong
in the tutorial / parity benchmark, not here. With real noisy counts at
this tiny a scale (tens of cells) the signal-to-negative-sampling-noise
ratio is too low for the lift to be reliably >0 run to run; a clean block
structure makes the "should be learnable" property unambiguous while
staying fast on CPU.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omicverse.ge._base import BaseGraphModel
from omicverse.ge._graph import HeteroGraph
from omicverse.ge._train import evaluate


def _block_graph(n_blocks=10, block_size=10, degree=6, seed=0):
    rng = np.random.default_rng(seed)
    n = n_blocks * block_size
    src, dst = [], []
    for b in range(n_blocks):
        cells = np.arange(b * block_size, (b + 1) * block_size)
        genes = np.arange(b * block_size, (b + 1) * block_size)
        for c in cells:
            chosen = rng.choice(genes, size=min(degree, block_size), replace=False)
            src.extend([c] * len(chosen))
            dst.extend(chosen.tolist())
    g = HeteroGraph()
    g.add_entity_type("C", np.arange(n))
    g.add_entity_type("G", np.arange(n))
    g.add_edges("C", "G", np.array(src), np.array(dst), name="r0", weight=1.0)
    g.build()
    return g


def test_training_improves_held_out_mrr():
    torch.manual_seed(0)
    graph = _block_graph(n_blocks=10, block_size=10, degree=6, seed=0)
    model = BaseGraphModel(dimension=8)
    model.graph = graph
    model.build()

    train_idx, eval_idx = graph.split(0.15, seed=0)
    rel_weights = torch.as_tensor(graph.relation_weights(), dtype=torch.float32)
    baseline = evaluate(graph, model.module, eval_idx, batch_size=200, rel_weights=rel_weights,
                        loss_fn="softmax", num_batch_negs=20, num_uniform_negs=20, device="cpu")

    model.train(num_epochs=60, batch_size=200, seed=0, progress=False, auto_wd=False, wd=0.0,
               eval_fraction=0.15, num_batch_negs=20, num_uniform_negs=20, lr=0.1)

    final_mrr = float(np.mean(model.history_["eval_mrr"][-5:]))
    final_loss = float(np.mean(model.history_["train_loss"][-5:]))

    assert final_mrr > baseline["mrr"] * 1.5, (
        f"expected held-out MRR to improve well above baseline "
        f"({baseline['mrr']:.4f}), got {final_mrr:.4f}")
    assert final_loss < model.history_["train_loss"][0]


def test_training_loss_decreases_monotonically_on_average():
    torch.manual_seed(1)
    graph = _block_graph(n_blocks=6, block_size=8, degree=4, seed=1)
    model = BaseGraphModel(dimension=6)
    model.graph = graph
    model.build()
    model.train(num_epochs=20, batch_size=150, seed=1, progress=False, auto_wd=False, wd=0.0)
    losses = model.history_["train_loss"]
    # not strictly monotone (minibatch SGD), but first-half mean should beat
    # a no-training baseline (first epoch) comfortably
    assert np.mean(losses[-5:]) < losses[0]
