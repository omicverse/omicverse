"""Decisive check (review round 3): train ov.ge.SIMBA 3 seeds against
PyTorch-BigGraph's *exact* fixed holdout (same edges excluded from
training for all 10 epochs, same as the reference run), then evaluate on
that same set. Settles whether the small residual MRR gap vs. the
reference is "training variance" or a real, fixable training-code
difference.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REF, alias_layer, seed_everything, set_threads

HERE = os.path.dirname(os.path.abspath(__file__))


def run_single():
    import anndata
    import omicverse as ov

    adata = anndata.read_h5ad(f"{REF}/single/shared/preprocessed.h5ad")
    alias_layer(adata, "disc", "simba")
    graph = ov.ge.gen_graph(list_CG=[adata], use_highly_variable=True)
    graph.build()
    holdout = graph.split(0.05, method="pbg")  # fixed, shared across seeds (as PBG does)

    out = []
    for seed in (1, 2, 3):
        seed_everything(seed)
        model = ov.ge.SIMBA(dimension=50)
        model.graph = graph
        model.build()
        t0 = time.time()
        model.train(num_epochs=10, batch_size=1000, lr=0.1, seed=seed, auto_wd=True,
                   num_batch_negs=50, num_uniform_negs=50, loss_fn="softmax",
                   device="cpu", progress=False, holdout_idx=holdout)
        dt = time.time() - t0
        rec = {"seed": seed, "train_time_sec": dt,
              "eval_loss": model.history_["eval_loss"][-1], "eval_mrr": model.history_["eval_mrr"][-1]}
        print("single", rec)
        out.append(rec)
    return out


def run_batch():
    import anndata
    import omicverse as ov

    donors = ("s3d7", "s2d1", "s1d3")
    adatas = {}
    for d in donors:
        a = anndata.read_h5ad(f"{REF}/batch/shared/preprocessed_{d}.h5ad")
        alias_layer(a, "disc", "simba")
        adatas[d] = a
    cc_s2d1 = anndata.read_h5ad(f"{REF}/batch/shared/CC_s3d7_vs_s2d1.h5ad")
    cc_s1d3 = anndata.read_h5ad(f"{REF}/batch/shared/CC_s3d7_vs_s1d3.h5ad")
    alias_layer(cc_s2d1, "conn", "simba")
    alias_layer(cc_s1d3, "conn", "simba")
    graph = ov.ge.gen_graph(list_CG=[adatas["s3d7"], adatas["s2d1"], adatas["s1d3"]],
                            list_CC=[cc_s2d1, cc_s1d3], use_highly_variable=True)
    graph.build()
    holdout = graph.split(0.05, method="pbg")

    out = []
    for seed in (1, 2, 3):
        seed_everything(seed)
        model = ov.ge.SIMBA(dimension=50)
        model.graph = graph
        model.build()
        t0 = time.time()
        model.train(num_epochs=10, batch_size=1000, lr=0.1, seed=seed, auto_wd=True,
                   num_batch_negs=50, num_uniform_negs=50, loss_fn="softmax",
                   device="cpu", progress=False, holdout_idx=holdout)
        dt = time.time() - t0
        rec = {"seed": seed, "train_time_sec": dt,
              "eval_loss": model.history_["eval_loss"][-1], "eval_mrr": model.history_["eval_mrr"][-1]}
        print("batch", rec)
        out.append(rec)
    return out


def main():
    set_threads()
    results = {"single": run_single(), "batch": run_batch()}
    with open(os.path.join(HERE, "..", "results_pbg_holdout_train.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
