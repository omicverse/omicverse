"""GPU timing only (one seed each, ``device='cuda'``) for the parity
benchmark's CPU-vs-GPU wall-time comparison. Not a correctness check --
graph/embedding-quality parity is covered by run_single.py / run_batch.py."""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REF, alias_layer, seed_everything, set_threads

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "results_single")
OUTDIR_B = os.path.join(HERE, "..", "results_batch")


def main():
    set_threads()
    import anndata
    import torch
    import omicverse as ov

    assert torch.cuda.is_available()
    out = {}

    # --- single (CG) ---
    adata = anndata.read_h5ad(f"{REF}/single/shared/preprocessed.h5ad")
    alias_layer(adata, "disc", "simba")
    graph = ov.ge.gen_graph(list_CG=[adata], use_highly_variable=True)
    seed_everything(1)
    model = ov.ge.SIMBA(dimension=50, device="cuda")
    model.graph = graph
    model.build()
    torch.cuda.synchronize()
    t0 = time.time()
    model.train(num_epochs=10, batch_size=1000, lr=0.1, seed=1, auto_wd=True,
               num_batch_negs=50, num_uniform_negs=50, loss_fn="softmax",
               eval_fraction=0.05, device="cuda", progress=False)
    torch.cuda.synchronize()
    out["single_gpu_sec"] = time.time() - t0
    print("single GPU:", out["single_gpu_sec"], "s; peak mem (GB):",
         torch.cuda.max_memory_allocated() / 1e9)
    del model, graph
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- batch (CG+CC) ---
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
    graph_b = ov.ge.gen_graph(list_CG=[adatas["s3d7"], adatas["s2d1"], adatas["s1d3"]],
                              list_CC=[cc_s2d1, cc_s1d3], use_highly_variable=True)
    seed_everything(1)
    model_b = ov.ge.SIMBA(dimension=50, device="cuda")
    model_b.graph = graph_b
    model_b.build()
    torch.cuda.synchronize()
    t0 = time.time()
    model_b.train(num_epochs=10, batch_size=1000, lr=0.1, seed=1, auto_wd=True,
                 num_batch_negs=50, num_uniform_negs=50, loss_fn="softmax",
                 eval_fraction=0.05, device="cuda", progress=False)
    torch.cuda.synchronize()
    out["batch_gpu_sec"] = time.time() - t0
    print("batch GPU:", out["batch_gpu_sec"], "s; peak mem (GB):",
         torch.cuda.max_memory_allocated() / 1e9)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "gpu_timing.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
