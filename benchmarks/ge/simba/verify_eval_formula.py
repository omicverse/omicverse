"""Root-cause check for the apparent held-out-MRR gap (review round 2):
load the REFERENCE's own trained embeddings and score them with
``ov.ge``'s ``evaluate()`` on the **exact same held-out edge set PBG used
during training** (PBG derives it deterministically:
``torch.Generator().manual_seed(hash((edge_path_idx, edge_chunk_idx,
bucket.lhs, bucket.rhs)))`` -- for this single-bucket/single-chunk data
that's always ``hash((0, 0, 0, 0))`` -- then samples
``torch.randint(num_edges, (num_eval_edges,), generator=g)`` **with
replacement** over the edges in on-disk/construction order).

If ``ov.ge.evaluate()`` reproduces PBG's self-reported final-epoch loss/MRR
on that exact edge set, the metric *formula* is confirmed correct, and any
gap against an *arbitrary* different held-out set (e.g. ``ov.ge``'s own
disjoint 5% split) is not a bug -- it is comparing performance on two
different edge sets, not two different metrics. See results_simba.md
section 6 for what this resolved.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REF, alias_layer, set_threads


def pbg_style_eval_idx(num_edges: int, eval_fraction: float = 0.05) -> np.ndarray:
    num_eval_edges = int(num_edges * eval_fraction)
    g = torch.Generator()
    g.manual_seed(hash((0, 0, 0, 0)))  # edge_path_idx=edge_chunk_idx=bucket.lhs=bucket.rhs=0
    return torch.randint(num_edges, (num_eval_edges,), dtype=torch.long, generator=g).numpy()


def main():
    set_threads()
    import anndata
    import omicverse as ov
    from omicverse.ge._train import evaluate

    results = {}

    # ---- single ----
    adata = anndata.read_h5ad(f"{REF}/single/shared/preprocessed.h5ad")
    alias_layer(adata, "disc", "simba")
    graph = ov.ge.gen_graph(list_CG=[adata], use_highly_variable=True)
    graph.build()
    cell_emb = anndata.read_h5ad(f"{REF}/single/seed1/cell_embedding.h5ad")
    gene_emb = anndata.read_h5ad(f"{REF}/single/seed1/gene_embedding.h5ad")
    kge = ov.ge.KGEModule({"C": graph.n_entities("C"), "G": graph.n_entities("G")},
                          graph.relations, dimension=50)
    with torch.no_grad():
        kge.embeddings["C"].copy_(torch.from_numpy(cell_emb[graph.entity_ids["C"]].X.astype(np.float32)))
        kge.embeddings["G"].copy_(torch.from_numpy(gene_emb[graph.entity_ids["G"]].X.astype(np.float32)))
    eval_idx = pbg_style_eval_idx(graph.n_edges)
    rel_weights = torch.as_tensor(graph.relation_weights(), dtype=torch.float32)
    res = evaluate(graph, kge, eval_idx, batch_size=1000, rel_weights=rel_weights,
                   loss_fn="softmax", num_batch_negs=50, num_uniform_negs=50, device="cpu")
    print("single, PBG's exact held-out set, ov.ge.evaluate():", res, "| ref self-report: loss=21.391, mrr=0.0724")
    results["single"] = {"ov_ge_on_pbg_holdout": res, "ref_self_report": {"loss": 21.391, "mrr": 0.0724}}

    # ---- batch ----
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
    graph_b.build()
    emb_s3d7 = anndata.read_h5ad(f"{REF}/batch/seed1/embedding_s3d7.h5ad")
    emb_s2d1 = anndata.read_h5ad(f"{REF}/batch/seed1/embedding_s2d1.h5ad")
    emb_s1d3 = anndata.read_h5ad(f"{REF}/batch/seed1/embedding_s1d3.h5ad")
    emb_genes = anndata.read_h5ad(f"{REF}/batch/seed1/embedding_genes.h5ad")
    counts = {k: graph_b.n_entities(k) for k in ("C", "C2", "C3", "G")}
    kge_b = ov.ge.KGEModule(counts, graph_b.relations, dimension=50)
    with torch.no_grad():
        kge_b.embeddings["C"].copy_(torch.from_numpy(emb_s3d7[graph_b.entity_ids["C"]].X.astype(np.float32)))
        kge_b.embeddings["C2"].copy_(torch.from_numpy(emb_s2d1[graph_b.entity_ids["C2"]].X.astype(np.float32)))
        kge_b.embeddings["C3"].copy_(torch.from_numpy(emb_s1d3[graph_b.entity_ids["C3"]].X.astype(np.float32)))
        kge_b.embeddings["G"].copy_(torch.from_numpy(emb_genes[graph_b.entity_ids["G"]].X.astype(np.float32)))
    eval_idx_b = pbg_style_eval_idx(graph_b.n_edges)
    rel_weights_b = torch.as_tensor(graph_b.relation_weights(), dtype=torch.float32)
    res_b = evaluate(graph_b, kge_b, eval_idx_b, batch_size=1000, rel_weights=rel_weights_b,
                     loss_fn="softmax", num_batch_negs=50, num_uniform_negs=50, device="cpu")
    print("batch, PBG's exact held-out set, ov.ge.evaluate():", res_b, "| ref self-report: loss=21.060, mrr=0.1231")
    results["batch"] = {"ov_ge_on_pbg_holdout": res_b, "ref_self_report": {"loss": 21.060, "mrr": 0.1231}}

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    with open(os.path.join(outdir, "results_eval_formula_check.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
