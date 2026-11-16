"""Parity benchmark (a): single-sample 10x PBMC 3k (CG graph).

ov.ge.SIMBA vs. the original simba+simba_pbg reference run under
/scratch/users/steorra/tmp/simba_parity/ref/single/ (see its README.md).
SPEC_ge_simba.md section 3.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (REF, PBMC3K_MARKERS, alias_layer, ari_nmi_leiden, jaccard,
                    knn_label_accuracy, ref_eval_trajectory, ref_final_eval,
                    seed_everything, set_threads, top_k_genes)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "results_single")
SEEDS = (1, 2, 3)


def check_graph(graph, adata) -> dict:
    ref_stats = json.load(open(f"{REF}/single/shared/graph_stats.json"))["graph0"]
    ref_edges = [ref_stats[f"relation{i}"]["n_edges"] for i in range(5)]
    ref_total = ref_stats["n_edges"]

    got = graph.edges_per_relation()
    # both orderings are ascending-weight (expr level 1..5), so position-wise
    # comparison is valid without re-deriving a name mapping.
    got_edges = [got[r["name"]] for r in sorted(graph.relations, key=lambda r: r["weight"])]

    report = {
        "n_entities_C": graph.n_entities("C"), "ref_n_entities_C": adata.n_obs,
        "n_entities_G": graph.n_entities("G"), "ref_n_entities_G": int(adata.var["highly_variable"].sum()),
        "n_relations": len(graph.relations), "ref_n_relations": 5,
        "edges_per_relation": got_edges, "ref_edges_per_relation": ref_edges,
        "n_edges": graph.n_edges, "ref_n_edges": ref_total,
        "edges_match": got_edges == ref_edges and graph.n_edges == ref_total,
        "auto_wd": graph.auto_wd(), "ref_wd": 0.049186,
    }
    report["wd_match"] = abs(report["auto_wd"] - report["ref_wd"]) < 1e-6
    return report


def per_celltype_marker_jaccard(cell_emb, gene_emb, labels, k=50):
    import omicverse as ov

    out = {}
    for ct in sorted(pd.unique(labels)):
        mask = labels == ct
        if mask.sum() < 5:
            continue
        sub_ref = cell_emb[np.asarray(mask)]
        cmp = ov.ge.compare_entities(sub_ref, gene_emb)
        top = set(top_k_genes(cmp, "max", k=k))
        markers = set(PBMC3K_MARKERS.get(ct, []))
        present = markers & set(cmp.var_names)
        ranks = pd.Series(np.asarray(cmp.var["max"]), index=cmp.var_names).rank(ascending=False)
        out[ct] = {
            "n_cells": int(mask.sum()),
            "markers_in_hvg": sorted(present), "markers_not_in_hvg": sorted(markers - present),
            "top_k_contains_all_present_markers": present.issubset(top) if present else None,
            "marker_ranks": {g: int(ranks[g]) for g in present},
        }
    return out


def main():
    set_threads()
    os.makedirs(OUTDIR, exist_ok=True)
    import anndata
    import omicverse as ov

    adata = anndata.read_h5ad(f"{REF}/single/shared/preprocessed.h5ad")
    alias_layer(adata, "disc", "simba")

    graph = ov.ge.gen_graph(list_CG=[adata], use_highly_variable=True)
    graph_report = check_graph(graph, adata)
    print("graph check:", json.dumps(graph_report, indent=1))

    ref_cmp = anndata.read_h5ad(f"{REF}/single/seed1/compare_entities_genes.h5ad")

    results = {"graph": graph_report, "seeds": []}
    for seed in SEEDS:
        seed_everything(seed)
        model = ov.ge.SIMBA(dimension=50)
        model.graph = graph
        model.build()
        t0 = time.time()
        model.train(num_epochs=10, batch_size=1000, lr=0.1, seed=seed, auto_wd=True,
                   num_batch_negs=50, num_uniform_negs=50, loss_fn="softmax",
                   eval_fraction=0.05, device="cpu", progress=False)
        train_time = time.time() - t0

        cell_emb = model.get_embedding("C")
        gene_emb = model.get_embedding("G")
        labels_s = adata.obs.loc[cell_emb.obs_names, "celltype"].astype(str)

        ari_nmi = ari_nmi_leiden(cell_emb.X, labels_s.values, seed=seed)
        knn_acc = knn_label_accuracy(cell_emb.X, labels_s.values, seed=seed)

        cmp = ov.ge.compare_entities(cell_emb, gene_emb)
        shared_genes = cmp.var_names.intersection(ref_cmp.var_names)
        corr = {}
        for metric in ("max", "std", "gini", "entropy"):
            a = np.asarray(cmp[:, shared_genes].var[metric])
            b = np.asarray(ref_cmp[:, shared_genes].var[metric])
            corr[metric] = float(np.corrcoef(a, b)[0, 1])
        jacc_top50 = {m: jaccard(top_k_genes(cmp, m, 50), top_k_genes(ref_cmp, m, 50))
                     for m in ("max", "std", "gini", "entropy")}

        markers = per_celltype_marker_jaccard(cell_emb, gene_emb, labels_s.values)

        ref_eval = ref_final_eval(f"{REF}/single/seed{seed}/training_stats.json")

        rec = {
            "seed": seed, "train_time_sec": train_time,
            "ari": ari_nmi["ari"], "nmi": ari_nmi["nmi"], "knn_acc": knn_acc,
            "eval_mrr": model.history_["eval_mrr"][-1], "eval_loss": model.history_["eval_loss"][-1],
            "ref_eval_mrr": ref_eval["mrr"], "ref_eval_loss": ref_eval["loss"],
            "compare_entities_corr": corr, "compare_entities_top50_jaccard": jacc_top50,
            "markers": markers,
        }
        print(f"seed {seed}: time={train_time:.1f}s ari={rec['ari']:.3f} nmi={rec['nmi']:.3f} "
              f"knn_acc={knn_acc:.3f} mrr={rec['eval_mrr']:.4f} (ref {ref_eval['mrr']:.4f})")
        results["seeds"].append(rec)

    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("wrote", os.path.join(OUTDIR, "results.json"))


if __name__ == "__main__":
    main()
