"""Parity benchmark (b): 3-donor NeurIPS 2021 BMMC RNA batch integration
(CG+CC graph).

ov.ge.SIMBA vs. the original simba+simba_pbg reference run under
/scratch/users/steorra/tmp/simba_parity/ref/batch/ (see its README.md).
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
from common import (REF, alias_layer, ari_nmi_leiden, ilisi, jaccard, kbet_score,
                    knn_label_accuracy, ref_final_eval, seed_everything, set_threads,
                    top_k_genes)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "results_batch")
SEEDS = (1, 2, 3)
DONORS = ("s3d7", "s2d1", "s1d3")  # s3d7 = reference (largest), s2d1/s1d3 = query
DONOR_TYPE = {"s3d7": "C", "s2d1": "C2", "s1d3": "C3"}


def check_graph(graph, adatas) -> dict:
    ref_stats = json.load(open(f"{REF}/batch/shared/graph_stats.json"))["graph_batch"]
    ref_total = ref_stats["n_edges"]
    ref_edges = [ref_stats[f"relation{i}"]["n_edges"] for i in range(17)]

    # gen_graph processes list_CG then list_CC in call order, same as the
    # reference's own gen_graph -- r0..r16 line up by position, no resorting
    # needed.
    got = graph.edges_per_relation()
    got_edges = [got[f"r{i}"] for i in range(len(graph.relations))]
    return {
        "n_entities": {k: graph.n_entities(k) for k in ("C", "C2", "C3", "G")},
        "ref_n_entities": {"C": 8000, "C2": 8000, "C3": 5935},
        "n_relations": len(graph.relations), "ref_n_relations": 17,
        "edges_per_relation": got_edges, "ref_edges_per_relation": ref_edges,
        "n_edges": graph.n_edges, "ref_n_edges": ref_total,
        "edges_match": got_edges == ref_edges and graph.n_edges == ref_total,
        "auto_wd": graph.auto_wd(), "ref_wd": 0.005578,
        "wd_match": abs(graph.auto_wd() - 0.005578) < 1e-6,
    }


def main():
    set_threads()
    os.makedirs(OUTDIR, exist_ok=True)
    import anndata
    import omicverse as ov

    adatas = {}
    for d in DONORS:
        a = anndata.read_h5ad(f"{REF}/batch/shared/preprocessed_{d}.h5ad")
        alias_layer(a, "disc", "simba")
        adatas[d] = a

    cc_s2d1 = anndata.read_h5ad(f"{REF}/batch/shared/CC_s3d7_vs_s2d1.h5ad")
    cc_s1d3 = anndata.read_h5ad(f"{REF}/batch/shared/CC_s3d7_vs_s1d3.h5ad")
    alias_layer(cc_s2d1, "conn", "simba")
    alias_layer(cc_s1d3, "conn", "simba")

    graph = ov.ge.gen_graph(list_CG=[adatas["s3d7"], adatas["s2d1"], adatas["s1d3"]],
                            list_CC=[cc_s2d1, cc_s1d3], use_highly_variable=True)
    graph_report = check_graph(graph, adatas)
    print("graph check:", json.dumps(graph_report, indent=1))

    ref_cmp = anndata.read_h5ad(f"{REF}/batch/seed1/compare_entities_genes.h5ad")

    # concatenated donor labels for integration metrics (batch = donor, bio = cell_type)
    cell_type = pd.concat([adatas[d].obs["cell_type"].astype(str) for d in DONORS])
    donor_of = pd.concat([pd.Series(d, index=adatas[d].obs_names) for d in DONORS])

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

        gene_emb = model.get_embedding("G")
        ref_emb = model.get_embedding("C")
        integrated = model.embed_query("C", ["C2", "C3"])  # obs['id_dataset'] in {ref,query_0,query_1}

        ids = integrated.obs_names
        labels_bio = cell_type.loc[ids].values
        labels_batch = donor_of.loc[ids].values

        ari_nmi = ari_nmi_leiden(integrated.X, labels_bio, seed=seed)
        knn_acc = knn_label_accuracy(integrated.X, labels_bio, seed=seed)
        ilisi_v = ilisi(integrated.X, labels_batch)
        kbet_v = kbet_score(integrated.X, labels_batch)

        cmp = ov.ge.compare_entities(ref_emb, gene_emb)
        shared_genes = cmp.var_names.intersection(ref_cmp.var_names)
        corr = {}
        for metric in ("max", "std", "gini", "entropy"):
            a = np.asarray(cmp[:, shared_genes].var[metric])
            b = np.asarray(ref_cmp[:, shared_genes].var[metric])
            corr[metric] = float(np.corrcoef(a, b)[0, 1])
        jacc_top50 = {m: jaccard(top_k_genes(cmp, m, 50), top_k_genes(ref_cmp, m, 50))
                     for m in ("max", "std", "gini", "entropy")}

        ref_eval = ref_final_eval(f"{REF}/batch/seed{seed}/training_stats.json")

        rec = {
            "seed": seed, "train_time_sec": train_time,
            "ari": ari_nmi["ari"], "nmi": ari_nmi["nmi"], "knn_acc": knn_acc,
            "ilisi": ilisi_v, "kbet": kbet_v,
            "eval_mrr": model.history_["eval_mrr"][-1], "eval_loss": model.history_["eval_loss"][-1],
            "ref_eval_mrr": ref_eval["mrr"], "ref_eval_loss": ref_eval["loss"],
            "compare_entities_corr": corr, "compare_entities_top50_jaccard": jacc_top50,
        }
        print(f"seed {seed}: time={train_time:.1f}s ari={rec['ari']:.3f} nmi={rec['nmi']:.3f} "
              f"knn_acc={knn_acc:.3f} iLISI={ilisi_v:.3f} kBET={kbet_v:.3f} "
              f"mrr={rec['eval_mrr']:.4f} (ref {ref_eval['mrr']:.4f})")
        results["seeds"].append(rec)

    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("wrote", os.path.join(OUTDIR, "results.json"))


if __name__ == "__main__":
    main()
