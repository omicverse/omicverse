"""ARI/NMI/kNN-accuracy computed on the *reference's own* trained cell
embeddings (single task), same metric code as run_single.py -- fills the
gap flagged in review: the single-task table had no reference ARI/NMI/kNN
to compare against."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REF, ari_nmi_leiden, knn_label_accuracy, set_threads


def main():
    set_threads()
    import anndata

    adata = anndata.read_h5ad(f"{REF}/single/shared/preprocessed.h5ad")
    out = []
    for seed in (1, 2, 3):
        emb = anndata.read_h5ad(f"{REF}/single/seed{seed}/cell_embedding.h5ad")
        labels = adata.obs.loc[emb.obs_names, "celltype"].astype(str).values
        res = ari_nmi_leiden(emb.X, labels, seed=seed)
        knn_acc = knn_label_accuracy(emb.X, labels, seed=seed)
        rec = {"seed": seed, "ari": res["ari"], "nmi": res["nmi"], "knn_acc": knn_acc}
        print(rec)
        out.append(rec)

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_single")
    with open(os.path.join(outdir, "ref_cluster_metrics.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
