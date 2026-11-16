"""Integration metrics (ARI/NMI/kNN-acc, iLISI, kBET) computed on the
*reference's own* `embed_integrated.h5ad` per seed -- a direct baseline
for run_batch.py's numbers on `ov.ge.SIMBA`'s integrated embedding
(the original reference run never computed these itself)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REF, ari_nmi_leiden, ilisi, kbet_score, knn_label_accuracy, set_threads

DONORS = ("s3d7", "s2d1", "s1d3")


def main():
    set_threads()
    import anndata

    cell_type = pd.concat([anndata.read_h5ad(f"{REF}/batch/shared/preprocessed_{d}.h5ad")
                           .obs["cell_type"].astype(str) for d in DONORS])
    donor_of = pd.concat([pd.Series(d, index=anndata.read_h5ad(
        f"{REF}/batch/shared/preprocessed_{d}.h5ad").obs_names) for d in DONORS])

    out = []
    for seed in (1, 2, 3):
        a = anndata.read_h5ad(f"{REF}/batch/seed{seed}/embed_integrated.h5ad")
        labels_bio = cell_type.loc[a.obs_names].values
        labels_batch = donor_of.loc[a.obs_names].values
        res = ari_nmi_leiden(a.X, labels_bio, seed=seed)
        knn_acc = knn_label_accuracy(a.X, labels_bio, seed=seed)
        il = ilisi(a.X, labels_batch)
        kb = kbet_score(a.X, labels_batch)
        rec = {"seed": seed, "ari": res["ari"], "nmi": res["nmi"], "knn_acc": knn_acc,
              "ilisi": il, "kbet": kb}
        print(rec)
        out.append(rec)

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_batch")
    with open(os.path.join(outdir, "ref_integration_metrics.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
