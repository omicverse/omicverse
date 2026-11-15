"""Compare ``ov.vi.DRVI`` (native) against ``drvi-py``'s ``DRVI`` (reference) on the
``train_native.py`` / ``train_reference.py`` outputs: scib-metrics integration
scores and the DRVI-paper-style per-dimension disentanglement score (each latent
dimension's best association with cell type / batch, aggregated with
``drvi-py``'s own ``most_similar_averaging_score`` / ``latent_matching_score``).

Usage: python validate.py [--dir RESULTS_DIR] [--seeds 0 1 2] [--tag memory]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def assoc_matrix(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``(n_latent, n_categories)``: ``|Spearman rho|`` of each latent dimension
    against the one-vs-rest indicator of each category (the per-dimension
    "association with a covariate" used in the DRVI paper's disentanglement
    figures)."""
    cats = pd.unique(labels)
    out = np.zeros((z.shape[1], len(cats)))
    for j, c in enumerate(cats):
        y = (labels == c).astype(float)
        for i in range(z.shape[1]):
            out[i, j] = abs(spearmanr(z[:, i], y).correlation)
    return out


def disentanglement_scores(z, cell_type, batch) -> dict:
    from drvi.utils.metrics._aggregation import latent_matching_score, most_similar_averaging_score
    m_ct = assoc_matrix(z, cell_type)
    m_bt = assoc_matrix(z, batch)
    return {
        "msas_cell_type": most_similar_averaging_score(m_ct),  # mean best-dim |rho| per cell type
        "lms_cell_type": latent_matching_score(m_ct),          # one-to-one dim<->cell-type matching
        "msas_batch": most_similar_averaging_score(m_bt),
        "lms_batch": latent_matching_score(m_bt),
    }


def scib_scores(z, cell_type, batch) -> dict:
    import scib_metrics
    from scib_metrics.nearest_neighbors import pynndescent

    neigh = pynndescent(z, n_neighbors=90, random_state=0)
    nmi_ari = scib_metrics.nmi_ari_cluster_labels_leiden(neigh, cell_type, seed=0)
    return {
        "nmi": nmi_ari["nmi"],
        "ari": nmi_ari["ari"],
        "silhouette_label": scib_metrics.silhouette_label(z, cell_type),
        "silhouette_batch": scib_metrics.silhouette_batch(z, cell_type, batch),
        "ilisi_batch": scib_metrics.ilisi_knn(neigh, batch),
        "clisi_label": scib_metrics.clisi_knn(neigh, cell_type),
    }


def load(dir_, impl, seed):
    path = os.path.join(dir_, f"{impl}_seed{seed}.npz")
    d = np.load(path, allow_pickle=True)
    return d["z"], d["cell_type"], d["batch"], float(d["elbo_train"]), float(d["time_s"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="/scratch/users/steorra/tmp/vi_p2/data/drvi/results")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()

    rows = []
    for impl in ["native", "reference"]:
        for seed in args.seeds:
            path = os.path.join(args.dir, f"{impl}_seed{seed}.npz")
            if not os.path.exists(path):
                print(f"missing {path}, skipping")
                continue
            z, ct, bt, elbo, t = load(args.dir, impl, seed)
            row = {"impl": impl, "seed": seed, "elbo_train": elbo, "time_s": t}
            row.update(disentanglement_scores(z, ct, bt))
            row.update(scib_scores(z, ct, bt))
            rows.append(row)
            print(f"{impl} seed={seed} done")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    summary = df.groupby("impl").agg(["mean", "std"])
    print(summary)

    out_md = os.path.join(os.path.dirname(args.dir), "..", "results_drvi_metrics.csv")
    df.to_csv(os.path.join(args.dir, "metrics.csv"), index=False)
    print(f"wrote {os.path.join(args.dir, 'metrics.csv')}")


if __name__ == "__main__":
    main()
