"""Train scvi.external.MRVI (scvi-tools 1.5.2, torch backend) on the same prepped
Wilk 2020 cohort with the same hyperparameters/seed as run_ours.py, and dump the same
metrics for comparison.

Usage: python run_ref.py <seed> <out_prefix> [cpu|cuda]
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from common import DATA, SAMPLE_KEY, LABELS_KEY, COV_KEY, MODEL_KW, TRAIN_KW  # noqa: E402


def main():
    seed = int(sys.argv[1])
    out = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"

    import anndata as ad
    import scvi

    scvi.settings.seed = seed
    a = ad.read_h5ad(DATA)
    scvi.external.MRVI.setup_anndata(a, layer="counts", sample_key=SAMPLE_KEY)
    t0 = time.time()
    m = scvi.external.MRVI(a, **MODEL_KW)
    accel = "gpu" if device == "cuda" else "cpu"
    m.train(accelerator=accel, devices=1, early_stopping=False, enable_progress_bar=False,
           **TRAIN_KW)
    train_time = time.time() - t0

    hist = m.history
    elbo_train = float(hist["elbo_train"].iloc[-1, 0])
    elbo_val = (float(hist["elbo_validation"].iloc[-1, 0])
               if "elbo_validation" in hist else float("nan"))

    u = m.get_latent_representation(give_z=False)
    z = m.get_latent_representation(give_z=True)

    torch.manual_seed(seed + 1000)
    dist = m.get_local_sample_distances(batch_size=256, groupby=LABELS_KEY, keep_cell=False,
                                        use_vmap=False)
    group_da = dist[LABELS_KEY]
    group_keys = group_da.coords[f"{LABELS_KEY}_name"].values
    group_vals = group_da.values

    da = m.differential_abundance(sample_cov_keys=[COV_KEY])
    de = m.differential_expression(sample_cov_keys=[COV_KEY], mc_samples=25, store_lfc=True,
                                   batch_size=128, use_vmap=False)

    np.savez(out + ".npz", u=u, z=z, elbo_train=elbo_train, elbo_val=elbo_val,
            train_time=train_time, sample_order=np.asarray(m.sample_order),
            cell_type=a.obs[LABELS_KEY].astype(str).values,
            disease=a.obs[COV_KEY].astype(str).values,
            donor=a.obs[SAMPLE_KEY].astype(str).values,
            group_dist_keys=np.asarray(group_keys), group_dist=group_vals,
            da_log_probs=da.log_probs.values, da_samples=da.coords["sample"].values,
            de_pvalue=de.pvalue.values, de_padj=de.padj.values,
            de_effect=de.effect_size.values,
            de_lfc_mean_gene=de.lfc.values[:, 0, :].mean(0),
            de_genes=de.coords["gene"].values)
    print(f"ref seed={seed}: elbo_train={elbo_train:.3f} elbo_val={elbo_val:.3f} "
         f"train_time={train_time:.1f}s")


if __name__ == "__main__":
    main()
