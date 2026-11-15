"""Train ov.vi.MRVI on the prepped Wilk 2020 COVID PBMC cohort and dump the metrics /
outputs used for the ours-vs-scvi-tools comparison (common.py has the shared config).

Usage: python run_ours.py <seed> <out_prefix> [cpu|cuda]
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from common import DATA, SAMPLE_KEY, LABELS_KEY, COV_KEY, MODEL_KW, OURS_TRAIN_KW as TRAIN_KW  # noqa: E402


def main():
    seed = int(sys.argv[1])
    out = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"

    import anndata as ad
    sys.path.insert(0, "../../..")  # the worktree root, for `import omicverse`
    from omicverse.vi import MRVI

    a = ad.read_h5ad(DATA)
    torch.manual_seed(seed)
    t0 = time.time()
    m = MRVI(a, sample_key=SAMPLE_KEY, layer="counts", seed=seed, **MODEL_KW)
    m.train(device=device, seed=seed, progress=False, **TRAIN_KW)
    train_time = time.time() - t0

    elbo_train = m.history_["elbo_train"][-1]
    elbo_val = m.history_.get("elbo_validation", [np.nan])[-1]

    u = m.get_latent_representation()
    z = m.get_latent_representation(give_z=True)

    torch.manual_seed(seed + 1000)
    dist = m.get_local_sample_distances(batch_size=512, groupby=LABELS_KEY, keep_cell=False)
    da = m.differential_abundance(sample_cov_keys=[COV_KEY])
    de = m.differential_expression(sample_cov_keys=[COV_KEY], mc_samples=25, store_lfc=True,
                                   mc_samples_lfc=15, batch_size=256)

    np.savez(out + ".npz", u=u, z=z, elbo_train=elbo_train, elbo_val=elbo_val,
            train_time=train_time, sample_order=np.asarray(m.sample_order),
            cell_type=a.obs[LABELS_KEY].astype(str).values,
            disease=a.obs[COV_KEY].astype(str).values,
            donor=a.obs[SAMPLE_KEY].astype(str).values,
            group_dist_keys=np.asarray(list(dist["groups"][LABELS_KEY].keys())),
            group_dist=np.stack(list(dist["groups"][LABELS_KEY].values())),
            da_log_probs=da["log_probs"], da_samples=np.asarray(da["samples"]),
            de_pvalue=de["pvalue"], de_padj=de["padj"], de_effect=de["effect_size"],
            de_lfc_mean_gene=de["lfc"][:, 0, :].mean(0), de_genes=de["genes"])
    print(f"ours seed={seed}: elbo_train={elbo_train:.3f} elbo_val={elbo_val:.3f} "
         f"train_time={train_time:.1f}s")


if __name__ == "__main__":
    main()
