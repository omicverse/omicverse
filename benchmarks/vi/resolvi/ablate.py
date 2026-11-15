"""Ablate the two training-default fixes (KL warmup=20, lr_extra=1e-2) to find
which one moved marker AUROC down, on the same real Xenium subsample."""
import sys
import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, ".")
from run_benchmark import load, purity_metrics, marker_auroc  # noqa: E402

DEVICE = "cuda"
N_CELLS, EPOCHS = 8000, 20
SEEDS = [0, 1]


def run(seed, kl_warmup, lr_extra):
    import torch
    torch.manual_seed(seed)
    from omicverse.vi import RESOLVI
    a = load(N_CELLS, seed=0)
    labels = a.obs["cell_type"].to_numpy()
    m = RESOLVI(a, layer="counts", spatial_key="spatial", n_neighbors=10, mixture_k=50,
               n_hidden=32, n_hidden_encoder=128, n_layers=2, dropout_rate=0.05, seed=seed,
               lr_extra=lr_extra)
    m.train(max_epochs=EPOCHS, device=DEVICE, progress=False, batch_size=512, seed=seed,
            plan_kwargs={"n_epochs_kl_warmup": kl_warmup})
    z = m.get_latent_representation()
    den = m.get_normalized_expression(library_size=1e4)
    pur = purity_metrics(z, labels)
    auc = marker_auroc(den, labels)
    return {**pur, "marker_auroc_mean": np.mean(list(auc.values()))}


rows = []
for name, kl, lrx in [("kl400_lr1x (orig baseline)", 400, 3e-3),
                      ("kl20_lr1x (KL fix only)", 20, 3e-3),
                      ("kl400_lrextra (lr fix only)", 400, 1e-2),
                      ("kl20_lrextra (both, committed)", 20, 1e-2)]:
    for seed in SEEDS:
        r = run(seed, kl, lrx)
        r.update(variant=name, seed=seed)
        rows.append(r)
        print(r)

df = pd.DataFrame(rows)
df.to_csv("ablation_resolvi.csv", index=False)
print(df.groupby("variant")[["ari", "nmi", "marker_auroc_mean"]].agg(["mean", "std"]))
