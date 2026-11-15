import sys
import anndata as ad
import numpy as np
import torch
from scipy.stats import spearmanr
import omicverse as ov

DATA = "/scratch/users/steorra/tmp/vi_p2/data/decipher/erythroid_neurips2021.h5ad"
seeds = [int(s) for s in sys.argv[1:]]
a = ad.read_h5ad(DATA)
pt = a.obs["GEX_pseudotime_order"].values.astype(np.float64)
m = ~np.isnan(pt)
for seed in seeds:
    torch.manual_seed(seed)
    mod = ov.vi.Decipher(a, layer="counts", seed=seed, dim_v=2, dim_z=10, beta=0.1)
    mod.train(max_epochs=300, batch_size=128, device="cpu", progress=False, seed=seed,
             early_stopping=True, check_val_every_n_epoch=5, early_stopping_patience=10)
    v = mod.get_latent_representation()
    r0 = spearmanr(v[m, 0], pt[m]).statistic
    r1 = spearmanr(v[m, 1], pt[m]).statistic
    best = max(abs(r0), abs(r1))
    print(f"seed={seed} epochs={mod._epochs_done} best={best:.4f}", flush=True)
