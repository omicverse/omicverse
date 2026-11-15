import sys
import anndata as ad
import numpy as np
import torch
from scipy.stats import spearmanr
import scvi

DATA = "/scratch/users/steorra/tmp/vi_p2/data/decipher/erythroid_neurips2021.h5ad"
seeds = [int(s) for s in sys.argv[1:]]
a = ad.read_h5ad(DATA)
pt = a.obs["GEX_pseudotime_order"].values.astype(np.float64)
m = ~np.isnan(pt)
for seed in seeds:
    scvi.settings.seed = seed
    torch.manual_seed(seed)
    aa = a.copy()
    scvi.external.Decipher.setup_anndata(aa, layer="counts")
    mod = scvi.external.Decipher(aa, dim_v=2, dim_z=10, beta=0.1)
    mod.train(max_epochs=300, batch_size=128, accelerator="cpu", enable_progress_bar=False,
             early_stopping=True, check_val_every_n_epoch=5, early_stopping_patience=10)
    v = mod.get_latent_representation()
    r0 = spearmanr(v[m, 0], pt[m]).statistic
    r1 = spearmanr(v[m, 1], pt[m]).statistic
    best = max(abs(r0), abs(r1))
    print(f"seed={seed} best={best:.4f}", flush=True)
