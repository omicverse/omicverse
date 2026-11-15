"""Train one (ov.vi.Decipher | scvi.external.Decipher) model on the erythroid
NeurIPS 2021 BMMC trajectory and dump its metrics + full-data latent v.

Usage: python run_one.py {ours,ref} <seed> [--epochs N] [--device cpu|cuda]
"""
import argparse
import json
import sys
import time

import anndata as ad
import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import best_axis_spearman, eval_networks  # noqa: E402

DATA = "/scratch/users/steorra/tmp/vi_p2/data/decipher/erythroid_neurips2021.h5ad"
LINEAGE = ["HSC", "MK/E prog", "Proerythroblast", "Erythroblast", "Normoblast", "Reticulocyte"]
COMMON = dict(dim_v=2, dim_z=10, beta=0.1)  # layers_v_to_z=(64,), layers_z_to_x=() (defaults)
OUT = "/scratch/users/steorra/tmp/vi_p2/data/decipher/results_{}_seed{}.json"


def run_ours(adata, seed, device, epochs):
    import omicverse as ov
    torch.manual_seed(seed)
    m = ov.vi.Decipher(adata, layer="counts", seed=seed, **COMMON)
    t0 = time.perf_counter()
    # early_stopping on nll_validation (the Decipher default monitor, both here and in
    # scvi-tools): the raw ClippedAdam/lr=5e-3 recipe eventually diverges on this small
    # dataset in BOTH implementations past ~150-250 epochs, so this is how both are meant
    # to be run in practice, not a port-specific workaround.
    m.train(max_epochs=epochs, batch_size=128, device=device, progress=False, seed=seed,
            early_stopping=True, check_val_every_n_epoch=5, early_stopping_patience=10)
    dt = time.perf_counter() - t0
    val = np.asarray(m.validation_indices)
    v_full = m.get_latent_representation()
    mod = m.module
    networks = (mod.encoder_x_to_z, mod.encoder_zx_to_v, mod.decoder_v_to_z,
               mod.decoder_z_to_x, mod.theta)
    return v_full, val, networks, dt, mod.beta


def run_ref(adata, seed, device, epochs):
    import scvi
    scvi.settings.seed = seed
    torch.manual_seed(seed)
    a = adata.copy()
    scvi.external.Decipher.setup_anndata(a, layer="counts")
    m = scvi.external.Decipher(a, **COMMON)
    accel = "gpu" if device == "cuda" else "cpu"
    t0 = time.perf_counter()
    m.train(max_epochs=epochs, batch_size=128, accelerator=accel, enable_progress_bar=False,
           early_stopping=True, check_val_every_n_epoch=5, early_stopping_patience=10)
    dt = time.perf_counter() - t0
    mod = m.module
    mod.eval()  # Lightning leaves the module in train() mode after fit(); freeze
                # BatchNorm to running stats, same as ov.vi.Decipher.train() does
    val = np.asarray(m.validation_indices)
    v_full = m.get_latent_representation()
    dev = mod.device
    networks = (mod.encoder_x_to_z, mod.encoder_zx_to_v, mod.decoder_v_to_z,
               mod.decoder_z_to_x, mod.theta.to(dev))
    return v_full, val, networks, dt, mod.beta, dev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["ours", "ref"])
    p.add_argument("seed", type=int)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    adata = ad.read_h5ad(DATA)
    pt = adata.obs["GEX_pseudotime_order"].values.astype(np.float64)
    pt_mask = ~np.isnan(pt)
    X = np.asarray(adata.layers["counts"]) if not hasattr(adata.layers["counts"], "toarray") \
        else adata.layers["counts"].toarray()

    if args.mode == "ours":
        v_full, val, networks, dt, beta = run_ours(adata, args.seed, args.device, args.epochs)
        device = next(networks[0].parameters()).device
    else:
        v_full, val, networks, dt, beta, device = run_ref(adata, args.seed, args.device,
                                                           args.epochs)

    rho_gt, axis = best_axis_spearman(v_full[pt_mask], pt[pt_mask])
    elbo, nll = eval_networks(networks, X[val], device, beta, n_iw=100)

    out = dict(mode=args.mode, seed=args.seed, epochs=args.epochs, device=str(device),
              train_time_s=dt, elbo_val=elbo, nll_val=nll, spearman_vs_gt=rho_gt,
              best_axis=axis)
    path = OUT.format(args.mode, args.seed)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    np.savez(path.replace(".json", "_v.npz"), v=v_full, val=val, pt=pt)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
