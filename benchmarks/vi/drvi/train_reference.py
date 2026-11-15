"""Train the original ``drvi-py`` (``theislab/drvi``) ``DRVI`` model (scvi-tools
backend) on the same BMMC3 data, with the same architecture / hyperparameters,
for ``validate.py`` to compare against ``train_native.py``.

Usage: python train_reference.py <seed> [--device cuda|cpu] [--data PATH] [--max-epochs N]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("seed", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--data", default=os.environ.get(
        "OV_DRVI_DATA", "/scratch/users/steorra/tmp/vi_p2/data/drvi") + "/bmmc3_hvg2000.h5ad")
    p.add_argument("--out", default=os.environ.get(
        "OV_DRVI_DATA", "/scratch/users/steorra/tmp/vi_p2/data/drvi") + "/results")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)

    import anndata as ad
    import scvi
    from drvi.model import DRVI as DRVIRef

    scvi.settings.seed = args.seed
    a = ad.read_h5ad(args.data)
    DRVIRef.setup_anndata(a, layer="counts", batch_key="batch")
    # same architecture as ov.vi.DRVI's defaults (both mirror the DRVI paper):
    # n_latent=32, encoder/decoder_dims=(128, 128), n_split_latent=-1 (one split per
    # dim), split_aggregation='logsumexp', gene_likelihood='pnb', dispersion='gene'.
    m = DRVIRef(a, n_latent=32)
    t0 = time.perf_counter()
    accelerator = "gpu" if args.device == "cuda" else "cpu"
    m.train(max_epochs=args.max_epochs, accelerator=accelerator, devices=1,
            batch_size=128, enable_progress_bar=False)
    dt = time.perf_counter() - t0
    z = m.get_latent_representation()
    elbo = float(m.history["elbo_train"].iloc[-1, 0])
    peak_mem = (torch.cuda.max_memory_allocated() / 2**20) if args.device == "cuda" else 0.0

    np.savez(f"{args.out}/reference_seed{args.seed}.npz", z=z, elbo_train=elbo, time_s=dt,
             peak_mem_mb=peak_mem, cell_type=a.obs["cell_type"].astype(str).values,
             batch=a.obs["batch"].astype(str).values, var_names=a.var_names.values)
    print(f"[reference] seed={args.seed} elbo_train={elbo:.2f} time={dt:.1f}s "
          f"peak_gpu_mb={peak_mem:.0f}")


if __name__ == "__main__":
    main()
