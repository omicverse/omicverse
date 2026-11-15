"""Train ``ov.vi.DRVI`` (this port) on the BMMC3 benchmark data (prep.py) and save
the latent representation + training metrics for ``validate.py``.

Usage: python train_native.py <seed> [--device cuda|cpu] [--data PATH] [--max-epochs N]
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
    p.add_argument("--storage", default="memory", choices=["memory", "disk", "dask"])
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)

    import anndata as ad
    import omicverse as ov

    if args.storage == "memory":
        a = ad.read_h5ad(args.data)
    elif args.storage == "disk":
        ov.settings.cpu_init()
        a = ov.read(args.data, storage="disk")
    else:
        from omicverse.pp._dask import read_h5ad
        # read_h5ad(layer=...) loads layers[layer] as X directly (layers/obsm/obsp
        # are not read lazily), so the model sees the counts as X (layer=None).
        a = read_h5ad(args.data, chunk_size=1024, device="cpu", layer="counts")

    model_layer = None if args.storage == "dask" else "counts"
    m = ov.vi.DRVI(a, layer=model_layer, batch_key="batch", n_latent=32, seed=args.seed)
    t0 = time.perf_counter()
    m.train(max_epochs=args.max_epochs, device=args.device, progress=False, batch_size=128,
            seed=args.seed)
    dt = time.perf_counter() - t0
    z = m.get_latent_representation()
    elbo = float(m.history_["elbo_train"][-1])
    peak_mem = (torch.cuda.max_memory_allocated() / 2**20) if args.device == "cuda" else 0.0

    a_mem = a if args.storage == "memory" else ad.read_h5ad(args.data)
    np.savez(f"{args.out}/native_{args.storage}_seed{args.seed}.npz", z=z, elbo_train=elbo,
             time_s=dt, peak_mem_mb=peak_mem,
             cell_type=a_mem.obs["cell_type"].astype(str).values,
             batch=a_mem.obs["batch"].astype(str).values,
             var_names=np.asarray(m.registry["var_names"]))
    print(f"[native/{args.storage}] seed={args.seed} elbo_train={elbo:.2f} time={dt:.1f}s "
          f"peak_gpu_mb={peak_mem:.0f}")


if __name__ == "__main__":
    main()
