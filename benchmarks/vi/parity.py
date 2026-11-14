"""Parity of ov.vi.SCVI with scvi-tools 1.5.2 (spec Validation 2) and the
block-shuffle experiment (Validation 3).

    python parity.py --data smoke --impl ov scvi --seeds 0 1 2 --device cpu
    python parity.py --data smoke --impl ov --shuffle block --order donor ...

Same hyper-parameters (scvi-tools defaults; batch = donor), same max_epochs
(scvi-tools heuristic unless --epochs).  Records final train / validation
-ELBO (each package's own ``get_elbo`` on its own split), training time and
scib-metrics of the latent; appends to ``<out>.csv``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def run_ov(a, seed, epochs, device, shuffle, block_size, buffer_blocks, **kw):
    from omicverse.vi import SCVI
    m = SCVI(a, layer="counts", batch_key="donor", seed=seed)
    t = time.perf_counter()
    m.train(max_epochs=epochs, device=device, seed=seed, progress=False, shuffle=shuffle,
            block_size=block_size, buffer_blocks=buffer_blocks, **kw)
    tt = time.perf_counter() - t
    z = m.get_latent_representation()
    return z, -m.get_elbo(indices=m.train_indices), -m.get_elbo(indices=m.validation_indices), tt


def run_scvi(a, seed, epochs, device):
    import scvi
    scvi.settings.seed = seed
    scvi.settings.progress_bar_style = "tqdm"
    b = a.copy()
    scvi.model.SCVI.setup_anndata(b, layer="counts", batch_key="donor")
    m = scvi.model.SCVI(b)
    acc = "gpu" if device.startswith("cuda") else "cpu"
    t = time.perf_counter()
    m.train(max_epochs=epochs, accelerator=acc, devices=1, enable_progress_bar=False)
    tt = time.perf_counter() - t
    z = m.get_latent_representation()
    return z, -m.get_elbo(indices=m.train_indices), -m.get_elbo(indices=m.validation_indices), tt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="smoke")
    p.add_argument("--impl", nargs="+", default=["ov", "scvi"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--shuffle", default="auto")
    p.add_argument("--order", default="file", help="'file' or 'donor' (rows sorted by donor)")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--buffer-blocks", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    common.set_threads(args.threads)
    a = common.load(args.data)
    if args.order == "donor":
        o = np.argsort(a.obs["donor"].astype(str).values, kind="stable")
        a = a[o].copy()
    epochs = args.epochs
    if epochs is None:
        epochs = max(min(round(20000 / a.n_obs * 400), 400), 1)
    out = args.out or f"parity_{args.data}_{args.device.split(':')[0]}"
    rows = []
    for seed in args.seeds:
        for impl in args.impl:
            if impl == "ov":
                z, et, ev, tt = run_ov(a, seed, epochs, args.device, args.shuffle,
                                       args.block_size, args.buffer_blocks)
            else:
                z, et, ev, tt = run_scvi(a, seed, epochs, args.device)
            os.makedirs(os.path.join(common.CACHE, "latents"), exist_ok=True)
            np.save(os.path.join(common.CACHE, "latents",
                                 f"{out}_{impl}_{args.shuffle}_{args.order}_s{seed}.npy"), z)
            a.obsm["Z"] = z
            met = common.scib(a, "Z", n_jobs=args.threads)
            r = dict(data=args.data, n_obs=a.n_obs, impl=impl, device=args.device, seed=seed,
                     epochs=epochs, shuffle=args.shuffle if impl == "ov" else "random",
                     order=args.order, block_size=args.block_size,
                     buffer_blocks=args.buffer_blocks, neg_elbo_train=et,
                     neg_elbo_validation=ev, train_time_s=tt, **met)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                  flush=True)
            rows.append(r)
            common.write_table([r], out)


if __name__ == "__main__":
    main()
