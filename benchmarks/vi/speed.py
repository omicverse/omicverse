"""Training speed of ov.vi.SCVI vs scvi-tools 1.5.2 (spec Validation 5).

    python speed.py --data smoke --device cpu --epochs 50 --threads 16
    python speed.py --data seaad300k --device cuda --epochs 10

Same hyper-parameters (scvi-tools defaults, batch = donor) and epochs; wall
time of ``train()`` only (model construction / data setup excluded for both;
ov.vi's one-off upload of the CSR to the device is included in train()).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="smoke")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--impl", nargs="+", default=["ov", "scvi"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--ov-kwargs", default="{}")
    args = p.parse_args()
    import json
    common.set_threads(args.threads)
    a = common.load(args.data)
    if args.device.startswith("cuda"):  # CUDA context / cuBLAS init outside the timings
        import torch
        x = torch.randn(256, 256, device=args.device)
        (x @ x).sum().item()
    rows = []
    for rep in range(args.repeats):
        for impl in args.impl:
            if impl == "ov":
                from omicverse.vi import SCVI
                m = SCVI(a, layer="counts", batch_key="donor", seed=rep)
                t = time.perf_counter()
                m.train(max_epochs=args.epochs, device=args.device, seed=rep, progress=False,
                        **json.loads(args.ov_kwargs))
                if args.device.startswith("cuda"):
                    import torch
                    torch.cuda.synchronize()
                tt = time.perf_counter() - t
                elbo = m.history_["elbo_train"][-1]
            else:
                import scvi
                scvi.settings.seed = rep
                b = a.copy()
                scvi.model.SCVI.setup_anndata(b, layer="counts", batch_key="donor")
                m = scvi.model.SCVI(b)
                acc = "gpu" if args.device.startswith("cuda") else "cpu"
                t = time.perf_counter()
                m.train(max_epochs=args.epochs, accelerator=acc, devices=1,
                        enable_progress_bar=False)
                tt = time.perf_counter() - t
                elbo = float(m.history["elbo_train"].iloc[-1, 0])
            r = dict(data=args.data, n_obs=a.n_obs, impl=impl + args.tag, device=args.device,
                     threads=args.threads, epochs=args.epochs, repeat=rep, train_time_s=tt,
                     s_per_epoch=tt / args.epochs, elbo_train_last=elbo)
            print(r, flush=True)
            rows.append(r)
            common.write_table([r], f"speed_{args.device.split(':')[0]}")


if __name__ == "__main__":
    main()
