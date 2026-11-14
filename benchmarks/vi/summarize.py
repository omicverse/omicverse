"""Collect the ov.vi benchmark CSVs into markdown tables (results.md)."""
from __future__ import annotations

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

HERE = common.HERE
METRICS = ["neg_elbo_train", "neg_elbo_validation", "iLISI", "cLISI", "kBET", "ASW_batch",
           "ASW_label", "NMI", "ARI", "graph_conn", "isolated_labels", "train_time_s"]


def _num(v):
    return float(re.sub(r"tensor\(([-0-9.e+]+).*", r"\1", str(v)))


def _read(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    for c in ("neg_elbo_train", "neg_elbo_validation"):
        if c in d:
            d[c] = d[c].map(_num)
    return d


def parity():
    out = []
    for f, title in (("parity_smoke_cuda.csv", "SEA-AD 5k cells (smoke), H100"),
                     ("parity_smoke_cpu.csv", "SEA-AD 5k cells (smoke), CPU 16 threads"),
                     ("parity_seaad50k_cuda.csv", "SEA-AD 50k cells, H100")):
        d = _read(f)
        if d is None or d.empty:
            continue
        n = d.groupby("impl").size().to_dict()
        s = common.summarize(d, ["impl"], [m for m in METRICS if m in d])
        out.append(f"### {title} (seeds per impl: {n}; epochs {int(d['epochs'].iloc[0])})\n")
        out.append(common.to_markdown(s))
        out.append("")
    return "\n".join(out)


def blockshuffle():
    d = _read("blockshuffle_seaad50k.csv")
    if d is None:
        return ""
    ref = _read("parity_seaad50k_cuda.csv")
    if ref is not None:
        r = ref[ref.impl == "ov"].copy()
        r["shuffle"], r["order"], r["block_size"], r["buffer_blocks"] = "random", "any", "-", "-"
        d = pd.concat([r, d], ignore_index=True)
    d["config"] = (d["shuffle"].astype(str) + " order=" + d["order"].astype(str) + " block="
                   + d["block_size"].astype(str) + " x" + d["buffer_blocks"].astype(str))
    s = common.summarize(d, ["config"], [m for m in METRICS if m in d and m != "train_time_s"])
    return "### Block shuffle vs random (SEA-AD 50k, 160 epochs, 3 seeds)\n\n" + \
        common.to_markdown(s)


def speed():
    out = []
    for f in ("speed_cpu.csv", "speed_cuda.csv"):
        d = _read(f)
        if d is None:
            continue
        g = d.groupby(["data", "device", "impl", "epochs"])["train_time_s"].agg(["mean", "min",
                                                                                 "count"])
        out.append(f"### {f}\n")
        out.append(common.to_markdown(g.reset_index().round(2)))
        out.append("")
    return "\n".join(out)


def disk():
    d = _read("disk_seaad.csv")
    if d is None:
        return ""
    cols = ["mode", "tag", "n_obs", "epochs", "epoch_s", "train_s", "latent_s",
            "rss_before_train_gb", "rss_peak_train_gb", "rss_growth_train_gb",
            "gpu_used_peak_train_gb", "torch_peak_alloc_gb", "iLISI", "cLISI", "kBET",
            "ASW_batch", "ASW_label", "NMI", "ARI"]
    d = d[[c for c in cols if c in d]].round(3)
    return "### Full SEA-AD (1,378,211 cells, 2000 HVGs) trained from disk on the H100\n\n" + \
        common.to_markdown(d)


def main():
    parts = ["# ov.vi benchmark results\n", "## Parity with scvi-tools 1.5.2\n", parity(),
             "## Block shuffle\n", blockshuffle(), "\n## Training speed\n", speed(),
             "## Disk training\n", disk()]
    lock = _read("lock_test.csv")
    if lock is not None:
        parts += ["\n## Concurrent reads (lock test)\n", common.to_markdown(lock)]
    text = "\n".join(parts) + "\n"
    with open(os.path.join(HERE, "results.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
