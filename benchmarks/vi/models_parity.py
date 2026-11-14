"""SCANVI and TOTALVI vs scvi-tools 1.5.2 on real data.

    python models_parity.py --model scanvi --data smoke --device cuda
    python models_parity.py --model totalvi --device cuda   # 10x PBMC CITE-seq (scvi.data)

SCANVI: SEA-AD cells with 70 % of the ``cell_type`` labels hidden (seeded);
reports label-transfer accuracy / macro-F1 on the hidden cells, ELBO and
scib-metrics of the latent.  TOTALVI: the two 10x PBMC CITE-seq datasets
(``scvi.data.pbmcs_10x_cite_seq``, batch = dataset); reports train /
validation -ELBO, iLISI of the latent and the correlation of denoised protein
with raw protein counts.  Same hyper-parameters (package defaults), 3 seeds.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def _hide_labels(a, seed=0, frac=0.7):
    lab = a.obs["cell_type"].astype(str).values.copy()
    lab[np.random.default_rng(seed).random(a.n_obs) < frac] = "Unknown"
    a.obs["lab"] = lab
    return lab == "Unknown"


def run_scanvi(a, impl, seed, device):
    if impl == "ov":
        from omicverse.vi import SCANVI
        m = SCANVI(a, layer="counts", batch_key="donor", labels_key="lab",
                   unlabeled_category="Unknown", seed=seed)
        t = time.perf_counter()
        m.train(device=device, seed=seed, progress=False)
        tt = time.perf_counter() - t
        return (m.get_latent_representation(), np.asarray(m.predict()), tt,
                -m.get_elbo(indices=m.train_indices), -m.get_elbo(indices=m.validation_indices))
    import scvi
    scvi.settings.seed = seed
    b = a.copy()
    scvi.model.SCANVI.setup_anndata(b, layer="counts", batch_key="donor", labels_key="lab",
                                    unlabeled_category="Unknown")
    m = scvi.model.SCANVI(b)
    t = time.perf_counter()
    m.train(accelerator="gpu" if device.startswith("cuda") else "cpu", devices=1,
            enable_progress_bar=False)
    tt = time.perf_counter() - t
    return (m.get_latent_representation(), np.asarray(m.predict()), tt,
            -float(m.get_elbo(indices=m.train_indices)),
            -float(m.get_elbo(indices=m.validation_indices)))


def _cite():
    import anndata as ad
    path = os.path.join(common.CACHE, "pbmc_cite_hvg4000.h5ad")
    if os.path.exists(path):
        return ad.read_h5ad(path)
    import scanpy as sc
    import scvi
    a = scvi.data.pbmcs_10x_cite_seq(save_path=os.path.join(common.CACHE, "scvi_data"))
    a.layers["counts"] = a.X.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=4000, flavor="seurat_v3", batch_key="batch",
                                layer="counts")
    a = a[:, a.var["highly_variable"].values].copy()
    a.obs["batch"] = a.obs["batch"].astype(str).astype("category")
    a.write_h5ad(path)
    return a


def run_totalvi(a, impl, seed, device):
    if impl == "ov":
        from omicverse.vi import TOTALVI
        m = TOTALVI(a, layer="counts", batch_key="batch",
                    protein_expression_obsm_key="protein_expression", seed=seed)
        t = time.perf_counter()
        m.train(device=device, seed=seed, progress=False)
        tt = time.perf_counter() - t
        _, pro = m.get_normalized_expression(n_samples=10, return_numpy=True)
        return (m.get_latent_representation(), pro, tt, -m.get_elbo(indices=m.train_indices),
                -m.get_elbo(indices=m.validation_indices), len(m.history_["time"]))
    import scvi
    scvi.settings.seed = seed
    b = a.copy()
    scvi.model.TOTALVI.setup_anndata(b, layer="counts", batch_key="batch",
                                     protein_expression_obsm_key="protein_expression")
    m = scvi.model.TOTALVI(b)
    t = time.perf_counter()
    m.train(accelerator="gpu" if device.startswith("cuda") else "cpu", devices=1,
            enable_progress_bar=False)
    tt = time.perf_counter() - t
    _, pro = m.get_normalized_expression(n_samples=10, return_numpy=True)
    return (m.get_latent_representation(), pro, tt,
            -float(m.get_elbo(indices=m.train_indices)),
            -float(m.get_elbo(indices=m.validation_indices)), len(m.history["elbo_train"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["scanvi", "totalvi"], default="scanvi")
    p.add_argument("--data", default="smoke")
    p.add_argument("--impl", nargs="+", default=["ov", "scvi"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=16)
    args = p.parse_args()
    common.set_threads(args.threads)
    from sklearn.metrics import accuracy_score, f1_score
    for seed in args.seeds:
        for impl in args.impl:
            if args.model == "scanvi":
                a = common.load(args.data)
                hidden = _hide_labels(a, seed)
                z, pred, tt, et, ev = run_scanvi(a, impl, seed, args.device)
                truth = a.obs["cell_type"].astype(str).values
                a.obsm["Z"] = z
                met = common.scib(a, "Z", n_jobs=args.threads)
                r = dict(model="SCANVI", data=args.data, impl=impl, seed=seed,
                         acc_hidden=accuracy_score(truth[hidden], pred[hidden]),
                         f1_macro_hidden=f1_score(truth[hidden], pred[hidden], average="macro"),
                         neg_elbo_train=et, neg_elbo_validation=ev, train_time_s=tt, **met)
                name = f"models_parity_scanvi_{args.data}"
            else:
                a = _cite()
                z, pro, tt, et, ev, ep = run_totalvi(a, impl, seed, args.device)
                raw = np.asarray(a.obsm["protein_expression"], dtype=np.float64)
                cors = [np.corrcoef(np.log1p(raw[:, j]), np.log1p(pro[:, j]))[0, 1]
                        for j in range(raw.shape[1])]
                from scib_metrics import ilisi_knn
                from scib_metrics.nearest_neighbors import pynndescent
                nn = pynndescent(np.asarray(z, np.float32), n_neighbors=90, n_jobs=args.threads)
                r = dict(model="TOTALVI", data="pbmc_cite", impl=impl, seed=seed, epochs=ep,
                         neg_elbo_train=et, neg_elbo_validation=ev, train_time_s=tt,
                         iLISI=float(ilisi_knn(nn, a.obs["batch"].astype(str).values)),
                         protein_denoised_vs_raw_r=float(np.nanmedian(cors)))
                name = "models_parity_totalvi"
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                  flush=True)
            common.write_table([r], name)


if __name__ == "__main__":
    main()
