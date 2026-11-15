"""ov.vi.RESOLVI vs scvi.external.RESOLVI (scvi-tools 1.5.2) on real imaging-based
spatial data: 10x Xenium FFPE human breast cancer Rep1 (Janesick et al. 2023),
313-gene panel, 10x's own supervised cell-type labels.

Metrics (3 seeds each):
  * final training ELBO (per-cell, higher is better)
  * latent cell-type purity: KMeans(k=n_types) ARI/NMI on get_latent_representation()
    against the 10x-supervised `cell_type` labels (labelled cells only)
  * denoised-marker specificity: for a handful of literature marker genes, AUROC of
    the denoised/normalized expression at separating its cell type from the rest
  * wall time (same GPU, back to back, same load)

Usage:
  python run_benchmark.py --n-cells 15000 --epochs 25 --seeds 0 1 2 --out results.csv
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from sklearn.cluster import KMeans

DATA = "/scratch/users/steorra/tmp/vi_p2/data/resolvi/xenium_breast_r1_crop.h5ad"

# A handful of canonical markers present in the 313-gene Xenium breast panel,
# each expected to be specific to one of 10x's supervised cell types.
MARKERS = {
    "KRT8": "Invasive_Tumor", "EPCAM": "Invasive_Tumor",
    "PTPRC": "CD4+_T_Cells", "CD3D": "CD8+_T_Cells",
    "CD68": "Macrophages_1", "PECAM1": "Endothelial",
    "MS4A1": "B_Cells", "ACTA2": "Myoepi_ACTA2+",
}


def load(n_cells, seed):
    a = ad.read_h5ad(DATA)
    rs = np.random.default_rng(seed)
    rows = rs.choice(a.n_obs, min(n_cells, a.n_obs), replace=False)
    a = a[np.sort(rows)].copy()
    a.layers["counts"] = a.layers["counts"].astype(np.float32)
    a.X = a.layers["counts"].copy()
    keep = np.asarray(a.layers["counts"].sum(1)).ravel() >= 10
    a = a[keep].copy()
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    return a


def purity_metrics(z, labels):
    mask = (labels != "Unlabeled") & ~pd.isna(labels)
    if mask.sum() < 10:
        return {}
    k = len(set(labels[mask]))
    pred = KMeans(k, n_init=5, random_state=0).fit_predict(z[mask])
    return {"ari": adjusted_rand_score(labels[mask], pred),
           "nmi": normalized_mutual_info_score(labels[mask], pred)}


def marker_auroc(denoised: pd.DataFrame, labels):
    out = {}
    for gene, cell_type in MARKERS.items():
        if gene not in denoised.columns:
            continue
        y = (labels == cell_type).astype(int)
        if y.sum() < 5 or y.sum() == len(y):
            continue
        out[gene] = roc_auc_score(y, denoised[gene].to_numpy())
    return out


def run_ours(a, epochs, seed, device, mode="memory", tmp_path=None):
    import torch
    torch.manual_seed(seed)
    import omicverse as ov
    from omicverse.vi import RESOLVI

    layer_arg = "counts"
    if mode == "memory":
        data = a
    elif mode == "oom":
        path = tmp_path / f"ours_{seed}.h5ad"
        a.write_h5ad(path)
        ov.settings.cpu_init()
        data = ov.read(str(path), storage="disk")
    elif mode == "dask":
        # Dask-backed AnnData without scanpy/rapids: `omicverse.pp._dask.read_h5ad` builds
        # a lazy dask-array X of row blocks (cupy CSR blocks on device="gpu", handed to
        # torch via DLPack by ov.vi's DaskSource) -- this is how omicverse's own t_vi
        # tutorial / tests/vi exercise Dask, no `ov.settings.gpu_init()` (RAPIDS) needed.
        # It does not read obsm/layers, so `layer="counts"` is read directly as X, and the
        # spatial coordinates are copied over from the (already in-memory) source AnnData.
        from omicverse.pp._dask import read_h5ad as dask_read_h5ad
        path = tmp_path / f"ours_dask_{seed}.h5ad"
        a.write_h5ad(path)
        data = dask_read_h5ad(str(path), chunk_size=512, device="cpu", layer="counts")
        data.obsm["spatial"] = np.asarray(a.obsm["spatial"])
        layer_arg = None
    else:
        raise ValueError(mode)

    t0 = time.perf_counter()
    m = RESOLVI(data, layer=layer_arg, spatial_key="spatial", n_neighbors=10, mixture_k=50,
               n_hidden=32, n_hidden_encoder=128, n_layers=2, dropout_rate=0.05, seed=seed)
    m.train(max_epochs=epochs, device=device, progress=False, batch_size=512, seed=seed)
    wall = time.perf_counter() - t0

    z = m.get_latent_representation()
    denoised = m.get_normalized_expression(library_size=1e4)
    elbo = m.get_elbo()
    return dict(wall=wall, elbo=elbo), z, denoised


def run_reference(a, epochs, seed, device):
    import torch
    torch.manual_seed(seed)
    import scvi
    from scvi.external import RESOLVI

    adata = a.copy()
    RESOLVI.setup_anndata(adata, layer="counts", batch_key=None,
                          prepare_data_kwargs={"spatial_rep": "spatial", "n_neighbors": 10})
    t0 = time.perf_counter()
    m = RESOLVI(adata, n_hidden=32, n_hidden_encoder=128, n_latent=10, n_layers=2,
               dropout_rate=0.05, mixture_k=50)
    m.train(max_epochs=epochs, batch_size=512, accelerator=device,
           enable_progress_bar=False)
    wall = time.perf_counter() - t0

    z = m.get_latent_representation()
    denoised = m.get_normalized_expression(library_size=1e4)
    try:
        elbo = float(np.mean(m.history["elbo_train"].to_numpy()[-5:]))
    except Exception:
        elbo = float("nan")
    return dict(wall=wall, elbo=elbo), z, denoised


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-cells", type=int, default=15000)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results_resolvi_raw.csv")
    p.add_argument("--skip-reference", action="store_true")
    p.add_argument("--extra-modes", action="store_true", help="also run oom/dask for ours")
    args = p.parse_args()

    import tempfile
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        tmp = Path(tmp)
        for seed in args.seeds:
            a = load(args.n_cells, seed=0)  # same subsample every seed; vary init/shuffle only
            labels = a.obs["cell_type"].to_numpy()

            stats, z, den = run_ours(a, args.epochs, seed, args.device)
            pur = purity_metrics(z, labels)
            auc = marker_auroc(den, labels)
            rows.append({"impl": "ov.vi.RESOLVI", "seed": seed, **stats, **pur,
                        "marker_auroc_mean": np.mean(list(auc.values())) if auc else np.nan,
                        "markers": json.dumps(auc)})
            print(rows[-1])

            if args.extra_modes:
                stats_o, z_o, _ = run_ours(a, args.epochs, seed, args.device, mode="oom",
                                          tmp_path=tmp)
                pur_o = purity_metrics(z_o, labels)
                rows.append({"impl": "ov.vi.RESOLVI[oom]", "seed": seed, **stats_o, **pur_o})
                print(rows[-1])
                stats_d, z_d, _ = run_ours(a, args.epochs, seed, args.device, mode="dask",
                                          tmp_path=tmp)
                pur_d = purity_metrics(z_d, labels)
                rows.append({"impl": "ov.vi.RESOLVI[dask]", "seed": seed, **stats_d, **pur_d})
                print(rows[-1])

            if not args.skip_reference:
                stats, z, den = run_reference(a, args.epochs, seed, args.device)
                pur = purity_metrics(z, labels)
                auc = marker_auroc(den, labels)
                rows.append({"impl": "scvi.external.RESOLVI", "seed": seed, **stats, **pur,
                            "marker_auroc_mean": np.mean(list(auc.values())) if auc else np.nan,
                            "markers": json.dumps(auc)})
                print(rows[-1])

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
