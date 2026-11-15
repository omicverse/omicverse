"""AnnDataOOM and Dask-backed training on the real Wilk 2020 cohort (not synthetic),
compared to in-memory training: same seed/hyperparameters as run_ours.py (common.py),
ELBO, ARI (ground-truth cell_type vs KMeans on u) and sample-distance matrix
correlation against the in-memory run.

Usage: python run_disk.py <seed> <out_prefix> [cuda|cpu]
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from common import DATA, SAMPLE_KEY, LABELS_KEY, MODEL_KW, OURS_TRAIN_KW as TRAIN_KW  # noqa: E402


def _run(adata, seed, device, label):
    from omicverse.vi import MRVI
    torch.manual_seed(seed)
    t0 = time.time()
    m = MRVI(adata, sample_key=SAMPLE_KEY, layer=None, seed=seed, **MODEL_KW)
    m.train(device=device, seed=seed, progress=False, **TRAIN_KW)
    dt = time.time() - t0
    u = m.get_latent_representation()
    torch.manual_seed(seed + 1000)
    dist = m.get_local_sample_distances(batch_size=256, groupby=LABELS_KEY, keep_cell=False)
    print(f"{label}: elbo_train={m.history_['elbo_train'][-1]:.3f} "
         f"kind={m._source().kind} time={dt:.1f}s")
    return u, dist, m.history_["elbo_train"][-1]


def main():
    seed = int(sys.argv[1])
    out = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"

    import anndata as ad
    import omicverse as ov

    # X already holds raw counts (prep_data.py); layer=None everywhere so the Dask
    # reader (which does not carry layers) and AnnDataOOM / in-memory agree exactly.
    a_mem = ad.read_h5ad(DATA)
    cell_type = a_mem.obs[LABELS_KEY].astype(str).values
    u_mem, dist_mem, elbo_mem = _run(a_mem, seed, device, "memory")

    ov.settings.cpu_init()
    a_oom = ov.read(DATA, storage="disk")
    u_oom, dist_oom, elbo_oom = _run(a_oom, seed, device, "oom")

    from omicverse.pp._dask import read_h5ad
    a_dsk = read_h5ad(DATA, chunk_size=2000, device="cpu")
    u_dsk, dist_dsk, elbo_dsk = _run(a_dsk, seed, device, "dask")

    def ari(u):
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        k = len(set(cell_type))
        km = KMeans(k, n_init=5, random_state=0).fit_predict(u)
        return adjusted_rand_score(cell_type, km)

    def dist_corr(d1, d2):
        from scipy.stats import pearsonr
        k1, k2 = list(d1["groups"][LABELS_KEY].keys()), list(d2["groups"][LABELS_KEY].keys())
        common = [g for g in k1 if g in k2]
        a = np.stack([d1["groups"][LABELS_KEY][g] for g in common])
        b = np.stack([d2["groups"][LABELS_KEY][g] for g in common])
        n = a.shape[-1]
        iu = np.triu_indices(n, k=1)
        return pearsonr(np.concatenate([a[i][iu] for i in range(len(a))]),
                        np.concatenate([b[i][iu] for i in range(len(b))]))[0]

    print(f"ari: memory={ari(u_mem):.4f} oom={ari(u_oom):.4f} dask={ari(u_dsk):.4f}")
    print(f"sample-distance r vs memory: oom={dist_corr(dist_mem, dist_oom):.4f} "
         f"dask={dist_corr(dist_mem, dist_dsk):.4f}")
    print(f"elbo_train: memory={elbo_mem:.3f} oom={elbo_oom:.3f} dask={elbo_dsk:.3f}")

    np.savez(out + ".npz", elbo_mem=elbo_mem, elbo_oom=elbo_oom, elbo_dsk=elbo_dsk,
            ari_mem=ari(u_mem), ari_oom=ari(u_oom), ari_dsk=ari(u_dsk),
            dist_r_oom=dist_corr(dist_mem, dist_oom), dist_r_dsk=dist_corr(dist_mem, dist_dsk))


if __name__ == "__main__":
    main()
