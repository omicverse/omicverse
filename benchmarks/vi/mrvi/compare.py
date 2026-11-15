"""Summarise the ours-vs-scvi-tools MrVI comparison across seeds: ELBO, latent scib-style
metrics, sample-distance matrix correlation, DA/DE concordance. Writes a markdown table
to stdout (redirect into ../results_mrvi.md)."""
import sys
import glob

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def _load(prefix):
    seeds = sorted(int(p.split("_s")[-1].split(".")[0]) for p in glob.glob(prefix + "_s*.npz"))
    return {s: dict(np.load(f"{prefix}_s{s}.npz", allow_pickle=True)) for s in seeds}


def _latent_metrics(u, cell_type, donor):
    k = len(set(cell_type))
    km = KMeans(k, n_init=5, random_state=0).fit_predict(u)
    ari = adjusted_rand_score(cell_type, km)
    nmi = normalized_mutual_info_score(cell_type, km)
    asw_batch = silhouette_score(u, donor)       # lower magnitude = better-mixed donors
    asw_bio = silhouette_score(u, cell_type)      # higher = better cell-type separation
    return dict(ari=ari, nmi=nmi, asw_batch=asw_batch, asw_bio=asw_bio)


def _dist_corr(a, b):
    # a, b: group_dist_keys-aligned (n_group, n_sample, n_sample); upper triangle only
    n = a.shape[-1]
    iu = np.triu_indices(n, k=1)
    xa = np.concatenate([a[i][iu] for i in range(a.shape[0])])
    xb = np.concatenate([b[i][iu] for i in range(b.shape[0])])
    return pearsonr(xa, xb)[0]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    ours = _load(f"{tag}_ours")
    ref = _load(f"{tag}_ref")
    seeds = sorted(set(ours) & set(ref))
    print(f"# MrVI validation: ours vs scvi-tools 1.5.2 (torch backend), seeds {seeds}\n")

    print("| seed | elbo_train (ours) | elbo_train (ref) | elbo_val (ours) | "
         "elbo_val (ref) | train_time ours (s) | train_time ref (s) |")
    print("|---|---|---|---|---|---|---|")
    for s in seeds:
        o, r = ours[s], ref[s]
        print(f"| {s} | {o['elbo_train']:.2f} | {r['elbo_train']:.2f} | "
             f"{o['elbo_val']:.2f} | {r['elbo_val']:.2f} | "
             f"{float(o['train_time']):.1f} | {float(r['train_time']):.1f} |")

    print("\n| seed | metric | ours (u) | ref (u) |")
    print("|---|---|---|---|")
    for s in seeds:
        o, r = ours[s], ref[s]
        mo = _latent_metrics(o["u"], o["cell_type"], o["donor"])
        mr = _latent_metrics(r["u"], r["cell_type"], r["donor"])
        for k in mo:
            print(f"| {s} | {k} | {mo[k]:.4f} | {mr[k]:.4f} |")

    print("\n| seed | sample-distance matrix Pearson r (ours vs ref, per cell_type group) |")
    print("|---|---|")
    for s in seeds:
        o, r = ours[s], ref[s]
        ok, rk = list(o["group_dist_keys"]), list(r["group_dist_keys"])
        common = [g for g in ok if g in rk]
        ao = np.stack([o["group_dist"][ok.index(g)] for g in common])
        ar = np.stack([r["group_dist"][rk.index(g)] for g in common])
        print(f"| {s} | {_dist_corr(ao, ar):.4f} |")

    print("\n| seed | DE effect_size Spearman r | DE -log10(pvalue) Spearman r | "
         "DE mean-LFC-per-gene Pearson r | DA log_probs Pearson r |")
    print("|---|---|---|---|---|")
    for s in seeds:
        o, r = ours[s], ref[s]
        r_eff = spearmanr(o["de_effect"].ravel(), r["de_effect"].ravel())[0]
        r_p = spearmanr(-np.log10(o["de_pvalue"].ravel() + 1e-300),
                        -np.log10(r["de_pvalue"].ravel() + 1e-300))[0]
        og, rg = list(o["de_genes"]), list(r["de_genes"])
        common_g = [g for g in og if g in rg]
        lo = np.array([o["de_lfc_mean_gene"][og.index(g)] for g in common_g])
        lr = np.array([r["de_lfc_mean_gene"][rg.index(g)] for g in common_g])
        r_lfc = pearsonr(lo, lr)[0]
        os_, rs_ = list(o["da_samples"]), list(r["da_samples"])
        common_s = [x for x in os_ if x in rs_]
        dao = np.concatenate([o["da_log_probs"][:, os_.index(x)] for x in common_s])
        dar = np.concatenate([r["da_log_probs"][:, rs_.index(x)] for x in common_s])
        r_da = pearsonr(dao, dar)[0]
        print(f"| {s} | {r_eff:.4f} | {r_p:.4f} | {r_lfc:.4f} | {r_da:.4f} |")


if __name__ == "__main__":
    main()
