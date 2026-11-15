"""Summarise the ours-vs-scvi-tools MrVI comparison across seeds: ELBO, latent scib-style
metrics, sample-distance matrix correlation, DA/DE concordance -- each measured both
ours-vs-ref (cross-implementation, same seed) and within each implementation across its
own seeds (the baseline: how much two independently-seeded runs of the *same*
implementation already disagree, from nothing but weight init / minibatch order /
MC sampling). Writes a markdown report to stdout (redirect into ../results_mrvi.md)."""
import sys
import glob
import itertools

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


def _pair_metrics(o, r):
    """All concordance metrics between two runs ``o``, ``r`` (same keys regardless of
    whether they come from the same implementation at different seeds, or different
    implementations at the same seed)."""
    ok, rk = list(o["group_dist_keys"]), list(r["group_dist_keys"])
    common = [g for g in ok if g in rk]
    ao = np.stack([o["group_dist"][ok.index(g)] for g in common])
    ar = np.stack([r["group_dist"][rk.index(g)] for g in common])
    dist_r = _dist_corr(ao, ar)

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

    return dict(dist=dist_r, de_effect=r_eff, de_pval=r_p, de_lfc=r_lfc, da=r_da)


def _mean_sd(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), float(v.std())


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

    # cross-implementation (same seed) and within-implementation (seed pairs) metrics
    cross = {s: _pair_metrics(ours[s], ref[s]) for s in seeds}
    within_ours = {(i, j): _pair_metrics(ours[i], ours[j])
                   for i, j in itertools.combinations(seeds, 2)}
    within_ref = {(i, j): _pair_metrics(ref[i], ref[j])
                  for i, j in itertools.combinations(seeds, 2)}

    print("\n| seed | sample-dist r (ours vs ref) | DE effect rho | DE -log10(p) rho | "
         "DE mean-log2FC r | DA log_probs r |")
    print("|---|---|---|---|---|---|")
    for s in seeds:
        m = cross[s]
        print(f"| {s} | {m['dist']:.4f} | {m['de_effect']:.4f} | {m['de_pval']:.4f} | "
             f"{m['de_lfc']:.4f} | {m['da']:.4f} |")

    print("\n## Baseline: within-implementation seed-pair concordance vs "
         "ours-vs-ref (same seed)\n")
    print("| metric | ours-vs-ref (mean +/- sd, n=3 seeds) | within-ours seed-pairs "
         "(mean +/- sd, n=3 pairs) | within-ref seed-pairs (mean +/- sd, n=3 pairs) |")
    print("|---|---|---|---|")
    for key, label in [("dist", "sample-distance r"), ("de_effect", "DE effect rho"),
                      ("de_pval", "DE -log10(p) rho"), ("de_lfc", "DE mean-log2FC r"),
                      ("da", "DA log_probs r")]:
        c_m, c_s = _mean_sd([cross[s][key] for s in seeds])
        wo_m, wo_s = _mean_sd([within_ours[p][key] for p in within_ours])
        wr_m, wr_s = _mean_sd([within_ref[p][key] for p in within_ref])
        print(f"| {label} | {c_m:.4f} +/- {c_s:.4f} | {wo_m:.4f} +/- {wo_s:.4f} | "
             f"{wr_m:.4f} +/- {wr_s:.4f} |")

    print("\n### within-ours / within-ref seed-pair detail\n")
    print("| pair | sample-dist r (ours) | sample-dist r (ref) | DE mean-log2FC r (ours) "
         "| DE mean-log2FC r (ref) |")
    print("|---|---|---|---|---|")
    for p in within_ours:
        print(f"| {p} | {within_ours[p]['dist']:.4f} | {within_ref[p]['dist']:.4f} | "
             f"{within_ours[p]['de_lfc']:.4f} | {within_ref[p]['de_lfc']:.4f} |")


if __name__ == "__main__":
    main()
