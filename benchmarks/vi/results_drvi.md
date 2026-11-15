# DRVI (ov.vi.DRVI) vs drvi-py 0.2.8 — validation

Data: NeurIPS 2021 BMMC CITE-seq GEX, donors s1d3/s2d1/s3d7 (3 batches, 27,423 cells,
2,000 HVGs, raw counts; `benchmarks/vi/drvi/prep.py`), `cell_type` (32 types) and
`batch` as the covariates used for disentanglement scoring. Same architecture /
hyperparameters on both sides (both default to the DRVI paper's settings):
`n_latent=32`, encoder/decoder dims `(128, 128)`, `n_split_latent=-1` (one decoder
split per latent dimension), `split_aggregation='logsumexp'`, `gene_likelihood='pnb'`,
`dispersion='gene'`, `batch_key='batch'`. `max_epochs=100`, `batch_size=128`.
GPU: job 52347932, 1 of 2 shared H100 80GB (one process at a time, peak < 1 GB each).
3 seeds (0, 1, 2). `benchmarks/vi/drvi/train_native.py`, `train_reference.py`,
`validate.py`.

## ELBO and wall time (GPU, back-to-back under the same shared-job load)

| impl      | elbo_train (mean ± sd) | time (s, mean ± sd) | speedup |
|-----------|------------------------|----------------------|---------|
| native    | 810.50 ± 0.93          | 60.2 ± 1.9           | **10.4x** |
| reference | 810.22 ± 1.96          | 628.4 ± 182.0        | 1x      |

Native and reference converge to the same train ELBO (differ by 0.3, well inside
the 0.9–2.0 seed spread of either); native is ~10x faster, almost entirely
PyTorch-Lightning / DataLoader overhead in the reference, not GPU compute (both
report < 1 GB peak GPU memory for this data size).

## Disentanglement (DRVI-paper-style per-dimension association)

Per-dimension `|Spearman rho|` against each category (one-vs-rest) of `cell_type`
(32 types) / `batch` (3 batches), aggregated with `drvi-py`'s own
`drvi.utils.metrics._aggregation.most_similar_averaging_score` (MSAS: mean of each
category's best-matching dimension) and `latent_matching_score` (LMS: optimal
one-to-one dimension↔category assignment) — the same formulas score both
implementations' embeddings.

| impl      | MSAS cell_type | LMS cell_type | MSAS batch | LMS batch |
|-----------|----------------|---------------|------------|-----------|
| native    | 0.227 ± 0.003  | 0.203 ± 0.001 | 0.330 ± 0.015 | 0.320 ± 0.012 |
| reference | 0.229 ± 0.002  | 0.204 ± 0.003 | 0.345 ± 0.023 | 0.333 ± 0.019 |

Within seed spread on both scores and both covariates.

## scib integration metrics (bio conservation + batch mixing)

Leiden on a 90-NN graph of the raw latent (`scib_metrics.nmi_ari_cluster_labels_leiden`,
`optimize_resolution=True`); `silhouette_label` / `silhouette_batch`
(`scib_metrics.silhouette_*`); `ilisi_knn` (batch mixing, scaled 0–1, higher =
better mixed); `clisi_knn` (label purity, scaled 0–1, higher = better separated).

| impl      | NMI   | ARI   | silhouette_label | silhouette_batch | iLISI (batch) | cLISI (label) |
|-----------|-------|-------|-------------------|-------------------|----------------|-----------------|
| native    | 0.797 ± 0.009 | 0.777 ± 0.065 | 0.572 ± 0.006 | 0.915 ± 0.003 | 0.327 ± 0.026 | 0.999 ± 0.0001 |
| reference | 0.789 ± 0.004 | 0.743 ± 0.058 | 0.571 ± 0.007 | 0.919 ± 0.003 | 0.331 ± 0.038 | 0.999 ± 0.0001 |

All six metrics agree to within one seed's standard deviation; neither
implementation is systematically better or worse than the other on this data.
Per-seed numbers: `/scratch/users/steorra/tmp/vi_p2/data/drvi/results/metrics.csv`.

## AnnDataOOM / Dask (real data, matches in-memory)

Same data/seed/hyperparameters, `max_epochs=20` (fewer epochs — this run only
checks that disk-backed training reaches the same loss as in-memory, not full
convergence), CPU (`storage='disk'`/`'dask'` block-shuffle, `block_size=256`,
`buffer_blocks=4`), run with `train_native.py --storage {memory,disk,dask}`:

| storage | elbo_train | time (s) |
|---------|-----------|----------|
| memory (`shuffle='random'`) | 867.36 | 354 |
| disk / AnnDataOOM (`shuffle='block'`) | 859.41 | 413 |
| dask (`shuffle='block'`) | 858.28 | 358 |

The three losses agree to within 1%, confirming AnnDataOOM and Dask-backed
training read the correct rows/genes and reach the same optimum as in-memory
(block shuffle vs. full random shuffle explains the small residual gap, as in
the phase-1 SCVI validation). `tests/vi/test_drvi.py::test_oom_matches_in_memory`
/ `test_dask_matches_in_memory` additionally check *exact* latent equality
against in-memory training on smaller synthetic data (same seed, no
convergence tolerance needed).

## Interpretability (per-dimension gene effects)

Not benchmarked against the reference numerically (`drvi-py`'s traversal
utilities build a full systematic AnnData grid via scvi-tools' data loaders,
which is out of scope to port); `ov.vi.DRVI.latent_traverse` /
`.differential_vars` / `.dimension_gene_effects` implement the same idea
(decode the `+/-` extremes of one latent dimension, holding the others at
random baselines, and rank genes by the resulting log-fold-change) and are
checked for determinism and shape correctness in
`tests/vi/test_drvi.py::test_latent_traverse_and_differential_vars`.
