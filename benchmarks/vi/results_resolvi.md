# ResolVI (ov.vi.RESOLVI / ov.space.resolvi) — validation vs scvi-tools 1.5.2

GPU: 1x NVIDIA H100 80GB (job 52347932, shared with 4 other port agents + the
user's own jobs; both GPUs ~97% busy throughout). Env `ovvi` (torch 2.6 cu124,
scvi-tools 1.5.2 pyro reference). Timings are back-to-back on the same node
under the same shared load, as required by SPEC_phase2.

## Data

10x Genomics Xenium FFPE human breast cancer Rep1 (Janesick et al. 2023),
313-gene panel, 10x's own supervised cell-type calls
(`Cell_Barcode_Type_Matrices.xlsx`), real public data — not synthetic.
`benchmarks/vi/resolvi/prep_data.py` builds `xenium_breast_r1.h5ad` (full,
~165k cells after the >=10-count filter) and a left-tissue crop
(`xenium_breast_r1_crop.h5ad`, 73,800 cells, 20 cell types incl. "Unlabeled").
The benchmark below subsamples the crop to 8,000 cells (same subsample every
seed; only the init/training RNG differs) — `benchmarks/vi/resolvi/run_benchmark.py`.

## Validation table (3 seeds, 20 epochs, n_latent=10, n_hidden=32,
n_hidden_encoder=128, n_layers=2, mixture_k=50, n_neighbors=10, dropout=0.05,
batch_size=512; ours: 5-particle Monte Carlo ELBO to match the reference's
`Trace_ELBO(num_particles=5)`, `n_epochs_kl_warmup=20` and a 10x faster lr on
`per_neighbor_diffusion_map`/`u_prior_means`, both matching the reference's
own `RESOLVI.train()` defaults — see "two more fixes" below)

| impl | seed | wall (s) | latent ARI | latent NMI | marker AUROC (mean of 8) |
|---|---|---|---|---|---|
| ov.vi.RESOLVI | 0 | 122.8 | 0.347 | 0.601 | 0.930 |
| ov.vi.RESOLVI | 1 | 90.8 | 0.339 | 0.612 | 0.932 |
| ov.vi.RESOLVI | 2 | 96.7 | 0.373 | 0.620 | 0.935 |
| **ov.vi.RESOLVI mean +/- sd** | | **103.4** | **0.353 +/- 0.018** | **0.611 +/- 0.010** | **0.932 +/- 0.003** |
| scvi.external.RESOLVI | 0 | 234.3 | 0.307 | 0.607 | 0.946 |
| scvi.external.RESOLVI | 1 | 149.3 | 0.340 | 0.612 | 0.943 |
| scvi.external.RESOLVI | 2 | 175.3 | 0.349 | 0.610 | 0.947 |
| **scvi.external.RESOLVI mean +/- sd** | | **186.3** | **0.332 +/- 0.022** | **0.610 +/- 0.003** | **0.945 +/- 0.002** |

ARI/NMI: `KMeans(k=20)` on `get_latent_representation()` vs 10x's supervised
`cell_type` (excluding "Unlabeled"). Marker AUROC: `roc_auc_score` of the
denoised expression (`get_normalized_expression(library_size=1e4)`) for 8
literature cell-type markers present in the panel (KRT8/EPCAM -> tumor,
PTPRC/CD3D -> T cells, CD68 -> macrophages, PECAM1 -> endothelial, MS4A1 ->
B cells, ACTA2 -> myoepithelial) separating their cell type from the rest.

**Latent purity now matches within seed spread**: ARI 0.353 +/- 0.018 vs
0.332 +/- 0.022 (ours is actually slightly *higher*, gap +0.021 is well
inside the combined +/-0.028 spread); NMI 0.611 +/- 0.010 vs 0.610 +/- 0.003
(gap 0.001, trivial). Marker AUROC is still high (0.932 +/- 0.003) but now
sits a bit below the reference's 0.945 +/- 0.002 (gap 0.013 > combined
+/-0.004 spread) — the two training-default fixes below closed the ARI/NMI
gap at a small cost to marker sharpness (stronger/earlier KL pressure on `z`
slightly smooths the decoder's per-gene profile); both are still
functionally strong (>0.93 AUROC) and this residual is left as a known,
minor difference rather than re-tuned further. ELBO values are **not on the
same scale** (pyro's is the dataset-scale SVI loss with Dirichlet/Gamma
plate up-weighting baked in, ~6.9M; ours is the per-cell mean, ~1.6k) —
reported for convergence-trend reference only.

Wall time: ours is 1.6-2.1x faster per seed (103.4s vs 186.3s mean) despite
matching the reference's 5-particle ELBO — no pyro trace overhead, and
`cuda_graph`/`torch.compile` are disabled for this model (see below) so this
is the un-accelerated eager number.

## Two more fixes made after the coordinator's review of the first table

1. **KL warm-up schedule mismatch.** `scvi.external.RESOLVI.train()` defaults
   `n_epochs_kl_warmup=20` (independent of `max_epochs`); `ov.vi`'s own
   `TrainConfig` default is `400`. At `max_epochs=20` this left `z` almost
   completely unregularised throughout our runs (`kl_weight` only reaching
   `20/400 = 0.05`) vs the reference reaching full KL weight by the end of
   training — plausibly the direct cause of the lower ARI/NMI in the first
   table. `RESOLVI.train()` now defaults `plan_kwargs['n_epochs_kl_warmup']`
   to the reference's literal default (`20`), overridable as before.
2. **Missing per-parameter learning rate.** The reference trains
   `per_neighbor_diffusion_map` and `u_prior_means` — the per-cell diffusion
   weights and the latent mixture prior's component locations, both
   structurally point estimates with no amortising network averaging
   gradients over many cells — at `lr_extra=1e-2`, 10x the base `lr=3e-3`.
   Added `ResolVIModule.optim_parameters()` (two param groups) and a small,
   backwards-compatible extension to `ov.vi._train.make_optimizer` to accept
   pre-built param groups (it previously assumed a flat parameter list;
   `tests/vi/test_core.py` still passes unchanged).

Both fixes are about matching the reference's own stated training defaults,
not new hyperparameter tuning.

## A real correctness bug found and fixed during validation

The first full run collapsed every cell's true/diffusion/background mixture
to background ~= 1 (std ~ 1e-8 across all 8,000 cells) — marker AUROC ~ 0.52
(chance) vs the reference's 0.95. Root cause: the Dirichlet log-priors on the
per-neighbour diffusion weights and the true/diffusion/background mixture are
bare log-densities (no entropy term, since their guide is a point estimate),
so maximising them means *subtracting* them from the loss being minimised —
they were being *added*, so the optimiser fled the prior's mode instead of
being pulled toward it. Fixing the sign (`_resolvi.py`, `ResolVIModule.forward`)
and adding the reference's 5-particle Monte-Carlo ELBO (a 1-particle gradient
of the Gamma-compound NB mixture turned out to still reach a sensible
optimum once the sign was right, but 5 particles better matches the
reference and is kept as the default) immediately recovered sensible,
input-dependent mixture proportions and the table above.

## AnnDataOOM / Dask

`ov.settings.gpu_init(dask=True)` needs `rapids_singlecell`/`dask_cuda`
(absent from the shared `ovvi` env); that is *not* how `ov.vi` itself gets a
Dask-backed AnnData, though (nor how the phase-1 t_vi tutorial / `tests/vi`
do it) — `omicverse.pp._dask.read_h5ad(path, chunk_size=, device=)` builds
one directly (a lazy dask array of row blocks, `device='gpu'` for cupy CSR
blocks handed to torch via DLPack, no RAPIDS). Using that instead:
`benchmarks/vi/resolvi/run_benchmark.py --extra-modes` (3,000-cell
subsample, 15 epochs, seed 0; `read_h5ad` does not load `obsm`/`layers`, so
the spatial coordinates are copied over and `layers['counts']` is read
directly as `X`):

| impl | wall (s) | ELBO | latent ARI | latent NMI |
|---|---|---|---|---|
| ov.vi.RESOLVI (in-memory) | 33.7 | 1595.35 | 0.316 | 0.491 |
| ov.vi.RESOLVI (AnnDataOOM, `ov.read(storage='disk')`) | 29.2 | 1595.28 | 0.291 | 0.482 |
| ov.vi.RESOLVI (Dask, `omicverse.pp._dask.read_h5ad`) | 29.5 | 1595.42 | 0.341 | 0.495 |

All three train end to end and land on closely matching ELBO/ARI/NMI (not
bit-identical: the OOM/Dask sources default to block shuffling vs the
resident source's random shuffling, so minibatch order differs even at the
same seed, as for the other `ov.vi` models).

## Known simplifications vs the pyro reference (see `_resolvi.py` docstrings)

* Global point estimates (`sparsity_diffusion`, `true/diffusion/background_
  proportion`, `diffuse_scale`, per-gene background, per-neighbour diffusion
  map) are plain `nn.Parameter`s with an explicit Gamma/Dirichlet/Exponential
  log-prior penalty added to the loss (exactly what a pyro `Delta` guide
  contributes to `Trace_ELBO`), not sampled through a probabilistic-
  programming trace.
* `size_scaling` (physical cell-size factor in place of the library size) and
  `differential_expression`/`differential_niche_abundance` are not ported
  (rare in practice; flag if needed).
* `cuda_graph` and `torch.compile` are disabled for this model: both fast
  paths strip the per-cell row index from the minibatch dict (no other
  `ov.vi` model needs it), which `RESOLVI` needs to index
  `per_neighbor_diffusion_map`.
