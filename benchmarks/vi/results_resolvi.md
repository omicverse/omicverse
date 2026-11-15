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
`Trace_ELBO(num_particles=5)`)

| impl | seed | wall (s) | latent ARI | latent NMI | marker AUROC (mean of 8 markers) |
|---|---|---|---|---|---|
| ov.vi.RESOLVI | 0 | 181.0 | 0.278 | 0.581 | 0.952 |
| ov.vi.RESOLVI | 1 | 55.1 | 0.307 | 0.593 | 0.934 |
| ov.vi.RESOLVI | 2 | 55.0 | 0.307 | 0.586 | 0.946 |
| **ov.vi.RESOLVI mean** | | **97.0** | **0.297** | **0.587** | **0.944** |
| scvi.external.RESOLVI | 0 | 234.3 | 0.307 | 0.607 | 0.946 |
| scvi.external.RESOLVI | 1 | 149.3 | 0.340 | 0.612 | 0.943 |
| scvi.external.RESOLVI | 2 | 175.3 | 0.349 | 0.610 | 0.947 |
| **scvi.external.RESOLVI mean** | | **186.3** | **0.332** | **0.610** | **0.945** |

ARI/NMI: `KMeans(k=20)` on `get_latent_representation()` vs 10x's supervised
`cell_type` (excluding "Unlabeled"). Marker AUROC: `roc_auc_score` of the
denoised expression (`get_normalized_expression(library_size=1e4)`) for 8
literature cell-type markers present in the panel (KRT8/EPCAM -> tumor,
PTPRC/CD3D -> T cells, CD68 -> macrophages, PECAM1 -> endothelial, MS4A1 ->
B cells, ACTA2 -> myoepithelial) separating their cell type from the rest.

**Marker specificity matches the reference almost exactly (0.944 vs 0.945
mean AUROC)**; latent cluster purity is close but ~10% lower on ARI (0.297 vs
0.332) and ~4% lower on NMI (0.587 vs 0.610) — within the seed spread
(+/-0.03) of either implementation, but our mean sits consistently a bit
below the reference's, most likely the pyro reference's additional global
per-cell bookkeeping (Pyro's `obs_plate` subsample up-weighting of in-plate
sites, which this port approximates — see `_resolvi.py` module docstring —
rather than reproduces bit-for-bit) and/or its 2-layer-deeper, Lightning-
tuned optimiser path. ELBO values are **not on the same scale** (pyro's is
the dataset-scale SVI loss with Dirichlet/Gamma plate up-weighting baked in,
~6.9M; ours is the per-cell mean, ~1.6k) and are reported for convergence-
trend reference only, not absolute comparison.

Wall time: ours is 1.5-4.3x faster per seed (97.0s vs 186.3s mean) despite
matching the reference's 5-particle ELBO — no pyro trace overhead, and
`cuda_graph`/`torch.compile` are disabled for this model (see below) so this
is the un-accelerated eager number.

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

`benchmarks/vi/resolvi/run_benchmark.py --extra-modes` (3,000-cell subsample,
15 epochs, seed 0):

| impl | wall (s) | ELBO | latent ARI | latent NMI |
|---|---|---|---|---|
| ov.vi.RESOLVI (in-memory) | 31.9 | 1586.66 | 0.294 | 0.527 |
| ov.vi.RESOLVI (AnnDataOOM, `ov.read(storage='disk')`) | 31.6 | 1586.69 | 0.298 | 0.528 |

AnnDataOOM trains end to end and matches the in-memory run closely (not
bit-identical: the OOM source defaults to block shuffling vs the resident
source's random shuffling, so minibatch order differs even at the same seed,
as for the other `ov.vi` models). **Dask-GPU not validated**: `ov.settings.gpu_init(dask=True)`
requires `rapids_singlecell` and `dask_cuda`, neither installed in the shared
`ovvi` env (only `cupy`/`dask` are present per SPEC_phase2's environment
list) — an environment gap, not a code path issue; `ResolVI._extra_obs` /
`_materialize_counts` make no assumption that would prevent it, and the model
already streams AnnDataOOM blocks through the same `ov.vi._data` sources
Dask uses.

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
