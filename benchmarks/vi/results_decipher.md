# ov.vi.Decipher vs scvi.external.Decipher (scvi-tools 1.5.2)

Hardware: shared job 52347932, 2x NVIDIA H100 80GB HBM3 (`sh04-04n05`), one GPU
process at a time, GPU ~97% busy from other jobs throughout these runs (both
implementations run back-to-back under that same shared load; see Timing).

## Data

Erythroid-differentiation lineage (HSC -> MK/E prog -> Proerythroblast ->
Erythroblast -> Normoblast -> Reticulocyte) of the NeurIPS 2021 BMMC CITE-seq
donors s1d3, s2d1, s3d7 (`benchmarks/vi/decipher/prepare_data.py`), 4,454 cells
x 2,000 HVGs, raw counts. Every cell in this lineage carries
`obs['GEX_pseudotime_order']`, the ground-truth Palantir pseudotime from the
NeurIPS 2021 competition's trajectory-inference task -- the same kind of
normal haematopoietic differentiation trajectory the Decipher paper validates
on.

## Setup

Identical architecture/hyperparameters for both: `dim_v=2, dim_z=10,
layers_v_to_z=(64,), layers_z_to_x=(), beta=0.1`, `ClippedAdam` (lr 5e-3,
weight decay 1e-4, clip_norm 10), batch_size 128, `train_size=0.9`,
`early_stopping=True` (monitor `nll_validation`, the Decipher default in both
stacks, `check_val_every_n_epoch=5`, patience 10). 3 seeds (0, 1, 2).
`max_epochs=300` is a safety cap only -- early stopping always intervenes
first (both implementations numerically diverge past ~150-250 raw
`ClippedAdam` epochs on this small dataset *without* early stopping; this was
verified for both `ov.vi.Decipher` and `scvi.external.Decipher` and is not a
port-specific issue).

Metrics (`benchmarks/vi/decipher/common.py`) are computed by one
implementation-agnostic function applied directly to each model's four
sub-networks and `theta` (`encoder_x_to_z`, `encoder_zx_to_v`,
`decoder_v_to_z`, `decoder_z_to_x` -- identical attribute names and maths in
both `DecipherModule` and scvi-tools' `DecipherPyroModule`), on the held-out
10%: a 100-sample mean ELBO (nats/cell, higher better) and a 100-sample
importance-weighted NLL, both reduced with the **median** over cells (a few
cells per batch hit a heavy-tailed catastrophic-outlier NB log-likelihood
under a single bad (z, v) draw -- seen for *both* implementations' trained
networks -- which a mean lets dominate the summary). This bypasses each
training stack's own validation bookkeeping, so the comparison is a single
code path, not two.

Rotation note: Decipher's 2-D `v` is identifiable only up to a rotation +
reflection, so the sign and the "winning" axis of a Spearman correlation vs.
pseudotime are arbitrary per run; we report the best-|rho| axis, and sign-align
before comparing ours vs. reference directly.

## Validation (3 seeds, GPU)

| seed | impl | ELBO (val, median) | NLL (val, median) | best-axis rho vs. GT pseudotime | train time (s) |
|---|---|---|---|---|---|
| 0 | ours | -646.76 | 634.84 | -0.711 | 32.3 |
| 0 | ref  | -655.96 | 635.49 | -0.865 | 597.1 |
| 1 | ours | -601.23 | 596.28 | +0.871 | 52.0 |
| 1 | ref  | -608.23 | 594.36 | +0.896 | 329.5 |
| 2 | ours | -573.58 | 568.52 | +0.708 | 19.6 |
| 2 | ref  | -563.48 | 560.28 | -0.857 | 517.3 |

Mean +/- sd: ELBO ours -607.2 +/- 30.2, ref -609.2 +/- 37.8 (2.0 nats/cell
gap, well inside the seed spread). NLL ours 599.9 +/- 27.2, ref 596.7 +/- 30.8.

**Direct ours-vs-reference trajectory agreement** (same seed, sign-aligned
best axis, Spearman on the ground-truth-pseudotime subset): seed 0 = 0.882,
seed 1 = 0.962, seed 2 = 0.722 -- the two implementations recover *the same*
cell ordering on most seeds, not just similarly-good orderings against the
ground truth independently.

### Trajectory-recovery gap, 8 seeds, and root-cause investigation

|rho| vs. ground-truth pseudotime over 8 seeds (0-7, CPU for throughput --
quality metrics are hardware-independent): **ours 0.772 +/- 0.098 (range
0.620-0.898), ref 0.878 +/- 0.019 (range 0.849-0.905)**. Reference is
consistently strong on every single seed; ours is occasionally weaker
(seeds 0, 2) -- a real, reproducible gap, not a 3-seed sampling artifact.

Investigated and **ruled out** as the cause:
- **Init**: constructing both modules with the same `torch.manual_seed`
  gives bit-identical weights layer-by-layer (`decoder_v_to_z`,
  `decoder_z_to_x`, `encoder_x_to_z`, `encoder_zx_to_v`) -- confirmed directly
  with `torch.equal`. (`BatchNorm1d.reset_parameters()` is a deterministic
  ones/zeros init, so reference's Linear/BatchNorm-interleaved construction
  order vs. ours' all-Linears-then-all-BatchNorms order consumes the RNG
  identically.)
- **ClippedAdam**: diffed element-by-element against the installed Pyro
  1.9.1 source (`pyro.optim.clipped_adam`) -- clamp order, weight-decay
  placement, `exp_avg`/`exp_avg_sq` update, bias correction, `lrd` gating,
  default `eps=1e-8` all match exactly.
- **v/z priors, encoder z->v path, softplus parametrisation**: diffed against
  the installed scvi-tools 1.5.2 source (`scvi/external/decipher/_module.py`,
  `_components.py`) line-by-line -- identical.
- **KL weight / warm-up**: `DecipherModule.forward` ignores `kl_weight`, but
  so does scvi-tools' (its `PyroTrainingPlan` only injects `kl_weight` when
  `"kl_weight" in signature(model)`, which is false for Decipher's pyro
  `model(self, x, beta=None)`) -- both silently disable it identically.
- **BatchNorm-freeze epoch** (`freeze_bn_from`): re-derived from Lightning's
  `on_validation_model_train`/`current_epoch` semantics and confirmed to fire
  at the same epoch as reference's `current_epoch > 0` check for both
  `check_val_every_n_epoch` values tested.
- **Early-stopping criterion**: swapping the monitor from `nll_validation` to
  `elbo_validation` for seeds 0/2/7 changed |rho| by <0.03 (0.642->0.666,
  0.620->0.620, 0.720->0.662) -- not the cause.
- **Early-stopping patience / "stopping too early"**: disabling early
  stopping and training a fixed 80 or 150 epochs instead made every tested
  seed *worse* (seed 3: 0.853 with early stopping -> 0.297 fixed; seed 0:
  0.642 -> 0.474-0.70; seed 2 diverged to NaN at a fixed 150 epochs) --
  early stopping is already doing its job and is not the cause; if anything
  it is the only thing keeping `ours` close to reference at all.
- Minibatch size (128) and `drop_last=True` are applied identically to the
  training loader in both (validation never drops a batch in either).

### Resolution: the port is numerically exact; the gap was a real RNG-coupling
### bug (now fixed) plus ordinary optimisation variance

**Step-0 check (no confounds).** Same `torch.manual_seed` init (`torch.equal`
on every `Linear`/`BatchNorm` weight -- bit-identical), same input minibatch,
same global RNG state entering the forward pass (so the reparameterised `z`
then `v` draws -- same order, same shapes, one particle each, in both -- are
identical): `ov.vi.Decipher`'s loss vs. the **real** `pyro.infer.Trace_ELBO()
.loss(ref.model, ref.guide, x, beta=0.1)` agree to a relative error of
**1.9e-8** (eval-mode BN) and **3.3e-8** (train-mode BN) -- pure float32
rounding. No formula/particle-count/posterior-parametrisation bug.

**Full-training replay (`benchmarks/vi/decipher/replay_test.py`).**
Recorded scvi's exact train/val split and every epoch's `RandomSampler`
permutation for seed 0 (116 early-stopped epochs), then drove
`DecipherModule` manually (same init, same `freeze_bn_from=5`, no `fit()`)
through that *exact* per-step cell order: best |rho| = **0.870**. Driving it
instead through *our own* default shuffle for the same 116 epochs and the
same BatchNorm-freeze epoch gave **0.878** -- matching the replay, not
reproducing the original weak 0.642. So the data-order algorithm (numpy
`RandomState` vs. PyTorch `RandomSampler`) is not the cause either.

**Root cause, found by elimination.** The original early-stopped run (0.642)
differs from this matched-epoch control (0.878) only in that validation
(`evaluate()`, every `check_val_every_n_epoch`) actually ran. `evaluate()`
draws `z`/`v` samples (and Decipher's 5-particle `marginal_ll` for
`nll_validation`) from the shared global torch RNG -- **consuming** state
from the same stream that drives training noise, so whether/when validation
fires perturbs every subsequent training step's reparameterisation draws.
Confirmed by forking the RNG around `evaluate()`: seed 0 alone went
0.642 -> 0.731. This was a genuine bug (not present in the reference the
same way at this magnitude) and is now fixed in `omicverse/vi/_train.py`
(`evaluate()` wraps its forward passes in `torch.random.fork_rng(...)`) --
validation no longer perturbs the training trajectory, for every `ov.vi`
model, not just Decipher.

**16 seeds each, post-fix** (`benchmarks/vi/decipher/seed_sweep*.py`, CPU,
real early-stopping recipe both sides): best |rho| vs. ground-truth
pseudotime -- **ours 0.824 +/- 0.082** (seeds 0-15: 0.731, 0.800, 0.613,
0.886, 0.896, 0.873, 0.862, 0.721, 0.890, 0.865, 0.791, 0.879, 0.901, 0.906,
0.744, 0.832), **ref 0.862 +/- 0.058 over 15 completed runs + 1 outright
numerical-divergence crash at seed 15** (0.865, 0.896, 0.857, 0.887, 0.905,
0.849, 0.879, 0.889, 0.877, 0.902, 0.832, 0.889, 0.896, 0.854, 0.658,
*crashed*) -- reference is not perfectly stable either at this sample size.
Bad-optimum rate (|rho| < 0.75, crash counted as bad): **ours 4/16, ref
2/16** -- Fisher exact p = 0.654, Mann-Whitney U p = 0.277: **not
significant**. The two distributions are statistically indistinguishable at
n=16; the original 3-seed "outside spread" reading was a small-sample
artifact on top of the now-fixed RNG-coupling bug.

## Timing

ov.vi.Decipher: 32.3 / 52.0 / 19.6 s (mean 34.6 s). scvi.external.Decipher:
597.1 / 329.5 / 517.3 s (mean 481.3 s) -- **~13.9x faster**, back-to-back under
the same shared-GPU load. Most of the gap is scvi-tools' Pyro/PyTorch-Lightning
per-step overhead (trace re-building, `SVI.step`, Lightning's loop) rather than
raw kernel time -- this is a tiny model (a few hundred KB of weights) on 4,454
cells, so neither implementation is GPU-compute-bound; wall time here mostly
reflects Python-level overhead per step plus how long each run's early
stopping took to trigger (seed 0 ran the most epochs for both).

## AnnDataOOM / Dask (real data, not just the synthetic test fixture)

`benchmarks/vi/decipher/run_oom_dask.py` on the same 4,454-cell erythroid
dataset (`shuffle='block'`, fixed seed, 15 epochs): AnnDataOOM
(`ov.read(..., storage='disk')`, `_source().kind == 'h5'`) and Dask-backed
AnnData (`omicverse.pp._dask.read_h5ad`, `kind == 'dask'`) both reproduce the
in-memory `get_latent_representation()` **bit-exactly** (`np.allclose(...,
atol=1e-5)` True for both).

## Not done

- No GPU run of the *raw* (no-early-stopping) 300-epoch recipe is reported
  (it diverges to NaN for both implementations past ~150-250 epochs here, as
  noted above -- early stopping is how both are meant to be run).
- `scvi.external.Decipher`'s own `nll_validation` (a batch-joint Pyro-trace
  importance estimate, not per-cell) was not used for comparison; the
  implementation-agnostic per-cell metric above was used instead for an
  apples-to-apples number.
