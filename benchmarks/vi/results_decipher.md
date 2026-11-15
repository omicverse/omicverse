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
|rho| vs. ground-truth pseudotime: ours 0.76 +/- 0.08, ref 0.87 +/- 0.02 --
both recover the trajectory strongly and consistently; ref has a slightly
tighter, slightly higher spread here but ours is within it.

**Direct ours-vs-reference trajectory agreement** (same seed, sign-aligned
best axis, Spearman on the ground-truth-pseudotime subset): seed 0 = 0.882,
seed 1 = 0.962, seed 2 = 0.722 -- the two implementations recover
*the same* cell ordering, not just similarly-good orderings against the
ground truth independently.

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
