# ov.vi multi-GPU DDP -- first hardware verification

Hardware: job 52347932, 2x NVIDIA H100 80GB (sh04-04n05), shared with the user's own
jobs (both GPUs ~97% busy throughout; absolute timings below are under that
contention, 1-GPU vs 2-GPU run back-to-back for each comparison). 8 CPUs
(`srun -c8`) for all GPU runs. Scripts: `benchmarks/vi/ddp/bench_ddp.py`.

## 1. NCCL DDP actually runs on 2 GPUs

`SCVI.train(devices=[0, 1])` and `train()` under `torchrun --nproc_per_node=2`
both ran with `torch.distributed.get_backend() == "nccl"`, `world_size == 2`,
on real multi-donor data (NeurIPS 2021 BMMC donors s1d3/s2d1/s3d7 concatenated
on common genes, 27,423 cells x 13,953 genes, `batch_key='batch'`), for all
three source kinds:

| source | devices=[0,1] | notes |
|---|---|---|
| in-memory AnnData | OK | `_world_size=2`, rank-0 gathered latent == parent re-inference (`atol=1e-4`) |
| AnnDataOOM (`ov.read(storage='disk')`) | OK | see lock test below |
| Dask (cupy row blocks, `ov.pp._dask.read_h5ad(device='gpu')`) | **bug, fixed** | see Section 6 |

`torchrun --standalone --nproc_per_node=2` (AnnDataOOM, `train()` with no
`devices=`) also ran on 2 real GPUs: backend `nccl`, `world=2`, 8 epochs,
`get_latent_representation()` finite, weights identical across ranks after
training, `save()`/`load()` round-trips the DDP-trained weights exactly
(`atol=1e-5`). Both entry paths (spawn via `devices=[...]` and `torchrun`)
work.

A benign environment warning appears on every NCCL init on this cluster:
`[c10d] The client socket cannot be initialized to connect to [localhost]:PORT
(errno: 97 - Address family not supported by protocol)`. This is torch's own
IPv6-first hostname resolution for `localhost`/the node's own hostname falling
back to IPv4 (no IPv6 on this node); setting `NCCL_SOCKET_FAMILY=AF_INET` /
`GLOO_SOCKET_FAMILY=AF_INET` does not suppress it. It is non-fatal (all runs
below completed correctly) and not specific to ov.vi's code -- left as-is.

## 2. No HDF5 lock errors with a concurrent writer

Realistic scenario: this process already has the AnnDataOOM open (its own
`ov.read()` succeeded before any writer appeared), then a **separate,
`spawn`-started** process opens the same file with `h5py.File(path, 'a')`
(and sets `HDF5_USE_FILE_LOCKING=FALSE`, the same convention `_distributed.py`
uses for its own workers) and keeps touching it for the whole run. 2-GPU
AnnDataOOM DDP training (`devices=[0, 1]`, block shuffle, 10 epochs) completed
with no lock errors, `world_size=2`, finite latents.

Two related findings, not bugs in `_distributed.py` but worth recording:

- If the writer is started with the default `fork` context instead of
  `spawn`, its `h5py.File(path, 'a')` call fails with `BlockingIOError:
  ... unable to lock file`, even with `HDF5_USE_FILE_LOCKING=FALSE` set
  *inside* the forked child's function body. A forked child inherits the
  parent's already-`dlopen`'d libhdf5 and its env-var state is read once
  at that point, not reread per `H5Fopen`; setting the env var after `fork()`
  is too late. `train_distributed()` already uses
  `mp.get_context("spawn")` for its own workers for exactly this reason
  (see the comment added next to it) -- any user-written concurrent writer
  needs the same `spawn` + "set `HDF5_USE_FILE_LOCKING=FALSE` before
  importing h5py" discipline.
- Opening an AnnDataOOM **for the first time** while a writer with *default*
  (locking-on) HDF5 access already holds the file open fails outright
  (`H5Fopen(): unable to lock file`), from `anndataoom-fast`'s `read()`
  itself, before any ov.vi / DDP code runs -- standard single-writer HDF5
  semantics, since that call does not set `HDF5_USE_FILE_LOCKING=FALSE`.
  Not in scope here (it happens in user code before `model.train()`), but
  worth knowing: a process meant to open a file a writer might already hold
  needs `HDF5_USE_FILE_LOCKING=FALSE` set *before calling `ov.read()`*.

## 3. Correctness

### Disjoint sharding (unit-level, exact)

`DataSource.iterate(rows, rank=r, world=2, seed=0, epoch=e)` for `r in (0, 1)`,
checked for 3 epochs, in-memory (`shuffle='block'` and `shuffle='random'`) and
AnnDataOOM (`shuffle='block'`): the two ranks' row sets are disjoint and their
union is exactly the full row set, every epoch. `fit()` calls `iterate()` with
this same `(rank, world, seed, epoch)` so this holds for real training too.

### Latent quality, 1-GPU vs 2-GPU, same epochs (BMMC, 27,423 cells, 60 epochs)

| impl | world | elbo_val | NMI (cell_type) | ARI (cell_type) |
|---|---|---|---|---|
| 1gpu | 1 | -3938.30 | 0.7061 | 0.2824 |
| 2gpu | 2 | -3937.15 | 0.7126 | 0.2867 |

### Latent quality, 1-GPU vs 2-GPU (SEA-AD 300k, 8 epochs)

| impl | elbo_val | NMI (cell_type) | ARI (cell_type) |
|---|---|---|---|
| 1gpu | -1085.43 | 0.8532 | 0.5954 |
| 2gpu | -1084.68 | 0.8435 | 0.5823 |

### Latent quality, 1-GPU vs 2-GPU (SEA-AD full, 1,378,211 cells, 4 epochs)

| impl | elbo_val |
|---|---|
| 1gpu | -1927.03 |
| 2gpu | -1927.92 |

DDP quality matches the 1-GPU run within normal seed-to-seed spread in all
three cases (never ran multiple seeds per side given GPU contention / time;
the CPU `tests/vi/test_distributed.py::test_ddp_matches_single_process_quality`
does assert this within explicit tolerances, seed-controlled, and passes).

## 4. Scaling: 1-GPU vs 2-GPU epoch time

Same architecture (`n_hidden=128, n_latent=10`), same epoch count, run
back-to-back on the same contended GPUs.

| dataset | n_obs | epochs | impl | mean epoch time (s, after 1st) | speedup |
|---|---|---|---|---|---|
| SEA-AD 300k (in-memory) | 300,000 | 8 | 1gpu | 2.47 | 1.00x |
| SEA-AD 300k (in-memory) | 300,000 | 8 | 2gpu (`threads_per_process` default = 2) | 6.38 | 0.39x |
| SEA-AD 300k (in-memory) | 300,000 | 8 | 2gpu (`threads_per_process=8`) | 4.56 | 0.54x |
| SEA-AD full (AnnDataOOM) | 1,378,211 | 4 | 1gpu | 44.91 | 1.00x |
| SEA-AD full (AnnDataOOM) | 1,378,211 | 4 | 2gpu | 64.38 | 0.70x |

**2-GPU DDP was slower than 1 GPU at both scales tested, under this
environment's constraints.** Two separable causes, isolated with a follow-up
run (`threads_per_process=8` instead of the default
`torch.get_num_threads() // world`, which halves to 2 with the 4-thread
`OMP_NUM_THREADS` this environment exports):

- CPU-thread starvation per rank: with only 8 CPUs shared across ranks (this
  task's GPU-job budget) and `OMP_NUM_THREADS=4`, the default
  `threads_per_process = 4 // 2 = 2` under-serves the CPU-side batching /
  (for AnnDataOOM) decode work each rank needs; forcing `threads_per_process=8`
  recovered about 30% of the gap (6.38s -> 4.56s/epoch) on SEA-AD 300k.
- Even with that fixed, 2-GPU remained slower than 1-GPU (4.56s vs 2.47s).
  ov.vi's VAE models are small and fast (single-GPU epochs are 2-45s even at
  1.38M cells) -- the NCCL gradient all-reduce once per minibatch (~1,100-2,300
  minibatches/epoch at these batch sizes) adds a fixed per-step cost that, for
  a model this cheap to compute, is not repaid by halving the per-GPU batch.
  This is the expected regime for DDP: it pays off when per-step compute (big
  model / big batch) dominates communication, or when a single GPU can't hold
  the desired batch/model -- not as a wall-clock speedup for an
  already-millisecond-scale step on fast, contended, shared H100s with a
  modest CPU budget. We did not have an uncontended multi-GPU node or a larger
  CPU allocation to test whether either removes the regression; worth
  re-checking there.

## 5. Entry paths

Both confirmed on real 2-GPU hardware (NCCL): `model.train(devices=[0, 1])`
(spawns processes, `mp.get_context("spawn")`) and `torchrun
--nproc_per_node=2 script.py` with plain `model.train()` (auto-detected via
`torchrun_world() > 1`). `get_latent_representation()` and `save()`/`load()`
after DDP training both work (Section 1, Section 3).

## 6. Bug found and fixed: Dask(cupy) DDP crashed on any rank other than 0

**Symptom:** `SCVI.train(devices=[0, 1])` on a Dask(cupy)-backed AnnData
(`ov.pp._dask.read_h5ad(path, device='gpu')`) crashed rank 1 with
`cudaErrorIllegalAddress: an illegal memory access was encountered` inside
`DaskSource._block_tensors`'s `cp.cuda.get_current_stream().synchronize()`,
taking down the NCCL process group (`ncclUnhandledCudaError`). Rank 0 (device
`cuda:0`) never showed the bug; neither did the existing single-GPU
`tests/vi/test_gpu.py::test_gpu_dask_cupy_blocks` test.

**Root cause:** cupy's "current device" is thread-local and defaults to `0`.
`DaskSource._compute()` ran the dask graph with `scheduler="threads"`, i.e. in
dask's own internal thread-pool threads -- which never see the
`torch.cuda.set_device(rank)` call made on the worker's *main* thread. Those
background threads' cupy ops (`cp.empty`, `cupyx.scipy.sparse.csr_matrix`,
etc., inside `omicverse/pp/_dask.py`'s `_read_block`) therefore always
allocated on device 0, regardless of which device the DDP rank was assigned.
On rank 0 this coincidentally matched (so single-GPU and rank-0-only testing
never caught it); on rank 1 (`cuda:1`) it meant a stream on the wrong device
was synchronized / a DLPack tensor from device-0 memory was being accessed in
a process whose CUDA context had been switched to device 1, corrupting state.

**Fix** (`omicverse/vi/_data.py`):
- `DaskSource._compute()`: run the dask graph with `scheduler="synchronous"`
  (in the *calling* thread, which does have the right device set) instead of
  `"threads"` when `self.device.type == "cuda"`; CPU Dask is unaffected.
- `DaskSource._load()`: wrap the per-block compute + tensor-conversion loop in
  a new `_cupy_device_ctx(dev)` helper (`cupy.cuda.Device(dev.index)` context
  manager, no-op off CUDA), so `_block_tensors`'s stream sync also runs under
  the right device.
- `DaskSource.__init__`'s existing `cupy.get_default_memory_pool().free_all_blocks()`
  (to give back cupy pool blocks from earlier passes) now also runs inside
  that device's context instead of whatever was cupy's default.

Verified: `benchmarks/vi/ddp/bench_ddp.py --check dask` (2-GPU DDP on real
BMMC data) now completes; `tests/vi/test_gpu.py` (single-GPU, unaffected) and
the full CPU suite (`tests/vi`, 45 passed / 2 skipped) still pass. Added
`tests/vi/test_distributed.py::test_ddp_dask_cupy_two_gpus`
(`skipif(torch.cuda.device_count() < 2)`) so a 2-GPU CI runner catches a
regression here.

## Not done

- Multiple seeds per quality comparison (ran once per side given GPU
  contention / time budget; CPU `test_ddp_matches_single_process_quality`
  does assert the DDP-vs-single-process tolerance explicitly and passes).
- Did not get a clean (uncontended, more CPUs) multi-GPU measurement to see
  whether the negative scaling in Section 4 is specific to this job's 8-CPU /
  shared-GPU budget or persists on a dedicated node.
- `devices='cpu:N'` (gloo) on CPU is already covered by the existing CPU test
  suite (`test_ddp_matches_single_process_quality`,
  `test_ddp_uneven_blocks_no_deadlock`, `test_torchrun_launch`); not re-run
  here since this task is specifically about the GPU/NCCL path.
