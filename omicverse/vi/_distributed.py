"""Multi-process (DDP) training for the ``ov.vi`` models.

``model.train(devices=[0, 1, 2, 3])`` (GPUs, NCCL) or ``devices='cpu:4'``
(CPU processes, gloo) spawns one process per device.  Workers never receive the
AnnData / AnnDataOOM object: they get a picklable *source spec* built once in
the parent —

* AnnDataOOM: file path, dataset key, the resolved row / column selections and
  the chunk tables of ``data`` / ``indices`` (``CSRFile``; built with h5py once,
  in the parent).  Workers read with anndataoom-fast's ``pread`` reader and
  never open the file with HDF5, so there is no HDF5 file lock to contend for;
  ``HDF5_USE_FILE_LOCKING=FALSE`` is set for them anyway.
* in-memory AnnData: the CSR arrays in shared memory (no copy per worker).
* Dask-backed AnnData: the dask array; each rank computes its own blocks on
  its own device.

plus the small per-cell arrays (batch codes, labels, covariates).

Sharding: every rank permutes the blocks with the same epoch seed and takes
blocks ``rank::world``; each rank runs the same number of minibatches (the
minimum over ranks; the rest of the epoch is dropped) so the gradient
all-reduce never waits.  The effective batch size is ``batch_size * world``;
the learning rate is not rescaled (pass ``lr=`` to change it).

Only rank 0 returns results: the trained weights and history, and — when
``latent_key`` is given — the latent representation computed by all ranks on
their shards and gathered to rank 0; the parent writes it to ``obsm``.  The
source file is only ever read.
"""
from __future__ import annotations

import os
import socket
import time

import numpy as np


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _parse_devices(devices):
    """[0, 1] / ['cuda:0', 'cuda:1'] -> CUDA; 'cpu:4' / ['cpu'] * 4 / 4 (no GPU) -> CPU."""
    import torch
    if isinstance(devices, str):
        if devices.startswith("cpu"):
            n = int(devices.split(":")[1]) if ":" in devices else 1
            return ["cpu"] * n, "gloo"
        devices = [devices]
    if isinstance(devices, int):
        if torch.cuda.is_available():
            devices = list(range(devices))
        else:
            return ["cpu"] * devices, "gloo"
    out = []
    for d in devices:
        if isinstance(d, int):
            out.append(f"cuda:{d}")
        else:
            out.append(str(d))
    backend = "nccl" if all(d.startswith("cuda") for d in out) else "gloo"
    if any(d.startswith("cuda") for d in out) and backend == "gloo":
        raise ValueError("mixing CPU and CUDA devices is not supported")
    return out, backend


def _worker(rank, world, backend, port, payload, queue):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from ._data import resolve_device, source_from_spec
    from ._train import fit

    device = resolve_device(payload["devices"][rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(payload["threads"])
    dist.init_process_group(backend, init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world)
    try:
        src = source_from_spec(payload["spec"], device)
        cls, args = payload["module"]
        module = cls(**args)
        module.load_state_dict(payload["state"])
        module.to(device)
        module.train_config = payload["cfg"]
        ddp = DDP(module, device_ids=[device.index] if device.type == "cuda" else None,
                  find_unused_parameters=payload.get("find_unused", False))

        class _M:  # what fit() needs from the model
            pass

        m = _M()
        m.module = module
        t0 = time.perf_counter()
        hist = fit(m, src, payload["train"], payload["val"], max_epochs=payload["max_epochs"],
                   batch_size=payload["batch_size"], shuffle=payload["shuffle"],
                   buffer_blocks=payload["buffer_blocks"], seed=payload["seed"],
                   cfg=payload["cfg"], progress=payload["progress"], rank=rank, world=world,
                   ddp_module=ddp)
        train_time = time.perf_counter() - t0
        # BatchNorm running statistics are updated per rank; use rank 0's (the
        # weights the parent gets) everywhere
        for buf in module.buffers():
            dist.broadcast(buf.data, 0)
        latent = None
        if payload.get("latent"):
            module.eval()
            idx, zs = [], []
            with torch.no_grad():
                for b in src.iterate(None, batch_size=4096, shuffle="none", rank=rank,
                                     world=world, buffer_blocks=payload["buffer_blocks"]):
                    inf = module.inference(b["x"], b["batch"], module._cats(b), b.get("cont"),
                                           sample=False)
                    idx.append(b["idx"].cpu())
                    zs.append(inf["qz_m"].float().cpu())
            part = (torch.cat(idx).numpy(), torch.cat(zs).numpy()) if idx else None
            gathered = [None] * world if rank == 0 else None
            dist.gather_object(part, gathered, dst=0)
            if rank == 0:
                n_lat = module.n_latent
                latent = np.zeros((src.n_obs, n_lat), np.float32)
                for p in gathered:
                    if p is not None:
                        latent[p[0]] = p[1]
        if rank == 0:
            # numpy (pickled by value): torch tensors would travel as shared-memory
            # handles that die with this process
            state = {k: v.detach().cpu().numpy().copy() for k, v in module.state_dict().items()}
            queue.put({"state": state, "history": hist, "latent": latent,
                       "train_time": train_time, "epochs": m.__dict__.get("_epochs_done")})
        dist.barrier()
    except BaseException as e:
        if rank == 0:
            queue.put({"error": repr(e)})
        raise
    finally:
        dist.destroy_process_group()


def train_distributed(model, devices, max_epochs=None, *, train_size=None, validation_size=None,
                      shuffle_set_split=True, batch_size=128, early_stopping=False,
                      check_val_every_n_epoch=None, shuffle="auto", block_size=None,
                      buffer_blocks=None, seed=0, precision="32", progress=True,
                      plan_kwargs=None, threads_per_process=None, latent_key=None, **kwargs):
    """Train ``model`` with one process per device (DDP).

    Arguments:
        devices: GPU ids (``[0, 1]``), ``'cpu:N'`` or an int (number of processes).
        threads_per_process: CPU threads per worker (default: current threads / N).
        latent_key: Also compute the latent representation (sharded over ranks,
            gathered to rank 0) and store it in ``adata.obsm[latent_key]``.
        Other arguments as :meth:`BaseVIModel.train`.
    """
    import torch
    import torch.multiprocessing as mp
    from ._base import _max_epochs_heuristic
    from ._data import DEFAULT_BUFFER_BLOCKS, split_train_val
    from ._train import TrainConfig

    devs, backend = _parse_devices(devices)
    world = len(devs)
    src = model._source(device="cpu", block_size=block_size)
    spec = src.spec()
    if max_epochs is None:
        max_epochs = _max_epochs_heuristic(src.n_obs)
    train, val = split_train_val(src.n_obs, train_size, validation_size, batch_size, seed,
                                 shuffle_set_split)
    model.train_indices, model.validation_indices = train, val
    if shuffle == "auto":
        shuffle = "random" if spec["kind"] == "csr" else "block"
    cfg = TrainConfig.from_kwargs(**{**(plan_kwargs or {}), **kwargs},
                                  early_stopping=early_stopping,
                                  check_val_every_n_epoch=check_val_every_n_epoch,
                                  precision=precision, compile=False, cuda_graph=False)
    model._configure_training(cfg)
    threads = threads_per_process or max(1, torch.get_num_threads() // world)
    payload = dict(
        devices=devs, threads=threads, spec=spec,
        module=(type(model.module), model._module_args),
        state={k: v.detach().cpu() for k, v in model.module.state_dict().items()},
        cfg=cfg, train=train, val=val, max_epochs=max_epochs, batch_size=batch_size,
        shuffle=shuffle, buffer_blocks=buffer_blocks or DEFAULT_BUFFER_BLOCKS, seed=seed,
        progress=progress, latent=latent_key is not None,
        find_unused=getattr(model, "_ddp_find_unused", False),
    )
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    old = os.environ.get("HDF5_USE_FILE_LOCKING")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # inherited by the spawned workers
    try:
        pc = mp.start_processes(_worker, args=(world, backend, port, payload, queue),
                                nprocs=world, join=False, start_method="spawn")
        # read rank 0's result while the workers run (a large state dict would
        # otherwise block rank 0 on a full pipe), then join
        import queue as _queue
        res = None
        while res is None:
            try:
                res = queue.get(timeout=1.0)
            except _queue.Empty:
                if pc.join(timeout=0):  # all exited (raises if one failed)
                    break
        while not pc.join():
            pass
    finally:
        if old is None:
            os.environ.pop("HDF5_USE_FILE_LOCKING", None)
        else:
            os.environ["HDF5_USE_FILE_LOCKING"] = old
    if res is None:
        raise RuntimeError("distributed training ended without a result from rank 0")
    if "error" in res:
        raise RuntimeError(f"distributed training failed: {res['error']}")
    model.module.load_state_dict({k: torch.from_numpy(v) for k, v in res["state"].items()})
    model._merge_history(res["history"])
    model._last_train_time = res["train_time"]
    model.is_trained_ = True
    model.module.eval()
    if latent_key is not None and res["latent"] is not None:
        model.adata.obsm[latent_key] = res["latent"]
    model._world_size = world
    return model


__all__ = ["train_distributed"]
