"""Plain PyTorch training loop for the ``ov.vi`` models (no Lightning).

Matches scvi-tools' ``TrainingPlan`` defaults: Adam (lr 1e-3, weight decay
1e-6, eps 0.01), KL warm-up over 400 epochs (linear in the epoch), ELBO =
reconstruction + local KL averaged over cells (+ global KL), optional early
stopping on the validation ELBO and reduce-LR-on-plateau.  Per-step metrics
are accumulated on the device and read once per epoch, so the loop never
waits on the GPU inside an epoch.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    eps: float = 0.01
    optimizer: str = "Adam"
    n_epochs_kl_warmup: int | None = 400
    n_steps_kl_warmup: int | None = None
    max_kl_weight: float = 1.0
    min_kl_weight: float = 0.0
    reduce_lr_on_plateau: bool = False
    lr_factor: float = 0.6
    lr_patience: int = 30
    lr_threshold: float = 0.0
    lr_min: float = 0.0
    lr_scheduler_metric: str = "elbo_validation"
    early_stopping: bool = False
    early_stopping_monitor: str = "elbo_validation"
    early_stopping_min_delta: float = 0.0
    early_stopping_patience: int = 45
    early_stopping_mode: str = "min"
    check_val_every_n_epoch: int | None = None
    precision: str = "32"          # '32' or 'bf16' (autocast of the networks on CUDA)
    compile: bool = False
    gradient_clip_val: float | None = None
    extra: dict = field(default_factory=dict)  # model-specific (e.g. classification ratio)

    @classmethod
    def from_kwargs(cls, **kw):
        known = {k: kw.pop(k) for k in list(kw) if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.extra.update(kw)
        return cfg


def kl_weight(cfg: TrainConfig, epoch: int, step: int) -> float:
    """scvi-tools ``_compute_kl_weight``."""
    slope = cfg.max_kl_weight - cfg.min_kl_weight
    if cfg.n_epochs_kl_warmup:
        if epoch < cfg.n_epochs_kl_warmup:
            return slope * (epoch / cfg.n_epochs_kl_warmup) + cfg.min_kl_weight
    elif cfg.n_steps_kl_warmup:
        if step < cfg.n_steps_kl_warmup:
            return slope * (step / cfg.n_steps_kl_warmup) + cfg.min_kl_weight
    return cfg.max_kl_weight


def make_optimizer(params, cfg: TrainConfig, device):
    opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[cfg.optimizer.lower()]
    params = [p for p in params if p.requires_grad]
    kw = dict(lr=cfg.lr, eps=cfg.eps, weight_decay=cfg.weight_decay)
    if device.type == "cuda":
        try:
            return opt_cls(params, fused=True, **kw)
        except (RuntimeError, TypeError, ValueError):
            pass
    return opt_cls(params, **kw)


class _Meter:
    """Device-side sums of the per-cell loss terms over an epoch."""

    def __init__(self, device):
        self.device = device
        self.sums = {}
        self.n = 0

    def add(self, out: dict, n: int):
        for k, v in out.items():
            if k in ("loss",) or v is None:
                continue
            v = v.detach()
            s = v.sum() if v.dim() else v * n
            self.sums[k] = self.sums[k] + s if k in self.sums else s
        self.n += n

    def result(self, suffix: str, reduce=None) -> dict:
        keys = sorted(self.sums)
        if not keys:
            return {}
        vec = torch.stack([self.sums[k].float() for k in keys] +
                          [torch.tensor(float(self.n), device=self.device)])
        if reduce is not None:
            vec = reduce(vec)
        vec = vec.double().cpu().numpy()
        n = max(vec[-1], 1.0)
        res = {f"{k}_{suffix}": float(v / n) for k, v in zip(keys, vec[:-1])}
        rec = res.get(f"reconstruction_loss_{suffix}", 0.0)
        kl = res.get(f"kl_local_{suffix}", 0.0)
        res[f"elbo_{suffix}"] = rec + kl + res.pop(f"kl_global_{suffix}", 0.0)
        return res


def _autocast(cfg: TrainConfig, device):
    if cfg.precision in ("bf16", "bf16-mixed") and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    import contextlib
    return contextlib.nullcontext()


def evaluate(module, batches, device, cfg: TrainConfig | None = None, kl_w: float = 1.0,
             reduce=None, suffix="validation") -> dict:
    """Mean ELBO terms of ``module`` over ``batches`` (eval mode, no grad)."""
    cfg = cfg or TrainConfig()
    was = module.training
    module.eval()
    meter = _Meter(device)
    with torch.no_grad(), _autocast(cfg, device):
        for b in batches:
            out = module(b, kl_weight=kl_w)
            meter.add(out, b["x"].shape[0])
    module.train(was)
    return meter.result(suffix, reduce)


def fit(model, source, train_rows, val_rows, *, max_epochs, batch_size, shuffle,
        buffer_blocks, seed, cfg: TrainConfig, progress=True, rank=0, world=1,
        ddp_module=None, log=print):
    """Train ``model.module`` (or its DDP wrapper) for ``max_epochs`` epochs.

    Returns the per-epoch history (dict of lists)."""
    device = source.device
    module = model.module
    net = ddp_module if ddp_module is not None else module
    if cfg.compile:
        net = torch.compile(net)
    opt = make_optimizer(module.parameters(), cfg, device)
    sched = None
    if cfg.reduce_lr_on_plateau:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience,
            threshold=cfg.lr_threshold, threshold_mode="abs", min_lr=cfg.lr_min)
    check_val = cfg.check_val_every_n_epoch
    if check_val is None and (cfg.early_stopping or (
            sched is not None and cfg.lr_scheduler_metric.endswith("validation"))):
        check_val = 1
    reduce = None
    if world > 1:
        import torch.distributed as dist

        def reduce(v):
            dist.all_reduce(v)
            return v
    history: dict = {}
    best, bad = math.inf, 0
    # every train() call starts a new schedule (KL warm-up from epoch 0), as scvi-tools
    step = 0
    start_epoch = 0
    sign = 1.0 if cfg.early_stopping_mode == "min" else -1.0
    show = progress and rank == 0
    bar = None
    if show:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=max_epochs, desc="Training", unit="epoch")
        except ImportError:
            bar = None
    t_start = time.perf_counter()
    for e in range(max_epochs):
        epoch = start_epoch + e
        t0 = time.perf_counter()
        module.train()
        meter = _Meter(device)
        n_batches = None
        if world > 1:
            counts = source.n_rows_per_rank(train_rows, world, seed, epoch)
            n_batches = min(c // batch_size for c in counts)
        kw = kl_weight(cfg, epoch, step)
        for b in source.iterate(train_rows, batch_size=batch_size, shuffle=shuffle, seed=seed,
                                epoch=epoch, buffer_blocks=buffer_blocks, rank=rank,
                                world=world, n_batches=n_batches, drop_last=world > 1):
            if cfg.n_steps_kl_warmup:
                kw = kl_weight(cfg, epoch, step)
            with _autocast(cfg, device):
                out = net(b, kl_weight=kw)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            if cfg.gradient_clip_val:
                torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.gradient_clip_val)
            opt.step()
            meter.add(out, b["x"].shape[0])
            step += 1
        rec = meter.result("train", reduce)
        rec["epoch"] = epoch
        rec["kl_weight"] = kw
        rec["lr"] = opt.param_groups[0]["lr"]
        if check_val and val_rows is not None and len(val_rows) and (e + 1) % check_val == 0:
            val_batches = source.iterate(val_rows, batch_size=max(batch_size, 1024),
                                         shuffle="none", rank=rank, world=world)
            rec.update(evaluate(module, val_batches, device, cfg, kw, reduce))
        rec["time"] = time.perf_counter() - t0
        for k, v in rec.items():
            history.setdefault(k, []).append(v)
        if sched is not None and cfg.lr_scheduler_metric in rec:
            sched.step(rec[cfg.lr_scheduler_metric])
        if bar is not None:
            bar.update(1)
            bar.set_postfix({k: f"{rec[k]:.4g}" for k in ("elbo_train", "elbo_validation")
                             if k in rec})
        elif show and (e == 0 or (e + 1) % 10 == 0 or e + 1 == max_epochs):
            msg = ", ".join(f"{k}={rec[k]:.4g}" for k in ("elbo_train", "elbo_validation") if k in rec)
            log(f"epoch {epoch + 1}/{start_epoch + max_epochs}: {msg} ({rec['time']:.2f}s)")
        if cfg.early_stopping and cfg.early_stopping_monitor in rec:
            cur = sign * rec[cfg.early_stopping_monitor]
            if cur < best - cfg.early_stopping_min_delta:
                best, bad = cur, 0
            else:
                bad += 1
                if bad >= cfg.early_stopping_patience:
                    if show:
                        log(f"Early stopping at epoch {epoch + 1}: {cfg.early_stopping_monitor} "
                            f"did not improve for {bad} checks.")
                    model._epochs_done = epoch + 1
                    model._global_step = step
                    break
        model._epochs_done = epoch + 1
        model._global_step = step
    if bar is not None:
        bar.close()
    model._last_train_time = time.perf_counter() - t_start
    return history


__all__ = ["TrainConfig", "fit", "evaluate", "kl_weight", "make_optimizer"]
