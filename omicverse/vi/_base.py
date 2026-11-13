"""``BaseVIModel``: data source + torch module + training / inference plumbing.

Configuration is passed to the constructor (no global ``setup_anndata``
registry): the layer with raw counts, the batch / label / covariate columns
and the training genes are recorded in ``model.registry`` and re-applied to
any AnnData given later (query data, ``load``).
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import torch

from ._data import build_source, resolve_device, split_train_val, DEFAULT_BUFFER_BLOCKS
from ._train import TrainConfig, evaluate, fit


def _categories(values) -> list:
    s = pd.Series(values)
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [c for c in s.cat.categories]
    return list(pd.Categorical(s.astype(str) if s.dtype == object else s).categories)


def _codes(values, categories, key) -> np.ndarray:
    s = pd.Series(np.asarray(values))
    cats = pd.Index(categories)
    try:
        codes = cats.get_indexer(s.values)
    except Exception:  # mixed dtypes
        codes = pd.Index([str(c) for c in categories]).get_indexer(s.astype(str).values)
    if (codes < 0).any():
        bad = pd.unique(s.values[codes < 0])[:5]
        raise ValueError(f"adata.obs[{key!r}] has categories unseen by the model: {list(bad)}")
    return codes.astype(np.int64)


def _max_epochs_heuristic(n_obs, epochs_cap=400, decay_at_n_obs=20000):
    return max(min(round((decay_at_n_obs / n_obs) * epochs_cap), epochs_cap), 1)


class BaseVIModel:
    """Shared base of :class:`SCVI`, :class:`SCANVI` and :class:`TOTALVI`."""

    _model_name = "base"

    def __init__(self, adata, *, layer=None, batch_key=None, labels_key=None,
                 categorical_covariate_keys=None, continuous_covariate_keys=None, genes=None,
                 registry=None):
        self.adata = adata
        if registry is None:
            registry = self._make_registry(adata, layer, batch_key, labels_key,
                                           categorical_covariate_keys,
                                           continuous_covariate_keys, genes)
        self.registry = registry
        self.module = None
        self.history_: dict = {}
        self.is_trained_ = False
        self.train_indices = self.validation_indices = None
        self._sources: dict = {}
        self._device = torch.device("cpu")
        self._epochs_done = 0
        self._global_step = 0
        self._last_train_time = None

    # ------------------------------------------------------------------
    # registry
    # ------------------------------------------------------------------
    def _make_registry(self, adata, layer, batch_key, labels_key, cat_keys, cont_keys, genes):
        from ._data import gene_selection
        obs = adata.obs
        for k in [batch_key, labels_key] + list(cat_keys or []) + list(cont_keys or []):
            if k is not None and k not in obs.columns:
                raise KeyError(f"{k!r} is not a column of adata.obs")
        sel = gene_selection(adata.var, genes)
        var_names = list(adata.var_names if sel is None else adata.var_names[sel])
        reg = {
            "layer": layer,
            "batch_key": batch_key,
            "labels_key": labels_key,
            "categorical_covariate_keys": list(cat_keys or []),
            "continuous_covariate_keys": list(cont_keys or []),
            "var_names": [str(v) for v in var_names],
            "batch_categories": _categories(obs[batch_key]) if batch_key else [0],
            "label_categories": _categories(obs[labels_key]) if labels_key else [0],
            "cat_categories": [_categories(obs[k]) for k in (cat_keys or [])],
        }
        return reg

    @property
    def n_batch(self):
        return len(self.registry["batch_categories"])

    @property
    def n_labels(self):
        return len(self.registry["label_categories"])

    @property
    def n_vars(self):
        return len(self.registry["var_names"])

    def _genes_for(self, adata):
        names = self.registry["var_names"]
        if len(names) == adata.n_vars and list(map(str, adata.var_names)) == names:
            return None
        idx = adata.var_names.get_indexer(names)
        if (idx < 0).any():
            raise ValueError("adata lacks genes the model was trained on "
                             f"(e.g. {[n for n, i in zip(names, idx) if i < 0][:5]})")
        return idx

    def _obs_arrays(self, adata) -> dict:
        r = self.registry
        n = adata.n_obs
        obs = adata.obs
        out = {}
        out["batch"] = (_codes(obs[r["batch_key"]].values, r["batch_categories"], r["batch_key"])
                        if r["batch_key"] else np.zeros(n, np.int64))
        out["labels"] = self._label_codes(adata)
        if r["categorical_covariate_keys"]:
            out["cat"] = np.stack([_codes(obs[k].values, cats, k) for k, cats in
                                   zip(r["categorical_covariate_keys"], r["cat_categories"])], 1)
        if r["continuous_covariate_keys"]:
            out["cont"] = np.stack([np.asarray(obs[k].values, dtype=np.float32)
                                    for k in r["continuous_covariate_keys"]], 1)
        out.update(self._extra_obs(adata))
        return out

    def _label_codes(self, adata):
        r = self.registry
        if not r["labels_key"]:
            return np.zeros(adata.n_obs, np.int64)
        return _codes(adata.obs[r["labels_key"]].values, r["label_categories"], r["labels_key"])

    def _extra_obs(self, adata) -> dict:
        return {}

    # ------------------------------------------------------------------
    # data sources
    # ------------------------------------------------------------------
    def _source(self, adata=None, device=None, **kw):
        """Minibatch source for ``adata`` on ``device`` (cached; inference reuses
        the training source)."""
        adata = self.adata if adata is None else adata
        device = resolve_device(device or self._device)
        key = (id(adata), str(device))
        kw = {k: v for k, v in kw.items() if v is not None and not (k == "resident" and v == "auto")}
        hit = self._sources.get(key)
        if hit is not None and (not kw or hit[0] == kw):
            return hit[1]
        if hit is not None:
            hit[1].close()
        src, _ = build_source(adata, layer=self.registry["layer"], genes=self._genes_for(adata),
                              obs=self._obs_arrays(adata), device=device, **kw)
        if adata is self.adata:  # one source per device for the model's own data
            for k in [k for k in self._sources if k[0] == id(adata)]:
                self._sources.pop(k)[1].close()
        self._sources[key] = (kw, src)
        self._check_counts(src)
        return src

    def _check_counts(self, src):
        try:
            x = src.first_rows(64)
        except Exception:  # never block training on the check
            return
        if x.size and (np.any(x < 0) or not np.allclose(x, np.round(x))):
            warnings.warn(f"{type(self).__name__}: the input does not look like raw counts "
                          f"(layer={self.registry['layer']!r}); the likelihood assumes counts.",
                          UserWarning, stacklevel=4)

    def to_device(self, device):
        self._device = resolve_device(device)
        if self.module is not None:
            self.module.to(self._device)
        return self

    @property
    def device(self):
        return self._device

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train(self, max_epochs: int | None = None, *, device="auto", devices=None,
              train_size: float | None = None, validation_size: float | None = None,
              shuffle_set_split: bool = True, batch_size: int = 128,
              early_stopping: bool = False, check_val_every_n_epoch: int | None = None,
              shuffle: str = "auto", block_size: int | None = None,
              buffer_blocks: int | None = None, resident="auto", seed: int = 0,
              precision: str = "32", compile: bool = False, progress: bool = True,
              plan_kwargs: dict | None = None, **kwargs):
        """Train the model.

        Arguments:
            max_epochs: Passes over the training cells; default
                ``min(round(20000 / n_cells * 400), 400)`` (scvi-tools heuristic).
            device: ``'auto'`` (CUDA if available), ``'cpu'``, ``'cuda'``, ``'cuda:1'``...
            devices: Several devices (e.g. ``[0, 1]`` or ``'cpu:4'``) to train with one
                process each (DDP; see :func:`omicverse.vi.train_distributed`).
            train_size, validation_size, shuffle_set_split: scvi-tools' split
                (default 90 % train, the rest validation).
            batch_size: Minibatch size.
            early_stopping: Stop when ``elbo_validation`` stops improving
                (``early_stopping_patience`` checks, default 45).
            check_val_every_n_epoch: Validation frequency (default: only with
                early stopping / LR plateau, as scvi-tools).
            shuffle: ``'random'`` (global permutation, in-memory data on the
                device), ``'block'`` (block shuffle with a buffer; disk data) or
                ``'auto'`` (random when resident, block otherwise).
            block_size, buffer_blocks: Block-shuffle geometry (cells per block,
                blocks per shuffle buffer).
            resident: Keep the matrix on the device (``'auto'``: in-memory data that fits).
            seed: Seed for initialisation-independent randomness (split, shuffle,
                sampling).  Set it before constructing the model for full determinism.
            precision: ``'32'`` or ``'bf16'`` (autocast networks on CUDA).
            compile: ``torch.compile`` the module.
            plan_kwargs: Optimisation options (``lr``, ``weight_decay``, ``eps``,
                ``n_epochs_kl_warmup``, ``n_steps_kl_warmup``, ``reduce_lr_on_plateau``,
                ``lr_factor``, ``lr_patience``, ...); also accepted as keywords.
        """
        if devices is not None and not (isinstance(devices, (list, tuple)) and len(devices) == 1):
            from ._distributed import train_distributed
            return train_distributed(self, devices=devices, max_epochs=max_epochs,
                                     train_size=train_size, validation_size=validation_size,
                                     shuffle_set_split=shuffle_set_split, batch_size=batch_size,
                                     early_stopping=early_stopping,
                                     check_val_every_n_epoch=check_val_every_n_epoch,
                                     shuffle=shuffle, block_size=block_size,
                                     buffer_blocks=buffer_blocks, seed=seed,
                                     precision=precision, progress=progress,
                                     plan_kwargs=plan_kwargs, **kwargs)
        if isinstance(devices, (list, tuple)):
            device = devices[0] if not isinstance(devices[0], int) else f"cuda:{devices[0]}"
        device = resolve_device(device)
        self.to_device(device)
        src = self._source(device=device, block_size=block_size, resident=resident)
        if max_epochs is None:
            max_epochs = _max_epochs_heuristic(src.n_obs)
        train, val = split_train_val(src.n_obs, train_size, validation_size, batch_size, seed,
                                     shuffle_set_split)
        self.train_indices, self.validation_indices = train, val
        if shuffle == "auto":
            shuffle = "random" if src.resident else "block"
        cfg = TrainConfig.from_kwargs(**{**(plan_kwargs or {}), **kwargs},
                                      early_stopping=early_stopping,
                                      check_val_every_n_epoch=check_val_every_n_epoch,
                                      precision=precision, compile=compile)
        self._configure_training(cfg)
        devs = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(seed + 1)  # dropout / reparameterisation noise
            hist = fit(self, src, train, val, max_epochs=max_epochs, batch_size=batch_size,
                       shuffle=shuffle, buffer_blocks=buffer_blocks or DEFAULT_BUFFER_BLOCKS,
                       seed=seed, cfg=cfg, progress=progress)
        self._merge_history(hist)
        self.is_trained_ = True
        self.module.eval()
        return self

    def _configure_training(self, cfg: TrainConfig):
        """Hook for model-specific training options (e.g. SCANVI classification ratio)."""
        self.module.train_config = cfg

    def _merge_history(self, hist):
        if not self.history_:
            self.history_ = {k: list(v) for k, v in hist.items()}
        else:
            for k, v in hist.items():
                self.history_.setdefault(k, []).extend(v)

    @property
    def history(self) -> dict:
        """Per-epoch metrics: dict of one-column DataFrames indexed by epoch (as scvi-tools)."""
        h = self.history_
        if not h:
            return {}
        epochs = pd.Index(range(len(h.get("epoch", []))), name="epoch")
        out = {}
        for k, v in h.items():
            if k == "epoch":
                continue
            if len(v) == len(epochs):
                out[k] = pd.DataFrame({k: v}, index=epochs)
            else:  # validation metrics checked every n epochs
                out[k] = pd.DataFrame({k: v})
        return out

    @property
    def training_time(self):
        """Wall time (s) of the last ``train()`` call."""
        return self._last_train_time

    # ------------------------------------------------------------------
    # inference helpers
    # ------------------------------------------------------------------
    def _check_trained(self):
        if self.module is None:
            raise RuntimeError("the model has no module")
        if not self.is_trained_:
            warnings.warn("the model has not been trained yet", UserWarning, stacklevel=3)

    def _batches(self, adata=None, indices=None, batch_size=None):
        adata = self.adata if adata is None else adata
        src = self._source(adata)
        rows = None
        order = None
        if indices is not None:
            indices = np.asarray(indices)
            if indices.dtype == bool:
                indices = np.flatnonzero(indices)
            rows, order = np.unique(indices.astype(np.int64), return_inverse=True)
        bs = batch_size or 1024
        n = src.n_obs if rows is None else len(rows)
        return src, src.iterate(rows, batch_size=bs, shuffle="none"), rows, order, n

    def _collect(self, fn, adata=None, indices=None, batch_size=None):
        """Run ``fn(batch) -> tensor (B, ...)`` over cells in order; numpy result
        in the order of ``indices`` (or of the cells)."""
        self._check_trained()
        src, it, rows, order, n = self._batches(adata, indices, batch_size)
        was = self.module.training
        self.module.eval()
        out = None
        with torch.no_grad():
            for b in it:
                v = fn(b)
                if out is None:
                    out = torch.empty((n,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
                pos = b["idx"] if rows is None else torch.from_numpy(
                    np.searchsorted(rows, b["idx"].cpu().numpy())).to(v.device)
                out[pos] = v
        self.module.train(was)
        res = out.cpu().numpy() if out is not None else np.zeros((0,))
        return res if order is None else res[order]

    def get_elbo(self, adata=None, indices=None, batch_size: int | None = None) -> float:
        """ELBO (higher is better; mean over cells) — scvi-tools ``get_elbo``."""
        self._check_trained()
        src, it, *_ = self._batches(adata, indices, batch_size)
        res = evaluate(self.module, it, src.device, suffix="eval")
        return -res["elbo_eval"]

    def get_reconstruction_error(self, adata=None, indices=None, batch_size=None) -> float:
        self._check_trained()
        src, it, *_ = self._batches(adata, indices, batch_size)
        res = evaluate(self.module, it, src.device, suffix="eval")
        return -res["reconstruction_loss_eval"]

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _init_params(self) -> dict:
        return dict(self._module_kwargs)

    def save(self, dir_path: str, overwrite: bool = False, save_anndata: bool = False):
        """Write ``model.pt`` (weights, registry, init parameters, history) to ``dir_path``."""
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, "model.pt")
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True")
        state = {
            "model": type(self).__name__,
            "format": 1,
            "registry": self.registry,
            "init_params": self._init_params(),
            "state_dict": {k: v.detach().cpu() for k, v in self.module.state_dict().items()},
            "history": self.history_,
            "is_trained": self.is_trained_,
            "train_indices": self.train_indices,
            "validation_indices": self.validation_indices,
        }
        torch.save(state, path)
        with open(os.path.join(dir_path, "registry.json"), "w") as f:
            json.dump({"model": type(self).__name__, "registry": self.registry,
                       "init_params": {k: v for k, v in self._init_params().items()
                                       if isinstance(v, (int, float, str, bool, list, type(None)))}},
                      f, indent=1, default=str)
        if save_anndata:
            if hasattr(self.adata, "write_h5ad"):
                self.adata.write_h5ad(os.path.join(dir_path, "adata.h5ad"))
        return path

    @classmethod
    def load(cls, dir_path: str, adata=None, device="auto"):
        """Rebuild a saved model on ``adata`` (same genes / categories as training)."""
        state = torch.load(os.path.join(dir_path, "model.pt"), map_location="cpu",
                           weights_only=False)
        if adata is None:
            p = os.path.join(dir_path, "adata.h5ad")
            if not os.path.exists(p):
                raise ValueError("pass adata= (the model was saved without its AnnData)")
            import anndata
            adata = anndata.read_h5ad(p)
        model = cls.__new__(cls)
        BaseVIModel.__init__(model, adata, registry=state["registry"])
        model._build(**state["init_params"])
        model.module.load_state_dict(state["state_dict"])
        model.history_ = state.get("history", {})
        model.is_trained_ = state.get("is_trained", True)
        model.train_indices = state.get("train_indices")
        model.validation_indices = state.get("validation_indices")
        model.to_device(device)
        model.module.eval()
        return model

    def __repr__(self):
        r = self.registry
        return (f"{type(self).__name__}(n_obs={self.adata.n_obs}, n_vars={self.n_vars}, "
                f"n_batch={self.n_batch}, n_labels={self.n_labels}, layer={r['layer']!r}, "
                f"trained={self.is_trained_})")


__all__ = ["BaseVIModel"]
