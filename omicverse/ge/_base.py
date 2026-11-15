"""``BaseGraphModel``: :class:`~omicverse.ge._graph.HeteroGraph` +
:class:`~omicverse.ge._kge.KGEModule` + training / persistence plumbing.

The graph-embedding analogue of ``ov.vi``'s ``BaseVIModel``: instead of a
minibatch-of-cells data source and an ELBO, the training unit here is a
batch of *edges* and the objective is PBG/SIMBA's contrastive loss (see
``_train.py``). Everything entity-type specific (how a model's graph gets
built from AnnData) lives in subclasses (:class:`~omicverse.ge._simba.SIMBA`
is the first and only one today).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ._graph import HeteroGraph
from ._kge import KGEModule
from ._train import fit


def _resolve_device(device="auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class BaseGraphModel:
    """Shared base of ``ov.ge`` graph-embedding models.

    Defaults mirror PBG/SIMBA's own (``_settings.py``): 50-d embeddings,
    dot comparator, softmax loss, 50/50 batch/uniform negatives, 10
    epochs, ``lr=0.1``.
    """

    _model_name = "base"

    def __init__(self, dimension: int = 50, init_scale: float = 1e-3,
                 max_norm: float | None = None, device: str = "auto"):
        self.dimension = dimension
        self.init_scale = init_scale
        self.max_norm = max_norm
        self.graph = HeteroGraph()
        self.module: KGEModule | None = None
        self.history_: dict = {}
        self.is_trained_ = False
        self.wd_ = 0.0
        self._device = _resolve_device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    def to_device(self, device) -> "BaseGraphModel":
        self._device = _resolve_device(device)
        if self.module is not None:
            self.module.to(self._device)
        return self

    # ------------------------------------------------------------------
    # build / train
    # ------------------------------------------------------------------
    def build(self) -> "BaseGraphModel":
        """(Re)build the embedding tables from the current graph's entity
        counts; call after the graph is fully assembled."""
        self.graph.build()
        counts = {name: self.graph.n_entities(name) for name in self.graph.entity_names()}
        self.module = KGEModule(counts, self.graph.relations, dimension=self.dimension,
                                init_scale=self.init_scale, max_norm=self.max_norm)
        self.module.to(self._device)
        return self

    def train(self, num_epochs: int = 10, *, batch_size: int = 1000, lr: float = 0.1,
              wd: float | None = None, auto_wd: bool = True, wd_interval: int = 50,
              loss_fn: str = "softmax", num_batch_negs: int = 50, num_uniform_negs: int = 50,
              eval_fraction: float = 0.05, seed: int = 0, device=None, progress: bool = True,
              early_stopping: bool = False, patience: int = 5) -> "BaseGraphModel":
        """Train the embeddings.

        ``wd=None, auto_wd=True`` (the default) reproduces SIMBA/PBG's
        ``pbg_train(auto_wd=True)`` sample-size-scaled weight decay
        (:meth:`HeteroGraph.auto_wd`); pass ``wd=`` directly to override.
        Calling ``train`` again resumes training (extends ``history``, does
        not reinitialise embeddings) unless :meth:`build` is called first.
        """
        if device is not None:
            self.to_device(device)
        if self.module is None:
            self.build()
        if wd is None:
            wd = self.graph.auto_wd() if auto_wd else 0.0
        self.wd_ = wd
        hist = fit(self.graph, self.module, num_epochs=num_epochs, batch_size=batch_size,
                  lr=lr, wd=wd, wd_interval=wd_interval, loss_fn=loss_fn,
                  num_batch_negs=num_batch_negs, num_uniform_negs=num_uniform_negs,
                  eval_fraction=eval_fraction, seed=seed, device=self._device,
                  progress=progress, early_stopping=early_stopping, patience=patience)
        if self.history_:
            for k, v in hist.items():
                if k == "_time":
                    self.history_["_time"] = self.history_.get("_time", 0.0) + v
                else:
                    self.history_.setdefault(k, []).extend(v)
        else:
            self.history_ = hist
        self.is_trained_ = True
        return self

    @property
    def history(self) -> dict:
        return self.history_

    @property
    def training_time(self):
        return self.history_.get("_time")

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def get_embedding(self, entity_type: str):
        """Embedding of ``entity_type`` as an :class:`anndata.AnnData`
        (``X`` = embedding, ``obs_names`` = the entity's original ids)."""
        import anndata
        import pandas as pd

        if self.module is None or entity_type not in self.module.embeddings:
            raise KeyError(f"no embedding for entity type {entity_type!r}; "
                           f"available: {self.graph.entity_names()}")
        X = self.module.embeddings[entity_type].detach().cpu().numpy().astype(np.float32)
        ids = self.graph.entity_ids[entity_type]
        obs = pd.DataFrame({"entity_type": [entity_type] * len(ids)},
                           index=[str(i) for i in ids])
        return anndata.AnnData(X=X, obs=obs)

    def write_embeddings(self, adata, *, cell_type: str, gene_type: str | None = None,
                         obsm_key: str = "X_simba", varm_key: str = "simba"):
        """Write the trained ``cell_type`` embedding into ``adata.obsm[obsm_key]``
        (aligned to ``adata.obs_names``) and, if given, the ``gene_type``
        embedding into ``adata.varm[varm_key]`` (aligned to ``adata.var_names``)."""
        emb = self.get_embedding(cell_type)
        order = emb.obs_names.get_indexer(adata.obs_names.astype(str))
        if (order < 0).any():
            raise ValueError(f"adata has obs not present in the {cell_type!r} embedding")
        adata.obsm[obsm_key] = emb.X[order]
        if gene_type is not None:
            gemb = self.get_embedding(gene_type)
            gorder = gemb.obs_names.get_indexer(adata.var_names.astype(str))
            if (gorder < 0).any():
                raise ValueError(f"adata has var not present in the {gene_type!r} embedding")
            adata.varm[varm_key] = gemb.X[gorder]
        return adata

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _init_params(self) -> dict:
        return {}

    def _apply_init_params(self, params: dict):
        pass

    def save(self, dir_path: str, overwrite: bool = False) -> str:
        """Write the graph (npz + json) and the model (``model.pt``: weights,
        history, init params) to ``dir_path``."""
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, "model.pt")
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True")
        self.graph.save(os.path.join(dir_path, "graph"))
        state = {
            "model": type(self).__name__,
            "format": 1,
            "dimension": self.dimension,
            "init_scale": self.init_scale,
            "max_norm": self.max_norm,
            "state_dict": ({} if self.module is None else
                           {k: v.detach().cpu() for k, v in self.module.state_dict().items()}),
            "history": self.history_,
            "is_trained": self.is_trained_,
            "wd": self.wd_,
            "init_params": self._init_params(),
        }
        torch.save(state, path)
        return path

    @classmethod
    def load(cls, dir_path: str, device: str = "auto") -> "BaseGraphModel":
        state = torch.load(os.path.join(dir_path, "model.pt"), map_location="cpu",
                           weights_only=False)
        model = cls.__new__(cls)
        BaseGraphModel.__init__(model, dimension=state["dimension"],
                               init_scale=state["init_scale"], max_norm=state["max_norm"],
                               device=device)
        model.graph = HeteroGraph.load(os.path.join(dir_path, "graph"))
        model._apply_init_params(state.get("init_params", {}))
        model.build()
        if state["state_dict"]:
            model.module.load_state_dict(state["state_dict"])
        model.history_ = state.get("history", {})
        model.is_trained_ = state.get("is_trained", True)
        model.wd_ = state.get("wd", 0.0)
        model.to_device(device)
        return model

    def __repr__(self):
        return (f"{type(self).__name__}(dimension={self.dimension}, "
                f"trained={self.is_trained_}, {self.graph!r})")


__all__ = ["BaseGraphModel"]
