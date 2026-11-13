r"""Variational inference for single-cell omics — ``ov.vi``.

omicverse's own implementation of the scVI model family (no scvi-tools
dependency), built to train and infer **directly on disk-backed data**:

* in-memory ``AnnData`` (CSR kept on the GPU, minibatches gathered there);
* ``AnnDataOOM`` from ``ov.read(path, storage='disk')`` (CPU / mixed mode):
  row blocks streamed with anndataoom-fast's ``pread`` reader and
  block-shuffled, host memory bounded by the shuffle buffer;
* Dask-backed ``AnnData`` from ``ov.read(path, storage='disk')`` in
  ``ov.settings.gpu_init()`` mode (cupy blocks handed to torch via DLPack).

Models
------
``SCVI``     unsupervised VAE (ZINB / NB / Poisson likelihood).
``SCANVI``   semi-supervised scANVI (label transfer, ``predict``).
``TOTALVI``  joint RNA + protein (CITE-seq) model.

The API mirrors scvi-tools, but configuration goes to the constructor::

    m = ov.vi.SCVI(adata, layer="counts", batch_key="batch", n_latent=30)
    m.train()
    adata.obsm["X_scVI"] = m.get_latent_representation()

``torch`` is imported only when a model is used.
"""
from __future__ import annotations

import importlib as _importlib

_LAZY_ATTRS = {
    "SCVI": ("._scvi", "SCVI"),
    "SCANVI": ("._scanvi", "SCANVI"),
    "TOTALVI": ("._totalvi", "TOTALVI"),
    "BaseVIModel": ("._base", "BaseVIModel"),
    "build_source": ("._data", "build_source"),
    "train_distributed": ("._distributed", "train_distributed"),
    "mde": ("._mde", "mde"),
    "distributions": ("._distributions", None),
    "nn": ("._nn", None),
    "priors": ("._priors", None),
}

__all__ = ["SCVI", "SCANVI", "TOTALVI", "BaseVIModel", "build_source", "train_distributed", "mde"]


def __getattr__(name):
    try:
        mod, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module 'omicverse.vi' has no attribute {name!r}") from None
    m = _importlib.import_module(mod, __name__)
    val = m if attr is None else getattr(m, attr)
    globals()[name] = val
    return val


def __dir__():
    return sorted(set(globals()) | set(__all__))
