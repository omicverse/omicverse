"""Vendored copy of CAMEX (Cross-species cell-type Annotation and integration
via Multi-species EXpression graphs).

CAMEX is a graph-neural-network method for cross-species single-cell
integration and annotation. Given one AnnData per species (raw counts, gene
symbols) plus many-to-many gene-homology tables linking the species, it builds
a DGL heterograph of cells and genes, connects homologous genes across species,
and trains a relational GNN autoencoder that embeds every species into a shared
latent space (integration) and, optionally, transfers cell-type labels from
annotated reference species to unannotated query species (annotation).

Reference
---------
Guo, Z.-H., et al. "CAMEX: cross-species cell-type annotation and integration
via multi-species expression graphs."
Upstream: https://github.com/zhanglabtools/CAMEX (branch ``main``, MIT license)

Why vendored
------------
CAMEX is not published on PyPI and its ``setup.py`` pins a conflicting set of
dependencies (``numpy>=1.22.4,<=1.24.4``, ``scanpy<=1.9.3``, ``pandas==1.5.3``,
``numba==0.56.4``, ``harmonypy==0.0.9``) that are incompatible with the
omicverse environment (numpy 2.x, newer scanpy). Rather than pip-installing it
(which would break the environment), the ``CAMEX/`` package source is copied
here and patched to run under the current environment:

* intra-package ``from CAMEX.xxx import ...`` imports rewritten as relative
  imports (``from .xxx import ...``);
* ``dgl.dataloading.NodeDataLoader`` (removed in DGL >=1.0) resolved via a
  compatibility shim to ``dgl.dataloading.DataLoader``;
* the heavy optional ``scib`` / ``scanpy.external`` imports in
  :mod:`integration_metrics` made lazy so ``import omicverse.external.camex``
  stays light.

The vendored modules (``base``, ``model``, ``layer``, ``loss``, ``trainer``,
``params``, ``preprocess_untils``, ``train_untils``, ``integration_metrics``)
are imported lazily so that merely importing this package does not require
``torch``/``dgl`` to be present.

Programmatic entry point
------------------------
Use :func:`omicverse.external.camex._run.run_camex` to run the CAMEX
integration pipeline from Python and get back a combined AnnData with the CAMEX
integration embedding in ``.obsm['X_CAMEX_Integration']`` and ``species`` /
``batch`` in ``.obs``.
"""

__all__ = ["run_camex", "Dataset", "Trainer"]


def __getattr__(name):
    # Lazy re-exports so ``import omicverse.external.camex`` does not eagerly
    # import torch/dgl-heavy submodules.
    if name == "run_camex":
        from ._run import run_camex
        return run_camex
    if name == "Dataset":
        from .base import Dataset
        return Dataset
    if name == "Trainer":
        from .trainer import Trainer
        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
