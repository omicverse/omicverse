"""Vendored copy of SATURN (Species Alignment Through Unification of Rna and proteiNs).

SATURN is a deep-learning method for cross-species single-cell integration. Per
species it takes an AnnData of raw counts plus a precomputed protein-embedding
dictionary (gene symbol -> ESM2 embedding vector) and trains a "macrogene"
conditional VAE that maps every species into a shared embedding space, enabling
integration of scRNA-seq datasets across species with non-orthologous genes.

Reference
---------
Rosen, Y., Brbic, M., Roohani, Y., Swanson, K., Li, Z., & Leskovec, J. (2023).
"Toward universal cell embeddings: integrating single-cell RNA-seq datasets
across species with SATURN." *Nature Methods*.
Upstream: https://github.com/snap-stanford/SATURN

Why vendored
------------
Upstream SATURN pins a set of dependencies that conflict with the omicverse
environment (fair-esm, faiss-gpu, scvi-tools, pytorch-metric-learning,
scikit-learn-intelex). Rather than pip-installing it (which would break the
environment), the source is copied here and its intra-repository imports are
rewritten as relative imports. ``faiss`` and ``scikit-learn-intelex`` are
treated as optional and imported lazily. ``utils/``, ``losses/``, ``miners/``,
``reducers/``, ``testers/`` and ``distances/`` are a vendored copy of
pytorch-metric-learning that SATURN depends on.

Programmatic entry point
------------------------
Use :func:`omicverse.external.saturn._run.run_saturn` to run the SATURN
pipeline (pretrain + metric learning) from Python and get back a combined
AnnData with the SATURN embedding in ``.obsm['X_saturn']`` and ``species`` in
``.obs``.
"""

from ._run import run_saturn

__all__ = ["run_saturn"]
