"""CellChat cell-cell communication inference, reimplemented in PyTorch for OmicVerse.

A port of R CellChat 2.2.0 (Jin et al., Nat Commun 2021; Nat Protoc 2025):

* ``_engine``   -- ``computeCommunProb`` (triMean, Hill probability with
  agonist/antagonist/co-receptor terms, label-permutation p-values) in PyTorch,
  CPU (float64, R parity) or CUDA/MPS.
* ``_pipeline`` -- the surrounding R steps (``extractGene``,
  ``identifyOverExpressedGenes``/``Interactions``, ``filterCommunication``,
  ``computeCommunProbPathway``, ``aggregateNet``, ``subsetCommunication``,
  ``netAnalysis_computeCentrality``, ``identifyCommunicationPatterns``).
* ``_db``       -- CellChatDB v2 tables exported from the R package (``data/``).

The user-facing API is :func:`omicverse.single.run_cellchat`. PyTorch is imported
lazily, only when inference runs.
"""
from ._db import CellChatDB, load_cellchat_db

__all__ = ["CellChatDB", "load_cellchat_db"]
