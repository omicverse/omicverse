r"""Graph embedding for single-cell omics -- ``ov.ge``.

omicverse's own re-implementation of knowledge-graph embedding (the family
PyTorch-BigGraph / SIMBA belong to): no ``simba`` / ``simba_pbg`` /
``torchbiggraph`` dependency.

Unlike ``ov.vi`` (VAEs trained minibatch-of-*cells*-at-a-time), these models
learn one free embedding row per **entity** (cell, gene, peak, motif, ...)
from a typed multigraph of **edges** (e.g. "this cell expresses this gene at
this level"), with a contrastive (softmax-over-sampled-negatives) objective
-- see ``SPEC_ge_simba.md`` for the full design and its relationship to
PyTorch-BigGraph.

Data layer
----------
``HeteroGraph``   typed entities + typed relations + an edge table; builders
                  from in-memory *and* disk-backed (``AnnDataOOM`` / Dask)
                  AnnData matrices stream row blocks rather than densifying.

Model layer
-----------
``KGEModule``       per-entity-type embedding tables, relation operators,
                     dot comparator, batch+uniform negative sampling.
``RowAdagrad``       PBG's row-wise Adagrad optimiser for embedding tables.
``BaseGraphModel``   graph + module + train()/save()/load()/get_embedding().
``SIMBA``            native SIMBA (cell/gene, optionally peak/motif/k-mer)
                     built on the above, with SIMBA's own preprocessing
                     (``normalize``, ``log_transform``, ``discretize``,
                     ``select_variable_genes``), graph recipe (``gen_graph``,
                     ``infer_edges``) and post-training analysis
                     (``embed``, ``compare_entities``, ``query``).

``ov.single.pySIMBA`` keeps its original API but now runs on :class:`SIMBA`
(no external install). ``torch`` is imported only once a model is used.
"""
from __future__ import annotations

import importlib as _importlib

_LAZY_ATTRS = {
    "HeteroGraph": ("._graph", "HeteroGraph"),
    "KGEModule": ("._kge", "KGEModule"),
    "RowAdagrad": ("._optim", "RowAdagrad"),
    "BaseGraphModel": ("._base", "BaseGraphModel"),
    "SIMBA": ("._simba", "SIMBA"),
    "gen_graph": ("._simba", "gen_graph"),
    "infer_edges": ("._simba", "infer_edges"),
    "trim_edges": ("._simba", "trim_edges"),
    "discretize": ("._simba", "discretize"),
    "normalize": ("._simba", "normalize"),
    "log_transform": ("._simba", "log_transform"),
    "binarize": ("._simba", "binarize"),
    "select_variable_genes": ("._simba", "select_variable_genes"),
    "softmax_transform": ("._simba", "softmax_transform"),
    "embed": ("._simba", "embed"),
    "compare_entities": ("._simba", "compare_entities"),
    "query": ("._simba", "query"),
    "fit": ("._train", "fit"),
    "evaluate": ("._train", "evaluate"),
    "plot_history": ("._utils", "plot_history"),
    "gini": ("._utils", "gini"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name):
    try:
        mod, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module 'omicverse.ge' has no attribute {name!r}") from None
    m = _importlib.import_module(mod, __name__)
    val = getattr(m, attr)
    globals()[name] = val
    return val


def __dir__():
    return sorted(set(globals()) | set(__all__))
