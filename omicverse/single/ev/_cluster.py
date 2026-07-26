"""Clustering of EV subpopulations for single-EV proteomics.

EV-subpopulation discovery is the core of single-EV analysis: an EV x protein
AnnData is partitioned into vesicle subpopulations that share a surface-marker
profile.

This module keeps only the EV-specific routines:

* :func:`flowsom` — a native, pure-Python FlowSOM: a self-organizing map
  (SOM) is trained on the EV x protein matrix, then the SOM nodes are
  consensus / hierarchically *metaclustered* into the requested number of
  EV subpopulations. FlowSOM is the cytometry-standard clustering for this
  kind of marker-panel data; the native implementation removes any R / Java
  dependency.
* :func:`subpopulation_abundance` builds the per-sample subpopulation-
  frequency table for downstream differential-abundance testing.

Graph-based clustering and the UMAP embedding are omicverse-native — use
:func:`omicverse.pp.leiden` and :func:`omicverse.pp.umap` (after
:func:`omicverse.pp.neighbors`) instead.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..._registry import register_function


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dense(x):
    """Return a dense float64 ndarray from a (possibly sparse) matrix."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def _cluster_matrix(adata, layer, use_rep):
    """Resolve the EV x feature matrix used for clustering."""
    if use_rep is not None and use_rep in adata.obsm:
        return _dense(adata.obsm[use_rep])
    if layer is not None and layer in adata.layers:
        return _dense(adata.layers[layer])
    return _dense(adata.X)


@register_function(
    aliases=["ev_flowsom", "flowsom_ev", "FlowSOM", "自组织映射聚类"],
    category="ev",
    description=(
        "Native pure-Python FlowSOM clustering of EV subpopulations. A "
        "self-organizing map (SOM) is trained on the EV x protein matrix, "
        "then the SOM nodes are hierarchically metaclustered into n_clusters "
        "EV subpopulations. No R / Java dependency. Writes the metacluster "
        "label into obs."
    ),
    examples=[
        "ov.single.ev.flowsom(adata, n_clusters=8)",
        "ov.single.ev.flowsom(adata, n_clusters=10, grid=(12, 12))",
        "ov.single.ev.flowsom(adata, n_clusters=6, use_rep='X_pca')",
    ],
    related=["pp.leiden", "single.ev.subpopulation_abundance"],
)
def flowsom(
    adata,
    *,
    n_clusters: int = 10,
    grid=(10, 10),
    n_epochs: int = 20,
    linkage: str = "ward",
    layer: Optional[str] = "scaled",
    use_rep: Optional[str] = None,
    key_added: str = "flowsom",
    random_state: int = 0,
):
    """Native FlowSOM clustering of EV subpopulations.

    The FlowSOM algorithm: (1) train a self-organizing map so each grid node
    becomes a prototype of a region of protein space; (2) assign every EV to
    its best-matching node; (3) *metacluster* the node prototypes with
    agglomerative hierarchical clustering into ``n_clusters`` groups; (4)
    propagate the node-level metacluster label back to every EV.

    Parameters
    ----------
    adata
        EV x protein AnnData.
    n_clusters
        Number of EV subpopulations (metaclusters) to recover.
    grid
        ``(nx, ny)`` size of the SOM grid; ``nx * ny`` should comfortably
        exceed ``n_clusters``.
    n_epochs
        Number of SOM training epochs.
    linkage
        Linkage for the agglomerative metaclustering step — ``'ward'``
        (default, the FlowSOM convention) | ``'average'`` | ``'complete'``
        | ``'single'``.
    layer
        Layer used for clustering (default ``'scaled'`` if present, else
        ``X``). Ignored when ``use_rep`` is given.
    use_rep
        ``obsm`` key to cluster on instead of a layer (e.g. ``'X_pca'``).
    key_added
        ``obs`` column the metacluster label is written to.
    random_state
        Random seed (SOM initialization and training order).

    Returns
    -------
    :class:`anndata.AnnData`
        The same object with the metacluster label in ``obs[key_added]``,
        the per-EV SOM node in ``obs[key_added + '_som']``, and the SOM /
        metacluster details in ``uns['ev']['flowsom']``.
    """
    # The SOM itself lives in `ov.flow._som`: FlowSOM is a flow-cytometry
    # algorithm that this module borrowed, and its maths is not EV-specific.
    # One implementation, two callers — the only thing that differs is where
    # the result is written.
    from ...flow._som import som_metacluster

    mat = _cluster_matrix(adata, layer, use_rep)
    n_nodes = int(grid[0]) * int(grid[1])
    node_of_ev, node_meta, _codes = som_metacluster(
        mat, n_clusters=n_clusters, grid=grid, n_epochs=n_epochs,
        linkage=linkage, random_state=random_state,
    )
    n_eff = int(len(np.unique(node_meta)))
    ev_meta = node_meta[node_of_ev]
    adata.obs[key_added] = pd.Categorical(
        [str(c) for c in ev_meta],
        categories=[str(c) for c in sorted(np.unique(node_meta))],
    )
    adata.obs[f"{key_added}_som"] = pd.Categorical([str(n) for n in node_of_ev])

    ev = adata.uns.setdefault("ev", {})
    ev["flowsom"] = {
        "grid": (int(grid[0]), int(grid[1])),
        "n_nodes": n_nodes,
        "n_clusters": n_eff,
        "n_epochs": int(n_epochs),
        "linkage": linkage,
        "node_metacluster": node_meta,
        "key_added": key_added,
    }
    return adata


# ---------------------------------------------------------------------------
# subpopulation_abundance
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "ev_subpopulation_abundance", "subpopulation_abundance_ev",
        "EV亚群丰度", "囊泡亚群丰度",
    ],
    category="ev",
    description=(
        "Per-sample / per-condition frequency of each EV subpopulation — a "
        "table (samples x subpopulations) of counts or fractions, the input "
        "for downstream differential-abundance testing."
    ),
    examples=[
        "tab = ov.single.ev.subpopulation_abundance(adata, groupby='sample', "
        "cluster_key='flowsom')",
        "tab = ov.single.ev.subpopulation_abundance(adata, groupby='condition', "
        "cluster_key='leiden', normalize=False)",
    ],
    related=["single.ev.flowsom", "pp.leiden"],
)
def subpopulation_abundance(
    adata,
    *,
    groupby: str,
    cluster_key: str,
    normalize: bool = True,
):
    """Per-sample frequency of each EV subpopulation.

    Parameters
    ----------
    adata
        EV x protein AnnData with a clustering in ``obs[cluster_key]``.
    groupby
        ``obs`` column defining the samples / conditions (rows of the
        output table).
    cluster_key
        ``obs`` column with the EV-subpopulation labels (columns of the
        output table).
    normalize
        Return per-sample fractions (default) instead of raw EV counts.

    Returns
    -------
    :class:`pandas.DataFrame`
        ``samples x subpopulations`` table of subpopulation frequencies (or
        counts), ready for differential-abundance testing.
    """
    for col in (groupby, cluster_key):
        if col not in adata.obs:
            raise KeyError(f"obs[{col!r}] not found.")
    tab = pd.crosstab(adata.obs[groupby], adata.obs[cluster_key])
    if normalize:
        tab = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return tab
