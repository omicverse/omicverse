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


# ---------------------------------------------------------------------------
# Native Self-Organizing Map
# ---------------------------------------------------------------------------
class _SOM:
    """A minimal rectangular-grid self-organizing map (numpy only).

    This is the SOM core of FlowSOM — a Kohonen map with a Gaussian
    neighborhood that shrinks over training. After training, each node holds
    a prototype (codebook vector) in protein space.
    """

    def __init__(self, grid=(10, 10), n_features=0, random_state=0):
        self.nx, self.ny = int(grid[0]), int(grid[1])
        self.n_nodes = self.nx * self.ny
        self.n_features = n_features
        self.rng = np.random.default_rng(random_state)
        self.codes = None
        # 2-D grid coordinate of every node (for neighborhood distances)
        gx, gy = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        self._grid = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)

    def _init_codes(self, data):
        """Seed codebook vectors from random data rows + small jitter."""
        idx = self.rng.choice(data.shape[0], size=self.n_nodes, replace=True)
        jitter = self.rng.normal(0.0, 1e-3, size=(self.n_nodes, data.shape[1]))
        self.codes = data[idx].astype(np.float64) + jitter

    def train(self, data, n_epochs=10, batch=None):
        """Train the SOM with an online Kohonen update over ``n_epochs``."""
        data = np.asarray(data, dtype=np.float64)
        self.n_features = data.shape[1]
        self._init_codes(data)
        n_obs = data.shape[0]
        batch = n_obs if batch is None else int(min(batch, n_obs))

        radius0 = max(self.nx, self.ny) / 2.0
        lr0 = 0.5
        total = max(1, n_epochs * (n_obs // batch + 1))
        step = 0
        for _ in range(n_epochs):
            order = self.rng.permutation(n_obs)
            for start in range(0, n_obs, batch):
                rows = data[order[start:start + batch]]
                # winning node (best-matching unit) for each row
                d = (
                    (rows[:, None, :] - self.codes[None, :, :]) ** 2
                ).sum(axis=2)
                bmu = d.argmin(axis=1)
                frac = step / total
                radius = max(radius0 * (1.0 - frac), 1.0)
                lr = lr0 * (1.0 - frac)
                # Gaussian neighborhood weight from grid distance to BMU
                gd = (
                    (self._grid[bmu][:, None, :] - self._grid[None, :, :]) ** 2
                ).sum(axis=2)
                infl = np.exp(-gd / (2.0 * radius ** 2))  # rows x nodes
                # weighted move of every node toward its assigned rows
                w = infl * lr
                num = w.T @ rows
                den = w.sum(axis=0)[:, None]
                den[den == 0] = 1.0
                self.codes += (num / den - self.codes) * (den > 0)
                step += 1
        return self

    def winners(self, data):
        """Best-matching-unit index for each row of ``data``."""
        data = np.asarray(data, dtype=np.float64)
        d = ((data[:, None, :] - self.codes[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)


# ---------------------------------------------------------------------------
# flowsom
# ---------------------------------------------------------------------------
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
    from sklearn.cluster import AgglomerativeClustering

    mat = _cluster_matrix(adata, layer, use_rep)
    n_obs = mat.shape[0]
    n_nodes = int(grid[0]) * int(grid[1])
    if n_clusters > n_nodes:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed the SOM node count "
            f"({n_nodes}); enlarge `grid`."
        )

    # 1-2. train SOM, assign every EV to its best-matching node
    som = _SOM(grid=grid, random_state=random_state)
    som.train(mat, n_epochs=n_epochs)
    node_of_ev = som.winners(mat)

    # 3. metacluster the SOM node prototypes
    n_eff = int(min(n_clusters, n_nodes))
    meta = AgglomerativeClustering(n_clusters=n_eff, linkage=linkage)
    node_meta = meta.fit_predict(som.codes)

    # 4. propagate node metacluster label back to every EV
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
