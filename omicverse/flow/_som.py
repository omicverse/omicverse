r"""Self-organizing map — the core of FlowSOM.

WHY THIS LIVES IN ``ov.flow``
-----------------------------
FlowSOM was invented for flow cytometry (Van Gassen et al., Cytometry A 2015)
and is the field's standard clustering for marker-panel data. omicverse already
had a complete, pure-numpy implementation — but inside ``ov.single.ev``, where
it had been borrowed for single-extracellular-vesicle proteomics.

The maths is not EV-specific: it takes any ``observations x features`` matrix.
So it is promoted here rather than duplicated, and ``ov.single.ev.flowsom``
keeps its own name and its own ``uns['ev']`` output while sharing this code.
There is one implementation, called from two places — the alternative was a
second SOM that would drift from the first.

Pure numpy on purpose: `pyFlowSOM` (angelolab) is NOT open source despite its
PyPI classifier reading "Apache Software License" — its "Modified Apache
License 2.0" grants only a non-commercial, academic copyright licence and routes
commercial users to Stanford, which is GPL-3 incompatible and unusable in a paid
product. The saeyslab `flowsom` package is GPL-3, which omicverse could use but
the proprietary UI could not.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["SOM", "som_metacluster"]


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


class SOM:
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


def som_metacluster(
    matrix: np.ndarray,
    *,
    n_clusters: int = 10,
    grid: Tuple[int, int] = (10, 10),
    n_epochs: int = 20,
    linkage: str = "ward",
    random_state: int = 0,
):
    """Train a SOM and metacluster its nodes — the whole FlowSOM algorithm.

    Returns ``(node_of_obs, node_metacluster, codes)``. Deliberately knows
    nothing about AnnData: the two callers write their results into different
    places (``uns['flow']`` and ``uns['ev']``) and that is the only thing that
    differs between them.
    """
    from sklearn.cluster import AgglomerativeClustering

    n_nodes = int(grid[0]) * int(grid[1])
    if n_clusters > n_nodes:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed the SOM node count "
            f"({n_nodes}); enlarge `grid`."
        )
    som = SOM(grid=grid, random_state=random_state)
    som.train(matrix, n_epochs=n_epochs)
    node_of_obs = som.winners(matrix)
    n_eff = int(min(n_clusters, n_nodes))
    node_meta = AgglomerativeClustering(n_clusters=n_eff, linkage=linkage).fit_predict(som.codes)
    return node_of_obs, node_meta, som.codes
