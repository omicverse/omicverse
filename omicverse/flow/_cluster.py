r"""FlowSOM for cytometry — ``ov.flow.flowsom``.

The same algorithm ``ov.single.ev.flowsom`` runs, on the same code
(:mod:`omicverse.flow._som`), writing into ``uns['flow']`` instead of
``uns['ev']`` and defaulting to the layer a cytometry workflow actually
produces.

Automated clustering is a complement to gating, not a replacement. A gate is
auditable and reproducible and a reviewer can argue with it; a metacluster is
neither. Use this to find populations a strategy missed, then draw the gate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .._registry import register_function
from ._som import _cluster_matrix, som_metacluster

__all__ = ["flowsom"]


@register_function(
    aliases=["flowsom", "flow_som", "自组织映射聚类", "细胞亚群聚类", "meta_clustering"],
    category="flow",
    description=(
        "FlowSOM clustering of cytometry events: train a self-organizing map "
        "over the marker panel, then metacluster the SOM nodes into "
        "populations. The cytometry-standard unsupervised method, pure numpy "
        "with no R or Java dependency. Complements manual gating — use it to "
        "find populations a gating strategy missed."
    ),
    examples=[
        "ov.flow.flowsom(adata, n_clusters=12)",
        "ov.flow.flowsom(adata, layer='compensated', markers=['CD3','CD4','CD8'])",
    ],
    related=["flow.GatingStrategy", "single.ev.flowsom", "pp.leiden"],
)
def flowsom(
    adata,
    *,
    n_clusters: int = 10,
    grid: Tuple[int, int] = (10, 10),
    n_epochs: int = 20,
    linkage: str = "ward",
    markers: Optional[list] = None,
    layer: Optional[str] = None,
    use_rep: Optional[str] = None,
    key_added: str = "flowsom",
    random_state: int = 0,
):
    """Cluster events with FlowSOM.

    Arguments
    ---------
    markers
        Restrict clustering to these channels. Strongly recommended: including
        FSC/SSC/Time lets scatter dominate the map, and the populations that
        come out are then shape clusters rather than phenotypes.
    layer
        Which matrix to cluster. Cluster on TRANSFORMED, compensated values —
        raw fluorescence is dominated by the brightest decade and the SOM will
        follow it.
    """
    sub = adata[:, list(markers)] if markers is not None else adata
    mat = _cluster_matrix(sub, layer, use_rep)

    node_of_obs, node_meta, codes = som_metacluster(
        mat, n_clusters=n_clusters, grid=grid, n_epochs=n_epochs,
        linkage=linkage, random_state=random_state,
    )
    labels = node_meta[node_of_obs]
    adata.obs[key_added] = pd.Categorical(
        [str(c) for c in labels],
        categories=[str(c) for c in sorted(np.unique(node_meta))],
    )
    adata.obs[f"{key_added}_som"] = pd.Categorical([str(n) for n in node_of_obs])
    adata.uns.setdefault("flow", {})["flowsom"] = {
        "grid": (int(grid[0]), int(grid[1])),
        "n_nodes": int(grid[0]) * int(grid[1]),
        "n_clusters": int(len(np.unique(node_meta))),
        "n_epochs": int(n_epochs),
        "linkage": linkage,
        "markers": list(markers) if markers is not None else None,
        "node_metacluster": node_meta,
        "key_added": key_added,
    }
    return adata
