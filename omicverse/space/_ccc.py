"""Spatial entry point for the unified communication API (``ov.space.run_ccc``)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import anndata

from .._registry import register_function


@register_function(
    aliases=["space.run_ccc", "空间细胞通讯", "spatial cell communication", "spatial ccc"],
    category="space",
    description=("Spatial cell-cell communication through the unified run_ccc API: method='cytosignal' "
                 "(cellular-resolution LR signaling) or 'commot' (optimal transport); non-spatial methods "
                 "can be added to the list for a consensus."),
    prerequisites={"functions": [], "optional_functions": ["pp.normalize_total", "pp.log1p"]},
    requires={"obs": ["groupby"], "obsm": ["spatial"]},
    produces={"uns": ["ccc"]},
    auto_fix="none",
    examples=[
        "comm = ov.space.run_ccc(adata, groupby='cell_type', method='cytosignal')",
        "comm = ov.space.run_ccc(adata, groupby='cell_type', method='commot', method_kwargs={'dis_thr': 200})",
        "comm = ov.space.run_ccc(adata, groupby='cell_type', method=['cytosignal', 'cellchat'])",
    ],
    related=["single.run_ccc", "single.compare_ccc", "space.run_commot", "pl.ccc_heatmap"],
)
def run_ccc(
    adata: anndata.AnnData,
    groupby: str,
    method: str | Sequence[str] = "cytosignal",
    *,
    spatial_key: str = "spatial",
    method_kwargs: Mapping | None = None,
    **kwargs,
) -> anndata.AnnData:
    """Spatial communication with :func:`omicverse.single.run_ccc`.

    Identical to ``ov.single.run_ccc`` except that ``spatial_key`` defaults to
    ``'spatial'`` and ``method`` to ``'cytosignal'``. Per-cell scores of
    spatial methods are written to ``adata.obsm`` / ``adata.obsp``; the
    returned communication AnnData is aggregated to sender/receiver groups.
    """
    from ..single._ccc import run_ccc as _run

    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial coordinates adata.obsm['{spatial_key}'] not found.")
    return _run(adata, groupby, method, spatial_key=spatial_key, method_kwargs=method_kwargs, **kwargs)
