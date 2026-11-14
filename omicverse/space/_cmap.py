r"""CMAP — cell mapping across positions.

Thin omicverse wrapper over `pycmap <https://github.com/omicverse/py-CMAP>`_,
the pure-Python port of the R package CMAP
(`SuoLab-GZLab/CMAP <https://github.com/SuoLab-GZLab/CMAP>`_, MIT).

CMAP places single cells at spatial coordinates in three levels: an SVM
assigns each cell to a spatial domain, an optimiser assigns it to a spot
inside that domain, and interpolation over neighbouring spots gives a
coordinate. It sits alongside :class:`~omicverse.space.CellMap` (SpatRio,
optimal transport) and :func:`~omicverse.space.Tangram` as a third way to do
the same job; the three differ in what they need and in what they return, not
only in speed.

Unlike those two, CMAP needs the spatial object to already carry **spatial
domain labels** — by default in ``adata_st.obs['HMRF_cluster']``. Any domain
labelling works (``ov.space.clusters``, STAGATE, Leiden on a spatial graph);
pycmap also ships the HMRF route the original package used, via
``pycmap.run_hmrf``.
"""
from typing import Any, List, Optional, Sequence

from .._registry import register_function


def _get_pycmap():
    """Import pycmap lazily, with an actionable message when it is absent."""
    try:
        from pycmap import run_cmap, run_cmap_fast
    except ImportError as err:  # pragma: no cover - import guard
        raise ImportError(
            "CMAP needs the `pycmap` package:\n"
            "    pip install pycmap\n"
            "Source: https://github.com/omicverse/py-CMAP"
        ) from err
    return run_cmap, run_cmap_fast


@register_function(
    aliases=["CMAP", "cmap_mapping", "细胞定位", "三级映射", "cell mapping across positions"],
    category="space",
    description=(
        "Place single cells at spatial coordinates with CMAP's three-level "
        "strategy: SVM domain assignment, per-spot optimisation, then "
        "interpolation to an exact coordinate"
    ),
    prerequisites={
        # Spatial domain labels must exist on adata_st before this runs; any
        # domain labelling will do. pycmap.run_hmrf reproduces the original
        # package's route.
        "functions": ["space.clusters"],
        "optional_functions": ["space.pySTAGATE", "pp.pca"],
    },
    # The registry's state keys are those of a single AnnData
    # (_registry.py: valid_keys = {layers, obsm, obsp, uns, var, obs, varm}),
    # so a two-object function declares the union, as space.CellMap does.
    # Both of these live on `adata_st`, not on `adata_sc`; which object each
    # belongs to is stated in the docstring and enforced in the body.
    requires={
        "obs": ["HMRF_cluster"],
        "obsm": ["spatial"],
    },
    produces={
        "obs": [
            "CMAP_domain",
            "CMAP_svm_prob",
            "CMAP_spot",
            "CMAP_spot_prob",
            "CMAP_entropy",
            "CMAP_norm_entropy",
            "CMAP_quality",
        ],
        "obsm": ["spatial_CMAP"],
    },
    auto_fix="none",
    examples=[
        "adata_sc, cell_spot_map = ov.space.CMAP(adata_sc, adata_st, spatial_genes=svg_list)",
        "adata_sc, _ = ov.space.CMAP(adata_sc, adata_st, svg_list, spatial_data_type='square')",
        "adata_sc, _ = ov.space.CMAP(adata_sc, adata_st, svg_list, fast=True)",
    ],
    related=["space.CellMap", "space.CellLoc", "space.Tangram", "space.svg"],
)
def CMAP(
    adata_sc: Any,
    adata_st: Any,
    spatial_genes: Sequence[str],
    spatial_data_type: str = "honeycomb",
    cluster_key: str = "HMRF_cluster",
    celltype_key: Optional[str] = None,
    fast: bool = False,
    **kwargs: Any,
):
    r"""Map single cells onto spatial coordinates with CMAP.

    Arguments
    ---------
    adata_sc
        Single-cell AnnData. Raw or log-normalised counts; raw counts are
        log-normalised internally (detected from the value range).
    adata_st
        Spatial AnnData. Must carry domain labels in ``obs[cluster_key]`` and
        coordinates in ``obsm['spatial']``.
    spatial_genes
        Genes used for the level-1 domain classifier — typically spatially
        variable genes, e.g. from :func:`~omicverse.space.svg`.
    spatial_data_type
        ``'honeycomb'`` for Visium, ``'square'`` for ST arrays,
        ``'singlecell_level'`` for MERFISH-like data. This sets the
        neighbourhood geometry used in level 3.
    cluster_key
        Column in ``adata_st.obs`` holding the domain labels.
    celltype_key
        Optional column in ``adata_sc.obs`` with cell-type annotations.
    fast
        Use the accelerated path (``pycmap.run_cmap_fast``) instead of the
        reference path. Same three levels.
    kwargs
        Passed through to pycmap: ``svm_threshold``, ``num_epochs``,
        ``para_distance``, ``para_density``, ``n_near_spot``, ``dis_cut``,
        ``radius``, ``device``, ``verbose`` and the two ``pca_method_*``.

    Returns
    -------
    adata_sc
        Modified in place: ``obs['CMAP_domain', 'CMAP_svm_prob', 'CMAP_spot',
        'CMAP_spot_prob', 'CMAP_entropy', 'CMAP_norm_entropy',
        'CMAP_quality']`` and ``obsm['spatial_CMAP']``.
    cell_spot_map
        A DataFrame of the cell-to-spot assignment, for inspection.

    Notes
    -----
    Numerical agreement with the R reference has **not** been measured; the
    port records ``parity.status: unverified``. Treat results as those of the
    Python implementation rather than as reproducing the R package exactly.

    Examples
    --------
    >>> import omicverse as ov
    >>> svg_list = ov.space.svg(adata_st, mode='prost', n_svgs=3000)
    >>> adata_sc, cell_spot_map = ov.space.CMAP(
    ...     adata_sc, adata_st, spatial_genes=svg_list,
    ... )
    >>> adata_sc.obsm['spatial_CMAP'][:3]
    """
    run_cmap, run_cmap_fast = _get_pycmap()

    if cluster_key not in adata_st.obs:
        raise KeyError(
            f"adata_st.obs has no '{cluster_key}'. CMAP needs spatial domain "
            "labels before it runs — use ov.space.clusters, STAGATE, or "
            "pycmap.run_hmrf for the original package's HMRF route."
        )
    if "spatial" not in adata_st.obsm:
        raise KeyError("adata_st.obsm has no 'spatial' coordinates")

    fn = run_cmap_fast if fast else run_cmap
    return fn(
        adata_sc,
        adata_st,
        spatial_genes=list(spatial_genes),
        spatial_data_type=spatial_data_type,
        cluster_key=cluster_key,
        celltype_key=celltype_key,
        **kwargs,
    )
