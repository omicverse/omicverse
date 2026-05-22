"""Real droplet-scRNA-seq datasets for the ``ov.pp.ambient`` tutorial.

Each loader downloads a dataset asset from the ``omicverse-data`` GitHub
release (``ambient-v1``) on first use, caches it under ``dir``, and
returns it ready for the ``ov.pp.ambient`` ambient / contamination-RNA
removal workflow (SoupX / DecontX / FastCAR / scCDC).

Ambient ("soup") RNA — cell-free transcripts released by lysed cells —
leaks into every droplet during library prep. Decontamination methods
need either the *raw unfiltered* droplet matrix (to read the soup
profile out of the empty droplets — SoupX, FastCAR) or a *filtered,
clustered* matrix of real cells (DecontX, scCDC). The two loaders here
supply exactly those inputs from real public 10x Genomics data:

| Loader | Returns | Use | Source |
|---|---|---|---|
| :func:`pbmc_raw_10x` | AnnData (cells), or ``(cells, raw)`` | end-to-end decontamination on real PBMCs | 10x ``pbmc_1k_v3`` |
| :func:`hgmm_mixture` | AnnData (cells), or ``(cells, raw)`` | ground-truth validation (cross-species reads = contamination) | 10x ``hgmm_1k_v3`` |

The **hgmm** human-mouse species-mixing dataset is the ambient-RNA
ground-truth standard: a transcript mapped to the *other* species in a
cell is unambiguous contamination, so the per-cell minor-species
fraction is a direct, label-free measure of soup the tutorial can
quantify a correction against.

All data is real, public 10x Genomics single-cell RNA-seq, redistributed
at tutorial scale (~1k cells each, a subset of empty droplets retained
for the soup profile) under its original Creative Commons license.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

from .._registry import register_function
from ._datasets import download_data

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData

_RELEASE = (
    "https://github.com/omicverse/omicverse-data/releases/download/ambient-v1"
)


def _fetch(asset: str, dir: str) -> str:
    """Download ``asset`` from the ambient-v1 release; return local path."""
    return download_data(f"{_RELEASE}/{asset}", file_path=asset, dir=dir)


@register_function(
    aliases=[
        "pbmc_raw_10x", "pbmc_raw", "pbmc_1k_v3", "pbmc_ambient",
        "raw_10x_pbmc", "原始10x_PBMC", "环境RNA_PBMC",
    ],
    category="datasets",
    description=(
        "Real raw 10x Genomics PBMC dataset for the ov.pp.ambient "
        "ambient-RNA-removal tutorial — 10x Genomics 'pbmc_1k_v3' "
        "(peripheral-blood mononuclear cells, Chromium 3' v3, Cell Ranger "
        "3.0.0). Carries BOTH the filtered real cells (1,222 cells x "
        "15,300 genes, raw UMI counts in .X) and the raw unfiltered "
        "droplet matrix (13,222 droplets = the 1,222 cells + 12,000 empty "
        "droplets at 10-500 UMIs) needed to build the SoupX / FastCAR soup "
        "profile. With raw_droplets=False (default) returns just the "
        "filtered-cell AnnData; with raw_droplets=True returns a "
        "(cells, raw) tuple — pass raw= to ov.pp.ambient.remove_ambient "
        "for the SoupX / FastCAR backends. obs carries n_counts / n_genes "
        "on the cells and droplet_type ('cell' / 'empty') on the raw "
        "matrix."
    ),
    examples=[
        "adata = ov.datasets.pbmc_raw_10x()",
        "cells, raw = ov.datasets.pbmc_raw_10x(raw_droplets=True)",
        "ov.pp.ambient.remove_ambient(cells, method='soupx', raw=raw)",
    ],
    related=[
        "datasets.hgmm_mixture", "pp.ambient.remove_ambient",
    ],
)
def pbmc_raw_10x(
    dir: str = "./data", raw_droplets: bool = False
) -> Union["AnnData", Tuple["AnnData", "AnnData"]]:
    """Load the real 10x ``pbmc_1k_v3`` raw droplet dataset.

    Parameters
    ----------
    dir
        Directory the asset is cached in. Default ``'./data'``.
    raw_droplets
        When ``True`` also load and return the raw unfiltered droplet
        matrix (filtered cells + empty droplets) — required for the SoupX
        and FastCAR backends of :func:`ov.pp.ambient.remove_ambient`.

    Returns
    -------
    :class:`anndata.AnnData` or tuple
        The filtered-cell AnnData (1,222 cells x 15,300 genes), or a
        ``(cells, raw)`` tuple when ``raw_droplets=True``.
    """
    import anndata as ad
    cells = ad.read_h5ad(_fetch("pbmc_raw_10x_cells.h5ad", dir))
    if not raw_droplets:
        return cells
    raw = ad.read_h5ad(_fetch("pbmc_raw_10x_raw.h5ad", dir))
    return cells, raw


@register_function(
    aliases=[
        "hgmm_mixture", "hgmm", "hgmm_1k", "species_mixing",
        "human_mouse_mixture", "barnyard", "人鼠混合", "物种混合",
    ],
    category="datasets",
    description=(
        "Real human-mouse species-mixing ('barnyard') 10x dataset for the "
        "ground-truth-validation arm of the ov.pp.ambient tutorial — 10x "
        "Genomics 'hgmm_1k_v3' (a 1:1 mix of human HEK293T and mouse NIH3T3 "
        "cells, Chromium 3' v3, Cell Ranger 3.0.0, hg19 + mm10 reference). "
        "Because the two species share no genes, any transcript mapped to "
        "the OTHER species in a cell is unambiguous ambient contamination — "
        "so the per-cell minor-species read fraction is a direct, "
        "label-free ground-truth measure of soup. Carries the filtered "
        "real cells (1,046 cells x 29,984 hg19+mm10 genes, raw UMI counts "
        "in .X) with obs['species'] ('human' / 'mouse' / 'mixed'), "
        "obs['hg_frac'] and obs['cross_species_frac'] (the ground-truth "
        "contamination), plus the raw unfiltered droplet matrix (7,046 "
        "droplets = 1,046 cells + 6,000 empty droplets) for the soup "
        "profile. With raw_droplets=True returns a (cells, raw) tuple."
    ),
    examples=[
        "adata = ov.datasets.hgmm_mixture()",
        "cells, raw = ov.datasets.hgmm_mixture(raw_droplets=True)",
        "adata.obs['cross_species_frac'].mean()  # ground-truth soup",
    ],
    related=[
        "datasets.pbmc_raw_10x", "pp.ambient.remove_ambient",
    ],
)
def hgmm_mixture(
    dir: str = "./data", raw_droplets: bool = False
) -> Union["AnnData", Tuple["AnnData", "AnnData"]]:
    """Load the real 10x ``hgmm_1k_v3`` human-mouse species-mixing dataset.

    Parameters
    ----------
    dir
        Directory the asset is cached in. Default ``'./data'``.
    raw_droplets
        When ``True`` also load and return the raw unfiltered droplet
        matrix (filtered cells + empty droplets) — required for the SoupX
        and FastCAR backends of :func:`ov.pp.ambient.remove_ambient`.

    Returns
    -------
    :class:`anndata.AnnData` or tuple
        The filtered-cell AnnData (1,046 cells x 29,984 genes) with the
        per-cell species label and ground-truth cross-species
        contamination fraction, or a ``(cells, raw)`` tuple when
        ``raw_droplets=True``.
    """
    import anndata as ad
    cells = ad.read_h5ad(_fetch("hgmm_mixture_cells.h5ad", dir))
    if not raw_droplets:
        return cells
    raw = ad.read_h5ad(_fetch("hgmm_mixture_raw.h5ad", dir))
    return cells, raw
