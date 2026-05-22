"""Real scRNA-seq data for the ``ov.single`` metabolism tutorial.

Loaders download a dataset asset from the ``omicverse-data`` GitHub
release (``metabolism-v1``) on first use, cache it under ``dir``, and
return it ready for the single-cell metabolism workflow
(:class:`ov.single.Metabolism` — scMetabolism / Compass / scFEA — and
:class:`ov.single.MetaboliteCCC` — MEBOCOST).

| Loader | Returns | Use |
|---|---|---|
| :func:`metabolism_hnsc` | AnnData | 200-cell HNSC tumour scRNA-seq — input for every metabolism backend |
| :func:`metabolism_compass` | str (directory) | precomputed Compass reaction-flux output for the same 200 cells |

Compass is a solver-bound CLI tool (it needs an IBM CPLEX or Gurobi
licence and runs for hours), so the tutorial loads a **precomputed**
Compass run rather than recomputing it — :func:`metabolism_compass`
supplies that output directory for
``ov.single.Metabolism(method='compass').run(compass_dir=...)``.

The data is the head-and-neck squamous-cell-carcinoma (HNSC) demo from
MEBOCOST (kaifuchenlab/MEBOCOST), itself a 200-cell subset of the
Puram et al. 2017 HNSC atlas (GSE103322).
"""
from __future__ import annotations

import os
import zipfile
from typing import TYPE_CHECKING

from .._registry import register_function
from ._datasets import download_data

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData

_RELEASE = (
    "https://github.com/omicverse/omicverse-data/releases/download/metabolism-v1"
)


@register_function(
    aliases=[
        "metabolism_hnsc", "hnsc_metabolism", "metabolism_demo",
        "scrna_metabolism", "单细胞代谢数据", "HNSC代谢",
    ],
    category="datasets",
    description=(
        "Real 200-cell HNSC tumour scRNA-seq dataset for the ov.single "
        "metabolism tutorial — a head-and-neck squamous-cell-carcinoma "
        "demo (subset of Puram et al. 2017, GSE103322) with 200 cells x "
        "18,241 genes, log-normalised expression in .X. obs carries the "
        "cell-type annotation ('celltype': Malignant, Fibroblasts, CD8T, "
        "CD8Tex, CD4Tconv, Mono/Macro, Endothelial, Plasma, Mast, Myocyte, "
        "Myofibroblasts) plus a precomputed UMAP. Input for every "
        "ov.single.Metabolism backend (scMetabolism / Compass / scFEA) "
        "and for ov.single.MetaboliteCCC (MEBOCOST)."
    ),
    examples=[
        "adata = ov.datasets.metabolism_hnsc()",
        "met = ov.single.Metabolism(adata, method='scmetabolism'); met.run()",
    ],
    related=[
        "datasets.metabolism_compass", "single.Metabolism",
        "single.MetaboliteCCC",
    ],
)
def metabolism_hnsc(dir: str = "./data") -> "AnnData":
    """Load the real 200-cell HNSC tumour scRNA-seq metabolism demo.

    Parameters
    ----------
    dir
        Directory the asset is cached in. Default ``'./data'``.

    Returns
    -------
    :class:`anndata.AnnData`
        200 cells x 18,241 genes, log-normalised, with ``obs['celltype']``
        and a precomputed UMAP.
    """
    import anndata as ad
    path = download_data(
        f"{_RELEASE}/metabolism_hnsc_200cell.h5ad",
        file_path="metabolism_hnsc_200cell.h5ad",
        dir=dir,
    )
    return ad.read_h5ad(path)


@register_function(
    aliases=[
        "metabolism_compass", "compass_precomputed", "compass_hnsc",
        "compass输出", "预计算Compass",
    ],
    category="datasets",
    description=(
        "Precomputed Compass reaction-flux output for the 200-cell HNSC "
        "metabolism demo. Compass is a constraint-based metabolic-flux "
        "tool that needs a commercial LP solver (IBM CPLEX / Gurobi) and "
        "runs for hours, so the tutorial loads this precomputed result "
        "instead. Downloads + unzips a directory holding Compass "
        "'reactions.tsv.gz', 'secretions.tsv.gz' and 'uptake.tsv.gz' "
        "(reactions x the 11 HNSC cell types). Pass the returned directory "
        "to ov.single.Metabolism(method='compass').run(compass_dir=...)."
    ),
    examples=[
        "compass_dir = ov.datasets.metabolism_compass()",
        "met = ov.single.Metabolism(adata, method='compass')",
        "met.run(compass_dir=compass_dir, group_key='celltype')",
    ],
    related=[
        "datasets.metabolism_hnsc", "single.Metabolism",
    ],
)
def metabolism_compass(dir: str = "./data") -> str:
    """Download the precomputed Compass output for the HNSC metabolism demo.

    Parameters
    ----------
    dir
        Directory the asset is cached in. Default ``'./data'``.

    Returns
    -------
    str
        Path to the directory holding the Compass output TSVs — pass it as
        ``compass_dir`` to :meth:`ov.single.Metabolism.run`.
    """
    zip_path = download_data(
        f"{_RELEASE}/compass_hnsc.zip",
        file_path="compass_hnsc.zip",
        dir=dir,
    )
    out_dir = os.path.join(os.path.dirname(zip_path), "compass_hnsc")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    return out_dir
