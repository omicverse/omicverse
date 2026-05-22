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
        "Real head-and-neck-cancer (HNSC) scRNA-seq atlas for the ov.single "
        "metabolism tutorial — the full Puram et al. 2017 cohort (Cell, "
        "GSE103322): 5,578 cells x 23,686 genes, log2(TPM/10+1) expression "
        "in .X. obs carries 'malignant' (Malignant / Non-malignant), "
        "'celltype' (Malignant, Fibroblast, T cell, Endothelial, B cell, "
        "Mast, Macrophage, Dendritic, myocyte), 'patient' (19 patients) and "
        "'lymph_node'. This is the dataset used by Xiao et al. 2019 "
        "(Nat Commun) to map the metabolic landscape of the tumour "
        "microenvironment — input for ov.single.Metabolism (scMetabolism / "
        "scFEA / Compass) and ov.single.MetaboliteCCC (MEBOCOST)."
    ),
    examples=[
        "adata = ov.datasets.metabolism_hnsc()",
        "met = ov.single.Metabolism(adata, method='scmetabolism'); met.run()",
    ],
    related=[
        "datasets.metabolism_compass", "single.Metabolism",
        "single.MetaboliteCCC", "single.differential_metabolism",
    ],
)
def metabolism_hnsc(dir: str = "./data") -> "AnnData":
    """Load the real Puram 2017 head-and-neck-cancer scRNA-seq atlas.

    The full GSE103322 cohort — the dataset Xiao et al. 2019 used to map
    the single-cell metabolic landscape of the tumour microenvironment.

    Parameters
    ----------
    dir
        Directory the asset is cached in. Default ``'./data'``.

    Returns
    -------
    :class:`anndata.AnnData`
        5,578 cells x 23,686 genes, log2(TPM/10+1) expression, with
        ``obs['malignant']``, ``obs['celltype']`` and ``obs['patient']``.
    """
    import anndata as ad
    path = download_data(
        f"{_RELEASE}/hnsc_puram2017_full.h5ad",
        file_path="hnsc_puram2017_full.h5ad",
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
