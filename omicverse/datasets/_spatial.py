"""Real spatial datasets for the ``ov.space`` SPATA2-inspired utility tutorial.

The loader downloads its asset from the ``omicverse-data`` GitHub release
(``spata2-v1``) on first use, caches it under ``dir``, and returns it ready for
the coordinate / variable / outline / outlier helpers in :mod:`omicverse.space`.

| Loader | Returns | Modality | Source |
|---|---|---|---|
| :func:`spata2_example` | AnnData (spots x genes) | 10x Visium | SPATA2 ``example_data`` |

The dataset is the one SPATA2's own initiation vignette uses, so a tutorial
written on it can be checked against the R package's published output rather
than only against itself.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .._registry import register_function
from ._datasets import download_data

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData

_RELEASE = (
    "https://github.com/omicverse/omicverse-data/releases/download/spata2-v1"
)


def _fetch(asset: str, dir: str) -> str:
    """Download ``asset`` from the spata2-v1 release; return local path."""
    return download_data(f"{_RELEASE}/{asset}", file_path=asset, dir=dir)


@register_function(
    aliases=[
        "spata2_example", "spata2_example_data", "SPATA2示例数据",
        "SPATA2空间数据", "空间转录组示例", "visium_spata2",
    ],
    category="datasets",
    description=(
        "Real 10x Visium dataset for the ``ov.space`` SPATA2 utility "
        "tutorial — the ``example_data`` that ships with "
        "theMILOlab/SPATA2 v3.1.4 and drives the package's own "
        "customized-initiation vignette. 3,733 spots x 2,000 genes; "
        "``.X`` holds the raw integer UMI counts carried over unchanged "
        "from the R object's ``count_mtr`` (1,026,186 non-zero, 3,494,849 "
        "total), ``.obsm['spatial']`` and ``.obs['x']`` / ``.obs['y']`` "
        "hold the original pixel coordinates from ``coords_df``. The "
        "histology images in the source ``.rda`` are omitted: they are S4 "
        "EBImage objects, which is also why that ``.rda`` cannot be read "
        "from Python at all. Feed straight into "
        "``ov.space.spata2_get_coords`` / ``spata2_tissue_outline`` / "
        "``spata2_identify_outliers``. Provenance is in ``.uns['source']``."
    ),
    examples=[
        "adata = ov.datasets.spata2_example()",
        "outline = ov.space.spata2_tissue_outline(adata)",
    ],
)
def spata2_example(dir: str = "./data") -> "AnnData":
    """Load SPATA2's own ``example_data`` as an AnnData (3,733 Visium spots)."""
    import anndata as ad
    return ad.read_h5ad(_fetch("spata2_example.h5ad", dir))
