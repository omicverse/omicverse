"""Real datasets for the unified cell-cell communication tutorial (``ov.single.run_ccc``).

Assets live in the ``omicverse-data`` GitHub release ``ccc-v1`` and are cached
under ``dir`` on first use.

| Loader | Returns | Modality | Source |
|---|---|---|---|
| :func:`slidetags_mouse_brain` | AnnData (nuclei x genes) | Slide-tags snRNA-seq | CytoSignal example (SCP2170) |
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .._registry import register_function
from ._datasets import download_data

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData

_RELEASE = "https://github.com/omicverse/omicverse-data/releases/download/ccc-v1"


@register_function(
    aliases=["slidetags_mouse_brain", "Slide-tags小鼠脑", "空间细胞通讯示例数据", "SCP2170", "cytosignal_example"],
    category="datasets",
    description=(
        "Real Slide-tags snRNA-seq of E14 mouse brain (Russell et al., Nature 2024; Broad SCP2170) as "
        "pre-processed for the CytoSignal tutorial: 4,623 nuclei x 26,944 genes with raw UMI counts in "
        "``.X``, ``.obs['cell_type']`` (21 types) and ``.obsm['spatial']`` (1 unit = 0.73 um). Input of the "
        "spatial section of the run_ccc tutorial (CytoSignal / COMMOT / SecAct)."
    ),
    examples=["adata = ov.datasets.slidetags_mouse_brain()",
              "comm = ov.space.run_ccc(adata, 'cell_type', method='cytosignal', species='mouse')"],
)
def slidetags_mouse_brain(dir: str = "./data") -> "AnnData":
    """Load the Slide-tags E14 mouse brain (raw counts, cell types, coordinates)."""
    import anndata as ad

    path = download_data(f"{_RELEASE}/slidetags_mouse_brain.h5ad", file_path="slidetags_mouse_brain.h5ad", dir=dir)
    adata = ad.read_h5ad(path)
    adata.var_names_make_unique()
    return adata
