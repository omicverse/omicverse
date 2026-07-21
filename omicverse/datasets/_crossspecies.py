"""Cross-species example datasets.

Currently ships the **frog + zebrafish embryogenesis** pair used by the SATURN
cross-species tutorial (Briggs et al. 2018, *Science*, Xenopus tropicalis
GSE113074; Wagner et al. 2018, *Science*, zebrafish). The two AnnData objects
carry **raw counts** with gene *symbols* as ``var_names`` and a ``cell_type``
label, ready for :func:`omicverse.single.cross_species_integrate`.

The h5ad files here are subsampled and gene-filtered conversions of SATURN's
official sources (a big dense TSV → a small sparse h5ad), hosted so the tutorial
runs in seconds instead of downloading ~4 GB.
"""
from __future__ import annotations

from typing import Tuple

import anndata

from ._datasets import download_data

# Small, pre-converted h5ads (raw counts + cell_type) hosted on the
# omicverse-tutorials release page.
_SATURN_FZ = {
    "frog": (
        "https://github.com/omicverse/omicverse-tutorials/releases/download/"
        "data-saturn-fz/frog_saturn.h5ad"),
    "zebrafish": (
        "https://github.com/omicverse/omicverse-tutorials/releases/download/"
        "data-saturn-fz/zebrafish_saturn.h5ad"),
}


def saturn_frog_zebrafish(dir: str = "./data") -> Tuple[anndata.AnnData, anndata.AnnData]:
    """Load the SATURN frog + zebrafish embryogenesis cross-species pair.

    Both objects hold raw counts with species-native gene symbols and a
    ``cell_type`` annotation. Frog is *Xenopus tropicalis* (query the ``'frog'``
    Ensembl species), zebrafish is *Danio rerio* (``'zebrafish'``).

    Parameters
    ----------
    dir
        Local cache directory for the downloaded h5ad files.

    Returns
    -------
    (frog, zebrafish) : tuple[AnnData, AnnData]

    Examples
    --------
    >>> import omicverse as ov
    >>> frog, zebrafish = ov.datasets.saturn_frog_zebrafish()
    >>> adata = ov.single.cross_species_integrate(
    ...     [zebrafish, frog], ['zebrafish', 'frog'],
    ...     ref_species='zebrafish', method='scVI')
    """
    frog_path = download_data(_SATURN_FZ["frog"], file_path="frog_saturn.h5ad", dir=dir)
    zeb_path = download_data(_SATURN_FZ["zebrafish"], file_path="zebrafish_saturn.h5ad", dir=dir)
    return anndata.read_h5ad(frog_path), anndata.read_h5ad(zeb_path)
