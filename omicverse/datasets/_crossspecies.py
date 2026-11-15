"""Cross-species example datasets.

Ships two cross-species pairs:

* **frog + zebrafish** embryogenesis, used by the SATURN cross-species
  tutorial (Briggs et al. 2018, *Science*, Xenopus tropicalis GSE113074;
  Wagner et al. 2018, *Science*, zebrafish). Raw counts, gene *symbols*,
  ``cell_type`` label, ready for :func:`omicverse.single.cross_species_integrate`.
* **Baron et al. 2016 pancreas** (human vs. mouse, GEO GSE84133), used by the
  SysVI cross-system integration tutorial (Hrovatin et al. 2023). Fetched
  directly from the public GEO FTP mirror and the public MGI/JAX
  human-mouse ortholog table — no pre-converted mirror.
"""
from __future__ import annotations

import os
from typing import Tuple

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._datasets import download_data, download_data_requests

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


# GEO GSE84133 supplementary files (public FTP mirror); {gsm}_{sample}_umifm_counts.csv.gz
_BARON_SAMPLES = {
    "human1_umifm_counts.csv.gz": "GSM2230757",
    "human2_umifm_counts.csv.gz": "GSM2230758",
    "human3_umifm_counts.csv.gz": "GSM2230759",
    "human4_umifm_counts.csv.gz": "GSM2230760",
    "mouse1_umifm_counts.csv.gz": "GSM2230761",
    "mouse2_umifm_counts.csv.gz": "GSM2230762",
}
_BARON_GEO_URL = "https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm_nnn}/{gsm}/suppl/{gsm}_{fname}"
# MGI/JAX human-mouse one-to-one ortholog table (public, not GEO/figshare).
_JAX_ORTHOLOG_URL = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"


def _baron_read_sample(path: str, batch: str) -> anndata.AnnData:
    """One Baron GSM csv.gz: rows are cells, first two columns are barcode and
    the authors' cluster label, the rest are gene UMI counts."""
    df = pd.read_csv(path, index_col=0)
    obs = df[["barcode", "assigned_cluster"]].copy()
    X = sp.csr_matrix(df.drop(columns=["barcode", "assigned_cluster"]).values.astype(np.float32))
    a = anndata.AnnData(X, obs=obs, var=pd.DataFrame(index=df.columns[2:]))
    a.obs["batch"] = batch
    return a


def _baron_one2one_orthologs(path: str) -> dict:
    """One mouse symbol -> one human symbol, from MGI's human-mouse homology
    classes, keeping only unambiguous (one gene per class and species) pairs."""
    t = pd.read_csv(path, sep="\t")
    t = t[["DB Class Key", "Common Organism Name", "Symbol"]]
    mm = t[t["Common Organism Name"].str.startswith("mouse")]
    hs = t[t["Common Organism Name"] == "human"]
    mm = mm[~mm["DB Class Key"].duplicated(keep=False)]
    hs = hs[~hs["DB Class Key"].duplicated(keep=False)]
    m = mm.merge(hs, on="DB Class Key", suffixes=("_mm", "_hs"))
    return dict(zip(m["Symbol_mm"], m["Symbol_hs"]))


def baron_pancreas_human_mouse(dir: str = "./data", n_top_genes: int = 2000) -> anndata.AnnData:
    """Load Baron et al. 2016 pancreas scRNA-seq, human and mouse, harmonized
    for cross-*system* integration (SysVI, Hrovatin et al. 2023).

    Downloads the six GEO GSE84133 supplementary count tables (4 human
    donors, 2 mouse strains) and the public MGI/JAX one-to-one human-mouse
    ortholog table, then: per-species ``normalize_total(1e4)`` + ``log1p``;
    maps mouse genes to their human ortholog symbol; keeps genes shared by
    both species; merges the authors' ``t_cell`` / ``T_cell`` labels; selects
    ``n_top_genes`` HVGs (Scanpy ``'seurat'`` flavour, ``batch_key='system'``).
    The result is cached as ``<dir>/baron_pancreas_hs_mm.h5ad`` (path in
    ``adata.uns['h5ad_path']``) so re-running the tutorial, or re-opening it
    with ``ov.read(path, storage='disk')``, does not re-download or re-build.

    Parameters
    ----------
    dir
        Local cache directory for the downloaded csv.gz / ortholog table and
        the merged h5ad.
    n_top_genes
        Number of highly-variable genes kept (default 2000, as the SysVI
        benchmark).

    Returns
    -------
    adata : AnnData
        ``obs['system']`` (human / mouse, the covariate SysVI corrects),
        ``obs['batch']`` (donor / strain, a weaker covariate),
        ``obs['cell_type']`` (authors' labels); raw counts in
        ``layers['counts']``, log-normalized expression in ``X``.

    Examples
    --------
    >>> import omicverse as ov
    >>> adata = ov.datasets.baron_pancreas_human_mouse()
    >>> m = ov.single.sysvi(adata, batch_key='system', categorical_covariate_keys=['batch'])
    """
    import scanpy as sc

    out_path = os.path.join(dir, "baron_pancreas_hs_mm.h5ad")
    if os.path.exists(out_path):
        adata = anndata.read_h5ad(out_path)
        adata.uns["h5ad_path"] = out_path
        return adata

    parts = {}
    for species in ("human", "mouse"):
        samples = []
        for fname, gsm in _BARON_SAMPLES.items():
            if not fname.startswith(species):
                continue
            url = _BARON_GEO_URL.format(gsm_nnn=gsm[:-3] + "nnn", gsm=gsm, fname=fname)
            path = download_data_requests(url, file_path=fname, dir=dir)
            samples.append(_baron_read_sample(path, batch=fname.split("_")[0]))
        a = anndata.concat(samples)
        a.obs_names_make_unique()
        a.layers["counts"] = a.X.copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        parts[species] = a

    ortholog_path = download_data_requests(
        _JAX_ORTHOLOG_URL, file_path="HOM_MouseHumanSequence.rpt", dir=dir)
    mm2hs = _baron_one2one_orthologs(ortholog_path)
    mm = parts["mouse"]
    mm = mm[:, [g in mm2hs for g in mm.var_names]].copy()
    mm.var_names = [mm2hs[g] for g in mm.var_names]
    mm = mm[:, ~mm.var_names.duplicated()].copy()
    hs = parts["human"]
    common = hs.var_names.intersection(mm.var_names)
    hs, mm = hs[:, common].copy(), mm[:, common].copy()
    hs.obs["system"], mm.obs["system"] = "human", "mouse"
    adata = anndata.concat([hs, mm], merge="same")
    adata.obs["cell_type"] = adata.obs["assigned_cluster"].astype(str).replace({"t_cell": "T_cell"})
    for k in ("system", "batch", "cell_type"):
        adata.obs[k] = adata.obs[k].astype("category")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat", batch_key="system")
    adata = adata[:, adata.var["highly_variable"].values].copy()
    adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
    adata.X.sort_indices()  # anndata-rs (ov.read(..., storage='disk')) needs sorted CSR indices
    adata.write_h5ad(out_path)
    adata.uns["h5ad_path"] = out_path
    return adata
