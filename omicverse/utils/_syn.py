"""
Copy from scVI (https://github.com/scverse/scvi-tools/blob/main/scvi/data/_built_in_data/_synthetic.py)

"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
#from scanpy import read
import scanpy as sc
from .._anndata_compat import as_dtype as _as_dtype


logger = logging.getLogger(__name__)

def synthetic_iid(
    batch_size: int = 200,
    n_genes: int = 10000,
    n_proteins: int = 100,
    n_regions: int = 100,
    n_batches: int = 2,
    n_labels: int = 3,
    dropout_ratio: float = 0.7,
    sparse_format: Optional[str] = None,
) -> AnnData:
    if n_batches < 1:
        raise ValueError("`n_batches` must be greater than 0")
    if n_genes < 1:
        raise ValueError("`n_genes` must be greater than 0")

    return _generate_synthetic(
        batch_size=batch_size,
        n_genes=n_genes,
        n_proteins=n_proteins,
        n_regions=n_regions,
        n_batches=n_batches,
        n_labels=n_labels,
        dropout_ratio=dropout_ratio,
        sparse_format=sparse_format,
    )

def _generate_synthetic(
    *,
    batch_size: int,
    n_genes: int,
    n_proteins: int,
    n_regions: int,
    n_batches: int,
    n_labels: int,
    dropout_ratio: float,
    sparse_format: Optional[str],
    batch_key: str = "batch",
    labels_key: str = "labels",
    protein_expression_key: str = "protein_expression",
    protein_names_key: str = "protein_names",
    accessibility_key: str = "accessibility",
) -> AnnData:
    n_obs = batch_size * n_batches

    def sparsify_data(data: np.ndarray):
        if sparse_format is not None:
            data = getattr(scipy.sparse, sparse_format)(data)
        return data

    rna = np.random.negative_binomial(5, 0.3, size=(n_obs, n_genes))
    mask = np.random.binomial(n=1, p=dropout_ratio, size=(n_obs, n_genes))
    rna = rna * mask
    rna = sparsify_data(rna)

    if n_proteins > 0:
        protein = np.random.negative_binomial(5, 0.3, size=(n_obs, n_proteins))
        protein_names = np.arange(n_proteins).astype(str)
        protein = sparsify_data(protein)

    if n_regions > 0:
        accessibility = np.random.negative_binomial(5, 0.3, size=(n_obs, n_regions))
        mask = np.random.binomial(n=1, p=dropout_ratio, size=(n_obs, n_regions))
        accessibility = accessibility * mask
        accessibility = sparsify_data(accessibility)

    batch = []
    for i in range(n_batches):
        batch += [f"batch_{i}"] * batch_size

    if n_labels > 0:
        labels = np.random.randint(0, n_labels, size=(n_obs,))
        labels = np.array([f"label_{i}" for i in labels])

    adata = AnnData(_as_dtype(rna, np.float32))
    if n_proteins > 0:
        adata.obsm[protein_expression_key] = protein
        adata.uns[protein_names_key] = protein_names
    if n_regions > 0:
        adata.obsm[accessibility_key] = accessibility

    adata.obs[batch_key] = pd.Categorical(batch)
    if n_labels > 0:
        adata.obs[labels_key] = pd.Categorical(labels)

    return adata

url_datadir = "https://github.com/theislab/scvelo_notebooks/raw/master/"

def pancreas(file_path= "data/Pancreas/endocrinogenesis_day15.h5ad"):
    """Pancreatic endocrinogenesis.

    Data from `Bastidas-Ponce et al. (2019) <https://doi.org/10.1242/dev.173849>`__.

    Pancreatic epithelial and Ngn3-Venus fusion (NVF) cells during secondary transition
    with transcriptome profiles sampled from embryonic day 15.5.

    Endocrine cells are derived from endocrine progenitors located in the pancreatic
    epithelium. Endocrine commitment terminates in four major fates: glucagon- producing
    α-cells, insulin-producing β-cells, somatostatin-producing δ-cells and
    ghrelin-producing ε-cells.

    .. image:: https://user-images.githubusercontent.com/31883718/67709134-a0989480-f9bd-11e9-8ae6-f6391f5d95a0.png
       :width: 600px

    Arguments
    ---------
    file_path
        Path where to save dataset and read it from.

    Returns
    -------
    Returns `adata` object
    """
    url = f"{url_datadir}data/Pancreas/endocrinogenesis_day15.h5ad"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    adata.var_names_make_unique()
    return adata