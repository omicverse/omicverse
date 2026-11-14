"""``ov.datasets.cite_seq`` — 10x PBMC 5k + 10k CITE-seq loader.

The assembly (gene intersection, protein join, batch column, counts layer) is
checked offline on two small files placed in the cache directory under the
loader's file names, so nothing is downloaded.  The real download runs only
with ``OV_TEST_NETWORK=1``::

    OV_TEST_NETWORK=1 pytest tests/test_datasets_cite_seq.py -m network
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from anndata import AnnData

import omicverse as ov
from omicverse.datasets._datasets import CITE_SEQ_PBMC_FILES


def _run(n_cells, genes, proteins, seed):
    rs = np.random.default_rng(seed)
    a = AnnData(X=rs.poisson(1.0, (n_cells, len(genes))).astype(np.float32))
    a.var_names = genes
    a.obs_names = [f"AAAC{i:04d}-1" for i in range(n_cells)]  # barcodes overlap across runs
    a.obsm["protein_expression"] = rs.poisson(20, (n_cells, len(proteins))).astype(np.int64)
    a.uns["protein_names"] = np.asarray(proteins)
    return a


@pytest.fixture()
def cache(tmp_path):
    (b10, f10, _), (b5, f5, _) = CITE_SEQ_PBMC_FILES
    _run(12, ["G1", "G2", "G3", "G4"], ["CD3", "CD4", "CD8a"], 0).write_h5ad(tmp_path / f10)
    _run(8, ["G2", "G3", "G4", "G5"], ["CD3", "CD4", "CD19", "CD8a"], 1).write_h5ad(tmp_path / f5)
    return tmp_path


def test_cite_seq_assembles_both_runs(cache):
    adata = ov.datasets.cite_seq(dir=str(cache))
    assert adata.shape == (20, 3)
    assert list(adata.var_names) == ["G2", "G3", "G4"]
    assert list(adata.obs["batch"].cat.categories) == ["PBMC10k", "PBMC5k"]
    assert (adata.obs["batch"] == "PBMC10k").sum() == 12
    assert adata.obs_names.is_unique
    pro = adata.obsm["protein_expression"]
    assert list(pro.columns) == ["CD3", "CD4", "CD8a"]  # inner join
    assert list(adata.uns["protein_names"]) == list(pro.columns)
    assert pro.shape == (20, 3) and not pro.isna().any().any()
    counts = adata.layers["counts"]
    assert np.array_equal(counts.toarray(), adata.X.toarray())
    assert np.allclose(counts.data, np.round(counts.data))


def test_cite_seq_outer_join_fills_missing_proteins_with_zero(cache):
    adata = ov.datasets.cite_seq(protein_join="outer", dir=str(cache))
    pro = adata.obsm["protein_expression"]
    assert set(pro.columns) == {"CD3", "CD4", "CD8a", "CD19"}
    assert (pro.loc[adata.obs["batch"] == "PBMC10k", "CD19"] == 0).all()


def test_cite_seq_rejects_unknown_join(cache):
    with pytest.raises(ValueError, match="protein_join"):
        ov.datasets.cite_seq(protein_join="left", dir=str(cache))


@pytest.mark.skipif(not os.environ.get("OV_TEST_NETWORK"),
                    reason="downloads ~43 MB from exampledata.scverse.org; set OV_TEST_NETWORK=1")
@pytest.mark.network
def test_cite_seq_real_download(tmp_path):
    adata = ov.datasets.cite_seq(dir=str(tmp_path))
    assert adata.shape == (10849, 15792)
    assert adata.obsm["protein_expression"].shape == (10849, 14)
    assert adata.obs["batch"].value_counts().to_dict() == {"PBMC10k": 6855, "PBMC5k": 3994}
