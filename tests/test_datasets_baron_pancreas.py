"""``ov.datasets.baron_pancreas_human_mouse`` — Baron et al. 2016 pancreas,
human vs. mouse (GEO GSE84133), harmonized via MGI/JAX one-to-one orthologs
for SysVI cross-system integration.

The assembly (per-species normalize+log1p, ortholog gene mapping, cell_type
merge, HVG selection, cache write/read) is checked offline on tiny synthetic
csv.gz / ortholog-table fixtures placed in the cache directory under the
loader's real file names, so nothing is downloaded. The real download runs
only with ``OV_TEST_NETWORK=1``::

    OV_TEST_NETWORK=1 pytest tests/test_datasets_baron_pancreas.py -m network
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import omicverse as ov
from omicverse.datasets._crossspecies import _BARON_SAMPLES


def _write_sample(path, genes, cell_ids, clusters, seed):
    rs = np.random.default_rng(seed)
    counts = rs.poisson(3.0, (len(cell_ids), len(genes))).astype(int)
    df = pd.DataFrame(counts, columns=genes, index=cell_ids)
    df.insert(0, "assigned_cluster", clusters)
    df.insert(0, "barcode", cell_ids)  # real files repeat a (non-unique) barcode here too
    df.to_csv(path, index=True)


@pytest.fixture()
def cache(tmp_path):
    hs_genes = ["INS", "GCG", "SST", "ACTB"]
    mm_genes = ["Ins1", "Gcg", "Sst", "Actb"]  # one-to-one orthologs of hs_genes, same order
    for i, fname in enumerate(f for f in _BARON_SAMPLES if f.startswith("human")):
        barcodes = [f"h{i}_{j}" for j in range(6)]
        clusters = (["alpha", "beta", "t_cell"] * 2)[:6]
        _write_sample(tmp_path / fname, hs_genes, barcodes, clusters, seed=i)
    for i, fname in enumerate(f for f in _BARON_SAMPLES if f.startswith("mouse")):
        barcodes = [f"m{i}_{j}" for j in range(5)]
        clusters = (["alpha", "T_cell"] * 3)[:5]
        _write_sample(tmp_path / fname, mm_genes, barcodes, clusters, seed=10 + i)
    ortholog = pd.DataFrame({
        "DB Class Key": [1, 1, 2, 2, 3, 3, 4, 4],
        "Common Organism Name": ["human", "mouse, laboratory"] * 4,
        "Symbol": ["INS", "Ins1", "GCG", "Gcg", "SST", "Sst", "ACTB", "Actb"],
    })
    ortholog.to_csv(tmp_path / "HOM_MouseHumanSequence.rpt", sep="\t", index=False)
    return tmp_path


def test_baron_pancreas_assembles_human_and_mouse(cache):
    adata = ov.datasets.baron_pancreas_human_mouse(dir=str(cache), n_top_genes=4)
    assert set(adata.obs["system"].cat.categories) == {"human", "mouse"}
    assert (adata.obs["system"] == "human").sum() == 4 * 6
    assert (adata.obs["system"] == "mouse").sum() == 2 * 5
    # mouse genes were renamed to their human ortholog symbol
    assert set(adata.var_names) <= {"INS", "GCG", "SST", "ACTB"}
    # the 't_cell' / 'T_cell' authors' label was merged
    assert "t_cell" not in adata.obs["cell_type"].cat.categories
    assert "T_cell" in adata.obs["cell_type"].cat.categories
    assert adata.obs_names.is_unique
    assert adata.layers["counts"].max() > 1  # raw counts kept pre-normalization
    assert adata.X.has_sorted_indices
    assert adata.uns["h5ad_path"] == str(cache / "baron_pancreas_hs_mm.h5ad")
    assert os.path.exists(adata.uns["h5ad_path"])


def test_baron_pancreas_second_call_reads_cache_without_sample_files(cache):
    ov.datasets.baron_pancreas_human_mouse(dir=str(cache), n_top_genes=4)
    for fname in _BARON_SAMPLES:
        os.remove(cache / fname)
    adata = ov.datasets.baron_pancreas_human_mouse(dir=str(cache), n_top_genes=4)
    assert adata.n_obs == 4 * 6 + 2 * 5


@pytest.mark.skipif(not os.environ.get("OV_TEST_NETWORK"),
                    reason="downloads ~25 MB from GEO/MGI; set OV_TEST_NETWORK=1")
@pytest.mark.network
def test_baron_pancreas_real_download(tmp_path):
    adata = ov.datasets.baron_pancreas_human_mouse(dir=str(tmp_path))
    assert adata.n_obs == 10455
    assert adata.n_vars == 2000
    assert set(adata.obs["system"].cat.categories) == {"human", "mouse"}
