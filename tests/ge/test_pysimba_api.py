"""``ov.single.pySIMBA`` keeps its original public API (preprocess ->
gen_graph -> train -> batch_correction) but now runs end to end on
``ov.ge.SIMBA`` -- no ``import simba`` anywhere."""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
anndata = pytest.importorskip("anndata")
pytest.importorskip("sklearn")

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SAMPLE = os.path.join(_HERE, "sample", "rna_test.h5ad")


def test_module_does_not_import_simba_or_torchbiggraph():
    import omicverse.single._simba as mod

    src = open(mod.__file__).read()
    assert not re.search(r"^\s*(import simba\b|from simba\b)", src, re.MULTILINE)
    assert not re.search(r"^\s*(import torchbiggraph|from torchbiggraph)", src, re.MULTILINE)
    assert not re.search(r"^\s*(import simba_pbg|from simba_pbg)", src, re.MULTILINE)


@pytest.fixture
def two_batch_adata():
    if not os.path.exists(_SAMPLE):
        pytest.skip(f"bundled sample not found: {_SAMPLE}")
    adata = anndata.read_h5ad(_SAMPLE)
    # Restrict to the file's own (previously computed) highly-variable genes
    # and its best-covered cells: SIMBA's `infer_edges` requires every cell
    # to have non-zero expression among the two batches' *shared* top
    # variable genes, which a uniform prefix slice of this small bundled
    # sample cannot guarantee (several cells end up all-zero on the ~250
    # shared genes). Sorting by coverage keeps the test fast and real-data
    # (no synthetic counts) while avoiding that degenerate corner.
    hv = adata.var["highly_variable"].to_numpy()
    adata = adata[:, hv].copy()
    totals = np.asarray(adata.X.sum(axis=1)).flatten()
    adata = adata[np.argsort(-totals)][:160, :300].copy()
    adata.obs["batch"] = np.array(["b1"] * 80 + ["b2"] * 80)
    return adata


def test_pysimba_full_workflow(two_batch_adata):
    import omicverse as ov

    sim = ov.single.pySIMBA(two_batch_adata, workdir="sample_simba_test_workdir")
    # span=1.0: this tiny 80-cell slice is numerically degenerate enough
    # that skmisc's loess (span=0.3, SIMBA's own default) raises a
    # near-singularity ValueError; a wider span is the standard loess
    # workaround and is exactly what `preprocess(span=...)` is for.
    sim.preprocess(batch_key="batch", min_n_cells=1, n_top_genes=250, n_bins=4, span=1.0)
    assert "simba_batch_dict" in sim.adata.uns
    assert set(sim.adata.uns["simba_batch_dict"]) == {"b1", "b2"}

    sim.gen_graph(n_components=5, k=5)
    assert sim.simba_model is not None
    assert sim.simba_model.module is not None
    assert "simba_batch_edge_dict" in sim.adata.uns

    sim.train(num_workers=1, auto_wd=False)
    assert sim.simba_model.is_trained_

    out = sim.batch_correction()
    assert "X_simba" in out.obsm
    assert out.obsm["X_simba"].shape[0] == out.n_obs
    assert np.isfinite(out.obsm["X_simba"]).all()


def test_pysimba_requires_gen_graph_before_train(two_batch_adata):
    import omicverse as ov

    sim = ov.single.pySIMBA(two_batch_adata)
    with pytest.raises(RuntimeError):
        sim.train()
