"""SAMap and SATURN are vendored under ``omicverse.external`` (their upstreams
pin conflicting deps — SAMap needs numpy==1.23.5/scanpy==1.9.3, SATURN needs
faiss/fair-esm/pytorch-metric-learning). These tests check that the vendored
packages import and are wired into ``ov.single`` — the heavy end-to-end runs
(BLAST proteomes / ESM embeddings) are validated separately, not in CI.
"""
import importlib

import pytest


def test_samap_vendored_imports():
    m = importlib.import_module("omicverse.external.samap")
    assert hasattr(m, "q") and hasattr(m, "ut")           # upstream namespace
    run = importlib.import_module("omicverse.external.samap._run")
    assert callable(run.samap_integrate)
    assert callable(run.build_blast_maps)


def test_samalg_runs_on_current_numpy():
    """The hardest compat risk: samalg (SAM) must actually *run* under the
    installed numpy/scanpy, not just import."""
    np = pytest.importorskip("numpy")
    sp = pytest.importorskip("scipy.sparse")
    ad = pytest.importorskip("anndata")
    pytest.importorskip("numba")
    from omicverse.external.samap.samalg import SAM

    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, size=(120, 200)).astype("float32"))
    a = ad.AnnData(X)
    a.var_names = [f"g{i}" for i in range(200)]
    s = SAM(counts=a)
    s.preprocess_data(sum_norm="cell_median", norm="log",
                      thresh_low=0.0, thresh_high=0.96, min_expression=1)
    s.run(preprocessing="StandardScaler", npcs=20, weight_PCs=False,
          k=10, n_genes=150, weight_mode="rms")
    assert "X_pca" in s.adata.obsm or s.adata.obsm  # SAM produced an embedding


def test_saturn_vendored_imports():
    importlib.import_module("omicverse.external.saturn")
    run = importlib.import_module("omicverse.external.saturn._run")
    assert callable(run.run_saturn)


def test_backends_exposed_via_method():
    """SAMap/SATURN are reached through the unified CrossSpecies method= API,
    not standalone functions."""
    import omicverse as ov
    assert callable(ov.single.cross_species_integrate)
    assert isinstance(ov.single.CrossSpecies, type)
    # no standalone per-method functions
    assert not hasattr(ov.single, "samap_integrate")
    assert not hasattr(ov.single, "saturn_integrate")
    # method routing is wired (constructing does not run anything heavy)
    from omicverse.single._cross_species import _HOMOLOGY_METHODS
    assert {"samap", "saturn"} <= _HOMOLOGY_METHODS


def test_crossspecies_save_load(tmp_path):
    """CrossSpecies persists via ov.io.save / ov.io.load."""
    import numpy as np
    import anndata as ad
    import omicverse as ov

    cs = ov.single.CrossSpecies(
        [ad.AnnData(np.ones((5, 3))), ad.AnnData(np.ones((4, 3)))],
        ["human", "mouse"], method="harmony")
    cs.adata = ad.AnnData(np.zeros((3, 2)))
    p = str(tmp_path / "cs.pkl")
    cs.save(p)
    cs2 = ov.single.CrossSpecies.load(p)
    assert cs2.method == "harmony" and cs2.species == ["human", "mouse"]
    assert cs2.adata.shape == (3, 2)
