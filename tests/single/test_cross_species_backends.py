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


def test_backends_exposed_in_single():
    import omicverse as ov
    assert callable(ov.single.samap_integrate)
    assert callable(ov.single.saturn_integrate)
    assert callable(ov.single.cross_species_integrate)
