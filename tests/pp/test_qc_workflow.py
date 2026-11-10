"""Tests for the QC compute/plot/filter workflow (issue #808).

ov.pp.qc_metrics (compute, no filter) -> ov.pl.qc (inspect distributions)
-> ov.pp.qc (filter).
"""
import numpy as np
import pytest


def _counts_adata(n=400, n_genes=60, seed=0):
    """Synthetic raw-count AnnData with MT-/RPS/HB gene names + a sample col."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(n, n_genes)).astype(np.float32)
    names = (
        [f"MT-{i}" for i in range(5)]
        + [f"RPS{i}" for i in range(5)]
        + [f"HB{i}" for i in range(3)]
        + [f"G{i}" for i in range(n_genes - 13)]
    )
    a = ad.AnnData(X)
    a.var_names = names
    a.obs["sample"] = rng.choice(["s1", "s2"], n)
    return a


def test_qc_metrics_computes_without_filtering():
    import omicverse as ov

    a = _counts_adata()
    n0 = a.n_obs
    ov.pp.qc_metrics(a)
    assert a.n_obs == n0  # nothing removed
    for col in ["nUMIs", "detected_genes", "mito_perc",
                "total_counts", "n_genes_by_counts",
                "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"]:
        assert col in a.obs, col
    # aliases consistent with scanpy metrics
    assert np.allclose(a.obs["nUMIs"], a.obs["total_counts"])
    assert np.allclose(a.obs["mito_perc"], a.obs["pct_counts_mt"] / 100.0)
    assert a.var["mt"].sum() == 5 and a.var["ribo"].sum() == 5


def test_qc_metrics_explicit_prefix():
    import omicverse as ov

    a = _counts_adata()
    ov.pp.qc_metrics(a, mt_startswith="MT-")
    assert a.var["mt"].sum() == 5


def test_pl_qc_runs():
    import matplotlib
    matplotlib.use("Agg")
    import omicverse as ov

    a = _counts_adata()
    ov.pp.qc_metrics(a)
    fig = ov.pl.qc(a, tresh={"mito_perc": 0.2, "nUMIs": 5, "detected_genes": 5})
    assert fig is not None and len(fig.axes) >= 3

    fig2 = ov.pl.qc(a, batch_key="sample", kind="violin")
    assert fig2 is not None


def test_pl_qc_requires_metrics():
    import anndata as ad
    import matplotlib
    matplotlib.use("Agg")
    import omicverse as ov

    a = ad.AnnData(np.zeros((5, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        ov.pl.qc(a)


def test_qc_cpu_gpu_mixed_in_memory_matches_cpu():
    """cpu-gpu-mixed qc on an in-memory AnnData (regression: NameError
    'backend' in 2.4.0) keeps the same cells and genes as cpu mode."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    import omicverse as ov
    rs = np.random.default_rng(0)
    X = sp.csr_matrix(rs.poisson(0.6, (1200, 500)).astype(np.float32))
    a = ad.AnnData(X)
    a.var_names = [f"MT-{i}" if i < 5 else f"G{i}" for i in range(500)]
    a.obs_names = [f"c{i}" for i in range(1200)]
    kw = dict(doublets=False, tresh={"mito_perc": 0.5, "nUMIs": 150, "detected_genes": 100},
              min_cells=3, min_genes=100)
    try:
        ov.settings.cpu_gpu_mixed_init()
        m = ov.pp.qc(a.copy(), **kw)
    finally:
        ov.settings.cpu_init()
    c = ov.pp.qc(a.copy(), **kw)
    assert list(m.obs_names) == list(c.obs_names)
    assert list(m.var_names) == list(c.var_names)
