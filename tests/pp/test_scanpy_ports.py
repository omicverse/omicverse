"""Parity tests for the native ports of scanpy functions in ``omicverse.pp``.

``calculate_qc_metrics``, ``score_genes``, ``score_genes_cell_cycle``,
``diffmap``, ``dpt``, ``draw_graph``, ``dendrogram``, ``subsample``, the
regress-out kernel behind ``ov.pp.regress`` and the Louvain backend behind
``ov.pp.louvain`` are compared against scanpy on a small synthetic count
matrix with cluster structure. scanpy is only needed for the comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

sc = pytest.importorskip("scanpy")


def _counts(n_obs=600, n_vars=300, n_groups=4, seed=0):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, n_groups, n_obs)
    base = rng.gamma(0.6, 1.0, n_vars)
    prog = rng.gamma(0.6, 2.0, (n_groups, n_vars)) * (rng.random((n_groups, n_vars)) < 0.15)
    mu = (base + prog[groups]) * rng.lognormal(0, 0.3, n_obs)[:, None]
    X = rng.poisson(mu).astype(np.float32)
    ad = AnnData(sp.csr_matrix(X))
    ad.obs_names = [f"c{i}" for i in range(n_obs)]
    names = [f"G{i}" for i in range(n_vars)]
    names[:5] = ["MT-A", "MT-B", "MT-C", "RPS1", "RPL2"]
    ad.var_names = names
    ad.obs["group"] = pd.Categorical([f"g{g}" for g in groups])
    ad.obs["cov1"] = rng.normal(size=n_obs)
    ad.obs["cov2"] = rng.normal(size=n_obs)
    ad.var["mt"] = ad.var_names.str.startswith("MT-")
    ad.var["ribo"] = ad.var_names.str.startswith(("RPS", "RPL"))
    return ad


@pytest.fixture(scope="module")
def raw_counts():
    return _counts()


@pytest.fixture(scope="module")
def processed(raw_counts):
    a = raw_counts.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.raw = a
    sc.pp.scale(a, max_value=10)
    sc.pp.pca(a, n_comps=20, random_state=0)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=20, random_state=0)
    return a


def _close(a, b, atol=1e-6):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), atol=atol, rtol=1e-6)


@pytest.mark.parametrize("dense", [False, True])
def test_calculate_qc_metrics_matches_scanpy(raw_counts, dense):
    import omicverse as ov

    a = raw_counts.copy()
    if dense:
        a.X = a.X.toarray()
    o1, v1 = sc.pp.calculate_qc_metrics(a, qc_vars=["mt", "ribo"], percent_top=(5, 10, 50))
    o2, v2 = ov.pp.calculate_qc_metrics(a, qc_vars=["mt", "ribo"], percent_top=(5, 10, 50))
    assert list(o1.columns) == list(o2.columns)
    assert list(v1.columns) == list(v2.columns)
    for c in o1:
        assert o1[c].dtype == o2[c].dtype, c
        _close(o1[c], o2[c])
    for c in v1:
        _close(v1[c], v2[c])

    b1, b2 = a.copy(), a.copy()
    sc.pp.calculate_qc_metrics(b1, qc_vars="mt", inplace=True, log1p=False, percent_top=(5, 50))
    ov.pp.calculate_qc_metrics(b2, qc_vars="mt", inplace=True, log1p=False, percent_top=(5, 50))
    assert list(b1.obs.columns) == list(b2.obs.columns)
    assert list(b1.var.columns) == list(b2.var.columns)


def test_score_genes_matches_scanpy(processed):
    import omicverse as ov

    genes = list(processed.var_names[10:30])
    a1, a2 = processed.copy(), processed.copy()
    sc.tl.score_genes(a1, genes, score_name="s", ctrl_size=20)
    ov.pp.score_genes(a2, genes, score_name="s", ctrl_size=20)
    _close(a1.obs["s"], a2.obs["s"], atol=0)
    a1, a2 = processed.copy(), processed.copy()
    sc.tl.score_genes(a1, genes, score_name="s", use_raw=False, random_state=5, n_bins=10)
    ov.pp.score_genes(a2, genes, score_name="s", use_raw=False, random_state=5, n_bins=10)
    _close(a1.obs["s"], a2.obs["s"], atol=0)


def test_score_genes_cell_cycle_matches_scanpy(processed):
    from omicverse.pp._score_genes import score_genes_cell_cycle

    s_genes = list(processed.var_names[40:60])
    g2m = list(processed.var_names[60:85])
    a1, a2 = processed.copy(), processed.copy()
    sc.tl.score_genes_cell_cycle(a1, s_genes=s_genes, g2m_genes=g2m)
    score_genes_cell_cycle(a2, s_genes=s_genes, g2m_genes=g2m)
    _close(a1.obs["S_score"], a2.obs["S_score"], atol=0)
    _close(a1.obs["G2M_score"], a2.obs["G2M_score"], atol=0)
    assert (a1.obs["phase"].values == a2.obs["phase"].values).all()


def test_regress_out_matches_scanpy(processed):
    from omicverse.pp._regress import regress_out

    a1 = sc.pp.regress_out(processed, ["cov1", "cov2"], copy=True)
    a2 = regress_out(processed, ["cov1", "cov2"], copy=True)
    _close(a1.X, a2.X, atol=1e-5)
    a1 = sc.pp.regress_out(processed, "group", copy=True)
    a2 = regress_out(processed, "group", copy=True)
    _close(a1.X, a2.X, atol=1e-5)
    # ov.pp.regress writes layers['regressed'] (and honours `layer`)
    import omicverse as ov
    a3 = processed.copy()
    ov.pp.regress(a3, keys=["cov1", "cov2"])
    a1 = sc.pp.regress_out(processed, ["cov1", "cov2"], copy=True)
    _close(a1.X, a3.layers["regressed"], atol=1e-5)


def test_louvain_matches_scanpy(processed):
    pytest.importorskip("louvain")
    from omicverse.pp._louvain import louvain

    a1, a2 = processed.copy(), processed.copy()
    sc.tl.louvain(a1, resolution=1.0, random_state=0)
    louvain(a2, resolution=1.0, random_state=0)
    assert (a1.obs["louvain"].values == a2.obs["louvain"].values).all()
    assert list(a1.obs["louvain"].cat.categories) == list(a2.obs["louvain"].cat.categories)
    assert a1.uns["louvain"] == a2.uns["louvain"]


def test_louvain_without_louvain_package(processed, monkeypatch):
    """Fallback backends recover the planted groups (no `louvain` package)."""
    import builtins

    from sklearn.metrics import adjusted_rand_score

    from omicverse.pp import _louvain as L

    real_import = builtins.__import__
    blocked = {"louvain"}

    def fake_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    a = processed.copy()
    L.louvain(a, resolution=0.3, random_state=0)
    assert L.LAST_BACKEND == "igraph.community_multilevel"
    assert a.obs["louvain"].dtype.name == "category"
    assert list(a.uns["louvain"]) == ["params"]
    assert adjusted_rand_score(a.obs["group"], a.obs["louvain"]) > 0.9
    b = processed.copy()
    L.louvain(b, resolution=0.3, random_state=0)
    assert (a.obs["louvain"].values == b.obs["louvain"].values).all()  # seeded


def test_louvain_igraph_flavor_matches_scanpy(processed):
    import random

    from omicverse.pp._louvain import louvain

    a1, a2 = processed.copy(), processed.copy()
    random.seed(0)
    sc.tl.louvain(a1, flavor="igraph")
    random.seed(0)
    louvain(a2, flavor="igraph")
    assert (a1.obs["louvain"].values == a2.obs["louvain"].values).all()


def test_diffmap_dpt_match_scanpy(processed):
    import omicverse as ov

    a1, a2 = processed.copy(), processed.copy()
    sc.tl.diffmap(a1, n_comps=10)
    ov.pp.diffmap(a2, n_comps=10)
    _close(a1.uns["diffmap_evals"], a2.uns["diffmap_evals"])
    _close(a1.obsm["X_diffmap"], a2.obsm["X_diffmap"])
    a1.uns["iroot"] = a2.uns["iroot"] = 0
    sc.tl.dpt(a1, n_dcs=10, n_branchings=1)
    ov.pp.dpt(a2, n_dcs=10, n_branchings=1)
    assert set(a1.obs.columns) == set(a2.obs.columns)
    assert set(a1.uns.keys()) == set(a2.uns.keys())
    _close(a1.obs["dpt_pseudotime"], a2.obs["dpt_pseudotime"])
    assert (a1.obs["dpt_groups"].values == a2.obs["dpt_groups"].values).all()
    assert (a1.obs["dpt_order"].values == a2.obs["dpt_order"].values).all()


@pytest.mark.parametrize("layout", ["fr", "drl"])
def test_draw_graph_matches_scanpy(processed, layout):
    import omicverse as ov

    a1, a2 = processed.copy(), processed.copy()
    kw = {"niter": 100} if layout == "fr" else {}
    sc.tl.draw_graph(a1, layout=layout, **kw)
    ov.pp.draw_graph(a2, layout=layout, **kw)
    key = f"X_draw_graph_{layout}"
    _close(a1.obsm[key], a2.obsm[key])
    assert a1.uns["draw_graph"] == a2.uns["draw_graph"]


def test_draw_graph_fa_matches_scanpy(processed):
    pytest.importorskip("fa2_modified")
    import omicverse as ov

    a1, a2 = processed.copy(), processed.copy()
    sc.tl.draw_graph(a1, iterations=50)
    ov.pp.draw_graph(a2, iterations=50)
    _close(a1.obsm["X_draw_graph_fa"], a2.obsm["X_draw_graph_fa"])


@pytest.mark.parametrize("kwargs", [{}, {"var_names": ["G10", "G11"]}, {"linkage_method": "average"}])
def test_dendrogram_matches_scanpy(processed, kwargs):
    import omicverse as ov

    a1, a2 = processed.copy(), processed.copy()
    sc.tl.dendrogram(a1, "group", **kwargs)
    ov.pp.dendrogram(a2, "group", **kwargs)
    d1, d2 = a1.uns["dendrogram_group"], a2.uns["dendrogram_group"]
    assert set(d1) == set(d2)
    _close(d1["linkage"], d2["linkage"])
    _close(d1["correlation_matrix"], d2["correlation_matrix"])
    assert d1["categories_ordered"] == d2["categories_ordered"]
    assert d1["categories_idx_ordered"] == d2["categories_idx_ordered"]


def test_subsample_matches_scanpy(raw_counts):
    import omicverse as ov

    for kw in ({"fraction": 0.25}, {"n_obs": 50, "random_state": 3}):
        s1 = sc.pp.subsample(raw_counts, copy=True, **kw)
        s2 = ov.pp.subsample(raw_counts, copy=True, **kw)
        assert list(s1.obs_names) == list(s2.obs_names)
    a = raw_counts.copy()
    ov.pp.subsample(a, n_obs=40)
    assert a.n_obs == 40
    X = raw_counts.X.toarray()
    r1 = sc.pp.subsample(X, n_obs=7)
    r2 = ov.pp.subsample(X, n_obs=7)
    np.testing.assert_array_equal(r1[1], r2[1])
    np.testing.assert_array_equal(r1[0], r2[0])
