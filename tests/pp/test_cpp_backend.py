"""backend='cpp' (anndataoom-fast kernels) must reproduce backend='python'.

Each step is run on identical inputs with both backends; QC, normalization,
HVG selection and markers must agree exactly, floating-point layers to
rounding, PCA up to sign, Leiden (same graph) label for label.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("anndataoom_fast")


def _counts(n=1500, n_genes=600, seed=0):
    import anndata as ad

    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 4, n)
    base = rng.gamma(0.4, 1.0, size=(4, n_genes))
    depth = rng.lognormal(7.0, 0.4, n)
    mu = base[groups] / base[groups].sum(1, keepdims=True) * depth[:, None]
    X = sp.csr_matrix(rng.poisson(mu).astype(np.float32))
    a = ad.AnnData(X)
    a.var_names = ([f"MT-{i}" for i in range(8)] + [f"RPS{i}" for i in range(8)]
                   + [f"G{i}" for i in range(n_genes - 16)])
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["group"] = pd.Categorical([f"g{g}" for g in groups])
    return a


def _preprocessed(backend):
    import omicverse as ov

    a = _counts()
    a = ov.pp.qc(a, mode="seurat", doublets=False, tresh={"mito_perc": 0.5, "nUMIs": 100,
                                                          "detected_genes": 50},
                 backend=backend) or a
    ov.pp.preprocess(a, mode="shiftlog|pearson", n_HVGs=300, backend=backend)
    ov.pp.scale(a, max_value=10, backend=backend)
    ov.pp.pca(a, n_pcs=20, layer="scaled", backend=backend)
    return a


@pytest.fixture(scope="module")
def pair():
    return _preprocessed("python"), _preprocessed("cpp")


def test_qc_and_hvg_identical(pair):
    a, b = pair
    assert list(a.obs_names) == list(b.obs_names)
    for k in ("nUMIs", "detected_genes", "mito_perc"):
        np.testing.assert_array_equal(a.obs[k].values, b.obs[k].values)
    np.testing.assert_array_equal(a.var["highly_variable_features"].values,
                                  b.var["highly_variable_features"].values)
    np.testing.assert_allclose(a.var["residual_variances"].values,
                               b.var["residual_variances"].values, rtol=1e-10)


def test_scale_and_pca_match(pair):
    a, b = pair
    sa, sb = np.asarray(a.layers["scaled"]), np.asarray(b.layers["scaled"])
    assert sa.shape == sb.shape and type(sa) is type(sb)
    np.testing.assert_allclose(sa, sb, rtol=1e-6, atol=1e-6)
    pa, pb = a.obsm["X_pca"], b.obsm["X_pca"]
    cos = np.abs((pa * pb).sum(0)) / (np.linalg.norm(pa, axis=0) * np.linalg.norm(pb, axis=0))
    assert cos.min() > 0.9999


def test_leiden_identical_on_same_graph(pair):
    # The exact C++ port reproduces leidenalg 0.12 (older releases differ in
    # a few labels).
    leidenalg = pytest.importorskip("leidenalg")
    if tuple(int(x) for x in leidenalg.version.split(".")[:2]) < (0, 12):
        pytest.skip(f"exact port matches leidenalg >= 0.12, found {leidenalg.version}")
    import omicverse as ov

    a = pair[0].copy()
    ov.pp.neighbors(a, n_neighbors=15, use_rep="X_pca", backend="python")
    for r in (0.5, 1.0):
        for it in (-1, 2):
            ov.pp.leiden(a, resolution=r, key_added="py", backend="python", n_iterations=it)
            ov.pp.leiden(a, resolution=r, key_added="cpp", backend="cpp",
                         flavor="leidenalg_exact", n_iterations=it)
            np.testing.assert_array_equal(a.obs["py"].values, a.obs["cpp"].values)
    ov.pp.leiden(a, resolution=[0.5, 1.0], key_added="multi", backend="cpp",
                 flavor="leidenalg_exact", n_iterations=-1)
    ov.pp.leiden(a, resolution=1.0, key_added="single", backend="python")
    np.testing.assert_array_equal(a.obs["multi_1.0"].values, a.obs["single"].values)


def test_leiden_brisc_default_matches_leidenalg_quality(pair):
    import omicverse as ov
    from sklearn.metrics import adjusted_rand_score

    a = pair[0].copy()
    ov.pp.neighbors(a, n_neighbors=15, use_rep="X_pca", backend="python")
    ov.pp.leiden(a, resolution=1.0, key_added="brisc", backend="cpp")  # default flavor
    ov.pp.leiden(a, resolution=1.0, key_added="ref", backend="python")
    # Same objective, different optimiser: compare the modularity reached,
    # not the partition itself.
    A = a.obsp["connectivities"].tocsr()
    k = np.asarray(A.sum(1)).ravel()
    m2 = k.sum()

    def modularity(lab):
        lab = pd.factorize(np.asarray(lab))[0]
        r = A.tocoo()
        inside = r.data[lab[r.row] == lab[r.col]].sum()
        return (inside - (np.bincount(lab, weights=k) ** 2).sum() / m2) / m2

    assert modularity(a.obs["brisc"]) > 0.98 * modularity(a.obs["ref"])
    assert adjusted_rand_score(a.obs["group"], a.obs["brisc"]) > 0.5


def test_cpp_graph_umap_and_markers_run(pair):
    import omicverse as ov

    b = pair[1].copy()
    ov.pp.neighbors(b, n_neighbors=15, use_rep="X_pca", backend="cpp")
    ov.pp.umap(b, backend="cpp")
    assert b.obsm["X_umap"].shape == (b.n_obs, 2) and np.isfinite(b.obsm["X_umap"]).all()
    c = b.copy()
    ov.single.find_markers(b, groupby="group", method="cosg", n_genes=20, backend="cpp")
    ov.single.find_markers(c, groupby="group", method="cosg", n_genes=20, backend="python")
    nb, nc = b.uns["rank_genes_groups"]["names"], c.uns["rank_genes_groups"]["names"]
    cols = list(nb.columns) if hasattr(nb, "columns") else list(nb.dtype.names)
    assert cols
    for g in cols:
        assert list(np.asarray(nb[g]).astype(str)) == list(np.asarray(nc[g]).astype(str))
