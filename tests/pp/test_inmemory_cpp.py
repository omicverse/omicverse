"""storage='memory' with backend='cpp': threaded load / cuts / implicit scale
give the same AnnData as the plain anndata + scanpy-style path."""
import numpy as np
import pytest
import scipy.sparse as sp

aof_memory = pytest.importorskip("anndataoom_fast.memory")
if not hasattr(aof_memory, "subset_csr"):
    pytest.skip("anndataoom-fast without subset_csr", allow_module_level=True)


def _counts(n=1500, g=400, seed=0):
    import anndata as ad
    rng = np.random.default_rng(seed)
    lam = rng.gamma(0.4, 2.0, size=g) * rng.lognormal(0, 0.6, size=(n, 1))
    lam[:, 0] *= 40  # one highly expressed gene
    X = sp.csr_matrix(rng.poisson(lam).astype(np.float32))
    X[:, 5] = 0  # a gene detected nowhere
    X.eliminate_zeros()
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    names = [f"G{j}" for j in range(g)]
    names[10:15] = [f"MT-{j}" for j in range(5)]
    a.var_names = names
    a.layers["raw"] = X.copy()
    return a


@pytest.fixture
def h5ad(tmp_path):
    p = tmp_path / "c.h5ad"
    a = _counts()
    del a.layers["raw"]
    a.write_h5ad(p)
    return p


def test_threaded_read_equals_anndata(h5ad):
    import anndata as ad
    import omicverse as ov
    from omicverse.io.single._read import _read_h5ad_threaded
    ov.settings.backend = "cpp"
    a, b = _read_h5ad_threaded(h5ad), ad.read_h5ad(h5ad)
    assert a is not None
    assert (a.X != b.X).nnz == 0 and a.X.dtype == b.X.dtype
    assert a.obs.equals(b.obs) and a.var.equals(b.var)


def test_subset_helpers_match_anndata():
    from omicverse.pp._qc import subset_copy, subset_inplace
    a = _counts()
    rng = np.random.default_rng(1)
    rows, cols = rng.random(a.n_obs) < 0.7, rng.random(a.n_vars) < 0.6
    ref = a[rows, cols].copy()
    got = subset_copy(a, obs=rows, var=cols, backend="cpp")
    assert a.shape == (1500, 400)  # input untouched
    assert list(got.obs_names) == list(ref.obs_names) and list(got.var_names) == list(ref.var_names)
    for k in (None, "raw"):
        x, y = (got.X, ref.X) if k is None else (got.layers[k], ref.layers[k])
        np.testing.assert_array_equal(x.toarray(), y.toarray())
    b = a.copy()
    subset_inplace(b, obs=rows, var=cols, backend="cpp")
    assert list(b.obs_names) == list(ref.obs_names)
    np.testing.assert_array_equal(b.layers["raw"].toarray(), ref.layers["raw"].toarray())


def test_cpp_qc_same_cells_and_genes_and_input_untouched():
    import omicverse as ov
    a = _counts()
    kw = dict(doublets=False, tresh={"mito_perc": 0.5, "nUMIs": 100, "detected_genes": 30},
              min_cells=20, min_genes=30)
    ov.settings.cpu_init()
    c = ov.pp.qc(a.copy(), backend="python", **kw)
    before = a.shape
    f = ov.pp.qc(a, backend="cpp", **kw)
    assert a.shape == before  # the user's object is not filtered in place
    assert list(f.obs_names) == list(c.obs_names)
    assert list(f.var_names) == list(c.var_names)
    np.testing.assert_array_equal(f.X.toarray(), c.X.toarray())


def test_implicit_scale_pca_equals_dense(tmp_path):
    import omicverse as ov
    ov.settings.cpu_init()

    def run(lazy):
        a = ov.pp.qc(_counts(), backend="cpp", doublets=False,
                     tresh={"mito_perc": 0.5, "nUMIs": 100, "detected_genes": 30})
        ov.pp.preprocess(a, mode="shiftlog|pearson", n_HVGs=100, backend="cpp")
        ov.pp.scale(a, max_value=10, use_implicit_centering=lazy, backend="cpp")
        ov.pp.pca(a, n_pcs=10, layer="scaled", backend="cpp")
        return a

    d, l = run(False), run(True)
    assert "scaled" in d.layers and "scaled" not in l.layers
    np.testing.assert_array_equal(d.obsm["X_pca"], l.obsm["X_pca"])
    np.testing.assert_array_equal(d.varm["PCs"], l.varm["PCs"])
    l.write_h5ad(tmp_path / "lazy.h5ad")  # only plain arrays are stored
