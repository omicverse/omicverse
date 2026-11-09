"""omicverse's rapids-singlecell wrappers give Scanpy's results with any rsc release."""
import numpy as np
import pytest
import scipy.sparse as sp

rsc = pytest.importorskip("rapids_singlecell")
cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover
    pytest.skip("no CUDA device", allow_module_level=True)


def _adata(seed=0, n=600, g=300):
    import anndata as ad
    rs = np.random.RandomState(seed)
    X = sp.random(n, g, density=0.08, random_state=rs, format="csr", dtype=np.float32)
    X.data = np.ceil(X.data * 20).astype(np.float32)
    X[:5, 0] = 5000.0          # a highly expressed gene in a few cells
    X[10:20] = 0               # empty cells
    X.eliminate_zeros()
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{j}" for j in range(g)]
    return a


def _to_gpu(a):
    a = a.copy()
    rsc.get.anndata_to_GPU(a)
    return a


@pytest.mark.parametrize("kw", [dict(min_genes=15), dict(max_genes=30), dict(min_counts=50),
                                dict(min_genes=10, max_counts=400)])
def test_filter_cells_matches_scanpy(kw):
    import scanpy as sc
    from omicverse.pp import _rsc_compat as rc
    a = _adata()
    ref = a.copy()
    for k, v in kw.items():
        sc.pp.filter_cells(ref, **{k: v})
    g = _to_gpu(a)
    rc.filter_cells(g, **kw)
    assert list(g.obs_names) == list(ref.obs_names)


@pytest.mark.parametrize("kw", [dict(min_cells=40), dict(max_cells=60), dict(min_counts=300)])
def test_filter_genes_matches_scanpy(kw):
    import scanpy as sc
    from omicverse.pp import _rsc_compat as rc
    a = _adata(1)
    ref = a.copy()
    sc.pp.filter_genes(ref, **kw)
    g = _to_gpu(a)
    rc.filter_genes(g, **kw)
    assert list(g.var_names) == list(ref.var_names)


@pytest.mark.parametrize("exclude", [False, True])
@pytest.mark.parametrize("target_sum", [None, 1e4])
def test_normalize_total_matches_scanpy(exclude, target_sum):
    import scanpy as sc
    from omicverse.pp import _rsc_compat as rc
    a = _adata(2)
    # Reference: Scanpy on a dense matrix, whose default target is the median
    # of the non-zero per-cell counts, as rapids-singlecell >= 0.14 (Scanpy's
    # sparse path takes the median over all cells, zeros included).
    ref = a.copy()
    ref.X = ref.X.toarray()
    sc.pp.normalize_total(ref, target_sum=target_sum, exclude_highly_expressed=exclude, max_fraction=0.2)
    g = _to_gpu(a)
    rc.normalize_total(g, target_sum=target_sum, exclude_highly_expressed=exclude, max_fraction=0.2)
    got = g.X.get().toarray() if hasattr(g.X, "get") else np.asarray(g.X)
    np.testing.assert_allclose(got, ref.X, rtol=2e-6, atol=1e-5)


def test_pca_kwargs_translation():
    from omicverse.pp import _rsc_compat as rc
    out = rc.pca_kwargs({"use_highly_variable": True, "svd_solver": "auto"})
    if "use_highly_variable" in rc._params("pp.pca"):
        assert out["use_highly_variable"] is True
    else:
        assert out == {"mask_var": "highly_variable", "svd_solver": "auto"}
