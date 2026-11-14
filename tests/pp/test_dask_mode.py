"""Chunked GPU mode: ``ov.read(storage='disk')`` in GPU mode + qc/preprocess/scale/pca
streams row blocks and gives the in-memory results.  Runs on CPU (scipy
blocks); on a CUDA machine the same tests also run with GPU blocks."""
import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("dask")


def _gpu_available():
    try:
        import cupy as cp
        import rapids_singlecell  # noqa: F401
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


DEVICES = ["cpu"] + (["gpu"] if _gpu_available() else [])


def _counts(n=2400, g=400, seed=0):
    import anndata as ad
    rs = np.random.default_rng(seed)
    groups = rs.integers(0, 4, n)
    base = rs.gamma(0.4, 1.0, size=(4, g))
    lib = rs.lognormal(7.3, 0.4, n)
    mu = base[groups] / base[groups].sum(1, keepdims=True) * lib[:, None]
    X = sp.csr_matrix(rs.poisson(mu).astype(np.float32))
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    names = [f"G{j}" for j in range(g)]
    names[:8] = [f"MT-{j}" for j in range(8)]
    names[8:20] = [f"RPL{j}" for j in range(12)]
    a.var_names = names
    a.obs["batch"] = np.where(np.arange(n) % 3 == 0, "a", "b")
    return a


@pytest.fixture(scope="module", params=[None, "gzip"], ids=["contiguous", "gzip"])
def h5ad(tmp_path_factory, request):
    path = tmp_path_factory.mktemp("dask") / f"counts_{request.param}.h5ad"
    _counts().write_h5ad(path, compression=request.param)
    return path


def _in_memory(path):
    import anndata as ad
    import omicverse as ov
    ov.settings.cpu_init()
    b = ad.read_h5ad(path)
    b = ov.pp.qc(b, mode="seurat", doublets=False, backend="python",
                 tresh={"mito_perc": 0.5, "nUMIs": 500, "detected_genes": 100})
    ov.pp.preprocess(b, mode="shiftlog|pearson", n_HVGs=150, exclude_highly_expressed=False,
                     backend="python")
    ov.pp.scale(b, max_value=10, backend="python")
    ov.pp.pca(b, n_pcs=10, layer="scaled", backend="python")
    return b


def _chunked(path, device):
    import omicverse as ov
    if device == "gpu":
        ov.settings.gpu_init()
    else:
        ov.settings.cpu_init()
    if device == "gpu":
        a = ov.read(path, storage="disk", chunk_size=700)
    else:  # the same Dask code on scipy blocks (no GPU needed)
        from omicverse.pp._dask import read_h5ad
        a = read_h5ad(path, chunk_size=700, device="cpu")
    a = ov.pp.qc(a, mode="seurat", doublets=False,
                 tresh={"mito_perc": 0.5, "nUMIs": 500, "detected_genes": 100})
    ov.pp.preprocess(a, mode="shiftlog|pearson", n_HVGs=150)
    ov.pp.scale(a, max_value=10)
    ov.pp.pca(a, n_pcs=10, layer="scaled")
    ov.settings.cpu_init()
    return a


@pytest.mark.parametrize("device", DEVICES)
def test_chunked_workflow_matches_in_memory(h5ad, device):
    from omicverse.pp._dask import is_dask
    a = _chunked(h5ad, device)
    b = _in_memory(h5ad)
    assert is_dask(a.X) and is_dask(a.layers["counts"])
    assert list(a.obs_names) == list(b.obs_names)
    assert list(a.var_names) == list(b.var_names)
    for c in ("nUMIs", "detected_genes", "mito_perc"):
        np.testing.assert_allclose(a.obs[c].values, b.obs[c].values, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(a.var["residual_variances"].values,
                               b.var["residual_variances"].values, rtol=1e-4)
    assert (a.var["highly_variable"].values == b.var["highly_variable"].values).all()
    pa, pb = np.asarray(a.obsm["X_pca"]), np.asarray(b.obsm["X_pca"])
    for i in range(5):  # same components up to sign
        assert abs(np.corrcoef(pa[:, i], pb[:, i])[0, 1]) > 0.9999
    np.testing.assert_allclose(a.uns["pca"]["variance_ratio"][:5],
                               b.uns["pca"]["variance_ratio"][:5], rtol=1e-4)


def test_workflow_reads_matrix_four_times(h5ad, monkeypatch):
    """QC, gene filter, HVG (+ scaling statistics), PCA: one pass each."""
    import omicverse as ov
    from omicverse.pp import _dask
    calls = []
    orig = _dask._read_block
    monkeypatch.setattr(_dask, "_read_block", lambda *a, **k: calls.append(1) or orig(*a, **k))
    _chunked(h5ad, "cpu")
    n_blocks = -(-_counts().n_obs // 700)
    assert len(calls) == 4 * n_blocks


def test_lazy_counts_layer_is_not_modified(h5ad):
    import anndata as ad
    a = _chunked(h5ad, "cpu")
    raw = ad.read_h5ad(h5ad)[a.obs_names, a.var_names].X
    counts = a.layers["counts"].compute()
    assert abs(counts - raw).max() == 0


def test_pearson_hvg_with_batches_matches_scanpy(h5ad):
    import anndata as ad
    import scanpy as sc
    import omicverse as ov
    from omicverse.pp import _dask
    a = _dask.read_h5ad(h5ad, chunk_size=500, device="cpu")
    _dask.highly_variable_genes_pearson(a, layer=None, n_top_genes=100, batch_key="batch")
    b = ad.read_h5ad(h5ad)
    sc.experimental.pp.highly_variable_genes(b, flavor="pearson_residuals", n_top_genes=100,
                                             batch_key="batch")
    np.testing.assert_allclose(a.var["residual_variances"].values,
                               b.var["residual_variances"].values, rtol=1e-4)
    assert (a.var["highly_variable"].values == b.var["highly_variable"].values).all()
    assert (a.var["highly_variable_nbatches"].values == b.var["highly_variable_nbatches"].values).all()


def test_doublets_on_dask_raise(h5ad):
    import omicverse as ov
    from omicverse.pp._dask import read_h5ad
    a = read_h5ad(h5ad, device="cpu")
    with pytest.raises(NotImplementedError, match="doublets=False"):
        ov.pp.qc(a, doublets=True)


def test_read_storage_follows_the_mode(h5ad, monkeypatch):
    """storage='disk': AnnDataOOM in the CPU modes, Dask row blocks in GPU
    mode; the default storage='memory' gives a plain AnnData."""
    import anndata as ad
    import omicverse as ov
    from omicverse.pp._dask import is_dask
    monkeypatch.setattr(ov.settings, "mode", "cpu")
    assert type(ov.read(h5ad)) is ad.AnnData  # default = memory
    pytest.importorskip("anndataoom")
    b = ov.read(h5ad, storage="disk")
    try:
        assert type(b).__name__ == "AnnDataOOM"
    finally:
        b.close()
    monkeypatch.setattr(ov.settings, "mode", "gpu")  # gpu_init() without RAPIDS
    a = ov.read(h5ad, storage="disk", chunk_size=500)
    assert is_dask(a.X) and a.X.chunks[0][0] == 500


def test_read_backend_is_a_deprecated_alias(h5ad):
    import anndata as ad
    import omicverse as ov
    with pytest.warns(FutureWarning, match="storage='memory'"):
        assert type(ov.read(h5ad, backend="python")) is ad.AnnData
    with pytest.raises(ValueError, match="storage"):
        ov.read(h5ad, storage="gpu")


def test_read_layer_as_x(tmp_path, monkeypatch):
    """layer= reads raw counts stored next to a normalised X (SEA-AD layout)."""
    import omicverse as ov
    from omicverse.pp._dask import read_h5ad
    a = _counts(n=300, g=50)
    a.layers["UMIs"] = a.X.copy()
    a.X = a.X.multiply(1 / (a.X.sum(1) + 1)).tocsr()
    path = tmp_path / "layered.h5ad"
    a.write_h5ad(path, compression="gzip")
    b = read_h5ad(path, chunk_size=128, device="cpu", layer="UMIs")
    assert abs(b.X.compute() - a.layers["UMIs"]).max() == 0
    monkeypatch.setattr(ov.settings, "mode", "cpu")
    with pytest.raises(NotImplementedError, match="layer="):
        ov.read(path, storage="disk", layer="UMIs")


@pytest.mark.skipif("gpu" not in DEVICES, reason="needs a CUDA device")
def test_staging_copy_waits_for_the_device():
    """The pinned staging buffer is refilled right after each copy; the copy
    must have landed by then even when the GPU queue is busy (regression:
    asynchronous copies corrupted ~5% of the values of SEA-AD)."""
    import cupy as cp
    from omicverse.pp import _dask
    busy = cp.RawKernel(r"""
    extern "C" __global__ void busy(long long cycles) {
        long long t0 = clock64();
        while (clock64() - t0 < cycles) {}
    }""", "busy")
    stage = _dask._staging()
    n = 1 << 20
    src = stage[:n]
    src[:] = 7
    dst = cp.zeros(n, dtype=cp.uint8)
    busy((1,), (1,), (cp.int64(400_000_000),))  # ~0.2 s ahead of the copy on the stream
    _dask._copy_from_stage(dst, src)
    src[:] = 0                                  # refill, as the next piece would
    assert int((dst == 7).sum()) == n


@pytest.mark.skipif("gpu" not in DEVICES, reason="needs a CUDA device")
@pytest.mark.parametrize("compression", [None, "gzip"])
def test_gpu_blocks_exact_through_pinned_buffer(tmp_path, compression):
    """Blocks larger than the pinned staging buffer are decoded and copied
    piece by piece; the device blocks equal the file exactly."""
    import threading
    import anndata as ad
    from omicverse.pp import _dask
    X = sp.random(20_000, 500, density=0.05, format="csr", dtype=np.float32,
                  random_state=np.random.default_rng(7))
    X.data = np.ceil(X.data * 30)
    path = tmp_path / f"big_{compression}.h5ad"
    ad.AnnData(X).write_h5ad(path, compression=compression)
    old_bytes, old_local = _dask._STAGE_BYTES, _dask._LOCAL
    _dask._STAGE_BYTES, _dask._LOCAL = 1 << 16, threading.local()  # many pieces per block
    try:
        b = _dask.read_h5ad(path, chunk_size=7_000, device="gpu")
        _dask._GENE_STATS.clear()
        st = _dask.gene_stats(b.X)
    finally:
        _dask._STAGE_BYTES, _dask._LOCAL = old_bytes, old_local
    np.testing.assert_array_equal(st["nnz"], np.diff(X.tocsc().indptr))
    np.testing.assert_allclose(st["sum"], np.asarray(X.sum(0)).ravel(), rtol=1e-12)
