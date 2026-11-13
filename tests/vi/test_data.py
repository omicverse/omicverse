"""ov.vi data sources: AnnData (device / host), AnnDataOOM and Dask inputs give the
same minibatches and the same trained model; streaming memory stays bounded."""
import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")
ad = pytest.importorskip("anndata")


def _counts(n=3000, g=300, seed=0):
    rs = np.random.default_rng(seed)
    grp = rs.integers(0, 5, n)
    base = rs.gamma(0.4, 1.0, (5, g))
    lib = rs.lognormal(7.0, 0.4, n)
    m = base[grp] / base[grp].sum(1, keepdims=True) * lib[:, None]
    X = sp.csr_matrix(rs.poisson(m).astype(np.float32))
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"G{j}" for j in range(g)]
    a.obs["batch"] = np.where(np.arange(n) < n // 2, "b0", "b1")  # batches in row order
    a.obs["label"] = np.array([f"t{k}" for k in range(5)])[grp]
    return a


@pytest.fixture(scope="module", params=[None, "gzip"], ids=["contiguous", "gzip"])
def h5ad(tmp_path_factory, request):
    path = tmp_path_factory.mktemp("vi") / f"counts_{request.param}.h5ad"
    _counts().write_h5ad(path, compression=request.param)
    return path


def _row_mask(n, seed=1):
    keep = np.random.default_rng(seed).random(n) > 0.1
    keep[:5] = True
    return keep


def _gene_mask(g):
    m = np.zeros(g, bool)
    m[np.random.default_rng(2).choice(g, 120, replace=False)] = True
    return m


_KEEP = []  # anndataoom closes the file when the parent object is collected


def _oom(path, rows, genes):
    import omicverse as ov
    ov.settings.cpu_init()
    a = ov.read(str(path), storage="disk")
    _KEEP.append(a)
    return a[rows]


def _mem(path, rows):
    return ad.read_h5ad(path)[rows].copy()


def _batches(src, rows=None, **kw):
    out = []
    for b in src.iterate(rows, **kw):
        out.append((b["idx"].numpy().copy(), b["x"].numpy().copy(), b["batch"].numpy().copy()))
    return out


def _same(b1, b2):
    assert len(b1) == len(b2)
    for (i1, x1, c1), (i2, x2, c2) in zip(b1, b2):
        assert np.array_equal(i1, i2)
        assert np.array_equal(x1, x2)
        assert np.array_equal(c1, c2)


def _build(adata, genes, **kw):
    from omicverse.vi._data import build_source
    codes = (np.asarray(adata.obs["batch"].values) == "b1").astype(np.int64)
    src, _ = build_source(adata, genes=genes, obs={"batch": codes}, device="cpu", **kw)
    return src


def test_sources_give_identical_minibatches(h5ad):
    rows = _row_mask(3000)
    gm = _gene_mask(300)
    mem = _mem(h5ad, rows)
    dev = _build(mem, gm, block_size=256)
    host = _build(mem, gm, block_size=256, source="host")
    oom = _build(_oom(h5ad, rows, gm), gm, block_size=256)
    assert dev.kind == "device" and host.kind == "host" and oom.kind == "h5"
    assert dev.n_obs == oom.n_obs == rows.sum() and oom.n_vars == gm.sum()
    train = np.sort(np.random.default_rng(0).choice(dev.n_obs, 2000, replace=False))
    kw = dict(batch_size=128, shuffle="block", seed=3, epoch=2, buffer_blocks=3)
    ref = _batches(dev, train, **kw)
    _same(ref, _batches(host, train, **kw))
    _same(ref, _batches(oom, train, **kw))
    # every training cell exactly once, minibatches straddle buffers
    idx = np.concatenate([r[0] for r in ref])
    assert np.array_equal(np.sort(idx), train)
    assert all(len(r[0]) == 128 for r in ref[:-1])
    # in-order pass (inference)
    _same(_batches(dev, None, batch_size=500), _batches(oom, None, batch_size=500))
    # dense rows equal the matrix
    X = mem.X[:, gm].toarray()
    for i, x, _ in ref[:3]:
        assert np.array_equal(x, X[i])


def test_dask_source_identical(h5ad):
    pytest.importorskip("dask")
    from omicverse.pp._dask import read_h5ad
    gm = _gene_mask(300)
    da_ad = read_h5ad(h5ad, chunk_size=256, device="cpu")
    mem = ad.read_h5ad(h5ad)
    src = _build(da_ad, gm)
    assert src.kind == "dask"
    ref = _build(mem, gm, block_size=256)
    assert np.array_equal(src.starts, ref.starts) and np.array_equal(src.ends, ref.ends)
    kw = dict(batch_size=100, shuffle="block", seed=5, epoch=0, buffer_blocks=2)
    _same(_batches(ref, None, **kw), _batches(src, None, **kw))
    # with a row subset the dask chunks shrink; the blocks follow the chunks
    rows = _row_mask(3000)
    sub = da_ad[rows]
    s2 = _build(sub, gm)
    r2 = _build(mem[rows].copy(), gm)
    r2.starts, r2.ends = s2.starts.copy(), s2.ends.copy()
    _same(_batches(r2, None, **kw), _batches(s2, None, **kw))


def test_random_shuffle_needs_resident(h5ad):
    rows = _row_mask(3000)
    oom = _build(_oom(h5ad, rows, None), None, block_size=256)
    with pytest.raises(ValueError):
        next(oom.iterate(None, shuffle="random"))


def test_distributed_sharding_partitions_blocks(h5ad):
    mem = _mem(h5ad, _row_mask(3000))
    src = _build(mem, None, block_size=200)
    rows = np.arange(src.n_obs)
    world = 3
    counts = src.n_rows_per_rank(rows, world, seed=1, epoch=4)
    nb = min(c // 64 for c in counts)
    seen = []
    for r in range(world):
        bs = _batches(src, rows, batch_size=64, shuffle="block", seed=1, epoch=4,
                      buffer_blocks=2, rank=r, world=world, n_batches=nb, drop_last=True)
        assert len(bs) == nb  # equal minibatch counts on every rank
        seen.append(np.concatenate([b[0] for b in bs]))
    allseen = np.concatenate(seen)
    assert len(np.unique(allseen)) == len(allseen)  # disjoint shards


def _train(adata, **kw):
    from omicverse.vi import SCVI
    m = SCVI(adata, batch_key="batch", genes=kw.pop("genes", None), n_hidden=32, n_latent=5,
             seed=0)
    m.train(max_epochs=3, device="cpu", progress=False, shuffle="block", block_size=256,
            buffer_blocks=4, seed=0, **kw)
    return m


def test_same_latent_from_anndata_oom_and_dask(h5ad):
    rows = _row_mask(3000)
    gm = _gene_mask(300)
    mem = _mem(h5ad, rows)
    oom = _oom(h5ad, rows, gm)
    z_mem = _train(mem, genes=gm).get_latent_representation()
    m_oom = _train(oom, genes=gm)
    assert m_oom._source().kind == "h5"
    assert np.array_equal(z_mem, m_oom.get_latent_representation())
    # Dask: same block layout as the in-memory run (no row filter, chunk = block size)
    pytest.importorskip("dask")
    from omicverse.pp._dask import read_h5ad
    full = ad.read_h5ad(h5ad)
    da_ad = read_h5ad(h5ad, chunk_size=256, device="cpu")
    z1 = _train(full, genes=gm).get_latent_representation()
    m_da = _train(da_ad, genes=gm)
    assert m_da._source().kind == "dask"
    assert np.array_equal(z1, m_da.get_latent_representation())


def test_oom_counts_layer_after_preprocess(h5ad):
    import omicverse as ov
    ov.settings.cpu_init()
    a0 = ov.read(str(h5ad), storage="disk")
    _KEEP.append(a0)
    a = a0[_row_mask(3000)]
    ov.pp.preprocess(a, mode="shiftlog|pearson", n_HVGs=100)
    from omicverse.vi import SCVI
    with pytest.raises(ValueError, match="raw counts"):
        SCVI(a, batch_key="batch", genes="highly_variable_features").train(
            max_epochs=1, device="cpu", progress=False)
    m = SCVI(a, layer="counts", batch_key="batch", genes="highly_variable_features",
             n_hidden=16)
    m.train(max_epochs=1, device="cpu", progress=False)
    assert m._source().kind == "h5" and m.n_vars == 100
    mem = _mem(h5ad, _row_mask(3000))
    hv = np.asarray(a.var["highly_variable_features"].values)
    m2 = SCVI(mem, batch_key="batch", genes=hv, n_hidden=16)
    m2.train(max_epochs=1, device="cpu", progress=False, shuffle="block", block_size=2048,
             buffer_blocks=32)
    assert np.array_equal(m.get_latent_representation(), m2.get_latent_representation())


def test_streaming_memory_is_bounded(tmp_path):
    psutil = pytest.importorskip("psutil")
    import gc
    n, g = 60000, 2000
    rs = np.random.default_rng(0)
    nnz_per = 400
    indptr = np.arange(0, (n + 1) * nnz_per, nnz_per, dtype=np.int64)
    indices = np.sort(rs.integers(0, g, (n, nnz_per)), axis=1)
    indices = np.unique(indices + np.arange(n)[:, None] * g).astype(np.int64)  # dedupe per row
    rows_of = indices // g
    indices = (indices % g).astype(np.int32)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows_of, minlength=n))])
    X = sp.csr_matrix((np.ones(len(indices), np.float32), indices, indptr), shape=(n, g))
    a = ad.AnnData(X)
    a.obs["batch"] = "b0"
    path = tmp_path / "big.h5ad"
    a.write_h5ad(path)
    file_mb = X.data.nbytes / 2**20 + X.indices.nbytes / 2**20
    del a, X, indices, rows_of, indptr
    gc.collect()
    import omicverse as ov
    ov.settings.cpu_init()
    oom = ov.read(str(path), storage="disk")
    src = _build(oom, None, block_size=256)
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak = base
    seen = 0
    for b in src.iterate(None, batch_size=256, shuffle="block", seed=0, buffer_blocks=2):
        seen += b["x"].shape[0]
        peak = max(peak, proc.memory_info().rss)
    assert seen == n
    grow_mb = (peak - base) / 2**20
    # shuffle buffer: 2 blocks x 256 cells x ~400 nnz x 8 B ~ 1.6 MB; a few buffers are
    # in flight (reading / queued / in use) plus 2 MB dense minibatches: O(buffer), not O(n)
    assert grow_mb < 0.25 * file_mb, (grow_mb, file_mb)
