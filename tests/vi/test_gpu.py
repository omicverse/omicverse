"""ov.vi on CUDA: device-resident / pinned-staging / AnnDataOOM / Dask(cupy) sources
give the same minibatches as on the CPU (skipped without a GPU)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from .test_data import _KEEP, _batches, _counts, _gene_mask, _mem, _row_mask, _same  # noqa: E402


@pytest.fixture(scope="module", params=[None, "gzip"], ids=["contiguous", "gzip"])
def h5ad(tmp_path_factory, request):
    path = tmp_path_factory.mktemp("vigpu") / f"counts_{request.param}.h5ad"
    _counts().write_h5ad(path, compression=request.param)
    return path


def _build(adata, genes, device, **kw):
    from omicverse.vi._data import build_source
    codes = (np.asarray(adata.obs["batch"].values) == "b1").astype(np.int64)
    src, _ = build_source(adata, genes=genes, obs={"batch": codes}, device=device, **kw)
    return src


def _cpu(batches):
    return [(i, x, c) for i, x, c in batches]


def _gpu_batches(src, rows=None, **kw):
    out = []
    for b in src.iterate(rows, **kw):
        assert b["x"].is_cuda
        out.append((b["idx"].cpu().numpy(), b["x"].cpu().numpy(), b["batch"].cpu().numpy()))
    return out


def test_gpu_sources_match_cpu(h5ad):
    import omicverse as ov
    rows = _row_mask(3000)
    gm = _gene_mask(300)
    mem = _mem(h5ad, rows)
    ov.settings.cpu_init()
    a = ov.read(str(h5ad), storage="disk")
    _KEEP.append(a)
    oom = a[rows]
    kw = dict(batch_size=128, shuffle="block", seed=3, epoch=1, buffer_blocks=3)
    ref = _batches(_build(mem, gm, "cpu", block_size=256), None, **kw)
    for src in (_build(mem, gm, "cuda", block_size=256),
                _build(mem, gm, "cuda", block_size=256, source="host"),
                _build(oom, gm, "cuda", block_size=256)):
        _same(ref, _gpu_batches(src, None, **kw))


def test_gpu_dask_cupy_blocks(h5ad):
    pytest.importorskip("cupy")
    from omicverse.pp._dask import read_h5ad
    gm = _gene_mask(300)
    da_ad = read_h5ad(h5ad, chunk_size=256, device="gpu")
    src = _build(da_ad, gm, "cuda")
    assert src.kind == "dask"
    import anndata as ad
    ref = _build(ad.read_h5ad(h5ad), gm, "cpu", block_size=256)
    kw = dict(batch_size=100, shuffle="block", seed=5, epoch=0, buffer_blocks=2)
    _same(_batches(ref, None, **kw), _gpu_batches(src, None, **kw))


def test_gpu_training_close_to_cpu(h5ad):
    import anndata as ad
    from omicverse.vi import SCVI
    mem = ad.read_h5ad(h5ad)
    elbos = []
    for dev in ("cpu", "cuda"):
        m = SCVI(mem, batch_key="batch", n_hidden=32, n_latent=5, seed=0)
        m.train(max_epochs=20, device=dev, progress=False, seed=0)
        assert np.isfinite(m.get_latent_representation()).all()
        elbos.append(m.get_elbo())
    # different RNG streams / kernels -> not bitwise, but the same fit quality
    assert abs(elbos[0] - elbos[1]) < 0.02 * abs(elbos[0]), elbos
