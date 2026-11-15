"""ov.vi.DRVI: likelihoods, additive split decoder, model on in-memory AnnData,
AnnDataOOM and Dask, plus the interpretability utilities (latent traversal /
per-dimension gene effects)."""
import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")
ad = pytest.importorskip("anndata")

from .test_data import _KEEP, _counts, _oom  # noqa: E402


def test_log_pnb_matches_log_nb():
    """DRVI's log-mean 'pnb' likelihood is the same NB, reparameterised."""
    from omicverse.vi._distributions import log_nb_positive, log_pnb_positive
    g = torch.Generator().manual_seed(0)
    x = torch.poisson(torch.rand(20, 15, generator=g) * 10)
    mu = torch.rand(20, 15, generator=g).exp() + 0.1
    theta = torch.rand(15, generator=g).exp() + 0.1
    ref = log_nb_positive(x, mu, theta)
    got = log_pnb_positive(x, torch.log(mu), torch.log(theta))
    assert torch.allclose(ref, got, atol=1e-5)


@pytest.mark.parametrize("gene_likelihood", ["pnb", "nb", "zinb", "poisson"])
def test_module_forward_backward(gene_likelihood):
    from omicverse.vi._drvi import DRVIModule
    torch.manual_seed(0)
    n, g, nb = 32, 40, 3
    mod = DRVIModule(n_input=g, n_batch=nb, n_latent=8, n_hidden=32, n_layers=1,
                     gene_likelihood=gene_likelihood)
    x = torch.poisson(torch.ones(n, g) * 3)
    b = {"x": x, "batch": torch.randint(0, nb, (n,))}
    out = mod(b, kl_weight=1.0)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert mod.decoder.split_w.grad is not None
    assert mod.decoder.mean_head.weight.grad is not None


def test_split_decoder_terms_depend_only_on_own_dimension():
    """The decoder's defining property: before aggregation, split i's
    per-gene term is a function of ``z[:, i]`` alone (and the shared decoder
    weights), never of the other latent dimensions — the "additive" part of
    the additive split decoder."""
    from omicverse.vi._drvi import SplitDecoderDRVI
    torch.manual_seed(0)
    dec = SplitDecoderDRVI(n_latent=4, n_output=10, n_hidden=16, n_layers=1)
    dec.eval()

    def per_split_logit(z):
        b, n_latent = z.shape
        z_r = z.reshape(b, dec.n_split, dec.split_size)
        z_s = torch.einsum("bsd,sdn->bsn", z_r, dec.split_w)
        h = dec.shared(z_s.reshape(b * dec.n_split, n_latent))
        return dec.mean_head(h).reshape(b, dec.n_split, -1)

    z1 = torch.randn(5, 4)
    z2 = z1.clone()
    z2[:, [0, 1, 3]] = torch.randn(5, 3)  # perturb every dim except dim 2
    with torch.no_grad():
        t1, t2 = per_split_logit(z1), per_split_logit(z2)
    assert torch.allclose(t1[:, 2, :], t2[:, 2, :], atol=1e-5)
    assert not torch.allclose(t1[:, 0, :], t2[:, 0, :])


def _disentangled(n=1200, g=80, seed=0):
    """Two independent discrete factors (cell type x batch), each driving a
    disjoint block of genes, so a disentangling model should recover one
    latent dimension that tracks cell type and another that tracks batch."""
    rs = np.random.default_rng(seed)
    ct = rs.integers(0, 3, n)
    bt = rs.integers(0, 2, n)
    ct_genes, bt_genes = slice(0, g // 2), slice(g // 2, g)
    base = np.full((n, g), 1.0)
    ct_mu = rs.gamma(2.0, 1.0, (3, g // 2))
    bt_mu = rs.gamma(2.0, 1.0, (2, g - g // 2))
    base[:, ct_genes] = ct_mu[ct]
    base[:, bt_genes] = bt_mu[bt]
    lib = rs.lognormal(6.5, 0.2, n)
    m = base / base.sum(1, keepdims=True) * lib[:, None]
    X = sp.csr_matrix(rs.poisson(m).astype(np.float32))
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{j}" for j in range(g)]
    a.obs["cell_type"] = np.array([f"ct{k}" for k in range(3)])[ct]
    a.obs["batch"] = np.array([f"b{k}" for k in range(2)])[bt]
    a.layers["counts"] = a.X.copy()
    return a


def test_drvi_model_train_and_latent(tmp_path):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from omicverse.vi import DRVI
    torch.set_num_threads(4)
    a = _disentangled()
    m = DRVI(a, layer="counts", batch_key="batch", n_latent=6, n_hidden=32, n_layers=1, seed=0)
    m.train(max_epochs=40, device="cpu", progress=False, batch_size=128)
    z = m.get_latent_representation()
    assert z.shape == (a.n_obs, 6)
    ari = adjusted_rand_score(a.obs["cell_type"],
                              KMeans(3, n_init=5, random_state=0).fit_predict(z))
    assert ari > 0.6
    ne = m.get_normalized_expression(library_size=1e4)
    assert ne.shape == (a.n_obs, 80)
    assert np.allclose(ne.values.sum(1), 1e4, rtol=1e-3)
    assert np.isfinite(m.get_elbo())

    m.save(str(tmp_path / "d"))
    m2 = DRVI.load(str(tmp_path / "d"), adata=a, device="cpu")
    assert np.allclose(m2.get_latent_representation(), z, atol=1e-5)


def test_latent_traverse_and_differential_vars():
    from omicverse.vi import DRVI
    a = _disentangled(n=600)
    m = DRVI(a, layer="counts", batch_key="batch", n_latent=4, n_hidden=16, n_layers=1, seed=0)
    m.train(max_epochs=10, device="cpu", progress=False, batch_size=128)
    eff = m.latent_traverse(n_samples=16, seed=0)
    assert eff.shape == (4, a.n_vars)
    assert np.isfinite(eff).all()
    genes, vals = m.differential_vars(dim=0, n_genes=5)
    assert len(genes) == 5 and len(vals) == 5
    assert (np.abs(vals) >= 0).all()
    # reproducible given the same seed
    eff2 = m.latent_traverse(n_samples=16, seed=0)
    assert np.allclose(eff, eff2)
    allg = m.dimension_gene_effects(n_genes=3, n_samples=16, seed=0)
    assert set(allg) == set(range(4))
    # a non-contiguous `dims=` subset must be keyed by the *actual* dimension
    # indices, not by their position in `dims` (regression: used to raise
    # KeyError / silently mislabel dimensions for any dims != range(k)).
    subset = m.dimension_gene_effects(n_genes=3, dims=[3, 1], n_samples=16, seed=0)
    assert set(subset) == {3, 1}
    g3, v3 = subset[3]
    g3_direct, v3_direct = m.differential_vars(dim=3, n_genes=3, n_samples=16, seed=0)
    assert list(g3) == list(g3_direct) and np.allclose(v3, v3_direct)


def test_oom_matches_in_memory(tmp_path):
    import omicverse as ov
    from omicverse.vi import DRVI
    a = _counts(n=1500, g=150, seed=2)

    def _train(adata, **kw):
        m = DRVI(adata, batch_key="batch", n_hidden=24, n_layers=1, n_latent=5, seed=0)
        m.train(max_epochs=4, device="cpu", progress=False, shuffle="block", block_size=256,
                buffer_blocks=4, seed=0, **kw)
        return m

    path = tmp_path / "drvi_counts.h5ad"
    a.write_h5ad(path)
    z_mem = _train(a).get_latent_representation()
    oom = _oom(path, np.ones(a.n_obs, bool), np.ones(a.n_vars, bool))
    m_oom = _train(oom)
    assert m_oom._source().kind == "h5"
    assert np.array_equal(z_mem, m_oom.get_latent_representation())


def test_dask_matches_in_memory(tmp_path):
    pytest.importorskip("dask")
    from omicverse.pp._dask import read_h5ad
    from omicverse.vi import DRVI
    a = _counts(n=1200, g=120, seed=4)
    path = tmp_path / "drvi_dask.h5ad"
    a.write_h5ad(path)

    def _train(adata, **kw):
        m = DRVI(adata, batch_key="batch", n_hidden=24, n_layers=1, n_latent=5, seed=0)
        m.train(max_epochs=4, device="cpu", progress=False, shuffle="block", block_size=256,
                buffer_blocks=4, seed=0, **kw)
        return m

    z1 = _train(a).get_latent_representation()
    da_ad = read_h5ad(path, chunk_size=256, device="cpu")
    m_da = _train(da_ad)
    assert m_da._source().kind == "dask"
    assert np.array_equal(z1, m_da.get_latent_representation())


def test_single_drvi_wrapper():
    import omicverse as ov
    a = _disentangled(n=300)
    m = ov.single.drvi(a, layer="counts", batch_key="batch",
                       model_kwargs=dict(n_latent=4, n_hidden=16, n_layers=1, seed=0),
                       train_kwargs=dict(max_epochs=5, progress=False, device="cpu"))
    assert "X_drvi" in a.obsm and a.obsm["X_drvi"].shape == (300, 4)
    with pytest.raises(ValueError, match="overwrite"):
        ov.single.drvi(a, layer="counts",
                       model_kwargs=dict(n_latent=4, n_hidden=16, n_layers=1))
