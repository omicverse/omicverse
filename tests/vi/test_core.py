"""ov.vi core: likelihoods, layers, SCVI module and model on in-memory AnnData."""
import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from omicverse.vi import _distributions as D  # noqa: E402


def _rand_params(B=16, G=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.rand(B, G, generator=g, dtype=torch.float64) * 20 + 1e-3
    theta = torch.rand(G, generator=g, dtype=torch.float64) * 5 + 0.05
    pi = torch.randn(B, G, generator=g, dtype=torch.float64)
    x = torch.poisson(mu * (torch.rand(B, G, generator=g, dtype=torch.float64) < 0.4))
    return x, mu, theta, pi


def test_nb_matches_torch_distribution():
    x, mu, theta, _ = _rand_params()
    ours = D.log_nb_positive(x, mu, theta, eps=0.0)
    ref = torch.distributions.NegativeBinomial(
        total_count=theta.expand_as(mu), logits=torch.log(mu) - torch.log(theta)).log_prob(x)
    assert torch.allclose(ours, ref, atol=1e-8, rtol=1e-8)


def test_zinb_closed_form():
    x, mu, theta, pi = _rand_params()
    ours = D.log_zinb_positive(x, mu, theta, pi)  # eps=1e-8 selects the zero branch
    nb = D.log_nb_positive(x, mu, theta, eps=0.0)
    p0 = torch.sigmoid(pi)
    ref = torch.where(x == 0, torch.log(p0 + (1 - p0) * torch.exp(nb)), torch.log1p(-p0) + nb)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_poisson_and_normal():
    x, mu, _, _ = _rand_params()
    assert torch.allclose(D.log_poisson(x, mu), torch.distributions.Poisson(mu).log_prob(x))
    loc, scale = torch.randn(5, 3), torch.rand(5, 3) + 0.1
    y = torch.randn(5, 3)
    assert torch.allclose(D.log_normal(y, loc, scale),
                          torch.distributions.Normal(loc, scale).log_prob(y), atol=1e-6)


def test_kl_closed_form():
    qm, qv = torch.randn(7, 4), torch.rand(7, 4) + 0.01
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(qm, qv.sqrt()),
                                            torch.distributions.Normal(0, 1)).sum(-1)
    assert torch.allclose(D.kl_normal_standard(qm, qv), ref, atol=1e-6)
    pm, pv = torch.randn(7, 4), torch.rand(7, 4) + 0.1
    ref2 = torch.distributions.kl_divergence(torch.distributions.Normal(qm, qv.sqrt()),
                                             torch.distributions.Normal(pm, pv.sqrt()))
    assert torch.allclose(D.kl_normal_normal(qm, qv, pm, pv), ref2, atol=1e-5)


@pytest.mark.parametrize("theta_shape", ["gene", "cell"])
def test_sparse_rowsums_equal_dense(theta_shape):
    x, mu, theta, pi = _rand_params(B=32, G=60, seed=3)
    if theta_shape == "cell":
        theta = theta.expand_as(mu).contiguous() * (1 + torch.rand_like(mu))
    row, col = torch.nonzero(x, as_tuple=True)
    val = x[row, col]
    nb = D.nb_rowsum(mu, theta, row, col, val)
    assert torch.allclose(nb, D.log_nb_positive(x, mu, theta).sum(-1), rtol=1e-10, atol=1e-8)
    zi = D.zinb_rowsum(mu, theta, pi, row, col, val)
    assert torch.allclose(zi, D.log_zinb_positive(x, mu, theta, pi).sum(-1), rtol=1e-10, atol=1e-8)


def test_sparse_rowsum_gradients_equal_dense():
    x, mu, theta, pi = _rand_params(B=8, G=20, seed=5)
    row, col = torch.nonzero(x, as_tuple=True)
    grads = []
    for sparse in (False, True):
        m, t, p = (v.clone().requires_grad_(True) for v in (mu, theta, pi))
        f = D.zinb_rowsum(m, t, p, row, col, x[row, col]) if sparse else \
            D.log_zinb_positive(x, m, t, p).sum(-1)
        f.sum().backward()
        grads.append((m.grad, t.grad, p.grad))
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-8)


def test_matches_scvi_tools_logprobs():
    sd = pytest.importorskip("scvi.distributions._negative_binomial")
    x, mu, theta, pi = (v.float() for v in _rand_params(B=64, G=100, seed=7))
    assert torch.allclose(D.log_nb_positive(x, mu, theta), sd.log_nb_positive(x, mu, theta),
                          atol=1e-5, rtol=1e-5)
    assert torch.allclose(D.log_zinb_positive(x, mu, theta, pi),
                          sd.log_zinb_positive(x, mu, theta, pi), atol=1e-5, rtol=1e-5)
    mu2 = mu * 3 + 1
    assert torch.allclose(D.log_mixture_nb(x, mu, mu2, theta, None, pi),
                          sd.log_mixture_nb(x, mu, mu2, theta, None, pi), atol=1e-5, rtol=1e-5)
    th2 = theta * 2
    assert torch.allclose(D.log_mixture_nb(x, mu, mu2, theta, th2, pi),
                          sd.log_mixture_nb(x, mu, mu2, theta, th2, pi), atol=1e-5, rtol=1e-5)
    row, col = torch.nonzero(x, as_tuple=True)
    assert torch.allclose(D.zinb_rowsum(mu, theta, pi, row, col, x[row, col]),
                          sd.log_zinb_positive(x, mu, theta, pi).sum(-1), rtol=1e-5, atol=1e-3)


# ---------------------------------------------------------------------------
# layers / module vs scvi-tools (same weights -> same outputs)
# ---------------------------------------------------------------------------
def _copy_fclayers(ours, theirs):
    """Load scvi-tools FCLayers weights into ours (same parameter layout)."""
    for i, (lin, post) in enumerate(zip(ours.layers, ours.norms)):
        seq = getattr(theirs.fc_layers, f"Layer {i}")
        lin.linear.load_state_dict(seq[0].state_dict())
        bn = [m for m in post if isinstance(m, torch.nn.BatchNorm1d)]
        if bn:
            bn[0].load_state_dict(seq[1].state_dict())


def test_fclayers_match_scvi():
    snn = pytest.importorskip("scvi.nn")
    from omicverse.vi._nn import FCLayers
    torch.manual_seed(0)
    theirs = snn.FCLayers(30, 16, n_cat_list=[3, 1, 4], n_layers=2, n_hidden=20,
                          dropout_rate=0.0)
    ours = FCLayers(30, 16, n_cat_list=[3, 1, 4], n_layers=2, n_hidden=20, dropout_rate=0.0)
    _copy_fclayers(ours, theirs)
    for m in (ours, theirs):
        m.eval()
    x = torch.randn(9, 30)
    c1, c2, c3 = torch.randint(0, 3, (9, 1)), torch.zeros(9, 1, dtype=torch.long), \
        torch.randint(0, 4, (9, 1))
    assert torch.allclose(ours(x, c1, c2, c3), theirs(x, c1, c2, c3), atol=1e-6)
    ours.train(); theirs.train()
    assert torch.allclose(ours(x, c1, c2, c3), theirs(x, c1, c2, c3), atol=1e-6)


def _toy_counts(n=600, g=80, n_batch=3, seed=0):
    import anndata as ad
    rs = np.random.default_rng(seed)
    grp = rs.integers(0, 4, n)
    base = rs.gamma(0.5, 1.0, (4, g))
    lib = rs.lognormal(6.5, 0.4, n)
    m = base[grp] / base[grp].sum(1, keepdims=True) * lib[:, None]
    X = sp.csr_matrix(rs.poisson(m).astype(np.float32))
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{j}" for j in range(g)]
    a.obs["batch"] = np.array(list("abcdefgh"))[np.arange(n) % n_batch]
    a.obs["label"] = np.array(["t0", "t1", "t2", "t3"])[grp]
    a.layers["counts"] = X.copy()
    return a


def test_vae_forward_matches_scvi():
    pytest.importorskip("scvi")
    from scvi.module import VAE
    from omicverse.vi._scvi import VAEModule
    torch.manual_seed(1)
    G, nb = 50, 3
    theirs = VAE(G, n_batch=nb, n_latent=6, n_hidden=32)
    ours = VAEModule(G, n_batch=nb, n_latent=6, n_hidden=32)
    _copy_fclayers(ours.z_encoder.encoder, theirs.z_encoder.encoder)
    ours.z_encoder.mean_encoder.load_state_dict(theirs.z_encoder.mean_encoder.state_dict())
    ours.z_encoder.var_encoder.load_state_dict(theirs.z_encoder.var_encoder.state_dict())
    _copy_fclayers(ours.decoder.px_decoder, theirs.decoder.px_decoder)
    ours.decoder.px_scale_decoder.load_state_dict(theirs.decoder.px_scale_decoder[0].state_dict())
    ours.decoder.px_dropout_decoder.load_state_dict(theirs.decoder.px_dropout_decoder.state_dict())
    with torch.no_grad():
        ours.px_r.copy_(theirs.px_r)
    ours.eval(); theirs.eval()
    x = torch.poisson(torch.rand(20, G) * 5)
    x[:, 0] += 1
    bi = torch.randint(0, nb, (20, 1))
    inf_t = theirs.inference(x, bi)
    inf_o = ours.inference(x, bi.squeeze(1), sample=False)
    assert torch.allclose(inf_o["qz_m"], inf_t["qz"].loc, atol=1e-5)
    assert torch.allclose(inf_o["qz_v"], inf_t["qz"].scale ** 2, atol=1e-5)
    z = inf_t["qz"].loc
    gen_t = theirs.generative(z, inf_t["library"], bi, y=torch.zeros(20, 1, dtype=torch.long))
    gen_o = ours.generative(z, inf_o["library"], bi.squeeze(1))
    assert torch.allclose(gen_o["px_rate"], gen_t["px"].mu, rtol=1e-5, atol=1e-6)
    lp_t = gen_t["px"].log_prob(x).sum(-1)
    row, col = torch.nonzero(x, as_tuple=True)
    b = {"x": x, "row": row, "col": col, "val": x[row, col]}
    assert torch.allclose(ours.log_likelihood(b, gen_o), lp_t, rtol=1e-5, atol=1e-3)


# ---------------------------------------------------------------------------
# model on in-memory AnnData
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy():
    return _toy_counts()


def test_scvi_shapes_and_outputs(toy):
    from omicverse.vi import SCVI
    m = SCVI(toy, layer="counts", batch_key="batch", n_latent=5, n_hidden=32)
    m.train(max_epochs=3, device="cpu", progress=False)
    z = m.get_latent_representation()
    assert z.shape == (toy.n_obs, 5) and np.isfinite(z).all()
    qm, qv = m.get_latent_representation(return_dist=True)
    assert np.allclose(qm, z) and (qv > 0).all()
    zi = m.get_latent_representation(indices=[5, 2, 2, 9])
    assert np.allclose(zi, z[[5, 2, 2, 9]], atol=1e-6)
    ne = m.get_normalized_expression(indices=np.arange(10), n_samples=3, library_size=1e4)
    assert ne.shape == (10, toy.n_vars)
    assert np.allclose(ne.values.sum(1), 1e4, rtol=1e-4)
    ne2 = m.get_normalized_expression(indices=np.arange(10), transform_batch=["a", "b"],
                                      gene_list=["g1", "g3"], return_numpy=True)
    assert ne2.shape == (10, 2)
    chunks = list(m.get_normalized_expression(as_generator=True, batch_size=256))
    assert sum(c[1].shape[0] for c in chunks) == toy.n_obs
    assert np.isfinite(m.get_elbo()) and m.get_elbo() < 0
    h = m.history
    assert len(h["elbo_train"]) == 3 and "kl_weight" in h
    pps = m.posterior_predictive_sample(indices=np.arange(4))
    assert pps.shape == (4, toy.n_vars) and (pps >= 0).all()


def test_scvi_gene_subset_and_nb(toy):
    from omicverse.vi import SCVI
    mask = np.zeros(toy.n_vars, bool)
    mask[::2] = True
    toy.var["hvg"] = mask
    m = SCVI(toy, layer="counts", batch_key="batch", genes="hvg", gene_likelihood="nb",
             dispersion="gene-batch", n_hidden=16)
    m.train(max_epochs=2, device="cpu", progress=False)
    assert m.n_vars == mask.sum()
    sub = toy[:, mask].copy()
    m2 = SCVI(sub, layer="counts", batch_key="batch", gene_likelihood="nb",
              dispersion="gene-batch", n_hidden=16)
    m2.train(max_epochs=2, device="cpu", progress=False)
    assert np.allclose(m.get_latent_representation(), m2.get_latent_representation(), atol=1e-5)


def test_deterministic_training_cpu(toy):
    from omicverse.vi import SCVI
    zs = []
    for _ in range(2):
        m = SCVI(toy, layer="counts", batch_key="batch", n_hidden=32, seed=3)
        m.train(max_epochs=4, device="cpu", progress=False, seed=11)
        zs.append(m.get_latent_representation())
    assert np.array_equal(zs[0], zs[1])
    m = SCVI(toy, layer="counts", batch_key="batch", n_hidden=32, seed=3)
    m.train(max_epochs=4, device="cpu", progress=False, seed=12)
    assert not np.allclose(zs[0], m.get_latent_representation())


def test_save_load_roundtrip(toy, tmp_path):
    from omicverse.vi import SCVI
    m = SCVI(toy, layer="counts", batch_key="batch", n_latent=4, n_hidden=16,
             continuous_covariate_keys=None)
    m.train(max_epochs=2, device="cpu", progress=False)
    z = m.get_latent_representation()
    m.save(str(tmp_path / "m"))
    with pytest.raises(FileExistsError):
        m.save(str(tmp_path / "m"))
    m2 = SCVI.load(str(tmp_path / "m"), adata=toy, device="cpu")
    assert np.allclose(m2.get_latent_representation(), z, atol=1e-6)
    assert m2.history_["elbo_train"] == m.history_["elbo_train"]
    # query data in a different gene order / with extra genes maps onto the trained genes
    q = toy[:50, ::-1].copy()
    assert np.allclose(m2.get_latent_representation(q), z[:50], atol=1e-5)


def test_covariates_and_early_stopping(toy):
    from omicverse.vi import SCVI
    toy.obs["cont"] = np.random.default_rng(0).normal(size=toy.n_obs).astype(np.float32)
    m = SCVI(toy, layer="counts", batch_key="batch", categorical_covariate_keys=["label"],
             continuous_covariate_keys=["cont"], n_hidden=16, encode_covariates=True)
    m.train(max_epochs=6, device="cpu", progress=False, early_stopping=True,
            early_stopping_patience=1, early_stopping_min_delta=1e9)
    assert len(m.history["elbo_validation"]) == 2  # stopped after patience 1
    assert m.get_latent_representation().shape[0] == toy.n_obs


def test_library_prior_mode(toy):
    from omicverse.vi import SCVI
    m = SCVI(toy, layer="counts", batch_key="batch", use_observed_lib_size=False, n_hidden=16)
    m.train(max_epochs=2, device="cpu", progress=False)
    assert np.isfinite(m.get_elbo())


def test_split_matches_scvi_datasplitter():
    from omicverse.vi._data import split_train_val
    tr, va = split_train_val(1000, batch_size=128, seed=0)
    assert len(tr) == 900 and len(va) == 100
    tr, va = split_train_val(1003, batch_size=128, seed=0)  # ceil(902.7)=903 -> 903 % 128 = 7
    assert len(tr) + len(va) == 1003
    tr2, va2 = split_train_val(258, batch_size=128, seed=0)  # 233 % 128 = 105 -> unchanged
    pytest.importorskip("scvi")
    from scvi.dataloaders._data_splitting import validate_data_split
    for n in (1000, 1003, 258, 2563, 129 * 10 + 1):
        n_train, n_val = validate_data_split(n, 0.9, None, 128, False, True)
        tr, va = split_train_val(n, batch_size=128, seed=0)
        assert (len(tr), len(va)) == (n_train, n_val)
        perm = np.random.RandomState(seed=0).permutation(n)
        assert np.array_equal(np.sort(perm[:n_val]), va)
