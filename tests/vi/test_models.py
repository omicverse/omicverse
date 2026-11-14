"""ov.vi SCANVI and TOTALVI."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")
ad = pytest.importorskip("anndata")

from .test_data import _KEEP, _counts  # noqa: E402


def _semi(n=3000, seed=1, frac_unlabeled=0.7):
    a = _counts(n=n, g=200, seed=seed)
    truth = a.obs["label"].astype(str).values.copy()
    lab = truth.copy()
    lab[np.random.default_rng(0).random(n) < frac_unlabeled] = "Unknown"
    a.obs["lab"] = lab
    return a, truth


def test_scanvi_predicts_unlabeled_cells(tmp_path):
    from omicverse.vi import SCANVI
    torch.set_num_threads(4)
    a, truth = _semi()
    m = SCANVI(a, batch_key="batch", labels_key="lab", unlabeled_category="Unknown",
               n_hidden=32)
    assert m.n_labels == 5 and "Unknown" not in m.registry["label_categories"]
    m.train(max_epochs=15, device="cpu", progress=False, check_val_every_n_epoch=5)
    assert "classification_loss_train" in m.history_
    assert "classification_loss_validation" in m.history_
    unl = a.obs["lab"].values == "Unknown"
    pred = m.predict()
    assert (pred[unl] == truth[unl]).mean() > 0.9
    soft = m.predict(soft=True, indices=np.arange(10))
    assert soft.shape == (10, 5) and np.allclose(soft.values.sum(1), 1, atol=1e-5)
    # labelled split is stratified: every split contains labelled cells
    labelled = np.flatnonzero(~unl)
    assert np.isin(m.train_indices, labelled).any() and np.isin(m.validation_indices,
                                                                labelled).any()
    m.save(str(tmp_path / "s"))
    m2 = SCANVI.load(str(tmp_path / "s"), adata=a, device="cpu")
    assert np.array_equal(m2.predict(), pred)


def test_scanvi_from_scvi_and_block_shuffle_on_disk(tmp_path):
    import omicverse as ov
    from omicverse.vi import SCANVI, SCVI
    a, truth = _semi(n=2500, seed=3)
    s = SCVI(a, batch_key="batch", n_hidden=32)
    s.train(max_epochs=10, device="cpu", progress=False)
    m = SCANVI.from_scvi_model(s, unlabeled_category="Unknown", labels_key="lab")
    assert m.was_pretrained
    ref = s.module.state_dict()
    got = m.module.state_dict()
    assert torch.equal(ref["z_encoder.mean_encoder.weight"], got["z_encoder.mean_encoder.weight"])
    m.train(device="cpu", progress=False)
    assert len(m.history_["elbo_train"]) == 10  # min(10, max(2, round(400 / 3)))
    unl = a.obs["lab"].values == "Unknown"
    assert (m.predict()[unl] == truth[unl]).mean() > 0.9
    # AnnDataOOM input: block shuffle, labelled stream drawn from the shuffle buffer
    path = tmp_path / "semi.h5ad"
    a.write_h5ad(path)
    ov.settings.cpu_init()
    oom = ov.read(str(path), storage="disk")
    _KEEP.append(oom)
    d = SCANVI(oom, batch_key="batch", labels_key="lab", unlabeled_category="Unknown",
               n_hidden=32)
    d.train(max_epochs=15, device="cpu", progress=False, block_size=256, buffer_blocks=4)
    assert d._source().kind == "h5"
    assert (d.predict()[unl] == truth[unl]).mean() > 0.9


def _cite(n=2000, g=150, p=14, seed=0, missing=False):
    rs = np.random.default_rng(seed)
    grp = rs.integers(0, 4, n)
    base = rs.gamma(0.5, 1, (4, g))
    lib = rs.lognormal(7, 0.3, n)
    X = sp.csr_matrix(rs.poisson(base[grp] / base[grp].sum(1, keepdims=True)
                                 * lib[:, None]).astype(np.float32))
    fg = rs.gamma(2, 30, (4, p)) * (rs.random((4, p)) < 0.5)
    pro = rs.poisson(rs.gamma(5, 2, (n, p)) + fg[grp]).astype(np.float32)
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs["batch"] = np.where(np.arange(n) % 2 == 0, "b0", "b1")
    a.obs["label"] = np.array(["A", "B", "C", "D"])[grp]
    if missing:
        pro[a.obs["batch"].values == "b1", :3] = 0
    a.obsm["protein_expression"] = pd.DataFrame(pro, index=a.obs_names,
                                                columns=[f"P{i}" for i in range(p)])
    return a


@pytest.mark.parametrize("missing", [False, True])
def test_totalvi(missing, tmp_path):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from omicverse.vi import TOTALVI
    a = _cite(missing=missing)
    m = TOTALVI(a, batch_key="batch", protein_expression_obsm_key="protein_expression")
    assert m.module.has_extra_step == missing
    assert (m.module.protein_batch_mask is not None) == missing
    m.train(max_epochs=15, device="cpu", progress=False)
    assert "elbo_validation" in m.history_  # reduce-LR-on-plateau validates every epoch
    z = m.get_latent_representation()
    assert z.shape == (a.n_obs, 20)
    ari = adjusted_rand_score(a.obs["label"], KMeans(4, n_init=5, random_state=0).fit_predict(z))
    assert ari > 0.8
    rna, pro = m.get_normalized_expression(n_samples=2, transform_batch=["b0", "b1"],
                                           library_size=1e4)
    assert rna.shape == (a.n_obs, 150) and pro.shape == (a.n_obs, 14)
    assert np.allclose(rna.values.sum(1), 1e4, rtol=1e-3)
    fg = m.get_protein_foreground_probability(protein_list=["P0", "P5"])
    assert fg.shape == (a.n_obs, 2) and ((fg.values >= 0) & (fg.values <= 1)).all()
    assert np.isfinite(m.get_elbo())
    m.save(str(tmp_path / "t"))
    m2 = TOTALVI.load(str(tmp_path / "t"), adata=a, device="cpu")
    assert np.allclose(m2.get_latent_representation(), z, atol=1e-5)


def test_totalvi_mixture_matches_scvi_tools():
    pytest.importorskip("scvi")
    from scvi.distributions import NegativeBinomialMixture
    from omicverse.vi._distributions import log_mixture_nb
    g = torch.Generator().manual_seed(0)
    y = torch.poisson(torch.rand(30, 12, generator=g) * 50)
    mu1 = torch.rand(30, 12, generator=g) * 10 + 0.1
    mu2 = mu1 * (1 + torch.rand(30, 12, generator=g) * 5)
    th = torch.rand(12, generator=g) * 3 + 0.1
    pi = torch.randn(30, 12, generator=g)
    ref = NegativeBinomialMixture(mu1=mu1, mu2=mu2, theta1=th, mixture_logits=pi).log_prob(y)
    assert torch.allclose(log_mixture_nb(y, mu1, mu2, th, None, pi), ref, atol=1e-4, rtol=1e-5)
