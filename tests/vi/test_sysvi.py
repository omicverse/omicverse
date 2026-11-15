"""ov.vi.SysVI: cVAE + VampPrior + latent cycle-consistency for cross-system
integration, and the VampPrior itself (omicverse.vi._priors.VampPrior)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ad = pytest.importorskip("anndata")

from .test_data import _KEEP  # noqa: E402


def _cross_system(n=600, g=50, seed=0, n_systems=2, donors_per_system=2):
    """Synthetic, already-"normalized" (continuous, Gaussian-ish) expression
    with a strong 'system' effect (what SysVI corrects via conditioning +
    cycle loss) and a weaker 'donor' effect (a covariate only)."""
    rs = np.random.default_rng(seed)
    sysno = rs.integers(0, n_systems, n)
    donor = sysno * donors_per_system + rs.integers(0, donors_per_system, n)
    grp = rs.integers(0, 3, n)  # the biology we want preserved
    bio = rs.normal(0, 1, (3, g))
    sys_shift = rs.normal(0, 1, (n_systems, g))
    donor_shift = rs.normal(0, 0.3, (n_systems * donors_per_system, g))
    X = (bio[grp] + sys_shift[sysno] + donor_shift[donor]
        + rs.normal(0, 0.2, (n, g))).astype(np.float32)
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs["system"] = [f"s{k}" for k in sysno]
    a.obs["donor"] = [f"d{k}" for k in donor]
    a.obs["label"] = [f"t{k}" for k in grp]
    for k in ("system", "donor", "label"):
        a.obs[k] = a.obs[k].astype("category")
    return a


def test_sysvi_basic_train_and_latent(tmp_path):
    from omicverse.vi import SysVI
    torch.set_num_threads(4)
    a = _cross_system()
    m = SysVI(a, batch_key="system", categorical_covariate_keys=["donor"],
              n_hidden=32, n_latent=8, n_prior_components=4, seed=0)
    assert m.n_batch == 2
    m.train(max_epochs=10, device="cpu", progress=False, batch_size=64)
    assert m.is_trained_
    for k in ("reconstruction_loss_train", "kl_local_train", "cycle_loss_train", "elbo_train"):
        assert k in m.history_ and np.isfinite(m.history_[k]).all()
    z = m.get_latent_representation()
    assert z.shape == (a.n_obs, 8) and np.isfinite(z).all()
    z_sample = m.get_latent_representation(give_mean=False)
    assert z_sample.shape == z.shape
    assert np.isfinite(m.get_elbo())
    m.save(str(tmp_path / "s"))
    m2 = SysVI.load(str(tmp_path / "s"), adata=a, device="cpu")
    assert np.allclose(m2.get_latent_representation(), z, atol=1e-5)


def test_sysvi_disables_kl_warmup_by_default():
    from omicverse.vi import SysVI
    a = _cross_system(n=300)
    m = SysVI(a, batch_key="system", n_hidden=16, n_latent=6, n_prior_components=3, seed=0)
    m.train(max_epochs=3, device="cpu", progress=False)
    # with n_epochs_kl_warmup=0, kl_weight is always 1 (scvi-tools' SysVI.train override)
    assert all(w == 1.0 for w in m.history_["kl_weight"])
    with pytest.warns(UserWarning, match="KL warm-up"):
        m.train(max_epochs=1, device="cpu", progress=False,
                plan_kwargs={"n_epochs_kl_warmup": 400})


def test_sysvi_needs_at_least_two_systems():
    from omicverse.vi import SysVI
    a = _cross_system(n_systems=1, donors_per_system=2)
    with pytest.raises(ValueError, match="at least two"):
        SysVI(a, batch_key="system", n_hidden=16, n_latent=6)
    with pytest.raises(ValueError, match="batch_key"):
        SysVI(a, batch_key=None)


def test_sysvi_standard_normal_prior():
    from omicverse.vi import SysVI
    a = _cross_system(n=300)
    m = SysVI(a, batch_key="system", prior="standard_normal", n_hidden=16, n_latent=6, seed=0)
    m.train(max_epochs=3, device="cpu", progress=False)
    assert np.isfinite(m.get_elbo())


def test_sysvi_disk_and_dask_give_same_latent_as_in_memory(tmp_path):
    import omicverse as ov
    from omicverse.vi import SysVI
    a = _cross_system(n=1200, g=40, seed=2)
    path = tmp_path / "cross_system.h5ad"
    a.write_h5ad(path)

    def _train(adata, **kw):
        m = SysVI(adata, batch_key="system", n_hidden=16, n_latent=6, n_prior_components=3,
                  seed=0)
        m.train(max_epochs=6, device="cpu", progress=False, seed=0, shuffle="block", **kw)
        return m

    # same block layout on every source (mem / AnnDataOOM / dask) -> identical
    # minibatches -> bitwise-identical trained latent (as tests/vi/test_data.py)
    m_mem = _train(ad.read_h5ad(path), block_size=128, buffer_blocks=4)
    z_mem = m_mem.get_latent_representation()

    ov.settings.cpu_init()
    oom = ov.read(str(path), storage="disk")
    _KEEP.append(oom)
    m_oom = _train(oom, block_size=128, buffer_blocks=4)
    assert m_oom._source().kind in ("h5", "oom")
    assert np.array_equal(z_mem, m_oom.get_latent_representation())

    pytest.importorskip("dask")
    from omicverse.pp._dask import read_h5ad
    da_ad = read_h5ad(path, chunk_size=128, device="cpu")
    m_da = _train(da_ad, block_size=128, buffer_blocks=4)
    assert m_da._source().kind == "dask"
    assert np.array_equal(z_mem, m_da.get_latent_representation())


def test_vampprior_matches_scvi_tools_sysvi():
    """omicverse.vi._priors.VampPrior reproduces scvi.external.sysvi._priors
    .VampPrior's log_prob / kl bit-for-bit (float32), including its two
    numerical quirks (per-dimension, un-summed mixture; an extra sqrt of the
    pseudo-input scale in log_prob) — see the VampPrior docstring."""
    scvi = pytest.importorskip("scvi")
    from scvi.external.sysvi._priors import VampPrior as RefVamp
    from omicverse.vi._priors import VampPrior as OVVamp

    torch.manual_seed(0)
    K, L, N, n_batch = 4, 5, 7, 2
    W_m, W_v = torch.randn(L, 3), torch.randn(L, 3)

    class TinyOV(torch.nn.Module):
        def forward(self, x, batch_index, cont=None, sample=False):
            oh = torch.nn.functional.one_hot(batch_index.reshape(-1).long(), n_batch).float()
            h = torch.cat([x, oh], -1)
            q_m = h @ W_m.t()
            q_v = torch.nn.functional.softplus(h @ W_v.t()) + 1e-4
            return q_m, q_v, q_m

    class TinySCVI(torch.nn.Module):
        def forward(self, x, batch_index, cat_list=None, cont=None):
            from torch.distributions import Normal
            oh = torch.nn.functional.one_hot(batch_index.reshape(-1).long(), n_batch).float()
            h = torch.cat([x, oh], -1)
            q_m = h @ W_m.t()
            q_v = torch.nn.functional.softplus(h @ W_v.t()) + 1e-4
            return {"q_dist": Normal(q_m, q_v.sqrt() + 1e-12)}

    u = torch.randn(K, 1)
    batch_idx = torch.randint(0, n_batch, (K,))
    ov_prior = OVVamp(K, TinyOV(), u, [n_batch], batch_idx, [], cont=None, trainable=True)
    ref_prior = RefVamp(K, TinySCVI(), u, [n_batch], batch_idx.unsqueeze(-1), [], cont=None,
                        trainable_priors=True)

    z = torch.randn(N, L)
    q_m = torch.randn(N, L)
    q_v = torch.nn.functional.softplus(torch.randn(N, L)) + 1e-4

    lp_ov, lp_ref = ov_prior.log_prob(z), ref_prior.log_prob(z)
    assert torch.allclose(lp_ov, lp_ref, atol=1e-5)

    from torch.distributions import Normal
    qz = Normal(q_m, q_v.sqrt() + 1e-12)
    kl_ov, kl_ref = ov_prior.kl(q_m, q_v, z), ref_prior.kl(qz, z)
    assert torch.allclose(kl_ov, kl_ref, atol=1e-5)
