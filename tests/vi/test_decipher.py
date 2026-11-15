"""ov.vi.Decipher: two-level VAE with an interpretable 2-D latent ``v``.

Fast CPU-only checks on a synthetic 1-D differentiation trajectory (continuous
pseudotime ``obs['time']`` plus a discretised ``obs['stage']``): the model
trains, its Decipher space recovers the trajectory (Spearman vs. the known
pseudotime), the trajectory / rotation utilities work, and in-memory,
AnnDataOOM and Dask-backed AnnData give identical results for a fixed seed.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")
ad = pytest.importorskip("anndata")

from .test_data import _KEEP  # noqa: E402


def _traj_counts(n=1500, g=150, seed=0):
    """Cells spread along a 1-D pseudotime ``t`` in [0, 1); gene log-means vary
    smoothly (linearly + quadratically) with ``t``, giving a continuous
    trajectory with a known ground-truth ordering."""
    rs = np.random.default_rng(seed)
    t = rs.random(n)
    a_coef = rs.normal(0, 1.3, g)
    b_coef = rs.normal(0, 0.8, g)
    base = np.log(rs.gamma(2.0, 1.0, g) + 0.1)
    log_mean = base[None, :] + a_coef[None, :] * t[:, None] + b_coef[None, :] * (t[:, None] ** 2)
    prop = np.exp(log_mean - log_mean.max(1, keepdims=True))
    prop /= prop.sum(1, keepdims=True)
    lib = rs.lognormal(7.3, 0.3, n)
    X = sp.csr_matrix(rs.poisson(prop * lib[:, None]).astype(np.float32))
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"G{j}" for j in range(g)]
    a.layers["counts"] = a.X.copy()
    a.obs["time"] = t
    a.obs["stage"] = pd.Categorical(pd.cut(t, bins=[-0.01, 0.33, 0.66, 1.01],
                                           labels=["early", "mid", "late"]))
    return a


def _train(adata, **kw):
    from omicverse.vi import Decipher
    m = Decipher(adata, layer=kw.pop("layer", "counts"), genes=kw.pop("genes", None),
                dim_v=2, dim_z=8, seed=0)
    m.train(max_epochs=kw.pop("max_epochs", 60), batch_size=kw.pop("batch_size", 128),
            device="cpu", progress=False, train_size=kw.pop("train_size", None), seed=0, **kw)
    return m


def test_decipher_trains_and_recovers_trajectory(tmp_path):
    from scipy.stats import spearmanr
    torch.set_num_threads(4)
    a = _traj_counts()
    m = _train(a)
    assert m.is_trained_
    # loss improves over training
    elbo = m.history_["elbo_train"]
    assert elbo[-1] < elbo[0]
    v = m.get_latent_representation()
    z = m.get_latent_representation(give_z=True)
    assert v.shape == (a.n_obs, 2) and z.shape == (a.n_obs, 8)
    assert np.isfinite(v).all() and np.isfinite(z).all()
    # at least one Decipher-space axis tracks the ground-truth pseudotime
    rho = max(abs(spearmanr(v[:, k], a.obs["time"]).statistic) for k in range(2))
    assert rho > 0.6, f"best |spearman| with true pseudotime = {rho:.3f}"
    nll = m.get_marginal_ll(n_samples=10)
    assert np.isfinite(nll)
    # save / load round trip
    m.save(str(tmp_path / "d"))
    m2 = type(m).load(str(tmp_path / "d"), adata=a, device="cpu")
    assert np.allclose(m2.get_latent_representation(), v, atol=1e-5)


def test_rotate_components_and_decipher_time():
    from omicverse.vi import Trajectory, rotate_decipher_components, compute_decipher_time
    from scipy.stats import spearmanr
    a = _traj_counts(seed=1)
    m = _train(a, max_epochs=80)
    a.obsm["decipher_v"] = m.get_latent_representation()
    a.obsm["decipher_z"] = m.get_latent_representation(give_z=True)
    v, z, rot = rotate_decipher_components(
        a, "decipher_v", "decipher_z", v1_obs_col="stage",
        v1_order=["early", "mid", "late"])
    assert v.shape == a.obsm["decipher_v"].shape and rot.shape == (2, 2)
    assert np.allclose(np.linalg.det(rot) ** 2, 1.0, atol=1e-6)  # rotation/reflection
    a.obsm["decipher_v_rot"] = v
    traj = Trajectory.from_clusters(a, "decipher_v_rot", "stage", ["early", "mid", "late"])
    assert traj.trajectory_latent.shape[1] == 2 and traj.n_points > 0
    dt = compute_decipher_time(a, "stage", traj)
    ok = np.isfinite(dt)
    assert ok.all()  # every cell falls in one of the trajectory's clusters
    rho = spearmanr(dt[ok], a.obs["time"].values[ok]).statistic
    assert abs(rho) > 0.5, f"decipher time vs. true pseudotime rho = {rho:.3f}"


def test_compute_imputed_expression_and_gene_patterns():
    from omicverse.vi import Trajectory
    a = _traj_counts(n=600, g=80, seed=2)
    m = _train(a, max_epochs=30)
    imputed = m.compute_imputed_gene_expression()
    assert imputed.shape == (a.n_obs, 80) and (imputed >= 0).all()
    a.obsm["decipher_v"] = m.get_latent_representation()
    a.obsm["decipher_z"] = m.get_latent_representation(give_z=True)
    cov_v, cov_z = m.compute_imputed_gene_expression(
        compute_covariances=True, v_obsm_key="decipher_v", z_obsm_key="decipher_z",
        return_imputed=False)
    assert cov_v.shape == (80, 2) and cov_z.shape == (80, 8) and np.isfinite(cov_v).all()
    traj = Trajectory.from_clusters(a, "decipher_v", "stage", ["early", "mid", "late"])
    pat = m.compute_gene_patterns(trajectory=traj, n_samples=20)
    npts = len(pat["times"])
    assert pat["mean"].shape == (npts, 80)
    assert pat["q25"].shape == pat["q75"].shape == (npts, 80)
    assert (pat["q25"] <= pat["q75"] + 1e-3).all()  # quantiles are ordered
    assert np.isfinite(pat["mean"]).all() and (pat["mean"] >= 0).all()


def test_compute_gene_patterns_rotation_undo():
    """Regression for a real trap hit in practice: overwriting the obsm key used
    to build a ``Trajectory`` with the *rotated* Decipher space (from
    ``rotate_decipher_components``, meant for plotting) and feeding that
    trajectory straight into ``compute_gene_patterns`` silently decodes
    meaningless coordinates, because ``decoder_v_to_z`` only understands the
    native (un-rotated) ``v``. ``compute_gene_patterns(..., rotation=rot)``
    must undo the rotation and recover the native-space result exactly."""
    from omicverse.vi import Trajectory, rotate_decipher_components
    a = _traj_counts(n=600, g=80, seed=2)
    m = _train(a, max_epochs=30)
    a.obsm["decipher_v"] = m.get_latent_representation()
    a.obsm["decipher_z"] = m.get_latent_representation(give_z=True)
    stages = ["early", "mid", "late"]

    traj_native = Trajectory.from_clusters(a, "decipher_v", "stage", stages)
    pat_native = m.compute_gene_patterns(trajectory=traj_native, n_samples=5)

    v_rot, z_rot, rot = rotate_decipher_components(
        a, "decipher_v", "decipher_z", v1_obs_col="stage", v1_order=stages)
    assert not np.allclose(rot, np.eye(2), atol=1e-2), "test needs a non-trivial rotation"
    a.obsm["decipher_v_rot"] = v_rot
    traj_rot = Trajectory.from_clusters(a, "decipher_v_rot", "stage", stages)

    # decoding the rotated trajectory directly (no `rotation=`) is the bug: it
    # feeds the decoder coordinates it never saw in training, and must not
    # match the native-space pattern.
    pat_wrong = m.compute_gene_patterns(trajectory=traj_rot, n_samples=5)
    assert not np.allclose(pat_wrong["mean"], pat_native["mean"], atol=1.0)

    # passing `rotation=rot` undoes it and exactly recovers the native-space
    # trajectory's decoded pattern (same points, same z_mean, deterministic).
    pat_fixed = m.compute_gene_patterns(trajectory=traj_rot, n_samples=5, rotation=rot)
    np.testing.assert_allclose(pat_fixed["mean"], pat_native["mean"], atol=1e-2, rtol=1e-4)


def test_decipher_oom_and_dask_match_in_memory(tmp_path):
    import omicverse as ov
    a = _traj_counts(n=800, g=100, seed=4)
    path = tmp_path / "traj.h5ad"
    a.write_h5ad(path)
    # layer=None (.X) throughout (dask-backed AnnData only exposes X, not layers);
    # the same explicit block shuffle on every source so minibatches line up exactly
    bkw = dict(shuffle="block", block_size=200, buffer_blocks=4)
    mem = ov.read(str(path))
    z_mem = _train(mem, layer=None, max_epochs=10, **bkw).get_latent_representation()

    ov.settings.cpu_init()
    oom = ov.read(str(path), storage="disk")
    _KEEP.append(oom)
    m_oom = _train(oom, layer=None, max_epochs=10, **bkw)
    assert m_oom._source().kind == "h5"
    assert np.allclose(z_mem, m_oom.get_latent_representation(), atol=1e-5)

    pytest.importorskip("dask")
    from omicverse.pp._dask import read_h5ad
    da_ad = read_h5ad(str(path), chunk_size=200, device="cpu")
    m_da = _train(da_ad, layer=None, max_epochs=10, **bkw)
    assert m_da._source().kind == "dask"
    assert np.allclose(z_mem, m_da.get_latent_representation(), atol=1e-5)


def test_single_decipher_wrapper():
    import omicverse as ov
    a = _traj_counts(n=500, g=60, seed=5)
    model = ov.single.decipher(a, layer="counts",
                               model_kwargs={"dim_v": 2, "dim_z": 6, "seed": 0},
                               train_kwargs={"max_epochs": 10, "device": "cpu",
                                             "progress": False})
    assert "X_decipher" in a.obsm and a.obsm["X_decipher"].shape == (500, 2)
    assert model.is_trained_
    with pytest.raises(KeyError):
        ov.single.decipher(a, layer="counts",
                           train_kwargs={"max_epochs": 1, "device": "cpu", "progress": False})
    model2 = ov.single.decipher(a, layer="counts", key_added_z="X_decipher_z", overwrite=True,
                                model_kwargs={"seed": 0},
                                train_kwargs={"max_epochs": 10, "device": "cpu",
                                              "progress": False})
    assert "X_decipher_z" in a.obsm and model2.is_trained_
