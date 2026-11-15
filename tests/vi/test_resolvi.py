"""ov.vi.RESOLVI / ov.space.resolvi: fast CPU checks on synthetic and a real-data
subsample (Xenium breast cancer, 10x Genomics; see benchmarks/vi/resolvi/prep_data.py)."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ad = pytest.importorskip("anndata")

_KEEP = []  # anndataoom closes the file when the parent object is collected
_XENIUM = "/scratch/users/steorra/tmp/vi_p2/data/resolvi/xenium_breast_r1_crop.h5ad"


def _spatial(n=900, g=60, k=3, seed=0, n_batch=1):
    """Cells of ``k`` types clustered in space (so spatial diffusion mixes
    neighbouring types) with type-specific Poisson expression."""
    rs = np.random.default_rng(seed)
    labels = rs.integers(0, k, n)
    centers = rs.normal(scale=12.0, size=(k, 2))
    coords = centers[labels] + rs.normal(scale=1.5, size=(n, 2))
    base = rs.gamma(0.6, 1.0, (k, g))
    X = rs.poisson(base[labels] * 6).astype(np.float32)
    a = ad.AnnData(X)
    a.var_names = [f"g{i}" for i in range(g)]
    a.layers["counts"] = a.X.copy()
    a.obs["cell_type"] = labels.astype(str)
    a.obs["slide"] = np.where(np.arange(n) % n_batch == 0, "s0", "s1") if n_batch > 1 else "s0"
    a.obsm["spatial"] = coords
    return a, labels


def test_resolvi_trains_and_outputs(tmp_path):
    from omicverse.vi import RESOLVI
    torch.set_num_threads(4)
    a, labels = _spatial(n_batch=2)
    m = RESOLVI(a, layer="counts", spatial_key="spatial", batch_key="slide", n_neighbors=8,
               mixture_k=5, n_hidden=16, n_hidden_encoder=32, n_layers=1, n_particles=1)
    assert m.module.full_counts.shape == (a.n_obs, a.n_vars)
    m.train(max_epochs=10, device="cpu", progress=False, batch_size=128, seed=0)
    assert np.isfinite(m.get_elbo())

    z = m.get_latent_representation()
    assert z.shape == (a.n_obs, 10)
    props = m.get_proportions()
    assert props.shape == (a.n_obs, 3)
    assert ((props.to_numpy() >= 0) & (props.to_numpy() <= 1.2)).all()  # +diffusion_eps slack

    denoised = m.get_normalized_expression()
    assert denoised.shape == (a.n_obs, a.n_vars) and (denoised.to_numpy() >= 0).all()
    corrected = m.get_corrected_counts()
    assert corrected.shape == denoised.shape

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels, KMeans(3, n_init=5, random_state=0).fit_predict(z))
    assert ari > 0.5  # z should separate the 3 spatially-clustered cell types

    m.save(str(tmp_path / "m"))
    m2 = RESOLVI.load(str(tmp_path / "m"), adata=a, device="cpu")
    assert np.allclose(m.get_latent_representation(), m2.get_latent_representation(), atol=1e-5)


def test_resolvi_semisupervised_predicts_unlabeled():
    from omicverse.vi import RESOLVI
    torch.set_num_threads(4)
    a, labels = _spatial(n=700, seed=2)
    truth = labels.astype(str)
    lab = truth.copy()
    rs = np.random.default_rng(0)
    lab[rs.random(len(lab)) < 0.6] = "Unknown"
    a.obs["lab"] = lab
    m = RESOLVI(a, layer="counts", spatial_key="spatial", labels_key="lab",
               unlabeled_category="Unknown", n_neighbors=8, mixture_k=5, n_hidden=16,
               n_hidden_encoder=32, n_layers=1, n_particles=1)
    assert m.semisupervised_ and m.module.n_labels == 3
    m.train(max_epochs=10, device="cpu", progress=False, batch_size=128, seed=0)
    unl = lab == "Unknown"
    pred = m.predict()
    assert (pred[unl] == truth[unl]).mean() > 0.8
    soft = m.predict(soft=True, indices=np.arange(10))
    assert soft.shape == (10, 3)


def test_resolvi_rejects_bad_n_neighbors():
    from omicverse.vi import RESOLVI
    a, _ = _spatial(n=100)
    with pytest.raises(ValueError, match="n_neighbors"):
        RESOLVI(a, spatial_key="spatial", layer="counts", n_neighbors=2)


def test_resolvi_disk_backed_trains(tmp_path):
    """AnnDataOOM input: the model streams minibatches from disk (not loaded
    into memory as a whole) and still trains end to end."""
    import omicverse as ov
    from omicverse.vi import RESOLVI
    torch.set_num_threads(4)
    a, _ = _spatial(n=500, g=40, seed=3, n_batch=2)
    path = tmp_path / "spatial.h5ad"
    a.write_h5ad(path)
    ov.settings.cpu_init()
    oom = ov.read(str(path), storage="disk")
    _KEEP.append(oom)
    m = RESOLVI(oom, layer="counts", spatial_key="spatial", batch_key="slide", n_neighbors=8,
               mixture_k=5, n_hidden=16, n_hidden_encoder=32, n_layers=1, n_particles=1)
    m.train(max_epochs=4, device="cpu", progress=False, batch_size=64, block_size=128)
    assert m._source().kind in ("oom", "h5")
    z = m.get_latent_representation()
    assert z.shape == (oom.n_obs, 10) and np.isfinite(z).all()


@pytest.mark.skipif(not os.path.exists(_XENIUM), reason="real Xenium benchmark data not staged")
def test_resolvi_real_xenium_subsample():
    """A subsample of the real 10x Xenium breast-cancer panel (Janesick et al.
    2023; benchmarks/vi/resolvi/prep_data.py) — not synthetic data."""
    from omicverse.vi import RESOLVI
    torch.set_num_threads(4)
    full = ad.read_h5ad(_XENIUM)
    rs = np.random.default_rng(0)
    rows = rs.choice(full.n_obs, 1500, replace=False)
    a = full[rows].copy()
    a = a[:, np.asarray(a.layers["counts"].sum(0)).ravel().argsort()[-80:]].copy()
    keep = np.asarray(a.layers["counts"].sum(1)).ravel() >= 10
    a = a[keep].copy()

    m = RESOLVI(a, layer="counts", spatial_key="spatial", n_neighbors=8, mixture_k=8,
               n_hidden=16, n_hidden_encoder=32, n_layers=1, n_particles=1)
    m.train(max_epochs=8, device="cpu", progress=False, batch_size=128, seed=0)
    assert np.isfinite(m.get_elbo())
    z = m.get_latent_representation()
    assert z.shape[0] == a.n_obs and np.isfinite(z).all()
    props = m.get_proportions()
    assert np.isfinite(props.to_numpy()).all()


def test_space_resolvi_protects_spatial_keys():
    from omicverse.space._resolvi import resolvi
    a, _ = _spatial(n=120, g=30)
    with pytest.raises(ValueError, match="spatial"):
        resolvi(a, layer="counts", spatial_key="spatial", key_added="spatial")


def test_space_resolvi_writes_outputs():
    import omicverse.space as sp
    a, _ = _spatial(n=200, g=40, seed=5)
    m = sp.resolvi(a, layer="counts", spatial_key="spatial", n_neighbors=8,
                  model_kwargs=dict(mixture_k=5, n_hidden=16, n_hidden_encoder=32, n_layers=1, n_particles=1),
                  train_kwargs=dict(max_epochs=5, device="cpu", progress=False, batch_size=64))
    assert a.obsm["X_resolvi"].shape == (a.n_obs, 10)
    assert a.obsm["resolvi_proportions"].shape == (a.n_obs, 3)
    assert "spatial" in a.obsm and a.obsm["spatial"].shape == (a.n_obs, 2)  # untouched
    with pytest.raises(ValueError, match="overwrite"):
        sp.resolvi(a, layer="counts", spatial_key="spatial",
                  train_kwargs=dict(max_epochs=1, device="cpu", progress=False))
