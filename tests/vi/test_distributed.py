"""ov.vi multi-process training (DDP, gloo on CPU processes)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

if not dist.is_available():
    pytest.skip("torch.distributed unavailable", allow_module_level=True)

from .test_data import _KEEP, _counts  # noqa: E402


def _ari(z, labels):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    k = len(np.unique(labels))
    pred = KMeans(k, n_init=5, random_state=0).fit_predict(z)
    return adjusted_rand_score(labels, pred)


@pytest.fixture(scope="module")
def data():
    return _counts(n=4000, g=200, seed=4)


def test_ddp_matches_single_process_quality(data):
    from omicverse.vi import SCVI
    torch.set_num_threads(4)
    single = SCVI(data, batch_key="batch", n_hidden=32, n_latent=5, seed=0)
    single.train(max_epochs=30, device="cpu", batch_size=256, progress=False, seed=0)
    ddp = SCVI(data, batch_key="batch", n_hidden=32, n_latent=5, seed=0)
    ddp.train(max_epochs=30, devices="cpu:2", batch_size=128, progress=False, seed=0,
              threads_per_process=2, latent_key="X_ddp")
    assert ddp.is_trained_ and ddp._world_size == 2
    z1 = single.get_latent_representation()
    z2 = ddp.get_latent_representation()
    # rank-0 gathered latent equals the parent's inference with the trained weights
    assert np.allclose(data.obsm["X_ddp"], z2, atol=1e-5)
    e1 = single.get_elbo(indices=single.validation_indices)
    e2 = ddp.get_elbo(indices=ddp.validation_indices)
    assert abs(e1 - e2) < 0.02 * abs(e1), (e1, e2)
    labels = data.obs["label"].values
    a1, a2 = _ari(z1, labels), _ari(z2, labels)
    assert a2 > a1 - 0.1, (a1, a2)


def test_ddp_uneven_blocks_no_deadlock(data, tmp_path):
    import omicverse as ov
    from omicverse.vi import SCVI
    path = tmp_path / "uneven.h5ad"
    data[:3333].copy().write_h5ad(path)  # 3333 cells -> uneven blocks per rank
    ov.settings.cpu_init()
    a = ov.read(str(path), storage="disk")
    _KEEP.append(a)
    m = SCVI(a, batch_key="batch", n_hidden=16, n_latent=4, seed=0)
    m.train(max_epochs=3, devices="cpu:3", block_size=500, buffer_blocks=2, batch_size=128,
            progress=False, threads_per_process=1, early_stopping=True,
            early_stopping_patience=5)
    assert len(m.history_["elbo_train"]) == 3
    assert "elbo_validation" in m.history_
    assert np.isfinite(m.get_latent_representation()).all()
