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


_TORCHRUN_SCRIPT = r"""
import os, sys, numpy as np
sys.path.insert(0, {root!r})
import torch
torch.set_num_threads(1)
import omicverse as ov
from omicverse.vi import SCVI
ov.settings.cpu_init()
a = ov.read({path!r}, storage="disk")
m = SCVI(a, batch_key="batch", n_hidden=16, n_latent=4, seed=0)
m.train(max_epochs=2, block_size=400, buffer_blocks=2, progress=False, device="cpu")
z = m.get_latent_representation()
w = m.module.z_encoder.mean_encoder.weight.detach().numpy()
np.save(os.path.join({out!r}, f"w{{os.environ['RANK']}}.npy"), w)
if os.environ["RANK"] == "0":
    np.save(os.path.join({out!r}, "z.npy"), z)
    print("SOURCE", m._source().kind, len(m.history_["elbo_train"]))
"""


def test_torchrun_launch(data, tmp_path):
    import subprocess
    import sys
    import os
    path = tmp_path / "tr.h5ad"
    data[:2000].copy().write_h5ad(path)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = tmp_path / "run.py"
    script.write_text(_TORCHRUN_SCRIPT.format(root=root, path=str(path), out=str(tmp_path)))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        "--nproc_per_node=2", str(script)], capture_output=True, text=True,
                       env=env, timeout=900, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-3000:]
    assert "SOURCE h5 2" in r.stdout
    w0, w1 = np.load(tmp_path / "w0.npy"), np.load(tmp_path / "w1.npy")
    assert np.allclose(w0, w1)  # DDP keeps the ranks' weights identical
    assert np.isfinite(np.load(tmp_path / "z.npy")).all()
