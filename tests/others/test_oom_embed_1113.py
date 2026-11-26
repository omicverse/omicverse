"""Issue #1113: pacmap/localmap on AnnDataOOM, kNN reuse, effective CPUs, h5ad pre-flight."""
import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp

import omicverse as ov
from omicverse import _backend, _oom_native

aof = pytest.importorskip("anndataoom_fast")


def _pcs(n=600, d=20, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(0, 6, (4, d))
    return (c[rng.integers(0, 4, n)] + rng.normal(0, 1, (n, d))).astype(np.float32)


def _neigh_adata(k=25, n=600):
    a = ad.AnnData(np.zeros((n, 3), np.float32))
    a.obsm["X_pca"] = _pcs(n)
    ov.pp.neighbors(a, n_neighbors=k, use_rep="X_pca", n_pcs=20)
    return a


# ---- 1. OOM_NATIVE ---------------------------------------------------------
def test_native_registered():
    assert "omicverse.pp._embed_brisc.pacmap" in _oom_native.OOM_NATIVE
    assert "omicverse.pp._embed_brisc.localmap" in _oom_native.OOM_NATIVE


@pytest.mark.parametrize("name", ["pacmap", "localmap"])
def test_no_materialisation_on_oom(tmp_path, capsys, name):
    pytest.importorskip("anndataoom")
    a = ad.AnnData(sp.random(500, 200, 0.2, format="csr", dtype=np.float32, random_state=0))
    a.X.sort_indices()
    a.obsm["X_pca"] = _pcs(500)
    f = tmp_path / "a.h5ad"
    a.write_h5ad(f)
    o = ov.read(str(f), storage="disk")
    ov.pp.scale(o)
    capsys.readouterr()
    getattr(ov.pp, name)(o, n_threads=2, num_iterations=(20, 20, 30))
    out = capsys.readouterr().out
    assert "into memory" not in out
    assert o.obsm[f"X_{name}"].shape == (500, 2)


# ---- 2. kNN reuse ----------------------------------------------------------
@pytest.mark.parametrize("name", ["pacmap", "localmap"])
def test_reuse_equals_explicit(name):
    from omicverse.pp._embed_brisc import _reuse_knn
    a = _neigh_adata(k=25)
    idx, dist = _reuse_knn(a, "X_pca", 20)
    assert idx.shape == (a.n_obs, 20)
    assert (np.diff(dist, axis=1) >= -1e-4).all()
    kw = dict(n_threads=2, num_iterations=(20, 20, 30))
    ov.pp.__dict__[name](a, **kw)
    assert a.uns[name]["params"]["knn"] == "reused obsp['distances']"
    ref = getattr(aof, name)(a.obsm["X_pca"], knn_indices=idx, knn_dists=dist, **kw)
    assert np.array_equal(a.obsm[f"X_{name}"], ref)
    ov.pp.__dict__[name](a, key_added="X_x", knn_indices=idx, knn_dists=dist, **kw)
    assert np.array_equal(a.obsm["X_x"], ref)
    assert a.uns["x"]["params"]["knn"] == "given"


def test_fallback_too_few_and_opt_out_and_other_rep():
    from omicverse.pp._embed_brisc import _reuse_knn
    a = _neigh_adata(k=15)  # 14 per row < 20
    assert _reuse_knn(a, "X_pca", 20) is None
    ov.pp.pacmap(a, n_threads=2, num_iterations=(10, 10, 20))
    assert a.uns["pacmap"]["params"]["knn"] == "anndataoom_fast.knn"
    b = _neigh_adata(k=25)
    assert _reuse_knn(b, "X_other", 20) is None
    b.obsm["X_other"] = b.obsm["X_pca"]
    assert _reuse_knn(b, "X_other", 20) is None  # neighbours were on X_pca
    ov.pp.pacmap(b, reuse_neighbors=False, n_threads=2, num_iterations=(10, 10, 20))
    assert b.uns["pacmap"]["params"]["knn"] == "anndataoom_fast.knn"


def test_reuse_ragged_rows():
    from omicverse.pp._embed_brisc import _reuse_knn
    a = _neigh_adata(k=25)
    D = a.obsp["distances"].tolil()
    D.rows[3], D.data[3] = D.rows[3][:-2], D.data[3][:-2]  # row 3 has 22, rest 24
    a.obsp["distances"] = D.tocsr()
    idx, dist = _reuse_knn(a, "X_pca", 20)
    assert idx.shape == (a.n_obs, 20) and (np.diff(dist, axis=1) >= -1e-4).all()


# ---- 3. effective CPUs -----------------------------------------------------
def test_cgroup_v2_v1_and_slurm(tmp_path):
    p = tmp_path / "cpu.max"
    p.write_text("1600000 100000\n")
    assert _backend._cgroup_cpu_limit(v2_path=str(p)) == 16
    p.write_text("max 100000\n")
    none = str(tmp_path / "none")
    assert _backend._cgroup_cpu_limit(v2_path=str(p), v1_quota=none, v1_period=none) is None
    p.write_text("150000 100000\n")
    assert _backend._cgroup_cpu_limit(v2_path=str(p)) == 2
    q, per = tmp_path / "q", tmp_path / "p"
    q.write_text("400000\n"); per.write_text("100000\n")
    assert _backend._cgroup_cpu_limit(v2_path=none, v1_quota=str(q), v1_period=str(per)) == 4
    q.write_text("-1\n")
    assert _backend._cgroup_cpu_limit(v2_path=none, v1_quota=str(q), v1_period=str(per)) is None
    assert _backend._slurm_cpu_limit({"SLURM_CPUS_PER_TASK": "6"}) == 6
    assert _backend._slurm_cpu_limit({"SLURM_JOB_CPUS_PER_NODE": "8(x2),4"}) == 8
    assert _backend._slurm_cpu_limit({}) is None


def test_effective_cpu_count(tmp_path, monkeypatch):
    import os
    p = tmp_path / "cpu.max"
    none = str(tmp_path / "none")
    kw = dict(v2_path=str(p), v1_quota=none, v1_period=none)
    p.write_text("200000 100000\n")
    aff = len(os.sched_getaffinity(0))
    assert _backend.effective_cpu_count({}, **kw) == min(aff, 2)
    assert _backend.effective_cpu_count({"SLURM_CPUS_PER_TASK": "1"}, **kw) == 1
    assert _backend.effective_cpu_count({"OMP_NUM_THREADS": "7"}, **kw) == 7  # user wins
    assert _backend.default_threads(3) == 3 and _backend.default_threads(None) >= 1


# ---- 4. pre-flight ---------------------------------------------------------
def _write_csr(path, n=3000, per=10, bad_lane=None):
    rng = np.random.default_rng(0)
    idx = np.sort(rng.integers(0, 500, (n, per)), axis=1).astype(np.int32)
    if bad_lane is not None:
        idx[bad_lane, :3] = [400, 5, 6]
    X = sp.csr_matrix((np.ones(n * per, np.float32), idx.ravel(),
                       np.arange(0, n * per + 1, per)), shape=(n, 500))
    a = ad.AnnData(X)
    a.write_h5ad(path)


@pytest.mark.parametrize("lane,expected", [(None, True), (1500, False), (2990, False),
                                           (5, False)])
def test_csr_sorted_probe(tmp_path, lane, expected):
    from omicverse.io.single._read import _h5ad_csr_sorted
    f = tmp_path / "x.h5ad"
    _write_csr(f, bad_lane=lane)
    assert _h5ad_csr_sorted(f) is expected


def test_check_sorted_flag(tmp_path, monkeypatch):
    pytest.importorskip("anndataoom")
    from omicverse.io.single import _read
    f = tmp_path / "x.h5ad"
    _write_csr(f)
    calls = []
    monkeypatch.setattr(_read, "_h5ad_csr_sorted", lambda p: calls.append(p) or True)
    ov.read(str(f), storage="disk").close()
    ov.read(str(f), storage="disk", check_sorted=False).close()
    assert len(calls) == 1
