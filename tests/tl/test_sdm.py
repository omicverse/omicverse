"""Tests for omicverse.tl._sdm (species-set prediction) on synthetic landscapes."""
import numpy as np
import pytest
from scipy import sparse

from omicverse.tl import _sdm as sdm


def _landscape(n=900, S=40, seed=0, n_groups=6):
    """Sites on a lon/lat rectangle; species respond to two smooth environmental gradients."""
    rng = np.random.default_rng(seed)
    lat = rng.uniform(44, 52, n)
    lon = rng.uniform(0, 16, n)
    env = np.c_[np.sin(lat / 2) + .1 * rng.normal(size=n), np.cos(lon / 3) + .1 * rng.normal(size=n)]
    opt = rng.uniform(-1, 1, (S, 2))
    width = rng.uniform(.3, .7, S)
    d2 = ((env[:, None, :] - opt[None]) ** 2).sum(2)
    p = 0.9 * np.exp(-d2 / (2 * width[None] ** 2))
    Y = (rng.random((n, S)) < p).astype(np.float32)
    Y[Y.sum(1) == 0, 0] = 1
    groups = np.minimum((lon / (16 / n_groups)).astype(int), n_groups - 1)
    return lat, lon, env, sparse.csr_matrix(Y), p, groups


def test_sample_f1_exact():
    truth = sparse.csr_matrix(np.array([[1, 1, 0, 0], [0, 0, 1, 0]]))
    ranking = np.array([[0, 2, 1, 3], [2, 0, 1, 3]])
    f = sdm.sdm_sample_f1(ranking, np.array([2, 1]), truth)
    assert np.allclose(f, [2 * 1 / 4, 1.0])
    # richness override counts species outside the columns
    f2 = sdm.sdm_sample_f1(ranking, np.array([2, 1]), truth, richness=np.array([3, 1]))
    assert np.allclose(f2, [2 / 5, 1.0])


def test_decode_adapts_set_size_and_beats_fixed_k():
    rng = np.random.default_rng(1)
    n, S = 400, 60
    # heterogeneous richness: half the sites species-poor, half species-rich
    P = np.where(np.arange(n)[:, None] < n // 2, rng.uniform(0, .08, (n, S)), rng.uniform(0, .6, (n, S)))
    Y = sparse.csr_matrix((rng.random((n, S)) < P).astype(float))
    r, s = sdm.sdm_decode_sets(P, kmax=S)
    assert s[: n // 2].mean() < s[n // 2:].mean()
    f_plug = sdm.sdm_sample_f1(r, s, Y).mean()
    best_fixed = max(sdm.sdm_sample_f1(r, np.full(n, k), Y).mean() for k in range(1, S + 1))
    assert f_plug >= best_fixed - 0.01


def test_sets_to_strings_sorted_unique():
    out = sdm.sdm_sets_to_strings(np.array([[2, 0, 1], [1, 2, 0]]), np.array([2, 0]), [30, 10, 20])
    assert out == ["20 30", ""]


def test_spatial_holdout_cells_groups_blocks():
    lat, lon, _, _, _, groups = _landscape()
    xyz = sdm._xyz(lat, lon)
    fit, val, w = sdm.sdm_spatial_holdout(lat, lon, groups=groups, scheme="cells", cell_km=20,
                                          fraction=.1, buffer_km=5)
    assert len(np.intersect1d(fit, val)) == 0 and len(w) == len(val)
    d = np.min(np.linalg.norm(xyz[fit][:, None] - xyz[val][None], axis=2), 1)
    assert d.min() > 5
    fit, val, w = sdm.sdm_spatial_holdout(lat, lon, groups=groups, scheme="groups", fraction=.2,
                                          min_group=10, buffer_km=10)
    held = np.unique(groups[val])
    assert not np.isin(groups[fit], held).any()          # whole groups held out
    assert np.isclose(w.mean(), 1)
    fit, val, _ = sdm.sdm_spatial_holdout(lat, lon, scheme="blocks", block_km=200, fraction=.2)
    assert len(val) > 0 and len(fit) > 0


def test_neighbor_composition_geo_and_cosine():
    Y = sparse.csr_matrix(np.eye(3))
    ref = (np.array([45., 46., 47.]), np.array([5., 5., 5.]))
    P = sdm.sdm_neighbor_composition(Y, ref, (np.array([46.001]), np.array([5.])), metric="geographic", k=3)
    assert P.argmax() == 1 and np.isclose(P.sum(), 1)
    E = np.eye(3) + 0.01
    P = sdm.sdm_neighbor_composition(Y, E, E[[2]], metric="cosine", k=3, tau=.05)
    assert P.argmax() == 2


def test_presence_only_histogram_rows_normalised():
    rng = np.random.default_rng(0)
    H, dist = sdm.sdm_presence_only_histogram(
        np.array([45., 50.]), np.array([5., 10.]), rng.uniform(44, 51, 500), rng.uniform(4, 11, 500),
        np.r_[rng.integers(0, 7, 499), -1], 7, k=50)
    assert H.shape == (2, 7)
    assert np.allclose(np.asarray(H.sum(1)).ravel(), 1)
    assert dist.shape == (2, 3) and (np.diff(dist, axis=1) >= 0).all()


def test_cube_summary_and_reader(tmp_path):
    cube = np.random.default_rng(0).normal(size=(5, 3, 4, 8)).astype(np.float32)
    miss = np.array([0, 0, 1, 0, 0], bool)
    F = sdm.sdm_cube_summary(cube, time_axis=-1, missing=miss)
    assert F.shape == (5, 3 * 3 * 4)
    assert np.isnan(F[2]).all() and np.isfinite(F[0]).all()
    assert np.allclose(F[0, :12], cube[0].mean(-1).ravel(), atol=1e-6)
    ids = [101, 102, 103]
    np.save(tmp_path / "101.npy", np.ones((2, 2)))
    np.save(tmp_path / "103_cube.npy", 3 * np.ones((2, 2)))
    arr, m = sdm.sdm_read_tensor_dir(tmp_path, ids, log=None)
    assert arr.shape == (3, 2, 2) and m.tolist() == [False, True, False] and arr[2, 0, 0] == 3


def test_standardize_missing_flags():
    a = np.array([[1., np.nan], [3., 4.], [5., 6.]])
    X = sdm.sdm_standardize([a, np.arange(3.)], fit_rows=[0, 1, 2])
    assert X.shape == (3, 2 + 1 + 1)
    assert X[0, 2] == 1 and X[1, 2] == 0 and np.isfinite(X).all()


def test_fuse_keeps_mass():
    rng = np.random.default_rng(0)
    A = rng.uniform(0, .5, (4, 10))
    B = rng.uniform(0, .5, (4, 10))
    F = sdm.sdm_fuse(A, [B], [0.3])
    assert np.allclose(F.sum(1), A.sum(1), rtol=1e-4)
    assert np.allclose(sdm.sdm_fuse(A, [B], [0.0]), A)


def test_pipeline_learns_niches_on_synthetic_landscape():
    pytest.importorskip("torch")
    lat, lon, env, Y, p, groups = _landscape(n=1100, S=40)
    n_tr = 900
    rng = np.random.default_rng(3)
    # presence-only records drawn from the true intensity (biased observers ignored here)
    po_site = rng.integers(0, len(lat), 6000)
    po_sp = np.array([rng.choice(p.shape[1], p=pp / pp.sum()) for pp in p[po_site]])
    res = sdm.sdm_community_pipeline(
        lat[:n_tr], lon[:n_tr], Y[:n_tr], lat[n_tr:], lon[n_tr:],
        covariates=[env], embedding=env + 0.05 * rng.normal(size=env.shape), groups=groups[:n_tr],
        po=(lat[po_site] + .01, lon[po_site], po_sp), species_ids=np.arange(100, 140),
        epochs=6, hidden=64, final_seeds=(0,), po_svd_dim=8, po_k=64, max_val=200,
        near_cell_km=30, near_fraction=.15, far_fraction=.2, n_threads=1, log=None)
    f = sdm.sdm_sample_f1(res["ranking"], res["sizes"], Y[n_tr:]).mean()
    # baseline: the globally most frequent species, best fixed k
    prev = np.asarray(Y[:n_tr].mean(0)).ravel()
    r0 = np.tile(np.argsort(-prev), (len(lat) - n_tr, 1))
    f0 = max(sdm.sdm_sample_f1(r0, np.full(len(r0), k), Y[n_tr:]).mean() for k in range(1, 41))
    assert f > f0 + 0.03, (f, f0)
    assert len(res["predictions"]) == len(lat) - n_tr
    s0 = res["predictions"][0].split()
    assert s0 == sorted(s0, key=int) and all(100 <= int(v) < 140 for v in s0)
    assert "near" in res["decisions"] and "report" in res


def test_group_holdout_folds_are_disjoint_with_skip_groups():
    lat, lon, _, _, _, groups = _landscape(n_groups=8)
    _, v1, _ = sdm.sdm_spatial_holdout(lat, lon, groups=groups, scheme="groups", fraction=.12,
                                       min_group=10, seed=0)
    g1 = set(groups[v1].astype(str))
    _, v2, _ = sdm.sdm_spatial_holdout(lat, lon, groups=groups, scheme="groups", fraction=.12,
                                       min_group=10, seed=0, skip_groups=sorted(g1))
    assert not g1 & set(groups[v2].astype(str))
