"""Synthetic tests for the open-set re-identification helpers in omicverse.tl._reid."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from omicverse.tl._reid import (reid_candidate_pairs, reid_fuse_similarity,
                                reid_open_set_cluster, reid_pair_auc,
                                reid_select_threshold, reid_view_center)


def _population(n_ids, per_id, dim=64, noise=0.35, seed=0):
    """Unit-norm embeddings: one random centre per identity plus noise."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_ids, dim))
    counts = rng.integers(1, per_id + 1, size=n_ids) if per_id > 1 else np.ones(n_ids, int)
    y = np.repeat(np.arange(n_ids), counts)
    x = centres[y] / np.sqrt(dim) + rng.normal(scale=noise / np.sqrt(dim), size=(len(y), dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, y


def test_select_threshold_recovers_partition():
    x, y = _population(60, 6, seed=1)
    sel = reid_select_threshold(x @ x.T, y, n_rep=3)
    from scipy.cluster.hierarchy import fcluster
    from omicverse.tl._reid import _linkage
    lab = fcluster(_linkage(x @ x.T, "average"), t=1 - sel["threshold"], criterion="distance")
    assert sel["ari"] > 0.9
    assert adjusted_rand_score(y, lab) > 0.9
    assert {"threshold", "ari", "abundance_error"} <= set(sel["table"].columns)


def test_open_set_with_gallery_attaches_known_and_clusters_new():
    x, y = _population(80, 6, seed=2)
    rng = np.random.default_rng(0)
    known = set(range(40))
    g_idx, q_idx = [], []
    for i in range(80):
        idx = rng.permutation(np.flatnonzero(y == i))
        if i in known and len(idx) > 1:
            g_idx += list(idx[: len(idx) // 2])
            q_idx += list(idx[len(idx) // 2:])
        elif i in known:
            g_idx += list(idx)
        else:
            q_idx += list(idx)
    q_idx, g_idx = np.array(q_idx), np.array(g_idx)
    xq, xg = x[q_idx], x[g_idx]
    res = reid_open_set_cluster(xq @ xq.T, sim_qg=xq @ xg.T, gallery_labels=y[g_idx],
                                sim_gg=xg @ xg.T, n_rep=2, log=None)
    assert res["mode"] == "reference-simulation"
    assert adjusted_rand_score(y[q_idx], res["labels"]) > 0.85
    assert res["n_attached"] > 0
    k_true = len(np.unique(y[q_idx]))
    assert abs(res["n_clusters"] - k_true) / k_true < 0.25


def test_open_set_without_reference_uses_verified_pairs():
    x, y = _population(50, 4, noise=0.5, seed=3)
    s = x @ x.T
    pairs = reid_candidate_pairs(s, k=5)
    # an (imperfect) verifier: true pairs accepted with prob 0.8, false never
    rng = np.random.default_rng(1)
    same = y[pairs["first"]] == y[pairs["second"]]
    pairs["accepted"] = same & (rng.random(len(pairs)) < 0.8)
    res = reid_open_set_cluster(s, verified=pairs, log=None)
    assert res["mode"] == "verified-pairs"
    assert adjusted_rand_score(y, res["labels"]) > 0.8


def test_hdbscan_and_nn_mixture_paths_run():
    x, y = _population(40, 3, seed=4)
    s = x @ x.T
    r1 = reid_open_set_cluster(s, method="hdbscan", log=None)
    r2 = reid_open_set_cluster(s, method="agglomerative", log=None)
    r3 = reid_open_set_cluster(s, log=None)
    assert len(r1["labels"]) == len(y) and len(r2["labels"]) == len(y)
    assert r2["mode"] == "nn-mixture" and r3["mode"] == "hdbscan"
    assert list(r3["labels"]) == list(r1["labels"])
    assert adjusted_rand_score(y, r2["labels"]) > 0.6
    assert all(isinstance(v, str) and v for v in r1["labels"])


def test_candidate_pairs_rank_and_reciprocity():
    s = np.array([[1, .9, .1, .2], [.9, 1, .3, .1], [.1, .3, 1, .8], [.2, .1, .8, 1.]])
    p = reid_candidate_pairs(s, k=1)
    assert set(zip(p["first"], p["second"])) == {(0, 1), (2, 3)}
    assert p["reciprocal"].all()


def test_fuse_and_auc():
    x, y = _population(30, 4, seed=5)
    rng = np.random.default_rng(2)
    noise = rng.normal(size=(len(y), 64))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    good, bad = x @ x.T, noise @ noise.T
    a_good, a_bad = reid_pair_auc(good, y), reid_pair_auc(bad, y)
    assert a_good > 0.95 and abs(a_bad - 0.5) < 0.1
    fused = reid_fuse_similarity([good, bad], weights=[a_good - .5, max(a_bad - .5, 0)])
    assert reid_pair_auc(fused, y) > 0.95
    assert fused.shape == good.shape


def test_view_center_removes_group_shift():
    x, y = _population(30, 4, seed=6)
    view = np.arange(len(y)) % 2
    shift = np.zeros(x.shape[1])
    shift[0] = 2.0
    xs = x + view[:, None] * shift
    xs /= np.linalg.norm(xs, axis=1, keepdims=True)
    xc = reid_view_center(xs, view, strength=1.0)
    assert reid_pair_auc(xc @ xc.T, y) > reid_pair_auc(xs @ xs.T, y)


def test_local_verify_separates_same_pattern(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from PIL import Image
    from omicverse.tl._reid import reid_local_verify
    rng = np.random.default_rng(0)
    paths, ident = [], []
    for i in range(6):
        base = np.full((240, 320), 128, np.uint8)
        for _ in range(120):
            c = (int(rng.integers(10, 310)), int(rng.integers(10, 230)))
            cv2.circle(base, c, int(rng.integers(2, 7)), int(rng.integers(0, 255)), -1)
        for v in range(2):
            ang = float(rng.uniform(-15, 15))
            M = cv2.getRotationMatrix2D((160, 120), ang, float(rng.uniform(0.9, 1.1)))
            im = cv2.warpAffine(base, M, (320, 240), borderValue=128)
            p = tmp_path / f"{i}_{v}.png"
            Image.fromarray(im).save(p)
            paths.append(str(p))
            ident.append(i)
    ident = np.array(ident)
    iu = np.triu_indices(len(paths), 1)
    pairs = pd.DataFrame({"first": iu[0], "second": iu[1]})
    ver = reid_local_verify(paths, pairs, n_null=0, n_jobs=2, log=None, min_inliers=40)
    same = ident[ver["first"]] == ident[ver["second"]]
    assert ver.loc[same, "inliers"].min() > ver.loc[~same, "inliers"].max()
    assert (ver["accepted"] == same).all()


def test_fit_projection_improves_heldout_and_crossfits():
    from omicverse.tl._reid import reid_fit_projection
    rng = np.random.default_rng(7)
    # identity signal lives in 8 dims, 120 nuisance dims of larger variance
    n_ids, per = 60, 6
    y = np.repeat(np.arange(n_ids), per)
    sig = rng.normal(size=(n_ids, 8))[y] + 0.3 * rng.normal(size=(len(y), 8))
    x = np.hstack([sig, 1.0 * rng.normal(size=(len(y), 120))])
    pj = reid_fit_projection(x, y, n_pca=128)
    assert pj["heldout_auc_proj"] > pj["heldout_auc_raw"] + 0.05
    oof = pj["reference_oof"]
    assert oof.shape[0] == len(y)
    # cross-fitted coordinates must not be (near-)perfectly separable in-sample
    from omicverse.tl._reid import reid_pair_auc
    p_in = pj["transform"](x)
    assert reid_pair_auc(oof @ oof.T, y) < reid_pair_auc(p_in @ p_in.T, y)


def test_embed_with_workers_and_cache(tmp_path):
    torch = pytest.importorskip("torch")
    from PIL import Image
    from omicverse.tl._reid import reid_embed
    paths = []
    rng = np.random.default_rng(0)
    for i in range(10):
        p = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 255, (20, 30, 3), dtype=np.uint8)).save(p)
        paths.append(p.name)

    class Net(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(2,)).flatten(1)

    def tf(im):
        return torch.tensor(np.asarray(im.resize((8, 8)), dtype=np.float32)).permute(2, 0, 1)

    e1 = reid_embed(paths, Net(), tf, image_root=str(tmp_path), n_workers=1, batch_size=3, log=None)
    e3 = reid_embed(paths, Net(), tf, image_root=str(tmp_path), n_workers=3, batch_size=3,
                    views=("full",), cache=str(tmp_path / "c.npy"), log=None)
    assert np.allclose(e1, e3, atol=1e-6)
    e4 = reid_embed(paths, None, tf, image_root=str(tmp_path), cache=str(tmp_path / "c.npy"), log=None)
    assert np.allclose(e3, e4)
    ef, per = reid_embed(paths, Net(), tf, image_root=str(tmp_path), views=("full", "hflip"),
                         return_views=True, log=None)
    assert set(per) == {"full", "hflip"} and np.allclose(np.linalg.norm(ef, axis=1), 1, atol=1e-5)
