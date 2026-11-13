"""ov.pp.pacmap / ov.pp.localmap (anndataoom-fast port of brisc's PaCMAP / LocalMAP)."""
import numpy as np
import pytest

aof = pytest.importorskip("anndataoom_fast")


def _adata(n=1200, d=20, seed=0):
    import anndata as ad

    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 6, (5, d))
    labels = rng.integers(0, 5, n)
    pcs = (centres[labels] + rng.normal(0, 1, (n, d))).astype(np.float32)
    a = ad.AnnData(np.zeros((n, 3), np.float32))
    a.obsm["X_pca"] = pcs
    a.obs["blob"] = labels.astype(str)
    return a


@pytest.mark.parametrize("name", ["pacmap", "localmap"])
def test_embedding_stored_and_matches_backend(name):
    import omicverse as ov

    a = _adata()
    getattr(ov.pp, name)(a, n_threads=2, num_iterations=(30, 30, 60))
    key = f"X_{name}"
    emb = a.obsm[key]
    assert emb.shape == (a.n_obs, 2) and emb.dtype == np.float32
    assert np.isfinite(emb).all()
    params = a.uns[name]["params"]
    assert params["use_rep"] == "X_pca" and params["random_state"] == 0
    # same as calling anndataoom-fast directly, for any thread count
    ref = getattr(aof, name)(a.obsm["X_pca"], n_threads=5, num_iterations=(30, 30, 60))
    assert np.array_equal(emb, ref)
    # blobs stay separated: nearest embedded neighbour mostly has the same label
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(emb).kneighbors(emb, return_distance=False)[:, 1]
    labels = a.obs["blob"].to_numpy()
    assert (labels[nn] == labels).mean() > 0.95


def test_precomputed_knn_and_key():
    import omicverse as ov

    a = _adata(n=800)
    idx, dist = aof._embed.brisc_knn(a.obsm["X_pca"], 20)
    ov.pp.localmap(a, key_added="X_lm", knn_indices=idx, knn_dists=dist, random_state=3,
                   num_iterations=50)
    ref = aof.localmap(a.obsm["X_pca"], knn_indices=idx, knn_dists=dist, random_state=3,
                       num_iterations=50)
    assert np.array_equal(a.obsm["X_lm"], ref)
    assert a.uns["lm"]["params"]["knn"] == "given"


def test_missing_rep():
    import omicverse as ov

    a = _adata(n=100)
    with pytest.raises(KeyError):
        ov.pp.pacmap(a, use_rep="X_nope")


def test_registered():
    import omicverse as ov
    from omicverse._registry import _global_registry

    assert ov.pp.localmap is not None and ov.pp.pacmap is not None
    found = _global_registry.find("localmap")
    assert found, "ov.pp.localmap is not in the function registry"
