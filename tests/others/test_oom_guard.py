"""Generic AnnDataOOM guard in ``register_function`` (omicverse/_oom_guard.py).

Non-native registered functions receiving a disk-backed AnnDataOOM run on an
in-memory copy and their results are copied back; OOM-native functions
(omicverse/_oom_native.py) see the AnnDataOOM itself.
"""

import importlib

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

matplotlib.use("Agg")

pytest.importorskip("anndataoom")

import omicverse as ov  # noqa: E402
from omicverse import _oom_guard, _oom_native  # noqa: E402
from omicverse._oom_compat import is_oom  # noqa: E402
from omicverse._registry import register_function  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _make_adata(n_obs=300, n_vars=600, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n_obs, n_vars)).astype(np.float32))
    X.sort_indices()
    obs = pd.DataFrame(
        {"group": pd.Categorical(np.where(np.arange(n_obs) % 3 == 0, "a",
                                          np.where(np.arange(n_obs) % 3 == 1, "b", "c"))),
         "batch": pd.Categorical(np.where(np.arange(n_obs) % 2 == 0, "b1", "b2"))},
        index=[f"cell{i}" for i in range(n_obs)])
    genes = [f"G{i}" for i in range(n_vars)]
    genes[:3] = ["MT-CO1", "MT-ND1", "MT-ATP6"]
    var = pd.DataFrame(index=genes)
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.obsm["spatial"] = rng.uniform(0, 100, size=(n_obs, 2))
    a.obsm["X_pre"] = rng.normal(size=(n_obs, 5))
    a.uns["keep"] = {"x": 1}
    return a


@pytest.fixture
def h5ad_path(tmp_path):
    p = tmp_path / "tiny.h5ad"
    _make_adata().write_h5ad(p)
    return str(p)


@pytest.fixture
def oom(h5ad_path):
    a = ov.read(h5ad_path, storage="disk")
    assert is_oom(a)
    yield a
    try:
        a.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# registered test functions (not allow-listed)
# ---------------------------------------------------------------------------

@register_function(aliases=["_oomguard_add_keys"], category="test",
                   description="test: adds obs/var/obsm/uns/obsp/layers entries")
def _oomguard_add_keys(adata, tag="t"):
    """Adds obs/var/obsm/uns/obsp/layers entries."""
    assert not is_oom(adata), "guard must hand over an in-memory AnnData"
    X = adata.X
    tot = np.asarray(X.sum(axis=1)).ravel()
    adata.obs[f"{tag}_total"] = tot
    adata.obs["group"] = adata.obs["group"].astype(str).str.upper().astype("category")  # changed
    adata.var[f"{tag}_mean"] = np.asarray(X.mean(axis=0)).ravel()
    adata.obsm[f"X_{tag}"] = np.c_[tot, tot]
    adata.obsm["X_pre"] = adata.obsm["X_pre"] * 0 + 7.0  # changed in place
    adata.obsp[f"{tag}_graph"] = sp.identity(adata.n_obs, format="csr")
    adata.uns[f"{tag}_info"] = {"n": adata.n_obs}
    adata.uns["keep"]["x"] = 2  # nested in-place change
    adata.layers[f"{tag}_counts"] = X.copy()
    return None


@register_function(aliases=["_oomguard_modify_X"], category="test",
                   description="test: modifies X in place")
def _oomguard_modify_X(adata):
    """Doubles X in place."""
    adata.X = adata.X * 2
    adata.obs["doubled"] = True


@register_function(aliases=["_oomguard_subset"], category="test",
                   description="test: returns a new AnnData")
def _oomguard_subset(adata, n=10):
    """Returns the first n cells as a new AnnData."""
    return adata[:n].copy()


@register_function(aliases=["_oomguard_returns_self"], category="test",
                   description="test: in place, returns its input")
def _oomguard_returns_self(adata):
    """Sets an obs flag and returns its input."""
    adata.obs["flag"] = 1
    return adata


@register_function(aliases=["_oomguard_native"], category="test",
                   description="test: allow-listed function")
def _oomguard_native(adata):
    """Allow-listed: reports whether it got the AnnDataOOM."""
    return is_oom(adata)


_oom_native.register_native(_oomguard_native)


# ---------------------------------------------------------------------------
# (i) non-native function: materialise + copy back
# ---------------------------------------------------------------------------

def test_non_native_copies_back_new_and_changed(oom, capsys):
    ref = _make_adata()
    out = _oomguard_add_keys(oom, tag="t")
    assert out is None
    msg = capsys.readouterr().out
    assert msg.count("[AnnDataOOM]") == 1 and "_oomguard_add_keys" in msg
    assert "to_memory" in msg
    assert is_oom(oom)
    np.testing.assert_allclose(oom.obs["t_total"].values, np.asarray(ref.X.sum(1)).ravel())
    assert list(oom.obs["group"].cat.categories) == ["A", "B", "C"]
    assert "t_mean" in oom.var.columns
    assert oom.obsm["X_t"].shape == (oom.n_obs, 2)
    assert np.all(oom.obsm["X_pre"] == 7.0)
    assert oom.obsp["t_graph"].shape == (oom.n_obs, oom.n_obs)
    assert oom.uns["t_info"] == {"n": oom.n_obs}
    assert oom.uns["keep"]["x"] == 2
    assert "t_counts" in oom.layers.keys()
    lay = oom.layers["t_counts"][:]
    np.testing.assert_allclose(np.asarray(lay.todense() if sp.issparse(lay) else lay),
                               ref.X.toarray())


def test_adata_kwarg_and_return_values(oom):
    sub = _oomguard_subset(adata=oom, n=10)
    assert isinstance(sub, ad.AnnData) and not is_oom(sub) and sub.n_obs == 10
    same = _oomguard_returns_self(oom)
    assert same is oom and "flag" in oom.obs.columns


def test_x_modification_returns_in_memory(oom):
    before = np.asarray(oom.X[:5].todense() if sp.issparse(oom.X[:5]) else oom.X[:5])
    with pytest.warns(UserWarning, match="modified X"):
        res = _oomguard_modify_X(oom)
    assert isinstance(res, ad.AnnData) and not is_oom(res)
    after_mem = res.X[:5].toarray() if sp.issparse(res.X) else res.X[:5]
    np.testing.assert_allclose(after_mem, before * 2)
    # obs still copied back; disk X untouched
    assert "doubled" in oom.obs.columns
    still = oom.X[:5]
    np.testing.assert_allclose(np.asarray(still.todense() if sp.issparse(still) else still), before)


def test_real_non_native_function(oom):
    ov.pp.normalize_total(oom)  # native, lazy
    ov.pp.log1p(oom)
    res = ov.single.get_celltype_marker(oom, clustertype="group", rank=True, key="rank_genes_groups")
    assert isinstance(res, dict) and set(res) == {"a", "b", "c"}
    assert "rank_genes_groups" in oom.uns  # computed on the in-memory copy, copied back


# ---------------------------------------------------------------------------
# (ii) native functions are never materialised
# ---------------------------------------------------------------------------

def _forbid_materialise(monkeypatch, obj):
    def boom(*a, **k):
        raise AssertionError("native function materialised the AnnDataOOM")
    monkeypatch.setattr(type(obj), "to_memory", boom)
    monkeypatch.setattr(type(obj), "to_adata", boom)
    monkeypatch.setattr(_oom_guard, "_materialise", boom)


def test_native_not_materialised(oom, monkeypatch):
    _forbid_materialise(monkeypatch, oom)
    assert _oomguard_native(oom) is True
    ov.pp.qc(oom, tresh={"mito_perc": 0.5, "nUMIs": 1, "detected_genes": 1}, doublets=False)
    ov.pl.embedding(oom, basis="spatial", color="group", show=False)


def test_allow_list_entries_resolve():
    """Every allow-list entry names a real function (catches renames)."""
    for name in sorted(_oom_native.OOM_NATIVE):
        if name.startswith(__name__):
            continue
        mod, _, attr = name.rpartition(".")
        m = importlib.import_module(mod)
        assert hasattr(m, attr), name


# ---------------------------------------------------------------------------
# (iii) MemoryError
# ---------------------------------------------------------------------------

def test_memory_error_when_too_large(oom, monkeypatch):
    monkeypatch.setattr(_oom_guard, "available_memory", lambda: 1024)
    with pytest.raises(MemoryError, match="to_memory"):
        _oomguard_add_keys(oom)
    assert "t_total" not in oom.obs.columns
    est = _oom_guard.estimate_nbytes(oom)
    assert 0 < est < 300 * 600 * 8 * 2


def test_guard_can_be_disabled(oom, monkeypatch):
    monkeypatch.setenv("OMICVERSE_OOM_GUARD", "0")
    assert _oomguard_subset.__wrapped__ is not None
    called = {}

    @register_function(aliases=["_oomguard_probe"], category="test", description="probe")
    def probe(adata):
        """Records whether it got the AnnDataOOM."""
        called["oom"] = is_oom(adata)

    probe(oom)
    assert called["oom"] is True


# ---------------------------------------------------------------------------
# (iv) AnnData inputs untouched; to_memory helper
# ---------------------------------------------------------------------------

def test_anndata_passthrough(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("guard ran on a plain AnnData")
    monkeypatch.setattr(_oom_guard, "call", boom)
    a = _make_adata()
    assert _oomguard_add_keys(a) is None
    assert "t_total" in a.obs.columns
    assert _oomguard_returns_self(a) is a
    assert ov.utils.to_memory(a) is a


def test_to_memory(oom):
    m = ov.utils.to_memory(oom)
    assert isinstance(m, ad.AnnData) and not is_oom(m)
    assert m.shape == oom.shape
    np.testing.assert_allclose(m.X.toarray(), _make_adata().X.toarray())


# ---------------------------------------------------------------------------
# smoke: allow-listed functions run on a tiny AnnDataOOM without the guard
# ---------------------------------------------------------------------------

def test_native_pipeline_smoke(oom, monkeypatch):
    monkeypatch.setattr(_oom_guard, "_materialise",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("materialised")))
    oom.obs["grp2"] = pd.Categorical(np.where(np.arange(oom.n_obs) % 2, "x", "y"))
    ov.pp.qc(oom, tresh={"mito_perc": 0.5, "nUMIs": 1, "detected_genes": 1}, doublets=False)
    ov.pp.preprocess(oom, mode="shiftlog|pearson", n_HVGs=50)
    ov.pp.scale(oom)
    ov.pp.pca(oom, layer="scaled", n_pcs=10)
    ov.pp.neighbors(oom, n_neighbors=10, use_rep="scaled|original|X_pca")
    ov.pp.umap(oom)
    ov.pp.leiden(oom, resolution=1)
    assert {"X_pca", "X_umap"} <= set(oom.obsm.keys()) and "leiden" in oom.obs
    ov.single.find_markers(oom, groupby="group")
    ov.single.get_markers(oom, n_genes=3)
    ov.pl.embedding(oom, basis="X_umap", color="leiden", show=False)
    ov.pl.dotplot(oom, list(oom.var_names[3:6]), groupby="group", show=False)
    ov.pl.violin(oom, keys="nUMIs", groupby="group", show=False)
    ov.pl.cellproportion(oom, celltype_clusters="group", groupby="grp2")
    ov.space.spatial_neighbors(oom)
    ov.space.spatial_autocorr(oom, genes=list(oom.var_names[3:13]))
    assert is_oom(oom)
