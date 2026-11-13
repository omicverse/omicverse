"""``method='soupx'`` must find the cluster column on its own, like DecontX.

The documented call ``remove_ambient(adata, method='soupx', raw=raw)`` passes
no ``cluster_key``. ``_run_soupx`` used to treat that as "no clusters", skip
``pysoupx.auto_est_cont`` and silently set rho = 0.1 -- even on an AnnData
that already had ``obs['leiden']``. It now resolves the column the same way
``_run_decontx`` does (clusters / leiden / louvain / cell_type / celltype)
and only falls back to 0.1, with a warning, when none exists.

The first group of tests runs the real pysoupx backend on a small synthetic
channel with a known contamination fraction; the second fakes the backend so
the wiring is still checked where pysoupx is not installed (e.g. CI, which
does not install ``omicverse[ambient]``).
"""
from __future__ import annotations

import types
import warnings

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import omicverse as ov
from omicverse.pp.ambient import _ambient

TRUE_RHO = 0.2          # far from the 0.1 fallback, so the two can't be confused
FALLBACK = "auto_est_cont"  # substring of the fallback warning


def _channel(seed=0, n_clusters=8, cells_per=40, n_genes=1000, n_markers=20,
             marker_mass=0.3, rho=TRUE_RHO, n_empty=1500):
    """Filtered cells (with obs['leiden']) + raw droplets (cells + empties).

    Each cluster has its own block of marker genes on top of a shared
    background; every cell takes a ``rho`` fraction of its counts from the
    soup (the mean profile), and the empty droplets (< 100 UMIs, SoupX's
    default soup range) are pure soup. Enough clusters keep the soup leaking
    into other clusters' marker genes low, so the markers pass SoupX's
    tf-idf > 1 cut-off.
    """
    rng = np.random.default_rng(seed)
    n_bg = n_genes - n_clusters * n_markers
    bg = rng.lognormal(0, 1, n_bg)
    bg /= bg.sum()
    prof = np.zeros((n_clusters, n_genes))
    for k in range(n_clusters):
        prof[k, :n_bg] = (1 - marker_mass) * bg
        start = n_bg + k * n_markers
        prof[k, start:start + n_markers] = marker_mass / n_markers
    soup = prof.mean(0)

    labels = np.repeat(np.arange(n_clusters), cells_per)
    depth = rng.lognormal(np.log(600), 0.3, labels.size).astype(int)
    X = []
    for k, d in zip(labels, depth):
        n_amb = rng.binomial(d, rho)
        X.append(rng.multinomial(d - n_amb, prof[k])
                 + rng.multinomial(n_amb, soup))
    genes = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    cells = anndata.AnnData(
        sp.csr_matrix(np.asarray(X, dtype=np.float32)),
        obs=pd.DataFrame(
            {"leiden": pd.Categorical(labels.astype(str))},
            index=[f"cell{i:04d}" for i in range(labels.size)]),
        var=genes)

    emp = [rng.multinomial(max(int(rng.lognormal(np.log(40), 0.4)), 1), soup)
           for _ in range(n_empty)]
    empties = anndata.AnnData(
        sp.csr_matrix(np.asarray(emp, dtype=np.float32)),
        obs=pd.DataFrame(index=[f"empty{i:05d}" for i in range(n_empty)]),
        var=genes.copy())
    return cells, anndata.concat([cells, empties])


def _assert_estimated_not_fallback(rho):
    # SoupX's marker-based estimate (gamma prior centred on 0.05) runs a bit
    # low -- ~0.16 for a true 0.2 here -- but it must beat the fixed 0.1.
    assert abs(rho - TRUE_RHO) < abs(0.1 - TRUE_RHO) - 0.02, rho


def _fallback_warnings(record):
    return [w for w in record
            if issubclass(w.category, UserWarning)
            and FALLBACK in str(w.message)]


# ---------------------------------------------------------------------------
# real pysoupx backend
# ---------------------------------------------------------------------------
@pytest.fixture
def pysoupx():
    return pytest.importorskip("pysoupx")


def test_leiden_without_cluster_key_reaches_auto_est_cont(pysoupx):
    cells, raw = _channel()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = ov.pp.ambient.remove_ambient(cells, method="soupx", raw=raw)

    assert not _fallback_warnings(record)
    info = out.uns["ambient"]
    assert info["auto_estimated"] is True
    assert info["cluster_key"] == "leiden"
    _assert_estimated_not_fallback(info["contamination_fraction"])
    np.testing.assert_allclose(
        out.obs["ambient_contamination"], info["contamination_fraction"])


def test_estimate_contamination_also_resolves_leiden(pysoupx):
    cells, raw = _channel()
    rho = ov.pp.ambient.estimate_contamination(cells, method="soupx", raw=raw)
    assert rho.attrs["auto_estimated"] is True
    assert rho.attrs["cluster_key"] == "leiden"
    _assert_estimated_not_fallback(rho.mean())


def test_no_cluster_column_warns_and_uses_fallback(pysoupx):
    cells, raw = _channel()
    del cells.obs["leiden"]

    with pytest.warns(UserWarning, match=FALLBACK) as record:
        out = ov.pp.ambient.remove_ambient(cells, method="soupx", raw=raw)

    message = str(_fallback_warnings(record)[0].message)
    assert "contamination_fraction=0.1" in message
    assert "cluster_key=" in message
    info = out.uns["ambient"]
    assert info["auto_estimated"] is False
    assert info["cluster_key"] is None
    assert info["contamination_fraction"] == pytest.approx(0.1)
    np.testing.assert_allclose(out.obs["ambient_contamination"], 0.1)


# ---------------------------------------------------------------------------
# faked backend -- runs without pysoupx
# ---------------------------------------------------------------------------
def _fake_soupx(monkeypatch):
    """Patch ``_require`` with a pysoupx stand-in that records its calls."""
    calls = {"auto_est_cont": 0, "set_rho": []}

    class SoupChannel:
        @classmethod
        def from_anndata(cls, adata, raw=None, cluster_key=None, **kwargs):
            sc = cls()
            sc.n_cells = adata.n_obs
            sc.toc = sp.csr_matrix(adata.X.T)
            sc.meta_data = pd.DataFrame(index=adata.obs_names)
            calls["cluster_key"] = cluster_key
            return sc

    def set_contamination_fraction(sc, rho, **kwargs):
        calls["set_rho"].append(rho)
        sc.meta_data["rho"] = rho

    def auto_est_cont(sc, verbose=False, **kwargs):
        calls["auto_est_cont"] += 1
        set_contamination_fraction(sc, 0.123)

    def adjust_counts(sc, clusters=None, **kwargs):
        calls["adjust_clusters"] = clusters
        return sc.toc

    fake = types.SimpleNamespace(
        SoupChannel=SoupChannel,
        set_contamination_fraction=set_contamination_fraction,
        auto_est_cont=auto_est_cont,
        adjust_counts=adjust_counts,
    )
    monkeypatch.setattr(
        _ambient, "_require",
        lambda modname, role, extra="ambient": fake)
    return calls


def _small(obs_cols):
    X = np.arange(1, 13, dtype=float).reshape(4, 3)
    return anndata.AnnData(X=X, obs=pd.DataFrame(
        {c: ["A", "A", "B", "B"] for c in obs_cols},
        index=[f"c{i}" for i in range(4)]))


@pytest.mark.parametrize("col", _ambient._CLUSTER_KEY_CANDIDATES)
def test_fake_backend_gets_resolved_cluster_key(monkeypatch, col):
    calls = _fake_soupx(monkeypatch)
    adata = _small([col])

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _, rho, _, meta = _ambient._run_soupx(adata, raw=adata)

    assert calls["cluster_key"] == col            # set on the SoupChannel
    assert calls["auto_est_cont"] == 1
    assert calls["adjust_clusters"] is None       # cluster-level adjustment
    assert meta["auto_estimated"] is True
    assert meta["cluster_key"] == col
    np.testing.assert_allclose(rho, 0.123)


def test_fake_backend_explicit_cluster_key_wins(monkeypatch):
    calls = _fake_soupx(monkeypatch)
    adata = _small(["leiden", "cell_type"])
    _, _, _, meta = _ambient._run_soupx(
        adata, raw=adata, cluster_key="cell_type")
    assert calls["cluster_key"] == "cell_type"
    assert meta["cluster_key"] == "cell_type"


def test_fake_backend_no_clusters_warns_and_falls_back(monkeypatch):
    calls = _fake_soupx(monkeypatch)
    adata = _small([])

    with pytest.warns(UserWarning, match=FALLBACK):
        _, rho, _, meta = _ambient._run_soupx(adata, raw=adata)

    assert calls["cluster_key"] is None
    assert calls["auto_est_cont"] == 0
    assert calls["set_rho"] == [0.1]
    assert calls["adjust_clusters"] is False
    assert meta["auto_estimated"] is False
    np.testing.assert_allclose(rho, 0.1)


def test_fake_backend_missing_explicit_key_names_it(monkeypatch):
    calls = _fake_soupx(monkeypatch)
    adata = _small(["leiden"])

    with pytest.warns(UserWarning, match="cluster_key='nope'"):
        _, _, _, meta = _ambient._run_soupx(adata, raw=adata,
                                            cluster_key="nope")
    assert calls["cluster_key"] is None
    assert calls["auto_est_cont"] == 0
    assert meta["auto_estimated"] is False


def test_fake_backend_explicit_rho_needs_no_clusters(monkeypatch):
    calls = _fake_soupx(monkeypatch)
    adata = _small([])

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _, rho, _, meta = _ambient._run_soupx(
            adata, raw=adata, contamination_fraction=0.05)

    assert calls["auto_est_cont"] == 0
    assert calls["set_rho"] == [0.05]
    assert meta["auto_estimated"] is False
    np.testing.assert_allclose(rho, 0.05)
