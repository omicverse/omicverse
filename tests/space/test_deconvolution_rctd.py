"""Regression test for the RCTD deconvolution backend (issue #682).

`ov.space.Deconvolution.deconvolution(method='RCTD', ...)` integrates
`rctd-py` (https://github.com/p-gueguen/rctd-py) — the canonical 10x
Visium HD deconvolution method — into the same `Deconvolution` class
that already wraps Tangram / cell2location / FlashDeconv / Starfysh.

The real RCTD algorithm spends most of its time auto-calibrating
sigma (~60–100s on tiny inputs) and downloads a Q-matrix table on
first run. Neither is appropriate for unit-test speed, so this test
**mocks `rctd.run_rctd` and `rctd.Reference`** and only exercises the
omicverse-side wiring:

- the method dispatcher recognises ``method='RCTD'``,
- raw counts from ``layers['counts']`` are routed to RCTD's expected
  ``adata.X`` interface,
- ``rctd_kwargs`` (mode / cell_min / config overrides) flow through
  to ``Reference`` / ``run_rctd``,
- the returned ``FullResult`` / ``DoubletResult`` is reshaped into the
  ``adata_cell2location`` schema (pixels × cell-types, row-normalised
  to proportions),
- ``pixel_mask`` correctly slices ``obs`` / ``obsm['spatial']``,
- ``self.rctd_result`` and ``self.adata_sp.obsm['rctd_proportions']``
  are set as documented.

A separate `pytest.importorskip`-gated test runs the real algorithm
against a synthetic 5-celltype × 80-pixel dataset so the integration
is also validated end-to-end (skipped automatically when ``rctd-py``
is not installed in the test environment).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData


# --------------------------------------------------------------------------- #
# Synthetic data fixtures
# --------------------------------------------------------------------------- #


def _synthetic_pair(seed: int = 0):
    """Tiny scRNA + spatial pair with raw counts in `layers['counts']`.

    Same shape used by the real-RCTD test below so behaviour is
    consistent across mocked and live runs.
    """
    rng = np.random.default_rng(seed)
    n_genes = 100
    n_sc = 200
    n_sp = 80
    counts_sc = rng.poisson(lam=2.0, size=(n_sc, n_genes)).astype(np.int32)
    ad_sc = AnnData(X=sp.csr_matrix(counts_sc.astype(np.float32)))
    ad_sc.var_names = [f"g{i:03d}" for i in range(n_genes)]
    ad_sc.obs["cell_type"] = (["A", "B", "C", "D", "E"] * (n_sc // 5))[:n_sc]
    ad_sc.layers["counts"] = ad_sc.X.copy()

    counts_sp = rng.poisson(lam=2.0, size=(n_sp, n_genes)).astype(np.int32)
    ad_sp = AnnData(X=sp.csr_matrix(counts_sp.astype(np.float32)))
    ad_sp.var_names = ad_sc.var_names.copy()
    ad_sp.layers["counts"] = ad_sp.X.copy()
    ad_sp.obsm["spatial"] = rng.uniform(0, 1000, size=(n_sp, 2)).astype(np.float32)
    ad_sp.obs["sample"] = "smoke"
    return ad_sp, ad_sc


# --------------------------------------------------------------------------- #
# Mocked `rctd` module — fast, deterministic, exercises every wiring branch
# --------------------------------------------------------------------------- #


def _install_fake_rctd(monkeypatch, seen: dict):
    """Patch `rctd` in `sys.modules` with a stand-in that records every call.

    The real `rctd` package ships a NamedTuple `RCTDConfig` and result
    types with a fixed contract — we replicate the surface our
    integration relies on without pulling in the heavy implementation.
    """
    fake = types.ModuleType("rctd")

    # NamedTuple-like config: stash kwargs on a plain dict-backed object.
    class FakeConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)
        def __repr__(self):
            return f"FakeConfig({self.__dict__!r})"

    class FakeReference:
        def __init__(self, adata, *, cell_type_col="cell_type",
                     cell_min=25, n_max_cells=10000, min_UMI=100):
            seen["reference_kwargs"] = dict(
                cell_type_col=cell_type_col,
                cell_min=cell_min,
                n_max_cells=n_max_cells,
                min_UMI=min_UMI,
            )
            seen["reference_adata"] = adata
            self.cell_type_names = sorted(set(adata.obs[cell_type_col]))
            self.n_genes = adata.n_vars
            self.gene_names = list(adata.var_names)

    class FakeFullResult:
        def __init__(self, n_pixels, cell_type_names):
            rng = np.random.default_rng(42)
            self.weights = rng.dirichlet(
                np.ones(len(cell_type_names)), size=n_pixels
            ).astype(np.float32) * 100  # unnormalised abundances
            self.cell_type_names = cell_type_names
            # Mark the *last* pixel as filtered out so we can verify that
            # pixel_mask actually slices obs / obsm.
            self.pixel_mask = np.ones(n_pixels + 1, dtype=bool)
            self.pixel_mask[-1] = False
            self.weights = self.weights[: n_pixels]   # one row dropped
            self.converged = np.ones(n_pixels, dtype=bool)

    def fake_run_rctd(spatial, reference, *, mode="doublet", config=None,
                      batch_size="auto", sigma_override=None):
        seen["run_kwargs"] = dict(
            mode=mode, config=config,
            batch_size=batch_size, sigma_override=sigma_override,
        )
        seen["spatial_in"] = spatial
        # We wire the result so that pixel_mask drops the last pixel.
        n_pixels = spatial.n_obs - 1
        return FakeFullResult(n_pixels=n_pixels,
                              cell_type_names=reference.cell_type_names)

    fake.RCTDConfig = FakeConfig
    fake.Reference = FakeReference
    fake.run_rctd = fake_run_rctd
    fake.RCTD = MagicMock()  # not used by the omicverse integration path
    monkeypatch.setitem(sys.modules, "rctd", fake)
    return fake


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_rctd_dispatcher_wires_kwargs_and_reshapes_result(monkeypatch):
    seen: dict = {}
    _install_fake_rctd(monkeypatch, seen)

    import omicverse as ov

    ad_sp, ad_sc = _synthetic_pair()
    decov = ov.space.Deconvolution(adata_sp=ad_sp, adata_sc=ad_sc)
    decov.deconvolution(
        method="RCTD",
        celltype_key_sc="cell_type",
        rctd_kwargs={
            "mode": "full",
            "cell_min": 7,
            "min_UMI": 13,
            "config": {"UMI_min": 13, "max_iter": 21},
            "batch_size": 17,
            "sigma_override": 84,
        },
    )

    # ── Reference args correctly forwarded ──
    assert seen["reference_kwargs"]["cell_type_col"] == "cell_type"
    assert seen["reference_kwargs"]["cell_min"] == 7
    assert seen["reference_kwargs"]["min_UMI"] == 13

    # ── Reference fed RAW counts (from `layers['counts']`), not log-norm .X ──
    ref_adata = seen["reference_adata"]
    # If raw counts were forwarded, the .X content equals the layer.
    raw = ad_sc.layers["counts"]
    np.testing.assert_array_equal(
        np.asarray(raw.toarray() if sp.issparse(raw) else raw),
        np.asarray(ref_adata.X.toarray() if sp.issparse(ref_adata.X) else ref_adata.X),
    )

    # ── run_rctd args correctly forwarded ──
    assert seen["run_kwargs"]["mode"] == "full"
    assert seen["run_kwargs"]["batch_size"] == 17
    assert seen["run_kwargs"]["sigma_override"] == 84
    cfg = seen["run_kwargs"]["config"]
    assert cfg.UMI_min == 13
    assert cfg.max_iter == 21

    # ── adata_cell2location reshape: row-normalised proportions, pixels sliced ──
    res = decov.adata_cell2location
    # 80 - 1 (last pixel filtered out via pixel_mask) = 79 kept pixels
    assert res.shape == (79, 5)
    assert list(res.var_names) == ["A", "B", "C", "D", "E"]
    row_sums = np.asarray(res.X).sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    # ── obs / obsm sliced to the kept pixels ──
    assert len(res.obs) == 79
    assert res.obsm["spatial"].shape == (79, 2)
    np.testing.assert_array_equal(
        res.obsm["spatial"], ad_sp.obsm["spatial"][:79]
    )
    assert res.obs_names.tolist() == ad_sp.obs_names[:79].tolist()

    # ── adata_sp.obsm['rctd_proportions'] mirror is set, indexed by all pixels ──
    prop = decov.adata_sp.obsm["rctd_proportions"]
    assert isinstance(prop, pd.DataFrame)
    assert prop.shape == (80, 5)
    # The dropped last pixel has all-zero proportions (filled with 0.0 on reindex).
    assert (prop.iloc[-1].to_numpy() == 0.0).all()

    # ── self.method + self.rctd_result are set ──
    assert decov.method == "RCTD"
    assert decov.rctd_result is not None
    assert decov.rctd_result.cell_type_names == ["A", "B", "C", "D", "E"]


def test_rctd_unknown_mode_raises(monkeypatch):
    """The reshape branch only handles 'full'/'doublet'/'multi' — anything
    else must raise rather than silently produce a malformed AnnData."""
    seen: dict = {}
    fake = _install_fake_rctd(monkeypatch, seen)

    # Force the fake to return a result we can't reshape ('bogus' mode).
    def fake_run_rctd(spatial, reference, *, mode="doublet", config=None,
                      batch_size="auto", sigma_override=None):
        # Return a plain object — `mode='bogus'` must be rejected before
        # we ever inspect it for `weights` / `weights_doublet`.
        class Stub:
            cell_type_names = reference.cell_type_names
            pixel_mask = np.ones(spatial.n_obs, dtype=bool)
            weights = np.zeros((spatial.n_obs, len(reference.cell_type_names)))
        return Stub()
    fake.run_rctd = fake_run_rctd

    import omicverse as ov

    ad_sp, ad_sc = _synthetic_pair()
    decov = ov.space.Deconvolution(adata_sp=ad_sp, adata_sc=ad_sc)
    with pytest.raises(ValueError, match="Unknown RCTD mode"):
        decov.deconvolution(
            method="RCTD",
            celltype_key_sc="cell_type",
            rctd_kwargs={"mode": "bogus"},
        )


def test_rctd_rejects_continuous_x_without_raw_count_layer(monkeypatch):
    seen: dict = {}
    _install_fake_rctd(monkeypatch, seen)

    import omicverse as ov

    ad_sp, ad_sc = _synthetic_pair()
    del ad_sc.layers["counts"]
    ad_sc.X = sp.csr_matrix(np.full(ad_sc.shape, 0.25, dtype=np.float32))
    decov = ov.space.Deconvolution(adata_sp=ad_sp, adata_sc=ad_sc)

    with pytest.raises(ValueError, match="raw integer-like counts.*reference"):
        decov.deconvolution(method="RCTD", celltype_key_sc="cell_type")


def test_rctd_routes_custom_raw_count_layers(monkeypatch):
    seen: dict = {}
    _install_fake_rctd(monkeypatch, seen)

    import omicverse as ov

    ad_sp, ad_sc = _synthetic_pair()
    ad_sc.layers["raw"] = ad_sc.layers["counts"].copy()
    ad_sp.layers["raw"] = ad_sp.layers["counts"].copy()
    del ad_sc.layers["counts"]
    del ad_sp.layers["counts"]
    ad_sc.X = sp.csr_matrix(np.full(ad_sc.shape, 0.25, dtype=np.float32))
    ad_sp.X = sp.csr_matrix(np.full(ad_sp.shape, 0.5, dtype=np.float32))

    decov = ov.space.Deconvolution(adata_sp=ad_sp, adata_sc=ad_sc)
    decov.deconvolution(
        method="rctd",
        celltype_key_sc="cell_type",
        counts_layer_sc="raw",
        counts_layer_sp="raw",
        rctd_kwargs={"mode": "full"},
    )

    expected = ad_sc.layers["raw"]
    observed = seen["reference_adata"].X
    np.testing.assert_array_equal(
        observed.toarray() if sp.issparse(observed) else observed,
        expected.toarray() if sp.issparse(expected) else expected,
    )


def test_rctd_unknown_method_raise_message_lists_rctd():
    """Future regression: the dispatcher's `else: raise ValueError` should
    list 'RCTD' alongside the other backends."""
    import omicverse as ov

    ad_sp, ad_sc = _synthetic_pair()
    decov = ov.space.Deconvolution(adata_sp=ad_sp, adata_sc=ad_sc)
    with pytest.raises(ValueError, match=r".*RCTD.*"):
        decov.deconvolution(method="NotARealMethod")


# --------------------------------------------------------------------------- #
# Real-rctd end-to-end smoke (skipped automatically when rctd-py is missing)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_rctd_full_mode_real_smoke():
    """End-to-end smoke against the real `rctd-py` algorithm.

    Skipped automatically when `rctd-py` is not installed (e.g. minimal
    CI environments). On full installs this catches API drift in the
    upstream library that the mocked tests would miss — e.g. if a
    future release renamed `weights` or moved `pixel_mask` to `mask`.
    """
    pytest.importorskip("rctd")
    import omicverse as ov

    ad_sp, ad_sc = _synthetic_pair()
    # The real sigma estimator requires spots with >300 UMI. The generic
    # routing fixture has ~200 UMI and is not a valid full-runtime fixture.
    ad_sp.layers['counts'] = ad_sp.layers['counts'] * 3
    decov = ov.space.Deconvolution(adata_sp=ad_sp, adata_sc=ad_sc)
    decov.deconvolution(
        method="RCTD",
        celltype_key_sc="cell_type",
        rctd_kwargs={
            "mode": "full",
            "cell_min": 10,
            "min_UMI": 10,
            "config": {"UMI_min": 10},
            "batch_size": 64,
        },
    )

    res = decov.adata_cell2location
    assert res.n_obs > 0
    assert res.n_vars == 5
    assert set(res.var_names) == {"A", "B", "C", "D", "E"}
    np.testing.assert_allclose(np.asarray(res.X).sum(axis=1), 1.0, rtol=1e-4)
