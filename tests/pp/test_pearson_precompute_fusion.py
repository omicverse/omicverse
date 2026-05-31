"""Fused-pass Pearson HVG (in-memory path) must equal the legacy version.

`omicverse.pp._normalization.normalize_total` now stashes per-gene
totals + per-gene E[count^2] + per-cell totals in
`adata.uns['_pearson_precompute']` as a side product of its existing
per-cell-sum pass. `omicverse.pp.experimental._highly_variable_pearson_residuals`
auto-consumes that dict to skip its own first scan over X.

This test pins down that the fast path produces the same residual
variances + highly_variable mask as the legacy path on the same input.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import anndata


def _make_synthetic(n_obs=400, n_vars=300, density=0.06, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.random(
        n_obs, n_vars, density=density, format="csr",
        dtype=np.float32, random_state=rng,
    )
    # Promote to integer-ish counts; Pearson HVG needs non-negative ints.
    X.data = np.ceil(X.data * 20).astype(np.float32)
    a = anndata.AnnData(X=X.copy())
    a.layers["counts"] = X.copy()
    return a


def _run_normalize(adata):
    from omicverse.pp._normalization import normalize_total, log1p
    normalize_total(adata, target_sum=1e4)
    log1p(adata)


def _run_hvg(adata, *, n_top=50, batch_key=None):
    from omicverse.pp.experimental import highly_variable_genes
    highly_variable_genes(
        adata, flavor="pearson_residuals",
        n_top_genes=n_top, layer="counts", batch_key=batch_key,
        subset=False, inplace=True,
    )


def test_normalize_stashes_precompute():
    a = _make_synthetic()
    _run_normalize(a)
    pre = a.uns.get("_pearson_precompute")
    assert pre is not None, "normalize_total did not stash precompute"
    assert pre["sums_genes"].shape == (a.n_vars,)
    assert pre["sq_sums_genes"].shape == (a.n_vars,)
    assert pre["sums_cells"].shape == (a.n_obs,)
    assert pre["n_obs"] == a.n_obs and pre["n_vars"] == a.n_vars
    # Recover per-cell library size sanity.
    raw_sums = np.asarray(a.layers["counts"].sum(axis=1)).ravel()
    np.testing.assert_allclose(pre["sums_cells"], raw_sums, rtol=1e-6)


def test_hvg_pearson_fast_path_matches_legacy():
    """Two runs on identical input — one with precompute auto-consumed,
    one with it disabled — yield bit-equal residual_variances and the
    same highly_variable mask."""
    # Path A: legacy (force two-pass by deleting the precompute).
    a = _make_synthetic()
    _run_normalize(a)
    a.uns.pop("_pearson_precompute", None)
    _run_hvg(a, n_top=50)
    rv_a = np.asarray(a.var["residual_variances"].values, dtype=np.float64)
    mask_a = a.var["highly_variable"].values.astype(bool)

    # Path B: fast (precompute auto-consumed).
    b = _make_synthetic()
    _run_normalize(b)
    assert "_pearson_precompute" in b.uns  # would be a regression
    _run_hvg(b, n_top=50)
    rv_b = np.asarray(b.var["residual_variances"].values, dtype=np.float64)
    mask_b = b.var["highly_variable"].values.astype(bool)

    np.testing.assert_allclose(rv_a, rv_b, atol=1e-9, rtol=1e-10)
    np.testing.assert_array_equal(mask_a, mask_b)


def test_subset_invalidates_precompute():
    """If the caller var-subsets the adata between normalize and HVG,
    the stale precompute must be rejected and the HVG must fall back to
    a from-scratch pass on the subset."""
    a = _make_synthetic()
    _run_normalize(a)
    # Drop half the genes — precompute n_vars no longer matches.
    keep = np.ones(a.n_vars, dtype=bool)
    keep[::2] = False
    a_sub = a[:, keep].copy()
    # Preserve the stale uns key on purpose.
    a_sub.uns["_pearson_precompute"] = a.uns["_pearson_precompute"]
    _run_hvg(a_sub, n_top=20)  # must not crash
    # If the staleness guard fired, the run completed and wrote HVG cols.
    assert "residual_variances" in a_sub.var.columns
    assert int(a_sub.var["highly_variable"].sum()) == 20


def test_exclude_highly_expressed_skips_precompute():
    """`exclude_highly_expressed=True` re-derives counts_per_cell on a
    filtered subset; the precompute would carry full-panel sums_cells
    that no longer matches the normalize-time per-cell totals. Verify
    the function skips the stash rather than shipping inconsistent data."""
    from omicverse.pp._normalization import normalize_total
    a = _make_synthetic()
    normalize_total(a, target_sum=1e4, exclude_highly_expressed=True)
    assert "_pearson_precompute" not in a.uns
