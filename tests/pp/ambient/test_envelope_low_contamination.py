"""``method='envelope'`` at low contamination with truly absent genes.

On the 10x hgmm human/mouse mixture (true contamination ~1 %) the envelope
used to put every cell at ``rho_lo``: it ranked genes by their *observed*
native-to-soup ratio and kept the lowest 50, so it picked genes that by chance
had zero counts in the group and read rho = 0 off them. And even at the right
rho, per-gene half-to-even rounding of sub-0.5 expected ambient counts gave
most of the removal back. This simulates that regime with a known truth:
two "species" with disjoint gene sets, ~1.5 % contamination, one soup.
"""
from __future__ import annotations

import warnings

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import omicverse as ov
from omicverse.pp.ambient._envelope import correct_counts


def _species_mixture(seed=0, n_genes=1600, n_cells=400, n_empty=3000,
                     rho_mean=0.015):
    rng = np.random.default_rng(seed)
    half = n_genes // 2
    native = np.zeros((2, n_genes))
    native[0, :half] = rng.lognormal(0, 1.5, half)      # species A genes
    native[1, half:] = rng.lognormal(0, 1.5, half)      # species B genes
    native /= native.sum(1, keepdims=True)
    soup = 0.5 * native[0] + 0.5 * native[1]

    species = rng.integers(0, 2, n_cells)
    depth = rng.lognormal(np.log(4000), 0.4, n_cells).astype(int)
    rho = rng.beta(3, 3 / rho_mean - 3, n_cells)
    X, Xn = [], []
    for c in range(n_cells):
        n_amb = rng.binomial(depth[c], rho[c])
        nat = rng.multinomial(depth[c] - n_amb, native[species[c]])
        X.append(nat + rng.multinomial(n_amb, soup))
        Xn.append(nat)
    names = [f"cell{c:04d}" for c in range(n_cells)]
    cells = anndata.AnnData(
        sp.csr_matrix(np.asarray(X, dtype=np.float32)),
        obs=pd.DataFrame({"species": np.where(species == 0, "A", "B")},
                         index=names),
        var=pd.DataFrame({"genome": ["A"] * half + ["B"] * half},
                         index=[f"g{i}" for i in range(n_genes)]))
    cells.obs["true_rho"] = 1 - np.asarray(Xn).sum(1) / np.asarray(X).sum(1)
    emp = [rng.multinomial(max(int(rng.lognormal(np.log(40), 0.4)), 1), soup)
           for _ in range(n_empty)]
    empties = anndata.AnnData(
        sp.csr_matrix(np.asarray(emp, dtype=np.float32)),
        obs=pd.DataFrame(index=[f"empty{e:05d}" for e in range(n_empty)]),
        var=cells.var.copy())
    raw = anndata.concat([cells, empties])
    return cells, raw, np.asarray(Xn, dtype=float)


@pytest.fixture(scope="module")
def mixture():
    return _species_mixture()


def _run(cells, raw, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = ov.pp.ambient.remove_ambient(
            cells.copy(), method="envelope", raw=raw,
            cluster_key="species", **kw)
    pinned = [w for w in rec if "pinned at a bound" in str(w.message)]
    return out, pinned


def test_in_sample_panel_reproduces_the_pin_and_warns(mixture):
    """The old selection collapses to rho_lo -- and now says so, loudly."""
    cells, raw, _ = mixture
    out, pinned = _run(cells, raw, crossfit=False, rounding="gene")
    rho = out.obs["ambient_contamination"].to_numpy()
    assert np.mean(rho <= 0.005) > 0.9
    assert pinned, "a pinned estimate must raise a UserWarning"
    diag = out.uns["ambient"]["envelope_diagnostics"]
    assert diag["frac_at_rho_lo"] > 0.9 and diag["pinned_warning"]
    # the diagnosis: the in-sample panel reads far fewer counts than the true
    # contamination puts on it (it was selected for having few)
    g = diag["groups"]
    truth = cells.obs["true_rho"].mean()
    assert (g["panel_observed"]
            < 0.5 * truth * g["panel_expected_per_unit_rho"]).all()


def test_crossfit_recovers_low_contamination(mixture):
    cells, raw, _ = mixture
    truth = cells.obs["true_rho"].to_numpy()
    out, pinned = _run(cells, raw, level_shrink=1.0)
    rho = out.obs["ambient_contamination"].to_numpy()
    assert not pinned
    # deep empties (single preparation): the soup is cross-fitted as well
    diag = out.uns["ambient"]["envelope_diagnostics"]
    assert diag["soup_crossfit"] and diag["soup_selection_stability"] >= 0.8
    assert np.mean(rho <= 0.005) < 0.25
    assert 0.6 * truth.mean() < rho.mean() < 1.5 * truth.mean()
    # the default (conservative level_shrink=0.75) is no longer pinned either
    out75, pinned75 = _run(cells, raw)
    assert not pinned75
    assert out75.obs["ambient_contamination"].mean() > 0.5 * truth.mean()


def test_total_rounding_removes_what_rho_reports(mixture):
    """Per-gene rounding of sub-0.5 expected counts gives the removal back."""
    cells, raw, _ = mixture
    A = cells.X.toarray().astype(float)
    truth = cells.obs["true_rho"].to_numpy()
    soup = np.asarray(raw.X[cells.n_obs:].sum(0)).reshape(1, -1)
    soup = soup / soup.sum()
    b = np.zeros(cells.n_obs, dtype=int)
    N = A.sum(1)

    C_gene = correct_counts(A, soup, b, truth)                  # default
    C_tot = correct_counts(A, soup, b, truth, rounding="total")
    assert (A.sum(1) - C_gene.sum(1)).sum() < 0.5 * (truth * N).sum()
    np.testing.assert_allclose(A.sum(1) - C_tot.sum(1), np.rint(truth * N),
                               atol=1)
    assert (C_tot >= 0).all() and (C_tot <= A).all()
    assert C_tot.dtype == np.int64

    # cross-species counts are pure soup: 'total' removes most of them
    gen = cells.var["genome"].to_numpy()
    sp_ = cells.obs["species"].to_numpy()
    other = np.where(sp_[:, None] == "A", gen[None, :] == "B",
                     gen[None, :] == "A")
    assert (C_tot * other).sum() < 0.5 * (A * other).sum()
    assert (C_gene * other).sum() > 0.8 * (A * other).sum()


def test_under_removal_warns_and_total_does_not(mixture):
    cells, raw, _ = mixture
    with pytest.warns(UserWarning, match="rounding='total'"):
        out = ov.pp.ambient.remove_ambient(
            cells.copy(), method="envelope", raw=raw, cluster_key="species")
    assert out.uns["ambient"]["removed_fraction_of_target"] < 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = ov.pp.ambient.remove_ambient(
            cells.copy(), method="envelope", raw=raw, cluster_key="species",
            rounding="total")
    assert out.uns["ambient"]["removed_fraction_of_target"] > 0.95


def test_upper_pin_warns(mixture):
    """Most cells clipped at rho_hi is reported as well."""
    cells, raw, _ = mixture
    out, pinned = _run(cells, raw, rho_lo=0.0, rho_hi=1e-3)
    assert pinned and "upper clip bound" in str(pinned[0].message)
    assert out.uns["ambient"]["envelope_diagnostics"]["frac_at_rho_hi"] > 0.5
