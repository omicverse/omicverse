"""``method='envelope'``: when to cross-fit the soup, and small lineages.

Cross-fitting the panel (#1030) has two halves. Ranking a group's panel on the
other fold of cells fixes the hgmm collapse (chance-zero genes read rho = 0)
and stays on. Ranking on one binomial half of the empty droplets' counts and
reading against the other half only works when the empties measure the soup
well. With few, shallow empties over many genes (several preparations, a few
hundred ~30-UMI empties each) a half-sample ranking is mostly sampling noise:
the panel stops being a lower envelope, carries native expression, and the
level was over-estimated by ~1.5x -- the regression that failed a
three-preparation benchmark after #1030. ``soup_crossfit='auto'`` measures
how much of a panel's soup share replicates in the independent half
(:func:`soup_selection_stability`) and splits only when it does.

The second regression of #1030: a rare lineage of a few dozen cells is ranked
on half of its cells, so the panel picks genes it expresses and its
contamination is over-stated. The same lineage's cells in the other batches
are independent of the read fold; ranking on them too fixes it.

Both regimes are simulated here with a known truth. The single-preparation,
deep-empties, disjoint-genome regime (hgmm) is
``test_envelope_low_contamination.py``.
"""
from __future__ import annotations

import warnings

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from omicverse.pp.ambient._envelope import run_envelope


def _multiprep(seed=0, n_genes=2000, n_prep=3, cells_per_prep=500,
               n_empty=100, empty_umi=30, rare_frac=0.03):
    """Preparations with different soups, no population-private gene, and a
    rare population (the last one) at ``rare_frac`` of every preparation."""
    rng = np.random.default_rng(seed)
    n_pop = 5
    base = rng.lognormal(0.0, 1.0, n_genes)
    native = base * rng.lognormal(0.0, 0.4, (n_pop + 1, n_genes))
    for k in range(n_pop + 1):
        m = rng.choice(n_genes, 40, replace=False)
        native[k, m] *= rng.uniform(20, 60, 40)
    native /= native.sum(1, keepdims=True)
    comp = rng.dirichlet(np.ones(n_pop) * 2, n_prep)
    soup = comp @ native[:n_pop] * rng.lognormal(0, 0.3, (n_prep, n_genes))
    soup /= soup.sum(1, keepdims=True)

    X, Xn, obs = [], [], []
    for j in range(n_prep):
        p = np.r_[comp[j] * (1 - rare_frac), rare_frac]
        pop = rng.choice(n_pop + 1, cells_per_prep, p=p)
        depth = rng.lognormal(np.log(2500), 0.5, cells_per_prep).astype(int)
        rho = rng.beta(4, 22, cells_per_prep)
        for c in range(cells_per_prep):
            n_amb = rng.binomial(depth[c], rho[c])
            nat = rng.multinomial(depth[c] - n_amb, native[pop[c]])
            X.append(nat + rng.multinomial(n_amb, soup[j]))
            Xn.append(nat)
            obs.append((f"p{j}_c{c:04d}", f"p{j}", f"pop{pop[c]}"))
    obs = pd.DataFrame(obs, columns=["bc", "prep", "population"]).set_index("bc")
    X, Xn = np.asarray(X, float), np.asarray(Xn, float)
    obs["true_rho"] = 1 - Xn.sum(1) / X.sum(1)
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    cells = anndata.AnnData(sp.csr_matrix(X.astype(np.float32)), obs=obs, var=var)

    E, eo = [], []
    for j in range(n_prep):
        for e in range(n_empty):
            n = max(int(rng.lognormal(np.log(empty_umi), 0.5)), 1)
            E.append(rng.multinomial(n, soup[j]))
            eo.append((f"p{j}_e{e:05d}", f"p{j}"))
    empties = anndata.AnnData(
        sp.csr_matrix(np.asarray(E, np.float32)),
        obs=pd.DataFrame(eo, columns=["bc", "prep"]).set_index("bc"),
        var=var.copy())
    raw = anndata.concat([cells, empties])
    raw.obs["prep"] = list(obs["prep"]) + list(empties.obs["prep"])
    return cells, raw


def _run(cells, raw, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C, rho, _, meta = run_envelope(cells.copy(), raw=raw, batch_key="prep",
                                       **kw)
    return rho, meta["envelope_diagnostics"]


@pytest.fixture(scope="module")
def shallow():
    return _multiprep(seed=0)


@pytest.fixture(scope="module")
def deep():
    return _multiprep(seed=0, n_empty=3000, empty_umi=200)


def test_auto_keeps_full_soup_when_empties_are_shallow(shallow):
    """Few shallow empties: the split ranking is noise; auto must not split,
    and its level must not be the inflated one the split produces."""
    cells, raw = shallow
    truth = cells.obs["true_rho"].mean()
    rho_auto, d_auto = _run(cells, raw, level_shrink=1.0)
    rho_split, d_split = _run(cells, raw, level_shrink=1.0,
                              soup_crossfit=True)
    assert d_auto["soup_selection_stability"] < 0.8
    assert d_auto["soup_crossfit"] is False and d_split["soup_crossfit"] is True
    # the split's panel carries native expression -> level well above truth
    assert rho_split.mean() > 1.3 * truth
    assert rho_split.mean() > 1.3 * rho_auto.mean()
    assert abs(rho_auto.mean() / truth - 1.0) < 0.25
    # the cell cross-fit stays on: nothing pinned
    assert d_auto["frac_at_rho_lo"] < 0.05


def test_auto_splits_the_soup_when_empties_measure_it(deep):
    """Same cells and soups, many deep empties: the panel's soup share
    replicates in the independent half, so the soup is cross-fitted."""
    cells, raw = deep
    rho, d = _run(cells, raw)
    assert d["soup_selection_stability"] >= 0.8
    assert d["soup_crossfit"] is True
    # well-measured soup: splitting costs nothing, both readings agree
    full, _ = _run(cells, raw, soup_crossfit=False)
    assert abs(rho.mean() / full.mean() - 1.0) < 0.1


def test_soup_crossfit_false_matches_auto_on_shallow(shallow):
    cells, raw = shallow
    a, _ = _run(cells, raw)
    b, d = _run(cells, raw, soup_crossfit=False)
    np.testing.assert_allclose(a, b)
    assert np.isnan(d["soup_selection_stability"])
    with pytest.raises(ValueError, match="soup_crossfit"):
        _run(cells, raw, soup_crossfit="sometimes")


def test_rare_lineage_borrows_its_other_batches(shallow):
    """A rare lineage (~15 cells per preparation) ranked on half its own
    cells is over-estimated; ranking on its cells in the other preparations
    as well brings it in line with everyone else."""
    cells, raw = shallow
    truth = cells.obs["true_rho"].to_numpy()
    rare = cells.obs["population"].to_numpy() == "pop5"
    assert 20 < rare.sum() < 0.05 * cells.n_obs

    def rel(rho):  # rare cells' estimate/truth relative to all cells'
        return (rho[rare].mean() / truth[rare].mean()) / (rho.mean() / truth.mean())

    on, _ = _run(cells, raw, cluster_key="population")
    off, _ = _run(cells, raw, cluster_key="population", borrow_lineage=False)
    assert 0.8 < rel(on) < 1.2
    assert rel(off) > rel(on) + 0.2
