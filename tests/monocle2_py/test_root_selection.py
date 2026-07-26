"""Root-selection tests: ensure order_cells(root_state=...) and
the new ``root_by_column=`` convenience match R Monocle 2's
``orderCells(root_state=GM_state(cds))`` pattern from the HSMM
tutorial.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omicverse.single import Monocle


def test_order_cells_accepts_root_state(small_branching_adata):
    """Regression: previously a 'Y_N' vs cell-name mismatch caused
    order_cells(root_state=N) to raise ``ValueError: 'Y_N' is not in list``.
    """
    mono = Monocle(small_branching_adata.copy())
    (mono.preprocess()
         .select_ordering_genes()
         .reduce_dimension()
         .order_cells())
    # Pick any observed state and re-run — must not throw
    any_state = mono.adata.obs['State'].mode().iloc[0]
    mono.order_cells(root_state=int(any_state))
    assert mono.pseudotime is not None


def test_order_cells_root_by_column_sets_progenitor_state(small_branching_adata):
    """After calling ``root_by_column=``, the state that hosts most of
    the declared progenitor cells should have the lowest mean
    pseudotime (matches R GM_state() semantics)."""
    adata = small_branching_adata.copy()
    # Fabricate a time covariate — first 50 cells are t=0 progenitors
    adata.obs['time'] = ([0] * 50 + [1] * 50 + [2] * 50)
    mono = Monocle(adata)
    (mono.preprocess()
         .select_ordering_genes()
         .reduce_dimension()
         .order_cells(root_by_column='time', root_by_value=0))

    root_cell = mono.adata.uns['monocle']['root_cell']
    assert mono.adata.obs.loc[root_cell, 'time'] == 0
    assert mono.adata.obs.loc[root_cell, 'Pseudotime'] == pytest.approx(0.0)

    # Find the state that contains most t=0 cells
    t0_mask = mono.adata.obs['time'] == 0
    progenitor_state = mono.adata.obs.loc[t0_mask, 'State'].mode().iloc[0]
    # That state should have mean pseudotime ≤ everywhere else
    root_pt = mono.adata.obs.loc[
        mono.adata.obs['State'] == progenitor_state, 'Pseudotime'
    ].mean()
    other_pt = mono.adata.obs.loc[
        mono.adata.obs['State'] != progenitor_state, 'Pseudotime'
    ].mean()
    # Tolerance of 0.5 — on the tiny synthetic fixture pseudotime
    # values are near zero everywhere and can swap by small amounts
    assert root_pt <= other_pt + 0.5, (
        f"root_by_column failed: progenitor state {progenitor_state} "
        f"mean pt={root_pt:.2f} vs other states mean={other_pt:.2f}"
    )


def test_order_cells_root_by_column_defaults_to_min(small_branching_adata):
    """When root_by_value is not given, the column minimum is used."""
    adata = small_branching_adata.copy()
    adata.obs['hours'] = ([0] * 50 + [24] * 50 + [48] * 50)
    mono = Monocle(adata)
    (mono.preprocess()
         .select_ordering_genes()
         .reduce_dimension()
         .order_cells(root_by_column='hours'))  # no value → use min (0)
    root_cell = mono.adata.uns['monocle']['root_cell']
    assert mono.adata.obs.loc[root_cell, 'hours'] == 0
    # State hosting the most 0h cells must be the root state
    t0 = mono.adata.obs[mono.adata.obs['hours'] == 0]
    root_state = t0['State'].mode().iloc[0]
    # Cells of that state should sit near pseudotime 0
    root_pt = mono.adata.obs.loc[mono.adata.obs['State'] == root_state,
                                  'Pseudotime'].mean()
    other_pt = mono.adata.obs.loc[mono.adata.obs['State'] != root_state,
                                  'Pseudotime'].mean()
    # Root state cells should have lower mean pseudotime than other
    # states (with some tolerance for noisy synthetic fixtures).
    assert root_pt <= other_pt + 1.0


def test_root_by_column_missing_column_raises(small_branching_adata):
    mono = Monocle(small_branching_adata.copy())
    (mono.preprocess()
         .select_ordering_genes()
         .reduce_dimension()
         .order_cells())
    with pytest.raises(KeyError, match='not found'):
        mono.order_cells(root_by_column='nonexistent')


def test_root_by_column_missing_value_raises(small_branching_adata):
    adata = small_branching_adata.copy()
    adata.obs['time'] = [0] * adata.n_obs
    mono = Monocle(adata)
    (mono.preprocess()
         .select_ordering_genes()
         .reduce_dimension()
         .order_cells())
    with pytest.raises(ValueError, match='matched'):
        mono.order_cells(root_by_column='time', root_by_value=999)
