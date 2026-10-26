"""``method='envelope'`` must refuse a ``raw`` it cannot line up with the cells.

Empty droplets are "raw barcodes that are not called cells", so a raw matrix
whose barcodes do not contain the called cells used to turn every raw row --
cells included -- into an empty droplet, and the soup profile silently became
the cells' own expression. Genes were taken positionally the same way.
"""
from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest

from omicverse.pp.ambient._envelope import _empties_from_raw


GENES = ["g0", "g1", "g2", "g3"]


def _pair(cell_names=("c0", "c1", "c2"), raw_names=None, raw_genes=GENES):
    rng = np.random.default_rng(0)
    cells = anndata.AnnData(
        X=rng.poisson(5, (3, 4)).astype(float),
        obs=pd.DataFrame(index=list(cell_names)),
        var=pd.DataFrame(index=GENES))
    raw_names = list(raw_names or list(cell_names) + ["e0", "e1"])
    X = np.vstack([cells.X, rng.poisson(1, (len(raw_names) - 3, 4))]).astype(float)
    raw = anndata.AnnData(
        X=X, obs=pd.DataFrame(index=raw_names), var=pd.DataFrame(index=GENES))
    return cells, raw[:, list(raw_genes)].copy()


def _empties(cells, raw):
    E, _ = _empties_from_raw(cells, raw, batch_key=None, empty_batch=None,
                             n_batch=1, uniq=np.array([0]))
    return np.asarray(E)


def test_matching_raw_keeps_only_the_empty_droplets():
    cells, raw = _pair()
    assert _empties(cells, raw).shape == (2, 4)


def test_disjoint_barcodes_raise_instead_of_using_cells_as_soup():
    cells, raw = _pair(raw_names=["r0", "r1", "r2", "e0", "e1"])
    with pytest.raises(ValueError, match="none of the called cells"):
        _empties(cells, raw)


def test_partially_missing_cells_warn():
    cells, raw = _pair(raw_names=["c0", "c1", "x2", "e0", "e1"])
    with pytest.warns(UserWarning, match="1 of 3 called cells"):
        _empties(cells, raw)


def test_raw_gene_order_is_aligned_to_adata():
    cells, raw = _pair()
    _, shuffled = _pair(raw_genes=["g3", "g1", "g0", "g2"])
    np.testing.assert_array_equal(_empties(cells, shuffled), _empties(cells, raw))


def test_raw_missing_genes_raise():
    cells, raw = _pair(raw_genes=["g0", "g1", "g2"])
    with pytest.raises(ValueError, match="genes of adata are missing"):
        _empties(cells, raw)
