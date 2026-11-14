"""Tests for ``ov.space.niche`` — the spatial and molecular niche flavours.

The clustering is Leiden over a feature matrix, run through omicverse's own
``ov.pp.neighbors`` / ``ov.pp.leiden`` (no scanpy). ``_cluster_rows`` must
ask for scanpy's igraph flavour (two iterations, undirected) so labels match
the earlier scanpy-backed release.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from omicverse import space
from omicverse.space import _niche


def _lattice(n_side: int = 10, step: float = 5.0, seed: int = 0) -> AnnData:
    """A small grid with two spatially segregated cell types."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float) * step
    labels = np.where(coords[:, 0] < coords[:, 0].mean(), "left", "right")

    adata = AnnData(rng.poisson(5, size=(len(coords), 6)).astype(np.float32))
    adata.obsm["spatial"] = coords
    adata.obs["cell_type"] = pd.Categorical(labels)
    space.spatial_neighbors(adata, n_neighs=6, coord_type="grid")
    return adata


# --------------------------------------------------------------------------- #
# the Leiden call
# --------------------------------------------------------------------------- #
def test_cluster_rows_uses_igraph_flavour(monkeypatch):
    import omicverse.pp as ovpp

    seen: dict = {}

    def leiden(adata, *, resolution=1.0, key_added="leiden", random_state=0, **kw):
        seen.update(kw)
        adata.obs[key_added] = pd.Categorical(["0"] * adata.n_obs)

    monkeypatch.setattr(ovpp, "leiden", leiden)
    rng = np.random.default_rng(0)
    labels = _niche._cluster_rows(rng.random((40, 5)), resolution=1.0, seed=0,
                                  n_neighbors=10)
    assert len(labels) == 40
    assert seen == {"flavor": "igraph", "n_iterations": 2, "directed": False}


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_neighborhood_labels_every_spot():
    adata = _lattice()
    space.niche.neighborhood(adata, "cell_type", resolution=0.5, key_added="nb")
    assert adata.obs["nb"].notna().all()
    assert adata.obsm["nb_composition"].shape == (adata.n_obs, 2)


def test_utag_labels_every_spot():
    adata = _lattice()
    space.niche.utag(adata, resolution=0.5, key_added="ut")
    assert adata.obs["ut"].notna().all()
    assert adata.obsm["ut_smoothed"].shape == adata.shape


# --------------------------------------------------------------------------- #
# molecular niche: the returned object is a dataclass, not a plotter
# --------------------------------------------------------------------------- #
def test_molecular_returns_a_plain_dataclass():
    """The registered example used to call a method that does not exist."""
    rng = np.random.default_rng(0)
    adata = AnnData(rng.poisson(3, size=(30, 6)).astype(np.float32))
    adata.obsm["q05_cell_abundance_w_sf"] = rng.exponential(1.0, size=(30, 5))

    zones = space.niche.molecular(adata, n_factors=3)
    assert zones.factor_loadings.shape == (5, 3)
    assert zones.spot_activations.shape == (30, 3)
    assert set(zones.factor_top_cell_types) == set(zones.factor_names)
    # what the registry says it writes, it writes
    assert adata.obsm["X_tissue_zones"].shape == (30, 3)
    assert "X_tissue_zones" in adata.uns["tissue_zones"]
