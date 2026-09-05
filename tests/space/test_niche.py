"""Tests for ``ov.space.niche`` — the spatial and molecular niche flavours.

The interesting part here is not the clustering, which is Leiden over a
feature matrix and is covered by scanpy's own tests. It is the call into
scanpy: ``sc.tl.leiden(..., flavor='igraph')`` needs scanpy >= 1.10, while
omicverse pins ``scanpy>=1.9``, so on a valid 1.9 install the whole module
used to raise ``TypeError``. The keyword is now passed only when the
signature has it, and both arms are exercised below with a stub — the
installed scanpy can only ever be one of the two.
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
# the scanpy version split
# --------------------------------------------------------------------------- #
def _stub_leiden(with_flavor: bool, seen: dict):
    """A stand-in for ``sc.tl.leiden`` with or without the 1.10 ``flavor=``."""
    if with_flavor:
        def leiden(adata, *, resolution=1.0, key_added="leiden", random_state=0,
                   flavor="leidenalg", n_iterations=-1, directed=None):
            seen.update(flavor=flavor, n_iterations=n_iterations,
                        directed=directed)
            adata.obs[key_added] = pd.Categorical(["0"] * adata.n_obs)
    else:
        # scanpy 1.9: no `flavor`, so passing it is a TypeError
        def leiden(adata, *, resolution=1.0, key_added="leiden", random_state=0,
                   n_iterations=-1, directed=True):
            seen.update(n_iterations=n_iterations, directed=directed)
            adata.obs[key_added] = pd.Categorical(["0"] * adata.n_obs)
    return leiden


def test_flavor_is_passed_when_scanpy_understands_it():
    seen: dict = {}
    kwargs = _niche._leiden_flavor_kwargs(_stub_leiden(True, seen))
    assert kwargs == {"flavor": "igraph", "n_iterations": 2, "directed": False}


def test_flavor_is_dropped_on_scanpy_1_9():
    """The keyword is a 1.10 addition; omicverse pins >=1.9, so it has to go."""
    seen: dict = {}
    assert _niche._leiden_flavor_kwargs(_stub_leiden(False, seen)) == {}


@pytest.mark.parametrize("with_flavor", [True, False])
def test_cluster_rows_runs_on_both_scanpy_generations(monkeypatch, with_flavor):
    """``_cluster_rows`` must not raise on either signature.

    Before the fix the 1.9 arm raised ``TypeError: leiden() got an unexpected
    keyword argument 'flavor'``, which took ``niche.neighborhood`` and
    ``niche.utag`` down with it.
    """
    import scanpy as sc

    seen: dict = {}
    monkeypatch.setattr(sc.tl, "leiden", _stub_leiden(with_flavor, seen))

    rng = np.random.default_rng(0)
    labels = _niche._cluster_rows(rng.random((40, 5)), resolution=1.0, seed=0,
                                  n_neighbors=10)
    assert len(labels) == 40
    assert ("flavor" in seen) is with_flavor


# --------------------------------------------------------------------------- #
# end to end, against whatever scanpy is actually installed
# --------------------------------------------------------------------------- #
def test_neighborhood_labels_every_spot():
    adata = _lattice()
    space.niche.neighborhood(adata, "cell_type", resolution=0.5, key_added="nb")
    assert adata.obs["nb"].notna().all()
    assert adata.obsm["nb_composition"].shape == (adata.n_obs, 2)


def test_neighborhood_does_not_count_missing_labels_as_first_category(monkeypatch):
    from scipy import sparse

    adata = AnnData(np.ones((3, 1), dtype=np.float32))
    adata.obs["cell_type"] = pd.Categorical(
        [np.nan, "A", "B"],
        categories=["A", "B"],
    )
    adata.obsp["spatial_connectivities"] = sparse.csr_matrix(
        np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=float,
        )
    )
    captured = {}

    def fake_cluster_rows(matrix, resolution, seed, n_neighbors):
        captured["matrix"] = matrix.copy()
        return np.zeros(matrix.shape[0], dtype=int)

    monkeypatch.setattr(_niche, "_cluster_rows", fake_cluster_rows)
    space.niche.neighborhood(
        adata,
        "cell_type",
        normalize=False,
        scale=False,
        key_added="nb_missing",
    )

    assert np.array_equal(captured["matrix"][1], [0, 0])


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
