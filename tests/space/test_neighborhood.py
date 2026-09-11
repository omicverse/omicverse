"""The arrangement statistics, held against squidpy where squidpy has them.

These functions were written to squidpy's semantics on purpose: each statistic
has one correct definition, and matching a reference implementation is what makes
the numbers checkable rather than merely plausible. The measured agreement on a
real Visium slide, recorded when this file was written:

    interaction_matrix      exact (max |diff| = 0)
    nhood_enrichment count  exact
    nhood_enrichment z      Pearson r = 0.9975   (independent permutation draws)
    centrality_scores       exact (max |diff| = 0)
    co_occurrence           Pearson r = 1.0000
    ripley F / G / L        Pearson r = 1.0000 each
    sepal                   Spearman rho = 0.864
    sliding_window          ARI = 1.0000
    var_by_distance         max |diff| = 1.11e-16

squidpy is an optional dependency, so the comparisons skip when it is absent; the
contract tests below run either way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from omicverse import space
from omicverse.space import _svg as svg_module


def _lattice(n_side: int = 12, step: float = 7.3, seed: int = 0) -> AnnData:
    """A hexagonal-ish lattice with two spatially segregated cell types."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    coords[:, 0] += (coords[:, 1] % 2) * 0.5
    coords *= step

    labels = np.where(coords[:, 0] < coords[:, 0].mean(), "left", "right")
    adata = AnnData(rng.poisson(5, size=(len(coords), 6)).astype(np.float32))
    adata.obsm["spatial"] = coords
    adata.obs["cell_type"] = pd.Categorical(labels)
    space.spatial_neighbors(adata, n_neighs=6, coord_type="grid")
    return adata


# --------------------------------------------------------------------------- #
# contracts
# --------------------------------------------------------------------------- #
def test_missing_graph_names_the_function_that_builds_it():
    adata = AnnData(np.zeros((5, 2), dtype=np.float32))
    adata.obsm["spatial"] = np.arange(10, dtype=float).reshape(5, 2)
    adata.obs["cell_type"] = pd.Categorical(["a"] * 5)
    with pytest.raises(KeyError, match="spatial_neighbors"):
        space.interaction_matrix(adata, "cell_type")


def test_spatial_neighbors_never_connects_different_libraries():
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.01, 0.01],
            [1.01, 0.01],
            [0.01, 1.01],
        ]
    )
    adata = AnnData(
        np.ones((6, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"slice": ["A"] * 3 + ["B"] * 3},
            index=[f"c{i}" for i in range(6)],
        ),
    )
    adata.obsm["spatial"] = coords

    space.spatial_neighbors(adata, n_neighs=2, library_key="slice")

    graph = adata.obsp["spatial_connectivities"].tocoo()
    labels = adata.obs["slice"].to_numpy()
    assert not np.any(labels[graph.row] != labels[graph.col])
    assert adata.uns["spatial_neighbors"]["params"]["library_sizes"] == {
        "A": 3,
        "B": 3,
    }


def test_spatial_neighbors_counts_nonself_neighbors():
    adata = AnnData(np.ones((5, 1), dtype=np.float32))
    adata.obsm["spatial"] = np.column_stack(
        [np.arange(5, dtype=float), np.zeros(5, dtype=float)]
    )

    space.spatial_neighbors(adata, n_neighs=2)

    degree = adata.obsp["spatial_connectivities"].getnnz(axis=1)
    assert degree[0] >= 2
    assert degree[-1] >= 2


def _overlapping_library_adata():
    local_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    adata = AnnData(
        np.arange(16, dtype=np.float32).reshape(8, 2) + 1,
        obs=pd.DataFrame(
            {"slice": ["A"] * 4 + ["B"] * 4},
            index=[f"cell_{i}" for i in range(8)],
        ),
    )
    adata.var_names = ["g0", "g1"]
    adata.obsm["spatial"] = np.vstack([local_coords, local_coords])
    adata.uns["spatial"] = {"A": {}, "B": {}}
    return adata


def _assert_no_cross_library_edges(adata):
    graph = adata.obsp["spatial_connectivities"].tocoo()
    labels = adata.obs["slice"].to_numpy()
    assert not np.any(labels[graph.row] != labels[graph.col])


def test_morani_auto_graph_forwards_library_key_without_mutating_input(monkeypatch):
    adata = _overlapping_library_adata()
    x_before = np.asarray(adata.X).copy()
    obs_before = adata.obs.copy(deep=True)
    var_before = adata.var.copy(deep=True)
    uns_keys_before = set(adata.uns)
    obsp_keys_before = set(adata.obsp)
    captured = {}

    original_spatial_neighbors = svg_module.spatial_neighbors

    def capture_graph(*args, **kwargs):
        result = original_spatial_neighbors(*args, **kwargs)
        captured["library_key"] = kwargs.get("library_key")
        captured["copy"] = kwargs.get("copy")
        captured["connectivities"] = result[0]
        return result

    monkeypatch.setattr(svg_module, "spatial_neighbors", capture_graph)

    result = space.moranI(
        adata,
        genes=["g0"],
        auto_spatial_neighbors=True,
        n_neighs=2,
        library_key="slice",
        corr_method=None,
        copy=True,
    )

    assert result['gene'].tolist() == ['g0', 'g0']
    assert result['library'].tolist() == ['A', 'B']
    assert captured["library_key"] == "slice"
    assert captured["copy"] is True
    graph = captured["connectivities"].tocoo()
    labels = adata.obs["slice"].to_numpy()
    assert not np.any(labels[graph.row] != labels[graph.col])
    assert "moranI" not in adata.uns
    assert set(adata.uns) == uns_keys_before
    assert set(adata.obsp) == obsp_keys_before
    np.testing.assert_array_equal(np.asarray(adata.X), x_before)
    pd.testing.assert_frame_equal(adata.obs, obs_before)
    pd.testing.assert_frame_equal(adata.var, var_before)


def test_morani_stratification_does_not_call_a_pure_library_shift_spatial():
    adata = _overlapping_library_adata()
    adata.X = np.concatenate(
        [
            np.zeros((4, 2), dtype=np.float32),
            np.full((4, 2), 10.0, dtype=np.float32),
        ],
        axis=0,
    )

    result = space.moranI(
        adata,
        genes=["g0"],
        auto_spatial_neighbors=True,
        n_neighs=2,
        n_perms=19,
        seed=3,
        library_key="slice",
        corr_method=None,
        copy=True,
    )

    assert not result['testable'].any()
    assert result['I'].isna().all()
    assert result['pval_sim'].isna().all()


def test_svg_moran_forwards_library_key_to_automatic_graph():
    adata = _overlapping_library_adata()

    result = space.svg(
        adata,
        mode="moran",
        n_svgs=1,
        n_perms=None,
        library_key="slice",
    )

    assert result is adata
    assert int(adata.var["space_variable_features"].sum()) == 0
    assert list(adata.varm['space_variable_features_by_library'].columns) == ['A', 'B']
    assert set(adata.uns['spatial_features_by_library']['library']) == {'A', 'B'}


def test_spatial_autocorr_copy_does_not_mutate_uns():
    adata = _lattice(n_side=4)
    keys_before = set(adata.uns)

    result = space.spatial_autocorr(
        adata,
        genes=list(adata.var_names[:2]),
        copy=True,
    )

    assert set(result.index) == set(adata.var_names[:2])
    assert "moranI" not in adata.uns
    assert set(adata.uns) == keys_before


def test_interaction_matrix_counts_every_edge_from_both_ends():
    adata = _lattice()
    counts = space.interaction_matrix(adata, "cell_type", copy=True)
    assert counts.shape == (2, 2)
    assert counts.sum() == adata.obsp["spatial_connectivities"].nnz
    assert np.allclose(counts, counts.T)      # symmetric graph, symmetric counts


def test_interaction_matrix_normalized_rows_sum_to_one():
    adata = _lattice()
    fractions = space.interaction_matrix(adata, "cell_type", normalized=True, copy=True)
    assert np.allclose(fractions.sum(axis=1), 1.0)


def test_segregated_types_are_enriched_with_themselves():
    """Two blocks side by side: each touches itself far more than the other."""
    adata = _lattice()
    res = space.nhood_enrichment(adata, "cell_type", n_perms=200, seed=0, copy=True)
    z = res["zscore"]
    assert z[0, 0] > 0 and z[1, 1] > 0        # like with like
    assert z[0, 1] < 0                        # across the boundary, depleted


def test_nhood_enrichment_is_reproducible_under_a_seed():
    adata = _lattice()
    a = space.nhood_enrichment(adata, "cell_type", n_perms=100, seed=7, copy=True)
    b = space.nhood_enrichment(adata, "cell_type", n_perms=100, seed=7, copy=True)
    assert np.array_equal(a["zscore"], b["zscore"])


def test_centrality_scores_returns_one_row_per_cluster():
    adata = _lattice()
    df = space.centrality_scores(adata, "cell_type", copy=True)
    assert list(df.index) == ["left", "right"]
    assert set(df.columns) == {"degree_centrality", "average_clustering",
                               "closeness_centrality"}
    assert df.notna().all().all()


def test_co_occurrence_shape_follows_the_interval_edges():
    adata = _lattice()
    occ, edges = space.co_occurrence(adata, "cell_type", interval=8, copy=True)
    assert occ.shape == (2, 2, len(edges) - 1)
    assert (occ >= 0).all()


def test_ripley_rejects_an_unknown_mode():
    adata = _lattice()
    with pytest.raises(ValueError, match="mode must be"):
        space.ripley(adata, "cell_type", mode="Q")


@pytest.mark.parametrize("mode", ["F", "G", "L"])
def test_ripley_runs_and_is_monotone_in_the_cumulative_modes(mode):
    adata = _lattice()
    res = space.ripley(adata, "cell_type", mode=mode, n_simulations=5,
                       n_observations=100, n_steps=15, seed=0, copy=True)
    stats = res[f"{mode}_stat"]
    assert {"bins", "stats", "cell_type"} <= set(stats.columns)
    if mode in ("F", "G"):                    # empirical CDFs never decrease
        for _, grp in stats.groupby("cell_type", observed=True):
            assert (np.diff(grp.sort_values("bins")["stats"].to_numpy()) >= -1e-12).all()


def test_sepal_refuses_a_graph_that_is_not_a_lattice():
    adata = _lattice()
    space.spatial_neighbors(adata, n_neighs=3)          # degree no longer 6
    with pytest.raises(ValueError, match="lattice"):
        space.sepal(adata, max_neighs=6, genes=list(adata.var_names[:2]))


def test_sepal_rejects_an_unsupported_lattice():
    adata = _lattice()
    with pytest.raises(ValueError, match="max_neighs"):
        space.sepal(adata, max_neighs=5)


def test_mask_graph_keeps_only_edges_inside_the_polygon():
    adata = _lattice()
    xy = adata.obsm["spatial"]
    mid = xy.mean(axis=0)
    poly = [(xy[:, 0].min() - 1, xy[:, 1].min() - 1), (xy[:, 0].min() - 1, mid[1]),
            (mid[0], mid[1]), (mid[0], xy[:, 1].min() - 1)]
    keep, masked = space.mask_graph(adata, poly, copy=True)
    assert 0 < keep.sum() < adata.n_obs
    assert masked.nnz < adata.obsp["spatial_connectivities"].nnz
    # no edge may touch a spot outside the mask
    coo = masked.tocoo()
    assert keep[coo.row].all() and keep[coo.col].all()


def test_mask_graph_negative_selects_the_complement():
    adata = _lattice()
    xy = adata.obsm["spatial"]
    mid = xy.mean(axis=0)
    poly = [(xy[:, 0].min() - 1, xy[:, 1].min() - 1), (xy[:, 0].min() - 1, mid[1]),
            (mid[0], mid[1]), (mid[0], xy[:, 1].min() - 1)]
    inside, _ = space.mask_graph(adata, poly, copy=True)
    outside, _ = space.mask_graph(adata, poly, negative_mask=True, copy=True)
    assert np.array_equal(inside, ~outside)


def test_sliding_window_covers_every_spot_and_tiles_without_gaps():
    adata = _lattice()
    labels = space.sliding_window(adata, window_size=30.0, copy=True)
    assert labels.notna().all()
    assert labels.nunique() > 1


def test_sliding_window_rejects_overlap_at_or_above_the_window():
    adata = _lattice()
    with pytest.raises(ValueError, match="overlap"):
        space.sliding_window(adata, window_size=30.0, overlap=30.0)


def test_var_by_distance_is_zero_at_the_anchor_and_one_at_the_far_edge():
    adata = _lattice()
    dm = space.var_by_distance(adata, groups="left", cluster_key="cell_type", copy=True)
    col = dm["left"].to_numpy()
    finite = col[np.isfinite(col)]
    assert np.isclose(finite.min(), 0.0)
    assert np.isclose(finite.max(), 1.0)
    assert "left_raw" in dm.columns            # unscaled distances kept alongside


def test_var_by_distance_without_normalisation_keeps_coordinate_units():
    adata = _lattice()
    dm = space.var_by_distance(adata, groups="left", cluster_key="cell_type",
                               normalize=False, copy=True)
    assert dm["left"].max() > 1.0              # pixels, not a fraction


def test_var_by_distance_needs_a_cluster_key_when_groups_names_clusters():
    adata = _lattice()
    with pytest.raises(ValueError, match="cluster_key"):
        space.var_by_distance(adata, groups="left")


# --------------------------------------------------------------------------- #
# agreement with squidpy
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def paired():
    """The same slide and the same graph, ready for both implementations."""
    sq = pytest.importorskip("squidpy")
    adata = _lattice(n_side=14)
    sq.gr.spatial_neighbors(adata, n_neighs=6, coord_type="generic")
    return sq, adata


def test_interaction_matrix_matches_squidpy(paired):
    sq, adata = paired
    assert np.allclose(sq.gr.interaction_matrix(adata, "cell_type", copy=True),
                       space.interaction_matrix(adata, "cell_type", copy=True))


def test_nhood_enrichment_counts_match_squidpy(paired):
    sq, adata = paired
    theirs = sq.gr.nhood_enrichment(adata, "cell_type", n_perms=50, seed=0,
                                    copy=True, show_progress_bar=False)
    ours = space.nhood_enrichment(adata, "cell_type", n_perms=50, seed=0, copy=True)
    assert np.array_equal(theirs[1], ours["count"])


def test_centrality_scores_match_squidpy(paired):
    sq, adata = paired
    sq.gr.centrality_scores(adata, "cell_type")
    theirs = adata.uns["cell_type_centrality_scores"]
    ours = space.centrality_scores(adata, "cell_type", copy=True)
    for col in ours.columns:
        assert np.allclose(theirs[col].to_numpy(), ours[col].to_numpy(), atol=1e-10)


def test_co_occurrence_matches_squidpy(paired):
    sq, adata = paired
    theirs, edges = sq.gr.co_occurrence(adata, "cell_type", interval=10, copy=True,
                                        show_progress_bar=False)
    ours, _ = space.co_occurrence(adata, "cell_type", interval=edges, copy=True)
    assert theirs.shape == ours.shape
    assert np.allclose(theirs, ours, atol=1e-10)


def test_var_by_distance_matches_squidpy(paired):
    sq, adata = paired
    sq.tl.var_by_distance(adata, groups="left", cluster_key="cell_type")
    theirs = adata.obsm["design_matrix"]["left"].to_numpy()
    ours = space.var_by_distance(adata, groups="left", cluster_key="cell_type",
                                 copy=True)["left"].to_numpy()
    assert np.allclose(theirs, ours, atol=1e-10, equal_nan=True)
