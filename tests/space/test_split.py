from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
import pytest
from scipy import sparse


def _split_fixture(sparse_counts=False):
    cell_types = pd.Index(["T_A", "T_B", "T_C"])
    genes = pd.Index([f"Gene_{i}" for i in range(6)])
    cells = pd.Index([f"Cell_{i}" for i in range(12)])
    reference = pd.DataFrame(
        np.array(
            [
                [10, 2, 1, 1, 0, 0],
                [1, 9, 2, 1, 0, 1],
                [0, 1, 8, 2, 1, 1],
            ],
            dtype=float,
        ),
        index=cell_types,
        columns=genes,
    )
    primary = pd.Series(np.repeat(cell_types.to_numpy(), 4), index=cells)
    secondary = pd.Series(
        ["T_B", "T_C", "T_B", "T_C", "T_A", "T_C", "T_A", "T_C", "T_A", "T_B", "T_A", "T_B"],
        index=cells,
    )
    weights = pd.DataFrame(0.0, index=cells, columns=cell_types)
    for cell in cells:
        weights.loc[cell, primary.loc[cell]] = 0.75
        weights.loc[cell, secondary.loc[cell]] = 0.25
    contaminated = weights.to_numpy() @ reference.to_numpy()
    clean_target = (
        0.75 * reference.loc[primary].to_numpy()
        + 1e-10 / (weights.to_numpy() > 0).sum(axis=1)[:, None]
    )
    counts = sparse.csr_matrix(contaminated) if sparse_counts else contaminated.copy()
    adata = ad.AnnData(
        X=counts.copy(),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=genes),
    )
    adata.layers["counts"] = counts.copy()
    adata.layers["clean_target"] = clean_target
    adata.obsm["spatial"] = np.column_stack([
        np.repeat(np.arange(4), 3),
        np.tile(np.arange(3), 4),
    ]).astype(float)
    return adata, weights, reference, primary, secondary


def _manual_split(counts, weights, reference, primary):
    counts_arr = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    w = weights.to_numpy(dtype=float)
    ref = reference.to_numpy(dtype=float)
    pidx = reference.index.get_indexer(primary.to_numpy())
    primary_weight = w[np.arange(w.shape[0]), pidx]
    n_types = np.maximum((w > 0).sum(axis=1), 1)
    denom = w @ ref + 1e-10
    numer = primary_weight[:, None] * ref[pidx] + 1e-10 / n_types[:, None]
    return counts_arr * numer / denom


def test_split_purify_matches_reference_formula_dense():
    import omicverse as ov

    adata, weights, reference, primary, _ = _split_fixture()
    ov.space.split_purify(adata, weights, reference, primary_cell_type=primary)
    expected = _manual_split(adata.layers["counts"], weights, reference, primary)
    assert np.allclose(adata.layers["split_purified"], expected, rtol=1e-10, atol=1e-10)
    assert list(adata.obs["first_type"]) == list(primary)
    assert set(adata.obs["purification_status"]) == {"purified"}


def test_split_purify_accepts_sparse_and_infers_primary():
    import omicverse as ov

    dense, weights, reference, primary, _ = _split_fixture()
    sparse_adata, _, _, _, _ = _split_fixture(sparse_counts=True)
    ov.space.split_purify(dense, weights, reference, primary_cell_type=primary)
    ov.space.split_purify(sparse_adata, weights, reference)
    assert sparse.issparse(sparse_adata.layers["split_purified"])
    assert np.allclose(
        sparse_adata.layers["split_purified"].toarray(),
        dense.layers["split_purified"],
    )


def test_split_purify_copy_mode_and_cells_to_purify():
    import omicverse as ov

    adata, weights, reference, primary, _ = _split_fixture()
    copied = ov.space.split_purify(
        adata,
        weights,
        reference.T,
        primary_cell_type=primary,
        cells_to_purify=adata.obs_names[:4],
        copy=True,
    )
    assert "split_purified" not in adata.layers
    assert "split_purified" in copied.layers
    assert np.allclose(copied.layers["split_purified"][4:], copied.layers["counts"][4:])
    assert set(copied.obs.iloc[4:]["purification_status"]) == {"raw"}


def test_split_spatial_score_writes_diffusion_metrics():
    import omicverse as ov

    adata, weights, _, primary, secondary = _split_fixture()
    ov.space.split_spatial_score(
        adata,
        weights,
        primary,
        secondary_cell_type=secondary,
        k=3,
        radius=2.0,
    )
    scores = adata.obs["neighborhood_weights_second_type"].to_numpy()
    assert "split_spatial_neighbors" in adata.uns
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert adata.uns["split_spatial_neighbors"]["indices"].shape[0] == adata.n_obs


def test_split_balance_selects_purified_profiles_by_score_and_spot_class():
    import omicverse as ov

    adata, weights, reference, primary, secondary = _split_fixture()
    ov.space.split_purify(adata, weights, reference, primary_cell_type=primary)
    ov.space.split_spatial_score(adata, weights, primary, secondary_cell_type=secondary, k=3)
    adata.obs["neighborhood_weights_second_type"] = 0.0
    adata.obs.iloc[:3, adata.obs.columns.get_loc("neighborhood_weights_second_type")] = 0.5
    adata.obs["spot_class"] = "singlet"
    adata.obs.iloc[3, adata.obs.columns.get_loc("spot_class")] = "doublet_uncertain"
    adata.obs.iloc[4, adata.obs.columns.get_loc("spot_class")] = "reject"
    ov.space.split_balance(adata, threshold=0.15, spot_class_key="spot_class")
    balanced = adata.layers["split_balanced"].toarray()
    purified = adata.layers["split_purified"]
    raw = adata.layers["counts"]
    assert np.allclose(balanced[:4], purified[:4])
    assert np.allclose(balanced[5:], raw[5:])
    assert np.allclose(balanced[4], 0.0)
    assert adata.obs["split_balance_status"].iloc[4] == "removed"


def test_split_balance_can_mark_split_shift_swaps():
    import omicverse as ov

    adata, weights, reference, primary, _ = _split_fixture()
    ov.space.split_purify(adata, weights, reference, primary_cell_type=primary)
    adata.obs["neighborhood_weights_second_type"] = 1.0
    adata.obs["first_type_neighborhood"] = adata.obs["second_type"].astype(str)
    ov.space.split_balance(adata, threshold=0.15, swap_labels=True)
    assert adata.obs["split_shift_swap"].all()


def test_split_reassign_residuals_preserves_nonnegative_counts_and_stats():
    import omicverse as ov

    adata, weights, reference, primary, _ = _split_fixture(sparse_counts=True)
    ov.space.split_purify(adata, weights, reference, primary_cell_type=primary)
    adata.obs["neighborhood_weights_second_type"] = 1.0
    ov.space.split_balance(adata, threshold=0.15)
    ov.space.split_reassign_residuals(adata, k=3, mode="uniform", self_keep=0.25)
    reassigned = adata.layers["split_reassigned"]
    assert sparse.issparse(reassigned)
    assert reassigned.data.min() >= 0
    assert "split_reassignment_operator" in adata.uns
    assert adata.uns["split_residual_stats"]["self_keep"] == 0.25


def test_purified_counts_are_closer_to_clean_target_than_contaminated_counts():
    import omicverse as ov

    adata, weights, reference, primary, _ = _split_fixture()
    ov.space.split_purify(adata, weights, reference, primary_cell_type=primary)
    contaminated = np.asarray(adata.layers["counts"])
    purified = np.asarray(adata.layers["split_purified"])
    clean = np.asarray(adata.layers["clean_target"])
    assert np.linalg.norm(purified - clean) < np.linalg.norm(contaminated - clean)


def test_split_rejects_invalid_inputs():
    import omicverse as ov

    adata, weights, reference, primary, secondary = _split_fixture()
    bad_weights = weights.copy()
    bad_weights.iloc[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        ov.space.split_purify(adata, bad_weights, reference, primary_cell_type=primary)

    bad_secondary = secondary.copy()
    bad_secondary.iloc[0] = "missing_type"
    with pytest.raises(ValueError, match="secondary_cell_type"):
        ov.space.split_spatial_score(adata, weights, primary, secondary_cell_type=bad_secondary)

    with pytest.raises(ValueError, match="radius"):
        ov.space.split_spatial_score(adata, weights, primary, radius=-1)
