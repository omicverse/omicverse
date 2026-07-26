"""Tests for Monocle 2-style residual model correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from omicverse.external.monocle2_py.dimension_reduction import (
    _residualize_model_effects,
    reduce_dimension,
)
from omicverse.single import Monocle


def test_residualize_categorical_batch_preserves_intercept_and_biology():
    """Batch offsets are removed while a balanced biological contrast remains."""
    obs = pd.DataFrame(
        {
            "batch": ["a", "a", "b", "b"],
            "state": ["early", "late", "early", "late"],
        },
        index=[f"c{i}" for i in range(4)],
    )
    # gene 0: batch-only; gene 1: state-only; gene 2: both effects
    FM = np.array(
        [
            [1.0, 1.0, 11.0, 11.0],
            [2.0, 8.0, 2.0, 8.0],
            [3.0, 9.0, 13.0, 19.0],
        ]
    )

    adjusted, metadata = _residualize_model_effects(FM, obs, "~ batch")

    np.testing.assert_allclose(adjusted[0], np.repeat(1.0, 4), atol=1e-12)
    np.testing.assert_allclose(adjusted[1], FM[1], atol=1e-12)
    np.testing.assert_allclose(adjusted[2], [3.0, 9.0, 3.0, 9.0], atol=1e-12)
    assert metadata["design_columns"] == ["Intercept", "batch[T.b]"]
    assert metadata["design_rank"] == 2


def test_residualize_multiple_terms_matches_vectorized_ols():
    obs = pd.DataFrame(
        {
            "batch": ["a", "a", "b", "b", "c", "c"],
            "qc": [0.0, 1.0, 0.5, 1.5, 1.0, 2.0],
        },
        index=[f"c{i}" for i in range(6)],
    )
    FM = np.array(
        [
            [2.0, 3.0, 7.5, 8.5, 13.0, 14.0],
            [5.0, 4.0, 6.0, 5.0, 7.0, 6.0],
        ]
    )

    adjusted, metadata = _residualize_model_effects(FM, obs, "~ batch + qc")

    # Residualized expression must have no linear association with any
    # non-intercept design column, up to floating-point tolerance.
    from patsy import dmatrix

    design = np.asarray(dmatrix("~ batch + qc", obs), dtype=float)
    for column in range(1, design.shape[1]):
        np.testing.assert_allclose(
            design[:, column] @ adjusted.T,
            design[:, column].sum() * adjusted.mean(axis=1),
            atol=1e-10,
        )
    assert metadata["n_terms"] == 4


def test_residual_model_rejects_missing_values():
    obs = pd.DataFrame({"batch": ["a", None, "b"]})
    FM = np.ones((2, 3))

    with pytest.raises(ValueError, match="Invalid residualModelFormulaStr"):
        _residualize_model_effects(FM, obs, "~ batch")


def test_residual_model_requires_intercept():
    obs = pd.DataFrame({"batch": ["a", "a", "b", "b"]})
    FM = np.ones((2, 4))

    with pytest.raises(ValueError, match="must include an intercept"):
        _residualize_model_effects(FM, obs, "~ 0 + batch")


def test_residual_model_rejects_rank_deficient_design():
    obs = pd.DataFrame(
        {
            "batch": ["a", "a", "b", "b"],
            "duplicate": ["a", "a", "b", "b"],
        }
    )
    FM = np.ones((2, 4))

    with pytest.raises(ValueError, match="rank-deficient"):
        _residualize_model_effects(FM, obs, "~ batch + duplicate")


def test_residual_model_drops_least_squares_roundoff_as_zero_variance():
    adata = AnnData(
        X=np.array([[1.0], [1.0], [11.0], [11.0]]),
        obs=pd.DataFrame(
            {"batch": ["a", "a", "b", "b"]},
            index=[f"c{i}" for i in range(4)],
        ),
    )
    adata.var["use_for_ordering"] = True

    with pytest.raises(
        ValueError,
        match="zero variance after applying residualModelFormulaStr",
    ):
        reduce_dimension(
            adata,
            reduction_method="ICA",
            norm_method="none",
            pseudo_expr=0,
            residualModelFormulaStr="~ batch",
        )


def test_public_monocle_api_records_residual_model(small_branching_adata):
    adata = small_branching_adata.copy()
    adata.obs["batch"] = np.where(
        np.arange(adata.n_obs) % 2 == 0, "batch_a", "batch_b"
    )
    mono = Monocle(adata)
    mono.preprocess().select_ordering_genes()
    mono.reduce_dimension(
        reduction_method="ICA",
        random_state=0,
        residualModelFormulaStr="~ batch",
    )

    metadata = mono.adata.uns["monocle"]["residual_model"]
    assert metadata["formula"] == "~ batch"
    assert metadata["design_columns"] == ["Intercept", "batch[T.batch_b]"]
    assert metadata["n_cells"] == adata.n_obs
    assert "X_ICA" in mono.adata.obsm


def test_reducing_again_without_model_clears_residual_metadata(
    small_branching_adata,
):
    adata = small_branching_adata.copy()
    adata.obs["batch"] = np.where(
        np.arange(adata.n_obs) % 2 == 0, "batch_a", "batch_b"
    )
    mono = Monocle(adata)
    mono.preprocess().select_ordering_genes()
    mono.reduce_dimension(
        reduction_method="ICA",
        random_state=0,
        residualModelFormulaStr="~ batch",
    )
    assert "residual_model" in mono.adata.uns["monocle"]

    mono.reduce_dimension(reduction_method="ICA", random_state=0)

    assert "residual_model" not in mono.adata.uns["monocle"]
