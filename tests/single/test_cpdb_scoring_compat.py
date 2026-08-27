from types import SimpleNamespace

import pandas as pd
import pytest

from omicverse.single._cpdb import _call_cpdb_analysis


def test_cpdb_scoring_deduplicates_genes() -> None:
    def build_scoring_matrix() -> pd.DataFrame:
        return pd.DataFrame(
            [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]],
            index=["NRXN1", "NRXN1", "CXCL12"],
            columns=["EVT", "dNK1"],
        )

    scoring_utils = SimpleNamespace(
        heteromer_geometric_expression_per_cell_type=build_scoring_matrix
    )
    original = scoring_utils.heteromer_geometric_expression_per_cell_type

    def call(**kwargs):
        matrix = scoring_utils.heteromer_geometric_expression_per_cell_type()
        # CellPhoneDB transposes this matrix before downstream dataframe
        # consumers inspect the columns.
        assert matrix.T.columns.is_unique
        return matrix

    analysis_method = SimpleNamespace(scoring_utils=scoring_utils, call=call)

    result = _call_cpdb_analysis(analysis_method, {})

    assert result.index.tolist() == ["NRXN1", "CXCL12"]
    assert result.loc["NRXN1"].tolist() == [1.0, 2.0]
    assert (
        scoring_utils.heteromer_geometric_expression_per_cell_type is original
    )


def test_cpdb_scoring_restores_patch_on_error() -> None:
    def build_scoring_matrix() -> pd.DataFrame:
        return pd.DataFrame([[1.0]], index=["NRXN1"], columns=["EVT"])

    scoring_utils = SimpleNamespace(
        heteromer_geometric_expression_per_cell_type=build_scoring_matrix
    )
    original = scoring_utils.heteromer_geometric_expression_per_cell_type

    def call(**kwargs):
        scoring_utils.heteromer_geometric_expression_per_cell_type()
        raise RuntimeError("analysis failed")

    analysis_method = SimpleNamespace(scoring_utils=scoring_utils, call=call)

    with pytest.raises(RuntimeError, match="analysis failed"):
        _call_cpdb_analysis(analysis_method, {})

    assert (
        scoring_utils.heteromer_geometric_expression_per_cell_type is original
    )
