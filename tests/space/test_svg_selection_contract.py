from __future__ import annotations

import warnings
import sys
import types

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from omicverse.space._svg import _select_significant_svg_names, svg


def test_qvalue_selection_never_promotes_nonsignificant_genes():
    qvals = pd.Series(
        [0.001, 0.2, np.nan, 0.01],
        index=["g1", "g2", "g3", "g4"],
    )

    selected = _select_significant_svg_names(
        qvals,
        n_svgs=100,
        qval_threshold=0.05,
    )

    assert selected.tolist() == ["g1", "g4"]


def test_qvalue_selection_validates_threshold():
    with pytest.raises(ValueError, match="qval_threshold"):
        _select_significant_svg_names(
            pd.Series([0.1], index=["g1"]),
            n_svgs=1,
            qval_threshold=1.1,
        )


def test_pearson_residual_mode_discloses_that_it_is_not_spatial(monkeypatch):
    adata = AnnData(
        X=np.array(
            [
                [1, 0, 2],
                [0, 2, 1],
                [3, 1, 0],
                [1, 1, 1],
            ],
            dtype=np.float32,
        )
    )
    adata.var_names = ["g1", "g2", "g3"]
    adata.obsm["spatial"] = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        dtype=np.float64,
    )

    def fake_preprocess(data, **kwargs):
        data.var["highly_variable"] = [True, False, True]
        return data

    monkeypatch.setattr("omicverse.pp.preprocess", fake_preprocess)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = svg(adata, mode="pearson_residuals", n_svgs=2)

    assert result.var["space_variable_features"].tolist() == [True, False, True]
    assert result.uns["space_svg_runs"]["pearson_residuals"]["spatial_evidence"] is False
    assert any("does not use spatial coordinates" in str(item.message) for item in caught)


def _spatial_adata_for_svg_backends():
    adata = AnnData(
        X=np.array(
            [
                [1, 2, 3],
                [2, 1, 2],
                [3, 2, 1],
                [2, 3, 2],
            ],
            dtype=np.float32,
        )
    )
    adata.var_names = ["g1", "g2", "g3"]
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["spatial"] = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        dtype=np.float64,
    )
    return adata


def test_somde_branch_applies_qvalue_threshold(monkeypatch):
    class FakeSomNode:
        def __init__(self, coords, k):
            self.genes = None

        def reTrain(self, epochs):
            return None

        def mtx(self, frame):
            self.genes = list(frame.index)
            return None, None

        def norm(self):
            return None

        def run(self, n_jobs=1):
            return (
                pd.DataFrame(
                    {
                        "g": self.genes,
                        "LLR": [5.0, 1.0, 4.0],
                        "pval": [0.001, 0.4, 0.01],
                        "qval": [0.002, 0.5, 0.02],
                        "FSV": [0.8, 0.1, 0.7],
                    }
                ),
                2,
            )

    fake_somde = types.ModuleType("omicverse.external.somde")
    fake_somde.SomNode = FakeSomNode
    monkeypatch.setitem(sys.modules, "omicverse.external.somde", fake_somde)

    result = svg(
        _spatial_adata_for_svg_backends(),
        mode="somde",
        n_svgs=3,
        qval_threshold=0.05,
    )

    assert result.var_names[result.var["space_variable_features"]].tolist() == ["g1", "g3"]


def test_spatialde_branch_applies_qvalue_threshold(monkeypatch):
    fake_spatialde = types.ModuleType("omicverse.external.SpatialDE")

    def fake_run(coords, expression, **kwargs):
        return pd.DataFrame(
            {
                "g": list(expression.columns),
                "LLR": [5.0, 1.0, 4.0],
                "pval": [0.001, 0.4, 0.01],
                "qval": [0.002, 0.5, 0.02],
                "FSV": [0.8, 0.1, 0.7],
                "l": [1.0, 1.0, 1.0],
            }
        )

    fake_spatialde.run = fake_run
    fake_naivede = types.ModuleType("omicverse.external.NaiveDE")
    fake_naivede.stabilize = lambda matrix: matrix
    fake_naivede.regress_out = lambda sample_info, matrix, formula: matrix
    monkeypatch.setitem(sys.modules, "omicverse.external.SpatialDE", fake_spatialde)
    monkeypatch.setitem(sys.modules, "omicverse.external.NaiveDE", fake_naivede)

    result = svg(
        _spatial_adata_for_svg_backends(),
        mode="spatialde",
        n_svgs=3,
        qval_threshold=0.05,
        show_progress=False,
    )

    assert result.var_names[result.var["space_variable_features"]].tolist() == ["g1", "g3"]
