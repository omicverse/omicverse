"""CPU-only contract tests for shaoleishen/pycellchat 0.1.0 result adaptation.

The synthetic tensors deliberately have asymmetric sender/receiver entries and
unsorted groups/L-R identifiers. No pycellchat installation or inference is needed.
Schema source: https://github.com/shaoleishen/pycellchat/tree/
697bbc8212331283a3db2dbc4a7481691da708df/python/pycellchat
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from omicverse.single import (
    extract_comm_adata,
    format_pycellchat_results,
    to_comm_adata,
)


@pytest.fixture
def cellchat_result():
    prob = np.arange(27, dtype=float).reshape(3, 3, 3) / 27
    # Include significance at zero, a boundary value, and nonsignificant edges.
    pval = np.arange(27, dtype=float).reshape(3, 3, 3) / 100
    pval[0, 1, 0] = 0.01
    pval[1, 0, 0] = 0.8
    return {
        "net": {"prob": prob, "pval": pval},
        "idents": {
            "names": ["T", "B", "Myeloid"],
            "codes": np.array([1, 0, 2, 1]),
            "column": "celltype",
        },
        "LR": {
            "LRsig": pd.DataFrame(
                {
                    "interaction_name": [
                        "MIF_CD74_CXCR4",
                        "CXCL13_CXCR5",
                        "TNF_TNFRSF1A",
                    ],
                    "interaction_name_2": [
                        "MIF - (CD74+CXCR4)",
                        "CXCL13 - CXCR5",
                        "TNF - TNFRSF1A",
                    ],
                    "ligand": ["MIF", "CXCL13", "TNF"],
                    "receptor": ["CD74_CXCR4", "CXCR5", "TNFRSF1A"],
                    "pathway_name": ["MIF", "CXCL", "TNF"],
                    "annotation": ["Secreted Signaling"] * 3,
                    "evidence": ["source A", "source B", "source C"],
                },
                index=["db42", "db5", "db17"],
            ),
            "interaction_name": ["MIF_CD74_CXCR4", "CXCL13_CXCR5", "TNF_TNFRSF1A"],
        },
        "options": {
            "mode": "single",
            "datatype": "RNA",
            "parameter": {"nboot": 100, "seed": 1},
        },
    }


def test_values_and_tensor_axis_order(cellchat_result):
    comm = format_pycellchat_results(cellchat_result)
    assert comm.shape == (9, 3)
    groups = cellchat_result["idents"]["names"]
    names = cellchat_result["LR"]["interaction_name"]
    assert comm.obs_names.tolist() == [f"{s}|{r}" for s in groups for r in groups]
    assert comm.var_names.tolist() == names
    for i, sender in enumerate(groups):
        for j, receiver in enumerate(groups):
            row = comm.obs_names.get_loc(f"{sender}|{receiver}")
            assert comm.obs.iloc[row]["sender"] == sender
            assert comm.obs.iloc[row]["receiver"] == receiver
            for k, interaction in enumerate(names):
                col = comm.var_names.get_loc(interaction)
                assert comm.X[row, col] == cellchat_result["net"]["prob"][i, j, k]
                assert (
                    comm.layers["means"][row, col]
                    == cellchat_result["net"]["prob"][i, j, k]
                )
                assert (
                    comm.layers["pvalues"][row, col]
                    == cellchat_result["net"]["pval"][i, j, k]
                )
    assert comm.layers["pvalues"][0, 0] == 0  # Do not invent p=1 for zero probability.
    assert comm.layers["pvalues"][3, 0] == 0.8  # Do not discard nonsignificant entries.


def test_metadata_and_no_input_mutation(cellchat_result):
    before = deepcopy(cellchat_result)
    comm = format_pycellchat_results(cellchat_result)
    assert comm.var["classification"].tolist() == ["MIF", "CXCL", "TNF"]
    assert comm.var["gene_a"].tolist() == ["MIF", "CXCL13", "TNF"]
    assert comm.var["gene_b"].tolist() == ["CD74_CXCR4", "CXCR5", "TNFRSF1A"]
    assert comm.var["annotation"].tolist() == ["Secreted Signaling"] * 3
    assert comm.var["evidence"].tolist() == ["source A", "source B", "source C"]
    assert comm.var.iloc[0]["interaction_name_2"] == "MIF - (CD74+CXCR4)"
    assert comm.uns["comm_source"] == "pycellchat"
    assert comm.uns["support_kind"] == "pvalue"
    assert comm.uns["pvalues_are_statistical"] is True
    assert comm.uns["pycellchat_groupby"] == "celltype"
    assert comm.uns["pycellchat_options"] == before["options"]
    for tensor in ("prob", "pval"):
        np.testing.assert_array_equal(
            cellchat_result["net"][tensor], before["net"][tensor]
        )
    pd.testing.assert_frame_equal(cellchat_result["LR"]["LRsig"], before["LR"]["LRsig"])
    comm.X[:] = 7
    comm.layers["means"][:] = 8
    comm.layers["pvalues"][:] = 0.9
    comm.var.iloc[0, comm.var.columns.get_loc("evidence")] = "changed"
    comm.uns["pycellchat_options"]["parameter"]["seed"] = 5
    for tensor in ("prob", "pval"):
        np.testing.assert_array_equal(
            cellchat_result["net"][tensor], before["net"][tensor]
        )
    pd.testing.assert_frame_equal(cellchat_result["LR"]["LRsig"], before["LR"]["LRsig"])
    assert cellchat_result["options"] == before["options"]


@pytest.mark.parametrize(
    "kind", ["mapping", "object", "adata", "explicit_adata", "custom_uns", "alias"]
)
def test_generic_dispatch(cellchat_result, kind):
    adata = AnnData(np.ones((4, 1)))
    adata.uns["cellchat"] = cellchat_result
    if kind == "mapping":
        comm = to_comm_adata(data=cellchat_result)
    elif kind == "object":
        comm = to_comm_adata(data=SimpleNamespace(cc=cellchat_result))
    elif kind == "adata":
        comm = to_comm_adata(adata)
    elif kind == "explicit_adata":
        comm = to_comm_adata(data=adata)
    elif kind == "custom_uns":
        adata.uns["custom_result"] = adata.uns.pop("cellchat")
        comm = to_comm_adata(adata, result_uns_key="custom_result")
    else:
        comm = extract_comm_adata(adata)
    assert comm.shape == (9, 3)
    assert comm.uns["comm_source"] == "pycellchat"
    assert to_comm_adata(comm) is comm
    assert "means" not in adata.layers
    assert adata.shape == (4, 1)


@pytest.mark.parametrize("kind", ["adata", "object"])
def test_direct_formatter_unwraps_result(cellchat_result, kind):
    adata = AnnData(np.ones((4, 1)))
    adata.uns["cellchat"] = cellchat_result
    value = adata if kind == "adata" else SimpleNamespace(cc=cellchat_result)
    np.testing.assert_array_equal(
        format_pycellchat_results(value).X, cellchat_result["net"]["prob"].reshape(9, 3)
    )


@pytest.mark.parametrize("n_groups,n_lr", [(0, 0), (0, 3), (3, 0)])
def test_empty_tensors(cellchat_result, n_groups, n_lr):
    cellchat_result["net"] = {
        key: np.zeros((n_groups, n_groups, n_lr)) for key in ("prob", "pval")
    }
    cellchat_result["idents"]["names"] = cellchat_result["idents"]["names"][:n_groups]
    cellchat_result["LR"]["LRsig"] = cellchat_result["LR"]["LRsig"].iloc[:n_lr]
    cellchat_result["LR"]["interaction_name"] = cellchat_result["LR"][
        "interaction_name"
    ][:n_lr]
    comm = format_pycellchat_results(cellchat_result)
    assert comm.shape == (n_groups**2, n_lr)
    assert comm.layers["pvalues"].shape == comm.shape


@pytest.mark.parametrize("key", ["prob", "pval"])
def test_missing_tensors_are_not_fabricated(cellchat_result, key):
    del cellchat_result["net"][key]
    with pytest.raises(ValueError, match="both 'prob' and 'pval'"):
        format_pycellchat_results(cellchat_result)


@pytest.mark.parametrize("shape", [(3, 3), (3, 2, 3), (3, 3, 2)])
def test_tensor_shape_mismatch(cellchat_result, shape):
    cellchat_result["net"]["pval"] = np.zeros(shape)
    with pytest.raises(ValueError, match="3D|same shape"):
        format_pycellchat_results(cellchat_result)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.01, 1.01])
def test_invalid_pvalues(cellchat_result, value):
    cellchat_result["net"]["pval"][0, 0, 0] = value
    with pytest.raises(ValueError, match="finite real|valid range"):
        format_pycellchat_results(cellchat_result)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.01])
def test_invalid_probabilities(cellchat_result, value):
    cellchat_result["net"]["prob"][0, 0, 0] = value
    with pytest.raises(ValueError, match="finite real|valid range"):
        format_pycellchat_results(cellchat_result)


@pytest.mark.parametrize("key", ["prob", "pval"])
def test_nonnumeric_tensors(cellchat_result, key):
    cellchat_result["net"][key] = cellchat_result["net"][key].astype(str)
    with pytest.raises(ValueError, match="finite real"):
        format_pycellchat_results(cellchat_result)


@pytest.mark.parametrize(
    "names,match",
    [
        (["T", "B"], "dimensions"),
        (["T", "T", "B"], "unique"),
        ([1, "1", "B"], "unique"),
        (["T", None, "B"], "missing"),
        (["T", " ", "B"], "empty"),
        ("TBM", "one-dimensional"),
        ([["T", "B", "M"]], "one-dimensional"),
    ],
)
def test_invalid_group_names(cellchat_result, names, match):
    cellchat_result["idents"]["names"] = names
    with pytest.raises(ValueError, match=match):
        format_pycellchat_results(cellchat_result)


def test_missing_group_names(cellchat_result):
    del cellchat_result["idents"]["names"]
    with pytest.raises(ValueError, match="requires 'names'"):
        format_pycellchat_results(cellchat_result)


def test_lr_count_must_match(cellchat_result):
    cellchat_result["LR"]["LRsig"] = cellchat_result["LR"]["LRsig"].iloc[:2]
    with pytest.raises(ValueError, match="row count"):
        format_pycellchat_results(cellchat_result)


def test_lr_order_must_match_redundant_names(cellchat_result):
    cellchat_result["LR"]["LRsig"] = cellchat_result["LR"]["LRsig"].iloc[::-1]
    with pytest.raises(ValueError, match="interaction order"):
        format_pycellchat_results(cellchat_result)


@pytest.mark.parametrize("column", ["ligand", "receptor"])
def test_required_lr_columns(cellchat_result, column):
    del cellchat_result["LR"]["LRsig"][column]
    with pytest.raises(ValueError, match="missing required columns"):
        format_pycellchat_results(cellchat_result)


@pytest.mark.parametrize("column", ["interaction_name", "ligand", "receptor"])
def test_missing_lr_labels(cellchat_result, column):
    cellchat_result["LR"]["LRsig"].iloc[
        0, cellchat_result["LR"]["LRsig"].columns.get_loc(column)
    ] = None
    with pytest.raises(ValueError, match="missing labels"):
        format_pycellchat_results(cellchat_result)


def test_duplicate_interaction_names(cellchat_result):
    cellchat_result["LR"]["LRsig"]["interaction_name"] = [
        "duplicate",
        "duplicate",
        "other",
    ]
    with pytest.raises(ValueError, match="unique"):
        format_pycellchat_results(cellchat_result)


def test_distinct_interactions_can_share_ligand_receptor(cellchat_result):
    cellchat_result["LR"]["LRsig"]["ligand"] = "MIF"
    cellchat_result["LR"]["LRsig"]["receptor"] = "CD74_CXCR4"
    comm = format_pycellchat_results(cellchat_result)
    assert comm.n_vars == 3
    assert comm.var_names.is_unique


def test_index_fallback_without_interaction_name(cellchat_result):
    del cellchat_result["LR"]["LRsig"]["interaction_name"]
    cellchat_result["LR"]["interaction_name"] = []
    comm = format_pycellchat_results(cellchat_result)
    assert comm.var_names.tolist() == ["db42", "db5", "db17"]
    cellchat_result["LR"]["LRsig"].index = ["same", "same", "different"]
    with pytest.raises(ValueError, match="unique"):
        format_pycellchat_results(cellchat_result)


def test_pathway_fallback(cellchat_result):
    cellchat_result["LR"]["LRsig"]["pathway_name"] = [None, "", "TNF"]
    assert format_pycellchat_results(cellchat_result).var[
        "classification"
    ].tolist() == ["Unclassified", "Unclassified", "TNF"]
    del cellchat_result["LR"]["LRsig"]["pathway_name"]
    assert (
        format_pycellchat_results(cellchat_result).var["classification"]
        == "Unclassified"
    ).all()


def test_separator_validation(cellchat_result):
    cellchat_result["idents"]["names"][0] = "T|memory"
    with pytest.raises(ValueError, match="choose another separator"):
        format_pycellchat_results(cellchat_result)
    comm = to_comm_adata(data=cellchat_result, separator="::")
    assert comm.obs_names[0] == "T|memory::T|memory"
    for separator in ("", None, 1):
        with pytest.raises(ValueError, match="nonempty string"):
            format_pycellchat_results(cellchat_result, separator=separator)


@pytest.mark.parametrize("key", ["net", "LR", "idents"])
def test_missing_namespaces(cellchat_result, key):
    del cellchat_result[key]
    with pytest.raises(ValueError, match="mapping"):
        format_pycellchat_results(cellchat_result)


def test_merged_results_rejected(cellchat_result):
    cellchat_result["options"]["mode"] = "merged"
    with pytest.raises(ValueError, match="single-dataset"):
        format_pycellchat_results(cellchat_result)


def test_h5ad_roundtrip(cellchat_result, tmp_path):
    comm = format_pycellchat_results(cellchat_result)
    path = tmp_path / "pycellchat_comm.h5ad"
    comm.write_h5ad(path)
    restored = read_h5ad(path)
    np.testing.assert_array_equal(restored.layers["means"], comm.layers["means"])
    np.testing.assert_array_equal(restored.layers["pvalues"], comm.layers["pvalues"])
    pd.testing.assert_frame_equal(restored.obs, comm.obs)
    pd.testing.assert_frame_equal(restored.var, comm.var)
    assert restored.uns["comm_source"] == "pycellchat"
    assert restored.uns["pycellchat_options"] == comm.uns["pycellchat_options"]


@pytest.mark.parametrize("separator", ["|", "::"])
def test_ccc_filtering_and_plot_smoke(cellchat_result, separator):
    from omicverse.pl import ccc_heatmap, ccc_network_plot, ccc_stat_plot
    from omicverse.pl._ccc import _communication_long_table

    if separator == "::":
        cellchat_result["idents"]["names"][2] = "Myeloid|resident"
    comm = format_pycellchat_results(cellchat_result, separator=separator)
    table = _communication_long_table(comm)
    row = table[
        (table["sender"] == "T")
        & (table["receiver"] == "B")
        & (table["ligand"] == "MIF")
    ]
    assert len(row) == 1
    assert row.iloc[0]["score"] == cellchat_result["net"]["prob"][0, 1, 0]
    assert row.iloc[0]["pvalue"] == 0.01
    assert not (
        (table["sender"] == "B")
        & (table["receiver"] == "T")
        & (table["ligand"] == "MIF")
    ).any()
    try:
        for plotter, plot_type in [
            (ccc_heatmap, "dot"),
            (ccc_network_plot, "circle"),
            (ccc_stat_plot, "bar"),
        ]:
            fig, ax = plotter(
                comm, plot_type=plot_type, display_by="interaction", top_n=3, show=False
            )
            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)
            fig.canvas.draw()
    finally:
        plt.close("all")


@pytest.mark.parametrize("key", ["net", "LR", "idents", "options"])
def test_wrong_namespace_types(cellchat_result, key):
    cellchat_result[key] = []
    with pytest.raises(TypeError, match="mapping"):
        format_pycellchat_results(cellchat_result)


def test_invalid_result_type():
    with pytest.raises(TypeError, match="Expected a pycellchat"):
        format_pycellchat_results(object())


def test_invalid_lr_table(cellchat_result):
    cellchat_result["LR"]["LRsig"] = []
    with pytest.raises(TypeError, match="DataFrame"):
        format_pycellchat_results(cellchat_result)


def test_duplicate_lr_columns(cellchat_result):
    frame = cellchat_result["LR"]["LRsig"]
    cellchat_result["LR"]["LRsig"] = pd.concat([frame, frame[["ligand"]]], axis=1)
    with pytest.raises(ValueError, match="unique column"):
        format_pycellchat_results(cellchat_result)


def test_nonsquare_tensors(cellchat_result):
    for key in ("prob", "pval"):
        cellchat_result["net"][key] = cellchat_result["net"][key][:, :2, :]
    with pytest.raises(ValueError, match="sender/receiver dimensions"):
        format_pycellchat_results(cellchat_result)


def test_fortran_order_arrays_keep_axis_semantics(cellchat_result):
    expected = cellchat_result["net"]["prob"].reshape(9, 3).copy()
    for key in ("prob", "pval"):
        cellchat_result["net"][key] = np.asfortranarray(cellchat_result["net"][key])
    np.testing.assert_array_equal(
        format_pycellchat_results(cellchat_result).X, expected
    )


def test_optional_metadata_and_numeric_groups(cellchat_result):
    del cellchat_result["options"]
    del cellchat_result["idents"]["column"]
    del cellchat_result["LR"]["interaction_name"]
    cellchat_result["idents"]["names"] = [3, 1, 2]
    comm = format_pycellchat_results(cellchat_result)
    assert comm.obs_names[1] == "3|1"
    assert comm.uns["pycellchat_options"] == {}
    assert "pycellchat_groupby" not in comm.uns


def test_multichar_separator_cannot_create_pair_name_collisions(cellchat_result):
    cellchat_result["idents"]["names"] = ["a:", "b", "a", ":b"]
    cellchat_result["net"] = {"prob": np.ones((4, 4, 3)), "pval": np.zeros((4, 4, 3))}
    # Neither label contains '::', but both a: -> b and a -> :b
    # would receive the same observation name, 'a:::b'.
    with pytest.raises(ValueError, match="names collide; choose another separator"):
        format_pycellchat_results(cellchat_result, separator="::")
