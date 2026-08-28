from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sys
import warnings
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path


matplotlib.use("Agg")

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _ccc_plot_data import (
    make_comm_adata as build_comm_adata,
    make_comm_adata_shifted as build_comm_adata_shifted,
    make_comm_adata_with_receiver_only_group as build_comm_adata_with_receiver_only_group,
)

try:
    import omicverse as ov
    import omicverse.pl._ccc as ccc_mod
    import omicverse.pl._cpdbviz as cpdbviz_mod
except Exception as exc:  # pragma: no cover - environment guard for optional deps
    ov = None
    ccc_mod = None
    cpdbviz_mod = None
    pytestmark = pytest.mark.skip(reason=f"omicverse import failed in test env: {exc}")

@pytest.fixture()
def comm_adata() -> AnnData:
    return build_comm_adata()


@pytest.fixture()
def comparison_comm_adata() -> AnnData:
    return build_comm_adata_shifted()


@pytest.fixture()
def comm_adata_with_receiver_only_group() -> AnnData:
    return build_comm_adata_with_receiver_only_group()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _assert_figure_and_axes(fig, ax) -> None:
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def _build_comm_adata_with_duplicate_pairs() -> AnnData:
    obs = pd.DataFrame(
        {
            "sender": ["A", "A", "B"],
            "receiver": ["B", "B", "A"],
        },
        index=["A|B#1", "A|B#2", "B|A#1"],
    )
    var = pd.DataFrame(
        {
            "interacting_pair": ["L1_R1", "L2_R2"],
            "interaction_name": ["L1_R1", "L2_R2"],
            "interaction_name_2": ["L1 - R1", "L2 - R2"],
            "classification": ["TNF", "TNF"],
            "gene_a": ["L1", "L2"],
            "gene_b": ["R1", "R2"],
        },
        index=["L1_R1", "L2_R2"],
    )
    means = np.array(
        [
            [1.0, 0.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ],
        dtype=float,
    )
    pvalues = np.array(
        [
            [0.01, 0.20],
            [0.03, 0.04],
            [0.01, 0.01],
        ],
        dtype=float,
    )
    adata = AnnData(X=means.copy(), obs=obs, var=var)
    adata.layers["means"] = means.copy()
    adata.layers["pvalues"] = pvalues.copy()
    return adata


def _build_comm_adata_with_high_nonsignificant_scores() -> AnnData:
    obs = pd.DataFrame(
        {
            "sender": ["A", "A", "B", "B"],
            "receiver": ["A", "B", "A", "B"],
        },
        index=["A|A", "A|B", "B|A", "B|B"],
    )
    var = pd.DataFrame(
        {
            "interacting_pair": ["L1_R1", "L2_R2"],
            "interaction_name": ["L1_R1", "L2_R2"],
            "interaction_name_2": ["L1 - R1", "L2 - R2"],
            "classification": ["Pathway", "Pathway"],
            "gene_a": ["L1", "L2"],
            "gene_b": ["R1", "R2"],
        },
        index=["L1_R1", "L2_R2"],
    )
    means = np.array(
        [
            [1.0, 100.0],
            [2.0, 200.0],
            [3.0, 300.0],
            [4.0, 400.0],
        ],
        dtype=float,
    )
    pvalues = np.array(
        [
            [0.01, 0.50],
            [0.02, 0.50],
            [0.50, 0.50],
            [0.50, 0.50],
        ],
        dtype=float,
    )
    adata = AnnData(X=means.copy(), obs=obs, var=var)
    adata.layers["means"] = means.copy()
    adata.layers["pvalues"] = pvalues.copy()
    return adata


def _build_dense_pathway_comm_adata(n_cell_types: int = 5) -> AnnData:
    cell_types = [f"Cell{i}" for i in range(n_cell_types)]
    obs = pd.DataFrame(
        [
            {"sender": sender, "receiver": receiver}
            for sender in cell_types
            for receiver in cell_types
        ],
        index=[f"{sender}|{receiver}" for sender in cell_types for receiver in cell_types],
    )
    var = pd.DataFrame(
        {
            "interacting_pair": ["L1_R1"],
            "interaction_name": ["L1_R1"],
            "interaction_name_2": ["L1 - R1"],
            "classification": ["Dense"],
            "gene_a": ["L1"],
            "gene_b": ["R1"],
        },
        index=["L1_R1"],
    )
    means = np.arange(1, n_cell_types * n_cell_types + 1, dtype=float).reshape(-1, 1)
    pvalues = np.full_like(means, 0.01, dtype=float)
    adata = AnnData(X=means.copy(), obs=obs, var=var)
    adata.layers["means"] = means.copy()
    adata.layers["pvalues"] = pvalues.copy()
    return adata


def _build_raw_liana_adata(*, uns_key: str = "liana_res", shifted: bool = False) -> AnnData:
    adata = AnnData(X=np.zeros((1, 1), dtype=float))
    specificity = [0.01, 0.04, 0.15, 0.03]
    magnitude = [0.02, 0.10, 0.30, 0.05]
    if shifted:
        specificity = [0.03, 0.07, 0.20, 0.05]
        magnitude = [0.06, 0.14, 0.34, 0.09]
    adata.uns[uns_key] = pd.DataFrame(
        {
            "source": ["B", "B", "T", "T"],
            "target": ["T", "Myeloid", "B", "Myeloid"],
            "ligand_complex": ["CXCL13", "TNF", "IL7", "MIF"],
            "receptor_complex": ["CXCR5", "TNFRSF1A", "IL7R", "CD74_CXCR4"],
            "magnitude_rank": magnitude,
            "specificity_rank": specificity,
        }
    )
    return adata


def _build_raw_liana_sample_adata() -> AnnData:
    base = _build_raw_liana_adata()
    rows = []
    for sample, offset in (("sample_1", 0.0), ("sample_2", 0.03)):
        frame = base.uns["liana_res"].copy()
        frame["sample"] = sample
        frame["specificity_rank"] = frame["specificity_rank"] + offset
        frame["magnitude_rank"] = frame["magnitude_rank"] + offset
        rows.append(frame)
    base.uns["liana_res"] = pd.concat(rows, ignore_index=True)
    return base


def _build_raw_liana_condition_adata() -> AnnData:
    base = _build_raw_liana_adata()
    rows = []
    for condition, offset in (("ctrl", 0.0), ("stim", 0.03)):
        frame = base.uns["liana_res"].copy()
        frame["condition"] = condition
        frame["specificity_rank"] = frame["specificity_rank"] + offset
        frame["magnitude_rank"] = frame["magnitude_rank"] + offset
        rows.append(frame)
    base.uns["liana_res"] = pd.concat(rows, ignore_index=True)
    return base


def _build_raw_cpdb_adata(*, uns_key: str = "cpdb_results") -> AnnData:
    adata = AnnData(X=np.zeros((1, 1), dtype=float))
    adata.uns[uns_key] = {
        "means": pd.DataFrame(
            {
                "id_cp_interaction": [1, 2],
                "interacting_pair": ["CXCL13_CXCR5", "TNF_TNFRSF1A"],
                "interaction_name_2": ["CXCL13 - CXCR5", "TNF - TNFRSF1A"],
                "gene_a": ["CXCL13", "TNF"],
                "gene_b": ["CXCR5", "TNFRSF1A"],
                "classification": ["CXCL", "TNF"],
                "B|T": [0.8, 0.5],
                "T|B": [0.1, 0.3],
            }
        ),
        "pvalues": pd.DataFrame(
            {
                "id_cp_interaction": [1, 2],
                "interacting_pair": ["CXCL13_CXCR5", "TNF_TNFRSF1A"],
                "interaction_name_2": ["CXCL13 - CXCR5", "TNF - TNFRSF1A"],
                "gene_a": ["CXCL13", "TNF"],
                "gene_b": ["CXCR5", "TNFRSF1A"],
                "classification": ["CXCL", "TNF"],
                "B|T": [0.01, 0.02],
                "T|B": [0.3, 0.2],
            }
        ),
    }
    return adata


@pytest.fixture()
def raw_liana_adata() -> AnnData:
    return _build_raw_liana_adata()


@pytest.fixture()
def raw_liana_comparison_adata() -> AnnData:
    return _build_raw_liana_adata(shifted=True)


@pytest.fixture()
def raw_cpdb_adata() -> AnnData:
    return _build_raw_cpdb_adata()


@pytest.mark.parametrize(
    ("plot_type", "kwargs"),
    [
        ("heatmap", {"display_by": "aggregation"}),
        ("focused_heatmap", {"display_by": "aggregation"}),
        ("heatmap", {"display_by": "interaction", "sender_use": "EVT_1", "facet_by": "sender", "top_n": 2}),
        ("dot", {"display_by": "interaction", "sender_use": "EVT_1", "top_n": 2}),
        ("bubble", {"display_by": "aggregation", "top_n": 2}),
        ("bubble_lr", {"pair_lr_use": "MDK_SDC1"}),
        ("bubble", {"display_by": "interaction", "receiver_use": "dNK1", "facet_by": "receiver", "top_n": 2}),
        ("pathway_bubble", {"signaling": "MK", "top_n": 2}),
        ("role_heatmap", {"pattern": "incoming", "top_n": 2}),
        ("role_network", {}),
        ("role_network_marsilea", {}),
    ],
)
def test_ccc_heatmap_variants_return_figure_and_axes(comm_adata: AnnData, plot_type: str, kwargs: dict) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        plot_type=plot_type,
        show=False,
        **kwargs,
    )
    _assert_figure_and_axes(fig, ax)


@pytest.mark.parametrize(
    ("plot_type", "kwargs"),
    [
        ("pathway_bubble", {"signaling": "MK", "top_n": 2}),
        ("role_heatmap", {"pattern": "incoming", "top_n": 2}),
    ],
)
def test_ccc_heatmap_marsilea_variants_leave_single_open_figure(
    comm_adata: AnnData, plot_type: str, kwargs: dict
) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        plot_type=plot_type,
        show=False,
        **kwargs,
    )
    _assert_figure_and_axes(fig, ax)
    assert plt.get_fignums() == [fig.number]


def test_ccc_heatmap_diff_heatmap_returns_figure_and_axes(
    comm_adata: AnnData, comparison_comm_adata: AnnData
) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        comparison_adata=comparison_comm_adata,
        plot_type="diff_heatmap",
        pattern="all",
        top_n=2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_heatmap_diff_heatmap_prefers_diverging_cmap_and_text_for_small_discrete_matrix(
    monkeypatch, comm_adata: AnnData, comparison_comm_adata: AnnData
) -> None:
    captured: dict[str, object] = {}

    def _fake_diff(*args, **kwargs):
        matrix = pd.DataFrame(
            np.ones((3, 3), dtype=float),
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        return matrix, "Delta score"

    def _fake_cluster(matrix, **kwargs):
        return matrix

    def _fake_plot(matrix, **kwargs):
        captured["kwargs"] = kwargs
        fig, ax = plt.subplots()
        return fig, ax

    monkeypatch.setattr(ccc_mod, "_diff_role_matrix", _fake_diff)
    monkeypatch.setattr(ccc_mod, "_apply_cluster_order", _fake_cluster)
    monkeypatch.setattr(ccc_mod, "_plot_heatmap_matrix", _fake_plot)

    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        comparison_adata=comparison_comm_adata,
        plot_type="diff_heatmap",
        show_row_names=True,
        show_col_names=True,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    assert captured["kwargs"]["cmap"] == "RdBu_r"
    assert captured["kwargs"]["add_text"] is True
    assert captured["kwargs"]["show_row_names"] is True
    assert captured["kwargs"]["show_col_names"] is True


def test_ccc_heatmap_diff_heatmap_requires_comparison_adata(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="comparison_adata"):
        ov.pl.ccc_heatmap(
            comm_adata,
            plot_type="diff_heatmap",
            show=False,
        )


def test_ccc_heatmap_aggregation_routes_through_cellchatviz_backend(monkeypatch, comm_adata: AnnData) -> None:
    called: dict[str, object] = {}

    class _StubViz:
        def netVisual_heatmap_marsilea(self, **kwargs):
            called["kwargs"] = kwargs
            return "plotter"

    def _fake_build(adata, *, palette=None):
        called["palette"] = palette
        called["adata"] = adata
        return _StubViz()

    def _fake_render(plotter, *, title=None, add_custom_legends=False):
        fig, ax = plt.subplots()
        called["plotter"] = plotter
        return fig, ax

    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", _fake_build)
    monkeypatch.setattr(ccc_mod, "_render_plotter_figure", _fake_render)

    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        plot_type="heatmap",
        display_by="aggregation",
        signaling="MK",
        show_row_names=True,
        show_col_names=True,
        show=False,
    )

    _assert_figure_and_axes(fig, ax)
    assert called["plotter"] == "plotter"
    assert called["kwargs"]["signaling"] == ["MK"]
    assert called["kwargs"]["show_row_names"] is True
    assert called["kwargs"]["show_col_names"] is True


def test_ccc_heatmap_count_measure(monkeypatch, comm_adata: AnnData) -> None:
    called: dict[str, object] = {}

    class _StubViz:
        def netVisual_heatmap_marsilea(self, **kwargs):
            called.update(kwargs)
            return "plotter"

    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", lambda adata, *, palette=None: _StubViz())
    monkeypatch.setattr(
        ccc_mod,
        "_render_plotter_figure",
        lambda plotter, *, title=None, add_custom_legends=False: plt.subplots(),
    )

    ov.pl.ccc_heatmap(
        comm_adata,
        plot_type="heatmap",
        display_by="aggregation",
        value="count",
        show=False,
    )

    assert called["measure"] == "count"


def test_cpdb_heatmap_count_direction(monkeypatch) -> None:
    captured: dict[str, object] = {"numbers": {}}

    class _Heatmap:
        def __init__(self, data, **kwargs):
            captured["matrix"] = data.to_numpy()
            captured["label"] = kwargs["label"]

        def add_left(self, plotter, **kwargs):
            return None

        def add_top(self, plotter, **kwargs):
            return None

        def add_bottom(self, plotter, **kwargs):
            return None

        def add_legends(self):
            return None

        def add_title(self, title):
            return None

        def add_dendrogram(self, side, **kwargs):
            return None

    class _Plotter:
        @staticmethod
        def Numbers(values, **kwargs):
            captured["numbers"][kwargs["label"]] = np.asarray(values)
            return object()

        @staticmethod
        def Colors(*args, **kwargs):
            return object()

        @staticmethod
        def Labels(*args, **kwargs):
            return object()

    class _Marsilea:
        Heatmap = _Heatmap
        plotter = _Plotter

    monkeypatch.setattr(cpdbviz_mod, "MARSILEA_AVAILABLE", True)
    monkeypatch.setattr(cpdbviz_mod, "ma", _Marsilea)

    viz = ov.pl.CellChatViz(_build_comm_adata_with_duplicate_pairs())
    viz.netVisual_heatmap_marsilea(measure="count", add_dendrogram=False)

    np.testing.assert_array_equal(captured["matrix"], np.array([[0.0, 3.0], [2.0, 0.0]]))
    np.testing.assert_array_equal(captured["numbers"]["Outgoing"], np.array([3.0, 2.0]))
    np.testing.assert_array_equal(captured["numbers"]["Incoming"], np.array([2.0, 3.0]))
    assert captured["label"] == "Number of interactions"


def test_ccc_count_legend(monkeypatch, comm_adata: AnnData) -> None:
    captured: dict[str, object] = {}

    def _fake_plot(matrix, size_matrix, **kwargs):
        captured.update(kwargs)
        return plt.subplots()

    monkeypatch.setattr(ccc_mod, "_dot_matrix_plot", _fake_plot)

    ov.pl.ccc_heatmap(
        comm_adata,
        plot_type="bubble",
        display_by="aggregation",
        value="count",
        show=False,
    )

    assert captured["color_label"] == "Number of interactions"


def test_ccc_heatmap_aggregation_rejects_interaction_filters(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="interaction_use"):
        ov.pl.ccc_heatmap(
            comm_adata,
            plot_type="heatmap",
            display_by="aggregation",
            interaction_use="TGFB1 - TGFBR1",
            show=False,
        )


def test_ccc_heatmap_role_heatmap_rejects_sender_filter(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="sender_use"):
        ov.pl.ccc_heatmap(
            comm_adata,
            plot_type="role_heatmap",
            sender_use="EVT_1",
            show=False,
        )


def test_ccc_heatmap_pathway_bubble_rejects_pair_lr_filter(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="pair_lr_use"):
        ov.pl.ccc_heatmap(
            comm_adata,
            plot_type="pathway_bubble",
            signaling="MK",
            pair_lr_use="MDK_SDC1",
            show=False,
        )


@pytest.mark.parametrize(
    ("plot_type", "kwargs"),
    [
        ("circle", {}),
        ("circle_focused", {}),
        ("individual_outgoing", {}),
        ("individual_incoming", {}),
        ("individual", {"signaling": "MK"}),
        ("arrow", {}),
        ("arrow", {"display_by": "interaction"}),
        ("sigmoid", {}),
        ("sigmoid", {"display_by": "interaction"}),
        ("embedding_network", {}),
        ("pathway", {"signaling": "MK"}),
        ("chord", {"signaling": "MK"}),
        ("lr_chord", {"pair_lr_use": "MDK_SDC1"}),
        ("gene_chord", {"signaling": "MK"}),
        ("diffusion", {}),
        ("individual_lr", {"pair_lr_use": "MDK_SDC1"}),
        ("bipartite", {"ligand": "TGFB1"}),
    ],
)
def test_ccc_network_plot_base_variants_return_figure_and_axes(
    comm_adata: AnnData, plot_type: str, kwargs: dict
) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        plot_type=plot_type,
        show=False,
        **kwargs,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_network_plot_diff_network_returns_figure_and_axes(
    comm_adata: AnnData, comparison_comm_adata: AnnData
) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        comparison_adata=comparison_comm_adata,
        plot_type="diff_network",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_network_plot_rejects_unknown_plot_type(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="Unsupported `plot_type='pathway1'`"):
        ov.pl.ccc_network_plot(
            comm_adata,
            plot_type="pathway1",
            signaling="MK",
            show=False,
        )


def test_ccc_network_plot_pathway_forwards_top_n(monkeypatch, comm_adata: AnnData) -> None:
    captured: dict[str, object] = {}

    class _StubViz:
        def netVisual_aggregate(self, **kwargs):
            captured.update(kwargs)
            return plt.subplots(figsize=kwargs.get("figsize", (7, 7)))

    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", lambda adata, *, palette=None: _StubViz())

    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        plot_type="pathway",
        signaling="MK",
        top_n=7,
        show=False,
    )

    _assert_figure_and_axes(fig, ax)
    assert captured["signaling"] == ["MK"]
    assert captured["layout"] == "circle"
    assert captured["top_n"] == 7


def test_ccc_circle_keeps_all_edges(monkeypatch) -> None:
    adata = _build_dense_pathway_comm_adata()
    captured_lengths: list[int] = []

    def _fake_draw_circle_network(edge_df, **kwargs):
        captured_lengths.append(len(edge_df))
        return plt.subplots(figsize=kwargs.get("figsize", (7, 7)))

    monkeypatch.setattr(ccc_mod, "_draw_circle_network", _fake_draw_circle_network)

    fig, ax = ov.pl.ccc_network_plot(
        adata,
        plot_type="circle",
        signaling="Dense",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)

    fig, ax = ov.pl.ccc_network_plot(
        adata,
        plot_type="circle",
        signaling="Dense",
        top_n=7,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)

    assert captured_lengths == [25, 7]


def test_ccc_circle_sender_legend() -> None:
    adata = _build_comm_adata_with_high_nonsignificant_scores()

    fig, ax = ov.pl.ccc_network_plot(
        adata,
        plot_type="circle",
        signaling="Pathway",
        show=False,
    )

    _assert_figure_and_axes(fig, ax)
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["A (sender)"]


@pytest.mark.parametrize(
    ("plot_type", "method_name"),
    [
        ("individual_outgoing", "netVisual_individual_circle"),
        ("individual_incoming", "netVisual_individual_circle_incoming"),
    ],
)
def test_ccc_network_plot_individual_circle_variants_use_more_visible_edge_defaults(
    monkeypatch, comm_adata: AnnData, plot_type: str, method_name: str
) -> None:
    called: dict[str, object] = {}

    class _StubViz:
        def netVisual_individual_circle(self, **kwargs):
            called["kwargs"] = kwargs
            fig, _ = plt.subplots()
            return fig

        def netVisual_individual_circle_incoming(self, **kwargs):
            called["kwargs"] = kwargs
            fig, _ = plt.subplots()
            return fig

    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", lambda adata, *, palette=None: _StubViz())

    fig, ax = ov.pl.ccc_network_plot(comm_adata, plot_type=plot_type, show=False)
    _assert_figure_and_axes(fig, ax)
    assert called["kwargs"]["edge_width_max"] == 12


@pytest.mark.parametrize(
    ("plot_type", "extra_kwargs"),
    [
        ("individual_outgoing", {"sender_use": ["EVT_1", "EVT_2"], "receiver_use": ["DSC_1", "DSC_2"]}),
        ("individual_incoming", {"sender_use": ["EVT_1", "EVT_2"], "receiver_use": ["DSC_1", "DSC_2"]}),
    ],
)
def test_ccc_network_plot_individual_circle_variants_forward_grid_and_celltype_filters(
    monkeypatch, comm_adata: AnnData, plot_type: str, extra_kwargs: dict
) -> None:
    called: dict[str, object] = {}

    class _StubViz:
        def netVisual_individual_circle(self, **kwargs):
            called["kwargs"] = kwargs
            fig, _ = plt.subplots()
            return fig

        def netVisual_individual_circle_incoming(self, **kwargs):
            called["kwargs"] = kwargs
            fig, _ = plt.subplots()
            return fig

    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", lambda adata, *, palette=None: _StubViz())

    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        plot_type=plot_type,
        ncols=3,
        nrows=2,
        show=False,
        **extra_kwargs,
    )
    _assert_figure_and_axes(fig, ax)
    assert called["kwargs"]["ncols"] == 3
    assert called["kwargs"]["nrows"] == 2
    assert called["kwargs"]["sender_use"] == extra_kwargs["sender_use"]
    assert called["kwargs"]["receiver_use"] == extra_kwargs["receiver_use"]


def test_cellchatviz_individual_outgoing_omits_redundant_legend(comm_adata: AnnData) -> None:
    viz = ov.pl.CellChatViz(comm_adata)
    fig = viz.netVisual_individual_circle(figsize=(6, 4), ncols=2)
    assert not fig.legends


def test_ccc_network_plot_diff_network_uses_cell_type_node_colors(
    comm_adata: AnnData, comparison_comm_adata: AnnData
) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        comparison_adata=comparison_comm_adata,
        plot_type="diff_network",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    facecolors = []
    for collection in ax.collections:
        if hasattr(collection, "get_facecolors"):
            colors = collection.get_facecolors()
            if len(colors):
                facecolors.extend([tuple(round(channel, 3) for channel in color[:3]) for color in colors])
    assert facecolors
    assert any(color != (0.957, 0.945, 0.918) for color in facecolors)


@pytest.mark.parametrize(
    ("plot_type", "expected_title"),
    [
        ("scatter", "Outgoing vs incoming communication"),
        ("role_scatter", "Signaling role scatter"),
    ],
)
def test_ccc_stat_plot_scatter_variants_route_through_cellchatviz_scatter(
    monkeypatch, comm_adata: AnnData, plot_type: str, expected_title: str
) -> None:
    called: dict[str, object] = {}

    class _StubViz:
        def netAnalysis_signalingRole_scatter(self, **kwargs):
            called["kwargs"] = kwargs
            fig, ax = plt.subplots()
            ax.text(0.1, 0.2, "EVT_1")
            return fig, ax

    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", lambda adata, *, palette=None: _StubViz())

    fig, ax = ov.pl.ccc_stat_plot(comm_adata, plot_type=plot_type, show=False)
    _assert_figure_and_axes(fig, ax)
    assert called["kwargs"]["x_measure"] == "outdegree"
    assert called["kwargs"]["y_measure"] == "indegree"
    assert called["kwargs"]["title"] == expected_title


@pytest.mark.parametrize(
    ("plot_type", "kwargs"),
    [
        ("bar", {}),
        ("sankey", {}),
        ("box", {}),
        ("violin", {}),
        ("scatter", {}),
        ("role_scatter", {}),
        ("role_network", {}),
        ("role_network_marsilea", {}),
        ("pathway_summary", {}),
        ("gene", {"signaling": "TGFb"}),
    ],
)
def test_ccc_stat_plot_core_variants_return_figure_and_axes(
    comm_adata: AnnData, plot_type: str, kwargs: dict
) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type=plot_type,
        top_n=2,
        show=False,
        **kwargs,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_stat_plot_interaction_sankey_returns_figure_and_axes(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="sankey",
        display_by="interaction",
        top_n=2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    interaction_texts = [
        text
        for text in ax.texts
        if "FN1" in text.get_text() or "MIF" in text.get_text()
    ]
    assert interaction_texts
    assert all(text.get_rotation() == 0 for text in interaction_texts)
    assert any(" - " in text.get_text() for text in interaction_texts)
    receiver_rectangles = [
        patch
        for patch in ax.patches
        if hasattr(patch, "get_facecolor")
        and hasattr(patch, "get_width")
        and hasattr(patch, "get_x")
        and patch.get_width() > 0.03
        and patch.get_x() >= 0.89
    ]
    assert receiver_rectangles
    assert any(tuple(round(channel, 3) for channel in patch.get_facecolor()[:3]) != (0.851, 0.851, 0.851) for patch in receiver_rectangles)
    assert max(float(patch.get_y()) + float(patch.get_height()) / 2.0 for patch in receiver_rectangles) < 0.87
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Interaction sankey"
    assert ax.get_title() == ""


def test_ccc_stat_plot_rejects_invalid_display_by(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="Unsupported `display_by='aggregate'`"):
        ov.pl.ccc_stat_plot(
            comm_adata,
            plot_type="sankey",
            display_by="aggregate",
            show=False,
        )


@pytest.mark.parametrize("plot_type", ["arrow", "sigmoid"])
def test_ccc_network_plot_interaction_flow_has_middle_stage_labels(comm_adata: AnnData, plot_type: str) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        plot_type=plot_type,
        display_by="interaction",
        top_n=3,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    texts = [text.get_text() for text in ax.texts]
    assert "Ligand-Receptor" in texts
    assert "nan" not in texts
    assert any("CXCL12" in text or "TGFB1" in text or "MDK" in text for text in texts)
    assert any(" - " in text for text in texts if "CXCL12" in text or "TGFB1" in text or "MDK" in text)
    middle_rectangles = [
        patch
        for patch in ax.patches
        if hasattr(patch, "get_width")
        and hasattr(patch, "get_height")
        and hasattr(patch, "get_x")
        and 0.45 <= patch.get_x() <= 0.52
        and patch.get_width() >= 0.03
        and patch.get_height() >= 0.03
    ]
    assert middle_rectangles


def test_build_flow_plot_frames_prunes_dense_interaction_branches() -> None:
    long_df = pd.DataFrame(
        {
            "sender": ["S1", "S2", "S3", "S4", "S5", "S6"] * 2,
            "receiver": ["R1", "R2", "R3", "R4", "R5", "R6"] * 2,
            "interaction": ["L1_R1"] * 6 + ["L2_R2"] * 6,
            "pair_lr": ["L1_R1"] * 6 + ["L2_R2"] * 6,
            "score": np.linspace(12.0, 1.0, 12),
            "significant": np.ones(12, dtype=float),
        }
    )
    node_df, edge_df, column_titles = ccc_mod._build_flow_plot_frames(
        long_df,
        display_by="interaction",
        value="sum",
        top_n=4,
    )

    assert column_titles == [(0.0, "Sender"), (0.5, "Ligand-Receptor"), (1.0, "Receiver")]
    interaction_nodes = node_df.loc[node_df["column"] == "interaction", "label"].astype(str).tolist()
    assert set(interaction_nodes) == {"L1_R1", "L2_R2"}
    sender_edges = edge_df.loc[edge_df["from_id"].astype(str).str.startswith("sender::"), :]
    receiver_edges = edge_df.loc[edge_df["to_id"].astype(str).str.startswith("receiver::"), :]
    assert sender_edges.groupby("to_id").size().max() <= 3
    assert receiver_edges.groupby("from_id").size().max() <= 3


def test_ccc_network_plot_bipartite_aligns_ligand_and_receptor_labels(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        plot_type="bipartite",
        ligand="TGFB1",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)

    text_lookup = {
        " ".join(text.get_text().split()): text
        for text in ax.texts
        if text.get_text().strip()
    }
    assert all(not label.startswith("Ligand focus:") for label in text_lookup)

    ligand_text = text_lookup["TGFB1"]
    receptor_raw = (
        comm_adata.var.loc[comm_adata.var["ligand"].astype(str) == "TGFB1", "receptor"].astype(str).iloc[0]
    )
    receptor_tokens = [token for token in receptor_raw.replace("_", " ").split() if token]
    receptor_label = next(
        label
        for label in text_lookup
        if all(any(token.lower() in part.lower() for part in label.split()) for token in receptor_tokens)
    )
    receptor_text = text_lookup[receptor_label]
    assert abs(ligand_text.get_position()[1] - receptor_text.get_position()[1]) < 1e-6


def test_ccc_network_plot_bipartite_receiver_labels_keep_vertical_separation(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        plot_type="bipartite",
        ligand="TGFB1",
        top_n=6,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    receiver_texts = [
        text for text in ax.texts
        if text.get_text().strip()
        and text.get_position()[0] > 2.9
    ]
    y_positions = sorted(float(text.get_position()[1]) for text in receiver_texts)
    if len(y_positions) > 1:
        diffs = np.diff(y_positions)
        assert float(diffs.min()) > 0.015


def test_ccc_network_plot_embedding_network_hides_on_plot_celltype_labels(
    comm_adata: AnnData,
) -> None:
    node_positions = pd.DataFrame(
        {
            "x": [-1.0, 1.0, 0.0],
            "y": [0.0, 0.0, 1.0],
        },
        index=["EVT_1", "VCT", "dNK1"],
    )
    embedding_points = pd.DataFrame(
        {
            "x": [-1.1, -0.9, 0.9, 1.1, -0.1, 0.1],
            "y": [0.1, -0.1, 0.1, -0.1, 0.9, 1.1],
            "cell_type": ["EVT_1", "EVT_1", "VCT", "VCT", "dNK1", "dNK1"],
        }
    )
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        plot_type="embedding_network",
        node_positions=node_positions,
        embedding_points=embedding_points,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    text_labels = {text.get_text().strip() for text in ax.texts if text.get_text().strip()}
    assert all("EVT_1" not in label for label in text_labels)
    assert all("VCT" not in label for label in text_labels)
    assert all("nCells:" not in label for label in text_labels)


def test_interaction_display_lookup_formats_lr_labels_and_cleans_complex_suffix() -> None:
    long_df = pd.DataFrame(
        {
            "pair_lr": ["complex:FN1_integrin_a5b1_complex"],
            "interaction": [""],
            "ligand": ["complex:FN1"],
            "receptor": ["integrin_a5b1_complex"],
        }
    )
    lookup = ccc_mod._interaction_display_lookup(long_df)
    assert lookup["complex:FN1_integrin_a5b1_complex"] == "FN1 - a5b1"


def test_ccc_stat_plot_lr_contribution_returns_figure_and_axes(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="lr_contribution",
        signaling="TGFb",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_stat_plot_bar_cleans_complex_interaction_labels(comm_adata: AnnData) -> None:
    adata = comm_adata.copy()
    adata.var = adata.var.copy()
    adata.var["interaction_name_2"] = ""
    adata.var.loc[adata.var_names[0], "interaction_name"] = "FN1_integrin_a5b1_complex"
    adata.var.loc[adata.var_names[0], "interacting_pair"] = "FN1_integrin_a5b1_complex"
    adata.layers["means"][:, 0] = 50.0

    fig, ax = ov.pl.ccc_stat_plot(
        adata,
        plot_type="bar",
        top_n=1,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    labels = [" ".join(tick.get_text().split()) for tick in ax.get_yticklabels() if tick.get_text()]
    assert any("FN1 integrin a5b1" in label for label in labels)
    assert all("complex" not in label.lower() for label in labels)


def test_ccc_stat_plot_gene_cleans_complex_gene_labels(comm_adata: AnnData) -> None:
    adata = comm_adata.copy()
    adata.var = adata.var.copy()
    adata.var.loc[adata.var_names[0], "classification"] = "TGFb"
    adata.var.loc[adata.var_names[0], "pathway_name"] = "TGFb"
    adata.var.loc[adata.var_names[0], "gene_a"] = "complex:IL27"
    adata.var.loc[adata.var_names[0], "gene_b"] = "complex:IL1_receptor"
    adata.layers["means"][:, 0] = 60.0

    fig, ax = ov.pl.ccc_stat_plot(
        adata,
        plot_type="gene",
        signaling="TGFb",
        top_n=2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    labels = " ".join(tick.get_text() for tick in ax.get_yticklabels() if tick.get_text())
    assert "complex:" not in labels.lower()
    assert "receptor" not in labels.lower()
    assert "IL27" in labels or "IL1" in labels


def test_ccc_stat_plot_gene_uses_source_expression_adata(comm_adata: AnnData) -> None:
    genes = list(
        dict.fromkeys(
            comm_adata.var["gene_a"].astype(str).tolist() + comm_adata.var["gene_b"].astype(str).tolist()
        )
    )
    obs = pd.DataFrame(
        {
            "cell_labels": ["EVT_1", "EVT_1", "dNK1", "dNK1", "VCT", "VCT"],
        },
        index=[f"cell_{idx}" for idx in range(6)],
    )
    matrix = np.arange(len(obs.index) * len(genes), dtype=float).reshape(len(obs.index), len(genes)) + 1.0
    expr_adata = AnnData(matrix, obs=obs, var=pd.DataFrame(index=genes))

    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="gene",
        signaling="TGFb",
        source_adata=expr_adata,
        source_groupby="cell_labels",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    assert ax.get_xlabel() == "cell_labels"
    assert "expression" in ax.get_title().lower()
    ytick_labels = [tick.get_text() for tick in ax.get_yticklabels() if tick.get_text()]
    assert any("(L)" in label or "(R)" in label for label in ytick_labels)


@pytest.mark.parametrize(
    ("plot_type", "kwargs"),
    [
        ("comparison", {"compare_by": "overall"}),
        ("comparison", {"compare_by": "celltype", "pattern": "incoming"}),
        ("ranknet", {}),
        ("role_change", {"idents_use": "EVT_1"}),
    ],
)
def test_ccc_stat_plot_comparison_variants_return_figure_and_axes(
    comm_adata: AnnData,
    comparison_comm_adata: AnnData,
    plot_type: str,
    kwargs: dict,
) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        comparison_adata=comparison_comm_adata,
        plot_type=plot_type,
        top_n=3,
        show=False,
        **kwargs,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_stat_plot_box_sender_facets_by_sender(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="box",
        facet_by="sender",
        top_n=3,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    titles = {axis.get_title() for axis in fig.axes if axis.get_title()}
    assert "EVT_1" in titles
    assert len(titles) >= 2


def test_ccc_stat_plot_role_change_labels_signaling_for_selected_identity(
    comm_adata: AnnData, comparison_comm_adata: AnnData
) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        comparison_adata=comparison_comm_adata,
        plot_type="role_change",
        idents_use="EVT_1",
        top_n=3,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    text_labels = [text.get_text() for text in ax.texts if text.get_text()]
    pathway_labels = set(comm_adata.var["classification"].astype(str).tolist())
    assert any(label in pathway_labels for label in text_labels)
    assert "EVT_1" not in text_labels


@pytest.mark.parametrize("plot_type", ["comparison", "ranknet", "role_change"])
def test_ccc_stat_plot_comparison_variants_require_comparison_adata(
    comm_adata: AnnData, plot_type: str
) -> None:
    with pytest.raises(ValueError, match="comparison_adata"):
        ov.pl.ccc_stat_plot(
            comm_adata,
            plot_type=plot_type,
            show=False,
        )


def test_ccc_heatmap_save_writes_output(comm_adata: AnnData, tmp_path) -> None:
    output = tmp_path / "ccc_heatmap.png"
    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        plot_type="bubble",
        display_by="interaction",
        save=str(output),
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    assert output.exists()


def test_ccc_network_plot_diff_network_requires_comparison_adata(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="comparison_adata"):
        ov.pl.ccc_network_plot(
            comm_adata,
            plot_type="diff_network",
            show=False,
        )


def test_ccc_network_plot_pathway_requires_signaling(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="signaling"):
        ov.pl.ccc_network_plot(
            comm_adata,
            plot_type="pathway",
            show=False,
        )


def test_ccc_network_plot_pathway_forwards_top_n_to_cellchatviz_backend(
    monkeypatch, comm_adata: AnnData
) -> None:
    called: dict[str, object] = {}

    class _StubViz:
        def netVisual_aggregate(self, **kwargs):
            called["kwargs"] = kwargs
            fig, ax = plt.subplots()
            return fig, ax

    def _fake_build(adata, *, palette=None):
        called["adata"] = adata
        called["palette"] = palette
        return _StubViz()

    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", _fake_build)

    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        plot_type="pathway",
        signaling="TGFb",
        top_n=5,
        show=False,
    )

    _assert_figure_and_axes(fig, ax)
    assert called["kwargs"]["signaling"] == ["TGFb"]
    assert called["kwargs"]["top_n"] == 5


def test_ccc_network_plot_pathway_rejects_interaction_filters(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="interaction_use"):
        ov.pl.ccc_network_plot(
            comm_adata,
            plot_type="pathway",
            signaling="MK",
            interaction_use="TGFB1 - TGFBR1",
            show=False,
        )


def test_ccc_network_plot_diffusion_rejects_sender_filter(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="sender_use"):
        ov.pl.ccc_network_plot(
            comm_adata,
            plot_type="diffusion",
            sender_use="EVT_1",
            show=False,
        )


def test_ccc_network_plot_rejects_invalid_display_by(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="Unsupported `display_by='aggregate'`"):
        ov.pl.ccc_network_plot(
            comm_adata,
            plot_type="arrow",
            display_by="aggregate",
            show=False,
        )


def test_ccc_network_plot_embedding_requires_positions(comm_adata: AnnData) -> None:
    comm_adata = comm_adata.copy()
    comm_adata.uns.pop("node_positions", None)
    with pytest.raises(ValueError, match="node_positions"):
        ov.pl.ccc_network_plot(
            comm_adata,
            plot_type="embedding_network",
            show=False,
        )


def test_ccc_stat_plot_raises_for_empty_filtered_data(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="No communication records remain"):
        ov.pl.ccc_stat_plot(
            comm_adata,
            plot_type="lr_contribution",
            signaling="NotAPathway",
            show=False,
        )


def test_ccc_stat_plot_lr_contribution_routes_through_cellchatviz_backend(monkeypatch, comm_adata: AnnData) -> None:
    called: dict[str, object] = {}

    class _StubViz:
        def netAnalysis_contribution(self, **kwargs):
            called["kwargs"] = kwargs
            fig, ax = plt.subplots()
            return None, fig, (ax,)

    def _fake_build(adata, *, palette=None):
        called["adata"] = adata
        called["palette"] = palette
        return _StubViz()

    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", _fake_build)

    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="lr_contribution",
        signaling="TGFb",
        top_n=4,
        show=False,
    )

    _assert_figure_and_axes(fig, ax)
    assert called["kwargs"]["signaling"] == ["TGFb"]
    assert called["kwargs"]["top_pairs"] == 4
    assert plt.get_fignums() == [fig.number]


def test_ccc_stat_plot_lr_contribution_rejects_sender_filter_with_signaling(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="sender_use"):
        ov.pl.ccc_stat_plot(
            comm_adata,
            plot_type="lr_contribution",
            signaling="TGFb",
            sender_use="EVT_1",
            show=False,
        )


def test_ccc_stat_plot_pathway_summary_rejects_pair_lr_filter(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="pair_lr_use"):
        ov.pl.ccc_stat_plot(
            comm_adata,
            plot_type="pathway_summary",
            pair_lr_use="MDK_SDC1",
            show=False,
        )


def test_ccc_heatmap_matrix_dot_has_been_removed(comm_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="has been removed"):
        ov.pl.ccc_heatmap(
            comm_adata,
            plot_type="matrix_dot",
            display_by="interaction",
            show=False,
        )


def test_ccc_heatmap_focused_heatmap_avoids_extra_annotation_text(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        plot_type="focused_heatmap",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    non_title_texts = [
        text.get_text().strip()
        for axis in fig.axes
        for text in axis.texts
        if text.get_text().strip() and "focused communication heatmap" not in text.get_text().strip().lower()
    ]
    assert non_title_texts == []


def test_cellchatviz_role_network_variants_hide_value_one_annotations_and_marsilea_uses_dark_text(
    comm_adata: AnnData,
) -> None:
    viz = ov.pl.CellChatViz(comm_adata)
    n_cells = len(viz.cell_types)
    viz.centrality_scores = {
        "outdegree": np.array([1.0, 0.42, 0.0][:n_cells], dtype=float),
        "indegree": np.array([0.18, 1.0, 0.0][:n_cells], dtype=float),
        "flow_betweenness": np.array([0.0, 0.27, 1.0][:n_cells], dtype=float),
        "information": np.array([0.66, 0.0, 1.0][:n_cells], dtype=float),
    }

    fig = viz.netAnalysis_signalingRole_network(show_values=True)
    ax = fig.axes[0]
    texts = [text.get_text().strip() for text in ax.texts if text.get_text().strip()]
    assert "1.00" not in texts
    assert any(text in {"0.42", "0.18", "0.27", "0.66"} for text in texts)
    plt.close(fig)

    plotter = viz.netAnalysis_signalingRole_network_marsilea(show_values=True)
    fig = ccc_mod._render_plot(plotter)
    text_items = [
        text
        for axis in fig.axes
        for text in axis.texts
        if text.get_text().strip() and text.get_text().strip().replace(".", "", 1).isdigit()
    ]
    assert text_items
    assert all(text.get_text().strip() != "1.00" for text in text_items)
    assert all(text.get_color() != "white" for text in text_items)
    plt.close(fig)


def test_ccc_heatmap_accepts_named_palette_string(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        comm_adata,
        plot_type="circle",
        palette="Set1",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_pair_lr_filter_prefers_interacting_pair(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        comm_adata,
        plot_type="heatmap",
        display_by="interaction",
        pair_lr_use="MDK_SDC1",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    ticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert ticklabels == ["MDK - SDC1"]


def test_ccc_scatter_includes_receiver_only_groups(comm_adata_with_receiver_only_group: AnnData) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata_with_receiver_only_group,
        plot_type="scatter",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    # adjustText can leave an empty helper text artist in some environments;
    # the test only cares that receiver-only group labels are present.
    labels = {text.get_text() for text in ax.texts if text.get_text()}
    assert labels == {"EVT_1", "dNK1", "SCT"}


def test_ccc_heatmap_auto_resolves_liana_adata(raw_liana_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type="heatmap",
        display_by="aggregation",
        signaling="TNF",
        top_n=3,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_heatmap_dot_rejects_aggregation_display(raw_liana_adata: AnnData) -> None:
    with pytest.raises(ValueError, match="only supports `display_by='interaction'`"):
        ov.pl.ccc_heatmap(
            raw_liana_adata,
            plot_type="dot",
            display_by="aggregation",
            top_n=3,
            show=False,
        )


@pytest.mark.parametrize("plot_type", ["source_target_dot", "tile"])
def test_ccc_heatmap_liana_style_views(raw_liana_adata: AnnData, plot_type: str) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type=plot_type,
        display_by="interaction",
        top_n=3,
        pvalue_threshold=0.2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_heatmap_tile_uses_side_specific_row_labels(raw_liana_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type="tile",
        display_by="interaction",
        top_n=3,
        pvalue_threshold=0.2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    main_axes = [axis for axis in fig.axes if axis.get_xlabel() == "Cell type"]
    assert len(main_axes) == 2
    left_axis, right_axis = main_axes
    assert left_axis.get_ylabel() == "Source ligand"
    assert right_axis.get_ylabel() == "Target receptor"
    assert right_axis.yaxis.get_label_position() == "right"
    assert any(tick.get_text() for tick in left_axis.get_yticklabels())
    assert any(tick.get_text() for tick in right_axis.get_yticklabels())


def test_ccc_heatmap_liana_sample_dot_view() -> None:
    raw_liana_adata = _build_raw_liana_sample_adata()
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type="sample_dot",
        display_by="interaction",
        top_n=2,
        pvalue_threshold=0.2,
        sample_key="sample",
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_heatmap_liana_sample_dot_autodetects_non_sample_context_key() -> None:
    raw_liana_adata = _build_raw_liana_condition_adata()
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type="sample_dot",
        display_by="interaction",
        top_n=2,
        pvalue_threshold=0.2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_heatmap_liana_sample_dot_hides_nonfirst_panel_pair_labels() -> None:
    raw_liana_adata = _build_raw_liana_sample_adata()
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type="sample_dot",
        display_by="interaction",
        top_n=4,
        pvalue_threshold=0.2,
        sample_key="sample",
        ncols=2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    titled_axes = [axis for axis in fig.axes if axis.get_title()]
    assert len(titled_axes) >= 2
    assert any(tick.get_text() for tick in titled_axes[0].get_yticklabels())
    assert all(not tick.get_text() for tick in titled_axes[1].get_yticklabels())
    assert any(text.get_text() == "Sender -> receiver" for text in fig.texts)


def test_ccc_heatmap_pathway_bubble_can_autoselect_top_pathways(raw_liana_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type="pathway_bubble",
        top_n=3,
        pvalue_threshold=0.2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_heatmap_pathway_bubble_autoselect_does_not_relimit_interactions(monkeypatch, raw_liana_adata: AnnData) -> None:
    called: dict[str, object] = {}

    class _StubViz:
        def netVisual_bubble_marsilea(self, **kwargs):
            called["kwargs"] = kwargs
            return "plotter"

    monkeypatch.setattr(
        ccc_mod,
        "_pathway_summary_table",
        lambda *args, **kwargs: pd.DataFrame({"pathway": ["P1", "P2", "P3"]}),
    )
    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", lambda adata, *, palette=None: _StubViz())
    monkeypatch.setattr(ccc_mod, "_render_plotter_figure", lambda plotter, *, title=None: plt.subplots())

    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type="pathway_bubble",
        top_n=3,
        pvalue_threshold=0.2,
        show=False,
    )

    _assert_figure_and_axes(fig, ax)
    assert called["kwargs"]["signaling"] is not None
    assert len(called["kwargs"]["signaling"]) == 3
    assert called["kwargs"]["top_interactions"] is None


def test_ccc_heatmap_role_heatmap_and_pathway_bubble_suppress_categorical_unit_warning(
    monkeypatch,
    raw_liana_adata: AnnData,
) -> None:
    class _StubViz:
        def netAnalysis_signalingRole_heatmap(self, **kwargs):
            warnings.warn(
                "Using categorical units to plot a list of strings that are all parsable as floats or dates.",
                UserWarning,
            )
            return "role_plotter", [], pd.DataFrame([[1.0]])

        def netVisual_bubble_marsilea(self, **kwargs):
            warnings.warn(
                "Using categorical units to plot a list of strings that are all parsable as floats or dates.",
                UserWarning,
            )
            return "bubble_plotter"

    monkeypatch.setattr(ccc_mod, "_build_cellchatviz", lambda adata, *, palette=None: _StubViz())
    monkeypatch.setattr(ccc_mod, "_render_plotter_figure", lambda plotter, *, title=None: plt.subplots())
    monkeypatch.setattr(
        ccc_mod,
        "_pathway_summary_table",
        lambda *args, **kwargs: pd.DataFrame({"pathway": ["P1", "P2", "P3"]}),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig1, ax1 = ov.pl.ccc_heatmap(
            raw_liana_adata,
            plot_type="role_heatmap",
            pattern="incoming",
            top_n=3,
            figsize=(4, 2),
            show=False,
        )
        fig2, ax2 = ov.pl.ccc_heatmap(
            raw_liana_adata,
            plot_type="pathway_bubble",
            top_n=3,
            figsize=(3, 3),
            show=False,
        )
    _assert_figure_and_axes(fig1, ax1)
    _assert_figure_and_axes(fig2, ax2)
    assert not any("Using categorical units to plot a list of strings" in str(w.message) for w in caught)


def test_ccc_stat_plot_pathway_summary_places_sig_labels_inside_bars(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="pathway_summary",
        top_n=4,
        min_expression=0.0,
        strength_threshold=0.0,
        min_significant_pairs=1,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    text_items = [text for text in ax.texts if "sig" in text.get_text()]
    assert text_items
    assert all(text.get_color() in {"#1F1F1F", "#FFFFFF", "#4A4A4A"} for text in text_items)


def test_ccc_stat_plot_pathway_summary_moves_short_bar_labels_outside(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="pathway_summary",
        top_n=8,
        min_expression=0.0,
        strength_threshold=0.0,
        min_significant_pairs=1,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    bar_patches = [patch for patch in ax.patches if patch.get_width() > 0]
    text_items = [text for text in ax.texts if "sig" in text.get_text()]
    assert text_items
    for text, patch in zip(text_items, bar_patches):
        if text.get_color() == "#4A4A4A":
            assert text.get_position()[0] > patch.get_x() + patch.get_width()


def test_ccc_stat_plot_pathway_summary_respects_verbose_false(comm_adata: AnnData, capsys) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="pathway_summary",
        top_n=4,
        min_expression=0.0,
        strength_threshold=0.0,
        min_significant_pairs=1,
        verbose=False,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    captured = capsys.readouterr()
    assert "Calculating cell communication strength" not in captured.out
    assert "Pathway significance analysis results" not in captured.out


def test_ccc_stat_plot_pathway_summary_respects_verbose_true(comm_adata: AnnData, capsys) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="pathway_summary",
        top_n=4,
        min_expression=0.0,
        strength_threshold=0.0,
        min_significant_pairs=1,
        verbose=True,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    captured = capsys.readouterr()
    assert "Calculating cell communication strength" in captured.out
    assert "Pathway significance analysis results" in captured.out


def test_ccc_heatmap_dot_supports_multirow_source_facets(raw_liana_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type="dot",
        display_by="interaction",
        top_n=4,
        pvalue_threshold=0.2,
        ncols=2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    assert len(fig.axes) >= 3


def test_ccc_heatmap_dot_hides_upper_row_target_labels(raw_liana_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type="dot",
        display_by="interaction",
        top_n=4,
        pvalue_threshold=0.2,
        ncols=1,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    facet_axes = [axis for axis in fig.axes if axis.get_title()]
    upper_row_axes = facet_axes[:-1]
    lower_row_axes = facet_axes[-1:]
    assert upper_row_axes
    assert lower_row_axes
    assert all(not any(tick.get_text() for tick in axis.get_xticklabels()) for axis in upper_row_axes)
    assert all(any(tick.get_text() for tick in axis.get_xticklabels()) for axis in lower_row_axes)


def test_ccc_heatmap_dot_shows_target_labels_on_last_visible_panel_per_column(raw_liana_adata: AnnData) -> None:
    adata = _build_raw_liana_adata()
    extra = adata.uns["liana_res"].copy()
    extra["source"] = "Myeloid"
    adata.uns["liana_res"] = pd.concat([adata.uns["liana_res"], extra], ignore_index=True)
    fig, ax = ov.pl.ccc_heatmap(
        adata,
        plot_type="dot",
        display_by="interaction",
        top_n=5,
        pvalue_threshold=0.2,
        ncols=2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    facet_axes = [axis for axis in fig.axes if axis.get_title()]
    assert len(facet_axes) >= 3
    # Last visible panel in column 2 should still show bottom x labels even
    # when the final grid row is incomplete.
    assert any(tick.get_text() for tick in facet_axes[1].get_xticklabels())


def test_ccc_heatmap_dot_collapses_to_single_facet_when_sender_filter_leaves_one_source(
    raw_liana_adata: AnnData,
) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type="dot",
        display_by="interaction",
        sender_use="B",
        top_n=4,
        pvalue_threshold=0.2,
        nrows=2,
        figsize=(8, 3),
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    facet_axes = [axis for axis in fig.axes if axis.get_title()]
    assert len(facet_axes) == 1
    assert any(tick.get_text() for tick in facet_axes[0].get_xticklabels())


def test_ccc_heatmap_dot_respects_explicit_figsize(raw_liana_adata: AnnData) -> None:
    target_size = (5.5, 2.75)
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type="dot",
        display_by="interaction",
        sender_use="B",
        top_n=4,
        pvalue_threshold=0.2,
        figsize=target_size,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    width, height = fig.get_size_inches()
    assert width == pytest.approx(target_size[0], abs=0.05)
    assert height == pytest.approx(target_size[1], abs=0.05)


def test_ccc_heatmap_dot_auto_sizes_when_figsize_is_none(raw_liana_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type="dot",
        display_by="interaction",
        top_n=4,
        pvalue_threshold=0.2,
        figsize=None,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    width, height = fig.get_size_inches()
    assert width > 0
    assert height > 0


def test_ccc_heatmap_diff_heatmap_auto_resolves_liana_comparison_adata(
    raw_liana_adata: AnnData,
    raw_liana_comparison_adata: AnnData,
) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        comparison_adata=raw_liana_comparison_adata,
        plot_type="diff_heatmap",
        top_n=2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_network_plot_auto_resolves_liana_adata_with_custom_uns_key() -> None:
    raw_adata = _build_raw_liana_adata(uns_key="custom_results")
    fig, ax = ov.pl.ccc_network_plot(
        raw_adata,
        plot_type="circle",
        display_by="interaction",
        result_uns_key="custom_results",
        top_n=3,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_heatmap_auto_resolves_cpdb_adata(raw_cpdb_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        raw_cpdb_adata,
        plot_type="heatmap",
        display_by="interaction",
        top_n=2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_network_plot_auto_resolves_cpdb_adata(raw_cpdb_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_network_plot(
        raw_cpdb_adata,
        plot_type="circle",
        display_by="interaction",
        top_n=2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_heatmap_source_target_dot_warns_as_compatibility_alias(raw_liana_adata: AnnData) -> None:
    with pytest.warns(FutureWarning, match="compatibility alias"):
        fig, ax = ov.pl.ccc_heatmap(
            raw_liana_adata,
            plot_type="source_target_dot",
            display_by="interaction",
            top_n=3,
            pvalue_threshold=0.2,
            show=False,
        )
    _assert_figure_and_axes(fig, ax)


def test_ccc_heatmap_source_target_tile_warns_as_compatibility_alias(raw_liana_adata: AnnData) -> None:
    with pytest.warns(FutureWarning, match="compatibility alias"):
        fig, ax = ov.pl.ccc_heatmap(
            raw_liana_adata,
            plot_type="source_target_tile",
            display_by="interaction",
            top_n=3,
            pvalue_threshold=0.2,
            show=False,
        )
    _assert_figure_and_axes(fig, ax)


def test_ccc_stat_plot_auto_resolves_liana_adata(raw_liana_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        raw_liana_adata,
        plot_type="pathway_summary",
        top_n=4,
        min_expression=0.0,
        strength_threshold=0.0,
        min_significant_pairs=1,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)


def test_ccc_heatmap_tile_uses_distinct_ligand_and_receptor_row_labels(raw_liana_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_heatmap(
        raw_liana_adata,
        plot_type="tile",
        display_by="interaction",
        top_n=6,
        pvalue_threshold=0.2,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    heatmap_axes = [axis for axis in fig.axes if axis.get_xlabel() == "Cell type"]
    assert len(heatmap_axes) >= 2
    left_labels = [tick.get_text() for tick in heatmap_axes[0].get_yticklabels() if tick.get_text()]
    right_labels = [tick.get_text() for tick in heatmap_axes[1].get_yticklabels() if tick.get_text()]
    assert left_labels
    assert right_labels
    assert left_labels != right_labels


def test_ccc_plot_plain_anndata_without_comm_or_liana_raises() -> None:
    plain = AnnData(X=np.zeros((1, 1), dtype=float))
    with pytest.raises(ValueError, match="Could not find a supported communication result"):
        ov.pl.ccc_heatmap(plain, show=False)


def test_cellchatviz_aggregates_duplicate_sender_receiver_rows() -> None:
    duplicate_comm_adata = _build_comm_adata_with_duplicate_pairs()
    viz = ov.pl.CellChatViz(duplicate_comm_adata)

    mean_df = viz.mean(count_min=1.0)
    pvalue_df = viz.pvalue(count_min=1.0)
    pathway_networks = viz.compute_pathway_network(pvalue_threshold=0.05)
    pathway_comm = viz.compute_pathway_communication(
        method="mean",
        min_lr_pairs=1,
        min_expression=1.0,
    )

    assert mean_df.loc["A", "B"] == pytest.approx(6.0)
    assert pvalue_df.loc["A", "B"] == pytest.approx((0.01 + 0.03 + 0.04) / 3.0)
    assert pathway_networks["TNF"][viz.cell_types.index("A"), viz.cell_types.index("B")] == pytest.approx(6.0)
    assert pathway_comm["TNF"]["communication_matrix"].loc["A", "B"] == pytest.approx(2.0)
    assert pathway_comm["TNF"]["pvalue_matrix"].loc["A", "B"] == pytest.approx(0.01)
    assert pathway_comm["TNF"]["n_valid_interactions"].loc["A", "B"] == pytest.approx(3.0)


def test_cellchatviz_get_signaling_pathways_respects_custom_threshold() -> None:
    duplicate_comm_adata = _build_comm_adata_with_duplicate_pairs()
    viz = ov.pl.CellChatViz(duplicate_comm_adata)

    significant_pathways, pathway_stats = viz.get_signaling_pathways(
        min_interactions=1,
        pathway_pvalue_threshold=0.02,
        method="mean",
        correction_method=None,
        min_expression=1.0,
    )
    tighter_pathways, _ = viz.get_signaling_pathways(
        min_interactions=1,
        pathway_pvalue_threshold=0.017,
        method="mean",
        correction_method=None,
        min_expression=1.0,
    )

    assert significant_pathways == ["TNF"]
    assert tighter_pathways == []
    assert pathway_stats["TNF"]["n_significant_interactions"] == 3
    assert pathway_stats["TNF"]["significance_rate"] == pytest.approx(3.0 / 5.0)
    assert pathway_stats["TNF"]["significant_cell_pairs"] == ["A|B", "B|A"]


def test_ccc_violin_uses_non_vertical_interaction_labels(comm_adata: AnnData) -> None:
    fig, ax = ov.pl.ccc_stat_plot(
        comm_adata,
        plot_type="violin",
        top_n=4,
        show=False,
    )
    _assert_figure_and_axes(fig, ax)
    assert all(tick.get_rotation() == 0 for tick in ax.get_xticklabels())
