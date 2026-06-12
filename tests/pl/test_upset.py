import matplotlib

matplotlib.use("Agg")

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import omicverse as ov


def _gene_sets():
    return {
        "A": {"g1", "g2", "g3", "g4"},
        "B": {"g2", "g3", "g5", "g6"},
        "C": {"g3", "g4", "g6", "g7"},
        "D": {"g1", "g3", "g8"},
        "E": {"g3", "g9", "g10"},
    }


def test_upset_is_exported_and_draws_many_sets(tmp_path):
    fig, axes = ov.pl.upset(_gene_sets(), top_n=10)

    assert fig is not None
    assert set(axes) == {"intersections", "matrix", "set_sizes"}
    assert axes["intersections"].get_ylabel() == "Intersection size"
    assert axes["matrix"].get_xlabel() == "Intersections"
    assert axes["intersections"].get_title() == ""

    out = tmp_path / "upset.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()
    assert out.stat().st_size > 0


def test_upset_validates_empty_sets():
    with pytest.raises(ValueError, match="empty sets"):
        ov.pl.upset({"A": {"g1"}, "B": set()})


def test_upset_validates_missing_dict_keys():
    with pytest.raises(KeyError, match="Keys not found in dictionary input: missing"):
        ov.pl.upset({"A": {"g1"}, "B": {"g2"}}, keys=["A", "missing"])


def test_upset_accepts_adata_obs_boolean_sets(tmp_path):
    obs = pd.DataFrame(
        {
            "high_CD3D": [True, True, False, False, True],
            "cycling": [False, True, True, False, True],
            "treated": [True, False, True, False, False],
        },
        index=[f"cell_{i}" for i in range(5)],
    )
    adata = AnnData(np.ones((5, 2)), obs=obs, var=pd.DataFrame(index=["g1", "g2"]))

    fig, axes = ov.pl.upset(adata, keys=["high_CD3D", "cycling", "treated"], axis="obs")

    assert fig is not None
    assert axes["set_sizes"].get_xlabel() == "Set size"
    out = tmp_path / "adata_upset.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.stat().st_size > 0


def test_upset_accepts_adata_var_boolean_sets(tmp_path):
    var = pd.DataFrame(
        {
            "pathway_A": [True, True, False, False],
            "pathway_B": [False, True, True, False],
        },
        index=[f"gene_{i}" for i in range(4)],
    )
    adata = AnnData(np.ones((2, 4)), obs=pd.DataFrame(index=["cell_1", "cell_2"]), var=var)

    fig, axes = ov.pl.upset(adata, keys=["pathway_A", "pathway_B"], axis="var")

    assert fig is not None
    out = tmp_path / "var_upset.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.stat().st_size > 0


def test_upset_requires_boolean_adata_keys():
    obs = pd.DataFrame({"cell_type": ["T", "B"]}, index=["cell_1", "cell_2"])
    adata = AnnData(np.ones((2, 1)), obs=obs, var=pd.DataFrame(index=["g1"]))

    with pytest.raises(TypeError, match="must be boolean"):
        ov.pl.upset(adata, keys=["cell_type"])


def test_upset_accepts_style_callable_and_uses_rc_cycle():
    calls = []

    def custom_style(show_monitor=True):
        calls.append(show_monitor)
        mpl.rcParams["axes.prop_cycle"] = cycler(color=["#123456", "#abcdef"])

    with mpl.rc_context():
        fig, axes = ov.pl.upset(_gene_sets(), top_n=3, style=custom_style)

    first_bar = axes["intersections"].patches[0]
    assert calls == [False]
    assert mpl.colors.to_hex(first_bar.get_facecolor()) == "#123456"
    plt.close(fig)


def test_upset_grid_false_overrides_global_grid():
    with mpl.rc_context({"axes.grid": True}):
        fig, axes = ov.pl.upset(_gene_sets(), top_n=3, grid=False)

    assert not any(line.get_visible() for line in axes["intersections"].get_ygridlines())
    assert not any(line.get_visible() for line in axes["set_sizes"].get_xgridlines())
    plt.close(fig)


def test_upset_grid_true_applies_to_set_size_axis():
    with mpl.rc_context({"axes.grid": False}):
        fig, axes = ov.pl.upset(_gene_sets(), top_n=3, grid=True)

    assert any(line.get_visible() for line in axes["intersections"].get_ygridlines())
    assert any(line.get_visible() for line in axes["set_sizes"].get_xgridlines())
    plt.close(fig)


def test_upset_accepts_per_intersection_bar_colors():
    colors = ["#111111", "#222222", "#333333"]

    fig, axes = ov.pl.upset(_gene_sets(), top_n=3, intersection_color=colors)

    bar_colors = [
        mpl.colors.to_hex(patch.get_facecolor())
        for patch in axes["intersections"].patches[:3]
    ]
    assert bar_colors == colors
    plt.close(fig)


def test_upset_dict_colors_only_override_matching_columns_and_rows():
    sets = {
        "A": {"shared_1", "shared_2", "shared_3", "a"},
        "B": {"shared_1", "shared_2", "shared_3", "b"},
        "C": {"shared_1", "shared_2", "shared_3", "c"},
    }

    with mpl.rc_context({"axes.prop_cycle": cycler(color=["#101010", "#202020"])}):
        fig, axes = ov.pl.upset(
            sets,
            top_n=10,
            intersection_color={"A&B&C": "#abcdef"},
            set_size_color={"A": "#fedcba"},
            matrix_color={"A": "#123456"},
        )

    intersection_colors = [
        mpl.colors.to_hex(patch.get_facecolor())
        for patch in axes["intersections"].patches
    ]
    set_size_colors = [
        mpl.colors.to_hex(patch.get_facecolor())
        for patch in axes["set_sizes"].patches
    ]
    active_dot_colors = [
        mpl.colors.to_hex(collection.get_facecolors()[0])
        for collection in axes["matrix"].collections[1::2]
    ]

    assert intersection_colors[0] == "#abcdef"
    assert all(color == "#101010" for color in intersection_colors[1:])
    assert set_size_colors[0] == "#fedcba"
    assert all(color == "#202020" for color in set_size_colors[1:])
    assert active_dot_colors[0] == "#123456"
    assert all(color == "#101010" for color in active_dot_colors[1:])
    plt.close(fig)


def test_upset_intersection_color_keys_are_order_insensitive():
    sets = {
        "A": {"shared_1", "shared_2", "a"},
        "B": {"shared_1", "shared_2", "b"},
        "C": {"c"},
    }

    fig, axes = ov.pl.upset(
        sets,
        top_n=3,
        intersection_color={"B&A": "#abcdef"},
    )

    first_bar_color = mpl.colors.to_hex(axes["intersections"].patches[0].get_facecolor())
    assert first_bar_color == "#abcdef"
    plt.close(fig)


def test_upset_accepts_size_and_grid_style_parameters():
    fig, axes = ov.pl.upset(
        _gene_sets(),
        top_n=3,
        bar_width=0.5,
        set_bar_height=0.35,
        dot_size=60,
        empty_dot_size=20,
        line_width=2.5,
        grid_color="#eeeeee",
        grid_linewidth=1.7,
        count_fontsize=9,
        count_offset=0.4,
        matrix_alpha=0.8,
        empty_alpha=0.3,
    )

    assert axes["intersections"].patches[0].get_width() == pytest.approx(0.5)
    assert axes["set_sizes"].patches[0].get_height() == pytest.approx(0.35)
    assert axes["matrix"].collections[0].get_sizes()[0] == pytest.approx(20)
    assert axes["matrix"].collections[1].get_sizes()[0] == pytest.approx(60)
    assert axes["matrix"].collections[0].get_alpha() == pytest.approx(0.3)
    assert axes["matrix"].collections[1].get_alpha() == pytest.approx(0.8)
    assert axes["matrix"].lines[0].get_linewidth() == pytest.approx(2.5)
    assert axes["intersections"].texts[0].get_fontsize() == pytest.approx(9)

    gridline = next(line for line in axes["intersections"].get_ygridlines() if line.get_visible())
    assert mpl.colors.to_hex(gridline.get_color()) == "#eeeeee"
    assert gridline.get_linewidth() == pytest.approx(1.7)
    plt.close(fig)


def test_upset_accepts_manual_set_and_intersection_order():
    fig, axes = ov.pl.upset(
        _gene_sets(),
        top_n=4,
        set_order=["C", "A"],
        intersection_order=["A&B&C&D&E", ("A", "D")],
        intersection_color={"A&B&C&D&E": "#abcdef"},
    )

    ytick_labels = [label.get_text() for label in axes["set_sizes"].get_yticklabels()]
    first_bar_color = mpl.colors.to_hex(axes["intersections"].patches[0].get_facecolor())
    assert ytick_labels[:2] == ["C", "A"]
    assert first_bar_color == "#abcdef"
    plt.close(fig)


def test_upset_intersection_order_is_order_insensitive():
    fig, axes = ov.pl.upset(
        _gene_sets(),
        top_n=4,
        intersection_order=[("E", "D", "C", "B", "A")],
        intersection_color={"A&B&C&D&E": "#abcdef"},
    )

    first_bar_color = mpl.colors.to_hex(axes["intersections"].patches[0].get_facecolor())
    assert first_bar_color == "#abcdef"
    plt.close(fig)


def test_upset_accepts_height_ratios():
    fig, axes = ov.pl.upset(_gene_sets(), top_n=3, height_ratios=(1, 3))

    top_height = axes["intersections"].get_position().height
    lower_height = axes["matrix"].get_position().height
    assert lower_height / top_height == pytest.approx(3, rel=0.05)
    plt.close(fig)


def test_upset_rejects_too_many_sets_by_default():
    obs = pd.DataFrame(
        {f"flag_{i}": [True, False, True] for i in range(13)},
        index=["cell_1", "cell_2", "cell_3"],
    )
    adata = AnnData(np.ones((3, 1)), obs=obs, var=pd.DataFrame(index=["gene_1"]))

    with pytest.raises(ValueError, match="max_sets"):
        ov.pl.upset(adata)
