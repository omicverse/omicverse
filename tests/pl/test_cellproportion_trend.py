from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def cells():
    return pd.DataFrame(
        {
            "cell_type": ["A", "A", "B", "A", "B", "B", "A", "B"],
            "sample": ["S1", "S1", "S1", "S2", "S2", "S2", "S3", "S3"],
        }
    )


def test_cellproportion_trend_connects_cumulative_boundaries(cells):
    from omicverse.pl import cellproportion

    fig, ax = plt.subplots()
    cellproportion(
        cells,
        celltype_clusters="cell_type",
        groupby="sample",
        groupby_li=["S1", "S2", "S3"],
        trend=True,
        trend_kwargs={"linewidth": 2.5, "linestyle": "--"},
        ax=ax,
    )

    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_ydata()) == pytest.approx([2 / 3, 1 / 3, 1 / 2])
    assert ax.lines[0].get_linewidth() == pytest.approx(2.5)
    assert ax.lines[0].get_linestyle() == "--"


def test_cellproportion_trend_respects_transpose(cells):
    from omicverse.pl import cellproportion

    fig, ax = plt.subplots()
    cellproportion(
        cells,
        celltype_clusters="cell_type",
        groupby="sample",
        groupby_li=["S1", "S2", "S3"],
        trend=True,
        transpose=True,
        ax=ax,
    )

    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_xdata()) == pytest.approx([2 / 3, 1 / 3, 1 / 2])


def test_cellproportion_default_has_no_trend_lines(cells):
    from omicverse.pl import cellproportion

    fig, ax = plt.subplots()
    cellproportion(
        cells,
        celltype_clusters="cell_type",
        groupby="sample",
        ax=ax,
    )

    assert not ax.lines
