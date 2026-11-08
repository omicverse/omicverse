import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import Wedge

import omicverse as ov


def test_celltype_tree_keeps_interleaved_subtrees_contiguous():
    cells = pd.DataFrame({
        "lineage": ["T", "B", "T", "T", "B"],
        "family": ["Shared", "Shared", "Memory", "Shared", "Shared"],
        "cell_type": ["T1", "B1", "T2", "T3", "B2"],
    })
    fig, ax = plt.subplots()
    try:
        result = ov.pl.celltype_tree(
            cells, hierarchy=["lineage", "family"], label="cell_type", ax=ax, show=False,
        )
        assert result is ax
        ordered_labels = []
        for text in ax.texts:
            if text.get_text() in set(cells["cell_type"]):
                x, y = text.get_position()
                angle = (90 - np.degrees(np.arctan2(y, x))) % 360
                ordered_labels.append((angle, text.get_text()))
        assert [label for _, label in sorted(ordered_labels)] == ["T1", "T3", "T2", "B1", "B2"]
    finally:
        plt.close(fig)


def test_celltype_tree_colors_each_track_from_its_own_column():
    cells = pd.DataFrame({
        "lineage": ["T", "B", "T"],
        "cell_type": ["CD4", "Naive B", "CD8"],
        "percent": [100, 0, 0],
        "ATAC": [-1.0, np.nan, 1.0],
        "RNA": [1.0, 0.0, -1.0],
        "rna_marker": ["*", "", None],
    })
    original = cells.copy(deep=True)
    ax = ov.pl.celltype_tree(
        cells, hierarchy=["lineage"], label="cell_type", proportion="percent",
        tracks=["ATAC", "RNA"], track_markers={"RNA": "rna_marker"}, vmax=1, show=False,
    )
    try:
        wedges = [p for p in ax.patches if isinstance(p, Wedge) and p.r > 1]
        expected = plt.get_cmap("RdBu_r")
        radii = sorted({p.r for p in wedges})
        assert len(radii) == 2
        tracks = [sorted((p for p in wedges if p.r == radius), key=lambda p: -p.theta1)
                  for radius in radii]
        np.testing.assert_allclose([p.get_facecolor() for p in tracks[0]],
                                   [expected(0.0), expected(1.0), to_rgba("#e0e0e0")])
        np.testing.assert_allclose([p.get_facecolor() for p in tracks[1]],
                                   [expected(1.0), expected(0.0), expected(0.5)])
        marker, = [t for t in ax.texts if t.get_text() == "*"]
        assert tracks[1][0].r - tracks[1][0].width < np.hypot(*marker.get_position()) < tracks[1][0].r
        assert "CD4 (100%)" in {t.get_text() for t in ax.texts}
        pd.testing.assert_frame_equal(cells, original)
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("columns, options, error, message", [
    ({"lineage": ["T", "T"], "cell_type": ["CD4", "CD4"]}, {}, ValueError, "Leaf paths"),
    ({"lineage": ["T"], "cell_type": ["CD4"]}, {"tracks": ["RNA"]}, KeyError, "RNA"),
    ({"lineage": ["T"], "cell_type": ["CD4"], "RNA": [np.inf]},
     {"tracks": ["RNA"]}, ValueError, "finite"),
])
def test_celltype_tree_rejects_invalid_inputs(columns, options, error, message):
    with pytest.raises(error, match=message):
        ov.pl.celltype_tree(pd.DataFrame(columns), hierarchy=["lineage"], label="cell_type",
                            show=False, **options)
