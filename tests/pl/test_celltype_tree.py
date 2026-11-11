import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.colors import to_rgba
from matplotlib.patches import Wedge
from scipy.sparse import csr_matrix

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


def _track_colors(ax):
    wedges = (p for p in ax.patches if isinstance(p, Wedge) and p.r > 1)
    return [p.get_facecolor() for p in sorted(wedges, key=lambda p: (p.r, -p.theta1))]


def test_celltype_tree_keeps_nullable_track_values_missing_after_standardization():
    cells = pd.DataFrame({
        "cell_type": ["A", "B", "C"],
        "score": pd.array([1, None, 3], dtype="Int64"),
        "constant": pd.array([True, None, True], dtype="boolean"),
    })
    ax = ov.pl.celltype_tree(cells, tracks=["score", "constant"], standardize=True, vmax=1, show=False)
    try:
        cmap = plt.get_cmap("RdBu_r")
        gray = to_rgba("#e0e0e0")
        np.testing.assert_allclose(_track_colors(ax), [cmap(0.0), gray, cmap(1.0), cmap(0.5), gray, cmap(0.5)])
    finally:
        plt.close(ax.figure)


def test_celltype_tree_summarizes_anndata_by_leaf():
    adata = AnnData(
        csr_matrix([[1.0], [3.0], [3.0], [1.0], [1.0]]),
        obs=pd.DataFrame({
            "lineage": pd.Categorical(["T", "B", "T", "T", "T"], categories=["B", "T", "NK"]),
            "cell_type": ["Shared", "Shared", "Shared", "Memory", "Memory"],
            "score": [1.0, -1.0, 1.0, 0.0, 0.0],
            "constant": [0.1, 0.1, 0.1, 0.1, 0.1],
            "marker": ["*", None, "*", "", ""],
        }, index=list("abcde")),
        var=pd.DataFrame(index=["gene"]),
    )
    original = adata.copy()
    ax = ov.pl.celltype_tree(
        adata, hierarchy="lineage", tracks=["gene", "score", "constant"],
        track_markers={"gene": "marker"}, standardize=True, vmax=2, show=False,
    )
    try:
        expected = plt.get_cmap("RdBu_r")
        extreme = np.sqrt(1.5)
        colors = expected((np.array([0, -extreme, extreme, extreme, 0, -extreme, 0, 0, 0]) + 2) / 4)
        np.testing.assert_allclose(_track_colors(ax), colors)
        assert [t.get_text() for t in ax.texts if "%" in t.get_text()] == [
            "Shared (40%)", "Memory (40%)", "Shared (20%)",
        ]
        assert [t.get_text() for t in ax.texts].count("*") == 1
        pd.testing.assert_frame_equal(adata.obs, original.obs)
        np.testing.assert_array_equal(adata.X.toarray(), original.X.toarray())
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("track, options, expected", [
    ("gene", {"layer": "counts"}, [1.0, 0.0]),
    ("raw_gene", {"use_raw": True}, [0.0, 1.0]),
])
def test_celltype_tree_reads_anndata_expression_source(track, options, expected):
    adata = AnnData(
        np.array([[-1.0, -2.0], [1.0, 2.0]]),
        obs=pd.DataFrame({"cell_type": ["A", "B"], "score": [5.0, 5.0]}, index=["a", "b"]),
        var=pd.DataFrame(index=["gene", "raw_gene"]),
    )
    adata.raw = adata.copy()
    adata = adata[:, ["gene"]].copy()
    adata.layers["counts"] = csr_matrix([[2.0], [-2.0]])
    ax = ov.pl.celltype_tree(
        adata, tracks=[track, "score"], proportion=False,
        standardize=True, vmax=1, show=False, **options,
    )
    try:
        np.testing.assert_allclose(_track_colors(ax), plt.get_cmap("RdBu_r")([*expected, 0.5, 0.5]))
        assert [t.get_text() for t in ax.texts] == ["A", "B", track.replace("_", " "), "score"]
    finally:
        plt.close(ax.figure)


def test_celltype_tree_rejects_missing_anndata_leaf_labels():
    adata = AnnData(obs=pd.DataFrame({"cell_type": ["A", None]}, index=["a", "b"]))
    with pytest.raises(ValueError, match="missing"):
        ov.pl.celltype_tree(adata, show=False)
