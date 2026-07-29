"""Tests for the embeddable marsilea-backed ov.pl.heatmap.

The load-bearing property is embedding: heatmap(data, figure=sub) must draw
entirely inside the caller's SubFigure and leave sibling subfigures untouched.
That is what distinguishes it from ov.pl.complexheatmap, which self-builds a
figure.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import SubFigure
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

marsilea = pytest.importorskip("marsilea")

from omicverse.pl._marsilea_heatmap import heatmap


def _block_data(seed=0):
    """Two clearly separated row blocks so clustering has something to do."""
    rng = np.random.RandomState(seed)
    top = rng.normal(5.0, 0.2, size=(6, 8))
    bottom = rng.normal(-5.0, 0.2, size=(6, 8))
    matrix = np.vstack([top, bottom])
    # Interleave the rows so the input order is not already clustered.
    order = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
    frame = pd.DataFrame(
        matrix[order],
        index=[f"r{i}" for i in order],
        columns=[f"c{j}" for j in range(8)],
    )
    return frame


def test_embeds_into_subfigure_and_leaves_sibling_untouched():
    fig = plt.figure(figsize=(10, 4))
    left, right = fig.subfigures(1, 2)

    data = _block_data()
    h = heatmap(data, figure=left, row_cluster=True, col_cluster=True)

    # Every axis the heatmap drew must belong to the target SubFigure...
    assert isinstance(left, SubFigure)
    assert len(left.axes) > 0
    for ax in left.axes:
        assert ax.figure is left
        assert ax.figure is not fig

    # ...and the sibling SubFigure must be completely untouched.
    assert len(right.axes) == 0

    # The marsilea object reports the SubFigure as its figure.
    assert h.figure is left
    plt.close(fig)


def test_subfigure_isolation_would_fail_against_parent_figure():
    """Guard: rendering into the parent fig (not the sub) breaks the isolation
    assertions. Confirms the test above actually tests embedding."""
    fig = plt.figure(figsize=(10, 4))
    left, right = fig.subfigures(1, 2)

    data = _block_data()
    # Deliberately render into the PARENT figure instead of the SubFigure.
    h = heatmap(data, figure=fig)

    # The isolation assertions from the real test would now fail: axes live in
    # fig, not in the sub, and neither sub gets its own axes.
    assert h.figure is fig
    assert len(left.axes) == 0 and len(right.axes) == 0
    with pytest.raises(AssertionError):
        assert h.figure is left
    plt.close(fig)


def test_standalone_creates_exactly_one_figure():
    before = set(plt.get_fignums())
    data = _block_data()
    h = heatmap(data, figure=None)
    after = set(plt.get_fignums())

    assert len(after - before) == 1
    assert h.figure is not None
    plt.close(h.figure)


def test_anndata_row_annotation_from_obs():
    data = _block_data()
    adata = AnnData(
        X=data.values,
        obs=pd.DataFrame(
            {"celltype": ["A"] * 6 + ["B"] * 6},
            index=data.index,
        ),
        var=pd.DataFrame(index=data.columns),
    )
    h = heatmap(adata, row_annotation="celltype", figure=None)

    # The annotation strip must carry one entry per row.
    from omicverse.pl._marsilea_heatmap import _collect_annotations

    ann = _collect_annotations("celltype", adata, adata.obs, adata.n_obs, "row")
    assert len(ann) == 1
    name, series = ann[0]
    assert name == "celltype"
    assert len(series) == adata.n_obs == 12
    plt.close(h.figure)


def test_zscore_rows_changes_drawn_data():
    data = _block_data()

    h_plain = heatmap(data, figure=None, z_score=None, row_cluster=False,
                      col_cluster=False)
    h_z = heatmap(data, figure=None, z_score=0, row_cluster=False,
                  col_cluster=False)

    # z_score=0 standardises each row -> row means ~0, unlike the raw data.
    from omicverse.pl._marsilea_heatmap import _apply_zscore, _as_frame

    frame, _ = _as_frame(data)
    z = _apply_zscore(frame, 0)
    assert np.allclose(z.values.mean(axis=1), 0.0, atol=1e-9)
    # The raw block data has strongly non-zero row means.
    assert np.abs(frame.values.mean(axis=1)).max() > 1.0
    # And the transform actually changed the values.
    assert not np.allclose(z.values, frame.values)
    plt.close(h_plain.figure)
    plt.close(h_z.figure)


def test_clustering_reorders_rows_and_exposes_order():
    data = _block_data()
    h = heatmap(data, figure=None, row_cluster=True, row_dendrogram=True)

    assert hasattr(h, "row_order")
    assert sorted(h.row_order) == sorted(data.index)
    # On this interleaved two-block data the clustered order must differ.
    assert list(h.row_order) != list(data.index)
    # The two blocks should end up contiguous after clustering.
    block = ["A" if int(name[1:]) < 6 else "B" for name in h.row_order]
    # Count transitions between blocks; a clean clustering has exactly one.
    transitions = sum(a != b for a, b in zip(block, block[1:]))
    assert transitions == 1
    plt.close(h.figure)


def test_mismatched_annotation_length_raises():
    data = _block_data()  # 12 rows
    with pytest.raises(ValueError, match="entries but the data has 12 rows"):
        heatmap(data, figure=None, row_annotation={"grp": ["x", "y", "z"]})


# --------------------------------------------------------------------------
# rect=: confining the block to part of a host figure
#
# marsilea derives a figure size from its own content, resizes whatever figure
# it is handed, and places its axes as fractions of that size — so a heatmap
# dropped into a multi-panel figure takes the whole canvas. `rect` sizes the
# cell so the block fills a region at true scale, then translates it there.
# --------------------------------------------------------------------------


def test_rect_keeps_the_host_figure_size():
    fig = plt.figure(figsize=(9.0, 5.0))
    heatmap(_block_data(), figure=fig, rect=(0.05, 0.05, 0.9, 0.4))
    assert tuple(fig.get_size_inches()) == (9.0, 5.0)


def test_rect_confines_every_axes_to_the_region():
    fig = plt.figure(figsize=(9.0, 5.0))
    x0, y0, w, h = 0.10, 0.08, 0.80, 0.44
    heatmap(_block_data(), figure=fig, rect=(x0, y0, w, h),
            row_cluster=True, col_cluster=True, row_dendrogram=True)

    assert fig.axes, "nothing was drawn"
    for ax in fig.axes:
        pos = ax.get_position()
        assert pos.x0 >= x0 - 1e-6 and pos.x1 <= x0 + w + 1e-6, "axes left the region in x"
        assert pos.y0 >= y0 - 1e-6 and pos.y1 <= y0 + h + 1e-6, "axes left the region in y"


def test_rect_does_not_disturb_a_panel_already_on_the_figure():
    fig = plt.figure(figsize=(9.0, 5.0))
    host = fig.add_axes([0.08, 0.62, 0.35, 0.30])
    before = host.get_position().bounds

    heatmap(_block_data(), figure=fig, rect=(0.05, 0.05, 0.9, 0.45))

    assert host.get_position().bounds == pytest.approx(before)


def test_rect_fills_the_region_rather_than_shrinking_inside_it():
    """A wider region must give a wider block — the cell is resized, not the
    block rescaled, so the block tracks the region it was given."""
    def block_width(region_w):
        fig = plt.figure(figsize=(12.0, 5.0))
        heatmap(_block_data(), figure=fig, rect=(0.05, 0.05, region_w, 0.45))
        return max(a.get_position().x1 for a in fig.axes) - \
            min(a.get_position().x0 for a in fig.axes)

    assert block_width(0.85) > block_width(0.45) * 1.5


def test_rect_without_a_figure_is_refused():
    with pytest.raises(TypeError, match="pass `figure="):
        heatmap(_block_data(), rect=(0, 0, 1, 1))
