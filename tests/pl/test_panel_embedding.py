"""`ov.pl.upset` and `ov.pl.dotplot` can be one panel of a figure.

Both used to build their own figure unconditionally — `upset` with
`plt.figure`, `dotplot` through marsilea's layout engine — so neither could sit
in a composition. `dotplot` even advertised an `ax=` argument that the marsilea
path ignored.

The contract tested here: the host figure keeps its size, an axes already on it
is untouched, and the new axes land inside the region asked for.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from omicverse.pl import dotplot, upset


def _sets():
    return {"RNA": {"a", "b", "c", "d"},
            "ATAC": {"b", "c", "e"},
            "WES": {"c", "d", "f", "g"}}


def _adata(n_obs=40, n_var=8):
    rng = np.random.default_rng(0)
    X = rng.gamma(2.0, 1.0, size=(n_obs, n_var))
    obs = pd.DataFrame({"grp": rng.choice(["A", "B", "C"], n_obs)},
                       index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_var)])
    return AnnData(X=X, obs=obs, var=var)


class TestUpsetAsPanel:
    def test_region_and_host_are_respected(self):
        fig = plt.figure(figsize=(12.0, 6.0))
        host = fig.add_axes([0.05, 0.62, 0.30, 0.30])
        before = host.get_position().bounds

        x0, y0, w, h = 0.45, 0.08, 0.50, 0.42
        upset(_sets(), figure=fig, rect=(x0, y0, w, h))

        assert tuple(fig.get_size_inches()) == (12.0, 6.0)
        assert host.get_position().bounds == pytest.approx(before)
        for axes in fig.axes:
            if axes is host:
                continue
            pos = axes.get_position()
            assert pos.x0 >= x0 - 1e-6 and pos.x1 <= x0 + w + 1e-6
            assert pos.y0 >= y0 - 1e-6 and pos.y1 <= y0 + h + 1e-6

    def test_standalone_behaviour_is_unchanged(self):
        fig, axes = upset(_sets(), figsize=(7.0, 4.0))
        assert tuple(fig.get_size_inches()) == (7.0, 4.0)
        assert axes, "no axes returned"

    def test_rect_without_a_figure_is_refused(self):
        with pytest.raises(TypeError, match="pass `figure="):
            upset(_sets(), rect=(0, 0, 1, 1))


class TestDotplotAsPanel:
    """`dotplot` can be confined to a region, and its legends can move.

    marsilea sizes its figure from the heatmap cell plus a fixed overhead, so
    the cell is solved for the region (one throwaway render measures the
    overhead) and the block is then translated — never rescaled.
    """

    def test_host_figure_and_neighbours_survive(self):
        adata = _adata()
        fig = plt.figure(figsize=(13.0, 9.0))
        host = fig.add_axes([0.05, 0.80, 0.25, 0.15])
        before = host.get_position().bounds

        dotplot(adata, list(adata.var_names[:4]), "grp", figure=fig,
                rect=(0.10, 0.10, 0.45, 0.55), show=False)

        assert tuple(fig.get_size_inches()) == (13.0, 9.0)
        assert host.get_position().bounds == pytest.approx(before)

    def test_block_stays_inside_the_region(self):
        adata = _adata()
        fig = plt.figure(figsize=(13.0, 9.0))
        existing = set(fig.axes)
        x0, y0, w, h = 0.10, 0.10, 0.45, 0.55

        dotplot(adata, list(adata.var_names[:4]), "grp", figure=fig,
                rect=(x0, y0, w, h), show=False)

        added = [a for a in fig.axes if a not in existing]
        assert added, "nothing was drawn"
        assert min(a.get_position().x0 for a in added) >= x0 - 1e-6
        assert max(a.get_position().x1 for a in added) <= x0 + w + 1e-6
        assert min(a.get_position().y0 for a in added) >= y0 - 1e-6
        assert max(a.get_position().y1 for a in added) <= y0 + h + 1e-6

    def test_a_wider_region_gives_a_wider_block(self):
        """The cell is solved for the region, so the block tracks it."""
        def block_width(region_w):
            adata = _adata()
            fig = plt.figure(figsize=(16.0, 9.0))
            existing = set(fig.axes)
            dotplot(adata, list(adata.var_names[:4]), "grp", figure=fig,
                    rect=(0.05, 0.10, region_w, 0.6), show=False)
            added = [a for a in fig.axes if a not in existing]
            return (max(a.get_position().x1 for a in added)
                    - min(a.get_position().x0 for a in added))

        assert block_width(0.80) > block_width(0.40) * 1.4

    def test_rect_without_a_figure_is_refused(self):
        adata = _adata()
        with pytest.raises(TypeError, match="pass `figure="):
            dotplot(adata, list(adata.var_names[:3]), "grp",
                    rect=(0, 0, 1, 1), show=False)

    def test_legend_side_reaches_marsilea(self):
        adata = _adata()
        for side in ("right", "bottom", "left", "top"):
            m = dotplot(adata, list(adata.var_names[:4]), "grp",
                        legend_side=side, show=False)
            assert m is not None, f"legend_side={side!r} produced nothing"

    def test_legends_can_be_switched_off(self):
        adata = _adata()
        assert dotplot(adata, list(adata.var_names[:4]), "grp",
                       legend=False, show=False) is not None

    def test_an_unknown_side_is_refused(self):
        adata = _adata()
        with pytest.raises(ValueError, match="must be 'right', 'left'"):
            dotplot(adata, list(adata.var_names[:3]), "grp",
                    legend_side="sideways", show=False)
