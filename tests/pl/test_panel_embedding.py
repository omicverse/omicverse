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
    """`dotplot` maps onto a region, and its legends can move.

    The block's extent cannot be read back from marsilea, so it is mapped onto
    the region proportionally rather than fitted to it — the same thing cnsplots
    does. Geometry scales, text does not, which is why a small `fontsize` is
    part of using it in a panel.
    """

    def test_block_lands_exactly_on_the_region(self):
        adata = _adata()
        fig = plt.figure(figsize=(13.0, 9.0))
        existing = set(fig.axes)
        x0, y0, w, h = 0.10, 0.10, 0.45, 0.55

        dotplot(adata, list(adata.var_names[:4]), "grp", figure=fig,
                rect=(x0, y0, w, h), show=False, fontsize=6)

        added = [a for a in fig.axes if a not in existing]
        assert added, "nothing was drawn"
        assert min(a.get_position().x0 for a in added) == pytest.approx(x0, abs=1e-6)
        assert max(a.get_position().x1 for a in added) == pytest.approx(x0 + w, abs=1e-6)
        assert min(a.get_position().y0 for a in added) == pytest.approx(y0, abs=1e-6)
        assert max(a.get_position().y1 for a in added) == pytest.approx(y0 + h, abs=1e-6)

    def test_host_figure_and_neighbours_survive(self):
        adata = _adata()
        fig = plt.figure(figsize=(13.0, 9.0))
        host = fig.add_axes([0.05, 0.80, 0.25, 0.15])
        before = host.get_position().bounds

        dotplot(adata, list(adata.var_names[:4]), "grp", figure=fig,
                rect=(0.10, 0.10, 0.45, 0.55), show=False, fontsize=6)

        assert tuple(fig.get_size_inches()) == (13.0, 9.0)
        assert host.get_position().bounds == pytest.approx(before)

    def test_rect_without_a_figure_is_refused(self):
        adata = _adata()
        with pytest.raises(TypeError, match="pass `figure="):
            dotplot(adata, list(adata.var_names[:3]), "grp",
                    rect=(0, 0, 1, 1), show=False)

    def test_every_side_works(self):
        adata = _adata()
        for side in ("right", "left", "top", "bottom"):
            assert dotplot(adata, list(adata.var_names[:4]), "grp",
                           legend_side=side, show=False) is not None

    def test_legends_can_be_switched_off(self):
        adata = _adata()
        assert dotplot(adata, list(adata.var_names[:4]), "grp",
                       legend=False, show=False) is not None

    def test_an_unknown_side_is_refused(self):
        adata = _adata()
        with pytest.raises(ValueError, match="must be 'right', 'left'"):
            dotplot(adata, list(adata.var_names[:3]), "grp",
                    legend_side="sideways", show=False)
