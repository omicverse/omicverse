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
    def test_host_figure_size_survives(self):
        adata = _adata()
        fig = plt.figure(figsize=(14.0, 8.0))
        host = fig.add_axes([0.05, 0.72, 0.30, 0.22])
        before = host.get_position().bounds

        dotplot(adata, list(adata.var_names[:5]), "grp", figure=fig,
                rect=(0.40, 0.10, 0.55, 0.55), show=False)

        assert tuple(fig.get_size_inches()) == (14.0, 8.0), \
            "marsilea resized the host figure"
        assert host.get_position().bounds == pytest.approx(before)

    def test_block_is_placed_at_the_region_origin(self):
        adata = _adata()
        fig = plt.figure(figsize=(14.0, 8.0))
        before = set(fig.axes)
        x0, y0 = 0.40, 0.10

        dotplot(adata, list(adata.var_names[:5]), "grp", figure=fig,
                rect=(x0, y0, 0.55, 0.55), show=False)

        added = [a for a in fig.axes if a not in before]
        assert added, "nothing was drawn"
        assert min(a.get_position().x0 for a in added) >= x0 - 1e-6
        assert min(a.get_position().y0 for a in added) >= y0 - 1e-6

    def test_an_undersized_region_warns_rather_than_rescaling(self):
        """Rescaling would shrink the axes without shrinking their text."""
        adata = _adata(n_var=20)
        fig = plt.figure(figsize=(14.0, 8.0))
        with pytest.warns(UserWarning, match="overflow the region"):
            dotplot(adata, list(adata.var_names), "grp", figure=fig,
                    rect=(0.1, 0.1, 0.06, 0.06), show=False)

    def test_rect_without_a_figure_is_refused(self):
        adata = _adata()
        with pytest.raises(TypeError, match="pass `figure="):
            dotplot(adata, list(adata.var_names[:3]), "grp",
                    rect=(0, 0, 1, 1), show=False)
