"""`ov.pl.boxplot(show_points=...)` regression tests.

boxplot always overlaid jittered scatter points, with no way to turn them off,
which clutters a hue split or a small multi-panel figure. `show_points` gates
that overlay; it defaults to True so every existing call draws exactly as
before.

Note on the "given ax" premise: this boxplot builds its own figure with
`plt.subplots` and returns `(fig, ax)` — it has no `ax` parameter. The decoy
here is therefore a pre-existing axes that boxplot must NOT draw onto: it opens
its own figure regardless.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

from omicverse.pl._bulk import boxplot


def _frame():
    rng = np.random.default_rng(0)
    rows = []
    for gene in ["G1", "G2"]:
        for cancer in ["BRCA", "LUAD"]:
            for _ in range(12):
                rows.append(dict(gene=gene, cancer=cancer,
                                 value=float(rng.normal(1.0, 0.3))))
    return pd.DataFrame(rows)


def _n_scatter(ax):
    return sum(isinstance(c, PathCollection) for c in ax.collections)


def test_show_points_false_draws_no_scatter():
    frame = _frame()
    fig, ax = boxplot(frame, hue="gene", x_value="cancer", y_value="value",
                      show_points=False)
    assert _n_scatter(ax) == 0
    plt.close(fig)


def test_show_points_true_is_the_default_and_draws_scatter():
    frame = _frame()
    # default (no show_points) and explicit True must both draw the points
    fig_default, ax_default = boxplot(frame, hue="gene", x_value="cancer",
                                      y_value="value")
    fig_true, ax_true = boxplot(frame, hue="gene", x_value="cancer",
                                y_value="value", show_points=True)
    assert _n_scatter(ax_default) > 0
    assert _n_scatter(ax_true) == _n_scatter(ax_default)
    plt.close(fig_default)
    plt.close(fig_true)


def test_boxplot_opens_its_own_figure_not_the_decoy():
    frame = _frame()
    # a decoy axes that boxplot must leave untouched — it makes its own figure
    decoy_fig, decoy_ax = plt.subplots()

    for show in (True, False):
        fig, ax = boxplot(frame, hue="gene", x_value="cancer",
                          y_value="value", show_points=show)
        assert fig is not decoy_fig
        assert ax is not decoy_ax
        # nothing was drawn onto the decoy
        assert _n_scatter(decoy_ax) == 0
        assert decoy_ax.get_lines() == []
        plt.close(fig)

    plt.close(decoy_fig)
