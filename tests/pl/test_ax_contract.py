"""Every single-axes plot in ``ov.pl`` draws into an ``ax=`` it is handed.

A caller assembling a figure with :func:`ov.pl.multipanel` hands each panel to
a plotting function and expects the drawing to land there. That only works if
the whole namespace agrees on the convention the table-first family already
follows: ``ax`` is keyword-only, defaults to ``None``, and when it is supplied
the function draws and nothing else — no new figure, no ``plt.show()``, no
resizing of the figure the panel belongs to.

Three things are checked for each retrofitted plot.

1. **Nothing is created.** ``plt.get_fignums()`` is identical across the call,
   and the parent figure keeps the size it had. A function that quietly opens
   its own canvas loses the drawing; one that honours ``figsize`` while given
   an ``ax`` rescales every other panel on the sheet.
2. **Something is drawn.** The artist count on the passed axes goes up. An
   ``ax`` that is accepted and then ignored passes assertion 1 perfectly while
   producing an empty panel, so the positive check has to be here too.
3. **The old path is untouched.** Called without ``ax`` the function still
   produces its own figure and returns what it always returned.

The final test is a guard rather than a case: it sweeps ``ov.pl.__all__`` for
callables that take ``figsize`` but no ``ax`` and fails on any name it does not
already know to be composite. A new single-axes plot added without ``ax`` shows
up there rather than being discovered by a user mid-figure.
"""
from __future__ import annotations

import inspect
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import omicverse as ov  # noqa: E402


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def _long_frame() -> pd.DataFrame:
    """A small long-form table: three categories, two groups, one measure."""
    rng = np.random.default_rng(20260729)
    n = 120
    return pd.DataFrame({
        "cell_type": rng.choice(["B", "Mono", "T"], n),
        "condition": rng.choice(["ctrl", "drug"], n),
        "score": rng.normal(size=n),
    })


def _fraction_tables():
    """A cell-fraction matrix plus the sample metadata it is grouped by."""
    rng = np.random.default_rng(11)
    index = [f"S{i}" for i in range(12)]
    res = pd.DataFrame(rng.random((12, 3)), index=index,
                       columns=["B", "Mono", "T"])
    obs = pd.DataFrame({"condition": rng.choice(["ctrl", "drug"], 12)},
                       index=index)
    colors = {"B": "#4C72B0", "Mono": "#DD8452", "T": "#55A868"}
    return res, obs, colors


def _small_network():
    """A graph with the two side tables ``plot_network`` needs."""
    import networkx as nx

    graph = nx.relabel_nodes(nx.karate_club_graph(),
                             {i: f"gene{i}" for i in range(34)})
    types = {node: ("hub" if index % 2 else "leaf")
             for index, node in enumerate(graph.nodes)}
    colors = {node: ("#C44E52" if types[node] == "hub" else "#4C72B0")
              for node in graph.nodes}
    return graph, types, colors


# --------------------------------------------------------------------------
# the plots that were given an `ax`
# --------------------------------------------------------------------------
#
# Each entry returns the callable and the keyword arguments to reach a drawing,
# without `ax` — the tests below add it. Keeping the calls in one table means a
# plot retrofitted later is covered by all three checks by adding one line.


def _case_boxplot():
    return ov.pl.boxplot, dict(data=_long_frame(), hue="condition",
                               x_value="cell_type", y_value="score")


def _case_boxplot_xy_aliases():
    # `x`/`y` are the names the sibling table-first plots use; they have to
    # reach the same axes as `x_value`/`y_value`.
    return ov.pl.boxplot, dict(data=_long_frame(), hue="condition",
                               x="cell_type", y="score")


def _case_plot_boxplot():
    return ov.pl.plot_boxplot, dict(data=_long_frame(), hue="condition",
                                    x_value="cell_type", y_value="score")


def _case_plot_grouped_fractions():
    res, obs, colors = _fraction_tables()
    return ov.pl.plot_grouped_fractions, dict(res=res, obs=obs,
                                              group_key="condition",
                                              color_dict=colors)


def _case_plot_network():
    pytest.importorskip("adjustText")
    graph, types, colors = _small_network()
    return ov.pl.plot_network, dict(G=graph, G_type_dict=types,
                                    G_color_dict=colors, plot_node_num=3)


AX_PLOTS = {
    "boxplot": _case_boxplot,
    "boxplot(x=,y=)": _case_boxplot_xy_aliases,
    "plot_boxplot": _case_plot_boxplot,
    "plot_grouped_fractions": _case_plot_grouped_fractions,
    "plot_network": _case_plot_network,
}


def _call(func, kwargs):
    """Run a plot, muffling the deprecation notices the shims emit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return func(**kwargs)


def _returned_axes(result):
    """The axes out of a result that is either an ``Axes`` or ``(fig, ax)``."""
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, matplotlib.axes.Axes):
                return item
        return None
    return result if isinstance(result, matplotlib.axes.Axes) else None


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _target_and_decoy(figsize=(5.0, 3.0)):
    """A panel to draw into, plus a later figure that owns ``plt.gca()``.

    The decoy is the point. A function that reaches for ``plt.scatter`` or
    ``nx.draw_networkx_nodes()`` without naming an axes draws on the *current*
    axes, and when the caller's panel happens to be the current axes — which it
    is in the simplest test — that bug is invisible. Opening a second figure
    afterwards puts the panel out of focus, which is the situation in every
    real multipanel figure: the panels were created before the plotting calls,
    in some other order.
    """
    fig, ax = plt.subplots(figsize=figsize)
    decoy_fig, decoy_ax = plt.subplots()
    assert plt.gca() is decoy_ax
    return fig, ax, decoy_fig, decoy_ax


# --------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(AX_PLOTS))
def test_ax_is_keyword_only_and_optional(name):
    func, _ = AX_PLOTS[name]()
    parameter = inspect.signature(func).parameters.get("ax")
    assert parameter is not None, f"{name} has no `ax` parameter"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"{name} takes `ax` positionally, which would shift existing calls"
    )
    assert parameter.default is None


@pytest.mark.parametrize("name", sorted(AX_PLOTS))
def test_drawing_into_a_given_ax_creates_no_figure(name, monkeypatch):
    func, kwargs = AX_PLOTS[name]()

    def _no_show(*args, **kwargs):
        raise AssertionError(f"{name} called plt.show() with an `ax` given")

    monkeypatch.setattr(plt, "show", _no_show)

    fig, ax, _, _ = _target_and_decoy()
    before = set(plt.get_fignums())
    size_before = tuple(fig.get_size_inches())

    result = _call(func, {**kwargs, "ax": ax})

    assert set(plt.get_fignums()) == before, (
        f"{name} opened a figure although it was given an `ax`"
    )
    assert tuple(fig.get_size_inches()) == size_before, (
        f"{name} resized the caller's figure — `figsize` must be ignored "
        f"when `ax` is given"
    )
    assert _returned_axes(result) is ax, (
        f"{name} did not return the axes it was handed"
    )


@pytest.mark.parametrize("name", sorted(AX_PLOTS))
def test_drawing_into_a_given_ax_actually_draws(name):
    func, kwargs = AX_PLOTS[name]()

    _, ax, _, decoy_ax = _target_and_decoy()
    before = len(ax.get_children())
    decoy_before = len(decoy_ax.get_children())

    _call(func, {**kwargs, "ax": ax})

    assert len(ax.get_children()) > before, (
        f"{name} accepted `ax` but drew nothing on it"
    )
    assert len(decoy_ax.get_children()) == decoy_before, (
        f"{name} drew on the current axes instead of the one it was given"
    )


@pytest.mark.parametrize("name", sorted(AX_PLOTS))
def test_without_ax_the_old_behaviour_is_unchanged(name):
    func, kwargs = AX_PLOTS[name]()

    before = set(plt.get_fignums())
    result = _call(func, kwargs)
    after = set(plt.get_fignums())

    assert after - before, f"{name} produced no figure when called without `ax`"
    axes = _returned_axes(result)
    assert axes is not None, f"{name} stopped returning an Axes"
    assert axes.get_figure().number in after - before
    assert len(axes.get_children()) > 0


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------
#
# Composite plots — a main panel plus dendrograms, colour bars or annotation
# strips, or a grid of panels — cannot be squeezed into one axes, so they are
# expected to lack `ax`. So are the rcParams helpers, which take `figsize` only
# to set a default. Everything else that takes `figsize` but no `ax` is a
# single-axes plot that a caller cannot place in a panel, which is the gap this
# module exists to close.

COMPOSITE_OR_NOT_A_PLOT = frozenset({
    # marsilea / PyComplexHeatmap builders: heatmap plus side plots
    "ccc_heatmap", "cell_cor_heatmap", "cnv_heatmap", "complexheatmap",
    "dynamic_heatmap", "feature_heatmap", "group_heatmap",
    # gridspec assemblies
    "dotplot_doublegroup", "embedding_celltype", "geneset_wordcloud",
    "stacking_vol", "upset",
    # panel grids
    "perturb_celloracle_layout", "perturb_development_layout", "qc",
    # dispatchers over many plot types, and animation
    "animate_streamplot", "ccc_network_plot", "ccc_stat_plot",
    # scanpy dotplot: main panel plus dendrogram and colour bar
    "markers_dotplot",
    # wrappers around the composites above
    "plot_embedding_celltype",
    # single-axes, but wrappers of functions that already take `ax`
    "cpdb_plot_curve_network", "plot_cellproportion", "plot_flowsig_network",
    # rcParams configuration, not plots
    "ov_plot_set", "plot_set", "plotset", "style",
})


def test_no_new_single_axes_plot_is_missing_ax():
    missing = set()
    for name in ov.pl.__all__:
        func = getattr(ov.pl, name, None)
        if not callable(func):
            continue
        try:
            parameters = inspect.signature(func).parameters
        except (TypeError, ValueError):  # C-level callables
            continue
        if "figsize" in parameters and "ax" not in parameters:
            missing.add(name)

    unexpected = missing - COMPOSITE_OR_NOT_A_PLOT
    assert not unexpected, (
        "these ov.pl functions take `figsize` but no `ax`, so they cannot be "
        f"placed in a panel: {sorted(unexpected)}. Give them a keyword-only "
        "`ax=None` and add a case to AX_PLOTS, or list them in "
        "COMPOSITE_OR_NOT_A_PLOT with the reason they need more than one axes."
    )
