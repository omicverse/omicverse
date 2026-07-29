"""Tests for ``ov.pl.sankey`` — the general flow / alluvial diagram.

These check the contract and the geometry, not a rendered image. The load-
bearing property is mass conservation: the ribbons leaving a node must sum back
to the node's height, and that is asserted from the drawn artists' own extents
(the node ``Rectangle``s and ribbon ``Polygon``s carry gids), so the test would
catch a layout that merely *looks* like a Sankey.

The axes contract mirrors the rest of ``ov.pl``: given an ``ax`` the function
draws into it and opens no new figure; without one it makes exactly one; and it
never leaks onto ``plt.gca()``.
"""
from __future__ import annotations

from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from omicverse.pl._categorical import sankey  # noqa: E402


@pytest.fixture
def flows():
    """A small table with three categorical columns of different arity."""
    rng = np.random.default_rng(20260729)
    n = 240
    return pd.DataFrame({
        "day": rng.choice(["Thur", "Fri", "Sat", "Sun"], n),
        "sex": rng.choice(["Female", "Male"], n),
        "smoker": rng.choice(["Yes", "No"], n, p=[0.4, 0.6]),
    })


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _leftedge_width(polygon) -> float:
    """Height of a ribbon polygon at its left edge, read off the artist."""
    xy = polygon.get_xy()
    x_min = xy[:, 0].min()
    ys = xy[np.isclose(xy[:, 0], x_min), 1]
    return float(ys.max() - ys.min())


# --------------------------------------------------------------------------
# axes contract
# --------------------------------------------------------------------------


def test_draws_into_given_axes_without_a_new_figure(flows):
    fig, ax = plt.subplots()
    before = set(plt.get_fignums())
    returned = sankey(flows, ["day", "sex"], ax=ax)
    assert returned is ax
    assert set(plt.get_fignums()) == before  # no figure was created
    assert ax.patches  # something was actually drawn


def test_creates_a_figure_when_no_axes_is_given(flows):
    before = set(plt.get_fignums())
    ax = sankey(flows, ["day", "sex"])
    new = set(plt.get_fignums()) - before
    assert len(new) == 1
    assert ax.figure.number in new


def test_does_not_leak_onto_the_current_axes(flows):
    fig, ax = plt.subplots()
    sankey(flows, ["day", "sex"], ax=ax)
    # A decoy figure opened afterwards must come up empty — nothing should have
    # been drawn through plt.gca().
    decoy = plt.figure()
    decoy_ax = decoy.gca()
    assert not decoy_ax.patches
    assert not decoy_ax.lines


# --------------------------------------------------------------------------
# it draws a flow for two and for three stages
# --------------------------------------------------------------------------


@pytest.mark.parametrize("columns", [["day", "sex"], ["day", "sex", "smoker"]])
def test_two_and_three_stage_both_draw(flows, columns):
    ax = sankey(flows, columns)
    nodes = [p for p in ax.patches if (p.get_gid() or "").startswith("node:")]
    ribbons = [p for p in ax.patches if (p.get_gid() or "").startswith("flow:")]
    stages = {int((p.get_gid()).split(":")[1]) for p in nodes}
    assert stages == set(range(len(columns)))  # every column became a stage
    assert ribbons  # and consecutive stages are joined


# --------------------------------------------------------------------------
# mass conservation — the defining property
# --------------------------------------------------------------------------


@pytest.mark.parametrize("columns", [["day", "sex"], ["day", "sex", "smoker"]])
def test_ribbons_conserve_mass(flows, columns):
    ax = sankey(flows, columns)
    node_height = {}
    outgoing = defaultdict(float)
    for patch in ax.patches:
        gid = patch.get_gid() or ""
        if gid.startswith("node:"):
            _, stage, level = gid.split(":", 2)
            node_height[(int(stage), level)] = patch.get_height()
        elif gid.startswith("flow:"):
            _, stage, left, right = gid.split(":", 3)
            outgoing[(int(stage), left)] += _leftedge_width(patch)

    # Every node that has any outgoing ribbon (all but the last stage) must see
    # its full height leave it.
    checked = 0
    for (stage, level), total in outgoing.items():
        assert (stage, level) in node_height
        assert total == pytest.approx(node_height[(stage, level)], abs=1e-9)
        checked += 1
    assert checked  # the assertion above actually ran


def test_mass_conservation_test_catches_a_broken_layout(flows):
    """The mass-conservation check must go red on a layout that violates it.

    This is the guard on the guard: mutate one drawn ribbon so its left edge no
    longer matches its share of the node, and confirm the same comparison the
    real test uses now fails. Without this, a test that always passes would be
    indistinguishable from a correct one.
    """
    ax = sankey(flows, ["day", "sex"])
    node_height = {}
    ribbons = {}
    for patch in ax.patches:
        gid = patch.get_gid() or ""
        if gid.startswith("node:0:"):
            node_height[gid.split(":", 2)[2]] = patch.get_height()
        elif gid.startswith("flow:0:"):
            ribbons[gid] = patch

    # Break one ribbon: push only its left-edge *top* vertex up, so the left
    # edge is genuinely wider than the flow it should carry.
    victim = next(iter(ribbons.values()))
    xy = victim.get_xy().copy()
    left_idx = np.flatnonzero(np.isclose(xy[:, 0], xy[:, 0].min()))
    top_vertex = left_idx[np.argmax(xy[left_idx, 1])]
    xy[top_vertex, 1] += 0.1
    victim.set_xy(xy)

    left_label = next(iter(ribbons)).split(":", 3)[2]
    outgoing = sum(_leftedge_width(p) for gid, p in ribbons.items()
                   if gid.split(":", 3)[2] == left_label)
    assert outgoing != pytest.approx(node_height[left_label], abs=1e-9)


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


def test_unknown_column_names_the_available_columns(flows):
    with pytest.raises(KeyError) as excinfo:
        sankey(flows, ["day", "gender"])
    message = str(excinfo.value)
    assert "gender" in message
    # the error must list what was actually available
    assert "day" in message and "sex" in message and "smoker" in message


def test_needs_at_least_two_columns(flows):
    with pytest.raises(TypeError):
        sankey(flows, ["day"])
