"""Gate geometry and the gating strategy tree."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omicverse.flow import (
    BooleanGate, EllipsoidGate, GatingStrategy, PolygonGate, QuadrantGate,
    RectangleGate, gate_from_dict, make_transform, points_in_polygon,
)

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


def frame(**cols):
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


# ── point in polygon ────────────────────────────────────────────────────────

def test_points_in_polygon_basics():
    pts = np.array([[5.0, 5.0], [-1.0, 5.0], [11.0, 5.0], [5.0, -1.0]])
    assert points_in_polygon(SQUARE, pts).tolist() == [True, False, False, False]


def test_boundary_points_are_inside():
    """A gate that visually contains a point must not report that it does not.
    Vertex-exact events are rare in real data and common in gates drawn by
    snapping."""
    assert points_in_polygon(SQUARE, np.array([[0.0, 0.0], [10.0, 5.0]])).all()


def test_concave_polygon_is_handled():
    """People draw crescents around a smear; a convex-hull shortcut would
    silently include the notch."""
    c = np.array([[0, 0], [10, 0], [10, 10], [5, 4], [0, 10]], dtype=float)
    assert points_in_polygon(c, np.array([[5.0, 1.0]]))[0]      # in the body
    assert not points_in_polygon(c, np.array([[5.0, 8.0]]))[0]  # in the notch


def test_degenerate_polygons_are_refused():
    with pytest.raises(ValueError, match="at least 3"):
        points_in_polygon(np.array([[0.0, 0.0], [1.0, 1.0]]), np.zeros((1, 2)))


# ── gates ───────────────────────────────────────────────────────────────────

def test_rectangle_bounds_are_half_open():
    """[min, max) per Gating-ML: adjacent rectangles must partition, not
    double-count the shared edge."""
    g = RectangleGate(name="r", dims=("x",), bounds=((0.0, 10.0),))
    assert g.mask(frame(x=[-0.001, 0.0, 5.0, 9.999, 10.0])).tolist() == \
        [False, True, True, True, False]


def test_rectangle_one_sided_threshold():
    """`None` is unbounded — how "CD3 positive" is expressed without inventing
    a ceiling."""
    g = RectangleGate(name="pos", dims=("x",), bounds=((100.0, None),))
    assert g.mask(frame(x=[99.0, 100.0, 1e9])).tolist() == [False, True, True]


def test_rectangle_rejects_inverted_bounds():
    with pytest.raises(ValueError, match=">="):
        RectangleGate(name="bad", dims=("x",), bounds=((10.0, 0.0),))


def test_a_gate_carries_the_transform_its_vertices_were_drawn_on():
    """THE correctness property. The same vertices read as linear values
    describe a completely different population, so a gate without its transform
    is not reapplicable — it is wrong."""
    lg = make_transform("logicle", t=262144, w=0.5, m=4.5, a=0.0)
    scaled = RectangleGate(name="s", dims=("CD3",), bounds=((0.4, None),),
                           transforms={"CD3": lg})
    linear = RectangleGate(name="l", dims=("CD3",), bounds=((0.4, None),))

    # On a logicle scale, 0.4 is about 460 data units, so 1e3 is positive and
    # -100 is not. Read as raw linear values the same bound of 0.4 admits
    # essentially everything above zero — a completely different population.
    data = frame(CD3=[-100.0, 0.0, 1.0, 1e3, 1e5])
    assert scaled.mask(data).tolist() == [False, False, False, True, True]
    assert linear.mask(data).tolist() == [False, False, True, True, True]
    assert not np.array_equal(scaled.mask(data), linear.mask(data)), \
        "if these agreed, the test would prove nothing"


def test_missing_dimension_names_the_gate_and_the_dimension():
    g = RectangleGate(name="r", dims=("CD8",), bounds=((0.0, 1.0),))
    with pytest.raises(KeyError, match="CD8"):
        g.mask(frame(CD3=[1.0]))


def test_ellipsoid_uses_mahalanobis_distance():
    g = EllipsoidGate(name="e", dims=("x", "y"), mean=np.array([0.0, 0.0]),
                      covariance=np.eye(2), distance_square=4.0)
    # unit covariance, d2=4 -> radius 2
    assert g.mask(frame(x=[0.0, 1.9, 2.1], y=[0.0, 0.0, 0.0])).tolist() == [True, True, False]


def test_quadrant_gate_partitions_every_event_exactly_once():
    q = QuadrantGate(name="Q", dims=("x", "y"), dividers=(0.0, 0.0))
    masks = q.masks(frame(x=[1, -1, 1, -1], y=[1, 1, -1, -1]))
    stack = np.column_stack(list(masks.values()))
    assert stack.sum(axis=1).tolist() == [1, 1, 1, 1], "quadrants must not overlap or leak"
    assert len(masks) == 4


def test_boolean_gate_needs_its_operands_computed_first():
    b = BooleanGate(name="b", dims=(), operator="and", operands=("A", "B"))
    with pytest.raises(KeyError, match="have not been computed"):
        b.combine({"A": np.array([True])})


@pytest.mark.parametrize("op,expected", [
    ("and", [False, False, False, True]),
    ("or", [False, True, True, True]),
])
def test_boolean_operators(op, expected):
    b = BooleanGate(name="b", dims=(), operator=op, operands=("A", "B"))
    resolved = {"A": np.array([0, 1, 0, 1], bool), "B": np.array([0, 0, 1, 1], bool)}
    assert b.combine(resolved).tolist() == expected


def test_all_gate_kinds_round_trip_through_json():
    lg = make_transform("logicle", t=262144, w=0.5, m=4.5, a=0.0)
    gates = [
        RectangleGate(name="r", dims=("x",), bounds=((0.0, 1.0),), transforms={"x": lg}),
        PolygonGate(name="p", dims=("x", "y"), vertices=SQUARE),
        EllipsoidGate(name="e", dims=("x", "y"), mean=np.zeros(2),
                      covariance=np.eye(2), distance_square=2.0),
        QuadrantGate(name="q", dims=("x", "y"), dividers=(0.0, 0.0)),
        BooleanGate(name="b", dims=(), operator="not", operands=("r",)),
    ]
    data = frame(x=[0.5, 2.0], y=[0.5, 2.0])
    for g in gates:
        back = gate_from_dict(g.to_dict())
        assert type(back) is type(g) and back.name == g.name
        if isinstance(g, (BooleanGate,)):
            continue
        if isinstance(g, QuadrantGate):
            assert list(back.masks(data)) == list(g.masks(data))
        else:
            assert np.array_equal(back.mask(data), g.mask(data))


# ── strategy ────────────────────────────────────────────────────────────────

def make_adata(n=5000, seed=0):
    import anndata as ad
    rng = np.random.default_rng(seed)
    fsc = np.abs(rng.normal(6e4, 1.5e4, n))
    cd3 = np.where(rng.random(n) < 0.4, rng.normal(2e4, 5e3, n), rng.normal(-50, 200, n))
    X = np.column_stack([fsc, cd3]).astype(np.float32)
    return ad.AnnData(X, var=pd.DataFrame(index=["FSC-A", "CD3"]))


def test_a_child_is_always_a_subset_of_its_parent():
    """The invariant parent-restriction exists to guarantee. If a child ever
    contains an event its parent does not, every downstream frequency is a lie.
    """
    a = make_adata()
    gs = (GatingStrategy()
          .add_gate(RectangleGate(name="Cells", dims=("FSC-A",), bounds=((3e4, 1e5),)))
          .add_gate(RectangleGate(name="CD3+", dims=("CD3",), bounds=((5e3, None),)),
                    parent="Cells"))
    res = gs.apply(a)
    assert (res.masks["CD3+"] & ~res.masks["Cells"]).sum() == 0


def test_parent_restriction_matches_the_naive_and():
    """Restricting to the parent's members must give exactly what gating
    everything and ANDing would — it is an evaluation strategy, not a different
    answer."""
    a = make_adata()
    cells = RectangleGate(name="Cells", dims=("FSC-A",), bounds=((3e4, 1e5),))
    cd3 = RectangleGate(name="CD3+", dims=("CD3",), bounds=((5e3, None),))
    gs = GatingStrategy().add_gate(cells).add_gate(cd3, parent="Cells")
    res = gs.apply(a)

    f = {"FSC-A": np.asarray(a.X)[:, 0], "CD3": np.asarray(a.X)[:, 1]}
    naive = cells.mask(f) & cd3.mask(f)
    assert np.array_equal(res.masks["CD3+"], naive)


def test_unknown_parent_is_refused_when_the_gate_is_added():
    gs = GatingStrategy()
    with pytest.raises(KeyError, match="Nonexistent"):
        gs.add_gate(RectangleGate(name="x", dims=("a",), bounds=((0.0, 1.0),)),
                    parent="Nonexistent")


def test_duplicate_gate_names_are_refused():
    gs = GatingStrategy().add_gate(RectangleGate(name="x", dims=("a",), bounds=((0.0, 1.0),)))
    with pytest.raises(ValueError, match="already in this strategy"):
        gs.add_gate(RectangleGate(name="x", dims=("b",), bounds=((0.0, 1.0),)))


def test_stats_report_the_denominator_they_used():
    a = make_adata()
    gs = (GatingStrategy()
          .add_gate(RectangleGate(name="Cells", dims=("FSC-A",), bounds=((3e4, 1e5),)))
          .add_gate(RectangleGate(name="CD3+", dims=("CD3",), bounds=((5e3, None),)),
                    parent="Cells"))
    df = gs.apply(a).stats()
    row = df[df["population"] == "CD3+"].iloc[0]
    assert row["parent"] == "Cells"
    assert row["parent_count"] == int(gs.apply(a).masks["Cells"].sum())
    assert 0 < row["freq_parent"] <= 1
    assert row["freq_parent"] >= row["freq_total"]


def test_small_populations_are_flagged():
    """A frequency off 30 events has a 95% CI of about +/-18 points. Reporting
    it to one decimal, as every cytometry tool does, is false precision."""
    a = make_adata()
    gs = (GatingStrategy()
          .add_gate(RectangleGate(name="Rare", dims=("CD3",), bounds=((1e9, None),))))
    df = gs.apply(a).stats(min_events=100)
    assert bool(df[df["population"] == "Rare"].iloc[0]["low_n"])


def test_gates_can_be_drawn_on_detector_names_too():
    """A strategy written against 'FITC-A' must still apply to a file indexed by
    the marker 'CD3'."""
    import anndata as ad
    rng = np.random.default_rng(0)
    X = rng.normal(1e4, 1e3, (500, 1)).astype(np.float32)
    var = pd.DataFrame({"channel": ["FITC-A"], "marker": ["CD3"]}, index=["CD3"])
    a = ad.AnnData(X, var=var)
    gs = GatingStrategy().add_gate(
        RectangleGate(name="pos", dims=("FITC-A",), bounds=((0.0, None),)))
    assert gs.apply(a).masks["pos"].all()


def test_strategy_round_trips_through_json():
    a = make_adata()
    lg = make_transform("logicle", t=262144, w=0.5, m=4.5, a=0.0)
    gs = (GatingStrategy("panel")
          .add_gate(RectangleGate(name="Cells", dims=("FSC-A",), bounds=((3e4, 1e5),)))
          .add_gate(RectangleGate(name="CD3+", dims=("CD3",), bounds=((0.4, None),),
                                  transforms={"CD3": lg}), parent="Cells"))
    res = gs.apply(a.copy())
    back = GatingStrategy.from_dict(gs.to_dict())
    res2 = back.apply(a.copy())
    assert back.name == "panel"
    assert all(np.array_equal(res.masks[k], res2.masks[k]) for k in res.masks)


def test_one_strategy_applies_to_many_samples():
    """The reason the tree belongs to the strategy and not to a sample."""
    gs = GatingStrategy().add_gate(
        RectangleGate(name="CD3+", dims=("CD3",), bounds=((5e3, None),)))
    fr = [gs.apply(make_adata(seed=s)).stats().iloc[0]["freq_total"] for s in range(4)]
    assert len(set(np.round(fr, 6))) == 4, "different samples, different numbers"
    assert all(0.3 < f < 0.5 for f in fr), "...but all near the 40% that was simulated"


# ── both evaluation branches ────────────────────────────────────────────────
# The traversal restricts to the parent's members only when the parent is
# selective (RESTRICT_BELOW). Both branches must give the same answer, and the
# pass-through branch is the one that can silently drop the parent constraint.

@pytest.mark.parametrize("keep_frac,restricts", [(0.9, False), (0.05, True)])
def test_child_is_a_subset_of_parent_on_both_branches(keep_frac, restricts):
    import anndata as ad
    from omicverse.flow._strategy import RESTRICT_BELOW

    rng = np.random.default_rng(3)
    n = 4000
    x = rng.random(n)
    y = rng.random(n)
    a = ad.AnnData(np.column_stack([x, y]).astype(np.float32),
                   var=pd.DataFrame(index=["x", "y"]))

    gs = (GatingStrategy()
          .add_gate(RectangleGate(name="P", dims=("x",), bounds=((0.0, keep_frac),)))
          # A child that would match plenty of events OUTSIDE the parent.
          .add_gate(RectangleGate(name="C", dims=("y",), bounds=((0.0, 0.5),)), parent="P"))
    res = gs.apply(a, write_obs=False)

    # bool(), not `is`: numpy comparisons give np.bool_, which is never the
    # Python True/False singleton, so `is` is always False here.
    assert bool(res.masks["P"].mean() <= RESTRICT_BELOW) == restricts, \
        "this case must exercise the branch it claims to"
    assert (res.masks["C"] & ~res.masks["P"]).sum() == 0, \
        "a child must never contain an event its parent excluded"
    # ...and it must equal the naive computation.
    f = {"x": x, "y": y}
    naive = gs.gates["P"].mask(f) & gs.gates["C"].mask(f)
    assert np.array_equal(res.masks["C"], naive)


def test_quadrants_also_respect_the_parent_on_the_pass_through_branch():
    import anndata as ad

    rng = np.random.default_rng(4)
    n = 4000
    x, y = rng.normal(0, 1, n), rng.normal(0, 1, n)
    a = ad.AnnData(np.column_stack([x, y]).astype(np.float32),
                   var=pd.DataFrame(index=["x", "y"]))
    gs = (GatingStrategy()
          # keeps ~84% -> pass-through branch
          .add_gate(RectangleGate(name="P", dims=("x",), bounds=((-1.0, None),)))
          .add_gate(QuadrantGate(name="Q", dims=("x", "y"), dividers=(0.0, 0.0)), parent="P"))
    res = gs.apply(a, write_obs=False)
    quads = [k for k in res.masks if k.startswith("Q") and k != "Q"]
    assert len(quads) == 4
    for q in quads:
        assert (res.masks[q] & ~res.masks["P"]).sum() == 0, f"{q} leaked past its parent"
    # The quadrants partition the PARENT, not the whole file.
    total = np.column_stack([res.masks[q] for q in quads]).sum(axis=1)
    assert np.array_equal(total > 0, res.masks["P"])


def test_ov_flow_is_a_registered_namespace():
    """`import omicverse.flow` works for a submodule whether or not it is
    registered, so the whole test suite can pass while `ov.flow` — the way
    every user and every agent reaches it — raises AttributeError."""
    import omicverse as ov

    assert hasattr(ov, "flow")
    assert "flowsom" in ov.flow.__all__
    assert callable(ov.flow.make_transform)


# ── a quadrant's sub-populations ARE populations ────────────────────────────
#
# `add_gate`'s own comment has always said each quadrant "must be addressable
# as a parent", and `apply()` has always handled it correctly. Only the
# validation refused, so gating within CD4+CD8- — the most ordinary thing to do
# after drawing a quadrant — raised KeyError. Meanwhile `to_dict()` emitted
# `"parent": "CD4+CD8-"` for a strategy built any other way, and `from_dict()`
# rejected that exact document: the module could not round-trip its own output.

def _quadrant_sample():
    """Four events, one per quadrant, repeated — so every octant holds 50."""
    import anndata as ad
    X = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]] * 50,
                 dtype=np.float32)
    return ad.AnnData(X, var=pd.DataFrame(index=["x", "y"]))


def _quadrant_strategy():
    return GatingStrategy().add_gate(
        QuadrantGate(name="Q", dims=("x", "y"), dividers=(5.0, 5.0),
                     quadrant_names=("DN", "C4", "C8", "DP")))


def test_a_quadrant_sub_population_can_be_a_parent():
    gs = _quadrant_strategy()
    gs.add_gate(RectangleGate(name="kid", dims=("x",), bounds=((0.0, None),)),
                parent="C4")
    res = gs.apply(_quadrant_sample(), write_obs=False)
    assert res.masks["C4"].sum() == 50
    # Restricted to its parent, not to the whole sample.
    assert res.masks["kid"].sum() == 50
    assert res.masks["kid"].sum() <= res.masks["C4"].sum()


def test_a_strategy_gated_inside_a_quadrant_round_trips_through_its_own_dict():
    gs = _quadrant_strategy()
    gs.add_gate(RectangleGate(name="kid", dims=("x",), bounds=((0.0, None),)),
                parent="C4")
    d = gs.to_dict()
    assert any(g.get("parent") == "C4" for g in d["gates"]), d["gates"]

    back = GatingStrategy.from_dict(d)
    assert back.parent_of("kid") == "C4"
    a = _quadrant_sample()
    r1, r2 = gs.apply(a, write_obs=False), back.apply(a, write_obs=False)
    for name in r1.masks:
        assert np.array_equal(r1.masks[name], r2.masks[name]), name


def test_a_parent_that_does_not_exist_is_still_refused():
    """Widening the check must not turn a typo into silence — and the message
    now lists the octants, which is what the caller was reaching for."""
    gs = _quadrant_strategy()
    with pytest.raises(KeyError) as e:
        gs.add_gate(RectangleGate(name="kid", dims=("x",), bounds=((0.0, None),)),
                    parent="C5")
    assert "C5" in str(e.value)
    assert "C4" in str(e.value), "the available octants should be listed"


def test_a_gate_before_its_parent_raises_instead_of_running_unrestricted():
    """The silent one, and why this is a correctness fix and not an ergonomic
    one.

    `masks.get(parent)` returned None for a parent not yet computed, and None
    means "no restriction" three lines later. Measured before the fix: a child
    of an octant, declared first, returned 200 of 200 where its parent held 50.
    No exception, no warning, and a perfectly plausible number.
    """
    gs = GatingStrategy()
    # Reach past add_gate, which keeps the order right by construction. This is
    # the state from_dict produces from a reordered document.
    gs._gates["kid"] = RectangleGate(name="kid", dims=("x",), bounds=((-99.0, None),))
    gs._parent["kid"] = "C4"
    gs._order.append("kid")
    gs.add_gate(QuadrantGate(name="Q", dims=("x", "y"), dividers=(5.0, 5.0),
                             quadrant_names=("DN", "C4", "C8", "DP")))
    with pytest.raises(KeyError) as e:
        gs.apply(_quadrant_sample(), write_obs=False)
    assert "has not been computed yet" in str(e.value)
