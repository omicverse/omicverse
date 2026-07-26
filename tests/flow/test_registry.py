"""Every ov.flow name an agent might ask for resolves to the right object.

The registry is how omicOS agents and `ov.find_function` reach the library by
natural-language query. Functions were registered from the start; the CLASSES
were not, and that was worse than it sounds — `find_function("GatingStrategy")`
did not come back empty, it came back with `ov.flow.hierarchy`, a plotting
function whose *description* happens to contain the words "gating strategy".
An agent asking for the strategy object got a chart.

`@register_function` works on classes exactly as it does on functions: the
decorator detects `inspect.isclass` and returns the class untouched rather than
wrapping it, so identity, `isinstance` and pickling all survive. This file pins
both halves — that the classes are registered, and that decorating them did not
turn them into something else.
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest

import omicverse as ov


#: query -> the fully-qualified name it must resolve to.
EXPECTED = {
    "GatingStrategy": "omicverse.flow._strategy.GatingStrategy",
    "门控策略": "omicverse.flow._strategy.GatingStrategy",
    "GatingResult": "omicverse.flow._strategy.GatingResult",
    "PolygonGate": "omicverse.flow._gates.PolygonGate",
    "多边形门": "omicverse.flow._gates.PolygonGate",
    "RectangleGate": "omicverse.flow._gates.RectangleGate",
    "QuadrantGate": "omicverse.flow._gates.QuadrantGate",
    "象限门": "omicverse.flow._gates.QuadrantGate",
    "BooleanGate": "omicverse.flow._gates.BooleanGate",
    "Logicle": "omicverse.flow._transforms.Logicle",
    "Hyperlog": "omicverse.flow._transforms.Hyperlog",
}

CLASSES = [
    "GatingStrategy", "GatingResult", "RectangleGate", "PolygonGate",
    "EllipsoidGate", "QuadrantGate", "BooleanGate",
    "Logicle", "Hyperlog", "Asinh", "Log", "Linear",
]


@pytest.fixture(scope="module", autouse=True)
def _import_flow():
    """ov.flow is lazily imported, and registration is an import side effect.

    Without this the whole file would pass or fail depending on what some other
    test happened to import first.
    """
    assert ov.flow.GatingStrategy is not None


@pytest.mark.parametrize("name", CLASSES)
def test_the_class_is_registered(name):
    cls = getattr(ov.flow, name)
    assert inspect.isclass(cls), f"ov.flow.{name} is not a class"
    assert hasattr(cls, "_registry_info"), (
        f"ov.flow.{name} carries no @register_function — an agent cannot reach it"
    )
    info = cls._registry_info
    assert info["aliases"], f"{name} registered with no aliases"
    assert info["description"], f"{name} registered with no description"


@pytest.mark.parametrize("name", CLASSES)
def test_registering_did_not_replace_the_class(name):
    """The decorator wraps functions. If it ever wrapped a class instead, the
    class would become a function: isinstance would fail, dataclass fields would
    vanish, and instances would stop pickling."""
    cls = getattr(ov.flow, name)
    assert inspect.isclass(cls)
    assert cls.__module__.startswith("omicverse.flow")


def test_the_gate_dataclasses_still_behave_like_dataclasses():
    gate = ov.flow.PolygonGate(name="p", dims=("a", "b"),
                               vertices=[[0, 0], [1, 0], [1, 1]])
    assert dataclasses.is_dataclass(gate)
    assert isinstance(gate, ov.flow.Gate)
    assert gate.to_dict()["kind"] == "polygon"


def test_a_registered_gate_still_pickles():
    """The comment in _registry.py gives this as the reason classes are not
    wrapped, so it is worth actually checking rather than trusting."""
    import pickle

    lg = ov.flow.Logicle(t=262144.0, m=4.5, w=1.0, a=0.0)
    assert pickle.loads(pickle.dumps(lg)).to_dict() == lg.to_dict()


@pytest.mark.parametrize("query,expected", sorted(EXPECTED.items()))
def test_the_query_resolves_to_the_class_not_to_something_that_mentions_it(
    query, expected, capsys
):
    """The regression this file exists for.

    Before the classes were registered, "GatingStrategy" returned
    ov.flow.hierarchy — a real function, plausibly ranked, and completely wrong.
    A miss is recoverable; a confident wrong answer is not.
    """
    ov.find_function(query)
    printed = capsys.readouterr().out
    first = [l for l in printed.splitlines() if l.strip().startswith("1. ")]
    assert first, f"{query!r} found nothing"
    assert expected in first[0], (
        f"{query!r} resolved to {first[0].strip()} instead of {expected}"
    )
