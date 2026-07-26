"""Parity tests for ov.flow's display transforms.

These are the module's stated EXIT CRITERION. logicle and hyperlog are derived
from the GatingML 2.0 / Parks 2006 specification rather than wrapped, so the
only thing that makes the derivation trustworthy is agreeing with the reference
implementation to float precision.

Two layers, deliberately:

* GOLDEN values below, captured from ``flowutils``' C extension. These run
  everywhere, including CI, where flowutils is not installed. Without them the
  whole parity suite would skip and the exit criterion would be decorative — a
  mistake already made once in this repo, where the Sanger tests silently
  skipped in CI because their fixture path did not exist.
* A live sweep against ``flowutils`` when it IS installed, over a wider
  parameter grid than the goldens cover.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from omicverse.flow._transforms import (
    Asinh, Hyperlog, Linear, Log, Logicle,
    make_transform, transform_from_dict,
)

HAS_FLOWUTILS = importlib.util.find_spec("flowutils") is not None

DATA = np.array([-10000.0, -1000.0, -100.0, -10.0, 0.0, 10.0, 100.0,
                 1000.0, 10000.0, 100000.0, 262144.0])

GOLDEN = {
    ("logicle", 262144, 0.5, 4.5, 0.0): [
        -0.4616104350043351, -0.23211535395017646, 0.009041134692025388, 0.099917946544774, 0.1111111111111111, 0.12230427567744821, 0.21318108753019682, 0.45433757617239867, 0.6838326572265573, 0.9069275914814589, 1.0,
    ],
    ("hyperlog", 262144, 0.5, 4.5, 0.0): [
        -0.4588606761913223, -0.21683562112984384, 0.026258145648262704, 0.10139614363057406, 0.1111111111111111, 0.12082607859164815, 0.1959640765739595, 0.43905784335206605, 0.6810828984135445, 0.906677584496927, 1.0,
    ],
    ("logicle", 262144, 0.5, 4.5, 1.0): [
        -0.19586308318536516, -0.008094380504689802, 0.1892154738389299, 0.26356922899117874, 0.2727272727272727, 0.2818853164633667, 0.3562390716156155, 0.5535489259592352, 0.7413176286399106, 0.923849847575739, 1.0,
    ],
    ("hyperlog", 262144, 0.5, 4.5, 1.0): [
        -0.1936132805201728, 0.0044072190755823915, 0.2033021191667605, 0.2647786629704697, 0.2727272727272727, 0.2806758824840757, 0.3421524262877849, 0.541047326378963, 0.7390678259747182, 0.9236452964065766, 1.0,
    ],
    ("logicle", 10000, 0.2, 4.0, 0.0): [
        -0.9, -0.6498237007691234, -0.3982231124629184, -0.14086137278881936, 0.05, 0.24086137278881936, 0.4982231124629184, 0.7498237007691234, 1.0, 1.2500179763568513, 1.3546541971087136,
    ],
    ("hyperlog", 10000, 0.2, 4.0, 0.0): [
        -0.9, -0.646826276176676, -0.37660357160464075, -0.07849405261605205, 0.05, 0.17849405261605206, 0.47660357160464073, 0.746826276176676, 1.0, 1.25043633411735, 1.3551084164891591,
    ],
    ("logicle", 1000000.0, 2.0, 5.0, 2.0): [
        0.5216264778939116, 0.5663949600158064, 0.5709251642305386, 0.5713782306634764, 0.5714285714285714, 0.5714789121936664, 0.5719319786266042, 0.5764621828413364, 0.6212306649632312, 0.8300452875243889, 0.9079212052822181,
    ],
    ("hyperlog", 1000000.0, 2.0, 5.0, 2.0): [
        0.5404502475984846, 0.5682937857868626, 0.5711147732608317, 0.5713971884617608, 0.5714285714285714, 0.571459954395382, 0.5717423695963111, 0.5745633570702802, 0.6024068952586582, 0.795973708579683, 0.8964547167731168,
    ],
}


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_matches_the_reference_implementation(key):
    """Bit-parity with flowutils' C extension, pinned as data.

    A tolerance of 1e-12 is ~4500x looser than the measured agreement
    (4.4e-16); it is here to absorb platform libm differences, not to hide a
    real divergence.
    """
    kind, t, w, m, a = key
    tr = make_transform(kind, t=t, w=w, m=m, a=a)
    assert np.allclose(tr.apply(DATA), np.asarray(GOLDEN[key]), rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", ["logicle", "hyperlog"])
def test_the_linearisation_point_maps_to_data_zero(kind):
    """B(x1) == 0 is the DEFINING constraint — x1 is the point the reflection is
    odd about. An earlier cut had `f = -mf_a*a` instead of `+`, which left
    B(x1) = -221.7: the top of scale still landed exactly on T, so it looked
    right end to end while every value under ~1e4 was wrong.
    """
    tr = make_transform(kind, t=262144, w=0.5, m=4.5, a=0.0)
    assert abs(float(tr.inverse(np.array([tr._p["x1"]]))[0])) < 1e-9
    # ...and therefore data 0 maps to x1.
    assert float(tr.apply(np.array([0.0]))[0]) == pytest.approx(tr._p["x1"], abs=1e-12)


@pytest.mark.parametrize("kind", ["logicle", "hyperlog"])
def test_the_negative_region_is_reflected_not_extrapolated(kind):
    """Below x1 both transforms are the ODD REFLECTION about x1.

    Hyperlog's spec formula (a*e^by + c*y - f) has no reflection in it, and
    omitting it is invisible above x1: the parameters come out identical and
    everything at or above the linearisation point agrees to 1e-16. Only the
    negative region diverges — by ~38 data units at scale 0 — which is the one
    region a biexponential exists for.
    """
    tr = make_transform(kind, t=262144, w=0.5, m=4.5, a=0.0)
    x1 = tr._p["x1"]
    below = np.array([0.0, 0.02, 0.05, 0.08, 0.10])
    assert np.allclose(tr.inverse(below), -tr.inverse(2 * x1 - below), rtol=0, atol=1e-9)


@pytest.mark.parametrize("kind", ["logicle", "hyperlog"])
def test_scale_is_not_clamped_to_the_unit_interval(kind):
    """[0, 1] is the DISPLAY range, not the domain.

    An event dimmer than B(0) has a negative scale value and that is meaningful
    — it is off the bottom of the axis. Clamping collapsed everything under
    about -139 (on a 262144/W=0.5 scale) onto exactly 0, i.e. precisely the
    deep-negative events a badly compensated channel produces.
    """
    tr = make_transform(kind, t=262144, w=0.5, m=4.5, a=0.0)
    deep = tr.apply(np.array([-1e4, -1e5]))
    assert (deep < 0).all()
    assert deep[1] < deep[0], "further below the axis must give a smaller scale"
    assert float(tr.apply(np.array([1e7]))[0]) > 1.0


@pytest.mark.parametrize("kind", ["logicle", "hyperlog", "asinh", "log", "linear"])
def test_round_trips(kind):
    params = {"logicle": dict(t=262144, w=0.5, m=4.5, a=0.0),
              "hyperlog": dict(t=262144, w=0.5, m=4.5, a=0.0),
              "asinh": dict(t=262144, m=4.0, a=1.0),
              "log": dict(t=262144, m=4.5),
              "linear": dict(t=262144, a=0.0)}[kind]
    tr = make_transform(kind, **params)
    x = DATA[DATA > 0] if kind == "log" else DATA
    assert np.allclose(tr.inverse(tr.apply(x)), x, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("kind", ["logicle", "hyperlog", "asinh", "log", "linear"])
def test_monotonic(kind):
    tr = make_transform(kind, t=262144)
    x = np.linspace(1.0, 262144.0, 500)
    y = tr.apply(x)
    assert np.all(np.diff(y) > 0)


def test_log_is_nan_below_zero_rather_than_silently_clamped():
    """A log scale genuinely cannot show a negative event. Returning NaN makes a
    mis-chosen scale visible; clamping to 0 would pile every negative event onto
    the axis origin and look like a real population."""
    y = Log(t=262144, m=4.5).apply(np.array([-1.0, 0.0, 1.0]))
    assert np.isnan(y[0]) and np.isnan(y[1]) and np.isfinite(y[2])


def test_serialisation_round_trip():
    """A gate is meaningless without the scaling its vertices were drawn on, so
    a strategy is only saveable if its transforms are."""
    for tr in [Logicle(t=1e5, w=0.7, m=4.2, a=0.3), Hyperlog(t=4096, w=0.5, m=4.0, a=1.0),
               Asinh(t=1e4, m=3.0, a=0.5), Log(t=1e4, m=3.0), Linear(t=1e4, a=-100.0)]:
        back = transform_from_dict(tr.to_dict())
        assert type(back) is type(tr)
        assert np.allclose(back.apply(DATA), tr.apply(DATA), equal_nan=True)


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError, match="linear region"):
        Logicle(t=262144, w=3.0, m=4.5, a=0.0)      # 2W > M+A
    with pytest.raises(ValueError, match="T must be positive"):
        Logicle(t=0)
    with pytest.raises(ValueError, match="unknown transform"):
        make_transform("bogus")


def test_ticks_land_on_decades_and_include_negatives():
    tr = Logicle(t=262144, w=0.5, m=4.5, a=0.0)
    ticks = tr.ticks()
    assert len(ticks["major"]) > 3
    assert np.all(np.diff(ticks["major"]) > 0)
    assert any(lbl.startswith("-") for lbl in ticks["labels"]), \
        "a biexponential axis must label negative decades — that is its whole point"


# ── live sweep, wider than the goldens ──────────────────────────────────────

@pytest.mark.skipif(not HAS_FLOWUTILS, reason="flowutils not installed (goldens still ran)")
@pytest.mark.parametrize("t,w,m,a", [
    (262144, 0.5, 4.5, 0.0), (262144, 0.5, 4.5, 1.0), (262144, 1.0, 4.5, 0.5),
    (10000, 0.2, 4.0, 0.0), (1e6, 2.0, 5.0, 2.0), (262144, 0.1, 4.5, 0.0),
    (4096, 0.5, 4.0, 1.5), (262144, 0.5, 4.5, 2.5),
])
@pytest.mark.parametrize("kind", ["logicle", "hyperlog"])
def test_live_parity_sweep(kind, t, w, m, a):
    import flowutils.transforms as FU

    ref_fwd = {"logicle": FU.logicle, "hyperlog": FU.hyperlog}[kind]
    ref_inv = {"logicle": FU.logicle_inverse, "hyperlog": FU.hyperlog_inverse}[kind]
    data = np.concatenate([DATA, np.linspace(-5e3, 2.6e5, 300)])
    tr = make_transform(kind, t=t, w=w, m=m, a=a)

    got = tr.apply(data)
    exp = ref_fwd(data.reshape(-1, 1).copy(), [0], t=t, m=m, w=w, a=a)[:, 0]
    assert np.nanmax(np.abs(got - exp)) < 1e-12

    y = np.linspace(0, 1, 300)
    exp_i = ref_inv(y.reshape(-1, 1).copy(), [0], t=t, m=m, w=w, a=a)[:, 0]
    assert np.nanmax(np.abs(tr.inverse(y) - exp_i)) / max(1.0, t) < 1e-12
