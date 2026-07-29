"""Panels sized in millimetres come out that size, and the extras that ride along.

``width`` plus ``width_ratios`` fixes the canvas and lets the panels fall where
they may, so the drawn panel depends on how much room the tick labels happened
to need. A journal that specifies the panel — "each panel 45 mm wide" — cannot
be satisfied that way. ``panel_size`` / ``panel_widths`` / ``panel_heights``
invert the derivation: the panels are given, the canvas follows.

These tests measure the drawn axes rather than trusting the arithmetic, because
the arithmetic is only right if the gridspec fraction conversion is right.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

import omicverse as ov

MM_PER_IN = 25.4


def panel_mm(ax, fig) -> tuple[float, float]:
    """The panel as drawn, in millimetres."""
    box = ax.get_position()
    fig_w_in, fig_h_in = fig.get_size_inches()
    return (box.width * fig_w_in * MM_PER_IN, box.height * fig_h_in * MM_PER_IN)


def test_uniform_panel_size_is_what_gets_drawn():
    fig, axes = ov.pl.multipanel((2, 3), panel_size=(45, 100),
                                 margins=14, spacing=8)
    for key in "abcdef":
        w, h = panel_mm(axes[key], fig)
        assert w == pytest.approx(45.0, abs=0.05), f"panel {key} is {w} mm wide"
        assert h == pytest.approx(100.0, abs=0.05), f"panel {key} is {h} mm tall"


def test_the_canvas_is_derived_from_the_panels():
    """3x45 + 2x8 gaps + 2x14 margins = 187 wide; 2x100 + 8 + 28 = 236 tall."""
    fig, _ = ov.pl.multipanel((2, 3), panel_size=(45, 100), margins=14, spacing=8)
    w_mm, h_mm = (v * MM_PER_IN for v in fig.get_size_inches())
    assert w_mm == pytest.approx(3 * 45 + 2 * 8 + 2 * 14, abs=0.05)
    assert h_mm == pytest.approx(2 * 100 + 1 * 8 + 2 * 14, abs=0.05)


def test_per_column_and_per_row_sizes():
    fig, axes = ov.pl.multipanel((2, 3), panel_widths=[45, 60, 30],
                                 panel_heights=[80, 50],
                                 margins=(20, 6, 16, 10), spacing=(9, 7))
    expected = {"a": (45, 80), "b": (60, 80), "c": (30, 80),
                "d": (45, 50), "e": (60, 50), "f": (30, 50)}
    for key, (want_w, want_h) in expected.items():
        w, h = panel_mm(axes[key], fig)
        assert (w, h) == (pytest.approx(want_w, abs=0.05),
                          pytest.approx(want_h, abs=0.05)), f"panel {key}"


def test_asymmetric_margins_do_not_leak_into_the_panel():
    """A bigger left margin must move the panels, not shrink them."""
    fig_a, axes_a = ov.pl.multipanel((1, 2), panel_size=(40, 40), margins=10)
    fig_b, axes_b = ov.pl.multipanel((1, 2), panel_size=(40, 40),
                                     margins=(40, 10, 10, 10))
    assert panel_mm(axes_a["a"], fig_a) == pytest.approx(
        panel_mm(axes_b["a"], fig_b), abs=0.05)
    assert axes_b["a"].get_position().x0 > axes_a["a"].get_position().x0


def test_units_are_honoured():
    fig_mm, axes_mm = ov.pl.multipanel((1, 1), panel_size=(50.8, 25.4),
                                       units="mm", margins=0, spacing=0)
    fig_in, axes_in = ov.pl.multipanel((1, 1), panel_size=(2.0, 1.0),
                                       units="in", margins=0, spacing=0)
    assert panel_mm(axes_mm["a"], fig_mm) == pytest.approx(
        panel_mm(axes_in["a"], fig_in), abs=0.05)


class TestRefusals:
    """Contradictory inputs must say so rather than silently picking one."""

    def test_figure_size_and_panel_size_together(self):
        with pytest.raises(ValueError, match="not both"):
            ov.pl.multipanel((1, 2), width=100, panel_size=(40, 40))

    def test_a_mosaic_has_no_single_panel_width(self):
        with pytest.raises(ValueError, match="regular grid"):
            ov.pl.multipanel("AB", panel_size=(40, 40))

    def test_relative_ratios_alongside_absolute_sizes(self):
        with pytest.raises(ValueError, match="relative"):
            ov.pl.multipanel((1, 2), panel_widths=[40, 40], panel_heights=[40],
                             width_ratios=[1, 2])

    def test_one_of_widths_or_heights_is_not_enough(self):
        with pytest.raises(ValueError, match="both"):
            ov.pl.multipanel((1, 2), panel_widths=[40, 40])

    def test_wrong_number_of_column_sizes(self):
        with pytest.raises(ValueError, match="one per column"):
            ov.pl.multipanel((1, 3), panel_widths=[40, 40], panel_heights=[40])

    def test_margins_that_leave_no_room(self):
        with pytest.raises(ValueError):
            ov.pl.multipanel((1, 1), panel_size=(40, 40), margins=(-30, -30, 0, 0))


class TestPanelStyles:
    def test_color_cycle_binds_to_the_panel(self):
        fig, axes = ov.pl.multipanel(
            (1, 2), width=120, height=50,
            panel_styles={"a": {"color_cycle": ["#4C72B0", "#DD8452"]}})
        first = axes["a"].plot([0, 1], [0, 1])[0].get_color()
        second = axes["a"].plot([0, 1], [1, 0])[0].get_color()
        assert (first, second) == ("#4C72B0", "#DD8452")
        # The other panel keeps the global cycle.
        assert axes["b"].plot([0, 1], [0, 1])[0].get_color() not in {"#4C72B0"}

    def test_other_keys_reach_axes_set(self):
        fig, axes = ov.pl.multipanel(
            (1, 1), width=80, height=50,
            panel_styles={"a": {"title": "Baseline", "xlabel": "t",
                                "xlim": (0, 10)}})
        assert axes["a"].get_title() == "Baseline"
        assert axes["a"].get_xlabel() == "t"
        assert axes["a"].get_xlim() == (0, 10)

    def test_a_dual_keyed_panel_is_styled_once(self):
        """A mosaic panel answers to 'A' and 'a'; applying twice would reset.

        ``set_prop_cycle`` restarts the cycle, so styling the same axes through
        both of its keys would put the first two lines on the same colour —
        silently, and only visible in the drawn figure.
        """
        fig, axes = ov.pl.multipanel(
            "AB", width=100, height=40,
            panel_styles={"A": {"color_cycle": ["#111111", "#222222"]},
                          "a": {"color_cycle": ["#111111", "#222222"]}})
        assert axes["A"] is axes["a"]
        first = axes["A"].plot([0, 1], [0, 1])[0].get_color()
        second = axes["A"].plot([0, 1], [1, 0])[0].get_color()
        assert first != second, "the prop cycle was reset by a second application"

    def test_unknown_panel_is_refused(self):
        with pytest.raises(KeyError, match="do not exist"):
            ov.pl.multipanel((1, 2), width=100, height=40,
                             panel_styles={"z": {"title": "nope"}})


class TestFigureTitle:
    def test_title_is_drawn(self):
        fig, _ = ov.pl.multipanel((1, 2), width=120, height=50, title="Figure 1")
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "Figure 1"

    def test_title_kw_reaches_suptitle(self):
        fig, _ = ov.pl.multipanel((1, 2), width=120, height=50, title="Figure 2",
                                  title_kw={"x": 0.02, "ha": "left"})
        assert fig._suptitle.get_ha() == "left"

    def test_no_title_by_default(self):
        fig, _ = ov.pl.multipanel((1, 2), width=120, height=50)
        assert fig._suptitle is None


class TestBackwardCompatibility:
    """Every pre-existing call shape must behave exactly as before."""

    def test_mosaic_with_ratios(self):
        fig, axes = ov.pl.multipanel("AAB\nCDB", width="nature-double",
                                     height=90, width_ratios=[1, 1, 1.3])
        assert set("ABCDabcd") <= set(axes)
        assert axes["A"] is axes["a"]
        w_mm = fig.get_size_inches()[0] * MM_PER_IN
        assert w_mm == pytest.approx(183.0, abs=0.05)

    def test_plain_grid(self):
        fig, axes = ov.pl.multipanel((2, 2), width=120, height=80)
        assert sorted(axes) == ["a", "b", "c", "d"]

    def test_unlabelled_grid_is_keyed_by_position(self):
        fig, axes = ov.pl.multipanel((2, 2), width=120, height=80, label=False)
        assert sorted(axes) == [(0, 0), (0, 1), (1, 0), (1, 1)]
