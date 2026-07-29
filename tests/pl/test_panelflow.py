"""The axes box is the size that was asked for, whatever got drawn into it.

``multipanel`` divides a canvas into cells and the axes box is the leftover, so
a long y tick label eats the panel. ``panelflow`` declares the box and measures
the decorations out of the rendered figure, reserving them beside it. These
tests read the drawn position back and convert to millimetres rather than
trusting the arithmetic, because the arithmetic is only right if the
inches-to-figure-fraction conversion is right — and because the reserve only
exists after a draw, every geometry assertion here forces one first.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import seaborn as sns

import matplotlib.pyplot as plt  # noqa: E402

import omicverse as ov  # noqa: E402
from omicverse.pl._panelflow import _Panel, _child_map  # noqa: E402

MM_PER_IN = 25.4


def panel_mm(ax, fig) -> tuple[float, float]:
    """The panel as drawn, in millimetres."""
    box = ax.get_position()
    fig_w_in, fig_h_in = fig.get_size_inches()
    return (box.width * fig_w_in * MM_PER_IN, box.height * fig_h_in * MM_PER_IN)


def left_edge_mm(ax, fig) -> float:
    """Distance from the canvas' left edge to the axes box, in millimetres."""
    return ax.get_position().x0 * fig.get_size_inches()[0] * MM_PER_IN


def top_edge_mm(ax, fig) -> float:
    """Distance from the canvas' top edge down to the axes box, in millimetres."""
    fig_h_mm = fig.get_size_inches()[1] * MM_PER_IN
    return fig_h_mm - ax.get_position().y1 * fig_h_mm


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def cells():
    rng = np.random.default_rng(20260729)
    n = 120
    return pd.DataFrame({
        "cell_type": rng.choice(["B", "Mono", "NK", "T"], n),
        "score": rng.normal(0.0, 1.0, n),
        "depth": rng.normal(5.0, 1.5, n),
    })


class TestTheBoxIsTheSizeAsked:
    def test_requested_size_is_the_drawn_size(self):
        fl = ov.pl.panelflow(max_width=183, units="mm")
        fl.panel("A", 45, 100)
        fl.panel("B", 60, 100)
        fl.fig.canvas.draw()  # the reserves do not exist until something renders
        assert panel_mm(fl["A"], fl.fig) == (pytest.approx(45.0, abs=0.05),
                                             pytest.approx(100.0, abs=0.05))
        assert panel_mm(fl["B"], fl.fig) == (pytest.approx(60.0, abs=0.05),
                                             pytest.approx(100.0, abs=0.05))

    def test_long_tick_labels_move_the_panel_instead_of_shrinking_it(self):
        """The property a gridspec does not have.

        Both panels ask for 45x40 mm. One of them is given cell-type names as
        y tick labels and a y axis label on top of that — some 44 mm of text,
        which in a gridspec comes straight out of the cell and leaves a panel
        barely wider than the one next to it is tall. Here it comes out of the
        reserve: the box stays 45 mm and the whole panel moves right.
        """
        fl = ov.pl.panelflow(max_width=183, units="mm")
        plain = fl.panel("A", 45, 40)
        plain.plot([0, 1], [0, 1])
        loud = fl.panel("B", 45, 40)
        loud.plot([0, 1], [0, 1])
        loud.set_yticks([0.0, 0.5, 1.0],
                        ["Oligodendrocyte precursor", "Endothelial cell", "T"])
        loud.set_ylabel("Normalised expression")
        fl.fig.canvas.draw()

        for key in "AB":
            width, height = panel_mm(fl[key], fl.fig)
            assert width == pytest.approx(45.0, abs=0.05), f"panel {key} width"
            assert height == pytest.approx(40.0, abs=0.05), f"panel {key} height"

        # ... and the decorations really were that big, i.e. the assertion
        # above is not passing because nothing was measured.
        plain_reserve = fl._panels[0].left_reserve
        loud_reserve = fl._panels[1].left_reserve
        assert (loud_reserve - plain_reserve) * MM_PER_IN > 30

    def test_units_agree(self):
        """50.8 mm and 2 in are the same panel."""
        fl_mm = ov.pl.panelflow(max_width=183, units="mm")
        fl_mm.panel("A", 50.8, 25.4)
        fl_mm.fig.canvas.draw()
        fl_in = ov.pl.panelflow(max_width=183 / MM_PER_IN, units="in")
        fl_in.panel("A", 2.0, 1.0)
        fl_in.fig.canvas.draw()
        assert panel_mm(fl_mm["A"], fl_mm.fig) == pytest.approx(
            panel_mm(fl_in["A"], fl_in.fig), abs=0.05)

    def test_journal_preset_sizes_the_canvas(self):
        fl = ov.pl.panelflow("nature-double", units="mm")
        fl.panel("A", 45, 40)
        fl.fig.canvas.draw()
        assert fl.fig.get_size_inches()[0] * MM_PER_IN == pytest.approx(183.0,
                                                                        abs=0.05)


class TestReserveEverythingThatEscapesTheBox:
    """Not just the decorations we thought to look for.

    An earlier version of ``_measure`` enumerated the artists it expected —
    both axes, the title, the legend — and so measured nothing at all for a
    plot that writes loose text outside its box. ``ov.pl.forest`` does exactly
    that: the estimate strings go at ``x=1.04`` in axes coordinates, some
    30 mm to the right of a 50 mm panel, and landed on top of whatever was
    next in the row. The reserve now comes from ``ax.get_tightbbox``, which
    unions everything the axes will draw.
    """

    @pytest.fixture
    def hazard_ratios(self):
        return pd.DataFrame({
            "cohort": ["Discovery", "Validation", "Pooled"],
            "HR": [0.68, 0.81, 0.74],
            "se": [0.14, 0.19, 0.11],
        })

    def test_forest_annotations_do_not_land_on_the_next_panel(self, hazard_ratios):
        fl = ov.pl.panelflow(max_width=183, units="mm")
        k = fl.panel("K", 50, 32)
        ov.pl.forest(data=hazard_ratios, estimate="HR", se="se", label="cohort",
                     ax=k, log_scale=True)
        loser = fl.panel("L", 50, 32)
        loser.plot([0, 1], [0, 1])
        fl.fig.canvas.draw()

        renderer = fl.fig.canvas.get_renderer()
        k_content = k.get_tightbbox(renderer)
        l_box = loser.get_window_extent(renderer)

        # The overhang was real and it was measured ...
        assert k_content.x1 > k.get_window_extent(renderer).x1 + 100, (
            "the reproducer needs forest to actually write outside the box")
        assert fl._panels[0].over_right * MM_PER_IN > 20

        # ... so the panel beside it starts clear of everything K draws.
        assert not k_content.overlaps(l_box)
        assert l_box.x0 > k_content.x1

        # And neither panel paid for it in size.
        for label in ("K", "L"):
            assert panel_mm(fl[label], fl.fig) == (pytest.approx(50.0, abs=0.05),
                                                   pytest.approx(32.0, abs=0.05))

    def test_a_caption_below_the_box_is_reserved_for(self):
        """The same question on the bottom side: an unclipped caption."""
        fl = ov.pl.panelflow(max_width=183, units="mm")
        plain = fl.panel("A", 45, 40)
        plain.plot([0, 1], [0, 1])
        captioned = fl.panel("B", 45, 40)
        captioned.plot([0, 1], [0, 1])
        captioned.set_xlabel("pseudotime")
        captioned.text(0.5, -0.45, "n = 3 donors, two-sided Wilcoxon",
                       transform=captioned.transAxes, ha="center", va="top",
                       clip_on=False)
        fl.fig.canvas.draw()

        overhangs = [panel.over_bottom * MM_PER_IN for panel in fl._panels]
        assert overhangs[1] > overhangs[0] + 10
        renderer = fl.fig.canvas.get_renderer()
        assert captioned.get_tightbbox(renderer).y0 >= -0.5, (
            "the caption should be inside the canvas, not hanging off it")

    def test_a_clipped_annotation_is_not_reserved_for(self):
        """``get_tightbbox`` honours clipping, and that is correct.

        An artist with ``clip_on=True`` never appears outside the box, so
        reserving room for it would be pure whitespace. Pinned so nobody
        "fixes" the clipping check later.
        """
        fl = ov.pl.panelflow(max_width=183, units="mm")
        plain = fl.panel("A", 45, 40)
        plain.plot([0, 1], [0, 1])
        clipped = fl.panel("B", 45, 40)
        clipped.plot([0, 1], [0, 1])
        clipped.text(0.5, -0.45, "never actually drawn out here",
                     transform=clipped.transAxes, ha="center", va="top",
                     clip_on=True)
        fl.fig.canvas.draw()
        assert (fl._panels[1].over_bottom
                == pytest.approx(fl._panels[0].over_bottom, abs=1e-9))

    def test_the_tag_is_not_counted_twice(self):
        """The tag is placed *from* the reserve, so it must not be *in* it.

        It is a child of the axes, so ``get_tightbbox`` would union it in and
        each pass would push it a further tag-width to the left. It is kept
        out with ``set_in_layout(False)`` and measured on its own.
        """
        def reserve(tag: bool):
            fl = ov.pl.panelflow(max_width=183, units="mm", tag=tag)
            fl.panel("A", 45, 40).plot([0, 1], [0, 1])
            fl.fig.canvas.draw()
            return fl

        tagged, bare = reserve(True), reserve(False)
        assert tagged._tags["A"].get_in_layout() is False
        # The decorations are identical; only the tag differs.
        assert tagged._panels[0].decor_left == pytest.approx(
            bare._panels[0].decor_left, abs=1e-9)
        assert tagged._panels[0].tag_width > 0
        assert tagged._panels[0].left_reserve > bare._panels[0].left_reserve

    def test_the_tag_lands_exactly_pad_left_from_the_decorations(self):
        """What a double count actually looks like from outside.

        ``pad_left`` is defined as the clear air between the tag and the
        nearest thing the axes drew, so that gap is the invariant. Counting the
        tag twice adds a tag-width to the reserve, which moves the tag out by
        that much, which adds it again — a runaway that stops only because the
        re-layout pass count stops it, leaving the reserve at roughly three
        times its true size and the gap wrong by the same amount. Measured
        against the y axis rather than the panel's own numbers, so the engine
        is not being asked to mark its own work.
        """
        fl = ov.pl.panelflow(max_width=183, units="mm", pad_left=1.5)
        ax = fl.panel("A", 45, 40)
        ax.plot([0, 1], [0, 1])
        fl.fig.canvas.draw()

        renderer = fl.fig.canvas.get_renderer()
        pad_px = 1.5 / MM_PER_IN * fl.fig.dpi
        tag = fl._tags["A"].get_window_extent(renderer=renderer)
        decorations = ax.yaxis.get_tightbbox(renderer)
        assert decorations.x0 - tag.x1 == pytest.approx(pad_px, abs=1.0)

        # ... and it does not creep on redraw.
        for _ in range(3):
            fl.fig.canvas.draw()
        renderer = fl.fig.canvas.get_renderer()
        assert fl._tags["A"].get_window_extent(renderer=renderer).x1 == (
            pytest.approx(tag.x1, abs=0.5))


class TestFixedAspectIsTheOneThingNotPromised:
    """A fixed aspect and an arbitrary box are contradictory requests.

    ``ax.pie`` leaves ``aspect=1, adjustable='box'`` behind, and matplotlib
    then shrinks the drawn box to the largest square inside the rectangle it
    was handed. No layout engine can honour both, so this pins which one wins
    and how to get the other — it is documented behaviour, not a leak.
    """

    def test_a_pie_shrinks_to_square_inside_the_requested_rectangle(self):
        fl = ov.pl.panelflow(max_width=183, units="mm")
        ax = fl.panel("A", 55, 45)
        ax.pie([3, 4, 5])
        fl.fig.canvas.draw()

        fig_w_mm, fig_h_mm = (v * MM_PER_IN for v in fl.fig.get_size_inches())
        assigned = ax.get_position(original=True)
        assert assigned.width * fig_w_mm == pytest.approx(55.0, abs=0.05)
        assert assigned.height * fig_h_mm == pytest.approx(45.0, abs=0.05)
        assert panel_mm(ax, fl.fig) == (pytest.approx(45.0, abs=0.05),
                                        pytest.approx(45.0, abs=0.05))

    def test_datalim_keeps_the_requested_box(self):
        fl = ov.pl.panelflow(max_width=183, units="mm")
        ax = fl.panel("A", 55, 45)
        ax.pie([3, 4, 5])
        ax.set_adjustable("datalim")
        fl.fig.canvas.draw()
        assert panel_mm(ax, fl.fig) == (pytest.approx(55.0, abs=0.05),
                                        pytest.approx(45.0, abs=0.05))

    def test_a_square_request_is_met_exactly(self):
        """Which is the answer for a pie: ask for the shape you want."""
        fl = ov.pl.panelflow(max_width=183, units="mm")
        ax = fl.panel("A", 45, 45)
        ax.pie([3, 4, 5])
        fl.fig.canvas.draw()
        assert panel_mm(ax, fl.fig) == (pytest.approx(45.0, abs=0.05),
                                        pytest.approx(45.0, abs=0.05))


class TestFlow:
    def test_a_row_wraps_at_max_width(self):
        """Two 45 mm panels fit in 120 mm, three do not."""
        fl = ov.pl.panelflow(max_width=120, units="mm", tag=False,
                             margins=0, pad_left=0, pad_top=0)
        for label in "ABC":
            fl.panel(label, 45, 20).plot([0, 1], [0, 1])
        fl.fig.canvas.draw()

        a_top = top_edge_mm(fl["A"], fl.fig)
        b_top = top_edge_mm(fl["B"], fl.fig)
        c_top = top_edge_mm(fl["C"], fl.fig)
        assert b_top == pytest.approx(a_top, abs=0.05), "A and B share a row"
        assert left_edge_mm(fl["B"], fl.fig) > left_edge_mm(fl["A"], fl.fig) + 45

        # C did not fit, so it starts a new row *below* the first one.
        assert c_top > a_top + 20, "C should be on the row under A"
        assert left_edge_mm(fl["C"], fl.fig) == pytest.approx(
            left_edge_mm(fl["A"], fl.fig), abs=0.05), "a new row restarts at the left"

    def test_below_stacks_in_the_same_column(self):
        """C under A, and B beside A is not pushed around by it.

        Tags are off and the two column members are the same size, so their
        measured reserves are identical and the axes boxes line up exactly. A
        tag would not change the column, only the box inside it, because 'A'
        and 'C' are not the same number of millimetres of ink.
        """
        fl = ov.pl.panelflow(max_width=183, units="mm", tag=False)
        for label in ("A", "B"):
            fl.panel(label, 45, 40).plot([0, 1], [0, 1])
        fl.panel("C", 45, 40, below="A").plot([0, 1], [0, 1])
        fl.fig.canvas.draw()

        a, b, c = fl["A"], fl["B"], fl["C"]
        assert left_edge_mm(c, fl.fig) == pytest.approx(left_edge_mm(a, fl.fig),
                                                        abs=0.05)
        assert top_edge_mm(c, fl.fig) > top_edge_mm(a, fl.fig) + 40
        assert b.get_position().x0 > a.get_position().x1
        assert b.get_position().x0 > c.get_position().x1

        # The neighbour is where it would have been without the stack at all.
        reference = ov.pl.panelflow(max_width=183, units="mm", tag=False)
        for label in ("A", "B"):
            reference.panel(label, 45, 40).plot([0, 1], [0, 1])
        reference.fig.canvas.draw()
        assert left_edge_mm(b, fl.fig) == pytest.approx(
            left_edge_mm(reference["B"], reference.fig), abs=0.05)
        assert panel_mm(b, fl.fig) == pytest.approx(
            panel_mm(reference["B"], reference.fig), abs=0.05)

    def test_a_stack_wraps_as_one_unit(self):
        """A column is packed by its widest member, not its parent's width."""
        fl = ov.pl.panelflow(max_width=120, units="mm", tag=False,
                             margins=0, pad_left=0, pad_top=0)
        fl.panel("A", 40, 20).plot([0, 1], [0, 1])
        fl.panel("B", 40, 20, below="A").plot([0, 1], [0, 1])
        fl.panel("C", 40, 20).plot([0, 1], [0, 1])
        fl.fig.canvas.draw()
        # A and C share a row; B rides under A rather than filling the gap.
        assert top_edge_mm(fl["C"], fl.fig) == pytest.approx(
            top_edge_mm(fl["A"], fl.fig), abs=0.05)
        assert top_edge_mm(fl["B"], fl.fig) > top_edge_mm(fl["A"], fl.fig) + 20


class TestRefusals:
    def test_unknown_parent(self):
        fl = ov.pl.panelflow(max_width=120, units="mm")
        fl.panel("A", 40, 30)
        with pytest.raises(ValueError, match="does not exist"):
            fl.panel("B", 40, 30, below="Z")

    def test_a_panel_cannot_sit_under_itself(self):
        fl = ov.pl.panelflow(max_width=120, units="mm")
        with pytest.raises(ValueError, match="does not exist"):
            fl.panel("A", 40, 30, below="A")

    def test_a_cycle_is_refused(self):
        """``panel()`` cannot build a cycle — this is what makes that true.

        ``below=`` has to name an already-declared panel, so the public API
        gives a forest by construction. The guard is tested directly because a
        cycle reaching the packer would otherwise recurse until the interpreter
        gave up, and "cannot happen" is only a guarantee if something checks.
        """
        def stub(label, below):
            return _Panel(label=label, width=1.0, height=1.0, pad_left=0.0,
                          pad_top=0.0, margin_left=0.0, margin_right=0.0,
                          margin_bottom=0.0, margin_top=0.0, below=below)

        with pytest.raises(ValueError, match="cycle"):
            _child_map([stub("A", "C"), stub("B", "A"), stub("C", "B")])

    def test_duplicate_label(self):
        fl = ov.pl.panelflow(max_width=120, units="mm")
        fl.panel("A", 40, 30)
        with pytest.raises(ValueError, match="already exists"):
            fl.panel("A", 40, 30)

    def test_unknown_panel_is_named_with_what_exists(self):
        fl = ov.pl.panelflow(max_width=120, units="mm")
        fl.panel("A", 40, 30)
        fl.panel("B", 40, 30)
        with pytest.raises(KeyError) as excinfo:
            fl["X"]
        message = str(excinfo.value)
        assert "'X'" in message and "'A'" in message and "'B'" in message

    def test_non_positive_size(self):
        fl = ov.pl.panelflow(max_width=120, units="mm")
        with pytest.raises(ValueError, match="positive"):
            fl.panel("A", 0, 30)


class TestCurrentAxes:
    def test_panel_does_not_steal_gca(self):
        """Drawing into the wrong panel has to be an error, not a no-op.

        If ``panel()`` set the current axes, a bare ``plt.plot`` or an
        ``ov.pl`` call that forgot ``ax=`` would land in the most recently
        created panel and look like it worked.
        """
        fig = plt.figure()
        before = fig.add_subplot(111)
        plt.sca(before)

        fl = ov.pl.panelflow(max_width=120, units="mm")
        returned = fl.panel("A", 40, 30)

        assert plt.gcf() is fig
        assert plt.gca() is before
        assert returned is not before
        assert returned is fl["A"]

    def test_the_flow_figure_is_not_the_current_figure(self):
        fig = plt.figure()
        plt.sca(fig.add_subplot(111))
        fl = ov.pl.panelflow(max_width=120, units="mm")
        fl.panel("A", 40, 30)
        fl.panel("B", 40, 30)
        assert plt.gcf() is fig
        assert fl.fig is not fig


class TestAgnosticToWhatDrewIt:
    """The engine positions an axes; it does not care who filled it."""

    def _assert_exact(self, fl, label):
        width, height = panel_mm(fl[label], fl.fig)
        assert width == pytest.approx(45.0, abs=0.05), f"panel {label} width"
        assert height == pytest.approx(40.0, abs=0.05), f"panel {label} height"

    def test_matplotlib_seaborn_and_ov_pl(self, cells):
        fl = ov.pl.panelflow(max_width=183, units="mm")

        mpl_ax = fl.panel("A", 45, 40)
        mpl_ax.plot(cells["depth"], cells["score"], "o", ms=2)
        mpl_ax.set_ylabel("score")

        sns_ax = fl.panel("B", 45, 40)
        sns.scatterplot(data=cells, x="depth", y="score", hue="cell_type",
                        ax=sns_ax, legend=False)

        ov_ax = fl.panel("C", 45, 40)
        ov.pl.barplot(data=cells, x="cell_type", y="score", ax=ov_ax)

        fl.fig.canvas.draw()
        for label in "ABC":
            self._assert_exact(fl, label)

        # Each of the three drew a different amount outside its box, and each
        # reserve was measured from what that library actually rendered — the
        # engine has no idea which of them drew.
        reserves = [panel.left_reserve * MM_PER_IN for panel in fl._panels]
        assert all(value > 0 for value in reserves)
        assert len(set(round(value, 3) for value in reserves)) == 3

    def test_seaborn_did_not_replace_the_axes(self, cells):
        """seaborn draws into the axes it is given; the flow still owns it."""
        fl = ov.pl.panelflow(max_width=183, units="mm")
        ax = fl.panel("A", 45, 40)
        sns.violinplot(data=cells, x="cell_type", y="score", ax=ax)
        fl.fig.canvas.draw()
        assert fl["A"] is ax
        assert ax.figure is fl.fig
        self._assert_exact(fl, "A")


class TestRelayout:
    def test_repeated_draws_settle(self):
        """A settled flow must not keep moving, or saving is a lottery."""
        fl = ov.pl.panelflow(max_width=183, units="mm", title="Figure 1")
        a = fl.panel("A", 45, 40)
        a.plot([0, 1], [0, 1e6])
        a.set_ylabel("a fairly long axis label")
        fl.panel("B", 60, 40).plot([0, 1], [0, 1])
        fl.fig.canvas.draw()
        first = [fl[label].get_position().frozen() for label in "AB"]
        first_size = tuple(fl.fig.get_size_inches())
        for _ in range(3):
            fl.fig.canvas.draw()
        later = [fl[label].get_position().frozen() for label in "AB"]
        assert tuple(fl.fig.get_size_inches()) == first_size
        for before, after in zip(first, later):
            assert before.bounds == pytest.approx(after.bounds, abs=1e-9)

    def test_title_reserves_a_band_above_the_first_row(self):
        titled = ov.pl.panelflow(max_width=183, units="mm", title="Figure 1")
        titled.panel("A", 45, 40).plot([0, 1], [0, 1])
        titled.fig.canvas.draw()
        bare = ov.pl.panelflow(max_width=183, units="mm")
        bare.panel("A", 45, 40).plot([0, 1], [0, 1])
        bare.fig.canvas.draw()
        assert top_edge_mm(titled["A"], titled.fig) > top_edge_mm(bare["A"],
                                                                  bare.fig) + 2
        assert panel_mm(titled["A"], titled.fig) == pytest.approx(
            panel_mm(bare["A"], bare.fig), abs=0.05)

    def test_saving_settles_the_layout_first(self, tmp_path):
        fl = ov.pl.panelflow(max_width=183, units="mm")
        a = fl.panel("A", 45, 40)
        a.plot([0, 1], [0, 1e6])
        written = fl.savefig(str(tmp_path / "figure1"), formats=["pdf", "svg"],
                             verbose=False)
        assert len(written) == 2
        assert all((tmp_path / name).exists()
                   for name in ("figure1.pdf", "figure1.svg"))
        # savefig draws, and drawing is what re-lays-out; the geometry it wrote
        # must be the settled one.
        before = fl["A"].get_position().frozen()
        fl.fig.canvas.draw()
        assert fl["A"].get_position().bounds == pytest.approx(before.bounds,
                                                              abs=1e-9)
        assert panel_mm(fl["A"], fl.fig) == (pytest.approx(45.0, abs=0.05),
                                             pytest.approx(40.0, abs=0.05))


class TestPanelOptions:
    def test_color_cycle_binds_to_the_panel(self):
        fl = ov.pl.panelflow(max_width=183, units="mm")
        ax = fl.panel("A", 45, 40, color_cycle=["#4C72B0", "#DD8452"])
        assert ax.plot([0, 1], [0, 1])[0].get_color() == "#4C72B0"
        assert ax.plot([0, 1], [1, 0])[0].get_color() == "#DD8452"
        other = fl.panel("B", 45, 40)
        assert other.plot([0, 1], [0, 1])[0].get_color() != "#4C72B0"

    def test_style_reaches_axes_set(self):
        fl = ov.pl.panelflow(max_width=183, units="mm")
        ax = fl.panel("A", 45, 40, style={"title": "Baseline", "xlabel": "t",
                                          "xlim": (0, 10)})
        assert ax.get_title() == "Baseline"
        assert ax.get_xlabel() == "t"
        assert ax.get_xlim() == (0, 10)

    def test_automatic_labels_follow_the_alphabet(self):
        fl = ov.pl.panelflow(max_width=183, units="mm")
        fl.panel(width=30, height=20)
        fl.panel(width=30, height=20)
        assert fl.labels == ["A", "B"]
        lower = ov.pl.panelflow(max_width=183, units="mm", label="a")
        lower.panel(width=30, height=20)
        assert lower.labels == ["a"]

    def test_a_tag_is_drawn_outside_the_box(self):
        fl = ov.pl.panelflow(max_width=183, units="mm")
        ax = fl.panel("A", 45, 40)
        ax.plot([0, 1], [0, 1])
        fl.fig.canvas.draw()
        tag = fl._tags["A"]
        renderer = fl.fig.canvas.get_renderer()
        assert tag.get_text() == "A"
        assert tag.get_window_extent(renderer=renderer).x1 <= (
            ax.get_window_extent(renderer).x0 + 0.5)

    def test_tag_false_keeps_the_label_addressable(self):
        fl = ov.pl.panelflow(max_width=183, units="mm", tag=False)
        ax = fl.panel("A", 45, 40)
        fl.fig.canvas.draw()
        assert fl["A"] is ax
        assert "A" not in fl._tags


# --------------------------------------------------------------------------
# points, and the measurement policy
# --------------------------------------------------------------------------


def test_points_are_a_unit():
    """Panel sizes quoted in points (72 pt = 1 in) — cnsplots' native unit."""
    import omicverse as ov

    fl_pt = ov.pl.panelflow(max_width=720, units="pt", margins=0)
    a = fl_pt.panel("A", 144, 72)
    fl_in = ov.pl.panelflow(max_width=10, units="in", margins=0)
    b = fl_in.panel("A", 2, 1)

    fl_pt.fig.canvas.draw()
    fl_in.fig.canvas.draw()
    wa, ha = (a.get_position().width * fl_pt.fig.get_size_inches()[0],
              a.get_position().height * fl_pt.fig.get_size_inches()[1])
    wb, hb = (b.get_position().width * fl_in.fig.get_size_inches()[0],
              b.get_position().height * fl_in.fig.get_size_inches()[1])
    assert wa == pytest.approx(wb, abs=1e-6)
    assert ha == pytest.approx(hb, abs=1e-6)


def test_reserve_axis_packs_tighter_than_tight():
    """'axis' reserves for the axis decorations only, 'tight' for everything.

    A legend thrown outside the axes is measured by 'tight' and ignored by
    'axis', so the same panels give a narrower canvas under 'axis'.
    """
    import omicverse as ov

    def next_panel_x0(policy):
        fl = ov.pl.panelflow(max_width=200, units="mm", reserve=policy,
                             margins=0)
        ax = fl.panel("A", 40, 40)
        ax.plot([0, 1], [0, 1], label="a long legend entry that sticks out")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        following = fl.panel("B", 40, 40)
        fl.fig.canvas.draw()
        return following.get_position().x0

    # 'tight' reserves the room the legend occupies, pushing B to the right;
    # 'axis' does not measure it, so B sits much closer to A.
    assert next_panel_x0("axis") < next_panel_x0("tight") - 0.05


def test_reserve_rejects_an_unknown_policy():
    import omicverse as ov

    with pytest.raises(ValueError, match="reserve must be"):
        ov.pl.panelflow(max_width=100, reserve="whatever")
