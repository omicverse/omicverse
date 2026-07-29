"""``ov.pl.declutter_ticks`` separates tick labels that actually collide.

The property under test is the outcome, not the remedy: after the pass, no two
visible tick labels overlap. Which of rotate / stagger / thin got there is an
implementation detail, and asserting on it would freeze the ladder.

The check has to happen at draw time, because whether labels collide depends on
the physical size the axes ends up at — the case that motivates the function is
a panel a layout engine shrinks *after* the plotting call.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from omicverse.pl import declutter_ticks
from omicverse.pl._declutter import _collides, _visible_labels


def _overlap_free(ax, which="x"):
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    return not _collides(_visible_labels(ax, which), renderer, which)


def _crowded_axes(width=1.6, n=8, label="LongCategory"):
    fig, ax = plt.subplots(figsize=(width, 1.6))
    # The limits have to cover the ticks: matplotlib drops out-of-view ticks at
    # draw time, so an axes left on the default (0, 1) would report three
    # labels however many were set.
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{label}{i}" for i in range(n)])
    return fig, ax


def test_crowded_x_labels_end_up_separated():
    fig, ax = _crowded_axes()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert _collides(_visible_labels(ax, "x"), renderer, "x"), \
        "fixture is not actually crowded"

    declutter_ticks(ax, on_draw=False)
    assert _overlap_free(ax, "x")


def test_roomy_labels_are_left_alone():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["a", "b", "c"])
    fig.canvas.draw()
    before = [(t.get_rotation(), t.get_visible()) for t in ax.get_xticklabels()]

    declutter_ticks(ax, on_draw=False)

    after = [(t.get_rotation(), t.get_visible()) for t in ax.get_xticklabels()]
    assert after == before, "an uncrowded axes was modified"


def test_draw_time_callback_reacts_to_a_later_resize():
    """The motivating case: the axes is shrunk after the plotting call."""
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_xlim(-0.5, 7.5)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"LongCategory{i}" for i in range(8)])
    declutter_ticks(ax)                      # on_draw=True, nothing to fix yet
    fig.canvas.draw()

    fig.set_size_inches(1.8, 1.6)            # now they would collide
    assert _overlap_free(ax, "x"), "the callback did not re-evaluate"


def test_rotation_ceiling_is_respected():
    fig, ax = _crowded_axes()
    declutter_ticks(ax, on_draw=False, max_rotation=30, stagger=False,
                    thin=False)
    assert all(abs(t.get_rotation()) <= 30 + 1e-6
               for t in ax.get_xticklabels())


def test_thinning_alone_can_clear_the_collision():
    # Moderate-length labels: 12 of them collide, every 4th does not — so
    # thinning alone is genuinely sufficient here. (With very long labels and
    # both other remedies disabled, `max_thin` can legitimately run out.)
    fig, ax = _crowded_axes(n=12, label="Cat")
    fig.canvas.draw()
    # get_xticklabels() returns only the visible labels, so a drop in its
    # length is the observable effect of thinning.
    before = len(ax.get_xticklabels())

    declutter_ticks(ax, on_draw=False, rotate=False, stagger=False)

    assert len(ax.get_xticklabels()) < before, "nothing was thinned"
    assert _overlap_free(ax, "x")


def test_thinning_is_undone_when_the_axes_grows_again():
    """Hidden labels must be recoverable: shrink then grow restores them."""
    fig, ax = _crowded_axes(width=1.6, n=12)
    declutter_ticks(ax)
    fig.canvas.draw()
    thinned = len(ax.get_xticklabels())

    fig.set_size_inches(16, 3)
    fig.canvas.draw()

    assert len(ax.get_xticklabels()) > thinned, \
        "labels hidden while small were never restored"


def test_repeated_passes_do_not_accumulate_offsets():
    fig, ax = _crowded_axes()
    declutter_ticks(ax, on_draw=False)
    fig.canvas.draw()
    first = [t.get_window_extent(fig.canvas.get_renderer()).y0
             for t in _visible_labels(ax, "x")]

    for _ in range(3):
        declutter_ticks(ax, on_draw=False)
    fig.canvas.draw()
    again = [t.get_window_extent(fig.canvas.get_renderer()).y0
             for t in _visible_labels(ax, "x")]

    assert again == pytest.approx(first, abs=0.5), \
        "labels drifted further on each pass"


def test_axis_argument_is_validated():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="must be 'x', 'y' or 'both'"):
        declutter_ticks(ax, axis="horizontal")


def test_style_axes_registers_the_pass_by_default():
    from omicverse.pl._plot_backend import set_tick_declutter, style_axes

    fig, ax = plt.subplots(figsize=(1.6, 1.6))
    ax.set_xlim(-0.5, 7.5)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"LongCategory{i}" for i in range(8)])
    style_axes(ax)
    assert _overlap_free(ax, "x")

    previous = set_tick_declutter(False)
    try:
        fig2, ax2 = plt.subplots(figsize=(1.6, 1.6))
        ax2.set_xlim(-0.5, 7.5)
        ax2.set_xticks(range(8))
        ax2.set_xticklabels([f"LongCategory{i}" for i in range(8)])
        style_axes(ax2)
        assert not _overlap_free(ax2, "x"), "the global switch was ignored"
    finally:
        set_tick_declutter(previous)


def _crowded_y_axes(height=0.9, n=5):
    """Crowded, but not hopelessly: collides at full size, fits at 70%."""
    fig, ax = plt.subplots(figsize=(3.0, height))
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"Group{i}" for i in range(n)])
    return fig, ax


def test_shrinking_is_preferred_over_dropping_labels():
    """A category axis should lose font size before it loses category names."""
    fig, ax = _crowded_y_axes()
    fig.canvas.draw()
    assert _collides(_visible_labels(ax, "y"), fig.canvas.get_renderer(), "y"), \
        "fixture is not actually crowded"
    before_n = len(ax.get_yticklabels())
    before_size = ax.get_yticklabels()[0].get_size()

    declutter_ticks(ax, axis="y", on_draw=False)

    assert ax.get_yticklabels()[0].get_size() < before_size, "nothing shrank"
    assert len(ax.get_yticklabels()) == before_n, "a label was dropped"
    assert _overlap_free(ax, "y")


def test_shrinking_is_undone_when_the_axes_grows_again():
    fig, ax = _crowded_y_axes()
    original = ax.get_yticklabels()[0].get_size()
    declutter_ticks(ax, axis="y")
    fig.canvas.draw()
    assert ax.get_yticklabels()[0].get_size() < original, "nothing shrank"

    fig.set_size_inches(3.0, 8.0)
    fig.canvas.draw()
    assert ax.get_yticklabels()[0].get_size() == pytest.approx(original), \
        "the font size was never given back"
