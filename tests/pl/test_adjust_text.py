"""``ov.pl.adjust_text`` keeps labels apart and, crucially, inside the axes.

The property that distinguishes it from the packaged ``adjustText`` is the hard
clamp: in a small axes, labels stay within the axes rectangle instead of being
pushed off-canvas. That is the failure this reimplementation exists to prevent,
so it is the first thing pinned here.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from omicverse.pl import adjust_text


def _overlap(a, b) -> bool:
    return (a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1)


def test_labels_are_clamped_inside_a_small_axes():
    # points crowd the top edge — naive repulsion would push labels above it
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    coords = [(0.50, 0.95), (0.52, 0.96), (0.48, 0.94), (0.50, 0.97)]
    texts = [ax.text(x, y, f"label_{i}") for i, (x, y) in enumerate(coords)]

    adjust_text(texts, ax=ax)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axbb = ax.get_window_extent()
    for t in texts:
        bb = t.get_window_extent(renderer)
        assert bb.x0 >= axbb.x0 - 1 and bb.x1 <= axbb.x1 + 1, "label left the axes in x"
        assert bb.y0 >= axbb.y0 - 1 and bb.y1 <= axbb.y1 + 1, "label left the axes in y"


def test_overlaps_are_reduced():
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    pts = [(5.0, 5.0), (5.1, 5.1), (4.9, 5.0), (5.0, 4.9)]
    texts = [ax.text(x, y, "GeneName") for x, y in pts]

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pairs = [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    before = sum(_overlap(texts[i].get_window_extent(r),
                          texts[j].get_window_extent(r)) for i, j in pairs)

    adjust_text(texts, ax=ax, arrows=False)

    fig.canvas.draw()
    after = sum(_overlap(texts[i].get_window_extent(r),
                         texts[j].get_window_extent(r)) for i, j in pairs)
    assert before > 0 and after < before


def test_arrows_are_drawn():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    texts = [ax.text(0.50, 0.50, "a"), ax.text(0.51, 0.51, "b")]
    n_before = len(ax.texts)

    adjust_text(texts, ax=ax, arrows=True)

    # one annotation (a leader line) added per label
    assert len(ax.texts) >= n_before + len(texts)


def test_empty_input_is_a_noop():
    fig, ax = plt.subplots()
    assert adjust_text([], ax=ax) == []
