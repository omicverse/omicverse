"""A small, dependency-free text-repel for label placement.

The third-party :mod:`adjustText` computes label positions against the axes
geometry at call time and, in a small or freshly-resized axes, can push labels
far outside the data limits (a volcano panel that ends up 60 mm tall in a
multi-panel figure sends its gene labels below the x-axis and off the canvas).

``adjust_text`` does the same job — move a set of labels so they neither
overlap each other nor sit on their anchor points — but **hard-clamps every
label inside the axes** on every iteration. That one invariant makes it safe to
call on a panel of any size, including *after* a layout engine has resized the
axes, which is exactly when the packaged version fails.

The algorithm is a short force relaxation in display (pixel) space: labels
repel each other along their smaller overlap, feel a gentle pull back toward
their anchor so they do not wander, and are clipped to the axes rectangle. Thin
leader lines connect each final label to its point.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from .._registry import register_function


def _boxes(texts, renderer) -> np.ndarray:
    """Display-space ``[x0, y0, x1, y1]`` for each text, one row each."""
    out = np.empty((len(texts), 4))
    for i, t in enumerate(texts):
        bb = t.get_window_extent(renderer)
        out[i] = (bb.x0, bb.y0, bb.x1, bb.y1)
    return out


@register_function(
    aliases=["adjust_text", "repel_labels", "标签排斥", "文字避让"],
    category="pl",
    description=(
        "Repel a set of text labels so they do not overlap each other or their "
        "points, hard-clamped inside the axes so labels never leave the panel"
    ),
    examples=[
        "texts = [ax.text(x, y, name) for x, y, name in top_genes]",
        "ov.pl.adjust_text(texts, ax=ax)",
        "# keep the original points explicitly when the texts were pre-moved",
        "ov.pl.adjust_text(texts, x=xs, y=ys, ax=ax, arrow_color='0.5')",
    ],
    related=["pl.volcano", "pl.scatterplot"],
)
def adjust_text(texts: Sequence[Any],
                *,
                ax=None,
                x: Optional[Sequence[float]] = None,
                y: Optional[Sequence[float]] = None,
                max_iter: int = 300,
                step: float = 0.6,
                pad: float = 2.0,
                pull: float = 0.015,
                arrows: bool = True,
                arrow_color: str = "0.6",
                arrow_lw: float = 0.5):
    r"""Move ``texts`` so they stop overlapping, without ever leaving the axes.

    Arguments
    ---------
    texts
        Already-created :class:`matplotlib.text.Text` objects (e.g. the return
        of repeated ``ax.text(...)``). They are moved in place.
    ax
        Axes the texts live on. Defaults to the first text's axes.
    x, y
        The anchor points the labels belong to, in data coordinates. Leader
        lines point here and the pull is toward here. Defaults to each text's
        current position (correct when the texts were created at their points).
    max_iter, step, pad, pull
        Relaxation controls: iteration cap, fraction of the computed push
        applied per step, padding in points kept between boxes and at the axes
        edge, and the strength of the pull back toward the anchor.
    arrows
        Draw a thin leader line from each label to its anchor.
    arrow_color, arrow_lw
        Leader-line style.

    Returns
    -------
    The same ``texts`` list, moved.

    Notes
    -----
    Unlike :mod:`adjustText`, a label is clamped to the axes rectangle on every
    iteration, so this is safe to call after the axes has been resized — the
    failure mode that sends labels off-canvas in a multi-panel layout cannot
    happen here.
    """
    texts = list(texts)
    if not texts:
        return texts
    if ax is None:
        ax = texts[0].axes
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    trans = ax.transData
    inv = trans.inverted()

    if x is None or y is None:
        pts = np.array([t.get_position() for t in texts], dtype=float)
    else:
        pts = np.column_stack([np.asarray(x, dtype=float),
                               np.asarray(y, dtype=float)])
    anchors = trans.transform(pts)                       # display space, fixed

    for t in texts:
        t.set_horizontalalignment("center")
        t.set_verticalalignment("center")

    n = len(texts)
    for _ in range(max_iter):
        boxes = _boxes(texts, renderer)
        centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                                   (boxes[:, 1] + boxes[:, 3]) / 2])
        push = np.zeros((n, 2))
        # pairwise separation along whichever axis they overlap less on
        for i in range(n):
            for j in range(i + 1, n):
                ox = (min(boxes[i, 2], boxes[j, 2])
                      - max(boxes[i, 0], boxes[j, 0])) + pad
                oy = (min(boxes[i, 3], boxes[j, 3])
                      - max(boxes[i, 1], boxes[j, 1])) + pad
                if ox > 0 and oy > 0:
                    if ox <= oy:
                        d = centers[i, 0] - centers[j, 0]
                        s = ox / 2 * (1.0 if d >= 0 else -1.0)
                        push[i, 0] += s
                        push[j, 0] -= s
                    else:
                        d = centers[i, 1] - centers[j, 1]
                        s = oy / 2 * (1.0 if d >= 0 else -1.0)
                        push[i, 1] += s
                        push[j, 1] -= s
        # gentle pull back toward the anchor so labels do not drift away
        push += (anchors - centers) * pull

        axbb = ax.get_window_extent()
        moved = 0.0
        for i, t in enumerate(texts):
            target = centers[i] + push[i] * step
            half_w = (boxes[i, 2] - boxes[i, 0]) / 2 + pad
            half_h = (boxes[i, 3] - boxes[i, 1]) / 2 + pad
            # hard clamp: the whole label box stays inside the axes rectangle
            target[0] = min(max(target[0], axbb.x0 + half_w), axbb.x1 - half_w)
            target[1] = min(max(target[1], axbb.y0 + half_h), axbb.y1 - half_h)
            moved += float(np.abs(target - centers[i]).sum())
            t.set_position(inv.transform(target))
        if moved < 1.0:
            break

    if arrows:
        for t, anchor in zip(texts, pts):
            ax.annotate("", xy=(anchor[0], anchor[1]),
                        xytext=t.get_position(),
                        arrowprops=dict(arrowstyle="-", color=arrow_color,
                                        lw=arrow_lw, shrinkA=1, shrinkB=1),
                        zorder=1, annotation_clip=False)
    return texts
