"""`slopeplot(color_by=<subject-constant column>)` regression tests.

The wide pivot in slopeplot reduces the long frame to y indexed by subject on
x alone, and `resolve_columns` only keeps the columns it is handed (x, y,
subject) — so a metadata column such as a healthy/disease label never reached
the point where `color_by` was validated, and colouring by it raised even
though it was a real column. These tests pin the fix: a subject-constant column
is carried through the pivot and colours one line per subject, a column that
varies within a subject is a genuine error, and an unknown name still reports
what is available.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.lines import Line2D

from omicverse.pl._categorical import slopeplot


def _paired_frame():
    """Long-form: each `pair` is one subject measured across `site`, with a
    `label` (healthy/disease) that is constant within a pair."""
    rows = []
    for subject in range(6):
        label = "healthy" if subject < 3 else "disease"
        base = float(subject)
        for site in ["A", "B", "C"]:
            rows.append(dict(value=base + (0.0 if site == "A" else 1.0),
                             site=site, label=label, pair=f"s{subject}"))
    return pd.DataFrame(rows)


def test_color_by_subject_constant_column_draws_one_line_per_subject():
    frame = _paired_frame()
    n_subjects = frame["pair"].nunique()

    # summary=None so the only line artists are the subject lines themselves,
    # not the overlaid group-mean line.
    ax = slopeplot(frame, "site", "value", subject="pair",
                   color_by="label", summary=None)

    subject_lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 1]
    assert len(subject_lines) == n_subjects

    legend = ax.get_legend()
    assert legend is not None
    assert {t.get_text() for t in legend.get_texts()} == {"healthy", "disease"}


def test_color_by_column_varying_within_subject_raises():
    frame = _paired_frame()
    # break the invariant: one subject now carries two labels, so it cannot be
    # one line of one colour.
    frame.loc[frame["pair"] == "s0", "label"] = ["healthy", "disease", "healthy"]

    with pytest.raises(ValueError, match="not constant within every subject"):
        slopeplot(frame, "site", "value", subject="pair", color_by="label")


def test_color_by_unknown_column_names_available_columns():
    frame = _paired_frame()
    with pytest.raises(ValueError) as excinfo:
        slopeplot(frame, "site", "value", subject="pair", color_by="not_a_col")
    message = str(excinfo.value)
    assert "not_a_col" in message
    # the error has to say what could have been used instead
    assert "label" in message and "Available columns" in message


# ---------------------------------------------------------------------------
# grouped paired mode (`group=`): the cnsplots "two conditions within each of
# several groups" layout that the single-cluster path cannot express.
# ---------------------------------------------------------------------------


def _grouped_frame(pairs_per_group=4):
    """Long-form: within each `site`, every `pair` has exactly one healthy and
    one disease row — the two conditions the slope connects."""
    rows = []
    for site in ["site1", "site2", "site3"]:
        for k in range(pairs_per_group):
            pair = f"{site}_{k}"
            rows.append(dict(value=float(k), site=site, label="healthy", pair=pair))
            rows.append(dict(value=float(k) + 1.0, site=site, label="disease",
                             pair=pair))
    return pd.DataFrame(rows)


def test_group_draws_one_line_per_pair_and_ticks_are_groups():
    frame = _grouped_frame(pairs_per_group=4)
    ax = slopeplot(frame, x="label", y="value", subject="pair", group="site",
                   order=["healthy", "disease"])

    # one connecting line per pair, across all groups (3 groups x 4 pairs)
    pair_lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) == 2]
    assert len(pair_lines) == 12

    # the x-axis is the groups, not the two conditions
    assert [t.get_text() for t in ax.get_xticklabels()] == ["site1", "site2", "site3"]

    # the legend keys off the two conditions
    legend = ax.get_legend()
    assert legend is not None
    assert {t.get_text() for t in legend.get_texts()} == {"healthy", "disease"}


def test_group_endpoints_sit_either_side_of_each_group_centre():
    frame = _grouped_frame(pairs_per_group=3)
    ax = slopeplot(frame, x="label", y="value", subject="pair", group="site",
                   order=["healthy", "disease"])
    # scatter offsets: healthy at centre-0.2, disease at centre+0.2 per group
    xs = sorted({round(float(x), 2)
                 for coll in ax.collections for x, _ in coll.get_offsets()})
    assert xs == [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2]


def test_group_requires_exactly_two_conditions():
    frame = _grouped_frame()
    # a third condition makes the "connect two" contract ambiguous
    extra = frame[frame["label"] == "healthy"].copy()
    extra["label"] = "recovered"
    frame = pd.concat([frame, extra], ignore_index=True)
    with pytest.raises(ValueError, match="exactly two"):
        slopeplot(frame, x="label", y="value", subject="pair", group="site")


def test_group_subject_may_not_span_two_groups():
    frame = _grouped_frame()
    # move one pair's disease row into a different group: the pair now straddles
    # two groups and cannot be one cluster's line.
    frame.loc[(frame["pair"] == "site1_0") & (frame["label"] == "disease"),
              "site"] = "site2"
    with pytest.raises(ValueError, match="spans more than one"):
        slopeplot(frame, x="label", y="value", subject="pair", group="site")
