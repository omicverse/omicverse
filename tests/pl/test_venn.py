"""Tests for ov.pl.venn's matplotlib-venn backend.

These pin the behaviour that matters after switching off the non-proportional
`venn` package: draw into the caller's axes without leaking a figure, keep the
passed-in set names, emit subset-count texts, and size labels off the font
(rcParams by default, or an explicit fontsize).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import omicverse as ov


SETS3 = {"A": {1, 2, 3}, "B": {2, 3, 4}, "C": {3, 4, 5}}
SETS2 = {"Up": {1, 2, 3}, "Down": {3, 4, 5}}


def test_venn3_draws_into_ax_without_new_figure():
    fig, ax = plt.subplots()
    n_before = len(plt.get_fignums())
    out = ov.pl.venn(sets=SETS3, ax=ax)
    # No stray figure minted when an axes is supplied (decoy/gca-leak guard).
    assert len(plt.get_fignums()) == n_before
    assert out is ax
    # Set names are the ones passed in.
    labels = {t.get_text() for t in ax.texts}
    assert {"A", "B", "C"}.issubset(labels)
    # Subset-count texts exist (numbers drawn inside the regions).
    assert any(t.get_text().isdigit() for t in ax.texts)
    plt.close(fig)


def test_venn2_two_sets():
    fig, ax = plt.subplots()
    ov.pl.venn(sets=SETS2, ax=ax)
    labels = {t.get_text() for t in ax.texts}
    assert {"Up", "Down"}.issubset(labels)
    assert any(t.get_text().isdigit() for t in ax.texts)
    plt.close(fig)


def test_explicit_fontsize_is_honoured():
    fig, ax = plt.subplots()
    ov.pl.venn(sets=SETS3, ax=ax, fontsize=7)
    sizes = {round(t.get_fontsize(), 3) for t in ax.texts if t.get_text()}
    assert sizes == {7.0}
    plt.close(fig)


def test_fontsize_defaults_to_rcparams():
    fig, ax = plt.subplots()
    old = plt.rcParams["font.size"]
    plt.rcParams["font.size"] = 6
    try:
        ov.pl.venn(sets=SETS3, ax=ax)
        sizes = {round(t.get_fontsize(), 3) for t in ax.texts if t.get_text()}
        assert sizes == {6.0}
    finally:
        plt.rcParams["font.size"] = old
    plt.close(fig)


def test_too_many_sets_falls_back_or_errors(tmp_path):
    # 4 sets are outside matplotlib-venn's 2/3 support; must not raise the
    # 2/3-only path error. It routes to the venny4py fallback instead.
    # out=tmp_path keeps the fallback's intersection files out of the repo.
    fig, ax = plt.subplots()
    four = {"A": {1}, "B": {2}, "C": {3}, "D": {4}}
    # Should not raise the "need at least 2" error; fallback handles it.
    ov.pl.venn(sets=four, ax=ax, out=str(tmp_path) + "/")
    plt.close("all")


def test_too_few_sets_errors():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ov.pl.venn(sets={"only": {1, 2}}, ax=ax)
    plt.close(fig)
