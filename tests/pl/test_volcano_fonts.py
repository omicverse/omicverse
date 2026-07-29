"""Tests for ov.pl.volcano axis-label / tick font sizing.

The axis labels used to inherit titlefont's fixed size (14), dwarfing the
5-6pt labels of neighbouring panels. They must now follow
rcParams['axes.labelsize'] by default and honour an explicit label_fontsize.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import omicverse as ov


def _fake_deg(n=60, seed=0):
    rng = np.random.default_rng(seed)
    fc = rng.normal(0, 2, n)
    q = rng.uniform(1e-6, 1, n)
    sig = np.where(fc > 1.5, "up", np.where(fc < -1.5, "down", "normal"))
    return pd.DataFrame({"log2FC": fc, "qvalue": q, "sig": sig})


def test_axis_labels_follow_rcparams_labelsize():
    df = _fake_deg()
    old = plt.rcParams["axes.labelsize"]
    plt.rcParams["axes.labelsize"] = 5
    try:
        fig, ax = plt.subplots()
        ov.pl.volcano(df, plot_genes_num=0, ax=ax)
        # ~5, not the old fixed 14.
        assert ax.xaxis.get_label().get_fontsize() == 5
        assert ax.yaxis.get_label().get_fontsize() == 5
        plt.close(fig)
    finally:
        plt.rcParams["axes.labelsize"] = old


def test_explicit_label_fontsize_honoured():
    df = _fake_deg()
    fig, ax = plt.subplots()
    ov.pl.volcano(df, plot_genes_num=0, label_fontsize=9, ax=ax)
    assert ax.xaxis.get_label().get_fontsize() == 9
    assert ax.yaxis.get_label().get_fontsize() == 9
    plt.close(fig)
