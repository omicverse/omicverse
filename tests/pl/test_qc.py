import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.figure import Figure

import omicverse as ov


def _qc_adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "nUMIs": [1000, 1200, 1500, 2000, 2400, 500000],
            "detected_genes": [300, 450, 600, 750, 900, 50000],
            "mito_perc": [1.0, 2.0, 3.0, 4.0, 5.0, 92.0],
            "sample": pd.Categorical(["s1", "s1", "s2", "s2", "s2", "s1"]),
        },
        index=[f"cell_{i}" for i in range(6)],
    )
    return AnnData(
        X=np.ones((6, 3), dtype=float),
        obs=obs,
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )


def _hist_right_edge(ax):
    return max(patch.get_x() + patch.get_width() for patch in ax.patches)


def test_qc_clip_at_returns_figure_and_does_not_modify_obs():
    adata = _qc_adata()
    original_umis = adata.obs["nUMIs"].copy(deep=True)
    original_genes = adata.obs["detected_genes"].copy(deep=True)

    fig = ov.pl.qc(adata, umi_clip_at=50000, gene_clip_at=8000, bins=5, log=False)

    assert isinstance(fig, Figure)
    pd.testing.assert_series_equal(adata.obs["nUMIs"], original_umis)
    pd.testing.assert_series_equal(adata.obs["detected_genes"], original_genes)
    plt.close(fig)


def test_qc_clip_at_limits_histogram_display_values_only_for_counts_and_genes():
    adata = _qc_adata()

    fig = ov.pl.qc(adata, umi_clip_at=50000, gene_clip_at=8000, bins=5, log=False)
    umi_ax, gene_ax, mito_ax = fig.axes[:3]

    assert _hist_right_edge(umi_ax) == pytest.approx(50000)
    assert _hist_right_edge(gene_ax) == pytest.approx(8000)
    assert _hist_right_edge(mito_ax) == pytest.approx(92.0)
    assert "clipped at 50000" in umi_ax.get_xlabel()
    assert "clipped at 8000" in gene_ax.get_xlabel()
    assert "clipped" not in mito_ax.get_xlabel()
    plt.close(fig)


def test_qc_clip_at_supports_grouped_histogram_and_violin():
    adata = _qc_adata()

    hist_fig = ov.pl.qc(
        adata,
        batch_key="sample",
        umi_clip_at=50000,
        gene_clip_at=8000,
        bins=5,
        log=False,
    )
    violin_fig = ov.pl.qc(
        adata,
        batch_key="sample",
        kind="violin",
        umi_clip_at=50000,
        gene_clip_at=8000,
        log=False,
    )

    assert isinstance(hist_fig, Figure)
    assert isinstance(violin_fig, Figure)
    assert "clipped at 50000" in hist_fig.axes[0].get_xlabel()
    assert "clipped at 50000" in violin_fig.axes[0].get_ylabel()
    plt.close(hist_fig)
    plt.close(violin_fig)


def test_qc_clip_at_applies_to_scanpy_fallback_metric_names():
    obs = pd.DataFrame(
        {
            "total_counts": [1000, 1200, 500000],
            "n_genes_by_counts": [300, 450, 50000],
            "pct_counts_mt": [1.0, 2.0, 92.0],
        },
        index=["cell_0", "cell_1", "cell_2"],
    )
    adata = AnnData(
        X=np.ones((3, 2), dtype=float),
        obs=obs,
        var=pd.DataFrame(index=["g1", "g2"]),
    )

    fig = ov.pl.qc(adata, umi_clip_at=50000, gene_clip_at=8000, bins=3, log=False)

    assert _hist_right_edge(fig.axes[0]) == pytest.approx(50000)
    assert _hist_right_edge(fig.axes[1]) == pytest.approx(8000)
    assert _hist_right_edge(fig.axes[2]) == pytest.approx(92.0)
    plt.close(fig)


@pytest.mark.parametrize("bad_value", [0, -1, "500", np.inf, np.nan, True])
def test_qc_clip_at_validates_umi_clip_at(bad_value):
    with pytest.raises(ValueError, match="umi_clip_at"):
        ov.pl.qc(_qc_adata(), umi_clip_at=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1, "500", np.inf, np.nan, True])
def test_qc_clip_at_validates_gene_clip_at(bad_value):
    with pytest.raises(ValueError, match="gene_clip_at"):
        ov.pl.qc(_qc_adata(), gene_clip_at=bad_value)
