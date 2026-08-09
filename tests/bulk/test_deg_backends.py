"""Tests for the edgeR / limma backends of ov.bulk.pyDEG.deg_analysis.

Both backends use the vendored pure-Python ports (omicverse.external.pyedger
and omicverse.external.pylimma) — no R / inmoose required. They are
R-parity tested in their own repos; here we only check that the omicverse
wiring produces the standard pyDEG result schema and recovers planted
differential genes.
"""

import numpy as np
import pandas as pd
import pytest

import omicverse as ov


N_GENES = 400
N_TRUE_DE = 40       # genes 0..39 are up-regulated in group 1


@pytest.fixture
def deg_counts():
    """Synthetic count matrix (genes × samples) with planted DE genes."""
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(
        rng.negative_binomial(20, 0.4, size=(N_GENES, 6)).astype(float),
        index=[f"g{i:04d}" for i in range(N_GENES)],
        columns=[f"s{i}" for i in range(6)],
    )
    counts.iloc[:N_TRUE_DE, :3] *= 4          # up in group 1
    return counts


_SCHEMA = {
    "pvalue", "qvalue", "log2FC", "abs(log2FC)", "BaseMean",
    "log2(BaseMean)", "-log(pvalue)", "-log(qvalue)", "sig",
}


@pytest.mark.parametrize("method", ["edger", "limma", "edgepy"])
def test_backend_schema_and_recovery(deg_counts, method):
    g1, g2 = ["s0", "s1", "s2"], ["s3", "s4", "s5"]
    dds = ov.bulk.pyDEG(deg_counts.copy())
    dds.drop_duplicates_index()
    res = dds.deg_analysis(g1, g2, method=method, alpha=0.05)

    # Standard pyDEG result schema.
    assert _SCHEMA.issubset(res.columns), f"missing columns: {_SCHEMA - set(res.columns)}"
    assert isinstance(res, pd.DataFrame)
    assert res.shape[0] == N_GENES

    # Planted DE genes (up in group 1) should be recovered, with positive log2FC.
    planted = [f"g{i:04d}" for i in range(N_TRUE_DE)]
    up = set(res.index[res["sig"] == "up"])
    recovered = sum(g in up for g in planted)
    assert recovered >= int(0.9 * N_TRUE_DE), f"{method}: only {recovered}/{N_TRUE_DE} recovered"
    assert res.loc[planted, "log2FC"].mean() > 0.5

    # Non-DE genes should be mostly normal.
    non_de = res.iloc[N_TRUE_DE:]
    assert (non_de["sig"] == "normal").mean() > 0.8

    # p / q values are valid probabilities.
    for col in ("pvalue", "qvalue"):
        v = res[col].dropna()
        assert ((v >= 0) & (v <= 1)).all()


def test_edger_specific_columns(deg_counts):
    dds = ov.bulk.pyDEG(deg_counts.copy())
    dds.drop_duplicates_index()
    res = dds.deg_analysis(["s0", "s1", "s2"], ["s3", "s4", "s5"], method="edger")
    # edgeR QL exposes the F statistic and logCPM.
    assert "F" in res.columns
    assert "logCPM" in res.columns


def test_limma_specific_columns(deg_counts):
    dds = ov.bulk.pyDEG(deg_counts.copy())
    dds.drop_duplicates_index()
    res = dds.deg_analysis(["s0", "s1", "s2"], ["s3", "s4", "s5"], method="limma")
    # limma exposes the moderated t statistic and average expression.
    assert "t" in res.columns
    assert "AveExpr" in res.columns


def test_unknown_method_raises(deg_counts):
    dds = ov.bulk.pyDEG(deg_counts.copy())
    dds.drop_duplicates_index()
    with pytest.raises(ValueError):
        dds.deg_analysis(["s0", "s1", "s2"], ["s3", "s4", "s5"], method="not_a_method")


@pytest.mark.parametrize("method", ["DEseq2", "edger", "limma", "edgepy", "ttest"])
def test_too_few_observations_raise_clear_error(deg_counts, method):
    dds = ov.bulk.pyDEG(deg_counts.copy())
    dds.drop_duplicates_index()
    with pytest.raises(ValueError, match="at least two observations"):
        dds.deg_analysis(["s0"], ["s3"], method=method)
