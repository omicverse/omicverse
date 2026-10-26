"""Tests for the edgeR / limma backends of ov.bulk.pyDEG.deg_analysis.

Both backends use pure-Python ports (the ``pyedger`` package from PyPI
and the vendored omicverse.external.pylimma) — no R / inmoose required. They are
R-parity tested in their own repos; here we only check that the omicverse
wiring produces the standard pyDEG result schema and recovers planted
differential genes.
"""

import sys

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


@pytest.fixture
def no_pydeseq2(monkeypatch):
    """Make ``import pydeseq2`` fail, as on an install without the extras."""
    monkeypatch.setitem(sys.modules, "pydeseq2", None)


# edger is left out only for speed (~1 min on 400 genes); it goes through the
# same code path before the method branch.
@pytest.mark.parametrize("method", ["limma", "edgepy", "ttest"])
def test_methods_run_without_pydeseq2(deg_counts, no_pydeseq2, method):
    dds = ov.bulk.pyDEG(deg_counts.copy())
    dds.drop_duplicates_index()
    res = dds.deg_analysis(["s0", "s1", "s2"], ["s3", "s4", "s5"], method=method)
    assert res.shape[0] == N_GENES


def test_deseq2_without_pydeseq2_raises_install_hint(deg_counts, no_pydeseq2):
    dds = ov.bulk.pyDEG(deg_counts.copy())
    dds.drop_duplicates_index()
    with pytest.raises(ImportError, match=r"pip install pydeseq2"):
        dds.deg_analysis(["s0", "s1", "s2"], ["s3", "s4", "s5"], method="DEseq2")


def test_unknown_method_raises(deg_counts):
    dds = ov.bulk.pyDEG(deg_counts.copy())
    dds.drop_duplicates_index()
    with pytest.raises(ValueError):
        dds.deg_analysis(["s0", "s1", "s2"], ["s3", "s4", "s5"], method="not_a_method")


@pytest.mark.parametrize("method", ["DEseq2", "edger", "limma", "edgepy", "ttest"])
def test_too_few_observations_raise_clear_error(deg_counts, method):
    dds = ov.bulk.pyDEG(deg_counts.copy())
    dds.drop_duplicates_index()
    with pytest.raises(ValueError, match="at least two distinct input columns"):
        dds.deg_analysis(["s0"], ["s3"], method=method)


@pytest.mark.parametrize(
    ("group1", "group2", "match"),
    [
        (["s0", "s0"], ["s3", "s4"], "group1 contains duplicate columns"),
        (["s0", "s1"], ["s3", "s3"], "group2 contains duplicate columns"),
        (["s0", "s1"], ["s1", "s3"], "group1 and group2 overlap"),
    ],
)
def test_invalid_group_columns_raise_clear_error(deg_counts, group1, group2, match):
    dds = ov.bulk.pyDEG(deg_counts.copy())
    dds.drop_duplicates_index()
    with pytest.raises(ValueError, match=match):
        dds.deg_analysis(group1, group2, method="ttest")


def test_missing_group_columns_raise_clear_error(deg_counts):
    dds = ov.bulk.pyDEG(deg_counts.copy())
    dds.drop_duplicates_index()
    with pytest.raises(KeyError, match="not found in the pyDEG count matrix"):
        dds.deg_analysis(["s0", "missing"], ["s3", "s4"], method="ttest")


def test_duplicate_count_matrix_columns_raise_clear_error(deg_counts):
    duplicated = deg_counts.copy()
    duplicated.columns = ["s0", "s0", "s2", "s3", "s4", "s5"]
    dds = ov.bulk.pyDEG(duplicated)
    dds.drop_duplicates_index()
    with pytest.raises(ValueError, match="count matrix columns must be unique"):
        dds.deg_analysis(["s0", "s2"], ["s3", "s4"], method="ttest")


# ---------------------------------------------------------------------------
# limma-voom library composition (TMM)
# ---------------------------------------------------------------------------
N_COMP, N_COMP_UP = 1000, 50


def _composition_counts(seed, scale):
    """Poisson counts (genes × 8 samples); genes 0..49 are multiplied by
    ``scale`` (per sample), the rest are unchanged. The planted genes make
    some libraries ~15% larger, which shifts every unchanged gene's logCPM
    unless voom gets TMM-normalized library sizes."""
    rng = np.random.default_rng(seed)
    mu = np.tile(rng.lognormal(np.log(200), 1.0, N_COMP)[:, None], (1, 8))
    mu[:N_COMP_UP] *= scale
    return pd.DataFrame(
        rng.poisson(mu).astype(float),
        index=[f"g{i:04d}" for i in range(N_COMP)],
        columns=[f"s{i}" for i in range(8)],
    )


def test_limma_unchanged_genes_not_shifted_by_composition():
    counts = _composition_counts(0, np.array([4.0] * 4 + [1.0] * 4))
    res = ov.bulk.pyDEG(counts).deg_analysis(
        list(counts.columns[:4]), list(counts.columns[4:]), method="limma")
    null = res.loc[counts.index[N_COMP_UP:]]
    assert abs(null["log2FC"].median()) < 0.05
    assert (null["qvalue"] < 0.05).mean() < 0.02
    assert (res.loc[counts.index[:N_COMP_UP], "sig"] == "up").all()


def test_continuous_deg_unchanged_genes_not_shifted_by_composition():
    trait = np.repeat([0.0, 1.0, 2.0, 3.0], 2)
    counts = _composition_counts(1, 1.0 + trait)     # genes 0..49 rise with trait
    res = ov.bulk.pyDEG(counts).continuous_deg(pd.Series(trait, index=counts.columns))
    null = res.loc[counts.index[N_COMP_UP:]]
    assert abs(null["log2FC"].median()) < 0.02       # slope per trait unit
    assert (null["qvalue"] < 0.05).mean() < 0.02
