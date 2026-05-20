"""Tests for ov.bulk.pyDEG.timecourse_deg — time-course / longitudinal DE.

``timecourse_deg`` is isomorphic to ``continuous_deg``: it is built on the
vendored pure-Python limma-voom (``omicverse.external.pylimma``) and answers
the three classic time-course questions with a *moderated F-test over a
block of design columns*:

1. which genes are temporally regulated (``group=None``),
2. whether trajectories differ between groups (``group=`` given — the
   group×time interaction F-test),
3. repeated-measures designs (``block=`` given — duplicateCorrelation).

These tests build small synthetic count matrices with a *known* temporal
pattern and assert that the method recovers the planted genes with high
power and controlled FDR on the null genes, that the group path flags the
interaction genes, and that the ``block=`` path runs.
"""

import numpy as np
import pandas as pd
import pytest

import omicverse as ov


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------
# Enough total genes that the planted (inflated) genes do not dominate the
# library size — otherwise voom's per-sample normalization would induce a
# spurious compositional time trend in the null genes.
N_GENES = 600
N_TRUE = 40          # genes 0..39 carry the planted temporal / interaction signal


def _temporal_counts(seed=0):
    """Counts (genes × samples) with genes 0..39 rising over time."""
    rng = np.random.default_rng(seed)
    times = np.array([0, 1, 2, 4, 8])
    reps = 4
    samp_t = np.repeat(times, reps).astype(float)
    n = len(samp_t)
    samples = [f"s{i}" for i in range(n)]
    genes = [f"g{i:04d}" for i in range(N_GENES)]
    base = rng.negative_binomial(50, 0.4, size=(N_GENES, n)).astype(float)
    for i in range(N_TRUE):                       # planted: rise with time
        base[i] *= (1.0 + 0.3 * samp_t)
    counts = pd.DataFrame(base, index=genes, columns=samples)
    return counts, pd.Series(samp_t, index=samples)


def _temporal_continuous(seed=0):
    """Continuous log-expression (genes × samples) — microarray-style.

    Genes 0..39 follow a smooth cell-cycle-like wave over time; the rest
    are flat plus noise. The matrix is already log-scaled (it carries
    negative values), so it must NOT be voom-transformed.
    """
    rng = np.random.default_rng(seed)
    times = np.array([0, 1, 2, 4, 8, 12])
    reps = 4
    samp_t = np.repeat(times, reps).astype(float)
    n = len(samp_t)
    samples = [f"s{i}" for i in range(n)]
    genes = [f"g{i:04d}" for i in range(N_GENES)]
    # log-expression centred at 8 with small biological noise.
    expr = rng.normal(8.0, 0.4, size=(N_GENES, n))
    for i in range(N_TRUE):                       # planted: smooth wave
        expr[i] += 2.0 * np.sin(samp_t / 12.0 * np.pi)
    # a couple of genes carry log-ratios straddling zero (negative values)
    expr[0] -= 8.0
    df = pd.DataFrame(expr, index=genes, columns=samples)
    return df, pd.Series(samp_t, index=samples)


def _interaction_counts(seed=2):
    """Counts where genes 0..39 rise over time ONLY in group B."""
    rng = np.random.default_rng(seed)
    times = np.array([0, 1, 2, 4])
    reps = 5
    samp_t = np.tile(np.repeat(times, reps), 2).astype(float)
    grp = np.array(["A"] * (len(times) * reps) + ["B"] * (len(times) * reps))
    n = len(samp_t)
    samples = [f"s{i}" for i in range(n)]
    genes = [f"g{i:04d}" for i in range(N_GENES)]
    base = rng.negative_binomial(60, 0.4, size=(N_GENES, n)).astype(float)
    for i in range(N_TRUE):                       # planted: time effect in B only
        base[i] *= np.where(grp == "B", 1.0 + 0.6 * samp_t, 1.0)
    counts = pd.DataFrame(base, index=genes, columns=samples)
    # subject IDs: 5 subjects, each sampled at all 4 time points, per group.
    subj = np.concatenate([
        np.tile(np.arange(reps), len(times)),                  # group A subjects 0..4
        np.tile(np.arange(reps, 2 * reps), len(times)),        # group B subjects 5..9
    ])
    return (counts, pd.Series(samp_t, index=samples),
            pd.Series(grp, index=samples), pd.Series(subj, index=samples))


_SCHEMA = {
    "F", "pvalue", "qvalue", "AveExpr", "BaseMean", "log2(BaseMean)",
    "log2FC", "abs(log2FC)", "-log(pvalue)", "-log(qvalue)", "sig",
}


# ---------------------------------------------------------------------------
# path 1 — temporal regulation (group=None)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("time_basis", ["factor", "spline", "auto"])
def test_temporal_recovery(time_basis):
    counts, time = _temporal_counts()
    dds = ov.bulk.pyDEG(counts.copy())
    dds.drop_duplicates_index()
    res = dds.timecourse_deg(time=time, time_basis=time_basis)

    # standard pyDEG-style schema.
    assert _SCHEMA.issubset(res.columns), f"missing: {_SCHEMA - set(res.columns)}"
    assert res is dds.result
    assert res.shape[0] == N_GENES

    # F-test => sig is temporal / normal, never up / down.
    assert set(res["sig"]).issubset({"temporal", "normal"})

    # trajectory-shape columns are present and recoverable.
    shape_cols = [c for c in res.columns if c.startswith("log2FC_")]
    assert len(shape_cols) >= 1

    # p / q values are valid probabilities; F is non-negative.
    for col in ("pvalue", "qvalue"):
        v = res[col].dropna()
        assert ((v >= 0) & (v <= 1)).all()
    assert (res["F"].dropna() >= 0).all()

    # high power on the planted temporally-regulated genes.
    planted = [f"g{i:04d}" for i in range(N_TRUE)]
    hit = set(res.index[res["sig"] == "temporal"])
    recovered = sum(g in hit for g in planted)
    assert recovered >= int(0.9 * N_TRUE), \
        f"{time_basis}: only {recovered}/{N_TRUE} temporal genes recovered"

    # controlled false-discovery on the null genes.
    null_genes = res.index[N_TRUE:]
    fdr = (res.loc[null_genes, "sig"] == "temporal").mean()
    assert fdr < 0.10, f"{time_basis}: null FDR too high ({fdr:.3f})"


def test_temporal_auto_picks_factor_then_spline():
    """auto => factor for <=4 time points, spline for more."""
    counts, time = _temporal_counts()
    dds = ov.bulk.pyDEG(counts.copy())
    dds.drop_duplicates_index()
    # 5 distinct time points -> auto resolves to spline (3 df by default).
    res = dds.timecourse_deg(time=time, time_basis="auto", spline_df=3)
    shape_cols = [c for c in res.columns if c.startswith("log2FC_time_s")]
    assert len(shape_cols) >= 2          # spline basis columns

    # collapse to 3 distinct time points -> auto resolves to factor.
    t3 = time.replace({4.0: 2.0, 8.0: 2.0})
    res3 = dds.timecourse_deg(time=t3, time_basis="auto")
    fac_cols = [c for c in res3.columns if c.startswith("log2FC_time_")]
    assert len(fac_cols) == 2            # one-hot of 3 levels, drop-first


def test_factor_columns_match_time_points():
    counts, time = _temporal_counts()
    dds = ov.bulk.pyDEG(counts.copy())
    dds.drop_duplicates_index()
    res = dds.timecourse_deg(time=time, time_basis="factor")
    # 5 distinct time points => 4 factor coefficients (drop-first).
    fac_cols = sorted(c for c in res.columns if c.startswith("log2FC_time_"))
    assert len(fac_cols) == 4


def test_null_pvalues_are_calibrated():
    """A pure-null matrix (no time signal) must give a uniform-ish p."""
    rng = np.random.default_rng(7)
    times = np.array([0, 1, 2, 4, 8])
    samp_t = np.repeat(times, 4).astype(float)
    samples = [f"s{i}" for i in range(len(samp_t))]
    counts = pd.DataFrame(
        rng.negative_binomial(50, 0.4, size=(800, len(samp_t))).astype(float),
        index=[f"g{i:04d}" for i in range(800)], columns=samples,
    )
    dds = ov.bulk.pyDEG(counts)
    dds.drop_duplicates_index()
    res = dds.timecourse_deg(time=pd.Series(samp_t, index=samples),
                             time_basis="factor")
    # under the null, ~5% of genes should fall below p<0.05 and FDR ~ 0.
    assert (res["pvalue"] < 0.05).mean() < 0.12
    assert (res["qvalue"] < 0.05).mean() < 0.02


# ---------------------------------------------------------------------------
# continuous (microarray / pre-normalized) input — data_type
# ---------------------------------------------------------------------------
def test_continuous_data_type_recovers_temporal_genes():
    """data_type='continuous' skips voom and runs lmFit directly on an
    already-log-scaled expression matrix (microarray-style)."""
    expr, time = _temporal_continuous()
    dds = ov.bulk.pyDEG(expr.copy())
    dds.drop_duplicates_index()
    res = dds.timecourse_deg(time=time, data_type="continuous",
                             time_basis="spline")

    assert _SCHEMA.issubset(res.columns)
    assert res.shape[0] == N_GENES
    assert set(res["sig"]).issubset({"temporal", "normal"})

    # high power on the planted smooth-wave genes.
    planted = [f"g{i:04d}" for i in range(N_TRUE)]
    hit = set(res.index[res["sig"] == "temporal"])
    recovered = sum(g in hit for g in planted)
    assert recovered >= int(0.9 * N_TRUE), \
        f"only {recovered}/{N_TRUE} continuous temporal genes recovered"

    # controlled false-discovery on the flat null genes.
    null_genes = res.index[N_TRUE:]
    fdr = (res.loc[null_genes, "sig"] == "temporal").mean()
    assert fdr < 0.10, f"continuous null FDR too high ({fdr:.3f})"


def test_data_type_auto_detection():
    """auto => 'continuous' for a matrix with negative / non-integer
    values, 'counts' for a non-negative near-integer matrix."""
    # continuous matrix (has negatives + non-integers) -> resolves to
    # the no-voom path; recovers the planted wave genes.
    expr, time = _temporal_continuous()
    dds = ov.bulk.pyDEG(expr.copy())
    dds.drop_duplicates_index()
    res_auto = dds.timecourse_deg(time=time, data_type="auto",
                                  time_basis="spline")
    res_cont = dds.timecourse_deg(time=time, data_type="continuous",
                                  time_basis="spline")
    # 'auto' on this matrix must behave exactly like 'continuous'.
    np.testing.assert_allclose(res_auto["F"].values, res_cont["F"].values)
    planted = [f"g{i:04d}" for i in range(N_TRUE)]
    hit = set(res_auto.index[res_auto["sig"] == "temporal"])
    assert sum(g in hit for g in planted) >= int(0.9 * N_TRUE)

    # a non-negative *integer* count matrix -> resolves to 'counts'
    # and matches an explicit data_type='counts' run.
    rng = np.random.default_rng(5)
    times = np.array([0, 1, 2, 4, 8])
    samp_t = np.repeat(times, 4).astype(float)
    csamples = [f"s{i}" for i in range(len(samp_t))]
    int_counts = pd.DataFrame(
        rng.negative_binomial(50, 0.4, size=(N_GENES, len(samp_t))),
        index=[f"g{i:04d}" for i in range(N_GENES)], columns=csamples,
    ).astype(float)
    for i in range(N_TRUE):                       # planted, kept integer
        int_counts.iloc[i] = np.round(
            int_counts.iloc[i].values * (1.0 + 0.3 * samp_t))
    ctime = pd.Series(samp_t, index=csamples)
    dc = ov.bulk.pyDEG(int_counts.copy())
    dc.drop_duplicates_index()
    r_auto = dc.timecourse_deg(time=ctime, data_type="auto",
                               time_basis="factor")
    r_counts = dc.timecourse_deg(time=ctime, data_type="counts",
                                 time_basis="factor")
    np.testing.assert_allclose(r_auto["F"].values, r_counts["F"].values)


def test_bad_data_type_raises():
    expr, time = _temporal_continuous()
    dds = ov.bulk.pyDEG(expr.copy())
    dds.drop_duplicates_index()
    with pytest.raises(ValueError):
        dds.timecourse_deg(time=time, data_type="not_a_type")


# ---------------------------------------------------------------------------
# path 2 — group × time interaction
# ---------------------------------------------------------------------------
def test_group_interaction_recovery():
    counts, time, group, _ = _interaction_counts()
    dds = ov.bulk.pyDEG(counts.copy())
    dds.drop_duplicates_index()
    res = dds.timecourse_deg(time=time, group=group, time_basis="factor")

    assert _SCHEMA.issubset(res.columns)
    assert set(res["sig"]).issubset({"temporal", "normal"})

    # the interaction F-test is over the group:time columns only.
    inter_cols = [c for c in res.columns if c.startswith("log2FC_group_")
                  and ":" in c]
    assert len(inter_cols) >= 1

    # genes with a group-specific trajectory are flagged.
    planted = [f"g{i:04d}" for i in range(N_TRUE)]
    hit = set(res.index[res["sig"] == "temporal"])
    recovered = sum(g in hit for g in planted)
    assert recovered >= int(0.85 * N_TRUE), \
        f"only {recovered}/{N_TRUE} interaction genes recovered"

    null_genes = res.index[N_TRUE:]
    fdr = (res.loc[null_genes, "sig"] == "temporal").mean()
    assert fdr < 0.10, f"interaction null FDR too high ({fdr:.3f})"


def test_group_interaction_ignores_shared_time_trend():
    """A time trend shared by BOTH groups must NOT be called by the
    interaction test (only group-specific trajectories should)."""
    rng = np.random.default_rng(11)
    times = np.array([0, 1, 2, 4])
    reps = 5
    samp_t = np.tile(np.repeat(times, reps), 2).astype(float)
    grp = np.array(["A"] * (len(times) * reps) + ["B"] * (len(times) * reps))
    samples = [f"s{i}" for i in range(len(samp_t))]
    base = rng.negative_binomial(60, 0.4, size=(N_GENES, len(samp_t))).astype(float)
    # genes 0..39 rise with time IDENTICALLY in both groups (no interaction).
    for i in range(N_TRUE):
        base[i] *= (1.0 + 0.5 * samp_t)
    counts = pd.DataFrame(base, index=[f"g{i:04d}" for i in range(N_GENES)],
                          columns=samples)
    dds = ov.bulk.pyDEG(counts)
    dds.drop_duplicates_index()
    res = dds.timecourse_deg(time=pd.Series(samp_t, index=samples),
                             group=pd.Series(grp, index=samples),
                             time_basis="factor")
    # the shared-trend genes should NOT be flagged by the interaction test.
    planted = [f"g{i:04d}" for i in range(N_TRUE)]
    flagged = (res.loc[planted, "sig"] == "temporal").mean()
    assert flagged < 0.25, \
        f"interaction test wrongly flagged shared-trend genes ({flagged:.2f})"


# ---------------------------------------------------------------------------
# path 3 — repeated measures (block=)
# ---------------------------------------------------------------------------
def test_block_repeated_measures_runs():
    counts, time, group, block = _interaction_counts()
    dds = ov.bulk.pyDEG(counts.copy())
    dds.drop_duplicates_index()
    res = dds.timecourse_deg(time=time, group=group, block=block,
                             time_basis="factor")
    # the repeated-measures path estimates a within-subject correlation.
    assert hasattr(dds, "timecourse_correlation")
    assert -1.0 < dds.timecourse_correlation < 1.0
    assert _SCHEMA.issubset(res.columns)
    assert res.shape[0] == N_GENES
    # still recovers the planted interaction genes.
    planted = [f"g{i:04d}" for i in range(N_TRUE)]
    hit = set(res.index[res["sig"] == "temporal"])
    assert sum(g in hit for g in planted) >= int(0.7 * N_TRUE)


def test_block_temporal_only_runs():
    """block= combines with the single-group temporal path too."""
    counts, time = _temporal_counts()
    # build per-subject block IDs: 4 subjects sampled at all 5 time points.
    samp_t = time.values
    times = sorted(set(samp_t))
    block = np.empty(len(samp_t), dtype=int)
    for t in times:
        idx = np.where(samp_t == t)[0]
        block[idx] = np.arange(len(idx))
    dds = ov.bulk.pyDEG(counts.copy())
    dds.drop_duplicates_index()
    res = dds.timecourse_deg(time=time,
                             block=pd.Series(block, index=time.index),
                             time_basis="factor")
    assert hasattr(dds, "timecourse_correlation")
    assert _SCHEMA.issubset(res.columns)


# ---------------------------------------------------------------------------
# covariates and edge cases
# ---------------------------------------------------------------------------
def test_covariates_numeric_and_categorical():
    counts, time = _temporal_counts()
    rng = np.random.default_rng(3)
    samples = list(time.index)
    cov = pd.DataFrame({
        "rin": rng.normal(8.0, 0.5, size=len(samples)),       # numeric
        "batch": rng.choice(["b1", "b2"], size=len(samples)),  # categorical
    }, index=samples)
    dds = ov.bulk.pyDEG(counts.copy())
    dds.drop_duplicates_index()
    res = dds.timecourse_deg(time=time, covariates=cov, time_basis="factor")
    planted = [f"g{i:04d}" for i in range(N_TRUE)]
    hit = set(res.index[res["sig"] == "temporal"])
    assert sum(g in hit for g in planted) >= int(0.8 * N_TRUE)


def test_missing_time_samples_dropped():
    counts, time = _temporal_counts()
    time = time.copy()
    time.iloc[:2] = np.nan          # two samples with missing time
    dds = ov.bulk.pyDEG(counts.copy())
    dds.drop_duplicates_index()
    res = dds.timecourse_deg(time=time, time_basis="factor")
    # method still runs and returns the full gene table.
    assert res.shape[0] == N_GENES


def test_bad_time_basis_raises():
    counts, time = _temporal_counts()
    dds = ov.bulk.pyDEG(counts.copy())
    dds.drop_duplicates_index()
    with pytest.raises(ValueError):
        dds.timecourse_deg(time=time, time_basis="not_a_basis")


def test_continuous_and_deg_analysis_untouched():
    """timecourse_deg must not perturb the sibling methods."""
    assert hasattr(ov.bulk.pyDEG, "continuous_deg")
    assert hasattr(ov.bulk.pyDEG, "deg_analysis")
    assert hasattr(ov.bulk.pyDEG, "timecourse_deg")


# ---------------------------------------------------------------------------
# temporal_clusters — soft-cluster the trajectories of time-course genes
# ---------------------------------------------------------------------------
def test_temporal_clusters():
    """temporal_clusters groups time-course genes by trajectory shape via
    the pymfuzz (Mfuzz) fuzzy c-means backend."""
    pytest.importorskip("pymfuzz")
    counts, time = _temporal_counts(seed=0)
    deg = ov.bulk.pyDEG(counts)
    res = deg.timecourse_deg(time=time)
    genes = list(res.index[res["sig"] == "temporal"])
    assert len(genes) >= 5

    tc = ov.bulk.temporal_clusters(counts, time, genes=genes,
                                   n_clusters=4, seed=0)
    assert list(tc.columns) == ["cluster", "membership"]
    assert len(tc) >= 5
    assert tc["cluster"].nunique() <= 4
    assert ((tc["membership"] >= 0) & (tc["membership"] <= 1)).all()
    assert tc.attrs["centers"].shape == (4, len(time.unique()))
    assert tc.attrs["membership_matrix"].shape[1] == 4
    # auto cluster-count selection also runs
    tc_auto = ov.bulk.temporal_clusters(counts, time, genes=genes, seed=0)
    assert tc_auto.attrs["n_clusters"] >= 2
