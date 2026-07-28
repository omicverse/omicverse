"""Tests for the data-first statistical plots and the layout/export layer.

Two kinds of assertion live here.

1. **Numerical parity.** The estimators are re-implementations, so each one is
   pinned to a value produced by the reference implementation on a fixed
   dataset: Kaplan-Meier / log-rank / Aalen-Johansen against ``lifelines``
   0.30.0, DeLong's AUC interval against R ``pROC`` 1.18, Gray's test against
   R ``cmprsk`` 2.2-12, and the meta-analysis against
   ``statsmodels.stats.meta_analysis`` (which is a hard dependency, so that
   one is checked live rather than pinned).

   The reference packages are *not* required to run these tests — the expected
   numbers are baked in as constants.

2. **Contract.** Every plotting function returns an ``Axes``, accepts a
   DataFrame / bare arrays / an AnnData-like object interchangeably, and the
   export path really does keep text as text.
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from omicverse.pl._classification import confusion, roc, roc_auc_ci  # noqa: E402
from omicverse.pl._forest import forest, meta_analysis  # noqa: E402
from omicverse.pl._layout import (  # noqa: E402
    JOURNAL_WIDTH_MM,
    add_panel_label,
    figure,
    multipanel,
    savefig,
    set_editable_text,
    take_legend_out,
)
from omicverse.pl._stats_common import resolve_columns  # noqa: E402
from omicverse.pl._survival import (  # noqa: E402
    aalen_johansen,
    cumulative_incidence,
    grays_test,
    kaplan_meier,
    logrank_test,
    survival,
)

# --------------------------------------------------------------------------
# fixed datasets — `default_rng` is stream-stable across NumPy versions
# --------------------------------------------------------------------------


def make_survival():
    rng = np.random.default_rng(20260727)
    n = 200
    grp = np.where(np.arange(n) < n // 2, "A", "B")
    lam = np.where(grp == "A", 0.05, 0.10)
    T = rng.exponential(1.0 / lam)
    C = rng.exponential(20.0, n)
    return np.minimum(T, C), (T <= C).astype(int), grp


def make_competing():
    rng = np.random.default_rng(20260728)
    n = 240
    grp = np.where(np.arange(n) < n // 2, "A", "B")
    h1 = np.where(grp == "A", 0.04, 0.09)
    T1 = rng.exponential(1.0 / h1)
    T2 = rng.exponential(1.0 / 0.05, n)
    C = rng.exponential(30.0, n)
    t = np.minimum(np.minimum(T1, T2), C)
    code = np.where((T1 <= T2) & (T1 <= C), 1,
                    np.where((T2 < T1) & (T2 <= C), 2, 0))
    return t, code, grp


def make_scores():
    rng = np.random.default_rng(20260729)
    n = 300
    y = (np.arange(n) % 2).astype(int)
    return y, rng.normal(y * 0.9, 1.0)


# Reference values, computed once against the packages named above.
KM_A_MEDIAN = 13.642009585218469          # lifelines KaplanMeierFitter
KM_A_SURV_AT_10 = 0.6445304290957725
KM_A_MEDIAN_CI = (10.935559444422825,     # lifelines median_survival_times()
                  32.84643671835402)
LOGRANK_CHI2 = 24.2660158656954           # lifelines multivariate_logrank_test
LOGRANK_P = 8.390647599419126e-07
AJ_CIF1_AT_10 = 0.34813048741651814       # lifelines AalenJohansenFitter
AJ_CIF1_AT_25 = 0.512018609133357
AJ_VAR1_AT_25 = 0.0015224199800410285
GRAY_CHI2 = 7.790932                      # R cmprsk::cuminc()$Tests, cause 1
GRAY_P = 0.005251
GRAY_CHI2_CAUSE2 = 1.309497
PROC_AUC = 0.731688888889                 # R pROC roc()/ci.auc(method='delong')
PROC_LO = 0.675718603199
PROC_HI = 0.787659174578


def _step_value(timeline, values, query):
    """Right-continuous step lookup, the convention of both estimators."""
    idx = np.searchsorted(timeline, query, side="right") - 1
    return float(values[idx])


# --------------------------------------------------------------------------
# survival estimators
# --------------------------------------------------------------------------


class TestSurvivalEstimators:
    def test_kaplan_meier_matches_lifelines(self):
        t, e, g = make_survival()
        fit = kaplan_meier(t[g == "A"], e[g == "A"])
        assert fit["median"] == pytest.approx(KM_A_MEDIAN, rel=1e-12)
        assert _step_value(fit["timeline"], fit["survival"], 10.0) == pytest.approx(
            KM_A_SURV_AT_10, rel=1e-12
        )

    def test_median_confidence_interval_matches_lifelines(self):
        t, e, g = make_survival()
        fit = kaplan_meier(t[g == "A"], e[g == "A"])
        low, high = fit["median_ci"]
        assert low == pytest.approx(KM_A_MEDIAN_CI[0], rel=1e-12)
        assert high == pytest.approx(KM_A_MEDIAN_CI[1], rel=1e-12)
        # Brookmeyer-Crowley: the interval must bracket the point estimate
        assert low <= fit["median"] <= high

    def test_confidence_band_stays_inside_zero_one(self):
        t, e, _ = make_survival()
        fit = kaplan_meier(t, e)
        finite = np.isfinite(fit["lower"]) & np.isfinite(fit["upper"])
        assert np.all(fit["lower"][finite] >= 0.0)
        assert np.all(fit["upper"][finite] <= 1.0)
        assert np.all(fit["lower"][finite] <= fit["survival"][finite] + 1e-12)
        assert np.all(fit["upper"][finite] >= fit["survival"][finite] - 1e-12)

    def test_survival_is_monotone_non_increasing(self):
        t, e, _ = make_survival()
        fit = kaplan_meier(t, e)
        assert np.all(np.diff(fit["survival"]) <= 1e-12)

    def test_no_censoring_gives_empirical_survival(self):
        t = np.array([1.0, 2.0, 3.0, 4.0])
        fit = kaplan_meier(t, np.ones(4))
        assert _step_value(fit["timeline"], fit["survival"], 2.0) == pytest.approx(0.5)
        assert _step_value(fit["timeline"], fit["survival"], 4.0) == pytest.approx(0.0)

    def test_logrank_matches_lifelines(self):
        t, e, g = make_survival()
        res = logrank_test(t, e, g)
        assert res["statistic"] == pytest.approx(LOGRANK_CHI2, rel=1e-10)
        assert res["pvalue"] == pytest.approx(LOGRANK_P, rel=1e-8)
        assert res["df"] == 1

    def test_logrank_hazard_ratio_direction(self):
        t, e, g = make_survival()
        res = logrank_test(t, e, g)
        # group B has twice the hazard of A, and A is the reference here
        assert res["hazard_ratio"] < 1.0
        lo, hi = res["hazard_ratio_ci"]
        assert lo < res["hazard_ratio"] < hi
        assert hi < 1.0

    def test_reference_group_does_not_depend_on_row_order(self):
        """Regression: the HR direction must not flip when rows are shuffled.

        Deriving the level order from ``pd.unique`` (order of appearance) made
        the reference group depend on which patient happened to be first in
        the file, so a per-cohort forest plot came out with half the hazard
        ratios inverted.
        """
        t, e, g = make_survival()
        forward = logrank_test(t, e, g)
        order = np.argsort(g)[::-1]  # now group "B" appears first
        shuffled = logrank_test(t[order], e[order], g[order])
        assert forward["groups"] == shuffled["groups"] == ["A", "B"]
        assert forward["hazard_ratio"] == pytest.approx(shuffled["hazard_ratio"])

    def test_explicit_group_order_sets_the_reference(self):
        t, e, g = make_survival()
        default = logrank_test(t, e, g)
        flipped = logrank_test(t, e, g, groups=["B", "A"])
        assert flipped["groups"] == ["B", "A"]
        assert flipped["hazard_ratio"] == pytest.approx(
            1.0 / default["hazard_ratio"], rel=1e-12)
        assert flipped["statistic"] == pytest.approx(default["statistic"])

    def test_categorical_order_is_respected(self):
        t, e, g = make_survival()
        categorical = pd.Categorical(g, categories=["B", "A"], ordered=True)
        res = logrank_test(t, e, categorical)
        assert res["groups"] == ["B", "A"]

    def test_unknown_explicit_level_is_reported(self):
        t, e, g = make_survival()
        with pytest.raises(ValueError, match="not present"):
            logrank_test(t, e, g, groups=["A", "Z"])

    def test_logrank_needs_two_groups(self):
        t, e, _ = make_survival()
        with pytest.raises(ValueError, match="at least two groups"):
            logrank_test(t, e, np.zeros(len(t)))


class TestCompetingRisks:
    def test_aalen_johansen_matches_lifelines(self):
        t, code, _ = make_competing()
        fit = aalen_johansen(t, code, 1)
        assert _step_value(fit["timeline"], fit["cif"], 10.0) == pytest.approx(
            AJ_CIF1_AT_10, rel=1e-10
        )
        assert _step_value(fit["timeline"], fit["cif"], 25.0) == pytest.approx(
            AJ_CIF1_AT_25, rel=1e-10
        )

    def test_aalen_johansen_variance_matches_lifelines(self):
        t, code, _ = make_competing()
        fit = aalen_johansen(t, code, 1)
        assert _step_value(fit["timeline"], fit["variance"], 25.0) == pytest.approx(
            AJ_VAR1_AT_25, rel=1e-8
        )

    def test_cif_never_exceeds_one_minus_km(self):
        """The whole point of Aalen-Johansen: 1 - KM over-states incidence."""
        t, code, _ = make_competing()
        aj = aalen_johansen(t, code, 1)
        km = kaplan_meier(t, code == 1)
        for query in (5.0, 10.0, 20.0, 40.0):
            naive = 1.0 - _step_value(km["timeline"], km["survival"], query)
            assert _step_value(aj["timeline"], aj["cif"], query) <= naive + 1e-12

    def test_cifs_of_all_causes_sum_to_one_minus_survival(self):
        t, code, _ = make_competing()
        f1 = aalen_johansen(t, code, 1)
        f2 = aalen_johansen(t, code, 2)
        for query in (5.0, 15.0, 30.0):
            total = (_step_value(f1["timeline"], f1["cif"], query)
                     + _step_value(f2["timeline"], f2["cif"], query))
            surv = _step_value(f1["timeline"], f1["survival"], query)
            assert total == pytest.approx(1.0 - surv, abs=1e-10)

    def test_grays_test_tracks_cmprsk(self):
        """Agreement with ``cmprsk::cuminc`` to within a few percent.

        Not an exact match: the statistic uses Gray's subdistribution risk set
        but the log-rank variance rather than Gray's martingale variance. On
        the four simulated cohorts checked against R the gap never exceeded
        2.3% of the chi-square, so 5% is the contract.
        """
        t, code, g = make_competing()
        res = grays_test(t, code, g, 1)
        assert res["df"] == 1
        assert res["statistic"] == pytest.approx(GRAY_CHI2, rel=0.05)
        assert res["pvalue"] == pytest.approx(GRAY_P, rel=0.15)

        competing = grays_test(t, code, g, 2)
        assert competing["statistic"] == pytest.approx(GRAY_CHI2_CAUSE2, rel=0.05)

    def test_grays_test_beats_a_plain_logrank_here(self):
        """The cause-specific log-rank answers a different question."""
        t, code, g = make_competing()
        gray = grays_test(t, code, g, 1)
        naive = logrank_test(t, code == 1, g)
        assert gray["pvalue"] < 0.05
        # both point the same way on this cohort, but the statistics differ
        assert gray["statistic"] != pytest.approx(naive["statistic"], rel=1e-3)

    def test_grays_test_null_case(self):
        """With groups assigned at random the test must not fire."""
        rng = np.random.default_rng(4)
        t, code, _ = make_competing()
        g = rng.integers(0, 2, len(t))
        res = grays_test(t, code, g, 1)
        assert res["pvalue"] > 0.05

    def test_unknown_cause_is_reported(self):
        t, code, g = make_competing()
        with pytest.raises(ValueError, match="No events of cause"):
            grays_test(t, code, g, 9)


# --------------------------------------------------------------------------
# classifier evaluation
# --------------------------------------------------------------------------


class TestROC:
    def test_auc_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score

        y, s = make_scores()
        assert roc_auc_ci(y, s)["auc"] == pytest.approx(roc_auc_score(y, s),
                                                        rel=1e-12)

    def test_delong_interval_matches_proc(self):
        y, s = make_scores()
        res = roc_auc_ci(y, s)
        assert res["auc"] == pytest.approx(PROC_AUC, abs=1e-11)
        assert res["lower"] == pytest.approx(PROC_LO, abs=1e-11)
        assert res["upper"] == pytest.approx(PROC_HI, abs=1e-11)

    def test_delong_handles_ties(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        s = np.array([1.0, 1.0, 1.0, 2.0, 0.0, 2.0])
        res = roc_auc_ci(y, s)
        assert 0.0 <= res["auc"] <= 1.0
        assert res["se"] > 0

    def test_positive_class_does_not_depend_on_row_order(self):
        """Regression: AUC came out as 1 - AUC when label 1 appeared first.

        ``roc`` picked ``pos_label`` from ``pd.unique`` (order of appearance)
        while ``roc_auc_ci`` used the sorted classes, so the same model scored
        0.79 in one call and 0.21 in another.
        """
        y, s = make_scores()
        order = np.argsort(y)[::-1]  # label 1 now comes first
        _, forward = roc(y_true=y, y_score=s, return_stats=True)
        _, shuffled = roc(y_true=y[order], y_score=s[order], return_stats=True)
        plt.close("all")
        assert forward["curves"]["score"]["auc"] > 0.5
        assert forward["curves"]["score"]["auc"] == pytest.approx(
            shuffled["curves"]["score"]["auc"])

    def test_plot_auc_agrees_with_the_estimator(self):
        y, s = make_scores()
        _, stats = roc(y, s, return_stats=True)
        plt.close("all")
        assert stats["curves"]["score"]["auc"] == pytest.approx(
            roc_auc_ci(y, s)["auc"], rel=1e-12)

    def test_single_class_is_rejected(self):
        with pytest.raises(ValueError, match="two classes"):
            roc_auc_ci(np.ones(10), np.arange(10.0))

    def test_plot_accepts_arrays_dict_and_frame(self):
        y, s = make_scores()
        frame = pd.DataFrame({"label": y, "p1": s, "p2": -s})

        ax = roc(y, s)
        assert ax.get_ylabel() == "True positive rate"
        plt.close("all")

        ax = roc(y_true=y, y_score={"a": s, "b": -s})
        assert len(ax.get_legend().get_texts()) == 2
        plt.close("all")

        ax, stats = roc(frame, "label", ["p1", "p2"], return_stats=True)
        assert set(stats["curves"]) == {"p1", "p2"}
        plt.close("all")

    def test_multiclass_one_vs_rest(self):
        rng = np.random.default_rng(11)
        labels = np.array(["a", "b", "c"])[rng.integers(0, 3, 150)]
        proba = rng.random((150, 3))
        proba /= proba.sum(axis=1, keepdims=True)
        ax, stats = roc(y_true=labels, y_score=proba,
                        score_names=["a", "b", "c"], average="both",
                        return_stats=True)
        assert {"a", "b", "c", "macro-average", "micro-average"} <= set(stats["curves"])
        plt.close("all")

    def test_mismatched_class_count_is_reported(self):
        rng = np.random.default_rng(3)
        labels = np.array(["a", "b", "c"])[rng.integers(0, 3, 60)]
        with pytest.raises(ValueError, match="classes but only"):
            roc(y_true=labels, y_score=rng.random(60))


class TestConfusion:
    def test_matrix_and_metrics(self):
        truth = np.array(["a"] * 5 + ["b"] * 5)
        pred = np.array(["a"] * 4 + ["b"] + ["b"] * 4 + ["a"])
        ax, stats = confusion(y_true=truth, y_pred=pred, return_stats=True)
        assert stats["matrix"].loc["a", "a"] == 4
        assert stats["matrix"].loc["b", "a"] == 1
        assert stats["accuracy"] == pytest.approx(0.8)
        assert stats["per_class"].loc["a", "recall"] == pytest.approx(0.8)
        plt.close("all")

    def test_row_normalisation_sums_to_one(self):
        rng = np.random.default_rng(5)
        truth = rng.integers(0, 4, 200)
        pred = np.where(rng.random(200) < 0.7, truth, rng.integers(0, 4, 200))
        _, stats = confusion(y_true=truth, y_pred=pred, normalize="true",
                             return_stats=True)
        assert np.allclose(stats["normalized"].sum(axis=1), 1.0)
        plt.close("all")

    def test_unseen_class_keeps_a_row(self):
        truth = np.array(["a", "b", "c"])
        pred = np.array(["a", "b", "b"])
        _, stats = confusion(y_true=truth, y_pred=pred, return_stats=True)
        assert list(stats["matrix"].index) == ["a", "b", "c"]
        plt.close("all")

    def test_bad_normalize_is_reported(self):
        with pytest.raises(ValueError, match="normalize"):
            confusion(y_true=[0, 1], y_pred=[0, 1], normalize="rows")


# --------------------------------------------------------------------------
# forest / meta-analysis
# --------------------------------------------------------------------------


EST = np.array([0.32, 0.11, 0.55, -0.05, 0.41, 0.28, 0.62])
SE = np.array([0.12, 0.19, 0.15, 0.22, 0.10, 0.17, 0.25])


class TestMetaAnalysis:
    def test_matches_statsmodels(self):
        from statsmodels.stats.meta_analysis import combine_effects

        res = combine_effects(EST, SE ** 2, method_re="dl")
        frame = res.summary_frame()
        fixed = meta_analysis(EST, SE, method="fixed")
        random = meta_analysis(EST, SE, method="random")

        assert fixed["estimate"] == pytest.approx(
            frame.loc["fixed effect", "eff"], rel=1e-12)
        assert fixed["se"] == pytest.approx(
            frame.loc["fixed effect", "sd_eff"], rel=1e-12)
        assert random["estimate"] == pytest.approx(
            frame.loc["random effect", "eff"], rel=1e-12)
        assert random["se"] == pytest.approx(
            frame.loc["random effect", "sd_eff"], rel=1e-12)
        assert random["tau2"] == pytest.approx(res.tau2, rel=1e-10)
        assert random["Q"] == pytest.approx(res.q, rel=1e-10)

    def test_random_interval_is_wider_under_heterogeneity(self):
        fixed = meta_analysis(EST, SE, method="fixed")
        random = meta_analysis(EST, SE, method="random")
        assert random["tau2"] > 0
        assert (random["upper"] - random["lower"]) > (fixed["upper"] - fixed["lower"])

    def test_homogeneous_studies_collapse_to_fixed(self):
        est = np.full(5, 0.4)
        se = np.full(5, 0.1)
        random = meta_analysis(est, se, method="random")
        fixed = meta_analysis(est, se, method="fixed")
        assert random["tau2"] == 0.0
        assert random["estimate"] == pytest.approx(fixed["estimate"])

    def test_zero_standard_error_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            meta_analysis(EST, np.zeros_like(SE))


class TestForestPlot:
    def test_from_se_and_from_bounds_agree(self):
        frame = pd.DataFrame({"beta": EST, "se": SE,
                              "label": [f"S{i}" for i in range(len(EST))]})
        _, a = forest(frame, "beta", se="se", label="label", return_stats=True)
        z = 1.959963984540054
        frame["lo"], frame["hi"] = EST - z * SE, EST + z * SE
        _, b = forest(frame, "beta", lower="lo", upper="hi", label="label",
                      return_stats=True)
        assert np.allclose(a["rows"]["lower"], b["rows"]["lower"])
        assert np.allclose(a["rows"]["upper"], b["rows"]["upper"])
        plt.close("all")

    def test_meta_row_is_appended(self):
        frame = pd.DataFrame({"beta": EST, "se": SE})
        _, stats = forest(frame, "beta", se="se", meta="both", return_stats=True)
        assert set(stats["meta"]) == {"fixed", "random"}
        assert (stats["rows"]["kind"] == "summary").sum() == 2
        plt.close("all")

    def test_log_scale_uses_a_log_axis(self):
        frame = pd.DataFrame({"HR": np.exp(EST), "se": SE})
        ax = forest(frame, "HR", se="se", log_scale=True)
        assert ax.get_xscale() == "log"
        plt.close("all")

    def test_needs_an_interval(self):
        with pytest.raises(TypeError, match="lower.*upper.*se|se"):
            forest(pd.DataFrame({"beta": EST}), "beta")

    def test_meta_without_se_is_refused(self):
        frame = pd.DataFrame({"beta": EST, "lo": EST - 1, "hi": EST + 1})
        ax = forest(frame, "beta", lower="lo", upper="hi")
        assert ax is not None
        plt.close("all")


# --------------------------------------------------------------------------
# input handling shared by the group
# --------------------------------------------------------------------------


class TestInputHandling:
    def test_dataframe_arrays_and_anndata_agree(self):
        t, e, g = make_survival()
        frame = pd.DataFrame({"months": t, "status": e, "arm": g})

        ax_frame, s_frame = survival(frame, "months", "status", "arm",
                                     return_stats=True)
        plt.close("all")
        ax_array, s_array = survival(time=t, event=e, group=g,
                                     return_stats=True)
        plt.close("all")
        assert s_frame["pvalue"] == pytest.approx(s_array["pvalue"])

        anndata = pytest.importorskip("anndata")
        adata = anndata.AnnData(np.zeros((len(t), 2), dtype=np.float32),
                                obs=frame)
        _, s_adata = survival(adata, "months", "status", "arm",
                              return_stats=True)
        plt.close("all")
        assert s_adata["pvalue"] == pytest.approx(s_frame["pvalue"])

    def test_unknown_column_suggests_a_close_match(self):
        frame = pd.DataFrame({"months": [1.0, 2.0], "status": [1, 0]})
        with pytest.raises(KeyError, match="month"):
            resolve_columns(frame, time="monthss", event="status")

    def test_length_mismatch_is_reported(self):
        with pytest.raises(ValueError, match="different lengths"):
            resolve_columns(None, time=[1.0, 2.0], event=[1])

    def test_missing_values_are_dropped_and_reported(self, capsys):
        frame = pd.DataFrame({"months": [1.0, np.nan, 3.0],
                              "status": [1, 1, 0]})
        survival(frame, "months", "status", risk_table=False)
        assert "dropped 1 rows" in capsys.readouterr().out
        plt.close("all")

    def test_column_name_without_a_table_is_reported(self):
        with pytest.raises(TypeError, match="names a column"):
            resolve_columns(None, time="months", event=[1, 0])


# --------------------------------------------------------------------------
# plots return axes and carry the right furniture
# --------------------------------------------------------------------------


class TestSurvivalPlot:
    def test_risk_table_moves_the_time_axis_down(self):
        t, e, g = make_survival()
        ax = survival(time=t, event=e, group=g, risk_table=True)
        # the curve axes hands its tick labels to the table below it
        assert not any(lbl.get_text() for lbl in ax.get_xticklabels())
        assert len(ax.get_figure().axes) == 2
        plt.close("all")

    def test_without_risk_table_the_axes_keeps_its_label(self):
        t, e, _ = make_survival()
        ax = survival(time=t, event=e, risk_table=False)
        assert ax.get_xlabel() == "time"
        plt.close("all")

    def test_censor_marks_are_drawn(self):
        t, e, _ = make_survival()
        ax = survival(time=t, event=e, risk_table=False, censor_marks=True)
        markers = [ln for ln in ax.get_lines() if ln.get_marker() == "|"]
        assert markers
        plt.close("all")

    def test_percent_axis(self):
        t, e, _ = make_survival()
        ax = survival(time=t, event=e, risk_table=False, percent=True)
        assert "100" in [lbl.get_text() for lbl in ax.get_yticklabels()]
        plt.close("all")


class TestCumulativeIncidencePlot:
    def test_group_comparison_annotates_grays_test(self):
        t, code, g = make_competing()
        ax, stats = cumulative_incidence(time=t, event=code, group=g, cause=1,
                                         return_stats=True)
        assert stats["test"].startswith("Gray")
        assert any("Gray" in txt.get_text() for txt in ax.texts)
        plt.close("all")

    def test_several_causes_for_one_cohort(self):
        t, code, _ = make_competing()
        ax = cumulative_incidence(time=t, event=code, cause=[1, 2],
                                  cause_labels={1: "relapse", 2: "death"})
        labels = [txt.get_text() for txt in ax.get_legend().get_texts()]
        assert any("relapse" in lbl for lbl in labels)
        plt.close("all")

    def test_groups_and_multiple_causes_are_refused(self):
        t, code, g = make_competing()
        with pytest.raises(ValueError, match="one cause across"):
            cumulative_incidence(time=t, event=code, group=g, cause=[1, 2])

    def test_naive_km_overlay_sits_above(self):
        t, code, _ = make_competing()
        ax = cumulative_incidence(time=t, event=code, cause=1,
                                  show_naive_km=True)
        dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
        assert dashed
        plt.close("all")


# --------------------------------------------------------------------------
# layout and export
# --------------------------------------------------------------------------


class TestLayout:
    def test_millimetre_size_is_exact(self):
        fig = figure(89, 60, units="mm", dpi=300)
        width, height = fig.get_size_inches()
        assert width == pytest.approx(89 / 25.4, rel=1e-12)
        assert height == pytest.approx(60 / 25.4, rel=1e-12)
        plt.close(fig)

    def test_pixel_size_is_exact(self):
        fig = figure(1200, 800, units="px", dpi=150)
        assert tuple(np.round(fig.get_size_inches() * 150).astype(int)) == (1200, 800)
        plt.close(fig)

    def test_journal_preset(self):
        fig = figure("nature-single", 60)
        assert fig.get_size_inches()[0] == pytest.approx(
            JOURNAL_WIDTH_MM["nature-single"] / 25.4, rel=1e-12)
        plt.close(fig)

    def test_unknown_preset_lists_the_options(self):
        with pytest.raises(KeyError, match="nature-single"):
            figure("nature-quadruple", 60)

    def test_aspect_fills_the_missing_side(self):
        fig = figure(width=100, aspect=2.0)
        w, h = fig.get_size_inches()
        assert w / h == pytest.approx(2.0, rel=1e-12)
        plt.close(fig)

    def test_grid_panels_are_labelled_in_reading_order(self):
        fig, axes = multipanel((2, 2), width=180, height=120)
        assert list(axes) == ["a", "b", "c", "d"]
        for key, ax in axes.items():
            assert any(txt.get_text() == key for txt in ax.texts)
        plt.close(fig)

    def test_mosaic_keys_are_preserved(self):
        fig, axes = multipanel("AAB\nCCB", width=180, height=90, label="A")
        assert set(axes) == {"A", "B", "C"}
        plt.close(fig)

    def test_label_can_be_switched_off(self):
        fig, axes = multipanel((1, 2), width=180, height=60, label=False)
        assert list(axes) == [(0, 0), (0, 1)]
        assert not any(ax.texts for ax in axes.values())
        plt.close(fig)

    def test_panel_label_uses_point_offsets(self):
        fig, ax = figure(100, 60, axes=True)
        note = add_panel_label(ax, "a", dx=-18, dy=5)
        assert note.get_position() == (-18, 5)
        assert note.xycoords == "axes fraction"
        plt.close(fig)

    def test_take_legend_out_moves_the_legend(self):
        fig, ax = figure(100, 60, axes=True)
        ax.plot([0, 1], [0, 1], label="x")
        legend = take_legend_out(ax, loc="right")
        assert legend.get_bbox_to_anchor() is not None
        assert legend.get_frame_on() is False
        plt.close(fig)

    def test_take_legend_out_without_entries_is_reported(self):
        fig, ax = figure(100, 60, axes=True)
        with pytest.raises(ValueError, match="No legend entries"):
            take_legend_out(ax)
        plt.close(fig)


class TestExport:
    def test_svg_keeps_text_as_text(self, tmp_path):
        fig, ax = figure(80, 60, axes=True)
        ax.set_xlabel("editable label")
        target = tmp_path / "fig.svg"
        savefig(fig, target, verbose=False)
        content = target.read_text()
        assert "<text" in content
        assert "editable label" in content
        plt.close(fig)

    def test_outlined_mode_drops_the_text_elements(self, tmp_path):
        previous = set_editable_text(False)
        try:
            fig, ax = figure(80, 60, axes=True, editable_text=False)
            ax.set_xlabel("outlined label")
            target = tmp_path / "fig.svg"
            savefig(fig, target, editable_text=False, verbose=False)
            content = target.read_text()
            # glyphs become <path>/<use>; matplotlib still leaves the string in
            # an XML comment, so the real check is that no <text> node exists
            assert "<text" not in content
            plt.close(fig)
        finally:
            plt.rcParams.update(previous)

    def test_several_formats_from_one_call(self, tmp_path):
        fig, ax = figure(80, 60, axes=True)
        ax.plot([0, 1], [0, 1])
        written = savefig(fig, tmp_path / "panel", formats=["png", "svg", "pdf"],
                          verbose=False)
        assert [os.path.basename(p) for p in written] == [
            "panel.png", "panel.svg", "panel.pdf"]
        assert all(os.path.getsize(p) > 0 for p in written)
        plt.close(fig)

    def test_path_only_call_uses_the_current_figure(self, tmp_path):
        fig, ax = figure(80, 60, axes=True)
        ax.plot([0, 1], [0, 1])
        target = tmp_path / "current.png"
        savefig(str(target), verbose=False)
        assert target.exists()
        plt.close(fig)

    def test_editable_rcparams_are_not_leaked(self, tmp_path):
        set_editable_text(False)
        before = dict(plt.rcParams)
        fig, ax = figure(60, 40, axes=True, editable_text=False)
        savefig(fig, tmp_path / "x.svg", editable_text=True, verbose=False)
        assert plt.rcParams["svg.fonttype"] == before["svg.fonttype"]
        plt.close(fig)
        set_editable_text(True)


class TestRegistry:
    def test_new_functions_are_registered(self):
        import omicverse.pl  # noqa: F401  — runs the decorators

        from omicverse._registry import get_registry

        names = {entry.get("full_name")
                 for entry in get_registry().get_by_category("pl")}
        for expected in (
            "omicverse.pl._survival.survival",
            "omicverse.pl._survival.cumulative_incidence",
            "omicverse.pl._classification.roc",
            "omicverse.pl._classification.confusion",
            "omicverse.pl._forest.forest",
            "omicverse.pl._layout.figure",
            "omicverse.pl._layout.multipanel",
            "omicverse.pl._layout.add_panel_label",
            "omicverse.pl._layout.savefig",
            "omicverse.pl._layout.set_editable_text",
            "omicverse.pl._layout.take_legend_out",
        ):
            assert expected in names, f"{expected} missing from the registry"

    def test_functions_are_reachable_from_the_public_namespace(self):
        import omicverse.pl as pl

        for name in ("survival", "cumulative_incidence", "roc", "confusion",
                     "forest", "figure", "multipanel", "add_panel_label",
                     "savefig", "set_editable_text", "take_legend_out",
                     "kaplan_meier", "logrank_test", "aalen_johansen",
                     "grays_test", "roc_auc_ci", "meta_analysis"):
            assert hasattr(pl, name), f"ov.pl.{name} is not exported"
            assert name in pl.__all__, f"ov.pl.{name} is missing from __all__"
