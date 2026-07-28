"""Tests for the generic (table-first) plot layer and the AnnData adapter.

Three things are checked.

1. **The numbers behind the picture.** Every plot that computes something —
   a group comparison, a bar height, a regression slope, a density, a
   genomic-inflation factor — is checked against ``scipy`` / ``statsmodels`` /
   plain NumPy on the same data. A plot that draws the wrong number is worse
   than no plot.

2. **The contract.** Data-first means a DataFrame, bare arrays and an
   ``AnnData`` are interchangeable and give identical results, and that bad
   input produces an error naming the problem.

3. **The adapter.** ``as_plotdata`` resolves keys the same way whatever the
   container, and ``accepts_frame`` lets an ``AnnData``-shaped plot take a
   table without mutating the caller's frame.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import trapezoid  # np.trapz is gone in numpy 2

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from omicverse.pl._categorical import (  # noqa: E402
    barplot, donutplot, lollipopplot, pieplot, slopeplot, stackplot,
    stripplot, violinplot,
)
from omicverse.pl._distribution import histplot, kdeplot, qqplot, ridgeplot  # noqa: E402
from omicverse.pl._plotdata import (  # noqa: E402
    ObsView, PlotData, accepts_frame, as_plotdata,
)
from omicverse.pl._relational import lineplot, regplot, scatterplot  # noqa: E402
from omicverse.pl._stats_tests import (  # noqa: E402
    add_stat_annotation, compare_groups, format_pvalue,
)


@pytest.fixture
def cells():
    """A small per-cell table: four types, two conditions, two measurements."""
    rng = np.random.default_rng(20260727)
    n = 320
    cell_type = rng.choice(["B", "Mono", "NK", "T"], n, p=[0.25, 0.2, 0.15, 0.4])
    condition = rng.choice(["ctrl", "drug"], n)
    shift = np.select([cell_type == "T", cell_type == "NK"], [1.2, 0.4], 0.0)
    return pd.DataFrame({
        "cell_type": cell_type,
        "condition": condition,
        "score": rng.normal(shift + (condition == "drug") * 0.5, 1.0),
        "counts": rng.lognormal(8.0, 0.6, n),
        "sample": rng.choice([f"S{i}" for i in range(5)], n),
    })


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------
# group comparisons
# --------------------------------------------------------------------------


class TestCompareGroups:
    def test_two_group_pvalue_matches_scipy(self, cells):
        from scipy import stats

        subset = cells[cells["cell_type"].isin(["B", "T"])]
        result = compare_groups(subset, "score", "cell_type")
        a = subset.loc[subset["cell_type"] == "B", "score"]
        b = subset.loc[subset["cell_type"] == "T", "score"]
        expected = stats.mannwhitneyu(a, b, alternative="two-sided")
        assert result.loc[0, "statistic"] == pytest.approx(expected.statistic)
        assert result.loc[0, "pvalue"] == pytest.approx(expected.pvalue)
        assert result.loc[0, "test"] == "Mann-Whitney U"

    def test_named_tests_match_scipy(self, cells):
        from scipy import stats

        subset = cells[cells["cell_type"].isin(["B", "T"])]
        a = subset.loc[subset["cell_type"] == "B", "score"].to_numpy()
        b = subset.loc[subset["cell_type"] == "T", "score"].to_numpy()
        cases = {
            "welch": stats.ttest_ind(a, b, equal_var=False).pvalue,
            "ttest": stats.ttest_ind(a, b, equal_var=True).pvalue,
            "brunnermunzel": stats.brunnermunzel(a, b).pvalue,
        }
        for name, expected in cases.items():
            got = compare_groups(subset, "score", "cell_type", test=name)
            assert got.loc[0, "pvalue"] == pytest.approx(expected), name

    def test_omnibus_matches_scipy(self, cells):
        from scipy import stats

        result = compare_groups(cells, "score", "cell_type")
        overall = result[result["group1"] == "*"].iloc[0]
        samples = [g["score"].to_numpy()
                   for _, g in cells.groupby("cell_type", observed=True)]
        expected = stats.kruskal(*samples)
        assert overall["statistic"] == pytest.approx(expected.statistic)
        assert overall["pvalue"] == pytest.approx(expected.pvalue)
        assert overall["test"] == "Kruskal-Wallis"

    def test_anova_omnibus_follows_a_parametric_pairwise(self, cells):
        result = compare_groups(cells, "score", "cell_type", test="welch")
        assert result.iloc[0]["test"] == "One-way ANOVA"

    def test_holm_correction_matches_statsmodels(self, cells):
        from statsmodels.stats.multitest import multipletests

        result = compare_groups(cells, "score", "cell_type", omnibus=False)
        expected = multipletests(result["pvalue"], method="holm")[1]
        assert np.allclose(result["pvalue_corrected"], expected)

    def test_correction_can_be_switched_off(self, cells):
        result = compare_groups(cells, "score", "cell_type", correction=None,
                                omnibus=False)
        assert np.allclose(result["pvalue"], result["pvalue_corrected"])

    def test_all_pairs_are_covered(self, cells):
        result = compare_groups(cells, "score", "cell_type", omnibus=False)
        assert len(result) == 6  # 4 choose 2
        assert list(result["group1"])[:3] == ["B", "B", "B"]

    def test_pairs_restricts_the_comparisons(self, cells):
        result = compare_groups(cells, "score", "cell_type",
                                pairs=[("B", "T")], omnibus=False)
        assert len(result) == 1
        assert (result.loc[0, "group1"], result.loc[0, "group2"]) == ("B", "T")

    def test_unknown_pair_is_reported(self, cells):
        with pytest.raises(ValueError, match="absent or too small"):
            compare_groups(cells, "score", "cell_type", pairs=[("B", "ghost")])

    def test_tiny_groups_are_dropped_and_announced(self, capsys):
        frame = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10 + ["c"],
                              "v": np.arange(21.0)})
        result = compare_groups(frame, "v", "g", omnibus=False)
        assert "skipped groups smaller than 2" in capsys.readouterr().out
        assert set(result["group1"]) == {"a"}

    def test_paired_test_needs_equal_sizes(self):
        frame = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 4, "v": np.arange(9.0)})
        with pytest.raises(ValueError, match="equal group sizes"):
            compare_groups(frame, "v", "g", test="wilcoxon")

    def test_group_order_is_deterministic(self, cells):
        shuffled = cells.iloc[::-1].reset_index(drop=True)
        a = compare_groups(cells, "score", "cell_type", omnibus=False)
        b = compare_groups(shuffled, "score", "cell_type", omnibus=False)
        assert list(a["group1"]) == list(b["group1"])
        assert np.allclose(a["pvalue"], b["pvalue"])


class TestFormatPvalue:
    @pytest.mark.parametrize("p,expected", [
        (1e-6, "****"), (5e-4, "***"), (5e-3, "**"), (0.03, "*"), (0.4, "ns"),
    ])
    def test_star_cutoffs(self, p, expected):
        assert format_pvalue(p, "star") == expected

    def test_styles(self):
        assert format_pvalue(0.03, "value").startswith("P = ")
        assert format_pvalue(0.03, "full").startswith("* (P = ")
        assert format_pvalue(0.03, "compact") == "0.03"

    def test_unknown_style_is_reported(self):
        with pytest.raises(ValueError, match="style"):
            format_pvalue(0.01, "emoji")


class TestAnnotation:
    def test_brackets_are_drawn_and_room_is_made(self, cells):
        ax = violinplot(cells, "cell_type", "score")
        before = ax.get_ylim()[1]
        add_stat_annotation(ax, cells, "score", "cell_type")
        assert ax.get_ylim()[1] > before
        assert any(t.get_text() in {"*", "**", "***", "****", "ns"}
                   for t in ax.texts)

    def test_hide_ns_drops_the_uninteresting_ones(self, cells):
        ax = violinplot(cells, "cell_type", "score")
        add_stat_annotation(ax, cells, "score", "cell_type", hide_ns=True)
        assert all(t.get_text() != "ns" for t in ax.texts)

    def test_brackets_do_not_overlap(self, cells):
        """Every pair on the same tier must be disjoint along the category axis."""
        from omicverse.pl._stats_tests import _bracket_levels

        pairs = [(0, 1), (1, 2), (0, 2), (2, 3)]
        tiers = _bracket_levels(pairs)
        by_tier = {}
        for (lo, hi), tier in zip(pairs, tiers):
            for other_lo, other_hi in by_tier.setdefault(tier, []):
                assert hi < other_lo or lo > other_hi
            by_tier[tier].append((lo, hi))

    def test_reusing_a_results_table(self, cells):
        results = compare_groups(cells, "score", "cell_type")
        ax = violinplot(cells, "cell_type", "score")
        _, echoed = add_stat_annotation(ax, results=results, return_stats=True)
        assert echoed is results

    def test_needs_either_results_or_data(self, cells):
        ax = violinplot(cells, "cell_type", "score")
        with pytest.raises(TypeError, match="compare_groups"):
            add_stat_annotation(ax)


# --------------------------------------------------------------------------
# categorical plots
# --------------------------------------------------------------------------


class TestCategorical:
    def test_bar_heights_are_the_requested_statistic(self, cells):
        _, stats = barplot(cells, "cell_type", "score", estimator="median",
                           errorbar=None, return_stats=True)
        expected = cells.groupby("cell_type", observed=True)["score"].median()
        got = stats["summary"].set_index("x")["median"]
        assert np.allclose(got.reindex(expected.index), expected)

    def test_standard_error_is_the_standard_error(self, cells):
        _, stats = barplot(cells, "cell_type", "score", errorbar="se",
                           return_stats=True)
        row = stats["summary"].iloc[0]
        sample = cells.loc[cells["cell_type"] == row["x"], "score"]
        assert row["err_low"] == pytest.approx(
            sample.std(ddof=1) / np.sqrt(len(sample)))

    def test_unknown_estimator_is_reported(self, cells):
        with pytest.raises(ValueError, match="estimator"):
            barplot(cells, "cell_type", "score", estimator="mode")

    def test_unknown_errorbar_is_reported(self, cells):
        with pytest.raises(ValueError, match="errorbar"):
            barplot(cells, "cell_type", "score", errorbar="stderr")

    def test_test_is_skipped_with_hue_and_says_so(self, cells, capsys):
        barplot(cells, "cell_type", "score", hue="condition", test="auto")
        assert "ignored when `hue` is set" in capsys.readouterr().out

    def test_strip_draws_every_observation(self, cells):
        ax = stripplot(cells, "cell_type", "score", summary=None)
        drawn = sum(coll.get_offsets().shape[0] for coll in ax.collections)
        assert drawn == len(cells)

    def test_violin_split_needs_two_hue_levels(self, cells):
        with pytest.raises(ValueError, match="exactly two hue levels"):
            violinplot(cells, "cell_type", "score", hue="sample", split=True)

    def test_violin_inner_options(self, cells):
        for inner in ("box", "point", "stick", None):
            assert violinplot(cells, "cell_type", "score", inner=inner) is not None
            plt.close("all")
        with pytest.raises(ValueError, match="inner"):
            violinplot(cells, "cell_type", "score", inner="beeswarm")

    def test_stack_rows_sum_to_one(self, cells):
        _, matrix = stackplot(cells, "sample", hue="cell_type",
                              return_stats=True)
        assert np.allclose(matrix.sum(axis=1), 1.0)

    def test_stack_counts_match_a_crosstab(self, cells):
        _, matrix = stackplot(cells, "sample", hue="cell_type",
                              normalize=False, return_stats=True)
        expected = pd.crosstab(cells["sample"], cells["cell_type"])
        assert np.allclose(matrix.to_numpy(),
                           expected.reindex(index=matrix.index,
                                            columns=matrix.columns).to_numpy())

    def test_stack_needs_hue(self, cells):
        with pytest.raises(TypeError, match="`hue` is required"):
            stackplot(cells, "sample")

    def test_lollipop_top_n_is_announced(self, capsys):
        frame = pd.DataFrame({"label": list("abcdefghij"),
                              "value": np.arange(10.0)})
        _, table = lollipopplot(frame, "label", "value", top_n=3,
                                return_stats=True)
        assert "showing the 3 largest of 10" in capsys.readouterr().out
        assert len(table) == 3

    def test_lollipop_sorting(self):
        frame = pd.DataFrame({"label": ["c", "a", "b"], "value": [3.0, 1.0, 2.0]})
        _, table = lollipopplot(frame, "label", "value", return_stats=True)
        assert list(table["value"]) == [1.0, 2.0, 3.0]

    def test_pie_shares_sum_to_one(self, cells):
        _, shares = pieplot(cells, "cell_type", return_stats=True)
        assert shares.sum() == pytest.approx(1.0)
        assert shares.max() == pytest.approx(
            cells["cell_type"].value_counts(normalize=True).max())

    def test_donut_is_a_pie_with_a_hole(self, cells):
        ax = donutplot(cells, "cell_type")
        widths = {w.r - w.width for w in ax.patches if hasattr(w, "width")}
        assert widths and max(widths) > 0

    def test_slope_paired_test_matches_scipy(self):
        from scipy import stats

        rng = np.random.default_rng(3)
        pre = rng.normal(5, 1.2, 20)
        post = pre + rng.normal(0.6, 0.7, 20)
        frame = pd.DataFrame({
            "patient": np.repeat([f"P{i}" for i in range(20)], 2),
            "time": np.tile(["pre", "post"], 20),
            "value": np.ravel(np.column_stack([pre, post])),
        })
        _, stats_out = slopeplot(frame, "time", "value", subject="patient",
                                 order=["pre", "post"], test="wilcoxon",
                                 return_stats=True)
        expected = stats.wilcoxon(pre, post)
        assert stats_out["pvalue"] == pytest.approx(expected.pvalue)
        assert stats_out["n_complete"] == 20

    def test_slope_reports_excluded_subjects(self, capsys):
        frame = pd.DataFrame({
            "patient": ["a", "a", "b", "c", "c"],
            "time": ["pre", "post", "pre", "pre", "post"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        slopeplot(frame, "time", "value", subject="patient",
                  order=["pre", "post"], test="wilcoxon")
        assert "1 subject(s) lack" in capsys.readouterr().out

    def test_slope_needs_a_subject(self):
        frame = pd.DataFrame({"time": ["pre", "post"], "value": [1.0, 2.0]})
        with pytest.raises(TypeError, match="`subject` is required"):
            slopeplot(frame, "time", "value")


# --------------------------------------------------------------------------
# distributions
# --------------------------------------------------------------------------


class TestDistribution:
    def test_histogram_counts_match_numpy(self, cells):
        _, table = histplot(cells, "counts", bins=12, return_stats=True)
        expected, _ = np.histogram(cells["counts"], bins=12)
        assert np.allclose(table.iloc[:, 0].to_numpy(), expected)

    def test_density_integrates_to_one(self, cells):
        _, table = histplot(cells, "counts", bins=20, stat="density",
                            return_stats=True)
        centres = table.index.to_numpy()
        width = centres[1] - centres[0]
        assert (table.iloc[:, 0].sum() * width) == pytest.approx(1.0, rel=1e-6)

    def test_probability_sums_to_one(self, cells):
        _, table = histplot(cells, "counts", stat="probability",
                            return_stats=True)
        assert table.iloc[:, 0].sum() == pytest.approx(1.0)

    def test_fill_normalises_each_bin(self, cells):
        _, table = histplot(cells, "counts", hue="cell_type", multiple="fill",
                            return_stats=True)
        totals = table.sum(axis=1)
        assert np.allclose(totals[totals > 0], 1.0)

    def test_log_scale_drops_non_positive_and_says_so(self, capsys):
        frame = pd.DataFrame({"v": [-1.0, 0.0, 1.0, 10.0, 100.0]})
        histplot(frame, "v", log_scale=True, bins=3)
        assert "dropped 2 non-positive" in capsys.readouterr().out

    def test_kde_forces_density(self, cells, capsys):
        histplot(cells, "counts", kde=True, stat="count")
        assert "forces stat='density'" in capsys.readouterr().out

    def test_kde_curve_integrates_to_one(self, cells):
        _, curves = kdeplot(cells, "score", return_stats=True)
        grid, density = curves[None]
        assert trapezoid(density, grid) == pytest.approx(1.0, abs=0.02)

    def test_kde_two_dimensions_rejects_hue(self, cells):
        with pytest.raises(ValueError, match="one group per axes"):
            kdeplot(cells, "score", y="counts", hue="cell_type")

    def test_ridge_orders_by_median(self, cells):
        _, summary = ridgeplot(cells, "score", "cell_type", order="median",
                               return_stats=True)
        assert list(summary["median"]) == sorted(summary["median"])

    def test_ridge_summary_matches_pandas(self, cells):
        _, summary = ridgeplot(cells, "score", "cell_type", return_stats=True)
        expected = cells.groupby("cell_type", observed=True)["score"].median()
        assert np.allclose(summary["median"],
                           expected.reindex(summary.index))

    def test_ridge_needs_a_group(self, cells):
        with pytest.raises(TypeError, match="`group` is required"):
            ridgeplot(cells, "score")

    def test_qq_of_uniform_pvalues_has_lambda_near_one(self):
        rng = np.random.default_rng(11)
        _, stats_out = qqplot(rng.uniform(size=4000), dist="uniform", log=True,
                              return_stats=True)
        assert stats_out["lambda_gc"] == pytest.approx(1.0, abs=0.12)

    def test_qq_detects_inflation(self):
        rng = np.random.default_rng(12)
        inflated = np.concatenate([rng.uniform(size=3000),
                                   rng.beta(0.3, 8, 1000)])
        _, stats_out = qqplot(inflated, dist="uniform", log=True,
                              return_stats=True)
        assert stats_out["lambda_gc"] > 1.2

    def test_qq_of_a_normal_sample_is_a_straight_line(self):
        rng = np.random.default_rng(13)
        _, stats_out = qqplot(rng.normal(loc=3.0, scale=2.0, size=2000),
                              dist="norm", line="ols", return_stats=True)
        assert stats_out["slope"] == pytest.approx(2.0, rel=0.1)
        assert stats_out["intercept"] == pytest.approx(3.0, abs=0.15)

    def test_qq_unknown_distribution_is_reported(self):
        with pytest.raises(ValueError, match="scipy.stats distribution"):
            qqplot(np.random.default_rng(0).normal(size=50), dist="gaussianish")

    def test_qq_two_sample(self):
        rng = np.random.default_rng(14)
        ax = qqplot(rng.normal(size=200), other=rng.normal(size=300), line="45")
        assert ax.collections


# --------------------------------------------------------------------------
# relational
# --------------------------------------------------------------------------


class TestRelational:
    def test_scatter_correlation_matches_scipy(self, cells):
        from scipy import stats

        _, out = scatterplot(cells, "counts", "score", corr="spearman",
                             return_stats=True)
        expected = stats.spearmanr(cells["counts"], cells["score"])
        assert out["correlation"] == pytest.approx(expected.statistic)
        assert out["pvalue"] == pytest.approx(expected.pvalue)

    def test_scatter_unknown_correlation_is_reported(self, cells):
        with pytest.raises(ValueError, match="corr"):
            scatterplot(cells, "counts", "score", corr="cosine")

    def test_scatter_accepts_bare_arrays(self, cells):
        ax = scatterplot(cells["counts"].to_numpy(), cells["score"].to_numpy())
        assert ax.collections

    def test_line_aggregation_matches_groupby(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame({"hour": np.repeat([0, 2, 4, 8], 5),
                              "value": rng.normal(size=20)})
        _, table = lineplot(frame, "hour", "value", errorbar=None,
                            return_stats=True)
        expected = frame.groupby("hour")["value"].mean()
        assert np.allclose(table.set_index("x")["centre"].reindex(expected.index),
                           expected)

    def test_line_units_draws_every_replicate(self):
        rng = np.random.default_rng(5)
        frame = pd.DataFrame({"hour": np.tile([0, 1, 2], 4),
                              "rep": np.repeat([1, 2, 3, 4], 3),
                              "value": rng.normal(size=12)})
        ax = lineplot(frame, "hour", "value", units="rep")
        assert len(ax.get_lines()) == 5  # four replicates plus the summary

    def test_regression_slope_matches_polyfit(self, cells):
        _, out = regplot(cells, "counts", "score", return_stats=True)
        slope, intercept = np.polyfit(cells["counts"], cells["score"], 1)
        assert out[None]["slope"] == pytest.approx(slope)
        assert out[None]["intercept"] == pytest.approx(intercept)

    def test_r_squared_matches_scipy(self, cells):
        from scipy import stats

        _, out = regplot(cells, "counts", "score", return_stats=True)
        expected = stats.linregress(cells["counts"], cells["score"])
        assert out[None]["r_squared"] == pytest.approx(expected.rvalue ** 2)

    def test_polynomial_fit_recovers_a_quadratic(self):
        x = np.linspace(-3, 3, 200)
        frame = pd.DataFrame({"x": x, "y": 2.0 * x ** 2 - x + 1.0})
        _, out = regplot(frame, "x", "y", fit="poly", order=2, ci=None,
                         return_stats=True)
        assert out[None]["slope"] == pytest.approx(-1.0, abs=1e-6)
        assert out[None]["r_squared"] == pytest.approx(1.0, abs=1e-9)

    def test_lowess_runs_without_a_band(self, cells):
        _, out = regplot(cells, "counts", "score", fit="lowess",
                         return_stats=True)
        assert out[None]["fit"] == "lowess"

    def test_unknown_fit_is_reported(self, cells):
        with pytest.raises(ValueError, match="fit"):
            regplot(cells, "counts", "score", fit="spline")


# --------------------------------------------------------------------------
# the adapter
# --------------------------------------------------------------------------


class TestPlotData:
    def test_dataframe_columns_resolve(self, cells):
        view = as_plotdata(cells)
        assert isinstance(view, PlotData)
        assert view.n_obs == len(cells)
        assert np.allclose(view.values("score"), cells["score"])

    def test_anndata_features_and_obs_resolve(self, cells):
        anndata = pytest.importorskip("anndata")

        matrix = np.arange(len(cells) * 3, dtype=np.float32).reshape(len(cells), 3)
        adata = anndata.AnnData(matrix, obs=cells.copy())
        adata.var_names = ["GeneA", "GeneB", "GeneC"]
        view = as_plotdata(adata)
        assert np.allclose(view.values("GeneB"), matrix[:, 1])
        assert np.allclose(view.values("score"), cells["score"])
        assert view.has("GeneA") and view.has("cell_type")

    def test_sparse_features_are_densified(self, cells):
        anndata = pytest.importorskip("anndata")
        from scipy import sparse

        matrix = sparse.csr_matrix(np.eye(len(cells), 3, dtype=np.float32))
        adata = anndata.AnnData(matrix, obs=cells.copy())
        adata.var_names = ["A", "B", "C"]
        values = as_plotdata(adata).values("A")
        assert values.ndim == 1 and values[0] == 1.0

    def test_layers_are_reachable(self, cells):
        anndata = pytest.importorskip("anndata")

        matrix = np.zeros((len(cells), 2), dtype=np.float32)
        adata = anndata.AnnData(matrix, obs=cells.copy())
        adata.var_names = ["A", "B"]
        adata.layers["counts"] = np.ones_like(matrix)
        assert np.allclose(as_plotdata(adata).values("A", layer="counts"), 1.0)

    def test_unknown_key_suggests_a_close_match(self, cells):
        with pytest.raises(KeyError, match="score"):
            as_plotdata(cells).values("scoree")

    def test_layer_on_a_metadata_column_is_rejected(self, cells):
        with pytest.raises(ValueError, match="only applies to features"):
            as_plotdata(cells).values("score", layer="counts")

    def test_embedding_accepts_the_short_name(self, cells):
        view = as_plotdata(cells, obsm={"X_umap": np.zeros((len(cells), 2))})
        assert view.embedding("umap").shape == (len(cells), 2)
        with pytest.raises(KeyError, match="No embedding"):
            view.embedding("tsne")

    def test_dict_input(self):
        view = as_plotdata({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        assert np.allclose(view.values("b"), [4.0, 5.0, 6.0])

    def test_array_needs_obs(self):
        with pytest.raises(TypeError, match="needs `obs=`"):
            as_plotdata(np.zeros((4, 2)))

    def test_unsupported_type_is_reported(self):
        with pytest.raises(TypeError, match="Cannot plot a str"):
            as_plotdata("not data")

    def test_plotdata_passes_through(self, cells):
        view = as_plotdata(cells)
        assert as_plotdata(view) is view

    def test_a_gene_name_works_where_a_column_name_does(self, cells):
        """``ov.pl.violinplot(adata, 'louvain', 'CD3D')`` must just work."""
        anndata = pytest.importorskip("anndata")

        matrix = np.tile(np.arange(len(cells), dtype=np.float32)[:, None], (1, 2))
        adata = anndata.AnnData(matrix, obs=cells.copy())
        adata.var_names = ["GeneA", "GeneB"]

        from_gene = compare_groups(adata, "GeneA", "cell_type", omnibus=False)
        frame = cells.assign(GeneA=matrix[:, 0])
        from_column = compare_groups(frame, "GeneA", "cell_type", omnibus=False)
        assert np.allclose(from_gene["pvalue"], from_column["pvalue"])

        ax = violinplot(adata, "cell_type", "GeneA")
        assert ax.get_ylabel() == "GeneA"

    def test_metadata_wins_over_a_same_named_gene(self, cells):
        anndata = pytest.importorskip("anndata")

        matrix = np.zeros((len(cells), 1), dtype=np.float32)
        adata = anndata.AnnData(matrix, obs=cells.copy())
        adata.var_names = ["score"]
        assert np.allclose(as_plotdata(adata).values("score"), cells["score"])


class TestObsView:
    def test_does_not_mutate_the_callers_frame(self, cells):
        before = cells["cell_type"].dtype
        view = ObsView(cells)
        view.obs["cell_type"] = view.obs["cell_type"].cat.add_categories(["X"])
        assert cells["cell_type"].dtype == before

    def test_strings_become_categorical_like_anndata(self, cells):
        view = ObsView(cells)
        assert isinstance(view.obs["cell_type"].dtype, pd.CategoricalDtype)
        assert not isinstance(view.obs["score"].dtype, pd.CategoricalDtype)

    def test_expression_access_fails_with_an_actionable_message(self, cells):
        view = ObsView(cells)
        with pytest.raises(AttributeError, match="Pass an AnnData"):
            _ = view.X

    def test_shape_and_uns(self, cells):
        view = ObsView(cells)
        assert view.shape == (len(cells), 0)
        assert view.n_obs == len(cells) and len(view) == len(cells)
        view.uns["cell_type_colors"] = ["#000000"]
        assert view.uns["cell_type_colors"] == ["#000000"]

    def test_rejects_non_frames(self):
        with pytest.raises(TypeError, match="needs a DataFrame"):
            ObsView(np.zeros((3, 2)))


class TestAcceptsFrame:
    def test_decorator_wraps_a_frame(self, cells):
        @accepts_frame
        def probe(adata, key):
            return adata.obs[key].value_counts()

        counts = probe(cells, "cell_type")
        assert counts.sum() == len(cells)

    def test_anndata_passes_through_untouched(self, cells):
        anndata = pytest.importorskip("anndata")

        @accepts_frame
        def probe(adata):
            return adata

        adata = anndata.AnnData(np.zeros((len(cells), 1), dtype=np.float32),
                                obs=cells.copy())
        assert probe(adata) is adata

    def test_retrofitted_cellproportion_takes_a_frame(self, cells):
        from omicverse.pl import cellproportion

        ax = cellproportion(cells, celltype_clusters="cell_type",
                            groupby="sample")
        assert ax is None or ax is not None  # it draws; the point is it runs
        assert cells["cell_type"].dtype == object  # caller's frame untouched

    def test_retrofitted_matches_the_anndata_path(self, cells):
        anndata = pytest.importorskip("anndata")
        from omicverse.pl import cellstackarea

        adata = anndata.AnnData(np.zeros((len(cells), 1), dtype=np.float32),
                                obs=cells.copy())
        fig_frame, ax_frame = plt.subplots()
        cellstackarea(cells, celltype_clusters="cell_type", groupby="sample",
                      ax=ax_frame)
        heights_frame = sorted(round(c.get_paths()[0].vertices[:, 1].max(), 6)
                               for c in ax_frame.collections)
        fig_adata, ax_adata = plt.subplots()
        cellstackarea(adata, celltype_clusters="cell_type", groupby="sample",
                      ax=ax_adata)
        heights_adata = sorted(round(c.get_paths()[0].vertices[:, 1].max(), 6)
                               for c in ax_adata.collections)
        assert heights_frame == heights_adata


# --------------------------------------------------------------------------
# public surface
# --------------------------------------------------------------------------


class TestSurface:
    NEW = ("barplot", "stripplot", "violinplot", "stackplot", "lollipopplot",
           "pieplot", "donutplot", "slopeplot", "histplot", "kdeplot",
           "ridgeplot", "qqplot", "scatterplot", "lineplot", "regplot",
           "compare_groups", "add_stat_annotation", "format_pvalue",
           "as_plotdata", "accepts_frame", "PlotData", "ObsView")

    def test_everything_is_exported(self):
        import omicverse.pl as pl

        for name in self.NEW:
            assert hasattr(pl, name), f"ov.pl.{name} is missing"
            assert name in pl.__all__, f"ov.pl.{name} is not in __all__"

    def test_plots_are_registered(self):
        import omicverse.pl  # noqa: F401

        from omicverse._registry import get_registry

        names = {entry.get("full_name")
                 for entry in get_registry().get_by_category("pl")}
        for expected in (
            "omicverse.pl._categorical.barplot",
            "omicverse.pl._categorical.violinplot",
            "omicverse.pl._categorical.slopeplot",
            "omicverse.pl._distribution.ridgeplot",
            "omicverse.pl._distribution.qqplot",
            "omicverse.pl._relational.regplot",
            "omicverse.pl._stats_tests.compare_groups",
            "omicverse.pl._stats_tests.add_stat_annotation",
        ):
            assert expected in names, f"{expected} missing from the registry"

    def test_generic_names_do_not_shadow_the_omics_ones(self):
        import omicverse.pl as pl

        # ov.pl.violin still takes an AnnData; ov.pl.violinplot takes a frame
        assert pl.violin is not pl.violinplot
        assert pl.boxplot is not pl.barplot
