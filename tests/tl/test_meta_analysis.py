"""Tests for omicverse.tl meta-analysis extraction and effect sizes (synthetic articles only)."""
import math

import numpy as np
import pandas as pd

from omicverse.tl._meta_analysis import (extract_group_stats, group_comparisons, hedges_g,
                                         meta_effect_sizes, parse_mean_sd, pool_group_stats,
                                         recompute_group_test, sd_from_summary)


def test_parse_cells():
    assert parse_mean_sd("12.3 (4.5)") == (12.3, 4.5)
    assert parse_mean_sd("16.20(S.D.: 0.78)") == (16.2, 0.78)
    assert parse_mean_sd("−1.5 ± 0.25") == (-1.5, 0.25)
    assert parse_mean_sd("1,064 (225)") == (1064.0, 225.0)
    assert parse_mean_sd("19:9") is None


def test_pool_matches_raw_data():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 17), rng.normal(13, 3, 23)
    n, m, s = pool_group_stats([17, 23], [a.mean(), b.mean()], [a.std(ddof=1), b.std(ddof=1)])
    allv = np.r_[a, b]
    assert n == 40 and math.isclose(m, allv.mean()) and math.isclose(s, allv.std(ddof=1))


def test_hedges_formula():
    r = hedges_g(20, 17.12, 5.02, 20, 20.1, 1.93).iloc[0]
    sp = math.sqrt((19 * 5.02 ** 2 + 19 * 1.93 ** 2) / 38)
    assert math.isclose(r["g"], (17.12 - 20.1) / sp * (1 - 3 / (4 * 40 - 9)))
    assert hedges_g(20, 1, 1, 20, 0, 1, correction="exact")["g"].iloc[0] < 1


def test_sd_from_se_and_ci():
    assert math.isclose(sd_from_summary(25, se=2.0), 10.0)
    assert math.isclose(sd_from_summary(400, ci=(9.0, 11.0), use_t=False), 20 * 2.0 / (2 * 1.959963984540054))


_JATS = """<article><body><p>Go RT was longer (Patients A 371.9±33.1, Patients B 356.8±37.5, HC 372.9±20.7).</p>
<table-wrap><label>Table 2</label><caption>Task performance</caption><table>
<thead><tr><th/><th colspan="2">Controls (n = 14)</th><th colspan="2">Depressed (n = 30)</th></tr>
<tr><th>Measure</th><th>M</th><th>SD</th><th>M</th><th>SD</th></tr></thead>
<tbody><tr><td>Switching task</td><td/><td/><td/><td/></tr>
<tr><td>Part A time (s)</td><td>47.3</td><td>4.0</td><td>73.5</td><td>18.1</td></tr>
<tr><td>Part B time (s)</td><td>69.8</td><td>5.7</td><td>146.1</td><td>65.7</td></tr>
</tbody></table></table-wrap>
<table-wrap><label>Table 3</label><table><tr><th/><th>MDD-1 n=10</th><th>MDD-2 n=12</th><th>HC n=20</th></tr>
<tr><td>Errors</td><td>4.0 (2.0)</td><td>6.0 (3.0)</td><td>2.0 (1.0)</td></tr></table></table-wrap>
</body></article>"""


def test_extract_and_compare():
    led = extract_group_stats(_JATS, kind="xml")
    tab = led[led["layout"] != "text"]
    assert len(tab) == 7
    row = tab[(tab["row"] == "Switching task: Part B time (s)") & tab["group"].str.startswith("Depressed")].iloc[0]
    assert (row["mean"], row["sd"], row["n_header"]) == (146.1, 65.7, 30)
    txt = led[led["layout"] == "text"]
    assert set(txt["group"]) >= {"Patients A", "Patients B", "HC"}
    cmp_ = group_comparisons(tab, case=r"depress|MDD", control=r"control|HC")
    err = cmp_[cmp_["row"] == "Errors"].iloc[0]
    n, m, s = pool_group_stats([10, 12], [4, 6], [2, 3])
    assert err["n_case"] == 22 and math.isclose(err["mean_case"], m) and math.isclose(err["sd_case"], s)
    assert len(cmp_) == 3


def test_ledger_effects_and_checks():
    df = pd.DataFrame({"doi": ["x", "x", "x"],
                       "n_case": [[15, 15], 20, 20], "mean_case": [[4.87, 4.13], 3.0, 3.0],
                       "sd_case": [[3.31, 4.36], 0.0, 0.0],
                       "n_control": [14, 20, 20], "mean_control": [3.35, 1.0, 1.0], "sd_control": [1.95, 1.0, 1.0]})
    out = meta_effect_sizes(df)
    assert out.loc[0, "n_case"] == 30 and math.isclose(out.loc[0, "mean_case"], 4.5)
    assert out.loc[0, "pooled_from"] == 2 and out.loc[0, "issues"] == ""
    assert "zero variance" in out.loc[1, "issues"] and "duplicate" in out.loc[1, "issues"]


def test_recompute_check():
    r = recompute_group_test(20, 146.07, 65.66, 20, 69.75, 5.72, reported_F=26.8)
    assert r["matches"] and r["df"] == 38


def test_apply_selection_rules():
    from omicverse.tl._meta_analysis import apply_selection_rules
    led = pd.DataFrame({
        "study": ["a", "a", "a", "b", "b", "a"],
        "family": ["sorting", "sorting", "fluency", "sorting", "fluency", "memory"],
        "stat_class": ["errors", "categories", "Phonemic count ", "categories", "other variant", "recall"]})
    rules = {"sorting": {"select": ["errors"], "fallback": {"categories": ["errors"]}},
             "fluency": {"select": ["phonemic count", "semantic count"]}}
    d = apply_selection_rules(led, rules)
    assert list(d["decision"]) == ["keep", "exclude", "keep", "keep", "exclude", "exclude"]


def test_comparison_workbook(tmp_path):
    from omicverse.tl._meta_analysis import comparison_workbook
    p = tmp_path / "a.xml"
    p.write_text(_JATS)
    wb = comparison_workbook({"s1": str(p)}, case=r"depress|MDD|patients")
    tab = wb[wb["layout"] != "text"]
    assert {"family", "stat_class"} <= set(wb.columns)
    assert set(tab["row"]) == {"Switching task: Part A time (s)", "Switching task: Part B time (s)", "Errors"}
    err = tab[tab["row"] == "Errors"].iloc[0]
    assert err["n_case"] == 22 and err["n_control"] == 20
    # in-text comparison is kept with unknown n so it can be completed by hand
    txt = wb[wb["layout"] == "text"]
    assert len(txt) == 1 and np.isnan(txt["n_case"].iloc[0])
