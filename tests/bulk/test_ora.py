"""Tests for the pure-NumPy/SciPy over-representation analysis backend
(omicverse.bulk._ora) that replaced gseapy's local enrichr mode.

Validates the hypergeometric statistic against an independent scipy
computation, the Benjamini-Hochberg correction, deterministic output, the
gseapy-compatible result object, and that ``ov.bulk.geneset_enrichment``
(dict input, offline) routes through it.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    universe = [f"G{i}" for i in range(2000)]
    gene_sets = {
        f"T{t}": list(rng.choice(universe, size=int(rng.integers(15, 120)),
                                 replace=False))
        for t in range(50)
    }
    query = set(rng.choice(universe, size=200, replace=False))
    # plant a strong hit in T0
    gene_sets["T0"] = list(query)[:40] + list(rng.choice(universe, size=20,
                                                         replace=False))
    return universe, gene_sets, query


def test_hypergeom_matches_scipy_reference():
    from omicverse.bulk._ora import _calc_term_stats

    universe, gene_sets, query = _synthetic()
    bg = set(universe)
    terms, pvals, oddr, olsz, gssz, hits = _calc_term_stats(set(query), gene_sets, bg)
    # independent recomputation
    for t, p, x, m in zip(terms, pvals, olsz, gssz):
        ref = hypergeom.sf(x - 1, len(bg), m, len(query))
        assert abs(p - ref) < 1e-12, t


def test_bh_fdr_is_monotone_and_bounded():
    from omicverse.bulk._ora import _bh_fdr

    p = np.array([1e-8, 1e-3, 0.02, 0.2, 0.9])
    q = _bh_fdr(p)
    assert np.all(q >= p - 1e-12)          # adjusted >= raw
    assert np.all(q <= 1.0)
    # monotone in the p-value order
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_enrichr_detects_planted_set_and_is_deterministic():
    from omicverse.bulk._ora import enrichr

    universe, gene_sets, query = _synthetic()
    a = enrichr(list(query), gene_sets, background=universe)
    b = enrichr(list(query), gene_sets, background=universe)
    assert a is not None
    # planted T0 should be the most significant
    assert a.res2d.sort_values("P-value").iloc[0]["Term"] == "T0"
    assert np.array_equal(a.res2d["P-value"].values, b.res2d["P-value"].values)


def test_result_object_is_gseapy_compatible():
    from omicverse.bulk._ora import enrichr

    universe, gene_sets, query = _synthetic()
    enr = enrichr(list(query), gene_sets, background=universe)
    for col in ["Gene_set", "Term", "Overlap", "P-value", "Adjusted P-value",
                "Odds Ratio", "Combined Score", "Genes"]:
        assert col in enr.res2d.columns
    # .results mirrors .res2d (callers use either)
    assert enr.results is enr.res2d
    # Overlap is "hits/size" parseable
    h, n = enr.res2d.iloc[0]["Overlap"].split("/")
    assert int(h) >= 1 and int(n) >= int(h)


def test_geneset_enrichment_uses_local_ora():
    import omicverse as ov

    universe, gene_sets, query = _synthetic()
    enr = ov.bulk.geneset_enrichment(list(query), gene_sets,
                                     pvalue_threshold=0.5, pvalue_type="raw")
    assert "logp" in enr.columns and "fraction" in enr.columns
    assert (enr["P-value"] < 0.5).all()


def test_geneset_plot_multi_returns_marsilea():
    import matplotlib
    matplotlib.use("Agg")
    import omicverse as ov
    import marsilea as ma

    universe, gene_sets, query1 = _synthetic(seed=0)
    _, _, query2 = _synthetic(seed=5)
    e1 = ov.bulk.geneset_enrichment(list(query1), gene_sets,
                                    pvalue_threshold=1.0, pvalue_type="raw")
    e2 = ov.bulk.geneset_enrichment(list(query2), gene_sets,
                                    pvalue_threshold=1.0, pvalue_type="raw")
    h = ov.bulk.geneset_plot_multi({"A": e1, "B": e2},
                                   {"A": "#1f77b4", "B": "#ff7f0e"}, num=5)
    # PyComplexHeatmap was dropped here — the panel is now a Marsilea board
    assert isinstance(h, ma.SizedHeatmap)
    assert h.figure is not None
