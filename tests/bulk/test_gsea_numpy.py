"""Tests for the pure-NumPy pre-ranked GSEA backend (omicverse.bulk._gsea_numpy).

Validates correctness (enrichment score matches the textbook walk), detection
of planted enrichment, determinism, the gseapy-compatible result object, and
that ov.bulk.pyGSEA defaults to this single-process backend (no loky/deadlock).
"""
import numpy as np
import pandas as pd
import pytest


def _planted_data(n=4000, seed=0):
    """Ranked list of n genes + gene sets: 10 enriched-at-top, 40 random."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n)]
    scores = np.sort(rng.normal(size=n))[::-1]
    rnk = pd.DataFrame({"gene": genes, "score": scores})
    gs = {}
    for s in range(10):
        top = rng.choice(np.arange(0, 600), size=40, replace=False)
        gs[f"ENRICH_{s}"] = [genes[i] for i in top]
    for s in range(40):
        idx = rng.choice(n, size=int(rng.integers(20, 100)), replace=False)
        gs[f"RANDOM_{s}"] = [genes[i] for i in idx]
    return rnk, gs


def _es_reference(ranked_genes, scores, members, weight=1.0):
    """Textbook weighted running-ES (Subramanian 2005) for one gene set."""
    n = len(ranked_genes)
    member = set(members)
    in_set = np.array([g in member for g in ranked_genes])
    k = int(in_set.sum())
    w = np.abs(scores) ** weight
    n_r = w[in_set].sum()
    inc = np.where(in_set, w / n_r, -1.0 / (n - k))
    res = np.cumsum(inc)
    return res[np.argmax(np.abs(res))]


def test_es_matches_reference_walk():
    from omicverse.bulk._gsea_numpy import prerank

    rnk, gs = _planted_data()
    r = prerank(rnk, gs, permutation_num=200, seed=1, min_size=15, max_size=500)
    ranked = r.ranking.index.to_numpy()
    scores = r.ranking.to_numpy()
    for term in list(r.res2d.index)[:8]:
        ref = _es_reference(ranked, scores, gs[term])
        assert abs(r.res2d.loc[term, "es"] - ref) < 1e-9, term


def test_detects_planted_enrichment():
    from omicverse.bulk._gsea_numpy import prerank

    rnk, gs = _planted_data()
    r = prerank(rnk, gs, permutation_num=500, seed=1)
    top10 = r.res2d.sort_values("nes", ascending=False).head(10).index.tolist()
    assert sum("ENRICH_" in t for t in top10) >= 9  # nearly all planted on top
    # planted sets are significant; random ones largely not
    sig = r.res2d[r.res2d["fdr"] < 0.25].index
    assert sum("ENRICH_" in t for t in sig) >= 9


def test_deterministic_for_seed():
    from omicverse.bulk._gsea_numpy import prerank

    rnk, gs = _planted_data()
    a = prerank(rnk, gs, permutation_num=300, seed=7)
    b = prerank(rnk, gs, permutation_num=300, seed=7)
    assert np.allclose(a.res2d["nes"].values, b.res2d["nes"].values)
    assert np.array_equal(a.res2d["es"].values, b.res2d["es"].values)


def test_result_object_is_gseapy_compatible():
    from omicverse.bulk._gsea_numpy import prerank

    rnk, gs = _planted_data()
    r = prerank(rnk, gs, permutation_num=100, seed=1)
    assert isinstance(r.ranking, pd.Series)
    for col in ["es", "nes", "pval", "fdr", "matched_size", "geneset_size"]:
        assert col in r.res2d.columns
    term = r.res2d.index[0]
    d = r.results[term]
    for key in ["es", "nes", "pval", "fdr", "hit_indices", "RES"]:
        assert key in d
    assert len(d["RES"]) == len(r.ranking)


def test_pygsea_defaults_to_numpy_backend():
    import omicverse as ov

    rnk, gs = _planted_data()
    g = ov.bulk.pyGSEA(rnk, gs)
    assert g.backend == "numpy" and g.processes == 1
    res = g.enrichment(pval=0.25)
    assert len(res) >= 9
    assert {"fdr", "nes", "Term"}.issubset(res.columns)
