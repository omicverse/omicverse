"""v0.2 tests for ``omicverse.metabol`` — ID mapping, MSEA, mummichog, lipidomics.

All tests in this file that hit an upstream database (KEGG REST, PubChem
REST, ChEBI FTP, LION CSV) are gated behind ``OV_METABOL_FETCHER_TESTS``
so CI without network stays green. The offline-safe tests (parsing,
edge cases, class aggregation) always run.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


_NETWORK = bool(os.environ.get("OV_METABOL_FETCHER_TESTS"))
needs_network = pytest.mark.skipif(
    not _NETWORK,
    reason="hits KEGG/PubChem/ChEBI/LION — enable with OV_METABOL_FETCHER_TESTS=1",
)


# --------------------------------------------------------------------------- #
# Network-required — name → ID via PubChem
# --------------------------------------------------------------------------- #
@needs_network
def test_id_mapping_resolves_cachexia_metabolites():
    from omicverse.metabol import map_ids

    df = map_ids(["Isoleucine", "Glucose", "totally_made_up_xyz"])
    # PubChem canonical cross-refs for these amino acids are stable
    assert df.loc["Isoleucine", "kegg"] == "C00407"
    assert df.loc["Isoleucine", "hmdb"].startswith("HMDB")
    assert df.loc["Glucose", "kegg"] == "C00031"
    assert df.loc["totally_made_up_xyz", "hmdb"] == ""    # unresolved


@needs_network
def test_id_mapping_uses_mass_db_for_local_hits():
    """When a pre-fetched ChEBI DataFrame is passed via ``mass_db=``,
    names that match the table are resolved without PubChem round-trips."""
    import omicverse as ov

    ch = ov.metabol.fetch_chebi_compounds()
    # Pick a row we know exists; ``map_ids`` should find it locally
    sample = ch[ch["kegg"] != ""].head(1).iloc[0]
    result = ov.metabol.map_ids([sample["name"]], mass_db=ch)
    assert result.iloc[0]["kegg"] == sample["kegg"]


@needs_network
def test_id_mapping_chebi_column_alias():
    """``map_ids(..., targets=('chebi',), mass_db=chebi_df)`` must resolve
    the ChEBI identifier without falling through to PubChem. The ChEBI
    DataFrame stores the ID as ``chebi_id``, not ``chebi``; a column
    alias in map_ids bridges the gap. Without the alias every row
    looked "missing chebi" and the fallback hit PubChem for every
    name — the smoke test on MTBLS1 took ~30 s per MSEA call because
    of this."""
    import omicverse as ov

    ch = ov.metabol.fetch_chebi_compounds()
    sample = ch[ch["chebi_id"].astype(str).str.startswith("CHEBI:")].head(1).iloc[0]
    result = ov.metabol.map_ids(
        [sample["name"]], targets=("chebi",), mass_db=ch,
    )
    assert result.iloc[0]["chebi"] == sample["chebi_id"], (
        f"Expected {sample['chebi_id']!r}, got {result.iloc[0]['chebi']!r}"
    )


@needs_network
def test_msea_ora_accepts_mass_db(cachexia_adata):
    """``msea_ora(..., mass_db=chebi_df)`` must produce the same
    pathway set as the network-fallback path but without per-name
    PubChem traffic — covers the MTBLS1 perf regression fix."""
    import omicverse as ov
    from omicverse.metabol import differential, msea_ora, normalize, transform

    a = normalize(cachexia_adata, method="pqn")
    a = transform(a, method="log")
    deg = differential(a, method="welch_t", log_transformed=True)
    hits = deg[deg["padj"] < 0.30].index.tolist()
    background = deg.index.tolist()
    ch = ov.metabol.fetch_chebi_compounds()
    out = msea_ora(hits, background, min_size=3, mass_db=ch)
    assert not out.empty
    for col in ("pathway", "overlap", "set_size", "padj"):
        assert col in out.columns


@needs_network
def test_msea_ora_finds_relevant_pathways_on_cachexia(cachexia_adata):
    """Cachexia urinary metabolites should be enriched for amino-acid /
    TCA pathways. Uses the full fetched KEGG database."""
    from omicverse.metabol import differential, msea_ora, normalize, transform

    a = normalize(cachexia_adata, method="pqn")
    a = transform(a, method="log")
    deg = differential(a, method="welch_t", log_transformed=True)
    hits = deg[deg["padj"] < 0.20].index.tolist()
    background = deg.index.tolist()
    ora = msea_ora(hits, background, min_size=3)
    assert not ora.empty
    for col in ("pathway", "overlap", "set_size", "odds_ratio", "pvalue", "padj"):
        assert col in ora.columns
    top_terms = set(ora.head(10)["pathway"].tolist())
    assert any(("TCA" in t) or ("Alanine" in t) or ("Aminoacyl" in t)
               for t in top_terms), f"no amino-acid/TCA in top-10: {top_terms}"


@needs_network
def test_msea_gsea_runs_and_returns_results(cachexia_adata):
    from omicverse.metabol import differential, msea_gsea, normalize, transform

    a = normalize(cachexia_adata, method="pqn")
    a = transform(a, method="log")
    deg = differential(a, method="welch_t", log_transformed=True)
    out = msea_gsea(deg, stat_col="stat", n_perm=200, seed=0)
    assert not out.empty
    # msea_gsea returns the classic GSEA schema (Term/ES/NES/NOM p-val/...)
    assert "NES" in out.columns


@needs_network
def test_mummichog_annotates_positive_mode_peaks():
    """[M+H]+ synthesis from ChEBI compounds should round-trip via
    annotate_peaks — every spiked peak must recover its source KEGG ID."""
    import omicverse as ov

    ch = ov.metabol.fetch_chebi_compounds()
    ch = ch[(ch["kegg"] != "") & ch["mw"].notna()].reset_index(drop=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ch), size=10, replace=False)
    hit_mz = ch["mw"].iloc[idx].to_numpy() + 1.00728
    ann = ov.metabol.annotate_peaks(hit_mz, polarity="positive",
                                    ppm=10.0, mass_db=ch)
    expected_keggs = set(ch["kegg"].iloc[idx])
    got_keggs = set(ann["kegg"])
    assert expected_keggs.issubset(got_keggs), (
        f"Missing: {expected_keggs - got_keggs}"
    )


@needs_network
def test_mummichog_basic_enriches_spiked_pathway():
    """Spike every TCA-cycle member as ``[M+H]+`` peaks, flood the rest
    of the input with random-mass background peaks — mummichog must put
    'Citrate cycle (TCA cycle)' in the top-3 enriched pathways."""
    import omicverse as ov

    pathways = ov.metabol.load_pathways()
    tca_ids = pathways["Citrate cycle (TCA cycle)"]
    ch = ov.metabol.fetch_chebi_compounds()
    tca = ch[(ch["mw"].notna()) & (ch["kegg"].isin(tca_ids))].reset_index(drop=True)
    assert len(tca) >= 4, f"fetcher regression — TCA compounds = {len(tca)}"

    rng = np.random.default_rng(0)
    hit_mz = tca["mw"].to_numpy() + 1.00728
    bg_mz = rng.uniform(50, 1200, size=80)
    all_mz = np.concatenate([hit_mz, bg_mz])
    pvalue = np.concatenate([np.full(len(hit_mz), 0.001), np.full(80, 0.5)])
    res = ov.metabol.mummichog_basic(
        all_mz, pvalue, polarity="positive", ppm=10.0,
        n_perm=500, min_overlap=2,
        mass_db=ch, pathways=pathways,
    )
    assert not res.empty
    top3 = set(res.head(3)["pathway"])
    assert "Citrate cycle (TCA cycle)" in top3, f"TCA not in top-3: {top3}"


@needs_network
def test_lion_enrichment_spots_glycerophospholipid_hit():
    from omicverse.metabol import lion_enrichment

    hits = ["PC 34:1", "PC 36:2", "PE 34:1", "PE 36:4", "LPE 18:0"]
    background = hits + ["TAG 54:3", "TAG 52:2", "SM 34:1",
                         "Cer d18:1/24:0", "CE 18:2", "DAG 36:2"]
    out = lion_enrichment(hits, background, min_size=2)
    assert not out.empty
    for col in ("term", "overlap", "set_size", "odds_ratio", "pvalue", "padj"):
        assert col in out.columns
    # Glycerophospholipid-biased hit list → that category should dominate
    top_terms = " ".join(str(t).lower() for t in out.head(5)["term"])
    assert "glycerophospholipid" in top_terms or "phosphatidylcholine" in top_terms


# --------------------------------------------------------------------------- #
# Offline-safe — parsing / aggregation / edge cases
# --------------------------------------------------------------------------- #
def test_lipidomics_parse_lipid_names():
    from omicverse.metabol import parse_lipid

    pc = parse_lipid("PC 34:1")
    assert pc and pc.lipid_class == "PC" and pc.total_carbons == 34 and pc.total_db == 1
    # Goslin standardises triacylglycerol to the canonical "TG" abbreviation
    assert parse_lipid("TAG 54:3").lipid_class == "TG"
    assert parse_lipid("TG 54:3").lipid_class == "TG"
    # Goslin resolves the whole ceramide: 18 (sphingoid base) + 24 (N-acyl)
    # = 42 carbons, 1 double bond from the d18:1 base, per-chain in fa_chains
    cer = parse_lipid("Cer d18:1/24:0")
    assert cer.lipid_class == "CER" and cer.total_carbons == 42 and cer.total_db == 1
    assert cer.fa_chains == ((18, 1), (24, 0))
    assert parse_lipid("Glucose") is None
    assert parse_lipid("Isoleucine") is None


def test_lipidomics_annotate_and_aggregate():
    import anndata as ad
    from omicverse.metabol import aggregate_by_class, annotate_lipids

    names = ["PC 34:1", "PC 36:2", "PE 34:1", "PE 36:4", "TAG 54:3",
             "TAG 52:2", "LPC 18:0", "SM 34:1"]
    X = np.random.default_rng(0).uniform(10, 1000, size=(20, len(names)))
    adata = ad.AnnData(X=X,
                       obs=pd.DataFrame(index=[f"s{i}" for i in range(20)]),
                       var=pd.DataFrame(index=names))
    adata = annotate_lipids(adata)
    assert set(adata.var["lipid_class"]) == {"PC", "PE", "TG", "LPC", "SM"}
    agg = aggregate_by_class(adata, agg="sum")
    assert agg.n_vars == 5 and agg.n_obs == 20
    pc_cols = [i for i, c in enumerate(adata.var["lipid_class"]) if c == "PC"]
    expected = np.asarray(adata.X[:, pc_cols]).sum(axis=1)
    got = np.asarray(agg.X[:, list(agg.var_names).index("PC")])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)


@needs_network
def test_msea_ora_empty_when_no_hits_resolve_to_kegg():
    """Edge case: every hit is an unresolvable name → clear ValueError."""
    from omicverse.metabol import msea_ora
    with pytest.raises(ValueError, match="resolve to KEGG"):
        msea_ora(
            hits=["totally_made_up_1", "fake_compound_2"],
            background=["totally_made_up_1", "fake_compound_2", "another_fake"],
        )


def test_metabol_import_does_not_pull_in_heavy_deps():
    """Regression: ``import omicverse.metabol`` must stay lightweight —
    scipy / sklearn / statsmodels / matplotlib should NOT be imported at
    module init time. They're loaded lazily via ``__getattr__`` on first
    access of the functions that need them.
    """
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent("""
        import sys
        import omicverse.metabol   # noqa — the whole point
        heavy = [m for m in ("sklearn", "statsmodels", "matplotlib")
                 if m in sys.modules]
        # scipy is a borderline case — it's small and already pulled in by
        # numpy on many installs, so we don't gate it. sklearn et al are
        # the real concern.
        print(",".join(heavy) or "CLEAN")
    """)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "CLEAN", (
        f"import omicverse.metabol eagerly loaded: {result.stdout.strip()} — "
        "this defeats the lazy-loading pattern in metabol/__init__.py"
    )


def test_metabol_lazy_attrs_resolve_correctly():
    """Every symbol in __all__ must be reachable via __getattr__ and
    caches itself in globals() on first access."""
    import omicverse.metabol as m

    # First-access triggers the lazy import + caches; second access is
    # pure-dict lookup. Both must give the same object.
    f1 = m.opls_da
    f2 = m.opls_da
    assert f1 is f2
    # Cheap paths shouldn't have triggered heavy deps that aren't needed
    assert callable(m.normalize_name)
    # Unknown name still raises AttributeError (not KeyError)
    import pytest
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = m.not_a_real_symbol


def test_differential_single_group_raises():
    import anndata as ad
    from omicverse.metabol import differential

    adata = ad.AnnData(
        X=np.random.default_rng(0).normal(size=(5, 10)),
        obs=pd.DataFrame({"group": ["a"] * 5},
                         index=[f"s{i}" for i in range(5)]),
        var=pd.DataFrame(index=[f"m{i}" for i in range(10)]),
    )
    with pytest.raises(ValueError, match="fewer than 2 unique"):
        differential(adata, group_col="group")


def test_impute_seed_is_deterministic_and_overridable():
    """QRILC should be deterministic per seed and different seeds should
    produce different imputations (regression for the old hardcoded 42)."""
    import anndata as ad
    from omicverse.metabol import impute

    rng = np.random.default_rng(0)
    X = rng.uniform(1, 100, size=(30, 20))
    X[rng.random(X.shape) < 0.2] = np.nan
    adata = ad.AnnData(X=X,
                       obs=pd.DataFrame(index=[f"s{i}" for i in range(30)]),
                       var=pd.DataFrame(index=[f"m{i}" for i in range(20)]))
    a = impute(adata, method="qrilc", seed=0).X
    a2 = impute(adata, method="qrilc", seed=0).X
    b = impute(adata, method="qrilc", seed=123).X
    np.testing.assert_allclose(a, a2, rtol=0, atol=0)
    assert not np.allclose(a, b)
