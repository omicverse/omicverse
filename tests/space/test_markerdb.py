"""Offline test of markerdb_signatures: every web service is replaced by a
fake that answers from small in-memory tables (names are arbitrary)."""
import importlib.util
import sys
import types
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "omicverse" / "space" / "_markerdb.py"


def _load():
    try:
        from omicverse.space import _markerdb as m
        return m
    except Exception:  # light-weight import when the full package is unavailable
        pass
    reg = types.ModuleType("omicverse._registry")
    reg.register_function = lambda **k: (lambda f: f)
    sys.modules.setdefault("omicverse", types.ModuleType("omicverse"))
    sys.modules["omicverse._registry"] = reg
    sys.modules.setdefault("omicverse.space", types.ModuleType("omicverse.space"))
    spec = importlib.util.spec_from_file_location("omicverse.space._mdb", _P)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mdb = _load()

ATLAS = [{"cell_name": "alpha cell", "cl_id": "CL:1"},
         {"cell_name": "beta-positive alpha cell", "cl_id": "CL:2"},
         {"cell_name": "gamma cell", "cl_id": "CL:3"}]
TEMPLATES = {
    "alpha cell": {"positive": [("GA", 9)], "negative": [("GG", 3)]},
    "beta-positive alpha cell": {"positive": [("GA", 5), ("GB", 7)], "negative": []},
    "gamma cell": {"positive": [("GG", 8)], "negative": [("GA", 2)]},
}
GENES = {"MA": ["GA"], "MB": ["GB"], "MG": ["GG"]}


def fake_get(url, params=None, headers=None, timeout=30):
    if url.endswith("/api/markers/gene_celltype_counts"):
        return {"counts": {"GA": {"positive": 2, "negative": 1}, "GB": {"positive": 1},
                           "GG": {"positive": 1, "negative": 1}}}
    if url.endswith("/api/markers/celltypes"):
        if params and params.get("tissue") == "organ x":
            return {"celltypes": [{"cell_name": "alpha cell", "cl_id": "CL:1"}]}
        return {"celltypes": ATLAS}
    if url.endswith("/api/markers/celltype_markers"):
        assert headers and headers.get("Authorization") == "Bearer T"
        names = params["cell_names"].split("|")
        return {"templates": [
            {"cell_name": n, "cl_id": None,
             "positive": [{"gene": g, "n_pmid": k} for g, k in TEMPLATES[n]["positive"]],
             "negative": [{"gene": g, "n_pmid": k} for g, k in TEMPLATES[n]["negative"]]}
            for n in names]}
    if "/fetch/symbol/" in url:
        q = url.rsplit("/", 1)[1]
        return {"response": {"docs": [{"symbol": g} for g in GENES.get(q, [])]}}
    if "ols4/api/search" in url:
        return {"response": {"docs": []}}
    if "hierarchicalParents" in url:
        par = {"CL_2": [{"obo_id": "CL:1"}], "CL_4": [{"obo_id": "CL:1"}], "CL_5": [{"obo_id": "CL:1"}]}
        key = next((k for k in par if k in url), "")
        return {"_embedded": {"terms": par.get(key, [])}, "_links": {}}
    if url.endswith("/api/markers/tissues"):
        return {"tissues": [{"tissue_name": "organ x"}, {"tissue_name": "organ y"}]}
    if url.endswith("/api/markers/diseases"):
        return {"diseases": [{"disease": "disease z"}]}
    return {"response": {"docs": []}, "hits": []}


def test_signatures_parents_and_diagnostics(monkeypatch=None):
    mdb._get = fake_get
    kb = mdb.markerdb_signatures(
        ["alpha cell", "beta+ alpha cell", "gamma cell", "delta cell"],
        ["MA", "MB", "MG", "DNA"], token="T", server_url="http://x")
    sig = kb["signatures"]
    assert sig["alpha cell"]["MA"] > 0 and sig["alpha cell"]["MG"] < 0
    assert sig["gamma cell"]["MG"] > 0 and sig["gamma cell"]["MA"] < 0
    assert sig["beta+ alpha cell"]["MB"] > 0
    assert kb["parents"] == {"beta+ alpha cell": "alpha cell"}
    assert kb["unresolved_labels"] == ["delta cell"]
    assert kb["unresolved_markers"] == ["DNA"]
    # relative support: GB (7 papers) outweighs GA (5) for the beta+ alpha cell
    assert sig["beta+ alpha cell"]["MB"] == 1.0
    assert sig["beta+ alpha cell"]["MB"] > sig["beta+ alpha cell"]["MA"]
    # inheritance: the alpha cell's negative marker reaches its CL child, halved
    assert -0.2 < sig["beta+ alpha cell"]["MG"] < 0
    assert kb["defining"]["alpha cell"] == ["MA"]


def test_descriptions_weighting_and_weak():
    mdb._get = fake_get
    kb = mdb.markerdb_signatures(
        ["L1", "gamma cell", "beta+ alpha cell", "alpha cell"], ["MA", "MB", "MG"],
        token="T", server_url="http://x",
        descriptions={"L1": "L1 covers alpha cells, things and other stuff"})
    assert [t["cell_name"] for t in kb["resolved"]["L1"]["terms"]] == ["alpha cell"]
    assert kb["resolved"]["L1"]["from_description"] == ["alpha"]
    # share weighting: GA (5 papers) is weaker than GB (7) within beta+ alpha cell
    s = kb["signatures"]["beta+ alpha cell"]
    assert 0 < s["MA"] < s["MB"]
    lit = mdb.markerdb_signatures(["beta+ alpha cell", "gamma cell", "alpha cell"], ["MA", "MB", "MG"],
                                  token="T", server_url="http://x", weighting="literature")
    assert lit["signatures"]["beta+ alpha cell"]["MB"] > 0
    assert set(kb["weak_signatures"]) <= set(kb["signatures"])
    assert kb["gene_map_source"]["MA"] == "symbol"


def test_gene_overrides_and_pick():
    mdb._get = fake_get
    kb = mdb.markerdb_signatures(["alpha cell", "gamma cell"], ["MA", "MG", "DNA"], token="T",
                                 server_url="http://x", gene_overrides={"DNA": ["GG"]})
    assert kb["gene_map"]["DNA"] == ["GG"] and kb["gene_map_source"]["DNA"] == "override"
    assert "DNA" not in kb["unresolved_markers"]
    # paralogues used about equally are both kept; a dominant one wins alone
    assert mdb._pick(["X1", "X2", "X3"], {"X1": 10, "X2": 7, "X3": 1}) == ["X1", "X2"]
    assert mdb._pick(["X1", "X2"], {"X1": 10, "X2": 3}) == ["X1"]
    assert mdb._desc_phrases("Stroma", "Stroma covers fibroblasts, vessels and matrix") == \
        ["fibroblasts", "vessels", "matrix"]


def test_missing_token_raises():
    mdb._get = fake_get
    import os
    os.environ.pop("OMICOS_USER_TOKEN", None)
    try:
        mdb.markerdb_signatures(["alpha cell"], ["MA"], server_url="http://x")
    except RuntimeError as e:
        assert "OMICOS_USER_TOKEN" in str(e) and "by hand" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


ATLAS2 = ATLAS + [{"cell_name": "red alpha cell", "cl_id": "CL:4"},
                  {"cell_name": "blue alpha cell", "cl_id": "CL:5"}]
TEMPLATES.update({
    "red alpha cell": {"positive": [("GR", 10), ("GA", 4)], "negative": [("GU", 2)]},
    "blue alpha cell": {"positive": [("GU", 10), ("GA", 4)], "negative": [("GR", 3)]},
})


def fake_get2(url, params=None, headers=None, timeout=30):
    if url.endswith("/api/markers/gene_celltype_counts"):
        c = fake_get(url, params, headers, timeout)
        c["counts"].update({"GR": {"positive": 1}, "GU": {"positive": 1}})
        return c
    if url.endswith("/api/markers/celltypes") and not (params and params.get("tissue")):
        return {"celltypes": ATLAS2}
    if "/search/alias_name:" in url and "red%20protein" in url:
        return {"response": {"docs": [{"symbol": "GR"}]}}
    if "/fetch/symbol/" in url:
        q = url.rsplit("/", 1)[1]
        g = {"MR": ["GR"], "MU": ["GU"]}.get(q)
        if g:
            return {"response": {"docs": [{"symbol": x} for x in g]}}
    return fake_get(url, params, headers, timeout)


def test_polar_context_overrides_and_cache(tmp_path=None):
    import tempfile, os
    mdb._get = fake_get2
    cache = os.path.join(tempfile.mkdtemp(), "c.json")
    labels = ["alpha cell", "red alpha cell", "blue alpha cell", "gamma cell"]
    kb = mdb.markerdb_signatures(labels, ["MA", "MR", "MU", "MG", "DNA"], token="T",
                                 server_url="http://x", cache=cache, context=["organ x"])
    assert kb["parents"] == {"red alpha cell": "alpha cell", "blue alpha cell": "alpha cell"}
    assert sorted(kb["polar"]["alpha cell"]) == ["blue alpha cell", "red alpha cell"]
    # context: gamma cell is never recorded in organ x; the children follow their parent
    assert kb["out_of_context"] == ["gamma cell"] and "red alpha cell" in kb["signatures"]
    assert kb["context_scopes"] == [("tissue", "organ x")]
    # nomenclature overrides apply even when the cache already holds a gene map
    kb2 = mdb.markerdb_signatures(labels, ["MA", "MR", "MU", "MG", "DNA"], token="T",
                                  server_url="http://x", cache=cache, gene_overrides={"DNA": ["GG"]})
    assert kb2["gene_map"]["DNA"] == ["GG"] and "DNA" not in kb2["unresolved_markers"]


def test_protein_names_resolve_abbreviations():
    import tempfile, os
    mdb._get = fake_get2
    cache = os.path.join(tempfile.mkdtemp(), "c.json")
    kb = mdb.markerdb_signatures(["alpha cell", "red alpha cell"], ["MA", "XYZ"], token="T",
                                 server_url="http://x", cache=cache,
                                 protein_names={"XYZ": "red protein"})
    assert kb["gene_map"]["XYZ"] == ["GR"] and "XYZ" not in kb["unresolved_markers"]
    rows = {r["marker"]: r for r in kb["gene_check"]}
    assert rows["XYZ"]["match"] == "given" and rows["MA"]["match"] == "symbol"


def test_name_variants_greek_prefix():
    v = mdb._name_variants("a.SMA")
    assert "alpha-SMA" in v and mdb._name_variants("Ki-67") == ["Ki-67"]
    assert "beta-2-Microglobulin" in mdb._name_variants("B2-Microglobulin")


def test_gene_under_distinct_names_shares_weight():
    import tempfile, os
    mdb._get = fake_get
    kb = mdb.markerdb_signatures(["alpha cell", "gamma cell"], ["MA", "MA-iso"], token="T",
                                 server_url="http://x", gene_overrides={"MA-iso": ["GA"]},
                                 cache=os.path.join(tempfile.mkdtemp(), "c.json"))
    s = kb["signatures"]["alpha cell"]
    assert abs(s["MA"] - s["MA-iso"]) < 1e-9 and s["MA"] <= 0.5 + 1e-9


# --- panel names -> genes: alias ownership and Greek-prefixed abbreviations ---
# The tables mirror the public records: NCBI lists ASMA ("alpha skeletal muscle
# actin") as an alias of ACTA1; neither HGNC nor NCBI lists aSMA / alpha-SMA
# for ACTA2; antibodies registered as "alpha-SMA" declare UniProt P62736 (ACTA2).
MYGENE = {
    "ASMA": [{"symbol": "ACTA1", "alias": ["ACTA", "ASMA", "NEM3"]}],
    "CD45": [{"symbol": "PTPRCAP", "alias": ["LPAP", "CD45-AP"]},     # ranked first
             {"symbol": "PTPRC", "alias": ["CD45", "LCA", "T200"]}],
    "PDL1": [{"symbol": "CD274", "alias": ["B7-H1", "PD-L1", "PDL1"]}],
}
HGNC_ALIAS = {"Ki-67": [{"symbol": "MKI67", "alias_symbol": ["KIA", "Ki-67"]}],
              "HER2": [{"symbol": "ERBB2", "alias_symbol": ["HER-2", "HER2", "NEU"]}]}
UNIPROT_NAMES = {   # protein_name query -> reviewed entries
    "beta-Catenin": [("CTNNBIP1", "Beta-catenin-interacting protein 1", []),
                     ("CTNNB1", "Catenin beta-1", ["Beta-catenin"])],
}
ACCESSIONS = {"P62736": "ACTA2", "P68032": "ACTC1", "P63267": "ACTG2", "P35222": "CTNNB1"}
ABREG = {"alpha-SMA": [("alpha-SMA", "P62736"), ("alpha-SMA", "p62736")],
         "alpha SMA": [("alpha SMA", "P62736"), ("alpha SMA", "P62736"),
                       ("alpha SMA", "P68032, P62736, P63267")]}


def fake_names(url, params=None, headers=None, timeout=30):
    import re
    from urllib.parse import unquote
    if "mygene.info" in url:
        v = params["q"].split(" OR ")[0].split(":", 1)[1]
        return {"hits": MYGENE.get(re.sub(r"[^A-Z0-9]", "", v.upper()), [])}
    if "genenames.org/fetch/alias_symbol/" in url:
        return {"response": {"docs": HGNC_ALIAS.get(unquote(url.rsplit("/", 1)[1]), [])}}
    if "genenames.org" in url:
        return {"response": {"docs": []}}
    if "uniprot" in url:
        q = params["query"]
        m = re.search(r'protein_name:"([^"]+)"', q)
        if m:
            rows = UNIPROT_NAMES.get(m.group(1), [])
            return {"results": [
                {"genes": [{"geneName": {"value": g}}],
                 "proteinDescription": {"recommendedName": {"fullName": {"value": rec}},
                                        "alternativeNames": [{"fullName": {"value": a}} for a in alt]}}
                for g, rec, alt in rows]}
        accs = re.findall(r"accession:(\w+)", q)
        return {"results": [{"genes": [{"geneName": {"value": ACCESSIONS[a]}}]}
                            for a in accs if a in ACCESSIONS]}
    if "antibodyregistry" in url:
        return {"items": [{"abTarget": t, "abTargetUniprotId": a}
                          for t, a in ABREG.get(params["q"], [])]}
    return {"response": {"docs": []}, "hits": []}


def test_panel_gene_map_greek_abbreviation_is_not_a_case_folded_alias():
    """aSMA is alpha-smooth-muscle actin (ACTA2), not the ACTA1 alias ASMA."""
    mdb._get = fake_names
    names = ["aSMA", "a-SMA", "a.SMA", "ASMA", "bCatenin"]
    src = {}
    gm, miss = mdb.panel_gene_map(names, sources=src, workers=1)
    assert gm["aSMA"] == gm["a-SMA"] == gm["a.SMA"] == ["ACTA2"]
    assert src["aSMA"] == "antibody"
    # an upper-case token *is* a symbol-style alias and keeps its NCBI meaning
    assert gm["ASMA"] == ["ACTA1"] and src["ASMA"] == "mygene"
    # beta-catenin: exact UniProt protein name, not the substring hit CTNNBIP1
    assert gm["bCatenin"] == ["CTNNB1"] and src["bCatenin"] == "uniprot"
    assert not miss


def test_panel_gene_map_common_protein_aliases():
    mdb._get = fake_names
    src = {}
    gm, miss = mdb.panel_gene_map(["Ki67", "CD45", "PD-L1", "HER2"], sources=src, workers=1)
    assert gm["Ki67"] == ["MKI67"]               # via the nomenclature spelling Ki-67
    assert gm["CD45"] == ["PTPRC"]               # the symbol whose alias list has CD45
    assert gm["PD-L1"] == ["CD274"] and gm["HER2"] == ["ERBB2"]
    assert src["HER2"] == "alias" and not miss


def test_greek_abbreviation_unresolved_rather_than_wrong_when_offline():
    """Without the protein-name sources, aSMA must not fall back to ACTA1."""
    def offline(url, params=None, headers=None, timeout=30):
        if "uniprot" in url or "antibodyregistry" in url:
            raise OSError("offline")
        return fake_names(url, params, headers, timeout)
    mdb._get = offline
    gm, miss = mdb.panel_gene_map(["aSMA"], workers=1)
    assert "aSMA" not in gm and miss == ["aSMA"]


def test_greek_prefix_detection_and_spellings():
    for t in ("aSMA", "a-SMA", "a.SMA", "bCatenin", "b2M", "gH2AX"):
        assert mdb._greek_prefixed(t), t
    for t in ("ASMA", "B2M", "Ki67", "CD45", "beta-catenin", "dsDNA", "Vimentin"):
        assert not mdb._greek_prefixed(t), t
    assert "alpha-SMA" in mdb._name_variants("aSMA")
    assert "Ki-67" in mdb._variants("Ki67")


def test_failed_lookup_is_reported_and_not_cached():
    """A nomenclature lookup that fails transiently (network / rate limit) must
    not be cached as 'no match': it is reported and retried on the next call."""
    import tempfile, os, json
    import requests
    state = {"down": True}

    def flaky(url, params=None, headers=None, timeout=30):
        if state["down"] and ("genenames" in url or "mygene" in url) and url.endswith("MB"):
            raise requests.ConnectionError("rate limited")
        if state["down"] and "mygene" in url and params and "MB" in str(params.get("q")):
            raise requests.ConnectionError("rate limited")
        return fake_get(url, params, headers, timeout)
    mdb._get = flaky
    cache = os.path.join(tempfile.mkdtemp(), "c.json")
    labels = ["alpha cell", "beta-positive alpha cell", "gamma cell"]
    kb = mdb.markerdb_signatures(labels, ["MA", "MB", "MG"], token="T", server_url="http://x",
                                 cache=cache, resolve_aliases=False)
    assert "MB" in kb["lookup_failed"] and "MB" not in kb["gene_map"]
    assert "MB" not in json.load(open(cache))["gene_cache"]
    state["down"] = False
    kb2 = mdb.markerdb_signatures(labels, ["MA", "MB", "MG"], token="T", server_url="http://x",
                                  cache=cache, resolve_aliases=False)
    assert kb2["gene_map"]["MB"] == ["GB"] and kb2["lookup_failed"] == []


ATLAS3 = [{"cell_name": "root cell", "cl_id": "CL:10"}, {"cell_name": "kid one", "cl_id": "CL:11"},
          {"cell_name": "kid two", "cl_id": "CL:12"}, {"cell_name": "other a", "cl_id": "CL:20"},
          {"cell_name": "other b", "cl_id": "CL:21"}, {"cell_name": "other c", "cl_id": "CL:22"}]
TEMPLATES3 = {
    "root cell": {"positive": [("GA", 10), ("GC", 10)], "negative": []},
    "kid one": {"positive": [("GB", 10), ("GA", 1)], "negative": []},   # a state of the root lineage
    "kid two": {"positive": [("GC", 10)], "negative": []},              # a subtype marked by GC
    "other a": {"positive": [("GD", 10)], "negative": []},
    "other b": {"positive": [("GE", 10)], "negative": []},
    "other c": {"positive": [("GF", 10)], "negative": []},
}


def fake_get3(url, params=None, headers=None, timeout=30):
    if url.endswith("/api/markers/gene_celltype_counts"):
        return {"counts": {g: {"positive": 1} for g in ("GA", "GB", "GC", "GD", "GE", "GF")}}
    if url.endswith("/api/markers/celltypes"):
        return {"celltypes": ATLAS3}
    if url.endswith("/api/markers/celltype_markers"):
        names = params["cell_names"].split("|")
        return {"templates": [
            {"cell_name": n, "cl_id": None,
             "positive": [{"gene": g, "n_pmid": k} for g, k in TEMPLATES3[n]["positive"]],
             "negative": []} for n in names]}
    if "/fetch/symbol/" in url:
        q = url.rsplit("/", 1)[1]
        g = {"MA": "GA", "MB": "GB", "MC": "GC", "MD": "GD", "ME": "GE", "MF": "GF"}.get(q)
        return {"response": {"docs": [{"symbol": g}] if g else []}}
    if "hierarchicalParents" in url:
        key = next((k for k in ("CL_11", "CL_12") if k in url), "")
        return {"_embedded": {"terms": [{"obo_id": "CL:10"}] if key else []}, "_links": {}}
    return {"response": {"docs": []}, "hits": []}


def test_lineage_markers_and_subtype_markers_not_inherited():
    mdb._get = fake_get3
    labels = ["root cell", "kid one", "kid two", "other a", "other b", "other c"]
    kb = mdb.markerdb_signatures(labels, ["MA", "MB", "MC", "MD", "ME", "MF"], token="T",
                                 server_url="http://x")
    # kid one's own entry lists the root's defining marker GA: its lineage marker
    assert kb["lineage_markers"].get("kid one") == ["MA"]
    # the root's GC defines kid two (another descendant of the root): kid one
    # does not inherit it; without the rule it would
    assert "MC" not in kb["signatures"]["kid one"]
    old = mdb.markerdb_signatures(labels, ["MA", "MB", "MC", "MD", "ME", "MF"], token="T",
                                  server_url="http://x", subtype_markers=False)
    assert old["signatures"]["kid one"].get("MC", 0) > 0


def test_failed_ontology_lookup_is_reported_and_not_cached():
    """A label whose ontology search fails transiently must not be cached with
    a loose fallback match; it is reported and resolved on the next call."""
    import tempfile, os, json
    import requests
    state = {"down": True}
    atlas = ATLAS + [{"cell_name": "inflamed alpha cell", "cl_id": "CL:7"}]

    def flaky(url, params=None, headers=None, timeout=30):
        if "ols4" in url and state["down"]:
            raise requests.ConnectionError("service unavailable")
        if url.endswith("/api/markers/celltypes"):
            return {"celltypes": atlas}
        if "ols4/api/search" in url and params and "A1 alpha" in str(params.get("q")):
            return {"response": {"docs": [{"obo_id": "CL:7", "label": "inflamed alpha cell",
                                           "synonym": ["A1 alpha cell"]}]}}
        if url.endswith("/api/markers/celltype_markers"):
            names = params["cell_names"].split("|")
            tm = dict(TEMPLATES, **{"inflamed alpha cell": {"positive": [("GA", 3), ("GB", 6)],
                                                            "negative": []}})
            return {"templates": [
                {"cell_name": n, "cl_id": None,
                 "positive": [{"gene": g, "n_pmid": k} for g, k in tm[n]["positive"]],
                 "negative": [{"gene": g, "n_pmid": k} for g, k in tm[n]["negative"]]}
                for n in names]}
        return fake_get(url, params, headers, timeout)
    mdb._get = flaky
    cache = os.path.join(tempfile.mkdtemp(), "c.json")
    labels = ["alpha cell", "A1 alpha cell", "gamma cell"]
    kb = mdb.markerdb_signatures(labels, ["MA", "MB", "MG"], token="T", server_url="http://x",
                                 cache=cache, resolve_aliases=False)
    assert "label: A1 alpha cell" in kb["lookup_failed"]
    assert "A1 alpha cell" not in json.load(open(cache))["resolved"]
    state["down"] = False
    kb2 = mdb.markerdb_signatures(labels, ["MA", "MB", "MG"], token="T", server_url="http://x",
                                  cache=cache, resolve_aliases=False)
    assert kb2["resolved"]["A1 alpha cell"]["terms"][0]["cell_name"] == "inflamed alpha cell"
    assert not [x for x in kb2["lookup_failed"] if x.startswith("label")]
