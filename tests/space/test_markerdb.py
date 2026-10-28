"""Offline test of markerdb_signatures: every web service is replaced by a
fake that answers from small in-memory tables (names are arbitrary)."""
import importlib.util
import sys
import types
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "omicverse" / "space" / "_markerdb.py"


def _load():
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
        assert "OMICOS_USER_TOKEN" in str(e)
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
