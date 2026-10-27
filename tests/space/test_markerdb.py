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
    if "hierarchicalAncestors" in url:
        anc = {"CL_2": [{"obo_id": "CL:1"}]}
        key = "CL_2" if "CL_2" in url else ""
        return {"_embedded": {"terms": anc.get(key, [])}, "_links": {}}
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
    # specificity: GA is used by two cell types, GB by one
    assert sig["beta+ alpha cell"]["MB"] > sig["beta+ alpha cell"]["MA"]


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
