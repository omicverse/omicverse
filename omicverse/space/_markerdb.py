r"""Signed marker signatures for a label set, from marker knowledge bases.

``markerdb_signatures`` turns a list of cell-type labels and the marker names
of a protein panel into the ``signatures`` / ``parents`` that
:func:`omicverse.space.score_cluster_signatures` consumes.  All biology comes
from external resources at run time — nothing about cell types or markers is
written here:

* **labels → ontology**: each label is matched to a Cell Ontology term (EBI
  OLS search) and to the atlas' canonical cell-type names;
* **hierarchy**: ``parents`` are the nearest Cell Ontology ancestors that are
  themselves in the label set;
* **markers**: positive and negative marker genes per cell type, with
  literature support, from the OmicosAnno atlas (``/api/markers``);
* **specificity**: the atlas' per-gene cell-type counts
  (``spec = 1 / (1 + ln n)``) down-weight promiscuous markers;
* **panel names → genes**: HGNC search (symbols, aliases, previous symbols),
  then MyGene.info, for names such as ``CD20`` or ``Podoplanin``.

Network access is required.  The atlas marker endpoint needs an OmicOS
account token (``OMICOS_USER_TOKEN``, injected into OmicOS kernels); the
server defaults to ``OMICOS_SERVER_URL``.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .._registry import register_function

__all__ = ["markerdb_signatures", "panel_gene_map"]

_OLS = "https://www.ebi.ac.uk/ols4/api"
_HGNC = "https://rest.genenames.org"
_MYGENE = "https://mygene.info/v3"


def _get(url, params=None, headers=None, timeout=30):
    import requests

    r = requests.get(url, params=params, headers=headers or {"Accept": "application/json"},
                     timeout=timeout)
    r.raise_for_status()
    return r.json()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _variants(name: str) -> List[str]:
    s = str(name).strip()
    out = [s, re.sub(r"[\s._\-]+", "", s), re.sub(r"[._\-]+", " ", s)]
    seen, res = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            res.append(v)
    return res


def _hgnc(path: str, timeout: float) -> List[dict]:
    try:
        return _get(f"{_HGNC}/{path}", timeout=timeout)["response"]["docs"]
    except Exception:
        return []


def _gene_candidates(name: str, species: str, timeout: float) -> List[str]:
    """Exact nomenclature matches first, free-text search last."""
    human = species.lower() in ("human", "homo sapiens", "hs")
    for v in _variants(name):
        q = urllib.parse.quote(v)
        if human:
            docs = _hgnc(f"fetch/symbol/{q}", timeout)
            if docs:
                return [d["symbol"] for d in docs]
        try:
            hits = _get(f"{_MYGENE}/query",
                        params={"q": f"symbol:{v} OR alias:{v}", "species": species,
                                "fields": "symbol", "size": 10}, timeout=timeout).get("hits", [])
        except Exception:
            hits = []
        if hits:
            return list(dict.fromkeys(h["symbol"] for h in hits))
        if human:
            docs = _hgnc(f"fetch/prev_symbol/{q}", timeout) + _hgnc(f"fetch/alias_symbol/{q}", timeout)
            if docs:
                return list(dict.fromkeys(d["symbol"] for d in docs))
    if human:   # symbol stems ("HLA-DR" -> HLA-DRA, HLA-DRB1, ...)
        for v in _variants(name):
            if " " in v or len(v) > 8:
                continue
            docs = _hgnc(f"search/symbol:{urllib.parse.quote(v)}*", timeout)
            # subunits / chains of the named protein: stem + one letter (+ digits)
            stem = re.escape(v.upper())
            sub = [d["symbol"] for d in docs if re.fullmatch(stem + r"[A-Z]\d*", d["symbol"].upper())]
            if sub:
                return sub
    if human:   # protein names ("Granzyme B", "Podoplanin"): HGNC full-text, clear winner only
        for v in _variants(name):
            docs = _hgnc(f"search/{urllib.parse.quote(v)}", timeout)
            if docs:
                top = docs[0]["score"]
                tied = [d["symbol"] for d in docs if d["score"] >= top]
                if len(docs) == len(tied) or docs[len(tied)]["score"] < 0.5 * top:
                    # accept only if the name really is the gene's (approved,
                    # alias or previous) name — free text alone mis-assigns
                    ok = []
                    for sym in tied:
                        rec = _hgnc(f"fetch/symbol/{urllib.parse.quote(sym)}", timeout)
                        rec = rec[0] if rec else {}
                        names = [rec.get("name", "")] + list(rec.get("alias_name") or []) \
                            + list(rec.get("prev_name") or [])
                        if any(_norm(v) and _norm(v) in _norm(n) for n in names):
                            ok.append(sym)
                    if ok:
                        return ok
    return []


def _pick(cands: List[str], marker_genes: Optional[Mapping[str, int]]) -> List[str]:
    """Disambiguate candidate genes by their use as markers in the atlas."""
    if not marker_genes:
        return cands if len(cands) <= 4 else []
    used = sorted(((marker_genes.get(c.upper(), 0), c) for c in cands), reverse=True)
    used = [(n, c) for n, c in used if n > 0]
    if not used:
        return []
    if len(used) == 1 or used[0][0] >= 2 * used[1][0]:
        return [used[0][1]]
    return [c for _, c in used] if len(used) <= 4 else []


@register_function(
    aliases=["蛋白名转基因", "panel 基因映射", "protein marker to gene symbol", "panel gene map"],
    category="space",
    description="Map protein-panel marker names (CD20, Granzyme B, Podoplanin, ...) to gene "
                "symbols through HGNC search and MyGene.info",
    requires={}, produces={}, auto_fix="none",
    examples=["gm, unresolved = ov.space.panel_gene_map(['CD20', 'Pax-5', 'Granzyme B'])"],
)
def panel_gene_map(markers: Iterable[str], species: str = "human", *,
                   overrides: Optional[Mapping[str, Sequence[str]]] = None,
                   marker_genes: Optional[Mapping[str, int]] = None,
                   timeout: float = 20.0):
    r"""Resolve panel marker names to gene symbols from public gene nomenclature.

    Arguments:
        markers: panel column names.
        species: species for MyGene.info (HGNC is used for human).
        overrides: optional ``{marker: [genes]}`` that take precedence — for
            nomenclature only (names no resource resolves), never for biology.
        marker_genes: optional ``{gene: n}`` (e.g. atlas marker usage counts);
            when a name matches several genes, the gene used as a marker far
            more often wins, otherwise all marker genes are kept (≤ 4).

    Returns:
        ``(gene_map, unresolved)`` — ``{marker: [gene symbols]}`` and the list
        of names that no resource resolved unambiguously.
    """
    gm, miss = {}, []
    for m in markers:
        if overrides and m in overrides:
            gm[m] = list(overrides[m])
            continue
        g = _pick(_gene_candidates(m, species, timeout), marker_genes)
        if g:
            gm[m] = g
        else:
            miss.append(m)
    return gm, miss


def _ols_terms(label: str, timeout: float, rows: int = 10) -> List[dict]:
    try:
        docs = _get(f"{_OLS}/search", params={"q": label, "ontology": "cl", "rows": rows,
                                              "fieldList": "obo_id,label,synonym"},
                    timeout=timeout)["response"]["docs"]
    except Exception:
        return []
    return [d for d in docs if str(d.get("obo_id", "")).startswith("CL:")]


def _ols_ancestors(cl_id: str, timeout: float) -> List[str]:
    iri = "http://purl.obolibrary.org/obo/" + cl_id.replace(":", "_")
    enc = urllib.parse.quote(urllib.parse.quote(iri, safe=""), safe="")
    out, url = [], f"{_OLS}/ontologies/cl/terms/{enc}/hierarchicalAncestors?size=200"
    try:
        while url:
            js = _get(url, timeout=timeout)
            out += [t.get("obo_id") for t in js.get("_embedded", {}).get("terms", []) if t.get("obo_id")]
            url = js.get("_links", {}).get("next", {}).get("href")
    except Exception:
        pass
    return out


def _ols_order(cl_ids: Sequence[str], timeout: float) -> Dict[str, List[str]]:
    """Ancestors of each term, nearest first (by ancestor depth, deepest first)."""
    anc = {c: _ols_ancestors(c, timeout) for c in cl_ids}
    depth = {}
    for c, a in anc.items():
        for x in a:
            depth.setdefault(x, None)
    # depth of an ancestor = number of its own ancestors (cheap proxy, cached)
    cache: Dict[str, int] = {}
    def d(x):
        if x not in cache:
            cache[x] = len(_ols_ancestors(x, timeout))
        return cache[x]
    return {c: sorted(a, key=lambda x: -d(x)) for c, a in anc.items()}


_STOP = {"cell", "cells", "of", "the", "a", "an"}


def _tokens(s: str) -> set:
    s = re.sub(r"\+", "-positive ", str(s))           # CL spells "+" as "-positive"
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP}


def _jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _resolve_one(label: str, atlas: List[dict], timeout: float):
    """Best atlas cell type for a label: (term, score)."""
    tl = _tokens(label)
    by_norm = {_norm(c["cell_name"]): c for c in atlas}
    for q in (label, re.sub(r"\+", "-positive", label)):
        if _norm(q) in by_norm:
            return by_norm[_norm(q)], 1.0
    by_cl = {}
    for c in atlas:
        if c.get("cl_id"):
            by_cl.setdefault(c["cl_id"], c)
    best, best_s = None, 0.0
    query = re.sub(r"\+", "-positive", label)
    for t in _ols_terms(query, timeout, rows=15):
        c = by_cl.get(t["obo_id"])
        if c is None:
            continue
        names = [t.get("label", "")] + list(t.get("synonym") or []) + [c["cell_name"]]
        sc = max(_jacc(tl, _tokens(n)) for n in names)
        if sc > best_s:
            best, best_s = c, sc
    for c in atlas:                                     # atlas names themselves
        sc = _jacc(tl, _tokens(c["cell_name"]))
        if sc > best_s:
            best, best_s = c, sc
    return best, best_s


def _resolve_label(label: str, atlas: List[dict], timeout: float, min_score: float = 0.5):
    """``{"terms": [...], "score": s}`` — a compound label ("A_B", "A/B") becomes
    the union of its parts when every part names a cell type exactly."""
    whole = re.sub(r"[_]+", " ", label).strip()
    c, sc = _resolve_one(whole, atlas, timeout)
    if sc < 0.99:
        parts = [p.strip() for p in re.split(r"[_/&]| and ", label) if p.strip()]
        parts = [p for p in parts if len(p) >= 2 and p.lower() not in _STOP]
        if len(parts) >= 2:
            got = [_resolve_one(p, atlas, timeout) for p in parts]
            if all(g is not None and gs >= 0.99 for g, gs in got):
                terms = list({g["cell_name"]: {"cell_name": g["cell_name"], "cl_id": g.get("cl_id")}
                              for g, _ in got}.values())
                return {"terms": terms, "score": 1.0}
    if c is None or sc < min_score:
        return None
    return {"terms": [{"cell_name": c["cell_name"], "cl_id": c.get("cl_id")}], "score": round(sc, 3)}


@register_function(
    aliases=["知识库签名", "marker 知识库签名", "markerdb signatures", "atlas signatures",
             "signed signatures from marker database"],
    category="space",
    description="Build signed (+/-) marker signatures and a parent map for a label set from "
                "marker knowledge bases (OmicosAnno atlas via Cell Ontology, HGNC/MyGene for "
                "panel names), ready for ov.space.score_cluster_signatures",
    requires={}, produces={}, auto_fix="none",
    examples=[
        "kb = ov.space.markerdb_signatures(vocab_labels, panel_markers, species='human')",
        "res = ov.space.score_cluster_signatures(prof, kb['signatures'], parents=kb['parents'])",
    ],
)
def markerdb_signatures(
    labels: Sequence[str],
    panel_markers: Sequence[str],
    *,
    species: str = "human",
    tissue: Optional[str] = None,
    disease: Optional[str] = None,
    server_url: Optional[str] = None,
    token: Optional[str] = None,
    gene_overrides: Optional[Mapping[str, Sequence[str]]] = None,
    min_pmid: int = 1,
    cache: Optional[str] = None,
    timeout: float = 30.0,
) -> dict:
    r"""Signed signatures for ``labels`` restricted to the markers of a panel.

    Arguments:
        labels: the allowed cell-type labels (e.g. a task vocabulary).
        panel_markers: marker names of one panel (or the union of panels;
            scoring ignores markers a dataset lacks).
        species / tissue / disease: atlas scope (``tissue`` / ``disease`` as
            the atlas names them; ``None`` = all).
        server_url / token: OmicOS server and account token (default
            ``OMICOS_SERVER_URL`` / ``OMICOS_USER_TOKEN``).
        gene_overrides: ``{marker: [genes]}`` for names no resource resolves.
        min_pmid: drop marker statements supported by fewer papers.
        cache: optional JSON path; network results are stored and reused.

    Returns:
        ``dict`` with ``signatures`` (``{label: {marker: weight}}``; positive
        weights = atlas positive markers, negative = atlas negative markers,
        each scaled by literature support ``ln(1 + n_pmid)`` and scope
        specificity), ``parents`` (``{label: parent_label}`` from Cell
        Ontology), ``resolved`` (``{label: {"terms": [{cell_name, cl_id}],
        "score"}}`` — the atlas cell type(s) each label maps to),
        ``unresolved_labels``, ``gene_map``, ``unresolved_markers``,
        ``empty_signatures`` (labels whose atlas markers miss the panel) and
        ``evidence`` (``{label: {marker: [genes, n_pmid, polarity]}}``).
    """
    store = {}
    if cache and Path(cache).exists():
        store = json.loads(Path(cache).read_text())
    base = (server_url or os.environ.get("OMICOS_SERVER_URL") or "https://auth.omicos.cn").rstrip("/")
    tok = token or os.environ.get("OMICOS_USER_TOKEN")
    # the atlas names species "Human" / "Mouse"; the gene services take "human"
    atlas_species = {"human": "Human", "homo sapiens": "Human", "hs": "Human",
                     "mouse": "Mouse", "mus musculus": "Mouse", "mm": "Mouse"}.get(
        str(species).lower(), species)
    scope = {k: v for k, v in dict(species=atlas_species, tissue=tissue,
                                   disease=disease).items() if v}

    # 0. atlas marker usage (public): specificity and gene disambiguation
    if "gene_counts" not in store:
        store["gene_counts"] = _get(f"{base}/api/markers/gene_celltype_counts",
                                    params=scope, timeout=timeout).get("counts", {})
    counts = {g.upper(): v for g, v in store["gene_counts"].items()}
    usage = {g: int(v.get("positive", 0) or 0) + int(v.get("negative", 0) or 0)
             for g, v in counts.items()}

    # 1. panel names -> genes
    gm = store.get("gene_map")
    if gm is None:
        gm, miss = panel_gene_map(panel_markers, species, overrides=gene_overrides,
                                  marker_genes=usage, timeout=timeout)
        store["gene_map"], store["unresolved_markers"] = gm, miss
    miss = store.get("unresolved_markers", [])
    gene_to_markers: Dict[str, List[str]] = {}
    for m, genes in gm.items():
        for g in genes:
            gene_to_markers.setdefault(g.upper(), []).append(m)

    # 2. labels -> atlas cell types via Cell Ontology
    if "atlas_celltypes" not in store:
        store["atlas_celltypes"] = _get(f"{base}/api/markers/celltypes",
                                        params={**scope, "limit": 2000}, timeout=timeout)["celltypes"]
    atlas = store["atlas_celltypes"]
    resolved = store.get("resolved", {})
    for lab in labels:
        if lab not in resolved:
            hit = _resolve_label(lab, atlas, timeout)
            if hit:
                resolved[lab] = hit
    store["resolved"] = resolved
    # an atlas cell type serves one label only (the closest name); a label that
    # fell back onto another label's cell type would get an identical signature
    owner: Dict[tuple, str] = {}
    for lab in labels:
        r = resolved.get(lab)
        if not r:
            continue
        key = tuple(sorted(t["cell_name"] for t in r["terms"]))
        if key not in owner or r["score"] > resolved[owner[key]]["score"]:
            owner[key] = lab
    resolved = {lab: r for lab, r in resolved.items() if lab in labels
                and owner.get(tuple(sorted(t["cell_name"] for t in r["terms"]))) == lab}
    unresolved_labels = [l for l in labels if l not in resolved]

    # 3. atlas markers (+/-)
    names = sorted({c["cell_name"] for v in resolved.values() for c in v["terms"]})
    have = {t["cell_name"] for t in store.get("templates", [])}
    todo = [n for n in names if n not in have]
    if todo:
        if not tok:
            raise RuntimeError("OMICOS_USER_TOKEN is not set; the atlas marker endpoint needs "
                               "an OmicOS account token")
        templates = list(store.get("templates", []))
        for i in range(0, len(todo), 20):
            js = _get(f"{base}/api/markers/celltype_markers",
                      params={**scope, "cell_names": "|".join(todo[i:i + 20]),
                              "include_negative": "true", "pmids_per_gene": 0},
                      headers={"Accept": "application/json", "Authorization": f"Bearer {tok}"},
                      timeout=timeout)
            templates += js.get("templates", [])
        store["templates"] = templates
    tmpl = {t["cell_name"]: t for t in store.get("templates", [])}

    def spec(g, pol):
        n = max(1, int(counts.get(g, {}).get(pol, 1) or 1))
        return 1.0 / (1.0 + math.log(n))

    signatures, evidence, empty = {}, {}, []
    for lab, r in resolved.items():
        sig, ev = {}, {}
        for term in r["terms"]:                       # compound labels: union
            t = tmpl.get(term["cell_name"])
            if t is None:
                continue
            for pol, sign in (("positive", 1.0), ("negative", -1.0)):
                for m in t.get(pol, []):
                    g = m["gene"].upper()
                    if m.get("n_pmid", 0) < min_pmid or g not in gene_to_markers:
                        continue
                    w = sign * spec(g, pol) * math.log1p(m.get("n_pmid", 1))
                    for mk in gene_to_markers[g]:
                        if mk in sig and (sig[mk] > 0) != (w > 0):
                            sig[mk] += w           # conflicting statements net out
                        elif abs(w) > abs(sig.get(mk, 0.0)):
                            sig[mk] = w
                        ev.setdefault(mk, []).append([term["cell_name"], g, m.get("n_pmid", 0), pol])
        if any(w > 0 for w in sig.values()):
            signatures[lab] = {k: round(v, 4) for k, v in sig.items() if abs(v) > 1e-9}
            evidence[lab] = ev
        else:
            empty.append(lab)

    # 4. parents from Cell Ontology
    cl_of = {lab: r["terms"][0]["cl_id"] for lab, r in resolved.items()
             if len(r["terms"]) == 1 and r["terms"][0].get("cl_id")}
    if "ancestors" not in store:
        store["ancestors"] = _ols_order(sorted(set(cl_of.values())), timeout)
    anc = store["ancestors"]
    lab_of_cl: Dict[str, List[str]] = {}
    for lab, c in cl_of.items():
        lab_of_cl.setdefault(c, []).append(lab)
    parents = {}
    for lab, c in cl_of.items():
        if lab not in signatures:
            continue
        for a in anc.get(c, []):
            cands = [p for p in lab_of_cl.get(a, []) if p in signatures and p != lab]
            if cands:
                parents[lab] = cands[0]
                break

    if cache:
        Path(cache).write_text(json.dumps(store))
    return dict(signatures=signatures, parents=parents, resolved=resolved,
                unresolved_labels=unresolved_labels, gene_map=gm, unresolved_markers=miss,
                empty_signatures=empty, evidence=evidence)
