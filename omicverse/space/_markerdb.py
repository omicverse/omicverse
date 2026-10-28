r"""Signed marker signatures for a label set, from marker knowledge bases.

``markerdb_signatures`` turns a list of cell-type labels and the marker names
of a protein panel into the ``signatures`` / ``parents`` / ``polar`` that
:func:`omicverse.space.score_cluster_signatures` consumes.  All biology comes
from external resources at run time — nothing about cell types or markers is
written here:

* **labels → atlas / ontology**: each label is matched to the OmicosAnno
  atlas' cell-type names and to a Cell Ontology (CL) term (EBI OLS);
* **markers**: positive and negative marker genes per cell type, with the
  number of supporting papers, from the OmicosAnno atlas (``/api/markers``).
  A statement's weight is its support relative to the cell type's
  best-supported positive marker (so the defining marker weighs 1 and a
  marker named in a handful of papers next to one named in hundreds weighs
  little);
* **inheritance**: a cell type *is a* its CL ancestors, so it carries their
  markers too (down-weighted per generation) — e.g. a T-cell state inherits
  the T-cell markers its own papers rarely restate; where the cell type's
  own entry makes a statement about a marker, that statement wins;
* **hierarchy**: ``parents`` come from the label set itself (a label whose
  words contain another label's words, within the same group when groups are
  given) and otherwise from CL ancestry;
* **polar splits**: two sibling labels are *opposed* when each one's defining
  panel marker is a negative marker of the other in the atlas; their parent
  is returned in ``polar``;
* **context**: optional tissue / disease names select atlas scopes; labels
  whose cell type the atlas never records there are left out;
* **panel names → genes**: HGNC (symbols, aliases, previous symbols, protein
  names), then MyGene.info.

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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


def _pmap(fn, items, workers: int = 8):
    """Order-preserving parallel map for I/O-bound lookups."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(fn, items))


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


def _match_kind(name: str, genes: Sequence[str]) -> str:
    """'symbol' when the panel name is (a spelling of) the approved symbol of
    one of its genes, otherwise 'alias / name'."""
    n = _norm(name)
    return "symbol" if any(_norm(g) == n for g in genes) else "alias / name"


def _gene_candidates(name: str, species: str, timeout: float) -> Tuple[List[str], str]:
    """Exact nomenclature matches first, free-text search last.

    Returns ``(candidates, source)``; ``source`` says which rule matched
    (``symbol``, ``alias``, ``mygene``, ``stem``, ``family``, ``name``) so that
    matches other than an approved symbol can be reviewed.
    """
    human = species.lower() in ("human", "homo sapiens", "hs")
    for v in _variants(name):
        q = urllib.parse.quote(v)
        if human:
            docs = _hgnc(f"fetch/symbol/{q}", timeout)
            if docs:
                return [d["symbol"] for d in docs], "symbol"
            # HGNC aliases / previous symbols are complete for human; MyGene's
            # alias index can miss paralogues (CD16 -> FCGR3A and FCGR3B)
            docs = _hgnc(f"fetch/alias_symbol/{q}", timeout) + _hgnc(f"fetch/prev_symbol/{q}", timeout)
            if docs:
                return list(dict.fromkeys(d["symbol"] for d in docs)), "alias"
        try:
            hits = _get(f"{_MYGENE}/query",
                        params={"q": f"symbol:{v} OR alias:{v}", "species": species,
                                "fields": "symbol", "size": 10}, timeout=timeout).get("hits", [])
        except Exception:
            hits = []
        if hits:
            return list(dict.fromkeys(h["symbol"] for h in hits)), "mygene"
    if human:   # symbol stems ("HLA-DR" -> HLA-DRA, HLA-DRB1, ...)
        for v in _variants(name):
            if " " in v or len(v) > 8:
                continue
            docs = _hgnc(f"search/symbol:{urllib.parse.quote(v)}*", timeout)
            # subunits / chains of the named protein: stem + one letter (+ digits)
            stem = re.escape(v.upper())
            sub = [d["symbol"] for d in docs if re.fullmatch(stem + r"[A-Z]\d*", d["symbol"].upper())]
            if sub:
                return sub, "stem"
    if human:   # free text naming a family ("Collagen I"): the top-scoring
        # hits that share one symbol stem (COL1A1 / COL1A2), as candidates
        for v in _variants(name):
            docs = _hgnc(f"search/{urllib.parse.quote(v)}", timeout)
            if len(docs) < 2:
                continue
            top = docs[0]["score"]
            head = [d["symbol"] for d in docs if d["score"] >= 0.75 * top]
            if len(head) < 2 or len(head) > 6:
                continue
            stem = re.match(r"[A-Z]+\d*", head[0].upper())
            stem = stem.group(0) if stem else ""
            fam = [h for h in head if stem and h.upper().startswith(stem)]
            if len(stem) >= 3 and len(fam) == len(head):
                return fam, "family"
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
                        return ok, "name"
    return [], ""


def _pick(cands: List[str], marker_genes: Optional[Mapping[str, int]]) -> List[str]:
    """Disambiguate candidate genes by their use as markers in the atlas.

    One gene used far more often wins; otherwise the genes used at least half
    as often as the most used one are kept (chains / paralogues the antibody
    may not distinguish), at most four.
    """
    if not marker_genes:
        return cands if len(cands) <= 4 else []
    used = sorted(((marker_genes.get(c.upper(), 0), c) for c in cands), reverse=True)
    used = [(n, c) for n, c in used if n > 0]
    if not used:
        return []
    if len(used) == 1 or used[0][0] >= 2 * used[1][0]:
        return [used[0][1]]
    keep = [c for n, c in used if n >= 0.5 * used[0][0]]
    return keep if len(keep) <= 4 else []


_UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
_ABREG = "https://www.antibodyregistry.org/api/fts-antibodies"
_ACC = re.compile(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b")


def _uniprot_query(query: str, timeout: float, size: int = 25) -> List[str]:
    try:
        js = _get(_UNIPROT, params={"query": query, "fields": "gene_primary", "size": size,
                                    "format": "json"}, timeout=timeout)
    except Exception:
        return []
    out = []
    for r in js.get("results", []):
        for g in r.get("genes", [])[:1]:
            v = g.get("geneName", {}).get("value")
            if v:
                out.append(v)
    return list(dict.fromkeys(out))


_GREEK = {"a": "alpha", "b": "beta", "g": "gamma", "d": "delta", "k": "kappa"}


def _name_variants(name: str) -> List[str]:
    """Spellings of a panel name: separators normalised, a one-letter Greek
    prefix spelled out (a.SMA -> alpha-SMA, b.Catenin -> beta-catenin,
    B2-microglobulin -> beta-2-microglobulin)."""
    s = str(name).strip()
    out = [s, re.sub(r"[._]+", " ", s).strip(), re.sub(r"[._ ]+", "-", s).strip("-")]
    m = re.match(r"^([abgdkABGDK])(?:[.\-_ ]|(?=\d))(\d?)[.\-_ ]?(.+)$", s)
    if m:
        g = _GREEK[m.group(1).lower()]
        rest = m.group(3)
        num = f"-{m.group(2)}" if m.group(2) else ""
        out += [f"{g}{num}-{rest}", f"{g}{num} {rest}"]
    return list(dict.fromkeys(v for v in out if v))


def _uniprot_genes(name: str, timeout: float, taxon: str = "9606") -> List[str]:
    """Genes whose reviewed UniProt entry lists ``name`` as a protein name
    (recommended, alternative or short name)."""
    out: List[str] = []
    for v in _name_variants(name):
        out += _uniprot_query(f'protein_name:"{v}" AND organism_id:{taxon} AND reviewed:true', timeout)
    return list(dict.fromkeys(out))


def _antibody_registry_genes(name: str, timeout: float, taxon: str = "9606") -> List[str]:
    """Genes of the UniProt targets that Antibody Registry (RRID) entries
    matching ``name`` declare."""
    acc = []
    for v in _name_variants(name):
        try:
            items = _get(_ABREG, params={"q": v, "page": 1, "size": 100}, timeout=timeout).get("items", [])
        except Exception:
            continue
        n = _norm(v)
        for it in items:
            tgt = _norm(it.get("abTarget") or "")
            if n and n not in tgt and n not in _norm(it.get("abName") or ""):
                continue
            for f in ("abTargetUniprotId", "uniprotId"):
                acc += _ACC.findall(str(it.get(f) or ""))
    acc = list(dict.fromkeys(acc))[:20]
    if not acc:
        return []
    q = " OR ".join(f"accession:{a}" for a in acc)
    return _uniprot_query(f"({q}) AND organism_id:{taxon}", timeout)


def _cl_synonym_terms(name: str, timeout: float) -> List[str]:
    """Cell Ontology terms that carry ``name`` as an exact label / synonym
    (panels sometimes name a channel after the cell type it identifies)."""
    out = []
    for v in dict.fromkeys([name, re.sub(r"[._]", " ", name)]):
        docs = _ols_docs({"q": v, "rows": 10, "exact": "true", "queryFields": "label,synonym"}, timeout)
        out += [d["obo_id"] for d in docs if any(_norm(x) == _norm(v) for x in
                                                 [d.get("label", "")] + list(d.get("synonym") or []))]
    return list(dict.fromkeys(out))


def _genes_by_protein_name(name: str, timeout: float) -> List[str]:
    """HGNC genes whose approved, alias or previous *name* is ``name``."""
    q = urllib.parse.quote(f'"{name}"')
    for field in ("name", "alias_name", "prev_name"):
        docs = _hgnc(f"search/{field}:{q}", timeout)
        if docs:
            return list(dict.fromkeys(d["symbol"] for d in docs))
    return []


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
                   timeout: float = 20.0, workers: int = 8,
                   sources: Optional[Dict[str, str]] = None):
    r"""Resolve panel marker names to gene symbols from public gene nomenclature.

    Arguments:
        markers: panel column names.
        species: species for MyGene.info (HGNC is used for human).
        overrides: optional ``{marker: [genes]}`` that take precedence — for
            nomenclature only (names no resource resolves), never for biology.
        marker_genes: optional ``{gene: n}`` (e.g. atlas marker usage counts);
            when a name matches several genes, the gene used as a marker far
            more often wins, otherwise all marker genes are kept (≤ 4).
        workers: parallel requests.
        sources: optional dict filled with ``{marker: rule}`` — how each name
            was matched (``symbol`` = approved HGNC symbol; ``alias`` /
            ``mygene`` / ``stem`` / ``family`` / ``name`` are indirect matches
            worth reviewing; ``override``).

    Returns:
        ``(gene_map, unresolved)`` — ``{marker: [gene symbols]}`` and the list
        of names that no resource resolved unambiguously.
    """
    markers = list(markers)
    todo = [m for m in markers if not (overrides and m in overrides)]
    cands = dict(zip(todo, _pmap(lambda m: _gene_candidates(m, species, timeout), todo, workers)))
    gm, miss = {}, []
    for m in markers:
        if overrides and m in overrides:
            gm[m] = list(overrides[m])
            if sources is not None:
                sources[m] = "override"
            continue
        c, src = cands[m]
        g = _pick(c, marker_genes)
        if g:
            gm[m] = g
            if sources is not None:
                sources[m] = src
        else:
            miss.append(m)
    return gm, miss


# ----------------------------------------------------------------------------
# Cell Ontology
# ----------------------------------------------------------------------------

def _ols_docs(params: dict, timeout: float) -> List[dict]:
    try:
        docs = _get(f"{_OLS}/search", params={"ontology": "cl", "fieldList": "obo_id,label,synonym",
                                              **params}, timeout=timeout)["response"]["docs"]
    except Exception:
        return []
    return [d for d in docs if str(d.get("obo_id", "")).startswith("CL:")]


def _ols_terms(label: str, timeout: float, rows: int = 10) -> List[dict]:
    return _ols_docs({"q": label, "rows": rows}, timeout)


def _cl_exact(names: Sequence[str], timeout: float, min_score: float = 0.6) -> Optional[str]:
    """CL id whose label or synonym equals one of ``names`` (OLS exact search);
    otherwise the search hit whose *label* shares the most words with a name
    (token Jaccard >= ``min_score``; synonyms are not used here, they often
    name narrower types)."""
    for n in names:
        for q in dict.fromkeys([n, re.sub(r"\+", "-positive", n)]):
            docs = _ols_docs({"q": q, "rows": 5, "exact": "true",
                              "queryFields": "label,synonym"}, timeout)
            if docs:
                return docs[0]["obo_id"]
    best, bs = None, (0.0, 0)
    for n in names:
        tn = _tokens(n)
        for d in _ols_docs({"q": re.sub(r"\+", "-positive", n), "rows": 25}, timeout):
            sc = (_jacc(tn, _tokens(d.get("label", ""))), -len(_tokens(d.get("label", ""))))
            if sc > bs:
                best, bs = d["obo_id"], sc
    return best if bs[0] >= min_score else None


def _iri_enc(cl_id: str) -> str:
    iri = "http://purl.obolibrary.org/obo/" + cl_id.replace(":", "_")
    return urllib.parse.quote(urllib.parse.quote(iri, safe=""), safe="")


def _ols_parents(cl_id: str, timeout: float) -> List[str]:
    out, url = [], f"{_OLS}/ontologies/cl/terms/{_iri_enc(cl_id)}/hierarchicalParents?size=200"
    try:
        while url:
            js = _get(url, timeout=timeout)
            out += [t.get("obo_id") for t in js.get("_embedded", {}).get("terms", [])
                    if str(t.get("obo_id", "")).startswith("CL:")]
            url = js.get("_links", {}).get("next", {}).get("href")
    except Exception:
        pass
    return out


def _cl_lineage(cl_ids: Iterable[str], parents: Dict[str, List[str]], timeout: float,
                workers: int, max_depth: int = 30) -> Dict[str, Dict[str, int]]:
    """``{term: {ancestor: generation}}`` by breadth-first search over CL
    ``is_a`` / ``part_of`` parents; ``parents`` is a shared cache."""
    cl_ids = [c for c in dict.fromkeys(cl_ids) if c]
    gen = {c: {} for c in cl_ids}
    front = {c: {c} for c in cl_ids}
    for depth in range(1, max_depth + 1):
        need = sorted({x for f in front.values() for x in f} - set(parents))
        for c, p in zip(need, _pmap(lambda x: _ols_parents(x, timeout), need, workers)):
            parents[c] = p
        nxt = {}
        for c, f in front.items():
            new = {p for x in f for p in parents.get(x, [])} - set(gen[c]) - {c}
            for p in new:
                gen[c][p] = depth
            if new:
                nxt[c] = new
        front = nxt
        if not front:
            break
    return gen


# ----------------------------------------------------------------------------
# label resolution
# ----------------------------------------------------------------------------

_STOP = {"cell", "cells", "of", "the", "a", "an"}
# grammar of a free-text label definition, not biology
_DESC_STOP = _STOP | {"all", "any", "other", "population", "populations", "type", "types",
                      "subset", "subsets", "one", "ones", "kind", "kinds", "respectively",
                      "etc", "various", "both"}


def _fold(w: str) -> str:
    """Crude plural folding so 'fibroblasts' meets 'fibroblast'."""
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _tokens(s: str) -> set:
    s = re.sub(r"\+", "-positive ", str(s))           # CL spells "+" as "-positive"
    s = re.sub(r"\bnon[\s\-]+", "non", s, flags=re.I)  # "non-classical" is one word
    return {_fold(w) for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP}


def _desc_phrases(label: str, text: str) -> List[str]:
    """Cell-type phrases of a free-text label definition ("Stroma covers
    fibroblasts, vessels and matrix" -> fibroblasts / vessels / matrix)."""
    parts = re.split(r"[,;:()\[\]/]|\band\b|\bor\b|\bincluding\b|\bsuch as\b|\bcovers?\b|"
                     r"\bmeans?\b|\bis\b|\bare\b|\bincludes?\b|\bdenotes?\b|\bas well as\b|"
                     r"\bplus\b|\bwith\b", str(text), flags=re.I)
    own = _norm(label)
    out = []
    for p in parts:
        words = [w for w in re.findall(r"[A-Za-z0-9+\-]+", p) if w.lower() not in _DESC_STOP]
        ph = " ".join(words).strip(" -")
        if len(ph) >= 3 and _norm(ph) != own and ph not in out:
            out.append(ph)
    return out


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


def _resolve_description(label: str, text: str, atlas: List[dict], timeout: float,
                         min_score: float = 0.8):
    """Atlas cell types named by a label's free-text definition (union of the
    phrases that match an atlas type closely); ``None`` if none does."""
    terms, scores, used = {}, [], []
    for ph in _desc_phrases(label, text):
        c, sc = _resolve_one(ph, atlas, timeout)
        if c is not None and sc >= min_score:
            terms.setdefault(c["cell_name"], {"cell_name": c["cell_name"], "cl_id": c.get("cl_id")})
            scores.append(sc)
            used.append(ph)
    if not terms:
        return None
    return {"terms": list(terms.values()), "score": round(min(scores), 3), "from_description": used}


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


def _lexical_parents(labels: Sequence[str], groups: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Parent = the most specific other label whose words are a proper subset
    of the label's words ("memory B cell" -> "B cell"), within the same group
    when ``groups`` is given."""
    tok = {l: _tokens(l) for l in labels}
    out = {}
    for l in labels:
        best = None
        for p in labels:
            if p == l or not tok[p] or not tok[p] < tok[l]:
                continue
            if groups is not None and groups.get(p) != groups.get(l):
                continue
            if best is None or len(tok[p]) > len(tok[best]):
                best = p
        if best:
            out[l] = best
    return out


def _match_scope(name: str, options: Sequence[str]) -> Optional[str]:
    by = {_norm(o): o for o in options}
    if _norm(name) in by:
        return by[_norm(name)]
    tn = _tokens(name)
    best, bs = None, 0.0
    for o in options:
        s = _jacc(tn, _tokens(o))
        if s > bs:
            best, bs = o, s
    return best if bs >= 0.5 else None


# ----------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------

@register_function(
    aliases=["知识库签名", "marker 知识库签名", "markerdb signatures", "atlas signatures",
             "signed signatures from marker database"],
    category="space",
    description="Build signed (+/-) marker signatures, a parent map and polar parents for a "
                "label set from marker knowledge bases (OmicosAnno atlas, Cell Ontology "
                "inheritance, HGNC/MyGene for panel names), optionally restricted to a "
                "tissue / disease context; ready for ov.space.score_cluster_signatures",
    requires={}, produces={}, auto_fix="none",
    examples=[
        "kb = ov.space.markerdb_signatures(vocab_labels, panel_markers, groups=vocab_groups, "
        "context=['lymph node'])",
        "res = ov.space.score_cluster_signatures(prof, kb['signatures'], parents=kb['parents'], "
        "polar=kb['polar'], min_weight=1)",
    ],
)
def markerdb_signatures(
    labels: Sequence[str],
    panel_markers: Sequence[str],
    *,
    species: str = "human",
    groups: Optional[Mapping[str, str]] = None,
    context: Optional[Sequence[str]] = None,
    tissue: Optional[str] = None,
    disease: Optional[str] = None,
    server_url: Optional[str] = None,
    token: Optional[str] = None,
    gene_overrides: Optional[Mapping[str, Sequence[str]]] = None,
    protein_names: Optional[Mapping[str, str]] = None,
    resolve_aliases: bool = True,
    remap_min: float = 0.2,
    descriptions: Optional[Mapping[str, str]] = None,
    weighting: str = "relative",
    share_power: float = 0.5,
    min_pmid: int = 1,
    min_rel: float = 0.05,
    inherit_depth: int = 2,
    inherit_decay: float = 0.5,
    ontology_parents: bool = False,
    cache: Optional[str] = None,
    timeout: float = 30.0,
    workers: int = 8,
) -> dict:
    r"""Signed signatures for ``labels`` restricted to the markers of a panel.

    Arguments:
        labels: the allowed cell-type labels (e.g. a task vocabulary).
        panel_markers: marker names of one panel (or the union of panels;
            scoring ignores markers a dataset lacks).
        species: atlas species.
        groups: optional ``{label: group}`` shipped with the label set (e.g. a
            vocabulary's compartment / lineage column).  Parents are only
            sought within a group.
        context: optional tissue and/or disease names the data come from
            (matched to the atlas' tissue and disease scopes).  Labels whose
            cell type (or CL term) the atlas never records in any matched
            scope are dropped (never a label with a ``descriptions`` entry), unless a child of theirs is recorded there;
            children of kept labels are kept (they are only used to split
            their parent).  ``None`` keeps every label.
        tissue / disease: restrict the *marker statements* to one atlas
            scope (``None`` = all; usually leave unset).
        server_url / token: OmicOS server and account token (default
            ``OMICOS_SERVER_URL`` / ``OMICOS_USER_TOKEN``).
        gene_overrides: ``{marker: [genes]}`` for panel names no resource
            resolves (nomenclature only).  Applied even when a cache exists.
        protein_names: ``{marker: full protein name}`` for abbreviated panel
            names; each full name is resolved to genes through HGNC approved
            / alias / previous names (the database decides the gene), and
            takes precedence over the abbreviation's own lookup.
        descriptions: optional ``{label: text}`` — the label's definition
            as the task / study words it (e.g. "X covers A, B and C"). Its
            cell-type phrases are matched to atlas types and the label's
            signature becomes their union; phrases that match nothing closely
            are ignored; without a match the label name itself is used.
        weighting: ``'relative'`` (default): a statement's weight is its
            paper count relative to the cell type's best-supported positive
            marker (defining marker = 1).  ``'share'``: ``sqrt(n / n_max) *
            ln(1 + n)`` times the atlas-wide specificity of the gene;
            ``'literature'``: ``ln(1 + n)`` times specificity.  The latter two
            are rescaled so the cell type's strongest positive weight is 1.
        resolve_aliases: panel names whose lookup only matched an alias of a
            gene that is a defining marker of no label in ``labels`` (or that
            stayed unresolved) are looked up again in UniProt protein names,
            Antibody Registry (RRID) target accessions and Cell Ontology
            synonyms (a channel named after the cell type it identifies →
            that cell type's best-supported atlas marker); a candidate that is a
            marker of a label (support >= ``remap_min`` of the label's best) replaces the mapping (most sources, then atlas usage).  Reported
            in ``remapped_markers``.
        min_pmid: drop marker statements supported by fewer papers.
        min_rel: drop statements whose support is below this fraction of the
            cell type's best-supported positive marker (incidental mentions).
        ontology_parents: also nest labels by Cell Ontology ancestry when the
            label set's own wording gives no parent (off by default: CL nests
            some vocabulary-level labels under broader ones, e.g. a lineage
            under a generic class the label set treats as a sibling).
        inherit_depth / inherit_decay: a label also carries the markers of its
            CL ancestors up to ``inherit_depth`` generations, scaled by
            ``inherit_decay ** generation`` (``0`` disables inheritance).
        cache: optional JSON path; network results are stored and reused.
        workers: parallel requests.

    Returns:
        ``dict`` with ``signatures`` (``{label: {marker: weight}}``; the
        label's best-supported positive marker weighs 1, other statements by
        their paper count relative to it; negative weights = atlas negative
        markers), ``parents`` (``{label: parent_label}``), ``polar``
        (``{parent: [child, child]}`` — opposed sibling pairs: each one's
        defining panel marker is an atlas negative marker of the other),
        ``defining`` (``{label: [panel markers]}`` — the label's own markers
        with at least half its top support), ``resolved``
        (``{label: {"terms": [{cell_name, cl_id}], "score"}}``), ``cl_ids``,
        ``unresolved_labels``, ``out_of_context`` (labels dropped by
        ``context``), ``context_scopes``, ``gene_map``, ``gene_names``
        (``{marker: {gene: approved HGNC name}}`` — check that each names
        the protein the panel measures), ``gene_check`` (one row per panel
        name: genes, approved names, ``match`` — approved symbol, alias /
        name, or given —, ``atlas_usage`` — cell types the atlas lists the
        gene for — and ``defines`` — candidate labels it is a defining marker
        of; alias matches that define nothing and are rarely used come
        first: the likeliest wrong genes),
        ``unresolved_markers``, ``empty_signatures``, ``weak_signatures``
        (labels whose strongest marker on this panel weighs less than a
        quarter of the median label's strongest — candidates for the residual
        / exclusion label), ``gene_map_source`` (``{marker: rule}``),
        ``gene_review`` (``{marker: ["GENE (approved name)"]}`` for indirect
        matches) and ``evidence``
        (``{label: {marker: [[cell type, gene, n_pmid, polarity, generation]]}}``).
    """
    labels = list(dict.fromkeys(labels))
    if weighting not in ("relative", "share", "literature"):
        raise ValueError("weighting must be 'relative', 'share' or 'literature'")
    store = {}
    if cache and Path(cache).exists():
        store = json.loads(Path(cache).read_text())
    base = (server_url or os.environ.get("OMICOS_SERVER_URL") or "https://auth.omicos.cn").rstrip("/")
    tok = token or os.environ.get("OMICOS_USER_TOKEN")
    # the atlas names species "Human" / "Mouse"; the gene services take "human"
    atlas_species = {"human": "Human", "homo sapiens": "Human", "hs": "Human",
                     "mouse": "Mouse", "mus musculus": "Mouse", "mm": "Mouse"}.get(
        str(species).lower(), species)
    sp = {"species": atlas_species}
    scope = {k: v for k, v in dict(species=atlas_species, tissue=tissue,
                                   disease=disease).items() if v}

    def save():
        if cache:
            Path(cache).write_text(json.dumps(store))

    # 0. atlas marker usage (public): gene disambiguation
    if "gene_counts" not in store:
        store["gene_counts"] = _get(f"{base}/api/markers/gene_celltype_counts",
                                    params=scope, timeout=timeout).get("counts", {})
    counts = {g.upper(): v for g, v in store["gene_counts"].items()}
    usage = {g: int(v.get("positive", 0) or 0) + int(v.get("negative", 0) or 0)
             for g, v in counts.items()}

    # 1. panel names -> genes (resolved names are cached per name; overrides always win)
    gcache = store.setdefault("gene_cache", {})
    gsrc = store.setdefault("gene_source", {})
    todo = [m for m in panel_markers if (m not in gcache or m not in gsrc)
            and not (gene_overrides and m in gene_overrides)]
    if todo:
        srcd: Dict[str, str] = {}
        got, _ = panel_gene_map(todo, species, marker_genes=usage, timeout=timeout,
                                workers=workers, sources=srcd)
        for m in todo:
            gcache[m] = got.get(m, [])
            gsrc[m] = srcd.get(m, "")
    pncache = store.setdefault("protein_name_cache", {})
    pn = {m: n for m, n in (protein_names or {}).items() if m in panel_markers}
    need = sorted({n for n in pn.values() if n not in pncache})
    for n, g in zip(need, _pmap(lambda n: _genes_by_protein_name(n, timeout), need, workers)):
        pncache[n] = g
    gm, miss = {}, []
    for m in panel_markers:
        g = gcache.get(m, [])
        if m in pn and pncache.get(pn[m]):
            g = _pick(pncache[pn[m]], usage) or pncache[pn[m]]
        if gene_overrides and m in gene_overrides:
            g = list(gene_overrides[m])
        if g:
            gm[m] = g
        else:
            miss.append(m)
    # approved gene names, so a reviewer can see whether each mapping names the protein
    gnames = store.setdefault("gene_names", {})
    need = sorted({g for genes in gm.values() for g in genes} - set(gnames))

    def _gname(g):
        d = _hgnc(f"fetch/symbol/{urllib.parse.quote(g)}", timeout)
        return d[0].get("name", "") if d else ""
    for g, n in zip(need, _pmap(_gname, need, workers)):
        gnames[g] = n
    gene_names = {m: {g: gnames.get(g, "") for g in genes} for m, genes in gm.items()}
    gene_to_markers: Dict[str, List[str]] = {}
    for m, genes in gm.items():
        for g in genes:
            gene_to_markers.setdefault(g.upper(), []).append(m)

    # 2. labels -> atlas cell types, labels -> CL
    if "atlas_celltypes" not in store:
        store["atlas_celltypes"] = _get(f"{base}/api/markers/celltypes",
                                        params={**sp, "limit": 2000}, timeout=timeout)["celltypes"]
    atlas = store["atlas_celltypes"]
    atlas_by_cl: Dict[str, str] = {}
    for c in atlas:
        if c.get("cl_id"):
            atlas_by_cl.setdefault(c["cl_id"], c["cell_name"])
    resolved = store.setdefault("resolved", {})
    desc_store = store.setdefault("descriptions", {})
    new = []
    for lab in labels:
        text = (descriptions or {}).get(lab)
        if text:
            if desc_store.get(lab) != text or lab not in resolved:
                new.append(lab)
        elif lab not in resolved or lab in desc_store:
            new.append(lab)

    def _res(lab):
        text = (descriptions or {}).get(lab)
        if text:
            return _resolve_description(lab, text, atlas, timeout) or _resolve_label(lab, atlas, timeout)
        return _resolve_label(lab, atlas, timeout)
    for l, hit in zip(new, _pmap(_res, new, workers)):
        resolved[l] = hit
        text = (descriptions or {}).get(l)
        if text:
            desc_store[l] = text
        else:
            desc_store.pop(l, None)
    # an atlas cell type serves one label only (the closest name)
    owner: Dict[tuple, str] = {}
    for lab in labels:
        r = resolved.get(lab)
        if not r:
            continue
        key = tuple(sorted(t["cell_name"] for t in r["terms"]))
        if key not in owner or r["score"] > resolved[owner[key]]["score"]:
            owner[key] = lab
    res = {lab: r for lab, r in resolved.items() if lab in labels and r
           and owner.get(tuple(sorted(t["cell_name"] for t in r["terms"]))) == lab}
    unresolved_labels = [l for l in labels if l not in res]

    clc = store.setdefault("label_cl", {})
    need = [l for l in labels if l not in clc]

    def _cl_of(l):
        r = res.get(l)
        if r and len(r["terms"]) == 1 and r["terms"][0].get("cl_id"):
            return r["terms"][0]["cl_id"]
        names = ([r["terms"][0]["cell_name"]] if r and len(r["terms"]) == 1 else []) + [l]
        return _cl_exact(names, timeout)
    for l, c in zip(need, _pmap(_cl_of, need, workers)):
        clc[l] = c
    cl_of = {l: clc.get(l) for l in labels if clc.get(l)}

    # 3. CL lineage (labels' terms) -> inherited atlas terms and CL parents
    pcache = store.setdefault("cl_parents", {})
    lineage = _cl_lineage(cl_of.values(), pcache, timeout, workers)
    save()

    def inherited(lab):
        c = cl_of.get(lab)
        if not c or inherit_depth <= 0:
            return []
        out = []
        for a, g in sorted(lineage.get(c, {}).items(), key=lambda x: x[1]):
            if g <= inherit_depth and a in atlas_by_cl:
                out.append((atlas_by_cl[a], g))
        return out

    # 4. atlas markers (+/-) for own and inherited terms
    names = sorted({t["cell_name"] for v in res.values() for t in v["terms"]}
                   | {n for lab in res for n, _ in inherited(lab)})
    tmpl_store = store.setdefault("templates", {})
    if isinstance(tmpl_store, list):                     # older cache layout
        tmpl_store = store["templates"] = {t["cell_name"]: t for t in tmpl_store}
    def fetch_templates(names_):
        todo = [n for n in dict.fromkeys(names_) if n not in tmpl_store]
        if not todo:
            return
        if not tok:
            raise RuntimeError("OMICOS_USER_TOKEN is not set; the atlas marker endpoint needs "
                               "an OmicOS account token")
        chunks = [todo[i:i + 20] for i in range(0, len(todo), 20)]

        def fetch(chunk):
            return _get(f"{base}/api/markers/celltype_markers",
                        params={**scope, "cell_names": "|".join(chunk),
                                "include_negative": "true", "pmids_per_gene": 0},
                        headers={"Accept": "application/json", "Authorization": f"Bearer {tok}"},
                        timeout=timeout).get("templates", [])
        for got in _pmap(fetch, chunks, min(workers, 4)):
            for t in got:
                tmpl_store[t["cell_name"]] = t
        for n in todo:
            tmpl_store.setdefault(n, {"cell_name": n, "positive": [], "negative": []})
    fetch_templates(names)
    save()

    scheme = weighting in ("share", "literature")

    def rel(t, floor=None):
        """per-gene relative support of one atlas term: {gene: {pol: r}}"""
        floor = min_rel if floor is None else floor
        pos = [m for m in t.get("positive", []) if m.get("n_pmid", 0) >= min_pmid]
        neg = [m for m in t.get("negative", []) if m.get("n_pmid", 0) >= min_pmid]
        top = max([m.get("n_pmid", 0) for m in pos] + [1])

        def wt(n, g, pol):           # weight of one statement under the chosen scheme
            if not scheme:
                return n / top
            k = max(1, int(counts.get(g, {}).get(pol, 1) or 1))
            w = math.log1p(max(n, 1)) / (1.0 + math.log(k))
            if weighting == "share":
                w *= min(1.0, n / top) ** share_power
            return w
        scale = max([wt(m.get("n_pmid", 0), m["gene"].upper(), "positive") for m in pos] + [1e-9]) \
            if scheme else 1.0
        out = {}
        for pol, ms in (("positive", pos), ("negative", neg)):
            for m in ms:
                n = m.get("n_pmid", 0)
                if n / top < floor:
                    continue
                g = m["gene"].upper()
                r = wt(n, g, pol) / scale if scheme else n / top
                out.setdefault(g, {})[pol] = max(out.get(g, {}).get(pol, 0.0), r)
        return out

    # 4b. panel names whose own lookup matched an alias of a gene that defines no
    # label: gather candidates from other nomenclature resources and keep the one
    # that is a defining marker of a label in the label set
    def top_genes(t, frac=0.5):
        return {g for g, d in rel(t, 0.0).items()
                if d.get("positive", 0.0) - d.get("negative", 0.0) >= frac}
    label_def: Dict[str, List[str]] = {}
    for lab, r in res.items():
        for term in r["terms"]:
            for g in top_genes(tmpl_store.get(term["cell_name"], {}), remap_min):
                label_def.setdefault(g, []).append(lab)
    fixed = set(gene_overrides or {}) | set(pn)
    suspects = [m for m in panel_markers if m not in fixed and
                ((m in gm and (gsrc.get(m) or _match_kind(m, gm[m])) != "symbol"
                  and not any(g.upper() in label_def for g in gm[m])) or m in miss)]
    if not resolve_aliases:
        suspects = []
    acache = store.setdefault("alias_candidates", {})
    need = [m for m in suspects if m not in acache]

    def _alt(m):
        src: Dict[str, List[str]] = {}
        for tag, genes in (("UniProt", _uniprot_genes(m, timeout)),
                           ("AntibodyRegistry", _antibody_registry_genes(m, timeout))):
            for g in genes:
                src.setdefault(g.upper(), []).append(tag)
        return {"genes": src, "cl": _cl_synonym_terms(m, timeout)}
    for m, got in zip(need, _pmap(_alt, need, workers)):
        acache[m] = got
    # cell-type synonyms -> the atlas markers of that cell type (or its nearest ancestor)
    cl_need = sorted({c for m in suspects for c in acache[m]["cl"]})
    cl_gen = _cl_lineage(cl_need, pcache, timeout, workers) if cl_need else {}
    cl_term = {}
    for c in cl_need:
        chain = [c] + [a for a, _ in sorted(cl_gen.get(c, {}).items(), key=lambda x: x[1])]
        hit = next((atlas_by_cl[a] for a in chain if a in atlas_by_cl), None)
        if hit:
            cl_term[c] = hit
    fetch_templates(cl_term.values())
    remapped = {}
    for m in suspects:
        src = {g: list(v) for g, v in acache[m]["genes"].items()}
        for c in acache[m]["cl"]:
            if c in cl_term:      # the cell type's best-supported marker
                r_ = rel(tmpl_store.get(cl_term[c], {}), 0.0)
                pos = {g: d.get("positive", 0.0) for g, d in r_.items() if d.get("positive")}
                if pos:
                    src.setdefault(max(pos, key=pos.get), []).append(f"CellOntology:{c}")
        # a gene another panel name already measures is not a new mapping (a
        # different name for it denotes an isoform / epitope, e.g. a splice form)
        taken = {x.upper() for k, gs in gm.items() if k != m and k not in suspects for x in gs}
        ok = {g: v for g, v in src.items() if g in label_def and g not in taken}
        if not ok:
            continue
        best = max(ok, key=lambda g: (len(set(ok[g])), usage.get(g, 0)))
        remapped[m] = dict(was=gm.get(m, []), now=[best], sources=sorted(set(ok[best])),
                           defines=sorted(set(label_def[best])))
        gm[m] = [best]
        if m in miss:
            miss.remove(m)
    if remapped:
        for g in {g for v in remapped.values() for g in v["now"]} - set(gnames):
            gnames[g] = _gname(g)
        gene_names = {m: {g: gnames.get(g, "") for g in genes} for m, genes in gm.items()}
        gene_to_markers = {}
        for m, genes in gm.items():
            for g in genes:
                gene_to_markers.setdefault(g.upper(), []).append(m)
    save()

    def own_defining(lab, frac=0.5):
        return sorted(m for m, (p, n) in own_markers(lab).items() if p - n >= frac)

    def own_markers(lab):
        """{panel marker: (pos, neg)} from the label's own atlas term(s)"""
        acc = {}
        for term in res[lab]["terms"]:      # every statement counts here (opposition evidence)
            for g, d in rel(tmpl_store.get(term["cell_name"], {}), 0.0).items():
                for mk in gene_to_markers.get(g, []):
                    a = acc.setdefault(mk, [0.0, 0.0])
                    a[0] = max(a[0], d.get("positive", 0.0))
                    a[1] = max(a[1], d.get("negative", 0.0))
        return acc

    signatures, evidence, empty, own = {}, {}, [], {}
    for lab in res:
        acc, ev = {}, {}
        sources = [(t["cell_name"], 0) for t in res[lab]["terms"]] + inherited(lab)
        spoken = set()          # markers the label's own atlas entry makes a statement about
        for name, gen in sources:
            scale = inherit_decay ** gen
            t = tmpl_store.get(name, {})
            for g, d in rel(t).items():
                for mk in gene_to_markers.get(g, []):
                    if gen > 0 and mk in spoken:
                        continue        # the specific statement overrides the inherited one
                    if gen == 0:
                        spoken.add(mk)
                    a = acc.setdefault(mk, [0.0, 0.0])
                    for i, pol in enumerate(("positive", "negative")):
                        if pol in d:
                            a[i] = max(a[i], scale * d[pol])
                            ev.setdefault(mk, []).append([name, g, round(d[pol], 4), pol, gen])
        # a gene measured under several distinct panel names (not spellings of one
        # name) shares its weight among them instead of counting once per name
        def _share(mk):
            names = {_norm(x) for g in gm.get(mk, []) for x in gene_to_markers.get(g.upper(), [])}
            return 1.0 / max(1, len(names))
        sig = {mk: round((a[0] - a[1]) * _share(mk), 4) for mk, a in acc.items()
               if abs(a[0] - a[1]) > 1e-4}
        own[lab] = own_markers(lab)
        if any(w > 0 for w in sig.values()):
            signatures[lab] = sig
            evidence[lab] = ev
        else:
            empty.append(lab)

    # 5. parents: the label set's own structure, then Cell Ontology
    lex = _lexical_parents([l for l in labels if l in signatures], groups)
    lab_of_cl: Dict[str, List[str]] = {}
    for lab, c in cl_of.items():
        if lab in signatures:
            lab_of_cl.setdefault(c, []).append(lab)
    parents = {}
    for lab in signatures:
        if lab in lex:
            parents[lab] = lex[lab]
            continue
        if not ontology_parents:
            continue
        c = cl_of.get(lab)
        for a, _ in sorted(lineage.get(c, {}).items(), key=lambda x: x[1]) if c else []:
            cands = [p for p in lab_of_cl.get(a, []) if p != lab
                     and (groups is None or groups.get(p) == groups.get(lab))]
            if cands:
                parents[lab] = cands[0]
                break
    # no cycles
    for lab in list(parents):
        seen, x = {lab}, parents.get(lab)
        while x is not None:
            if x in seen:
                parents.pop(lab, None)
                break
            seen.add(x)
            x = parents.get(x)

    # 6. context
    out_of_context, ctx_scopes = [], []
    if context:
        if "scopes" not in store:
            store["scopes"] = {
                "tissue": [t["tissue_name"] for t in _get(f"{base}/api/markers/tissues",
                                                          timeout=timeout).get("tissues", [])],
                "disease": [d["disease"] for d in _get(f"{base}/api/markers/diseases",
                                                       timeout=timeout).get("diseases", [])]}
        for name in context:
            for kind in ("tissue", "disease"):
                hit = _match_scope(name, store["scopes"][kind])
                if hit and (kind, hit) not in ctx_scopes:
                    ctx_scopes.append((kind, hit))
        ctx = store.setdefault("context_celltypes", {})
        need = [s for s in ctx_scopes if f"{s[0]}:{s[1]}" not in ctx]
        for (kind, hit), got in zip(need, _pmap(
                lambda s: _get(f"{base}/api/markers/celltypes",
                               params={**sp, s[0]: s[1], "limit": 2000}, timeout=timeout)["celltypes"],
                need, workers)):
            ctx[f"{kind}:{hit}"] = [[c["cell_name"], c.get("cl_id")] for c in got]
        save()
        seen_names, seen_cl = set(), set()
        for kind, hit in ctx_scopes:
            for n, c in ctx[f"{kind}:{hit}"]:
                seen_names.add(n)
                if c:
                    seen_cl.add(c)
        if ctx_scopes:
            # a label the task defines itself (descriptions) is in context by definition
            keep = {lab for lab in signatures
                    if any(t["cell_name"] in seen_names for t in res[lab]["terms"])
                    or (cl_of.get(lab) in seen_cl) or (descriptions or {}).get(lab)}
            for lab in list(keep):                       # ancestors of kept labels
                x = parents.get(lab)
                while x is not None:
                    keep.add(x)
                    x = parents.get(x)
            for _ in range(len(labels)):                 # descendants of kept labels
                more = {c for c, p in parents.items() if p in keep and c not in keep}
                if not more:
                    break
                keep |= more
            out_of_context = [l for l in signatures if l not in keep]
            signatures = {l: s for l, s in signatures.items() if l in keep}
            parents = {c: p for c, p in parents.items() if c in signatures and p in signatures}

    # 7. opposed siblings -> polar parents
    kids: Dict[str, List[str]] = {}
    for c, p in parents.items():
        kids.setdefault(p, []).append(c)
    polar: Dict[str, List[str]] = {}
    for p, ks in kids.items():
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                oa, ob = own.get(a, {}), own.get(b, {})
                if not oa or not ob:
                    continue
                ta = max(oa, key=lambda m: oa[m][0])
                tb = max(ob, key=lambda m: ob[m][0])
                if ta != tb and ob.get(ta, (0, 0))[1] > 0 and oa.get(tb, (0, 0))[1] > 0 \
                        and oa[ta][0] > oa[ta][1] and ob[tb][0] > ob[tb][1]:
                    polar.setdefault(p, [])
                    polar[p] += [x for x in (a, b) if x not in polar[p]]

    # labels the panel can hardly define (little positive evidence on it)
    # testability = strongest positive marker on the panel
    top_w = {lab: max([w for w in sig.values() if w > 0] + [0.0]) for lab, sig in signatures.items()}
    weak = []
    if len(top_w) >= 3:
        med = sorted(top_w.values())[len(top_w) // 2]
        weak = [lab for lab, v in top_w.items() if v < 0.25 * med]
    gene_map_source = {m: ("override" if gene_overrides and m in gene_overrides
                           else "protein_name" if m in pn and pncache.get(pn[m])
                           else "remapped" if m in remapped else gsrc.get(m, ""))
                       for m in gm}
    gene_review = {m: [f"{g} ({gnames.get(g, '')})" for g in genes] for m, genes in gm.items()
                   if gene_map_source.get(m) not in ("symbol", "override")}
    defining = {lab: own_defining(lab) for lab in signatures}
    # review table for the panel-name -> gene mapping: how it was matched, how
    # often the atlas uses the gene as a marker, which candidate labels it defines
    defines: Dict[str, List[str]] = {}
    for lab, ms in defining.items():
        for m in ms:
            defines.setdefault(m, []).append(lab)
    gene_check = sorted(
        [dict(marker=m, genes=genes, names=[gene_names[m].get(g, "") for g in genes],
              match=("given" if (gene_overrides and m in gene_overrides) or m in pn
                     else "remapped" if m in remapped
                     else ("symbol" if (gsrc.get(m) or _match_kind(m, genes)) == "symbol"
                           else f"alias / name ({gsrc.get(m) or 'lookup'})")),
              atlas_usage=max([usage.get(g.upper(), 0) for g in genes] + [0]),
              defines=sorted(defines.get(m, [])))
         for m, genes in gm.items()],
        key=lambda r: (r["match"] != "alias / name", bool(r["defines"]), r["atlas_usage"]))
    save()
    return dict(signatures=signatures, parents=parents, polar=polar, defining=defining,
                resolved=res,
                cl_ids=cl_of, unresolved_labels=unresolved_labels,
                out_of_context=out_of_context, context_scopes=ctx_scopes,
                gene_map=gm, gene_names=gene_names, gene_check=gene_check,
                remapped_markers=remapped,
                unresolved_markers=miss,
                empty_signatures=empty, weak_signatures=weak,
                gene_map_source=gene_map_source, gene_review=gene_review,
                evidence=evidence)
