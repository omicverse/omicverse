r"""Proposed admissibility decisions for meta-analysis candidate comparisons.

:func:`propose_selection` turns the three inputs a reviewer would use -- the
comparison workbook of the source articles, the method paper's measure-defining
sentences and the abbreviations defined in the articles / method -- into one
proposed decision per workbook row (``keep`` / ``exclude`` / ``review``) with
the reason and the method sentence it rests on.  The reviewer then only reads
the ``review`` rows instead of deciding every row by judgement.

How a row is decided
--------------------
1. **Task identity.** Row text (section path + row label, abbreviations
   expanded) is matched to the method statements by IDF-weighted shared
   content words (simple stemming). The best statement names the task family;
   ``coverage`` is the share of the row's task words the statement explains.
2. **Measure.** The measure clause of the statement ("the dependent measures
   are X and Y", "X (when reported), or Y (when X is not reported)") is parsed
   into measure *types* (speed, accuracy/errors) and qualifier words
   ("perseverative", "between-search", "moves", ...). A row is kept when its
   measure type or qualifier is one the clause names; a row with no measure
   word at all is the task's single score; a row naming another measure is
   excluded; a fallback measure is kept only when the study has no row of the
   preferred measure.
3. **Coding rule** (``prefer_counts``): within a study and family, percentage
   rows of the accuracy/error type are dropped when count rows of that type
   exist.
4. Uncertain identity (partial coverage, task words the method never uses,
   measure rows without any task name) is ``review``, never silently excluded.

The vocabulary of measure types (time / latency / RT; accuracy / correct /
errors / hits / false alarms / omissions / commissions) is generic
psychometric vocabulary; task knowledge comes only from the method text and
the articles' own abbreviation definitions.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._registry import register_function
from ._meta_analysis import (comparison_workbook, extract_group_stats, meta_effect_sizes,
                             method_measure_statements, _strip_tags, _CONTROL_DEFAULT)

__all__ = ["abbreviation_map", "propose_selection", "reconstruct_study_effects", "write_study_effects",
           "record_deliverable", "verify_deliverable"]

_STOP = set("""a an the of and or in on for to with by from as at is are was were be been this that these
those which who it its their than then also both each either per vs versus into within across between
after before during over under not no all any most more less other such only one two three first second
mean means sd s.d se n m total score scores raw t task tasks test tests condition conditions
measure measures number numbers percent percentage participant participants subject subjects group groups
control controls patient patients study studies analysis analyzed reported report reporting value values
used using use k p d effect size sizes composite individually included include includes including
variable variables dependent primary common most reported version versions et al table note response responses
completion characteristic characteristics there upon significant significantly effect effects difference
differences found observed showed compared higher lower greater average""".split())

_SPEED = r"\b(rt|rts|reaction|latenc\w*|response time\w*|time\w*|speed|ms|msec|millisecond\w*|second\w*|sec|duration|slow\w*)\b|\(s\)"
_ACC = (r"\b(accura\w*|correct\w*|hits?|errors?|error rates?|omissions?|commissions?|false[- ]?alarms?|"
        r"false a\.?|incorrect|mistakes?|misses|missing|failures?)\b")
_PCT = r"%|\bpercent\w*|\bproportion|\brate\b"
_MEASURE_CLAUSE = re.compile(
    r"(dependent (?:measure|variable)s?(?: reported)?(?: (?:is|are|was|were|include[sd]?))?|primary (?:dependent )?measures?"
    r"(?: reported)?(?: [^.;:]{0,60}?)?(?: is| are| was| were)|(?:the )?measures? (?:is|are|was|were|include[sd]?)"
    r"|dependent variables?:?)(?P<clause>[^.;]*)", re.I)


def _stem(w: str) -> str:
    w = w.lower()
    for suf in ("ings", "ing", "ies", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    return w


def _words(text: str) -> List[str]:
    # "Benton's FAS": a possessive capitalised word is an author / test-owner name, not a task word
    text = re.sub(r"\b[A-Z][a-z]+['’]s\b", " ", str(text))
    text = re.sub(r"['’]s\b", "", text)
    return [w for w in re.findall(r"[A-Za-z][A-Za-z']+", text)]


def _content(text: str) -> List[str]:
    out = []
    text = re.sub(_ACC, " ", str(text), flags=re.I)
    text = re.sub(_SPEED, " ", text, flags=re.I)
    # joined forms of hyphen / slash compounds ("go/no-go" -> "gonogo", "nogo")
    for comp in re.findall(r"[A-Za-z]+(?:[/-][A-Za-z]+)+", text):
        parts = re.split(r"[/-]", comp)
        out.append("".join(parts).lower())
        if len(parts) > 2:
            out.append("".join(parts[1:]).lower())
    for w in _words(text):
        lw = w.lower().strip("'")
        if lw in _STOP or len(lw) < 2:
            continue
        if re.fullmatch(_SPEED, lw) or re.fullmatch(_ACC, lw) or re.fullmatch(r"\w*%", lw):
            continue
        out.append(_stem(lw))
    return out


_ABBR_DEF = re.compile(r"((?:[A-Za-z][\w'’-]*\s+){0,7}[A-Za-z][\w'’-]*)\s*\(\s*(?:[A-Z][a-z]+['’]s\s+)?"
                       r"([A-Z][A-Za-z0-9/&-]{1,9})\s*[),;]")
_ABBR_EQ = re.compile(r"\b([A-Z][A-Z0-9/&-]{1,9})\s*[=:]\s*([A-Za-z][^;=.]{3,80})")


def _acronym(words: Sequence[str]) -> str:
    return "".join(w[0] for w in words if w and w[0].isupper())


@register_function(
    aliases=["缩写表", "abbreviation map", "acronym definitions from article text"],
    category="utils",
    description="Collect abbreviation definitions from article / method text: 'Full Name (ABBR)', 'ABBR = Full "
                "Name' (table notes), and acronyms of capitalised multi-word names; returns {ABBR: expansion}.",
    examples=["ab = ov.tl.abbreviation_map(open('/app/data/article.xml').read())"],
    related=["tl.propose_selection"],
)
def abbreviation_map(*texts: str) -> Dict[str, str]:
    """Return ``{abbreviation: expansion}`` collected from one or more texts (markup is stripped)."""
    out: Dict[str, str] = {}
    gen: Dict[str, str] = {}
    for text in texts:
        t = _strip_tags(str(text))
        # glossaries in table notes: "SCIP, Screen for ...; VLT-L, verbal learning test-learning; ..."
        for m in re.finditer(r"(?:^|[;.:]\s*|\s)([A-Z][A-Za-z]*[A-Z][A-Za-z0-9/&-]{0,8}),\s+([A-Za-z][^;=.,]{3,80}?)(?=\s*[;.]|$)", t):
            exp = m.group(2).strip()
            if exp.split()[0][:1].lower() == m.group(1)[0].lower():
                out.setdefault(m.group(1), exp)
        for m in re.finditer(r"(?:^|[;.,]\s*|\s)([A-Z][A-Z0-9]{1,5})\s+[–—]\s+([A-Z][A-Za-z][^;–—]{2,60}?)(?=\s*;|\s*\.|$)", t):
            out.setdefault(m.group(1), m.group(2).strip())
        for m in _ABBR_DEF.finditer(t):
            phrase, ab = m.group(1).split(), m.group(2)
            # keep the shortest tail of the phrase whose initials spell the abbreviation
            letters = re.sub(r"[^A-Za-z]", "", ab).lower()
            best = None
            for k in range(1, len(phrase) + 1):
                tail = phrase[-k:]
                ini = "".join(w[0].lower() for w in tail if w.lower() not in ("of", "and", "the", "to", "in", "for"))
                if ini == letters or (len(letters) >= 2 and ini.startswith(letters[:2]) and k >= len(letters) - 1):
                    best = " ".join(tail)
                    if ini == letters:
                        break
            out.setdefault(ab, best or " ".join(phrase[-max(2, len(letters)):]))
        for m in _ABBR_EQ.finditer(t):
            out.setdefault(m.group(1), m.group(2).strip())
        for m in re.finditer(r"\b([A-Z][a-z]+(?:[\s-]+(?:of|to|and|the|[A-Z][a-z]+)){1,6})", t):
            ws = re.split(r"[\s-]+", m.group(1))
            for k in range(len(ws) - 1):          # every tail starting at a capitalised word
                if not ws[k][:1].isupper():
                    continue
                ac = _acronym(ws[k:])
                if len(ac) >= 3:
                    gen.setdefault(ac, " ".join(ws[k:]))
    for k, v in gen.items():
        out.setdefault(k, v)
    return out


def _expand(text: str, abbr: Dict[str, str]) -> str:
    extra = []
    for tok in re.findall(r"\b[A-Z][A-Za-z0-9/&-]{1,9}\b", str(text)):
        for key in (tok, tok.split("-")[0], tok.upper()):
            if key in abbr:
                extra.append(abbr[key])
                break
    return str(text) + (" " + " ".join(extra) if extra else "")


_COMPOSITE = r"\b(index|indices|quotient|composite|factor score|scale score|full[- ]scale|iq)\b"


def _types(text: str) -> set:
    t = str(text).lower()
    out = set()
    if re.search(_COMPOSITE, t):
        out.add("composite index")
    if re.search(_SPEED, t):
        out.add("speed")
    if re.search(_ACC, t):
        out.add("accuracy/errors")
    return out


def _parse_measures(stmt: str) -> dict:
    """Measure clause(s) of a method statement -> preferred and fallback measures."""
    pref, fb = [], []
    for m in _MEASURE_CLAUSE.finditer(stmt):
        clause = m.group("clause")
        # "X (when reported), or Y (when X were not reported)"
        parts = re.split(r",?\s+or\s+(?=[^,]*\(when[^)]*not)", clause)
        pref.append(re.sub(r"\(when[^)]*\)", "", parts[0]))
        for p in parts[1:]:
            fb.append(re.sub(r"\(when[^)]*\)", "", p))
    if not pref:
        return {}
    def summ(chunks):
        txt = " ".join(chunks)
        quals = set(_content(txt)) - {"time", "limit", "complete", "condition", "participant"}
        # parts of a hyphenated qualifier ("no-cue") are not qualifiers on their own
        for comp in re.findall(r"[A-Za-z]+(?:[/-][A-Za-z]+)+", txt):
            quals -= {_stem(w.lower()) for w in re.split(r"[/-]", comp)}
        acr = {"".join(w[0] for w in _words(c) if w.lower() not in _STOP).lower() for c in re.split(r",| and ", txt)}
        return {"types": _types(txt), "quals": quals, "acronyms": {a for a in acr if len(a) >= 2}, "text": txt.strip()}
    return {"pref": summ(pref), "fallback": summ(fb) if fb else None}


@register_function(
    aliases=["提议纳入决策", "propose admissibility decisions", "meta-analysis selection proposals",
             "label workbook rows from method statements"],
    category="utils",
    description="Propose keep / exclude / review for every row of a comparison workbook from the method "
                "paper's measure-defining sentences: task identity by IDF-weighted word match (abbreviations "
                "expanded from the articles and method), measure by the statement's measure clause (speed vs "
                "accuracy/error types, qualifiers, 'X (when reported), or Y' fallbacks), counts preferred over "
                "percentages when asked. Returns family, statement, measure, decision, reason, coverage.",
    examples=["prop = ov.tl.propose_selection(wb, statements, abbreviations=ab, prefer_counts=True)",
              "prop[prop.decision == 'review']"],
    related=["tl.comparison_workbook", "tl.method_measure_statements", "tl.reconstruct_study_effects"],
)
def propose_selection(wb: pd.DataFrame, statements: pd.DataFrame, abbreviations: Optional[Dict[str, str]] = None,
                      prefer_counts: bool = True, min_coverage: float = 0.6,
                      study_abbreviations: Optional[Dict[str, Dict[str, str]]] = None,
                      study_texts: Optional[Dict[str, str]] = None,
                      text_cols: Sequence[str] = ("section", "row")) -> pd.DataFrame:
    """Return ``wb`` with ``family, statement, measure, measure_type, coverage, decision, reason``.

    ``statements``: output of :func:`method_measure_statements` (column ``sentence``).
    ``abbreviations``: {ABBR: expansion}, e.g. :func:`abbreviation_map` of the method + articles;
    ``study_abbreviations``: {study: {ABBR: expansion}} from each article (take precedence);
    ``study_texts``: {study: full text}, used to suggest the task of measure rows that carry no
    task name (e.g. "RT in ms" in a table of one task).
    """
    abbr = abbreviations or {}
    per_study = study_abbreviations or {}
    st = statements.reset_index(drop=True).copy()
    for c in ("context", "context_after"):
        if c not in st:
            st[c] = ""
    # results / discussion sentences (effect sizes, significance) are not coding rules
    st = st[~st["sentence"].str.contains(r"\bd\s*=\s*-?\d|significant|impair|\bfound\b|=\s*[A-Z]",
                                          case=False, regex=True)].reset_index(drop=True)
    incl = re.compile(r"(was|were) (included|analy[sz]ed) (individually|separately|as (a |an )?(non-\S+ )?"
                      r"(comparison )?measure)", re.I)
    st = st[~st["sentence"].str.contains(r"moderator|meta-regression|covariate|continuous variable|coded (as|for each)",
                                          case=False, regex=True)].reset_index(drop=True)

    def _meas(x):
        m = _parse_measures(x)
        if not m and incl.search(x):
            # the task is coded but no measure is named: its score(s)
            return {"pref": {"types": set(), "quals": set(), "acronyms": set(), "text": "(task coded; score)"},
                    "fallback": None}
        return m

    st["measures"] = st["sentence"].map(_meas)
    st = st[st["measures"].map(bool)].reset_index(drop=True)
    cur_tok = [set(_content(_expand(x, abbr))) for x in st["sentence"]]
    prev_tok = [set(_content(_expand(x, abbr))) for x in st["context"].fillna("")]
    next_tok = [set(_content(_expand(x, abbr))) for x in st["context_after"].fillna("")]
    ctx_tok = []
    # a measure sentence without task words ("The dependent measures are X and Y.") refers back to
    # the sentence before it, or -- when that names no task either -- forward to the next one
    for j in range(len(cur_tok)):
        if len(cur_tok[j]) < 3:
            cur_tok[j] = cur_tok[j] | prev_tok[j]
            ctx_tok.append(next_tok[j])
        else:
            ctx_tok.append(prev_tok[j] | next_tok[j])
    strong = [bool(re.search(r"dependent|primary", x, re.I)) for x in st["sentence"]]
    # coded test names (coding tables): all their words in a row identify the test outright
    test_name_tok = []
    for x in st["context"].fillna(""):
        mm = re.match(r"\s*(.+?) is a coded test\.", str(x))
        test_name_tok.append(set(_content(mm.group(1))) if mm else set())
    # statements that EXCLUDE ("... are not admissible", "is not coded", "were excluded")
    negative = [bool(re.search(r"\bnot\s+(?:admissible|coded|included|eligible|part of)|\bare excluded\b|"
                               r"\binadmissible\b", x, re.I)) for x in st["sentence"]]
    # task-name words of each statement: the sentence without its measure clause, plus context
    task_tok, cur_id = [], []
    for j, x in enumerate(st["sentence"]):
        mt_ = st.at[j, "measures"]["pref"]["text"]
        rest_ = set(_content(_expand(x.replace(mt_, " "), abbr)))
        task_tok.append(rest_ | ctx_tok[j])
        # identity words: the statement without the words only its measure clause uses
        cur_id.append(cur_tok[j] - (set(_content(mt_)) - rest_))
    qual_tok = [m["pref"]["quals"] | (m["fallback"]["quals"] if m.get("fallback") else set())
                | m["pref"]["acronyms"] for m in st["measures"]]
    df_ = Counter(t for a_, b_ in zip(cur_tok, ctx_tok) for t in (a_ | b_))
    nst = max(len(cur_tok), 1)
    idf = lambda t: math.log((nst + 1) / (df_.get(t, 0) + 0.5))

    def _subseq(a, b):
        it = iter(b)
        return all(ch in it for ch in a)

    def hit(t, pool):
        if t in pool:
            return True
        if len(t) >= 4 and any(len(u) >= 4 and (u.startswith(t) or t.startswith(u)) for u in pool):
            return True
        # abbreviated words ("persv" for "perseverative")
        return 4 <= len(t) <= 6 and any(len(u) > len(t) and u[0] == t[0] and _subseq(t, u) for u in pool)

    df = wb.copy()
    rec = []
    _art_cache: Dict[str, set] = {}
    for i in df.index:
        sec = str(df.at[i, "section"]) if "section" in df else ""
        inner = sec.split(">")[-1] if sec and sec != "nan" else ""
        row = str(df.at[i, "row"])
        lab = f"{inner} {row}"
        # per-study abbreviations take precedence
        ab_i = abbr
        if per_study and "study" in df:
            ab_i = {**abbr, **per_study.get(df.at[i, "study"], {})}
        rtypes = _types(_expand(row, ab_i))
        art_toks = set()
        if study_texts and "study" in df:
            sid_ = df.at[i, "study"]
            if sid_ not in _art_cache:
                _art_cache[sid_] = set(_content(_strip_tags(study_texts.get(sid_, "") or "")[:400000]))
            art_toks = _art_cache[sid_]
        # concepts: plain words, and abbreviations with their expansion words
        concepts = []
        caps = {t.lower() for t in re.findall(r"\b[A-Z]{2,}[A-Z0-9]*\b", lab)}
        caps |= {t.lower() for t in re.findall(r"\b([A-Z]{2,})\d", lab)}
        for tok in re.findall(r"\b[A-Z][A-Za-z0-9/&-]{1,9}\b", lab):
            if tok.lower() in _STOP or re.fullmatch(_SPEED, tok.lower()) or re.fullmatch(_ACC, tok.lower()):
                continue
            key = next((k for k in (tok, tok.split("-")[0]) if k in ab_i), None)
            if key and tok.upper() == tok:
                concepts.append((tok.lower(), set(_content(ab_i[key])) | {tok.lower()}, True))
        abbr_words = {c for c, _, _ in concepts}
        for c in list(abbr_words):          # "VLT-L" also yields the words "vlt", "vltl"
            parts_ = [x for x in re.split(r"[-/]", c) if x]
            abbr_words |= set(parts_) | {"".join(parts_)} | {_stem(x) for x in parts_}
        for w in set(_content(lab)):
            if w not in abbr_words and not any(w in e for _, e, _ in concepts):
                concepts.append((w, {w}, False))
        if not concepts:
            why, dec0, base0 = "no task words", ("review" if rtypes else "exclude"), {}
            if rtypes:
                why = "a measure row without a task name: identify the task in the article"
                txt = (study_texts or {}).get(df.at[i, "study"]) if "study" in df else None
                if txt:
                    atoks = Counter(_content(_strip_tags(txt)))
                    cand = sorted(((sum(idf(t) * min(atoks.get(t, 0), 3) for t in (cur_tok[j] | ctx_tok[j]))
                                    / (1 + len(cur_tok[j] | ctx_tok[j])), j) for j in range(len(cur_tok))
                                   if rtypes & st.at[j, "measures"]["pref"]["types"]), reverse=True)[:2]
                    def _desc(j):
                        ctxs = " ".join(x for x in (st.at[j, "context"], st.at[j, "context_after"]) if x)
                        return f"'{st.at[j, 'sentence'][:90]}' [about: {ctxs[:160]}]"
                    why += ("; the article's text best matches the method sentence(s): " +
                            " || ".join(_desc(j) for _, j in cand) +
                            " -- keep if the article's task is that task and the sentence names this statistic")
                    if cand and cand[0][0] > 0:
                        j0 = cand[0][1]
                        m0 = st.at[j0, "measures"]["pref"]
                        clear = len(cand) < 2 or cand[0][0] >= 1.1 * cand[1][0]
                        if strong[j0] and clear and rtypes & m0["types"]:
                            dec0 = "keep"
                            why = (f"task inferred from the article text (measure row without a task name): "
                                   f"'{st.at[j0, 'sentence'][:90]}' -- check it is the article's task")
                            base0 = dict(statement=st.at[j0, "sentence"][:300], measure=m0["text"][:120])
            rec.append({**dict(family="", statement="", measure="", measure_type=",".join(sorted(rtypes)),
                               coverage=0.0, decision=dec0, reason=why), **base0})
            continue
        weight = lambda forms: max(idf(f) for f in forms)

        def frac(forms, pool):
            fs = [f for f in forms if f not in abbr_words] or list(forms)
            return sum(hit(f, pool) for f in fs) / len(fs)

        best = None
        lab_tok = set(_content(_expand(lab, ab_i)))
        for j in range(len(cur_tok)):
            if test_name_tok[j] and test_name_tok[j] <= lab_tok and not negative[j]:
                sc_ = 100.0 + len(test_name_tok[j])          # longest fully named test wins
                if best is None or sc_ > best[0]:
                    best = (sc_, j, 1.0, [])
                continue
            if best is not None and best[0] >= 100:
                continue
            sc, cov_w, tot_w, unm = 0.0, 0.0, 0.0, []
            for name, forms, is_ab in concepts:
                w_ = weight(forms)
                unknown_code = (name in caps) and not is_ab      # an undefined code like "NM3M"
                tot_w += (0.1 if re.search(r"\d", lab.split(name.upper())[-1][:3] if name.upper() in lab else "")
                          else 0.3) * w_ if unknown_code else w_
                need = (0.25 if len(forms) >= 4 else 0.5) if is_ab else 1.0
                f1 = 1.0 if (is_ab and name in cur_tok[j]) else frac(forms, cur_tok[j])
                f2 = frac(forms, cur_tok[j] | ctx_tok[j])
                # coded test names (coding tables) are exact: a partly matching abbreviation is partly covered
                full = 1.0 if (not is_ab or not test_name_tok[j] or name in cur_tok[j] or name in ctx_tok[j]) else f2
                if f1 >= need:
                    sc += w_ * f1; cov_w += w_ * full
                elif f2 >= need:
                    sc += 0.6 * w_ * f2; cov_w += w_ * full
                elif any(hit(f, qual_tok[j]) for f in forms):
                    cov_w += 0.3 * w_ if unknown_code else w_
                else:
                    unm.append(name)
            sc *= 1.0 if strong[j] else 0.7
            # the article's own subject breaks ties between similarly named tests of the method
            if sc > 0 and art_toks:
                extra = [idf(t) for t in cur_tok[j] if t in art_toks and not any(t in f for _, f, _ in concepts)]
                sc += 0.25 * (max(extra) if extra else 0.0)
            if sc > 0 and (best is None or sc > best[0]) and not (best is not None and best[0] >= 100):
                best = (sc, j, cov_w / tot_w if tot_w else 0.0, unm)
        if best is None:
            rec.append(dict(family="", statement="", measure="", measure_type=",".join(sorted(rtypes)), coverage=0.0,
                            decision="exclude",
                            reason="no method statement names this task (outside the method's scope)"))
            continue
        sc, j, cov, unm = best
        if inner:
            # the section may be a domain heading while the row names the task itself
            lab_only = row.split(": ")[-1]
            rset = set(_content(_expand(lab_only, ab_i))) | {t.lower() for t in re.findall(r"[A-Z]{2,}", lab_only)}
            rconc = [(n, f) for n, f, _ in concepts if n in rset]
            if rconc and (len(rconc) >= 2 or any(n in caps for n, _ in rconc)):
                tot_r = sum(max(idf(x) for x in f) for _, f in rconc)
                um_r = [n for n, f in rconc if n in unm]
                cov_r = 1 - sum(max(idf(x) for x in f) for n, f in rconc if n in unm) / tot_r if tot_r else 0
                if cov_r > cov:
                    cov, unm = cov_r, um_r
        meas = st.at[j, "measures"]
        stmt = st.at[j, "sentence"]
        fam = (str(st.at[j, "section"]) + ": " + stmt[:60]).strip()
        codes = {c for c in caps if c not in {n for n, _, ab_ in concepts if ab_}}
        codes = {c for c in codes if not hit(c, qual_tok[j])}
        mwords = ((set(unm) & set(_content(row))) | (codes & set(re.findall(r"[a-z0-9]+", row.lower())))) \
            - {"part", "section", "list", "block", "level"}
        # a hyphen-suffixed code ("VLT-L" = "verbal learning test-learning") names its measure in the suffix
        row_x = row
        for tok in re.findall(r"\b[A-Z][A-Za-z0-9]*-[A-Za-z0-9]+\b", row):
            if tok in ab_i and "-" in ab_i[tok]:
                row_x += " " + ab_i[tok].split("-")[-1]
        rowc = set(_content(re.sub(r"[-/]", " ", row_x))) | set(re.findall(r"[a-z]+", row_x.lower()))

        row_low = row.lower()
        row_cmp = rowc | {a + b for a, b in zip(re.findall(r"[a-z]+", row_low), re.findall(r"[a-z]+", row_low)[1:])}

        def fits(m):
            if not m:
                return False
            mt = m["text"].lower()
            if re.fullmatch(r"\s*(the\s+)?(total\s+)?scores?\s*", mt):
                return True              # the coded test's score, whatever the article calls it
            # numbered conditions the measure names ("part 3") must match the row's
            conds = set(re.findall(r"\b(part|section|trial|block|list|condition|level|phase|session)\s*(\d+|[a-z])\b", mt))
            rconds = set(re.findall(r"\b(part|section|trial|block|list|condition|level|phase|session)\s*(\d+|[a-z])\b", row_low))
            if conds and rconds and not (conds & rconds):
                return False
            # a measure counting correct responses does not cover error / omission counts
            if re.search(r"\bcorrect", mt) and not re.search(r"error|accura|incorrect", mt) and \
                    re.search(r"error|omission|missing|incorrect|false|miss\b", row_low) and not re.search(r"\bcorrect", row_low):
                return False
            # condition qualifiers ("incongruent no-cue"): every condition part of the row must be one of them
            cq = {q for q in m["quals"] if len(q) >= 4}
            if len(cq) >= 2 and any(hit(q, row_cmp) for q in cq) and ": " in row:
                for part in row.split(": "):
                    if re.search(r"\b[A-Z]{2,}", part):
                        continue                     # an abbreviated test name, not a condition
                    pw = set(_content(part)) | {a + b for a, b in zip(re.findall(r"[a-z]+", part.lower()),
                                                                     re.findall(r"[a-z]+", part.lower())[1:])}
                    if not pw or any(hit(w, task_tok[j]) for w in set(_content(part))) or _types(part):
                        continue
                    if not any(hit(q, pw) for q in cq):
                        return False
            if rtypes & m["types"]:
                return True
            if not m["types"] and re.search(r"\bcorrect\b", row, re.I) and not re.search(_PCT, row):
                return True          # "total / number correct" is the count the task's measure names
            if any(hit(q, rowc) for q in m["quals"]):
                return True
            ac = {re.sub(r"[^a-z]", "", w.lower()) for w in re.findall(r"\b[A-Z]{2,5}\b", row)}
            return bool(ac & m["acronyms"])

        if negative[j]:
            rec.append(dict(family=fam, statement=stmt[:300], measure="", measure_type=",".join(sorted(rtypes)),
                            coverage=round(cov, 2), decision="exclude",
                            reason=f"the coding rules exclude it: {stmt[:160]}"))
            continue
        derived = re.search(r"\b(decrement|minus|difference score|change score|gain score)\b", row, re.I)
        if derived and not re.search(re.escape(derived.group(1)), meas["pref"]["text"], re.I):
            rec.append(dict(family=fam, statement=stmt[:300], measure=meas["pref"]["text"][:120],
                            measure_type=",".join(sorted(rtypes)), coverage=round(cov, 2), decision="exclude",
                            reason="a derived difference score (not one of the coded scores)"))
            continue
        if "composite index" in rtypes:
            rec.append(dict(family=fam, statement=stmt[:300], measure=meas["pref"]["text"][:120],
                            measure_type=",".join(sorted(rtypes)), coverage=round(cov, 2), decision="exclude",
                            reason="a composite / index score of a test battery, not a task's score"))
            continue
        base = dict(family=fam, statement=stmt[:300], measure=meas["pref"]["text"][:120],
                    measure_type=",".join(sorted(rtypes)), coverage=round(cov, 2))
        hidden = [u for u in unm if u in caps and u not in {n for n, _, ab_ in concepts if ab_}]
        if cov < 0.45 and not (hidden and cov >= 0.15):
            rec.append({**base, "decision": "exclude",
                        "reason": f"task words {sorted(unm)} are not in any method statement (outside scope)"})
            continue
        if fits(meas["pref"]):
            dec, why = "keep", f"measure named by the method ({meas['pref']['text'][:80]})"
        elif meas.get("fallback") and fits(meas["fallback"]):
            dec, why = "fallback", f"fallback measure ({meas['fallback']['text'][:80]})"
        elif not rtypes and not mwords and (not meas["pref"]["types"] or sc >= 100):
            dec, why = "keep", "the task's single score"
        elif not meas["pref"]["types"] and not meas["pref"]["quals"] and rtypes and not mwords - {"section", "part"}:
            desc = " ".join(x for x in (st.at[j, "context"], st.at[j, "context_after"]) if x)[:220]
            dec, why = "review", ("the method codes this task but names no measure for it: keep the statistics of "
                                  "the condition the method describes as the task's critical condition "
                                  f"(method text: {desc})")
        else:
            dec, why = "exclude", f"statistic not among the method's measures for this task ({meas['pref']['text'][:80]})"
        if dec in ("keep", "fallback") and cov < min_coverage:
            dec, why = "exclude", (f"a variant / different task: {sorted(unm)} not in the method statement "
                                   f"(coverage {cov:.2f}) -- the method does not name it")
        rec.append({**base, "decision": dec, "reason": why})
    R = pd.DataFrame(rec, index=df.index) if rec else pd.DataFrame(
        columns=["family", "statement", "measure", "measure_type", "coverage", "decision", "reason"])
    for c in R:
        df[c] = R[c]
    sid = df["study"] if "study" in df else pd.Series("", index=df.index)
    # fallbacks: kept only when the study has no preferred-measure row of that family
    for i in df.index[df["decision"] == "fallback"]:
        same = (sid == sid[i]) & (df["statement"] == df.at[i, "statement"]) & (df["decision"] == "keep")
        df.at[i, "decision"] = "exclude" if same.any() else "keep"
        df.at[i, "reason"] += " -- not used: preferred measure reported" if same.any() else " -- used: preferred measure absent"
    if prefer_counts:
        rows_ = df["row"].astype(str)
        pct = rows_.str.contains(_PCT, case=False, regex=True)
        inner = (df["section"].fillna("").astype(str).str.split(">").str[-1] if "section" in df
                 else pd.Series("", index=df.index))
        acc = df["measure_type"].str.contains("accuracy", na=False)
        for i in df.index[df["decision"].isin(["keep", "review"]) & pct & acc]:
            same = ((sid == sid[i]) & acc & ~pct & (inner == inner[i]) & (df["source"] == df.at[i, "source"]
                    if "source" in df else True) & df["decision"].isin(["keep", "review"]))
            if same.any():
                df.at[i, "decision"] = "exclude"
                df.at[i, "reason"] = "percentage of an accuracy/error measure while counts are reported (counts coded)"
    df["proposed"] = df["decision"]
    return df


@register_function(
    aliases=["研究效应重建", "reconstruct study effects", "meta-analysis one call", "study-level effects pipeline"],
    category="utils",
    description="One call from articles + method paper to proposed study-level effects: comparison_workbook "
                "over all article files, method_measure_statements of the method paper, abbreviation_map of all "
                "texts, propose_selection, and meta_effect_sizes for the kept rows. Prints the decisions grouped "
                "by study (keep / review / exclude with reasons) and returns (proposals, effects).",
    examples=["prop, eff = ov.tl.reconstruct_study_effects(src, '/app/data/method.xml', case=r'depress|MDD')",
              "prop[prop.decision == 'review']"],
    related=["tl.propose_selection", "tl.comparison_workbook", "tl.meta_effect_sizes"],
)
def reconstruct_study_effects(sources: dict, method_source: Union[str, Path], case=None, control=None, exclude=None,
                              n_map: Optional[dict] = None, prefer_counts: bool = True,
                              out_dir: Optional[str] = None, out: Optional[str] = None,
                              case_label: str = "case", out_columns: Optional[Sequence[str]] = None,
                              fetch_missing: bool = True, correction: str = "approx", verbose: bool = True):
    """Return ``(proposals, effects)``.

    ``proposals``: every workbook row with family, measure, decision (keep / exclude / review),
    reason and the method sentence used. ``effects``: :func:`meta_effect_sizes` of the ``keep``
    rows (columns ``n_case, mean_case, sd_case, n_control, mean_control, sd_control, d``).
    ``case=None`` treats every group that is not a control group as a case group (subgroups pooled;
    use ``exclude`` for groups of another diagnosis). ``prop.attrs['study_summary']`` (also printed)
    lists per study the statistics found, comparisons formed and the action needed (PDF only,
    groups not matched, unknown n). ``out_dir``: also write ``proposals.csv`` there. ``out``: write
    the deliverable (the ``keep`` rows, via :func:`write_study_effects`, columns renamed with
    ``case_label`` and ordered as ``out_columns``) -- rewrite it with :func:`write_study_effects`
    after resolving the review rows, never by hand. ``fetch_missing``: for a study (keyed by its
    DOI) whose files are PDF only or yield no comparison, try to download a full text with real
    tables (Europe PMC JATS XML, then the publisher page via doi.org) next to its files and use it;
    PDF tables printed column by column are parsed as well. Review the ``review`` rows (and spot-check
    the others) against the articles; change a decision in ``proposals`` and recompute with
    ``meta_effect_sizes(proposals[proposals.decision == 'keep'], case='case')``.
    """
    sources = {k: ([v] if isinstance(v, (str, Path)) else list(v)) for k, v in sources.items()}
    if fetch_missing:
        for sid, paths in sources.items():
            if not re.match(r"10\.\d{4,9}/\S+", str(sid)):
                continue
            pdf_only = all(str(p).lower().endswith(".pdf") for p in paths)
            nstat = 0
            if not pdf_only:
                for pth in paths:
                    try:
                        nstat += int((extract_group_stats(pth)["layout"] != "text").sum())
                    except Exception:
                        pass
            if pdf_only or nstat == 0:
                got = _fetch_fulltext(str(sid), Path(str(paths[0])).parent if paths else Path("."))
                if got:
                    if verbose:
                        print(f"[{sid}] fetched a full text with tables: {got}")
                    sources[sid] = [got] + paths
    kw = {} if control is None else {"control": control}
    # groups the method / instruction leaves out ("participants still in the acute phase ... are not part
    # of the case group: leave them out"): words of such sentences that name a group in an article
    try:
        _mtxt = Path(str(method_source)).read_text(errors="ignore") if len(str(method_source)) < 4096 else str(method_source)
    except Exception:
        _mtxt = str(method_source)
    excl_words = set()
    excl_sents = []
    for sent in re.split(r"(?<=[.])\s+", _strip_tags(_mtxt)):
        if re.search(r"\b(?:leave (?:them|it) out|left out|not part of|are excluded|is excluded|exclude[sd]? (?:them|those))\b",
                     sent, re.I) and re.search(r"\b(?:participants?|patients?|subjects?|subgroups?|groups?|sample)\b", sent, re.I):
            excl_words |= {w.lower() for w in re.findall(r"[A-Za-z]{4,}", sent)}
            excl_sents.append(sent.strip())
    excl_words -= {w.lower() for w in re.findall(r"[A-Za-z]+", str(case_label))}
    excl_words -= {"participants", "participant", "patients", "subjects", "group", "groups", "subgroup", "subgroups",
                   "sample", "they", "them", "when", "were", "still", "part", "leave", "left", "remaining",
                   "pool", "article", "reports", "separate", "excluded", "exclude", "tested", "test", "defines",
                   "described", "classified", "early", "illness", "with", "from", "that", "this", "those", "other"}
    auto_excl = r"\b(?:" + "|".join(sorted(map(re.escape, excl_words))) + r")\b" if excl_words else None
    # columns that are not groups of people (norm-based expectations, test statistics) never pair
    non_group = (r"\bexpected\b|\bpredicted\b|\bestimated\b|p[- ]?value|^\s*p\W*$|^\s*[tFz]\s*$|statistic|effect size"
                 r"|^\s*(?:\[control\]\s*)?(?:all|total|overall|whole sample|full sample|entire sample)\b")
    if auto_excl:
        non_group = f"{non_group}|{auto_excl}"
    if isinstance(exclude, dict):
        exclude = {k: (f"(?:{exclude[k]})|{non_group}" if exclude.get(k) else non_group) for k in sources}
    else:
        exclude = f"(?:{exclude})|{non_group}" if exclude else non_group
    # group sizes printed in one place of an article (a table header, an "N" row, the text
    # "acute group (n = 15)") apply to the same group named elsewhere in it
    user_n = set(n_map or {})
    n_map = {k: dict(v) for k, v in (n_map or {}).items()}
    case_by_study = {}
    for sid, paths in sources.items():
        found: Dict[str, set] = {}
        names = set()
        tab_sets = []
        text = ""
        for pth in ([paths] if isinstance(paths, (str, Path)) else paths):
            try:
                led = extract_group_stats(pth)
            except Exception:
                continue
            names |= set(led["group"].astype(str))
            tab_sets.extend(set(tg["group"].astype(str)) for _, tg in led[led["layout"] != "text"].groupby("table"))
            for g_, nh in zip(led["group"].astype(str), led["n_header"]):
                if pd.notna(nh):
                    base = re.split(r"\(|\bn\s*=|\bN\s*=", g_)[0].strip(" :,")
                    if 1 <= len(base) <= 40:
                        found.setdefault(base, set()).add(int(nh))
            try:
                if not str(pth).lower().endswith(".pdf"):
                    text += " " + _strip_tags(Path(str(pth)).read_text(errors="ignore"))
            except Exception:
                pass
        text_found: Dict[str, set] = {}
        for name, n in _named_sizes_in_text(text):
            text_found.setdefault(name, set()).add(n)
        auto = {}
        for g_ in names:
            if re.search(r"\bn\s*=", g_, re.I):
                continue
            n_ = _match_group_size(g_, found)
            if n_ is None:
                n_ = _match_group_size(g_, text_found)
            if n_ is not None:
                auto[rf"^\s*(?:\[control\]\s*)?{re.escape(g_.strip())}\s*$"] = n_
        auto.update({rf"(?:^|[\]\s(]){re.escape(b)}\b": next(iter(v)) for b, v in found.items() if len(v) == 1})
        if auto:
            n_map[sid] = {**auto, **n_map.get(sid, {})}
        # the case group is the one the task names (case_label), when an article has such a column
        if case is None and case_label and case_label != "case":
            lab = re.escape(case_label)
            ctl = control if isinstance(control, str) else _CONTROL_DEFAULT
            exc = exclude.get(sid) if isinstance(exclude, dict) else exclude
            keep_ = lambda g_: not (exc and re.search(exc, g_, re.I))
            alts = [lab]
            labelled = False
            for gs in tab_sets:
                kept = [g_ for g_ in gs if keep_(g_)]
                if any(re.search(lab, g_, re.I) and not re.search(ctl, g_, re.I) for g_ in kept):
                    labelled = True        # in this table only the label-named group is the case group
                else:                      # a table of subgroups not named by the label: all are cases
                    alts += [re.escape(g_.strip()) for g_ in kept if not re.search(ctl, g_, re.I)]
            if labelled:
                case_by_study[sid] = "|".join(dict.fromkeys(alts))
    if case is None:
        case = {sid: case_by_study.get(sid, r".") for sid in sources}   # else every non-control group
    wb = comparison_workbook(sources, case=case, exclude=exclude, n_map=n_map, verbose=False, **kw)
    # studies whose tables print no group sizes at all: take them from the text ("Forty-four
    # patients ... and 44 healthy comparison subjects"), reported in the summary
    inferred_n = {}
    if len(wb):
        for sid, g in wb.groupby("study"):
            if g["n_case"].notna().any() or g["n_control"].notna().any() or sid in user_n:
                continue
            txt = ""
            for pth in sources[sid]:
                try:
                    txt += _strip_tags(Path(str(pth)).read_text(errors="ignore")) if not str(pth).lower().endswith(
                        ".pdf") else ""
                except Exception:
                    pass
            nc, nk = _sample_sizes_from_text(txt)
            if nc and nk:
                cg = {c.strip() for x in g["case_groups"].astype(str) for c in x.split(" + ")}
                kg = set(g["control_group"].astype(str))
                n_map[sid] = {**n_map.get(sid, {}),
                              **{rf"(?:^|[\]\s(]){re.escape(c)}\b": nc for c in cg},
                              **{rf"(?:^|[\]\s(]){re.escape(c)}\b": nk for c in kg}}
                inferred_n[sid] = (nc, nk)
        if inferred_n:
            wb = comparison_workbook(sources, case=case, exclude=exclude, n_map=n_map, verbose=False, **kw)
    if wb.empty:
        wb = pd.DataFrame(columns=["study", "source", "table", "section", "row", "layout", "case_groups",
                                   "case_subgroups", "control_group", "n_case", "mean_case", "sd_case",
                                   "n_control", "mean_control", "sd_control", "d", "family", "stat_class"])
    st = method_measure_statements(method_source)
    try:
        mtext = Path(str(method_source)).read_text(errors="ignore") if len(str(method_source)) < 4096 else str(method_source)
    except Exception:
        mtext = str(method_source)
    stexts: Dict[str, str] = {}
    for sid, paths in sources.items():
        parts = []
        for p in ([paths] if isinstance(paths, (str, Path)) else paths):
            try:
                if str(p).lower().endswith(".pdf"):
                    from pypdf import PdfReader
                    parts.append("\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages))
                else:
                    parts.append(Path(str(p)).read_text(errors="ignore"))
            except Exception:
                pass
        stexts[sid] = "\n".join(parts)
    ab = abbreviation_map(mtext)
    sab = {sid: abbreviation_map(t) for sid, t in stexts.items()}
    prop = propose_selection(wb, st, abbreviations=ab, prefer_counts=prefer_counts, study_abbreviations=sab,
                             study_texts=stexts)
    prop = _aggregate_vs_subgroups(prop, stexts, excl_sents)
    prop = _check_reported_effects(prop, stexts)
    prop["proposed"] = prop["decision"]
    prop["stats_key"] = [_stats_key(r) for _, r in prop.iterrows()]
    kept = prop[(prop["decision"] == "keep")]
    kept = kept[kept[["n_case", "mean_case", "sd_case", "n_control", "mean_control", "sd_control"]].notna().all(axis=1)]
    eff = meta_effect_sizes(kept, case="case", control="control") if len(kept) else kept
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        prop.drop(columns=["case_subgroups"], errors="ignore").to_csv(Path(out_dir) / "proposals.csv", index=False)
    if out:
        prop.attrs["correction"] = correction
        write_study_effects(prop.assign(decision=prop["decision"].replace({"review": "exclude"}),
                                        proposed=prop["decision"].replace({"review": "exclude"})),
                            out, case_label=case_label, columns=out_columns, correction=correction)
    # per-study health check: statistics found vs comparisons formed
    summ = []
    for sid, paths in sources.items():
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        groups, nstat = Counter(), 0
        for pth in paths:
            try:
                led = extract_group_stats(pth)
            except Exception:
                continue
            nstat += len(led)
            groups.update(led.loc[led["layout"] != "text", "group"].astype(str))
        g = prop[prop["study"] == sid] if len(prop) else prop
        msgs = []
        if all(str(p).lower().endswith(".pdf") for p in paths) and not len(g):
            msgs.append("PDF only and no table could be read: get the XML / HTML full text and pass it")
        if nstat and not len(g):
            msgs.append(f"{nstat} statistics found but no case-vs-control pair: group names are "
                        f"{[k for k, _ in groups.most_common(8)]} -- pass case / control for this study")
        if not nstat:
            msgs.append("no statistics found in the file(s): wrong file, or a format to transcribe by hand")
        elif not len(g) or not (g["decision"] == "keep").any() if len(g) else True:
            if not len(g):
                msgs.append("no case-vs-control comparison found: if the article puts its group statistics in "
                            "supplementary files, download them and add them to this study's file list")
        if sid in inferred_n:
            msgs.append(f"group sizes taken from the text: case {inferred_n[sid][0]}, control {inferred_n[sid][1]} "
                        f"(check against the participants section)")
        if len(g) and (g["n_case"].isna() | g["n_control"].isna()).any():
            msgs.append("some rows have n=? (sizes not printed for these groups): add them via n_map only if "
                        "the article gives sizes for exactly these groups")
        summ.append({"study": sid, "statistics": nstat, "comparisons": len(g),
                     "keep": int((g["decision"] == "keep").sum()) if len(g) else 0,
                     "review": int((g["decision"] == "review").sum()) if len(g) else 0,
                     "action": "; ".join(msgs)})
    summ = pd.DataFrame(summ)
    prop.attrs["study_summary"] = summ
    if verbose:
        for sid, g in prop.groupby("study", sort=False):
            print(f"== {sid}")
            for i, r in g.iterrows():
                nn = "n=?" if pd.isna(r["n_case"]) or pd.isna(r["n_control"]) else f"{r['n_case']:g}/{r['n_control']:g}"
                print(f"  [{i}] {r['decision']:7s} {str(r['row'])[-60:]} | {nn} | {r['reason'][:110]}")
        print("\nPER-STUDY CHECK (fix every 'action' and call again before reviewing):")
        print(summ.to_string(index=False))
        if out:
            nrev = int((prop.decision == "review").sum())
            todo = [a for a in summ["action"] if a] if len(summ) else []
            if nrev == 0 and not todo:
                print(f"FINAL: the deliverable {out} is complete and final. Do not edit it; finish with "
                      f"ov.tl.verify_deliverable('{out}') (must say 'unchanged').")
            else:
                print(f"DELIVERABLE written to {out} from the keep rows. Remaining: {nrev} review rows"
                      + (f", {len(todo)} per-study actions" if todo else "") + ". Resolve them, then rewrite ONLY "
                      f"with ov.tl.write_study_effects(prop, '{out}', case_label='{case_label}', columns=...) and "
                      f"finish with ov.tl.verify_deliverable('{out}'). Never add, drop or edit rows by hand.")
        print(f"keep {int((prop.decision == 'keep').sum())}, review {int((prop.decision == 'review').sum())}, "
              f"exclude {int((prop.decision == 'exclude').sum())}; rows with unknown n are not in the effects "
              f"until n_map supplies them")
    return prop, eff


@register_function(
    aliases=["写出研究效应", "write study effects", "deliver meta-analysis rows"],
    category="utils",
    description="Write the deliverable from reviewed proposals: every 'keep' row with complete statistics, "
                "Hedges' g recomputed, columns renamed to the task's header. Refuses unresolved 'review' rows "
                "and changes of a proposed keep / exclude that carry no cited reason (overrides={index: "
                "'quoted method / instruction sentence'}).",
    examples=["ov.tl.write_study_effects(prop, '/app/results/study_effects.csv', case_label='mdd', "
              "columns=['doi','task','n_mdd','mean_mdd','sd_mdd','n_control','mean_control','sd_control','d'])"],
    related=["tl.reconstruct_study_effects", "tl.meta_effect_sizes"],
)
def write_study_effects(prop: pd.DataFrame, path: str, case_label: str = "case", control_label: str = "control",
                        columns: Optional[Sequence[str]] = None, study_column: str = "doi",
                        task_column: str = "task", overrides: Optional[Dict] = None,
                        extra_rows: Optional[pd.DataFrame] = None, correction: Optional[str] = None) -> pd.DataFrame:
    """Write and return the delivered table.

    ``extra_rows``: comparisons transcribed by hand (columns ``study, row, n_case, mean_case, sd_case,
    n_control, mean_control, sd_control``), appended after the kept rows.
    """
    overrides = overrides or {}
    if "proposed" not in prop:
        raise ValueError("write_study_effects takes the `prop` table returned by reconstruct_study_effects (it "
                         "carries the proposed decisions); comparisons transcribed by hand go in extra_rows, and "
                         "only for studies the function could not read")
    if extra_rows is not None and len(extra_rows):
        read = set(prop.loc[prop["proposed"].notna(), "study"].astype(str))
        clash = sorted(set(extra_rows["study"].astype(str)) & read)
        if clash:
            raise ValueError(f"extra_rows for studies the function already read: {clash}. Their rows and "
                             f"decisions come from the proposals; do not transcribe them by hand")
    if "proposed" in prop:
        unresolved = prop.index[prop["decision"] == "review"].tolist()
        if unresolved:
            raise ValueError(f"resolve the review rows first (set decision to keep / exclude): {unresolved}")
        changed = prop.index[(prop["proposed"].isin(["keep", "exclude"])) & (prop["decision"] != prop["proposed"])]
        bad = [i for i in changed if len(str(overrides.get(i, ""))) < 25]
        if bad:
            raise ValueError(
                f"rows {bad} changed from the proposed decision without a cited reason. Restore them "
                f"(prop.loc[i, 'decision'] = prop.loc[i, 'proposed']) or pass overrides={{i: '<quoted method or "
                f"instruction sentence that decides it>'}}. A measure you consider secondary is not a reason.")
        for i in changed:
            print(f"override [{i}] {prop.at[i, 'row']}: {prop.at[i, 'proposed']} -> {prop.at[i, 'decision']} "
                  f"({overrides[i][:100]})")
    if "stats_key" in prop:
        edited = [i for i in prop.index[prop["decision"] == "keep"]
                  if isinstance(prop.at[i, "stats_key"], str) and prop.at[i, "stats_key"] != _stats_key(prop.loc[i])]
        if edited:
            raise ValueError(
                f"rows {edited}: statistics differ from what reconstruct_study_effects read from the article. Values "
                f"are never edited by hand: wrong group sizes -> call reconstruct_study_effects again with n_map for "
                f"that study; a misprint -> the function recovers it from the article's own reported effect size or "
                f"leaves the row out.")
    kept = prop[prop["decision"] == "keep"]
    if extra_rows is not None and len(extra_rows):
        kept = pd.concat([kept, extra_rows], ignore_index=True)
    stat = ["n_case", "mean_case", "sd_case", "n_control", "mean_control", "sd_control"]
    miss = kept[kept[stat].isna().any(axis=1)]
    if len(miss):
        print(f"skipped {len(miss)} kept rows with missing statistics (fill n via n_map): "
              f"{miss['row'].astype(str).str[-50:].tolist()}")
    kept = kept.dropna(subset=stat)
    if correction is None:
        correction = prop.attrs.get("correction", "approx")
    eff = meta_effect_sizes(kept, case="case", control="control", correction=correction)
    out = pd.DataFrame({study_column: eff["study"].values, task_column: eff["row"].astype(str).values})
    for c in stat:
        out[c.replace("_case", f"_{case_label}").replace("_control", f"_{control_label}")] = eff[c].values
    out["d"] = eff["d"].values
    for c in out.columns:
        if c.startswith("n_"):
            out[c] = out[c].round().astype(int)
    out = out.drop_duplicates()
    if columns:
        out = out[list(columns)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    record_deliverable(path, "omicverse.tl.write_study_effects", rows=len(out))
    print(f"wrote {len(out)} rows to {path} (fingerprint recorded; verify with ov.tl.verify_deliverable)")
    return out


def _prov_file(path) -> Path:
    import hashlib
    import os
    root = Path(os.environ.get("OV_PROVENANCE_DIR", "/tmp/ov_provenance"))
    key = hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:16]
    return root / f"{key}.json"


@register_function(
    aliases=["记录交付物指纹", "record deliverable fingerprint"],
    category="utils",
    description="Record the sha256 fingerprint of a deliverable written by a function (in $OV_PROVENANCE_DIR, "
                "default /tmp/ov_provenance; nothing is added next to the deliverable), so later hand edits "
                "can be detected by verify_deliverable / check_deliverables.py.",
    examples=["ov.tl.record_deliverable('/app/results/out.csv', 'my_function')"],
    related=["tl.verify_deliverable"],
)
def record_deliverable(path, producer: str, **info) -> dict:
    """Write ``{path, sha256, producer, ...}`` for ``path``; returns the record."""
    import hashlib
    import json
    import time
    rec = {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
           "producer": producer, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
    f = _prov_file(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, indent=1))
    return rec


@register_function(
    aliases=["校验交付物指纹", "verify deliverable fingerprint", "was the deliverable hand-modified"],
    category="utils",
    description="Check that a deliverable is byte-identical to what its producing function wrote "
                "(sha256 recorded by record_deliverable). Returns {'status': 'unchanged' | 'modified' | "
                "'unrecorded', ...}.",
    examples=["ov.tl.verify_deliverable('/app/results/out.csv')"],
    related=["tl.record_deliverable", "tl.write_study_effects"],
)
def verify_deliverable(path) -> dict:
    import hashlib
    import json
    f = _prov_file(path)
    if not f.exists():
        return {"status": "unrecorded", "path": str(path)}
    rec = json.loads(f.read_text())
    now = hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None
    status = "unchanged" if now == rec["sha256"] else "modified"
    msg = ("file is exactly what " + rec["producer"] + " wrote" if status == "unchanged" else
           "file was changed after " + rec["producer"] + " wrote it: rewrite it with that function "
           "(hand edits bypass the method's decisions)")
    print(f"{path}: {status} -- {msg}")
    return {**rec, "status": status, "message": msg}


def _fetch_fulltext(doi: str, dest: Path, timeout: float = 60.0) -> Optional[str]:
    """Europe PMC JATS XML, else the publisher's HTML via doi.org, accepted only when it contains
    tables and names the DOI. Returns the saved path or None (offline / not found)."""
    import json
    import urllib.parse
    import urllib.request
    ua = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36", "Accept": "text/html,application/xhtml+xml,*/*"}
    safe = re.sub(r"[^\w.-]", "_", doi)

    def get(url):
        req = urllib.request.Request(url, headers=ua)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore")
    try:
        q = urllib.parse.quote(f"DOI:{doi}")
        res = json.loads(get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}"
                             f"&format=json&resultType=core"))["resultList"]["result"]
        pmcid = next((r.get("pmcid") for r in res if r.get("pmcid")), None)
        if pmcid:
            for url in (f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
                        f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"):
                try:
                    txt = get(url)
                except Exception:
                    continue
                if "<table" in txt:
                    f = dest / f"{safe}.{'xml' if url.endswith('XML') else 'html'}"
                    f.write_text(txt)
                    return str(f)
    except Exception:
        pass
    try:
        txt = get(f"https://doi.org/{doi}")
        if "<table" in txt and doi.lower() in txt.lower():
            f = dest / f"{safe}.publisher.html"
            f.write_text(txt)
            return str(f)
    except Exception:
        pass
    return None


_NUMWORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}


def _num_word(w: str) -> Optional[int]:
    w = w.lower().strip()
    if w.isdigit():
        return int(w)
    if w in _NUMWORDS:
        return _NUMWORDS[w]
    parts = re.split(r"[- ]", w)
    if parts[0] in _TENS:
        return _TENS[parts[0]] + (_NUMWORDS.get(parts[1], 0) if len(parts) > 1 else 0)
    return None


def _sample_sizes_from_text(text: str):
    """First case and control sample sizes stated in prose; (None, None) when ambiguous."""
    nw = r"(\d{1,4}|(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen))"
    pat = re.compile(nw + r"\s+((?:[\w-]+\s+){0,6}?)(patients|subjects|participants|individuals|volunteers|controls)\b",
                     re.I)
    case_n, ctrl_n = [], []
    for m in pat.finditer(text):
        n = _num_word(m.group(1))
        if not n or n < 3:
            continue
        ctx = (m.group(2) + " " + m.group(3)).lower()
        (ctrl_n if re.search(r"healthy|control|comparison|non-?\w+ed\b", ctx) else case_n).append(n)
    if case_n and ctrl_n:
        return case_n[0], ctrl_n[0]
    return None, None


def _norm_name(x: str) -> tuple:
    toks = re.findall(r"[a-z0-9]+", str(x).lower().replace("-", " "))
    return tuple(_stem(t) for t in toks if t not in {"the", "group", "groups", "subjects", "participants",
                                                       "patients", "individuals", "with", "of", "n"})


def _named_sizes_in_text(text: str):
    """("acute COVID-19", 15) for every "<name> (n = 15)" / "<name>, n = 15" in prose."""
    out = []
    for m in re.finditer(r"([A-Za-z][\w\-/’' ]{1,40}?)\s*[(,]\s*[nN]\s*[=:]\s*(\d{1,5})\s*\)?", str(text)):
        name = re.split(r"\b(?:in|of|and|with|were|was|the|as follows|follows|:)\b", m.group(1))[-1].strip(" ,:")
        if 1 <= len(name) <= 40 and int(m.group(2)) >= 2:
            out.append((name, int(m.group(2))))
    return out


def _match_group_size(group: str, found: Dict[str, set]) -> Optional[int]:
    g = set(_norm_name(group))
    if not g:
        return None
    exact = [v for k, v in found.items() if set(_norm_name(k)) == g and len(v) == 1]
    if exact:
        return next(iter(exact[0]))
    sup = sorted(((len(set(_norm_name(k)) - g), v) for k, v in found.items()
                  if g < set(_norm_name(k)) and len(v) == 1), key=lambda x: x[0])
    if sup and (len(sup) == 1 or sup[0][0] < sup[1][0]):
        return next(iter(sup[0][1]))
    return None


def _check_reported_effects(prop: pd.DataFrame, texts: Dict[str, str]) -> pd.DataFrame:
    """A kept comparison whose printed statistics contradict the effect size the article itself reports
    for that measure (a misprinted mean / SD) is excluded, with the reported value in the reason."""
    stat = ["n_case", "mean_case", "sd_case", "n_control", "mean_control", "sd_control"]
    for i in prop.index[(prop["decision"] == "keep") & prop[stat].notna().all(axis=1)]:
        txt = _strip_tags(texts.get(prop.at[i, "study"], "") or "")
        if not txt:
            continue
        row = str(prop.at[i, "row"]).split(": ")[-1]
        keys = re.findall(r"\b[A-Z][A-Z0-9]{1,6}(?:-[A-Za-z0-9]+)?\b", row) or [" ".join(_words(row)[:3])]
        keys = [k for k in keys if k.lower() not in ("sd", "se", "sem", "ms", "rt", "n")]
        reported = []
        for k in keys:
            if len(k) < 2:
                continue
            for m in re.finditer(re.escape(k), txt, re.I):
                win = txt[m.end():m.end() + 160]
                nxt = re.search(r"\b[A-Z][A-Z0-9]{1,6}(?:-[A-Za-z0-9]+)?\s*:", win[3:])
                if nxt:                                   # stop at the next measure's statistics
                    win = win[:nxt.start() + 3]
                for d in re.findall(r"\b(?:Cohen['’]s\s+)?d\s*=\s*([−–-]?\s*\d*\.\d+|[−–-]?\s*\d+)", win):
                    v = _to_float_safe(d)
                    if v is not None:
                        reported.append(v)
        if not reported:
            continue
        r = prop.loc[i]
        sp = math.sqrt(((r.n_case - 1) * r.sd_case ** 2 + (r.n_control - 1) * r.sd_control ** 2) /
                       (r.n_case + r.n_control - 2))
        d = (r.mean_case - r.mean_control) / sp if sp > 0 else float("nan")
        if np.isfinite(d) and min(min(abs(d - v), abs(abs(d) - abs(v))) for v in reported) > 1.0:
            rec = _recover_mean(r, reported, sp)
            if rec is not None:
                col, val, dv = rec
                prop.at[i, col] = val
                prop.at[i, "reason"] = (f"printed {col.replace('_', ' ')} {r[col]:g} is a misprint (it gives d = {d:.2f}, "
                                        f"the article reports d = {dv:.2f}); recovered {col.replace('_', ' ')} = {val:.4f} "
                                        f"from the reported d and the pooled SD")
            else:
                prop.at[i, "decision"] = "exclude"
                prop.at[i, "reason"] = (f"printed statistics give d = {d:.2f} but the article reports d = "
                                        f"{', '.join(f'{v:.2f}' for v in reported)} for this measure: a misprint that "
                                        f"cannot be recovered unambiguously (left out)")
    return prop


def _to_float_safe(x: str) -> Optional[float]:
    try:
        return float(str(x).replace("−", "-").replace("–", "-").replace(" ", ""))
    except ValueError:
        return None


def _aggregate_vs_subgroups(prop: pd.DataFrame, texts: Optional[Dict[str, str]] = None,
                            rules: Sequence[str] = ()) -> pd.DataFrame:
    """The same measure printed for an aggregate case column and for its subgroups: one row only.
    If the aggregate's n differs from the subgroups' total, the aggregate holds participants outside
    the reported subgroups (e.g. a subgroup the coding rules exclude) -> the pooled subgroups are kept."""
    if not len(prop):
        return prop
    norm = lambda x: re.sub(r"[^a-z0-9%]+", " ", str(x).lower()).strip()
    keep = prop["decision"] == "keep"
    for sid, g in prop[keep].groupby("study"):
        g = g.assign(_nr=g["row"].map(norm))
        for _, gg in g.groupby(["_nr", "mean_control", "sd_control"]):
            if len(gg) < 2:
                continue
            multi = gg[gg["case_groups"].astype(str).str.contains(r" \+ ")]
            single = gg[~gg.index.isin(multi.index)]
            if multi.empty or single.empty:
                continue
            m0 = multi.iloc[0]
            for i, r in single.iterrows():
                if pd.notna(r["n_case"]) and pd.notna(m0["n_case"]) and r["n_case"] > m0["n_case"]:
                    delta = int(round(r["n_case"] - m0["n_case"]))
                    ev = _sentence_with_count(_strip_tags((texts or {}).get(sid, "") or ""), delta)
                    rule = next((x for x in rules), "")
                    prop.at[i, "decision"] = "exclude"
                    prop.at[i, "reason"] = (f"aggregate case column (n={r['n_case']:g}) holds {delta} participants outside its "
                                            f"reported subgroups (total {m0['n_case']:g}); the pooled subgroups are used"
                                            + (f". Article: '{ev[:200]}'" if ev else "")
                                            + (f". Rules: '{rule[:220]}'" if rule else ""))
                    for j in multi.index:
                        prop.at[j, "reason"] = ("pooled subgroups replace the aggregate column, which includes "
                                                f"{delta} participants outside them" + (f" ('{ev[:120]}')" if ev else ""))
                else:
                    # equal totals: the printed aggregate; larger subgroup totals: repeated measurements of
                    # the same participants (e.g. follow-up waves), not separate people
                    for j in multi.index:
                        prop.at[j, "decision"] = "exclude"
                        prop.at[j, "reason"] = ("same participants as the printed aggregate column (subgroups total "
                                                f"{m0['n_case']:g} vs {r['n_case']:g})")
    return prop


_NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
              "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}


def _sentence_with_count(text: str, n: int) -> str:
    """First sentence of the article that states the count n of participants (digits or words)."""
    words = [w for w, v in _NUM_WORDS.items() if v == n]
    pat = re.compile(r"\b(?:%d%s)\b" % (n, "".join("|" + w for w in words)), re.I)
    for sent in re.split(r"(?<=[.])\s+", text):
        if pat.search(sent) and re.search(r"participants?|patients?|subjects?|individuals?|people|cases?", sent, re.I):
            return sent.strip()
    return ""


def _digits(x: float) -> str:
    return re.sub(r"[^0-9]", "", f"{abs(x):.4f}").lstrip("0")


def _recover_mean(r, reported, sp):
    """Which printed mean is the misprint, and its value from a reported d (case - control, pooled SD).
    The recovered value must keep the printed digits (a dropped / shifted digit), else None."""
    cands = []
    for v in reported:
        for col, val in (("mean_control", r.mean_case - v * sp), ("mean_case", r.mean_control + v * sp)):
            pd_, rd_ = _digits(r[col]).rstrip("0"), _digits(val)
            if pd_ and (rd_.startswith(pd_[:2]) and len(pd_) >= 2) and abs(val - r[col]) > 0.5 * sp:
                cands.append((col, round(val, 6), v))
    return cands[0] if len(cands) == 1 else None


def _stats_key(r) -> str:
    vals = []
    for c in ("n_case", "mean_case", "sd_case", "n_control", "mean_control", "sd_control"):
        v = r.get(c) if hasattr(r, "get") else r[c]
        vals.append("nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):.6g}")
    return "|".join(vals)
