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

__all__ = ["abbreviation_map", "propose_selection", "reconstruct_study_effects"]

_STOP = set("""a an the of and or in on for to with by from as at is are was were be been this that these
those which who it its their than then also both each either per vs versus into within across between
after before during over under not no all any most more less other such only one two three first second
mean means sd s.d se n m total score scores raw t task tasks test tests trial trials condition conditions
measure measures number numbers percent percentage participant participants subject subjects group groups
control controls patient patients study studies analysis analyzed reported report reporting value values
used using use k p d effect size sizes composite individually included include includes including
variable variables dependent primary common most reported version versions et al table note response responses
completion characteristic characteristics there upon significant significantly effect effects difference
differences found observed showed compared higher lower greater average performance average""".split())

_SPEED = r"\b(rt|rts|reaction|latenc\w*|response time\w*|time\w*|speed|ms|msec|millisecond\w*|second\w*|sec|duration|slow\w*)\b|\(s\)"
_ACC = (r"\b(accura\w*|correct\w*|hits?|errors?|error rates?|omissions?|commissions?|false[- ]?alarms?|"
        r"false a\.?|incorrect|mistakes?|misses|failures?)\b")
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
    text = re.sub(r"['’]s\b", "", str(text))
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


_COMPOSITE = r"\b(index|indices|quotient|composite|factor score|scale score|full[- ]scale)\b"


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
        # concepts: plain words, and abbreviations with their expansion words
        concepts = []
        caps = {t.lower() for t in re.findall(r"\b[A-Z]{2,}[A-Z0-9]*\b", lab)}
        caps |= {t.lower() for t in re.findall(r"\b([A-Z]{2,})\d", lab)}
        for tok in re.findall(r"\b[A-Z][A-Za-z0-9/&-]{1,9}\b", lab):
            if tok.lower() in _STOP or re.fullmatch(_SPEED, tok.lower()) or re.fullmatch(_ACC, tok.lower()):
                continue
            key = next((k for k in (tok, tok.split("-")[0]) if k in ab_i), None)
            if key and tok.upper() == tok:
                concepts.append((tok.lower(), set(_content(ab_i[key])) or {tok.lower()}, True))
        abbr_words = {c for c, _, _ in concepts}
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
                    why += "; the article's text best matches: " + " || ".join(
                        f"'{st.at[j, 'sentence'][:90]}'" for _, j in cand)
                    if cand and cand[0][0] > 0:
                        j0 = cand[0][1]
                        m0 = st.at[j0, "measures"]["pref"]
                        clear = len(cand) < 2 or cand[0][0] >= 1.3 * cand[1][0]
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
            return sum(hit(f, pool) for f in forms) / len(forms)

        best = None
        for j in range(len(cur_tok)):
            sc, cov_w, tot_w, unm = 0.0, 0.0, 0.0, []
            for name, forms, is_ab in concepts:
                w_ = weight(forms)
                unknown_code = (name in caps) and not is_ab      # an undefined code like "NM3M"
                tot_w += (0.1 if re.search(r"\d", lab.split(name.upper())[-1][:3] if name.upper() in lab else "")
                          else 0.3) * w_ if unknown_code else w_
                need = (0.25 if len(forms) >= 4 else 0.5) if is_ab else 1.0
                if frac(forms, cur_tok[j]) >= need:
                    sc += w_; cov_w += w_
                elif frac(forms, cur_tok[j] | ctx_tok[j]) >= need:
                    sc += 0.6 * w_; cov_w += w_
                elif any(hit(f, qual_tok[j]) for f in forms):
                    cov_w += 0.3 * w_ if unknown_code else w_
                else:
                    unm.append(name)
            sc *= 1.0 if strong[j] else 0.7
            if sc > 0 and (best is None or sc > best[0]):
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
        rowc = set(_content(re.sub(r"[-/]", " ", row))) | set(re.findall(r"[a-z]+", row.lower()))

        def fits(m):
            if not m:
                return False
            if rtypes & m["types"]:
                return True
            if not m["types"] and re.search(r"\bcorrect\b", row, re.I) and not re.search(_PCT, row):
                return True          # "total / number correct" is the count the task's measure names
            if any(hit(q, rowc) for q in m["quals"]):
                return True
            ac = {re.sub(r"[^a-z]", "", w.lower()) for w in re.findall(r"\b[A-Z]{2,5}\b", row)}
            return bool(ac & m["acronyms"])

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
        elif not rtypes and not mwords:
            dec, why = "keep", "the task's single score"
        else:
            dec, why = "exclude", f"statistic not among the method's measures for this task ({meas['pref']['text'][:80]})"
        if dec in ("keep", "fallback") and cov < min_coverage:
            dec, why = "review", (f"task identity uncertain (coverage {cov:.2f}): {sorted(unm)} not in the method "
                                  f"statement -- same task, or a variant the method does not name?")
        rec.append({**base, "decision": dec, "reason": why})
    R = pd.DataFrame(rec, index=df.index)
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
            same = ((sid == sid[i]) & ((df["statement"] == df.at[i, "statement"]) | (df["statement"] == ""))
                    & acc & ~pct & (inner == inner[i]) & df["decision"].isin(["keep", "review"]))
            if same.any():
                df.at[i, "decision"] = "exclude"
                df.at[i, "reason"] = "percentage of an accuracy/error measure while counts are reported (counts coded)"
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
                              out_dir: Optional[str] = None, verbose: bool = True):
    """Return ``(proposals, effects)``.

    ``proposals``: every workbook row with family, measure, decision (keep / exclude / review),
    reason and the method sentence used. ``effects``: :func:`meta_effect_sizes` of the ``keep``
    rows (columns ``n_case, mean_case, sd_case, n_control, mean_control, sd_control, d``).
    ``case=None`` treats every group that is not a control group as a case group (subgroups pooled;
    use ``exclude`` for groups of another diagnosis). ``prop.attrs['study_summary']`` (also printed)
    lists per study the statistics found, comparisons formed and the action needed (PDF only,
    groups not matched, unknown n). ``out_dir``: also write ``proposals.csv`` there. Review the ``review`` rows (and spot-check
    the others) against the articles; change a decision in ``proposals`` and recompute with
    ``meta_effect_sizes(proposals[proposals.decision == 'keep'], case='case')``.
    """
    kw = {} if control is None else {"control": control}
    if case is None:
        case = r"."          # every group that is not a control group is a case group
    # group sizes printed in one place of an article (a table header) apply to the same group
    # named elsewhere in it (another table, the running text)
    n_map = {k: dict(v) for k, v in (n_map or {}).items()}
    for sid, paths in sources.items():
        found = {}
        for pth in ([paths] if isinstance(paths, (str, Path)) else paths):
            try:
                led = extract_group_stats(pth)
            except Exception:
                continue
            for g_, nh in zip(led["group"].astype(str), led["n_header"]):
                if pd.notna(nh):
                    base = re.split(r"\(|\bn\s*=|\bN\s*=", g_)[0].strip(" :,")
                    if 1 <= len(base) <= 40:
                        found.setdefault(base, set()).add(int(nh))
        auto = {rf"(?:^|[\]\s(]){re.escape(b)}\b": next(iter(v)) for b, v in found.items() if len(v) == 1}
        if auto:
            n_map[sid] = {**auto, **n_map.get(sid, {})}
    wb = comparison_workbook(sources, case=case, exclude=exclude, n_map=n_map, verbose=False, **kw)
    if wb.empty:
        wb = pd.DataFrame(columns=["study", "source", "table", "section", "row", "layout", "case_groups",
                                   "case_subgroups", "control_group", "n_case", "mean_case", "sd_case",
                                   "n_control", "mean_control", "sd_control", "d", "family", "stat_class"])
    st = method_measure_statements(method_source)
    mtext = Path(str(method_source)).read_text(errors="ignore")
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
    kept = prop[(prop["decision"] == "keep")]
    kept = kept[kept[["n_case", "mean_case", "sd_case", "n_control", "mean_control", "sd_control"]].notna().all(axis=1)]
    eff = meta_effect_sizes(kept, case="case", control="control") if len(kept) else kept
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        prop.drop(columns=["case_subgroups"], errors="ignore").to_csv(Path(out_dir) / "proposals.csv", index=False)
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
        if all(str(p).lower().endswith(".pdf") for p in paths):
            msgs.append("PDF only: tables are often unreadable -- get the XML / HTML full text or transcribe")
        if nstat and not len(g):
            msgs.append(f"{nstat} statistics found but no case-vs-control pair: group names are "
                        f"{[k for k, _ in groups.most_common(8)]} -- pass case / control for this study")
        if not nstat:
            msgs.append("no statistics found in the file(s): wrong file, or a format to transcribe by hand")
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
        print(f"keep {int((prop.decision == 'keep').sum())}, review {int((prop.decision == 'review').sum())}, "
              f"exclude {int((prop.decision == 'exclude').sum())}; rows with unknown n are not in the effects "
              f"until n_map supplies them")
    return prop, eff
