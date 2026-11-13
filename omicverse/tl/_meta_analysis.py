r"""Study-level effect sizes for meta-analysis: extraction ledger and arithmetic.

Reconstructing a meta-analysis from its source articles fails in two places:

1. **Extraction is incomplete.** Articles print group statistics in tables
   (``12.3 (4.5)``, ``12.3 ± 4.5``, separate M / SD columns, group sizes in the
   column header) and in running text (``M = 12.3, SD = 4.5``).  Reading a
   table and keeping "the main outcome" silently drops the secondary measures
   (reaction times, error counts, each condition / part of a task) that the
   meta-analytic method codes.  :func:`extract_group_stats` turns every table
   and every in-text statistic of an article (JATS XML, HTML, PDF text or plain
   text) into one long candidate ledger, so that each candidate is *decided*
   (kept / excluded with a reason) instead of being overlooked.
2. **Arithmetic.**  :func:`pool_group_stats` combines subgroups into one sample
   including the between-subgroup variance (Cochrane Handbook 6.5.2.10),
   :func:`hedges_g` gives Cohen's d with the small-sample correction
   ``J = 1 - 3 / (4 N - 9)`` (and variance / CI), :func:`sd_from_summary`
   converts SE or a confidence interval to an SD, and
   :func:`meta_effect_sizes` applies all of it to a ledger table.
"""
from __future__ import annotations

import io
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._registry import register_function

__all__ = ["to_mean_sd", "parse_mean_sd", "extract_group_stats", "group_comparisons", "pool_group_stats", "hedges_g",
           "sd_from_summary", "recompute_group_test", "meta_effect_sizes", "apply_selection_rules",
           "comparison_workbook", "format_workbook", "method_measure_statements"]

_NUM = r"[-−–]?\s?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:[.,·]\d+)?"


def _to_float(tok: str) -> Optional[float]:
    if tok is None:
        return None
    t = str(tok).strip().replace("−", "-").replace("–", "-").replace("·", ".").replace(" ", "")
    t = t.rstrip("*†‡§¶#abcdefg")
    if not t or t in {"-", "."}:
        return None
    if "," in t and "." not in t:
        # 1,064 -> thousands; 4,51 -> decimal comma
        t = t.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", t) else t.replace(",", ".")
    else:
        t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


_PAIR_PATTERNS = [
    # 12.3 (4.5) / 12.3(S.D.: 4.5) / 12.3 [4.5] / 12.3 (SD 4.5)
    re.compile(r"^\s*(?P<m>" + _NUM + r")\s*[\(\[]\s*(?:s\.?\s*[de]\.?\s*m?\.?\s*[:=]?\s*)?[+±]?\s*(?P<s>" + _NUM
               + r")\s*[\)\]]\s*[*∗†‡§a-g]*\s*$", re.I),
    # 12.3 ± 4.5 / 12.3 +/- 4.5
    re.compile(r"^\s*(?P<m>" + _NUM + r")\s*(?:±|\+/-|\+-|\+/−)\s*(?P<s>" + _NUM + r")\s*[*∗†‡§a-g]*\s*$"),
]
# median with a range / IQR: "54; 22–71", "54 (22–71)", "54 [22-71]"
_MED_RANGE = re.compile(r"^\s*(?P<m>" + _NUM + r")\s*[;,]?\s*[\(\[]?\s*(?P<a>" + _NUM + r")\s*(?:–|-|—|to)\s*(?P<b>"
                        + _NUM + r")\s*[\)\]]?\s*[*∗†‡§a-g]*\s*$")


def _parse_median_range(cell) -> Optional[Tuple[float, float, float]]:
    s = str(cell).replace("\xa0", " ").strip()
    m = _MED_RANGE.match(s)
    if not m:
        return None
    vals = [_to_float(m.group(k)) for k in ("m", "a", "b")]
    if any(v is None for v in vals) or not (vals[1] <= vals[0] <= vals[2]):
        return None
    return tuple(vals)


def _stat_kind(hint: str) -> Tuple[str, str]:
    """('mean_sd' | 'mean_se', 'median_range' | 'median_iqr') declared by a caption / header text."""
    h = str(hint)
    se = bool(re.search(r"\bS\.?E\.?M?\b|standard error", h)) and not re.search(r"\bS\.?D\b|standard deviation", h)
    iqr = bool(re.search(r"\bIQR\b|interquartile|25(th)?\s*[-–,]\s*75", h, re.I))
    return ("mean_se" if se else "mean_sd"), ("median_iqr" if iqr else "median_range")


@register_function(
    aliases=["解析均值标准差", "parse mean sd cell", "parse M (SD)", "parse mean ± sd"],
    category="utils",
    description="Parse one reported group statistic cell into (mean, sd): '12.3 (4.5)', '12.3(S.D.: 4.5)', "
                "'12.3 ± 4.5', '12.3 [4.5]', unicode minus, thousands commas and decimal commas.",
    examples=["ov.tl.parse_mean_sd('16.20(S.D.: 0.78)')  # (16.2, 0.78)"],
    related=["tl.extract_group_stats"],
)
def parse_mean_sd(cell) -> Optional[Tuple[float, float]]:
    """Return ``(mean, sd)`` or ``None`` when the cell is not a mean/SD pair."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return None
    s = str(cell).replace(" ", " ").replace("\xa0", " ").strip()
    for p in _PAIR_PATTERNS:
        m = p.match(s)
        if m:
            a, b = _to_float(m.group("m")), _to_float(m.group("s"))
            if a is not None and b is not None:
                return a, b
    return None


# ---------------------------------------------------------------------------
# table extraction
# ---------------------------------------------------------------------------

_N_RE = re.compile(r"\b[nN]\s*[=:]\s*(\d+)")
_M_HDR = re.compile(r"^\s*(m|mean|means|average|avg|x̄)\.?\s*$", re.I)
_SD_HDR = re.compile(r"^\s*(sd|s\.d\.|std|std\.?\s*dev\.?|standard deviation|\(sd\))\s*$", re.I)


def _clean(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = re.sub(r"\s+", " ", _ZW.sub("", str(x))).strip()
    return "" if s.lower().startswith("unnamed:") or s.lower() == "nan" else s


def _tables_from_markup(text: str) -> List[Tuple[str, str]]:
    """Return [(caption, table_html)] for every <table> in HTML or JATS XML."""
    out = []
    wraps = list(re.finditer(r"<table-wrap\b.*?</table-wrap>", text, re.S | re.I))
    if wraps:
        for w in wraps:
            body = w.group(0)
            cap = " ".join(re.findall(r"<(?:label|caption)\b.*?</(?:label|caption)>", body, re.S | re.I))
            for t in re.findall(r"<table\b.*?</table>", body, re.S | re.I):
                out.append((_strip_tags(cap), t))
        return out
    for m in re.finditer(r"<table\b.*?</table>", text, re.S | re.I):
        t = m.group(0)
        cap = re.search(r"<caption\b.*?</caption>", t, re.S | re.I)
        before = text[max(0, m.start() - 600):m.start()]
        heads = re.findall(r"(Table\s+\d+[^<]{0,200})", _strip_tags(before))
        out.append((_strip_tags(cap.group(0)) if cap else (heads[-1] if heads else ""), t))
    return out


_ZW = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")


def _strip_tags(s: str) -> str:
    s = _ZW.sub("", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", s).strip()


def _grid(table_html: str) -> List[List[str]]:
    """Expand an HTML table (rowspan / colspan) into a rectangular grid of strings."""
    rows = re.findall(r"<tr\b.*?</tr>", table_html, re.S | re.I)
    grid: List[List[Optional[str]]] = []
    carry = {}
    for r_i, tr in enumerate(rows):
        cells = re.findall(r"<t([hd])\b([^>]*?)(/>|>(.*?)</t[hd]>)", tr, re.S | re.I)
        row: List[str] = []
        c_i = 0
        for cell in cells:
            attrs = cell[1]
            txt = _strip_tags(cell[3]) if cell[2] != "/>" else ""
            while (r_i, c_i) in carry:
                row.append(carry.pop((r_i, c_i)))
                c_i += 1
            cs = int((re.search(r'colspan="?(\d+)', attrs) or [None, 1])[1])
            rs = int((re.search(r'rowspan="?(\d+)', attrs) or [None, 1])[1])
            if c_i == 0 and cs > 1 and txt and re.search(r'center', attrs, re.I):
                txt = "\x01" + txt          # a centred spanning label: top-level section
            for k in range(cs):
                # continuation cells of a column span are marked so a value is read once
                row.append(txt if k == 0 or r_i == 0 or not txt else "\x02" + txt)
                for d in range(1, rs):
                    carry[(r_i + d, c_i)] = txt
                c_i += 1
        while (r_i, c_i) in carry:
            row.append(carry.pop((r_i, c_i)))
            c_i += 1
        grid.append(row)
    w = max((len(r) for r in grid), default=0)
    return [r + [""] * (w - len(r)) for r in grid]


def _is_num(s: str) -> bool:
    return _to_float(s) is not None or parse_mean_sd(s) is not None or _parse_median_range(s) is not None


def _table_candidates(caption: str, grid: List[List[str]], table_id: str) -> List[dict]:
    if not grid:
        return []
    top = [bool(r) and str(r[0]).startswith("\x01") for r in grid]
    cont = [[str(c).startswith("\x02") for c in r] for r in grid]
    grid = [[str(c).replace("\x01", "").replace("\x02", "") for c in r] for r in grid]
    ncol = len(grid[0])
    # header rows: leading rows with no numeric cell outside the first column
    h = 0
    def _section_like(row):
        # only a label (a spanning section cell repeats the label in every column)
        first = _clean(row[0])
        return bool(first) and all(_clean(c) in ("", first) for c in row[1:])

    def _datalike(c):
        c = _clean(c)
        return _is_num(c) or bool(re.fullmatch(r"[-−]?\d[\d.,;:%()\s/–-]*", c))

    while h < len(grid) and h < 4 and not any(_datalike(c) for c in grid[h][1:]) \
            and not (h > 0 and _section_like(grid[h])):
        h += 1
    header = [" ".join(dict.fromkeys(_clean(grid[r][c]) for r in range(h) if _clean(grid[r][c])))
              for c in range(ncol)]
    body = grid[h:]
    # a body row made of M / SD tokens is a sub-header (M | SD | M | SD)
    msd_row = None
    for i, row in enumerate(body[:3]):
        toks = [_clean(c) for c in row[1:]]
        if any(_M_HDR.match(t) for t in toks) and any(_SD_HDR.match(t) for t in toks):
            msd_row = row
            body = body[:i] + body[i + 1:]
            break
    sub = [_clean(c) for c in (msd_row or [""] * ncol)]
    # also allow M/SD tokens as the LAST header row
    if msd_row is None and h >= 1:
        last = [_clean(c) for c in grid[h - 1]]
        if any(_M_HDR.match(t) for t in last) and any(_SD_HDR.match(t) for t in last):
            sub = last
            header = [" ".join(dict.fromkeys(_clean(grid[r][c]) for r in range(h - 1) if _clean(grid[r][c])))
                      for c in range(ncol)]
    # label columns: leading columns that are mostly non-numeric
    nlab = 0
    for c in range(ncol):
        vals = [_clean(r[c]) for r in body if _clean(r[c])]
        if c > 0 and any(parse_mean_sd(v) is not None or _parse_median_range(v) is not None for v in vals):
            break
        if vals and sum(_is_num(v) for v in vals) / len(vals) < 0.5:
            nlab += 1
        else:
            break
    nlab = max(nlab, 1)
    out = []
    # what the cells hold: caption, header and the text-only body cells declare SE / median-range / IQR
    hint = " ".join([caption] + header + [_clean(c) for r in body for c in r if _clean(c) and not _is_num(_clean(c))])
    kind_ms, kind_med = _stat_kind(hint)
    # orientation: groups as ROWS when the same row labels repeat under several section rows
    lab_secs: Dict[str, set] = {}
    cur = ""
    for row in body:
        cells_ = [_clean(c) for c in row]
        lab_ = " ".join(dict.fromkeys(c for c in cells_[:nlab] if c))
        if lab_ and all(d in ("", lab_) for d in cells_[nlab:]):
            cur = lab_
        elif lab_:
            lab_secs.setdefault(lab_, set()).add(cur)
    rep = [k for k, v in lab_secs.items() if len(v) >= 2]
    transposed = len(rep) >= 2 and any(re.search(r"control|healthy|\bHC\b|comparison", k, re.I) for k in rep)
    # a row of group sizes ("N | 120 | 30")
    n_by_col: Dict[int, int] = {}
    for row in body:
        cells_ = [_clean(c) for c in row]
        if re.fullmatch(r"(n|no\.?|number of (subjects|participants|patients)|sample size)\s*:?", cells_[0], re.I):
            for c in range(nlab, ncol):
                v = _to_float(cells_[c])
                if v is not None and float(v).is_integer() and v >= 2:
                    n_by_col[c] = int(v)
    section: List[str] = []
    prev_data = False
    styled = False
    body_top = top[len(top) - len(body):] if msd_row is None else None
    for bi, row in enumerate(body):
        cells = [_clean(c) for c in row]
        is_top = bool(body_top[bi]) if body_top is not None else False
        label = " ".join(dict.fromkeys(c for c in cells[:nlab] if c))
        data = cells[nlab:]
        if label and all(d in ("", label) for d in data):
            # section header row (indentation is lost): after a data row it starts a sibling of
            # the innermost section, after another section row it nests under it
            if is_top:
                section, styled = [label], True
            else:
                section = (section[:-1] + [label]) if prev_data else (section + [label])[-3:]
            prev_data = False
            continue
        if not label:
            continue
        prev_data = True
        # the innermost section names the task; outer levels are ambiguous once indentation is lost
        # (unless the markup marks top-level sections, e.g. centred spanning cells)
        if styled:
            full = ": ".join(section + [label])
        else:
            full = f"{section[-1]}: {label}" if section else label
        c = nlab
        gi = len(grid) - len(body) + bi
        rc = cont[gi] if gi < len(cont) and msd_row is None else [False] * ncol
        used_hdr = {header[k] for k in range(nlab, ncol) if not rc[k] and k < len(header)}
        while c < ncol:
            cell = cells[c]
            if rc[c]:
                c += 1
                continue
            grp = header[c] if c < len(header) else ""
            # a value spanning several columns belongs to the spanned header no other value uses
            span = [c] + [k for k in range(c + 1, ncol) if rc[k] and all(rc[q] for q in range(c + 1, k + 1))]
            if len(span) > 1:
                others = {header[k] for k in range(nlab, ncol) if k not in span and not rc[k]}
                cand = [header[k] for k in span if header[k] not in others]
                if cand:
                    grp = cand[0]
            row_lab, row_no = full, bi
            if transposed:
                grp = label
                row_lab = f"{section[-1]}: {header[c]}" if section else header[c]
                row_no = (tuple(section), c)
            nh = _n_of(grp) if _n_of(grp) is not None else n_by_col.get(c)
            pair = parse_mean_sd(cell)
            if pair is not None:
                out.append({"table": table_id, "caption": caption, "row": row_lab, "group": grp,
                            "n_header": nh, "mean": pair[0], "sd": pair[1], "raw": cell,
                            "layout": "cell", "row_no": row_no, "section": " > ".join(section),
                            "stat": kind_ms})
                c += 1
                continue
            med = _parse_median_range(cell)
            if med is not None:
                out.append({"table": table_id, "caption": caption, "row": row_lab, "group": grp,
                            "n_header": nh, "mean": med[0], "sd": float("nan"), "lo": med[1], "hi": med[2],
                            "raw": cell, "layout": "cell", "row_no": row_no, "section": " > ".join(section),
                            "stat": kind_med})
                c += 1
                continue
            if sub[c] and _M_HDR.match(sub[c]) and c + 1 < ncol and _SD_HDR.match(sub[c + 1] or ""):
                m_, s_ = _to_float(cell), _to_float(cells[c + 1])
                if m_ is not None and s_ is not None:
                    out.append({"table": table_id, "caption": caption, "row": full, "group": grp,
                                "n_header": nh, "mean": m_, "sd": s_,
                                "raw": f"{cell} | {cells[c + 1]}", "layout": "M|SD columns",
                                "row_no": bi, "section": " > ".join(section), "stat": kind_ms})
                c += 2
                continue
            c += 1
    return out


def _n_of(text: str) -> Optional[int]:
    m = _N_RE.search(text or "")
    return int(m.group(1)) if m else None


_TEXT_PATTERNS = [
    re.compile(r"\bM\s*=\s*(?P<m>" + _NUM + r")\s*[,;]?\s*SD\s*=\s*(?P<s>" + _NUM + r")", re.I),
    re.compile(r"\bmean\s*(?:=|of|was|:)?\s*(?P<m>" + _NUM + r")\s*[,;(]?\s*(?:SD|S\.D\.|standard deviation)\s*(?:=|:|of)?\s*(?P<s>" + _NUM + r")", re.I),
    re.compile(r"(?P<m>" + _NUM + r")\s*(?:±|\+/-)\s*(?P<s>" + _NUM + r")"),
    re.compile(r"(?<![\w.])(?P<m>\d+(?:[.,]\d+)?)\s*\(\s*(?:SD\s*[=:]?\s*)?(?P<s>\d+(?:[.,]\d+)?)\s*\)"),
]


def _text_candidates(text: str, source: str) -> List[dict]:
    out = []
    lines = text.splitlines() if "\n" in text else re.split(r"(?<=[.;])\s+", text)
    for li, line in enumerate(lines):
        hits = []
        for p in _TEXT_PATTERNS:
            for m in p.finditer(line):
                a, b = _to_float(m.group("m")), _to_float(m.group("s"))
                if a is None or b is None or b < 0:
                    continue
                hits.append((m.start(), m.end(), a, b, m.group(0)))
        if not hits:
            # a table row flattened to text: "<label> m1 s1 m2 s2 [stat ...]"
            mrow = re.match(r"^\s*(?P<label>[^\d\s][^\d]*?)\s+(?P<nums>(?:[-−]?\d*[.,]?\d+\s+){3,}[-−]?\d*[.,]?\d+)\s*\S*$", line)
            if mrow:
                nums = [_to_float(x) for x in mrow.group("nums").split()]
                for k in range(len(nums) // 2):
                    a, b = nums[2 * k], nums[2 * k + 1]
                    if a is None or b is None or b < 0:
                        continue
                    out.append({"table": f"{source}:line{li + 1}", "caption": "", "row": mrow.group("label").strip()[-120:],
                                "group": f"col{2 * k + 1}-{2 * k + 2}", "n_header": None, "mean": a, "sd": b,
                                "raw": line.strip()[-160:], "layout": "text-row (verify column order)",
                                "context": line.strip()[-200:]})
            continue
        hits.sort()
        dedup = []
        for h_ in hits:
            if dedup and h_[0] < dedup[-1][1]:
                continue
            dedup.append(h_)
        for k, (s0, s1, a, b, raw) in enumerate(dedup):
            ctx = line[max(0, s0 - 160):s1 + 60]
            before = line[max(0, s0 - 200):s0]
            # measure label: the clause before the statistic (or before its group list)
            clause = re.split(r"[;.]\s|\)\s*(?:or|and|,)\s", before)[-1]
            grp_m = re.search(r"([A-Za-z][\w-]*(?:\s[A-Za-z][\w-]*)?)\s*[:=,]?\s*$", clause)
            group = grp_m.group(1) if grp_m else f"#{k + 1}"
            label = re.sub(r"\(\s*[^()]*$", "", clause).strip(" :,(|\t") or clause.strip()
            # "... upon reaction time for Go responses (...) or for NoGo responses (...)": the second
            # clause inherits the measure named in the first
            if re.match(r"(for|in|on|of|at|during|with)\b", label, re.I) and out and \
                    out[-1]["table"] == f"{source}:line{li + 1}":
                prev = out[-1]["row"]
                cut = list(re.finditer(r"\b(for|in|on|of|at|during|with)\b", prev, re.I))
                if cut:
                    label = (prev[:cut[-1].start()].rstrip() + " " + label).strip()
            out.append({"table": f"{source}:line{li + 1}", "caption": "", "row": label[-120:], "group": group,
                        "n_header": _n_of(ctx), "mean": a, "sd": b, "raw": raw, "layout": "text",
                        "context": ctx.strip()})
    return out


_NUMTOK = re.compile(r"^[-−–]?\d*[.,]?\d+$")
_GROUP_HDR = re.compile(r"([A-Z][A-Za-z][\w '/-]{0,40}?)\s*\(?\s*[nN]\s*=\s*(\d+)\s*\)?")


def _split_labels(seg: str, k: int) -> Tuple[List[str], List[List[str]]]:
    """Split a run-together label segment into ``k`` leaf labels (+ the section path of each)."""
    parts = [x.strip() for x in re.split(r"\s{3,}", seg) if x.strip()]
    if len(parts) < k:
        parts = [x.strip() for x in re.split(r"\s{2,}", seg) if x.strip()]
    if len(parts) < k:
        return [], []
    # drop section-like labels until k leaves remain, most section-like first
    from ._meta_selection import _types   # local import (shared vocabulary)

    def rank(i):
        x = parts[i]
        nxt = parts[i + 1] if i + 1 < len(parts) else ""
        if x.endswith(":"):
            return 0
        if i == 0:
            return 1
        if x[:1].isupper() and nxt[:1].islower():
            return 2
        if len(x.split()) == 1 and not _types(x):
            return 3
        if not _types(x):
            return 4
        return 9
    is_sec = [False] * len(parts)
    while sum(not v for v in is_sec) > k:
        cand = [(rank(i), i) for i in range(len(parts)) if not is_sec[i] and rank(i) < 9]
        if not cand:
            return [], []
        is_sec[min(cand)[1]] = True
    leaves, paths, sec = [], [], []
    for x, sct in zip(parts, is_sec):
        if sct:
            # a new top-level section (first or colon-less after leaves) replaces; colon ones nest
            sec = (sec[:1] + [x.rstrip(" :")]) if (x.endswith(":") and sec) else [x.rstrip(" :")]
        else:
            leaves.append(x)
            paths.append(list(sec))
    return leaves, paths


def _pdf_column_tables(text: str, source: str) -> List[dict]:
    """Tables that PDF text extraction emits column by column: a block of run-together row labels,
    then one text line per table COLUMN (all means of group 1, all SDs of group 1, ...). Group names
    and sizes come from the nearest preceding header ('<Group> n = 20 ... mean SD mean SD')."""
    lines = text.split("\n")
    out = []
    groups: List[Tuple[str, int]] = []
    tcount = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        hdr = _GROUP_HDR.findall(ln)
        if len(hdr) >= 2:
            groups = [(g.strip(), int(n)) for g, n in hdr]
            tcount += 1
        toks = ln.split()
        is_num = len(toks) >= 2 and all(_NUMTOK.match(t) for t in toks)
        if is_num and groups:
            # collect consecutive numeric lines of equal length
            block = [toks]
            j = i + 1
            while j < len(lines):
                t2 = lines[j].split()
                if len(t2) == len(toks) and all(_NUMTOK.match(t) for t in t2):
                    block.append(t2); j += 1
                else:
                    break
            k = len(toks)
            if len(block) >= 2 * len(groups):
                # the label segment is the text right before the block (after the last block / header)
                seg = lines[i - 1] if i > 0 else ""
                for hg, _ in groups:
                    seg = seg.split(hg)[-1] if hg in seg else seg
                # labels start after the last statistic of the previous block / header line
                stat = r"(?:[<≤>]?[-−]?\d*[.,]?\d+\*?|\[\d+,\s*\d+\]|n\s*=\s*\d+)"
                m_last = None
                for m_last in re.finditer(stat + r"(?=\s{2,}|\s*$)", seg):
                    pass
                if m_last is not None:
                    seg = seg[m_last.end():]
                hdrw = r"(?:mean|means|SD|s\.d\.|t|df|p|F|MANOVA|ANOVA|ANCOVA|Student['’]s t test)"
                seg = re.sub(rf"^\s*(?:{hdrw}\s+)*", "", seg)
                seg = re.sub(rf"(?:\s+{hdrw})*\s*$", "", seg)
                leaves, paths = _split_labels(seg, k)
                if leaves:
                    for c in range(k):
                        for gi, (g, n) in enumerate(groups):
                            m_ = _to_float(block[2 * gi][c])
                            s_ = _to_float(block[2 * gi + 1][c])
                            if m_ is None or s_ is None:
                                continue
                            path = paths[c]
                            out.append({"table": f"{source}:pdf-table{tcount}", "caption": "",
                                        "section": " > ".join(path),
                                        "row": (f"{path[-1]}: {leaves[c]}" if path else leaves[c]),
                                        "group": f"{g} (n = {n})", "n_header": n, "mean": m_, "sd": s_,
                                        "raw": " | ".join(b[c] for b in block), "layout": "pdf columns",
                                        "row_no": c, "context": seg[:200]})
            i = j
            continue
        i += 1
    return out


_JUNK_LINE = re.compile(r"licen[cs]e|preprint|copyright|perpetuity|medrxiv|biorxiv|doi\.org|certified by peer", re.I)
_NGRP = re.compile(r"\(\s*[nN]\s*=\s*(\d[\d,]*)\s*\)")


def _pdf_stream_tables(text: str, source: str) -> List[dict]:
    """Tables that PDF text extraction emits row by row with cells broken over lines.

    A header names the groups with their sizes ("No COVID (n = 185) COVID (n = 181)"). Then either
    every row is "<label> m (sd) m (sd) ..." -- one pair per group, possibly split across lines -- or
    every row is "<label> v1 v2 ..." with a fixed set of columns per group that the header lists
    ("M SD <other> ..." repeated per group)."""
    out: List[dict] = []
    hs = list(_NGRP.finditer(text))
    i, tcount = 0, 0
    while i < len(hs):
        clus = [hs[i]]
        while i + 1 < len(hs) and hs[i + 1].start() - clus[-1].end() < 250:
            i += 1
            clus.append(hs[i])
        i += 1
        if len(clus) < 2:
            continue
        tcount += 1
        groups, prev = [], max(0, clus[0].start() - 120)
        for m in clus:
            seg = "\n".join(ln for ln in text[prev:m.start()].split("\n") if not _JUNK_LINE.search(ln))
            seg = re.sub(r"\s+", " ", seg).strip()
            seg = re.split(r"\)\s|\.\s|:\s", seg)[-1] if prev == max(0, clus[0].start() - 120) else seg
            words = [w for w in seg.split() if not re.fullmatch(r"[\d.,%()]+|Measure|Variable|Characteristic", w)]
            groups.append((" ".join(words[-5:]).strip(" ,;"), int(m.group(1).replace(",", ""))))
            prev = m.end()
        body_start = clus[-1].end()
        nxt = re.search(r"\n\s*(?:Supplementary\s+)?Table\s+\d+[.:]|\(\s*[nN]\s*=\s*\d", text[body_start + 20:])
        body = text[body_start: body_start + 20 + (nxt.start() if nxt else 8000)]
        # the header continues with column titles up to the statistic names ("Mean (SD) T p ...")
        hm = None
        for hm in re.finditer(r"(?:Mean\s*\(\s*SD\s*\)|\bM\s+SD\b)[^\n]*\n", body[:800]):
            pass
        if hm is not None and not re.search(r"\d\s*\(\s*\d", body[:hm.start()]):
            body = body[hm.end():]
        k = len(groups)
        lines = body.split("\n")
        # flat mode: per-group column names listed in the header (M SD ... repeated)
        head_toks = re.findall(r"[A-Za-z]+", body[:400])
        m_pos = [q for q, t in enumerate(head_toks) if t in ("M", "Mean", "mean")]
        flat = None
        if len(m_pos) >= k >= 2 and "SD" in head_toks:
            c = m_pos[1] - m_pos[0]
            sd_off = head_toks.index("SD", m_pos[0]) - m_pos[0]
            if 2 <= c <= 8 and 0 < sd_off < c:
                flat = (c, sd_off)
        pair = r"([-−]?\d*\.?\d+)\s*\(\s*([-−]?\d*\.?\d+)\s*\)"
        section = ""
        if flat:
            c, sd_off = flat
            num = r"[-−]?\d*\.?\d+"
            pat = re.compile(r"(?:^|\s)([A-Z][A-Za-z0-9/&’'-]{0,30}(?:\s[A-Za-z][A-Za-z0-9/&’'-]{0,30}){0,3})\s+((?:" + num
                             + r"\s+){" + str(k * c - 1) + "}" + num + ")(?=\s|$)")
            hdr_words = set(head_toks[:m_pos[-1] + c])
            for rn, m in enumerate(pat.finditer(re.sub(r"\s+", " ", body))):
                vals = [_to_float(v) for v in m.group(2).split()]
                lab_ = " ".join(w for w in m.group(1).split() if w not in hdr_words) or m.group(1)
                for g, (gname, n) in enumerate(groups):
                    mu, sd_ = vals[g * c], vals[g * c + sd_off]
                    if mu is None or sd_ is None:
                        continue
                    out.append({"table": f"{source}:pdf-stream{tcount}", "caption": "", "section": "",
                                "row": lab_, "group": f"{gname} (n = {n})", "n_header": n, "mean": mu,
                                "sd": sd_, "raw": m.group(0).strip()[:200], "layout": "pdf rows",
                                "row_no": rn, "context": ""})
            continue
        stream = "\n".join(ln for ln in lines if not _JUNK_LINE.search(ln))
        pat = re.compile(r"(?m)^[ \t]*([^\n\d]*?[A-Za-z%’'()][^\n\d]*?)\s+" + r"\s*".join([pair] * k))
        last_end = 0
        path: List[str] = []
        measure_word = re.compile(r"\b(mean|SD|s\.d\.|accuracy|correct|%|RT|score|reaction|time|errors?|n)\b|%|\(SD\)", re.I)
        for rn, m in enumerate(pat.finditer(stream)):
            # text lines between the previous row and this one: headings (task / condition names)
            # and the first lines of a label broken over several lines
            above = [ln.strip() for ln in stream[last_end:m.start()].split("\n") if ln.strip()]
            above = [ln for ln in above if not re.search(r"\d[.,]\d|\(\s*\d|\d\s*\)|^\s*[\d.]+\s*$|[=<>]|"
                                                         r"\b(?:T/U|F/H|GLM|p\s*$|df)\b", ln)
                     and re.search(r"[A-Za-z]{2}", ln) and 1 < len(ln) < 80]
            lab_lines, heads = [], []
            for ln in reversed(above):          # trailing measure-like lines continue the label
                if not heads and measure_word.search(ln):
                    lab_lines.insert(0, ln)
                else:
                    heads.insert(0, ln)
            if heads:
                path = heads if len(heads) >= 2 or not path else path[:-1] + heads
            last_end = m.end()
            label = re.sub(r"\s+", " ", " ".join(lab_lines + [m.group(1)])).strip()
            section = " > ".join(re.sub(r"\s+", " ", x) for x in path)
            if path:
                label_full = ": ".join([re.sub(r"\s+", " ", x) for x in path[-2:]] + [label])
            else:
                label_full = label
            for g, (gname, n) in enumerate(groups):
                mu, sd_ = _to_float(m.group(2 + 2 * g)), _to_float(m.group(3 + 2 * g))
                if mu is None or sd_ is None:
                    continue
                out.append({"table": f"{source}:pdf-stream{tcount}", "caption": "", "section": section,
                            "row": label_full, "group": f"{gname} (n = {n})",
                            "n_header": n, "mean": mu, "sd": sd_, "raw": re.sub(r"\s+", " ", m.group(0))[:200],
                            "layout": "pdf rows", "row_no": rn, "context": ""})
    return out


@register_function(
    aliases=["提取组统计量", "提取均值标准差", "meta-analysis data extraction", "extract group statistics",
             "extract mean SD from article tables", "candidate ledger for meta-analysis"],
    category="utils",
    description="Turn an article (JATS / PMC XML, publisher HTML, PDF via pypdf, or plain text) into a long "
                "candidate ledger of EVERY reported group statistic: one row per (table, row label with its "
                "section headers, group column) with mean, SD, group n from the column header, the raw cell "
                "and caption; plus in-text statistics (M = x, SD = y; x ± y; table lines of PDF text). "
                "Handles 'm (sd)', 'm ± sd', 'm(S.D.: sd)', separate M | SD columns, rowspan/colspan.",
    examples=["led = ov.tl.extract_group_stats('/app/data/study1.xml')",
              "led.pivot_table(index=['table','row'], columns='group', values=['mean','sd'], aggfunc='first')"],
    related=["tl.meta_effect_sizes", "tl.parse_mean_sd"],
)
def extract_group_stats(source: Union[str, Path], kind: Optional[str] = None,
                        include_text: bool = True) -> pd.DataFrame:
    """Return the candidate ledger of an article.

    Parameters
    ----------
    source
        Path to the article file, or the markup / text itself.
    kind
        ``'xml'``, ``'html'``, ``'pdf'`` or ``'text'``; guessed from the suffix / content.
    include_text
        Also scan the running text (and PDF text lines) for in-text statistics.

    Columns: ``table, caption, section, row, group, n_header, mean, sd, raw, layout, row_no, context``
    (``row`` = innermost section + row label; ``section`` = the section path, whose outer levels
    are a best guess when the markup carries no indentation; ``row_no`` orders table rows).
    Every candidate still has to be *decided* (which group is case / control, whether the
    measure is admissible); the ledger only guarantees that nothing printed is overlooked.
    """
    text = None
    name = "text"
    p = Path(str(source)) if len(str(source)) < 4096 else None
    if p is not None and p.exists():
        name = p.name
        suf = p.suffix.lower()
        if kind is None:
            kind = {".xml": "xml", ".nxml": "xml", ".html": "html", ".htm": "html", ".pdf": "pdf"}.get(suf, "text")
        if kind == "pdf":
            from pypdf import PdfReader  # optional dependency
            reader = PdfReader(str(p))
            text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
        else:
            text = p.read_text(errors="ignore")
    else:
        text = str(source)
        if kind is None:
            kind = "xml" if "<table-wrap" in text else ("html" if "<table" in text.lower() else "text")
    rows: List[dict] = []
    if kind in ("xml", "html"):
        for i, (cap, t) in enumerate(_tables_from_markup(text)):
            tid = re.search(r"Table\s*\w+", cap)
            rows += _table_candidates(cap, _grid(t), tid.group(0) if tid else f"table{i + 1}")
        if include_text:
            body = re.sub(r"<table-wrap\b.*?</table-wrap>|<table\b.*?</table>|<ref-list\b.*?</ref-list>", " ",
                          text, flags=re.S | re.I)
            paras = re.findall(r"<p\b.*?</p>", body, re.S | re.I)
            plain = "\n".join(_strip_tags(x) for x in paras) if paras else _strip_tags(body)
            rows += _text_candidates(plain, name)
    else:
        if kind == "pdf":
            rows += _pdf_column_tables(text, name)
            rows += _pdf_stream_tables(text, name)
        rows += _text_candidates(text, name)
    led = pd.DataFrame(rows, columns=["table", "caption", "section", "row", "group", "n_header", "mean", "sd",
                                      "raw", "layout", "row_no", "context", "stat", "lo", "hi"])
    led["stat"] = led["stat"].fillna("mean_sd")
    led.attrs["source"] = name
    return led


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------

@register_function(
    aliases=["合并亚组", "pool subgroups", "combine groups mean sd", "Cochrane combine groups"],
    category="utils",
    description="Combine several subgroups (n, mean, SD) into one sample: n = sum n_g, mean = weighted mean, "
                "SD from within- plus between-subgroup sums of squares (Cochrane Handbook 6.5.2.10).",
    examples=["ov.tl.pool_group_stats([15, 15], [4.2, 4.8], [3.1, 4.4])"],
    related=["tl.hedges_g", "tl.meta_effect_sizes"],
)
def pool_group_stats(n: Sequence[float], mean: Sequence[float], sd: Sequence[float]) -> Tuple[float, float, float]:
    """Return ``(n, mean, sd)`` of the union of the subgroups (unrounded inputs, please)."""
    n = np.asarray(n, float)
    m = np.asarray(mean, float)
    s = np.asarray(sd, float)
    N = n.sum()
    M = (n * m).sum() / N
    ss = ((n - 1) * s ** 2).sum() + (n * (m - M) ** 2).sum()
    return float(N), float(M), float(math.sqrt(ss / (N - 1)))


@register_function(
    aliases=["Hedges g", "效应量", "standardized mean difference", "Cohen d small sample correction"],
    category="utils",
    description="Standardised mean difference of two groups: pooled SD, Cohen's d, Hedges' g with "
                "J = 1 - 3/(4N - 9) (or the exact gamma-function J), its variance, SE and 95% CI. "
                "Vectorised; sign is case - control.",
    examples=["ov.tl.hedges_g(20, 17.1, 5.0, 20, 20.1, 1.9)"],
    related=["tl.pool_group_stats", "tl.meta_effect_sizes"],
)
def hedges_g(n1, m1, s1, n2, m2, s2, correction: str = "approx") -> pd.DataFrame:
    """Return a DataFrame with ``sd_pooled, d, J, g, var_g, se_g, ci_low, ci_high``.

    ``correction='approx'`` uses ``J = 1 - 3 / (4 (n1 + n2) - 9)``; ``'exact'`` the gamma form;
    ``'none'`` returns g = d.
    """
    n1, m1, s1, n2, m2, s2 = (np.atleast_1d(np.asarray(x, float)) for x in (n1, m1, s1, n2, m2, s2))
    N = n1 + n2
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (N - 2))
    d = (m1 - m2) / sp
    if correction == "approx":
        J = 1 - 3 / (4 * N - 9)
    elif correction == "exact":
        from scipy.special import gammaln
        df = N - 2
        J = np.exp(gammaln(df / 2) - 0.5 * np.log(df / 2) - gammaln((df - 1) / 2))
    else:
        J = np.ones_like(N)
    g = d * J
    var_d = N / (n1 * n2) + d ** 2 / (2 * N)
    var_g = J ** 2 * var_d
    se = np.sqrt(var_g)
    return pd.DataFrame({"sd_pooled": sp, "d": d, "J": J, "g": g, "var_g": var_g, "se_g": se,
                         "ci_low": g - 1.959964 * se, "ci_high": g + 1.959964 * se})


@register_function(
    aliases=["标准误转标准差", "SE to SD", "CI to SD", "standard deviation from confidence interval"],
    category="utils",
    description="Recover a group SD from a standard error (SD = SE sqrt(n)) or from a confidence interval of "
                "the mean (SD = sqrt(n) (upper - lower) / (2 t_{n-1}) or z for large n).",
    examples=["ov.tl.sd_from_summary(n=20, se=1.1)", "ov.tl.sd_from_summary(n=20, ci=(10.2, 14.8))"],
    related=["tl.hedges_g"],
)
def sd_from_summary(n: float, se: Optional[float] = None, ci: Optional[Tuple[float, float]] = None,
                    level: float = 0.95, use_t: bool = True) -> float:
    """Return the SD of a group of size ``n`` from its standard error or a CI of its mean."""
    if se is not None:
        return float(se) * math.sqrt(n)
    if ci is not None:
        from scipy import stats
        q = stats.t.ppf(0.5 + level / 2, n - 1) if use_t else stats.norm.ppf(0.5 + level / 2)
        return math.sqrt(n) * (ci[1] - ci[0]) / (2 * q)
    raise ValueError("give se or ci")


@register_function(
    aliases=["研究效应量表", "meta effect sizes", "study-level effects table", "compute hedges g for ledger"],
    category="utils",
    description="Compute study-level effects for a ledger of kept comparisons: pools case subgroups given as "
                "lists, checks the six statistics (integer n >= 2, SD > 0, finite), computes Hedges' g "
                "(raw case - control sign unless told otherwise), flags duplicate rows with identical "
                "statistics, and returns the table ready to write.",
    examples=["out = ov.tl.meta_effect_sizes(kept, case='mdd', control='control')"],
    related=["tl.extract_group_stats", "tl.pool_group_stats", "tl.hedges_g"],
)
def meta_effect_sizes(ledger: pd.DataFrame, case: str = "case", control: str = "control",
                      correction: str = "approx", flip: Optional[pd.Series] = None,
                      digits: Optional[int] = None) -> pd.DataFrame:
    """Return ``ledger`` with pooled case statistics, ``d`` (Hedges-corrected) and checks.

    Expected columns: ``n_<case>, mean_<case>, sd_<case>, n_<control>, mean_<control>, sd_<control>``.
    A case cell may hold a list / tuple (one entry per subgroup); those rows are pooled with
    :func:`pool_group_stats`.  ``flip`` (bool Series) reverses the sign for rows where the method
    orients effects by poorer performance -- leave it ``None`` to keep the raw difference.
    Added columns: ``d``, ``var_g``, ``pooled_from`` (number of case subgroups), ``issues``.
    """
    df = ledger.copy()
    nc, mc, sc = f"n_{case}", f"mean_{case}", f"sd_{case}"
    nk, mk, sk = f"n_{control}", f"mean_{control}", f"sd_{control}"
    pooled = []
    for i, r in df.iterrows():
        if isinstance(r[nc], (list, tuple, np.ndarray)):
            N, M, S = pool_group_stats(r[nc], r[mc], r[sc])
            df.at[i, nc], df.at[i, mc], df.at[i, sc] = N, M, S
            pooled.append(len(r[nc]))
        else:
            pooled.append(1)
    for c in (nc, mc, sc, nk, mk, sk):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    es = hedges_g(df[nc], df[mc], df[sc], df[nk], df[mk], df[sk], correction=correction)
    sign = np.where(flip.values, -1.0, 1.0) if flip is not None else 1.0
    df["d"] = es["g"].values * sign
    df["var_g"] = es["var_g"].values
    df["pooled_from"] = pooled
    issues = []
    stat_cols = [nc, mc, sc, nk, mk, sk]
    dup = df.duplicated(subset=[c for c in (["doi"] if "doi" in df else []) + stat_cols], keep=False)
    for i, r in df.iterrows():
        msg = []
        if any(pd.isna(r[c]) for c in stat_cols):
            msg.append("missing statistic")
        for c in (nc, nk):
            if not pd.isna(r[c]) and (r[c] < 2 or float(r[c]) != int(r[c])):
                msg.append(f"{c} not an integer >= 2")
        for c in (sc, sk):
            if not pd.isna(r[c]) and r[c] <= 0:
                msg.append(f"{c} <= 0 (zero variance: check the source)")
        if dup.loc[i]:
            msg.append("duplicate statistics")
        issues.append("; ".join(msg))
    df["issues"] = issues
    if digits is not None:
        df["d"] = df["d"].round(digits)
    return df


@register_function(
    aliases=["组间比较表", "case control comparisons from ledger", "pair groups for meta-analysis"],
    category="utils",
    description="From an extract_group_stats ledger, build one case-vs-control comparison per (table, row): "
                "case columns matched by a regex are pooled (subgroups), the control column by another regex; "
                "n from the column header unless an explicit {group regex: n} map is given; Hedges' g "
                "(case - control). Rows lacking either side are dropped.",
    examples=["cmp = ov.tl.group_comparisons(led, case=r'depress|MDD', control=r'control|healthy|HC')"],
    related=["tl.extract_group_stats", "tl.meta_effect_sizes"],
)
def group_comparisons(ledger: pd.DataFrame, case: str, control: str, n_map: Optional[dict] = None,
                      exclude: Optional[str] = None, flags: int = re.I) -> pd.DataFrame:
    """Return ``table, row, case_groups, n_case, mean_case, sd_case, n_control, mean_control,
    sd_control, d`` (d = Hedges' g, case - control), one row per (table, row) that has both sides.

    ``exclude`` drops groups matching it before matching ``case`` (e.g. a bipolar or a
    comorbid group whose name also contains the case word). Group n comes from ``n_map``
    (first regex that matches the group name) or the column header.
    """
    led = ledger.copy()
    if exclude:
        led = led[~led["group"].astype(str).str.contains(exclude, flags=flags, regex=True)]

    def _n(g, nh):
        for pat, n in (n_map or {}).items():
            if re.search(pat, str(g), flags):
                return n
        return nh

    led["n"] = pd.to_numeric(pd.Series([_n(g, nh) for g, nh in zip(led["group"], led["n_header"])], index=led.index), errors="coerce")
    # printed summary statistics -> mean / SD (SE, median + range / IQR), with each group's n
    if "stat" in led:
        conv = [to_mean_sd(k, n, m, sd_, lo, hi) for k, n, m, sd_, lo, hi in
                zip(led["stat"].fillna("mean_sd"), led["n"], led["mean"], led["sd"],
                    led["lo"] if "lo" in led else [np.nan] * len(led), led["hi"] if "hi" in led else [np.nan] * len(led))]
        led["mean"] = [c[0] for c in conv]
        led["sd"] = [c[1] for c in conv]
    # control wins for names matching both patterns ("non-depressed", "healthy non-patients")
    led["is_ctrl"] = led["group"].astype(str).str.contains(control, flags=flags, regex=True)
    led["is_case"] = led["group"].astype(str).str.contains(case, flags=flags, regex=True) & ~led["is_ctrl"]
    out = []
    key = ["table", "row", "row_no"] if "row_no" in led else ["table", "row"]
    led = led.assign(row_no=led["row_no"].fillna(-1)) if "row_no" in led else led
    norm_row = lambda x: re.sub(r"[^a-z0-9%]+", " ", ": ".join(str(x).split(": ")[-2:]).lower()).strip()
    ctrl_rows = led[led["is_ctrl"]].assign(_nr=lambda d: d["row"].map(norm_row))
    for kk, g in led.groupby(key, sort=False):
        t, r = kk[0], kk[1]
        ca, co = g[g["is_case"]], g[g["is_ctrl"]]
        if not ca.empty and co.empty:
            # a table of case subgroups only: the control is printed for the same measure elsewhere
            other = ctrl_rows[(ctrl_rows["_nr"] == norm_row(r)) & (ctrl_rows["table"] != t)]
            if len(other):
                co = other.drop(columns=["_nr"]).iloc[[0]]
        if ca.empty or co.empty:
            continue
        if ca["n"].isna().any():
            # sizes unknown: keep the row visible (n / pooled stats NaN) so it is decided, not lost
            N = float("nan")
            M = float(ca["mean"].iloc[0]) if len(ca) == 1 else float("nan")
            S = float(ca["sd"].iloc[0]) if len(ca) == 1 else float("nan")
        elif len(ca) > 1:
            N, M, S = pool_group_stats(ca["n"].astype(float), ca["mean"], ca["sd"])
        else:
            N, M, S = float(ca["n"].iloc[0]), float(ca["mean"].iloc[0]), float(ca["sd"].iloc[0])
        stats_ = ",".join(sorted(set(g["stat"].fillna("mean_sd")))) if "stat" in g else "mean_sd"
        for _, c in co.iterrows():
            out.append({"table": t, "section": g["section"].iloc[0] if "section" in g else "", "row": r,
                        "printed": stats_,
                        "layout": g["layout"].iloc[0], "case_groups": " + ".join(ca["group"].astype(str)),
                        "case_subgroups": [(None if pd.isna(a) else float(a), float(b), float(c_))
                                           for a, b, c_ in zip(ca["n"], ca["mean"], ca["sd"])],
                        "control_group": c["group"], "n_case": N, "mean_case": M, "sd_case": S,
                        "n_control": float(c["n"]) if pd.notna(c["n"]) else float("nan"),
                        "mean_control": float(c["mean"]),
                        "sd_control": float(c["sd"])})
    res = pd.DataFrame(out)
    if len(res):
        res["d"] = hedges_g(res["n_case"], res["mean_case"], res["sd_case"], res["n_control"],
                            res["mean_control"], res["sd_control"])["g"].values
    return res


@register_function(
    aliases=["汇总统计量转均值标准差", "SE to SD", "median range to mean SD", "Hozo", "Wan IQR conversion"],
    category="utils",
    description="Convert a printed group summary to mean and SD: mean (SE) -> SD = SE sqrt(n); median with "
                "range -> Hozo et al. 2005 (mean (a+2m+b)/4 for n <= 25 else m; SD from the range with the "
                "n <= 15 / <= 70 / > 70 rules); median with IQR -> Wan et al. 2014 (mean (q1+m+q3)/3, SD = "
                "(q3-q1)/1.35). kind: mean_sd | mean_se | median_range | median_iqr.",
    examples=["ov.tl.to_mean_sd('median_range', 120, 54, None, 22, 71)"],
    related=["tl.group_comparisons", "tl.sd_from_summary"],
)
def to_mean_sd(kind: str, n, mean, sd=None, lo=None, hi=None) -> Tuple[float, float]:
    """Return ``(mean, sd)``; NaN where n is needed but unknown."""
    nan = float("nan")
    n = float(n) if n is not None and not (isinstance(n, float) and math.isnan(n)) else nan
    if kind in (None, "mean_sd") or (isinstance(kind, float)):
        return float(mean), (float(sd) if sd is not None else nan)
    if kind == "mean_se":
        return float(mean), (float(sd) * math.sqrt(n) if not math.isnan(n) else nan)
    m, a, b = float(mean), float(lo), float(hi)
    if kind == "median_iqr":
        return (a + m + b) / 3.0, (b - a) / 1.35
    if math.isnan(n):
        return nan, nan
    mu = (a + 2 * m + b) / 4.0 if n <= 25 else m
    if n <= 15:
        sdv = math.sqrt(((a - 2 * m + b) ** 2 / 4.0 + (b - a) ** 2) / 12.0)
    elif n <= 70:
        sdv = (b - a) / 4.0
    else:
        sdv = (b - a) / 6.0
    return mu, sdv


@register_function(
    aliases=["重算检验统计量", "recompute t test from mean sd", "check reported F against extracted stats"],
    category="utils",
    description="Recompute the pooled-variance two-group t (and F = t^2, two-sided p) from extracted N / mean / "
                "SD, to check an extraction against the t or F the article reports (wrong column order, a "
                "swapped group, SE printed as SD, or a misread digit show up as a mismatch).",
    examples=["ov.tl.recompute_group_test(20, 146.07, 65.66, 20, 69.75, 5.72, reported_F=23.9)"],
    related=["tl.extract_group_stats", "tl.hedges_g"],
)
def recompute_group_test(n1, m1, s1, n2, m2, s2, reported_t: Optional[float] = None,
                         reported_F: Optional[float] = None, rtol: float = 0.15) -> dict:
    """Return ``{'t', 'F', 'p', 'df', 'matches'}``; ``matches`` is None without a reported value.

    Reported statistics from ANCOVA / covariate-adjusted or Welch tests differ legitimately, so a
    mismatch is a prompt to re-read the source, not proof of an error; an F that matches
    ``t^2`` within ``rtol`` confirms the six numbers and their group assignment up to sign.
    """
    from scipy import stats
    df = n1 + n2 - 2
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df)
    t = (m1 - m2) / (sp * math.sqrt(1 / n1 + 1 / n2)) if sp > 0 else float("nan")
    p = 2 * stats.t.sf(abs(t), df) if np.isfinite(t) else float("nan")
    match = None
    if reported_F is not None:
        match = bool(abs(t * t - reported_F) <= rtol * max(abs(reported_F), 1e-9))
    elif reported_t is not None:
        match = bool(abs(abs(t) - abs(reported_t)) <= rtol * max(abs(reported_t), 1e-9))
    return {"t": t, "F": t * t, "p": p, "df": df, "matches": match}


@register_function(
    aliases=["应用测量选择规则", "apply measure selection rules", "meta-analysis admissibility rules",
             "decide candidate comparisons"],
    category="utils",
    description="Decide every candidate comparison of a meta-analysis ledger mechanically from a written "
                "rule table: each candidate carries (study, task family, statistic class); a family absent "
                "from the rules is out of scope; a class listed in the family's 'select' is kept; a class "
                "listed under 'fallback' is kept only when the same study reports none of its preferred "
                "classes for that family; anything else is excluded. Returns decision + reason per row and "
                "flags families / classes used inconsistently.",
    examples=["rules = {'task A': {'pattern': r'task a|ta-test', 'select': {'reaction time': r'\\bRT\\b|latency',\n"
              "                                                             'errors': r'error|commission|omission'}},\n"
              "         'task B': {'pattern': r'task b', 'select': {'score X': r'score x'},\n"
              "                    'fallback': {'score Y': ['score X']}, 'class_patterns': {'score Y': r'score y'}}}",
              "dec = ov.tl.apply_selection_rules(ledger, rules, study='doi')"],
    related=["tl.extract_group_stats", "tl.meta_effect_sizes"],
)
def apply_selection_rules(ledger: pd.DataFrame, rules: dict, study: str = "study",
                          family: str = "family", stat_class: str = "stat_class",
                          text_cols: Sequence[str] = ("section", "row")) -> pd.DataFrame:
    """Return ``ledger`` with ``family``, ``stat_class``, ``decision`` ('keep' / 'exclude') and
    ``reason`` columns.

    ``rules`` maps a task family (or a comparison-measure family) to::

        {"pattern": r"<regex naming the task in a row / section>",          # optional
         "select": {"<class>": r"<regex of the statistic in the row>", ...}  # or a list of names
         "fallback": {"<class>": ["<preferred class>", ...]},                # optional
         "class_patterns": {"<fallback class>": r"<regex>"},                 # regexes for fallbacks
         "quote": "<the method's sentence>"}

    Labelling: rows whose ``family`` / ``stat_class`` columns are filled keep their labels;
    otherwise the family is the first rule whose ``pattern`` matches the row's text
    (``text_cols`` joined) and the class the first selected / fallback class whose regex matches
    it. Write the table from the method paper / instruction BEFORE deciding, then let this
    function decide; review rows reported as unmatched. Decision: family absent -> out of scope;
    class in ``select`` -> keep; class in ``fallback`` -> keep only when the same study reports
    none of its preferred classes for that family; else exclude.
    """
    norm = lambda x: re.sub(r"\s+", " ", str(x)).strip().casefold()
    R = {}
    for f, r in rules.items():
        sel = r.get("select", [])
        sel_pat = dict(sel) if isinstance(sel, dict) else {c: None for c in sel}
        fb = r.get("fallback") or {}
        cpat = dict(r.get("class_patterns") or {})
        R[norm(f)] = {"pattern": r.get("pattern"),
                      "select": {norm(c) for c in sel_pat},
                      "fallback": {norm(c): {norm(p) for p in ps} for c, ps in fb.items()},
                      "class_re": [(norm(c), pt) for c, pt in list(sel_pat.items()) + list(cpat.items()) if pt]}
    df = ledger.copy()
    for c in (family, stat_class):
        if c not in df:
            df[c] = ""
    text = df[[c for c in text_cols if c in df]].astype(str).agg(" ".join, axis=1) if any(
        c in df for c in text_cols) else pd.Series("", index=df.index)
    fam_l, cls_l = [], []
    for i, t in text.items():
        f_ = norm(df.at[i, family]) if pd.notna(df.at[i, family]) else ""
        if f_ in ("", "nan"):
            f_ = next((k for k, v in R.items() if v["pattern"] and re.search(v["pattern"], t, re.I)), "")
        c_ = norm(df.at[i, stat_class]) if pd.notna(df.at[i, stat_class]) else ""
        if c_ in ("", "nan") and f_ in R:
            c_ = next((k for k, pt in R[f_]["class_re"] if re.search(pt, t, re.I)), "")
        fam_l.append(f_)
        cls_l.append(c_)
    df[family], df[stat_class] = fam_l, cls_l
    present = set(zip(df[study].map(norm), df[family], df[stat_class]))
    dec, why = [], []
    for s_, f_, c_ in zip(df[study].map(norm), df[family], df[stat_class]):
        if not f_:
            dec.append("exclude"); why.append("no family of the rule table matches (out of scope, or label it)")
        elif f_ not in R:
            dec.append("exclude"); why.append(f"family '{f_}' not in the method's scope")
        elif not c_:
            dec.append("exclude"); why.append(f"no selected statistic of '{f_}' matches this row")
        elif c_ in R[f_]["select"]:
            dec.append("keep"); why.append(f"'{c_}' is a selected statistic of '{f_}'")
        elif c_ in R[f_]["fallback"]:
            pref = R[f_]["fallback"][c_]
            if any((s_, f_, p) in present for p in pref):
                dec.append("exclude"); why.append(f"fallback '{c_}' not used: study reports {sorted(pref)}")
            else:
                dec.append("keep"); why.append(f"fallback '{c_}' used: study reports none of {sorted(pref)}")
        else:
            dec.append("exclude"); why.append(f"'{c_}' is not a selected statistic of '{f_}'")
    df["decision"], df["reason"] = dec, why
    return df

_CONTROL_DEFAULT = r"control|healthy|\bHC\b|\bNC\b|\bCTL\b|comparison|non[- ]?\w+|\bno\s+\w+|\bnorm(?:ative)?\b"


@register_function(
    aliases=["荟萃分析工作簿", "comparison workbook", "meta-analysis extraction workbook",
             "all case-control comparisons of all articles"],
    category="utils",
    description="One table of EVERY case-vs-control comparison printed in a set of articles: runs "
                "extract_group_stats on each file, pairs case groups (pooled when several) with the "
                "control group per table row / in-text statistic, adds Hedges' g, and leaves empty "
                "family / stat_class columns to be labelled and decided with apply_selection_rules. "
                "Rows whose group sizes are unknown are kept with NaN n (fill them from the text).",
    examples=["wb = ov.tl.comparison_workbook({doi: path, ...}, case=r'depress|patient', "
              "n_map={doi: {r'HC': 14}})",
              "wb.to_csv('/app/data/workbook.csv', index=False)"],
    related=["tl.extract_group_stats", "tl.apply_selection_rules", "tl.meta_effect_sizes"],
)
def comparison_workbook(sources: dict, case: Union[str, dict], control: Union[str, dict] = _CONTROL_DEFAULT,
                        exclude: Union[None, str, dict] = None, n_map: Optional[dict] = None,
                        include_text: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Return the workbook: one row per (study, table row / text statistic, control group).

    ``sources`` maps a study id (e.g. the DOI) to one path or a list of paths (XML / HTML / PDF /
    text of the same article). ``case``, ``control`` and ``exclude`` are regexes on group
    names, or dicts study -> regex for study-specific names (``control`` is tried first, so a
    group named e.g. "non-depressed" is a control). ``n_map``: study -> {group regex: n}.

    Columns: ``study, source, table, section, row, layout, case_groups, case_subgroups,
    control_group, n_case, mean_case, sd_case, n_control, mean_control, sd_control, d,
    family, stat_class`` (the last two empty, to label).
    """
    pick = lambda v, k: (v.get(k) if isinstance(v, dict) else v)
    frames = []
    for sid, paths in sources.items():
        if isinstance(paths, (str, Path)):
            paths = [paths]
        for pth in paths:
            led = extract_group_stats(pth, include_text=include_text)
            if led.empty:
                continue
            ctl = pick(control, sid) or _CONTROL_DEFAULT
            cs = pick(case, sid)
            # control wins over case for names matching both ("non-depressed")
            led = led.copy()
            led["group"] = led["group"].astype(str)
            is_ctl = led["group"].str.contains(ctl, flags=re.I, regex=True)
            led.loc[is_ctl, "group"] = "[control] " + led.loc[is_ctl, "group"]
            cmp_ = group_comparisons(led, case=cs, control=r"^\[control\] ", n_map=(n_map or {}).get(sid),
                                     exclude=pick(exclude, sid))
            if cmp_.empty:
                continue
            cmp_["control_group"] = cmp_["control_group"].str.replace(r"^\[control\] ", "", regex=True)
            cmp_.insert(0, "source", Path(str(pth)).name)
            cmp_.insert(0, "study", sid)
            frames.append(cmp_)
    if not frames:
        return pd.DataFrame()
    wb = pd.concat(frames, ignore_index=True)
    wb["family"] = ""
    wb["stat_class"] = ""
    if verbose:
        print(format_workbook(wb))
    return wb


def _fmt(n, m, s_):
    n_ = "n=?" if pd.isna(n) else f"n={n:g}"
    return f"{n_} {m:.4g} ({s_:.4g})" if pd.notna(m) else f"{n_} pooled?"


def format_workbook(wb: pd.DataFrame, max_label: int = 70) -> str:
    """Compact text listing of a comparison workbook, one line per comparison, grouped by study:
    ``[index] row | case n mean (sd) | control n mean (sd) | d``; rows with unknown n are marked."""
    lines = []
    for sid, g in wb.groupby("study", sort=False):
        unk = int(g["n_case"].isna().sum() + g["n_control"].isna().sum())
        lines.append(f"== {sid}: {len(g)} comparisons" + (f", {unk} group sizes unknown (fill n)" if unk else ""))
        for i, r in g.iterrows():
            lab = str(r["row"])[-max_label:]
            d = "" if pd.isna(r.get("d")) else f" | d={r['d']:.3f}"
            lines.append(f"  [{i}] {lab} | {r['case_groups']} {_fmt(r['n_case'], r['mean_case'], r['sd_case'])}"
                         f" | {r['control_group']} {_fmt(r['n_control'], r['mean_control'], r['sd_control'])}{d}")
    return "\n".join(lines)


_MEASURE_PAT = re.compile(
    r"(dependent (measure|variable)s?|measures? included|other measures|primary (outcome|measure)|outcome measures?|(the )?measures? (is|are|was|were)"
    r"|were (coded|included|analy[sz]ed|computed|combined|averaged|excluded)|was (coded|used|included|analy[sz]ed)"
    r"|only when|if (no|not) |rather than|instead of|composite|analy[sz]ed (individually|separately|together)"
    r"|effect sizes? (was|were) (computed|calculated)|pooled|subgroups?|direction|sign)", re.I)


@register_function(
    aliases=["方法学测量选择语句", "method measure statements", "extract measure selection rules from method paper"],
    category="utils",
    description="List the sentences of a method paper (meta-analysis / systematic review) that define what is "
                "coded: dependent / outcome measures per task, fallbacks ('only when', 'instead of'), "
                "composites, inclusion / exclusion, effect-size computation and pooling -- each with the "
                "section heading it sits under, so a measure-selection rule table can be written from the "
                "method's own words without reading the whole paper.",
    examples=["st = ov.tl.method_measure_statements('/app/data/method_paper.xml'); print(st.to_string())"],
    related=["tl.apply_selection_rules", "tl.comparison_workbook"],
)
def method_measure_statements(source: Union[str, Path], extra_pattern: Optional[str] = None) -> pd.DataFrame:
    """Return ``section, sentence, context`` for every measure-defining sentence (tables and references
    removed); ``context`` is the sentence before it, which often names the task."""
    p = Path(str(source)) if len(str(source)) < 4096 else None
    text = p.read_text(errors="ignore") if (p is not None and p.exists()) else str(source)
    if p is not None and p.exists() and p.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        text = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
    table_rows = _coding_table_statements(text)
    text = re.sub(r"<table-wrap\b.*?</table-wrap>|<table\b.*?</table>|<ref-list\b.*?</ref-list>"
                  r"|<script\b.*?</script>|<style\b.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"(?m)^\s*\|.*\|\s*$", " ", text)          # markdown tables handled above
    blocks: List[Tuple[str, str]] = []
    if re.search(r"<(p|title|h[1-6])\b", text, re.I):
        sec = ""
        for m in re.finditer(r"<(title|h[1-6])\b[^>]*>(.*?)</\1>|<p\b[^>]*>(.*?)</p>", text, re.S | re.I):
            if m.group(1):
                sec = _strip_tags(m.group(2))[:120]
            else:
                blocks.append((sec, _strip_tags(m.group(3))))
    else:
        sec = ""
        for para in re.split(r"\n\s*\n", text):
            para = re.sub(r"\s+", " ", para).strip()
            if 0 < len(para) < 80 and not para.endswith("."):
                sec = para
            elif para:
                blocks.append((sec, para))
    pat = re.compile(_MEASURE_PAT.pattern + (f"|{extra_pattern}" if extra_pattern else ""), re.I)
    rows = []
    for sec, para in blocks:
        sents = re.split(r"(?<=[.])\s+(?=[A-Z(])", para)
        for i, sent in enumerate(sents):
            if 25 <= len(sent) <= 800 and pat.search(sent):
                # the neighbouring sentences often name the task a measure sentence refers to
                rows.append({"section": sec, "sentence": sent.strip(),
                             "context": sents[i - 1].strip() if i > 0 else "",
                             "context_after": sents[i + 1].strip() if i + 1 < len(sents) else ""})
    return pd.DataFrame(table_rows + rows,
                        columns=["section", "sentence", "context", "context_after"]).drop_duplicates()


def _coding_table_statements(text: str) -> List[dict]:
    """A coding table (markdown ``| domain | test | scores coded |``, as instructions and protocols
    give it) -> one measure statement per test: "For the <test>, the dependent measures are <scores>."
    Rows of the same test are merged; a blank first cell continues the previous domain."""
    out: Dict[str, dict] = {}
    lines = [ln.strip() for ln in str(text).splitlines()]
    i = 0
    while i < len(lines):
        if not (lines[i].startswith("|") and lines[i].endswith("|")):
            i += 1
            continue
        block = []
        while i < len(lines) and lines[i].startswith("|") and lines[i].endswith("|"):
            block.append([c.strip() for c in lines[i].strip("|").split("|")])
            i += 1
        if len(block) < 3 or not all(set(c) <= set("-: ") for c in block[1]):
            continue
        hdr = [h.lower() for h in block[0]]
        sc = next((k for k, h in enumerate(hdr) if re.search(r"score|measure|outcome|statistic|variable", h)), None)
        tc = next((k for k, h in enumerate(hdr) if re.search(r"\btest|task|instrument|scale", h) and k != sc), None)
        if sc is None or tc is None:
            continue
        dc = next((k for k, h in enumerate(hdr) if k not in (sc, tc)), None)
        dom = ""
        for row in block[2:]:
            if len(row) <= max(sc, tc):
                continue
            if dc is not None and dc < len(row) and row[dc]:
                dom = row[dc]
            if not row[tc]:
                continue
            for test in [t.strip() for t in row[tc].split(";") if t.strip()]:
                scores = row[sc]
                key = test.lower()
                if key in out:
                    out[key]["scores"].append(scores)
                    out[key]["domains"].append(dom)
                else:
                    out[key] = {"test": test, "scores": [scores], "domains": [dom]}
    rows = []
    for v in out.values():
        scores = ", ".join(dict.fromkeys(x.replace(";", ",") for x in v["scores"] if x))
        doms = ", ".join(dict.fromkeys(d for d in v["domains"] if d))
        rows.append({"section": doms or "coding table",
                     "sentence": f"For the {v['test']}, the dependent measures are {scores}.",
                     "context": f"{v['test']} is a coded test.",
                     "context_after": ""})
    return rows
