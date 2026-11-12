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
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._registry import register_function

__all__ = ["parse_mean_sd", "extract_group_stats", "group_comparisons", "pool_group_stats", "hedges_g",
           "sd_from_summary", "recompute_group_test", "meta_effect_sizes", "apply_selection_rules"]

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
    re.compile(r"^\s*(?P<m>" + _NUM + r")\s*[\(\[]\s*(?:s\.?\s*d\.?\s*[:=]?\s*)?(?P<s>" + _NUM + r")\s*[\)\]]\s*[*†‡§a-g]*\s*$", re.I),
    # 12.3 ± 4.5 / 12.3 +/- 4.5
    re.compile(r"^\s*(?P<m>" + _NUM + r")\s*(?:±|\+/-|\+-|\+/−)\s*(?P<s>" + _NUM + r")\s*[*†‡§a-g]*\s*$"),
]


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
    s = re.sub(r"\s+", " ", str(x)).strip()
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


def _strip_tags(s: str) -> str:
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
                row.append(txt)
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
    return _to_float(s) is not None or parse_mean_sd(s) is not None


def _table_candidates(caption: str, grid: List[List[str]], table_id: str) -> List[dict]:
    if not grid:
        return []
    top = [bool(r) and str(r[0]).startswith("\x01") for r in grid]
    grid = [[str(c).replace("\x01", "") for c in r] for r in grid]
    ncol = len(grid[0])
    # header rows: leading rows with no numeric cell outside the first column
    h = 0
    def _section_like(row):
        # only a label (a spanning section cell repeats the label in every column)
        first = _clean(row[0])
        return bool(first) and all(_clean(c) in ("", first) for c in row[1:])

    while h < len(grid) and h < 4 and not any(_is_num(c) for c in grid[h][1:]) \
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
        if c > 0 and any(parse_mean_sd(v) is not None for v in vals):
            break
        if vals and sum(_is_num(v) for v in vals) / len(vals) < 0.5:
            nlab += 1
        else:
            break
    nlab = max(nlab, 1)
    out = []
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
        while c < ncol:
            cell = cells[c]
            grp = header[c] if c < len(header) else ""
            pair = parse_mean_sd(cell)
            if pair is not None:
                out.append({"table": table_id, "caption": caption, "row": full, "group": grp,
                            "n_header": _n_of(grp), "mean": pair[0], "sd": pair[1], "raw": cell,
                            "layout": "cell", "row_no": bi, "section": " > ".join(section)})
                c += 1
                continue
            if sub[c] and _M_HDR.match(sub[c]) and c + 1 < ncol and _SD_HDR.match(sub[c + 1] or ""):
                m_, s_ = _to_float(cell), _to_float(cells[c + 1])
                if m_ is not None and s_ is not None:
                    out.append({"table": table_id, "caption": caption, "row": full, "group": grp,
                                "n_header": _n_of(grp), "mean": m_, "sd": s_,
                                "raw": f"{cell} | {cells[c + 1]}", "layout": "M|SD columns",
                                "row_no": bi, "section": " > ".join(section)})
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
            out.append({"table": f"{source}:line{li + 1}", "caption": "", "row": label[-120:], "group": group,
                        "n_header": _n_of(ctx), "mean": a, "sd": b, "raw": raw, "layout": "text",
                        "context": ctx.strip()})
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
        rows += _text_candidates(text, name)
    led = pd.DataFrame(rows, columns=["table", "caption", "section", "row", "group", "n_header", "mean", "sd",
                                      "raw", "layout", "row_no", "context"])
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

    led["n"] = [_n(g, nh) for g, nh in zip(led["group"], led["n_header"])]
    led["is_case"] = led["group"].astype(str).str.contains(case, flags=flags, regex=True)
    led["is_ctrl"] = led["group"].astype(str).str.contains(control, flags=flags, regex=True) & ~led["is_case"]
    out = []
    key = ["table", "row", "row_no"] if "row_no" in led else ["table", "row"]
    led = led.assign(row_no=led["row_no"].fillna(-1)) if "row_no" in led else led
    for kk, g in led.groupby(key, sort=False):
        t, r = kk[0], kk[1]
        ca, co = g[g["is_case"]], g[g["is_ctrl"]]
        if ca.empty or co.empty or ca["n"].isna().any() or co["n"].isna().any():
            continue
        if len(ca) > 1:
            N, M, S = pool_group_stats(ca["n"], ca["mean"], ca["sd"])
        else:
            N, M, S = float(ca["n"].iloc[0]), float(ca["mean"].iloc[0]), float(ca["sd"].iloc[0])
        for _, c in co.iterrows():
            out.append({"table": t, "row": r, "case_groups": " + ".join(ca["group"].astype(str)),
                        "control_group": c["group"], "n_case": N, "mean_case": M, "sd_case": S,
                        "n_control": float(c["n"]), "mean_control": float(c["mean"]),
                        "sd_control": float(c["sd"])})
    res = pd.DataFrame(out)
    if len(res):
        res["d"] = hedges_g(res["n_case"], res["mean_case"], res["sd_case"], res["n_control"],
                            res["mean_control"], res["sd_control"])["g"].values
    return res


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
    examples=["rules = {'task family A': {'select': ['reaction time', 'errors'], 'quote': '...'},\n"
              "         'task family B': {'select': ['score X'], 'fallback': {'score Y': ['score X']}}}",
              "dec = ov.tl.apply_selection_rules(ledger, rules, study='doi')"],
    related=["tl.extract_group_stats", "tl.meta_effect_sizes"],
)
def apply_selection_rules(ledger: pd.DataFrame, rules: dict, study: str = "study",
                          family: str = "family", stat_class: str = "stat_class") -> pd.DataFrame:
    """Return ``ledger`` with ``decision`` ('keep' / 'exclude') and ``reason`` columns.

    ``rules`` maps a task family (or a comparison-measure family) to
    ``{"select": [classes], "fallback": {class: [preferred classes]}, "quote": "..."}``.
    Write the rule table from the method paper / instruction BEFORE deciding, with the quote
    that names the classes; label each candidate's family and class from the article; then let
    this function decide. Names are compared case-insensitively after stripping.
    Rows whose family or class is empty are returned with decision 'undecided'.
    """
    norm = lambda x: re.sub(r"\s+", " ", str(x)).strip().casefold()
    R = {norm(f): {"select": {norm(c) for c in r.get("select", [])},
                   "fallback": {norm(c): {norm(p) for p in ps} for c, ps in (r.get("fallback") or {}).items()}}
         for f, r in rules.items()}
    df = ledger.copy()
    fam = df[family].map(norm)
    cls = df[stat_class].map(norm)
    present = set(zip(df[study].map(norm), fam, cls))
    dec, why = [], []
    for s_, f_, c_ in zip(df[study].map(norm), fam, cls):
        if not f_ or f_ == "nan" or not c_ or c_ == "nan":
            dec.append("undecided"); why.append("family / class not labelled")
        elif f_ not in R:
            dec.append("exclude"); why.append(f"family '{f_}' not in the method's scope")
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
