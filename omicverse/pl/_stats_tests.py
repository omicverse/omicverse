r"""Group comparisons and the brackets that report them.

``ov.pl.add_palue`` can draw a significance bracket, but the caller has to
supply the coordinates *and* run the test. That is where hand-annotated
figures go wrong: the bracket ends up over the wrong pair, or the P value is
uncorrected, or the test does not match what the plot shows.

:func:`compare_groups` runs the comparison and returns a tidy table;
:func:`add_stat_annotation` takes that table (or computes it) and draws the
brackets at heights that do not collide. Every categorical plot in ``ov.pl``
accepts ``test=`` and routes through here, so the annotation and the numbers
can never drift apart.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._registry import register_function
from ._stats_common import group_levels

__all__ = ["compare_groups", "format_pvalue", "add_stat_annotation"]


#: Tests that compare the *distributions* of a continuous variable.
_CONTINUOUS_TESTS = {
    "mannwhitney", "mannwhitneyu", "ttest", "welch", "ttest_ind",
    "wilcoxon", "ttest_rel", "kruskal", "anova", "brunnermunzel",
}
#: Tests on a contingency table of counts.
_CATEGORICAL_TESTS = {"fisher", "chi2"}

_STAR_CUTOFFS = ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*"))


@register_function(
    aliases=["P值格式化", "format_pvalue", "显著性星号", "p_value_stars", "星号标注"],
    category="pl",
    description=(
        "Render a P value as stars, as a number, or as both — the same formatting the significance brackets use"
    ),
    examples=[
        "ov.pl.format_pvalue(0.003)            # -> '**'",
        'ov.pl.format_pvalue(0.003, "value")   # -> \'P = 0.003\'',
        'ov.pl.format_pvalue(1e-8, "full")     # -> \'**** (P = 1.0e-08)\'',
    ],
    related=["pl.compare_groups", "pl.add_stat_annotation"],
)
def format_pvalue(p: float, style: str = "star") -> str:
    """Render a P value as stars, as a number, or as both.

    ``style`` is ``'star'`` (``***``), ``'value'`` (``P = 3.2e-05``),
    ``'full'`` (``*** (P = 3.2e-05)``) or ``'compact'`` (``3.2e-05``).
    """
    if not np.isfinite(p):
        return "n/a"
    stars = "ns"
    for cutoff, symbol in _STAR_CUTOFFS:
        if p < cutoff:
            stars = symbol
            break
    if style == "star":
        return stars
    if p < 1e-4:
        text = f"{p:.1e}"
    elif p < 0.001:
        text = f"{p:.2e}"
    else:
        text = f"{p:.3g}"
    if style == "compact":
        return text
    if style == "value":
        return f"P = {text}"
    if style == "full":
        return f"{stars} (P = {text})"
    raise ValueError(
        f"`style` must be 'star', 'value', 'full' or 'compact', got {style!r}."
    )


def _run_test(a: np.ndarray, b: np.ndarray, name: str) -> Tuple[float, float, str]:
    from scipy import stats

    if name in {"mannwhitney", "mannwhitneyu"}:
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue), "Mann-Whitney U"
    if name in {"welch"}:
        res = stats.ttest_ind(a, b, equal_var=False)
        return float(res.statistic), float(res.pvalue), "Welch's t-test"
    if name in {"ttest", "ttest_ind"}:
        res = stats.ttest_ind(a, b, equal_var=True)
        return float(res.statistic), float(res.pvalue), "Student's t-test"
    if name == "brunnermunzel":
        res = stats.brunnermunzel(a, b)
        return float(res.statistic), float(res.pvalue), "Brunner-Munzel"
    if name in {"wilcoxon", "ttest_rel"}:
        if len(a) != len(b):
            raise ValueError(
                f"A paired test needs equal group sizes, got {len(a)} and "
                f"{len(b)}. Reshape to one row per pair, or use an unpaired "
                f"test."
            )
        if name == "wilcoxon":
            res = stats.wilcoxon(a, b)
            return float(res.statistic), float(res.pvalue), "Wilcoxon signed-rank"
        res = stats.ttest_rel(a, b)
        return float(res.statistic), float(res.pvalue), "Paired t-test"
    raise ValueError(f"Unknown pairwise test {name!r}.")


def _omnibus(samples: List[np.ndarray], name: str) -> Tuple[float, float, str]:
    from scipy import stats

    if name == "kruskal":
        res = stats.kruskal(*samples)
        return float(res.statistic), float(res.pvalue), "Kruskal-Wallis"
    if name == "anova":
        res = stats.f_oneway(*samples)
        return float(res.statistic), float(res.pvalue), "One-way ANOVA"
    raise ValueError(f"Unknown omnibus test {name!r}.")


def _correct(pvalues: Sequence[float], method: Optional[str]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    if method in (None, "none") or values.size <= 1:
        return values
    aliases = {"holm": "holm", "bonferroni": "bonferroni", "bh": "fdr_bh",
               "fdr_bh": "fdr_bh", "by": "fdr_by", "fdr_by": "fdr_by"}
    if method not in aliases:
        raise ValueError(
            f"`correction` must be one of {sorted(aliases)} or None, got "
            f"{method!r}."
        )
    from statsmodels.stats.multitest import multipletests

    finite = np.isfinite(values)
    out = values.copy()
    if finite.any():
        out[finite] = multipletests(values[finite], method=aliases[method])[1]
    return out


@register_function(
    aliases=["组间比较", "compare_groups", "统计检验", "significance_test",
             "pairwise_test", "显著性检验"],
    category="pl",
    description=(
        "Run pairwise (and omnibus) group comparisons with automatic test "
        "selection and multiple-testing correction, returning a tidy table"
    ),
    examples=[
        "# Two groups -> Mann-Whitney by default",
        "stats = ov.pl.compare_groups(df, 'expression', 'condition')",
        "# More than two -> Kruskal-Wallis omnibus plus corrected pairwise",
        "stats = ov.pl.compare_groups(df, 'score', 'cell_type', correction='holm')",
        "# Only the comparisons you care about",
        "stats = ov.pl.compare_groups(df, 'score', 'arm',",
        "                             pairs=[('drug', 'vehicle')], test='welch')",
    ],
    related=["pl.add_stat_annotation", "pl.violinplot", "pl.barplot", "pl.stripplot"],
)
def compare_groups(data: Any = None,
                   value: Any = None,
                   group: Any = None,
                   *,
                   groups: Optional[Sequence[Any]] = None,
                   pairs: Optional[Sequence[Tuple[Any, Any]]] = None,
                   test: str = "auto",
                   correction: Optional[str] = "holm",
                   omnibus: bool = True,
                   min_size: int = 2) -> pd.DataFrame:
    r"""Compare a continuous variable across groups.

    Arguments
    ---------
    data
        ``DataFrame``, ``dict``, ``AnnData`` (uses ``.obs``), or ``None`` when
        ``value``/``group`` are arrays.
    value
        Column name or array of the measured quantity.
    group
        Column name or array of group labels.
    groups
        Explicit level order; also restricts the comparison to those levels.
    pairs
        Which comparisons to run, e.g. ``[('a', 'b'), ('a', 'c')]``. Defaults
        to every pair, which is why the correction matters.
    test
        ``'auto'`` (default) uses Mann-Whitney U — it makes no normality
        assumption and expression-like data is rarely normal. Others:
        ``'welch'``, ``'ttest'``, ``'brunnermunzel'``, and for paired designs
        ``'wilcoxon'`` / ``'ttest_rel'``.
    correction
        ``'holm'`` (default), ``'bonferroni'``, ``'bh'``, ``'by'``, or ``None``.
        Applied across the pairwise tests only.
    omnibus
        With three or more groups, also run a Kruskal-Wallis test over all of
        them and include it as a row named ``('*', '*')``.
    min_size
        Groups smaller than this are dropped, with a message.

    Returns
    -------
    ``DataFrame`` with one row per comparison: ``group1``, ``group2``, ``n1``,
    ``n2``, ``statistic``, ``pvalue``, ``pvalue_corrected``, ``test``.
    """
    from ._stats_common import resolve_columns

    if data is not None and value is not None and group is None:
        from ._stats_common import as_frame

        if as_frame(data) is None:
            data, value, group = None, data, value

    frame, _ = resolve_columns(data, value=value, group=group,
                               require=("value", "group"))
    values = frame["value"].to_numpy(dtype=float)
    levels = group_levels(frame["group"], groups)

    samples: Dict[Any, np.ndarray] = {}
    dropped = []
    for level in levels:
        sample = values[frame["group"].to_numpy() == level]
        if sample.size < min_size:
            dropped.append((level, sample.size))
            continue
        samples[level] = sample
    if dropped:
        detail = ", ".join(f"{lv} (n={n})" for lv, n in dropped)
        print(f"compare_groups: skipped groups smaller than {min_size}: {detail}")
    if len(samples) < 2:
        raise ValueError(
            f"Need at least two groups with >= {min_size} observations; found "
            f"{len(samples)}."
        )

    kept = [lv for lv in levels if lv in samples]
    if pairs is None:
        pairs = list(itertools.combinations(kept, 2))
    else:
        pairs = [tuple(p) for p in pairs]
        unknown = {lv for p in pairs for lv in p} - set(samples)
        if unknown:
            raise ValueError(
                f"`pairs` names groups that are absent or too small: "
                f"{sorted(map(str, unknown))}. Available: {list(map(str, kept))}."
            )

    pairwise = "mannwhitney" if test == "auto" else test
    records = []
    for left, right in pairs:
        statistic, pvalue, label = _run_test(samples[left], samples[right],
                                             pairwise)
        records.append({"group1": left, "group2": right,
                        "n1": samples[left].size, "n2": samples[right].size,
                        "statistic": statistic, "pvalue": pvalue,
                        "test": label})
    result = pd.DataFrame.from_records(records)
    result["pvalue_corrected"] = _correct(result["pvalue"], correction)
    result.attrs["correction"] = correction or "none"

    if omnibus and len(kept) > 2:
        name = "anova" if pairwise in {"ttest", "welch", "ttest_rel"} else "kruskal"
        statistic, pvalue, label = _omnibus([samples[lv] for lv in kept], name)
        overall = pd.DataFrame([{
            "group1": "*", "group2": "*", "n1": len(values), "n2": len(kept),
            "statistic": statistic, "pvalue": pvalue,
            "pvalue_corrected": pvalue, "test": label,
        }])
        result = pd.concat([overall, result], ignore_index=True)
    return result[["group1", "group2", "n1", "n2", "statistic", "pvalue",
                   "pvalue_corrected", "test"]]


def _bracket_levels(pairs: Sequence[Tuple[float, float]]) -> List[int]:
    """Assign each bracket the lowest tier where it does not overlap."""
    tiers: List[List[Tuple[float, float]]] = []
    out = []
    for left, right in pairs:
        lo, hi = min(left, right), max(left, right)
        for index, tier in enumerate(tiers):
            if all(hi < a or lo > b for a, b in tier):
                tier.append((lo, hi))
                out.append(index)
                break
        else:
            tiers.append([(lo, hi)])
            out.append(len(tiers) - 1)
    return out


@register_function(
    aliases=["显著性标注", "add_stat_annotation", "significance_brackets",
             "统计括号", "star_annotation"],
    category="pl",
    description=(
        "Draw significance brackets on a categorical plot, running the test "
        "and stacking the brackets so they never overlap"
    ),
    examples=[
        "ax = ov.pl.violinplot(df, 'cell_type', 'score')",
        "ov.pl.add_stat_annotation(ax, df, 'score', 'cell_type')",
        "# Only some comparisons, showing the P value rather than stars",
        "ov.pl.add_stat_annotation(ax, df, 'score', 'arm',",
        "                          pairs=[('drug', 'vehicle')], style='full')",
        "# Reuse a table you already computed",
        "res = ov.pl.compare_groups(df, 'score', 'arm')",
        "ov.pl.add_stat_annotation(ax, results=res, order=['vehicle', 'drug'])",
    ],
    related=["pl.compare_groups", "pl.violinplot", "pl.barplot", "pl.add_palue"],
)
def add_stat_annotation(ax,
                        data: Any = None,
                        value: Any = None,
                        group: Any = None,
                        *,
                        results: Optional[pd.DataFrame] = None,
                        order: Optional[Sequence[Any]] = None,
                        pairs: Optional[Sequence[Tuple[Any, Any]]] = None,
                        test: str = "auto",
                        correction: Optional[str] = "holm",
                        style: str = "star",
                        hide_ns: bool = False,
                        use_corrected: bool = True,
                        orient: str = "v",
                        line_offset: float = 0.06,
                        line_height: float = 0.05,
                        text_offset: float = 0.008,
                        linewidth: float = 1.0,
                        color: str = "black",
                        fontsize: float = 9,
                        return_stats: bool = False):
    r"""Annotate an existing categorical axes with significance brackets.

    Arguments
    ---------
    ax
        The axes to annotate. Category positions are read from its tick
        locations, so this works on any categorical plot — including the
        ``ov.pl`` ones and hand-drawn matplotlib.
    data, value, group
        Passed to :func:`compare_groups` when ``results`` is not given.
    results
        A table from :func:`compare_groups`, if the comparison is already done.
    order
        Category order along the axis. Defaults to the axis tick labels, which
        is what the plot actually drew.
    style
        ``'star'``, ``'value'``, ``'full'`` or ``'compact'`` — see
        :func:`format_pvalue`.
    hide_ns
        Skip brackets for non-significant comparisons instead of writing "ns".
    use_corrected
        Annotate the multiplicity-corrected P value. Leaving this on is the
        honest default when every pair is tested.
    orient
        ``'v'`` for categories on the x-axis (default), ``'h'`` for the y-axis.
    line_offset, line_height, text_offset
        Geometry in axis fractions: gap above the data before the first tier,
        height of each tier, and the gap between bracket and text.
    return_stats
        Also return the results table.

    Returns
    -------
    ``ax``, or ``(ax, results)`` when ``return_stats=True``.
    """
    if results is None:
        if value is None or group is None:
            raise TypeError(
                "Give either `results=` from compare_groups(), or "
                "`data`/`value`/`group` so the test can be run here."
            )
        results = compare_groups(data, value, group, pairs=pairs, test=test,
                                 correction=correction,
                                 groups=list(order) if order else None)

    if order is None:
        labels = [t.get_text() for t in
                  (ax.get_xticklabels() if orient == "v" else ax.get_yticklabels())]
        ticks = list(ax.get_xticks() if orient == "v" else ax.get_yticks())
        order = labels
        positions = {label: tick for label, tick in zip(labels, ticks) if label}
    else:
        positions = {str(level): float(i) for i, level in enumerate(order)}
    if not positions:
        raise ValueError(
            "Could not read category positions from the axes — pass `order=` "
            "with the categories in the order they were plotted."
        )

    column = "pvalue_corrected" if use_corrected else "pvalue"
    rows = results[(results["group1"] != "*")].copy()
    keep = []
    for _, row in rows.iterrows():
        left, right = str(row["group1"]), str(row["group2"])
        if left not in positions or right not in positions:
            continue
        if hide_ns and row[column] >= 0.05:
            continue
        keep.append((positions[left], positions[right], row[column]))
    if not keep:
        return (ax, results) if return_stats else ax

    tiers = _bracket_levels([(a, b) for a, b, _ in keep])
    lo, hi = (ax.get_ylim() if orient == "v" else ax.get_xlim())
    span = hi - lo
    top = max(
        (max(np.max(line.get_ydata() if orient == "v" else line.get_xdata())
             for line in ax.get_lines()) if ax.get_lines() else lo),
        lo,
    )
    base = max(top, lo + 0.5 * span)
    base = min(base, hi)

    transform = ax.transData
    reached = base
    for (left, right, pvalue), tier in zip(keep, tiers):
        level = base + span * (line_offset + tier * line_height)
        cap = span * line_height * 0.25
        text = format_pvalue(pvalue, style)
        if orient == "v":
            ax.plot([left, left, right, right],
                    [level - cap, level, level, level - cap],
                    linewidth=linewidth, color=color, clip_on=False,
                    transform=transform, solid_joinstyle="miter")
            ax.text((left + right) / 2, level + span * text_offset, text,
                    ha="center", va="bottom", fontsize=fontsize, color=color,
                    clip_on=False)
        else:
            ax.plot([level - cap, level, level, level - cap],
                    [left, left, right, right],
                    linewidth=linewidth, color=color, clip_on=False,
                    transform=transform, solid_joinstyle="miter")
            ax.text(level + span * text_offset, (left + right) / 2, text,
                    ha="left", va="center", fontsize=fontsize, color=color,
                    rotation=270, clip_on=False)
        reached = max(reached, level + span * (text_offset + line_height * 0.6))

    # make room for the brackets rather than letting them run off the axes
    if orient == "v":
        ax.set_ylim(lo, max(hi, reached))
    else:
        ax.set_xlim(lo, max(hi, reached))
    return (ax, results) if return_stats else ax
