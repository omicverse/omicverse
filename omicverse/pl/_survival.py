r"""Time-to-event plots: Kaplan-Meier curves and competing-risk incidence.

Both estimators are implemented here directly on top of NumPy/SciPy rather
than pulled from ``lifelines``, so survival plotting works in a bare
``omicverse`` install. The implementations follow the standard references and
are checked against ``lifelines`` (Kaplan-Meier, log-rank, Aalen-Johansen) and
R's ``cmprsk`` (Gray's test) in ``tests/pl/test_stats.py``.

Public entry points
-------------------
``survival``             Kaplan-Meier curves + confidence band + numbers at
                         risk + log-rank test + hazard ratio
``cumulative_incidence`` Aalen-Johansen cumulative incidence under competing
                         risks + Gray's test

Both take a ``DataFrame`` (or ``AnnData``, or bare arrays) — no ``AnnData`` is
required anywhere.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._registry import register_function
from ._plot_backend import style_axes
from ._stats_common import (as_frame, default_palette, font_size,
                            group_levels, resolve_columns)

__all__ = [
    "kaplan_meier",
    "logrank_test",
    "aalen_johansen",
    "grays_test",
    "survival",
    "cumulative_incidence",
]


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------


def _event_table(time: np.ndarray, indicator: np.ndarray) -> Dict[str, np.ndarray]:
    """Risk set / event / censor counts at each distinct observed time."""
    t = np.asarray(time, dtype=float)
    times = np.unique(t)
    n_risk = np.array([(t >= ti).sum() for ti in times], dtype=float)
    n_event = np.array([((t == ti) & indicator).sum() for ti in times], dtype=float)
    n_censor = np.array([((t == ti) & ~indicator).sum() for ti in times], dtype=float)
    return {"times": times, "n_risk": n_risk, "n_event": n_event,
            "n_censor": n_censor}


@register_function(
    aliases=["KM估计", "kaplan_meier", "生存估计", "product_limit", "乘积极限估计"],
    category="pl",
    description=(
        "Kaplan-Meier product-limit estimator with a log-log confidence band, median and its Brookmeyer-Crowley interval — numbers only, no plot"
    ),
    examples=[
        'fit = ov.pl.kaplan_meier(df["months"], df["dead"])',
        'print(fit["median"], fit["median_ci"])',
        '# the curve itself, for a custom figure',
        'ax.step(fit["timeline"], fit["survival"], where="post")',
    ],
    related=["pl.survival", "pl.logrank_test", "pl.aalen_johansen"],
)
def kaplan_meier(time: Sequence[float],
                 event: Sequence[Any],
                 *,
                 alpha: float = 0.05) -> Dict[str, np.ndarray]:
    r"""Kaplan-Meier product-limit estimator with a log-log confidence band.

    Arguments
    ---------
    time
        Follow-up time for each subject.
    event
        1 / ``True`` if the event was observed, 0 / ``False`` if censored.
    alpha
        Two-sided significance for the confidence band (0.05 -> 95%).

    Returns
    -------
    dict with ``timeline`` (starting at 0), ``survival``, ``lower``,
    ``upper``, ``std_error``, the per-time ``n_risk`` / ``n_event`` /
    ``n_censor`` counts, ``median`` and ``median_ci``.

    Notes
    -----
    The band uses the log-log (``log(-log S)``) transform with Greenwood's
    variance, which keeps the interval inside :math:`[0, 1]` — unlike the
    plain linear band, which routinely runs past 1 in the early follow-up
    where survival is high.
    """
    from scipy.stats import norm

    t = np.asarray(time, dtype=float)
    e = np.asarray(event)
    if e.dtype == bool:
        ind = e
    else:
        ind = np.asarray(e, dtype=float) > 0
    if t.size == 0:
        raise ValueError("No observations passed to kaplan_meier().")
    if np.any(t < 0):
        raise ValueError("Negative follow-up times are not allowed.")

    table = _event_table(t, ind)
    times, n_risk = table["times"], table["n_risk"]
    n_event, n_censor = table["n_event"], table["n_censor"]

    with np.errstate(divide="ignore", invalid="ignore"):
        factors = 1.0 - n_event / n_risk
        surv = np.cumprod(factors)
        # Greenwood: Var(S) = S^2 * sum d / (n (n - d))
        denom = n_risk * (n_risk - n_event)
        increments = np.where(denom > 0, n_event / denom, np.nan)
        cum = np.nancumsum(np.where(n_event > 0, increments, 0.0))
        # once the risk set is exhausted by events the variance is undefined
        broken = np.cumsum((n_event > 0) & (denom <= 0)) > 0
        cum = np.where(broken, np.nan, cum)
        var = surv ** 2 * cum

    z = norm.ppf(1.0 - alpha / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # se of log(-log S)
        se_ll = np.sqrt(cum) / np.abs(np.log(surv))
        expo = np.exp(np.where(surv > 0, z * se_ll, np.nan))
        lower = surv ** expo
        upper = surv ** (1.0 / expo)
    lower = np.where(surv >= 1.0, 1.0, lower)
    upper = np.where(surv >= 1.0, 1.0, upper)
    lower = np.where(surv <= 0.0, 0.0, lower)
    upper = np.where(surv <= 0.0, 0.0, upper)

    timeline = np.concatenate([[0.0], times])
    surv_full = np.concatenate([[1.0], surv])
    lower_full = np.concatenate([[1.0], lower])
    upper_full = np.concatenate([[1.0], upper])

    median = np.nan
    below = np.flatnonzero(surv <= 0.5)
    if below.size:
        median = float(times[below[0]])
    # Brookmeyer-Crowley: the interval is the set of times whose confidence
    # band still covers 0.5. The lower band drops through 0.5 first, so it
    # gives the *earlier* bound; the upper band gives the later one.
    med_lo, med_hi = np.nan, np.nan
    hit = np.flatnonzero(lower <= 0.5)
    if hit.size:
        med_lo = float(times[hit[0]])
    hit = np.flatnonzero(upper <= 0.5)
    if hit.size:
        med_hi = float(times[hit[0]])

    return {
        "timeline": timeline,
        "survival": surv_full,
        "lower": lower_full,
        "upper": upper_full,
        "std_error": np.concatenate([[0.0], np.sqrt(var)]),
        "event_times": times,
        "n_risk": n_risk,
        "n_event": n_event,
        "n_censor": n_censor,
        "median": median,
        "median_ci": (med_lo, med_hi),
        "n": int(t.size),
        "n_events": int(ind.sum()),
    }


def _at_risk(time: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.array([(time >= q).sum() for q in query], dtype=int)


@register_function(
    aliases=["log-rank检验", "logrank_test", "生存差异检验", "mantel_cox", "风险比"],
    category="pl",
    description=(
        "Multivariate log-rank (Mantel-Cox) test with the Mantel-Haenszel hazard ratio for two groups; `groups=` fixes which level is the reference"
    ),
    examples=[
        'res = ov.pl.logrank_test(df["months"], df["dead"], df["arm"])',
        'print(res["pvalue"], res["hazard_ratio"])',
        '# choose the reference explicitly',
        'res = ov.pl.logrank_test(t, e, g, groups=["advanced", "early"])',
    ],
    related=["pl.survival", "pl.kaplan_meier", "pl.grays_test"],
)
def logrank_test(time: Sequence[float],
                 event: Sequence[Any],
                 group: Sequence[Any],
                 *,
                 groups: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    r"""Multivariate log-rank (Mantel-Cox) test across two or more groups.

    Arguments
    ---------
    time, event
        Follow-up time and event indicator.
    group
        Group label per subject.
    groups
        Explicit level order. The **first** level is the numerator of the
        hazard ratio, so pass this when the direction matters —
        ``groups=['treated', 'control']`` reports treated vs control.
        Without it the order comes from the ``Categorical`` categories if
        there are any, and otherwise from sorting the observed labels.

    Returns
    -------
    dict with ``statistic`` (chi-square), ``pvalue``, ``df``, ``groups``,
    ``observed``, ``expected``, and — for exactly two groups — the log-rank
    ``hazard_ratio`` with its confidence interval.

    Notes
    -----
    The hazard ratio reported is the Mantel-Haenszel / Peto estimate
    :math:`(O_1/E_1)/(O_2/E_2)` with
    :math:`\mathrm{SE}(\log \mathrm{HR}) = \sqrt{1/E_1 + 1/E_2}`. It agrees
    closely with a Cox model for a single binary covariate but is not
    identical, and it degrades when hazards are strongly non-proportional —
    exactly the situation where the log-rank test is itself questionable.
    """
    from scipy.stats import chi2 as chi2_dist
    from scipy.stats import norm

    t = np.asarray(time, dtype=float)
    e = np.asarray(event)
    ind = e if e.dtype == bool else np.asarray(e, dtype=float) > 0
    levels = group_levels(group, groups)
    g = pd.Series(group).to_numpy()
    k = len(levels)
    if k < 2:
        raise ValueError("The log-rank test needs at least two groups.")
    groups = levels

    event_times = np.unique(t[ind])
    observed = np.zeros(k)
    expected = np.zeros(k)
    V = np.zeros((k, k))

    masks = [g == grp for grp in groups]
    for ti in event_times:
        n_i = float((t >= ti).sum())
        d_i = float(((t == ti) & ind).sum())
        if n_i <= 1 or d_i == 0:
            # still accumulate O and E, but the variance term is 0
            pass
        n_ij = np.array([float((t[m] >= ti).sum()) for m in masks])
        d_ij = np.array([float(((t[m] == ti) & ind[m]).sum()) for m in masks])
        observed += d_ij
        expected += n_ij * d_i / n_i
        if n_i > 1:
            factor = d_i * (n_i - d_i) / (n_i - 1) / (n_i ** 2)
            V += factor * (np.diag(n_ij * n_i) - np.outer(n_ij, n_ij))

    diff = (observed - expected)[: k - 1]
    Vk = V[: k - 1, : k - 1]
    statistic = float(diff @ np.linalg.pinv(Vk) @ diff)
    df = k - 1
    pvalue = float(chi2_dist.sf(statistic, df))

    out: Dict[str, Any] = {
        "statistic": statistic,
        "pvalue": pvalue,
        "df": df,
        "groups": groups,
        "observed": observed,
        "expected": expected,
        "test": "log-rank (Mantel-Cox)",
    }
    if k == 2 and np.all(expected > 0):
        hr = float((observed[0] / expected[0]) / (observed[1] / expected[1]))
        se = float(np.sqrt(1.0 / expected[0] + 1.0 / expected[1]))
        z = norm.ppf(0.975)
        out["hazard_ratio"] = hr
        out["hazard_ratio_ci"] = (float(hr * np.exp(-z * se)),
                                  float(hr * np.exp(z * se)))
        out["hazard_ratio_reference"] = groups[1]
    return out


@register_function(
    aliases=["AJ估计", "aalen_johansen", "累积发病率估计", "竞争风险估计", "CIF"],
    category="pl",
    description=(
        "Aalen-Johansen cumulative incidence for one cause under competing risks, with the delta-method variance — the correct alternative to 1 - KM"
    ),
    examples=[
        'fit = ov.pl.aalen_johansen(df["months"], df["cause"], cause=1)',
        'print(fit["cif"][-1])',
        '# how wrong the naive estimator would have been',
        'km = ov.pl.kaplan_meier(df["months"], df["cause"] == 1)',
    ],
    related=["pl.cumulative_incidence", "pl.grays_test", "pl.kaplan_meier"],
)
def aalen_johansen(time: Sequence[float],
                   event: Sequence[Any],
                   cause: Any = 1,
                   *,
                   alpha: float = 0.05) -> Dict[str, np.ndarray]:
    r"""Aalen-Johansen cumulative incidence for one cause under competing risks.

    Arguments
    ---------
    time
        Follow-up time.
    event
        Cause code per subject: ``0`` (or ``False``) means censored, any other
        value identifies which event happened.
    cause
        The cause whose incidence is estimated. Every other non-zero code is a
        competing event.
    alpha
        Two-sided level of the pointwise confidence band.

    Returns
    -------
    dict with ``timeline``, ``cif``, ``lower``, ``upper``, ``variance``,
    ``n_risk`` and the overall (all-cause) ``survival``.

    Notes
    -----
    Treating competing events as censoring and running ``1 - KM`` **over-states**
    the incidence, sometimes badly, because it implicitly assumes the subjects
    removed by a competing event could still have had the event of interest.
    The Aalen-Johansen estimator weights each cause-specific increment by the
    overall survival just before that time, which is the quantity that actually
    adds up to the observed proportion of failures.

    The variance follows the standard delta-method expression (Aalen 1978; see
    Marubini & Valsecchi §10), and the band is built on the log-log scale.
    """
    from scipy.stats import norm

    t = np.asarray(time, dtype=float)
    codes = np.asarray(event)
    if codes.dtype == bool:
        codes = codes.astype(int)
    is_event = codes != 0
    is_cause = codes == cause

    times = np.unique(t[is_event])
    if times.size == 0:
        raise ValueError("No events of any cause were observed.")

    n_risk = np.array([(t >= ti).sum() for ti in times], dtype=float)
    d_any = np.array([((t == ti) & is_event).sum() for ti in times], dtype=float)
    d_k = np.array([((t == ti) & is_cause).sum() for ti in times], dtype=float)

    # overall (all-cause) Kaplan-Meier, shifted to give S(t-)
    surv = np.cumprod(1.0 - d_any / n_risk)
    surv_prev = np.concatenate([[1.0], surv[:-1]])

    increments = surv_prev * d_k / n_risk
    cif = np.cumsum(increments)

    # Aalen's delta-method variance
    m = times.size
    var = np.zeros(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(n_risk > d_any, d_any / (n_risk * (n_risk - d_any)), np.nan)
        b = surv_prev ** 2 * ((n_risk - d_k) / n_risk) * d_k / (n_risk ** 2)
        c = surv_prev * d_k / (n_risk ** 2)
    for i in range(m):
        d = cif[i] - cif[: i + 1]
        term1 = np.nansum(d ** 2 * a[: i + 1])
        term2 = np.nansum(b[: i + 1])
        term3 = 2.0 * np.nansum(d * c[: i + 1])
        var[i] = term1 + term2 - term3
    var = np.clip(var, 0.0, None)

    z = norm.ppf(1.0 - alpha / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # log-log band on the CIF, mirroring cmprsk's default
        se_ll = np.sqrt(var) / (cif * np.abs(np.log(cif)))
        expo = np.exp(z * se_ll)
        lower = cif ** expo
        upper = cif ** (1.0 / expo)
    lower = np.where(cif <= 0, 0.0, np.nan_to_num(lower, nan=0.0))
    upper = np.where(cif <= 0, 0.0, np.clip(np.nan_to_num(upper, nan=0.0), 0.0, 1.0))

    return {
        "timeline": np.concatenate([[0.0], times]),
        "cif": np.concatenate([[0.0], cif]),
        "lower": np.concatenate([[0.0], lower]),
        "upper": np.concatenate([[0.0], upper]),
        "variance": np.concatenate([[0.0], var]),
        "survival": np.concatenate([[1.0], surv]),
        "event_times": times,
        "n_risk": n_risk,
        "n_event": d_k,
        "n": int(t.size),
        "n_events": int(is_cause.sum()),
    }


def _left_limit(times: np.ndarray, values: np.ndarray, query: float,
                initial: float) -> float:
    """Value of a right-continuous step function just *before* ``query``."""
    index = int(np.searchsorted(times, query, side="left")) - 1
    return float(values[index]) if index >= 0 else float(initial)


def _subdistribution_curves(t: np.ndarray, codes: np.ndarray, cause: Any):
    """Per-group overall survival and cause-specific CIF, for Gray's weights."""
    times = np.unique(t)
    n_risk = np.array([(t >= x).sum() for x in times], dtype=float)
    d_any = np.array([((t == x) & (codes != 0)).sum() for x in times], dtype=float)
    d_k = np.array([((t == x) & (codes == cause)).sum() for x in times], dtype=float)
    surv = np.cumprod(1.0 - d_any / n_risk)
    surv_prev = np.concatenate([[1.0], surv[:-1]])
    cif = np.cumsum(surv_prev * d_k / n_risk)
    return times, surv, cif


@register_function(
    aliases=["Gray检验", "grays_test", "gray_test", "竞争风险检验", "次分布检验"],
    category="pl",
    description=(
        "Gray's test for equality of cumulative incidence functions — the test that matches what a competing-risk plot shows, unlike a plain log-rank"
    ),
    examples=[
        'res = ov.pl.grays_test(df["months"], df["cause"], df["arm"], cause=1)',
        'print(res["pvalue"])',
        '# contrast with the cause-specific log-rank, a different question',
        'ov.pl.logrank_test(df["months"], df["cause"] == 1, df["arm"])',
    ],
    related=["pl.cumulative_incidence", "pl.aalen_johansen", "pl.logrank_test"],
)
def grays_test(time: Sequence[float],
               event: Sequence[Any],
               group: Sequence[Any],
               cause: Any = 1,
               *,
               groups: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    r"""Gray's test for equality of cumulative incidence functions.

    Compares the *subdistribution* hazard of ``cause`` across groups, which is
    the test that matches what a cumulative-incidence plot shows. A plain
    log-rank on the same data answers a different question (the cause-specific
    hazard) and can point the other way when the competing risk differs between
    groups.

    Returns a dict with ``statistic`` (chi-square), ``pvalue``, ``df`` and
    ``groups``.

    Notes
    -----
    Implemented as a log-rank statistic over Gray's **subdistribution risk
    set**: within group :math:`j`,

    .. math::
        R_j(t) = Y_j(t)\,\frac{1 - \hat F_{kj}(t^-)}{\hat S_j(t^-)}

    where :math:`Y_j` is the ordinary number at risk, :math:`\hat F_{kj}` the
    Aalen-Johansen incidence of the cause of interest and :math:`\hat S_j` the
    all-cause survival. The inflation factor is exactly the subjects who left
    through a competing event and must stay in the denominator.

    The variance uses the log-rank (hypergeometric) estimator rather than
    Gray's full martingale variance, which also propagates the uncertainty in
    :math:`\hat F` and :math:`\hat S`. The consequence is that the chi-square
    lands within a few percent of ``cmprsk::cuminc`` rather than matching it
    digit for digit — the test suite pins that agreement to 5%. Treat a P
    value hovering at the significance threshold accordingly.
    """
    from scipy.stats import chi2 as chi2_dist

    t = np.asarray(time, dtype=float)
    codes = np.asarray(event)
    if codes.dtype == bool:
        codes = codes.astype(int)
    is_cause = codes == cause

    levels = group_levels(group, groups)
    g = pd.Series(group).to_numpy()
    k = len(levels)
    if k < 2:
        raise ValueError("Gray's test needs at least two groups.")
    groups = levels
    masks = [g == grp for grp in groups]

    event_times = np.unique(t[is_cause])
    if event_times.size == 0:
        raise ValueError(f"No events of cause {cause!r} were observed.")

    curves = [_subdistribution_curves(t[m], codes[m], cause) for m in masks]

    observed = np.zeros(k)
    expected = np.zeros(k)
    V = np.zeros((k, k))

    for ti in event_times:
        risk = np.zeros(k)
        events = np.zeros(k)
        for j, mask in enumerate(masks):
            times_j, surv_j, cif_j = curves[j]
            at_risk = float((t[mask] >= ti).sum())
            surv_prev = _left_limit(times_j, surv_j, ti, 1.0)
            cif_prev = _left_limit(times_j, cif_j, ti, 0.0)
            risk[j] = at_risk * (1.0 - cif_prev) / surv_prev if surv_prev > 0 else 0.0
            events[j] = float(((t[mask] == ti) & is_cause[mask]).sum())

        total_risk = risk.sum()
        total_events = events.sum()
        if total_risk <= 0:
            continue
        observed += events
        expected += risk * total_events / total_risk
        if total_risk > 1:
            factor = (total_events * (total_risk - total_events)
                      / (total_risk - 1) / (total_risk ** 2))
            V += factor * (np.diag(risk * total_risk) - np.outer(risk, risk))

    diff = (observed - expected)[: k - 1]
    Vk = V[: k - 1, : k - 1]
    statistic = float(diff @ np.linalg.pinv(Vk) @ diff)
    df = k - 1
    return {
        "statistic": statistic,
        "pvalue": float(chi2_dist.sf(statistic, df)),
        "df": df,
        "groups": groups,
        "observed": observed,
        "expected": expected,
        "test": "Gray's test (subdistribution)",
    }


# --------------------------------------------------------------------------
# plotting helpers
# --------------------------------------------------------------------------


#: kept as a private alias — several modules import it from here
_default_palette = default_palette


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "P = n/a"
    if p < 1e-4:
        return "P < 0.0001"
    if p < 0.001:
        return f"P = {p:.1e}"
    return f"P = {p:.3g}"


def _make_axes(ax, figsize, risk_table: bool, n_groups: int):
    """Return ``(fig, ax, ax_risk)``, creating a risk-table axes if asked."""
    if ax is None:
        fig = plt.figure(figsize=figsize)
        if risk_table:
            ratio = 0.10 + 0.075 * n_groups
            gs = fig.add_gridspec(2, 1, height_ratios=[1.0, ratio], hspace=0.10)
            main = fig.add_subplot(gs[0])
            table = fig.add_subplot(gs[1], sharex=main)
        else:
            main, table = fig.add_subplot(111), None
        return fig, main, table
    fig = ax.get_figure()
    table = None
    if risk_table:
        height = 0.08 + 0.065 * n_groups
        table = ax.inset_axes([0.0, -0.16 - height, 1.0, height])
        table.sharex(ax)
    return fig, ax, table


def _draw_risk_table(ax_risk, times, labels, counts, colors, fontsize,
                     xlim, xlabel=None):
    """Numbers-at-risk table below the curves.

    The main axes hands its x tick labels over to this one, so the time axis
    is written once, underneath the table — the layout every journal uses.
    """
    n = len(labels)
    span = (xlim[1] - xlim[0]) or 1.0
    ax_risk.set_ylim(-0.5, n - 0.5)
    ax_risk.invert_yaxis()
    ax_risk.set_yticks(range(n))
    ax_risk.set_yticklabels(labels, fontsize=font_size(fontsize))
    for tick, color in zip(ax_risk.get_yticklabels(), colors):
        tick.set_color(color)
    ax_risk.set_xticks(times)
    ax_risk.set_xticklabels([f"{v:g}" for v in times], fontsize=font_size(fontsize) + 0.5)
    ax_risk.tick_params(axis="x", length=0, labelbottom=True)
    ax_risk.tick_params(axis="y", length=0)
    for spine in ax_risk.spines.values():
        spine.set_visible(False)
    for row, row_counts in enumerate(counts):
        for x, value in zip(times, row_counts):
            # a count centred on the very edge would hang outside the axes
            rel = (x - xlim[0]) / span
            ha = "left" if rel < 0.02 else ("right" if rel > 0.98 else "center")
            ax_risk.text(x, row, f"{int(value)}", ha=ha, va="center",
                         fontsize=font_size(fontsize))
    ax_risk.set_ylabel("")
    if xlabel:
        ax_risk.set_xlabel(xlabel, fontsize=font_size(fontsize, "label", 0.5))
    ax_risk.set_title("Number at risk", fontsize=font_size(fontsize), loc="left", pad=3)


def _step_ci(ax, x, lo, hi, color, alpha):
    ax.fill_between(x, lo, hi, step="post", color=color, alpha=alpha,
                    linewidth=0)


# --------------------------------------------------------------------------
# public plots
# --------------------------------------------------------------------------


@register_function(
    aliases=["生存曲线", "survival", "KM曲线", "km_plot", "生存分析图", "生存曲线图"],
    category="pl",
    description=(
        "Kaplan-Meier survival curves with confidence band, censoring marks, "
        "numbers-at-risk table, log-rank test and hazard ratio"
    ),
    examples=[
        "# From a clinical table",
        "ax = ov.pl.survival(clinical, time='OS_months', event='OS_status',",
        "                    group='risk_group')",
        "# Bare arrays, no DataFrame needed",
        "ax = ov.pl.survival(time=t, event=e, group=labels)",
        "# From AnnData .obs, and keep the statistics",
        "ax, stats = ov.pl.survival(adata, time='days', event='dead',",
        "                           group='cluster', return_stats=True)",
        "print(stats['pvalue'], stats['hazard_ratio'])",
    ],
    related=["pl.cumulative_incidence", "pl.forest", "pl.savefig"],
)
def survival(data: Any = None,
             time: Any = None,
             event: Any = None,
             group: Any = None,
             *,
             groups: Optional[Sequence[Any]] = None,
             ax=None,
             figsize: Tuple[float, float] = (4.2, 4.2),
             palette=None,
             ci: bool = True,
             ci_alpha: float = 0.05,
             ci_opacity: float = 0.15,
             censor_marks: bool = True,
             risk_table: bool = True,
             risk_table_times: Optional[Sequence[float]] = None,
             test: bool = True,
             hazard_ratio: bool = True,
             median_line: bool = False,
             percent: bool = False,
             xlabel: Optional[str] = None,
             ylabel: Optional[str] = None,
             title: Optional[str] = None,
             xlim: Optional[Tuple[float, float]] = None,
             ylim: Optional[Tuple[float, float]] = None,
             legend: bool = True,
             legend_loc: str = "best",
             fontsize: Optional[float] = None,
             linewidth: float = 1.6,
             return_stats: bool = False):
    r"""Plot Kaplan-Meier survival curves.

    Arguments
    ---------
    data
        ``DataFrame``, ``dict``, or ``AnnData`` (its ``.obs`` is used). May be
        ``None`` when ``time``/``event``/``group`` are given as arrays.
    time
        Column name or array of follow-up times.
    event
        Column name or array; ``1``/``True`` = event observed, ``0``/``False``
        = censored.
    group
        Optional column name or array splitting the cohort into strata. With
        no ``group`` a single curve is drawn and no test is run.
    groups
        Explicit order of the strata. The first one is the **reference** of
        the hazard ratio, so ``groups=['advanced', 'early']`` reports advanced
        vs early. Left out, the order is the ``Categorical`` category order if
        the column has one, and otherwise the sorted labels — never the order
        rows happen to appear in the file.
    ci
        Draw the pointwise log-log confidence band at level ``1 - ci_alpha``.
    censor_marks
        Mark censoring times with a vertical tick on the curve — without them
        a flat tail is indistinguishable from "everyone still at risk".
    risk_table
        Draw the numbers-at-risk table underneath. Reviewers ask for it; it is
        also the only way to see that a late plateau rests on three patients.
    risk_table_times
        Times to tabulate. Defaults to the x-axis ticks.
    test
        Annotate the log-rank P value (needs ``group`` with >= 2 levels).
    hazard_ratio
        With exactly two groups, also annotate the log-rank hazard ratio and
        its 95% CI.
    median_line
        Drop guide lines from ``S = 0.5`` to each group's median survival.
    percent
        Label the y-axis 0-100% instead of 0-1.
    return_stats
        Return ``(ax, stats)`` where ``stats`` holds the per-group fits and
        the test result.

    Returns
    -------
    The main ``Axes`` (or ``(Axes, dict)`` when ``return_stats=True``).
    """
    if data is not None and as_frame(data) is None:
        # survival(time_array, event_array, group_array) without a table
        data, time, event, group = None, data, time, event
    frame, names = resolve_columns(data, time=time, event=event, group=group,
                                   require=("time", "event"))
    dropped = frame.attrs.get("n_dropped", 0)
    if dropped:
        print(f"survival: dropped {dropped} rows with missing values.")
    t = frame["time"].to_numpy(dtype=float)
    e = frame["event"].to_numpy()

    if "group" in frame:
        levels = group_levels(frame["group"], groups)
        g = frame["group"].to_numpy()
    else:
        g = np.zeros(len(t), dtype=int)
        levels = [0]

    colors = _default_palette(len(levels), palette)
    fig, ax, ax_risk = _make_axes(ax, figsize, risk_table, len(levels))

    fits: Dict[Any, Dict[str, np.ndarray]] = {}
    for level, color in zip(levels, colors):
        mask = g == level
        fit = kaplan_meier(t[mask], e[mask], alpha=ci_alpha)
        fits[level] = fit
        label = str(level) if "group" in frame else (names.get("time", "cohort"))
        if "group" not in frame:
            label = "all"
        ax.step(fit["timeline"], fit["survival"], where="post", color=color,
                linewidth=linewidth, label=f"{label} (n={fit['n']})")
        if ci:
            _step_ci(ax, fit["timeline"], fit["lower"], fit["upper"], color,
                     ci_opacity)
        if censor_marks:
            cens_times = fit["event_times"][fit["n_censor"] > 0]
            if cens_times.size:
                idx = np.searchsorted(fit["timeline"], cens_times, side="right") - 1
                ax.plot(cens_times, fit["survival"][idx], linestyle="none",
                        marker="|", markersize=6, markeredgewidth=1.2,
                        color=color)
        if median_line and np.isfinite(fit["median"]):
            ax.plot([0, fit["median"], fit["median"]], [0.5, 0.5, 0.0],
                    color=color, linewidth=0.8, linestyle=":", zorder=0)

    stats: Dict[str, Any] = {"fits": fits}
    annotation: List[str] = []
    if "group" in frame and len(levels) >= 2 and (test or hazard_ratio):
        result = logrank_test(t, e, g, groups=levels)
        stats.update(result)
        if test:
            annotation.append(f"Log-rank {_format_p(result['pvalue'])}")
        if hazard_ratio and "hazard_ratio" in result:
            lo, hi = result["hazard_ratio_ci"]
            annotation.append(
                f"HR = {result['hazard_ratio']:.2f} ({lo:.2f}-{hi:.2f})"
            )
            annotation.append(
                f"{levels[0]} vs {result['hazard_ratio_reference']}"
            )
    if annotation:
        ax.text(0.98, 0.98, "\n".join(annotation), transform=ax.transAxes,
                ha="right", va="top", fontsize=font_size(fontsize),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.8", alpha=0.85))

    ax.set_xlim(xlim if xlim is not None else (0, float(np.max(t)) * 1.02))
    ax.set_ylim(ylim if ylim is not None else (0.0, 1.02))
    if percent:
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([f"{v:.0f}" for v in np.linspace(0, 100, 6)])
    time_label = xlabel if xlabel is not None else names.get("time", "Time")
    if ax_risk is None:
        ax.set_xlabel(time_label, fontsize=font_size(fontsize, "label"))
    else:
        # the risk table below carries the time axis instead
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel(
        ylabel if ylabel is not None
        else ("Survival probability (%)" if percent else "Survival probability"),
        fontsize=font_size(fontsize, "label"),
    )
    if title:
        ax.set_title(title, fontsize=font_size(fontsize, "title"))
    ax.tick_params(labelsize=font_size(fontsize))
    style_axes(ax)
    if legend and "group" in frame:
        ax.legend(loc=legend_loc, frameon=False, fontsize=font_size(fontsize))

    if ax_risk is not None:
        ticks = (np.asarray(risk_table_times, dtype=float)
                 if risk_table_times is not None
                 else np.asarray([v for v in ax.get_xticks()
                                  if ax.get_xlim()[0] <= v <= ax.get_xlim()[1]]))
        counts = [_at_risk(t[g == level], ticks) for level in levels]
        _draw_risk_table(ax_risk, ticks, [str(lv) for lv in levels], counts,
                         colors, font_size(fontsize, "tick", -0.5), ax.get_xlim(), time_label)
        ax_risk.set_xlim(ax.get_xlim())

    return (ax, stats) if return_stats else ax


@register_function(
    aliases=["累积发病率", "cumulative_incidence", "竞争风险", "competing_risks",
             "累积发生率曲线"],
    category="pl",
    description=(
        "Aalen-Johansen cumulative incidence curves under competing risks, "
        "with Gray's test — the correct alternative to 1 - Kaplan-Meier"
    ),
    examples=[
        "# 0 = censored, 1 = relapse, 2 = death without relapse",
        "ax = ov.pl.cumulative_incidence(clinical, time='months', event='status',",
        "                                group='arm', cause=1)",
        "# Show every cause at once for one cohort",
        "ax = ov.pl.cumulative_incidence(clinical, time='months', event='status',",
        "                                cause=[1, 2])",
        "# Compare against the naive 1 - KM curve",
        "ax = ov.pl.cumulative_incidence(clinical, time='months', event='status',",
        "                                cause=1, show_naive_km=True)",
    ],
    related=["pl.survival", "pl.forest"],
)
def cumulative_incidence(data: Any = None,
                         time: Any = None,
                         event: Any = None,
                         group: Any = None,
                         *,
                         groups: Optional[Sequence[Any]] = None,
                         cause: Union[Any, Sequence[Any]] = 1,
                         cause_labels: Optional[Mapping[Any, str]] = None,
                         ax=None,
                         figsize: Tuple[float, float] = (4.2, 4.0),
                         palette=None,
                         ci: bool = False,
                         ci_alpha: float = 0.05,
                         ci_opacity: float = 0.15,
                         test: bool = True,
                         show_naive_km: bool = False,
                         risk_table: bool = False,
                         risk_table_times: Optional[Sequence[float]] = None,
                         percent: bool = False,
                         xlabel: Optional[str] = None,
                         ylabel: Optional[str] = None,
                         title: Optional[str] = None,
                         xlim: Optional[Tuple[float, float]] = None,
                         ylim: Optional[Tuple[float, float]] = None,
                         legend: bool = True,
                         legend_loc: str = "best",
                         fontsize: Optional[float] = None,
                         linewidth: float = 1.6,
                         return_stats: bool = False):
    r"""Plot cumulative incidence functions under competing risks.

    Arguments
    ---------
    data, time, group, groups
        As in :func:`survival`.
    event
        Cause code per subject: ``0`` = censored, other values identify which
        event occurred.
    cause
        Which cause to plot. Pass a list to overlay several causes; combining
        a multi-cause list with ``group`` is refused, because the resulting
        curve set is unreadable.
    cause_labels
        Display names for the cause codes.
    ci
        Draw the pointwise confidence band (off by default — with several
        groups the bands overlap into mud).
    test
        Annotate Gray's test P value when ``group`` has >= 2 levels.
    show_naive_km
        Also draw the ``1 - KM`` curve that treats competing events as
        censoring, as a dashed line. It is always at or above the correct
        curve; showing both makes the size of the bias visible.
    return_stats
        Return ``(ax, stats)``.

    Returns
    -------
    The ``Axes`` (or ``(Axes, dict)`` when ``return_stats=True``).
    """
    if data is not None and as_frame(data) is None:
        data, time, event, group = None, data, time, event
    frame, names = resolve_columns(data, time=time, event=event, group=group,
                                   require=("time", "event"))
    dropped = frame.attrs.get("n_dropped", 0)
    if dropped:
        print(f"cumulative_incidence: dropped {dropped} rows with missing values.")
    t = frame["time"].to_numpy(dtype=float)
    e = frame["event"].to_numpy()

    causes = list(cause) if isinstance(cause, (list, tuple, np.ndarray)) else [cause]
    has_group = "group" in frame
    if has_group and len(causes) > 1:
        raise ValueError(
            "Plot either several causes for one cohort, or one cause across "
            "groups — not both. Call this once per cause instead."
        )

    if has_group:
        levels = group_levels(frame["group"], groups)
        g = frame["group"].to_numpy()
        series = [(lv, g == lv, causes[0], str(lv)) for lv in levels]
    else:
        g = None
        levels = [0]
        series = [
            (c, np.ones(len(t), dtype=bool), c,
             (cause_labels or {}).get(c, f"cause {c}"))
            for c in causes
        ]

    colors = _default_palette(len(series), palette)
    fig, ax, ax_risk = _make_axes(ax, figsize, risk_table, len(series))

    fits: Dict[Any, Dict[str, np.ndarray]] = {}
    for (key, mask, this_cause, label), color in zip(series, colors):
        fit = aalen_johansen(t[mask], e[mask], this_cause, alpha=ci_alpha)
        fits[key] = fit
        ax.step(fit["timeline"], fit["cif"], where="post", color=color,
                linewidth=linewidth, label=f"{label} (n={fit['n']})")
        if ci:
            _step_ci(ax, fit["timeline"], fit["lower"], fit["upper"], color,
                     ci_opacity)
        if show_naive_km:
            km = kaplan_meier(t[mask], np.asarray(e[mask]) == this_cause)
            ax.step(km["timeline"], 1.0 - km["survival"], where="post",
                    color=color, linewidth=linewidth * 0.8, linestyle="--",
                    alpha=0.75, label=f"{label}, 1 - KM")

    stats: Dict[str, Any] = {"fits": fits}
    if has_group and test and len(levels) >= 2:
        result = grays_test(t, e, g, causes[0], groups=levels)
        stats.update(result)
        ax.text(0.02, 0.98, f"Gray's test {_format_p(result['pvalue'])}",
                transform=ax.transAxes, ha="left", va="top", fontsize=font_size(fontsize),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.8", alpha=0.85))

    ax.set_xlim(xlim if xlim is not None else (0, float(np.max(t)) * 1.02))
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0.0, None)
    if percent:
        ticks = ax.get_yticks()
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v * 100:.0f}" for v in ticks])
    time_label = xlabel if xlabel is not None else names.get("time", "Time")
    if ax_risk is None:
        ax.set_xlabel(time_label, fontsize=font_size(fontsize, "label"))
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel(
        ylabel if ylabel is not None
        else ("Cumulative incidence (%)" if percent else "Cumulative incidence"),
        fontsize=font_size(fontsize, "label"),
    )
    if title:
        ax.set_title(title, fontsize=font_size(fontsize, "title"))
    ax.tick_params(labelsize=font_size(fontsize))
    style_axes(ax)
    if legend:
        ax.legend(loc=legend_loc, frameon=False, fontsize=font_size(fontsize))

    if ax_risk is not None:
        ticks = (np.asarray(risk_table_times, dtype=float)
                 if risk_table_times is not None
                 else np.asarray([v for v in ax.get_xticks()
                                  if ax.get_xlim()[0] <= v <= ax.get_xlim()[1]]))
        counts = [_at_risk(t[mask], ticks) for _, mask, _, _ in series]
        _draw_risk_table(ax_risk, ticks, [lab for *_, lab in series], counts,
                         colors, font_size(fontsize, "tick", -0.5), ax.get_xlim(), time_label)
        ax_risk.set_xlim(ax.get_xlim())

    return (ax, stats) if return_stats else ax
