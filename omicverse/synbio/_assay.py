r"""The **T** in design-build-test-learn — turning measurements into parameters.

A plate reader produces a few hundred OD traces per run, and that is the single
most common measurement in the field. ``ov.synbio`` could *simulate* growth
(:func:`~omicverse.synbio.dynamic_fba`, :func:`~omicverse.synbio.rba`) and had
nothing at all that could *fit* it.

* :func:`fit_growth_curve` — µmax, lag, carrying capacity and doubling time from
  one OD trace, under a named growth model.
* :func:`fit_growth_curves` — the same over a plate, returned tidy.
* :func:`read_plate_reader` — parse the wide or long CSV a plate reader exports.
* :func:`dose_response` — Hill fit giving EC50/IC50, slope, and the MIC.
* :func:`compare_growth_to_model` — the coupling that only this package can make:
  a *measured* µmax against the growth an FBA or RBA model predicts, on the same
  axis and in the same units.
* :func:`plot_growth_curves` / :func:`plot_dose_response`.

**On µmax.** ``mu_max`` is the specific growth rate at the inflection point,
``(dOD/dt)_max / OD(t_inflection)``, in 1/h — derived from the fitted curve
rather than read off whichever parameter a model happens to call µ. The published
parameterisations disagree about that: Gompertz and logistic define µ as the
steepest slope of OD against time (OD/h) while Baranyi's is already a specific
rate, and on one synthetic curve the two conventions read 0.43 and 1.43 for
identical growth. The sigmoid's own slope remains available as ``slope_max``.

Only a specific rate can be compared with an FBA or RBA growth rate, or turned
into a doubling time — which is why :func:`compare_growth_to_model` uses this one.
A residual model dependence survives, because the models put the inflection at
different heights (A/e for Gompertz, A/2 for logistic); that difference is real,
and :func:`compare_growth_models` shows it rather than averaging it away. When the
specific rate is the number that matters, fitting with ``log_transform=True``
makes µ a specific rate by construction.

**On the models.** Four are offered and they are not interchangeable. Logistic
is symmetric and has no explicit lag term, so fitting it to a curve with a real
lag pushes the error into µmax. The modified Gompertz is asymmetric and
parameterises lag directly, which is why it dominates the microbiology
literature. Richards adds a shape parameter for when neither fits. Baranyi is
mechanistic and the reference model in predictive microbiology. The default is
Gompertz, and :attr:`GrowthFit.r_squared` plus :func:`compare_growth_models` are
there so the choice can be checked rather than assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


GROWTH_MODELS = ("gompertz", "logistic", "richards", "baranyi", "exponential")


# ---------------------------------------------------------------------------
# growth models
# ---------------------------------------------------------------------------

def _gompertz(t, A, mu, lag):
    """Modified Gompertz (Zwietering 1990), parameterised in µmax and lag.

    ``A`` is the asymptotic increase in OD, ``mu`` the maximum specific growth
    rate (the slope of the steepest tangent) and ``lag`` the intercept of that
    tangent with the time axis — so both come straight out of the fit rather
    than being read off the plot.
    """
    import numpy as np
    return A * np.exp(-np.exp(mu * math.e / A * (lag - t) + 1.0))


def _logistic(t, A, mu, lag):
    """Logistic, in the same (A, µmax, lag) parameterisation.

    Symmetric about the inflection: it *can* absorb a lag through the shift
    term, but it cannot represent a flat lag followed by a sharp acceleration,
    which is what a real lag looks like.
    """
    import numpy as np
    return A / (1.0 + np.exp(4.0 * mu / A * (lag - t) + 2.0))


def _richards(t, A, mu, lag, nu):
    """Richards — logistic with a shape parameter ``nu`` controlling asymmetry."""
    import numpy as np
    nu = max(nu, 1e-6)
    inner = 1.0 + nu * np.exp(1.0 + nu) * np.exp(
        mu / A * (1.0 + nu) ** (1.0 + 1.0 / nu) * (lag - t))
    return A * np.power(np.clip(inner, 1e-12, None), -1.0 / nu)


def _baranyi(t, A, mu, lag, y0=1e-3):
    """Baranyi-Roberts (1994), the reference model in predictive microbiology.

    Lag enters through an *adjustment function* rather than as a shifted
    sigmoid, which is what makes it mechanistic: the cell must build up an
    internal resource before growth begins, and ``A(t)`` is the physiological
    time that accumulates while it does.

        A(t) = t + (1/µ)·ln(e^(−µt) + e^(−h₀) − e^(−µt−h₀)),  h₀ = µ·λ
        y(t) = y₀·e^(µ·A(t)) / (1 + (e^(µ·A(t)) − 1)·y₀/A)

    The second line matters. An earlier version used a Monod-shaped
    ``A·(1 − e^(−µ·A(t)/A))`` saturation instead, which fits the curve just as
    well (R² 0.998 on a synthetic Gompertz trace) while making ``µ`` a different
    quantity — it reported µmax = 1.50 for data generated with µmax = 0.45. A
    parameter named µmax has to be µmax in every model, or the models cannot be
    compared and :func:`compare_growth_models` becomes misleading.
    """
    import numpy as np
    mu = max(float(mu), 1e-9)
    A = max(float(A), 1e-9)
    y0 = min(max(float(y0), 1e-9), A * 0.99)
    h0 = mu * lag
    with np.errstate(over="ignore", invalid="ignore"):
        inner = np.exp(-mu * t) + np.exp(-h0) - np.exp(-mu * t - h0)
        at = t + (1.0 / mu) * np.log(np.clip(inner, 1e-300, None))
        growth = np.exp(np.clip(mu * at, -700, 700))
        return y0 * growth / (1.0 + (growth - 1.0) * y0 / A)


def _exponential(t, y0, mu):
    """Pure exponential — only valid on a window that really is log phase."""
    import numpy as np
    return y0 * np.exp(mu * t)


@dataclass
class GrowthFit:
    """Parameters of one fitted growth curve."""

    model: str
    mu_max: float                      # 1/h — maximum SPECIFIC growth rate
    slope_max: float = 0.0             # OD/h — max dOD/dt, the sigmoid's own µ
    lag: float = 0.0                   # h
    carrying_capacity: float = 0.0     # OD units above the blank
    y0: float = 0.0
    r_squared: float = 0.0
    rmse: float = 0.0
    n_points: int = 0
    converged: bool = True
    parameters: Dict[str, float] = field(default_factory=dict)
    well: str = ""
    label: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def doubling_time(self) -> float:
        """Hours per doubling at µmax.

        Uses the *specific* rate, which is the only one that gives a doubling
        time — an absolute slope in OD/h cannot, since the time to double
        depends on where you start.
        """
        return math.log(2.0) / self.mu_max if self.mu_max > 1e-12 else float("inf")

    @property
    def good_fit(self) -> bool:
        return self.converged and self.r_squared >= 0.95

    def predict(self, t):
        """Model curve at times ``t``, for overlaying on the data."""
        import numpy as np
        t = np.asarray(t, dtype=float)
        p = self.parameters
        if self.model == "gompertz":
            return _gompertz(t, p["A"], p["mu"], p["lag"])
        if self.model == "logistic":
            return _logistic(t, p["A"], p["mu"], p["lag"])
        if self.model == "richards":
            return _richards(t, p["A"], p["mu"], p["lag"], p["nu"])
        if self.model == "baranyi":
            return _baranyi(t, p["A"], p["mu"], p["lag"], p.get("y0", self.y0))
        return _exponential(t, p["y0"], p["mu"])

    def __repr__(self) -> str:  # pragma: no cover
        tag = "" if self.good_fit else "  [poor fit]"
        return (f"GrowthFit({self.model}, µmax={self.mu_max:.4f} /h, "
                f"td={self.doubling_time:.2f} h, lag={self.lag:.2f} h, "
                f"A={self.carrying_capacity:.3f}, R²={self.r_squared:.4f}{tag})")


@register_function(
    aliases=["fit_growth_curve", "生长曲线拟合", "生长速率", "growth_curve",
             "umax", "倍增时间", "Gompertz", "迟滞期", "doubling_time"],
    category="synthetic_biology",
    description="从一条 OD 时序拟合生长参数:µmax(最大比生长速率)、迟滞期 λ、载容量 A、倍增时间、R²。model='gompertz'(默认,不对称、显式含迟滞,微生物学文献主流)/'logistic'(对称,无显式迟滞项 —— 有真实迟滞时误差会挤进 µmax)/'richards'(多一个形状参数)/'baranyi'(预测微生物学的机理参考模型)/'exponential'(仅对数期窗口)。酶标仪一次跑几百条曲线,这是最常见的测量。Fit growth parameters from an OD time course.",
    examples=[
        "fit = ov.synbio.fit_growth_curve(time_h, od600)",
        "fit.mu_max, fit.doubling_time, fit.lag, fit.r_squared",
        "fit = ov.synbio.fit_growth_curve(t, od, model='baranyi', blank=0.05)",
    ],
    related=["synbio.fit_growth_curves", "synbio.compare_growth_models",
             "synbio.compare_growth_to_model", "synbio.dynamic_fba",
             "synbio.plot_growth_curves"],
    requires={},
    produces={},
)
def fit_growth_curve(
    time,
    od,
    *,
    model: str = "gompertz",
    blank: float = 0.0,
    log_transform: bool = False,
    min_amplitude: float = 0.05,
    min_fold_change: float = 1.5,
    mu_max_ceiling: float = 6.0,
    label: str = "",
    well: str = "",
    maxfev: int = 20000,
) -> GrowthFit:
    """Fit a growth model to one OD trace.

    Parameters
    ----------
    time
        Times, in hours. Minutes are detected and converted — a trace whose last
        timepoint is in the hundreds is almost certainly minutes, and silently
        fitting it as hours puts µmax out by 60x.
    od
        Optical density, same length as ``time``.
    model
        See :data:`GROWTH_MODELS`.
    blank
        Media blank subtracted before fitting. Not optional in spirit: an
        unsubtracted blank inflates the apparent carrying capacity and flattens
        µmax.
    log_transform
        Fit ``ln(OD)`` instead. Weights the early exponential phase more, which
        helps when the plateau is noisy.
    label, well
        Carried through to the result for plotting and tables.

    Returns
    -------
    GrowthFit
    """
    if model not in GROWTH_MODELS:
        raise ValueError(f"model must be one of {list(GROWTH_MODELS)}, got {model!r}")
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "生长曲线拟合需要 scipy(随 omicverse 主依赖安装)。") from exc

    t = np.asarray(time, dtype=float)
    y = np.asarray(od, dtype=float)
    if t.shape != y.shape:
        raise ValueError(f"time 与 od 长度不一致:{t.shape} vs {y.shape}")
    if t.size < 5:
        raise ValueError(
            f"只有 {t.size} 个点,拟合不出 4 个参数。生长曲线至少需要 5 个时间点。")

    notes: List[str] = []
    if np.nanmax(t) > 100:
        notes.append(
            f"时间轴最大值 {np.nanmax(t):.0f} —— 判定为**分钟**并已转换为小时。"
            f"按小时拟合会让 µmax 差 60 倍。")
        t = t / 60.0

    y = y - float(blank)
    finite = np.isfinite(t) & np.isfinite(y)
    t, y = t[finite], y[finite]
    order = np.argsort(t)
    t, y = t[order], y[order]

    if np.nanmax(y) <= 0:
        return GrowthFit(model=model, mu_max=0.0, converged=False,
                         n_points=int(t.size), label=label, well=well,
                         notes=notes + ["扣除空白后 OD 全部 <= 0,没有生长可拟合。"])

    fit_y = np.log(np.clip(y, 1e-6, None)) if log_transform else y

    A0 = float(np.nanmax(y))
    # slope of the steepest 3-point window as the initial µmax
    with np.errstate(all="ignore"):
        slopes = np.gradient(np.log(np.clip(y, 1e-6, None)), t)
    mu0 = float(np.nanmax(slopes[np.isfinite(slopes)])) if slopes.size else 0.5
    mu0 = max(min(mu0, 5.0), 1e-3)
    lag0 = float(t[int(np.argmax(y > 0.1 * A0))]) if A0 > 0 else 0.0

    if model == "exponential":
        fn: Callable = _exponential
        p0 = [max(float(y[0]), 1e-4), mu0]
        bounds = ([1e-9, 1e-6], [A0 * 10 + 1e-6, 10.0])
        names = ["y0", "mu"]
    elif model == "richards":
        fn = _richards
        p0 = [A0, mu0, lag0, 1.0]
        bounds = ([1e-6, 1e-6, 0.0, 1e-3], [A0 * 5, 10.0, max(t.max(), 1.0), 20.0])
        names = ["A", "mu", "lag", "nu"]
    elif model == "baranyi":
        fn = _baranyi
        y00 = max(float(np.nanmin(y[y > 0])) if np.any(y > 0) else 1e-3, 1e-6)
        p0 = [A0, mu0, lag0, y00]
        bounds = ([1e-6, 1e-6, 0.0, 1e-9],
                  [A0 * 5, 10.0, max(t.max(), 1.0), max(A0, 1e-6)])
        names = ["A", "mu", "lag", "y0"]
    else:
        fn = {"gompertz": _gompertz, "logistic": _logistic}[model]
        p0 = [A0, mu0, lag0]
        bounds = ([1e-6, 1e-6, 0.0], [A0 * 5, 10.0, max(t.max(), 1.0)])
        names = ["A", "mu", "lag"]

    def wrapped(tt, *params):
        out = fn(tt, *params)
        return np.log(np.clip(out, 1e-6, None)) if log_transform else out

    converged = True
    try:
        popt, _ = curve_fit(wrapped, t, fit_y, p0=p0, bounds=bounds, maxfev=maxfev)
    except Exception as exc:
        converged = False
        popt = np.asarray(p0, dtype=float)
        notes.append(f"拟合未收敛({type(exc).__name__}),返回的是初值估计。")

    params = {n: float(v) for n, v in zip(names, popt)}
    pred = fn(t, *popt)
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / t.size))

    if model == "exponential":
        native = params["mu"]
        A = float(np.nanmax(y))
        lag = 0.0
        notes.append("exponential 模型不估计迟滞期与载容量;lag 报 0。")
    else:
        native, A, lag = params["mu"], params["A"], params["lag"]

    # The specific growth rate at the inflection point, for every model.
    #
    # Two things forced this. The published parameterisations do not agree on
    # what µ means: Gompertz and logistic define it as the steepest slope of OD
    # against time (OD/h), Baranyi's is already a specific rate (1/h), and on one
    # synthetic curve the two conventions read 0.43 and 1.43 for identical
    # growth. And max d(ln y)/dt — the obvious repair — is unbounded as y → 0, so
    # it is dominated by wherever the noise floor happens to be cut.
    #
    # The inflection is where every sigmoid has a well-defined answer:
    # µ_specific = (dOD/dt)_max / OD(t_inflection). A residual model dependence
    # remains, because the models place the inflection at different heights
    # (A/e for Gompertz, A/2 for logistic) — that difference is real and is
    # reported rather than hidden. compare_growth_models shows it side by side.
    dense = np.linspace(float(t.min()), float(t.max()), 800)
    with np.errstate(all="ignore"):
        curve = np.clip(np.asarray(fn(dense, *popt), dtype=float), 1e-12, None)
        slope = np.gradient(curve, dense)
    idx = int(np.nanargmax(slope))
    slope_max = float(slope[idx])
    od_at_inflection = float(curve[idx])
    specific = (slope_max / od_at_inflection
                if od_at_inflection > 1e-9 else float(native))
    if not np.isfinite(specific) or specific <= 0:
        specific = float(native)
        notes.append("拐点处的比生长速率算不出来,回退到模型自身的 µ 参数。")

    if r2 < 0.95 and converged:
        notes.append(
            f"R² = {r2:.3f} < 0.95 —— 换个模型试试(compare_growth_models),"
            f"或检查是否有污染、蒸发、或分光光度计的非线性上限。")

    # Amplitude gate. A four-parameter sigmoid fits any monotone trace, so R² is
    # not quality control: a well whose whole excursion is inside the blank noise
    # still converges. And µ = slope/OD explodes as OD -> 0, so a flat trace comes
    # back at 28 /h — a 1.5-minute doubling time — which then tops any nlargest().
    # NaN rather than a plausible-looking number, because NaN propagates.
    observed_range = float(np.nanmax(y) - np.nanmin(y))
    fold_change = (float(np.nanmax(y) / max(np.nanmin(y), 1e-12))
                   if np.nanmin(y) > 0 else float("inf"))
    if observed_range < min_amplitude or fold_change < min_fold_change:
        notes.append(
            f"OD 只变化了 {observed_range:.4f}(≈{fold_change:.2f} 倍),低于 "
            f"min_amplitude={min_amplitude} / min_fold_change={min_fold_change} —— "
            f"这个孔没有长起来。µmax 返回 nan 而不是拟合值:比速率是 slope/OD,"
            f"OD 趋近 0 时它会爆掉(平坦曲线能给出 28 /h,即 1.5 分钟倍增)。")
        specific = float("nan")
        converged = False
    elif specific > mu_max_ceiling:
        notes.append(
            f"拟合出的 µmax = {specific:.3g} /h 超过上限 {mu_max_ceiling} /h"
            f"(约 {60 * math.log(2) / mu_max_ceiling:.0f} 分钟倍增,已快于任何已知"
            f"细菌) —— 返回 nan。通常是曲线幅度太小、或 OD 进入了读数器的非线性区。")
        specific = float("nan")
        converged = False

    return GrowthFit(
        model=model, mu_max=float(specific), slope_max=slope_max,
        lag=float(lag),
        carrying_capacity=float(A), y0=float(y[0]), r_squared=float(r2),
        rmse=rmse, n_points=int(t.size), converged=converged,
        parameters=params, well=well, label=label, notes=notes)


@register_function(
    aliases=["compare_growth_models", "生长模型比较", "模型选择",
             "compare_models", "选模型"],
    category="synthetic_biology",
    description="把同一条曲线用几个生长模型都拟合一遍并比较 R²/RMSE/AIC —— logistic 是对称的、没有显式迟滞项,遇到真实迟滞会把误差挤进 µmax,所以模型该选哪个要看数据而不是默认。Fit one curve under several growth models and compare.",
    examples=["df = ov.synbio.compare_growth_models(t, od)"],
    related=["synbio.fit_growth_curve"],
    requires={},
    produces={},
)
def compare_growth_models(time, od, *, models: Optional[Sequence[str]] = None,
                          blank: float = 0.0) -> "pd.DataFrame":
    """One row per model: µmax, lag, R², RMSE, AIC."""
    import numpy as np
    import pandas as pd

    models = list(models or ("gompertz", "logistic", "richards", "baranyi"))
    rows = []
    for m in models:
        try:
            fit = fit_growth_curve(time, od, model=m, blank=blank)
        except Exception as exc:
            rows.append({"model": m, "mu_max": np.nan, "slope_max": np.nan,
                         "lag": np.nan, "r_squared": np.nan, "rmse": np.nan,
                         "aic": np.nan, "converged": False,
                         "note": type(exc).__name__})
            continue
        k = len(fit.parameters)
        n = max(fit.n_points, k + 2)
        aic = n * math.log(max(fit.rmse ** 2, 1e-12)) + 2 * k
        rows.append({"model": m, "mu_max": fit.mu_max,
                     "slope_max": fit.slope_max, "lag": fit.lag,
                     "r_squared": fit.r_squared, "rmse": fit.rmse, "aic": aic,
                     "converged": fit.converged, "note": ""})
    frame = pd.DataFrame(rows).set_index("model").sort_values("aic")
    # Mark the AIC-preferred model. Without this the table is easy to print and
    # then ignore: on a real well the default (Gompertz) came last by 233 AIC
    # units while its mu_max was 1.4x the other three, and nothing said so.
    usable = frame[frame["converged"] & frame["mu_max"].notna()]
    if len(usable):
        best = usable["aic"].idxmin()
        frame["preferred"] = frame.index == best
        spread = usable["mu_max"].max() / max(usable["mu_max"].min(), 1e-12)
        frame.attrs["preferred_model"] = best
        frame.attrs["mu_max_spread"] = float(spread)
        frame.attrs["note"] = (
            f"AIC 最优:{best!r}。可用模型之间 µmax 相差 {spread:.2f} 倍 —— "
            f"各模型把拐点放在不同高度(Gompertz 在 A/e,logistic 在 A/2),"
            f"所以它们对'比速率'的定义本就不同。要把 µmax 和 FBA 比,"
            f"先在这里选模型,再传给 fit_growth_curves(model=...)。")
    else:
        frame["preferred"] = False
        frame.attrs["note"] = "没有模型给出可用的 µmax —— 这个孔大概没有长起来。"
    return frame


@register_function(
    aliases=["fit_growth_curves", "批量生长曲线", "板级拟合",
             "fit_plate", "growth_curves"],
    category="synthetic_biology",
    description="对一整块板的 OD 时序批量拟合生长参数,返回整齐的 DataFrame(每孔一行:µmax、迟滞、载容量、倍增时间、R²)。可给 blanks= 指定空白孔并自动扣除,可给 plate= 把孔位映射回样品名。Fit growth parameters across a whole plate.",
    examples=[
        "df = ov.synbio.fit_growth_curves(od_frame, time_col='time_h')",
        "df = ov.synbio.fit_growth_curves(od_frame, blanks=['A1','A12'], plate=plate)",
    ],
    related=["synbio.fit_growth_curve", "synbio.read_plate_reader",
             "synbio.plot_growth_curves"],
    requires={},
    produces={},
)
def fit_growth_curves(
    data,
    *,
    time_col: str = "time_h",
    model: str = "gompertz",
    blanks: Optional[Sequence[str]] = None,
    blank: float = 0.0,
    plate=None,
    min_r_squared: float = 0.95,
) -> "pd.DataFrame":
    """Fit every well of a plate.

    Parameters
    ----------
    data
        DataFrame with a time column and one column per well.
    blanks
        Well names holding media only. Their mean is subtracted from every other
        well, which is the correct blank rather than a number typed in by hand.
    plate
        A :class:`~omicverse.synbio.Plate`, used to label each well with what is
        in it.
    min_r_squared
        Wells below this are kept but flagged, not dropped — a bad fit is
        information about that well, and silently dropping it biases whatever
        summary comes next.
    """
    import numpy as np
    import pandas as pd

    if time_col not in data.columns:
        raise ValueError(
            f"找不到时间列 {time_col!r}。现有列:{list(data.columns)[:8]}…")

    t = data[time_col].to_numpy(dtype=float)
    wells = [c for c in data.columns if c != time_col]
    if not wells:
        raise ValueError("除时间列外没有任何孔位列。")

    blank_value = float(blank)
    if blanks:
        missing = [b for b in blanks if b not in data.columns]
        if missing:
            raise ValueError(f"空白孔不在数据里:{missing}")
        blank_value = float(np.nanmean(data[list(blanks)].to_numpy(dtype=float)))
        wells = [w for w in wells if w not in set(blanks)]

    rows = []
    for w in wells:
        label = ""
        if plate is not None:
            label = plate.contents.get(w, "")
        fit = fit_growth_curve(t, data[w].to_numpy(dtype=float), model=model,
                              blank=blank_value, well=w, label=label)
        rows.append({
            "well": w, "label": label, "mu_max": fit.mu_max,
            "slope_max": fit.slope_max,
            "doubling_time_h": fit.doubling_time, "lag_h": fit.lag,
            "carrying_capacity": fit.carrying_capacity,
            "r_squared": fit.r_squared, "rmse": fit.rmse,
            "converged": fit.converged,
            "good_fit": fit.converged and fit.r_squared >= min_r_squared,
        })
    out = pd.DataFrame(rows).set_index("well")
    out.attrs["blank"] = blank_value
    out.attrs["model"] = model
    return out


@register_function(
    aliases=["read_plate_reader", "读酶标仪", "plate_reader", "OD数据读取",
             "读板数据"],
    category="synthetic_biology",
    description="读取酶标仪导出的 CSV/TSV/Excel:宽表(第一列时间、其余每列一个孔)与长表(time/well/value 三列)都能识别,时间单位可自动换算成小时。Read a plate-reader export in wide or long layout.",
    examples=[
        "df = ov.synbio.read_plate_reader('od600.csv')",
        "df = ov.synbio.read_plate_reader('run.xlsx', layout='long')",
    ],
    related=["synbio.fit_growth_curves", "synbio.plot_growth_curves"],
    requires={},
    produces={},
)
def read_plate_reader(
    path: str,
    *,
    layout: str = "auto",
    time_col: Optional[str] = None,
    well_col: str = "well",
    value_col: str = "value",
    time_unit: str = "auto",
) -> "pd.DataFrame":
    """Return a wide DataFrame: a ``time_h`` column plus one column per well."""
    import numpy as np
    import pandas as pd

    if layout not in ("auto", "wide", "long"):
        raise ValueError(
            f"layout must be one of ['auto', 'wide', 'long'], got {layout!r}")
    if time_unit not in ("auto", "h", "min", "s"):
        raise ValueError(
            f"time_unit must be one of ['auto', 'h', 'min', 's'], got {time_unit!r}")

    lower = str(path).lower()
    if lower.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path)
    elif lower.endswith((".tsv", ".txt")):
        raw = pd.read_csv(path, sep="\t")
    else:
        raw = pd.read_csv(path)

    cols = list(raw.columns)
    if layout == "auto":
        layout = "long" if (well_col in cols and value_col in cols) else "wide"

    if layout == "long":
        tcol = time_col or next(
            (c for c in cols if "time" in str(c).lower()), cols[0])
        wide = raw.pivot_table(index=tcol, columns=well_col, values=value_col,
                              aggfunc="mean").reset_index()
        wide = wide.rename(columns={tcol: "time_raw"})
    else:
        tcol = time_col or cols[0]
        wide = raw.rename(columns={tcol: "time_raw"})

    times = pd.to_numeric(wide["time_raw"], errors="coerce").to_numpy(dtype=float)
    unit = time_unit
    if unit == "auto":
        span = float(np.nanmax(times) - np.nanmin(times)) if times.size else 0.0
        unit = "s" if span > 6000 else ("min" if span > 100 else "h")
    factor = {"h": 1.0, "min": 1.0 / 60.0, "s": 1.0 / 3600.0}[unit]

    wide = wide.drop(columns=["time_raw"])
    wide.insert(0, "time_h", times * factor)
    wide.attrs["time_unit_detected"] = unit
    return wide


# ---------------------------------------------------------------------------
# dose response
# ---------------------------------------------------------------------------

@dataclass
class DoseResponse:
    """A fitted dose-response curve."""

    ec50: float
    hill_slope: float
    top: float
    bottom: float
    r_squared: float = 0.0
    mic: Optional[float] = None
    inhibitory: bool = False
    n_points: int = 0
    converged: bool = True
    notes: List[str] = field(default_factory=list)

    @property
    def ic50(self) -> Optional[float]:
        """EC50 under its inhibitory name — the same number, reported as IC50
        only when the response actually decreases with dose."""
        return self.ec50 if self.inhibitory else None

    def predict(self, concentration):
        import numpy as np
        c = np.asarray(concentration, dtype=float)
        return _hill(c, self.bottom, self.top, self.ec50, self.hill_slope)

    def __repr__(self) -> str:  # pragma: no cover
        name = "IC50" if self.inhibitory else "EC50"
        mic = "" if self.mic is None else f", MIC={self.mic:.4g}"
        return (f"DoseResponse({name}={self.ec50:.4g}, hill={self.hill_slope:.2f}, "
                f"R²={self.r_squared:.4f}{mic})")


def _hill(c, bottom, top, ec50, hill):
    import numpy as np
    c = np.clip(np.asarray(c, dtype=float), 1e-12, None)
    return bottom + (top - bottom) / (1.0 + (ec50 / c) ** hill)


@register_function(
    aliases=["dose_response", "剂量反应", "IC50", "EC50", "MIC", "Hill拟合",
             "抑制曲线", "剂量效应"],
    category="synthetic_biology",
    description="四参数 Hill 方程拟合剂量反应曲线,给出 EC50/IC50、Hill 斜率、上下平台与 R²;响应随剂量下降时同时给 MIC(达到指定抑制比例的最低浓度)。Fit a four-parameter Hill dose-response curve.",
    examples=[
        "res = ov.synbio.dose_response(concentrations, od)",
        "res.ic50, res.hill_slope, res.mic",
    ],
    related=["synbio.fit_growth_curve", "synbio.plot_dose_response"],
    requires={},
    produces={},
)
def dose_response(
    concentration,
    response,
    *,
    mic_threshold: float = 0.9,
    maxfev: int = 20000,
) -> DoseResponse:
    """Fit ``response`` against ``concentration``.

    Parameters
    ----------
    concentration
        Doses. Zero is allowed and handled — a true zero cannot appear on the
        log axis the Hill equation lives on, so it is used for the top plateau
        rather than dropped.
    response
        Measured response (OD, fluorescence, viability).
    mic_threshold
        Fraction of inhibition defining the MIC, e.g. 0.9 for MIC90.
    """
    import numpy as np
    from scipy.optimize import curve_fit

    c = np.asarray(concentration, dtype=float)
    r = np.asarray(response, dtype=float)
    if c.shape != r.shape:
        raise ValueError(f"concentration 与 response 长度不一致:{c.shape} vs {r.shape}")
    if c.size < 4:
        raise ValueError(
            f"只有 {c.size} 个点,四参数 Hill 拟合至少需要 4 个浓度。")

    finite = np.isfinite(c) & np.isfinite(r)
    c, r = c[finite], r[finite]
    notes: List[str] = []

    zero = c <= 0
    if zero.any():
        notes.append(
            f"{int(zero.sum())} 个零浓度点用于确定上平台,不参与对数轴拟合"
            f"(Hill 方程在 c=0 处无定义)。")

    nz = ~zero
    if nz.sum() < 4:
        raise ValueError("非零浓度点少于 4 个,无法拟合。")

    inhibitory = bool(np.nanmean(r[c > np.median(c[nz])])
                      < np.nanmean(r[c < np.median(c[nz])]))

    top0 = float(np.nanmax(r))
    bottom0 = float(np.nanmin(r))
    ec0 = float(np.exp(np.nanmean(np.log(c[nz]))))
    hill0 = -1.0 if inhibitory else 1.0

    converged = True
    try:
        popt, _ = curve_fit(
            _hill, c[nz], r[nz], p0=[bottom0, top0, ec0, hill0],
            bounds=([-np.inf, -np.inf, c[nz].min() * 1e-3, -20.0],
                    [np.inf, np.inf, c[nz].max() * 1e3, 20.0]),
            maxfev=maxfev)
    except Exception as exc:
        converged = False
        popt = np.array([bottom0, top0, ec0, hill0], dtype=float)
        notes.append(f"拟合未收敛({type(exc).__name__}),返回初值估计。")

    bottom, top, ec50, hill = (float(v) for v in popt)
    pred = _hill(c[nz], bottom, top, ec50, hill)
    ss_res = float(np.sum((r[nz] - pred) ** 2))
    ss_tot = float(np.sum((r[nz] - np.mean(r[nz])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mic = None
    if inhibitory and top != bottom:
        target = top - mic_threshold * (top - bottom)
        # invert the Hill equation for the concentration giving `target`
        frac = (target - bottom) / (top - bottom)
        if 0 < frac < 1:
            mic = float(ec50 * ((1.0 - frac) / frac) ** (1.0 / hill))

    return DoseResponse(ec50=ec50, hill_slope=hill, top=top, bottom=bottom,
                        r_squared=r2, mic=mic, inhibitory=inhibitory,
                        n_points=int(nz.sum()), converged=converged, notes=notes)


# ---------------------------------------------------------------------------
# the coupling: measured growth against a model's prediction
# ---------------------------------------------------------------------------

@register_function(
    aliases=["compare_growth_to_model", "实测对模型", "生长速率比对",
             "validate_model_growth", "measured_vs_predicted"],
    category="synthetic_biology",
    description="把**实测**的 µmax 和 GEM 预测的生长速率放在同一个坐标里比 —— FBA 是上界、RBA 加了蛋白质组预算、MOMA 是敲除后的近邻解,实测落在哪里本身就是信息。这是 omicverse 独有的咬合:同时握着实测生长与代谢模型两端。Compare measured µmax against a model's predicted growth rate.",
    examples=[
        "df = ov.synbio.compare_growth_to_model(fits, model)",
        "df = ov.synbio.compare_growth_to_model({'wt': 0.62, 'ko': 0.31}, model)",
    ],
    related=["synbio.fit_growth_curves", "synbio.fba", "synbio.rba",
             "synbio.knockout_flux"],
    requires={},
    produces={},
)
def compare_growth_to_model(
    measured,
    model,
    *,
    total_protein: Optional[float] = 0.55,
    knockouts: Optional[Mapping[str, Sequence[str]]] = None,
    growth_model: str = "gompertz",
) -> "pd.DataFrame":
    """Measured µmax beside FBA, RBA and (optionally) MOMA predictions.

    Parameters
    ----------
    measured
        ``{label: mu_max}``, or the DataFrame from :func:`fit_growth_curves`
        (its ``label`` column is used when present, else the well).
    model
        A :class:`cobra.Model`.
    total_protein
        Proteome budget for the RBA column; None skips it.
    knockouts
        ``{label: [reaction_ids]}`` — for labels that are knockout strains, adds
        FBA and MOMA predictions *for that deletion* rather than for wild type.
        Without this every row is compared against the wild-type prediction,
        which is only meaningful for wild-type wells.
    growth_model
        Recorded in ``.attrs`` and named in the ratio warning, because the choice
        moves the answer: on the same plate and the same GEM, Gompertz gives
        measured/FBA = 1.29 and Richards or Baranyi give 0.91. The models place the
        inflection at different heights, so they disagree about what "the" specific
        rate is by about 1.4x — see :func:`compare_growth_models`.

    Returns
    -------
    pandas.DataFrame
        One row per label, with ``measured``, ``fba``, ``rba``, ``moma`` where
        available, plus the ratio of measured to each prediction.
    """
    import numpy as np
    import pandas as pd

    if hasattr(measured, "columns"):
        col = "mu_max"
        if col not in measured.columns:
            raise ValueError(
                f"DataFrame 里没有 {col!r} 列 —— 期望 fit_growth_curves 的输出。")
        labels = (measured["label"].where(measured["label"].astype(bool),
                                          measured.index.to_series())
                  if "label" in measured.columns else measured.index.to_series())
        obs = dict(zip(labels, measured[col]))
    else:
        obs = {str(k): float(v) for k, v in dict(measured).items()}

    wt_fba = model.slim_optimize()
    wt_fba = 0.0 if wt_fba is None or wt_fba != wt_fba else float(wt_fba)

    wt_rba = None
    if total_protein is not None:
        try:
            from ._community import rba
            wt_rba = rba(model, total_protein=total_protein).growth
        except Exception:
            wt_rba = None

    knockouts = dict(knockouts or {})
    rows = []
    for label, mu in obs.items():
        fba_pred, moma_pred = wt_fba, None
        if label in knockouts:
            kos = list(knockouts[label])
            try:
                from ._perturb import knockout_flux
                fba_pred = knockout_flux(model, kos, method="fba").growth
                moma_pred = knockout_flux(model, kos,
                                          method="linear_moma").growth
            except Exception:
                pass
        row = {"label": label, "measured": float(mu), "fba": fba_pred}
        if wt_rba is not None:
            row["rba"] = wt_rba
        if moma_pred is not None:
            row["moma"] = moma_pred
        for key in ("fba", "rba", "moma"):
            if key in row and row[key] and row[key] > 1e-12:
                row[f"measured_over_{key}"] = float(mu) / row[key]
        rows.append(row)

    out = pd.DataFrame(rows).set_index("label")
    out.attrs["growth_model"] = growth_model
    exceed = out[out.get("measured_over_fba", pd.Series(dtype=float)) > 1.05]
    if len(exceed):
        out.attrs["warning"] = (
            f"{len(exceed)} 个样品的实测 µmax 超过 FBA 上界 5% 以上 —— FBA 是"
            f"化学计量上界,实测不该超过它。按先后顺序查:"
            f"(1) 生长模型的选择 —— 这些 µmax 来自 {growth_model!r},换成 AIC 更优的"
            f"模型可能整体降低约 1.4 倍,先跑 compare_growth_models;"
            f"(2) 培养基约束是否比实际更严(模型的碳源上限);"
            f"(3) 时间单位是否都是小时。"
            f"注意:OD 到干重的换算不会改变这个比值 —— µmax 是比速率,"
            f"OD 与干重之间的任何常数因子都会约掉。")
    return out


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

@register_function(
    aliases=["plot_growth_curves", "生长曲线图", "plot_growth", "OD曲线",
             "生长图"],
    category="synthetic_biology",
    description="画生长曲线与拟合叠加,并标出 µmax 切线与迟滞期截距 —— 迟滞期就是那条切线与时间轴的交点,画出来才看得出拟合是否合理。可同时画多孔并按 µmax 排序。Plot growth curves with the fitted model, µmax tangent and lag intercept.",
    examples=[
        "ov.synbio.plot_growth_curves(t, od, fit)",
        "ov.synbio.plot_growth_curves(frame, fits=fit_df)",
    ],
    related=["synbio.fit_growth_curve", "synbio.fit_growth_curves"],
    requires={},
    produces={},
)
def plot_growth_curves(time, od=None, fit=None, *, blank: float = 0.0,
                       max_curves: int = 24, axes=None):
    """Overlay data and fit. ``time`` may also be the wide DataFrame."""
    from ._plot import _mpl
    plt = _mpl()
    import numpy as np

    # wide-frame form
    if od is None and hasattr(time, "columns"):
        data = time
        tvals = data["time_h"].to_numpy(dtype=float)
        wells = [c for c in data.columns if c != "time_h"][:max_curves]
        if axes is None:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
        else:
            ax = axes if not isinstance(axes, (list, tuple)) else axes[0]
            fig = ax.figure
        for w in wells:
            ax.plot(tvals, data[w].to_numpy(dtype=float) - blank, lw=1.0,
                    alpha=0.75, label=w)
        ax.set_xlabel("time (h)")
        ax.set_ylabel("OD (blank-subtracted)")
        ax.set_title(f"{len(wells)} wells")
        if len(wells) <= 12:
            ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        return fig, ax

    t = np.asarray(time, dtype=float)
    y = np.asarray(od, dtype=float) - blank
    if axes is None:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
    else:
        ax = axes if not isinstance(axes, (list, tuple)) else axes[0]
        fig = ax.figure

    ax.plot(t, y, "o", ms=4, color="#377EB8", label="measured")
    if fit is not None:
        dense = np.linspace(float(np.nanmin(t)), float(np.nanmax(t)), 300)
        ax.plot(dense, fit.predict(dense), "-", color="#E41A1C", lw=1.6,
                label=f"{fit.model} (R²={fit.r_squared:.3f})")
        # The tangent lives on an OD-vs-time axis, so its slope is slope_max
        # (OD/h) — NOT mu_max, which is a *specific* rate in 1/h. Using mu_max
        # drew a line 5.95x too steep on a real well, and put the inflection at
        # 5.60 h instead of 6.35 h. The line was algebraically consistent with the
        # lag but was not tangent to the fitted curve, so it could not do the job
        # the caption claimed for it.
        if fit.slope_max > 0 and fit.carrying_capacity > 0:
            t_inf = fit.lag + fit.carrying_capacity / (fit.slope_max * math.e)
            y_inf = float(fit.predict(np.array([t_inf]))[0])
            span = np.array([fit.lag, min(t_inf + 2.0, float(np.nanmax(t)))])
            ax.plot(span, y_inf + fit.slope_max * (span - t_inf), "--",
                    color="#4DAF4A", lw=1.2,
                    label=f"tangent: {fit.slope_max:.3f} OD/h "
                          f"(µmax = {fit.mu_max:.3f} /h)")
            ax.axvline(fit.lag, ls=":", color="#984EA3", lw=1.1,
                       label=f"lag = {fit.lag:.2f} h")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("OD (blank-subtracted)")
    ax.set_title(fit.label or fit.well or "growth curve" if fit else "growth curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


@register_function(
    aliases=["plot_dose_response", "剂量反应图", "IC50图", "抑制曲线图",
             "plot_hill"],
    category="synthetic_biology",
    description="画剂量反应曲线(对数横轴)与 Hill 拟合,标出 EC50/IC50 与 MIC。Plot a dose-response curve with the Hill fit, EC50 and MIC.",
    examples=["ov.synbio.plot_dose_response(conc, resp, res)"],
    related=["synbio.dose_response"],
    requires={},
    produces={},
)
def plot_dose_response(concentration, response, fit: DoseResponse, ax=None):
    """Semi-log dose-response plot with the fit and its landmarks."""
    from ._plot import _mpl
    plt = _mpl()
    import numpy as np

    c = np.asarray(concentration, dtype=float)
    r = np.asarray(response, dtype=float)
    nz = c > 0

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
    else:
        fig = ax.figure

    ax.semilogx(c[nz], r[nz], "o", ms=5, color="#377EB8", label="measured")
    if (~nz).any():
        ax.axhline(float(np.nanmean(r[~nz])), ls=":", color="#999999", lw=1.0,
                   label="zero-dose control")
    dense = np.logspace(np.log10(c[nz].min() / 3), np.log10(c[nz].max() * 3), 300)
    ax.semilogx(dense, fit.predict(dense), "-", color="#E41A1C", lw=1.6,
                label=f"Hill (R²={fit.r_squared:.3f})")
    label = "IC50" if fit.inhibitory else "EC50"
    ax.axvline(fit.ec50, ls="--", color="#4DAF4A", lw=1.2,
               label=f"{label} = {fit.ec50:.3g}")
    if fit.mic is not None:
        ax.axvline(fit.mic, ls="--", color="#984EA3", lw=1.2,
                   label=f"MIC = {fit.mic:.3g}")
    ax.set_xlabel("concentration")
    ax.set_ylabel("response")
    ax.set_title(f"hill slope {fit.hill_slope:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


__all__ = [
    "fit_growth_curve", "GrowthFit", "GROWTH_MODELS", "compare_growth_models",
    "fetch_growth_dataset",
    "fit_growth_curves", "read_plate_reader",
    "dose_response", "DoseResponse",
    "compare_growth_to_model",
    "plot_growth_curves", "plot_dose_response",
]


_GROWTHDATA_URL = (
    "https://raw.githubusercontent.com/sprouffske/growthcurver/master/"
    "data/growthdata.rda"
)


@register_function(
    aliases=["fetch_growth_dataset", "示例生长数据", "真实生长曲线",
             "growthdata", "酶标仪示例数据"],
    category="synthetic_biology",
    description="下载一份**真实的**96 孔酶标仪生长数据(Growthcurver 随包发布的 growthdata:145 个时间点 × 96 孔,0–24 小时),用作 fit_growth_curves 的输入。**这份数据里没有培养基空白孔** —— 96 个孔里 95 个都有明显生长(6–13 倍),所以要么不扣空白,要么用 t=0 的读数作基线;把某一列当成空白扣掉会把数据毁掉。缓存到本地,只下载一次。Fetch a real 96-well plate-reader growth dataset.",
    examples=[
        "od = ov.synbio.fetch_growth_dataset()",
        "fits = ov.synbio.fit_growth_curves(od)                    # no blank wells in this set",
        "fits = ov.synbio.fit_growth_curves(od, blank=float(od.iloc[0, 1:].mean()))",
    ],
    related=["synbio.fit_growth_curves", "synbio.read_plate_reader",
             "synbio.compare_growth_to_model"],
    requires={},
    produces={},
)
def fetch_growth_dataset(cache_dir: Optional[str] = None,
                         timeout: float = 60.0) -> "pd.DataFrame":
    """A real 96-well OD time course, as ``time_h`` plus one column per well.

    Downloaded once and cached. This is measured plate-reader data rather than a
    curve generated from a model — which matters for a demonstration, since a
    fitter shown only on data produced by its own equations has not been shown to
    work on anything.

    **There are no blank wells in this set.** 95 of the 96 wells grow 6–13 fold,
    including all of column 12, so nominating a column as the blank and
    subtracting it removes real signal — doing that dropped the median R² to
    0.39. Fit it as-is, or pass ``blank=`` the mean t=0 reading (~0.05).
    """
    import os

    root = cache_dir or os.environ.get(
        "OMICOS_SYNBIO_GEM_DIR",
        os.path.join(os.path.expanduser("~"), ".omicverse", "synbio_gems"))
    os.makedirs(root, exist_ok=True)
    local = os.path.join(root, "growthdata.rda")

    if not os.path.exists(local):
        import urllib.request
        try:
            urllib.request.urlretrieve(_GROWTHDATA_URL, local)
        except Exception as exc:
            raise RuntimeError(
                f"无法下载示例生长数据({exc})。可手动取 {_GROWTHDATA_URL} "
                f"放到 {local},或用 ov.synbio.read_plate_reader 读自己的数据。"
            ) from exc

    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError(
            "这份数据集是 R 的 .rda 格式,读取需要 pyreadr"
            "(pip install pyreadr)。或用 ov.synbio.read_plate_reader "
            "读自己导出的 CSV。") from exc

    frame = pyreadr.read_r(local)["growthdata"]
    frame = frame.rename(columns={frame.columns[0]: "time_h"})
    return frame
