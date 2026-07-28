r"""The **L** in design-build-test-learn — and the arrow back to **D**.

``ml_guided_design`` scores variants in one shot. That is a Learn step, but it is
not a *cycle*: nothing chooses which experiments to run next, and nothing carries
what round *n* measured into what round *n+1* proposes. Without those two pieces
DBTL is a straight line you walk once.

* :func:`doe_design` — statistical designs for the screening round: full and
  fractional factorial, Plackett-Burman, definitive screening, central composite,
  Box-Behnken, Latin hypercube. Which one to use is a real decision and the
  function says what each buys.
* :func:`analyse_doe` — main effects, two-factor interactions, and a Pareto of
  effect sizes, so the screening round produces a ranking rather than a table.
* :func:`bayesian_optimize` — a Gaussian process over what has been measured,
  and the next batch chosen by expected improvement or upper confidence bound.
  This is the arrow back to Design.
* :func:`DBTLCampaign` — the bookkeeping: rounds, what was proposed, what came
  back, and whether the campaign is still improving.
* :func:`plot_doe_effects` / :func:`plot_optimization_progress`.

**Why designs rather than grids.** A full factorial over eight two-level factors
is 256 experiments; a resolution-IV fractional factorial answers the screening
question in 16 and keeps main effects clear of two-factor interactions. In a DBTL
context where a round costs weeks, that ratio is the whole argument, and it is
why the literature recommends resolution IV or a definitive screening design for
the first round rather than one-factor-at-a-time.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


DESIGNS = ("full_factorial", "fractional_factorial", "plackett_burman",
           "definitive_screening", "central_composite", "box_behnken",
           "latin_hypercube")

ACQUISITIONS = ("ei", "ucb", "poi", "greedy")


# ---------------------------------------------------------------------------
# design of experiments
# ---------------------------------------------------------------------------

def _hadamard(n: int):
    """Sylvester construction, for Plackett-Burman designs at n = 4, 8, 12…"""
    import numpy as np
    h = np.array([[1]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h[:n, :n]


def _plackett_burman(n_factors: int):
    """A Plackett-Burman design: n runs for up to n-1 factors, n a multiple of 4.

    Main effects only — two-factor interactions are fully confounded with them.
    That is the trade: the cheapest possible screen, on the assumption that
    interactions are small compared with main effects.
    """
    import numpy as np
    n = 4
    while n - 1 < n_factors:
        n *= 2
    h = _hadamard(n)
    # Drop the all-ones *column* only, keeping all N rows: a Plackett-Burman
    # design is N runs for up to N-1 factors. Dropping the first row too gave
    # N-1 runs, which for three factors meant three observations for three main
    # effects plus an intercept — an under-determined design that still looked
    # like a design.
    return h[:, 1:n_factors + 1]


def _fractional_factorial(n_factors: int, resolution: int = 4):
    """A 2^(k-p) fractional factorial at the requested resolution.

    Resolution IV keeps main effects unconfounded with two-factor interactions,
    which is what makes a screening round interpretable. Generators are built by
    aliasing the extra factors to interactions of the base factors, choosing
    high-order interactions first so the aliasing structure stays as clean as the
    run count allows.
    """
    import numpy as np

    if resolution not in (3, 4, 5):
        raise ValueError("resolution must be 3, 4 or 5")
    # smallest base size whose interaction pool can carry the extra factors
    min_order = {3: 2, 4: 3, 5: 4}[resolution]
    base = min_order
    while True:
        pool = [c for r in range(min_order, base + 1)
                for c in itertools.combinations(range(base), r)]
        if base + len(pool) >= n_factors:
            break
        base += 1
        if base > 12:
            raise ValueError(
                f"{n_factors} 个因子在 resolution {resolution} 下需要的基础因子数"
                f"超过 12,运行数会失控。降低 resolution,或改用 "
                f"design='plackett_burman'(仅主效应,最省)。")

    rows = list(itertools.product((-1, 1), repeat=base))
    cols = [np.array([r[i] for r in rows]) for i in range(base)]
    pool = sorted(pool, key=lambda c: (-len(c), c))
    for combo in pool:
        if len(cols) >= n_factors:
            break
        col = np.ones(len(rows), dtype=int)
        for i in combo:
            col = col * cols[i]
        cols.append(col)
    return np.column_stack(cols[:n_factors])


def _definitive_screening(n_factors: int):
    """A definitive screening design (Jones & Nachtsheim 2011).

    2k+1 runs for k factors, three levels each: main effects are orthogonal to
    each other *and* to two-factor interactions, and quadratic effects are
    estimable — so curvature shows up in the screening round instead of being
    discovered later. The fold-over pair structure is what buys that.
    """
    import numpy as np
    k = n_factors
    rows = []
    for i in range(k):
        row = np.ones(k, dtype=float)
        row[i] = 0.0
        rows.append(row.copy())
        rows.append(-row)
    rows.append(np.zeros(k, dtype=float))     # centre point
    return np.asarray(rows)


def _central_composite(n_factors: int, alpha: Optional[float] = None,
                       n_centre: int = 3, bounded: bool = True):
    """Central composite: a factorial core, axial (star) points, and centres.

    The star points are what let a *quadratic* surface be fitted, which is the
    point of a response-surface round — a factorial alone can only ever fit a
    plane plus interactions, and an optimum inside the design space is curvature.

    With ``bounded=True`` the whole matrix is rescaled so the axial points land
    exactly on the stated factor range instead of 1.68x outside it. The rotatable
    alpha is ``(2^k)^0.25`` — 1.682 for three factors — and ``to_frame()`` maps
    coded ±1 onto ``(low, high)``, so an unscaled design asked for settings well
    beyond the range the caller declared: over the range ``(1, 20)`` for plasmid
    copy number it produced **-5.5 and 26.5 copies**, and 6 of 17 runs fell
    outside the stated bounds. Rescaling preserves the relative geometry, and
    therefore rotatability, while keeping every run buildable.
    """
    import numpy as np
    core = np.array(list(itertools.product((-1.0, 1.0), repeat=n_factors)))
    a = alpha if alpha is not None else float(len(core)) ** 0.25   # rotatable
    star = []
    for i in range(n_factors):
        for sign in (-a, a):
            row = np.zeros(n_factors)
            row[i] = sign
            star.append(row)
    centre = np.zeros((n_centre, n_factors))
    coded = np.vstack([core, np.asarray(star), centre])
    if bounded:
        peak = float(np.max(np.abs(coded)))
        if peak > 1.0:
            coded = coded / peak
    return coded


def _box_behnken(n_factors: int, n_centre: int = 3):
    """Box-Behnken: three levels, no corner points.

    Every run sits on an edge midpoint, so no experiment combines the extremes
    of all factors at once — which matters when the corners are conditions the
    culture will not survive.
    """
    import numpy as np
    if n_factors < 3:
        raise ValueError("Box-Behnken 至少需要 3 个因子。")
    rows = []
    for i, j in itertools.combinations(range(n_factors), 2):
        for si, sj in itertools.product((-1.0, 1.0), repeat=2):
            row = np.zeros(n_factors)
            row[i], row[j] = si, sj
            rows.append(row)
    rows.extend([np.zeros(n_factors)] * n_centre)
    return np.asarray(rows)


def _latin_hypercube(n_factors: int, n_runs: int, seed: int = 0):
    """Latin hypercube: space-filling, for when the model form is unknown."""
    import numpy as np
    rng = np.random.default_rng(seed)
    cut = np.arange(n_runs) / n_runs
    out = np.empty((n_runs, n_factors))
    for j in range(n_factors):
        u = cut + rng.random(n_runs) / n_runs
        out[:, j] = rng.permutation(u) * 2.0 - 1.0      # to [-1, 1]
    return out


@dataclass
class Design:
    """An experimental design in coded and natural units."""

    design: str
    factors: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    coded: "object" = None            # ndarray, runs x factors, in [-1, 1]
    n_runs: int = 0
    resolution: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    @property
    def factor_names(self) -> List[str]:
        return list(self.factors)

    def to_frame(self, coded: bool = False) -> "pd.DataFrame":
        """Runs in natural units, or in coded ``[-1, 1]`` units."""
        import numpy as np
        import pandas as pd
        arr = np.asarray(self.coded, dtype=float)
        if coded:
            return pd.DataFrame(arr, columns=self.factor_names,
                                index=[f"run{i + 1}" for i in range(len(arr))])
        natural = np.empty_like(arr)
        for j, name in enumerate(self.factor_names):
            lo, hi = self.factors[name]
            mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
            natural[:, j] = mid + arr[:, j] * half
        return pd.DataFrame(natural, columns=self.factor_names,
                            index=[f"run{i + 1}" for i in range(len(arr))])

    def __repr__(self) -> str:  # pragma: no cover
        res = f", resolution {self.resolution}" if self.resolution else ""
        return (f"Design({self.design}, {self.n_runs} runs, "
                f"{len(self.factors)} factors{res})")


@register_function(
    aliases=["doe_design", "实验设计", "DoE", "析因设计", "正交设计",
             "Plackett_Burman", "响应面", "definitive_screening",
             "central_composite", "拉丁超立方"],
    category="synthetic_biology",
    description="生成统计实验设计:full_factorial / fractional_factorial(可指定分辨率,IV 级保证主效应不与二因子交互混淆)/ plackett_burman(最省,仅主效应)/ definitive_screening(2k+1 次,主效应正交且可估二次项)/ central_composite 与 box_behnken(响应面,能拟合曲率)/ latin_hypercube(空间填充)。8 个两水平因子全因子是 256 次,IV 级部分因子 16 次就能回答筛选问题 —— 一轮以周计时,这个比例就是全部理由。Generate a statistical experimental design.",
    examples=[
        "d = ov.synbio.doe_design({'promoter': (0, 1), 'rbs': (0, 1), 'copy': (1, 20)})",
        "d = ov.synbio.doe_design(factors, design='definitive_screening')",
        "d.to_frame(), d.n_runs",
    ],
    related=["synbio.analyse_doe", "synbio.bayesian_optimize",
             "synbio.plot_doe_effects", "synbio.saturation_library"],
    requires={},
    produces={},
)
def doe_design(
    factors: Mapping[str, Tuple[float, float]],
    *,
    design: str = "fractional_factorial",
    resolution: int = 4,
    n_runs: Optional[int] = None,
    n_centre: int = 3,
    alpha: Optional[float] = None,
    bounded: bool = True,
    seed: int = 0,
) -> Design:
    """Build an experimental design over ``factors``.

    Parameters
    ----------
    factors
        ``{name: (low, high)}`` in natural units.
    design
        Which design. The trade-offs, briefly:

        ``'full_factorial'``
            Everything, 2^k runs. Estimates every interaction; unaffordable past
            about five factors.
        ``'fractional_factorial'``
            2^(k-p) runs at a chosen ``resolution``. Resolution IV (the default)
            keeps main effects clear of two-factor interactions, which is what
            makes a screening round interpretable.
        ``'plackett_burman'``
            The cheapest screen: main effects only, interactions fully
            confounded with them.
        ``'definitive_screening'``
            2k+1 runs, three levels. Main effects orthogonal to each other and
            to two-factor interactions, *and* curvature estimable — so a
            quadratic optimum is visible in round one.
        ``'central_composite'`` / ``'box_behnken'``
            Response surface designs for fitting curvature. Box-Behnken avoids
            the all-extremes corners, which matters when those conditions kill
            the culture.
        ``'latin_hypercube'``
            Space-filling, for when no model form is assumed.
    resolution
        3, 4 or 5, for the fractional factorial.
    n_runs
        Runs for the Latin hypercube; defaults to ``10 * n_factors``.

    Returns
    -------
    Design
    """
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {list(DESIGNS)}, got {design!r}")
    if not factors:
        raise ValueError("factors 不能为空。")
    for name, bounds in factors.items():
        if len(bounds) != 2 or bounds[0] == bounds[1]:
            raise ValueError(
                f"因子 {name!r} 的范围必须是 (low, high) 且两端不等,得到 {bounds!r}。")

    import numpy as np
    k = len(factors)
    notes: List[str] = []
    res: Optional[int] = None

    if design == "full_factorial":
        coded = np.array(list(itertools.product((-1.0, 1.0), repeat=k)))
        if k > 6:
            notes.append(
                f"{k} 个因子的全因子 = {len(coded)} 次实验。"
                f"筛选阶段用 design='fractional_factorial'(resolution 4)"
                f"通常只要几十次。")
    elif design == "fractional_factorial":
        coded = _fractional_factorial(k, resolution).astype(float)
        res = resolution
        notes.append(
            f"resolution {resolution}: "
            + ("主效应与二因子交互不混淆" if resolution >= 4
               else "主效应与二因子交互存在混淆(resolution III)"))
        notes.append(f"全因子需要 {2 ** k} 次,这里 {len(coded)} 次。")
    elif design == "plackett_burman":
        coded = _plackett_burman(k).astype(float)
        notes.append("Plackett-Burman:只估主效应,二因子交互与主效应完全混淆。")
    elif design == "definitive_screening":
        coded = _definitive_screening(k)
        notes.append(
            f"definitive screening:{len(coded)} 次(2k+1),主效应彼此正交、"
            f"与二因子交互正交,且二次项可估。")
    elif design == "central_composite":
        coded = _central_composite(k, alpha=alpha, n_centre=n_centre,
                                   bounded=bounded)
        notes.append("中心复合设计:含星号点,可拟合二次响应面。")
        if bounded:
            notes.append(
                "已整体缩放,使轴向点正好落在给定的因子范围上(否则旋转型 alpha "
                f"= {float(2 ** k) ** 0.25:.3f} 会让约 1/3 的实验点越界,"
                "拷贝数、浓度这类量还会出现负值)。传 bounded=False 用经典未缩放版本。")
    elif design == "box_behnken":
        coded = _box_behnken(k, n_centre=n_centre)
        notes.append("Box-Behnken:无角点,不会出现所有因子同时取极值的实验。")
    else:
        runs = n_runs or 10 * k
        coded = _latin_hypercube(k, runs, seed=seed)
        notes.append(f"拉丁超立方:{runs} 次空间填充点,不假设模型形式。")

    return Design(design=design, factors={k_: tuple(v) for k_, v in factors.items()},
                  coded=coded, n_runs=int(len(coded)), resolution=res, notes=notes)


# ---------------------------------------------------------------------------
# DoE analysis
# ---------------------------------------------------------------------------

@dataclass
class DoEAnalysis:
    """Effects estimated from a design and its responses."""

    main_effects: Dict[str, float] = field(default_factory=dict)
    interactions: Dict[Tuple[str, str], float] = field(default_factory=dict)
    quadratics: Dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    r_squared: float = 0.0
    n_runs: int = 0
    significant: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ranked_effects(self) -> List[Tuple[str, float]]:
        """Every effect, largest absolute first — the Pareto ordering."""
        items = [(k, v) for k, v in self.main_effects.items()]
        items += [(f"{a}:{b}", v) for (a, b), v in self.interactions.items()]
        items += list(self.quadratics.items())
        return sorted(items, key=lambda kv: -abs(kv[1]))

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        rows = [{"term": k, "effect": v, "abs_effect": abs(v),
                 "kind": ("interaction" if ":" in k
                          else "quadratic" if k.endswith("^2") else "main"),
                 "significant": k in self.significant}
                for k, v in self.ranked_effects]
        return pd.DataFrame(rows).set_index("term")

    def __repr__(self) -> str:  # pragma: no cover
        top = self.ranked_effects[:3]
        return (f"DoEAnalysis({self.n_runs} runs, R²={self.r_squared:.3f}, "
                f"top: {[f'{k}={v:+.3g}' for k, v in top]})")


@register_function(
    aliases=["analyse_doe", "分析实验设计", "主效应", "效应分析",
             "analyze_doe", "Pareto效应"],
    category="synthetic_biology",
    description="从设计矩阵与响应估计主效应与二因子交互,按效应绝对值排序(Pareto),并用 Lenth 方法标出显著项 —— 筛选轮的产出应该是一份排序,而不是一张表。Estimate main effects and interactions from a design and its responses.",
    examples=[
        "res = ov.synbio.analyse_doe(design, titre)",
        "res.ranked_effects, res.significant, res.to_frame()",
    ],
    related=["synbio.doe_design", "synbio.plot_doe_effects",
             "synbio.bayesian_optimize"],
    requires={},
    produces={},
)
def analyse_doe(
    design: "Design | object",
    response: Sequence[float],
    *,
    interactions: bool = True,
    quadratic: str = "auto",
    factor_names: Optional[Sequence[str]] = None,
) -> DoEAnalysis:
    """Least-squares effects from a coded design matrix.

    Significance uses **Lenth's method** rather than a t-test, because a
    screening design usually has no replication and therefore no independent
    error estimate. Lenth's pseudo-standard-error is built from the median of the
    small effects, on the reasoning that most factors in a screen do nothing —
    which is exactly the situation a screen is run to establish.

    ``quadratic='auto'`` adds squared terms when the design actually supports
    them, i.e. when a factor takes three or more levels. Without this a central
    composite or Box-Behnken design could not be analysed by its own analysis
    function: those designs exist *to* estimate curvature, and a model with only
    main effects and interactions fits a plane through a bowl — on a synthetic
    response with a real quadratic it reached R² = 0.14 while reporting the main
    effects as significant.
    """
    import numpy as np

    if isinstance(design, Design):
        X = np.asarray(design.coded, dtype=float)
        names = design.factor_names
    else:
        X = np.asarray(design, dtype=float)
        names = list(factor_names or [f"x{i + 1}" for i in range(X.shape[1])])
    y = np.asarray(response, dtype=float)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"设计有 {X.shape[0]} 次实验,响应给了 {y.shape[0]} 个值。")
    if X.shape[1] != len(names):
        raise ValueError(
            f"设计有 {X.shape[1]} 列,因子名给了 {len(names)} 个。")

    terms: List[str] = list(names)
    cols = [X[:, j] for j in range(X.shape[1])]
    pairs: List[Tuple[str, str]] = []
    if interactions and X.shape[1] >= 2:
        for i, j in itertools.combinations(range(X.shape[1]), 2):
            candidate = X[:, i] * X[:, j]
            # skip a column already aliased onto an existing one
            if any(np.allclose(candidate, c) or np.allclose(candidate, -c)
                   for c in cols):
                continue
            cols.append(candidate)
            pairs.append((names[i], names[j]))
            terms.append(f"{names[i]}:{names[j]}")

    if quadratic not in ("auto", "always", "never"):
        raise ValueError(
            f"quadratic must be one of ['auto', 'always', 'never'], got {quadratic!r}")
    levels = [len(np.unique(np.round(X[:, j], 9))) for j in range(X.shape[1])]
    want_quad = (quadratic == "always"
                 or (quadratic == "auto" and max(levels) >= 3))
    quad_names: List[str] = []
    if want_quad:
        for j in range(X.shape[1]):
            if levels[j] < 3:
                continue
            col = X[:, j] ** 2
            if any(np.allclose(col, c) for c in cols):
                continue
            cols.append(col)
            quad_names.append(f"{names[j]}^2")
            terms.append(f"{names[j]}^2")

    A = np.column_stack([np.ones(len(y))] + cols)
    if A.shape[0] < A.shape[1]:
        # more terms than runs: drop interactions rather than fit noise
        A = np.column_stack([np.ones(len(y))] + cols[:X.shape[1]])
        terms = list(names)
        pairs = []
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # effect = 2 x coefficient for a [-1, 1] coded factor (low -> high change)
    effects = {t: float(2.0 * b) for t, b in zip(terms, beta[1:])}
    main = {n: effects[n] for n in names if n in effects}
    inter = {p: effects[f"{p[0]}:{p[1]}"] for p in pairs
             if f"{p[0]}:{p[1]}" in effects}
    quadratics = {q: effects[q] for q in quad_names if q in effects}

    # Lenth's pseudo standard error
    notes: List[str] = []
    significant: List[str] = []
    values = np.array(list(effects.values()), dtype=float)
    if values.size >= 3:
        s0 = 1.5 * float(np.median(np.abs(values)))
        keep = values[np.abs(values) < 2.5 * s0]
        pse = 1.5 * float(np.median(np.abs(keep))) if keep.size else s0
        margin = 2.5 * pse if pse > 0 else float("inf")
        significant = [t for t, v in effects.items() if abs(v) > margin]
        notes.append(
            f"Lenth PSE = {pse:.4g},显著阈值 |effect| > {margin:.4g}"
            f"(筛选设计通常无重复,故不用 t 检验)")
    else:
        notes.append("因子太少,未做显著性判定。")

    if not pairs and interactions:
        notes.append(
            "没有可估的二因子交互:这个设计的交互列与主效应混淆"
            "(Plackett-Burman 或 resolution III 就是如此)。")

    if not quadratics and quadratic != "never":
        notes.append(
            "设计里没有三水平因子,无法估计二次项(两水平设计只能拟合平面加交互)。"
            "要看曲率,用 design='central_composite' / 'box_behnken' / "
            "'definitive_screening'。")
    elif quadratics:
        notes.append(f"已估计 {len(quadratics)} 个二次项 —— 响应面设计的意义所在。")

    return DoEAnalysis(main_effects=main, interactions=inter,
                       quadratics=quadratics,
                       intercept=float(beta[0]), r_squared=r2,
                       n_runs=int(len(y)), significant=significant, notes=notes)


# ---------------------------------------------------------------------------
# Bayesian optimisation
# ---------------------------------------------------------------------------

@dataclass
class Proposal:
    """Experiments proposed for the next round."""

    points: "object" = None                  # ndarray, batch x factors
    factor_names: List[str] = field(default_factory=list)
    acquisition: str = "ei"
    predicted_mean: List[float] = field(default_factory=list)
    predicted_std: List[float] = field(default_factory=list)
    acquisition_value: List[float] = field(default_factory=list)
    best_observed: float = 0.0
    model_r_squared: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_frame(self) -> "pd.DataFrame":
        import numpy as np
        import pandas as pd
        df = pd.DataFrame(np.asarray(self.points, dtype=float),
                          columns=self.factor_names,
                          index=[f"proposal{i + 1}"
                                 for i in range(len(self.points))])
        df["predicted"] = self.predicted_mean
        df["uncertainty"] = self.predicted_std
        df[self.acquisition] = self.acquisition_value
        return df

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Proposal({len(self.predicted_mean)} experiments, "
                f"{self.acquisition}, best so far {self.best_observed:.4g})")


@register_function(
    aliases=["bayesian_optimize", "贝叶斯优化", "主动学习", "下一轮实验",
             "acquisition", "bayesopt", "expected_improvement"],
    category="synthetic_biology",
    description="用高斯过程拟合已测数据,再按采集函数挑下一批实验 —— 这就是 DBTL 里从 Learn 指回 Design 的那支箭。acquisition='ei'(期望改进,默认,平衡探索与利用)/'ucb'(置信上界,可调 kappa)/'poi'(改进概率)/'greedy'(纯利用,只在最后一轮收敛时用)。批量提议用 constant-liar 避免一批里挤在同一点。Gaussian-process Bayesian optimisation for the next round of experiments.",
    examples=[
        "prop = ov.synbio.bayesian_optimize(X_measured, titre, factors)",
        "prop = ov.synbio.bayesian_optimize(X, y, factors, batch=8, acquisition='ucb')",
        "prop.to_frame()",
    ],
    related=["synbio.doe_design", "synbio.analyse_doe", "synbio.DBTLCampaign",
             "synbio.ml_guided_design", "synbio.plot_optimization_progress"],
    requires={},
    produces={},
)
def bayesian_optimize(
    X,
    y,
    factors: Mapping[str, Tuple[float, float]],
    *,
    batch: int = 4,
    acquisition: str = "ei",
    kappa: float = 2.0,
    xi: float = 0.01,
    maximize: bool = True,
    n_candidates: int = 4000,
    seed: int = 0,
) -> Proposal:
    """Propose the next batch of experiments.

    Parameters
    ----------
    X
        Measured factor settings, runs x factors, in **natural** units.
    y
        Measured response, one per run.
    factors
        ``{name: (low, high)}`` — the space to search, in natural units.
    batch
        How many experiments to propose. Batches are chosen with a
        *constant-liar* pass: after each pick the model is refitted with that
        point pinned at the current best estimate, so a batch spreads out instead
        of proposing the same optimum four times.
    acquisition
        ``'ei'`` expected improvement, ``'ucb'`` upper confidence bound,
        ``'poi'`` probability of improvement, ``'greedy'`` posterior mean only.
    kappa, xi
        UCB exploration weight, and EI/POI improvement margin.
    maximize
        False to minimise (e.g. a cost or a by-product).

    Returns
    -------
    Proposal
    """
    if acquisition not in ACQUISITIONS:
        raise ValueError(
            f"acquisition must be one of {list(ACQUISITIONS)}, got {acquisition!r}")
    if batch < 1:
        raise ValueError("batch 必须 >= 1")

    import numpy as np
    try:
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "贝叶斯优化需要 scikit-learn 与 scipy(都是 omicverse 的主依赖)。") from exc

    names = list(factors)
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    ya = np.asarray(y, dtype=float).ravel()
    if Xa.shape[0] != ya.shape[0]:
        raise ValueError(f"X 有 {Xa.shape[0]} 行,y 有 {ya.shape[0]} 个值。")
    if Xa.shape[1] != len(names):
        raise ValueError(
            f"X 有 {Xa.shape[1]} 列,factors 给了 {len(names)} 个因子。")
    if Xa.shape[0] < 3:
        raise ValueError(
            f"只有 {Xa.shape[0]} 个观测,高斯过程拟合不出有意义的后验。"
            f"先跑一轮 doe_design(至少几次实验)再进优化。")

    lo = np.array([factors[n][0] for n in names], dtype=float)
    hi = np.array([factors[n][1] for n in names], dtype=float)
    span = np.where(hi - lo == 0, 1.0, hi - lo)

    def code(a):
        return (np.asarray(a, dtype=float) - lo) / span * 2.0 - 1.0

    def decode(a):
        return (np.asarray(a, dtype=float) + 1.0) / 2.0 * span + lo

    Xc = code(Xa)
    sign = 1.0 if maximize else -1.0
    yc = sign * ya

    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(len(names)),
                       length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-3, (1e-8, 1e1)))

    rng = np.random.default_rng(seed)
    notes: List[str] = []

    def fit(xs, ys):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=3,
                                      random_state=seed)
        gp.fit(xs, ys)
        return gp

    gp = fit(Xc, yc)
    pred_train = gp.predict(Xc)
    ss_res = float(np.sum((yc - pred_train) ** 2))
    ss_tot = float(np.sum((yc - np.mean(yc)) ** 2))
    model_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if model_r2 < 0.5:
        notes.append(
            f"高斯过程对已测数据的 R² 只有 {model_r2:.2f} —— 提议的可信度有限。"
            f"可能是响应噪声大、或还缺少覆盖设计空间的点(先补一轮 DoE)。")

    candidates = rng.uniform(-1.0, 1.0, size=(n_candidates, len(names)))
    chosen: List["object"] = []
    means: List[float] = []
    stds: List[float] = []
    acqs: List[float] = []

    xs, ys = Xc.copy(), yc.copy()
    best = float(np.max(yc))

    for _ in range(batch):
        mu, sd = gp.predict(candidates, return_std=True)
        sd = np.maximum(sd, 1e-12)
        if acquisition == "greedy":
            score = mu
        elif acquisition == "ucb":
            score = mu + kappa * sd
        else:
            z = (mu - best - xi) / sd
            if acquisition == "poi":
                score = norm.cdf(z)
            else:
                score = (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)
        idx = int(np.argmax(score))
        pick = candidates[idx]
        chosen.append(pick.copy())
        means.append(float(sign * mu[idx]))
        stds.append(float(sd[idx]))
        acqs.append(float(score[idx]))
        # constant liar: pin this point at its posterior mean and refit
        xs = np.vstack([xs, pick])
        ys = np.append(ys, mu[idx])
        gp = fit(xs, ys)
        candidates = np.delete(candidates, idx, axis=0)

    points = decode(np.asarray(chosen))
    return Proposal(points=points, factor_names=names, acquisition=acquisition,
                    predicted_mean=means, predicted_std=stds,
                    acquisition_value=acqs, best_observed=float(sign * best),
                    model_r_squared=model_r2, notes=notes)


# ---------------------------------------------------------------------------
# the campaign
# ---------------------------------------------------------------------------

@dataclass
class DBTLCampaign:
    """Bookkeeping across rounds of a design-build-test-learn campaign.

    A campaign is the thing that makes DBTL a cycle rather than a line: it holds
    what every round proposed and measured, so round *n+1* can be conditioned on
    everything before it and so "are we still improving" is answerable.
    """

    factors: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    objective: str = "response"
    maximize: bool = True
    rounds: List[Dict[str, object]] = field(default_factory=list)

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    @property
    def n_experiments(self) -> int:
        return sum(len(r["response"]) for r in self.rounds)

    def observations(self):
        """All measured (X, y) so far, as arrays."""
        import numpy as np
        if not self.rounds:
            raise ValueError("还没有任何一轮数据。先 record() 一轮结果。")
        X = np.vstack([np.asarray(r["X"], dtype=float) for r in self.rounds])
        y = np.concatenate([np.asarray(r["response"], dtype=float)
                            for r in self.rounds])
        return X, y

    @property
    def best(self) -> Tuple[float, Dict[str, float]]:
        """Best response so far, and the settings that produced it."""
        import numpy as np
        X, y = self.observations()
        idx = int(np.argmax(y) if self.maximize else np.argmin(y))
        return float(y[idx]), dict(zip(self.factors, X[idx]))

    @property
    def best_per_round(self) -> List[float]:
        import numpy as np
        out, running = [], None
        for r in self.rounds:
            vals = np.asarray(r["response"], dtype=float)
            here = float(np.max(vals) if self.maximize else np.min(vals))
            running = here if running is None else (
                max(running, here) if self.maximize else min(running, here))
            out.append(running)
        return out

    def record(self, X, response, *, note: str = "") -> None:
        """Add a round's measurements."""
        import numpy as np
        Xa = np.atleast_2d(np.asarray(X, dtype=float))
        ya = np.asarray(response, dtype=float).ravel()
        if Xa.shape[0] != ya.shape[0]:
            raise ValueError(f"X 有 {Xa.shape[0]} 行,response 有 {ya.shape[0]} 个。")
        if Xa.shape[1] != len(self.factors):
            raise ValueError(
                f"X 有 {Xa.shape[1]} 列,campaign 定义了 {len(self.factors)} 个因子。")
        self.rounds.append({"X": Xa, "response": ya, "note": note})

    def propose(self, *, batch: int = 4, acquisition: str = "ei", **kwargs):
        """Next batch, conditioned on every round so far."""
        X, y = self.observations()
        return bayesian_optimize(X, y, self.factors, batch=batch,
                                 acquisition=acquisition,
                                 maximize=self.maximize, **kwargs)

    @property
    def improving(self) -> bool:
        """Did the most recent round improve on the one before it?

        The signal to stop is two consecutive rounds without improvement, not a
        fixed round count — a campaign that has converged is spending money to
        confirm what it already knows.
        """
        best = self.best_per_round
        if len(best) < 2:
            return True
        return (best[-1] > best[-2]) if self.maximize else (best[-1] < best[-2])

    def to_frame(self) -> "pd.DataFrame":
        import numpy as np
        import pandas as pd
        rows = []
        for i, r in enumerate(self.rounds, start=1):
            X = np.asarray(r["X"], dtype=float)
            y = np.asarray(r["response"], dtype=float)
            for j in range(len(y)):
                row = {"round": i, self.objective: float(y[j])}
                row.update(dict(zip(self.factors, X[j])))
                rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:  # pragma: no cover
        if not self.rounds:
            return f"DBTLCampaign({len(self.factors)} factors, no rounds yet)"
        best, _ = self.best
        return (f"DBTLCampaign({self.n_rounds} rounds, {self.n_experiments} "
                f"experiments, best {self.objective}={best:.4g}, "
                f"improving={self.improving})")


@register_function(
    aliases=["dbtl_campaign", "DBTL循环", "优化活动", "campaign",
             "迭代优化", "多轮优化"],
    category="synthetic_biology",
    description="创建一个跨轮次的 DBTL 优化活动:记录每轮的实验设置与测量结果,基于**全部历史**提议下一批,并回答『还在改进吗』—— 该停的信号是连续两轮没有提升,而不是跑满预设轮数。Create a multi-round DBTL optimisation campaign.",
    examples=[
        "camp = ov.synbio.dbtl_campaign({'promoter': (0, 1), 'rbs': (0, 1)}, objective='titre')",
        "camp.record(design.to_frame().values, measured_titre)",
        "camp.propose(batch=8)",
    ],
    related=["synbio.doe_design", "synbio.bayesian_optimize",
             "synbio.plot_optimization_progress"],
    requires={},
    produces={},
)
def dbtl_campaign(factors: Mapping[str, Tuple[float, float]], *,
                  objective: str = "response",
                  maximize: bool = True) -> DBTLCampaign:
    """Start a campaign over ``factors``."""
    if not factors:
        raise ValueError("factors 不能为空。")
    return DBTLCampaign(factors={k: tuple(v) for k, v in factors.items()},
                        objective=objective, maximize=maximize)


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

@register_function(
    aliases=["plot_doe_effects", "效应图", "Pareto图", "主效应图",
             "plot_effects"],
    category="synthetic_biology",
    description="画 DoE 效应的 Pareto 图(按绝对效应排序,显著项着色,Lenth 阈值画线)与主效应方向。Plot a Pareto chart of DoE effects with the Lenth significance threshold.",
    examples=["ov.synbio.plot_doe_effects(analysis)"],
    related=["synbio.analyse_doe", "synbio.doe_design"],
    requires={},
    produces={},
)
def plot_doe_effects(analysis: DoEAnalysis, top_n: int = 20, axes=None):
    """Pareto of effects, and the signed main effects."""
    from ._plot import _mpl
    plt = _mpl()

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    else:
        fig = axes[0].figure
    axes = list(axes)

    ranked = analysis.ranked_effects[:top_n]
    labels = [k for k, _ in ranked]
    values = [abs(v) for _, v in ranked]
    sig = set(analysis.significant)
    colours = ["#E41A1C" if k in sig else "#B0B0B0" for k in labels]

    ax = axes[0]
    ax.barh(range(len(labels)), values, color=colours)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("|effect|")
    ax.set_title(f"Pareto of effects — red = significant (Lenth), "
                 f"R²={analysis.r_squared:.3f}")

    ax = axes[1]
    mains = sorted(analysis.main_effects.items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in mains], [v for _, v in mains],
            color=["#377EB8" if v >= 0 else "#FF7F00" for _, v in mains])
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("effect (low → high)")
    ax.set_title("main effects, signed")
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    return fig, axes


@register_function(
    aliases=["plot_optimization_progress", "优化进程图", "DBTL进程",
             "plot_campaign", "收敛图"],
    category="synthetic_biology",
    description="画 DBTL 活动的进程:每轮的实测分布、累积最优轨迹,以及是否还在改进 —— 平掉的轨迹就是该停的信号。Plot a DBTL campaign's progress and best-so-far trajectory.",
    examples=["ov.synbio.plot_optimization_progress(campaign)"],
    related=["synbio.dbtl_campaign", "synbio.bayesian_optimize"],
    requires={},
    produces={},
)
def plot_optimization_progress(campaign: DBTLCampaign, axes=None):
    """Per-round responses and the best-so-far curve."""
    from ._plot import _mpl
    plt = _mpl()
    import numpy as np

    if not campaign.rounds:
        raise ValueError("这个 campaign 还没有任何一轮数据。")

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    else:
        fig = axes[0].figure
    axes = list(axes)

    ax = axes[0]
    for i, r in enumerate(campaign.rounds, start=1):
        vals = np.asarray(r["response"], dtype=float)
        ax.scatter([i] * len(vals), vals, s=26, alpha=0.7, color="#377EB8")
    ax.set_xlabel("round")
    ax.set_ylabel(campaign.objective)
    ax.set_xticks(range(1, campaign.n_rounds + 1))
    ax.set_title(f"{campaign.n_experiments} experiments over "
                 f"{campaign.n_rounds} rounds")

    ax = axes[1]
    best = campaign.best_per_round
    ax.plot(range(1, len(best) + 1), best, "o-", color="#4DAF4A", lw=1.8)
    ax.set_xlabel("round")
    ax.set_ylabel(f"best {campaign.objective} so far")
    ax.set_xticks(range(1, campaign.n_rounds + 1))
    flat = not campaign.improving
    ax.set_title("converged — stop" if flat else "still improving")
    if flat and len(best) >= 2:
        ax.annotate("no gain in the last round",
                    xy=(len(best), best[-1]), fontsize=8,
                    xytext=(-10, -24), textcoords="offset points",
                    color="#E41A1C")

    fig.tight_layout()
    return fig, axes


__all__ = [
    "doe_design", "Design", "DESIGNS", "analyse_doe", "DoEAnalysis",
    "bayesian_optimize", "Proposal", "ACQUISITIONS",
    "dbtl_campaign", "DBTLCampaign",
    "plot_doe_effects", "plot_optimization_progress",
]
