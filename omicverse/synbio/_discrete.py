r"""Designs over **discrete part choices** — the design space synthetic biology
actually has.

:func:`~omicverse.synbio.doe_design` describes a factor as a continuous range,
``{'promoter': (0.05, 1.0)}``. Almost no real construct is built that way. There
is no continuous promoter dial: there is a plate of characterised promoters, and
the design question is *which one*. The same is true of RBSs, terminators,
signal peptides, host strains, insertion loci and codon tables. A continuous
design over a quantity you cannot set is a design you cannot build.

* :func:`orthogonal_array` — a strength-2 orthogonal array, so every pair of
  factors sees every level combination equally often.
* :func:`combinatorial_design` — a design over named part levels, optionally
  blocked.
* :func:`analyse_parts` — per-level effects, per-factor variance explained, and a
  real F-test (a blocked orthogonal array *has* residual degrees of freedom, so
  unlike an unreplicated screen it does not need Lenth's method).
* :func:`propose_combinations` — the next batch, chosen from a finite catalogue of
  buildable combinations rather than from a continuous box.
* :func:`plot_part_effects` / :func:`plot_design_balance`.

**Why the orthogonal array matters here.** Four factors at eight levels is 4,096
combinations. ``OA(64, 5, 8, 2)`` gets every main effect and every pairwise
balance in **64 runs** — one plate. That ratio is the entire argument for
designing rather than sampling, and it is exact rather than asymptotic: in a
strength-2 array each ordered pair of levels between any two columns occurs
exactly once, which is what makes the main-effect estimates mutually
uncorrelated.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


#: Primitive polynomials for the supported prime-power field sizes, as integer
#: coefficient lists from the highest power down, over GF(p).
_PRIMITIVE: Dict[int, Tuple[int, int, List[int]]] = {
    4: (2, 2, [1, 1, 1]),          # x^2 + x + 1 over GF(2)
    8: (2, 3, [1, 0, 1, 1]),       # x^3 + x + 1
    9: (3, 2, [1, 0, 1]),          # x^2 + 1 over GF(3)
    16: (2, 4, [1, 0, 0, 1, 1]),   # x^4 + x + 1
    25: (5, 2, [1, 0, 2]),         # x^2 + 2 over GF(5)
    27: (3, 3, [1, 0, 2, 1]),      # x^3 + 2x + 1
    32: (2, 5, [1, 0, 0, 1, 0, 1]),  # x^5 + x^2 + 1
}


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def _gf_tables(q: int):
    """Addition and multiplication tables for GF(q), q a prime or prime power."""
    import numpy as np

    if _is_prime(q):
        idx = np.arange(q)
        add = (idx[:, None] + idx[None, :]) % q
        mul = (idx[:, None] * idx[None, :]) % q
        return add, mul

    if q not in _PRIMITIVE:
        raise ValueError(
            f"orthogonal_array 需要素数或素数幂的水平数,得到 {q}。"
            f"支持:素数,以及 {sorted(_PRIMITIVE)}。"
            f"水平数不是素数幂时用 combinatorial_design(design='balanced') —— "
            f"它保证每个因子的水平均衡,但不保证两因子间严格正交。")

    p, m, poly = _PRIMITIVE[q]

    def to_digits(a: int) -> List[int]:
        out = []
        for _ in range(m):
            out.append(a % p)
            a //= p
        return out[::-1]                      # highest power first

    def from_digits(d: Sequence[int]) -> int:
        val = 0
        for c in d:
            val = val * p + (c % p)
        return val

    def poly_mul(a: int, b: int) -> int:
        da, db = to_digits(a), to_digits(b)
        prod = [0] * (2 * m - 1)
        for i, ca in enumerate(da):
            for j, cb in enumerate(db):
                prod[i + j] = (prod[i + j] + ca * cb) % p
        # reduce modulo the primitive polynomial
        for i in range(len(prod) - m):
            coeff = prod[i]
            if coeff == 0:
                continue
            for j, pc in enumerate(poly):
                prod[i + j] = (prod[i + j] - coeff * pc) % p
        return from_digits(prod[-m:])

    add = np.zeros((q, q), dtype=int)
    mul = np.zeros((q, q), dtype=int)
    for a in range(q):
        da = to_digits(a)
        for b in range(q):
            db = to_digits(b)
            add[a, b] = from_digits([(x + y) % p for x, y in zip(da, db)])
            mul[a, b] = poly_mul(a, b)
    return add, mul


@register_function(
    aliases=["orthogonal_array", "正交表", "正交array", "OA", "田口正交表",
             "Taguchi"],
    category="synthetic_biology",
    description="构造强度 2 的正交表 OA(q², k, q, 2):q 个水平、k ≤ q+1 个因子、q² 次实验,任意两列的每个有序水平组合恰好出现一次 —— 所以主效应估计互不相关。4 个 8 水平因子的全组合是 4096,正交表只要 64 次(一块板)。q 必须是素数或素数幂(2,3,4,5,7,8,9,11,13,16,25,27,32)。Construct a strength-2 orthogonal array.",
    examples=[
        "oa = ov.synbio.orthogonal_array(8, 4)      # 64 runs, 4 factors, 8 levels",
        "oa.shape                                    # (64, 4)",
    ],
    related=["synbio.combinatorial_design", "synbio.doe_design",
             "synbio.analyse_parts"],
    requires={},
    produces={},
)
def orthogonal_array(levels: int, n_factors: int):
    """A strength-2 orthogonal array with ``levels`` levels per factor.

    Uses the Rao-Hamming construction over GF(q): rows are indexed by all pairs
    ``(x, y)`` in GF(q)², and column *a* takes the value ``y + a*x`` for each
    field element ``a``, with one further column taking ``x``. That gives
    ``q + 1`` mutually orthogonal columns in ``q**2`` runs.

    Returns
    -------
    numpy.ndarray
        ``(levels**2, n_factors)`` of integer level indices in ``[0, levels)``.

    Notes
    -----
    Strength 2 means every *pair* of columns is balanced, which is what makes
    main effects estimable without bias from one another. It says nothing about
    three-factor interactions — those are confounded, exactly as in a
    resolution-IV fractional factorial.
    """
    import numpy as np

    q = int(levels)
    if q < 2:
        raise ValueError(f"水平数必须 >= 2,得到 {q}")
    k = int(n_factors)
    if k < 1:
        raise ValueError(f"因子数必须 >= 1,得到 {k}")
    if k > q + 1:
        raise ValueError(
            f"q={q} 的强度 2 正交表最多容纳 q+1 = {q + 1} 个因子,要 {k} 个。"
            f"要么减少因子,要么提高水平数,要么用 "
            f"combinatorial_design(design='balanced')。")

    add, mul = _gf_tables(q)
    rows = []
    for x in range(q):
        for y in range(q):
            row = [add[y, mul[a, x]] for a in range(q)]
            row.append(x)
            rows.append(row[:k])
    return np.asarray(rows, dtype=int)


# ---------------------------------------------------------------------------
# the design
# ---------------------------------------------------------------------------

DISCRETE_DESIGNS = ("orthogonal_array", "full_factorial", "balanced", "random")


@dataclass
class PartDesign:
    """A design over named part levels."""

    design: str
    parts: Dict[str, List[Any]] = field(default_factory=dict)
    runs: List[Dict[str, Any]] = field(default_factory=list)
    block: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def factor_names(self) -> List[str]:
        return [n for n in self.parts if n != self.block]

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def to_frame(self) -> "pd.DataFrame":
        """One row per run, level *labels* rather than codes."""
        import pandas as pd
        frame = pd.DataFrame(self.runs, columns=list(self.parts))
        frame.index = [f"run{i + 1}" for i in range(len(frame))]
        return frame

    def balance(self) -> "pd.DataFrame":
        """How often each level of each factor appears — the balance check."""
        import pandas as pd
        frame = self.to_frame()
        rows = []
        for name in self.parts:
            counts = frame[name].value_counts()
            rows.append({"factor": name, "levels": int(counts.size),
                         "min_count": int(counts.min()),
                         "max_count": int(counts.max()),
                         "balanced": bool(counts.min() == counts.max())})
        return pd.DataFrame(rows).set_index("factor")

    def full_size(self) -> int:
        """How many runs the exhaustive combination would have taken."""
        total = 1
        for levels in self.parts.values():
            total *= len(levels)
        return total

    def __repr__(self) -> str:  # pragma: no cover
        blk = f", block={self.block!r}" if self.block else ""
        return (f"PartDesign({self.design}, {self.n_runs} runs, "
                f"{len(self.factor_names)} factors{blk}, "
                f"full would be {self.full_size()})")


@register_function(
    aliases=["combinatorial_design", "组合设计", "部件设计", "离散设计",
             "part_design", "文库设计"],
    category="synthetic_biology",
    description="在**离散部件水平**上做实验设计 —— 真实的合成生物学设计空间就是这样:没有连续的启动子旋钮,只有一盘表征过的启动子,问题是选哪一个。design='orthogonal_array' 用强度 2 正交表(任意两因子的水平组合均衡出现,主效应估计互不相关);'full_factorial' 穷举;'balanced' 保证每因子水平均衡但不保证两因子正交(水平数不是素数幂时用);'random' 随机抽。block= 指定一个区组因子(如基因组位点、批次、板),它与各因子正交。Design over discrete part choices.",
    examples=[
        "d = ov.synbio.combinatorial_design({'minus35': m35, 'spacer': sp, 'minus10': m10})",
        "d = ov.synbio.combinatorial_design(parts, block='background')",
        "d.to_frame(), d.balance(), d.n_runs, d.full_size()",
    ],
    related=["synbio.orthogonal_array", "synbio.analyse_parts",
             "synbio.propose_combinations", "synbio.fetch_promoter_library",
             "synbio.doe_design"],
    requires={},
    produces={},
)
def combinatorial_design(
    parts: Mapping[str, Sequence[Any]],
    *,
    design: str = "orthogonal_array",
    block: Optional[str] = None,
    n_runs: Optional[int] = None,
    seed: int = 0,
) -> PartDesign:
    """Design over discrete part choices.

    Parameters
    ----------
    parts
        ``{factor: [level, level, ...]}`` — the parts actually available. Level
        labels can be anything hashable; they are carried through untouched so
        the design can be joined straight back onto a part catalogue.
    design
        ``'orthogonal_array'`` (default) needs every factor to have the same
        number of levels and that number to be a prime power, and it accepts at
        most ``q + 1`` factors including the block. ``'balanced'`` is the
        fallback that always works. ``'full_factorial'`` enumerates everything.
    block
        Name of a factor in ``parts`` to treat as a **nuisance block** rather
        than a factor of interest — a genomic locus, a batch, a plate. It is
        still balanced and still orthogonal to the real factors, so its effect
        can be removed instead of inflating the residual.
    n_runs
        For ``'balanced'`` and ``'random'``. Defaults to the smallest multiple of
        the largest level count that is at least ``3 x`` the number of estimated
        parameters.

    Returns
    -------
    PartDesign
    """
    import numpy as np

    if design not in DISCRETE_DESIGNS:
        raise ValueError(
            f"design must be one of {list(DISCRETE_DESIGNS)}, got {design!r}")
    if not parts:
        raise ValueError("parts 不能为空。")
    levels = {name: list(vals) for name, vals in parts.items()}
    for name, vals in levels.items():
        if len(vals) < 2:
            raise ValueError(
                f"因子 {name!r} 只有 {len(vals)} 个水平,无法估计效应。")
        if len(set(map(str, vals))) != len(vals):
            raise ValueError(f"因子 {name!r} 的水平有重复。")
    if block is not None and block not in levels:
        raise ValueError(
            f"block={block!r} 不在 parts 里。可选:{list(levels)}")

    names = list(levels)
    counts = [len(levels[n]) for n in names]
    rng = np.random.default_rng(seed)
    notes: List[str] = []

    if design == "full_factorial":
        codes = np.array(list(itertools.product(*[range(c) for c in counts])),
                         dtype=int)
        notes.append(f"全组合 {len(codes)} 次 —— 每个交互都可估,代价是次数。")
    elif design == "orthogonal_array":
        if len(set(counts)) != 1:
            raise ValueError(
                f"正交表要求所有因子的水平数相同,得到 {dict(zip(names, counts))}。"
                f"混合水平请用 design='balanced',或把水平数不同的因子"
                f"单独固定/作为区组处理。")
        q = counts[0]
        oa = orthogonal_array(q, len(names))
        codes = oa
        notes.append(
            f"强度 2 正交表:{len(codes)} 次实验覆盖 {len(names)} 个 {q} 水平因子"
            f"(全组合要 {q ** len(names)} 次)。任意两因子的每个水平组合"
            f"恰好出现一次,主效应估计互不相关。")
        notes.append("三因子及更高阶交互与主效应混淆 —— 与 resolution IV 部分因子同理。")
    else:
        n_params = 1 + sum(c - 1 for c in counts)
        step = int(np.lcm.reduce(counts))
        want = n_runs or int(step * max(1, -(-3 * n_params // step)))
        if want % step:
            want = int(step * (want // step + 1))
            notes.append(f"次数上调到 {want},使每个因子的水平数整除得开。")
        if design == "balanced":
            cols = []
            for c in counts:
                base = np.tile(np.arange(c), want // c)
                rng.shuffle(base)
                cols.append(base)
            codes = np.column_stack(cols)
            notes.append(
                f"均衡设计:{want} 次,每个因子的每个水平各出现 {want // max(counts)}–"
                f"{want // min(counts)} 次。两因子间只是近似正交 —— "
                f"用 d.balance() 与交叉表自行确认。")
        else:
            codes = np.column_stack(
                [rng.integers(0, c, size=want) for c in counts])
            notes.append(f"随机抽 {want} 次:不保证均衡,只作基线对照。")

    runs = [{name: levels[name][int(code[j])] for j, name in enumerate(names)}
            for code in codes]
    if block is not None:
        notes.append(
            f"{block!r} 作为区组:它与各因子正交,分析时会把它的效应扣掉"
            f"而不是留在残差里。")
    return PartDesign(design=design, parts=levels, runs=runs, block=block,
                      notes=notes)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

@dataclass
class PartEffects:
    """Per-level effects for a discrete design."""

    level_effects: Dict[str, Dict[Any, float]] = field(default_factory=dict)
    variance_explained: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    block_effects: Dict[Any, float] = field(default_factory=dict)
    intercept: float = 0.0
    r_squared: float = 0.0
    n_runs: int = 0
    residual_df: int = 0
    response_name: str = "response"
    notes: List[str] = field(default_factory=list)

    @property
    def ranked_factors(self) -> List[Tuple[str, float]]:
        """Factors by variance explained, largest first."""
        return sorted(self.variance_explained.items(), key=lambda kv: -kv[1])

    def best_levels(self) -> Dict[str, Any]:
        """The level of each factor with the largest positive effect."""
        return {name: max(eff.items(), key=lambda kv: kv[1])[0]
                for name, eff in self.level_effects.items()}

    def predict(self, combination: Mapping[str, Any]) -> float:
        """Additive prediction for one combination of levels."""
        total = self.intercept
        for name, level in combination.items():
            if name in self.level_effects:
                if level not in self.level_effects[name]:
                    raise KeyError(
                        f"因子 {name!r} 没有见过水平 {level!r};"
                        f"已估计的水平:{list(self.level_effects[name])}")
                total += self.level_effects[name][level]
        return float(total)

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        rows = []
        for name, eff in self.level_effects.items():
            for level, value in sorted(eff.items(), key=lambda kv: -kv[1]):
                rows.append({"factor": name, "level": level, "effect": value,
                             "variance_explained": self.variance_explained.get(name),
                             "p_value": self.p_values.get(name)})
        return pd.DataFrame(rows)

    def factor_frame(self) -> "pd.DataFrame":
        """One row per factor: variance explained, spread, p-value."""
        import pandas as pd
        rows = []
        for name, eff in self.level_effects.items():
            values = list(eff.values())
            rows.append({"factor": name,
                         "n_levels": len(values),
                         "variance_explained": self.variance_explained.get(name, 0.0),
                         "effect_spread": max(values) - min(values),
                         "best_level": max(eff.items(), key=lambda kv: kv[1])[0],
                         "p_value": self.p_values.get(name)})
        frame = pd.DataFrame(rows).set_index("factor")
        return frame.sort_values("variance_explained", ascending=False)

    def __repr__(self) -> str:  # pragma: no cover
        top = self.ranked_factors[:3]
        return (f"PartEffects({self.n_runs} runs, R²={self.r_squared:.3f}, "
                f"df_resid={self.residual_df}, "
                f"top: {[f'{k}={v:.3f}' for k, v in top]})")


@register_function(
    aliases=["analyse_parts", "analyze_parts", "部件效应", "水平效应",
             "分析组合设计", "ANOVA"],
    category="synthetic_biology",
    description="从离散部件设计与响应估计**每个水平**的效应(效应编码、和为零)、每个因子解释的方差比例 η²,以及真正的 F 检验 p 值 —— 有区组的正交表有残差自由度,所以不需要 Lenth 那种无重复筛选设计的近似。block= 把区组(位点/批次/板)的效应扣掉而不是留在残差里。返回的 best_levels() 是加性模型下的最优组合,predict() 给任意组合的预测值。Estimate per-level effects for a discrete part design.",
    examples=[
        "eff = ov.synbio.analyse_parts(frame, response, block='background')",
        "eff.factor_frame()          # eta^2 and p per factor, ranked",
        "eff.best_levels()           # the additive optimum",
    ],
    related=["synbio.combinatorial_design", "synbio.propose_combinations",
             "synbio.plot_part_effects", "synbio.analyse_doe"],
    requires={},
    produces={},
)
def analyse_parts(
    design: "PartDesign | object",
    response: Sequence[float],
    *,
    factors: Optional[Sequence[str]] = None,
    block: Optional[str] = None,
    response_name: str = "response",
) -> PartEffects:
    """Effects-coded least squares over discrete factors.

    Parameters
    ----------
    design
        A :class:`PartDesign`, or a DataFrame of level labels (one column per
        factor, one row per run).
    response
        One value per run. **Use a log scale** if the response is multiplicative,
        which expression and titre both are — additive effects on a log response
        are fold-changes, and that is what a part is usually claimed to have.
    factors
        Restrict to these columns. Defaults to every column except ``block``.
    block
        A nuisance column whose effect is estimated and removed, but which is not
        reported as a finding. Taken from the :class:`PartDesign` if not given.

    Returns
    -------
    PartEffects

    Notes
    -----
    Each factor is coded as deviations from the grand mean with the level effects
    summing to zero, so an effect reads as "this level against the average level"
    and the intercept is the grand mean. Variance explained is
    ``SS_factor / SS_total`` computed by dropping that factor's columns and
    taking the increase in residual sum of squares; for a balanced design this
    coincides with the sequential (Type I) decomposition, which is one of the
    practical reasons to balance the design in the first place.
    """
    import numpy as np

    if isinstance(design, PartDesign):
        frame = design.to_frame()
        block = block if block is not None else design.block
    else:
        import pandas as pd
        frame = pd.DataFrame(design).reset_index(drop=True)

    y = np.asarray(response, dtype=float).ravel()
    if len(frame) != len(y):
        raise ValueError(
            f"设计有 {len(frame)} 次实验,响应给了 {len(y)} 个值。")
    if block is not None and block not in frame.columns:
        raise ValueError(
            f"block={block!r} 不是设计里的列。可选:{list(frame.columns)}")

    wanted = list(factors) if factors is not None else [
        c for c in frame.columns if c != block]
    missing = [c for c in wanted if c not in frame.columns]
    if missing:
        raise ValueError(f"设计里没有这些列:{missing}")
    if not wanted:
        raise ValueError("没有可分析的因子。")

    notes: List[str] = []

    def columns_for(name: str):
        """Effects-coded columns for one factor: L-1 columns, sum-to-zero."""
        vals = frame[name].to_numpy()
        levels = list(dict.fromkeys(vals))          # first-seen order, stable
        cols = []
        for level in levels[:-1]:
            col = np.where(vals == level, 1.0, 0.0)
            col = np.where(vals == levels[-1], -1.0, col)
            cols.append(col)
        return levels, (np.column_stack(cols) if cols
                        else np.zeros((len(vals), 0)))

    blocks: Dict[str, Tuple[List[Any], "object"]] = {}
    for name in wanted:
        blocks[name] = columns_for(name)
    block_levels, block_cols = (columns_for(block) if block is not None
                                else ([], np.zeros((len(y), 0))))

    pieces = [np.ones((len(y), 1))]
    spans: Dict[str, Tuple[int, int]] = {}
    for name in wanted:
        _levels, cols = blocks[name]
        start = sum(p.shape[1] for p in pieces)
        pieces.append(cols)
        spans[name] = (start, start + cols.shape[1])
    block_span = None
    if block_cols.shape[1]:
        start = sum(p.shape[1] for p in pieces)
        pieces.append(block_cols)
        block_span = (start, start + block_cols.shape[1])

    A = np.column_stack(pieces)
    n, p = A.shape
    if n <= p:
        raise ValueError(
            f"{n} 次实验估不出 {p} 个参数(截距 + 各因子的 L-1 个水平"
            f"{' + 区组' if block else ''})。"
            f"至少需要 {p + 1} 次,或减少因子/水平。")

    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    df_resid = n - p

    level_effects: Dict[str, Dict[Any, float]] = {}
    for name in wanted:
        levels, _cols = blocks[name]
        lo, hi = spans[name]
        coefs = list(beta[lo:hi])
        effects = {lvl: float(c) for lvl, c in zip(levels[:-1], coefs)}
        effects[levels[-1]] = float(-sum(coefs))     # sum-to-zero constraint
        level_effects[name] = effects

    block_effects: Dict[Any, float] = {}
    if block_span is not None:
        lo, hi = block_span
        coefs = list(beta[lo:hi])
        block_effects = {lvl: float(c) for lvl, c in zip(block_levels[:-1], coefs)}
        block_effects[block_levels[-1]] = float(-sum(coefs))

    # variance explained and F-test, by dropping each factor's columns
    variance: Dict[str, float] = {}
    pvals: Dict[str, float] = {}
    try:
        from scipy.stats import f as f_dist
    except ImportError:  # pragma: no cover
        f_dist = None
        notes.append("没有 scipy,跳过 F 检验的 p 值。")
    for name in wanted:
        lo, hi = spans[name]
        keep = [j for j in range(p) if not (lo <= j < hi)]
        reduced = A[:, keep]
        rbeta, *_ = np.linalg.lstsq(reduced, y, rcond=None)
        ss_reduced = float(np.sum((y - reduced @ rbeta) ** 2))
        ss_factor = ss_reduced - ss_res
        variance[name] = ss_factor / ss_tot if ss_tot > 0 else 0.0
        df_factor = hi - lo
        if f_dist is not None and df_factor > 0 and df_resid > 0 and ss_res > 0:
            f_stat = (ss_factor / df_factor) / (ss_res / df_resid)
            pvals[name] = float(f_dist.sf(f_stat, df_factor, df_resid))

    notes.append(
        f"{n} 次实验,{p} 个参数,残差自由度 {df_resid} —— "
        f"有残差自由度所以能做真正的 F 检验,不必退到 Lenth 近似。")
    if block is not None and block_effects:
        spread = max(block_effects.values()) - min(block_effects.values())
        notes.append(
            f"区组 {block!r} 的水平间跨度 {spread:.3g};已从各因子效应中扣除。")
    # "Nothing mattered this round" has to be judged on the F-test, corrected for
    # the number of factors tested — not on eta-squared. With 64 runs and 22
    # parameters a *pure noise* response gives the top factor eta2 ~ 0.24, so an
    # "eta2 < 0.05" rule almost never fires and is one more check incapable of
    # failing. Three factors also means a naive p < 0.05 fires on noise about 14%
    # of the time, hence the Bonferroni threshold.
    ranked = sorted(variance.items(), key=lambda kv: -kv[1])
    if pvals:
        alpha = 0.05 / max(len(pvals), 1)
        best_p = min(pvals.values())
        notes.append(
            f"显著性阈值 p < {alpha:.4f}(0.05 经 {len(pvals)} 个因子的 Bonferroni "
            f"校正);最小 p = {best_p:.3g}。")
        if best_p >= alpha:
            notes.append(
                f"没有任何因子通过校正后的显著性阈值 —— 这一轮没测出效应,"
                f"别在噪声上排序。解释方差最大的是 {ranked[0][0]!r}(η² = "
                f"{ranked[0][1]:.1%}),但 64 次实验、{p} 个参数时纯噪声也能给出"
                f"这个量级的 η²,所以 η² 本身不足以判断。")

    return PartEffects(level_effects=level_effects, variance_explained=variance,
                       p_values=pvals, block_effects=block_effects,
                       intercept=float(beta[0]), r_squared=r2, n_runs=n,
                       residual_df=int(df_resid), response_name=response_name,
                       notes=notes)


# ---------------------------------------------------------------------------
# discrete proposals
# ---------------------------------------------------------------------------

@dataclass
class CombinationProposal:
    """The next batch, drawn from a finite catalogue."""

    combinations: List[Dict[str, Any]] = field(default_factory=list)
    predicted_mean: List[float] = field(default_factory=list)
    predicted_std: List[float] = field(default_factory=list)
    acquisition_value: List[float] = field(default_factory=list)
    acquisition: str = "ei"
    best_observed: float = 0.0
    model_r_squared: float = 0.0
    n_candidates: int = 0
    notes: List[str] = field(default_factory=list)

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        frame = pd.DataFrame(self.combinations)
        frame["predicted"] = self.predicted_mean
        frame["uncertainty"] = self.predicted_std
        frame[self.acquisition] = self.acquisition_value
        frame.index = [f"proposal{i + 1}" for i in range(len(frame))]
        return frame

    def __repr__(self) -> str:  # pragma: no cover
        return (f"CombinationProposal({len(self.combinations)} from "
                f"{self.n_candidates} candidates, {self.acquisition}, "
                f"best so far {self.best_observed:.4g})")


@register_function(
    aliases=["propose_combinations", "提议组合", "离散贝叶斯优化",
             "下一批组合", "discrete_bo"],
    category="synthetic_biology",
    description="从一个**有限的可建组合目录**里挑下一批实验(离散贝叶斯优化)。连续版 bayesian_optimize 会给出 'promoter=0.672' 这种建不出来的点;这个函数只会提议目录里真实存在的组合。用 one-hot 编码 + 高斯过程,批次用 constant-liar 展开(否则一批四个点会是同一个最优点)。已测过的组合自动排除。Propose the next batch from a finite catalogue of buildable combinations.",
    examples=[
        "prop = ov.synbio.propose_combinations(observed, y, catalogue, batch=8)",
        "prop.to_frame()",
    ],
    related=["synbio.analyse_parts", "synbio.combinatorial_design",
             "synbio.bayesian_optimize", "synbio.fetch_promoter_library"],
    requires={},
    produces={},
)
def propose_combinations(
    observed: "object",
    response: Sequence[float],
    catalogue: "object",
    *,
    batch: int = 4,
    acquisition: str = "ei",
    kappa: float = 2.0,
    xi: float = 0.01,
    maximize: bool = True,
    factors: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> CombinationProposal:
    """Bayesian optimisation restricted to combinations that can be built.

    Parameters
    ----------
    observed
        DataFrame of level labels already measured, one row per run.
    response
        Measured values, one per row of ``observed``.
    catalogue
        DataFrame of every combination that *could* be built — the search space.
        Rows already present in ``observed`` are excluded automatically.
    factors
        Columns to model. Defaults to the columns shared by both frames.

    Returns
    -------
    CombinationProposal

    Notes
    -----
    The continuous :func:`~omicverse.synbio.bayesian_optimize` will happily
    return ``promoter = 0.672``, and there is no such promoter. Restricting the
    acquisition to a catalogue is not a convenience — it is the difference
    between a proposal and a buildable experiment.
    """
    import numpy as np
    import pandas as pd

    if acquisition not in ("ei", "ucb", "poi", "greedy"):
        raise ValueError(
            f"acquisition must be one of ['ei', 'ucb', 'poi', 'greedy'], "
            f"got {acquisition!r}")
    if batch < 1:
        raise ValueError("batch 必须 >= 1")
    try:
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                                      WhiteKernel)
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 scikit-learn 与 scipy。") from exc

    obs = pd.DataFrame(observed).reset_index(drop=True)
    cat = pd.DataFrame(catalogue).reset_index(drop=True)
    y = np.asarray(response, dtype=float).ravel()
    if len(obs) != len(y):
        raise ValueError(f"observed 有 {len(obs)} 行,response 有 {len(y)} 个值。")
    if len(obs) < 3:
        raise ValueError(
            f"只有 {len(obs)} 个观测,高斯过程给不出有意义的后验。先跑一轮设计。")

    cols = list(factors) if factors is not None else [
        c for c in obs.columns if c in cat.columns]
    if not cols:
        raise ValueError(
            f"observed 与 catalogue 没有共同列。observed: {list(obs.columns)};"
            f"catalogue: {list(cat.columns)}")

    notes: List[str] = []
    # exclude what has already been measured
    key = lambda f: f[cols].astype(str).agg("|".join, axis=1)  # noqa: E731
    seen = set(key(obs))
    mask = ~key(cat).isin(seen)
    pool = cat.loc[mask].reset_index(drop=True)
    dropped = int((~mask).sum())
    if dropped:
        notes.append(f"目录里有 {dropped} 个组合已经测过,已排除。")
    if pool.empty:
        raise ValueError(
            "目录里的组合全部测过了,没有可提议的新实验。"
            "扩大 catalogue,或换一个响应目标。")

    # one-hot on the union of levels seen anywhere, so observed and pool align
    everything = pd.concat([obs[cols], pool[cols]], ignore_index=True)
    dummies = pd.get_dummies(everything.astype(str), columns=cols)
    Xo = dummies.iloc[:len(obs)].to_numpy(dtype=float)
    Xp = dummies.iloc[len(obs):].to_numpy(dtype=float)

    sign = 1.0 if maximize else -1.0
    yc = sign * y
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
              + WhiteKernel(1e-3, (1e-8, 1e1)))

    def fit(xs, ys):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=3, random_state=seed)
        gp.fit(xs, ys)
        return gp

    gp = fit(Xo, yc)
    pred_train = gp.predict(Xo)
    ss_res = float(np.sum((yc - pred_train) ** 2))
    ss_tot = float(np.sum((yc - yc.mean()) ** 2))
    model_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if model_r2 < 0.5:
        notes.append(
            f"高斯过程对已测数据的 R² 只有 {model_r2:.2f} —— 提议可信度有限。"
            f"要么响应噪声大,要么设计还没覆盖到足够的水平组合。")

    chosen_idx: List[int] = []
    means: List[float] = []
    stds: List[float] = []
    acqs: List[float] = []
    xs, ys = Xo.copy(), yc.copy()
    best = float(np.max(yc))
    available = np.ones(len(Xp), dtype=bool)

    for _ in range(min(batch, len(Xp))):
        mu, sd = gp.predict(Xp, return_std=True)
        sd = np.maximum(sd, 1e-12)
        if acquisition == "greedy":
            score = mu
        elif acquisition == "ucb":
            score = mu + kappa * sd
        else:
            z = (mu - best - xi) / sd
            score = (norm.cdf(z) if acquisition == "poi"
                     else (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z))
        score = np.where(available, score, -np.inf)
        idx = int(np.argmax(score))
        chosen_idx.append(idx)
        means.append(float(sign * mu[idx]))
        stds.append(float(sd[idx]))
        acqs.append(float(score[idx]))
        available[idx] = False
        # constant liar so the batch spreads instead of repeating one optimum
        xs = np.vstack([xs, Xp[idx]])
        ys = np.append(ys, mu[idx])
        gp = fit(xs, ys)

    if batch > len(Xp):
        notes.append(f"目录里只剩 {len(Xp)} 个未测组合,少于 batch={batch}。")

    combos = [pool.iloc[i][cols].to_dict() for i in chosen_idx]
    return CombinationProposal(
        combinations=combos, predicted_mean=means, predicted_std=stds,
        acquisition_value=acqs, acquisition=acquisition,
        best_observed=float(sign * best), model_r_squared=model_r2,
        n_candidates=int(len(pool)), notes=notes)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

@register_function(
    aliases=["plot_part_effects", "画部件效应", "水平效应图"],
    category="synthetic_biology",
    description="画离散设计的效应:左边是每个因子解释的方差比例(η²,带 F 检验显著性标注),右边是选定因子每个水平的效应值 —— 直接读出该选哪个部件。Plot per-factor variance explained and per-level effects.",
    examples=["ov.synbio.plot_part_effects(eff)",
              "ov.synbio.plot_part_effects(eff, factor='minus35')"],
    related=["synbio.analyse_parts", "synbio.plot_design_balance"],
    requires={}, produces={},
)
def plot_part_effects(effects: PartEffects, factor: Optional[str] = None,
                      axes=None):
    """Variance explained per factor, and per-level effects for one factor."""
    import matplotlib.pyplot as plt
    import numpy as np

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    else:
        fig = np.ravel(axes)[0].figure
    ax1, ax2 = np.ravel(axes)[:2]

    ranked = effects.ranked_factors
    names = [n for n, _ in ranked]
    values = [v for _, v in ranked]
    colours = ["#c0392b" if effects.p_values.get(n, 1.0) < 0.05 else "#95a5a6"
               for n in names]
    ax1.barh(range(len(names)), values, color=colours)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    ax1.invert_yaxis()
    ax1.set_xlabel("variance explained (η²)")
    ax1.set_title("red = F-test p < 0.05")
    for i, n in enumerate(names):
        p = effects.p_values.get(n)
        if p is not None:
            ax1.text(values[i], i, f"  p={p:.1e}", va="center", fontsize=7)

    pick = factor or (names[0] if names else None)
    if pick is not None and pick in effects.level_effects:
        eff = effects.level_effects[pick]
        items = sorted(eff.items(), key=lambda kv: kv[1])
        labels = [str(k)[:22] for k, _ in items]
        vals = [v for _, v in items]
        ax2.barh(range(len(vals)), vals,
                 color=["#2980b9" if v < 0 else "#27ae60" for v in vals])
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=7)
        ax2.axvline(0, color="black", lw=0.8)
        ax2.set_xlabel(f"effect on {effects.response_name} (vs mean level)")
        ax2.set_title(f"{pick}: pick the top bar")
    fig.tight_layout()
    return fig, axes


@register_function(
    aliases=["plot_design_balance", "画设计均衡性", "设计平衡检查"],
    category="synthetic_biology",
    description="检查设计的均衡性与正交性:左边是每个因子各水平出现的次数(均衡设计应该齐平),右边是任意两个因子的水平交叉表热图(强度 2 正交表应该处处相等)。设计做坏了这里一眼看得出来 —— 位置效应与因子混淆正是这样溜过去的。Check the balance and pairwise orthogonality of a design.",
    examples=["ov.synbio.plot_design_balance(design)",
              "ov.synbio.plot_design_balance(design, pair=('minus35', 'minus10'))"],
    related=["synbio.combinatorial_design", "synbio.plot_part_effects"],
    requires={}, produces={},
)
def plot_design_balance(design: "PartDesign | object",
                        pair: Optional[Tuple[str, str]] = None, axes=None):
    """Level counts per factor, and the pairwise crosstab for one pair."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    frame = design.to_frame() if isinstance(design, PartDesign) else pd.DataFrame(design)
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    else:
        fig = np.ravel(axes)[0].figure
    ax1, ax2 = np.ravel(axes)[:2]

    offset = 0.0
    ticks, labels = [], []
    for name in frame.columns:
        counts = frame[name].value_counts().sort_index()
        xs = np.arange(len(counts)) + offset
        balanced = counts.min() == counts.max()
        ax1.bar(xs, counts.to_numpy(),
                color="#27ae60" if balanced else "#e67e22")
        ticks.append(float(xs.mean()))
        labels.append(f"{name}\n{'balanced' if balanced else 'UNBALANCED'}")
        offset += len(counts) + 1.0
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("runs per level")
    ax1.set_title("level balance (green = equal counts)")

    cols = list(frame.columns)
    a, b = pair if pair else (cols[0], cols[1] if len(cols) > 1 else cols[0])
    table = pd.crosstab(frame[a], frame[b])
    im = ax2.imshow(table.to_numpy(), cmap="viridis", aspect="auto")
    ax2.set_xlabel(b)
    ax2.set_ylabel(a)
    uniform = table.to_numpy().min() == table.to_numpy().max()
    ax2.set_title(f"{a} x {b}: "
                  + ("every pair equally often (orthogonal)" if uniform
                     else "uneven — pairs confounded"), fontsize=9)
    ax2.set_xticks(range(table.shape[1]))
    ax2.set_xticklabels([str(c)[:10] for c in table.columns], rotation=90, fontsize=6)
    ax2.set_yticks(range(table.shape[0]))
    ax2.set_yticklabels([str(i)[:10] for i in table.index], fontsize=6)
    fig.colorbar(im, ax=ax2, label="runs")
    fig.tight_layout()
    return fig, axes


@register_function(
    aliases=["lookup_combination", "查组合", "回查实测值", "combination_lookup"],
    category="synthetic_biology",
    description="在一份已测数据里查出某个具体组合的行 —— 闭环的收尾动作:模型推荐了一个组合,那它实测是多少?返回匹配的行(通常一行),找不到时返回空表而不是报错,因为'这个组合没被建出来'本身就是一个结果。Look up the measured rows for one combination of levels.",
    examples=[
        "row = ov.synbio.lookup_combination(measured, effects.best_levels())",
        "row = ov.synbio.lookup_combination(measured, proposal.combinations[0])",
    ],
    related=["synbio.analyse_parts", "synbio.propose_combinations",
             "synbio.compare_part_effects"],
    requires={}, produces={},
)
def lookup_combination(frame: "object", combination: Mapping[str, Any]
                       ) -> "pd.DataFrame":
    """Rows of ``frame`` matching every level in ``combination``.

    An empty result means that combination was never built — which is a finding,
    not an error, and is why this does not raise.
    """
    import pandas as pd

    table = pd.DataFrame(frame)
    missing = [name for name in combination if name not in table.columns]
    if missing:
        raise KeyError(
            f"这些因子不在表里:{missing}。可选列:{list(table.columns)}")
    mask = pd.Series(True, index=table.index)
    for name, level in combination.items():
        mask &= table[name].astype(str) == str(level)
    return table.loc[mask]


@register_function(
    aliases=["compare_part_effects", "对比效应", "效应一致性", "验证设计",
             "effects_agreement"],
    category="synthetic_biology",
    description="把一份设计估计出的每水平效应与另一份(通常是全组合实测的真值)对比:逐因子给出效应的 Pearson 相关、解释方差之差、最佳水平是否一致。这是'这个设计到底有没有奏效'的直接检验 —— 只有在真值已知时才做得到,而那正是应该验证方法本身的场合。Compare per-level effects from two analyses, factor by factor.",
    examples=[
        "ov.synbio.compare_part_effects(from_64_runs, from_all_512)",
    ],
    related=["synbio.analyse_parts", "synbio.plot_part_effects"],
    requires={}, produces={},
)
def compare_part_effects(estimate: PartEffects, reference: PartEffects
                         ) -> "pd.DataFrame":
    """Per-factor agreement between an estimate and a reference analysis."""
    import numpy as np
    import pandas as pd

    rows = []
    shared = [f for f in estimate.level_effects if f in reference.level_effects]
    if not shared:
        raise ValueError(
            f"两次分析没有共同因子。estimate: {list(estimate.level_effects)};"
            f" reference: {list(reference.level_effects)}")
    for factor in shared:
        a, b = estimate.level_effects[factor], reference.level_effects[factor]
        levels = [lvl for lvl in a if lvl in b]
        va = np.array([a[lvl] for lvl in levels], dtype=float)
        vb = np.array([b[lvl] for lvl in levels], dtype=float)
        correlation = (float(np.corrcoef(va, vb)[0, 1])
                       if len(levels) > 2 and va.std() > 0 and vb.std() > 0
                       else float("nan"))
        best_a = max(a, key=a.get)
        best_b = max(b, key=b.get)
        rows.append({
            "factor": factor,
            "n_levels_compared": len(levels),
            "effect_r": correlation,
            "eta2_estimate": estimate.variance_explained.get(factor, float("nan")),
            "eta2_reference": reference.variance_explained.get(factor, float("nan")),
            "best_level_estimate": best_a,
            "best_level_reference": best_b,
            "best_level_agrees": best_a == best_b,
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values("eta2_reference", ascending=False).reset_index(drop=True)


@register_function(
    aliases=["plot_round_progress", "画轮次进展", "闭环进展", "DBTL进展",
             "round_progress"],
    category="synthetic_biology",
    description="画每一轮实测响应的分布与累计最优,可选画出全局最优参考线 —— 用来回答'这个闭环到底有没有奏效'。左图每轮一列散点加中位数,右图累计最优随轮次上升。传 best_possible= 时会标出距离真实最优还有多远;只有在真值已知(如回溯性地用一份已测全组合的数据集)时才给得出这条线,而那正是能验证方法本身的场合。Plot the measured response per round and the running best.",
    examples=[
        "ov.synbio.plot_round_progress({'round 1': y1, 'round 2': y2})",
        "ov.synbio.plot_round_progress(rounds, best_possible=truth.max())",
    ],
    related=["synbio.propose_combinations", "synbio.analyse_parts",
             "synbio.dbtl_campaign"],
    requires={}, produces={},
)
def plot_round_progress(rounds: Mapping[str, Sequence[float]],
                        best_possible: Optional[float] = None,
                        response_name: str = "response", axes=None):
    """Per-round measured response, and the running best against the ceiling."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not rounds:
        raise ValueError("rounds 不能为空。")
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    else:
        fig = np.ravel(axes)[0].figure
    ax1, ax2 = np.ravel(axes)[:2]

    labels = list(rounds)
    rng = np.random.default_rng(0)
    for i, label in enumerate(labels):
        values = np.asarray(list(rounds[label]), dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        jitter = rng.uniform(-0.13, 0.13, size=values.size)
        ax1.scatter(np.full(values.size, i) + jitter, values, s=26, alpha=0.75,
                    edgecolor="none")
        ax1.hlines(np.median(values), i - 0.25, i + 0.25, color="black", lw=1.6)
    if best_possible is not None:
        ax1.axhline(best_possible, ls="--", color="#c0392b", lw=1.2,
                    label="best in the whole library")
        ax1.legend(fontsize=8)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel(response_name)
    ax1.set_title("measured per round (bar = median)")

    running, best = [], -np.inf
    for label in labels:
        values = [v for v in rounds[label] if v == v]
        best = max([best] + list(values))
        running.append(best)
    ax2.plot(range(len(labels)), running, "o-", color="#2980b9")
    if best_possible is not None:
        ax2.axhline(best_possible, ls="--", color="#c0392b", lw=1.2)
        for i, value in enumerate(running):
            ax2.annotate(f"{100 * 2 ** (value - best_possible):.0f}% of best",
                         (i, value), textcoords="offset points", xytext=(4, -12),
                         fontsize=7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel(f"best {response_name} so far")
    ax2.set_title("running best")
    fig.tight_layout()
    return fig, axes


__all__ = [
    "orthogonal_array", "combinatorial_design", "PartDesign",
    "plot_round_progress", "lookup_combination", "compare_part_effects",
    "analyse_parts", "PartEffects",
    "propose_combinations", "CombinationProposal",
    "plot_part_effects", "plot_design_balance",
    "DISCRETE_DESIGNS",
]
