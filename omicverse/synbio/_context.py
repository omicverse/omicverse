r"""The Learn → Design bridge: constrain a GEM with measured expression.

``ov.bulk`` / ``ov.single`` produce expression; ``ov.synbio`` holds a genome-scale
model. Until now there was no path between them — the GEM answered "what could
this organism do" while the omics answered "what is it doing", and nothing joined
the two. :func:`contextualize_gem` is that join: it turns a transcriptome or
proteome into reaction-level constraints, so FBA answers the narrower and far
more useful question "what can *these cells, in this state* do".

Four published methods, all reachable through ``method=``:

* ``'gimme'`` — Gene Inactivity Moderated by Metabolism and Expression
  (Becker & Palsson 2008). LP. Requires the objective to hold a fraction of its
  optimum, then minimises use of below-threshold reactions. Reports the
  *inconsistency score*: how much low-expression flux the required function
  still forced.
* ``'imat'`` — Integrative Metabolic Analysis Tool (Shlomi 2008). MILP.
  Three-way high/moderate/low call, then maximise agreement: high reactions
  carrying flux, low reactions carrying none. Needs no objective at all, which
  makes it the method of choice when you do not know what the cells optimise.
* ``'init'`` / ``'tinit'`` — Integrative Network Inference for Tissues
  (Agren 2012/2014). MILP. Keeps reactions in proportion to positive evidence
  and drops the rest, optionally protecting metabolic tasks.
* ``'riptide'`` — Reaction Inclusion by Parsimony and Transcript Distribution
  (Jenior 2020). LP. Weighted pFBA where cost is *inverse* to transcript
  abundance, so flux is pushed onto well-expressed enzymes rather than switched
  off in poorly-expressed ones.

Gene → reaction mapping follows the standard convention: ``and`` takes the
minimum (a complex is limited by its scarcest subunit), ``or`` takes the maximum
(isozymes are alternatives). Pass ``or_op='sum'`` for the additive convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .._registry import register_function
from ._reconstruct import _GPR_TOKEN, _is_structural

if TYPE_CHECKING:  # pragma: no cover - typing only
    import cobra
    import pandas as pd
    from anndata import AnnData


_MISSING = (
    "ov.synbio.{fn} 需要额外依赖 COBRApy。请 pip install 'omicverse[synbio]' "
    "(或直接 pip install cobra optlang)。"
)


def _cobra(fn: str):
    try:
        import cobra
        return cobra
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_MISSING.format(fn=fn)) from exc


# ---------------------------------------------------------------------------
# expression input → {gene: value}
# ---------------------------------------------------------------------------

def gene_expression(
    expression,
    *,
    layer: Optional[str] = None,
    groupby: Optional[str] = None,
    group: Optional[str] = None,
    gene_col: Optional[str] = None,
    value_col: Optional[str] = None,
    agg: str = "mean",
) -> Dict[str, float]:
    """Coerce anything omicverse produces into ``{gene_id: expression}``.

    Accepts an :class:`~anndata.AnnData` (genes in ``var``, cells/samples in
    ``obs`` — the omicverse layout), a :class:`pandas.DataFrame` (either
    gene-indexed with a value column, or samples × genes), a
    :class:`pandas.Series`, or a plain mapping.

    Parameters
    ----------
    layer
        AnnData layer to read instead of ``.X`` (e.g. ``'counts'``, ``'lognorm'``).
    groupby, group
        Restrict an AnnData to one cell type / condition before aggregating:
        ``groupby='cell_type', group='hepatocyte'``. Without them every
        observation is aggregated together.
    gene_col, value_col
        For a long-format DataFrame, which columns hold the gene id and value.
    agg
        How to collapse observations per gene: ``'mean'`` | ``'median'`` |
        ``'sum'`` | ``'max'``.

    Returns
    -------
    dict
    """
    import numpy as np

    if isinstance(expression, Mapping):
        return {str(k): float(v) for k, v in expression.items()}

    mod = type(expression).__module__ or ""
    cls = type(expression).__name__

    # --- AnnData -----------------------------------------------------------
    if cls == "AnnData" or "anndata" in mod:
        ad = expression
        if groupby is not None:
            if groupby not in ad.obs:
                raise KeyError(f"groupby={groupby!r} not in adata.obs")
            if group is None:
                raise ValueError("groupby= 需要同时给 group=(要取哪一组)")
            mask = (ad.obs[groupby].astype(str) == str(group)).values
            if not mask.any():
                raise ValueError(f"adata.obs[{groupby!r}] 中没有 {group!r}")
            ad = ad[mask]
        X = ad.layers[layer] if layer else ad.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=float)
        vals = _aggregate(X, agg, axis=0)
        genes = [str(g) for g in ad.var_names]
        return dict(zip(genes, (float(v) for v in vals)))

    # --- pandas ------------------------------------------------------------
    if cls == "Series":
        return {str(k): float(v) for k, v in expression.items()}

    if cls == "DataFrame":
        df = expression
        if gene_col is not None:
            vc = value_col
            if vc is None:
                numeric = [c for c in df.columns
                           if c != gene_col and str(df[c].dtype) != "object"]
                if not numeric:
                    raise ValueError("找不到数值列;请传 value_col=")
                vc = numeric[0]
            return {str(g): float(v) for g, v in zip(df[gene_col], df[vc])}
        if df.shape[1] == 1:
            col = df.columns[0]
            return {str(i): float(v) for i, v in zip(df.index, df[col])}
        # samples × genes — aggregate down the samples
        vals = _aggregate(df.to_numpy(dtype=float), agg, axis=0)
        return {str(g): float(v) for g, v in zip(df.columns, vals)}

    raise TypeError(
        f"不支持的 expression 类型 {cls!r}。可以传 AnnData / DataFrame / Series / dict。")


_ECOLI_ABUNDANCE_URL = (
    "https://raw.githubusercontent.com/tibbdc/ECMpy/master/data/gene_abundance.csv"
)


@register_function(
    aliases=["fetch_ecoli_abundance", "大肠杆菌丰度", "ecoli_proteome",
             "大肠杆菌蛋白组", "fetch_ecoli_proteome"],
    category="synthetic_biology",
    description="下载真实的大肠杆菌基因丰度表(b 编号 → 蛋白丰度,来自已发表蛋白组,经 ECMpy 整理),用作 contextualize_gem 的输入。基因 id 就是 BiGG 模型用的 b 编号,与 e_coli_core / iML1515 直接对得上(textbook 模型 137 个基因里覆盖 136 个),不需要 id 映射。缓存到本地,只下载一次。Fetch a real E. coli gene-abundance table keyed by b-number.",
    examples=[
        "abund = ov.synbio.fetch_ecoli_abundance()",
        "res = ov.synbio.contextualize_gem(abund, model, method='gimme')",
    ],
    related=["synbio.contextualize_gem", "synbio.reaction_expression"],
    requires={},
    produces={},
)
def fetch_ecoli_abundance(cache_dir: Optional[str] = None,
                          timeout: float = 60.0) -> Dict[str, float]:
    """Real *E. coli* protein abundances as ``{b_number: abundance}``.

    Downloaded once and cached. The identifiers are the same b-numbers BiGG
    models use for genes, so this drops straight into
    :func:`contextualize_gem` without an id-mapping step — which is where most
    omics-to-GEM attempts actually fail.
    """
    import os
    import urllib.request

    root = cache_dir or os.environ.get(
        "OMICOS_SYNBIO_GEM_DIR",
        os.path.join(os.path.expanduser("~"), ".omicverse", "synbio_gems"))
    os.makedirs(root, exist_ok=True)
    local = os.path.join(root, "ecoli_gene_abundance.csv")

    if not os.path.exists(local):
        try:
            urllib.request.urlretrieve(_ECOLI_ABUNDANCE_URL, local)
        except Exception as exc:
            raise RuntimeError(
                f"无法下载大肠杆菌丰度表({exc})。可以手动取 "
                f"{_ECOLI_ABUNDANCE_URL} 放到 {local},或直接把自己的 "
                f"{{gene: value}} 传给 contextualize_gem。") from exc

    out: Dict[str, float] = {}
    with open(local, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            gene = parts[0].strip().strip('"')
            try:
                out[gene] = float(parts[1])
            except ValueError:
                continue
    if not out:
        raise RuntimeError(f"丰度表 {local} 解析后是空的。")
    return out


def _aggregate(X, agg: str, axis: int):
    import numpy as np
    fns = {"mean": np.nanmean, "median": np.nanmedian,
           "sum": np.nansum, "max": np.nanmax}
    if agg not in fns:
        raise ValueError(f"agg must be one of {sorted(fns)}, got {agg!r}")
    if X.ndim == 1:
        return X
    return fns[agg](X, axis=axis)


# ---------------------------------------------------------------------------
# gene → reaction score
# ---------------------------------------------------------------------------

def _gpr_score(rule: str, values: Mapping[str, float],
               or_op: str = "max") -> Optional[float]:
    """Fold gene values through a GPR: ``and`` → min, ``or`` → max (or sum).

    Returns None when the rule mentions no gene we have a measurement for —
    "unknown", which every method below treats differently from "zero".
    """
    rule = (rule or "").strip()
    if not rule:
        return None

    tokens = [t for t in _GPR_TOKEN.findall(rule) if t.strip()]
    prec = {"or": 1, "and": 2}
    out: List[Optional[float]] = []
    ops: List[str] = []

    def _combine(op: str) -> None:
        if len(out) < 2:
            return
        b = out.pop()
        a = out.pop()
        if a is None:
            out.append(b)
            return
        if b is None:
            out.append(a)
            return
        if op == "and":
            out.append(min(a, b))
        elif or_op == "sum":
            out.append(a + b)
        else:
            out.append(max(a, b))

    for tok in tokens:
        low = tok.lower()
        if low in ("and", "or"):
            while ops and ops[-1] in prec and prec[ops[-1]] >= prec[low]:
                _combine(ops.pop())
            ops.append(low)
        elif tok == "(":
            ops.append(tok)
        elif tok == ")":
            while ops and ops[-1] != "(":
                _combine(ops.pop())
            if ops:
                ops.pop()
        else:
            v = values.get(tok)
            out.append(None if v is None else float(v))
    while ops:
        op = ops.pop()
        if op in prec:
            _combine(op)
    return out[-1] if out else None


@register_function(
    aliases=["reaction_expression", "反应表达量", "gpr_score", "基因映射反应",
             "map_expression_to_reactions", "反应活性"],
    category="synthetic_biology",
    description="把基因表达按 GPR 规则折叠成反应级表达量(and 取 min = 复合物受最少亚基限制,or 取 max = 同工酶互为替代)。没有任何测量值的反应返回 None,与表达量为 0 区分开。Map gene expression onto reactions through their GPR rules.",
    examples=[
        "rxn_expr = ov.synbio.reaction_expression(adata, model)",
        "rxn_expr = ov.synbio.reaction_expression({'b0351': 12.0}, model, or_op='sum')",
    ],
    related=["synbio.contextualize_gem"],
    requires={},
    produces={},
)
def reaction_expression(
    expression,
    model: "cobra.Model",
    *,
    or_op: str = "max",
    layer: Optional[str] = None,
    groupby: Optional[str] = None,
    group: Optional[str] = None,
    gene_col: Optional[str] = None,
    value_col: Optional[str] = None,
    agg: str = "mean",
) -> Dict[str, Optional[float]]:
    """Reaction-level expression, ``{reaction_id: value_or_None}``."""
    if or_op not in ("max", "sum"):
        raise ValueError(f"or_op must be one of ['max', 'sum'], got {or_op!r}")
    genes = gene_expression(expression, layer=layer, groupby=groupby, group=group,
                            gene_col=gene_col, value_col=value_col, agg=agg)
    return {r.id: _gpr_score(r.gene_reaction_rule, genes, or_op=or_op)
            for r in model.reactions}


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------

@dataclass
class ContextResult:
    """A GEM narrowed to one measured state."""

    model: "cobra.Model"
    method: str
    fluxes: Dict[str, float] = field(default_factory=dict)
    active_reactions: List[str] = field(default_factory=list)
    inactive_reactions: List[str] = field(default_factory=list)
    removed_reactions: List[str] = field(default_factory=list)
    reaction_expression: Dict[str, Optional[float]] = field(default_factory=dict)
    #: The biological objective **re-maximised** on the returned model. With
    #: ``prune=False`` (the default) nothing has been removed and no bound has
    #: moved, so this is simply the unconstrained wild-type optimum and carries no
    #: information about the context. Compare :attr:`objective_in_state`.
    objective_value: float = 0.0
    #: The biological objective attained **in the flux state actually returned**.
    #: For iMAT this is routinely 0: the MILP optimises expression agreement, not
    #: growth, so the state it hands back is not a growing one. Quoting
    #: ``objective_value`` alongside those fluxes described two different LPs.
    objective_in_state: float = 0.0
    #: How many reaction bounds differ from the input model.
    n_bounds_changed: int = 0
    inconsistency: Optional[float] = None       # GIMME
    agreement: Optional[float] = None           # iMAT / INIT
    thresholds: Dict[str, float] = field(default_factory=dict)
    hit_time_limit: bool = False                # MILP stopped early
    solver_status: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def changed_the_model(self) -> bool:
        """Whether contextualisation actually narrowed the model at all."""
        return bool(self.n_bounds_changed or self.removed_reactions)

    @property
    def n_active(self) -> int:
        return len(self.active_reactions)

    def to_frame(self) -> "pd.DataFrame":
        """Per-reaction table: expression, flux, active call."""
        import pandas as pd
        active = set(self.active_reactions)
        rows = []
        for rid, expr in self.reaction_expression.items():
            rows.append({
                "reaction": rid,
                "expression": expr,
                "flux": self.fluxes.get(rid, 0.0),
                "active": rid in active,
            })
        return pd.DataFrame(rows).set_index("reaction")

    def __repr__(self) -> str:  # pragma: no cover
        extra = ""
        if self.inconsistency is not None:
            extra = f", inconsistency={self.inconsistency:.3f}"
        elif self.agreement is not None:
            extra = f", agreement={self.agreement:.3f}"
        return (f"ContextResult({self.method}, {self.n_active} active rxns, "
                f"objective={self.objective_value:.4f}{extra})")


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------

def _apply_time_limit(work, seconds: float) -> bool:
    """Cap the solver's wall clock, and say so out loud when it cannot be set.

    optlang's GLPK ``timeout`` is passed straight to ``glp_smcp_tm_lim_set``,
    which is typed ``int`` — handing it a float raises TypeError. That used to be
    swallowed by a bare ``except: pass`` here, so the limit silently never
    applied and iMAT ran for hours instead of stopping. A limit that fails to
    apply must be loud: the whole point of it is to bound the wait.
    """
    import warnings

    try:
        work.solver.configuration.timeout = int(max(1, round(float(seconds))))
        return True
    except Exception as exc:  # pragma: no cover - solver without a timeout knob
        warnings.warn(
            f"无法给求解器设置时限({type(exc).__name__}: {exc})。MILP 方法"
            f"(imat/init/tinit)可能长时间不返回;可以改用 method='gimme'/'riptide'"
            f"(纯 LP)。",
            RuntimeWarning, stacklevel=3,
        )
        return False


def _solve_tolerant(work, label: str) -> Tuple[Dict[str, float], float, str, bool]:
    """Solve a MILP and accept the incumbent if the solver stopped early.

    Branch-and-bound on these formulations does not always prove optimality
    inside the time limit. A feasible incumbent is still a usable contextual
    model — far better than raising — so long as the caller is told, which is
    what the returned flag and ``solver_status`` are for.
    """
    solver = work.solver
    try:
        solver.optimize()
    except Exception as exc:
        raise RuntimeError(
            f"{label} 的 MILP 求解失败({exc})。如果求解器不支持 MILP,"
            f"改用 method='gimme' 或 'riptide'(纯 LP)。") from exc

    status = str(solver.status or "")
    try:
        primal = {k: float(v) for k, v in solver.primal_values.items()}
    except Exception:
        primal = {}

    fluxes: Dict[str, float] = {}
    for rxn in work.reactions:
        fwd = primal.get(rxn.forward_variable.name, 0.0)
        rev = primal.get(rxn.reverse_variable.name, 0.0)
        fluxes[rxn.id] = fwd - rev

    try:
        obj = float(solver.objective.value)
    except Exception:
        obj = 0.0
    if obj != obj:
        obj = 0.0

    early = status not in ("optimal",)
    if early and not primal:
        raise RuntimeError(
            f"{label} 的 MILP 没有得到可用解(solver status={status!r})。"
            f"可以放宽 time_limit,或改用 method='gimme'/'riptide'(纯 LP)。")
    return fluxes, obj, status, early


def _percentile_thresholds(values: Sequence[float], low_pct: float,
                           high_pct: float) -> Tuple[float, float]:
    import numpy as np
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0, 0.0
    return (float(np.percentile(vals, low_pct)),
            float(np.percentile(vals, high_pct)))


# ---------------------------------------------------------------------------
# contextualize_gem
# ---------------------------------------------------------------------------

@register_function(
    aliases=["contextualize_gem", "情境化模型", "组学约束模型", "GIMME", "iMAT",
             "tINIT", "RIPTiDe", "context_specific_model", "表达约束GEM",
             "转录组约束代谢模型", "omics_to_gem"],
    category="synthetic_biology",
    description="用转录组/蛋白组把通用 GEM 收窄成特定状态的模型 —— DBTL 里 Learn→Design 的那座桥。method='gimme'(LP,需要目标函数)/'imat'(MILP,不需要目标函数)/'init'|'tinit'(MILP,按证据保留反应)/'riptide'(LP,加权 pFBA,通量压向高表达酶)。直接吃 ov.bulk / ov.single 的 AnnData。Constrain a GEM with measured expression.",
    examples=[
        "res = ov.synbio.contextualize_gem(adata, model, method='gimme')",
        "res = ov.synbio.contextualize_gem(adata, model, method='imat', groupby='cell_type', group='tumor')",
        "res.model, res.n_active, res.to_frame().head()",
    ],
    related=["synbio.reaction_expression", "synbio.fba", "synbio.ec_model",
             "synbio.plot_contextualization", "bulk.pyDEG", "single.pyVIA"],
    requires={},
    produces={},
)
def contextualize_gem(
    expression,
    model: "cobra.Model",
    method: str = "gimme",
    *,
    layer: Optional[str] = None,
    groupby: Optional[str] = None,
    group: Optional[str] = None,
    gene_col: Optional[str] = None,
    value_col: Optional[str] = None,
    agg: str = "mean",
    or_op: str = "max",
    threshold: Optional[float] = None,
    low_percentile: float = 25.0,
    high_percentile: float = 75.0,
    fraction_of_optimum: float = 0.9,
    epsilon: float = 1.0,
    prune: bool = False,
    protected_reactions: Optional[Sequence[str]] = None,
    sampling: int = 0,
    time_limit: Optional[float] = 60.0,
) -> ContextResult:
    """Narrow ``model`` to the state described by ``expression``.

    Parameters
    ----------
    expression
        AnnData / DataFrame / Series / mapping — see :func:`gene_expression`.
    model
        The generic :class:`cobra.Model`. Never mutated; work happens on a copy.
    method
        ``'gimme'`` | ``'imat'`` | ``'init'`` | ``'tinit'`` | ``'riptide'``.
    threshold
        Expression level separating "on" from "off". Defaults to the
        ``low_percentile`` of observed reaction expression, so it adapts to the
        dataset's scale instead of assuming TPM or log-counts.
    low_percentile, high_percentile
        Used by ``'imat'`` / ``'init'`` for the three-way call, and to derive
        ``threshold`` when it is not given.
    fraction_of_optimum
        GIMME's required metabolic functionality — how much of the objective the
        contextual model must still achieve.
    epsilon
        Minimum flux that counts as "carrying flux" in the MILP methods.
    prune
        Delete inactive reactions from the returned model rather than only
        reporting them. A pruned model is smaller and faster but can no longer
        answer counterfactuals.
    protected_reactions
        Never called inactive and never pruned. Boundary and biomass reactions
        are protected automatically.
    sampling
        ``'riptide'`` only: draw this many flux samples from the contextualised
        model (0 = skip). Sampling is what RIPTiDe uses to summarise the
        remaining solution space.
    time_limit
        Seconds allowed to the MILP methods (``'imat'``, ``'init'``,
        ``'tinit'``), after which the best solution found so far is used and
        ``result.hit_time_limit`` is set. iMAT is genuinely NP-hard: when a large
        share of reactions is called "high", every one of them is required to
        carry flux simultaneously and a branch-and-bound solver can grind for
        hours. A library function must not hang, so it stops and says so. Pass
        None to allow unlimited time.

    Returns
    -------
    ContextResult
    """
    method = method.lower()
    valid = ("gimme", "imat", "init", "tinit", "riptide")
    if method not in valid:
        raise ValueError(f"method must be one of {list(valid)}, got {method!r}")
    _cobra("contextualize_gem")

    rxn_expr = reaction_expression(
        expression, model, or_op=or_op, layer=layer, groupby=groupby,
        group=group, gene_col=gene_col, value_col=value_col, agg=agg,
    )

    observed = [v for v in rxn_expr.values() if v is not None]
    if not observed:
        raise ValueError(
            "没有任何反应拿到表达值 —— 基因 id 和模型的 gene id 对不上。"
            "检查命名体系(例如 b 编号 vs 基因名),或先做一次 id 映射。")

    lo, hi = _percentile_thresholds(observed, low_percentile, high_percentile)
    thr = float(lo if threshold is None else threshold)

    protected: Set[str] = set(protected_reactions or [])
    for r in model.reactions:
        if _is_structural(r):
            protected.add(r.id)

    work = model.copy()
    if method in ("imat", "init", "tinit") and time_limit is not None:
        _apply_time_limit(work, time_limit)

    if method == "gimme":
        res = _gimme(work, rxn_expr, thr, fraction_of_optimum, protected)
    elif method == "imat":
        if hi <= lo:
            raise ValueError(
                "iMAT 需要表达量有分布才能做高/中/低三分,但 "
                f"low={lo:g} 与 high={hi:g} 相等(数据没有离散度,或百分位取得太近)。"
                "请给不同的 low_percentile/high_percentile,或改用 "
                "method='gimme'/'riptide'(只需要一个阈值)。")
        res = _imat(work, rxn_expr, lo, hi, epsilon, protected)
    elif method in ("init", "tinit"):
        res = _init(work, rxn_expr, thr, epsilon, protected, tissue=(method == "tinit"))
    else:
        res = _riptide(work, rxn_expr, thr, fraction_of_optimum, protected, sampling)

    res.reaction_expression = rxn_expr
    res.thresholds = {"threshold": thr, "low": lo, "high": hi}

    # Did anything actually change? A "contextualised" model whose bounds are
    # identical to the input is the wild-type model, and every number taken from
    # it is a wild-type number.
    changed = 0
    for rxn in res.model.reactions:
        try:
            original = model.reactions.get_by_id(rxn.id)
        except Exception:
            continue
        if (abs(rxn.lower_bound - original.lower_bound) > 1e-9
                or abs(rxn.upper_bound - original.upper_bound) > 1e-9):
            changed += 1
    res.n_bounds_changed = changed

    # the objective in the state actually returned, not a re-maximisation
    objective_ids = [r.id for r in model.reactions
                     if getattr(r, "objective_coefficient", 0.0)]
    if objective_ids and res.fluxes:
        res.objective_in_state = float(sum(
            res.fluxes.get(rid, 0.0) for rid in objective_ids))

    if not res.changed_the_model:
        res.notes.append(
            f"method={res.method!r} 没有改动模型的任何边界,也没有删除反应 —— "
            f"返回的就是野生型模型,objective_value={res.objective_value:.4f} 就是"
            f"未约束最优值,与表达数据无关。要得到真正被收窄的模型,传 "
            f"prune=True(删掉不活跃反应),或自行按 res.inactive_reactions 收紧边界。")
    if (objective_ids and res.fluxes
            and abs(res.objective_in_state) < 1e-6 < abs(res.objective_value)):
        res.notes.append(
            f"注意:返回的通量状态里目标函数是 {res.objective_in_state:.4g}"
            f"(不生长),而 objective_value 报的是重新最大化后的 "
            f"{res.objective_value:.4f} —— 这两个数来自不同的 LP。"
            f"要讨论这个状态,用 objective_in_state。")

    if prune and res.inactive_reactions:
        drop = [r for r in res.inactive_reactions if r not in protected]
        if drop:
            res.model.remove_reactions(
                [res.model.reactions.get_by_id(r) for r in drop],
                remove_orphans=True)
            res.removed_reactions = drop
    return res


# ---------------------------------------------------------------------------
# GIMME  (Becker & Palsson 2008)
# ---------------------------------------------------------------------------

def _gimme(work, rxn_expr, threshold, fraction, protected) -> ContextResult:
    """Minimise flux through below-threshold reactions, weighted by how far
    below they sit, while the objective holds ``fraction`` of its optimum.

    The optimal value of that penalty *is* GIMME's inconsistency score: the
    amount of poorly-supported flux the required function could not avoid.
    """
    prob = work.problem
    base = float(work.slim_optimize() or 0.0)
    if base != base:
        base = 0.0

    obj_expr = work.objective.expression
    work.add_cons_vars([prob.Constraint(obj_expr, lb=fraction * base,
                                        name="_gimme_rmf")])

    aux, terms = [], []
    for rid, expr in rxn_expr.items():
        if expr is None or rid in protected:
            continue
        if expr >= threshold:
            continue
        rxn = work.reactions.get_by_id(rid)
        a = prob.Variable(f"_gimme_abs_{rid}", lb=0)
        aux.append(a)
        # The variable must be registered before any constraint mentions it:
        # add_cons_vars pulls referenced variables in implicitly, so adding the
        # constraints first and the variables afterwards raises
        # ContainerAlreadyContains on the second insert.
        work.add_cons_vars([a])
        work.add_cons_vars([
            prob.Constraint(a - rxn.flux_expression, lb=0, name=f"_gimme_p_{rid}"),
            prob.Constraint(a + rxn.flux_expression, lb=0, name=f"_gimme_n_{rid}"),
        ])
        terms.append((threshold - expr) * a)

    if terms:
        work.objective = prob.Objective(sum(terms), direction="min")
        sol = work.optimize()
        inconsistency = float(sol.objective_value or 0.0)
        fluxes = {k: float(v) for k, v in sol.fluxes.items()}
    else:
        inconsistency = 0.0
        sol = work.optimize()
        fluxes = {k: float(v) for k, v in sol.fluxes.items()}

    active = [rid for rid, v in fluxes.items() if abs(v) > 1e-8]
    inactive = [rid for rid in fluxes if abs(fluxes[rid]) <= 1e-8]

    # restore the biological objective on the returned model
    work.objective = prob.Objective(obj_expr, direction="max")
    objective_value = float(work.slim_optimize() or 0.0)
    if objective_value != objective_value:
        objective_value = 0.0

    return ContextResult(
        model=work, method="gimme", fluxes=fluxes,
        active_reactions=active, inactive_reactions=inactive,
        objective_value=objective_value, inconsistency=inconsistency,
    )


# ---------------------------------------------------------------------------
# iMAT  (Shlomi 2008)
# ---------------------------------------------------------------------------

def _imat(work, rxn_expr, lo, hi, epsilon, protected) -> ContextResult:
    """MILP maximising expression agreement — no objective function needed.

    For each highly-expressed reaction a binary says "carries |flux| >= eps";
    for each lowly-expressed one a binary says "carries no flux". Maximise the
    number of satisfied binaries. This is the method to use when you do not
    know what the cells are optimising, which is most of the time outside of
    exponential-phase batch culture.
    """
    prob = work.problem
    bin_vars, cons = [], []
    high, low = [], []

    for rid, expr in rxn_expr.items():
        if expr is None or rid in protected:
            continue
        rxn = work.reactions.get_by_id(rid)
        ub = max(abs(rxn.upper_bound), abs(rxn.lower_bound), 1.0)
        if expr >= hi:
            high.append(rid)
            # y_pos: v >= eps ;  y_neg: v <= -eps  (either satisfies "active")
            yp = prob.Variable(f"_imat_hp_{rid}", type="binary")
            yn = prob.Variable(f"_imat_hn_{rid}", type="binary")
            bin_vars += [yp, yn]
            cons.append(prob.Constraint(
                rxn.flux_expression - epsilon * yp + ub * (1 - yp), lb=0,
                name=f"_imat_hpc_{rid}"))
            cons.append(prob.Constraint(
                rxn.flux_expression + epsilon * yn - ub * (1 - yn), ub=0,
                name=f"_imat_hnc_{rid}"))
        elif expr <= lo:
            low.append(rid)
            yz = prob.Variable(f"_imat_lz_{rid}", type="binary")
            bin_vars.append(yz)
            # yz = 1 forces -0 <= v <= 0
            cons.append(prob.Constraint(
                rxn.flux_expression - ub * (1 - yz), ub=0, name=f"_imat_lzu_{rid}"))
            cons.append(prob.Constraint(
                rxn.flux_expression + ub * (1 - yz), lb=0, name=f"_imat_lzl_{rid}"))

    if not bin_vars:
        sol = work.optimize()
        fluxes = {k: float(v) for k, v in sol.fluxes.items()}
        active = [r for r, v in fluxes.items() if abs(v) > 1e-8]
        return ContextResult(
            model=work, method="imat", fluxes=fluxes, active_reactions=active,
            inactive_reactions=[r for r in fluxes if abs(fluxes[r]) <= 1e-8],
            objective_value=float(sol.objective_value or 0.0), agreement=1.0)

    obj_expr = work.objective.expression
    work.add_cons_vars(bin_vars + cons)
    work.objective = prob.Objective(sum(bin_vars), direction="max")
    fluxes, satisfied, status, early = _solve_tolerant(work, "iMAT")

    n_targets = len(high) + len(low)
    active = [r for r, v in fluxes.items() if abs(v) > 1e-8]
    inactive = [r for r in fluxes if abs(fluxes[r]) <= 1e-8]

    # ``objective_value`` means the same thing for every method — the biological
    # objective of the contextualised model. The count of satisfied expression
    # targets is a property of the fit, and belongs in ``agreement``.
    work.objective = prob.Objective(obj_expr, direction="max")
    objv = float(work.slim_optimize() or 0.0)

    return ContextResult(
        model=work, method="imat", fluxes=fluxes, active_reactions=active,
        inactive_reactions=inactive,
        objective_value=objv if objv == objv else 0.0,
        agreement=(satisfied / n_targets if n_targets else 1.0),
        hit_time_limit=early, solver_status=status,
    )


# ---------------------------------------------------------------------------
# INIT / tINIT  (Agren 2012 / 2014)
# ---------------------------------------------------------------------------

def _init(work, rxn_expr, threshold, epsilon, protected, tissue: bool) -> ContextResult:
    """Keep reactions in proportion to positive evidence, drop the rest.

    Weight ``w = expr - threshold`` — positive for supported reactions, negative
    for contradicted ones — and maximise ``sum(w_i * y_i)`` over binaries that
    force flux when set. ``tinit`` additionally requires every protected
    reaction to stay usable, which is the "metabolic tasks" part of tINIT.
    """
    prob = work.problem
    bin_vars, cons, terms = [], [], []

    for rid, expr in rxn_expr.items():
        if expr is None or rid in protected:
            continue
        rxn = work.reactions.get_by_id(rid)
        ub = max(abs(rxn.upper_bound), abs(rxn.lower_bound), 1.0)
        w = float(expr) - threshold
        y = prob.Variable(f"_init_y_{rid}", type="binary")
        bin_vars.append(y)
        # y = 1 -> the reaction may carry flux; y = 0 -> pinned to zero
        cons.append(prob.Constraint(rxn.flux_expression - ub * y, ub=0,
                                    name=f"_init_u_{rid}"))
        cons.append(prob.Constraint(rxn.flux_expression + ub * y, lb=0,
                                    name=f"_init_l_{rid}"))
        terms.append(w * y)

    if tissue:
        base = float(work.slim_optimize() or 0.0)
        if base == base and base > 1e-9:
            cons.append(prob.Constraint(work.objective.expression,
                                        lb=0.1 * base, name="_tinit_task"))

    if not bin_vars:
        sol = work.optimize()
        fluxes = {k: float(v) for k, v in sol.fluxes.items()}
        return ContextResult(
            model=work, method="tinit" if tissue else "init", fluxes=fluxes,
            active_reactions=[r for r, v in fluxes.items() if abs(v) > 1e-8],
            inactive_reactions=[r for r in fluxes if abs(fluxes[r]) <= 1e-8],
            objective_value=float(sol.objective_value or 0.0), agreement=1.0)

    obj_expr = work.objective.expression
    work.add_cons_vars(bin_vars + cons)
    work.objective = prob.Objective(sum(terms), direction="max")
    fluxes, score, status, early = _solve_tolerant(work, "INIT")

    def _on(v) -> bool:
        return v.primal is not None and v.primal > 0.5

    kept = [v.name[len("_init_y_"):] for v in bin_vars if _on(v)]
    dropped = [v.name[len("_init_y_"):] for v in bin_vars if not _on(v)]

    work.objective = prob.Objective(obj_expr, direction="max")
    objv = float(work.slim_optimize() or 0.0)

    return ContextResult(
        model=work, method="tinit" if tissue else "init", fluxes=fluxes,
        active_reactions=sorted(set(kept) | set(protected)),
        inactive_reactions=dropped,
        objective_value=objv if objv == objv else 0.0,
        agreement=score, hit_time_limit=early, solver_status=status,
    )


# ---------------------------------------------------------------------------
# RIPTiDe  (Jenior 2020)
# ---------------------------------------------------------------------------

def _riptide(work, rxn_expr, threshold, fraction, protected, sampling) -> ContextResult:
    """Weighted pFBA where cost is inverse to transcript abundance.

    Rather than switching low-expression reactions off, RIPTiDe makes them
    *expensive*: minimise ``sum(|v_i| / (1 + expr_i))``, so the same total flux
    prefers routes through well-expressed enzymes. Pure LP, and it never forces
    a reaction to exactly zero — which is why it degrades gracefully on noisy
    single-cell data where a hard on/off call is not defensible.
    """
    prob = work.problem
    base = float(work.slim_optimize() or 0.0)
    if base != base:
        base = 0.0
    obj_expr = work.objective.expression
    work.add_cons_vars([prob.Constraint(obj_expr, lb=fraction * base,
                                        name="_riptide_rmf")])

    observed = [v for v in rxn_expr.values() if v is not None]
    scale = max(observed) if observed else 1.0

    aux, terms = [], []
    for rid, expr in rxn_expr.items():
        rxn = work.reactions.get_by_id(rid)
        if rid in protected:
            continue
        e = 0.0 if expr is None else float(expr) / (scale or 1.0)
        cost = 1.0 / (1.0 + e)
        a = prob.Variable(f"_rip_abs_{rid}", lb=0)
        aux.append(a)
        work.add_cons_vars([a])          # before the constraints — see _gimme
        work.add_cons_vars([
            prob.Constraint(a - rxn.flux_expression, lb=0, name=f"_rip_p_{rid}"),
            prob.Constraint(a + rxn.flux_expression, lb=0, name=f"_rip_n_{rid}"),
        ])
        terms.append(cost * a)

    work.objective = prob.Objective(sum(terms) if terms else obj_expr,
                                    direction="min" if terms else "max")
    sol = work.optimize()
    fluxes = {k: float(v) for k, v in sol.fluxes.items()}
    active = [r for r, v in fluxes.items() if abs(v) > 1e-8]
    inactive = [r for r in fluxes if abs(fluxes[r]) <= 1e-8]

    work.objective = prob.Objective(obj_expr, direction="max")
    objv = float(work.slim_optimize() or 0.0)

    res = ContextResult(
        model=work, method="riptide", fluxes=fluxes, active_reactions=active,
        inactive_reactions=inactive,
        objective_value=objv if objv == objv else 0.0,
    )

    if sampling:
        try:
            from cobra.sampling import sample
            res.samples = sample(work, int(sampling))  # type: ignore[attr-defined]
        except Exception:
            pass
    return res


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

@register_function(
    aliases=["plot_contextualization", "情境化图", "plot_context_model",
             "表达通量图", "组学约束图"],
    category="synthetic_biology",
    description="画组学情境化的结果:反应表达量分布与阈值、活跃/沉默反应计数、表达量 vs 通量散点(检验约束是否真的起作用)。Visualise how expression narrowed the model.",
    examples=["ov.synbio.plot_contextualization(res)"],
    related=["synbio.contextualize_gem"],
    requires={},
    produces={},
)
def plot_contextualization(result: ContextResult, axes=None):
    """Three panels: expression histogram with thresholds, active/inactive
    counts, and expression versus attained flux."""
    from ._plot import _mpl
    plt = _mpl()
    import numpy as np

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    else:
        fig = axes[0].figure
    axes = list(axes)

    expr = [v for v in result.reaction_expression.values() if v is not None]
    thr = result.thresholds

    ax = axes[0]
    if expr:
        ax.hist(expr, bins=min(30, max(5, len(expr) // 3)), color="#377EB8",
                alpha=0.85)
    for key, colour, style in (("threshold", "k", "--"),
                               ("low", "#E41A1C", ":"),
                               ("high", "#4DAF4A", ":")):
        if key in thr:
            ax.axvline(thr[key], color=colour, ls=style, lw=1.2,
                       label=f"{key} = {thr[key]:.3g}")
    ax.set_xlabel("reaction expression")
    ax.set_ylabel("reactions")
    ax.set_title("expression mapped through GPRs")
    ax.legend(fontsize=7)

    ax = axes[1]
    n_act = len(result.active_reactions)
    n_inact = len(result.inactive_reactions)
    n_removed = len(result.removed_reactions)
    labels = ["active", "inactive"]
    vals = [n_act, n_inact]
    colours = ["#4DAF4A", "#E0E0E0"]
    if n_removed:
        labels.append("pruned")
        vals.append(n_removed)
        colours.append("#E41A1C")
    ax.bar(labels, vals, color=colours)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("reactions")
    title = f"{result.method}"
    if result.inconsistency is not None:
        title += f"  (inconsistency {result.inconsistency:.3g})"
    elif result.agreement is not None:
        title += f"  (agreement {result.agreement:.3g})"
    ax.set_title(title)

    ax = axes[2]
    xs, ys = [], []
    for rid, e in result.reaction_expression.items():
        if e is None:
            continue
        xs.append(e)
        ys.append(abs(result.fluxes.get(rid, 0.0)))
    if xs:
        ax.scatter(xs, ys, s=12, alpha=0.6, color="#984EA3")
        if len(xs) > 2 and np.std(xs) > 0 and np.std(ys) > 0:
            r = float(np.corrcoef(xs, ys)[0, 1])
            ax.set_title(f"expression vs |flux|  (r = {r:.2f})")
        else:
            ax.set_title("expression vs |flux|")
    ax.set_xlabel("reaction expression")
    ax.set_ylabel("|flux|")
    ax.set_yscale("symlog", linthresh=1e-6)

    fig.tight_layout()
    return fig, axes


__all__ = [
    "contextualize_gem", "ContextResult", "reaction_expression",
    "gene_expression", "plot_contextualization",
]
