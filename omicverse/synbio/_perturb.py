r"""What a cell actually does after a knockout — not what it would do if it
could re-optimise from scratch.

Plain FBA answers "given this deletion, what is the best the network could
possibly do", and a freshly-deleted cell is not that cell. It still has the
wild-type proteome, so it lands on the *nearest* feasible flux state, not the
globally optimal one. FBA is therefore systematically optimistic about
knockouts, which matters precisely when you are using it to pick knockout
targets.

* :func:`knockout_flux` — recompute flux after a deletion under
  ``method='fba'`` (re-optimise), ``'moma'`` / ``'linear_moma'``
  (Minimisation Of Metabolic Adjustment, Segrè 2002 — stay close to the
  wild-type flux vector), or ``'room'`` (Regulatory On/Off Minimisation,
  Shlomi 2005 — change as *few* fluxes as possible, however much each changes).
* :func:`compare_knockout_methods` — the same deletion under every method side
  by side. This is the table to look at before trusting an FBA knockout screen.
* :func:`plot_knockout_response` — growth by method, and where the flux moved.

COBRApy supplies the solvers for MOMA and ROOM; what this module adds is a
single comparable answer. Their native ``objective_value`` is *not* growth —
linear MOMA returns the total flux deviation (70.97 on a textbook PGI knockout)
and ROOM returns the count of significantly changed fluxes (1.29). Reading
either as a growth rate silently mixes units, so :class:`PerturbationResult`
keeps them apart: ``growth`` is always biomass flux, ``method_objective`` is
whatever the method itself minimised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover - typing only
    import cobra
    import pandas as pd


_MISSING = (
    "ov.synbio.{fn} 需要额外依赖 COBRApy。请 pip install 'omicverse[synbio]' "
    "(或直接 pip install cobra optlang)。"
)

METHODS = ("fba", "moma", "linear_moma", "room")


def _cobra(fn: str):
    try:
        import cobra
        return cobra
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_MISSING.format(fn=fn)) from exc


def _biomass_reaction(model) -> "Optional[cobra.Reaction]":
    """The reaction the model's objective is built on."""
    try:
        from cobra.util.solver import linear_reaction_coefficients
        coeffs = linear_reaction_coefficients(model)
        if coeffs:
            return max(coeffs, key=lambda r: abs(coeffs[r]))
    except Exception:
        pass
    for rxn in model.reactions:
        if "BIOMASS" in rxn.id.upper():
            return rxn
    return None


@dataclass
class PerturbationResult:
    """Flux state after a deletion, with growth kept separate from the
    method's own objective."""

    method: str
    knockouts: List[str] = field(default_factory=list)
    growth: float = 0.0
    reference_growth: float = 0.0
    fluxes: Dict[str, float] = field(default_factory=dict)
    reference_fluxes: Dict[str, float] = field(default_factory=dict)
    method_objective: Optional[float] = None
    method_objective_meaning: str = ""
    status: str = ""

    @property
    def growth_ratio(self) -> float:
        """Post-knockout growth as a fraction of wild type."""
        if self.reference_growth <= 1e-12:
            return 0.0
        return self.growth / self.reference_growth

    @property
    def lethal(self) -> bool:
        return self.growth <= 1e-6

    def changed_fluxes(self, threshold: float = 1e-3) -> Dict[str, float]:
        """``{reaction: delta}`` for reactions that moved by more than
        ``threshold`` relative to the reference."""
        out = {}
        for rid, v in self.fluxes.items():
            w = self.reference_fluxes.get(rid, 0.0)
            if abs(v - w) > threshold:
                out[rid] = v - w
        return out

    @property
    def n_changed(self) -> int:
        return len(self.changed_fluxes())

    def __repr__(self) -> str:  # pragma: no cover
        return (f"PerturbationResult({self.method}, ko={self.knockouts}, "
                f"growth={self.growth:.4f} "
                f"({self.growth_ratio:.0%} of wt), {self.n_changed} fluxes moved)")


def _resolve_knockouts(model, knockouts, genes) -> List[str]:
    if knockouts is None and genes is None:
        raise ValueError("需要给 knockouts=(反应 id)或 genes=(基因 id)之一。")
    ids: List[str] = []
    if knockouts is not None:
        ids += [k if isinstance(k, str) else k.id
                for k in ([knockouts] if isinstance(knockouts, str) else knockouts)]
    if genes is not None:
        ids += [g if isinstance(g, str) else g.id
                for g in ([genes] if isinstance(genes, str) else genes)]
    return ids


def _apply_knockouts(m, knockouts, genes) -> None:
    if knockouts is not None:
        for k in ([knockouts] if isinstance(knockouts, str) else knockouts):
            rid = k if isinstance(k, str) else k.id
            m.reactions.get_by_id(rid).knock_out()
    if genes is not None:
        for g in ([genes] if isinstance(genes, str) else genes):
            gid = g if isinstance(g, str) else g.id
            m.genes.get_by_id(gid).knock_out()


@register_function(
    aliases=["knockout_flux", "敲除通量", "MOMA", "ROOM", "moma", "room",
             "敲除响应", "knockout_response", "最小代谢调整"],
    category="synthetic_biology",
    description="敲除后重算通量。method='fba'(重新全局最优,系统性偏乐观)/'moma'(二次,需 QP 求解器)/'linear_moma'(线性 MOMA,靠近野生型通量)/'room'(改动的通量条数最少)。刚敲除的细胞还带着野生型蛋白组,走的是离原状态最近的次优解,不是重新全局最优。Recompute flux after a knockout with FBA / MOMA / ROOM.",
    examples=[
        "res = ov.synbio.knockout_flux(m, genes=['b4025'], method='linear_moma')",
        "res.growth, res.growth_ratio, res.n_changed",
        "df = ov.synbio.compare_knockout_methods(m, genes=['b4025'])",
    ],
    related=["synbio.compare_knockout_methods", "synbio.single_gene_deletion",
             "synbio.strain_design", "synbio.plot_knockout_response"],
    requires={},
    produces={},
)
def knockout_flux(
    model: "cobra.Model",
    knockouts: "Optional[str | Sequence[str]]" = None,
    *,
    genes: "Optional[str | Sequence[str]]" = None,
    method: str = "linear_moma",
    reference: "Optional[cobra.core.Solution]" = None,
    delta: float = 0.03,
    epsilon: float = 0.001,
) -> PerturbationResult:
    """Flux state after knocking out reactions and/or genes.

    Parameters
    ----------
    model
        The model. Never mutated — the knockout is applied inside a context.
    knockouts
        Reaction id(s) to disable.
    genes
        Gene id(s) to disable; COBRApy re-evaluates the GPRs.
    method
        ``'fba'`` — re-optimise the objective. Fast, and optimistic: it lets the
        cell find the best state the *deleted* network allows, which no cell
        reaches immediately after losing a gene.
        ``'linear_moma'`` (default) — minimise the L1 distance to the wild-type
        flux vector. This is the honest default: it needs only an LP solver and
        it captures the "nearest feasible state" idea.
        ``'moma'`` — the original quadratic form. Needs a QP-capable solver
        (GLPK is not one); raises with a pointer to ``'linear_moma'`` if absent.
        ``'room'`` — minimise the *number* of fluxes that change appreciably,
        which models regulatory rewiring being costly rather than flux change
        itself.
    reference
        Wild-type solution to stay close to. Computed by pFBA when omitted —
        pFBA rather than FBA because the reference must be a single well-defined
        flux vector, and plain FBA leaves it arbitrary among optima.
    delta, epsilon
        ROOM's relative and absolute tolerances for "this flux changed".

    Returns
    -------
    PerturbationResult
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")
    cobra = _cobra("knockout_flux")

    ko_ids = _resolve_knockouts(model, knockouts, genes)
    biomass = _biomass_reaction(model)

    if reference is None:
        # pFBA, not FBA: MOMA/ROOM measure distance *to a specific vector*, and
        # plain FBA does not pin one down when the optimum is degenerate — the
        # reference would then depend on solver tie-breaking.
        reference = cobra.flux_analysis.pfba(model)
    ref_fluxes = {k: float(v) for k, v in reference.fluxes.items()}
    ref_growth = float(ref_fluxes.get(biomass.id, 0.0)) if biomass else 0.0

    with model as m:
        _apply_knockouts(m, knockouts, genes)
        if method == "fba":
            sol = m.optimize()
            obj, meaning = None, ""
        elif method == "room":
            sol = cobra.flux_analysis.room(m, solution=reference, linear=True,
                                           delta=delta, epsilon=epsilon)
            obj = None if sol.objective_value is None else float(sol.objective_value)
            meaning = "number of significantly changed fluxes"
        else:
            linear = (method == "linear_moma")
            try:
                sol = cobra.flux_analysis.moma(m, solution=reference, linear=linear)
            except Exception as exc:
                if not linear and "QP" in str(exc).upper():
                    raise RuntimeError(
                        "method='moma'(二次型)需要支持 QP 的求解器,当前环境只有 "
                        "GLPK/SCIPY。请改用 method='linear_moma'(线性 MOMA,只需 LP,"
                        "论文里也是被推荐的可扩展替代),或安装 CPLEX/Gurobi/OSQP。"
                    ) from exc
                raise
            obj = None if sol.objective_value is None else float(sol.objective_value)
            meaning = ("total absolute flux deviation from wild type" if linear
                       else "squared flux deviation from wild type")

        fluxes = {k: float(v) for k, v in sol.fluxes.items()}
        status = str(sol.status)

    # Growth is read off the biomass flux, never from ``objective_value`` — for
    # MOMA/ROOM that field is the deviation they minimised, not a growth rate.
    growth = float(fluxes.get(biomass.id, 0.0)) if biomass else 0.0

    return PerturbationResult(
        method=method, knockouts=ko_ids, growth=growth,
        reference_growth=ref_growth, fluxes=fluxes, reference_fluxes=ref_fluxes,
        method_objective=obj, method_objective_meaning=meaning, status=status,
    )


@register_function(
    aliases=["compare_knockout_methods", "敲除方法比较", "FBA_vs_MOMA",
             "敲除预测比较", "compare_moma_room"],
    category="synthetic_biology",
    description="同一个敲除在 FBA / linear MOMA / ROOM 下的结果并排比较,用来看 FBA 偏乐观多少。做敲除筛选前应该先看这张表。Compare a knockout across FBA / MOMA / ROOM.",
    examples=[
        "df = ov.synbio.compare_knockout_methods(m, genes=['b4025'])",
        "df[['growth', 'growth_ratio', 'n_changed']]",
    ],
    related=["synbio.knockout_flux", "synbio.plot_knockout_response"],
    requires={},
    produces={},
)
def compare_knockout_methods(
    model: "cobra.Model",
    knockouts: "Optional[str | Sequence[str]]" = None,
    *,
    genes: "Optional[str | Sequence[str]]" = None,
    methods: Sequence[str] = ("fba", "linear_moma", "room"),
    delta: float = 0.03,
    epsilon: float = 0.001,
) -> "pd.DataFrame":
    """One row per method: growth, growth ratio, fluxes moved, method objective.

    ``'moma'`` is excluded from the default set because it needs a QP solver
    that most installs do not have; add it explicitly if yours does.
    """
    import pandas as pd
    cobra = _cobra("compare_knockout_methods")

    reference = cobra.flux_analysis.pfba(model)
    rows = []
    for method in methods:
        try:
            res = knockout_flux(model, knockouts, genes=genes, method=method,
                                reference=reference, delta=delta, epsilon=epsilon)
        except Exception as exc:
            rows.append({"method": method, "growth": float("nan"),
                         "growth_ratio": float("nan"), "n_changed": -1,
                         "method_objective": float("nan"),
                         "status": f"error: {type(exc).__name__}"})
            continue
        rows.append({
            "method": method,
            "growth": res.growth,
            "growth_ratio": res.growth_ratio,
            "n_changed": res.n_changed,
            "method_objective": (float("nan") if res.method_objective is None
                                 else res.method_objective),
            "status": res.status,
        })
    return pd.DataFrame(rows).set_index("method")


@register_function(
    aliases=["plot_knockout_response", "敲除响应图", "plot_moma", "敲除比较图",
             "plot_knockout_comparison"],
    category="synthetic_biology",
    description="画敲除响应:各方法预测的生长速率对比(FBA 的乐观偏差一目了然)、通量改动条数,以及野生型 vs 敲除后的通量散点。Plot knockout response across methods.",
    examples=[
        "ov.synbio.plot_knockout_response(df)",
        "ov.synbio.plot_knockout_response(res)",
    ],
    related=["synbio.knockout_flux", "synbio.compare_knockout_methods"],
    requires={},
    produces={},
)
def plot_knockout_response(result, axes=None):
    """Accepts a :class:`PerturbationResult` or the DataFrame from
    :func:`compare_knockout_methods`."""
    from ._plot import _mpl
    plt = _mpl()

    is_frame = hasattr(result, "index") and hasattr(result, "columns")

    if is_frame:
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
        else:
            fig = axes[0].figure
        axes = list(axes)
        df = result

        ax = axes[0]
        colours = ["#E41A1C" if m == "fba" else "#377EB8" for m in df.index]
        ax.bar(list(df.index), df["growth"].values, color=colours)
        for i, v in enumerate(df["growth"].values):
            if v == v:
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("post-knockout growth (1/h)")
        ax.set_title("red = FBA (re-optimises freely)")
        ax.tick_params(axis="x", rotation=20)

        ax = axes[1]
        ax.bar(list(df.index), df["n_changed"].values, color="#984EA3")
        ax.set_ylabel("reactions whose flux moved")
        ax.set_title("extent of rewiring")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        return fig, axes

    res = result
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    else:
        fig = axes[0].figure
    axes = list(axes)

    ax = axes[0]
    ax.bar(["wild type", f"{res.method}"], [res.reference_growth, res.growth],
           color=["#B0B0B0", "#377EB8"])
    for i, v in enumerate([res.reference_growth, res.growth]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("growth (1/h)")
    ax.set_title(f"KO {', '.join(res.knockouts) or '-'} "
                 f"({res.growth_ratio:.0%} of wt)")

    ax = axes[1]
    xs = [res.reference_fluxes.get(r, 0.0) for r in res.fluxes]
    ys = list(res.fluxes.values())
    ax.scatter(xs, ys, s=10, alpha=0.5, color="#4DAF4A")
    lim = max(1e-9, max(max(map(abs, xs), default=1.0), max(map(abs, ys), default=1.0)))
    ax.plot([-lim, lim], [-lim, lim], ls="--", c="k", lw=0.8)
    ax.set_xlabel("wild-type flux")
    ax.set_ylabel("post-knockout flux")
    ax.set_title(f"{res.n_changed} reactions moved")

    fig.tight_layout()
    return fig, axes


__all__ = ["knockout_flux", "compare_knockout_methods", "PerturbationResult",
           "plot_knockout_response", "METHODS"]
