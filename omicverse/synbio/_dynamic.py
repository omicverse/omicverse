r"""Dynamic FBA — time-resolved fermentation from a genome-scale model.

Flux Balance Analysis gives a single steady-state snapshot; **dynamic FBA**
(dFBA, the *static-optimisation* approach of Mahadevan *et al.* 2002) turns it
into a batch-fermentation time course. At each time step the substrate uptake
bound is set from the current substrate concentration by Michaelis–Menten
kinetics, an FBA is solved for the instantaneous growth and exchange fluxes, and
biomass / substrate / product concentrations are integrated forward:

    dX/dt = μ·X,   dS/dt = −v_S·X,   dP/dt = v_P·X

* :func:`dynamic_fba` — simulate a batch culture (growth, substrate depletion,
  by-product secretion) and return a tidy time-course ``DataFrame``.
* :func:`plot_dynamic_fba` — plot biomass / substrate / products vs time.

Pure COBRApy + NumPy; CPU-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .._registry import register_function
from ._gem import _cobra


@register_function(
    aliases=["dynamic_fba", "动态FBA", "dFBA", "批式发酵模拟", "dynamic_flux",
             "发酵时间曲线", "dfba"],
    category="synthetic_biology",
    description="动态 FBA(dFBA):把基因组尺度模型变成批式发酵时间曲线——每步按 Michaelis-Menten 摄取设边界、解 FBA、积分生物量/底物/产物(Mahadevan 2002 静态优化法)。Dynamic FBA batch-fermentation time course.",
    examples=[
        "df = ov.synbio.dynamic_fba(m, 'EX_glc__D_e', biomass_0=0.01, substrate_0=15)",
        "ov.synbio.plot_dynamic_fba(df)",
    ],
    related=["synbio.fba", "synbio.production_envelope", "synbio.strain_design"],
    requires={},
    produces={},
)
def dynamic_fba(model: "cobra.Model", substrate_exchange: str,
                biomass_0: float = 0.01, substrate_0: float = 10.0,
                t_end: float = 12.0, dt: float = 0.05,
                vmax: float = 10.0, km: float = 0.5,
                products: Optional[List[str]] = None,
                product_0: Optional[Dict[str, float]] = None) -> "pd.DataFrame":
    """Simulate a batch culture by dynamic FBA.

    Parameters
    ----------
    model : cobra.Model
    substrate_exchange : str
        Exchange reaction id of the limiting substrate (e.g. ``'EX_glc__D_e'``).
    biomass_0, substrate_0 : float
        Initial biomass (gDW/L) and substrate concentration (mmol/L).
    t_end, dt : float
        Simulation horizon and Euler step (h).
    vmax, km : float
        Michaelis–Menten parameters for substrate uptake
        (``v = vmax·S/(km+S)``, mmol/gDW/h).
    products : list[str], optional
        Exchange reactions to track as secreted products (e.g. acetate). If
        ``None``, the substrate and biomass are tracked and any positive-flux
        exchange is *not* auto-added (pass the ones you care about).
    product_0 : dict, optional
        Initial product concentrations (default 0).

    Returns
    -------
    pandas.DataFrame
        Columns ``time``, ``biomass``, ``substrate`` and one per product.
    """
    cobra = _cobra("dynamic_fba")
    import numpy as np
    import pandas as pd

    if substrate_exchange not in [r.id for r in model.reactions]:
        raise ValueError(f"substrate_exchange '{substrate_exchange}' 不在模型中。")
    products = list(products or [])
    for p in products:
        if p not in [r.id for r in model.reactions]:
            raise ValueError(f"product exchange '{p}' 不在模型中。")
    prod_conc = {p: float((product_0 or {}).get(p, 0.0)) for p in products}

    X = float(biomass_0)
    S = float(substrate_0)
    n_steps = int(round(t_end / dt))
    rows = []

    with model as m:
        sub_rxn = m.reactions.get_by_id(substrate_exchange)
        biomass = None
        for r in m.reactions:
            if r.objective_coefficient != 0:
                biomass = r
                break
        if biomass is None:
            raise ValueError("模型没有 biomass 目标反应。")

        for step in range(n_steps + 1):
            t = step * dt
            row = {"time": t, "biomass": X, "substrate": S}
            for p in products:
                row[_ex_label(p)] = prod_conc[p]
            rows.append(row)
            if S <= 1e-9 or X <= 1e-12:
                # substrate exhausted / washout — coast to the horizon
                continue

            # Michaelis–Menten uptake, capped by what is actually available
            v_uptake = vmax * S / (km + S)
            v_avail = S / (dt * X) if X > 0 else v_uptake
            v_uptake = min(v_uptake, v_avail)
            sub_rxn.lower_bound = -float(v_uptake)      # uptake is negative flux

            sol = m.optimize()
            if sol.status != "optimal" or sol.objective_value is None:
                continue
            mu = float(sol.objective_value)
            v_s = -float(sol.fluxes[substrate_exchange])   # >0 consumption
            # integrate (explicit Euler)
            dX = mu * X * dt
            dS = -v_s * X * dt
            for p in products:
                v_p = float(sol.fluxes[p])                 # >0 secretion
                prod_conc[p] = max(0.0, prod_conc[p] + v_p * X * dt)
            X = max(0.0, X + dX)
            S = max(0.0, S + dS)

    return pd.DataFrame(rows)


def _ex_label(rxn_id: str) -> str:
    """A short concentration label from an exchange reaction id."""
    lab = rxn_id
    if lab.startswith("EX_"):
        lab = lab[3:]
    return lab


@register_function(
    aliases=["plot_dynamic_fba", "dFBA图", "发酵曲线图", "plot_dfba",
             "dynamic_fba_plot", "批式发酵图"],
    category="synthetic_biology",
    description="画动态 FBA 的批式发酵时间曲线:生物量(左轴)与底物/产物浓度(右轴)对时间。Plot a dynamic-FBA batch time course (biomass + substrate/products vs time).",
    examples=["ov.synbio.plot_dynamic_fba(df)"],
    related=["synbio.dynamic_fba"],
    requires={},
    produces={},
)
def plot_dynamic_fba(df: "pd.DataFrame", ax=None):
    """Plot biomass (left axis) and substrate/products (right axis) vs time."""
    from ._plot import _mpl
    plt = _mpl()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
    else:
        fig = ax.figure
    ax.plot(df["time"], df["biomass"], color="#333333", lw=2.2, label="biomass")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("biomass (gDW/L)")
    ax2 = ax.twinx()
    metabolites = [c for c in df.columns if c not in ("time", "biomass")]
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(metabolites):
        ax2.plot(df["time"], df[c], color=cmap(i % 10), lw=1.8, ls="--", label=c)
    ax2.set_ylabel("metabolite (mmol/L)")
    # merged legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.set_title("Dynamic FBA — batch fermentation")
    fig.tight_layout()
    return fig, ax


__all__ = ["dynamic_fba", "plot_dynamic_fba"]
