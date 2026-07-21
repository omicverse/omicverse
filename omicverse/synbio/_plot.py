r"""Visualisation helpers for :mod:`omicverse.synbio`.

Keeps tutorial/notebook cells to a single call (the omicverse convention: plot
logic lives in the package, not inline in notebooks):

* :func:`view_structure` — interactive 3D (py3Dmol) coloured by pLDDT bands.
* :func:`plot_variant_effect` — deep-mutational-scan heatmap (position × AA).
* :func:`plot_enzyme_yield_response` — the A↔B coupling curve (k_cat → yield).
* :func:`plot_production_envelope` — growth vs product-yield trade-off (Layer A).

All third-party imports (matplotlib / py3Dmol) are inside the functions, so the
module imports with no extra installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

_AA20 = list("ACDEFGHIKLMNPQRSTVWY")

# AlphaFold / ESMFold pLDDT confidence bands (0–100 scale).
_PLDDT_BANDS = [
    (90, "#0053D6", "very high (>90)"),
    (70, "#65CBF3", "confident (70–90)"),
    (50, "#FFDB13", "low (50–70)"),
    (0, "#FF7D45", "very low (<50)"),
]


def _mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.synbio 绘图需要 matplotlib。请 pip install matplotlib "
            "(或 pip install 'omicverse[synbio]')。"
        ) from exc


def _pdb_text(structure) -> str:
    """Accept a StructurePrediction, a PDB path, or raw PDB text → PDB string."""
    import os
    pdb = getattr(structure, "pdb", None)
    if pdb is not None:
        return pdb
    if isinstance(structure, str):
        if os.path.exists(structure):
            with open(structure) as fh:
                return fh.read()
        if "ATOM" in structure:
            return structure
    raise ValueError("view_structure 需要 StructurePrediction、PDB 路径或 PDB 文本。")


def _ca_plddt(pdb_text: str):
    """Return {resid: plddt(0–100)} from CA B-factors (auto-rescales 0–1)."""
    vals = {}
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                resid = int(line[22:26])
                b = float(line[60:66])
                vals[resid] = b
            except ValueError:
                continue
    if vals and max(vals.values()) <= 1.0:      # ESMFold 0–1 scale
        vals = {k: v * 100.0 for k, v in vals.items()}
    return vals


@register_function(
    aliases=["view_structure", "结构可视化", "蛋白结构可视化", "view_pdb",
             "show_structure", "3d结构"],
    category="synthetic_biology",
    description="交互式 3D 结构可视化 (py3Dmol):按 pLDDT 置信度分带着色,内联渲染且在 nbconvert HTML 中保留。Interactive 3D structure view coloured by pLDDT.",
    examples=[
        "pred = ov.synbio.predict_structure(seq); ov.synbio.view_structure(pred)",
        "ov.synbio.view_structure('mypro.pdb', highlight=[23, 30])",
    ],
    related=["synbio.predict_structure", "synbio.stability_ddg"],
    requires={},
    produces={},
)
def view_structure(structure, color_by: str = "plddt",
                   highlight: Optional[Sequence[int]] = None,
                   width: int = 640, height: int = 480):
    """Interactive 3D cartoon of a predicted structure, coloured by pLDDT bands.

    Parameters
    ----------
    structure
        A :class:`StructurePrediction`, a PDB file path, or raw PDB text.
    color_by
        ``'plddt'`` (default, confidence bands) or ``'spectrum'``.
    highlight
        Residue ids to draw as orange sticks (e.g. mutated positions).
    width, height
        Viewport size in pixels.

    Returns
    -------
    py3Dmol.view
        Jupyter renders it as the cell's last expression.
    """
    try:
        import py3Dmol
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.synbio.view_structure 需要 py3Dmol。请 pip install py3Dmol "
            "(或 pip install 'omicverse[synbio]')。"
        ) from exc

    pdb = _pdb_text(structure)
    v = py3Dmol.view(width=width, height=height)
    v.addModel(pdb, "pdb")

    if color_by == "spectrum":
        v.setStyle({}, {"cartoon": {"color": "spectrum"}})
    else:
        plddt = _ca_plddt(pdb)
        v.setStyle({}, {"cartoon": {"color": "lightgrey"}})
        for cutoff, hexcol, _ in _PLDDT_BANDS:
            sel = [r for r, val in plddt.items() if val >= cutoff]
            if sel:
                v.setStyle({"resi": sel}, {"cartoon": {"color": hexcol}})
            # remove assigned residues so lower bands don't overwrite
            plddt = {r: val for r, val in plddt.items() if val < cutoff}

    if highlight:
        sel = {"resi": [int(r) for r in highlight]}
        v.addStyle(sel, {"stick": {"colorscheme": "orangeCarbon", "radius": 0.3}})

    v.zoomTo()
    return v


@register_function(
    aliases=["plot_method_comparison", "方法对比图", "算法对比图", "baseline_vs_sota",
             "plot_comparison", "对比柱状图"],
    category="synthetic_biology",
    description="baseline↔SOTA 方法对比柱状图:同一任务不同 method= 的结果并排(如 kcat/ΔG/脱靶/on-target 的基线 vs 真实模型)。Grouped bar chart comparing method= backends side by side.",
    examples=[
        "ov.synbio.plot_method_comparison(['PfkA'], {'baseline':[33.3],'dlkcat':[0.04]}, ylabel='kcat /s')",
    ],
    related=["synbio.enzyme_kcat", "synbio.reaction_dg", "synbio.offtarget_search"],
    requires={},
    produces={},
)
def plot_method_comparison(labels, series, ylabel: str = "value",
                           title: str = "baseline vs SOTA", ax=None,
                           log: bool = False):
    """Grouped bar chart comparing several ``method=`` backends on the same task.

    Parameters
    ----------
    labels
        Category names (x-axis) — e.g. the reactions, sites, or a single item.
    series
        ``{method_name: [values aligned to labels]}``; the first method is drawn
        as the "baseline" colour, the rest as SOTA colours.
    """
    import numpy as np
    plt = _mpl()

    methods = list(series.keys())
    n_lab = len(labels)
    n_met = len(methods)
    x = np.arange(n_lab)
    width = 0.8 / max(1, n_met)
    palette = ["#B0B0B0", "#DD8452", "#4C72B0", "#55A868", "#C44E52"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 1.3 * n_lab * n_met ** 0.5), 3.6))
    else:
        fig = ax.figure
    for i, mth in enumerate(methods):
        ax.bar(x + (i - (n_met - 1) / 2) * width, series[mth], width,
               label=mth, color=palette[i % len(palette)],
               edgecolor="k", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.axhline(0, c="grey", lw=0.8)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


@register_function(
    aliases=["plot_variant_effect", "突变热图", "饱和突变热图", "dms_heatmap",
             "变体效应图", "plot_dms"],
    category="synthetic_biology",
    description="深度突变扫描热图:把 variant_effect 的全饱和打分画成 位点×氨基酸 热图(红=有利/蓝=不利)。Deep-mutational-scan heatmap from a variant_effect table.",
    examples=[
        "df = ov.synbio.variant_effect(seq); ov.synbio.plot_variant_effect(df)",
    ],
    related=["synbio.variant_effect", "synbio.stability_ddg"],
    requires={},
    produces={},
)
def plot_variant_effect(df: "pd.DataFrame", value_col: str = "score",
                        figsize=None, cmap: str = "RdBu_r"):
    """Heatmap of a saturation scan: rows = amino acids, columns = positions.

    Accepts the output of :func:`variant_effect` (columns ``pos``, ``mut``,
    ``wt``, ``score``) or :func:`stability_ddg` (``ddg_proxy``). Wild-type cells
    are marked. Returns ``(fig, ax)``."""
    import numpy as np
    plt = _mpl()

    col = value_col
    if col not in df.columns:
        col = next((c for c in ("score", "ddg_proxy", "ddg") if c in df.columns),
                   df.columns[-1])
    positions = sorted(df["pos"].unique())
    pos_index = {p: i for i, p in enumerate(positions)}
    mat = np.full((len(_AA20), len(positions)), np.nan)
    for _, row in df.iterrows():
        if row["mut"] in _AA20:
            mat[_AA20.index(row["mut"]), pos_index[row["pos"]]] = row[col]

    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    figsize = figsize or (max(6, len(positions) * 0.16), 5)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_yticks(range(len(_AA20)))
    ax.set_yticklabels(_AA20, fontsize=7)
    step = max(1, len(positions) // 20)
    ax.set_xticks(range(0, len(positions), step))
    ax.set_xticklabels([positions[i] for i in range(0, len(positions), step)],
                       fontsize=7)
    ax.set_xlabel("residue position")
    ax.set_ylabel("substituted amino acid")

    # mark wild-type residue at each position
    wt_map = df.drop_duplicates("pos").set_index("pos")["wt"].to_dict()
    for p, wt in wt_map.items():
        if wt in _AA20:
            ax.scatter(pos_index[p], _AA20.index(wt), marker=".", c="k", s=6)

    label = {"score": "ESM score (Δ log-likelihood)",
             "ddg_proxy": "ΔΔG proxy", "ddg": "ΔΔG (kcal/mol)"}.get(col, col)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label=label)
    ax.set_title("Saturation scan  (black dot = wild-type)")
    fig.tight_layout()
    return fig, ax


@register_function(
    aliases=["plot_enzyme_yield_response", "酶得率曲线", "kcat得率曲线",
             "coupling_curve", "咬合曲线", "plot_coupling"],
    category="synthetic_biology",
    description="A↔B 咬合曲线:在固定蛋白池下扫描某反应的 kcat,画出 kcat → 生长/得率 的响应曲线(改酶→重算得率的可视化)。Plot k_cat → yield under a fixed enzyme budget.",
    examples=[
        "ov.synbio.plot_enzyme_yield_response(m, 'PFK', base_kcat=33.0)",
    ],
    related=["synbio.ec_model", "synbio.enzyme_kcat", "synbio.fba"],
    requires={},
    produces={},
)
def plot_enzyme_yield_response(model, reaction: str,
                              kcats: Optional[Sequence[float]] = None,
                              base_kcat: float = 10.0,
                              total_protein: Optional[float] = None,
                              ax=None):
    """Plot growth/yield as a function of an enzyme's k_cat under a fixed pool.

    For each k_cat, an enzyme-constrained model is built with the *same* protein
    budget and re-optimised — the visual proof that a faster enzyme raises the
    attainable yield. Returns ``(fig_or_ax, DataFrame)``."""
    import numpy as np
    import pandas as pd
    from ._ec import ec_model
    from ._gem import fba
    plt = _mpl()

    if kcats is None:
        kcats = list(np.round(base_kcat * np.array(
            [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]), 3))
    if total_protein is None:
        ref = ec_model(model, {reaction: base_kcat})
        total_protein = ref.synbio_ec["total_protein"]

    rows = []
    for k in kcats:
        ecm = ec_model(model, {reaction: float(k)}, total_protein=total_protein)
        g = fba(ecm).objective_value
        rows.append((float(k), float(g)))
    dfr = pd.DataFrame(rows, columns=["kcat", "growth"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    else:
        fig = ax.figure
    ax.plot(dfr["kcat"], dfr["growth"], "o-", color="#377EB8")
    ax.axvline(base_kcat, ls="--", c="grey", lw=1, label=f"predicted kcat={base_kcat:g}")
    ax.set_xscale("log")
    ax.set_xlabel(f"k$_{{cat}}$ of {reaction}  (1/s, log scale)")
    ax.set_ylabel("max growth  (1/h)")
    ax.set_title(f"Enzyme edit → yield  (fixed protein budget)\n{reaction}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, dfr


@register_function(
    aliases=["plot_production_envelope", "生产包络图", "得率生长权衡图",
             "production_envelope_plot"],
    category="synthetic_biology",
    description="生产包络图:画出目标产物得率随生长速率变化的可行区间(生长-得率权衡)。Plot the growth vs product-yield production envelope.",
    examples=["ov.synbio.plot_production_envelope(m, 'EX_ac_e')"],
    related=["synbio.production_envelope", "synbio.strain_design"],
    requires={},
    produces={},
)
def plot_production_envelope(model, target: str, ax=None):
    """Plot the production envelope (max/min product flux vs growth rate)."""
    from ._strain import production_envelope
    plt = _mpl()

    # cobra varies the target flux and reports the objective (growth) bounds:
    # env[target] is the product-flux grid, flux_minimum/maximum are growth.
    env = production_envelope(model, target)
    prod = env[target] if target in env.columns else env.iloc[:, 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    else:
        fig = ax.figure
    ax.fill_between(prod, env["flux_minimum"], env["flux_maximum"],
                    alpha=0.3, color="#4DAF4A", label="feasible growth range")
    ax.plot(prod, env["flux_maximum"], "-", color="#4DAF4A")
    ax.plot(prod, env["flux_minimum"], "--", color="#4DAF4A")
    ax.set_xlabel(f"{target} flux  (product)")
    ax.set_ylabel("growth rate (1/h)")
    ax.set_title(f"Production envelope: {target}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, env


__all__ = ["view_structure", "plot_variant_effect",
           "plot_enzyme_yield_response", "plot_production_envelope"]
