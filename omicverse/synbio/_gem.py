r"""Layer A — genome-scale metabolic models (GEMs) via COBRApy.

Constraint-based reconstruction and analysis: read a model (local SBML or a
BiGG identifier such as ``e_coli_core`` / ``iML1515``), run FBA/FVA/pFBA, and
screen single / double gene knockouts.  All CPU, all built on
`COBRApy <https://github.com/opencobra/cobrapy>`_ with the free GLPK solver
(via optlang).

Nothing here touches AnnData — the currency of this layer is a
:class:`cobra.Model`.  These functions register into the omicOS function index
under the ``synthetic_biology`` category with empty ``requires`` / ``produces``
(the AnnData-slot contract does not apply), exactly like the ``fetch_*``
data-loaders in :mod:`omicverse.micro`.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import cobra
    import pandas as pd


# ---------------------------------------------------------------------------
# dependency gate + cache
# ---------------------------------------------------------------------------

_MISSING = (
    "ov.synbio.{fn} 需要额外依赖 COBRApy。请 pip install 'omicverse[synbio]' "
    "(或直接 pip install cobra optlang)。"
)


def _cobra(fn: str):
    """Import COBRApy on demand with an actionable error."""
    try:
        import cobra
        return cobra
    except ImportError as exc:  # pragma: no cover - exercised only w/o extra
        raise ImportError(_MISSING.format(fn=fn)) from exc


def _gem_cache_dir() -> str:
    """``~/.omicverse/synbio_gems`` (override with ``OMICOS_SYNBIO_GEM_DIR``)."""
    root = os.environ.get(
        "OMICOS_SYNBIO_GEM_DIR",
        os.path.join(os.path.expanduser("~"), ".omicverse", "synbio_gems"),
    )
    os.makedirs(root, exist_ok=True)
    return root


# BiGG models that ship bundled with COBRApy (no download needed).
_COBRA_BUNDLED = {"e_coli_core", "textbook", "salmonella", "iJO1366", "mini"}

_BIGG_URL = "http://bigg.ucsd.edu/static/models/{model_id}.xml"


# ---------------------------------------------------------------------------
# load_gem
# ---------------------------------------------------------------------------

@register_function(
    aliases=[
        "load_gem", "代谢模型", "基因组尺度代谢模型", "读取GEM",
        "read_sbml", "cobra_model", "genome_scale_model",
    ],
    category="synthetic_biology",
    description="加载基因组尺度代谢模型 (GEM):本地 SBML 文件或 BiGG 模型 id (如 e_coli_core / iML1515) → cobra.Model。Load a genome-scale metabolic model from an SBML path or BiGG id.",
    examples=[
        "m = ov.synbio.load_gem('e_coli_core')",
        "m = ov.synbio.load_gem('iML1515')",
        "m = ov.synbio.load_gem('/path/to/model.xml')",
    ],
    related=["synbio.fba", "synbio.fva", "synbio.strain_design", "synbio.ec_model"],
    requires={},
    produces={},
)
def load_gem(path_or_id: str, cache_dir: Optional[str] = None) -> "cobra.Model":
    """Load a genome-scale metabolic model.

    Parameters
    ----------
    path_or_id
        Either a path to a local SBML file (``.xml``/``.sbml``/``.json``/
        ``.mat``/``.yml``) or a BiGG model identifier (e.g. ``"e_coli_core"``,
        ``"iML1515"``, ``"iCW773"``).  Bundled test models load offline;
        other BiGG ids are downloaded once to the cache directory.
    cache_dir
        Where downloaded BiGG SBMLs are stored (default
        ``~/.omicverse/synbio_gems``).

    Returns
    -------
    cobra.Model
    """
    cobra = _cobra("load_gem")

    # 1) an existing local file — dispatch on extension.
    if os.path.exists(path_or_id):
        ext = os.path.splitext(path_or_id)[1].lower()
        if ext == ".json":
            return cobra.io.load_json_model(path_or_id)
        if ext in (".mat",):
            return cobra.io.load_matlab_model(path_or_id)
        if ext in (".yml", ".yaml"):
            return cobra.io.load_yaml_model(path_or_id)
        return cobra.io.read_sbml_model(path_or_id)

    # 2) a model bundled inside COBRApy's test data.
    if path_or_id in _COBRA_BUNDLED:
        try:
            from cobra.io import load_model
            return load_model(path_or_id)
        except Exception:
            # older cobra: fall back to the bundled sample data path
            import cobra.test  # type: ignore

            return cobra.test.create_test_model(
                "textbook" if path_or_id in ("e_coli_core", "textbook") else path_or_id
            )

    # 3) a BiGG id — try cobra's built-in loader (fetches + caches), then a
    #    manual download to our cache dir.
    cache_dir = cache_dir or _gem_cache_dir()
    local = os.path.join(cache_dir, f"{path_or_id}.xml")
    if os.path.exists(local):
        return cobra.io.read_sbml_model(local)

    try:
        from cobra.io import load_model
        return load_model(path_or_id)  # handles BiGG + BioModels ids
    except Exception:
        pass

    # manual fetch from BiGG
    import urllib.request

    url = _BIGG_URL.format(model_id=path_or_id)
    try:
        urllib.request.urlretrieve(url, local)
    except Exception as exc:
        raise ValueError(
            f"无法加载 GEM '{path_or_id}':既不是本地文件,也不是可识别的 BiGG id。"
            f" 尝试下载 {url} 失败 ({exc})。请检查 id 或提供本地 SBML 路径。"
        ) from exc
    return cobra.io.read_sbml_model(local)


# ---------------------------------------------------------------------------
# fba
# ---------------------------------------------------------------------------

@register_function(
    aliases=["fba", "通量平衡分析", "flux_balance_analysis", "optimize", "通量分析"],
    category="synthetic_biology",
    description="通量平衡分析 (FBA):在稳态与化学计量约束下最大化目标反应(默认生物量),返回最优目标值与各反应通量。Flux Balance Analysis on a cobra.Model.",
    examples=[
        "sol = ov.synbio.fba(m)",
        "sol = ov.synbio.fba(m, objective='BIOMASS_Ecoli_core_w_GAM')",
    ],
    related=["synbio.load_gem", "synbio.pfba", "synbio.fva"],
    requires={},
    produces={},
)
def fba(
    model: "cobra.Model",
    objective: Optional[str] = None,
    fraction_of_optimum: float = 1.0,
) -> "cobra.core.Solution":
    """Run Flux Balance Analysis.

    Parameters
    ----------
    model
        A :class:`cobra.Model`.
    objective
        Reaction id to maximise; defaults to the model's existing objective
        (usually biomass).
    fraction_of_optimum
        If < 1, the objective is constrained to this fraction of its optimum
        (useful when probing sub-optimal flux distributions).

    Returns
    -------
    cobra.core.Solution
        ``.objective_value`` is the growth/production rate; ``.fluxes`` is a
        pandas Series over all reactions.
    """
    _cobra("fba")
    with model as m:
        if objective is not None:
            m.objective = objective
        sol = m.optimize()
        if fraction_of_optimum < 1.0 and sol.status == "optimal":
            obj_expr = m.objective.expression
            m.add_cons_vars(
                m.problem.Constraint(
                    obj_expr,
                    lb=fraction_of_optimum * sol.objective_value,
                    name="fba_fraction",
                )
            )
            sol = m.optimize()
    return sol


@register_function(
    aliases=["pfba", "简约FBA", "parsimonious_fba", "最小总通量"],
    category="synthetic_biology",
    description="简约 FBA (pFBA):在最优目标下最小化总通量,得到唯一且更符合生理的通量分布。Parsimonious FBA — minimal total flux at the FBA optimum.",
    examples=["sol = ov.synbio.pfba(m)"],
    related=["synbio.fba", "synbio.fva"],
    requires={},
    produces={},
)
def pfba(model: "cobra.Model", objective: Optional[str] = None) -> "cobra.core.Solution":
    """Parsimonious FBA — the flux distribution with minimal total flux at the
    growth optimum (a proxy for minimal enzyme usage)."""
    cobra = _cobra("pfba")
    with model as m:
        if objective is not None:
            m.objective = objective
        return cobra.flux_analysis.pfba(m)


# ---------------------------------------------------------------------------
# fva
# ---------------------------------------------------------------------------

@register_function(
    aliases=["fva", "通量可变性分析", "flux_variability_analysis", "通量范围"],
    category="synthetic_biology",
    description="通量可变性分析 (FVA):在(接近)最优生长下,计算每个反应通量的可行区间 [min,max]。Flux Variability Analysis → DataFrame of per-reaction min/max flux.",
    examples=[
        "df = ov.synbio.fva(m, fraction_of_optimum=0.9)",
        "df = ov.synbio.fva(m, reaction_list=['PGI','PFK'])",
    ],
    related=["synbio.fba", "synbio.pfba"],
    requires={},
    produces={},
)
def fva(
    model: "cobra.Model",
    reaction_list: Optional[Sequence[str]] = None,
    fraction_of_optimum: float = 1.0,
    loopless: bool = False,
) -> "pd.DataFrame":
    """Flux Variability Analysis.

    Returns a DataFrame indexed by reaction id with ``minimum`` / ``maximum``
    columns — the range each reaction's flux can take while the objective stays
    at ``fraction_of_optimum`` of its maximum.
    """
    cobra = _cobra("fva")
    rxns = None
    if reaction_list is not None:
        rxns = [model.reactions.get_by_id(r) if isinstance(r, str) else r
                for r in reaction_list]
    return cobra.flux_analysis.flux_variability_analysis(
        model, reaction_list=rxns,
        fraction_of_optimum=fraction_of_optimum, loopless=loopless,
    )


# ---------------------------------------------------------------------------
# gene deletions
# ---------------------------------------------------------------------------

@register_function(
    aliases=[
        "single_gene_deletion", "单基因敲除", "基因敲除", "gene_knockout",
        "single_ko",
    ],
    category="synthetic_biology",
    description="单基因敲除扫描:逐个敲除基因并用 FBA 重算生长,返回每个敲除后的生长速率。Single-gene deletion scan → DataFrame of post-KO growth.",
    examples=["df = ov.synbio.single_gene_deletion(m)"],
    related=["synbio.double_gene_deletion", "synbio.strain_design"],
    requires={},
    produces={},
)
def single_gene_deletion(
    model: "cobra.Model",
    gene_list: Optional[Sequence[str]] = None,
    method: str = "fba",
) -> "pd.DataFrame":
    """Knock out each gene (optionally a subset) and recompute growth by FBA."""
    cobra = _cobra("single_gene_deletion")
    return cobra.flux_analysis.single_gene_deletion(
        model, gene_list=gene_list, method=method
    )


@register_function(
    aliases=[
        "double_gene_deletion", "双基因敲除", "double_ko", "synthetic_lethal",
        "合成致死",
    ],
    category="synthetic_biology",
    description="双基因敲除扫描:成对敲除基因并重算生长,用于发现合成致死/上位效应。Double-gene deletion scan → DataFrame of post-KO growth for gene pairs.",
    examples=["df = ov.synbio.double_gene_deletion(m, gene_list=['b0008','b0114'])"],
    related=["synbio.single_gene_deletion"],
    requires={},
    produces={},
)
def double_gene_deletion(
    model: "cobra.Model",
    gene_list: Optional[Sequence[str]] = None,
    gene_list2: Optional[Sequence[str]] = None,
    method: str = "fba",
    number_of_processes: Optional[int] = None,
) -> "pd.DataFrame":
    """Knock out gene pairs and recompute growth (synthetic-lethality screen).

    ``gene_list`` defaults to all genes — pass a subset for large models, the
    full pairwise scan is O(n²)."""
    cobra = _cobra("double_gene_deletion")
    return cobra.flux_analysis.double_gene_deletion(
        model, gene_list1=gene_list, gene_list2=gene_list2,
        method=method, processes=number_of_processes,
    )


__all__ = [
    "load_gem", "fba", "pfba", "fva",
    "single_gene_deletion", "double_gene_deletion",
]
