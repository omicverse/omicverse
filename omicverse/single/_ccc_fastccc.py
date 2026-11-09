"""FastCCC backend for :func:`omicverse.single.run_ccc` (permutation-free CCC).

FastCCC (Hou, Ma & Zhou, *Nat Commun* 2025) scores every
sender-receiver-interaction triple of the CellPhoneDB database like
CellPhoneDB does (mean expression, complex = min of subunits, arithmetic
mean of ligand and receptor, >10% expressing cells) but replaces the label
permutations with the exact null law of a group mean, obtained by FFT
convolution of the gene's whole-dataset expression histogram (normal
approximation for groups of 30+ cells). It therefore needs no permutations
and returns the same p-values on every run. The engine lives in
``omicverse.external.fastccc`` and reproduces FastCCC 1.0.1 bit for bit.
"""
from __future__ import annotations

from collections.abc import Sequence

import anndata
import numpy as np
import pandas as pd

from .._registry import register_function
from ._cellchat import _get_matrix, _looks_like_counts

SCORE_SCALE = "expression"
NEEDS_SPATIAL = False

LONG_COLUMNS = ["source", "target", "ligand", "receptor", "ligand_genes", "receptor_genes", "interaction",
                "pathway", "annotation", "score", "pvalue", "significant"]

_ENGINE_KWARGS = ("single_unit_summary", "complex_aggregation", "LR_combination", "min_percentile", "cauchy",
                  "single_unit_summary_list", "complex_aggregation_list", "LR_combination_list", "use_DEG",
                  "convert_type", "select_list", "filter_")


def _resolve_db(cpdb_file_path):
    from ..external.fastccc import load_cpdb_tables
    from ._cpdb import download_cellphonedb_database

    if cpdb_file_path is None:
        cpdb_file_path = download_cellphonedb_database()
    return load_cpdb_tables(cpdb_file_path)


def _interaction_annotation(db, interactions: pd.DataFrame) -> pd.DataFrame:
    """Per-interaction names, subunit genes, pathway and direction."""
    names, genes = db.partner_names_and_genes()
    it = db.interaction_table.drop_duplicates("id_cp_interaction").set_index("id_cp_interaction")
    ann = pd.DataFrame(index=interactions.index.astype(str))
    m1, m2 = interactions["multidata_1_id"].to_numpy(), interactions["multidata_2_id"].to_numpy()
    ann["multidata_1_id"], ann["multidata_2_id"] = m1, m2
    ann["ligand"] = [names[m] for m in m1]
    ann["receptor"] = [names[m] for m in m2]
    ann["ligand_genes"] = ["_".join(genes[m]) for m in m1]
    ann["receptor_genes"] = ["_".join(genes[m]) for m in m2]
    ann["interaction"] = ann["ligand"] + "_" + ann["receptor"]
    sub = it.reindex(interactions.index)
    ann["pathway"] = sub["classification"].fillna("").astype(str).to_numpy()
    ann["annotation"] = sub["directionality"].fillna("").astype(str).to_numpy()
    return ann


def _long_table(res, ann: pd.DataFrame, pvalue_threshold: float, min_cells: int) -> pd.DataFrame:
    strength = res.strength.to_numpy(np.float64)
    pvals = np.clip(res.pvalues.to_numpy(np.float64), 0.0, 1.0)
    tested = res.tested.to_numpy(bool)
    pairs = [p.split("|") for p in res.strength.index]
    small = {g for g, n in res.group_sizes.items() if n <= min_cells}
    keep_pair = np.array([s not in small and r not in small for s, r in pairs])
    i, j = np.where(tested & (strength > 0) & keep_pair[:, None])
    a = ann.iloc[j]
    long = pd.DataFrame({
        "source": [pairs[k][0] for k in i],
        "target": [pairs[k][1] for k in i],
        "ligand": a["ligand"].to_numpy(),
        "receptor": a["receptor"].to_numpy(),
        "ligand_genes": a["ligand_genes"].to_numpy(),
        "receptor_genes": a["receptor_genes"].to_numpy(),
        "interaction": a["interaction"].to_numpy(),
        "pathway": a["pathway"].to_numpy(),
        "annotation": a["annotation"].to_numpy(),
        "score": strength[i, j],
        "pvalue": pvals[i, j],
    })
    long["significant"] = long["pvalue"].to_numpy() < pvalue_threshold
    return long[LONG_COLUMNS].reset_index(drop=True)


def run(adata, *, groupby, species="human", layer=None, min_cells=10, n_perms=None, pvalue_threshold=0.05,
        seed=0, use_gpu=False, spatial_key=None, progress=True, **method_kwargs) -> dict:
    """FastCCC backend of ``run_ccc``; returns ``{"raw", "long", "params"}``.

    Expects log1p-normalised expression (FastCCC's requirement; values must be
    below 14). ``n_perms``, ``seed`` and ``use_gpu`` are ignored: the p-values
    are analytic. Groups with ``<= min_cells`` cells are computed (the null
    uses every cell) but left out of ``long``, as for the other backends.

    ``method_kwargs``: ``cpdb_file_path`` (CellPhoneDB v5 ``cellphonedb.zip``;
    downloaded when ``None``), ``single_unit_summary`` (``'Mean'``, ``'Median'``,
    ``'Q3'``, ``'Quantile_0.9'`` ...), ``complex_aggregation`` (``'Minimum'`` /
    ``'Average'``), ``LR_combination`` (``'Arithmetic'`` / ``'Geometric'``),
    ``min_percentile`` (0.1), ``cauchy`` (run every branch of the ``*_list``
    arguments and Cauchy-combine), ``use_DEG``, ``convert_type``.
    """
    from ..external.fastccc import run_fastccc_engine

    unknown = set(method_kwargs) - set(_ENGINE_KWARGS) - {"cpdb_file_path"}
    if unknown:
        raise TypeError(f"Unknown FastCCC argument(s): {sorted(unknown)}.")
    if str(species).lower() != "human":
        raise ValueError("FastCCC uses the human CellPhoneDB database; map genes to human orthologs "
                         f"(species='{species}' is not supported).")
    if groupby not in adata.obs.columns:
        raise KeyError(f"`{groupby}` not found in adata.obs.")
    labels = adata.obs[groupby]
    if labels.isna().any():
        raise ValueError(f"adata.obs['{groupby}'] contains missing labels; drop or fill them first.")
    X = _get_matrix(adata, layer)
    if _looks_like_counts(X):
        raise ValueError("FastCCC expects log1p-normalised expression, but the selected matrix looks like raw "
                         "counts. Run ov.pp.normalize_total(adata, target_sum=1e4) and ov.pp.log1p(adata), "
                         "or pass `layer=`.")
    var_names = (adata.raw.var_names if layer == "raw" else adata.var_names).astype(str)

    cpdb_file_path = method_kwargs.pop("cpdb_file_path", None)
    db = _resolve_db(cpdb_file_path)
    engine_kwargs = {k: v for k, v in method_kwargs.items() if k in _ENGINE_KWARGS}
    res = run_fastccc_engine(X, labels.astype(str).to_numpy(), var_names, db, **engine_kwargs)
    ann = _interaction_annotation(db, res.interactions)
    long = _long_table(res, ann, pvalue_threshold, min_cells)

    params = {
        "method": "fastccc", "groupby": groupby, "species": "human",
        "database": "CellPhoneDB v5", "cpdb_file_path": str(db.source),
        "single_unit_summary": engine_kwargs.get("single_unit_summary", "Mean"),
        "complex_aggregation": engine_kwargs.get("complex_aggregation", "Minimum"),
        "LR_combination": engine_kwargs.get("LR_combination", "Arithmetic"),
        "min_percentile": float(engine_kwargs.get("min_percentile", 0.1)),
        "cauchy": bool(engine_kwargs.get("cauchy", False)),
        "use_DEG": bool(engine_kwargs.get("use_DEG", False)),
        "min_cells": int(min_cells), "pvalue_threshold": float(pvalue_threshold),
        "n_interactions": int(res.interactions.shape[0]),
    }
    raw = {
        "method": "fastccc",
        "groups": np.asarray(res.groups, dtype=object),
        "group_sizes": {str(k): int(v) for k, v in res.group_sizes.items()},
        "pairs": np.asarray(res.strength.index.astype(str), dtype=object),
        "interaction_ids": np.asarray(res.strength.columns.astype(str), dtype=object),
        "strength": res.strength.to_numpy(),
        "pvalues": res.pvalues.to_numpy(),
        "tested": res.tested.to_numpy(bool),
        "interactions": ann.reset_index(names="id_cp_interaction"),
        "parameters": params,
    }
    if res.branches:
        raw["branches"] = {k: v.loc[res.strength.index, res.strength.columns].to_numpy()
                           for k, v in res.branches.items()}
    return {"raw": raw, "long": long, "params": params}


@register_function(
    aliases=["FastCCC", "run_fastccc", "fastccc", "免置换细胞通讯", "FastCCC细胞通讯"],
    category="single",
    description=(
        "Permutation-free cell-cell communication (FastCCC): CellPhoneDB-style scores with analytic "
        "p-values from FFT-convolved null distributions; optional Cauchy combination of scoring branches."
    ),
    prerequisites={"functions": [], "optional_functions": ["pp.normalize_total", "pp.log1p"]},
    requires={"obs": ["groupby"]},
    produces={"uns": ["fastccc_res"]},
    auto_fix="none",
    examples=[
        "ov.single.run_fastccc(adata, groupby='cell_type', cpdb_file_path='cellphonedb.zip')",
        "ov.single.run_fastccc(adata, groupby='cell_type', cauchy=True)",
        "df = adata.uns['fastccc_res']['long']",
    ],
    related=["single.run_ccc", "single.run_cellphonedb_v5", "pl.ccc_heatmap"],
)
def run_fastccc(
    adata: anndata.AnnData,
    *,
    groupby: str,
    cpdb_file_path: str | None = None,
    layer: str | None = None,
    single_unit_summary: str = "Mean",
    complex_aggregation: str = "Minimum",
    LR_combination: str = "Arithmetic",
    min_percentile: float = 0.1,
    cauchy: bool = False,
    single_unit_summary_list: Sequence[str] = ("Mean", "Median", "Q3", "Quantile_0.9"),
    complex_aggregation_list: Sequence[str] = ("Minimum", "Average"),
    LR_combination_list: Sequence[str] = ("Arithmetic", "Geometric"),
    use_DEG: bool = False,
    min_cells: int = 10,
    pvalue_threshold: float = 0.05,
    key_added: str = "fastccc_res",
    inplace: bool = True,
):
    r"""Infer cell-cell communication with FastCCC (Hou et al., *Nat Commun* 2025).

    For each sender ``a``, receiver ``b`` and CellPhoneDB interaction ``l-r``
    the communication score is ``CS = h(s(x_{a,l}), s(x_{b,r}))``. The default
    (``'Mean'``, ``'Minimum'``, ``'Arithmetic'``) is CellPhoneDB's score; only
    triples whose ligand and receptor are detected in more than
    ``min_percentile`` of the cells of their group are tested. The p-value is
    the upper tail of the exact null law of ``CS`` for random groups of the same
    size, so no permutations are needed and results are deterministic.

    Parameters
    ----------
    adata
        AnnData with log1p-normalised expression (all genes; do not subset to HVGs).
    groupby
        ``adata.obs`` column with cell-group labels (must not contain ``'|'``).
    cpdb_file_path
        CellPhoneDB v5 ``cellphonedb.zip`` (or its extracted folder). Downloaded
        with :func:`download_cellphonedb_database` when ``None``.
    layer
        ``None`` (``adata.X``), a key of ``adata.layers`` or ``'raw'``.
    single_unit_summary
        ``'Mean'``, ``'Median'``/``'Q2'``, ``'Q3'`` or ``'Quantile_<q>'``.
    complex_aggregation
        ``'Minimum'`` or ``'Average'`` over complex subunits.
    LR_combination
        ``'Arithmetic'`` (zero unless both > 0) or ``'Geometric'`` mean.
    min_percentile
        Minimum expressing fraction; ``max(min_percentile, 1 - q)`` for quantiles.
    cauchy
        Run every combination of the three ``*_list`` arguments and combine
        their p-values with the Cauchy combination test; the score is the
        branch average.
    use_DEG
        Additionally require ligand and receptor to be individually above their
        null (FastCCC ``use_DEG``).
    min_cells
        Groups with ``<= min_cells`` cells are omitted from the long table.
    pvalue_threshold
        Significance cutoff for ``long['significant']``.
    key_added, inplace
        Store ``{"raw", "long", "params"}`` contents in ``adata.uns[key_added]``
        and return ``adata``; otherwise return the dict.

    Returns
    -------
    AnnData or dict
        ``uns[key_added]`` holds the full ``strength`` / ``pvalues`` / ``tested``
        matrices (``pairs`` x ``interaction_ids``), the interaction annotation and
        ``long``, the tested interactions with positive score.
    """
    kwargs = dict(cpdb_file_path=cpdb_file_path, single_unit_summary=single_unit_summary,
                  complex_aggregation=complex_aggregation, LR_combination=LR_combination,
                  min_percentile=min_percentile, cauchy=cauchy,
                  single_unit_summary_list=list(single_unit_summary_list),
                  complex_aggregation_list=list(complex_aggregation_list),
                  LR_combination_list=list(LR_combination_list), use_DEG=use_DEG)
    out = run(adata, groupby=groupby, layer=layer, min_cells=min_cells, pvalue_threshold=pvalue_threshold,
              **kwargs)
    result = dict(out["raw"])
    result["long"] = out["long"]
    if inplace:
        adata.uns[key_added] = result
        return adata
    return result


def db_pairs(species="human", cpdb_file_path=None, **_):
    """LR pairs of the CellPhoneDB database FastCCC used (for ``run_ccc(resource='shared')``)."""
    from ._ccc import _cpdb_db_pairs

    return _cpdb_db_pairs(species, cpdb_file_path=cpdb_file_path)
