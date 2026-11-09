"""Unified cell-cell communication entry point: ``ov.single.run_ccc(method=...)``.

Every backend is reduced to one *long table* with the columns in
:data:`LONG_COLUMNS` (one row per sender, receiver and ligand-receptor pair).
The long table is what the consensus, the method comparison and the
``ov.pl.ccc_*`` communication AnnData are built from, so a new method only has
to provide an adapter.

Backend interface
-----------------
An adapter module exposes ``run(adata, *, groupby, species, layer, min_cells,
n_perms, pvalue_threshold, seed, use_gpu, spatial_key, progress, **kwargs)``
returning ``{"raw": ..., "long": DataFrame, "params": dict}`` (optionally
``"obsm"`` / ``"obsp"`` dicts of per-cell results), plus the module constants
``SCORE_SCALE`` and ``NEEDS_SPATIAL``. ``ligand_genes`` / ``receptor_genes``
hold the sorted subunit gene symbols joined by ``"_"`` and are the key that
aligns interactions across databases.
"""
from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._registry import register_function

LONG_COLUMNS = ["source", "target", "ligand", "receptor", "ligand_genes", "receptor_genes", "interaction",
                "pathway", "annotation", "score", "pvalue", "significant"]
_KEY = ["source", "target", "ligand_genes", "receptor_genes"]
_ALIASES = {"cpdb": "cellphonedb", "cellphone": "cellphonedb", "cellphonedb_v5": "cellphonedb",
            "liana+": "liana", "lianaplus": "liana", "fast_ccc": "fastccc", "sec_act": "secact",
            "cyto_signal": "cytosignal"}
# Display-name priority when several methods describe the same LR pair.
_NAME_PRIORITY = ("cellchat", "cellphonedb", "liana", "fastccc", "cytosignal", "commot", "secact")


@dataclass(frozen=True)
class CCCBackend:
    """A registered communication method."""

    name: str
    run: Callable
    score_scale: str
    needs_spatial: bool = False
    legacy_key: str | None = None
    db_pairs: Callable | None = None
    description: str = ""


_REGISTRY: dict[str, CCCBackend] = {}
# Backends contributed by optional adapter modules, loaded on first use.
_LAZY = {"fastccc": "._ccc_fastccc", "secact": "._ccc_secact", "cytosignal": "._ccc_cytosignal"}


def register_ccc_method(name: str, run: Callable, *, score_scale: str, needs_spatial: bool = False,
                        legacy_key: str | None = None, db_pairs: Callable | None = None,
                        description: str = "") -> None:
    """Register a communication backend for :func:`run_ccc` (see module docstring)."""
    _REGISTRY[name] = CCCBackend(name, run, score_scale, needs_spatial, legacy_key, db_pairs, description)


def _canonical(method: str) -> str:
    key = str(method).strip().lower()
    return _ALIASES.get(key, key)


def _get_backend(method: str) -> CCCBackend:
    name = _canonical(method)
    if name not in _REGISTRY and name in _LAZY:
        module = importlib.import_module(_LAZY[name], package=__package__)
        register_ccc_method(name, module.run, score_scale=getattr(module, "SCORE_SCALE", "expression"),
                            needs_spatial=bool(getattr(module, "NEEDS_SPATIAL", False)),
                            legacy_key=getattr(module, "LEGACY_KEY", f"{name}_res"),
                            db_pairs=getattr(module, "db_pairs", None),
                            description=(module.__doc__ or "").strip().splitlines()[0] if module.__doc__ else "")
    if name not in _REGISTRY:
        raise ValueError(f"Unknown CCC method '{method}'. Available: {available_ccc_methods()}.")
    return _REGISTRY[name]


def available_ccc_methods() -> list[str]:
    """Names accepted by ``run_ccc(method=...)``."""
    return sorted(set(_REGISTRY) | set(_LAZY))


# ----------------------------------------------------------------------------- helpers
def _gene_key(genes) -> str:
    return "_".join(sorted({str(g) for g in genes if str(g) and str(g) != "nan"}))


def _norm_key(key: str) -> str:
    """Case-insensitive gene key: mouse symbols (``Nrg3``) match human-style ones (``NRG3``)."""
    return "_".join(sorted({g.upper() for g in str(key).split("_") if g})) if key else ""


def _norm_keys(values) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=object))
    uniq = pd.unique(s)
    return s.map(dict(zip(uniq, (_norm_key(u) for u in uniq)))).to_numpy(dtype=object)


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float if c in ("score", "pvalue") else
                                      bool if c == "significant" else object) for c in LONG_COLUMNS})


def _validate_long(long: pd.DataFrame, method: str) -> pd.DataFrame:
    missing = [c for c in LONG_COLUMNS if c not in long.columns]
    if missing:
        raise ValueError(f"Backend '{method}' long table lacks columns {missing}.")
    long = long[LONG_COLUMNS].copy()
    for col in LONG_COLUMNS:
        if col in ("score", "pvalue"):
            long[col] = pd.to_numeric(long[col], errors="coerce").astype(float)
        elif col == "significant":
            long[col] = long[col].fillna(False).astype(bool)
        else:
            long[col] = np.asarray(long[col].astype(object).where(long[col].notna(), "").astype(str), dtype=object)
    return long.reset_index(drop=True)


def _get_matrix(adata: anndata.AnnData, layer: str | None):
    from ._cellchat import _get_matrix as get
    return get(adata, layer)


def _minimal_adata(adata: anndata.AnnData, layer: str | None, groupby: str, keep_groups=None,
                   spatial_key: str | None = None) -> anndata.AnnData:
    """Expression + labels (+ coordinates) only, so wrapped tools never touch the user's object."""
    X = _get_matrix(adata, layer)
    var_names = adata.raw.var_names if layer == "raw" else adata.var_names
    obs = pd.DataFrame({groupby: adata.obs[groupby].astype(str).to_numpy()}, index=adata.obs_names.copy())
    mask = np.ones(adata.n_obs, dtype=bool) if keep_groups is None else obs[groupby].isin(keep_groups).to_numpy()
    out = anndata.AnnData(X=X[mask] if not sp.issparse(X) else sp.csr_matrix(X)[mask],
                          obs=obs.loc[mask], var=pd.DataFrame(index=pd.Index(var_names.astype(str))))
    out.obs[groupby] = pd.Categorical(out.obs[groupby])
    if spatial_key is not None and spatial_key in adata.obsm:
        out.obsm[spatial_key] = np.asarray(adata.obsm[spatial_key])[mask]
    out.var_names_make_unique()
    return out


def _kept_groups(adata: anndata.AnnData, groupby: str, min_cells: int) -> list[str]:
    counts = adata.obs[groupby].astype(str).value_counts()
    keep = sorted(counts.index[counts > min_cells])
    if len(keep) < 2:
        raise ValueError(f"Fewer than two groups in '{groupby}' have more than min_cells={min_cells} cells.")
    return keep


def _uns_safe(value):
    """Recursively turn a result into something ``write_h5ad`` accepts."""
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        frame.columns = [str(c) for c in frame.columns]
        for col in frame.columns:
            s = frame[col]
            if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
                frame[col] = np.asarray(s.astype(object).where(s.notna(), "").astype(str), dtype=object)
        frame.index = np.asarray(frame.index.astype(str), dtype=object)
        return frame
    if isinstance(value, Mapping):
        return {str(k): _uns_safe(v) for k, v in value.items() if v is not pd.DataFrame and v is not None}
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        return np.asarray(value, dtype=object)
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        return value.astype(str).astype(object)
    return value


# ----------------------------------------------------------------------------- pathways
_PATHWAY_CACHE: dict[str, tuple[dict, dict]] = {}


def _cellchat_pathway_maps(species: str):
    if species not in _PATHWAY_CACHE:
        from ..external.cellchat._db import SPECIES, load_cellchat_db
        if species not in SPECIES:
            _PATHWAY_CACHE[species] = ({}, {})
        else:
            db = load_cellchat_db(species)
            sub = db.complex_subunits()
            it = db.interaction
            lg = [_norm_key(_gene_key(sub.get(x, [x]))) for x in it["ligand"]]
            rg = [_norm_key(_gene_key(sub.get(x, [x]))) for x in it["receptor"]]
            pair = {}
            for a, b, p in zip(lg, rg, it["pathway_name"]):
                pair.setdefault((a, b), p)
            lig = pd.Series(it["pathway_name"].to_numpy(), index=lg)
            lig_map = lig.groupby(level=0).agg(lambda s: s.value_counts().index[0]).to_dict()
            _PATHWAY_CACHE[species] = (pair, lig_map)
    return _PATHWAY_CACHE[species]


def annotate_pathways(long: pd.DataFrame, species: str = "human") -> pd.DataFrame:
    """Fill empty ``pathway`` entries from CellChatDB (exact LR pair, then ligand; case-insensitive)."""
    long = long.copy()
    empty = long["pathway"].astype(str).isin(["", "nan", "None"])
    if empty.any():
        pair, lig = _cellchat_pathway_maps(species)
        keys = list(zip(_norm_keys(long.loc[empty, "ligand_genes"]), _norm_keys(long.loc[empty, "receptor_genes"])))
        filled = [pair.get(k) or lig.get(k[0]) or "Unclassified" for k in keys]
        long.loc[empty, "pathway"] = filled
    return long


# ----------------------------------------------------------------------------- built-in adapters
def _cellchat_db_pairs(species: str = "human", **_) -> set:
    from ..external.cellchat._db import load_cellchat_db
    db = load_cellchat_db(species)
    sub = db.complex_subunits()
    return {(_gene_key(sub.get(a, [a])), _gene_key(sub.get(b, [b])))
            for a, b in zip(db.interaction["ligand"], db.interaction["receptor"])}


def _run_cellchat_backend(adata, *, groupby, species="human", layer=None, min_cells=10, n_perms=None,
                          pvalue_threshold=0.05, seed=0, use_gpu=False, spatial_key=None, progress=True, **kw):
    from ..external.cellchat._db import load_cellchat_db
    from ._cellchat import run_cellchat

    n_perms = 100 if n_perms is None else int(n_perms)
    res = run_cellchat(adata, groupby=groupby, species=species, layer=layer, n_perms=n_perms, seed=seed,
                       min_cells=min_cells, pvalue_threshold=pvalue_threshold, use_gpu=use_gpu,
                       progress=progress, inplace=False, **kw)
    prob, pval = np.asarray(res["prob"]), np.asarray(res["pval"])
    groups = np.asarray([str(g) for g in res["groups"]], dtype=object)
    lr = pd.DataFrame(res["LRsig"])
    sub = load_cellchat_db(species).complex_subunits()
    s, r, k = np.nonzero(prob > 0)
    lg = np.asarray([_gene_key(sub.get(x, [x])) for x in lr["ligand"]], dtype=object)
    rg = np.asarray([_gene_key(sub.get(x, [x])) for x in lr["receptor"]], dtype=object)
    long = pd.DataFrame({
        "source": groups[s], "target": groups[r],
        "ligand": lr["ligand"].to_numpy()[k], "receptor": lr["receptor"].to_numpy()[k],
        "ligand_genes": lg[k], "receptor_genes": rg[k],
        "interaction": lr["interaction_name_2"].to_numpy()[k] if "interaction_name_2" in lr else lr["interaction_name"].to_numpy()[k],
        "pathway": lr["pathway_name"].to_numpy()[k],
        "annotation": lr["annotation"].to_numpy()[k] if "annotation" in lr else "",
        "score": prob[s, r, k], "pvalue": pval[s, r, k],
    })
    long["significant"] = long["pvalue"] < pvalue_threshold
    return {"raw": res, "long": long, "params": dict(res["parameters"])}


def _cpdb_db_pairs(species: str = "human", cpdb_file_path: str | None = None, **_) -> set:
    import zipfile
    if cpdb_file_path is None:
        return set()
    zf = zipfile.ZipFile(cpdb_file_path)
    md = pd.read_csv(zf.open("multidata_table.csv"))
    comp = pd.read_csv(zf.open("complex_composition_table.csv"))
    gene = pd.read_csv(zf.open("gene_table.csv"))
    it = pd.read_csv(zf.open("interaction_table.csv"))
    pmd2gene = gene.drop_duplicates("protein_id").set_index("protein_id")["gene_name"]  # protein_id = id_multidata
    cplx = {k: _gene_key(pmd2gene.reindex(v["protein_multidata_id"]).dropna())
            for k, v in comp.groupby("complex_multidata_id")}
    mdx = md.set_index("id_multidata")

    def genes(i):
        if bool(mdx.loc[i, "is_complex"]):
            return cplx.get(i, "")
        return _gene_key([pmd2gene.get(i, mdx.loc[i, "name"])])

    rec = mdx["receptor"].astype(bool) if "receptor" in mdx else pd.Series(False, index=mdx.index)
    out = set()
    for a, b in zip(it["multidata_1_id"], it["multidata_2_id"]):
        ga, gb = genes(a), genes(b)
        out.add((gb, ga) if rec.get(a, False) and not rec.get(b, False) else (ga, gb))
    return out


def _cpdb_partner_genes(results) -> dict:
    """``partner`` (``simple:UNIPROT`` / ``complex:NAME``) -> sorted gene key, from the deconvoluted table."""
    dec = results.get("deconvoluted")
    out = {}
    if isinstance(dec, pd.DataFrame) and "gene_name" in dec:
        cpx = dec[dec["is_complex"].astype(bool)] if "is_complex" in dec else dec.iloc[:0]
        for name, rows in cpx.groupby("complex_name"):
            out[f"complex:{name}"] = _gene_key(rows["gene_name"])
        simple = dec[~dec["is_complex"].astype(bool)] if "is_complex" in dec else dec
        for uni, g in zip(simple["uniprot"], simple["gene_name"]):
            out.setdefault(f"simple:{uni}", _gene_key([g]))
    return out


def _run_cellphonedb_backend(adata, *, groupby, species="human", layer=None, min_cells=10, n_perms=None,
                             pvalue_threshold=0.05, seed=0, use_gpu=False, spatial_key=None, progress=True,
                             cpdb_file_path=None, gene_min_cells=3, min_genes=200, threshold=0.1, threads=4,
                             **kw):
    from ._cpdb import download_cellphonedb_database, run_cellphonedb_v5

    if species != "human":
        raise ValueError("CellPhoneDB v5 is a human database; use species='human' (or method='cellchat'/'liana').")
    if cpdb_file_path is None:
        cpdb_file_path = download_cellphonedb_database()
    keep = _kept_groups(adata, groupby, min_cells)
    work = _minimal_adata(adata, layer, groupby, keep)
    results, _ = run_cellphonedb_v5(
        work, cpdb_file_path=cpdb_file_path, celltype_key=groupby, min_cell_fraction=0.0, min_genes=min_genes,
        min_cells=gene_min_cells, iterations=1000 if n_perms is None else int(n_perms), threshold=threshold,
        pvalue=pvalue_threshold, threads=threads, debug_seed=seed, **kw)
    means, pvals = results["means"], results.get("pvalues")
    pair_cols = [c for c in means.columns if "|" in str(c)]
    partner = _cpdb_partner_genes(results)
    info = means.drop(columns=pair_cols)
    if pvals is not None:
        pv = pvals.set_index("id_cp_interaction").reindex(info["id_cp_interaction"])[pair_cols].to_numpy(float)
    else:  # DEGs-based method: relevance only
        rel = results["relevant_interactions"].set_index("id_cp_interaction").reindex(info["id_cp_interaction"])
        pv = np.where(rel[pair_cols].fillna(0).to_numpy(float) > 0, 0.0, 1.0)
    mv = means[pair_cols].to_numpy(float)

    def genes_of(p, g):
        if isinstance(p, str) and p in partner:
            return partner[p]
        return _gene_key([g]) if isinstance(g, str) and g else ""

    ga = np.asarray([genes_of(p, g) for p, g in zip(info["partner_a"], info.get("gene_a", pd.Series([""] * len(info))))], dtype=object)
    gb = np.asarray([genes_of(p, g) for p, g in zip(info["partner_b"], info.get("gene_b", pd.Series([""] * len(info))))], dtype=object)
    na = np.asarray([str(g) if isinstance(g, str) and g else str(p).split(":", 1)[-1]
                     for p, g in zip(info["partner_a"], info.get("gene_a", info["partner_a"]))], dtype=object)
    nb = np.asarray([str(g) if isinstance(g, str) and g else str(p).split(":", 1)[-1]
                     for p, g in zip(info["partner_b"], info.get("gene_b", info["partner_b"]))], dtype=object)
    # Partner A expresses in the first cell type of "A|B"; orient receptor-first rows ligand -> receptor.
    flip = (info.get("receptor_a", False).astype(bool) & ~info.get("receptor_b", False).astype(bool)).to_numpy() \
        if "receptor_a" in info else np.zeros(len(info), bool)
    i, j = np.nonzero(mv > 0)
    a_cell = np.asarray([c.split("|")[0] for c in pair_cols], dtype=object)
    b_cell = np.asarray([c.split("|")[1] for c in pair_cols], dtype=object)
    f = flip[i]
    long = pd.DataFrame({
        "source": np.where(f, b_cell[j], a_cell[j]), "target": np.where(f, a_cell[j], b_cell[j]),
        "ligand": np.where(f, nb[i], na[i]), "receptor": np.where(f, na[i], nb[i]),
        "ligand_genes": np.where(f, gb[i], ga[i]), "receptor_genes": np.where(f, ga[i], gb[i]),
        "interaction": info["interacting_pair"].to_numpy()[i],
        "pathway": info["classification"].fillna("").to_numpy()[i] if "classification" in info else "",
        "annotation": info["directionality"].fillna("").to_numpy()[i] if "directionality" in info else "",
        "score": mv[i, j], "pvalue": pv[i, j],
    })
    long["significant"] = long["pvalue"] < pvalue_threshold
    long["pathway"] = long["pathway"].astype(str).str.replace(r"^Signaling by ", "", regex=True)
    params = {"cpdb_file_path": str(cpdb_file_path), "iterations": 1000 if n_perms is None else int(n_perms),
              "threshold": float(threshold), "min_genes": int(min_genes), "gene_min_cells": int(gene_min_cells),
              "groups": keep}
    return {"raw": results, "long": long, "params": params}


def _liana_db_pairs(species: str = "human", resource_name: str | None = None, resource=None, **_) -> set:
    import liana as li
    if resource is None:
        resource = li.resource.select_resource(resource_name or ("mouseconsensus" if species == "mouse" else "consensus"))
    return {(_gene_key(str(a).split("_")), _gene_key(str(b).split("_")))
            for a, b in zip(resource["ligand"], resource["receptor"])}


def _run_liana_backend(adata, *, groupby, species="human", layer=None, min_cells=10, n_perms=None,
                       pvalue_threshold=0.05, seed=0, use_gpu=False, spatial_key=None, progress=True,
                       liana_method="rank_aggregate", resource_name=None, expr_prop=0.1, **kw):
    try:
        import liana as li
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("method='liana' needs the optional dependency `liana` (pip install liana).") from exc
    if not hasattr(li.mt, liana_method):
        raise ValueError(f"Unknown LIANA method '{liana_method}'.")
    method_obj = getattr(li.mt, liana_method)
    if resource_name is None and "resource" not in kw:
        resource_name = "mouseconsensus" if species == "mouse" else "consensus"
    work = _minimal_adata(adata, layer, groupby)
    call = dict(groupby=groupby, expr_prop=expr_prop, min_cells=min_cells, use_raw=False, verbose=False,
                inplace=False, seed=seed, n_perms=1000 if n_perms is None else int(n_perms))
    if resource_name is not None:
        call["resource_name"] = resource_name
    call.update(kw)
    res = method_obj(work, **call)
    mag, spec = method_obj.magnitude, method_obj.specificity
    score = res[mag].to_numpy(float) if mag else np.full(len(res), np.nan)
    if mag and method_obj.magnitude_ascending:
        score = 1.0 - score if np.nanmax(score) <= 1 else -score
    is_p = spec is not None and (spec.endswith("pvals") or spec == "specificity_rank")
    pvalue = res[spec].to_numpy(float) if is_p else np.full(len(res), np.nan)
    long = pd.DataFrame({
        "source": res["source"].astype(str), "target": res["target"].astype(str),
        "ligand": res["ligand_complex"].astype(str), "receptor": res["receptor_complex"].astype(str),
        "ligand_genes": [_gene_key(str(x).split("_")) for x in res["ligand_complex"]],
        "receptor_genes": [_gene_key(str(x).split("_")) for x in res["receptor_complex"]],
        "interaction": (res["ligand_complex"].astype(str) + " - " + res["receptor_complex"].astype(str)),
        "pathway": "", "annotation": "", "score": score, "pvalue": pvalue,
    })
    long["significant"] = (long["pvalue"] < pvalue_threshold) if is_p else False
    long = long[np.isfinite(long["score"]) & (long["score"] > 0 if not (mag and method_obj.magnitude_ascending) else True)]
    params = {"liana_method": liana_method, "resource_name": resource_name or "custom", "expr_prop": float(expr_prop),
              "magnitude": mag or "", "specificity": spec or "", "n_perms": call["n_perms"]}
    return {"raw": res, "long": long, "params": params}


def _commot_db_pairs(species: str = "human", **_) -> set:
    return _cellchat_db_pairs(species)


def _run_commot_backend(adata, *, groupby, species="human", layer=None, min_cells=10, n_perms=None,
                        pvalue_threshold=0.05, seed=0, use_gpu=False, spatial_key="spatial", progress=True,
                        dis_thr=None, distance_unit="arbitrary", db_categories=("Secreted Signaling",),
                        pathways=None, statistic="sum", **kw):
    """COMMOT (optimal transport) on CellChatDB, summarised per sender/receiver group.

    The run time grows with the number of LR pairs times cells squared, so
    restrict large datasets with ``pathways=[...]`` (CellChatDB pathway names).
    """
    from ..external.cellchat._db import load_cellchat_db
    from ..space._commot import create_communication_anndata
    from ..space._communication import run_commot

    if dis_thr is None:
        raise ValueError("method='commot' needs dis_thr= (signaling range in coordinate units).")
    db = load_cellchat_db(species)
    if db_categories:
        db = db.subset(list(db_categories))
    sub = db.complex_subunits()
    it = db.interaction
    if pathways is not None:
        it = it[it["pathway_name"].isin(list(pathways))]
        if it.empty:
            raise ValueError(f"No CellChatDB interactions in pathways {list(pathways)}.")
    df = pd.DataFrame({"ligand": ["_".join(sub.get(x, [x])) for x in it["ligand"]],
                       "receptor": ["_".join(sub.get(x, [x])) for x in it["receptor"]],
                       "pathway": it["pathway_name"].to_numpy()}).drop_duplicates()
    keep = _kept_groups(adata, groupby, min_cells)
    work = _minimal_adata(adata, layer, groupby, keep, spatial_key=spatial_key)
    run_commot(work, database_name="ccc", df_ligrec=df, dis_thr=float(dis_thr), distance_unit=distance_unit,
               spatial_key=spatial_key, inplace=True, **kw)
    comm = create_communication_anndata(work, groupby, 100 if n_perms is None else int(n_perms),
                                        database_name="ccc", level="lr", statistic=statistic,
                                        use_gpu=use_gpu, seed=seed)
    M, P = np.asarray(comm.layers["means"]), np.asarray(comm.layers["pvalues"])
    i, j = np.nonzero(M > 0)
    v = comm.var
    long = pd.DataFrame({
        "source": comm.obs["sender"].to_numpy()[i], "target": comm.obs["receiver"].to_numpy()[i],
        "ligand": v["gene_a"].to_numpy()[j], "receptor": v["gene_b"].to_numpy()[j],
        "ligand_genes": [_gene_key(str(x).split("_")) for x in v["gene_a"].to_numpy()[j]],
        "receptor_genes": [_gene_key(str(x).split("_")) for x in v["gene_b"].to_numpy()[j]],
        "interaction": v["interacting_pair"].to_numpy()[j], "pathway": v["classification"].to_numpy()[j],
        "annotation": "", "score": M[i, j], "pvalue": P[i, j],
    })
    long["significant"] = long["pvalue"] < pvalue_threshold
    obsp = {k: work.obsp[k] for k in work.obsp if k.startswith("commot-ccc-")}
    raw = {"means": pd.DataFrame(M, index=comm.obs_names, columns=comm.var_names),
           "pvalues": pd.DataFrame(P, index=comm.obs_names, columns=comm.var_names)}
    params = {"dis_thr": float(dis_thr), "distance_unit": distance_unit, "statistic": statistic,
              "db_categories": list(db_categories or []), "pathways": list(pathways) if pathways else []}
    return {"raw": raw, "long": long, "params": params, "obsp": obsp,
            "row_index": work.obs_names}


register_ccc_method("cellchat", _run_cellchat_backend, score_scale="probability", legacy_key="cellchat_res",
                    db_pairs=_cellchat_db_pairs, description="CellChat v2 law-of-mass-action probability (PyTorch port).")
register_ccc_method("cellphonedb", _run_cellphonedb_backend, score_scale="expression", legacy_key="cpdb_results",
                    db_pairs=_cpdb_db_pairs, description="CellPhoneDB v5 statistical permutation test.")
register_ccc_method("liana", _run_liana_backend, score_scale="rank", legacy_key="liana_res",
                    db_pairs=_liana_db_pairs, description="LIANA+ (rank_aggregate by default).")
register_ccc_method("commot", _run_commot_backend, score_scale="expression", needs_spatial=True,
                    legacy_key="commot_res", db_pairs=_commot_db_pairs,
                    description="COMMOT collective optimal transport (spatial).")


# ----------------------------------------------------------------------------- comm AnnData
def long_to_comm_adata(long: pd.DataFrame, groups: Sequence[str] | None = None, *, method: str = "custom",
                       score_scale: str = "expression", separator: str = "|",
                       extra_layers: Mapping[str, pd.DataFrame] | None = None) -> anndata.AnnData:
    """Build the ``ov.pl.ccc_*`` communication AnnData from a long table.

    ``obs`` are all ordered sender-receiver pairs, ``var`` the LR pairs keyed by
    ``ligand_genes -> receptor_genes``; ``layers['means']`` holds the score
    (0 where untested) and ``layers['pvalues']`` the p-value (1 where untested).
    """
    long = _validate_long(long, method)
    if groups is None:
        groups = sorted(set(long["source"]) | set(long["target"]))
    groups = [str(g) for g in groups]
    if any(separator in g for g in groups):
        raise ValueError(f"Separator '{separator}' occurs in a group label; choose another separator.")
    n = len(groups)
    senders, receivers = np.repeat(groups, n), np.tile(groups, n)
    obs = pd.DataFrame({"sender": senders, "receiver": receivers},
                       index=[f"{s}{separator}{r}" for s, r in zip(senders, receivers)])
    obs["cell_type_pair"] = obs.index.astype(str)

    long = long[long["source"].isin(groups) & long["target"].isin(groups)].copy()
    long["lr_key"] = np.where(long["ligand_genes"].astype(bool) | long["receptor_genes"].astype(bool),
                              long["ligand_genes"] + "->" + long["receptor_genes"],
                              long["ligand"] + "->" + long["receptor"])
    first = long.drop_duplicates("lr_key").set_index("lr_key")
    keys = list(first.index)
    var = pd.DataFrame(index=pd.Index(keys, dtype=object))
    var["interaction_name"] = first["interaction"].to_numpy()
    var["interacting_pair"] = first["interaction"].to_numpy()
    var["interaction_name_2"] = first["interaction"].to_numpy()
    var["classification"] = first["pathway"].replace("", "Unclassified").to_numpy()
    var["gene_a"] = first["ligand"].to_numpy()
    var["gene_b"] = first["receptor"].to_numpy()
    var["ligand_genes"] = first["ligand_genes"].to_numpy()
    var["receptor_genes"] = first["receptor_genes"].to_numpy()
    var["annotation"] = first["annotation"].to_numpy()

    row = pd.Index(obs.index).get_indexer(long["source"] + separator + long["target"])
    col = pd.Index(keys).get_indexer(long["lr_key"])
    means = np.zeros((n * n, len(keys)))
    pvals = np.ones((n * n, len(keys)))
    # Duplicate (pair, LR) rows (e.g. isoforms) keep the strongest score.
    order = np.argsort(long["score"].to_numpy(), kind="stable")
    means[row[order], col[order]] = long["score"].to_numpy()[order]
    pv = long["pvalue"].to_numpy()
    pv = np.where(np.isfinite(pv), pv, np.where(long["significant"].to_numpy(), 0.0, 1.0))
    pvals[row[order], col[order]] = pv[order]
    comm = anndata.AnnData(X=means.copy(), obs=obs, var=var)
    comm.layers["means"] = means
    comm.layers["pvalues"] = pvals
    for name, frame in (extra_layers or {}).items():
        comm.layers[name] = frame.reindex(index=comm.obs_names, columns=comm.var_names).fillna(0).to_numpy(float)
    comm.uns["comm_source"] = method
    comm.uns["comm_score_scale"] = score_scale
    return comm


# ----------------------------------------------------------------------------- consensus
def _canon_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _in_universe(lg, rg, universe: set) -> np.ndarray:
    lg, rg = _norm_keys(lg), _norm_keys(rg)
    return np.fromiter((_canon_pair(a, b) in universe for a, b in zip(lg, rg)), dtype=bool, count=len(lg))


def _per_method_key_table(long: pd.DataFrame) -> pd.DataFrame:
    """One row per interaction, matched without direction.

    The key is (cell expressing gene set A, cell expressing gene set B, A, B)
    with A <= B, so a ligand->receptor row and the same co-expression written
    the other way round (CellPhoneDB adhesion pairs have no direction) align.
    Gene symbols are compared case-insensitively (mouse ``Nrg3`` = ``NRG3``).
    The strongest row keeps its oriented ``source`` / ``target`` / genes.
    """
    df = long[(long["ligand_genes"] != "") | (long["receptor_genes"] != "")].copy()
    lg, rg = _norm_keys(df["ligand_genes"]), _norm_keys(df["receptor_genes"])
    swap = lg > rg
    df["k_s"] = np.where(swap, df["target"], df["source"])
    df["k_t"] = np.where(swap, df["source"], df["target"])
    df["k_a"] = np.where(swap, rg, lg)
    df["k_b"] = np.where(swap, lg, rg)
    df = df.sort_values("score", ascending=False, kind="stable")
    agg = df.groupby(["k_s", "k_t", "k_a", "k_b"], sort=False).agg(
        score=("score", "max"), pvalue=("pvalue", "min"), significant=("significant", "any"),
        source=("source", "first"), target=("target", "first"), ligand_genes=("ligand_genes", "first"),
        receptor_genes=("receptor_genes", "first"), ligand=("ligand", "first"), receptor=("receptor", "first"),
        interaction=("interaction", "first"), pathway=("pathway", "first"), annotation=("annotation", "first"))
    return agg


def _resolve_level(longs: Mapping[str, pd.DataFrame], level: str) -> str:
    """``'lr'`` unless a method reports ligands only (SecAct), then ``'ligand'``."""
    if level not in ("auto", "lr", "ligand"):
        raise ValueError("level must be 'auto', 'lr' or 'ligand'.")
    if level != "auto":
        return level
    ligand_only = any(len(l) and (l["receptor_genes"].astype(str) == "").all() for l in longs.values())
    return "ligand" if ligand_only else "lr"


def _to_ligand_level(long: pd.DataFrame) -> pd.DataFrame:
    """Drop the receptor so sender -> receiver signalling is compared per ligand."""
    out = long.copy()
    out["receptor"] = ""
    out["receptor_genes"] = ""
    out["interaction"] = out["ligand"].astype(str)
    return out


def _rra_rho(ranks: np.ndarray) -> np.ndarray:
    """Robust rank aggregation rho (Kolde et al. 2012), Bonferroni-corrected over methods."""
    from scipy.stats import beta
    k = ranks.shape[1]
    srt = np.sort(ranks, axis=1)
    j = np.arange(1, k + 1)
    p = beta.cdf(srt, j, k - j + 1)
    return np.minimum(1.0, p.min(axis=1) * k)


def ccc_consensus_table(longs: Mapping[str, pd.DataFrame], *, how: str = "rra", min_methods: int | None = None,
                        pvalue_threshold: float = 0.05, universe: set | None = None,
                        level: str = "auto") -> pd.DataFrame:
    """Combine per-method long tables into one consensus long table.

    Each method ranks its interactions by score; untested or zero-score
    interactions get rank 1. ``how='rra'`` calls an interaction significant when
    the robust-rank-aggregation rho is below ``pvalue_threshold``;
    ``how='vote'`` when at least ``min_methods`` methods call it significant
    (default: majority). ``universe`` restricts to LR pairs given as
    ``(ligand_genes, receptor_genes)`` tuples, e.g. the pairs in every database.
    ``level='ligand'`` (chosen automatically when a method such as SecAct has
    no receptors) matches sender -> receiver signalling per ligand.
    """
    if how not in ("rra", "vote"):
        raise ValueError("how must be 'rra' or 'vote'.")
    if _resolve_level(longs, level) == "ligand":
        longs = {m: _to_ligand_level(l) for m, l in longs.items()}
    methods = list(longs)
    tables = {m: _per_method_key_table(longs[m]) for m in methods}
    if universe is not None:
        tables = {m: t[_in_universe(t.index.get_level_values(2), t.index.get_level_values(3), universe)]
                  for m, t in tables.items()}
    idx = tables[methods[0]].index
    for m in methods[1:]:
        idx = idx.union(tables[m].index)
    out = pd.DataFrame(index=idx)
    ranks = np.ones((len(idx), len(methods)))
    nsig = np.zeros(len(idx), dtype=int)
    ndet = np.zeros(len(idx), dtype=int)
    for c, m in enumerate(methods):
        t = tables[m]
        nrank = t["score"].rank(ascending=False, method="average") / max(len(t), 1)
        ranks[:, c] = nrank.reindex(idx).fillna(1.0).to_numpy()
        out[f"{m}_score"] = t["score"].reindex(idx).to_numpy()
        out[f"{m}_pvalue"] = t["pvalue"].reindex(idx).to_numpy()
        sig = t["significant"].reindex(idx).fillna(False).to_numpy(bool)
        out[f"{m}_significant"] = sig
        nsig += sig
        ndet += t["score"].reindex(idx).notna().to_numpy()
    rho = _rra_rho(ranks) if len(methods) > 1 else ranks[:, 0]
    if min_methods is None:
        min_methods = len(methods) // 2 + 1
    meta_cols = ["source", "target", "ligand_genes", "receptor_genes", "ligand", "receptor", "interaction",
                 "pathway", "annotation"]
    meta = pd.DataFrame("", index=idx, columns=meta_cols, dtype=object)
    # Names come from the highest-priority method that has the interaction.
    for m in sorted(methods, key=lambda x: _NAME_PRIORITY.index(x) if x in _NAME_PRIORITY else 99):
        t = tables[m][meta_cols].reindex(idx)
        for col in meta_cols:
            empty = meta[col].isna() | (meta[col].astype(str) == "")
            meta.loc[empty, col] = t.loc[empty, col]
    out = out.join(meta.fillna(""))
    out.index = out.index.set_names(["k_s", "k_t", "k_a", "k_b"])
    out["score"] = 1.0 - ranks.mean(axis=1)
    out["rra_rho"] = rho
    out["n_methods_detected"] = ndet
    out["n_methods_significant"] = nsig
    if how == "rra":
        out["pvalue"] = rho
        out["significant"] = rho < pvalue_threshold
    else:
        out["significant"] = nsig >= min_methods
        out["pvalue"] = np.where(out["significant"], 0.0, 1.0)
    out = out.reset_index(drop=True)
    front = LONG_COLUMNS + ["rra_rho", "n_methods_detected", "n_methods_significant"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].sort_values(["significant", "score"], ascending=False, kind="stable").reset_index(drop=True)


def shared_universe(methods: Sequence[str], species: str = "human", params: Mapping | None = None,
                    longs: Mapping[str, pd.DataFrame] | None = None, level: str = "lr") -> set:
    """Undirected LR pairs present in every method's database.

    Adapters without a database listing fall back to the pairs they tested
    (with a warning), which can only shrink the shared set.
    """
    import warnings

    sets = []
    for m in methods:
        backend = _get_backend(m)
        pairs = set()
        kw = {k: v for k, v in dict((params or {}).get(m, {})).items() if k != "species"}
        if backend.db_pairs is not None:
            try:
                pairs = backend.db_pairs(species=species, **kw)
            except (ImportError, OSError, KeyError, ValueError) as exc:
                warnings.warn(f"Database pairs of '{m}' unavailable ({exc}); using its tested pairs.", stacklevel=2)
        if not pairs and longs is not None and m in longs:
            pairs = set(zip(longs[m]["ligand_genes"], longs[m]["receptor_genes"]))
        if level == "ligand":
            pairs = {(a, "") for a, _ in pairs}
        sets.append({_canon_pair(_norm_key(a), _norm_key(b)) for a, b in pairs})
    return set.intersection(*sets) if sets else set()


# ----------------------------------------------------------------------------- main entry
def _get_ccc(adata: anndata.AnnData, key: str) -> Mapping:
    if key not in adata.uns or not isinstance(adata.uns[key], Mapping) or "long" not in adata.uns[key]:
        raise KeyError(f"adata.uns['{key}'] has no run_ccc result; run ov.single.run_ccc first.")
    return adata.uns[key]


def _looks_like_ccc_results(value) -> bool:
    return isinstance(value, Mapping) and value.get("kind") == "run_ccc" and "long" in value


@register_function(
    aliases=["run_ccc", "细胞通讯", "cell communication", "cell-cell communication", "CCC", "配体受体分析",
             "多方法细胞通讯", "consensus ccc"],
    category="single",
    description=("Unified cell-cell communication inference: method='cellchat' | 'cellphonedb' | 'liana' | "
                 "'fastccc' | 'secact' | 'cytosignal' | 'commot', or a list of methods with a consensus. "
                 "Returns communication AnnData for ov.pl.ccc_*."),
    prerequisites={"functions": [], "optional_functions": ["pp.normalize_total", "pp.log1p"]},
    requires={"obs": ["groupby"]},
    produces={"uns": ["ccc"]},
    auto_fix="none",
    examples=[
        "comm = ov.single.run_ccc(adata, groupby='cell_type', method='cellchat')",
        "comm = ov.single.run_ccc(adata, groupby='cell_type', method=['cellchat', 'cellphonedb', 'liana'])",
        "ov.pl.ccc_heatmap(comm, plot_type='heatmap')",
        "res = ov.single.compare_ccc(adata)",
    ],
    related=["single.compare_ccc", "single.get_ccc", "single.run_cellchat", "single.run_cellphonedb_v5",
             "single.run_liana", "pl.ccc_heatmap", "pl.ccc_network_plot", "pl.ccc_stat_plot", "pl.ccc_compare"],
)
def run_ccc(
    adata: anndata.AnnData,
    groupby: str,
    method: str | Sequence[str] = "cellchat",
    *,
    species: str = "human",
    resource: str | pd.DataFrame = "native",
    layer: str | None = None,
    min_cells: int = 10,
    n_perms: int | None = None,
    pvalue_threshold: float = 0.05,
    seed: int = 0,
    use_gpu: bool = False,
    spatial_key: str | None = None,
    consensus: str | None = "rra",
    min_methods: int | None = None,
    consensus_level: str = "auto",
    method_kwargs: Mapping[str, Mapping] | Mapping | None = None,
    key_added: str = "ccc",
    progress: bool = True,
    separator: str = "|",
) -> anndata.AnnData:
    r"""Infer cell-cell communication with one or several methods.

    Parameters
    ----------
    adata
        AnnData with **log-normalised** expression (library size + ``log1p``);
        backends that need another input (e.g. counts) convert internally.
    groupby
        ``adata.obs`` column with cell-group labels.
    method
        A method name from :func:`available_ccc_methods` or a list of them.
        ``'cellchat'`` (PyTorch port of CellChat v2), ``'cellphonedb'``
        (CellPhoneDB v5), ``'liana'`` (LIANA+ ``rank_aggregate``; other LIANA
        methods via ``method_kwargs={'liana': {'liana_method': ...}}``),
        ``'fastccc'`` (permutation-free), ``'secact'`` (secreted-protein
        activity), and the spatial ``'cytosignal'`` / ``'commot'`` (need
        ``spatial_key``).
    species
        ``'human'`` or ``'mouse'`` (CellChat also ``'zebrafish'``).
    resource
        ``'native'`` uses each method's own database; ``'shared'`` keeps only
        LR pairs present in every selected method's database, so methods are
        compared on the same interactions; a DataFrame with ``ligand`` /
        ``receptor`` columns (``_``-joined subunits) keeps only those pairs
        (and is passed to LIANA as its resource).
    layer
        Expression source: ``None`` (``X``), a layer name or ``'raw'``.
    min_cells
        Groups with ``<= min_cells`` cells are not analysed.
    n_perms
        Permutations for permutation-based methods; ``None`` uses each
        method's default (CellChat 100, CellPhoneDB/LIANA 1000).
    pvalue_threshold
        Significance cutoff for every method and for the RRA consensus.
    seed, use_gpu
        RNG seed; GPU for backends that support it (CellChat, SecAct, CytoSignal).
    spatial_key
        ``adata.obsm`` key of coordinates; required by spatial methods.
    consensus
        For several methods: ``'rra'`` (robust rank aggregation of per-method
        ranks; ``pvalue`` = rho), ``'vote'`` (significant in ``>= min_methods``
        methods) or ``None`` (no consensus layer).
    min_methods
        Votes needed with ``consensus='vote'`` (default: majority).
    consensus_level
        ``'lr'`` matches interactions per LR pair, ``'ligand'`` per ligand
        (sender -> receiver). ``'auto'`` uses ``'ligand'`` when a selected
        method reports no receptors (SecAct), else ``'lr'``.
    method_kwargs
        Extra arguments per backend, ``{'cellphonedb': {'cpdb_file_path': ...}}``;
        with a single method a flat dict is accepted.
    key_added
        Results go to ``adata.uns[key_added]``: per-method long tables
        (``['long'][method]``), parameters and the consensus table. Native
        results are also kept under each method's legacy key
        (``cellchat_res``, ``cpdb_results``, ``liana_res``, ...), so the
        method-specific helpers keep working.

    Returns
    -------
    AnnData
        Communication AnnData (sender|receiver x LR pair). For several methods
        it is the consensus, with ``layers['<method>_score']`` and
        ``layers['<method>_pvalue']`` per method.
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"`{groupby}` not found in adata.obs.")
    if adata.obs[groupby].isna().any():
        raise ValueError(f"adata.obs['{groupby}'] contains missing labels; drop or fill them first.")
    methods = [method] if isinstance(method, str) else list(method)
    methods = list(dict.fromkeys(_canonical(m) for m in methods))
    if not methods:
        raise ValueError("Select at least one method.")
    backends = {m: _get_backend(m) for m in methods}
    if method_kwargs is None:
        per_kw = {m: {} for m in methods}
    elif len(methods) == 1 and not any(_canonical(k) in backends for k in method_kwargs):
        per_kw = {methods[0]: dict(method_kwargs)}
    else:
        per_kw = {m: dict(method_kwargs.get(m, method_kwargs.get(_inverse_alias(m), {}))) for m in methods}
    for m, b in backends.items():
        if b.needs_spatial:
            spatial_key = spatial_key or "spatial"
            if spatial_key not in adata.obsm:
                raise KeyError(f"method='{m}' needs coordinates in adata.obsm['{spatial_key}'].")
    if isinstance(resource, pd.DataFrame):
        for m in methods:
            if m == "liana":
                per_kw[m].setdefault("resource", resource[["ligand", "receptor"]].astype(str))

    longs, params, scales = {}, {}, {}
    store = adata.uns.get(key_added) if isinstance(adata.uns.get(key_added), Mapping) else {}
    store = dict(store)
    for m in methods:
        b = backends[m]
        # Shared arguments are defaults; method_kwargs may override them per method.
        call = dict(species=species, layer=layer, min_cells=min_cells, n_perms=n_perms,
                    pvalue_threshold=pvalue_threshold, seed=seed, use_gpu=use_gpu,
                    spatial_key=spatial_key if b.needs_spatial or spatial_key else None, progress=progress)
        call.update(per_kw[m])
        out = b.run(adata, groupby=groupby, **call)
        long = annotate_pathways(_validate_long(out["long"], m), species)
        longs[m] = long
        p = dict(out.get("params", {}))
        p.update({k: v for k, v in per_kw[m].items() if isinstance(v, (str, int, float, bool))})
        params[m] = p
        scales[m] = b.score_scale
        if b.legacy_key and out.get("raw") is not None:
            adata.uns[b.legacy_key] = _uns_safe(out["raw"])
        for k, v in (out.get("obsm") or {}).items():
            # Per-cell results keep their column names (cells x interactions / proteins).
            if isinstance(v, pd.DataFrame):
                v = v.reindex(adata.obs_names).fillna(0)
                v.columns = v.columns.astype(str)
            adata.obsm[k] = v
        for k, v in (out.get("obsp") or {}).items():
            if v.shape[0] == adata.n_obs:
                adata.obsp[k] = v

    groups = _all_groups(adata, groupby, longs)
    if isinstance(resource, pd.DataFrame):
        universe = {_canon_pair(_norm_key(_gene_key(str(a).split("_"))), _norm_key(_gene_key(str(b).split("_"))))
                    for a, b in zip(resource["ligand"], resource["receptor"])}
    elif resource == "shared" and len(methods) > 1:
        universe = shared_universe(methods, species, {m: {**params[m], **per_kw[m]} for m in methods}, longs,
                                   level=_resolve_level(longs, consensus_level))
    elif resource in ("native", "shared"):
        universe = None
    else:
        raise ValueError("resource must be 'native', 'shared' or a DataFrame with ligand/receptor columns.")
    if universe is not None:
        by_ligand = _resolve_level(longs, consensus_level) == "ligand" and resource == "shared"
        longs = {m: l[_in_universe(l["ligand_genes"], [""] * len(l) if by_ligand else l["receptor_genes"],
                                   universe)].reset_index(drop=True)
                 for m, l in longs.items()}

    long_old = dict(store.get("long", {}))
    long_old.update(longs)
    params_old = dict(store.get("params", {}))
    params_old.update(params)
    scale_old = dict(store.get("score_scale", {}))
    scale_old.update(scales)
    store.update({"kind": "run_ccc", "groupby": groupby, "species": species, "groups": np.asarray(groups, dtype=object),
                  "methods": np.asarray(list(long_old), dtype=object), "long": long_old, "params": _uns_safe(params_old),
                  "score_scale": scale_old, "resource": resource if isinstance(resource, str) else "custom",
                  "pvalue_threshold": float(pvalue_threshold)})
    if len(methods) > 1 and consensus is not None:
        cons = ccc_consensus_table(longs, how=consensus, min_methods=min_methods, pvalue_threshold=pvalue_threshold,
                                   level=consensus_level)
        store["consensus"] = cons
        store["consensus_params"] = {"how": consensus, "methods": np.asarray(methods, dtype=object),
                                     "min_methods": int(min_methods or len(methods) // 2 + 1)}
    adata.uns[key_added] = store
    if len(methods) == 1:
        return get_ccc(adata, method=methods[0], key=key_added, separator=separator)
    return get_ccc(adata, method="consensus" if consensus is not None else methods[0], key=key_added,
                   separator=separator)


def _inverse_alias(name: str) -> str:
    for k, v in _ALIASES.items():
        if v == name:
            return k
    return name


def _all_groups(adata, groupby, longs) -> list[str]:
    present = set()
    for l in longs.values():
        present |= set(l["source"]) | set(l["target"])
    labels = [str(g) for g in pd.unique(adata.obs[groupby].astype(str))]
    return sorted(g for g in labels if g in present) or sorted(present)


@register_function(
    aliases=["get_ccc", "提取细胞通讯结果", "ccc结果"],
    category="single",
    description="Communication AnnData (or long table) of one method / the consensus stored by run_ccc.",
    prerequisites={"functions": ["run_ccc"], "optional_functions": []},
    requires={"uns": ["ccc"]},
    produces={},
    auto_fix="none",
    examples=["comm = ov.single.get_ccc(adata, method='cellphonedb')",
              "df = ov.single.get_ccc(adata, method='consensus', as_long=True)"],
    related=["single.run_ccc", "single.compare_ccc"],
)
def get_ccc(adata: anndata.AnnData, method: str = "consensus", *, key: str = "ccc", as_long: bool = False,
            significant_only: bool = False, separator: str = "|"):
    """Return the communication AnnData (``as_long=False``) or long table of ``method``."""
    store = _get_ccc(adata, key)
    m = _canonical(method)
    if m == "consensus":
        if "consensus" not in store:
            raise KeyError("No consensus stored; run run_ccc with several methods and consensus='rra'/'vote'.")
        long = pd.DataFrame(store["consensus"])
        scale, source = "rank", "consensus"
    else:
        if m not in store["long"]:
            raise KeyError(f"Method '{method}' not in adata.uns['{key}']; available: {list(store['long'])}.")
        long = pd.DataFrame(store["long"][m])
        scale, source = store["score_scale"].get(m, "expression"), m
    if significant_only:
        long = long[long["significant"].astype(bool)]
    if as_long:
        return long.reset_index(drop=True)
    extra = {}
    if m == "consensus":
        lk = long["ligand_genes"] + "->" + long["receptor_genes"]
        rk = long["source"] + separator + long["target"]
        for meth in store.get("consensus_params", {}).get("methods", []):
            for stat in ("score", "pvalue"):
                col = f"{meth}_{stat}"
                if col in long:
                    vals = long[col].astype(float)
                    if stat == "pvalue":
                        vals = vals.fillna(1.0)
                    extra[col] = pd.DataFrame({"r": rk, "c": lk, "v": vals}).pivot_table(
                        index="r", columns="c", values="v", aggfunc="first")
        extra["n_methods_significant"] = pd.DataFrame({"r": rk, "c": lk, "v": long["n_methods_significant"]}).pivot_table(
            index="r", columns="c", values="v", aggfunc="first")
    comm = long_to_comm_adata(long[LONG_COLUMNS], [str(g) for g in store["groups"]], method=source,
                              score_scale=scale, separator=separator, extra_layers=extra)
    if source == "cellchat":
        comm.uns["cellchat_parameters"] = dict(store["params"].get("cellchat", {}))
    return comm


# ----------------------------------------------------------------------------- comparison
@register_function(
    aliases=["compare_ccc", "细胞通讯方法比较", "ccc方法一致性", "compare communication methods"],
    category="single",
    description="Agreement between communication methods run by run_ccc: overlap, Jaccard, rank correlations.",
    prerequisites={"functions": ["run_ccc"], "optional_functions": []},
    requires={"uns": ["ccc"]},
    produces={},
    auto_fix="none",
    examples=["res = ov.single.compare_ccc(adata)", "res['jaccard']", "ov.pl.ccc_compare(res, plot_type='upset')"],
    related=["single.run_ccc", "pl.ccc_compare"],
)
def compare_ccc(adata: anndata.AnnData, *, key: str = "ccc", methods: Sequence[str] | None = None,
                shared_only: bool = True, top_n: int = 20, level: str = "auto") -> dict:
    """Compare the significant interactions of several methods.

    Interactions are matched on (sender, receiver, ligand genes, receptor
    genes), so complexes from different databases align. With
    ``shared_only=True`` only LR pairs present in every method's database are
    compared - otherwise differences in database coverage dominate.
    ``level='ligand'`` (automatic when a method such as SecAct reports
    ligands without receptors) compares sender -> receiver signalling per
    ligand instead of per LR pair.

    Returns
    -------
    dict
        ``n_significant`` (Series), ``jaccard`` / ``overlap`` / ``recall``
        (method x method DataFrames; ``recall.loc[a, b]`` = share of ``a``'s
        calls also made by ``b``), ``score_spearman`` (on interactions scored
        by both), ``lr_spearman`` (number of significant cell pairs per LR),
        ``cellpair_pearson`` (significant-interaction counts per cell pair),
        ``top_lr`` (top LR pairs per method), ``sets`` (significant keys per
        method, for UpSet plots) and ``universe_size``.
    """
    from scipy.stats import pearsonr, spearmanr

    store = _get_ccc(adata, key)
    methods = [_canonical(m) for m in (methods or list(store["long"]))]
    if len(methods) < 2:
        raise ValueError("Need results of at least two methods.")
    longs = {m: pd.DataFrame(store["long"][m]) for m in methods}
    level = _resolve_level(longs, level)
    if level == "ligand":
        longs = {m: _to_ligand_level(l) for m, l in longs.items()}
    universe = None
    if shared_only:
        params = {m: dict(store.get("params", {}).get(m, {})) for m in methods}
        universe = shared_universe(methods, str(store.get("species", "human")), params, longs, level=level)
    tabs = {}
    for m in methods:
        t = _per_method_key_table(longs[m])
        if universe is not None:
            t = t[_in_universe(t.index.get_level_values(2), t.index.get_level_values(3), universe)]
        tabs[m] = t
    sets = {m: set(t.index[t["significant"].to_numpy(bool)]) for m, t in tabs.items()}
    jac = pd.DataFrame(1.0, index=methods, columns=methods)
    ovl = pd.DataFrame(0, index=methods, columns=methods)
    rec = pd.DataFrame(1.0, index=methods, columns=methods)
    for a in methods:
        ovl.loc[a, a] = len(sets[a])
        for b in methods:
            if a != b:
                inter = len(sets[a] & sets[b])
                ovl.loc[a, b] = inter
                jac.loc[a, b] = inter / max(len(sets[a] | sets[b]), 1)
                rec.loc[a, b] = inter / max(len(sets[a]), 1)
    sc = pd.DataFrame(1.0, index=methods, columns=methods)
    for a, b in combinations(methods, 2):
        common = tabs[a].index.intersection(tabs[b].index)
        v = spearmanr(tabs[a].loc[common, "score"], tabs[b].loc[common, "score"])[0] if len(common) > 2 else np.nan
        sc.loc[a, b] = sc.loc[b, a] = v
    # LR-level agreement: number of significant cell pairs per (undirected) LR pair.
    names = {}
    lr_counts = {}
    for m, t in tabs.items():
        sig = t[t["significant"].to_numpy(bool)]
        for k, lg, rg in zip(zip(sig.index.get_level_values(2), sig.index.get_level_values(3)),
                             sig["ligand_genes"], sig["receptor_genes"]):
            names.setdefault(k, f"{_norm_key(lg)} -> {_norm_key(rg)}" if rg else _norm_key(lg))
        lr_counts[m] = sig.groupby(level=[2, 3]).size().sort_values(ascending=False, kind="stable")
    allk = sorted(set().union(*[set(v.index) for v in lr_counts.values()]))
    lrm = pd.DataFrame({m: v.reindex(allk).fillna(0).to_numpy() for m, v in lr_counts.items()},
                       index=pd.Index([names[k] for k in allk], dtype=object))
    lr_sp = pd.DataFrame(1.0, index=methods, columns=methods)
    for a, b in combinations(methods, 2):
        lr_sp.loc[a, b] = lr_sp.loc[b, a] = spearmanr(lrm[a], lrm[b])[0] if len(lrm) > 2 else np.nan
    groups = [str(g) for g in store["groups"]]
    cnt = {}
    for m, t in tabs.items():
        sig = t[t["significant"].to_numpy(bool)]
        c = sig.groupby(["source", "target"]).size()
        cnt[m] = np.array([[c.get((s, r), 0) for r in groups] for s in groups], float)
    cp = pd.DataFrame(1.0, index=methods, columns=methods)
    for a, b in combinations(methods, 2):
        x, y = cnt[a].ravel(), cnt[b].ravel()
        cp.loc[a, b] = cp.loc[b, a] = pearsonr(x, y)[0] if x.std() > 0 and y.std() > 0 else np.nan
    top = {m: [f"{names[k]} ({int(n)})" for k, n in v.head(top_n).items()] for m, v in lr_counts.items()}
    top_df = pd.DataFrame({m: pd.Series(v, dtype=object) for m, v in top.items()})
    return {"methods": methods, "n_significant": pd.Series({m: len(s) for m, s in sets.items()}),
            "jaccard": jac, "overlap": ovl, "recall": rec, "score_spearman": sc, "lr_spearman": lr_sp,
            "cellpair_pearson": cp, "cellpair_counts": {m: pd.DataFrame(v, index=groups, columns=groups) for m, v in cnt.items()},
            "lr_counts": lrm, "top_lr": top_df, "sets": sets,
            "universe_size": len(universe) if universe is not None else None, "level": level}
