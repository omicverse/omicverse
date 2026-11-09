"""SecAct secreted-protein communication backend for ``ov.single.run_ccc``.

SecAct (Ru et al., Nat Methods 2026) does not score ligand-receptor
co-expression. A sender -> receiver edge for secreted protein *P* means

1. *P* is up-regulated in the **sender** (Wilcoxon, BH ``< padj``, logFC,
   mean and detection cut-offs), and
2. the **signalling activity** of *P* -- inferred by ridge regression of the
   receiver's expression change on the SecAct signature of *P* -- is
   increased in the **receiver** (permutation z-score, BH ``< padj``).

There is no receptor gene: in the unified long table ``receptor`` and
``receptor_genes`` are ``""`` and ``interaction`` is ``"<P>→activity"``. The
edge score is SecAct's ``overall_strength`` (sender logFC x receiver activity
z) and the p-value its Fisher-combined, BH-adjusted ``overall_pv.adj``.

With ``spatial_key`` the backend runs ``SecAct.CCC.scST`` instead: per-cell
activity, then neighbouring sender expression x receiver activity against a
random cross-type background; the score is the mean product over neighbouring
pairs and the p-value its BH-adjusted permutation p.
"""
from __future__ import annotations


import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._registry import register_function

SCORE_SCALE = "activity"
NEEDS_SPATIAL = False  # spatial mode is opted into with ``spatial_key``

LONG_COLUMNS = ["source", "target", "ligand", "receptor", "ligand_genes", "receptor_genes", "interaction",
                "pathway", "annotation", "score", "pvalue", "significant"]


def _looks_like_counts(X) -> bool:
    values = X.data[:200000] if sp.issparse(X) else np.asarray(X).ravel()[:200000]
    values = values[values != 0]
    return values.size > 0 and bool(np.all(values >= 0)) and np.allclose(values, np.round(values))


def _counts_matrix(adata: anndata.AnnData, layer: str | None):
    """SecAct needs raw UMI counts: ``layer``, else ``layers['counts']``, ``raw.X`` or ``X``."""
    if layer is not None:
        if layer == "raw":
            if adata.raw is None:
                raise ValueError("layer='raw' requested but adata.raw is None.")
            X, names, where = adata.raw.X, adata.raw.var_names, "adata.raw.X"
        elif layer in adata.layers:
            X, names, where = adata.layers[layer], adata.var_names, f"adata.layers['{layer}']"
        else:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        if not _looks_like_counts(X):
            raise ValueError(f"SecAct needs raw counts, but {where} is not integer-valued.")
        return X, names, where
    candidates = []
    if "counts" in adata.layers:
        candidates.append((adata.layers["counts"], adata.var_names, "adata.layers['counts']"))
    if adata.raw is not None:
        candidates.append((adata.raw.X, adata.raw.var_names, "adata.raw.X"))
    candidates.append((adata.X, adata.var_names, "adata.X"))
    for X, names, where in candidates:
        if _looks_like_counts(X):
            return X, names, where
    raise ValueError(
        "SecAct needs raw counts (it normalises to log2 TPM / log2(CP100k) itself). None of "
        "adata.layers['counts'], adata.raw.X or adata.X is integer-valued; pass `layer=` with counts."
    )


def _flat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """h5ad-safe copy: string object columns, string index."""
    frame = frame.copy()
    for col in frame.columns:
        if not (pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col])):
            frame[col] = np.asarray(frame[col].astype(str), dtype=object)
    frame.index = np.asarray(frame.index.astype(str), dtype=object)
    return frame


def _activity_long(activity: dict) -> pd.DataFrame:
    """``{beta, se, zscore, pvalue}`` (proteins x samples) -> one long, h5ad-safe table."""
    z = activity["zscore"]
    out = z.stack().rename("zscore").reset_index()
    out.columns = ["protein", "cell_type", "zscore"]
    for k in ("beta", "se", "pvalue"):
        out[k] = activity[k].loc[z.index, z.columns].to_numpy().ravel()
    return _flat_frame(out[["protein", "cell_type", "beta", "se", "zscore", "pvalue"]])


def _long_table(edges: pd.DataFrame, score_col: str, pvalue_col: str, cutoff: float) -> pd.DataFrame:
    sp_ = edges["secretedProtein"].astype(str)
    long = pd.DataFrame({
        "source": edges["sender"].astype(str).to_numpy(),
        "target": edges["receiver"].astype(str).to_numpy(),
        "ligand": sp_.to_numpy(),
        "receptor": "",
        "ligand_genes": sp_.str.upper().to_numpy(),
        "receptor_genes": "",
        "interaction": (sp_ + "→activity").to_numpy(),
        "pathway": "",
        "annotation": "Secreted protein activity",
        "score": edges[score_col].to_numpy(dtype=np.float64),
        "pvalue": edges[pvalue_col].to_numpy(dtype=np.float64),
    })
    long["significant"] = long["pvalue"] < cutoff
    return long[LONG_COLUMNS].reset_index(drop=True)


def run(adata, *, groupby, species="human", layer=None, min_cells=10, n_perms=1000, pvalue_threshold=0.05,
        seed=0, use_gpu=False, spatial_key=None, progress=True, **method_kwargs) -> dict:
    """``run_ccc`` adapter: SecAct secreted-protein communication.

    Input is **raw counts** (``layer`` / ``layers['counts']`` / ``raw.X`` / ``X``,
    first integer-valued one); SecAct normalises internally. ``n_perms`` is
    SecAct's ``nrand``, ``seed`` seeds the GSL MT19937 permutation table (0
    reproduces R), and ``pvalue_threshold`` is SecAct's BH ``padj_cutoff``
    (R default 0.01), used for the step-1/2 filters and the final call.
    Groups with ``<= min_cells`` cells are skipped, the rule every ``run_ccc``
    backend uses (R hard-codes ``>= 30``; :func:`run_secact` keeps R's rule).
    ``n_perms=None`` means 1000.

    ``method_kwargs`` are forwarded to :func:`omicverse.external.secact.ccc_scrnaseq`
    (e.g. ``condition_key``/``condition_case``/``condition_control``,
    ``act_diff_cutoff``, ``exp_logfc_cutoff``, ``sig_matrix``, ``compat``,
    ``map_alias``, ``cache_dir``) or, with ``spatial_key``, to
    :func:`~omicverse.external.secact.ccc_spatial` (``radius``, ``k``,
    ``ratio_cutoff``, ``n_background``, ``background_seed``, ``activity_key``).
    """
    from ..external import secact as sec

    if groupby not in adata.obs.columns:
        raise KeyError(f"`{groupby}` not found in adata.obs.")
    labels = adata.obs[groupby]
    if labels.isna().any():
        raise ValueError(f"adata.obs['{groupby}'] contains missing labels; drop or fill them first.")
    labels = labels.astype(str).to_numpy()
    X, var_names, source = _counts_matrix(adata, layer)
    n_perms = 1000 if n_perms is None else int(n_perms)
    kw = dict(method_kwargs)
    # The engine keeps groups with >= its min_cells (R semantics); run_ccc drops <= min_cells.
    engine_min = int(kw.pop("_engine_min_cells", int(min_cells) + 1))
    cache = kw.pop("cache_dir", None)
    sig_matrix = kw.pop("sig_matrix", "SecAct")
    map_alias = kw.pop("map_alias", True)
    padj = float(kw.pop("padj_cutoff", pvalue_threshold))
    params = {"method": "secact", "groupby": groupby, "species": species, "counts_source": source,
              "nrand": int(n_perms), "seed": int(seed), "padj_cutoff": padj, "min_cells": int(min_cells),
              "sig_matrix": sig_matrix if isinstance(sig_matrix, str) else "custom DataFrame",
              "map_alias": bool(map_alias)}

    if spatial_key is None:
        cond_key = kw.pop("condition_key", None)
        conditions = None if cond_key is None else adata.obs[cond_key].astype(str).to_numpy()
        res = sec.ccc_scrnaseq(
            X, var_names, labels, conditions=conditions, condition_case=kw.pop("condition_case", None),
            condition_control=kw.pop("condition_control", None), padj_cutoff=padj, min_cells=engine_min,
            sig_matrix=sig_matrix, nrand=int(n_perms), seed=int(seed), species=species, map_alias=map_alias,
            use_gpu=use_gpu, cache_dir=cache, progress=progress, **kw)
        params.update({"mode": "scRNAseq", "rng_method": res["rng_method"], "condition_key": cond_key or ""})
        expression = pd.concat([df.assign(cell_type=c).rename_axis("protein").reset_index()
                                for c, df in res["expression"].items()], ignore_index=True)
        raw = {"method": "secact", "mode": "scRNAseq", "ccc": _flat_frame(res["ccc"]),
               "candidates": _flat_frame(res["candidates"]), "expression": _flat_frame(expression),
               "activity": _activity_long(res["activity"]), "parameters": params}
        long = _long_table(res["candidates"], "overall_strength", "overall_pv.adj", padj)
        return {"raw": raw, "long": long, "params": params}

    if spatial_key not in adata.obsm:
        raise KeyError(f"`{spatial_key}` not found in adata.obsm.")
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
    act_key = kw.pop("activity_key", None)
    obsm = {}
    if act_key is not None:
        act = adata.obsm[act_key]
        act = act if isinstance(act, pd.DataFrame) else pd.DataFrame(act, index=adata.obs_names)
        activity_z = act.T
    else:
        act = sec.spot_activity(X, var_names, scale_factor=float(kw.pop("activity_scale_factor", 1e5)),
                                sig_matrix=sig_matrix, nrand=int(n_perms), seed=int(seed), species=species,
                                map_alias=map_alias, use_gpu=use_gpu, cache_dir=cache, progress=progress)
        activity_z = act["zscore"]
        obsm["secact_activity"] = pd.DataFrame(activity_z.T.to_numpy(), index=adata.obs_names,
                                               columns=activity_z.index.astype(str))
    res = sec.ccc_spatial(X, var_names, coords, labels, activity_z, padj_cutoff=padj, species=species,
                          map_alias=map_alias, cache_dir=cache, progress=progress,
                          seed=int(kw.pop("background_seed", 123)), **kw)
    params.update({"mode": "scST", "spatial_key": spatial_key, "activity_key": act_key or ""})
    raw = {"method": "secact", "mode": "scST", "ccc": _flat_frame(res["ccc"]),
           "candidates": _flat_frame(res["candidates"]),
           "correlation": _flat_frame(res["correlation"].rename_axis("protein").reset_index()),
           "parameters": params}
    long = _long_table(res["candidates"], "strength", "pv.adj", padj)
    out = {"raw": raw, "long": long, "params": params}
    if obsm:
        out["obsm"] = obsm
    return out


@register_function(
    aliases=["SecAct", "run_secact", "secact", "分泌蛋白通讯", "SecAct细胞通讯", "secreted protein activity CCC"],
    category="single",
    description=(
        "SecAct secreted-protein cell-cell communication (Nat Methods 2026): sender expression of a "
        "secreted protein linked to its inferred signalling activity in receivers; scRNA-seq or spatial."
    ),
    prerequisites={"functions": [], "optional_functions": []},
    requires={"obs": ["groupby"]},
    produces={"uns": ["secact_res"]},
    auto_fix="none",
    examples=[
        "ov.single.run_secact(adata, groupby='cell_type')",
        "ov.single.run_secact(adata, groupby='cell_type', spatial_key='spatial', radius=20)",
        "adata.uns['secact_res']['ccc'].head()",
    ],
    related=["single.secact_activity", "single.run_cellchat", "single.run_liana"],
)
def run_secact(adata: anndata.AnnData, *, groupby: str, species: str = "human", layer: str | None = None,
               spatial_key: str | None = None, n_perms: int = 1000, padj_cutoff: float = 0.01,
               min_cells: int = 30, seed: int = 0, use_gpu: bool = False, progress: bool = True,
               key_added: str = "secact_res", inplace: bool = True, **kwargs):
    r"""Secreted-protein cell-cell communication with SecAct (R defaults).

    Re-implements R ``SecAct.CCC.scRNAseq`` (or ``SecAct.CCC.scST`` with
    ``spatial_key``) without R, scanpy or secactpy; results match R SecAct to
    floating-point precision. The 80 MB SecAct signature is downloaded once to
    ``$OMICVERSE_SECACT_DIR`` (default ``./data/secact``).

    Parameters
    ----------
    adata
        AnnData with **raw counts** in ``layer``, ``layers['counts']``, ``raw.X`` or ``X``.
    groupby
        Cell-type column in ``adata.obs``.
    species
        ``'human'`` or ``'mouse'`` (mouse genes are summed onto human orthologues).
    spatial_key
        ``adata.obsm`` key with cell coordinates (same unit as ``radius``, default 20 µm)
        to run the single-cell-resolution spatial pipeline.
    n_perms, seed
        Ridge permutations (R ``nrand``) and seed of the GSL MT19937 table (0 = R).
    padj_cutoff
        BH-adjusted p cut-off for every step (R default 0.01).
    min_cells
        Minimum cells per cell type, ``>= min_cells`` kept (R: 30).
    **kwargs
        Further SecAct options, e.g. ``condition_key``, ``condition_case``,
        ``condition_control``, ``act_diff_cutoff``, ``exp_logfc_cutoff``,
        ``sig_matrix='CytoSig'``, ``compat='secactpy'``, ``radius``, ``activity_key``.

    Returns
    -------
    AnnData or dict
        ``adata`` with ``adata.uns[key_added]`` (``ccc`` significant edges,
        ``candidates`` all tested edges, ``expression``/``activity`` step tables,
        ``long`` unified table, ``parameters``); the dict itself if ``inplace=False``.
    """
    out = run(adata, groupby=groupby, species=species, layer=layer, min_cells=min_cells, n_perms=n_perms,
              pvalue_threshold=padj_cutoff, seed=seed, use_gpu=use_gpu, spatial_key=spatial_key,
              progress=progress, _engine_min_cells=min_cells, **kwargs)
    result = dict(out["raw"])
    result["long"] = _flat_frame(out["long"])
    if not inplace:
        if "obsm" in out:
            result["cell_activity"] = out["obsm"]["secact_activity"]
        return result
    for key, value in out.get("obsm", {}).items():
        adata.obsm[key] = value
    adata.uns[key_added] = result
    return adata


@register_function(
    aliases=["secact_activity", "SecAct活性", "secreted protein activity", "分泌蛋白活性推断"],
    category="single",
    description="Infer signalling activity of 1,170 secreted proteins per cell type or per cell with SecAct.",
    prerequisites={"functions": [], "optional_functions": []},
    requires={},
    produces={"uns": ["secact_activity"], "obsm": ["secact_activity"]},
    auto_fix="none",
    examples=[
        "z = ov.single.secact_activity(adata, groupby='cell_type')['zscore']",
        "ov.single.secact_activity(adata)  # per cell -> adata.obsm['secact_activity']",
    ],
    related=["single.run_secact"],
)
def secact_activity(adata: anndata.AnnData, *, groupby: str | None = None, layer: str | None = None,
                    spatial: bool = False, species: str = "human", sig_matrix="SecAct", n_perms: int = 1000,
                    seed: int = 0, use_gpu: bool = False, progress: bool = True,
                    key_added: str = "secact_activity", inplace: bool = True, **kwargs):
    r"""Secreted-protein signalling activity (R ``SecAct.activity.inference.*``).

    ``groupby`` set: pseudo-bulk per group, log2 TPM centred on the mean group
    (``SecAct.activity.inference.scRNAseq``). ``groupby=None``: every cell,
    centred on the mean cell (``is.singleCellLevel=TRUE``; with ``spatial=True``
    the ``.ST`` variant, which drops all-zero genes first). Input is raw counts.

    Returns
    -------
    dict
        ``beta``, ``se``, ``zscore``, ``pvalue`` DataFrames (proteins x groups/cells).
        With ``inplace=True`` the per-cell z-scores also go to ``adata.obsm[key_added]``
        (cells x proteins) and group results to ``adata.uns[key_added]``.
    """
    from ..external import secact as sec

    X, var_names, _ = _counts_matrix(adata, layer)
    common = dict(sig_matrix=sig_matrix, nrand=int(n_perms), seed=int(seed), species=species, use_gpu=use_gpu,
                  progress=progress, **kwargs)
    if groupby is not None:
        res = sec.scrnaseq_activity(X, var_names, adata.obs[groupby].astype(str).to_numpy(), **common)
    elif spatial:
        res = sec.spot_activity(X, var_names, **common)
    else:
        res = sec.scrnaseq_activity(X, var_names, None, **common)
    if groupby is None:
        for k in res:
            res[k].columns = adata.obs_names
    if inplace:
        if groupby is None:
            adata.obsm[key_added] = pd.DataFrame(res["zscore"].T.to_numpy(), index=adata.obs_names,
                                                 columns=res["zscore"].index.astype(str))
        else:
            adata.uns[key_added] = _activity_long(res)
    return res


def db_pairs(species="human", sig_matrix="SecAct", cache_dir=None, **_):
    """Secreted proteins of the cached signature as ``(gene, '')`` pairs (no download)."""
    from ..external.secact import _signature as sg

    if not isinstance(sig_matrix, str):
        return set()
    if not any(sg.cache_dir(cache_dir).glob(f"{sig_matrix}*")):
        raise OSError(f"{sig_matrix} signature not cached; run SecAct once to download it.")
    sig, _, _ = sg.load_signature(sig_matrix, cache_dir=cache_dir)
    return {(str(g), "") for g in sig.columns}
