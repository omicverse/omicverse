"""CytoSignal backend for :func:`omicverse.single.run_ccc` / :func:`omicverse.space.run_ccc`.

CytoSignal (Liu et al., *Nat Genet* 2026) scores ligand-receptor signalling at
every cell of a spatial transcriptomics section and tests it with a spatial
permutation null and a density-weighted spatial FDR. The engine in
``omicverse.external.cytosignal`` is a NumPy/SciPy port of the R package
(v0.5.1) that reproduces its neighbour graphs, LR-scores, p-values and
significant cells (``set.seed`` included).
"""
from __future__ import annotations

from collections.abc import Sequence

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._registry import register_function

SCORE_SCALE = "expression"
NEEDS_SPATIAL = True

_SPECIES = ("human", "mouse")
_TYPES = {"diffusion": "diffusion", "contact": "contact"}


def _counts_matrix(adata: anndata.AnnData, layer: str | None):
    """Raw counts and matching var_names (CytoSignal normalises internally)."""
    def is_counts(X) -> bool:
        v = X.data[:200000] if sp.issparse(X) else np.asarray(X).ravel()[:200000]
        v = v[v != 0]
        return v.size > 0 and bool(np.all(v > 0)) and bool(np.allclose(v, np.round(v)))

    if layer is not None:
        if layer == "raw":
            if adata.raw is None:
                raise ValueError("layer='raw' requested but adata.raw is None.")
            X, var = adata.raw.X, adata.raw.var_names
        elif layer in adata.layers:
            X, var = adata.layers[layer], adata.var_names
        else:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        if not is_counts(X):
            raise ValueError(f"CytoSignal needs raw counts, but layer '{layer}' is not integer counts.")
        return X, var
    candidates = []
    if "counts" in adata.layers:
        candidates.append((adata.layers["counts"], adata.var_names))
    candidates.append((adata.X, adata.var_names))
    if adata.raw is not None:
        candidates.append((adata.raw.X, adata.raw.var_names))
    for X, var in candidates:
        if is_counts(X):
            return X, var
    raise ValueError(
        "CytoSignal needs raw counts (it library-normalises imputed counts itself). Put them in "
        "adata.layers['counts'] or pass layer=<counts layer>.")


def _gene_meta(db, intr_ids: Sequence[str], intr_db: str) -> pd.DataFrame:
    comp = db.components(intr_db)
    u2g = db.uniprot_to_gene
    rows = []
    for i in intr_ids:
        lg = [u2g.get(u, u) for u in comp[i]["ligands"]]
        rg = [u2g.get(u, u) for u in comp[i]["receptors"]]
        lig, rec = "_".join(lg), "_".join(rg)
        rows.append({"id": i, "ligand": lig, "receptor": rec,
                     "ligand_genes": "_".join(sorted(set(lg))), "receptor_genes": "_".join(sorted(set(rg))),
                     "interaction": f"{lig}-{rec}", "pathway": "", "annotation": _TYPES[intr_db]})
    return pd.DataFrame(rows, columns=["id", "ligand", "receptor", "ligand_genes", "receptor_genes",
                                       "interaction", "pathway", "annotation"])


def _cytosignal(adata, *, spatial_key, layer, species, seed, perm_size, scale_factor, r_eps_real, thresh,
                counts_thresh, recep_smooth, self_weight, dt_mode, norm_method, p_thresh, reads_thresh,
                sig_thresh, fdr_method, progress):
    from ..external.cytosignal import _core
    from ..external.cytosignal._db import load_cytosignal_db

    if species not in _SPECIES:
        raise ValueError(f"CytoSignal supports species {_SPECIES} (human gene symbols; mouse symbols are "
                         f"upper-cased as in the R package), got {species!r}.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial coordinates adata.obsm['{spatial_key}'] not found.")
    X, var = _counts_matrix(adata, layer)
    loc = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
    db = load_cytosignal_db()
    keep, st, res = _core.run_pipeline(
        X, list(var), loc, db=db, scale_factor=scale_factor, r_eps_real=r_eps_real, thresh=thresh,
        counts_thresh=counts_thresh, perm_size=perm_size, seed=seed, recep_smooth=recep_smooth,
        self_weight=self_weight, dt_mode=dt_mode, norm_method=norm_method, p_thresh=p_thresh,
        reads_thresh=reads_thresh, sig_thresh=sig_thresh, fdr_method=fdr_method, progress=progress)
    return db, keep, st, res


def _cell_frames(adata, keep, res, metas):
    """Cells x interactions DataFrames (full obs index; NaN for QC-removed cells)."""
    cells = adata.obs_names[keep]
    score_cols, padj_cols, scores, padjs = [], [], [], []
    for slot, r in res.items():
        meta = metas[slot]
        names = [f"{n} ({slot})" if slot == "raw" else n for n in meta["interaction"]]
        scores.append(r.score)
        score_cols += names
        p = np.full(r.score.shape, np.nan)
        p[r.has_neighbor] = r.padj
        padjs.append(p)
        padj_cols += names
    score = pd.DataFrame(np.hstack(scores), index=cells, columns=score_cols).reindex(adata.obs_names)
    padj = pd.DataFrame(np.hstack(padjs), index=cells, columns=padj_cols).reindex(adata.obs_names)
    return score, padj


def run(adata, *, groupby, species="human", layer=None, min_cells=10, n_perms=None, pvalue_threshold=0.05,
        seed=0, use_gpu=False, spatial_key="spatial", progress=True, perm_size=100000, scale_factor=1.0,
        r_eps_real=200.0, thresh=0.001, counts_thresh=300, recep_smooth=False, self_weight="auto",
        dt_mode="weight_sum_2", norm_method="default", p_thresh=0.05, reads_thresh=100, sig_thresh=100,
        fdr_method="spatialFDR", hq_only=True, slots=("diffusion", "contact"), **_ignored) -> dict:
    r"""CytoSignal through the unified ``run_ccc`` contract.

    Cell level (exactly the R package): ``removeLowQuality(counts_thresh)``
    -> ``changeUniprot`` -> ``inferEpsParams(scale_factor, r_eps_real)`` ->
    ``findNN`` (Gaussian epsilon ball + Delaunay) -> ``imputeLR`` ->
    ``inferIntrScore(perm_size)`` -> ``inferSignif(p_thresh, reads_thresh,
    sig_thresh)``. Input must be **raw counts** (``adata.layers['counts']``,
    ``layer=`` or integer ``X``); ``scale_factor`` converts coordinate units
    to micrometres (diffusion radius ``r_eps_real`` = 200 µm).

    Group level (``long``): each receiver cell's score is split across the
    sender groups by their share of the ligand imputed at that cell; the
    ``score`` of ``source -> target`` is the mean received score over target
    cells. ``pvalue`` comes from ``n_perms`` (default 1000) permutations of
    the receiver labels applied to the same statistic computed on CytoSignal's
    significant cells only (high-quality interactions when ``hq_only``). See
    :mod:`omicverse.external.cytosignal._group`. Groups with ``<= min_cells``
    cells are left out of the table.
    """
    from ..external.cytosignal import _group

    n_perms = 1000 if n_perms is None else int(n_perms)
    if groupby not in adata.obs.columns:
        raise KeyError(f"`{groupby}` not found in adata.obs.")
    db, keep, st, res = _cytosignal(
        adata, spatial_key=spatial_key, layer=layer, species=species, seed=seed, perm_size=perm_size,
        scale_factor=scale_factor, r_eps_real=r_eps_real, thresh=thresh, counts_thresh=counts_thresh,
        recep_smooth=recep_smooth, self_weight=self_weight, dt_mode=dt_mode, norm_method=norm_method,
        p_thresh=p_thresh, reads_thresh=reads_thresh, sig_thresh=sig_thresh, fdr_method=fdr_method,
        progress=progress)
    labels = adata.obs[groupby].astype(str).to_numpy()[keep]
    groups = sorted(pd.unique(labels))
    codes = np.searchsorted(np.asarray(groups, dtype=object), labels).astype(np.int64)
    sizes = np.bincount(codes, minlength=len(groups))
    keep_groups = np.where(sizes > min_cells)[0]  # same rule as the other backends

    metas = {s: _gene_meta(db, r.interactions, r.db) for s, r in res.items()}
    longs, info = [], []
    for slot in slots:
        if slot not in res:
            continue
        r, meta = res[slot], metas[slot]
        sig = np.zeros(r.score.shape, dtype=bool)
        sig[r.has_neighbor] = r.padj < p_thresh
        if hq_only:
            sig[:, ~np.isin(r.interactions, list(r.result_hq))] = False
        w = st.graphs[r.lig_slot]
        lc = _group.ligand_counts(st.counts, r.intr.lig)
        score, pval = _group.group_summary(w, lc, r.score, sig, codes, len(groups), n_perms=n_perms,
                                           seed=seed, use_gpu=use_gpu)
        longs.append(_group.long_table(score, pval, groups, meta, keep_groups=keep_groups,
                                       pvalue_threshold=pvalue_threshold))
    long = pd.concat(longs, ignore_index=True) if longs else pd.DataFrame(columns=[
        "source", "target", "ligand", "receptor", "ligand_genes", "receptor_genes", "interaction", "pathway",
        "annotation", "score", "pvalue", "significant"])

    cell_names = adata.obs_names[keep]
    for slot, r in res.items():
        meta = metas[slot].copy()
        meta["type"] = slot
        meta["n_sig_cells"] = [len(r.result.get(i, ())) for i in r.interactions]
        meta["high_quality"] = [i in r.result_hq for i in r.interactions]
        meta["n_null"] = r.n_null
        info.append(meta)
    interactions = pd.concat(info, ignore_index=True)
    sig_rows = [(slot, i, cell_names[c]) for slot, r in res.items() for i, cs in r.result.items() for c in cs]
    sig_cells = pd.DataFrame(sig_rows, columns=["type", "id", "cell"])
    params = {"method": "cytosignal", "groupby": groupby, "species": species, "spatial_key": spatial_key,
              "perm_size": int(perm_size), "n_perms": n_perms, "seed": int(seed), "scale_factor": float(scale_factor),
              "r_eps_real": float(r_eps_real), "eps": float(st.eps), "sigma": float(st.sigma),
              "counts_thresh": float(counts_thresh), "recep_smooth": bool(recep_smooth),
              "self_weight": str(self_weight), "dt_mode": dt_mode, "norm_method": norm_method,
              "p_thresh": float(p_thresh), "reads_thresh": float(reads_thresh), "sig_thresh": float(sig_thresh),
              "fdr_method": fdr_method, "hq_only": bool(hq_only), "min_cells": int(min_cells),
              "pvalue_threshold": float(pvalue_threshold), "n_cells_used": int(keep.sum()),
              "db_version": "CytoSignal 0.5.1 (CellPhoneDB-derived)"}
    raw = {"method": "cytosignal", "groups": np.asarray(groups, dtype=object),
           "interactions": interactions, "significant_cells": sig_cells, "parameters": params}
    score_df, padj_df = _cell_frames(adata, keep, res, metas)
    return {"raw": raw, "long": long, "params": params,
            "obsm": {"cytosignal_scores": score_df, "cytosignal_padj": padj_df}}


@register_function(
    aliases=["CytoSignal", "run_cytosignal", "cytosignal", "空间细胞通讯", "单细胞分辨率配体受体信号"],
    category="space",
    description=("Cellular-resolution spatial ligand-receptor signalling with CytoSignal (Gaussian/Delaunay "
                 "neighbourhoods, spatial permutation test, spatial FDR); R-package parity."),
    prerequisites={"functions": [], "optional_functions": []},
    requires={"obsm": ["spatial"], "layers": ["counts"]},
    produces={"obsm": ["cytosignal_scores", "cytosignal_padj"], "uns": ["cytosignal"]},
    auto_fix="none",
    examples=[
        "ov.single.run_cytosignal(adata, spatial_key='spatial', scale_factor=0.73)",
        "adata.uns['cytosignal']['interactions'].query('high_quality')",
        "ov.space.run_ccc(adata, groupby='cell_type', method='cytosignal', "
        "method_kwargs={'scale_factor': 0.73})",
    ],
    related=["space.run_ccc", "single.run_ccc"],
)
def run_cytosignal(adata: anndata.AnnData, *, spatial_key: str = "spatial", layer: str | None = None,
                   species: str = "human", scale_factor: float = 1.0, r_eps_real: float = 200.0,
                   thresh: float = 0.001, counts_thresh: float = 300, perm_size: int = 100000, seed: int = 42,
                   recep_smooth: bool = False, self_weight="auto", dt_mode: str = "weight_sum_2",
                   norm_method: str = "default", p_thresh: float = 0.05, reads_thresh: float = 100,
                   sig_thresh: float = 100, fdr_method: str = "spatialFDR", progress: bool = True,
                   key_added: str = "cytosignal", inplace: bool = True):
    r"""Infer cellular-resolution spatial signalling with CytoSignal (Liu et al., Nat Genet 2026).

    Ports the default R workflow (``createCytoSignal`` ... ``inferSignif``)
    and reproduces it to floating-point precision, including p-values: with
    the same ``seed`` the permutation draws equal R's ``set.seed(seed)``.

    Parameters
    ----------
    adata
        Spatial AnnData with **raw counts** (``layers['counts']``, ``layer=``
        or integer ``X``) and coordinates in ``obsm[spatial_key]``.
    scale_factor
        Micrometres per coordinate unit (R ``inferEpsParams(scale.factor)``);
        e.g. ``0.73`` for the Slide-tags tutorial data, ``1`` for µm coordinates.
    r_eps_real, thresh
        Diffusion radius in µm and the Gaussian tail mass outside it.
    counts_thresh
        Cells with library size ``<= counts_thresh`` are removed.
    perm_size
        Size of the spatial permutation null (R ``perm.size``, default 1e5).
    seed
        R-compatible seed (``set.seed``); the vignette uses 42.
    recep_smooth, self_weight, dt_mode, norm_method
        R ``inferIntrScore`` / ``findNN`` options.
    p_thresh, reads_thresh, sig_thresh, fdr_method
        R ``inferSignif`` options: adjusted p-value cutoff, minimum cells
        expressing every component, minimum significant cells per
        interaction, and ``'spatialFDR'`` or per-cell ``'fdr'``.
    key_added
        ``adata.uns`` key for the result summary.

    Returns
    -------
    AnnData or dict
        With ``inplace=True``: ``adata`` with ``obsm['cytosignal_scores']`` /
        ``obsm['cytosignal_padj']`` (cells x interactions, column names in
        ``uns[key_added]['score_columns']``) and ``uns[key_added]`` holding
        ``interactions`` (per-interaction summary incl. ``n_sig_cells`` and
        ``high_quality``), ``significant_cells`` and ``parameters``.
        Otherwise the dict with the per-slot engine results.
    """
    db, keep, st, res = _cytosignal(
        adata, spatial_key=spatial_key, layer=layer, species=species, seed=seed, perm_size=perm_size,
        scale_factor=scale_factor, r_eps_real=r_eps_real, thresh=thresh, counts_thresh=counts_thresh,
        recep_smooth=recep_smooth, self_weight=self_weight, dt_mode=dt_mode, norm_method=norm_method,
        p_thresh=p_thresh, reads_thresh=reads_thresh, sig_thresh=sig_thresh, fdr_method=fdr_method,
        progress=progress)
    metas = {s: _gene_meta(db, r.interactions, r.db) for s, r in res.items()}
    if not inplace:
        return {"keep": keep, "state": st, "slots": res, "interactions": metas}
    score_df, padj_df = _cell_frames(adata, keep, res, metas)
    adata.obsm["cytosignal_scores"] = score_df.to_numpy()
    adata.obsm["cytosignal_padj"] = padj_df.to_numpy()
    info = []
    cell_names = adata.obs_names[keep]
    for slot, r in res.items():
        m = metas[slot].copy()
        m["type"] = slot
        m["n_sig_cells"] = [len(r.result.get(i, ())) for i in r.interactions]
        m["high_quality"] = [i in r.result_hq for i in r.interactions]
        info.append(m)
    sig_rows = [(slot, i, cell_names[c]) for slot, r in res.items() for i, cs in r.result.items() for c in cs]
    adata.uns[key_added] = {
        "interactions": pd.concat(info, ignore_index=True),
        "significant_cells": pd.DataFrame(sig_rows, columns=["type", "id", "cell"]),
        "score_columns": np.asarray(score_df.columns.astype(str), dtype=object),
        "parameters": {"perm_size": int(perm_size), "seed": int(seed), "scale_factor": float(scale_factor),
                       "eps": float(st.eps), "sigma": float(st.sigma), "p_thresh": float(p_thresh),
                       "reads_thresh": float(reads_thresh), "sig_thresh": float(sig_thresh),
                       "fdr_method": fdr_method, "n_cells_used": int(keep.sum())},
    }
    return adata
