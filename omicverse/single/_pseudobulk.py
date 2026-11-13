"""Pseudobulk aggregation of single-cell data.

``ov.single.pseudobulk`` collapses a single-cell AnnData into per-sample
(optionally per-sample-per-celltype) pseudobulk profiles — the correct unit
for cohort-level differential expression with DESeq2 / edgeR / limma
(``ov.bulk.pyDEG``).

The API mirrors :func:`decoupler.pp.pseudobulk`
(https://decoupler.readthedocs.io/en/latest/api/generated/decoupler.pp.pseudobulk.html):
aggregate by ``sample_col`` and optionally ``groups_col``, with the same
QC fields written to the output (``obs['psbulk_n_cells']``,
``obs['psbulk_counts']``, ``layers['psbulk_props']``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._registry import register_function


def _get_matrix(adata, layer: Optional[str], raw: bool):
    """Resolve the cell × gene matrix to aggregate."""
    if raw:
        if adata.raw is None:
            raise ValueError("raw=True but adata.raw is None.")
        return adata.raw.X, list(adata.raw.var_names)
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"layer={layer!r} not in adata.layers.")
        return adata.layers[layer], list(adata.var_names)
    return adata.X, list(adata.var_names)


@register_function(
    aliases=["拟蛋白", "pseudobulk", "pseudo_bulk", "拟bulk", "伪bulk", "样本聚合"],
    category="single",
    description=(
        "Aggregate single-cell profiles into per-sample (optionally "
        "per-sample-per-celltype) pseudobulk profiles — the correct unit "
        "for cohort-level DE with DESeq2 / edgeR / limma. API mirrors "
        "decoupler.pp.pseudobulk."
    ),
    requires={
        "obs": [
            "sample_col — sample / donor / replicate label",
            "groups_col — optional cell-type / cluster label",
        ],
    },
    produces={
        "obs": ["psbulk_n_cells", "psbulk_counts"],
        "layers": ["psbulk_props"],
    },
    auto_fix="none",
    examples=[
        "# One pseudobulk profile per sample",
        "pb = ov.single.pseudobulk(adata, sample_col='donor')",
        "",
        "# Per-sample × per-celltype (the usual input for pyDEG)",
        "pb = ov.single.pseudobulk(adata, sample_col='donor',",
        "                          groups_col='celltype', layer='counts')",
        "",
        "# Mean instead of sum, drop tiny groups",
        "pb = ov.single.pseudobulk(adata, sample_col='donor',",
        "                          groups_col='celltype', mode='mean',",
        "                          min_cells=10)",
    ],
    related=["bulk.pyDEG", "single.get_obs_value", "single.MetaCell"],
)
def pseudobulk(
    adata,
    sample_col: str,
    groups_col: Optional[str] = None,
    layer: Optional[str] = None,
    raw: bool = False,
    mode: str = "sum",
    min_cells: int = 0,
    min_counts: int = 0,
    dtype=np.float32,
):
    """Summarise single-cell profiles into pseudobulk profiles.

    Parameters
    ----------
    adata : AnnData
        Single-cell AnnData. For DE downstream, point ``layer`` at raw counts.
    sample_col : str
        ``adata.obs`` column with the sample / donor / replicate label.
    groups_col : str or None, default=None
        ``adata.obs`` column with the cell-type / cluster label. When given,
        one pseudobulk profile is produced per ``(sample, group)`` pair;
        when ``None``, one per sample.
    layer : str or None, default=None
        Layer to aggregate. ``None`` uses ``adata.X``. For DESeq2 / edgeR
        this must be **raw counts**.
    raw : bool, default=False
        Aggregate ``adata.raw.X`` instead of ``adata.X`` / a layer.
    mode : str, default='sum'
        Aggregation: ``'sum'`` (counts — required by DESeq2 / edgeR),
        ``'mean'`` or ``'median'``.
    min_cells : int, default=0
        Drop pseudobulk profiles built from fewer than this many cells.
    min_counts : int, default=0
        Drop pseudobulk profiles whose total aggregated count is below this.
    dtype : numpy dtype, default=float32
        dtype of the output matrix.

    Returns
    -------
    AnnData
        Pseudobulk AnnData: rows = ``(sample, group)`` profiles, columns =
        genes. ``obs`` carries ``sample_col``, ``groups_col`` (if used),
        ``psbulk_n_cells`` (cells per profile), ``psbulk_counts`` (total
        aggregated count), plus any other ``obs`` column that is constant
        within every profile. ``layers['psbulk_props']`` holds, per gene,
        the fraction of cells in the profile with non-zero expression.
    """
    import anndata as ad

    if mode not in ("sum", "mean", "median"):
        raise ValueError(f"mode must be 'sum' / 'mean' / 'median', got {mode!r}")
    if sample_col not in adata.obs:
        raise KeyError(f"sample_col={sample_col!r} not in adata.obs.")
    if groups_col is not None and groups_col not in adata.obs:
        raise KeyError(f"groups_col={groups_col!r} not in adata.obs.")

    X, var_names = _get_matrix(adata, layer, raw)
    n_cells, n_genes = X.shape

    # ---- profile key: (sample) or (sample, group) ---------------------------
    # Profiles are the sorted unique (sample[, group]) tuples, encoded as ints.
    key_cols = [sample_col] if groups_col is None else [sample_col, groups_col]
    key_frame = pd.DataFrame({c: adata.obs[c].astype(str).to_numpy() for c in key_cols})
    codes, levels = zip(*(pd.factorize(key_frame[c].to_numpy(), sort=True) for c in key_cols))
    shape = [len(lev) for lev in levels]
    uniq, inv = np.unique(np.ravel_multi_index(codes, shape), return_inverse=True)
    n_profiles = len(uniq)

    profile_codes = np.unravel_index(uniq, shape)
    labels = levels[0][profile_codes[0]]
    for lev, code in zip(levels[1:], profile_codes[1:]):
        labels = labels + " | " + lev[code]

    # Cells grouped by profile (ascending cell index within each profile).
    cell_order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=n_profiles)
    indptr = np.concatenate(([0], np.cumsum(counts)))
    first_idx = cell_order[indptr[:-1]]
    counts_per_profile = counts.astype(float)

    # ---- aggregate cell × gene → profile × gene -----------------------------
    # Sparse (n_profiles × n_cells) indicator P; P @ X aggregates.
    Pcount = sp.csr_matrix(
        (np.ones(n_cells), cell_order, indptr), shape=(n_profiles, n_cells)
    )

    if mode == "median":
        # Median needs a per-profile pass (no linear-algebra shortcut).
        Xd = X.toarray() if sp.issparse(X) else np.asarray(X)
        Xd = Xd[cell_order]
        agg = np.vstack([
            np.median(Xd[indptr[p]:indptr[p + 1]], axis=0) for p in range(n_profiles)
        ]).astype(dtype)
    else:
        if mode == "sum":
            P = Pcount
        else:
            weight = 1.0 / counts_per_profile[inv[cell_order]]
            P = sp.csr_matrix((weight, cell_order, indptr), shape=(n_profiles, n_cells))
        agg = P @ X
        agg = np.asarray(agg.todense() if sp.issparse(agg) else agg, dtype=dtype)

    # ---- QC: expressed-gene proportion per profile --------------------------
    # Counts are integers, so float32 sums stay exact and halve the memory traffic.
    nz = (X != 0)
    if sp.issparse(nz):
        nz = nz.astype(np.float32)
    else:
        nz = np.asarray(nz, dtype=np.float32)
    props = Pcount.astype(np.float32) @ nz
    props = np.asarray(props.todense() if sp.issparse(props) else props, dtype=dtype)
    props /= counts_per_profile[:, None]          # → fraction of cells expressing

    psbulk_counts = np.asarray(agg.sum(axis=1)).ravel()

    # ---- assemble output AnnData --------------------------------------------
    obs = pd.DataFrame(index=pd.Index(labels, name="pseudobulk"))
    obs[sample_col] = key_frame[sample_col].to_numpy()[first_idx]
    if groups_col is not None:
        obs[groups_col] = key_frame[groups_col].to_numpy()[first_idx]
    obs["psbulk_n_cells"] = counts_per_profile.astype(int)
    obs["psbulk_counts"] = psbulk_counts

    # Carry over any other obs column that is constant within every profile.
    for col in adata.obs.columns:
        if col in (sample_col, groups_col):
            continue
        vals = adata.obs[col].astype(str).to_numpy()
        rep = vals[first_idx]
        if (vals == rep[inv]).all():
            obs[col] = rep

    out = ad.AnnData(X=agg, obs=obs, var=adata.var.copy())
    out.layers["psbulk_props"] = props
    out.uns["pseudobulk"] = {
        "sample_col": sample_col,
        "groups_col": groups_col,
        "mode": mode,
        "layer": layer if layer is not None else ("raw" if raw else "X"),
        "n_source_cells": int(n_cells),
    }

    # ---- filter tiny / low-count profiles -----------------------------------
    keep = np.ones(n_profiles, dtype=bool)
    if min_cells > 0:
        keep &= out.obs["psbulk_n_cells"].to_numpy() >= min_cells
    if min_counts > 0:
        keep &= out.obs["psbulk_counts"].to_numpy() >= min_counts
    if not keep.all():
        dropped = int((~keep).sum())
        print(f"[pseudobulk] dropped {dropped}/{n_profiles} profiles "
              f"below min_cells={min_cells} / min_counts={min_counts}")
        out = out[keep].copy()

    print(f"[pseudobulk] {out.n_obs} profiles × {out.n_vars} genes "
          f"(mode={mode}, from {n_cells} cells)")
    return out
