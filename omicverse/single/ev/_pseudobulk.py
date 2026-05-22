"""Pseudo-bulk aggregation and bulk-style DE for single-EV proteomics.

Single-EV proteomic atlases hold 10^4-10^6 vesicles; collapsing the per-EV
signal to a sample x protein matrix recovers a classic "bulk" measurement on
which moderated-t / OLS differential expression can be run for biomarker
discovery.

* :func:`pseudobulk`    — aggregate per-EV signal to a sample x protein
  matrix (sum / mean), returned as a new AnnData.
* :func:`pseudobulk_de` — bulk-style differential expression on the
  pseudo-bulk matrix (moderated-t / OLS) with BH-FDR.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..._registry import register_function


def _dense(adata):
    """Return ``adata.X`` as a dense float ndarray."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction for a 1-D array of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    out = np.full(n, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[ok] = np.clip(ranked, 0, 1)[np.argsort(order)]
    return out


@register_function(
    aliases=[
        "pseudobulk", "ev_pseudobulk", "pseudo_bulk",
        "拟bulk", "EV拟bulk聚合",
    ],
    category="ev",
    description=(
        "Aggregate per-EV single-EV signal into a sample x protein "
        "pseudo-bulk matrix (sum or mean over EVs of each sample). Returns a "
        "new AnnData whose obs are samples and whose var are proteins; "
        "carries per-sample EV counts and any constant per-sample metadata."
    ),
    examples=[
        "pb = ov.single.ev.pseudobulk(adata, sample_key='sample')",
        "pb = ov.single.ev.pseudobulk(adata, sample_key='sample', "
        "mode='mean', condition_key='condition')",
    ],
    related=["single.ev.pseudobulk_de", "single.ev.differential_abundance"],
)
def pseudobulk(
    adata,
    *,
    sample_key: str,
    mode: str = "sum",
    condition_key: Optional[str] = None,
    layer: Optional[str] = None,
    min_evs: int = 1,
):
    """Aggregate per-EV signal to a sample x protein pseudo-bulk matrix.

    Parameters
    ----------
    adata
        Single-EV AnnData.
    sample_key
        ``obs`` column with the sample ids — one pseudo-bulk row per sample.
    mode
        ``'sum'`` (default) or ``'mean'`` aggregation across the EVs of a
        sample.
    condition_key
        Optional ``obs`` column to carry through; the (constant) condition
        value of each sample is copied to ``pb.obs``.
    layer
        Optional ``layers`` key to aggregate instead of ``X``.
    min_evs
        Drop samples that contributed fewer than ``min_evs`` EVs.

    Returns
    -------
    :class:`anndata.AnnData`
        Pseudo-bulk AnnData — ``obs`` indexed by sample (with ``n_evs`` and,
        if requested, the condition column), ``var`` the proteins,
        ``uns['ev']`` copied from the input.
    """
    import anndata as ad

    if sample_key not in adata.obs:
        raise KeyError(f"obs[{sample_key!r}] not found.")
    if mode not in ("sum", "mean"):
        raise ValueError("mode must be 'sum' or 'mean'.")
    if layer is not None:
        X = adata.layers[layer]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, float)
    else:
        X = _dense(adata)
    samples = adata.obs[sample_key].astype(str).values
    order = list(pd.unique(samples))

    mats, obs_rows, keep = [], [], []
    for s in order:
        mask = samples == s
        n = int(mask.sum())
        if n < min_evs:
            continue
        agg = X[mask].sum(axis=0) if mode == "sum" else X[mask].mean(axis=0)
        mats.append(agg)
        row = {"sample": s, "n_evs": n}
        if condition_key is not None and condition_key in adata.obs:
            vals = pd.unique(adata.obs[condition_key][mask].astype(str))
            row[condition_key] = vals[0] if len(vals) == 1 else "mixed"
        obs_rows.append(row)
        keep.append(s)

    if not mats:
        raise ValueError("No sample passed the min_evs filter.")
    pb_X = np.vstack(mats).astype(float)
    obs = pd.DataFrame(obs_rows).set_index("sample")
    obs.index = obs.index.astype(str)
    var = adata.var.copy()
    pb = ad.AnnData(X=pb_X, obs=obs, var=var)
    pb.uns["ev"] = dict(adata.uns.get("ev", {}))
    pb.uns["ev"]["pseudobulk_mode"] = mode
    return pb


@register_function(
    aliases=[
        "pseudobulk_de", "ev_pseudobulk_de", "pb_de",
        "拟bulk差异表达", "EV生物标志物发现",
    ],
    category="ev",
    description=(
        "Classic bulk-style differential expression on a single-EV "
        "pseudo-bulk matrix: per-protein moderated-t (limma-style empirical-"
        "Bayes variance shrinkage) or ordinary OLS t-test between two "
        "conditions, with log2 fold-change and BH-FDR — for biomarker "
        "discovery."
    ),
    examples=[
        "res = ov.single.ev.pseudobulk_de(pb, condition_key='condition')",
        "res = ov.single.ev.pseudobulk_de(pb, condition_key='condition', "
        "group_a='tumor', group_b='healthy', method='ols')",
    ],
    related=["single.ev.pseudobulk", "single.ev.differential_abundance"],
)
def pseudobulk_de(
    pb,
    *,
    condition_key: str,
    group_a: Optional[str] = None,
    group_b: Optional[str] = None,
    method: str = "moderated_t",
    log_transform: bool = True,
):
    """Bulk-style differential expression on a pseudo-bulk matrix.

    Parameters
    ----------
    pb
        Pseudo-bulk AnnData from :func:`pseudobulk` (samples x proteins).
    condition_key
        ``obs`` column with the condition labels.
    group_a, group_b
        Condition values to compare; ``None`` uses the first two observed
        values. A positive ``log2fc`` means higher in ``group_a``.
    method
        ``'moderated_t'`` (limma-style empirical-Bayes shrinkage, default)
        or ``'ols'`` (Welch t-test).
    log_transform
        ``log1p`` the matrix before testing (recommended for ``count`` /
        ``intensity`` value types).

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per protein — ``protein``, ``log2fc``, ``mean_a``,
        ``mean_b``, ``t``, ``pval``, ``padj`` — sorted by ascending
        ``padj``.
    """
    from scipy import stats

    if condition_key not in pb.obs:
        raise KeyError(f"obs[{condition_key!r}] not found.")
    if method not in ("moderated_t", "ols"):
        raise ValueError("method must be 'moderated_t' or 'ols'.")
    cond = pb.obs[condition_key].astype(str)
    conds = list(pd.unique(cond))
    if group_a is None or group_b is None:
        if len(conds) < 2:
            raise ValueError("Need >= 2 conditions to compare.")
        group_a, group_b = conds[0], conds[1]
    group_a, group_b = str(group_a), str(group_b)
    mask_a = (cond == group_a).values
    mask_b = (cond == group_b).values
    if mask_a.sum() < 2 or mask_b.sum() < 2:
        raise ValueError(
            "Each condition needs >= 2 pseudo-bulk samples for DE."
        )
    X = pb.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, float)
    X = np.asarray(X, dtype=float)
    if log_transform:
        X = np.log1p(np.clip(X, 0, None))
    Xa, Xb = X[mask_a], X[mask_b]
    na, nb = Xa.shape[0], Xb.shape[0]
    mean_a, mean_b = Xa.mean(axis=0), Xb.mean(axis=0)
    var_a = Xa.var(axis=0, ddof=1)
    var_b = Xb.var(axis=0, ddof=1)

    diff = mean_a - mean_b
    if method == "ols":
        se = np.sqrt(var_a / na + var_b / nb)
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = np.where(se > 0, diff / se, 0.0)
        df_t = np.full(X.shape[1], na + nb - 2)
        pval = 2 * stats.t.sf(np.abs(tstat), df=df_t)
    else:
        # limma-style pooled variance + empirical-Bayes shrinkage
        dfree = na + nb - 2
        s2 = ((na - 1) * var_a + (nb - 1) * var_b) / dfree
        s2 = np.clip(s2, 1e-12, None)
        s2_prior = float(np.exp(np.mean(np.log(s2))))  # geometric mean
        d0 = 4.0  # prior degrees of freedom (limma default-ish)
        s2_post = (d0 * s2_prior + dfree * s2) / (d0 + dfree)
        se = np.sqrt(s2_post * (1.0 / na + 1.0 / nb))
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = np.where(se > 0, diff / se, 0.0)
        df_total = dfree + d0
        pval = 2 * stats.t.sf(np.abs(tstat), df=df_total)

    log2fc = (diff / np.log(2)) if log_transform else np.log2(
        (mean_a + 1e-9) / (mean_b + 1e-9)
    )
    df = pd.DataFrame({
        "protein": list(pb.var_names),
        "log2fc": np.asarray(log2fc, dtype=float),
        "mean_a": mean_a, "mean_b": mean_b,
        "t": np.asarray(tstat, dtype=float),
        "pval": np.asarray(pval, dtype=float),
    })
    df["padj"] = _bh_fdr(df["pval"].values)
    return df.sort_values("padj").reset_index(drop=True)
