"""Marker identification and differential analysis for single-EV proteomics.

Single-extracellular-vesicle (single-EV) proteomic data is an EV x protein
matrix — a vesicle behaves like a "cell" and a protein marker like a "gene".
This module provides:

* :func:`rank_markers` — per-EV-subpopulation marker-protein identification
  (Wilcoxon rank-sum / t-test of a protein in a cluster vs the rest).
* :func:`differential_abundance` — per-protein differential abundance between
  two conditions across single EVs.
* :func:`differential_subpopulation` — test whether EV-subpopulation
  *frequencies* shift between two conditions (proportion test / GLM).

Pure Python on numpy / scipy / pandas; every public function is
``@register_function``-decorated under ``category='ev'``.
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
    ranked = np.clip(ranked, 0, 1)
    adj = np.empty(len(pv))
    adj[order] = ranked
    out[ok] = adj
    return out


@register_function(
    aliases=[
        "rank_markers", "ev_rank_markers", "rank_marker_proteins",
        "EV标志蛋白", "亚群标志物",
    ],
    category="ev",
    description=(
        "Per-EV-subpopulation marker-protein identification: for every "
        "cluster, test each protein in that cluster versus the rest with a "
        "Wilcoxon rank-sum (or t-test), report log fold-change / effect size "
        "and BH-FDR. Returns a tidy ranked table."
    ),
    examples=[
        "df = ov.single.ev.rank_markers(adata, groupby='leiden')",
        "df = ov.single.ev.rank_markers(adata, groupby='flowsom', "
        "method='t-test', n_top=10)",
    ],
    related=[
        "single.ev.differential_abundance",
        "pl.dotplot",
    ],
)
def rank_markers(
    adata,
    *,
    groupby: str,
    method: str = "wilcoxon",
    n_top: Optional[int] = None,
    min_fraction: float = 0.0,
):
    """Identify marker proteins for each EV subpopulation.

    Each protein is compared in one cluster against all other EVs; the test
    yields an effect size, a log fold-change and a BH-FDR-adjusted p-value.

    Parameters
    ----------
    adata
        Single-EV AnnData — ``X`` is the EV x protein matrix, ``var`` index
        holds protein names.
    groupby
        ``obs`` column with the EV-subpopulation / cluster labels.
    method
        ``'wilcoxon'`` (rank-sum, default) or ``'t-test'`` (Welch).
    n_top
        Keep only the ``n_top`` top-ranked proteins per cluster; ``None``
        keeps all proteins.
    min_fraction
        Drop a protein from a cluster's table when it is detected (value
        ``> 0``) in fewer than this fraction of the cluster's EVs.

    Returns
    -------
    :class:`pandas.DataFrame`
        Tidy table — one row per (cluster, protein): ``group``, ``protein``,
        ``log2fc``, ``effect_size``, ``mean_in``, ``mean_rest``,
        ``frac_in``, ``pval``, ``padj``, ``rank``.
    """
    from scipy import stats

    if groupby not in adata.obs:
        raise KeyError(f"obs[{groupby!r}] not found.")
    if method not in ("wilcoxon", "t-test"):
        raise ValueError("method must be 'wilcoxon' or 't-test'.")
    X = _dense(adata)
    proteins = list(adata.var_names)
    labels = adata.obs[groupby].astype(str).values
    groups = pd.unique(labels)

    rows = []
    for g in groups:
        mask = labels == g
        if mask.sum() == 0 or (~mask).sum() == 0:
            continue
        Xin, Xrest = X[mask], X[~mask]
        mean_in = Xin.mean(axis=0)
        mean_rest = Xrest.mean(axis=0)
        frac_in = (Xin > 0).mean(axis=0)
        for j, prot in enumerate(proteins):
            a, b = Xin[:, j], Xrest[:, j]
            if frac_in[j] < min_fraction:
                continue
            if np.allclose(a, a[0]) and np.allclose(b, b[0]) and \
                    np.isclose(a[0], b[0]):
                pval = 1.0
            elif method == "wilcoxon":
                try:
                    pval = float(
                        stats.mannwhitneyu(a, b, alternative="two-sided")[1]
                    )
                except ValueError:
                    pval = 1.0
            else:
                pval = float(stats.ttest_ind(a, b, equal_var=False)[1])
            sd = np.sqrt(
                (a.var(ddof=1) if len(a) > 1 else 0.0)
                + (b.var(ddof=1) if len(b) > 1 else 0.0)
            )
            effect = float((mean_in[j] - mean_rest[j]) / sd) if sd > 0 else 0.0
            log2fc = float(np.log2((mean_in[j] + 1e-9) / (mean_rest[j] + 1e-9)))
            rows.append({
                "group": g, "protein": prot,
                "log2fc": log2fc, "effect_size": effect,
                "mean_in": float(mean_in[j]),
                "mean_rest": float(mean_rest[j]),
                "frac_in": float(frac_in[j]),
                "pval": pval if pval == pval else 1.0,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["padj"] = _bh_fdr(df["pval"].values)
    df = df.sort_values(
        ["group", "effect_size"], ascending=[True, False]
    ).reset_index(drop=True)
    df["rank"] = df.groupby("group").cumcount() + 1
    if n_top is not None:
        df = df[df["rank"] <= n_top].reset_index(drop=True)
    return df


@register_function(
    aliases=[
        "differential_abundance", "ev_differential_abundance",
        "ev_da", "差异丰度", "蛋白差异丰度",
    ],
    category="ev",
    description=(
        "Per-protein differential abundance between two conditions across "
        "single EVs: Wilcoxon rank-sum or Welch t-test of each protein in "
        "condition A versus condition B, with log2 fold-change, effect size "
        "and BH-FDR."
    ),
    examples=[
        "df = ov.single.ev.differential_abundance(adata, "
        "condition_key='condition', group_a='tumor', group_b='healthy')",
    ],
    related=[
        "single.ev.rank_markers",
        "single.ev.differential_subpopulation",
        "single.ev.pseudobulk_de",
    ],
)
def differential_abundance(
    adata,
    *,
    condition_key: str,
    group_a: str,
    group_b: str,
    method: str = "wilcoxon",
):
    """Per-protein differential abundance between two conditions.

    Parameters
    ----------
    adata
        Single-EV AnnData.
    condition_key
        ``obs`` column holding the condition labels.
    group_a, group_b
        The two condition values to compare (``A`` vs ``B``); a positive
        ``log2fc`` means higher in ``group_a``.
    method
        ``'wilcoxon'`` (default) or ``'t-test'`` (Welch).

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per protein — ``protein``, ``log2fc``, ``effect_size``,
        ``mean_a``, ``mean_b``, ``n_a``, ``n_b``, ``pval``, ``padj`` —
        sorted by ascending ``padj``.
    """
    from scipy import stats

    if condition_key not in adata.obs:
        raise KeyError(f"obs[{condition_key!r}] not found.")
    if method not in ("wilcoxon", "t-test"):
        raise ValueError("method must be 'wilcoxon' or 't-test'.")
    labels = adata.obs[condition_key].astype(str).values
    mask_a = labels == str(group_a)
    mask_b = labels == str(group_b)
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        raise ValueError(
            f"condition '{group_a}' or '{group_b}' has no EVs in "
            f"obs[{condition_key!r}]."
        )
    X = _dense(adata)
    Xa, Xb = X[mask_a], X[mask_b]
    mean_a = Xa.mean(axis=0)
    mean_b = Xb.mean(axis=0)

    rows = []
    for j, prot in enumerate(adata.var_names):
        a, b = Xa[:, j], Xb[:, j]
        if np.allclose(a, a[0]) and np.allclose(b, b[0]) and \
                np.isclose(a[0], b[0]):
            pval = 1.0
        elif method == "wilcoxon":
            try:
                pval = float(
                    stats.mannwhitneyu(a, b, alternative="two-sided")[1]
                )
            except ValueError:
                pval = 1.0
        else:
            pval = float(stats.ttest_ind(a, b, equal_var=False)[1])
        sd = np.sqrt(
            (a.var(ddof=1) if len(a) > 1 else 0.0)
            + (b.var(ddof=1) if len(b) > 1 else 0.0)
        )
        effect = float((mean_a[j] - mean_b[j]) / sd) if sd > 0 else 0.0
        log2fc = float(np.log2((mean_a[j] + 1e-9) / (mean_b[j] + 1e-9)))
        rows.append({
            "protein": prot, "log2fc": log2fc, "effect_size": effect,
            "mean_a": float(mean_a[j]), "mean_b": float(mean_b[j]),
            "n_a": int(mask_a.sum()), "n_b": int(mask_b.sum()),
            "pval": pval if pval == pval else 1.0,
        })

    df = pd.DataFrame(rows)
    df["padj"] = _bh_fdr(df["pval"].values)
    return df.sort_values("padj").reset_index(drop=True)


@register_function(
    aliases=[
        "differential_subpopulation", "ev_differential_subpopulation",
        "ev_da_subpop", "亚群频率差异", "亚群组成差异",
    ],
    category="ev",
    description=(
        "Test whether EV-subpopulation frequencies shift between two "
        "conditions: per-cluster two-proportion z-test (pooled across EVs) "
        "or, when replicate samples are available, a per-sample-fraction "
        "Welch t-test (GLM-style). Reports BH-FDR-adjusted p-values."
    ),
    examples=[
        "df = ov.single.ev.differential_subpopulation(adata, "
        "condition_key='condition', cluster_key='leiden')",
        "df = ov.single.ev.differential_subpopulation(adata, "
        "condition_key='condition', cluster_key='leiden', sample_key='sample')",
    ],
    related=[
        "single.ev.differential_abundance",
        "pl.cellproportion",
    ],
)
def differential_subpopulation(
    adata,
    *,
    condition_key: str,
    cluster_key: str,
    group_a: Optional[str] = None,
    group_b: Optional[str] = None,
    sample_key: Optional[str] = None,
):
    """Test EV-subpopulation frequency shifts between two conditions.

    With ``sample_key`` the per-sample cluster fractions are compared by a
    Welch t-test (the recommended replicate-aware test); without it the EVs
    are pooled and a two-proportion z-test is used.

    Parameters
    ----------
    adata
        Single-EV AnnData.
    condition_key
        ``obs`` column with the condition labels.
    cluster_key
        ``obs`` column with the EV-subpopulation labels.
    group_a, group_b
        Condition values to compare; ``None`` uses the first two observed
        condition values. A positive ``delta_frac`` means enriched in
        ``group_a``.
    sample_key
        Optional ``obs`` column with replicate-sample ids; enables the
        per-sample-fraction Welch t-test.

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per cluster — ``cluster``, ``frac_a``, ``frac_b``,
        ``delta_frac``, ``log2_ratio``, ``stat``, ``test``, ``pval``,
        ``padj``.
    """
    from scipy import stats

    for k in (condition_key, cluster_key):
        if k not in adata.obs:
            raise KeyError(f"obs[{k!r}] not found.")
    obs = adata.obs
    conds = list(pd.unique(obs[condition_key].astype(str)))
    if group_a is None or group_b is None:
        if len(conds) < 2:
            raise ValueError("Need >= 2 conditions to compare.")
        group_a, group_b = conds[0], conds[1]
    group_a, group_b = str(group_a), str(group_b)
    cond = obs[condition_key].astype(str)
    clusters = list(pd.unique(obs[cluster_key].astype(str)))
    clu = obs[cluster_key].astype(str)

    rows = []
    if sample_key is not None and sample_key in obs:
        smp = obs[sample_key].astype(str)
        # per-sample cluster fractions
        fr = (
            pd.crosstab(smp, clu, normalize="index")
            .reindex(columns=clusters, fill_value=0.0)
        )
        samp_cond = (
            obs[[sample_key, condition_key]].astype(str)
            .drop_duplicates().set_index(sample_key)[condition_key]
        )
        sa = [s for s in fr.index if samp_cond.get(s) == group_a]
        sb = [s for s in fr.index if samp_cond.get(s) == group_b]
        for c in clusters:
            va = fr.loc[sa, c].values if sa else np.array([])
            vb = fr.loc[sb, c].values if sb else np.array([])
            fa = float(va.mean()) if va.size else 0.0
            fb = float(vb.mean()) if vb.size else 0.0
            if va.size > 1 and vb.size > 1:
                stat, pval = stats.ttest_ind(va, vb, equal_var=False)
                stat, pval = float(stat), float(pval)
            else:
                stat, pval = np.nan, 1.0
            rows.append({
                "cluster": c, "frac_a": fa, "frac_b": fb,
                "delta_frac": fa - fb,
                "log2_ratio": float(np.log2((fa + 1e-9) / (fb + 1e-9))),
                "stat": stat, "test": "welch_t", "pval": pval,
            })
    else:
        na = int((cond == group_a).sum())
        nb = int((cond == group_b).sum())
        if na == 0 or nb == 0:
            raise ValueError(
                f"condition '{group_a}' or '{group_b}' has no EVs."
            )
        for c in clusters:
            ka = int(((cond == group_a) & (clu == c)).sum())
            kb = int(((cond == group_b) & (clu == c)).sum())
            fa, fb = ka / na, kb / nb
            p_pool = (ka + kb) / (na + nb)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / na + 1 / nb))
            if se > 0:
                z = (fa - fb) / se
                pval = float(2 * stats.norm.sf(abs(z)))
            else:
                z, pval = 0.0, 1.0
            rows.append({
                "cluster": c, "frac_a": fa, "frac_b": fb,
                "delta_frac": fa - fb,
                "log2_ratio": float(np.log2((fa + 1e-9) / (fb + 1e-9))),
                "stat": float(z), "test": "two_proportion_z", "pval": pval,
            })

    df = pd.DataFrame(rows)
    df["padj"] = _bh_fdr(df["pval"].values)
    return df.sort_values("padj").reset_index(drop=True)
