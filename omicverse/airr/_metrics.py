"""Repertoire metrics for single-cell AIRR data.

Diversity, repertoire overlap, group abundance, spectratype, clonotype
modularity and V(D)J gene usage — an AnnData-native reimplementation of the
core of scirpy's ``tl`` repertoire-metric functions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .._registry import register_function


def _vc(series):
    """``value_counts`` that drops zero-count (unobserved categorical) levels."""
    vc = series.value_counts()
    return vc[vc > 0]


def _clone_counts(adata, groupby: Optional[str], target_col: str):
    """Per-group clonotype-size table (``group`` -> {clone -> count})."""
    sub = adata.obs.dropna(subset=[target_col])
    if groupby is None:
        return {"all": _vc(sub[target_col])}
    return {
        g: _vc(gd[target_col])
        for g, gd in sub.groupby(groupby, observed=True)
    }


@register_function(
    aliases=["alpha_diversity", "airr_diversity", "多样性", "Alpha多样性"],
    category="airr",
    description=(
        "Compute alpha-diversity of the clonotype distribution per cell "
        "group: Shannon entropy, normalized Shannon, inverse Simpson, Gini-"
        "Simpson, observed richness or D50."
    ),
    requires={"obs": ["clone_id"]},
    examples=[
        "df = ov.airr.alpha_diversity(adata, groupby='group')",
        "df = ov.airr.alpha_diversity(adata, groupby='sample', metric='shannon')",
    ],
    related=["airr.define_clonotypes", "airr.repertoire_overlap"],
)
def alpha_diversity(
    adata,
    groupby: Optional[str] = None,
    *,
    target_col: str = "clone_id",
    metric: str = "shannon",
):
    """Alpha-diversity of the clonotype distribution.

    Parameters
    ----------
    adata
        AnnData with a clonotype column from
        :func:`omicverse.airr.define_clonotypes`.
    groupby
        ``obs`` column to compute diversity per group; ``None`` pools all
        cells.
    target_col
        Clonotype id column (default ``'clone_id'``).
    metric
        ``'shannon'`` | ``'normalized_shannon'`` | ``'inverse_simpson'`` |
        ``'gini_simpson'`` | ``'richness'`` | ``'d50'``.

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per group, columns ``n_cells``, ``n_clonotypes`` and the
        chosen ``metric``.
    """
    valid = {
        "shannon", "normalized_shannon", "inverse_simpson",
        "gini_simpson", "richness", "d50",
    }
    if metric not in valid:
        raise ValueError(f"metric must be one of {sorted(valid)}")
    counts = _clone_counts(adata, groupby, target_col)

    rows = []
    for g, vc in counts.items():
        n = int(vc.sum())
        if n == 0:
            continue
        p = vc.values / n
        rich = int(len(vc))
        if metric == "shannon":
            val = float(-(p * np.log(p)).sum())
        elif metric == "normalized_shannon":
            h = -(p * np.log(p)).sum()
            val = float(h / np.log(rich)) if rich > 1 else 0.0
        elif metric == "inverse_simpson":
            val = float(1.0 / (p ** 2).sum())
        elif metric == "gini_simpson":
            val = float(1.0 - (p ** 2).sum())
        elif metric == "richness":
            val = rich
        else:  # d50
            srt = np.sort(vc.values)[::-1]
            cum = np.cumsum(srt)
            val = int(np.searchsorted(cum, n / 2.0) + 1)
        rows.append({
            "group": g, "n_cells": n, "n_clonotypes": rich, metric: val,
        })
    return pd.DataFrame(rows).set_index("group")


@register_function(
    aliases=["repertoire_overlap", "airr_overlap", "组库重叠", "克隆型重叠"],
    category="airr",
    description=(
        "Compute a pairwise repertoire-overlap matrix between cell groups "
        "of a single-cell AIRR AnnData (jaccard / public / morisita / "
        "cosine)."
    ),
    requires={"obs": ["clone_id"]},
    examples=[
        "mat = ov.airr.repertoire_overlap(adata, groupby='sample')",
        "mat = ov.airr.repertoire_overlap(adata, groupby='group', metric='morisita')",
    ],
    related=["airr.alpha_diversity", "airr.plotting.repertoire_overlap_plot"],
)
def repertoire_overlap(
    adata,
    groupby: str,
    *,
    target_col: str = "clone_id",
    metric: str = "jaccard",
):
    """Pairwise repertoire-overlap matrix between cell groups.

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData with a clonotype column.
    groupby
        ``obs`` column defining the groups (samples / conditions).
    target_col
        Clonotype id column (default ``'clone_id'``).
    metric
        ``'jaccard'`` | ``'public'`` (shared count) | ``'morisita'`` |
        ``'cosine'``.

    Returns
    -------
    :class:`pandas.DataFrame`
        A symmetric ``n_groups x n_groups`` overlap matrix.
    """
    valid = {"jaccard", "public", "morisita", "cosine"}
    if metric not in valid:
        raise ValueError(f"metric must be one of {sorted(valid)}")
    counts = _clone_counts(adata, groupby, target_col)
    groups = list(counts.keys())
    all_clones = sorted(set().union(*[set(c.index) for c in counts.values()]))
    mat = pd.DataFrame(
        np.zeros((len(groups), len(groups))), index=groups, columns=groups
    )
    vecs = {g: counts[g].reindex(all_clones, fill_value=0).values for g in groups}
    sets = {g: set(counts[g].index) for g in groups}
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            if metric == "jaccard":
                u = len(sets[gi] | sets[gj])
                v = len(sets[gi] & sets[gj]) / u if u else 0.0
            elif metric == "public":
                v = float(len(sets[gi] & sets[gj]))
            elif metric == "cosine":
                a, b = vecs[gi].astype(float), vecs[gj].astype(float)
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                v = float(a @ b / denom) if denom else 0.0
            else:  # morisita
                a, b = vecs[gi].astype(float), vecs[gj].astype(float)
                na, nb = a.sum(), b.sum()
                if na == 0 or nb == 0:
                    v = 0.0
                else:
                    pa, pb = a / na, b / nb
                    num = 2 * (pa * pb).sum()
                    den = (pa ** 2).sum() + (pb ** 2).sum()
                    v = float(num / den) if den else 0.0
            mat.iloc[i, j] = v
    return mat


@register_function(
    aliases=["group_abundance", "airr_group_abundance", "分组丰度", "克隆型丰度"],
    category="airr",
    description=(
        "Cross-tabulate clonotype (or any obs category) abundance against a "
        "cell-group column — counts or fractions, the basis of stacked-bar "
        "repertoire plots."
    ),
    examples=[
        "df = ov.airr.group_abundance(adata, groupby='group', target_col='clone_id')",
        "df = ov.airr.group_abundance(adata, groupby='sample', normalize=True)",
    ],
    related=["airr.clonal_expansion", "airr.spectratype"],
)
def group_abundance(
    adata,
    groupby: str,
    *,
    target_col: str = "clone_id",
    normalize: bool = False,
    max_cols: Optional[int] = None,
):
    """Cross-tabulate a category against cell groups.

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData.
    groupby
        ``obs`` column with the cell groups (rows of the output).
    target_col
        ``obs`` column whose categories form the columns (e.g.
        ``'clone_id'``, ``'clonal_expansion'``).
    normalize
        Return per-group fractions instead of raw counts.
    max_cols
        Keep only the ``max_cols`` most abundant categories.

    Returns
    -------
    :class:`pandas.DataFrame`
        ``groups x categories`` count (or fraction) table.
    """
    sub = adata.obs.dropna(subset=[target_col])
    tab = pd.crosstab(sub[groupby], sub[target_col])
    if max_cols is not None and tab.shape[1] > max_cols:
        top = tab.sum(axis=0).sort_values(ascending=False).index[:max_cols]
        tab = tab[top]
    if normalize:
        tab = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return tab


@register_function(
    aliases=["spectratype", "airr_spectratype", "谱型分析", "CDR3长度分布"],
    category="airr",
    description=(
        "Compute the spectratype — the distribution of CDR3 lengths per cell "
        "group — for a single-cell AIRR AnnData."
    ),
    examples=[
        "df = ov.airr.spectratype(adata, groupby='group')",
        "df = ov.airr.spectratype(adata, groupby='sample', chain='VJ_1')",
    ],
    related=["airr.group_abundance", "airr.vdj_usage"],
)
def spectratype(
    adata,
    groupby: Optional[str] = None,
    *,
    chain: str = "VDJ_1",
    sequence: str = "aa",
):
    """CDR3-length distribution (spectratype) per cell group.

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData.
    groupby
        ``obs`` column for the groups; ``None`` pools all cells.
    chain
        Chain slot — ``'VJ_1'`` / ``'VJ_2'`` / ``'VDJ_1'`` / ``'VDJ_2'``.
    sequence
        ``'aa'`` (default) or ``'nt'``.

    Returns
    -------
    :class:`pandas.DataFrame`
        ``groups x CDR3-length`` count table.
    """
    field = "junction_aa" if sequence == "aa" else "junction"
    col = f"{chain}_{field}"
    if col not in adata.obs:
        raise KeyError(f"obs[{col!r}] not found.")
    df = adata.obs[[col]].copy()
    df["length"] = df[col].map(
        lambda s: len(str(s)) if (s is not None and s == s
                                  and str(s) not in ("None", "nan")) else np.nan
    )
    df = df.dropna(subset=["length"])
    df["length"] = df["length"].astype(int)
    if groupby is None:
        return (
            df["length"].value_counts().sort_index().to_frame("all").T
        )
    df["__g"] = adata.obs.loc[df.index, groupby].values
    return pd.crosstab(df["__g"], df["length"])


@register_function(
    aliases=["vdj_usage", "gene_usage", "airr_vdj_usage", "基因使用", "VDJ使用"],
    category="airr",
    description=(
        "Compute V/D/J gene-segment usage frequencies per cell group for a "
        "single-cell AIRR AnnData."
    ),
    examples=[
        "df = ov.airr.vdj_usage(adata, gene='v', chain='VDJ_1')",
        "df = ov.airr.vdj_usage(adata, gene='j', groupby='group', normalize=True)",
    ],
    related=["airr.spectratype", "airr.plotting.vdj_usage_plot"],
)
def vdj_usage(
    adata,
    *,
    gene: str = "v",
    chain: str = "VDJ_1",
    groupby: Optional[str] = None,
    normalize: bool = True,
):
    """V/D/J gene-segment usage frequencies.

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData.
    gene
        ``'v'`` | ``'d'`` | ``'j'`` | ``'c'``.
    chain
        Chain slot — ``'VJ_1'`` / ``'VDJ_1'`` etc.
    groupby
        ``obs`` column for per-group usage; ``None`` pools all cells.
    normalize
        Return frequencies (default) instead of raw counts.

    Returns
    -------
    :class:`pandas.DataFrame`
        ``groups x gene`` usage table (frequencies or counts).
    """
    col = f"{chain}_{gene}_gene"
    if col not in adata.obs:
        raise KeyError(f"obs[{col!r}] not found.")
    df = adata.obs[[col]].copy()
    df = df[df[col].map(
        lambda s: s is not None and s == s and str(s) not in ("None", "nan")
    )]
    if groupby is None:
        vc = df[col].value_counts()
        out = vc.to_frame("all").T
    else:
        df["__g"] = adata.obs.loc[df.index, groupby].values
        out = pd.crosstab(df["__g"], df[col])
    if normalize:
        out = out.div(out.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return out


@register_function(
    aliases=["clonotype_modularity", "airr_modularity", "克隆型模块度"],
    category="airr",
    description=(
        "Score how transcriptionally connected the cells of each clonotype "
        "are: the fraction of a clonotype's cells that fall in the same "
        "transcriptomic cluster (a lightweight modularity proxy)."
    ),
    requires={"obs": ["clone_id"]},
    examples=[
        "df = ov.airr.clonotype_modularity(adata, cluster_key='leiden')",
    ],
    related=["airr.define_clonotypes", "airr.clonotype_network"],
)
def clonotype_modularity(
    adata,
    cluster_key: str,
    *,
    target_col: str = "clone_id",
):
    """Clonotype transcriptomic-modularity score.

    For each clonotype, the score is the largest fraction of its cells that
    share a single transcriptomic cluster — a value near ``1`` means the
    clonotype's cells are transcriptionally homogeneous.

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData (also carrying a transcriptomic clustering).
    cluster_key
        ``obs`` column with the transcriptomic cluster labels.
    target_col
        Clonotype id column (default ``'clone_id'``).

    Returns
    -------
    :class:`pandas.DataFrame`
        Per-clonotype ``size``, ``modularity_score``, ``dominant_cluster``.
    """
    if cluster_key not in adata.obs:
        raise KeyError(f"obs[{cluster_key!r}] not found.")
    sub = adata.obs.dropna(subset=[target_col])
    rows = []
    for clone, cd in sub.groupby(target_col, observed=True):
        n = len(cd)
        vc = cd[cluster_key].value_counts()
        rows.append({
            "clone_id": clone, "size": n,
            "modularity_score": float(vc.iloc[0] / n),
            "dominant_cluster": vc.index[0],
        })
    return pd.DataFrame(rows).sort_values(
        "size", ascending=False
    ).reset_index(drop=True)
