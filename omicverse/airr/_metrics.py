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


def _diversity_value(vc, metric: str):
    """Single alpha-diversity value for one clonotype-count vector ``vc``."""
    n = int(vc.sum())
    p = vc.values / n
    rich = int(len(vc))
    if metric == "shannon":
        return float(-(p * np.log(p)).sum())
    if metric == "normalized_shannon":
        h = -(p * np.log(p)).sum()
        return float(h / np.log(rich)) if rich > 1 else 0.0
    if metric == "inverse_simpson":
        return float(1.0 / (p ** 2).sum())
    if metric == "gini_simpson":
        return float(1.0 - (p ** 2).sum())
    if metric == "richness":
        return rich
    # d50
    srt = np.sort(vc.values)[::-1]
    cum = np.cumsum(srt)
    return int(np.searchsorted(cum, n / 2.0) + 1)


@register_function(
    aliases=["alpha_diversity", "airr_diversity", "多样性", "Alpha多样性"],
    category="airr",
    description=(
        "Compute alpha-diversity of the clonotype distribution per cell "
        "group: Shannon entropy, normalized Shannon, inverse Simpson, Gini-"
        "Simpson, observed richness or D50. metric may be a list to return "
        "several diversity columns in one table."
    ),
    requires={"obs": ["clone_id"]},
    examples=[
        "df = ov.airr.alpha_diversity(adata, groupby='group')",
        "df = ov.airr.alpha_diversity(adata, groupby='sample', metric='shannon')",
        "df = ov.airr.alpha_diversity(adata, groupby='source', "
        "metric=['normalized_shannon', 'gini_simpson', 'd50'])",
    ],
    related=["airr.define_clonotypes", "airr.repertoire_overlap"],
)
def alpha_diversity(
    adata,
    groupby: Optional[str] = None,
    *,
    target_col: str = "clone_id",
    metric="shannon",
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
        ``'gini_simpson'`` | ``'richness'`` | ``'d50'``. May also be a list
        of metric names — every requested metric is returned as its own
        column in a single table.

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per group, columns ``n_cells``, ``n_clonotypes`` and one
        column per requested ``metric``.
    """
    valid = {
        "shannon", "normalized_shannon", "inverse_simpson",
        "gini_simpson", "richness", "d50",
    }
    single = isinstance(metric, str)
    metrics = [metric] if single else list(metric)
    bad = [m for m in metrics if m not in valid]
    if bad:
        raise ValueError(f"metric must be one of {sorted(valid)}, got {bad}")
    counts = _clone_counts(adata, groupby, target_col)

    rows = []
    for g, vc in counts.items():
        n = int(vc.sum())
        if n == 0:
            continue
        row = {"group": g, "n_cells": n, "n_clonotypes": int(len(vc))}
        for m in metrics:
            row[m] = _diversity_value(vc, m)
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


@register_function(
    aliases=[
        "summarize_by_group", "airr_summarize_by_group",
        "分组汇总", "分组统计",
    ],
    category="airr",
    description=(
        "Generic group reducer for repertoire metrics: join a Series or "
        "DataFrame to a group label, then groupby + aggregate (mean/std/min/"
        "max/...), optionally pivoting a second grouping level into columns."
    ),
    examples=[
        "tab = ov.airr.summarize_by_group(top10, group=meta['group'])",
        "tab = ov.airr.summarize_by_group(homeo, group='group', agg='mean')",
        "tab = ov.airr.summarize_by_group(hill, group=['group', 'Q'], "
        "value='Value', agg='mean', pivot='group')",
    ],
    related=["airr.alpha_diversity", "airr.group_abundance"],
)
def summarize_by_group(
    data,
    *,
    group,
    value=None,
    agg=("mean", "std"),
    pivot=False,
):
    """Group-and-aggregate any repertoire metric table.

    A generic reducer for the repeated ``.join(meta).groupby(group).agg(...)``
    pattern found across the bulk / BCR tutorials.

    Parameters
    ----------
    data
        A :class:`pandas.Series` or :class:`pandas.DataFrame` of metric
        values, indexed by sample (or any key).
    group
        The group label(s). Either a column name (or list of names) already
        present in ``data``, or a :class:`pandas.Series` / DataFrame that is
        index-aligned and joined onto ``data`` first.
    value
        Column(s) of ``data`` to aggregate. ``None`` keeps every numeric
        column not used for grouping.
    agg
        Aggregation function name or list of names passed to
        :meth:`pandas.core.groupby.GroupBy.agg` (e.g. ``'mean'``,
        ``('mean', 'std')``, ``['mean', 'min', 'max']``).
    pivot
        If a group level name (or ``True``), ``unstack`` that level so its
        categories become columns — the wide ``hill_profile`` layout.

    Returns
    -------
    :class:`pandas.DataFrame`
        The aggregated (and optionally pivoted) summary table.
    """
    df = data.to_frame() if isinstance(data, pd.Series) else data.copy()

    # resolve / attach the grouping column(s)
    if isinstance(group, (pd.Series, pd.DataFrame)):
        gobj = group.to_frame() if isinstance(group, pd.Series) else group
        if isinstance(group, pd.Series) and group.name is None:
            gobj = gobj.rename(columns={gobj.columns[0]: "group"})
        df = df.join(gobj, how="inner")
        group_cols = list(gobj.columns)
    else:
        group_cols = [group] if isinstance(group, str) else list(group)
    missing = [g for g in group_cols if g not in df.columns]
    if missing:
        raise KeyError(f"group column(s) {missing} not found in data.")

    if value is None:
        value_cols = [c for c in df.columns if c not in group_cols]
        value_cols = [
            c for c in value_cols
            if pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        value_cols = [value] if isinstance(value, str) else list(value)

    # avoid index-level / column-label ambiguity on groupby
    clash = [g for g in group_cols if g in (df.index.names or [])]
    if clash:
        df = df.reset_index(drop=True)
    gb = df.groupby(group_cols, observed=True)[value_cols]
    out = gb.agg(agg)

    if pivot is not False and pivot is not None:
        level = group_cols[-1] if pivot is True else pivot
        out = out.unstack(level)
    return out


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


@register_function(
    aliases=["cluster_purity", "airr_cluster_purity", "聚类纯度", "簇纯度"],
    category="airr",
    description=(
        "Score clustering quality against a ground-truth label: per-cluster "
        "purity (fraction of the dominant truth label) plus a size-weighted "
        "mean purity over all clusters."
    ),
    examples=[
        "tab, wmp = ov.airr.cluster_purity(labels, truth_epitope)",
        "tab, wmp = ov.airr.cluster_purity(nb['labels'], truth, ignore_label=-1)",
    ],
    related=["airr.benchmark_clustering", "airr.label_agreement"],
)
def cluster_purity(labels, truth, *, ignore_label=-1):
    """Per-cluster purity vs a ground-truth label, with a weighted summary.

    For each cluster the *purity* is the fraction of its members carrying
    the most common ground-truth label; clusters equal to ``ignore_label``
    (the unclustered / singleton bucket) are skipped.

    Parameters
    ----------
    labels
        Predicted cluster labels (array-like, one per item).
    truth
        Ground-truth labels (array-like, same length / order as ``labels``).
    ignore_label
        Cluster label treated as "not clustered" and excluded (default
        ``-1``); set to ``None`` to keep every cluster.

    Returns
    -------
    tab : :class:`pandas.DataFrame`
        One row per cluster — ``cluster``, ``size``, ``top_epitope``,
        ``purity``.
    weighted_purity : float
        Size-weighted mean of the per-cluster purities.
    """
    labels = np.asarray(labels)
    truth = np.asarray(truth)
    rows = []
    for cl in np.unique(labels):
        if ignore_label is not None and cl == ignore_label:
            continue
        if ignore_label is not None and np.isscalar(cl):
            try:
                if cl < 0 and ignore_label == -1:
                    continue
            except TypeError:
                pass
        sel = labels == cl
        vc = pd.Series(truth[sel]).value_counts()
        rows.append({
            "cluster": cl, "size": int(sel.sum()),
            "top_epitope": vc.index[0],
            "purity": float(vc.iloc[0] / sel.sum()),
        })
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab, float("nan")
    wmp = float((tab["purity"] * tab["size"]).sum() / tab["size"].sum())
    return tab, wmp


@register_function(
    aliases=["label_agreement", "airr_label_agreement", "标签一致性", "标签吻合度"],
    category="airr",
    description=(
        "Compare a predicted-label array against a ground-truth label array "
        "(ignoring missing predictions): accuracy / agreement fraction, the "
        "number of comparable items and the number that agree."
    ),
    examples=[
        "res = ov.airr.label_agreement(pred_epitope, truth_epitope)",
        "print(res['agreement'])",
    ],
    related=["airr.cluster_purity", "airr.benchmark_clustering"],
)
def label_agreement(pred, truth):
    """Accuracy / agreement of a predicted-label array vs a truth array.

    Items whose prediction is missing (``NaN`` / ``None``) are ignored; the
    agreement is computed over the remaining comparable items only.

    Parameters
    ----------
    pred
        Predicted labels (array-like or :class:`pandas.Series`). Missing
        values are dropped before scoring.
    truth
        Ground-truth labels (array-like, same length / order as ``pred``).

    Returns
    -------
    dict
        ``n_total`` (items), ``n_compared`` (non-missing predictions),
        ``n_agree`` (matching predictions) and ``agreement`` (the
        agreement fraction, ``nan`` when nothing is comparable).
    """
    pred = pd.Series(np.asarray(pred, dtype=object)).reset_index(drop=True)
    truth = np.asarray(truth, dtype=object)
    mask = pred.notna().to_numpy()
    p = pred.to_numpy()[mask]
    t = truth[mask]
    n_agree = int((p == t).sum())
    n_cmp = int(mask.sum())
    return {
        "n_total": int(len(pred)),
        "n_compared": n_cmp,
        "n_agree": n_agree,
        "agreement": float(n_agree / n_cmp) if n_cmp else float("nan"),
    }


@register_function(
    aliases=[
        "benchmark_clustering", "airr_benchmark_clustering",
        "聚类基准", "聚类比较",
    ],
    category="airr",
    description=(
        "Benchmark several clustering results against a ground-truth label: "
        "for each method report n_clusters, n_clustered items and the size-"
        "weighted mean purity (via cluster_purity)."
    ),
    examples=[
        "tab = ov.airr.benchmark_clustering("
        "{'neighbors': nb_lab, 'hclust': hc_lab}, truth_epitope)",
    ],
    related=["airr.cluster_purity", "airr.label_agreement"],
)
def benchmark_clustering(results, truth):
    """Compare several clustering methods against a ground-truth label.

    Parameters
    ----------
    results
        Mapping ``{method_name: label_array}`` — every value is a predicted
        cluster-label array aligned to ``truth``.
    truth
        Ground-truth labels shared by all methods.

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per method — ``method``, ``n_clusters``, ``n_clustered``
        (items assigned to a real cluster) and ``weighted_purity``.
    """
    rows = []
    for name, labels in results.items():
        tab, wmp = cluster_purity(labels, truth)
        rows.append({
            "method": name,
            "n_clusters": int(len(tab)),
            "n_clustered": int(tab["size"].sum()) if not tab.empty else 0,
            "weighted_purity": wmp,
        })
    return pd.DataFrame(rows)
