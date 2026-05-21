"""Clonotype definition, distance and networks for single-cell AIRR data.

A clean, AnnData-native reimplementation of the core of scirpy's clonotype
machinery:

* :func:`ir_dist` — pairwise CDR3 sequence-distance matrices
  (identity / hamming / levenshtein / alignment).
* :func:`define_clonotypes` — exact-identity clonotypes (cells with the same
  CDR3 nucleotide sequences belong to one clonotype).
* :func:`define_clonotype_clusters` — distance-based clonotype *clusters*
  (cells within a CDR3 distance cutoff are merged).
* :func:`clonal_expansion` — per-cell clonal-expansion category.
* :func:`clonotype_network` — a 2-D layout of the clonotype graph.
* :func:`clonotype_imbalance` — clonotypes enriched between two cell groups.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from .._registry import register_function


# ---------------------------------------------------------------------------
# Sequence-distance primitives
# ---------------------------------------------------------------------------
def _hamming(a: str, b: str) -> int:
    """Hamming distance — ``inf`` when lengths differ."""
    if len(a) != len(b):
        return 10 ** 9
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def _levenshtein(a: str, b: str) -> int:
    """Plain Levenshtein edit distance."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            )
        prev = cur
    return prev[lb]


# BLOSUM-ish: distance = (alignment penalty); approximated by length-normalised
# Levenshtein scaled to integer for the 'alignment' metric.
def _alignment_dist(a: str, b: str) -> int:
    """Cheap alignment-style distance — Levenshtein on the longer scale."""
    d = _levenshtein(a, b)
    return d


_METRICS = {
    "identity": lambda a, b: 0 if a == b else 10 ** 9,
    "hamming": _hamming,
    "levenshtein": _levenshtein,
    "alignment": _alignment_dist,
}


def _primary_cdr3(adata, sequence: str = "aa") -> pd.Series:
    """Concatenated VJ_1 + VDJ_1 CDR3 string per cell (the clonotype key)."""
    field = "junction_aa" if sequence == "aa" else "junction"
    vj = adata.obs.get(f"VJ_1_{field}")
    vdj = adata.obs.get(f"VDJ_1_{field}")

    def _fmt(v):
        return "" if (v is None or v != v or str(v) in ("None", "nan")) else str(v)

    keys = []
    for i in range(adata.n_obs):
        a = _fmt(vj.iloc[i]) if vj is not None else ""
        d = _fmt(vdj.iloc[i]) if vdj is not None else ""
        keys.append(f"{a}|{d}")
    return pd.Series(keys, index=adata.obs_names)


@register_function(
    aliases=["ir_dist", "airr_ir_dist", "免疫受体距离", "CDR3距离"],
    category="airr",
    description=(
        "Compute a pairwise CDR3 sequence-distance matrix between the unique "
        "receptor sequences of a single-cell AIRR AnnData. metric selects "
        "'identity', 'hamming', 'levenshtein' or 'alignment'. Result stored "
        "in adata.uns['ir_dist']."
    ),
    produces={"uns": ["ir_dist"]},
    examples=[
        "ov.airr.ir_dist(adata, metric='hamming', cutoff=2)",
        "ov.airr.ir_dist(adata, metric='identity')",
    ],
    related=["airr.define_clonotype_clusters", "airr.define_clonotypes"],
)
def ir_dist(
    adata,
    *,
    metric: str = "identity",
    sequence: str = "aa",
    cutoff: int = 0,
):
    """Pairwise CDR3 sequence-distance matrix over unique receptors.

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData.
    metric
        ``'identity'`` | ``'hamming'`` | ``'levenshtein'`` | ``'alignment'``.
    sequence
        ``'aa'`` (CDR3 amino-acid, default) or ``'nt'`` (nucleotide).
    cutoff
        Distances strictly greater than ``cutoff`` are treated as "no edge"
        and dropped from the sparse result.

    Returns
    -------
    AnnData
        ``adata.uns['ir_dist']`` is set to a dict with the unique sequence
        list, the (sparse) distance pairs and the metric/cutoff used.
    """
    if metric not in _METRICS:
        raise ValueError(
            f"metric must be one of {sorted(_METRICS)}, got {metric!r}"
        )
    keys = _primary_cdr3(adata, sequence)
    unique = sorted(set(keys) - {"|"})
    idx = {u: i for i, u in enumerate(unique)}
    fn = _METRICS[metric]

    pairs = []
    for i in range(len(unique)):
        for j in range(i, len(unique)):
            si, sj = unique[i], unique[j]
            d = fn(si, sj)
            if d <= cutoff:
                pairs.append((i, j, int(d)))

    adata.uns["ir_dist"] = {
        "sequences": unique,
        "pairs": pairs,
        "metric": metric,
        "sequence": sequence,
        "cutoff": cutoff,
    }
    adata.uns["_ir_dist_seq_index"] = idx
    return adata


@register_function(
    aliases=["define_clonotypes", "airr_clonotypes", "定义克隆型", "克隆型鉴定"],
    category="airr",
    description=(
        "Define exact-identity clonotypes for a single-cell AIRR AnnData: "
        "cells whose primary VJ + VDJ CDR3 sequences are identical share a "
        "clonotype id. Writes obs['clone_id'] and obs['clone_size']."
    ),
    requires={"obs": ["VJ_1_junction_aa", "VDJ_1_junction_aa"]},
    produces={"obs": ["clone_id", "clone_size"]},
    examples=[
        "ov.airr.define_clonotypes(adata)",
        "ov.airr.define_clonotypes(adata, sequence='nt')",
    ],
    related=["airr.define_clonotype_clusters", "airr.clonal_expansion"],
)
def define_clonotypes(
    adata,
    *,
    sequence: str = "aa",
    key_added: str = "clone_id",
):
    """Define exact-identity clonotypes.

    Cells with an identical primary CDR3 (VJ_1 + VDJ_1) sequence pair are
    assigned the same clonotype id; clone ids are ordered by clone size
    (``clonotype_0`` is the largest).

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData.
    sequence
        ``'aa'`` (default) or ``'nt'``.
    key_added
        ``obs`` column name for the clonotype id (default ``'clone_id'``).

    Returns
    -------
    AnnData
        ``obs[key_added]`` (clonotype id) and ``obs[key_added + '_size']``
        (clone size). Cells lacking any receptor get ``NaN``.
    """
    keys = _primary_cdr3(adata, sequence)
    has_ir = keys != "|"
    counts = keys[has_ir].value_counts()
    order = {k: f"clonotype_{i}" for i, k in enumerate(counts.index)}

    clone_id = keys.map(lambda k: order.get(k, np.nan))
    clone_id[~has_ir] = np.nan
    size = keys.map(lambda k: int(counts.get(k, 0)))
    size[~has_ir] = np.nan

    adata.obs[key_added] = pd.Categorical(clone_id)
    adata.obs[f"{key_added}_size"] = size.astype("float")
    adata.uns["clonotype"] = {
        "sequence": sequence,
        "n_clonotypes": int(len(counts)),
        "key": key_added,
    }
    return adata


def _connected_components(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Union-find connected components."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return np.array([find(i) for i in range(n)])


@register_function(
    aliases=[
        "define_clonotype_clusters", "airr_clonotype_clusters",
        "克隆型聚类", "距离克隆型",
    ],
    category="airr",
    description=(
        "Define distance-based clonotype CLUSTERS: cells whose CDR3 sequences "
        "are within a distance cutoff (hamming/levenshtein/alignment) are "
        "merged into one cluster via connected components. Writes "
        "obs['cc_clone_id'] and obs['cc_clone_size']."
    ),
    produces={"obs": ["cc_clone_id", "cc_clone_size"], "uns": ["ir_dist"]},
    examples=[
        "ov.airr.define_clonotype_clusters(adata, metric='hamming', cutoff=2)",
        "ov.airr.define_clonotype_clusters(adata, metric='levenshtein', cutoff=3)",
    ],
    related=["airr.ir_dist", "airr.define_clonotypes"],
)
def define_clonotype_clusters(
    adata,
    *,
    metric: str = "hamming",
    sequence: str = "aa",
    cutoff: int = 2,
    key_added: str = "cc_clone_id",
):
    """Define distance-based clonotype clusters.

    Receptor sequences within ``cutoff`` of each other (under ``metric``)
    are connected; connected components form clonotype *clusters* — looser
    than exact :func:`define_clonotypes`, capturing convergent / mutated
    receptors.

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData.
    metric
        ``'hamming'`` (default) | ``'levenshtein'`` | ``'alignment'`` |
        ``'identity'``.
    sequence
        ``'aa'`` (default) or ``'nt'``.
    cutoff
        Maximum CDR3 distance for two receptors to be linked.
    key_added
        ``obs`` column for the cluster id (default ``'cc_clone_id'``).

    Returns
    -------
    AnnData
        ``obs[key_added]`` and ``obs[key_added + '_size']``.
    """
    ir_dist(adata, metric=metric, sequence=sequence, cutoff=cutoff)
    info = adata.uns["ir_dist"]
    unique = info["sequences"]
    edges = [(i, j) for i, j, _ in info["pairs"]]
    comp = _connected_components(len(unique), edges)
    seq2comp = {unique[i]: comp[i] for i in range(len(unique))}

    keys = _primary_cdr3(adata, sequence)
    has_ir = keys != "|"
    raw = keys.map(lambda k: seq2comp.get(k, -1))
    # order clusters by size
    valid = raw[has_ir & (raw >= 0)]
    sizes = valid.value_counts()
    order = {c: f"ct_cluster_{i}" for i, c in enumerate(sizes.index)}

    cid = raw.map(lambda c: order.get(c, np.nan))
    cid[~has_ir] = np.nan
    csize = raw.map(lambda c: int(sizes.get(c, 0)))
    csize[~has_ir] = np.nan

    adata.obs[key_added] = pd.Categorical(cid)
    adata.obs[f"{key_added}_size"] = csize.astype("float")
    adata.uns["clonotype_clusters"] = {
        "metric": metric, "cutoff": cutoff, "sequence": sequence,
        "n_clusters": int(len(sizes)), "key": key_added,
    }
    return adata


@register_function(
    aliases=["clonal_expansion", "airr_clonal_expansion", "克隆扩增", "扩增分析"],
    category="airr",
    description=(
        "Categorise every cell by the size of the clonotype it belongs to "
        "('1 (single)', '2', '3', '>= 4' by default). Writes "
        "obs['clonal_expansion']."
    ),
    requires={"obs": ["clone_id"]},
    produces={"obs": ["clonal_expansion"]},
    examples=[
        "ov.airr.clonal_expansion(adata)",
        "ov.airr.clonal_expansion(adata, clip_at=10)",
    ],
    related=["airr.define_clonotypes", "airr.plotting.clonal_expansion_plot"],
)
def clonal_expansion(
    adata,
    *,
    target_col: str = "clone_id",
    clip_at: int = 4,
    key_added: str = "clonal_expansion",
):
    """Categorise cells by clonal-expansion level.

    Parameters
    ----------
    adata
        AnnData with a clonotype column from :func:`define_clonotypes`.
    target_col
        Clonotype id column (default ``'clone_id'``).
    clip_at
        Clone sizes ``>= clip_at`` are pooled into one ``'>= N'`` bucket.
    key_added
        ``obs`` column for the category (default ``'clonal_expansion'``).

    Returns
    -------
    AnnData
        ``obs[key_added]`` — an ordered categorical:
        ``'1 (single)'``, ``'2'``, …, ``'>= clip_at'``.
    """
    if target_col not in adata.obs:
        raise KeyError(
            f"obs[{target_col!r}] not found — run define_clonotypes first."
        )
    sizes = adata.obs[target_col].map(
        adata.obs[target_col].value_counts()
    )

    def _bucket(s):
        if s != s:
            return np.nan
        s = int(s)
        if s == 1:
            return "1 (single)"
        if s >= clip_at:
            return f">= {clip_at}"
        return str(s)

    cats = ["1 (single)"] + [str(i) for i in range(2, clip_at)] + [f">= {clip_at}"]
    vals = sizes.map(_bucket)
    adata.obs[key_added] = pd.Categorical(
        vals, categories=cats, ordered=True
    )
    return adata


@register_function(
    aliases=["clonotype_network", "airr_clonotype_network", "克隆型网络", "克隆网络图"],
    category="airr",
    description=(
        "Build a 2-D layout of the clonotype graph for a single-cell AIRR "
        "AnnData: nodes are clonotype clusters, edges connect cells within a "
        "CDR3 distance cutoff. Writes the layout to obsm['X_clonotype_network']."
    ),
    produces={"obsm": ["X_clonotype_network"], "uns": ["clonotype_network"]},
    examples=[
        "ov.airr.clonotype_network(adata, min_cells=2)",
        "ov.airr.clonotype_network(adata, metric='hamming', cutoff=2)",
    ],
    related=["airr.define_clonotype_clusters", "airr.plotting.clonotype_network_plot"],
)
def clonotype_network(
    adata,
    *,
    metric: str = "identity",
    sequence: str = "aa",
    cutoff: int = 0,
    min_cells: int = 1,
    layout_seed: int = 0,
):
    """Compute a 2-D layout of the clonotype network.

    Cells are nodes; an edge links two cells whose receptor sequences are
    within ``cutoff`` distance. A simple force-directed (spring) layout is
    computed per connected component and packed onto a grid.

    Parameters
    ----------
    adata
        Single-cell AIRR AnnData.
    metric, sequence, cutoff
        Passed to :func:`ir_dist` to build the edge set.
    min_cells
        Drop clonotypes smaller than this from the layout (their coordinates
        are ``NaN``).
    layout_seed
        Random seed for the spring layout.

    Returns
    -------
    AnnData
        ``obsm['X_clonotype_network']`` — an ``(n_obs, 2)`` coordinate array
        (``NaN`` for excluded cells).
    """
    keys = _primary_cdr3(adata, sequence)
    has_ir = (keys != "|").values
    # clone sizes
    sizes = keys.map(keys[keys != "|"].value_counts()).fillna(0).values
    keep = has_ir & (sizes >= min_cells)

    # edges between kept cells with identical / near keys
    ir_dist(adata, metric=metric, sequence=sequence, cutoff=cutoff)
    info = adata.uns["ir_dist"]
    seq_pairs = {
        (info["sequences"][i], info["sequences"][j])
        for i, j, _ in info["pairs"]
    }
    key_arr = keys.values
    kept_idx = np.where(keep)[0]
    # group kept cells by key for component merging
    edges: list[tuple[int, int]] = []
    by_key: dict[str, list[int]] = defaultdict(list)
    for i in kept_idx:
        by_key[key_arr[i]].append(i)
    # intra-key edges (identical receptors)
    for members in by_key.values():
        for a in range(1, len(members)):
            edges.append((members[a - 1], members[a]))
    # inter-key edges from distance pairs
    for ka, kb in seq_pairs:
        if ka == kb:
            continue
        for a in by_key.get(ka, []):
            for b in by_key.get(kb, []):
                edges.append((a, b))

    comp = _connected_components(adata.n_obs, edges)
    rng = np.random.default_rng(layout_seed)
    coords = np.full((adata.n_obs, 2), np.nan)

    # one packed circular layout per component
    comp_of_kept = {c for c in comp[kept_idx]}
    grid = int(np.ceil(np.sqrt(max(len(comp_of_kept), 1))))
    for ci, c in enumerate(sorted(comp_of_kept)):
        members = np.where((comp == c) & keep)[0]
        gx, gy = ci % grid, ci // grid
        m = len(members)
        if m == 1:
            coords[members[0]] = [gx * 3.0, gy * 3.0]
        else:
            ang = np.linspace(0, 2 * np.pi, m, endpoint=False)
            r = 0.8 + 0.05 * m
            jitter = rng.normal(0, 0.05, size=(m, 2))
            coords[members, 0] = gx * 3.0 + r * np.cos(ang) + jitter[:, 0]
            coords[members, 1] = gy * 3.0 + r * np.sin(ang) + jitter[:, 1]

    adata.obsm["X_clonotype_network"] = coords
    adata.uns["clonotype_network"] = {
        "metric": metric, "cutoff": cutoff, "min_cells": min_cells,
        "n_components": int(len(comp_of_kept)),
    }
    return adata


@register_function(
    aliases=["clonotype_imbalance", "airr_clonotype_imbalance", "克隆型失衡", "克隆型富集"],
    category="airr",
    description=(
        "Find clonotypes whose abundance is imbalanced between two cell "
        "groups: a per-clonotype Fisher exact test on a group contingency "
        "table, with fold-change and BH-corrected p-values."
    ),
    requires={"obs": ["clone_id"]},
    examples=[
        "df = ov.airr.clonotype_imbalance(adata, groupby='group')",
    ],
    related=["airr.define_clonotypes", "airr.group_abundance"],
)
def clonotype_imbalance(
    adata,
    groupby: str,
    *,
    target_col: str = "clone_id",
    case=None,
    control=None,
):
    """Test clonotypes for abundance imbalance between two groups.

    Parameters
    ----------
    adata
        AnnData with a clonotype column.
    groupby
        ``obs`` column with two (or more) groups.
    target_col
        Clonotype id column (default ``'clone_id'``).
    case, control
        The two group labels to compare. If ``None`` the first two unique
        values of ``obs[groupby]`` are used.

    Returns
    -------
    :class:`pandas.DataFrame`
        Per-clonotype ``n_case``, ``n_control``, ``log2_fold_change``,
        ``pvalue``, ``pvalue_adj`` — sorted by ``pvalue``.
    """
    from scipy import stats

    sub = adata.obs.dropna(subset=[target_col])
    groups = list(pd.unique(sub[groupby]))
    if case is None or control is None:
        if len(groups) < 2:
            raise ValueError(f"obs[{groupby!r}] needs at least two groups.")
        case, control = groups[0], groups[1]
    n_case = int((sub[groupby] == case).sum())
    n_control = int((sub[groupby] == control).sum())

    rows = []
    for clone, cd in sub.groupby(target_col, observed=True):
        a = int((cd[groupby] == case).sum())
        b = int((cd[groupby] == control).sum())
        if a == 0 and b == 0:
            continue
        table = [[a, n_case - a], [b, n_control - b]]
        _, p = stats.fisher_exact(table)
        fa = (a + 1) / (n_case + 1)
        fb = (b + 1) / (n_control + 1)
        rows.append({
            "clone_id": clone, "n_case": a, "n_control": b,
            "log2_fold_change": float(np.log2(fa / fb)), "pvalue": p,
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res = res.sort_values("pvalue").reset_index(drop=True)
    # BH correction
    m = len(res)
    ranks = np.arange(1, m + 1)
    adj = (res["pvalue"].values * m / ranks)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res["pvalue_adj"] = np.clip(adj, 0, 1)
    return res
