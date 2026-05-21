"""Joint single-cell TCR + gene-expression analysis — a CoNGA-style layer.

A clean, AnnData-native reimplementation of the core of **CoNGA**
(Schattgen *et al.*, *Nat Biotechnol* 2022; github ``phbradley/conga``):
"Clonotype Neighbor Graph Analysis" couples the gene-expression (GEX)
similarity graph with the TCR-sequence similarity graph and looks for
cells / clonotypes where the two graphs *agree* (or for genes that are
localized on either graph).

Functions operate on an :class:`~anndata.AnnData` that carries **both**

* a gene-expression embedding — ``obsm['X_pca']`` (and optionally
  ``obsm['X_umap']``) plus a GEX cluster column in ``obs`` — and
* per-cell TCR data in the ``ov.airr`` obs schema (``VJ_1_*`` / ``VDJ_1_*``
  chain slots) **plus** a clonotype / ``clone_id`` column
  (from :func:`omicverse.airr.define_clonotypes`).

Pipeline
--------
* :func:`conga_score`     — graph-vs-graph CoNGA score: hypergeometric
  significance of the overlap between a cell's GEX-graph and TCR-graph
  neighborhoods. Per-cell scores written to ``obs``.
* :func:`conga_clusters`  — group "CoNGA hits" (low-score clonotypes) by
  their shared (GEX-cluster x TCR-cluster) identity.
* :func:`tcr_clumping`    — detect TCR clonotype clusters that are more
  concentrated than expected under a V/J-matched permutation background.
* :func:`hotspot_features`— graph-vs-features: genes / TCR biochemical
  features whose values are spatially autocorrelated on the GEX (or TCR)
  graph (the HotSpot local-autocorrelation statistic).
* :func:`conga_score_plot`   — CoNGA score on a UMAP embedding.
* :func:`tcr_clumping_plot`  — bar / logo summary of TCR clumping hits.
* :func:`hotspot_features_plot` — bar plot of top HotSpot features.

TCR distances are taken from a sibling :mod:`omicverse.airr._tcr` module
(``tcrdist``) when available; otherwise a shared-clonotype / Hamming-CDR3
fallback graph is used, so this module is self-contained.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .._registry import register_function


def _require(modname: str, role: str):
    """Lazy-import a backend with an actionable error message."""
    import importlib

    try:
        return importlib.import_module(modname)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"{role} needs the '{modname}' backend. Install with: "
            f"pip install omicverse[airr]   (or pip install {modname})."
        ) from exc


# ---------------------------------------------------------------------------
# TCR primitives — CDR3 keys, distances, V/J usage
# ---------------------------------------------------------------------------
def _cdr3_key(adata, sequence: str = "aa") -> pd.Series:
    """Concatenated VJ_1 + VDJ_1 CDR3 string per cell (the TCR identity)."""
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


def _vj_genes(adata) -> pd.DataFrame:
    """Per-cell V/J gene table for VJ_1 and VDJ_1 chains (string-coerced)."""
    def _col(name):
        s = adata.obs.get(name)
        if s is None:
            return pd.Series(["None"] * adata.n_obs, index=adata.obs_names)
        return s.astype(str).replace({"nan": "None", "": "None"})

    return pd.DataFrame({
        "vj_v": _col("VJ_1_v_gene"),
        "vj_j": _col("VJ_1_j_gene"),
        "vdj_v": _col("VDJ_1_v_gene"),
        "vdj_j": _col("VDJ_1_j_gene"),
    }, index=adata.obs_names)


def _hamming(a: str, b: str) -> int:
    """Hamming distance — large value when lengths differ."""
    if len(a) != len(b):
        return 10 ** 6
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def _cdr3_distance_matrix(cdr3: list[str]) -> np.ndarray:
    """Fallback CDR3 distance — Hamming, with a length-gap penalty.

    Used when :mod:`omicverse.airr._tcr` (``tcrdist``) is unavailable. The
    two CDR3 halves (VJ + VDJ) are compared independently and summed.
    """
    n = len(cdr3)
    halves = []
    for k in cdr3:
        a, _, d = k.partition("|")
        halves.append((a, d))
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        ai, di = halves[i]
        for j in range(i + 1, n):
            aj, dj = halves[j]
            da = _hamming(ai, aj) if (ai and aj) else (3 if ai != aj else 0)
            dd = _hamming(di, dj) if (di and dj) else (3 if di != dj else 0)
            # cap exploding length-gap penalties
            d_ij = float(min(da, 12) + min(dd, 12))
            D[i, j] = D[j, i] = d_ij
    return D


def _tcr_distance_matrix(adata, sequence: str = "aa"):
    """Pairwise TCR distance matrix over cells.

    Tries the sibling :mod:`omicverse.airr._tcr` ``tcrdist`` implementation;
    falls back to a CDR3 Hamming distance when that module is not present.

    Returns
    -------
    tuple
        ``(D, backend)`` — an ``(n_obs, n_obs)`` distance array and the name
        of the backend used (``'tcrdist'`` or ``'cdr3_hamming'``).
    """
    cdr3 = list(_cdr3_key(adata, sequence).values)
    n = adata.n_obs
    try:  # pragma: no cover - depends on a sibling module built in parallel
        from ._tcr import tcrdist as _tcrdist  # type: ignore

        # ov.airr.tcrdist returns a (D, df) tuple — unpack the matrix.
        # It also drops cells with no CDR3 ('both' keeps any-chain cells),
        # so scatter the result back onto the full (n_obs, n_obs) grid.
        out = _tcrdist(adata, chain="both")
        D = out[0] if isinstance(out, tuple) else out
        D = np.asarray(getattr(D, "values", D), dtype=float)
        if D.ndim == 2 and D.shape[0] == D.shape[1]:
            if D.shape == (n, n):
                return D, "tcrdist"
            # rows kept by tcrdist == cells carrying at least one CDR3
            kept = np.array([k != "|" for k in cdr3])
            if int(kept.sum()) == D.shape[0]:
                far = float(D.max() * 2.0) if D.size else 1e6
                full = np.full((n, n), far, dtype=float)
                idx = np.where(kept)[0]
                full[np.ix_(idx, idx)] = D
                np.fill_diagonal(full, 0.0)
                return full, "tcrdist"
    except Exception:
        pass
    return _cdr3_distance_matrix(cdr3), "cdr3_hamming"


# ---------------------------------------------------------------------------
# kNN graphs
# ---------------------------------------------------------------------------
def _knn_from_embedding(X: np.ndarray, n_neighbors: int) -> list[set[int]]:
    """kNN neighbor sets from a (dense) embedding via Euclidean distance."""
    n = X.shape[0]
    k = min(n_neighbors, n - 1)
    # squared Euclidean distances
    sq = np.sum(X * X, axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(D, np.inf)
    nbrs = []
    for i in range(n):
        idx = np.argpartition(D[i], k)[:k]
        nbrs.append(set(int(j) for j in idx))
    return nbrs


def _knn_from_distance(D: np.ndarray, n_neighbors: int) -> list[set[int]]:
    """kNN neighbor sets from a precomputed distance matrix."""
    n = D.shape[0]
    k = min(n_neighbors, n - 1)
    Dw = D.astype(float).copy()
    np.fill_diagonal(Dw, np.inf)
    nbrs = []
    for i in range(n):
        idx = np.argpartition(Dw[i], k)[:k]
        nbrs.append(set(int(j) for j in idx))
    return nbrs


def _tcr_knn(adata, n_neighbors: int, sequence: str):
    """TCR-graph neighbor sets — shared clonotype first, then TCR distance.

    Cells of the same exact clonotype are always neighbors; the rest of the
    neighborhood is filled by the nearest TCR-distance cells.
    """
    D, backend = _tcr_distance_matrix(adata, sequence)
    keys = _cdr3_key(adata, sequence).values
    nbrs = _knn_from_distance(D, n_neighbors)
    # guarantee same-clonotype cells share an edge
    by_key: dict[str, list[int]] = {}
    for i, kk in enumerate(keys):
        if kk != "|":
            by_key.setdefault(kk, []).append(i)
    for members in by_key.values():
        if len(members) < 2:
            continue
        ms = set(members)
        for i in members:
            nbrs[i] |= (ms - {i})
    return nbrs, backend


# ---------------------------------------------------------------------------
# Hypergeometric overlap p-value
# ---------------------------------------------------------------------------
def _hypergeom_overlap_pvalue(overlap: int, k_gex: int, k_tcr: int,
                              n_total: int) -> float:
    """Upper-tail hypergeometric p-value of a neighborhood overlap.

    Probability of observing ``>= overlap`` shared neighbors when drawing
    ``k_tcr`` items (TCR neighbors) from a population of ``n_total`` of which
    ``k_gex`` are "successes" (GEX neighbors).
    """
    from scipy import stats

    if overlap <= 0 or k_gex <= 0 or k_tcr <= 0:
        return 1.0
    # P(X >= overlap) = sf(overlap - 1)
    p = stats.hypergeom.sf(overlap - 1, n_total, k_gex, k_tcr)
    return float(min(max(p, 0.0), 1.0))


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment (NaN-safe)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv)
    ranks = np.arange(1, m + 1)
    adj = pv[order] * m / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.clip(adj, 0, 1)
    out[valid] = res
    return out


# ---------------------------------------------------------------------------
# 1. graph-vs-graph CoNGA score
# ---------------------------------------------------------------------------
@register_function(
    aliases=["conga_score", "tcr_gex_conga_score", "CoNGA评分", "图图CoNGA评分"],
    category="airr",
    description=(
        "Graph-vs-graph CoNGA score for joint single-cell TCR + GEX data. "
        "Builds a kNN graph from the gene-expression embedding and a second "
        "graph from TCR similarity (TCRdist or shared clonotype), then for "
        "every cell scores the hypergeometric significance of the overlap "
        "between its GEX-graph and TCR-graph neighborhoods. Writes per-cell "
        "obs['conga_pvalue'], obs['conga_score'] and obs['conga_overlap']."
    ),
    requires={"obsm": ["X_pca"], "obs": ["clone_id"]},
    produces={"obs": ["conga_pvalue", "conga_score", "conga_overlap"]},
    examples=[
        "ov.airr.conga_score(adata, gex_rep='X_pca', n_neighbors=10)",
        "ov.airr.conga_score(adata, gex_rep='X_umap', sequence='aa')",
    ],
    related=["airr.conga_clusters", "airr.hotspot_features"],
)
def conga_score(
    adata,
    *,
    gex_rep: str = "X_pca",
    n_neighbors: int = 10,
    sequence: str = "aa",
    key_added: str = "conga",
):
    """Graph-vs-graph CoNGA score.

    For every cell two neighborhoods are formed: the ``n_neighbors`` nearest
    cells in the gene-expression embedding (``gex_rep``) and the
    ``n_neighbors`` nearest cells in TCR space. A CoNGA hit is a cell whose
    two neighborhoods overlap far more than expected by chance — the overlap
    is scored with an upper-tail hypergeometric test. Low ``conga_pvalue``
    (high ``conga_score = -log10 p``) flags cells where transcriptome and
    TCR co-vary.

    Parameters
    ----------
    adata
        AnnData with a GEX embedding in ``obsm[gex_rep]`` and per-cell TCR
        data plus a ``clone_id`` column.
    gex_rep
        ``obsm`` key of the gene-expression embedding (default ``'X_pca'``).
    n_neighbors
        Neighborhood size ``k`` for both graphs.
    sequence
        ``'aa'`` (CDR3 amino-acid, default) or ``'nt'`` for the TCR graph.
    key_added
        Prefix for the written ``obs`` columns (default ``'conga'``).

    Returns
    -------
    AnnData
        ``obs[key_added + '_pvalue']`` (raw hypergeometric p-value),
        ``obs[key_added + '_pvalue_adj']`` (BH-adjusted),
        ``obs[key_added + '_score']`` (``-log10`` adjusted p-value) and
        ``obs[key_added + '_overlap']`` (neighbor-set overlap count). The run
        summary is stored in ``uns[key_added]``.
    """
    if gex_rep not in adata.obsm:
        raise KeyError(
            f"obsm[{gex_rep!r}] not found — compute a GEX embedding "
            "(e.g. ov.pp.pca) first."
        )
    X = np.asarray(adata.obsm[gex_rep], dtype=float)
    if X.ndim != 2 or X.shape[0] != adata.n_obs:
        raise ValueError(f"obsm[{gex_rep!r}] must be an (n_obs, n_dim) array.")

    n = adata.n_obs
    gex_nbrs = _knn_from_embedding(X, n_neighbors)
    tcr_nbrs, backend = _tcr_knn(adata, n_neighbors, sequence)

    pvals = np.full(n, np.nan)
    overlaps = np.zeros(n, dtype=int)
    for i in range(n):
        g, t = gex_nbrs[i], tcr_nbrs[i]
        ov = len(g & t)
        overlaps[i] = ov
        pvals[i] = _hypergeom_overlap_pvalue(ov, len(g), len(t), n - 1)

    padj = _bh_adjust(pvals)
    score = -np.log10(np.clip(padj, 1e-300, 1.0))

    adata.obs[f"{key_added}_pvalue"] = pvals
    adata.obs[f"{key_added}_pvalue_adj"] = padj
    adata.obs[f"{key_added}_score"] = score
    adata.obs[f"{key_added}_overlap"] = overlaps.astype(float)
    adata.uns[key_added] = {
        "gex_rep": gex_rep,
        "n_neighbors": int(n_neighbors),
        "sequence": sequence,
        "tcr_backend": backend,
        "n_hits": int(np.sum(padj < 0.05)),
    }
    return adata


# ---------------------------------------------------------------------------
# 2. CoNGA clusters
# ---------------------------------------------------------------------------
@register_function(
    aliases=["conga_clusters", "tcr_gex_conga_clusters", "CoNGA簇", "CoNGA聚类"],
    category="airr",
    description=(
        "Group CoNGA hits into clusters by shared (GEX-cluster x TCR-cluster) "
        "identity. After conga_score, cells passing a CoNGA p-value cutoff "
        "are partitioned by the combination of their gene-expression cluster "
        "and their TCR cluster. Writes obs['conga_cluster']."
    ),
    requires={"obs": ["conga_pvalue"]},
    produces={"obs": ["conga_cluster"]},
    examples=[
        "ov.airr.conga_clusters(adata, gex_cluster='leiden', max_pvalue=0.05)",
        "ov.airr.conga_clusters(adata, tcr_cluster='cc_clone_id')",
    ],
    related=["airr.conga_score", "airr.tcr_clumping"],
)
def conga_clusters(
    adata,
    *,
    gex_cluster: str = "leiden",
    tcr_cluster: Optional[str] = None,
    score_key: str = "conga",
    max_pvalue: float = 0.05,
    min_cluster_size: int = 3,
    key_added: str = "conga_cluster",
):
    """Group low-CoNGA-score clonotypes by (GEX x TCR) identity.

    A CoNGA cluster collects cells that (i) pass the CoNGA significance
    cutoff and (ii) share the *same* gene-expression cluster **and** the same
    TCR cluster — i.e. a transcriptionally coherent group of related TCRs.

    Parameters
    ----------
    adata
        AnnData after :func:`conga_score`.
    gex_cluster
        ``obs`` column with gene-expression cluster labels (default
        ``'leiden'``).
    tcr_cluster
        ``obs`` column with a TCR clonotype / clonotype-cluster label. When
        ``None`` it is auto-detected from ``'cc_clone_id'`` then
        ``'clone_id'``.
    score_key
        Prefix used by :func:`conga_score` (default ``'conga'``).
    max_pvalue
        CoNGA adjusted-p-value cutoff for a cell to be a hit.
    min_cluster_size
        Drop CoNGA clusters smaller than this.
    key_added
        ``obs`` column for the CoNGA-cluster label (default
        ``'conga_cluster'``).

    Returns
    -------
    AnnData
        ``obs[key_added]`` — a categorical CoNGA-cluster id (``NaN`` for
        non-hits / small clusters); summary in ``uns[key_added]``.
    """
    padj_col = f"{score_key}_pvalue_adj"
    pcol = padj_col if padj_col in adata.obs else f"{score_key}_pvalue"
    if pcol not in adata.obs:
        raise KeyError(
            f"obs[{pcol!r}] not found — run ov.airr.conga_score first."
        )
    if gex_cluster not in adata.obs:
        raise KeyError(f"obs[{gex_cluster!r}] (GEX clusters) not found.")
    if tcr_cluster is None:
        for cand in ("cc_clone_id", "clone_id"):
            if cand in adata.obs:
                tcr_cluster = cand
                break
    if tcr_cluster is None or tcr_cluster not in adata.obs:
        raise KeyError(
            "No TCR cluster column found — pass tcr_cluster= or run "
            "ov.airr.define_clonotypes / define_clonotype_clusters."
        )

    obs = adata.obs
    pvals = pd.to_numeric(obs[pcol], errors="coerce")
    hit = (pvals < max_pvalue).fillna(False).values

    gex = obs[gex_cluster].astype(str).values
    tcr = obs[tcr_cluster].astype(str).values
    combo = np.array([
        f"{g}__{t}" if hit[i] and t not in ("nan", "None") else None
        for i, (g, t) in enumerate(zip(gex, tcr))
    ], dtype=object)

    counts = pd.Series([c for c in combo if c is not None]).value_counts()
    keep = {c for c, n in counts.items() if n >= min_cluster_size}
    order = {c: f"conga_{i}" for i, c in enumerate(
        [c for c in counts.index if c in keep]
    )}
    labels = np.array([
        order.get(c) if (c is not None and c in keep) else None
        for c in combo
    ], dtype=object)

    adata.obs[key_added] = pd.Categorical(labels)
    members = {}
    for cid in order.values():
        sel = labels == cid
        gx = pd.Series(gex[sel]).mode()
        tx = pd.Series(tcr[sel]).mode()
        members[cid] = {
            "n_cells": int(sel.sum()),
            "gex_cluster": (gx.iloc[0] if len(gx) else None),
            "tcr_cluster": (tx.iloc[0] if len(tx) else None),
        }
    adata.uns[key_added] = {
        "gex_cluster": gex_cluster,
        "tcr_cluster": tcr_cluster,
        "max_pvalue": float(max_pvalue),
        "min_cluster_size": int(min_cluster_size),
        "n_clusters": int(len(order)),
        "clusters": members,
    }
    return adata


# ---------------------------------------------------------------------------
# 2b. CoNGA cluster / score summaries
# ---------------------------------------------------------------------------
@register_function(
    aliases=["conga_cluster_table", "tcr_gex_conga_cluster_table",
             "CoNGA簇表", "CoNGA聚类表"],
    category="airr",
    description=(
        "Flatten the nested uns['conga_cluster'] dict written by "
        "ov.airr.conga_clusters into a tidy per-CoNGA-cluster DataFrame — "
        "one row per cluster with its cell count and its (GEX-cluster x "
        "TCR-cluster) identity, sorted by size."
    ),
    requires={"uns": ["conga_cluster"]},
    examples=[
        "ov.airr.conga_clusters(adata, gex_cluster='leiden')",
        "tab = ov.airr.conga_cluster_table(adata)",
    ],
    related=["airr.conga_clusters", "airr.conga_score"],
)
def conga_cluster_table(adata) -> pd.DataFrame:
    """Tidy per-CoNGA-cluster summary table.

    Unpacks the nested ``adata.uns['conga_cluster']['clusters']`` mapping
    produced by :func:`conga_clusters` into a flat DataFrame — one row per
    CoNGA cluster — for printing / plotting.

    Parameters
    ----------
    adata
        AnnData after :func:`conga_clusters`.

    Returns
    -------
    :class:`pandas.DataFrame`
        Columns ``conga_cluster``, ``n_cells``, ``gex_cluster`` and
        ``tcr_cluster``, sorted by ``n_cells`` descending.
    """
    if "conga_cluster" not in adata.uns:
        raise KeyError(
            "uns['conga_cluster'] not found — run ov.airr.conga_clusters first."
        )
    clusters = adata.uns["conga_cluster"].get("clusters", {})
    rows = []
    for cid, meta in clusters.items():
        rows.append({
            "conga_cluster": cid,
            "n_cells": int(meta.get("n_cells", 0)),
            "gex_cluster": meta.get("gex_cluster"),
            "tcr_cluster": meta.get("tcr_cluster"),
        })
    return (
        pd.DataFrame(
            rows,
            columns=["conga_cluster", "n_cells", "gex_cluster", "tcr_cluster"],
        )
        .sort_values("n_cells", ascending=False)
        .reset_index(drop=True)
    )


@register_function(
    aliases=["conga_score_summary", "tcr_gex_conga_score_summary",
             "CoNGA评分汇总", "CoNGA评分摘要"],
    category="airr",
    description=(
        "Per-group summary of the CoNGA score after ov.airr.conga_score — "
        "for every level of a grouping obs column, the cell count, the mean "
        "CoNGA score and the fraction of cells that are CoNGA hits "
        "(conga_pvalue_adj < 0.05). Sorted by hit fraction."
    ),
    requires={"obs": ["conga_score"]},
    examples=[
        "summ = ov.airr.conga_score_summary(adata, groupby='antigen_species')",
        "summ = ov.airr.conga_score_summary(adata, groupby='leiden')",
    ],
    related=["airr.conga_score", "airr.conga_clusters"],
)
def conga_score_summary(
    adata,
    *,
    groupby: str = "antigen_species",
    score_key: str = "conga",
) -> pd.DataFrame:
    """Per-group CoNGA-score summary.

    Aggregates the per-cell CoNGA score by a grouping ``obs`` column,
    reporting how strongly transcriptome and TCR co-vary within each group.

    Parameters
    ----------
    adata
        AnnData after :func:`conga_score`.
    groupby
        ``obs`` column to group cells by (default ``'antigen_species'``).
    score_key
        Prefix used by :func:`conga_score` (default ``'conga'``).

    Returns
    -------
    :class:`pandas.DataFrame`
        Indexed by group, with columns ``n_cells``, ``mean_conga_score`` and
        ``hit_fraction`` (fraction of cells with adjusted CoNGA p-value
        ``< 0.05``), sorted by ``hit_fraction`` descending.
    """
    score_col = f"{score_key}_score"
    if score_col not in adata.obs:
        raise KeyError(
            f"obs[{score_col!r}] not found — run ov.airr.conga_score first."
        )
    if groupby not in adata.obs:
        raise KeyError(f"obs[{groupby!r}] not found.")
    padj_col = f"{score_key}_pvalue_adj"
    pcol = padj_col if padj_col in adata.obs else f"{score_key}_pvalue"

    df = pd.DataFrame({
        "group": adata.obs[groupby].astype(object).values,
        "score": pd.to_numeric(adata.obs[score_col], errors="coerce").values,
        "hit": (
            pd.to_numeric(adata.obs[pcol], errors="coerce") < 0.05
        ).fillna(False).values,
    })
    summary = df.groupby("group", observed=True).agg(
        n_cells=("score", "size"),
        mean_conga_score=("score", "mean"),
        hit_fraction=("hit", "mean"),
    ).round(3)
    return summary.sort_values("hit_fraction", ascending=False)


@register_function(
    aliases=["expression_by_group", "tcr_gex_expression_by_group",
             "分组表达均值", "基因分组表达"],
    category="airr",
    description=(
        "Sparse-safe genes x groups mean-expression table — for a list of "
        "genes and a grouping obs column, returns the mean expression of "
        "each gene within each group. Works on a sparse adata.X or a dense "
        "layer without materialising the full matrix."
    ),
    examples=[
        "tab = ov.airr.expression_by_group(adata, genes=['GZMB', 'IL7R'], "
        "groupby='conga_hit')",
        "tab = ov.airr.expression_by_group(adata, genes=markers, "
        "groupby='leiden', layer='counts')",
    ],
    related=["airr.conga_score", "airr.hotspot_features"],
)
def expression_by_group(
    adata,
    *,
    genes,
    groupby: str,
    layer: Optional[str] = None,
) -> pd.DataFrame:
    """Genes x groups mean-expression table (sparse-safe).

    For each gene in ``genes`` and each level of ``groupby``, computes the
    mean expression — the standard input for a marker-gene bar plot or
    heatmap. Genes not present in ``adata.var_names`` are silently dropped.

    Parameters
    ----------
    adata
        AnnData with the expression matrix (or ``layer``).
    genes
        Iterable of gene names; only those present in ``adata.var_names`` are
        used, in their input order.
    groupby
        ``obs`` column used to group cells.
    layer
        Optional ``adata.layers`` key to read expression from; the default
        (``None``) uses ``adata.X``.

    Returns
    -------
    :class:`pandas.DataFrame`
        Genes on the rows, groups on the columns, holding the mean
        expression of each gene within each group.
    """
    if groupby not in adata.obs:
        raise KeyError(f"obs[{groupby!r}] not found.")
    present = [g for g in genes if g in adata.var_names]
    if not present:
        raise ValueError("None of the requested genes are in adata.var_names.")
    sub = adata[:, present]
    mat = sub.layers[layer] if layer is not None else sub.X
    dense = (
        np.asarray(mat.todense()) if hasattr(mat, "todense")
        else np.asarray(mat)
    )
    expr = pd.DataFrame(
        dense.astype(float), columns=present, index=adata.obs_names
    )
    expr["__group__"] = adata.obs[groupby].astype(object).values
    return expr.groupby("__group__", observed=True).mean().T


# ---------------------------------------------------------------------------
# 3. TCR clumping
# ---------------------------------------------------------------------------
@register_function(
    aliases=["tcr_clumping", "tcr_gex_clumping", "TCR聚集", "TCR聚集检测"],
    category="airr",
    description=(
        "Detect TCR clonotype clusters (clumps) more concentrated in TCR "
        "space than expected by chance. Counts each clonotype's TCR neighbors "
        "within a distance radius and compares it to a background model "
        "(background='recombination' — a random-recombination CDR3 shuffle; "
        "or background='vj_permutation' — a V/J-matched permutation), giving "
        "a per-clonotype clumping p-value. Writes obs['tcr_clump_pvalue'] and "
        "obs['tcr_clump_id']."
    ),
    requires={"obs": ["clone_id"]},
    produces={"obs": ["tcr_clump_pvalue", "tcr_clump_id"]},
    examples=[
        "df = ov.airr.tcr_clumping(adata, radius=24, n_permutations=100)",
        "df = ov.airr.tcr_clumping(adata, background='vj_permutation', radius=12)",
    ],
    related=["airr.conga_score", "airr.conga_clusters"],
)
def tcr_clumping(
    adata,
    *,
    radius: float = 24.0,
    sequence: str = "aa",
    background: str = "recombination",
    n_permutations: int = 100,
    max_pvalue: float = 0.05,
    min_clump_size: int = 3,
    seed: int = 0,
    key_added: str = "tcr_clump",
):
    """Detect TCR clumping vs a recombination / permutation background.

    For each clonotype the observed number of *other* clonotypes within a TCR
    distance ``radius`` is counted, then compared to a background model:

    * ``background='recombination'`` (default) — a random-recombination null:
      CDR3s are rebuilt by independently resampling, position by position,
      from the observed residue pool (per CDR3 length), so the null reflects
      what random V(D)J recombination would produce. Real antigen-driven
      clumps stand out against it even within a single V/J stratum.
    * ``background='vj_permutation'`` — clonotype identities are permuted
      *within* V/J-gene strata, preserving V/J usage while breaking CDR3
      co-localisation.

    Clonotypes with significantly more close neighbors than the background
    are "clumping"; connected clumping clonotypes are merged into clumps.

    Parameters
    ----------
    adata
        AnnData with per-cell TCR data and a ``clone_id`` column.
    radius
        TCR-distance radius defining a "close" pair. Scale matches the TCR
        backend (TCRdist units, or CDR3 Hamming sum for the fallback).
    sequence
        ``'aa'`` (default) or ``'nt'``.
    background
        ``'recombination'`` (default) or ``'vj_permutation'`` — the null
        model (see above).
    n_permutations
        Number of background draws.
    max_pvalue
        P-value cutoff for a clonotype to be called clumping.
    min_clump_size
        Drop clumps with fewer than this many clonotypes.
    seed
        Random seed for the background draws.
    key_added
        Prefix for the written ``obs`` columns (default ``'tcr_clump'``).

    Returns
    -------
    :class:`pandas.DataFrame`
        Per-clonotype clumping table — ``clone_id``, ``n_close``,
        ``expected``, ``pvalue``, ``pvalue_adj``, ``clump_id`` — sorted by
        ``pvalue``. ``obs[key_added + '_pvalue']`` and
        ``obs[key_added + '_id']`` are also written back per cell.
    """
    if "clone_id" not in adata.obs:
        raise KeyError(
            "obs['clone_id'] not found — run ov.airr.define_clonotypes first."
        )
    if background not in ("recombination", "vj_permutation"):
        raise ValueError(
            "background must be 'recombination' or 'vj_permutation'."
        )
    rng = np.random.default_rng(seed)

    # collapse to one representative cell per clonotype
    clone = adata.obs["clone_id"].astype(str)
    valid = clone[~clone.isin(["nan", "None"])]
    clones = list(pd.unique(valid))
    if len(clones) < 3:
        raise ValueError("Need at least 3 clonotypes for TCR clumping.")
    rep = {c: valid.index[valid.values == c][0] for c in clones}
    rep_pos = {c: adata.obs_names.get_loc(rep[c]) for c in clones}
    idx = np.array([rep_pos[c] for c in clones])
    nc = len(clones)

    D_full, backend = _tcr_distance_matrix(adata, sequence)
    D = D_full[np.ix_(idx, idx)]
    np.fill_diagonal(D, np.inf)
    close = D <= radius
    n_close = close.sum(axis=1).astype(int)

    perm_counts = np.zeros((n_permutations, nc), dtype=float)
    if background == "recombination":
        # random-recombination null — rebuild CDR3s by per-length, per-position
        # residue resampling, then recompute the close-neighbor counts.
        keys = _cdr3_key(adata, sequence)
        clone_cdr3 = [keys.iloc[rep_pos[c]] for c in clones]
        halves = [k.partition("|") for k in clone_cdr3]
        vj_seqs = [h[0] for h in halves]
        vdj_seqs = [h[2] for h in halves]

        def _pos_pool(seqs):
            """{length: {pos: [residues observed at that pos]}}."""
            pool: dict[int, dict[int, list]] = {}
            for s in seqs:
                if not s:
                    continue
                pool.setdefault(len(s), {})
                for p, ch in enumerate(s):
                    pool[len(s)].setdefault(p, []).append(ch)
            return pool

        vj_pool, vdj_pool = _pos_pool(vj_seqs), _pos_pool(vdj_seqs)

        def _resample(seqs, pool):
            out = []
            for s in seqs:
                if not s:
                    out.append("")
                    continue
                pp = pool.get(len(s), {})
                out.append("".join(
                    rng.choice(pp[p]) if pp.get(p) else c
                    for p, c in enumerate(s)
                ))
            return out

        for p in range(n_permutations):
            rvj = _resample(vj_seqs, vj_pool)
            rvdj = _resample(vdj_seqs, vdj_pool)
            keys_p = [f"{a}|{d}" for a, d in zip(rvj, rvdj)]
            Dp = _cdr3_distance_matrix(keys_p)
            np.fill_diagonal(Dp, np.inf)
            perm_counts[p] = (Dp <= radius).sum(axis=1)
    else:
        # V/J-matched permutation — shuffle clonotype identity within strata.
        vj = _vj_genes(adata)
        strata = (vj["vj_v"].astype(str) + "|" + vj["vdj_v"].astype(str)
                  + "|" + vj["vj_j"].astype(str) + "|" + vj["vdj_j"].astype(str))
        clone_stratum = np.array([strata.iloc[rep_pos[c]] for c in clones])
        stratum_groups: dict[str, np.ndarray] = {}
        for s in np.unique(clone_stratum):
            stratum_groups[s] = np.where(clone_stratum == s)[0]
        for p in range(n_permutations):
            order = np.arange(nc)
            for members in stratum_groups.values():
                if members.size > 1:
                    order[members] = rng.permutation(members)
            Dp = D[np.ix_(order, order)]
            perm_counts[p] = (Dp <= radius).sum(axis=1)

    expected = perm_counts.mean(axis=0)
    # Analytic one-sided p-value: model the close-neighbour count as Poisson
    # with the permutation-estimated background rate. A raw permutation
    # fraction is floored at 1/(n_permutations+1) (~5e-3 at 200 perms), so
    # after BH correction across the whole repertoire no clonotype can reach
    # significance even when it is genuinely enriched. The permutations still
    # estimate the background rate robustly; the Poisson tail then yields a
    # continuous p-value that survives multiple-testing correction — the same
    # model CoNGA uses for TCR clumping.
    from scipy.stats import poisson
    pvals = poisson.sf(n_close - 1, np.maximum(expected, 1e-9))
    pvals = np.clip(np.asarray(pvals, dtype=float), 1e-300, 1.0)
    # empirical permutation fraction kept for transparency
    ge = (perm_counts >= n_close[None, :]).sum(axis=0)
    pvals_emp = (ge + 1.0) / (n_permutations + 1.0)
    padj = _bh_adjust(pvals)

    # merge clumping clonotypes (close + significant) into clumps
    sig = (padj < max_pvalue) & (n_close > 0)
    parent = list(range(nc))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(nc):
        if not sig[i]:
            continue
        for j in range(i + 1, nc):
            if sig[j] and close[i, j]:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[rj] = ri
    comp = np.array([_find(i) for i in range(nc)])
    comp_counts = pd.Series(comp[sig]).value_counts() if sig.any() \
        else pd.Series(dtype=int)
    keep = {c for c, n in comp_counts.items() if n >= min_clump_size}
    clump_order = {c: f"clump_{i}" for i, c in enumerate(
        [c for c in comp_counts.index if c in keep]
    )}
    clump_id = np.array([
        clump_order.get(comp[i]) if (sig[i] and comp[i] in keep) else None
        for i in range(nc)
    ], dtype=object)

    res = pd.DataFrame({
        "clone_id": clones,
        "n_close": n_close,
        "expected": expected,
        "pvalue": pvals,
        "pvalue_adj": padj,
        "pvalue_empirical": pvals_emp,
        "clump_id": clump_id,
    }).sort_values("pvalue").reset_index(drop=True)

    # write per-cell columns
    p_map = dict(zip(res["clone_id"], res["pvalue_adj"]))
    c_map = dict(zip(res["clone_id"], res["clump_id"]))
    cell_clone = adata.obs["clone_id"].astype(str)
    adata.obs[f"{key_added}_pvalue"] = cell_clone.map(p_map).astype(float)
    adata.obs[f"{key_added}_id"] = pd.Categorical(cell_clone.map(c_map))
    adata.uns[key_added] = {
        "radius": float(radius),
        "background": background,
        "n_permutations": int(n_permutations),
        "tcr_backend": backend,
        "n_clumping_clonotypes": int(sig.sum()),
        "n_clumps": int(len(clump_order)),
    }
    return res


# ---------------------------------------------------------------------------
# 4. Graph-vs-features — HotSpot local autocorrelation
# ---------------------------------------------------------------------------
def _build_graph_weights(nbrs: list[set[int]], n: int):
    """Symmetric binary kNN weight matrix (row-degree normalised entries)."""
    W = np.zeros((n, n), dtype=float)
    for i, nb in enumerate(nbrs):
        for j in nb:
            W[i, j] = 1.0
            W[j, i] = 1.0
    return W


def _hotspot_autocorrelation(values: np.ndarray, W: np.ndarray,
                             n_permutations: int, rng) -> tuple:
    """HotSpot-style local autocorrelation of one feature on a graph.

    Computes a Geary/Moran-like local-autocorrelation statistic

    ``H = sum_ij W_ij * z_i * z_j``

    on the standardised feature ``z`` and a permutation-based z-score /
    p-value of ``H`` against label-shuffled nulls.
    """
    v = np.asarray(values, dtype=float)
    mask = ~np.isnan(v)
    if mask.sum() < 3 or np.nanstd(v[mask]) == 0:
        return 0.0, 0.0, 1.0
    z = np.zeros_like(v)
    z[mask] = (v[mask] - v[mask].mean()) / v[mask].std()
    obs = float(z @ W @ z)
    null = np.empty(n_permutations)
    zc = z.copy()
    for p in range(n_permutations):
        perm = rng.permutation(len(zc))
        zp = zc[perm]
        null[p] = zp @ W @ zp
    mu, sd = null.mean(), null.std()
    zscore = (obs - mu) / sd if sd > 0 else 0.0
    pval = (np.sum(null >= obs) + 1.0) / (n_permutations + 1.0)
    return obs, float(zscore), float(pval)


@register_function(
    aliases=["hotspot_features", "tcr_gex_hotspot", "HotSpot特征", "图特征自相关"],
    category="airr",
    description=(
        "Graph-vs-features (HotSpot) analysis for joint TCR + GEX data. "
        "Finds genes (or TCR biochemical features) whose values are spatially "
        "autocorrelated on the GEX or TCR kNN graph — i.e. localized to graph "
        "neighborhoods rather than spread randomly — using a HotSpot-style "
        "local-autocorrelation statistic with a permutation null."
    ),
    requires={"obsm": ["X_pca"]},
    examples=[
        "df = ov.airr.hotspot_features(adata, graph='gex', n_top_genes=200)",
        "df = ov.airr.hotspot_features(adata, graph='tcr', features=['CD8A'])",
    ],
    related=["airr.conga_score", "airr.conga_clusters"],
)
def hotspot_features(
    adata,
    *,
    graph: str = "gex",
    gex_rep: str = "X_pca",
    n_neighbors: int = 10,
    features: Optional[list] = None,
    n_top_genes: int = 200,
    sequence: str = "aa",
    n_permutations: int = 100,
    seed: int = 0,
    layer: Optional[str] = None,
    key_added: str = "hotspot",
):
    """HotSpot graph-vs-features local autocorrelation.

    Builds a kNN graph (from the gene-expression embedding when
    ``graph='gex'`` or from TCR similarity when ``graph='tcr'``) and, for
    each candidate feature, measures how localized its values are on that
    graph. Genes (or per-cell TCR biochemical scores) that are concentrated
    in graph neighborhoods get a high autocorrelation z-score and a low
    p-value.

    Candidate features are, in order of precedence: ``features`` (explicit
    list of ``var_names`` and/or numeric ``obs`` columns) → the ``n_top_genes``
    most-variable genes of ``adata.X`` → automatically derived TCR
    biochemical features (CDR3 length, charge, hydrophobicity).

    Parameters
    ----------
    adata
        AnnData with a GEX embedding and per-cell TCR data.
    graph
        ``'gex'`` (gene-expression graph, default) or ``'tcr'`` (TCR graph).
    gex_rep
        ``obsm`` key of the GEX embedding for the GEX graph.
    n_neighbors
        kNN neighborhood size for the chosen graph.
    features
        Explicit list of features (``var_names`` or numeric ``obs`` columns).
    n_top_genes
        When ``features`` is ``None``, number of most-variable genes tested.
    sequence
        CDR3 sequence space for the TCR graph (``'aa'`` / ``'nt'``).
    n_permutations
        Permutations for the autocorrelation null distribution.
    seed
        Random seed.
    layer
        Optional ``adata.layers`` key to read gene values from.
    key_added
        ``uns`` key for the result summary (default ``'hotspot'``).

    Returns
    -------
    :class:`pandas.DataFrame`
        Per-feature table — ``feature``, ``feature_type``, ``autocorrelation``
        (raw statistic), ``zscore``, ``pvalue``, ``pvalue_adj`` — sorted by
        ``zscore`` descending. Also stored in ``uns[key_added]['results']``.
    """
    if graph not in ("gex", "tcr"):
        raise ValueError("graph must be 'gex' or 'tcr'.")
    rng = np.random.default_rng(seed)
    n = adata.n_obs

    # --- build the chosen graph ---
    if graph == "gex":
        if gex_rep not in adata.obsm:
            raise KeyError(f"obsm[{gex_rep!r}] not found.")
        X = np.asarray(adata.obsm[gex_rep], dtype=float)
        nbrs = _knn_from_embedding(X, n_neighbors)
        backend = gex_rep
    else:
        nbrs, backend = _tcr_knn(adata, n_neighbors, sequence)
    W = _build_graph_weights(nbrs, n)

    # --- collect candidate feature vectors ---
    feat_values: dict[str, np.ndarray] = {}
    feat_types: dict[str, str] = {}

    def _gene_vec(name):
        if name not in adata.var_names:
            return None
        col = adata.var_names.get_loc(name)
        mat = adata.layers[layer] if layer is not None else adata.X
        v = mat[:, col]
        v = np.asarray(v.todense()).ravel() if hasattr(v, "todense") \
            else np.asarray(v).ravel()
        return v.astype(float)

    if features is not None:
        for f in features:
            if f in adata.obs and pd.api.types.is_numeric_dtype(adata.obs[f]):
                feat_values[f] = pd.to_numeric(
                    adata.obs[f], errors="coerce"
                ).values.astype(float)
                feat_types[f] = "obs"
            else:
                gv = _gene_vec(f)
                if gv is not None:
                    feat_values[f] = gv
                    feat_types[f] = "gene"
    else:
        # top-variable genes of X
        if adata.n_vars > 1 and not (
            adata.n_vars == 1 and adata.var_names[0] == "_ir_placeholder"
        ):
            mat = adata.layers[layer] if layer is not None else adata.X
            dense = np.asarray(mat.todense()) if hasattr(mat, "todense") \
                else np.asarray(mat)
            dense = dense.astype(float)
            var = dense.var(axis=0)
            top = np.argsort(var)[::-1][:min(n_top_genes, adata.n_vars)]
            for j in top:
                if var[j] > 0:
                    name = str(adata.var_names[j])
                    feat_values[name] = dense[:, j]
                    feat_types[name] = "gene"

    # always offer TCR biochemical features (esp. for graph='tcr')
    if not feat_values or graph == "tcr":
        for name, vec in _tcr_biochem_features(adata, sequence).items():
            feat_values.setdefault(name, vec)
            feat_types.setdefault(name, "tcr_feature")

    if not feat_values:
        raise ValueError("No candidate features found to test.")

    # --- score every feature ---
    rows = []
    for name, vec in feat_values.items():
        auto, zscore, pval = _hotspot_autocorrelation(
            vec, W, n_permutations, rng
        )
        rows.append({
            "feature": name,
            "feature_type": feat_types[name],
            "autocorrelation": auto,
            "zscore": zscore,
            "pvalue": pval,
        })
    res = pd.DataFrame(rows)
    res["pvalue_adj"] = _bh_adjust(res["pvalue"].values)
    res = res.sort_values("zscore", ascending=False).reset_index(drop=True)

    adata.uns[key_added] = {
        "graph": graph,
        "backend": backend,
        "n_neighbors": int(n_neighbors),
        "n_features": int(len(res)),
        "n_significant": int(np.sum(res["pvalue_adj"] < 0.05)),
        "results": res,
    }
    return res


def _tcr_biochem_features(adata, sequence: str = "aa") -> dict:
    """Per-cell TCR biochemical features — CDR3 length / charge / hydropathy."""
    # Kyte-Doolittle hydropathy
    kd = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
        "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
        "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
        "Y": -1.3, "V": 4.2,
    }
    charge = {"R": 1.0, "K": 1.0, "H": 0.5, "D": -1.0, "E": -1.0}
    keys = _cdr3_key(adata, sequence)

    length = np.zeros(adata.n_obs)
    net_charge = np.full(adata.n_obs, np.nan)
    hydropathy = np.full(adata.n_obs, np.nan)
    for i, k in enumerate(keys.values):
        seq = k.replace("|", "")
        if not seq:
            length[i] = np.nan
            continue
        length[i] = len(seq)
        if sequence == "aa":
            net_charge[i] = sum(charge.get(c, 0.0) for c in seq)
            vals = [kd[c] for c in seq if c in kd]
            hydropathy[i] = float(np.mean(vals)) if vals else np.nan
    out = {"tcr_cdr3_length": length}
    if sequence == "aa":
        out["tcr_cdr3_charge"] = net_charge
        out["tcr_cdr3_hydropathy"] = hydropathy
    return out


# ---------------------------------------------------------------------------
# 5. Plotting helpers
# ---------------------------------------------------------------------------
def _ax(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


@register_function(
    aliases=["conga_score_plot", "plot_conga_score", "CoNGA评分图"],
    category="airr",
    description=(
        "Scatter a 2-D embedding (obsm['X_umap']) with cells coloured by "
        "their CoNGA score, highlighting graph-vs-graph CoNGA hits."
    ),
    requires={"obs": ["conga_score"]},
    examples=[
        "ov.airr.conga_score_plot(adata, basis='X_umap')",
    ],
    related=["airr.conga_score", "airr.conga_clusters"],
)
def conga_score_plot(
    adata,
    *,
    basis: str = "X_umap",
    score_key: str = "conga",
    ax=None,
    figsize=(6, 5),
    size: float = 18,
    cmap: str = "magma_r",
    title: str = "CoNGA score",
):
    """Plot per-cell CoNGA scores on a 2-D embedding.

    Parameters
    ----------
    adata
        AnnData after :func:`conga_score`.
    basis
        ``obsm`` key of the 2-D embedding (default ``'X_umap'``).
    score_key
        Prefix used by :func:`conga_score` (default ``'conga'``).
    ax, figsize, size, cmap, title
        Standard matplotlib styling controls.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    score_col = f"{score_key}_score"
    if score_col not in adata.obs:
        raise KeyError(f"obs[{score_col!r}] not found — run conga_score first.")
    if basis not in adata.obsm:
        raise KeyError(f"obsm[{basis!r}] not found.")
    ax = _ax(ax, figsize)
    coords = np.asarray(adata.obsm[basis], dtype=float)[:, :2]
    score = pd.to_numeric(adata.obs[score_col], errors="coerce").values
    order = np.argsort(np.nan_to_num(score))
    sc = ax.scatter(
        coords[order, 0], coords[order, 1], c=score[order], s=size,
        cmap=cmap, edgecolors="none",
    )
    cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("-log10 CoNGA p")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


@register_function(
    aliases=["tcr_clumping_plot", "plot_tcr_clumping", "TCR聚集图"],
    category="airr",
    description=(
        "Bar / logo-style summary of TCR clumping hits — observed vs expected "
        "close-neighbor counts for the top clumping clonotypes."
    ),
    examples=[
        "ov.airr.tcr_clumping_plot(clump_df, top_n=15)",
    ],
    related=["airr.tcr_clumping"],
)
def tcr_clumping_plot(
    clumping,
    *,
    top_n: int = 15,
    ax=None,
    figsize=(7, 5),
    title: str = "TCR clumping",
):
    """Plot observed-vs-expected close neighbors for top clumping clonotypes.

    Parameters
    ----------
    clumping
        The per-clonotype DataFrame returned by :func:`tcr_clumping`.
    top_n
        Number of top (lowest-p-value) clonotypes to display.
    ax, figsize, title
        Standard matplotlib styling controls.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    if not isinstance(clumping, pd.DataFrame) or "n_close" not in clumping:
        raise ValueError("Pass the DataFrame returned by ov.airr.tcr_clumping.")
    ax = _ax(ax, figsize)
    df = clumping.head(top_n).iloc[::-1]
    y = np.arange(len(df))
    ax.barh(y, df["n_close"].values, color="#D1495B", label="observed")
    ax.scatter(df["expected"].values, y, color="#1F4E79", zorder=3,
               label="expected", s=30)
    ax.set_yticks(y)
    ax.set_yticklabels(df["clone_id"].astype(str).values, fontsize=8)
    ax.set_xlabel("# close TCR neighbors")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


@register_function(
    aliases=["hotspot_features_plot", "plot_hotspot_features", "HotSpot特征图"],
    category="airr",
    description=(
        "Horizontal bar plot of the top HotSpot graph-vs-features results — "
        "features ranked by local-autocorrelation z-score."
    ),
    examples=[
        "ov.airr.hotspot_features_plot(hotspot_df, top_n=20)",
    ],
    related=["airr.hotspot_features"],
)
def hotspot_features_plot(
    hotspot,
    *,
    top_n: int = 20,
    ax=None,
    figsize=(6, 6),
    title: str = "HotSpot features",
):
    """Bar plot of the top spatially-autocorrelated features.

    Parameters
    ----------
    hotspot
        The per-feature DataFrame returned by :func:`hotspot_features`.
    top_n
        Number of top features (by z-score) to display.
    ax, figsize, title
        Standard matplotlib styling controls.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    if not isinstance(hotspot, pd.DataFrame) or "zscore" not in hotspot:
        raise ValueError(
            "Pass the DataFrame returned by ov.airr.hotspot_features."
        )
    import matplotlib.pyplot as plt

    ax = _ax(ax, figsize)
    df = hotspot.head(top_n).iloc[::-1]
    y = np.arange(len(df))
    sig = (df["pvalue_adj"] < 0.05).values
    colors = np.where(sig, "#2A9D8F", "#BDBDBD")
    ax.barh(y, df["zscore"].values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"].astype(str).values, fontsize=8)
    ax.set_xlabel("autocorrelation z-score")
    ax.set_title(title)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2A9D8F"),
        plt.Rectangle((0, 0), 1, 1, color="#BDBDBD"),
    ]
    ax.legend(handles, ["FDR < 0.05", "n.s."], frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax
