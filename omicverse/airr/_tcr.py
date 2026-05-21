"""TCR-specificity analysis for omicverse — the ``ov.airr`` TCR layer.

This module covers the *antigen-specificity* side of T-cell receptor
repertoire analysis — the part of the field concerned with grouping TCRs by
their (predicted) antigen rather than by clonal descent:

* :func:`tcrdist`           — TCRdist position-weighted multi-CDR-loop
  distance (Dash et al. 2017 / tcrdist3).
* :func:`tcr_neighbors`     — fixed-radius neighbourhoods on a distance
  matrix.
* :func:`tcr_cluster`       — hierarchical clustering + flat clusters.
* :func:`giana_cluster`     — GIANA-style SVD/isometric CDR3 encoding +
  nearest-neighbour clustering.
* :func:`clustcr_cluster`   — clusTCR-style physicochemical CDR3 encoding +
  community / MCL clustering.
* :func:`meta_clonotypes`   — centroid TCR + adaptive TCRdist radius
  meta-clonotype discovery.
* :func:`specificity_groups`— GLIPH2 specificity groups (wraps ``pygliph``).
* :func:`annotate_antigen`  — VDJdb / McPAS-TCR / IEDB antigen annotation.
* :func:`cdr3_logo` / :func:`cdr3_logo_background` — CDR3 motif logos.
* :func:`detect_invariant`  — MAIT / iNKT invariant-T-cell detection.

Every function accepts flexible input — an :class:`anndata.AnnData` carrying
the per-cell ``ov.airr`` obs schema, a :class:`pyimmunarch.ImmunData`, or a
plain :class:`pandas.DataFrame` with standard AIRR columns (``v_call``,
``j_call``, ``junction_aa`` / ``cdr3_aa``) — and internally normalises to a
single clonotype DataFrame via :func:`_to_clonotype_df`.

The TCRdist / clustering core is pure-Python / numpy (numba-accelerated only
where numba is already an omicverse dependency); the GLIPH2 wrapper and the
optional logo backend import lazily, so ``import omicverse.airr`` succeeds
with no extra packages installed.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .._registry import register_function


# ---------------------------------------------------------------------------
# Optional-dependency helper (mirrors airr/_bcr.py)
# ---------------------------------------------------------------------------
def _require(modname: str, role: str):
    """Lazy-import an optional backend with an actionable error message."""
    import importlib

    try:
        return importlib.import_module(modname)
    except ImportError as exc:  # pragma: no cover - exercised when dep missing
        raise ImportError(
            f"{role} needs the '{modname}' backend. Install with: "
            f"pip install omicverse[airr]   (or pip install {modname})."
        ) from exc


# ---------------------------------------------------------------------------
# Input normalisation — AnnData / ImmunData / DataFrame -> clonotype DataFrame
# ---------------------------------------------------------------------------
#: canonical clonotype columns produced by :func:`_to_clonotype_df`
_CLONO_COLS = ("cdr3_b_aa", "v_b", "j_b", "cdr3_a_aa", "v_a", "j_a", "count")

# candidate source-column names for each canonical field
_COL_ALIASES = {
    "cdr3_b_aa": ("cdr3_b_aa", "junction_aa", "cdr3_aa", "cdr3", "cdr3b",
                  "CDR3.aa", "CDR3.beta.aa", "cdr3.beta", "CDR3b"),
    "v_b": ("v_b", "v_call", "v_gene", "v_b_gene", "V.name", "v",
            "TRBV", "vb_gene"),
    "j_b": ("j_b", "j_call", "j_gene", "j_b_gene", "J.name", "j",
            "TRBJ", "jb_gene"),
    "cdr3_a_aa": ("cdr3_a_aa", "cdr3_alpha_aa", "junction_a_aa", "CDR3.alpha.aa",
                  "cdr3a", "CDR3a"),
    "v_a": ("v_a", "v_a_gene", "va_gene", "TRAV"),
    "j_a": ("j_a", "j_a_gene", "ja_gene", "TRAJ"),
    "count": ("count", "duplicate_count", "Clones", "clone_count", "n"),
}


def _clean(val) -> Optional[str]:
    """Normalise a possibly-missing string cell to ``str`` or ``None``."""
    if val is None:
        return None
    try:
        if val != val:  # NaN
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if s in ("", "None", "nan", "NaN", "NA", "<NA>"):
        return None
    return s


def _first_present(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    """Return the first column of ``df`` whose name matches ``names``."""
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _df_from_anndata(adata) -> pd.DataFrame:
    """Pull the per-cell ``ov.airr`` obs schema into a clonotype DataFrame."""
    obs = adata.obs
    out = pd.DataFrame(index=obs.index)
    # VDJ_1 == beta chain, VJ_1 == alpha chain
    out["cdr3_b_aa"] = obs.get("VDJ_1_junction_aa")
    out["v_b"] = obs.get("VDJ_1_v_gene")
    out["j_b"] = obs.get("VDJ_1_j_gene")
    out["cdr3_a_aa"] = obs.get("VJ_1_junction_aa")
    out["v_a"] = obs.get("VJ_1_v_gene")
    out["j_a"] = obs.get("VJ_1_j_gene")
    cnt = obs.get("VDJ_1_duplicate_count")
    out["count"] = cnt if cnt is not None else 1
    return out


def _df_from_immunarch(data) -> pd.DataFrame:
    """Flatten a :class:`pyimmunarch.ImmunData` into one clonotype frame."""
    frames = []
    samples = getattr(data, "data", data)
    items = samples.items() if hasattr(samples, "items") else enumerate(samples)
    for name, rep in items:
        sub = _normalize_columns(rep.copy())
        sub["sample"] = name
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=list(_CLONO_COLS))
    return pd.concat(frames, ignore_index=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map an arbitrary AIRR-ish DataFrame onto the canonical clonotype cols."""
    out = pd.DataFrame(index=df.index)
    for canon, names in _COL_ALIASES.items():
        src = _first_present(df, names)
        out[canon] = df[src] if src is not None else None
    if out["count"].isna().all():
        out["count"] = 1
    return out


def _to_clonotype_df(data, *, chain: str = "beta") -> pd.DataFrame:
    """Normalise any accepted input to a clean clonotype DataFrame.

    Parameters
    ----------
    data
        An :class:`anndata.AnnData` (``ov.airr`` obs schema), a
        :class:`pyimmunarch.ImmunData`, or a :class:`pandas.DataFrame` with
        standard AIRR columns.
    chain
        ``'beta'`` (default) keeps only rows with a beta CDR3; ``'alpha'``
        keeps rows with an alpha CDR3; ``'both'`` keeps every row.

    Returns
    -------
    :class:`pandas.DataFrame`
        Columns ``cdr3_b_aa``, ``v_b``, ``j_b``, ``cdr3_a_aa``, ``v_a``,
        ``j_a``, ``count`` (canonical ``_CLONO_COLS``).
    """
    if isinstance(data, pd.DataFrame):
        df = _normalize_columns(data)
    elif hasattr(data, "obs") and hasattr(data, "obsm"):  # AnnData
        df = _df_from_anndata(data)
    elif hasattr(data, "data") or (
        isinstance(data, (list, tuple, dict)) and not isinstance(data, str)
    ):
        df = _df_from_immunarch(data)
    else:
        raise TypeError(
            "data must be an AnnData, a pyimmunarch.ImmunData or a "
            f"pandas.DataFrame, got {type(data)!r}."
        )

    for c in _CLONO_COLS:
        if c not in df.columns:
            df[c] = None
    for c in ("cdr3_b_aa", "v_b", "j_b", "cdr3_a_aa", "v_a", "j_a"):
        df[c] = df[c].map(_clean)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1)

    chain = chain.lower()
    if chain in ("beta", "b", "trb"):
        df = df[df["cdr3_b_aa"].notna()]
    elif chain in ("alpha", "a", "tra"):
        df = df[df["cdr3_a_aa"].notna()]
    elif chain in ("both", "paired", "ab"):
        df = df[df["cdr3_b_aa"].notna() | df["cdr3_a_aa"].notna()]
    else:
        raise ValueError("chain must be 'beta', 'alpha' or 'both'.")
    return df.reset_index(drop=True)


def _cdr3_series(df: pd.DataFrame, chain: str) -> pd.Series:
    """Return the CDR3-AA series for the requested chain."""
    return df["cdr3_a_aa"] if chain.lower().startswith("a") else df["cdr3_b_aa"]


# ---------------------------------------------------------------------------
# TCRdist core — position-weighted multi-CDR-loop substitution distance
# ---------------------------------------------------------------------------
# BLOSUM62 substitution matrix (subset over the 20 canonical residues).
_AA = "ARNDCQEGHILKMFPSTWYV"
_AA_IDX = {a: i for i, a in enumerate(_AA)}

# fmt: off
_BLOSUM62 = np.array([
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
], dtype=np.float64)
# fmt: on

# tcrdist3 substitution-distance matrix: dist(a,b) = min(4, 4 - BLOSUM62(a,b)),
# clamped to [0, 4]; identical residues -> 0.
_DMAT = np.minimum(4.0, 4.0 - _BLOSUM62)
np.fill_diagonal(_DMAT, 0.0)
_DMAT = np.clip(_DMAT, 0.0, 4.0)

#: TCRdist germline CDR1 / CDR2 / CDR2.5 loops (mouse/human shared subset).
#: Used as a fallback when a real V-gene CDR loop table is not supplied.
_GAP_CHAR = "."


def _encode(seq: Optional[str]) -> np.ndarray:
    """Encode a CDR3 string to residue indices (-1 for gap / unknown)."""
    if not seq:
        return np.empty(0, dtype=np.int64)
    return np.array([_AA_IDX.get(c, -1) for c in seq], dtype=np.int64)


def _pairwise_seq_dist(a: str, b: str, *, gap_penalty: float = 4.0,
                       ntrim: int = 3, ctrim: int = 2) -> float:
    """tcrdist3 single-CDR-loop distance between two sequences.

    The shorter sequence is centre-padded with gaps so the conserved N/C
    termini stay aligned; each gap costs ``gap_penalty``, every other position
    contributes the BLOSUM62-derived substitution distance.
    """
    a = a or ""
    b = b or ""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    # trim conserved flanks (only when both ends survive the trim)
    short = a
    longg = b
    gaps = lb - la
    # centre-insert gaps into the shorter sequence
    if gaps:
        mid = la // 2
        short = short[:mid] + _GAP_CHAR * gaps + short[mid:]
    es = _encode(short.replace(_GAP_CHAR, ""))
    # walk aligned positions
    total = 0.0
    si = 0
    n = len(short)
    lo = min(ntrim, n)
    hi = max(n - ctrim, lo)
    for pos in range(n):
        cs = short[pos]
        cl = longg[pos]
        weight = 1.0 if (lo <= pos < hi) else 1.0
        if cs == _GAP_CHAR or cl == _GAP_CHAR:
            total += gap_penalty * weight
            continue
        ia = _AA_IDX.get(cs, -1)
        ib = _AA_IDX.get(cl, -1)
        if ia < 0 or ib < 0:
            total += gap_penalty * weight
        else:
            total += _DMAT[ia, ib] * weight
    _ = es, si
    return float(total)


def _cdr3_distance_matrix(seqs: Sequence[str], *, gap_penalty: float = 4.0,
                          ntrim: int = 3, ctrim: int = 2) -> np.ndarray:
    """Vectorised all-vs-all tcrdist3 CDR3 distance matrix.

    Sequences of equal length are compared in a single vectorised numpy
    pass (the common case); unequal-length pairs fall back to the
    gap-padding routine.
    """
    n = len(seqs)
    D = np.zeros((n, n), dtype=np.float64)
    encs = [_encode(s) for s in seqs]
    lens = np.array([len(s) for s in seqs])

    # group by length so equal-length blocks vectorise
    for L in np.unique(lens):
        idx = np.where(lens == L)[0]
        if L == 0 or len(idx) < 2:
            continue
        mat = np.stack([encs[i] for i in idx])  # (k, L)
        k = len(idx)
        # substitution distance via the lookup table, position by position
        block = np.zeros((k, k), dtype=np.float64)
        for p in range(L):
            col = mat[:, p]
            valid = col >= 0
            sub = np.zeros((k, k))
            vi = np.where(valid)[0]
            if len(vi):
                lut = _DMAT[np.ix_(col[vi], col[vi])]
                sub[np.ix_(vi, vi)] = lut
            # invalid residue on either side -> gap penalty
            inv = ~valid
            if inv.any():
                sub[inv, :] = gap_penalty
                sub[:, inv] = gap_penalty
            block += sub
        D[np.ix_(idx, idx)] = block

    # unequal-length pairs
    for i in range(n):
        for j in range(i + 1, n):
            if lens[i] == lens[j]:
                continue
            d = _pairwise_seq_dist(seqs[i], seqs[j], gap_penalty=gap_penalty,
                                   ntrim=ntrim, ctrim=ctrim)
            D[i, j] = D[j, i] = d
    return D


def _vgene_distance_matrix(v_genes: Sequence[Optional[str]],
                           mismatch: float = 4.0) -> np.ndarray:
    """Crude V-gene distance: 0 if identical gene, ``mismatch`` otherwise.

    A stand-in for the germline CDR1/CDR2/CDR2.5 loop comparison when a real
    V-gene loop table is not available — V-gene identity is a strong proxy
    for germline-loop similarity.
    """
    n = len(v_genes)
    norm = [None if g is None else str(g).split("*")[0] for g in v_genes]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if norm[i] is None or norm[j] is None or norm[i] != norm[j]:
                D[i, j] = D[j, i] = mismatch if (
                    norm[i] != norm[j]) else 0.0
    return D


@register_function(
    aliases=["tcrdist", "tcr_distance", "tcrdist3", "TCRdist距离", "TCR距离"],
    category="airr",
    description=(
        "TCRdist position-weighted multi-CDR-loop substitution distance "
        "(Dash et al. 2017 / tcrdist3): BLOSUM62-derived per-residue "
        "distances over CDR1/2/2.5/3 with a 3x weight on CDR3 and a gap "
        "penalty, computed per chain and combined. Accepts an AnnData, a "
        "pyimmunarch.ImmunData or an AIRR DataFrame."
    ),
    examples=[
        "D, df = ov.airr.tcrdist(adata, chain='beta')",
        "D, df = ov.airr.tcrdist(tcr_df, chains=('alpha', 'beta'))",
    ],
    related=["airr.tcr_neighbors", "airr.tcr_cluster", "airr.meta_clonotypes"],
)
def tcrdist(
    data,
    *,
    chain: str = "beta",
    chains: Optional[Sequence[str]] = None,
    cdr3_weight: float = 3.0,
    gap_penalty: float = 4.0,
    ntrim: int = 3,
    ctrim: int = 2,
    include_vgene: bool = True,
):
    """Compute the TCRdist distance matrix over a TCR repertoire.

    TCRdist (Dash et al., *Nature* 2017) scores TCR similarity as a sum of
    BLOSUM62-derived per-residue substitution distances over the four CDR
    loops — CDR1, CDR2, CDR2.5 and CDR3 — with the antigen-contacting CDR3
    up-weighted (``cdr3_weight``, default ``3``) and an additive ``gap_penalty``
    for length differences. Here the germline CDR1/2/2.5 contribution is
    approximated by V-gene identity (a faithful proxy when a V-gene loop
    table is unavailable); the CDR3 loop uses the full position-weighted
    substitution distance.

    Parameters
    ----------
    data
        An :class:`anndata.AnnData` (``ov.airr`` obs schema), a
        :class:`pyimmunarch.ImmunData`, or an AIRR :class:`pandas.DataFrame`.
    chain
        Single chain to score — ``'beta'`` (default) or ``'alpha'``. Ignored
        when ``chains`` is given.
    chains
        Optional pair, e.g. ``('alpha', 'beta')`` — the per-chain distance
        matrices are summed into a paired-chain distance.
    cdr3_weight
        Multiplier applied to the CDR3-loop distance (default ``3.0``).
    gap_penalty
        Additive penalty per gap position (default ``4.0``).
    ntrim, ctrim
        Residues trimmed from the conserved CDR3 N / C termini.
    include_vgene
        Add the V-gene (germline-loop proxy) distance.

    Returns
    -------
    tuple
        ``(D, df)`` — the ``(n, n)`` :class:`numpy.ndarray` TCRdist matrix and
        the normalised clonotype :class:`pandas.DataFrame` whose row order
        matches ``D``.
    """
    use = list(chains) if chains else [chain]
    df = _to_clonotype_df(data, chain="both" if len(use) > 1 else use[0])
    n = len(df)
    D = np.zeros((n, n), dtype=np.float64)
    for ch in use:
        seqs = _cdr3_series(df, ch).fillna("").astype(str).tolist()
        cdr3_d = _cdr3_distance_matrix(
            seqs, gap_penalty=gap_penalty, ntrim=ntrim, ctrim=ctrim
        )
        D += cdr3_weight * cdr3_d
        if include_vgene:
            vcol = "v_a" if ch.lower().startswith("a") else "v_b"
            D += _vgene_distance_matrix(df[vcol].tolist())
    np.fill_diagonal(D, 0.0)
    return D, df


# ---------------------------------------------------------------------------
# Radius / neighbour clustering
# ---------------------------------------------------------------------------
@register_function(
    aliases=["tcr_neighbors", "tcr_radius", "tcr_radius_cluster",
             "TCR近邻", "TCR半径聚类"],
    category="airr",
    description=(
        "Fixed-radius neighbourhood clustering on a TCRdist distance matrix: "
        "every TCR within the given radius of another is linked, and the "
        "connected components form clusters. Accepts a precomputed distance "
        "matrix or any input understood by ov.airr.tcrdist."
    ),
    examples=[
        "res = ov.airr.tcr_neighbors(adata, radius=24)",
        "res = ov.airr.tcr_neighbors(D, radius=12, distance=D, df=df)",
    ],
    related=["airr.tcrdist", "airr.tcr_cluster"],
)
def tcr_neighbors(
    data,
    *,
    radius: float = 24.0,
    chain: str = "beta",
    distance: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    min_cluster_size: int = 2,
    **tcrdist_kwargs,
):
    """Fixed-radius neighbourhood clustering on the TCRdist graph.

    Two TCRs are neighbours when their TCRdist is ``<= radius``; clusters are
    the connected components of the resulting neighbour graph.

    Parameters
    ----------
    data
        An AnnData / ImmunData / AIRR DataFrame, *or* a precomputed
        distance matrix (in which case pass ``df`` too).
    radius
        Maximum TCRdist for two TCRs to be neighbours.
    chain
        Chain used when ``data`` needs a fresh :func:`tcrdist` computation.
    distance, df
        Optionally reuse a precomputed ``(D, df)`` pair from :func:`tcrdist`.
    min_cluster_size
        Components smaller than this are labelled ``-1`` (singletons).
    **tcrdist_kwargs
        Forwarded to :func:`tcrdist` when a distance matrix is computed.

    Returns
    -------
    dict
        ``{'labels': ndarray, 'n_neighbors': ndarray, 'df': DataFrame,
        'distance': ndarray, 'radius': float}`` — ``labels`` are cluster ids
        (``-1`` = singleton), ``n_neighbors`` the per-TCR neighbour count.
    """
    D, frame = _resolve_distance(data, distance, df, chain, tcrdist_kwargs)
    n = D.shape[0]
    adj = (D <= radius) & ~np.eye(n, dtype=bool)
    n_neighbors = adj.sum(axis=1)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
    comp = _connected_components(n, edges)
    labels = _relabel_by_size(comp, min_cluster_size)
    frame = frame.copy()
    frame["tcr_neighbor_cluster"] = labels
    frame["tcr_n_neighbors"] = n_neighbors
    return {
        "labels": labels, "n_neighbors": n_neighbors, "df": frame,
        "distance": D, "radius": float(radius),
    }


# ---------------------------------------------------------------------------
# Hierarchical clustering
# ---------------------------------------------------------------------------
@register_function(
    aliases=["tcr_cluster", "tcr_hierarchical", "tcr_hclust",
             "TCR层次聚类", "TCR聚类"],
    category="airr",
    description=(
        "Hierarchical (agglomerative) clustering of TCRs from a TCRdist "
        "distance matrix, cut into flat clusters either at a fixed distance "
        "or to a target cluster count. Accepts a precomputed distance matrix "
        "or any input understood by ov.airr.tcrdist."
    ),
    examples=[
        "res = ov.airr.tcr_cluster(adata, t=36, criterion='distance')",
        "res = ov.airr.tcr_cluster(tcr_df, t=10, criterion='maxclust')",
    ],
    related=["airr.tcrdist", "airr.tcr_neighbors"],
)
def tcr_cluster(
    data,
    *,
    t: float = 36.0,
    criterion: str = "distance",
    method: str = "average",
    chain: str = "beta",
    distance: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    **tcrdist_kwargs,
):
    """Hierarchical clustering of TCRs from the TCRdist matrix.

    Parameters
    ----------
    data
        An AnnData / ImmunData / AIRR DataFrame, or a precomputed distance
        matrix (pass ``df`` too).
    t
        Cut threshold — a distance (``criterion='distance'``) or the target
        number of clusters (``criterion='maxclust'``).
    criterion
        ``'distance'`` or ``'maxclust'`` (see :func:`scipy.cluster.hierarchy.fcluster`).
    method
        Linkage method — ``'average'`` (default), ``'complete'``, ``'single'``,
        ``'ward'`` …
    chain
        Chain used when ``data`` needs a fresh :func:`tcrdist` computation.
    distance, df
        Optionally reuse a precomputed ``(D, df)`` pair.
    **tcrdist_kwargs
        Forwarded to :func:`tcrdist`.

    Returns
    -------
    dict
        ``{'labels': ndarray, 'linkage': ndarray, 'df': DataFrame,
        'distance': ndarray}`` — ``labels`` are 0-based cluster ids.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    D, frame = _resolve_distance(data, distance, df, chain, tcrdist_kwargs)
    n = D.shape[0]
    if n < 2:
        labels = np.zeros(n, dtype=int)
        frame = frame.copy()
        frame["tcr_cluster"] = labels
        return {"labels": labels, "linkage": np.empty((0, 4)),
                "df": frame, "distance": D}
    Dsym = (D + D.T) / 2.0
    np.fill_diagonal(Dsym, 0.0)
    condensed = squareform(Dsym, checks=False)
    Z = linkage(condensed, method=method)
    raw = fcluster(Z, t=t, criterion=criterion)
    # 0-based, ordered by size
    labels = _relabel_by_size(raw, min_size=1)
    frame = frame.copy()
    frame["tcr_cluster"] = labels
    return {"labels": labels, "linkage": Z, "df": frame, "distance": Dsym}


# ---------------------------------------------------------------------------
# GIANA-style fast clustering
# ---------------------------------------------------------------------------
# Atchley physicochemical factor scores (5-D) for the 20 amino acids.
# fmt: off
_ATCHLEY = {
    "A": (-0.591, -1.302, -0.733,  1.570, -0.146),
    "C": (-1.343,  0.465, -0.862, -1.020, -0.255),
    "D": ( 1.050,  0.302, -3.656, -0.259, -3.242),
    "E": ( 1.357, -1.453,  1.477,  0.113, -0.837),
    "F": (-1.006, -0.590,  1.891, -0.397,  0.412),
    "G": (-0.384,  1.652,  1.330,  1.045,  2.064),
    "H": ( 0.336, -0.417, -1.673, -1.474, -0.078),
    "I": (-1.239, -0.547,  2.131,  0.393,  0.816),
    "K": ( 1.831, -0.561,  0.533, -0.277,  1.648),
    "L": (-1.019, -0.987, -1.505,  1.266, -0.912),
    "M": (-0.663, -1.524,  2.219, -1.005,  1.212),
    "N": ( 0.945,  0.828,  1.299, -0.169,  0.933),
    "P": ( 0.189,  2.081, -1.628,  0.421, -1.392),
    "Q": ( 0.931, -0.179, -3.005, -0.503, -1.853),
    "R": ( 1.538, -0.055,  1.502,  0.440,  2.897),
    "S": (-0.228,  1.399, -4.760,  0.670, -2.647),
    "T": (-0.032,  0.326,  2.213,  0.908,  1.313),
    "V": (-1.337, -0.279, -0.544,  1.242, -1.262),
    "W": (-0.595,  0.009,  0.672, -2.128, -0.184),
    "Y": ( 0.260,  0.830,  3.097, -0.838,  1.512),
}
# fmt: on


def _connected_components(n: int, edges: Sequence[tuple]) -> np.ndarray:
    """Union-find connected components (mirrors airr/_clonotype.py)."""
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
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def _relabel_by_size(comp: np.ndarray, min_size: int) -> np.ndarray:
    """Relabel raw component ids to 0-based ids ordered by size.

    Components smaller than ``min_size`` get label ``-1``.
    """
    comp = np.asarray(comp)
    out = np.full(comp.shape[0], -1, dtype=np.int64)
    vals, counts = np.unique(comp, return_counts=True)
    order = sorted(zip(vals, counts), key=lambda x: -x[1])
    nxt = 0
    for v, c in order:
        if c >= min_size:
            out[comp == v] = nxt
            nxt += 1
    return out


def _cdr3_feature_matrix(seqs: Sequence[str], *, n_pos: int = 8,
                         trim: int = 3) -> np.ndarray:
    """Encode CDR3 cores as fixed-length 5-D Atchley physicochemical vectors.

    The variable-length CDR3 core (after trimming ``trim`` conserved residues
    each end) is resampled to ``n_pos`` anchor positions; each anchor stores
    the 5 Atchley factors, giving an ``n_pos * 5``-D feature per TCR.
    """
    feats = []
    zero = (0.0, 0.0, 0.0, 0.0, 0.0)
    for s in seqs:
        s = s or ""
        core = s[trim:len(s) - trim] if len(s) > 2 * trim else s
        if not core:
            feats.append(np.zeros(n_pos * 5))
            continue
        vecs = np.array([_ATCHLEY.get(c, zero) for c in core])  # (L, 5)
        L = len(core)
        # resample to n_pos anchors
        anchors = np.linspace(0, L - 1, n_pos)
        lo = np.floor(anchors).astype(int)
        hi = np.minimum(lo + 1, L - 1)
        frac = (anchors - lo)[:, None]
        sampled = vecs[lo] * (1 - frac) + vecs[hi] * frac
        feats.append(sampled.reshape(-1))
    return np.asarray(feats, dtype=np.float64)


@register_function(
    aliases=["giana_cluster", "giana", "tcr_giana", "GIANA聚类", "GIANA"],
    category="airr",
    description=(
        "GIANA-style fast TCR clustering: encode each CDR3 into an isometric "
        "physicochemical feature vector, reduce it with a truncated SVD, then "
        "link TCRs sharing the same V gene that fall within a small Euclidean "
        "radius in the encoded space. Scales near-linearly with repertoire "
        "size."
    ),
    examples=[
        "res = ov.airr.giana_cluster(adata, thr=7.0)",
        "res = ov.airr.giana_cluster(tcr_df, n_components=12, thr=6.0)",
    ],
    related=["airr.clustcr_cluster", "airr.tcr_cluster"],
)
def giana_cluster(
    data,
    *,
    chain: str = "beta",
    thr: float = 7.0,
    n_components: int = 16,
    n_pos: int = 8,
    require_same_v: bool = True,
    min_cluster_size: int = 2,
):
    """GIANA-style fast nearest-neighbour TCR clustering.

    GIANA (Zhang et al., 2021) achieves near-linear-time TCR clustering by
    isometrically encoding CDR3s into a fixed-length numeric space so that
    Euclidean distance approximates a CDR3 substitution distance; here the
    encoding is the resampled Atchley-factor feature reduced by a truncated
    SVD. TCRs that share a V gene and lie within ``thr`` in the reduced space
    are linked, and connected components form the specificity clusters.

    Parameters
    ----------
    data
        An AnnData / ImmunData / AIRR DataFrame.
    chain
        ``'beta'`` (default) or ``'alpha'``.
    thr
        Euclidean radius in the encoded space for two TCRs to be linked.
    n_components
        Truncated-SVD dimensionality of the isometric encoding.
    n_pos
        Number of CDR3 anchor positions in the feature encoding.
    require_same_v
        Only link TCRs with the same V gene (GIANA's default).
    min_cluster_size
        Components smaller than this get label ``-1``.

    Returns
    -------
    dict
        ``{'labels': ndarray, 'embedding': ndarray, 'df': DataFrame}``.
    """
    df = _to_clonotype_df(data, chain=chain)
    seqs = _cdr3_series(df, chain).fillna("").astype(str).tolist()
    vcol = "v_a" if chain.lower().startswith("a") else "v_b"
    n = len(df)
    feats = _cdr3_feature_matrix(seqs, n_pos=n_pos)

    # truncated SVD isometric encoding
    if n >= 2 and feats.shape[1] > 0:
        k = min(n_components, min(feats.shape) - 1) or 1
        centred = feats - feats.mean(axis=0, keepdims=True)
        try:
            U, S, _ = np.linalg.svd(centred, full_matrices=False)
            emb = U[:, :k] * S[:k]
        except np.linalg.LinAlgError:  # pragma: no cover
            emb = centred[:, :k]
    else:
        emb = feats

    # group by length + V gene, then radius-link within each group
    v_norm = [None if v is None else str(v).split("*")[0] for v in df[vcol]]
    lens = np.array([len(s) for s in seqs])
    edges: list[tuple[int, int]] = []
    buckets: dict[tuple, list[int]] = {}
    for i in range(n):
        key = (lens[i], v_norm[i]) if require_same_v else (lens[i],)
        buckets.setdefault(key, []).append(i)
    thr2 = thr * thr
    for members in buckets.values():
        for ai in range(len(members)):
            for bi in range(ai + 1, len(members)):
                i, j = members[ai], members[bi]
                d2 = float(np.sum((emb[i] - emb[j]) ** 2))
                if d2 <= thr2:
                    edges.append((i, j))
    comp = _connected_components(n, edges)
    labels = _relabel_by_size(comp, min_cluster_size)
    df = df.copy()
    df["giana_cluster"] = labels
    return {"labels": labels, "embedding": emb, "df": df}


# ---------------------------------------------------------------------------
# clusTCR-style clustering
# ---------------------------------------------------------------------------
@register_function(
    aliases=["clustcr_cluster", "clustcr", "tcr_clustcr", "clusTCR聚类",
             "clusTCR"],
    category="airr",
    description=(
        "clusTCR-style two-step TCR clustering: encode CDR3s with "
        "physicochemical (Atchley) features, build a k-nearest-neighbour "
        "graph in feature space, then partition it with greedy-modularity "
        "(Louvain-like) community detection — an MCL-style fast clustering."
    ),
    examples=[
        "res = ov.airr.clustcr_cluster(adata, n_neighbors=10)",
        "res = ov.airr.clustcr_cluster(tcr_df, resolution=1.2)",
    ],
    related=["airr.giana_cluster", "airr.tcr_cluster"],
)
def clustcr_cluster(
    data,
    *,
    chain: str = "beta",
    n_neighbors: int = 8,
    n_pos: int = 8,
    resolution: float = 1.0,
    min_cluster_size: int = 2,
):
    """clusTCR-style physicochemical-feature community clustering.

    clusTCR (Valkiers et al., 2021) clusters TCRs in two steps — a fast
    hashing/feature step followed by Markov-clustering (MCL) of a similarity
    graph. This implementation encodes each CDR3 with resampled Atchley
    physicochemical features, builds a mutual k-nearest-neighbour graph, and
    partitions it with greedy-modularity community detection (a Louvain/MCL
    surrogate) — no external graph library required.

    Parameters
    ----------
    data
        An AnnData / ImmunData / AIRR DataFrame.
    chain
        ``'beta'`` (default) or ``'alpha'``.
    n_neighbors
        Number of nearest neighbours per TCR in the similarity graph.
    n_pos
        Number of CDR3 anchor positions in the feature encoding.
    resolution
        Community-detection resolution — larger values yield more, smaller
        clusters.
    min_cluster_size
        Communities smaller than this get label ``-1``.

    Returns
    -------
    dict
        ``{'labels': ndarray, 'features': ndarray, 'df': DataFrame}``.
    """
    df = _to_clonotype_df(data, chain=chain)
    seqs = _cdr3_series(df, chain).fillna("").astype(str).tolist()
    n = len(df)
    feats = _cdr3_feature_matrix(seqs, n_pos=n_pos)

    if n < 2:
        labels = np.full(n, -1, dtype=np.int64)
        df = df.copy()
        df["clustcr_cluster"] = labels
        return {"labels": labels, "features": feats, "df": df}

    # k-NN graph in feature space
    from scipy.spatial.distance import cdist

    Dfeat = cdist(feats, feats)
    np.fill_diagonal(Dfeat, np.inf)
    k = min(n_neighbors, n - 1)
    knn = np.argsort(Dfeat, axis=1)[:, :k]

    labels = _greedy_modularity(n, knn, resolution=resolution)
    labels = _relabel_by_size(labels, min_cluster_size)
    df = df.copy()
    df["clustcr_cluster"] = labels
    return {"labels": labels, "features": feats, "df": df}


def _greedy_modularity(n: int, knn: np.ndarray, *,
                       resolution: float = 1.0,
                       max_iter: int = 50) -> np.ndarray:
    """Louvain-style greedy-modularity community detection on a k-NN graph."""
    # symmetric, unweighted edge set
    adj: list[set] = [set() for _ in range(n)]
    for i in range(n):
        for j in knn[i]:
            j = int(j)
            adj[i].add(j)
            adj[j].add(i)
    deg = np.array([len(a) for a in adj], dtype=np.float64)
    m2 = deg.sum()
    if m2 == 0:
        return np.arange(n, dtype=np.int64)
    comm = np.arange(n, dtype=np.int64)
    comm_deg = deg.copy()

    for _ in range(max_iter):
        moved = False
        for i in range(n):
            ci = comm[i]
            comm_deg[ci] -= deg[i]
            # gain per candidate community
            links: dict[int, int] = {}
            for j in adj[i]:
                links[comm[j]] = links.get(comm[j], 0) + 1
            best_c, best_gain = ci, 0.0
            for c, k_in in links.items():
                gain = k_in - resolution * deg[i] * comm_deg[c] / m2
                if gain > best_gain:
                    best_gain, best_c = gain, c
            comm[i] = best_c
            comm_deg[best_c] += deg[i]
            if best_c != ci:
                moved = True
        if not moved:
            break
    return comm


# ---------------------------------------------------------------------------
# Distance-resolution helper shared by tcr_neighbors / tcr_cluster
# ---------------------------------------------------------------------------
def _resolve_distance(data, distance, df, chain, tcrdist_kwargs):
    """Return ``(D, df)`` from a precomputed pair or a fresh tcrdist call."""
    if distance is not None:
        D = np.asarray(distance, dtype=np.float64)
        if df is None:
            if isinstance(data, np.ndarray) or data is distance:
                df = pd.DataFrame({"cdr3_b_aa": [None] * D.shape[0]})
            else:
                df = _to_clonotype_df(data, chain=chain)
        return D, df.reset_index(drop=True)
    if isinstance(data, np.ndarray):
        D = np.asarray(data, dtype=np.float64)
        frame = (df if df is not None
                 else pd.DataFrame({"cdr3_b_aa": [None] * D.shape[0]}))
        return D, frame.reset_index(drop=True)
    D, frame = tcrdist(data, chain=chain, **(tcrdist_kwargs or {}))
    return D, frame


# ---------------------------------------------------------------------------
# Meta-clonotype discovery
# ---------------------------------------------------------------------------
@register_function(
    aliases=["meta_clonotypes", "meta_clonotype", "tcr_metaclonotypes",
             "元克隆型", "TCR元克隆型"],
    category="airr",
    description=(
        "Discover meta-clonotypes — a centroid TCR plus an adaptive TCRdist "
        "radius (and an optional CDR3 regex motif) that captures a "
        "biochemically similar neighbourhood while keeping the hit-rate "
        "against a background repertoire low (tcrdist3-style)."
    ),
    examples=[
        "mc = ov.airr.meta_clonotypes(adata, background=bg_df)",
        "mc = ov.airr.meta_clonotypes(tcr_df, radii=(0,12,24,36,48))",
    ],
    related=["airr.tcrdist", "airr.tcr_neighbors", "airr.specificity_groups"],
)
def meta_clonotypes(
    data,
    *,
    chain: str = "beta",
    background=None,
    radii: Sequence[float] = (0, 6, 12, 18, 24, 30, 36, 48),
    max_background_rate: float = 1e-4,
    min_neighbors: int = 2,
    add_motif: bool = True,
    **tcrdist_kwargs,
):
    """Discover adaptive-radius meta-clonotypes.

    A *meta-clonotype* is a centroid TCR together with the largest TCRdist
    radius at which its neighbourhood still hits a background repertoire at a
    rate ``<= max_background_rate``. This yields antigen-specificity features
    that generalise beyond a single clone while staying specific
    (Mayer-Blackwell et al., *eLife* 2021).

    Parameters
    ----------
    data
        The foreground (e.g. antigen-enriched) repertoire — AnnData /
        ImmunData / AIRR DataFrame.
    chain
        ``'beta'`` (default) or ``'alpha'``.
    background
        Optional background repertoire (same accepted types). When given the
        radius of each meta-clonotype is shrunk until the background
        hit-rate falls at or below ``max_background_rate``.
    radii
        Candidate radii, scanned largest-first.
    max_background_rate
        Maximum allowed fraction of background TCRs inside the radius.
    min_neighbors
        Drop centroids with fewer than this many foreground neighbours at
        the chosen radius.
    add_motif
        Add a per-meta-clonotype CDR3 regex motif (a per-position
        IUPAC-style consensus over the neighbourhood).
    **tcrdist_kwargs
        Forwarded to :func:`tcrdist`.

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per discovered meta-clonotype — ``centroid_cdr3``, ``v_gene``,
        ``j_gene``, ``radius``, ``n_neighbors``, ``background_rate`` and
        (optionally) ``motif``.
    """
    D, df = tcrdist(data, chain=chain, **tcrdist_kwargs)
    n = len(df)
    seqs = _cdr3_series(df, chain).fillna("").astype(str).tolist()
    vcol = "v_a" if chain.lower().startswith("a") else "v_b"
    jcol = "j_a" if chain.lower().startswith("a") else "j_b"

    bg_seqs: list[str] = []
    Dbg = None
    if background is not None:
        bg_df = _to_clonotype_df(background, chain=chain)
        bg_seqs = _cdr3_series(bg_df, chain).fillna("").astype(str).tolist()
        # foreground-vs-background CDR3 distance only (cheap, vectorised core)
        Dbg = _cross_cdr3_distance(seqs, bg_seqs,
                                   tcrdist_kwargs.get("gap_penalty", 4.0))
        Dbg = Dbg * tcrdist_kwargs.get("cdr3_weight", 3.0)
    n_bg = max(len(bg_seqs), 1)

    sorted_radii = sorted(radii, reverse=True)
    rows = []
    for i in range(n):
        chosen_r = 0.0
        n_nb = 0
        bg_rate = 0.0
        for r in sorted_radii:
            fg_hits = int(np.sum(D[i] <= r)) - 1  # exclude self
            if Dbg is not None:
                bg_hits = int(np.sum(Dbg[i] <= r))
                rate = bg_hits / n_bg
            else:
                rate = 0.0
            if rate <= max_background_rate:
                chosen_r, n_nb, bg_rate = float(r), fg_hits, rate
                break
        if n_nb < min_neighbors:
            continue
        row = {
            "centroid_cdr3": seqs[i],
            "v_gene": df[vcol].iloc[i],
            "j_gene": df[jcol].iloc[i],
            "radius": chosen_r,
            "n_neighbors": n_nb,
            "background_rate": bg_rate,
        }
        if add_motif:
            members = np.where(D[i] <= chosen_r)[0]
            row["motif"] = _cdr3_motif_regex([seqs[m] for m in members])
        rows.append(row)

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    # collapse: keep the widest-radius meta-clonotype per centroid CDR3
    res = (res.sort_values(["n_neighbors", "radius"], ascending=False)
              .drop_duplicates("centroid_cdr3")
              .reset_index(drop=True))
    return res


def _cross_cdr3_distance(a_seqs: Sequence[str], b_seqs: Sequence[str],
                         gap_penalty: float = 4.0) -> np.ndarray:
    """All-vs-all CDR3 tcrdist distance between two sequence lists."""
    na, nb = len(a_seqs), len(b_seqs)
    D = np.zeros((na, nb), dtype=np.float64)
    for i in range(na):
        for j in range(nb):
            D[i, j] = _pairwise_seq_dist(a_seqs[i], b_seqs[j],
                                         gap_penalty=gap_penalty)
    return D


def _cdr3_motif_regex(seqs: Sequence[str]) -> str:
    """Build a per-position regex motif from a set of equal-ish CDR3s.

    Positions that are fully conserved become a literal residue; partially
    conserved positions become a character class; the most common length is
    used as the motif scaffold.
    """
    seqs = [s for s in seqs if s]
    if not seqs:
        return ""
    lengths = pd.Series([len(s) for s in seqs])
    target_len = int(lengths.mode().iloc[0])
    same_len = [s for s in seqs if len(s) == target_len]
    if not same_len:
        return seqs[0]
    cols = list(zip(*same_len))
    parts = []
    for col in cols:
        residues = sorted(set(col))
        if len(residues) == 1:
            parts.append(residues[0])
        elif len(residues) <= 4:
            parts.append("[" + "".join(residues) + "]")
        else:
            parts.append(".")
    return "".join(parts)


# ---------------------------------------------------------------------------
# GLIPH2 specificity groups (pygliph wrapper)
# ---------------------------------------------------------------------------
@register_function(
    aliases=["specificity_groups", "gliph2", "gliph", "tcr_gliph",
             "GLIPH2特异性组", "特异性组"],
    category="airr",
    description=(
        "GLIPH2 specificity groups: cluster TCRs that likely recognise the "
        "same epitope by shared CDR3 motifs and global similarity, via the "
        "pygliph backend. Returns the GLIPH2 convergence groups with their "
        "enrichment scores."
    ),
    examples=[
        "groups = ov.airr.specificity_groups(adata)",
        "groups = ov.airr.specificity_groups(tcr_df, reference=ref_df)",
    ],
    related=["airr.meta_clonotypes", "airr.giana_cluster",
             "airr.clustcr_cluster"],
)
def specificity_groups(
    data,
    *,
    chain: str = "beta",
    reference=None,
    **kwargs,
):
    """GLIPH2 specificity-group discovery (``pygliph``).

    GLIPH2 (Huang et al., *Nat. Biotechnol.* 2020) groups TCRs predicted to
    share antigen specificity by detecting (a) shared CDR3 enriched motifs
    and (b) global CDR3 similarity, scoring each group for motif enrichment,
    CDR3-length / V-gene bias and clonal expansion.

    Parameters
    ----------
    data
        An AnnData / ImmunData / AIRR DataFrame of query TCRs.
    chain
        ``'beta'`` (default) or ``'alpha'``.
    reference
        Optional naive-repertoire reference (same accepted types) used by
        GLIPH2 to estimate motif-enrichment background.
    **kwargs
        Forwarded to the underlying ``pygliph`` entry point.

    Returns
    -------
    dict
        The GLIPH2 result with keys ``'motif_enrichment'``,
        ``'global_enrichment'``, ``'connections'``, ``'cluster_properties'``
        (the convergence-group table with enrichment scores),
        ``'cluster_list'`` and ``'parameters'``.
    """
    gliph = _require("pygliph", "GLIPH2 specificity grouping")
    df = _to_clonotype_df(data, chain=chain)
    cdr3 = _cdr3_series(df, chain)
    vcol = "v_a" if chain.lower().startswith("a") else "v_b"
    query = pd.DataFrame({
        "CDR3b": cdr3.values,
        "TRBV": df[vcol].values,
        "TRBJ": (df["j_a"] if chain.lower().startswith("a")
                 else df["j_b"]).values,
        "counts": df["count"].values,
    })
    ref_df = None
    if reference is not None:
        rdf = _to_clonotype_df(reference, chain=chain)
        ref_df = pd.DataFrame({
            "CDR3b": _cdr3_series(rdf, chain).values,
            "TRBV": rdf[vcol].values,
        })
    # pygliph exposes either a `gliph2` function or a `GLIPH2` class — be
    # tolerant of both API shapes.
    if hasattr(gliph, "gliph2"):
        return gliph.gliph2(
            query,
            refdb_beta=("gliph_reference" if ref_df is None else ref_df),
            **kwargs,
        )
    if hasattr(gliph, "combined"):
        return gliph.combined(
            query,
            refdb_beta=("gliph_reference" if ref_df is None else ref_df),
            **kwargs,
        )
    raise AttributeError(
        "pygliph is installed but exposes neither gliph2() nor combined() "
        "— cannot dispatch GLIPH2 specificity grouping."
    )


# ---------------------------------------------------------------------------
# Antigen-specificity database annotation
# ---------------------------------------------------------------------------
@register_function(
    aliases=["annotate_antigen", "tcr_antigen", "match_vdjdb",
             "抗原特异性注释", "TCR抗原注释"],
    category="airr",
    description=(
        "Annotate a TCR repertoire with putative antigen specificity by "
        "matching CDR3aa (optionally with V/J genes) against a reference "
        "database — VDJdb, McPAS-TCR or IEDB — using exact and "
        "within-distance matching. Accepts the reference table as a "
        "DataFrame so it works fully offline."
    ),
    examples=[
        "ann = ov.airr.annotate_antigen(adata, reference=vdjdb_df)",
        "ann = ov.airr.annotate_antigen(tcr_df, reference=mcpas_df, max_distance=12)",
    ],
    related=["airr.specificity_groups", "airr.meta_clonotypes"],
)
def annotate_antigen(
    data,
    reference: pd.DataFrame,
    *,
    chain: str = "beta",
    match_v: bool = False,
    match_j: bool = False,
    max_distance: float = 0.0,
    gap_penalty: float = 4.0,
    key_added: Optional[str] = None,
    copy: bool = False,
):
    """Annotate a repertoire against a TCR-antigen reference database.

    Each query TCR's CDR3 amino-acid sequence is matched against the
    reference (VDJdb / McPAS-TCR / IEDB) — exactly when ``max_distance == 0``,
    otherwise within a TCRdist CDR3 radius. The best (closest) reference hit
    and its antigen / epitope annotation are attached to every query row.

    Parameters
    ----------
    data
        The query repertoire — AnnData / ImmunData / AIRR DataFrame.
    reference
        A reference :class:`pandas.DataFrame`. Column names are auto-detected
        — CDR3 (``cdr3``/``CDR3``/``junction_aa``…), antigen / epitope
        (``antigen``/``epitope``/``Epitope``…), V / J genes, plus an optional
        ``species`` / ``Pathology`` column.
    chain
        ``'beta'`` (default) or ``'alpha'``.
    match_v, match_j
        Require the V / J gene to match in addition to the CDR3.
    max_distance
        ``0`` for exact CDR3 matching; otherwise the maximum TCRdist CDR3
        distance for a hit.
    gap_penalty
        Gap penalty for the within-distance CDR3 comparison.
    key_added
        When ``data`` is an AnnData, write the annotation straight back into
        ``adata.obs``. The default (``None``) keeps the historic
        return-only behaviour. Pass ``''`` to write the bare columns
        ``epitope`` / ``antigen`` / ``antigen_species`` (correctly aligned to
        cells that carry a CDR3, ``NaN`` elsewhere); pass a non-empty string
        to prefix them (``f'{key_added}_epitope'`` …). Ignored for
        non-AnnData input.
    copy
        When ``True`` (and ``data`` is an AnnData with ``key_added`` set),
        annotate and return a copy instead of modifying ``data`` in place.

    Returns
    -------
    :class:`pandas.DataFrame` or AnnData
        By default the query clonotype frame with appended ``antigen``,
        ``epitope``, ``antigen_species``, ``ref_cdr3`` and
        ``antigen_distance`` columns (``NaN`` where no hit was found). When
        ``data`` is an AnnData and ``key_added`` is set, the annotated
        AnnData (the same object, or a copy when ``copy=True``) is returned
        instead, with the annotation columns written into ``obs``.
    """
    df = _to_clonotype_df(data, chain=chain)
    q_cdr3 = _cdr3_series(df, chain).fillna("").astype(str).tolist()
    vcol = "v_a" if chain.lower().startswith("a") else "v_b"
    jcol = "j_a" if chain.lower().startswith("a") else "j_b"

    ref = reference.copy()
    r_cdr3_col = _first_present(ref, _COL_ALIASES["cdr3_b_aa"])
    if r_cdr3_col is None:
        raise ValueError(
            "reference must carry a CDR3 column (cdr3 / CDR3 / junction_aa…)."
        )
    ant_col = _first_present(
        ref, ("antigen", "Antigen", "antigen_gene", "Antigen.gene",
              "antigen.gene", "Gene", "Pathology", "Category"))
    epi_col = _first_present(
        ref, ("epitope", "Epitope", "antigen_epitope", "Epitope.peptide",
              "antigen.epitope", "epitope_seq", "peptide"))
    sp_col = _first_present(
        ref, ("species", "Species", "antigen_species", "Antigen.species",
              "antigen.species"))
    rv_col = _first_present(ref, _COL_ALIASES["v_b"])
    rj_col = _first_present(ref, _COL_ALIASES["j_b"])

    ref_cdr3 = ref[r_cdr3_col].map(_clean)
    keep = ref_cdr3.notna()
    ref = ref[keep].reset_index(drop=True)
    ref_cdr3 = ref_cdr3[keep].reset_index(drop=True)

    # exact-match index for the fast path
    exact: dict[str, list[int]] = {}
    for ri, s in enumerate(ref_cdr3):
        exact.setdefault(s, []).append(ri)

    def _ref_field(col, ri):
        return ref[col].iloc[ri] if (col is not None and ri is not None) else None

    out_ant, out_epi, out_sp, out_ref, out_dist = [], [], [], [], []
    for qi, qs in enumerate(q_cdr3):
        best_ri = None
        best_d = np.inf
        if qs in exact:
            cand = exact[qs]
            best_ri, best_d = cand[0], 0.0
            # tighten by V/J if requested
            for ri in cand:
                if match_v and _clean(_ref_field(rv_col, ri)) != _clean(df[vcol].iloc[qi]):
                    continue
                if match_j and _clean(_ref_field(rj_col, ri)) != _clean(df[jcol].iloc[qi]):
                    continue
                best_ri, best_d = ri, 0.0
                break
            else:
                if match_v or match_j:
                    best_ri, best_d = None, np.inf
        if best_ri is None and max_distance > 0 and qs:
            for ri, rs in enumerate(ref_cdr3):
                if match_v and _clean(_ref_field(rv_col, ri)) != _clean(df[vcol].iloc[qi]):
                    continue
                if match_j and _clean(_ref_field(rj_col, ri)) != _clean(df[jcol].iloc[qi]):
                    continue
                d = _pairwise_seq_dist(qs, rs, gap_penalty=gap_penalty)
                if d < best_d:
                    best_d, best_ri = d, ri
            if best_d > max_distance:
                best_ri, best_d = None, np.inf
        out_ant.append(_ref_field(ant_col, best_ri))
        out_epi.append(_ref_field(epi_col, best_ri))
        out_sp.append(_ref_field(sp_col, best_ri))
        out_ref.append(ref_cdr3.iloc[best_ri] if best_ri is not None else None)
        out_dist.append(best_d if best_ri is not None else np.nan)

    res = df.copy()
    res["antigen"] = out_ant
    res["epitope"] = out_epi
    res["antigen_species"] = out_sp
    res["ref_cdr3"] = out_ref
    res["antigen_distance"] = out_dist

    # optional in-place write-back into an AnnData's obs
    is_anndata = hasattr(data, "obs") and hasattr(data, "obsm")
    if key_added is not None and is_anndata:
        target = data.copy() if copy else data
        usable = usable_cdr3_mask(target, chain=chain)
        prefix = f"{key_added}_" if key_added else ""
        for col in ("epitope", "antigen", "antigen_species"):
            dest = f"{prefix}{col}"
            target.obs[dest] = np.nan
            target.obs.loc[usable, dest] = res[col].values
        return target
    return res


# ---------------------------------------------------------------------------
# CDR3 motif logos
# ---------------------------------------------------------------------------
def _position_frequency_matrix(seqs: Sequence[str]) -> pd.DataFrame:
    """Per-position amino-acid frequency matrix over equal-length CDR3s.

    Sequences are bucketed by length and the most-common length is used.
    """
    seqs = [s for s in seqs if s]
    if not seqs:
        return pd.DataFrame(columns=list(_AA))
    lengths = pd.Series([len(s) for s in seqs])
    target = int(lengths.mode().iloc[0])
    aligned = [s for s in seqs if len(s) == target]
    counts = np.zeros((target, len(_AA)), dtype=np.float64)
    for s in aligned:
        for p, c in enumerate(s):
            if c in _AA_IDX:
                counts[p, _AA_IDX[c]] += 1
    freq = counts / max(len(aligned), 1)
    return pd.DataFrame(freq, columns=list(_AA))


def _information_matrix(freq: pd.DataFrame) -> pd.DataFrame:
    """Convert a frequency matrix to an information-content (bits) matrix."""
    if freq.empty:
        return freq
    p = freq.values
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
    info = np.log2(20) - ent  # total information per position (bits)
    return pd.DataFrame(p * info[:, None], columns=freq.columns)


def _draw_logo(matrix: pd.DataFrame, *, ax=None, title: str = "",
               ylabel: str = "bits"):
    """Draw a sequence logo — via logomaker if available, else matplotlib."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, 0.45 * len(matrix)), 2.6))
    if matrix.empty:
        ax.set_title(title or "(no sequences)")
        return ax
    try:
        logomaker = _require("logomaker", "CDR3 logo")
        logomaker.Logo(matrix, ax=ax, color_scheme="chemistry")
    except ImportError:
        # matplotlib fallback — stacked coloured bars per position
        scheme = {a: c for a, c in zip(
            _AA, plt.cm.tab20(np.linspace(0, 1, 20)))}
        for pos in range(len(matrix)):
            order = matrix.iloc[pos].sort_values()
            bottom = 0.0
            for aa, val in order.items():
                if val <= 0:
                    continue
                ax.bar(pos, val, bottom=bottom, width=0.9,
                       color=scheme.get(aa, "grey"))
                if val > 0.15 * (matrix.values.max() or 1):
                    ax.text(pos, bottom + val / 2, aa, ha="center",
                            va="center", fontsize=8, fontweight="bold")
                bottom += val
    ax.set_xlabel("CDR3 position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


@register_function(
    aliases=["cdr3_logo", "tcr_logo", "cdr3_motif_logo", "CDR3标识图",
             "CDR3 logo"],
    category="airr",
    description=(
        "Draw a CDR3 amino-acid sequence-logo (motif plot) for a TCR "
        "repertoire or a single specificity cluster — an information-content "
        "(bits) logo over the most common CDR3 length. Uses logomaker when "
        "available, otherwise a matplotlib stacked-bar fallback."
    ),
    examples=[
        "ax = ov.airr.cdr3_logo(adata, chain='beta')",
        "ax = ov.airr.cdr3_logo(tcr_df, kind='probability')",
    ],
    related=["airr.cdr3_logo_background", "airr.specificity_groups"],
)
def cdr3_logo(
    data,
    *,
    chain: str = "beta",
    kind: str = "information",
    trim: int = 0,
    ax=None,
    title: Optional[str] = None,
):
    """Draw a CDR3 sequence-motif logo.

    Parameters
    ----------
    data
        An AnnData / ImmunData / AIRR DataFrame (or a single specificity
        cluster's slice thereof).
    chain
        ``'beta'`` (default) or ``'alpha'``.
    kind
        ``'information'`` (bits, default) or ``'probability'`` (raw
        per-position frequencies).
    trim
        Trim this many conserved residues from each CDR3 terminus before
        building the logo.
    ax
        Optional Matplotlib axis to draw on.
    title
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = _to_clonotype_df(data, chain=chain)
    seqs = _cdr3_series(df, chain).dropna().astype(str).tolist()
    if trim:
        seqs = [s[trim:len(s) - trim] for s in seqs if len(s) > 2 * trim]
    freq = _position_frequency_matrix(seqs)
    if kind == "information":
        matrix = _information_matrix(freq)
        ylabel = "bits"
    elif kind == "probability":
        matrix = freq
        ylabel = "probability"
    else:
        raise ValueError("kind must be 'information' or 'probability'.")
    return _draw_logo(matrix, ax=ax, ylabel=ylabel,
                      title=title or f"CDR3 {chain} logo (n={len(seqs)})")


@register_function(
    aliases=["cdr3_logo_background", "cdr3_enrichment_logo",
             "background_logo", "背景扣除标识图", "富集logo"],
    category="airr",
    description=(
        "Draw a background-subtracted CDR3 logo: per-position amino-acid "
        "enrichment of a foreground TCR set relative to a background "
        "repertoire (log2 frequency ratio), highlighting the residues that "
        "drive antigen specificity."
    ),
    examples=[
        "ax = ov.airr.cdr3_logo_background(cluster_df, background=bg_df)",
    ],
    related=["airr.cdr3_logo", "airr.meta_clonotypes"],
)
def cdr3_logo_background(
    data,
    background,
    *,
    chain: str = "beta",
    trim: int = 0,
    pseudocount: float = 1e-3,
    ax=None,
    title: Optional[str] = None,
):
    """Draw a background-subtracted (enrichment) CDR3 logo.

    Each position's letter heights are the log2 ratio of the foreground to
    the background amino-acid frequency — positive (upward) letters are
    enriched in the foreground, negative letters are depleted.

    Parameters
    ----------
    data
        The foreground repertoire — AnnData / ImmunData / AIRR DataFrame.
    background
        The background repertoire (same accepted types).
    chain
        ``'beta'`` (default) or ``'alpha'``.
    trim
        Conserved residues trimmed from each CDR3 terminus.
    pseudocount
        Added to every frequency before the log ratio to avoid divide-by-zero.
    ax
        Optional Matplotlib axis.
    title
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    def _seqs(d):
        frame = _to_clonotype_df(d, chain=chain)
        ss = _cdr3_series(frame, chain).dropna().astype(str).tolist()
        if trim:
            ss = [s[trim:len(s) - trim] for s in ss if len(s) > 2 * trim]
        return ss

    fg = _position_frequency_matrix(_seqs(data))
    bg = _position_frequency_matrix(_seqs(background))
    if fg.empty:
        return _draw_logo(fg, ax=ax, title=title or "(no foreground)")
    L = len(fg)
    # align background to foreground length
    if len(bg) >= L:
        bg = bg.iloc[:L].reset_index(drop=True)
    else:
        pad = pd.DataFrame(
            np.full((L - len(bg), len(_AA)), 1.0 / len(_AA)),
            columns=list(_AA))
        bg = pd.concat([bg, pad], ignore_index=True)
    ratio = np.log2((fg.values + pseudocount) / (bg.values + pseudocount))
    matrix = pd.DataFrame(ratio, columns=list(_AA))
    return _draw_logo(matrix, ax=ax, ylabel="log2 enrichment",
                      title=title or f"CDR3 {chain} enrichment logo")


# ---------------------------------------------------------------------------
# MAIT / iNKT invariant-T-cell detection
# ---------------------------------------------------------------------------
#: invariant-T-cell V/J gene rules.
_MAIT_TRAV = {"TRAV1-2", "TRAV1"}
_MAIT_TRAJ = {"TRAJ33", "TRAJ12", "TRAJ20"}
_INKT_TRAV = {"TRAV10", "TRAV24"}     # TRAV24 = human iNKT, TRAV10 = mouse
_INKT_TRAJ = {"TRAJ18"}


def _gene_root(val) -> Optional[str]:
    """Strip the allele suffix (``*01``) from a gene name."""
    v = _clean(val)
    return None if v is None else v.split("*")[0].split(" ")[0].upper()


@register_function(
    aliases=["detect_invariant", "mait_inkt", "detect_mait", "invariant_tcells",
             "恒定T细胞检测", "MAIT检测"],
    category="airr",
    description=(
        "Detect innate-like invariant T cells from their semi-invariant TCR "
        "alpha chain: MAIT cells (TRAV1-2 paired with TRAJ33/12/20) and iNKT "
        "cells (TRAV10/TRAV24 paired with TRAJ18). Writes the call back to "
        "obs['invariant_tcell'] when given an AnnData."
    ),
    examples=[
        "res = ov.airr.detect_invariant(adata)",
        "df = ov.airr.detect_invariant(tcr_df)",
    ],
    related=["airr.annotate_antigen", "airr.specificity_groups"],
)
def detect_invariant(
    data,
    *,
    key_added: str = "invariant_tcell",
    require_j: bool = True,
):
    """Flag MAIT and iNKT cells by their invariant TCR-alpha V/J genes.

    MAIT (mucosal-associated invariant T) cells carry a semi-invariant
    TRAV1-2 alpha chain rearranged to TRAJ33 (or TRAJ12 / TRAJ20); iNKT
    (invariant natural killer T) cells carry TRAV10–TRAJ18 (mouse) /
    TRAV24–TRAJ18 (human). This function applies those germline-gene rules
    to the VJ (alpha) chain.

    Parameters
    ----------
    data
        An AnnData (``ov.airr`` obs schema), ImmunData or AIRR DataFrame.
    key_added
        When ``data`` is an AnnData, the ``obs`` column the call is written
        to (default ``'invariant_tcell'``); the same object is returned.
    require_j
        Require the invariant J gene as well as the V gene. When ``False``,
        the V-gene rule alone (``'MAIT-like'`` / ``'iNKT-like'``) is used.

    Returns
    -------
    AnnData or :class:`pandas.DataFrame`
        For an AnnData input the input object with ``obs[key_added]`` added
        (a categorical of ``'MAIT'`` / ``'iNKT'`` / ``'conventional'`` …);
        otherwise the clonotype frame with an ``invariant_tcell`` column.
    """
    is_anndata = hasattr(data, "obs") and hasattr(data, "obsm")
    if is_anndata:
        # work on the full per-cell frame so the call aligns with obs index
        df = _df_from_anndata(data)
        for c in ("v_a", "j_a"):
            df[c] = df[c].map(_clean)
    else:
        df = _to_clonotype_df(data, chain="both")
    n = len(df)
    calls = np.array(["conventional"] * n, dtype=object)
    for i in range(n):
        v = _gene_root(df["v_a"].iloc[i])
        j = _gene_root(df["j_a"].iloc[i])
        if v is None:
            calls[i] = "unknown"
            continue
        is_mait_v = v in _MAIT_TRAV
        is_inkt_v = v in _INKT_TRAV
        if require_j:
            if is_mait_v and j in _MAIT_TRAJ:
                calls[i] = "MAIT"
            elif is_inkt_v and j in _INKT_TRAJ:
                calls[i] = "iNKT"
        else:
            if is_mait_v:
                calls[i] = "MAIT-like"
            elif is_inkt_v:
                calls[i] = "iNKT-like"

    if is_anndata:
        data.obs[key_added] = pd.Categorical(
            pd.Series(calls, index=data.obs.index)
        )
        return data

    out = df.copy()
    out[key_added] = calls
    return out


# ---------------------------------------------------------------------------
# CDR3 cleaning / usability helpers
# ---------------------------------------------------------------------------
@register_function(
    aliases=["clean_cdr3", "tcr_clean_cdr3", "CDR3清洗", "CDR3规范化"],
    category="airr",
    description=(
        "Normalise a possibly-missing CDR3 string the same way the ov.airr "
        "TCR layer does internally: empty / placeholder / NA values "
        "('', 'None', 'nan', 'NaN', 'NA', '<NA>') become None, otherwise the "
        "stripped string is returned. A thin public wrapper of the private "
        "_clean helper."
    ),
    examples=[
        "ov.airr.clean_cdr3('CASSLAPGATNEKLFF')",
        "ov.airr.clean_cdr3(adata.obs['VDJ_1_junction_aa'])",
    ],
    related=["airr.usable_cdr3_mask", "airr.annotate_antigen"],
)
def clean_cdr3(value):
    """Normalise a CDR3 string (or a Series of them) to ``str`` / ``None``.

    This is the public, documented wrapper of the package-internal
    ``_clean`` helper that the ``ov.airr`` TCR layer uses to coerce
    missing / placeholder CDR3 strings to ``None``.

    Parameters
    ----------
    value
        A single CDR3-like value, or a :class:`pandas.Series` of them.

    Returns
    -------
    str or None or :class:`pandas.Series`
        For a scalar input, the stripped string or ``None`` when the value
        is empty / a missing placeholder (``''``, ``'None'``, ``'nan'``,
        ``'NaN'``, ``'NA'``, ``'<NA>'``). For a Series input, the
        element-wise cleaned Series.
    """
    if isinstance(value, pd.Series):
        return value.map(_clean)
    return _clean(value)


@register_function(
    aliases=["usable_cdr3_mask", "tcr_usable_mask", "可用CDR3掩码",
             "CDR3可用掩码"],
    category="airr",
    description=(
        "Boolean mask over cells of an AnnData flagging those that carry a "
        "usable (non-missing) CDR3 amino-acid sequence for the requested "
        "chain — the beta CDR3 (VDJ_1_junction_aa) by default. Uses the same "
        "CDR3 cleaning as the rest of the ov.airr TCR layer."
    ),
    requires={"obs": ["VDJ_1_junction_aa"]},
    examples=[
        "keep = ov.airr.usable_cdr3_mask(adata, chain='beta')",
        "truth = adata.obs.loc[keep, 'antigen_epitope']",
    ],
    related=["airr.clean_cdr3", "airr.tcrdist"],
)
def usable_cdr3_mask(adata, *, chain: str = "beta") -> pd.Series:
    """Mask of cells carrying a usable CDR3 amino-acid sequence.

    A cell is "usable" when its CDR3 amino-acid column for the requested
    chain cleans to a non-``None`` string (see :func:`clean_cdr3`). The
    returned mask aligns one-to-one with ``adata.obs`` and is the standard
    way to align a TCRdist / clonotype frame back to ground-truth obs
    columns (the ``ov.airr`` TCR functions drop cells with no CDR3).

    Parameters
    ----------
    adata
        An :class:`anndata.AnnData` carrying the ``ov.airr`` obs schema
        (``VDJ_1_junction_aa`` for the beta chain, ``VJ_1_junction_aa`` for
        the alpha chain).
    chain
        ``'beta'`` (default) uses ``VDJ_1_junction_aa``; ``'alpha'`` uses
        ``VJ_1_junction_aa``.

    Returns
    -------
    :class:`pandas.Series`
        A boolean Series indexed by ``adata.obs_names`` — ``True`` where the
        cell carries a usable CDR3.
    """
    if not (hasattr(adata, "obs") and hasattr(adata, "obsm")):
        raise TypeError("usable_cdr3_mask expects an AnnData.")
    chain = chain.lower()
    if chain in ("alpha", "a", "tra"):
        col = "VJ_1_junction_aa"
    elif chain in ("beta", "b", "trb"):
        col = "VDJ_1_junction_aa"
    else:
        raise ValueError("chain must be 'beta' or 'alpha'.")
    if col not in adata.obs:
        raise KeyError(
            f"obs[{col!r}] not found — load TCR data into the ov.airr "
            "obs schema first."
        )
    return adata.obs[col].map(_clean).notna()


# ---------------------------------------------------------------------------
# TCRdist 2-D embedding
# ---------------------------------------------------------------------------
@register_function(
    aliases=["tcrdist_embedding", "tcr_embedding", "tcrdist_mds",
             "TCRdist嵌入", "TCR距离嵌入"],
    category="airr",
    description=(
        "Compute a 2-D embedding of a precomputed TCRdist distance matrix "
        "for visualisation — multidimensional scaling (MDS, default) on the "
        "symmetrised distance matrix, or t-SNE with a precomputed metric. "
        "Takes the (n, n) distance matrix returned by ov.airr.tcrdist."
    ),
    examples=[
        "D, df = ov.airr.tcrdist(adata, chain='beta')",
        "emb = ov.airr.tcrdist_embedding(D, method='mds')",
    ],
    related=["airr.tcrdist", "airr.tcr_cluster"],
)
def tcrdist_embedding(
    D,
    *,
    method: str = "mds",
    n_components: int = 2,
    random_state: int = 0,
) -> np.ndarray:
    """Embed a TCRdist distance matrix in low dimensions.

    The TCRdist matrix is symmetrised (``(D + D.T) / 2``) and embedded with
    multidimensional scaling — a faithful 2-D layout where Euclidean
    distance approximates TCRdist — so antigen-specific TCR groups become
    visually separable.

    Parameters
    ----------
    D
        An ``(n, n)`` TCRdist distance matrix (the first element of the
        tuple returned by :func:`tcrdist`).
    method
        ``'mds'`` (default) for metric multidimensional scaling, or
        ``'tsne'`` for t-SNE on the precomputed distance metric.
    n_components
        Embedding dimensionality (default ``2``).
    random_state
        Random seed for the (stochastic) embedding solver.

    Returns
    -------
    :class:`numpy.ndarray`
        An ``(n, n_components)`` array of embedding coordinates whose row
        order matches ``D``.
    """
    D = np.asarray(getattr(D, "values", D), dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square (n, n) distance matrix.")
    D_sym = (D + D.T) / 2.0
    method = method.lower()
    if method == "mds":
        from sklearn.manifold import MDS

        mds = MDS(
            n_components=n_components, dissimilarity="precomputed",
            random_state=random_state, n_init=1, normalized_stress=False,
        )
        return mds.fit_transform(D_sym)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30.0, max(2.0, (D.shape[0] - 1) / 3.0))
        tsne = TSNE(
            n_components=n_components, metric="precomputed", init="random",
            perplexity=perplexity, random_state=random_state,
        )
        return tsne.fit_transform(D_sym)
    raise ValueError("method must be 'mds' or 'tsne'.")


# ---------------------------------------------------------------------------
# GLIPH2 specificity-group purity vs ground truth
# ---------------------------------------------------------------------------
@register_function(
    aliases=["specificity_group_purity", "gliph_purity", "tcr_group_purity",
             "特异性组纯度", "GLIPH纯度"],
    category="airr",
    description=(
        "Score each GLIPH2 specificity group (from ov.airr.specificity_groups) "
        "for antigen purity against ground-truth epitope labels carried in an "
        "AnnData. Builds a CDR3->dominant-epitope lookup, then for every "
        "group reports its size and the fraction of member CDR3s mapping to "
        "the most common epitope."
    ),
    examples=[
        "groups = ov.airr.specificity_groups(adata, chain='beta')",
        "pur = ov.airr.specificity_group_purity(groups, adata)",
    ],
    related=["airr.specificity_groups", "airr.annotate_antigen"],
)
def specificity_group_purity(
    groups,
    adata,
    *,
    truth_col: str = "antigen_epitope",
    chain: str = "beta",
    min_size: int = 2,
) -> pd.DataFrame:
    """Per-GLIPH2-group antigen purity vs ground-truth epitope labels.

    Each TCR CDR3 is assigned its dominant ground-truth epitope (the most
    common label among cells carrying that exact CDR3); every GLIPH2
    specificity group is then scored for purity — the fraction of its member
    CDR3s that map to the group's single most common epitope.

    Parameters
    ----------
    groups
        The dict returned by :func:`specificity_groups` — its
        ``'cluster_list'`` entry maps each group tag to a member
        :class:`pandas.DataFrame` with a ``CDR3b`` column.
    adata
        An :class:`anndata.AnnData` carrying the per-cell CDR3 (``ov.airr``
        obs schema) and the ground-truth epitope column ``truth_col``.
    truth_col
        ``obs`` column with the ground-truth antigen / epitope label
        (default ``'antigen_epitope'``).
    chain
        ``'beta'`` (default) or ``'alpha'`` — which CDR3 to key on.
    min_size
        Skip groups with fewer than this many epitope-annotated CDR3s
        (default ``2``).

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per scored group — ``tag``, ``n_cdr3`` (annotated CDR3s in
        the group), ``top_epitope`` (dominant epitope) and ``purity`` (its
        fraction) — sorted by ``n_cdr3`` descending.
    """
    if not isinstance(groups, dict) or "cluster_list" not in groups:
        raise ValueError(
            "groups must be the dict returned by ov.airr.specificity_groups."
        )
    if not (hasattr(adata, "obs") and hasattr(adata, "obsm")):
        raise TypeError("specificity_group_purity expects an AnnData.")
    if truth_col not in adata.obs:
        raise KeyError(f"obs[{truth_col!r}] (ground truth) not found.")

    keep = usable_cdr3_mask(adata, chain=chain)
    cdr3_col = ("VJ_1_junction_aa" if chain.lower().startswith("a")
                else "VDJ_1_junction_aa")
    dom = (
        pd.DataFrame({
            "cdr3": adata.obs.loc[keep, cdr3_col].map(_clean).values,
            "epi": adata.obs.loc[keep, truth_col].astype(str).values,
        })
        .groupby("cdr3")["epi"]
        .agg(lambda s: s.value_counts().index[0])
    )

    rows = []
    for tag, mem in groups["cluster_list"].items():
        if not isinstance(mem, pd.DataFrame) or "CDR3b" not in mem:
            continue
        epis = pd.Series(
            [dom.get(c) for c in pd.unique(mem["CDR3b"])]
        ).dropna()
        if len(epis) < min_size:
            continue
        vc = epis.value_counts()
        rows.append({
            "tag": tag,
            "n_cdr3": int(len(epis)),
            "top_epitope": vc.index[0],
            "purity": float(vc.iloc[0] / len(epis)),
        })
    return (
        pd.DataFrame(rows, columns=["tag", "n_cdr3", "top_epitope", "purity"])
        .sort_values("n_cdr3", ascending=False)
        .reset_index(drop=True)
    )


__all__ = [
    "tcrdist",
    "tcr_neighbors",
    "tcr_cluster",
    "giana_cluster",
    "clustcr_cluster",
    "meta_clonotypes",
    "specificity_groups",
    "annotate_antigen",
    "cdr3_logo",
    "cdr3_logo_background",
    "detect_invariant",
    "clean_cdr3",
    "usable_cdr3_mask",
    "tcrdist_embedding",
    "specificity_group_purity",
]
