"""Pure-NumPy pre-ranked GSEA — a fast, single-process replacement for the
gseapy ``prerank`` backend.

Why this exists
---------------
``gseapy.prerank`` runs the permutation null with a joblib/loky **process
pool** (``processes=8`` by default). Inside a long-lived, multi-threaded
Python kernel (notebooks, the omicOS agent kernel, macOS ``spawn``) that pool
can **dead-lock** — the call never returns and ``SIGINT`` cannot break the
``loky`` join. It is also slow: it recomputes the running enrichment score in
Python for every gene set × permutation.

This module reimplements pre-ranked GSEA (Subramanian 2005, the same algorithm
``clusterProfiler``/``fgsea`` use) in vectorised NumPy:

* the running enrichment score is evaluated only at hit positions, in closed
  form, so the whole permutation null for a gene set is one batched NumPy op;
* it runs in a **single process** (no ``loky``, no dead-lock, fully
  deterministic for a given ``seed``);
* the null uses the gene-permutation scheme (a gene set of size *k* is compared
  to random *k*-gene sets), identical to ``gseapy``'s pre-rank null.

The returned object mimics the gseapy result (``.ranking``, ``.res2d``,
``.results``) so it is a drop-in for :class:`omicverse.bulk.pyGSEA` plotting.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def _progress_iter(iterable, total=None, desc="", enabled=True):
    """Wrap an iterable in a tqdm bar when available and ``enabled``."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=desc, leave=False)
    except Exception:
        return iterable


class PrerankResult:
    """gseapy-compatible result container.

    Attributes
    ----------
    ranking : pandas.Series
        The ranked metric, sorted descending, indexed by gene.
    res2d : pandas.DataFrame
        One row per tested gene set, indexed by term name. Columns:
        ``es, nes, pval, fdr, geneset_size, matched_size, lead_genes``.
    results : dict
        ``term -> {es, nes, pval, fdr, hit_indices, RES}`` for plotting.
    """

    def __init__(self, ranking: pd.Series, res2d: pd.DataFrame, results: dict):
        self.ranking = ranking
        self.res2d = res2d
        self.results = results


def _parse_rnk(rnk) -> pd.Series:
    """Normalise a ``.rnk`` input (Series / 2-col DataFrame / path) to a
    descending, de-duplicated, NaN-free float Series indexed by gene."""
    if isinstance(rnk, pd.Series):
        s = rnk.copy()
    elif isinstance(rnk, pd.DataFrame):
        if rnk.shape[1] == 1:
            s = rnk.iloc[:, 0].copy()
        else:
            s = rnk.set_index(rnk.columns[0])[rnk.columns[1]].copy()
    else:  # path to a .rnk / tsv
        df = pd.read_csv(rnk, header=None, sep="\t")
        s = df.set_index(df.columns[0])[df.columns[1]]
    s = s[~s.index.astype(str).duplicated(keep="first")]
    s = s.dropna().astype(float)
    s.index = s.index.astype(str)
    return s.sort_values(ascending=False)


def _running_es(hit_pos: np.ndarray, w_hit: np.ndarray, n: int, k: int,
                n_r: float):
    """Full running-ES curve over the ``n`` ranked genes for one gene set.

    Returns ``(RES, es, peak)`` where ``RES`` is the length-``n`` curve, ``es``
    its signed maximum-deviation value, and ``peak`` the index of that maximum.
    """
    inc = np.full(n, -1.0 / (n - k), dtype=np.float64)
    inc[hit_pos] = w_hit / n_r
    res = np.cumsum(inc)
    peak = int(np.argmax(np.abs(res)))
    return res, float(res[peak]), peak


def _es_null_batch(pos: np.ndarray, w_at: np.ndarray, n: int, k: int) -> np.ndarray:
    """Signed ES for a batch of random gene sets, fully vectorised.

    ``pos`` (P, k) ascending hit positions, ``w_at`` (P, k) their weights.
    The running ES only changes at hits, so its max/min over the walk equals
    the max/min of the values just-before and just-after each hit.
    """
    n_r = w_at.sum(axis=1, keepdims=True)
    cum = np.cumsum(w_at, axis=1)                  # cumulative weight after hit j
    j = np.arange(1, k + 1)                         # #hits seen after hit j
    miss = (n - k)
    res_after = cum / n_r - (pos + 1 - j) / miss
    res_before = (cum - w_at) / n_r - (pos - (j - 1)) / miss
    top = np.maximum(res_after.max(axis=1), res_before.max(axis=1))
    bot = np.minimum(res_after.min(axis=1), res_before.min(axis=1))
    return np.where(np.abs(top) >= np.abs(bot), top, bot)


def _sample_positions(n: int, k: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """``p`` random size-``k`` position sets (sorted), as a (p, k) int array.

    Uses ``argpartition`` of a random matrix — O(p·n) and vectorised, no
    Python-level sampling loop."""
    keys = rng.random((p, n), dtype=np.float32)
    idx = np.argpartition(keys, k - 1, axis=1)[:, :k]
    idx.sort(axis=1)
    return idx


def prerank(
    rnk,
    gene_sets: Dict[str, List[str]],
    *,
    permutation_num: int = 1000,
    weight: float = 1.0,
    min_size: int = 15,
    max_size: int = 500,
    seed: int = 0,
    verbose: bool = False,
    progress: bool = True,
    **_ignored,
) -> PrerankResult:
    """Run pre-ranked GSEA on a ranked gene list (single process, NumPy).

    Parameters
    ----------
    rnk
        Ranked metric: a pandas Series, a 2-column DataFrame ``[gene, score]``,
        or a path to a ``.rnk`` file.
    gene_sets
        ``{term: [genes]}`` mapping.
    permutation_num
        Size of the gene-permutation null per gene set (default 1000).
    weight
        Enrichment-score weighting exponent ``p`` (1.0 = classic weighted GSEA).
    min_size, max_size
        Keep gene sets whose matched size is within ``[min_size, max_size]``.
    seed
        Seed for the permutation RNG (deterministic results).

    Returns
    -------
    PrerankResult
        gseapy-compatible result (``.ranking`` / ``.res2d`` / ``.results``).
    """
    ranking = _parse_rnk(rnk)
    genes = ranking.index.to_numpy()
    scores = ranking.to_numpy(dtype=np.float64)
    n = genes.size
    pos_of = {g: i for i, g in enumerate(genes)}
    weights = np.abs(scores) ** weight

    rng = np.random.default_rng(seed)

    # --- observed ES for every retained gene set ---
    observed = []  # (term, es, peak, hit_pos, RES, matched, geneset_size)
    for term, members in _progress_iter(
            gene_sets.items(), total=len(gene_sets),
            desc="GSEA: scoring gene sets", enabled=progress):
        uniq = set(map(str, members))
        idx = np.fromiter((pos_of[g] for g in uniq if g in pos_of),
                          dtype=np.int64)
        matched = idx.size
        if matched < min_size or matched > max_size:
            continue
        hit_pos = np.sort(idx)
        w_hit = weights[hit_pos]
        n_r = w_hit.sum()
        if n_r <= 0:
            continue
        res, es, peak = _running_es(hit_pos, w_hit, n, matched, n_r)
        observed.append((term, es, peak, hit_pos, res, matched, len(uniq)))

    if not observed:
        empty = pd.DataFrame(
            columns=["es", "nes", "pval", "fdr", "geneset_size",
                     "matched_size", "lead_genes"]
        )
        return PrerankResult(ranking, empty, {})

    # --- permutation null, one batched computation per distinct gene-set size ---
    # A random size-k gene set is just the genes holding the k smallest keys of
    # a random vector. We draw ONE random key matrix and argsort it ONCE; every
    # size then slices the first k columns of that shared ordering. This avoids
    # rebuilding a (permutation_num x n) random matrix for each distinct size —
    # the dominant cost on big libraries (GO/Reactome have hundreds of sizes).
    sizes = sorted({rec[5] for rec in observed})
    keys = rng.random((permutation_num, n), dtype=np.float32)
    order = np.argsort(keys, axis=1).astype(np.int32, copy=False)
    del keys
    null_by_size: Dict[int, np.ndarray] = {}
    for k in _progress_iter(sizes, total=len(sizes),
                            desc="GSEA: permutation null", enabled=progress):
        pos = np.sort(order[:, :k], axis=1)
        null_by_size[k] = _es_null_batch(pos, weights[pos], n, k)
        if verbose:
            print(f"   null for size {k} done")

    # --- NES + nominal p-value, and pooled normalised null for FDR ---
    pooled_pos, pooled_neg = [], []
    inter = []  # (term, es, nes, pval, peak, hit_pos, RES, matched, gsize)
    for term, es, peak, hit_pos, res, matched, gsize in observed:
        null = null_by_size[matched]
        pn, nn = null[null > 0], null[null < 0]
        if es >= 0:
            mean_pos = pn.mean() if pn.size else 1.0
            nes = es / mean_pos if mean_pos else 0.0
            pval = (pn >= es).sum() / pn.size if pn.size else 1.0
        else:
            mean_neg = np.abs(nn).mean() if nn.size else 1.0
            nes = es / mean_neg if mean_neg else 0.0
            pval = (nn <= es).sum() / nn.size if nn.size else 1.0
        if pn.size:
            pooled_pos.append(pn / pn.mean())
        if nn.size:
            pooled_neg.append(nn / np.abs(nn).mean())
        inter.append((term, es, nes, max(pval, 1.0 / permutation_num),
                      peak, hit_pos, res, matched, gsize))

    null_nes_pos = np.concatenate(pooled_pos) if pooled_pos else np.array([0.0])
    null_nes_neg = np.concatenate(pooled_neg) if pooled_neg else np.array([0.0])
    obs_nes = np.array([r[2] for r in inter])
    n_obs_pos = max(int((obs_nes >= 0).sum()), 1)
    n_obs_neg = max(int((obs_nes < 0).sum()), 1)
    n_null_pos = max(int((null_nes_pos >= 0).sum()), 1)
    n_null_neg = max(int((null_nes_neg < 0).sum()), 1)

    # --- standard GSEA FDR, leading edge, assemble table ---
    rows, results = [], {}
    for term, es, nes, pval, peak, hit_pos, res, matched, gsize in inter:
        if nes >= 0:
            fdr_num = (null_nes_pos >= nes).sum() / n_null_pos
            fdr_den = (obs_nes[obs_nes >= 0] >= nes).sum() / n_obs_pos
        else:
            fdr_num = (null_nes_neg <= nes).sum() / n_null_neg
            fdr_den = (obs_nes[obs_nes < 0] <= nes).sum() / n_obs_neg
        fdr = float(min(fdr_num / fdr_den, 1.0)) if fdr_den > 0 else 1.0

        # leading-edge genes: hits up to (es>=0) / from (es<0) the ES peak
        if es >= 0:
            lead_idx = hit_pos[hit_pos <= peak]
        else:
            lead_idx = hit_pos[hit_pos >= peak]
        lead_genes = ";".join(genes[lead_idx].tolist())

        rows.append({
            "Term": term, "es": es, "nes": nes, "pval": pval, "fdr": fdr,
            "geneset_size": gsize, "matched_size": matched,
            "lead_genes": lead_genes,
        })
        results[term] = {
            "es": es, "nes": nes, "pval": pval, "fdr": fdr,
            "hit_indices": hit_pos.tolist(), "RES": res,
        }

    res2d = pd.DataFrame(rows).set_index("Term")
    res2d = res2d.reindex(res2d["nes"].abs().sort_values(ascending=False).index)
    return PrerankResult(ranking, res2d, results)
