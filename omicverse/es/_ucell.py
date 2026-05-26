"""UCell — per-cell signature scoring via Mann-Whitney U on descending ranks.

Reference: Andreatta & Carmona, *Comput Struct Biotechnol J* 19 (2021).
Reproduces the canonical algorithm from the R package
``UCell::ScoreSignatures_UCell`` exactly — see ``HelperFunctions.R`` /
``u_stat`` in the upstream R source.

Algorithm
---------
For each cell ``i`` and signature ``S = S⁺ ∪ S⁻`` (``-``-suffixed genes
form the negative subset):

1. Rank all features in the cell by descending expression (highest =
   rank 1). Ties broken by the average rank (R ``frankv(.., order=-1L,
   ties.method='average')``).
2. Any rank ``> maxRank`` is clamped to ``maxRank``.
3. Missing genes (signature genes not in ``adata.var_names``): if
   ``missing_genes='impute'`` (default), contribute rank ``maxRank``
   each — i.e. treated as the lowest-expressed. If ``'skip'``, dropped
   from the signature.
4. ``rank_sum = Σ ranks(S)`` ; ``rank_sum_min = |S| × (|S| + 1) / 2``.
5. ``UCell_score = 1 - (rank_sum − rank_sum_min) / (|S| × maxRank − rank_sum_min)``.
6. For the negative subset: ``score = max(0, UCell(S⁺) − w_neg × UCell(S⁻))``.

The score lies in ``[0, 1]`` — cells where signature genes top the
ranking get scores near 1; cells where they're at the bottom (or
absent) get scores near 0.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sps
import scipy.stats as sts
from anndata import AnnData
from tqdm.auto import tqdm

from .._monitor import monitor
from .._registry import register_function

__all__ = ["ucell"]


def _u_score(ranks: np.ndarray, gene_idx: np.ndarray, n_missing: int,
             max_rank: int) -> float:
    """Single-signature UCell U-score for one cell.

    Mirrors R ``u_stat`` exactly. ``gene_idx`` is the list of present-
    gene indices (≥ 0); ``n_missing`` is the count of signature genes
    NOT in the data matrix (under ``missing_genes='impute'`` policy
    they contribute ``max_rank`` each to ``rank_sum``).
    """
    len_sig = int(gene_idx.size + n_missing)
    if len_sig <= 0:
        return 0.0
    # Imputed missing genes contribute max_rank each
    rank_sum = float(n_missing * max_rank)
    if gene_idx.size > 0:
        # Clamp any rank > max_rank to max_rank (the R code does this
        # via `insig <- rank_sub >= maxRank; rank_sub[insig] <- maxRank`)
        sub = ranks[gene_idx]
        sub = np.minimum(sub, max_rank)
        rank_sum += float(sub.sum())
    rank_sum_min = len_sig * (len_sig + 1) / 2.0
    denom = len_sig * max_rank - rank_sum_min
    if denom <= 0:
        return 0.0
    return float(1.0 - (rank_sum - rank_sum_min) / denom)


def _split_signature(genes):
    """Split a signature into (positive_genes, negative_genes).

    R UCell convention: a gene name ending in ``-`` is a negative-set
    member, ``+`` (or no suffix) is positive. We strip the suffix and
    return two clean lists.
    """
    pos, neg = [], []
    for g in genes:
        if isinstance(g, str) and g.endswith("-"):
            neg.append(g[:-1])
        elif isinstance(g, str) and g.endswith("+"):
            pos.append(g[:-1])
        else:
            pos.append(g)
    return pos, neg


def _signature_indices(
    var_names_to_idx: Mapping[str, int],
    sig_genes: Sequence[str],
    missing_policy: str,
) -> tuple[np.ndarray, int]:
    """Resolve signature gene names to indices in ``adata.var_names``.

    Returns ``(present_idx_array, n_missing)``. Under ``missing_policy='impute'``
    missing genes are counted (and later contribute ``max_rank`` each);
    under ``'skip'`` they are dropped entirely.
    """
    present, missing = [], 0
    seen = set()
    for g in sig_genes:
        if g in seen:  # match R behaviour: dedup
            continue
        seen.add(g)
        ix = var_names_to_idx.get(g)
        if ix is None:
            if missing_policy == "impute":
                missing += 1
            # else: skip — don't increment anything
            continue
        present.append(ix)
    return np.asarray(present, dtype=np.int64), missing


def _ucell_gpu_score(ranks_np, resolved, *, max_rank: int, w_neg: float) -> np.ndarray:
    """GPU UCell score for a chunk of cells.

    ``ranks_np``: ``(n_cells, n_genes)`` averaged ranks (descending,
    computed on CPU via :func:`scipy.stats.rankdata`).
    ``resolved``: list of ``(pos_idx, pos_miss, neg_idx, neg_miss)`` —
    one per signature.

    Strategy: build a ``(n_genes, n_sig)`` membership matrix per +/-
    subset, clamp ranks at ``max_rank``, then a single matmul
    ``(n_cells, n_genes) @ (n_genes, n_sig)`` gives the rank sums of
    present genes per (cell, signature). Add the missing-gene
    contribution ``pos_miss * max_rank`` per signature, apply the
    closed-form UCell formula, combine with the negative subset.
    """
    import torch
    from ._engine import torch_device

    device = torch_device()
    n_cells, n_genes = ranks_np.shape
    n_sig = len(resolved)

    # Clamp ranks at max_rank — same as the CPU kernel
    ranks_clamped = np.minimum(ranks_np, float(max_rank))
    R = torch.as_tensor(ranks_clamped, dtype=torch.float64, device=device)

    # Build membership matrices: M_pos[g, s] = 1 if g ∈ S⁺_s, else 0
    M_pos = torch.zeros((n_genes, n_sig), dtype=torch.float64, device=device)
    M_neg = torch.zeros((n_genes, n_sig), dtype=torch.float64, device=device)
    len_pos = torch.zeros(n_sig, dtype=torch.float64, device=device)
    len_neg = torch.zeros(n_sig, dtype=torch.float64, device=device)
    rank_sum_min_pos_const = torch.zeros(n_sig, dtype=torch.float64, device=device)
    rank_sum_min_neg_const = torch.zeros(n_sig, dtype=torch.float64, device=device)
    miss_rank_pos = torch.zeros(n_sig, dtype=torch.float64, device=device)
    miss_rank_neg = torch.zeros(n_sig, dtype=torch.float64, device=device)

    for j, (pos_idx, pos_miss, neg_idx, neg_miss) in enumerate(resolved):
        if pos_idx.size > 0:
            M_pos[torch.as_tensor(pos_idx, dtype=torch.long, device=device), j] = 1.0
        if neg_idx.size > 0:
            M_neg[torch.as_tensor(neg_idx, dtype=torch.long, device=device), j] = 1.0
        n_p = int(pos_idx.size) + int(pos_miss)
        n_n = int(neg_idx.size) + int(neg_miss)
        len_pos[j] = n_p
        len_neg[j] = n_n
        rank_sum_min_pos_const[j] = n_p * (n_p + 1) / 2.0
        rank_sum_min_neg_const[j] = n_n * (n_n + 1) / 2.0
        miss_rank_pos[j] = float(pos_miss) * float(max_rank)
        miss_rank_neg[j] = float(neg_miss) * float(max_rank)

    # rank_sum_present[c, s] = Σ_g R[c, g] · M[g, s]  → (n_cells, n_sig)
    rs_pos = R @ M_pos + miss_rank_pos[None, :]
    rs_neg = R @ M_neg + miss_rank_neg[None, :]

    denom_pos = len_pos * float(max_rank) - rank_sum_min_pos_const
    denom_neg = len_neg * float(max_rank) - rank_sum_min_neg_const
    safe_pos = torch.where(denom_pos > 0, denom_pos, torch.ones_like(denom_pos))
    safe_neg = torch.where(denom_neg > 0, denom_neg, torch.ones_like(denom_neg))

    u_p = 1.0 - (rs_pos - rank_sum_min_pos_const[None, :]) / safe_pos[None, :]
    u_n = 1.0 - (rs_neg - rank_sum_min_neg_const[None, :]) / safe_neg[None, :]

    # Empty subsets contribute 0 (matches the CPU path)
    u_p = torch.where(len_pos[None, :] > 0, u_p, torch.zeros_like(u_p))
    u_n = torch.where(len_neg[None, :] > 0, u_n, torch.zeros_like(u_n))

    diff = u_p - float(w_neg) * u_n
    diff = torch.clamp(diff, min=0.0)
    return diff.cpu().numpy()


@register_function(
    aliases=[
        "ucell", "UCell", "u_cell_score",
        "andreatta_carmona_ucell", "mann_whitney_signature_score",
        "ov.es.ucell",
    ],
    category="es",
    description=(
        "Per-cell signature scoring via UCell (Mann-Whitney U on "
        "per-cell descending ranks, Andreatta & Carmona 2021). "
        "Bit-for-bit equivalent to the R `UCell::ScoreSignatures_UCell` "
        "function: handles `+/-` gene-name suffixes for "
        "up/down sub-signatures, `w_neg` weight on the down subset, "
        "`maxRank` truncation, and the `missing_genes='impute'/'skip'` "
        "policy. Returns a (cells × signatures) DataFrame in "
        "`adata.obsm['score_ucell']`."
    ),
    requires={"var": ["var_names (used as gene IDs)"]},
    produces={"obsm": ["score_ucell"]},
    auto_fix="none",
    examples=[
        "sigs = {'IFN_response': ['IFI6', 'ISG15', 'MX1']}",
        "ov.es.ucell(adata, signatures=sigs)",
        "adata.obsm['score_ucell']",
        "# Up- + down-regulated combined (gene-name suffix syntax):",
        "sigs = {'M1_polarisation': ['TNF', 'IL6', 'IL10-', 'TGFB1-']}",
        "ov.es.ucell(adata, signatures=sigs, w_neg=1.0)",
    ],
    related=["aucell", "gsea", "ora", "viper"],
)
@monitor
def ucell(
    data: Union[AnnData, "pd.DataFrame", np.ndarray],
    signatures: Optional[Mapping[str, Sequence[str]]] = None,
    *,
    max_rank: int = 1500,
    w_neg: float = 1.0,
    missing_genes: str = "impute",
    layer: Optional[str] = None,
    raw: bool = False,
    chunk_size: int = 100,
    verbose: bool = False,
    engine: str = "auto",
    key_added: str = "score_ucell",
    copy: bool = False,
) -> Optional[pd.DataFrame]:
    r"""Score per-cell signature activity using UCell.

    UCell ranks every feature per cell (descending expression) and
    derives a normalised Mann-Whitney U statistic against the signature
    genes — see module-level docstring for the precise formula. Output
    range: ``[0, 1]``, where 1 means signature genes are uniformly at
    the top of the ranking.

    Reference: `Andreatta & Carmona, *Comput Struct Biotechnol J* 19
    (2021) <https://doi.org/10.1016/j.csbj.2021.06.043>`_.

    Args:
        data: ``AnnData`` (or a ``DataFrame``/``ndarray`` shaped
            ``cells × genes``).
        signatures: ``{name → [gene, ...]}``. Append ``-`` to a gene
            name to make it a member of the *negative* subset of the
            signature (``UCell(S+) − w_neg·UCell(S−)``); append ``+``
            or no suffix for the positive subset.
        max_rank: Rank cutoff (R default ``1500``). Any rank greater
            than this is clamped to ``max_rank``; signatures longer
            than ``max_rank`` raise an error.
        w_neg: Weight on the negative subset (default ``1.0``).
        missing_genes: ``'impute'`` (default — missing genes contribute
            ``max_rank`` each to the rank sum) or ``'skip'`` (dropped
            from the signature, effectively shrinking ``|S|``).
        layer: AnnData layer name to score (default ``None`` = ``adata.X``).
        raw: Score ``adata.raw.X`` instead.
        chunk_size: Cells per processing chunk; controls peak memory
            on sparse inputs.
        verbose: Show per-chunk tqdm.
        key_added: ``adata.obsm`` key for the score table.
        copy: If ``True``, return a copy of the AnnData with scores
            written in. If ``False`` (default), writes in place and
            returns ``None``. For ``DataFrame``/``ndarray`` inputs,
            always returns the cells × signatures DataFrame.

    Returns:
        ``None`` (writes ``adata.obsm[key_added]``) or, for non-AnnData
        inputs, a ``(cells × signatures)`` DataFrame.

    Examples:
        >>> import omicverse as ov
        >>> sigs = {'IFN_response': ['IFI6', 'ISG15', 'MX1']}
        >>> ov.es.ucell(adata, signatures=sigs)
        >>> adata.obsm['score_ucell'].head()
        >>> # Up/down combined signature
        >>> sigs = {'M1_polarisation': ['TNF', 'IL6', 'IL10-', 'TGFB1-']}
        >>> ov.es.ucell(adata, signatures=sigs, w_neg=1.0)
    """
    if missing_genes not in ("impute", "skip"):
        raise ValueError(
            f"missing_genes must be 'impute' or 'skip'; got {missing_genes!r}"
        )
    if w_neg < 0:
        raise ValueError("w_neg must be >= 0")
    if not signatures:
        raise ValueError("signatures dict cannot be empty")

    # Pull (mat, var_names, obs_names) from the input
    if isinstance(data, AnnData):
        adata = data.copy() if copy else data
        if layer is not None:
            mat = adata.layers[layer]
        elif raw:
            if adata.raw is None:
                raise ValueError("raw=True but adata.raw is None")
            mat = adata.raw.X
            var_names = list(adata.raw.var_names)
            obs_names = list(adata.obs_names)
        else:
            mat = adata.X
        if layer is not None or not raw:
            var_names = list(adata.var_names)
            obs_names = list(adata.obs_names)
    elif isinstance(data, pd.DataFrame):
        adata = None
        mat = data.values
        var_names = list(data.columns)
        obs_names = list(data.index)
    else:
        mat = np.asarray(data)
        adata = None
        var_names = [f"gene_{i}" for i in range(mat.shape[1])]
        obs_names = [f"cell_{i}" for i in range(mat.shape[0])]

    n_obs, n_var = mat.shape
    max_rank = int(min(max_rank, n_var))

    var_to_idx = {g: i for i, g in enumerate(var_names)}

    # Resolve every signature's (+/− present indices, + missing counts) once
    sig_names = list(signatures.keys())
    resolved = []
    for name in sig_names:
        pos_genes, neg_genes = _split_signature(signatures[name])
        for sub, label in ((pos_genes, "+"), (neg_genes, "-")):
            if len(sub) > max_rank:
                raise ValueError(
                    f"signature {name!r} {label} subset has "
                    f"{len(sub)} genes > max_rank={max_rank}. "
                    "Increase max_rank or shorten the signature."
                )
        pos_idx, pos_miss = _signature_indices(var_to_idx, pos_genes, missing_genes)
        neg_idx, neg_miss = _signature_indices(var_to_idx, neg_genes, missing_genes)
        resolved.append((pos_idx, pos_miss, neg_idx, neg_miss))

    # Allocate output; iterate cells in chunks (per R chunk_size=100)
    n_sig = len(sig_names)
    scores = np.zeros((n_obs, n_sig), dtype=np.float64)
    is_sparse = sps.issparse(mat)

    # Engine dispatch: "cpu" → numpy loop (this file's _u_score),
    # "gpu" → CPU rank + GPU matmul-style U-score (see _ucell_gpu_score).
    # Both produce **bit-for-bit** identical output on fp64 — the GPU path
    # vectorises the per-signature U-score formula as one matmul.
    from ._engine import resolve_engine
    eng = resolve_engine(engine, has_torch_kernel=True)

    if eng == "gpu":
        # GPU path uses ov.utils.gpuex.scipy.rankdata for the avg-tie
        # ranking (batched, single searchsorted-based kernel) AND a
        # matmul for the per-signature U-score — neither step touches
        # the Python loop, so the speedup grows with both n_cells and
        # n_signatures.
        from ..utils.gpuex.scipy import rankdata as _gpu_rankdata
        for start in tqdm(range(0, n_obs, chunk_size), disable=not verbose):
            end = min(start + chunk_size, n_obs)
            chunk = mat[start:end].toarray() if is_sparse else np.asarray(mat[start:end])
            ranks = _gpu_rankdata(-chunk, axis=1, method="average")
            scores[start:end] = _ucell_gpu_score(
                ranks, resolved, max_rank=max_rank, w_neg=w_neg,
            )
    else:
        for start in tqdm(range(0, n_obs, chunk_size), disable=not verbose):
            end = min(start + chunk_size, n_obs)
            chunk = mat[start:end].toarray() if is_sparse else np.asarray(mat[start:end])
            ranks = sts.rankdata(-chunk, axis=1, method="average")
            for j, (pos_idx, pos_miss, neg_idx, neg_miss) in enumerate(resolved):
                n_p = pos_idx.size + pos_miss
                for r, row_ix in enumerate(range(start, end)):
                    u_p = _u_score(ranks[r], pos_idx, pos_miss, max_rank) if n_p else 0.0
                    if neg_idx.size + neg_miss > 0:
                        u_n = _u_score(ranks[r], neg_idx, neg_miss, max_rank)
                    else:
                        u_n = 0.0
                    s = u_p - w_neg * u_n
                    if s < 0:
                        s = 0.0
                    scores[row_ix, j] = s

    out = pd.DataFrame(scores, index=obs_names, columns=sig_names)
    if adata is not None:
        adata.obsm[key_added] = out
        return adata if copy else None
    return out
