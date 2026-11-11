# Vendored from `decoupler` (https://github.com/scverse/decoupler) by
# omicverse for in-tree GPU acceleration work. Original copyright by
# the decoupler authors, redistributed under decoupler's GPL-3.0
# license. Cross-module imports rewritten from `decoupler.*` to
# `omicverse.es.*` (see scripts/vendor_decoupler.py).
"""AUCell — per-cell pathway activity via area under the recovery curve."""

from __future__ import annotations

import numba as nb
import numpy as np
import scipy.sparse as sps
from tqdm.auto import tqdm

from .._monitor import monitor
from .._registry import register_function

from ._net import _getset, _resolve_net
from ._run import _run


# ───────────────────────── numba CPU kernel ─────────────────────────

@nb.njit(parallel=True, cache=True)
def _auc(
    neg: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    n_up: int,
    nsrc: int,
) -> np.ndarray:
    nobs, nvar = neg.shape
    max_auc = np.zeros(nsrc)
    for j in range(nsrc):
        fset = _getset(cnct, starts, offsets, j)
        x_th = np.arange(1, stop=fset.shape[0] + 1)
        x_th = x_th[x_th < n_up]
        max_auc[j] = np.sum(np.diff(np.append(x_th, n_up)) * x_th)
    es = np.zeros((nobs, nsrc))
    for i in nb.prange(nobs):
        row = neg[i]
        # Ordinal rank of the top n_up features (ties keep gene order); 0 marks
        # features outside the top n_up. Rows with NaN rank as all-NaN in
        # scipy, so none of their features is recovered.
        rank = np.zeros(nvar, dtype=np.int64)
        if not np.any(row != row):
            # Select the n_up-th smallest value, stably sort the features below
            # it and fill the remaining ranks with its ties in gene order.
            kth = np.partition(row, n_up - 1)[n_up - 1]
            top = np.nonzero(row < kth)[0]
            top = top[np.argsort(row[top], kind="mergesort")]
            r = top.shape[0]
            rank[top] = np.arange(1, r + 1)
            for g in range(nvar):
                if r == n_up:
                    break
                if row[g] == kth:
                    r += 1
                    rank[g] = r
        for j in range(nsrc):
            # Area under the step recovery curve: sum_k k * (x_{k+1} - x_k)
            # with x_{m+1} = n_up telescopes to m * n_up - sum(x).
            m = 0
            total = 0
            for g in _getset(cnct, starts, offsets, j):
                if rank[g] > 0:
                    m += 1
                    total += rank[g]
            es[i, j] = (m * n_up - total) / max_auc[j]
    return es


def _validate_n_up(
    nvar: int,
    n_up: int | float | None = None,
) -> int:
    assert isinstance(n_up, int | float) or n_up is None, "n_up must be numerical or None"
    if n_up is None:
        n_up = np.ceil(0.05 * nvar)
        n_up = int(np.clip(n_up, a_min=2, a_max=nvar))
    else:
        n_up = int(np.ceil(n_up))
    assert nvar >= n_up > 1, f"For nvar={nvar}, n_up={n_up} must be between 1 and {nvar}"
    return n_up


def _func_aucell(
    mat: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    n_up: int | float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, None]:
    """Numba CPU kernel — see :func:`aucell` for the public API."""
    nobs, nvar = mat.shape
    nsrc = starts.size
    n_up = _validate_n_up(nvar, n_up)
    es = np.zeros(shape=(nobs, nsrc))
    # Rank cells in dense blocks of about 4M values to bound memory on CSR input.
    bsize = max(1, 2**22 // nvar)
    for b0 in tqdm(range(0, nobs, bsize), disable=not verbose):
        block = mat[b0 : b0 + bsize]
        block = block.toarray() if isinstance(mat, sps.csr_matrix) else np.asarray(block)
        es[b0 : b0 + bsize] = _auc(neg=-block, cnct=cnct, starts=starts, offsets=offsets, n_up=n_up, nsrc=nsrc)
    return es, None


# ───────────────────────── torch GPU kernel ─────────────────────────

def _func_aucell_torch(
    mat,
    cnct,
    starts,
    offsets,
    n_up=None,
    verbose: bool = False,
):
    """Torch GPU port of :func:`_func_aucell` — batched recovery-curve cumsum.

    Replaces the per-signature sort + Python loop with one ``argsort + gather +
    cumsum`` pass over the gene→signature membership matrix. Bit-for-bit
    equivalent to the CPU kernel on fp64.
    """
    import torch
    from ._engine import torch_device, chunk_size_for, to_gpu_dense

    device = torch_device()
    nobs, nvar = mat.shape
    nsrc = starts.size
    n_up = _validate_n_up(nvar, n_up)
    n_up_int = int(n_up)

    M = to_gpu_dense(mat, device, dtype=torch.float64)

    sort_idx = torch.argsort(-M, dim=1, stable=True)
    top_idx = sort_idx[:, :n_up_int].contiguous()

    membership = torch.zeros(nvar, nsrc, dtype=torch.float32, device=device)
    cnct_t = torch.as_tensor(cnct, dtype=torch.long, device=device)
    sig_id_per_target = torch.as_tensor(
        np.repeat(np.arange(nsrc, dtype=np.int64), offsets),
        device=device,
    )
    membership[cnct_t, sig_id_per_target] = 1.0

    # Per-signature scalar max_auc (closed form, matches the numba kernel).
    sig_sizes = offsets.astype(np.int64)
    max_aucs = np.where(
        sig_sizes < n_up_int,
        sig_sizes * (sig_sizes - 1) / 2.0 + sig_sizes * (n_up_int - sig_sizes),
        (n_up_int - 1) * n_up_int / 2.0,
    ).astype(np.float64)
    max_aucs_t = torch.as_tensor(max_aucs, dtype=torch.float64, device=device)
    safe_max = torch.where(max_aucs_t == 0, torch.ones_like(max_aucs_t), max_aucs_t)

    cells_per_batch = chunk_size_for(n_up_int * nsrc, max_units=nobs)

    es = torch.zeros(nobs, nsrc, dtype=torch.float64, device=device)
    for b0 in range(0, nobs, cells_per_batch):
        b1 = min(b0 + cells_per_batch, nobs)
        m_top = membership[top_idx[b0:b1]]
        cs = m_top.cumsum(dim=1)
        total = cs.sum(dim=1)
        k_valid = m_top.sum(dim=1)
        es[b0:b1] = (total - k_valid).to(torch.float64) / safe_max

    zero_cols = (max_aucs_t == 0).nonzero(as_tuple=False).squeeze(-1)
    if zero_cols.numel() > 0:
        es[:, zero_cols] = 0.0

    return es.cpu().numpy(), None


_func_aucell_torch._accepts_sparse = True


# ───────────────────────── public API ─────────────────────────

@monitor
@register_function(
    aliases=["aucell", "AUCell", "通路活性", "基因集活性", "geneset_activity"],
    category="enrichment",
    description=(
        "Score per-cell pathway activity by integrating the recovery curve "
        "of signature genes in each cell's expression ranking. Robust to "
        "library-size variation, GPU-accelerated via the torch kernel."
    ),
    prerequisites={"optional_functions": ["preprocess"]},
    requires={"var": ["gene symbols matching signature keys"]},
    produces={"obsm": ["score_aucell"]},
    auto_fix="none",
    examples=[
        "ov.es.aucell(adata, signatures={'IFN_response': ifn_genes})",
        "ov.es.aucell(adata, signatures=pathway_dict, engine='gpu', tmin=3)",
    ],
    related=["gsea", "gsva", "ora", "decoupler"],
)
def aucell(
    data,
    signatures=None,
    *,
    net=None,
    tmin: int | float = 5,
    raw: bool = False,
    empty: bool = True,
    bsize: int | float = 250_000,
    verbose: bool = False,
    engine: str = "auto",
    n_up: int | float | None = None,
):
    r"""Score per-cell pathway activity using AUCell.

    AUCell ranks features per cell and computes the area under the recovery
    curve — how early signature genes appear in the ranking. The enrichment
    score for cell :math:`i` and signature :math:`F` is

    .. math::

        ES_{i,F} = \int_0^1 R_{i,F}(r)\, dr

    where :math:`R_{i,F}(r)` is the proportion of :math:`F` recovered in the
    top-:math:`r` fraction of cell :math:`i`'s expression ranking.

    Reference: `Aibar et al., Nat Methods (2017)
    <https://doi.org/10.1038/nmeth.4463>`_.

    Args:
        data: AnnData (or DataFrame) containing the expression matrix.
        signatures: Mapping ``{name → [gene, ...]}`` (binary) or
            ``{name → {gene: weight}}`` (weighted/signed). Mutually exclusive
            with ``net``.
        net: Long-format DataFrame with ``source / target / weight`` columns
            (decoupler convention). Power-user escape hatch; ``signatures`` is
            the default entry point.
        tmin: Minimum number of targets per signature; sets below this are
            silently dropped. Default 5.
        raw: Score against ``adata.raw.X`` instead of ``adata.X``. Default False.
        empty: Whether to write all-zero rows for signatures filtered out by
            ``tmin``. Default True.
        bsize: Cells per processing chunk (controls peak memory for sparse
            inputs). Default 250 000.
        verbose: Show per-cell tqdm progress bars. Default False.
        engine: ``"auto"`` (default) picks GPU when available, ``"cpu"`` forces
            the numba kernel, ``"gpu"`` forces the torch kernel.
        n_up: Number of top-ranked features to include in the AUC integration.
            ``None`` (default) picks the top 5 % of features.

    Returns:
        None. Writes ``adata.obsm['score_aucell']`` as a
        ``(cells × signatures)`` DataFrame of AUC scores.

    Examples:
        >>> import omicverse as ov
        >>> adata = ov.datasets.pancreatic_endocrinogenesis()
        >>> ov.pp.preprocess(adata, target_sum=1e4)
        >>> sigs = {'response_to_vitamin': ['Vdr', 'Cyp27a1', 'Cyp2r1']}
        >>> ov.es.aucell(adata, signatures=sigs, tmin=3)
        >>> adata.obsm['score_aucell'].head()
    """
    from ._engine import resolve_engine

    eng = resolve_engine(engine, has_torch_kernel=True)
    func = _func_aucell_torch if eng == "gpu" else _func_aucell
    resolved_net = _resolve_net(signatures, net)
    return _run(
        name="aucell",
        func=func,
        adj=False,
        test=False,
        data=data,
        net=resolved_net,
        tmin=tmin,
        raw=raw,
        empty=empty,
        bsize=bsize,
        verbose=verbose,
        n_up=n_up,
    )
