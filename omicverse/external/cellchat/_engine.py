"""PyTorch implementation of R CellChat ``computeCommunProb`` (sc/snRNA-seq mode).

For every LR pair ``i`` and sender/receiver groups ``(s, r)``::

    avg      = triMean of data.signaling / max(data.signaling) per group
    L, R     = ligand / receptor expression (complex = geometric mean of subunits)
    R        = R * prod(1 + co_A_receptor) / prod(1 + co_I_receptor)
    P1       = Hill(L_s * R_r)                    Hill(x) = x^n / (Kh^n + x^n)
    P2       = ag_s * ag_r,  ag = prod(1 + Hill(agonist genes))
    P3       = an_s * an_r,  an = prod(Kh^n / (Kh^n + agonist genes^n))
    P4       = f_s * f_r (population.size only), f = group size / n cells
    prob     = P1 * P2 * P3 * P4
    pval     = #{permutations with prob_perm > prob} / n_perm;  pval[prob == 0] = 1

triMean is mean(q25, q50, q50, q75) with type-7 quantiles (``collapse::fquantile``).
All groups of one permutation are summarised by two stable sorts per gene block
(by value, then by group label), so each group is a contiguous,
value-sorted block of every gene column.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import scipy.sparse as sp


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ov.single.run_cellchat requires PyTorch (CPU build is enough): pip install torch") from exc
    return torch


def resolve_device(use_gpu: bool):
    torch = _torch()
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


@dataclass
class LRDesign:
    """Gene-index layout of the LR pairs over the compact gene axis ``genes``."""

    genes: List[str]
    lig: np.ndarray          # (nLR, max_subunits) gene index, -1 = padding
    rec: np.ndarray
    co_a: np.ndarray         # (nLR, max_cofactors) gene index, -1 = padding
    co_i: np.ndarray
    agonist: np.ndarray
    antagonist: np.ndarray


def _pad(rows: Sequence[Sequence[int]]) -> np.ndarray:
    width = max([len(r) for r in rows] + [1])
    out = np.full((len(rows), width), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def build_design(lrsig, db, available: Sequence[str]) -> LRDesign:
    """Map each LR row to gene indices exactly as R ``computeExpr_*`` resolve them.

    Ligand/receptor: a gene present in the data is used as is, otherwise the
    name is a complex whose subunits are averaged geometrically. Cofactors
    (co-receptors, agonist, antagonist) keep only the genes present in the data.
    """
    available = set(available)
    complexes = db.complex_subunits()
    genes: List[str] = []
    index = {}

    def idx(gene: str) -> int:
        if gene not in index:
            index[gene] = len(genes)
            genes.append(gene)
        return index[gene]

    def lr_part(name: str) -> List[int]:
        if name in available:
            return [idx(name)]
        subunits = complexes.get(name)
        if not subunits:
            raise KeyError(f"'{name}' is neither an expressed gene nor a CellChatDB complex.")
        missing = [g for g in subunits if g not in available]
        if missing:
            raise KeyError(f"Complex '{name}' has subunits absent from the data: {missing}.")
        return [idx(g) for g in subunits]

    def cofactor(name: str) -> List[int]:
        return [idx(g) for g in dict.fromkeys(db.cofactor_genes(name)) if g in available]

    col = lambda c: lrsig[c].to_numpy() if c in lrsig.columns else np.repeat("", len(lrsig))
    return LRDesign(
        genes=genes,
        lig=_pad([lr_part(x) for x in col("ligand")]),
        rec=_pad([lr_part(x) for x in col("receptor")]),
        co_a=_pad([cofactor(x) for x in col("co_A_receptor")]),
        co_i=_pad([cofactor(x) for x in col("co_I_receptor")]),
        agonist=_pad([cofactor(x) for x in col("agonist")]),
        antagonist=_pad([cofactor(x) for x in col("antagonist")]),
    )


class _GroupTriMean:
    """Group-wise triMean of a fixed (cells x genes) matrix for many labelings."""

    def __init__(self, data, n_groups: int, device, dtype, gene_chunk: int):
        torch = _torch()
        self.torch, self.n_groups = torch, n_groups
        self.data, self.device, self.dtype, self.gene_chunk = data, device, dtype, gene_chunk

    def __call__(self, codes: np.ndarray):
        torch = self.torch
        sizes = np.bincount(codes, minlength=self.n_groups)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        # type-7 quantile positions h = (n - 1) p inside each group block
        probs = np.array([0.25, 0.5, 0.75])
        h = (np.maximum(sizes, 1)[:, None] - 1) * probs[None, :]
        lo = np.floor(h).astype(np.int64)
        hi = np.minimum(lo + 1, np.maximum(sizes, 1)[:, None] - 1)
        frac = torch.as_tensor(h - lo, dtype=self.dtype, device=self.device)
        lo = torch.as_tensor(starts[:, None] + lo, device=self.device)
        hi = torch.as_tensor(starts[:, None] + hi, device=self.device)
        empty = torch.as_tensor(sizes == 0, device=self.device)
        codes_t = torch.as_tensor(codes, dtype=torch.int64, device=self.device)

        n_genes = self.data.shape[1]
        out = torch.empty((n_genes, self.n_groups), dtype=self.dtype, device=self.device)
        for g0 in range(0, n_genes, self.gene_chunk):
            block = self.data[:, g0:g0 + self.gene_chunk]
            # sort by value, then stable-sort by group: each group becomes a
            # contiguous value-sorted block per gene column (values untouched).
            values, order = torch.sort(block, dim=0)
            _, by_group = torch.sort(codes_t[order], dim=0, stable=True)
            ordered = torch.gather(values, 0, by_group)
            ql = ordered[lo.reshape(-1)].reshape(self.n_groups, 3, -1)
            qh = ordered[hi.reshape(-1)].reshape(self.n_groups, 3, -1)
            q = ql + frac[:, :, None] * (qh - ql)
            tri = (q[:, 0] + 2.0 * q[:, 1] + q[:, 2]) / 4.0          # mean(q25, q50, q50, q75)
            tri[empty] = 0.0
            out[g0:g0 + block.shape[1]] = tri.T
        return out


def _gather(avg, index: np.ndarray):
    """(nLR, width, K) expression for a padded index matrix; padding -> NaN."""
    torch = _torch()
    idx = torch.as_tensor(index, device=avg.device)
    vals = avg[idx.clamp(min=0)]
    return torch.where((idx >= 0)[:, :, None], vals, torch.full_like(vals, float("nan")))


def _prob_tensor(avg, design_t, kh: float, n: float, pop):
    """(nLR, K, K) communication probability from group averages ``avg`` (G x K)."""
    torch = _torch()

    def geo_mean(index):  # R geometricMean: exp(mean(log(x)))
        vals = _gather(avg, index)
        logs = torch.log(vals)
        valid = ~torch.isnan(vals)
        logs = torch.where(valid, logs, torch.zeros_like(logs))
        return torch.exp(logs.sum(1) / valid.sum(1).clamp(min=1))

    def prod_of(index, fn):
        vals = _gather(avg, index)
        f = fn(vals)
        return torch.where(torch.isnan(vals), torch.ones_like(f), f).prod(1)

    hill = lambda x: x ** n / (kh ** n + x ** n)
    lig = geo_mean(design_t["lig"])
    rec = geo_mean(design_t["rec"])
    rec = rec * prod_of(design_t["co_a"], lambda x: 1.0 + x) / prod_of(design_t["co_i"], lambda x: 1.0 + x)
    p = hill(lig[:, :, None] * rec[:, None, :])
    ag = prod_of(design_t["agonist"], lambda x: 1.0 + hill(x))
    an = prod_of(design_t["antagonist"], lambda x: kh ** n / (kh ** n + x ** n))
    p = p * (ag[:, :, None] * ag[:, None, :]) * (an[:, :, None] * an[:, None, :])
    if pop is not None:
        p = p * (pop[:, None] * pop[None, :])[None]
    return p


def compute_commun_prob(X, codes: np.ndarray, n_groups: int, design: LRDesign, gene_names: Sequence[str],
                        scale: float, *, n_perms: int = 100, seed: int = 1, kh: float = 0.5, n: float = 1.0,
                        population_size: bool = False, use_gpu: bool = False, dtype: str = "float64",
                        gene_chunk: int = 256, progress: bool = False):
    """Return ``(prob, pval)`` as ``(sender, receiver, LR)`` numpy arrays.

    ``X`` is cells x genes (dense or sparse) with columns ``gene_names``;
    ``scale`` is ``max(data.signaling)``; ``codes`` are integer group labels.
    """
    torch = _torch()
    device = resolve_device(use_gpu)
    tdtype = torch.float64 if dtype == "float64" and device.type != "mps" else torch.float32
    col = {g: i for i, g in enumerate(gene_names)}
    cols = [col[g] for g in design.genes]
    sub = X[:, cols]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    data = torch.as_tensor(sub / scale, dtype=tdtype, device=device)

    trimean = _GroupTriMean(data, n_groups, device, tdtype, gene_chunk)
    design_t = {k: getattr(design, k) for k in ("lig", "rec", "co_a", "co_i", "agonist", "antagonist")}
    n_cells = len(codes)

    def pop_weights(c):
        if not population_size:
            return None
        return torch.as_tensor(np.bincount(c, minlength=n_groups) / n_cells, dtype=tdtype, device=device)

    prob = _prob_tensor(trimean(codes), design_t, kh, n, pop_weights(codes))
    exceed = torch.zeros_like(prob)
    rng = np.random.default_rng(seed)
    iterator = range(n_perms)
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, desc="CellChat permutations")
    for _ in iterator:
        perm = codes[rng.permutation(n_cells)]
        exceed += (_prob_tensor(trimean(perm), design_t, kh, n, pop_weights(perm)) > prob)
    pval = (exceed / max(n_perms, 1)).cpu().numpy()
    prob = prob.cpu().numpy()
    pval[prob == 0] = 1.0
    return np.moveaxis(prob, 0, -1).astype(np.float64), np.moveaxis(pval, 0, -1).astype(np.float64)
