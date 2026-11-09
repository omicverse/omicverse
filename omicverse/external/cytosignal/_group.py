"""Cell-group (sender -> receiver) summary of cellular-resolution CytoSignal scores.

CytoSignal tests every interaction *at each receiving cell* ``i``: the score
``S[i, k]`` combines the ligand that reaches ``i`` from its spatial neighbours
(Gaussian epsilon ball for diffusible, Delaunay neighbours for contact-dependent
ligands) with the receptor expressed at ``i``. The R package stops there; to
report source -> target communication, we attribute each receiver's score to
the neighbours that delivered the ligand:

* **Sender share.** The imputed ligand count at ``i`` is
  ``sum_j W[j, i] * Lc[j, k]`` (``W`` = the ligand-imputation graph,
  ``Lc`` = summed raw counts of the ligand subunits). Sender group ``s``
  contributes ``F_s[i, k] = sum_{j in s} W[j, i] Lc[j, k] / sum_j W[j, i] Lc[j, k]``.
* **Score.** ``score(s, t, k) = mean_{i in t} S[i, k] * F_s[i, k]`` -- the
  average CytoSignal score received by ``t`` cells that originates from ``s``
  cells. Summed over senders it equals the mean score of the receivers.
* **Significance.** The same statistic restricted to cells where CytoSignal
  itself calls the interaction significant (spatial-FDR ``padj < p_thresh``,
  high-quality interactions only), ``T(s, t, k)``, is compared with ``n_perms``
  random relabellings of the receiving cells (sender shares fixed):
  ``p = (1 + #{T_perm >= T}) / (1 + n_perms)``; ``p = 1`` when ``T = 0``.
  This asks whether ``t`` cells receive more significant ``s``-derived
  ``k`` signal than random cells of the tissue.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def ligand_counts(counts: sp.csr_matrix, lig_groups: list[list[int]]) -> np.ndarray:
    """Summed raw counts of each interaction's ligand subunits (cells x I)."""
    a = sp.csr_matrix((np.ones(sum(len(g) for g in lig_groups)),
                       (np.concatenate([np.asarray(g, dtype=np.int64) for g in lig_groups]) if lig_groups else [],
                        np.repeat(np.arange(len(lig_groups)), [len(g) for g in lig_groups]))),
                      shape=(counts.shape[1], len(lig_groups)))
    return np.asarray((counts @ a).todense())


def group_summary(w: sp.spmatrix, lc: np.ndarray, score: np.ndarray, sig: np.ndarray, codes: np.ndarray,
                  n_groups: int, *, n_perms: int = 1000, seed: int = 0, use_gpu: bool = False, batch: int = 64):
    """Group-level score and receiver-label permutation p-values.

    ``w`` is the ligand-imputation graph (column = receiver), ``lc`` the ligand
    counts (cells x I), ``score`` the CytoSignal score and ``sig`` the boolean
    cell-level significance (cells x I). Returns ``(score, pvalue)`` arrays of
    shape ``(senders, receivers, I)``.
    """
    w = sp.csr_matrix(w)
    n, n_intr = score.shape
    sizes = np.bincount(codes, minlength=n_groups).astype(np.float64)
    sizes_safe = np.where(sizes > 0, sizes, 1.0)
    onehot = np.zeros((n_groups, n))
    onehot[codes, np.arange(n)] = 1.0
    total = np.asarray(w.T @ lc)
    tested = np.where(sig.any(axis=0))[0]
    full = np.zeros((n_groups, n_groups, n_intr))
    parts = []
    for s in range(n_groups):
        rows = np.where(codes == s)[0]
        if rows.size == 0:
            parts.append(np.zeros((n, tested.size)))
            continue
        part = np.asarray(w[rows].T @ lc[rows])
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(total > 0, part / total, 0.0) * score   # S[i,k] * F_s[i,k]
        full[s] = onehot @ q / sizes_safe[:, None]
        parts.append(q[:, tested] * sig[:, tested])
    pval = np.ones((n_groups, n_groups, n_intr))
    if tested.size == 0 or n_perms <= 0:
        return full, pval
    m = np.ascontiguousarray(np.concatenate(parts, axis=1))  # (N, S*I'), sender-major

    if use_gpu:
        import torch

        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float64 if dev != "mps" else torch.float32
        mt = torch.as_tensor(m, device=dev, dtype=dtype)
        sz = torch.as_tensor(sizes_safe, device=dev, dtype=dtype)

        def agg(oh):
            o = torch.as_tensor(oh, device=dev, dtype=dtype)
            return (o @ mt / sz.repeat(oh.shape[0] // n_groups)[:, None]).cpu().numpy()
    else:
        def agg(oh):
            return oh @ m / np.tile(sizes_safe, oh.shape[0] // n_groups)[:, None]

    obs = agg(onehot)                                 # (T, S*I')
    exceed = np.zeros(obs.shape, dtype=np.int64)
    rng = np.random.default_rng(seed)
    cols = np.arange(n)
    done = 0
    while done < n_perms:
        b = min(batch, n_perms - done)
        oh = np.zeros((b * n_groups, n))
        for r in range(b):
            oh[r * n_groups + rng.permutation(codes), cols] = 1.0
        exceed += (agg(oh).reshape(b, n_groups, -1) >= obs[None]).sum(axis=0)
        done += b
    p = (1.0 + exceed) / (1.0 + n_perms)
    p[obs <= 0] = 1.0
    p = p.reshape(n_groups, n_groups, tested.size).transpose(1, 0, 2)  # (T, S, I') -> (S, T, I')
    pval[:, :, tested] = p
    return full, pval


def long_table(score, pval, groups, meta: pd.DataFrame, *, keep_groups=None, pvalue_threshold=0.05):
    """Flatten ``(S, T, I)`` arrays into the run_ccc long table (rows with score > 0)."""
    s, t, k = np.where(score > 0)
    if keep_groups is not None:
        ok = np.isin(s, keep_groups) & np.isin(t, keep_groups)
        s, t, k = s[ok], t[ok], k[ok]
    g = np.asarray(groups, dtype=object)
    m = meta.iloc[k]
    df = pd.DataFrame({
        "source": g[s], "target": g[t],
        "ligand": m["ligand"].to_numpy(), "receptor": m["receptor"].to_numpy(),
        "ligand_genes": m["ligand_genes"].to_numpy(), "receptor_genes": m["receptor_genes"].to_numpy(),
        "interaction": m["interaction"].to_numpy(), "pathway": m["pathway"].to_numpy(),
        "annotation": m["annotation"].to_numpy(),
        "score": score[s, t, k], "pvalue": pval[s, t, k],
    })
    df["significant"] = df["pvalue"] < pvalue_threshold
    return df.reset_index(drop=True)
