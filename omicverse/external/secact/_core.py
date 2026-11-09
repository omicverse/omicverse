"""SecAct pipelines: activity inference and secreted-protein cell-cell communication.

Ports of R SecAct (Ru et al., Nat Methods 2026)::

    SecAct.activity.inference   -> activity_inference
    SecAct.CCC.scRNAseq         -> ccc_scrnaseq
    SecAct.activity.inference.ST + SecAct.CCC.scST -> spot_activity + ccc_spatial

``compat="R"`` (default) follows the R package; ``compat="secactpy"`` follows
the Python port (SecActpy, GitHub main 13bb6d2), which differs in three places:
library sizes for the step-1 expression test are summed over secreted-protein
genes only, gene symbols are upper-cased with ``.N`` suffixes stripped, and
permutations use glibc ``rand()`` (``rng_method="srand"``) instead of GSL
MT19937.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from ._ridge import ridge_permutation
from ._signature import fetch_resource, group_signatures, load_signature

# ---------------------------------------------------------------------------
# small statistics helpers (R semantics)
# ---------------------------------------------------------------------------


def p_adjust_bh(p) -> np.ndarray:
    """R ``p.adjust(p, "BH")``; NaN entries stay NaN and are not counted."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    n = q.size
    if n:
        order = np.argsort(q)[::-1]  # decreasing, as R
        ranked = np.minimum.accumulate(n / np.arange(n, 0, -1) * q[order])
        res = np.empty(n)
        res[order] = np.minimum(ranked, 1.0)
        out[ok] = res
    return out


def fisher_combine(p1, p2) -> np.ndarray:
    """``metap::sumlog`` for two p-values per row (Fisher's method, 4 d.o.f.)."""
    chi = -2.0 * (np.log(p1) + np.log(p2))
    return stats.chi2.sf(chi, 4)


def _wilcox_from_ranks(rank_sum, n1, n2, tie_term):
    """Two-sided normal-approximation Wilcoxon rank-sum p (continuity corrected)."""
    z = rank_sum - n1 * (n1 + 1) / 2.0 - n1 * n2 / 2.0
    n = n1 + n2
    sigma = np.sqrt((n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))))
    with np.errstate(divide="ignore", invalid="ignore"):
        zc = (z - np.sign(z) * 0.5) / sigma
    return 2.0 * np.minimum(stats.norm.cdf(zc), stats.norm.sf(zc))


def _row_ranks(M: np.ndarray):
    """Average ranks per row plus ``sum(t^3 - t)`` over tie groups and a has-ties flag."""
    ranks = stats.rankdata(M, axis=1, method="average")
    tie_term = np.zeros(M.shape[0])
    has_ties = np.zeros(M.shape[0], dtype=bool)
    for g in range(M.shape[0]):
        _, counts = np.unique(M[g], return_counts=True)
        t = counts[counts > 1].astype(np.float64)
        tie_term[g] = np.sum(t ** 3 - t)
        has_ties[g] = t.size > 0
    return ranks, tie_term, has_ties


def _wilcox_r(case: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    """R ``wilcox.test(x, y)$p.value`` row-wise (exact if n<50 and no ties)."""
    n1, n2 = case.shape[1], ctrl.shape[1]
    ranks, tie_term, has_ties = _row_ranks(np.hstack([case, ctrl]))
    pv = _wilcox_from_ranks(ranks[:, :n1].sum(axis=1), n1, n2, tie_term)
    if n1 < 50 and n2 < 50:
        for g in np.where(~has_ties)[0]:
            pv[g] = stats.mannwhitneyu(case[g], ctrl[g], alternative="two-sided", method="exact").pvalue
    return pv


# ---------------------------------------------------------------------------
# expression preprocessing
# ---------------------------------------------------------------------------


def _alias_to_symbol(genes: np.ndarray, cache_dir=None) -> np.ndarray:
    """R ``transferSymbol``: NCBI alias -> official symbol (first match wins, as ``match``).

    The table is parsed the way R's ``read.csv`` reads it: a line with more than
    two fields wraps onto the next row (e.g. ``NERF-1a,b,ELF2``).
    """
    import gzip

    with gzip.open(fetch_resource("alias2symbol", cache_dir), "rt") as handle:
        handle.readline()
        fields = []
        for line in handle:
            parts = line.rstrip("\n").split(",")
            parts += [""] * (-len(parts) % 2)
            fields += parts
    first: dict[str, str] = {}
    for alias, symbol in zip(fields[0::2], fields[1::2]):
        first.setdefault(alias, symbol)
    return np.asarray([first.get(g, g) for g in genes], dtype=object)


def mouse_to_human(counts: sp.csr_matrix, genes: np.ndarray, cache_dir=None):
    """R ``mouse2human_mat``: sum mouse genes onto their human orthologues."""
    m2h = pd.read_csv(fetch_resource("mouse2human", cache_dir), index_col=0)
    lookup = dict(zip(m2h["mouse"].astype(str), m2h["human"].astype(str)))
    human = np.asarray([lookup.get(g) for g in genes], dtype=object)
    keep = np.asarray([h is not None for h in human])
    if not keep.any():
        raise ValueError("No mouse genes matched the SecAct mouse-to-human table.")
    levels, codes = np.unique(human[keep].astype(str), return_inverse=True)
    agg = sp.csr_matrix((np.ones(keep.sum()), (np.where(keep)[0], codes)), shape=(len(genes), len(levels)))
    return (sp.csr_matrix(counts) @ agg).tocsr(), levels.astype(object)


def prepare_counts(counts, genes, *, compat: str = "R", species: str = "human", map_alias: bool = True,
                   cache_dir=None):
    """Cells x genes counts -> (CSR float64, gene symbols) after SecAct's symbol clean-up.

    R: ``transferSymbol`` (``map_alias``, on by default) then ``rm_duplicates`` (keep the
    copy with the largest total count). SecActpy: upper-case, strip ``.N``
    suffixes, keep the copy with the largest mean.
    """
    X = sp.csr_matrix(counts, dtype=np.float64)
    genes = np.asarray(genes, dtype=object).astype(str).astype(object)
    if str(species).lower() == "mouse":
        X, genes = mouse_to_human(X, genes, cache_dir)
    elif str(species).lower() != "human":
        raise ValueError("SecAct signatures are human; species must be 'human' or 'mouse'.")
    if compat == "secactpy":
        genes = np.asarray([g.upper().split(".")[0] for g in genes], dtype=object)
    elif map_alias:
        genes = _alias_to_symbol(genes, cache_dir)
    if len(set(genes)) != len(genes):
        if compat == "secactpy":  # SecActpy ranks copies by scipy's sparse mean; its rounding breaks ties
            totals = np.asarray(X.T.mean(axis=1)).ravel()
        else:  # R rm_duplicates: largest rowSums, first copy on ties
            totals = np.asarray(X.sum(axis=0)).ravel()
        best: dict[str, int] = {}
        for i, g in enumerate(genes):
            if g not in best or totals[i] > totals[best[g]]:
                best[g] = i
        keep = np.sort(np.fromiter(best.values(), dtype=np.int64))
        X, genes = X[:, keep], genes[keep]
    return X.tocsr(), genes


def normalize_log(X: sp.csr_matrix, scale_factor: float, libsize=None) -> sp.csr_matrix:
    """R ``normalize_log_sparse``: ``log2(x / libsize * scale_factor + 1)`` per cell."""
    X = sp.csr_matrix(X, dtype=np.float64, copy=True)
    lib = np.asarray(X.sum(axis=1)).ravel() if libsize is None else np.asarray(libsize, dtype=np.float64)
    lib = np.where(lib == 0, 1.0, lib)
    X.data = np.log2(X.data / np.repeat(lib, np.diff(X.indptr)) * scale_factor + 1.0)
    return X


# ---------------------------------------------------------------------------
# activity inference
# ---------------------------------------------------------------------------


def _expand(frame: pd.DataFrame) -> pd.DataFrame:
    """R ``expand_rows`` + ``sort(rownames)``: one row per protein of each group."""
    rows, names = [], []
    for idx, values in zip(frame.index, frame.to_numpy()):
        for name in str(idx).split("|"):
            rows.append(values)
            names.append(name)
    out = pd.DataFrame(np.vstack(rows), index=names, columns=frame.columns)
    return out.loc[sorted(out.index)]


def activity_inference(Y: pd.DataFrame, *, sig_matrix="SecAct", is_group_sig: bool | None = None,
                       is_group_cor: float = 0.9, lambda_: float | None = None, nrand: int = 1000,
                       is_filter_sig: bool = False, rng_method: str = "mt19937", seed: int = 0,
                       use_gpu: bool = False, cache_dir=None, progress: bool = False) -> dict:
    """R ``SecAct.activity.inference`` on an already-differential profile ``Y`` (genes x samples).

    Returns ``{"beta", "se", "zscore", "pvalue"}`` DataFrames (proteins x samples).
    """
    X, default_lambda, default_group = load_signature(sig_matrix, cache_dir)
    lam = default_lambda if lambda_ is None else float(lambda_)
    group = default_group if is_group_sig is None else bool(is_group_sig)
    Y = Y.copy()
    Y.index = Y.index.astype(str)
    olp = Y.index[Y.index.isin(X.index)]
    olp = olp[~olp.duplicated()]
    if len(olp) < 20:
        raise ValueError("Fewer than 20 genes overlap the SecAct signature; check gene symbols / species.")
    if is_filter_sig:
        X = X.loc[:, X.columns.isin(Y.index)]
    if group:
        X, _ = group_signatures(X, is_group_cor)
    Xs = X.loc[olp].to_numpy(dtype=np.float64)
    Ys = Y.loc[olp].to_numpy(dtype=np.float64)

    def _scale(M):  # R scale(): centre, divide by sd (n - 1); constant columns -> 0
        M = M - M.mean(axis=0)
        sd = M.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            M = np.where(sd > 0, M / sd, 0.0)
        return M

    res = ridge_permutation(_scale(Xs), _scale(Ys), lam, int(nrand), rng_method=rng_method, seed=seed,
                            use_gpu=use_gpu, progress=progress)
    out = {k: pd.DataFrame(v, index=X.columns, columns=Y.columns) for k, v in res.items()}
    if group:
        out = {k: _expand(v) for k, v in out.items()}
    return out


# ---------------------------------------------------------------------------
# scRNA-seq CCC
# ---------------------------------------------------------------------------


def _signature_proteins(sig_matrix, cache_dir) -> list[str]:
    X, _, _ = load_signature(sig_matrix, cache_dir)
    return list(X.columns)


def ccc_scrnaseq(counts, genes, cell_types, *, conditions=None, condition_case=None, condition_control=None,
                 scale_factor: float = 1e5, act_diff_cutoff: float = 2.0, exp_logfc_cutoff: float = 0.2,
                 exp_mean_all_cutoff: float = 2.0, exp_fraction_case_cutoff: float = 0.1,
                 padj_cutoff: float = 0.01, min_cells: int = 30, sig_matrix="SecAct",
                 is_group_sig: bool | None = True, is_group_cor: float = 0.9, lambda_: float | None = 5e5,
                 nrand: int = 1000, rng_method: str | None = None, seed: int = 0, compat: str = "R",
                 species: str = "human", map_alias: bool = True, use_gpu: bool = False, cache_dir=None,
                 progress: bool = False) -> dict:
    """R ``SecAct.CCC.scRNAseq`` on a cells x genes count matrix.

    Returns ``expression`` ({cell type: DataFrame}), ``activity`` (``beta/se/zscore/pvalue``),
    ``candidates`` (every sender -> protein -> receiver edge that passed the step-1/2
    filters, with ``overall_pv.adj``) and ``ccc`` (``overall_pv.adj < padj_cutoff``, R output).
    """
    if compat not in ("R", "secactpy"):
        raise ValueError("compat must be 'R' or 'secactpy'.")
    if rng_method is None:
        rng_method = "srand" if compat == "secactpy" else "mt19937"
    X, genes = prepare_counts(counts, genes, compat=compat, species=species, map_alias=map_alias,
                              cache_dir=cache_dir)
    ct = np.asarray(cell_types).astype(str)
    cond = None if conditions is None else np.asarray(conditions).astype(str)
    if cond is not None and (condition_case is None or condition_control is None):
        raise ValueError("condition_case and condition_control are required with conditions.")

    if cond is None:
        cell_type_list = list(pd.unique(ct))
    else:
        case_set = set(ct[cond == str(condition_case)])
        cell_type_list = [c for c in pd.unique(ct[cond == str(condition_case)])
                          if c in set(ct[cond == str(condition_control)]) and c in case_set]

    def idx(c):
        if cond is None:
            return np.where(ct == c)[0], np.where(ct != c)[0]
        return (np.where((cond == str(condition_case)) & (ct == c))[0],
                np.where((cond == str(condition_control)) & (ct == c))[0])

    proteins = _signature_proteins(sig_matrix, cache_dir)
    gpos = {g: i for i, g in enumerate(genes)}
    if compat == "secactpy":
        sp_genes = [p.upper() for p in proteins if p.upper() in gpos]
    else:
        protein_set = set(proteins)
        sp_genes = [g for g in genes if g in protein_set]  # R: intersect(rownames, SPs)
    sp_idx = np.asarray([gpos[g] for g in sp_genes], dtype=np.int64)
    if sp_idx.size == 0:
        raise ValueError("No secreted-protein genes of the signature are present in the data.")

    # Step 1: secreted-protein expression change per cell type ----------------
    X_sp = X[:, sp_idx]
    libsize = np.asarray(X_sp.sum(axis=1)).ravel() if compat == "secactpy" else np.asarray(X.sum(axis=1)).ravel()
    E = normalize_log(X_sp, scale_factor, libsize)  # per-cell, so identical inside case and control

    global_ranks = None
    if cond is None and compat == "R":
        dense = E.T.toarray()
        global_ranks = _row_ranks(dense)

    expression: dict[str, pd.DataFrame] = {}
    for c in cell_type_list:
        case, ctrl = idx(c)
        if case.size < min_cells:
            continue
        cm = E[case].T.toarray()
        km = E[ctrl].T.toarray()
        mean_case, mean_ctrl = cm.mean(axis=1), km.mean(axis=1)
        n1, n2 = cm.shape[1], km.shape[1]
        if compat == "secactpy":
            with np.errstate(invalid="ignore"):
                pv = stats.mannwhitneyu(cm, km, axis=1, alternative="two-sided").pvalue
        elif global_ranks is not None and not (n1 < 50 and n2 < 50):
            ranks, tie_term, _ = global_ranks
            pv = _wilcox_from_ranks(ranks[:, case].sum(axis=1), n1, n2, tie_term)
        else:
            pv = _wilcox_r(cm, km)
        df = pd.DataFrame({
            "exp_logFC": mean_case - mean_ctrl,
            "exp_mean_all": (mean_case * n1 + mean_ctrl * n2) / (n1 + n2),
            "exp_mean_case": mean_case,
            "exp_mean_control": mean_ctrl,
            "exp_fraction_case": (cm > 0).mean(axis=1),
            "exp_fraction_control": (km > 0).mean(axis=1),
            "exp_pv": pv,
        }, index=sp_genes)
        df = df[df["exp_mean_all"] > 0].copy()
        df["exp_pv.adj"] = p_adjust_bh(df["exp_pv"].to_numpy())
        expression[c] = df.sort_values("exp_pv.adj", kind="mergesort")
    kept = list(expression)
    if len(kept) < 2:
        raise ValueError(f"Fewer than two cell types have >= {min_cells} cells; cannot infer CCC.")

    # Step 2: activity change per cell type (pseudo-bulk log2 TPM difference) --
    def log2tpm(v):
        total = v.sum()
        return np.log2(v * 1e6 / (total if total else 1.0) + 1.0)

    diff = {}
    for c in kept:
        case, ctrl = idx(c)
        a = np.asarray(X[case].sum(axis=0)).ravel()
        b = np.asarray(X[ctrl].sum(axis=0)).ravel()
        diff[c] = log2tpm(a) - log2tpm(b)
    bulk_diff = pd.DataFrame(diff, index=genes.astype(str))
    activity = activity_inference(bulk_diff, sig_matrix=sig_matrix, is_group_sig=is_group_sig,
                                  is_group_cor=is_group_cor, lambda_=lambda_, nrand=nrand,
                                  rng_method=rng_method, seed=seed, use_gpu=use_gpu, cache_dir=cache_dir,
                                  progress=progress)
    z, p = activity["zscore"], activity["pvalue"]

    # Step 3: link up-expressed senders to up-active receivers -----------------
    senders = []
    for c in kept:
        df = expression[c]
        up = df[(df["exp_logFC"] > exp_logfc_cutoff) & (df["exp_mean_all"] > exp_mean_all_cutoff)
                & (df["exp_fraction_case"] > exp_fraction_case_cutoff) & (df["exp_pv.adj"] < padj_cutoff)]
        senders += list(zip([c] * len(up), up.index, up["exp_logFC"], up["exp_pv"], up["exp_pv.adj"]))
    receivers: dict[str, list] = {}
    for c in kept:
        padj = p_adjust_bh(p[c].to_numpy())
        zc = z[c].to_numpy()
        for k in np.where((zc > act_diff_cutoff) & (padj < padj_cutoff))[0]:
            receivers.setdefault(str(z.index[k]).upper(), []).append(
                (c, float(zc[k]), float(p[c].iat[k]), float(padj[k])))
    rows = []
    for s, g, lfc, epv, epadj in senders:
        for r, zd, apv, apadj in receivers.get(str(g).upper(), []):
            if r != s:
                rows.append((s, g, r, lfc, epv, epadj, zd, apv, apadj))
    cols = ["sender", "secretedProtein", "receiver", "sender_exp_logFC", "sender_exp_pv", "sender_exp_pv.adj",
            "receiver_act_diff", "receiver_act_pv", "receiver_act_pv.adj"]
    cand = pd.DataFrame(rows, columns=cols)
    if len(cand):
        cand["overall_strength"] = cand["sender_exp_logFC"] * cand["receiver_act_diff"]
        tiny = np.finfo(float).tiny  # R: .Machine$double.xmin for zero sender p-values
        if compat == "R":  # R overwrites the column; SecActpy only floors inside the combination
            cand.loc[cand["sender_exp_pv"] == 0, "sender_exp_pv"] = tiny
        cand["overall_pv"] = fisher_combine(np.maximum(cand["sender_exp_pv"].to_numpy(), tiny),
                                            np.maximum(cand["receiver_act_pv"].to_numpy(), tiny))
        cand["overall_pv.adj"] = p_adjust_bh(cand["overall_pv"].to_numpy())
    else:
        cand = pd.DataFrame(columns=cols + ["overall_strength", "overall_pv", "overall_pv.adj"])
    ccc = cand[cand["overall_pv.adj"] < padj_cutoff].sort_values("overall_pv.adj", kind="mergesort")
    return {"expression": expression, "activity": activity, "candidates": cand.reset_index(drop=True),
            "ccc": ccc.reset_index(drop=True), "genes": genes, "rng_method": rng_method}


# ---------------------------------------------------------------------------
# spatial (single-cell resolution) CCC
# ---------------------------------------------------------------------------


def spot_activity(counts, genes, *, scale_factor: float = 1e5, sig_matrix="SecAct",
                  is_group_sig: bool | None = True, is_group_cor: float = 0.9, lambda_: float | None = 5e5,
                  nrand: int = 1000, rng_method: str = "mt19937", seed: int = 0, species: str = "human",
                  map_alias: bool = True, use_gpu: bool = False, cache_dir=None, progress: bool = False) -> dict:
    """R ``SecAct.activity.inference.ST``: per-cell activity vs the mean cell (cells x genes counts)."""
    X = sp.csr_matrix(counts, dtype=np.float64)
    keep = np.asarray(X.sum(axis=0)).ravel() > 0
    X, genes = prepare_counts(X[:, keep], np.asarray(genes)[keep], species=species, map_alias=map_alias,
                              cache_dir=cache_dir)
    E = normalize_log(X, scale_factor).toarray().T  # genes x cells
    diff = pd.DataFrame(E - E.mean(axis=1, keepdims=True), index=genes.astype(str))
    return activity_inference(diff, sig_matrix=sig_matrix, is_group_sig=is_group_sig, is_group_cor=is_group_cor,
                              lambda_=lambda_, nrand=nrand, rng_method=rng_method, seed=seed, use_gpu=use_gpu,
                              cache_dir=cache_dir, progress=progress)


def scrnaseq_activity(counts, genes, cell_types=None, *, sig_matrix="SecAct", is_group_sig: bool | None = True,
                      is_group_cor: float = 0.9, lambda_: float | None = 5e5, nrand: int = 1000,
                      rng_method: str = "mt19937", seed: int = 0, species: str = "human", map_alias: bool = True,
                      use_gpu: bool = False, cache_dir=None, progress: bool = False) -> dict:
    """R ``SecAct.activity.inference.scRNAseq``.

    With ``cell_types``: pseudo-bulk per cell type (sorted labels), log2 TPM, centred
    on the mean cell type. Without: every cell (``is.singleCellLevel=TRUE``),
    ``log2(count / libsize * 1e5 + 1)`` centred on the mean cell.
    """
    X, genes = prepare_counts(counts, genes, species=species, map_alias=map_alias, cache_dir=cache_dir)
    if cell_types is not None:
        ct = np.asarray(cell_types).astype(str)
        labels = sorted(pd.unique(ct))
        bulk = np.column_stack([np.asarray(X[ct == c].sum(axis=0)).ravel() for c in labels])
        expr = np.log2(bulk / bulk.sum(axis=0) * 1e6 + 1.0)
        columns = labels
    else:
        expr = normalize_log(X, 1e5).toarray().T
        columns = None
    diff = pd.DataFrame(expr - expr.mean(axis=1, keepdims=True), index=genes.astype(str), columns=columns)
    return activity_inference(diff, sig_matrix=sig_matrix, is_group_sig=is_group_sig, is_group_cor=is_group_cor,
                              lambda_=lambda_, nrand=nrand, rng_method=rng_method, seed=seed, use_gpu=use_gpu,
                              cache_dir=cache_dir, progress=progress)


def find_neighbors(coords: np.ndarray, radius: float, k: int = 100):
    """R ``find_neighbors``: up to ``k`` nearest cells within ``radius`` (self excluded)."""
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    kk = min(int(k), len(coords))
    dist, nn = cKDTree(coords).query(coords, k=kk, distance_upper_bound=radius)
    dist, nn = dist.reshape(len(coords), -1), nn.reshape(len(coords), -1)
    i = np.repeat(np.arange(len(coords)), dist.shape[1])
    j, x = nn.ravel(), dist.ravel()
    valid = (x <= radius) & (x > 0)
    return i[valid], j[valid], x[valid]


def _spearman_r(a: np.ndarray, b: np.ndarray):
    """R ``cor.test(a, b, method="spearman")`` (asymptotic t; SecAct data always has ties)."""
    n = a.size
    ra, rb = stats.rankdata(a), stats.rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(ra, rb)[0, 1])
    if abs(r) >= 1.0:
        return r, 0.0
    t = r / np.sqrt((1 - r * r) / (n - 2))
    return r, float(min(2.0 * stats.t.sf(abs(t), n - 2), 1.0))


def _pair_stats(exp_sp, act_sp, src, dst, bg_src, bg_dst, ratio_cutoff, n_nb):
    vec = exp_sp[src] * act_sp[dst]
    n_comm = int((vec > 0).sum())
    ratio = n_comm / n_nb
    if not ratio > ratio_cutoff:
        return None
    raw = vec.mean()
    null = (exp_sp[bg_src] * act_sp[bg_dst]).reshape(-1, n_nb).mean(axis=1)
    return n_comm, ratio, raw, (np.sum(null >= raw) + 1.0) / (null.size + 1.0)


def ccc_spatial(counts, genes, coords, cell_types, activity_z: pd.DataFrame, *, scale_factor: float = 1000.0,
                radius: float = 20.0, k: int = 100, ratio_cutoff: float = 0.2, padj_cutoff: float = 0.01,
                n_background: int = 1000, seed: int = 123, species: str = "human", map_alias: bool = True,
                cache_dir=None, progress: bool = False) -> dict:
    """R ``SecAct.CCC.scST``: neighbouring sender expression x receiver activity.

    ``activity_z`` is the per-cell activity z-score (proteins x cells, columns in
    the order of ``counts`` rows). Returns ``correlation`` (per-protein spatial
    Spearman test), ``candidates`` (all tested edges) and ``ccc`` (``pv.adj < padj_cutoff``).
    """
    from ._rng import r_sample_replace

    X, genes = prepare_counts(counts, genes, species=species, map_alias=map_alias, cache_dir=cache_dir)
    E = normalize_log(X, scale_factor).tocsc()
    gpos = {g: i for i, g in enumerate(genes)}
    act = activity_z.to_numpy(dtype=np.float64).copy()
    act[act < 0] = 0.0
    proteins = [str(p) for p in activity_z.index]
    ct = np.asarray(cell_types).astype(str)
    n = len(ct)

    i, j, _ = find_neighbors(coords, radius, k)
    W = sp.csr_matrix((np.ones(i.size), (i, j)), shape=(n, n))
    W.data = (W.data > 0).astype(np.float64)

    # Step 1: keep proteins whose activity follows neighbouring expression.
    rs, ps = np.full(len(proteins), np.nan), np.full(len(proteins), np.nan)
    for a, prot in enumerate(proteins):
        if prot in gpos:
            e = E[:, gpos[prot]].toarray().ravel()
            rs[a], ps[a] = _spearman_r(act[a], W.T @ e)  # exp %*% weights: sum over neighbours
    corr = pd.DataFrame({"r": rs, "p": ps, "padj": p_adjust_bh(ps)}, index=proteins)
    keep = corr.index[(corr["r"].fillna(-np.inf) > 0.05) & (corr["padj"].fillna(np.inf) < 0.01)]

    # Step 2: per cell-type pair, observed mean vs background of random cross-type pairs.
    types = list(pd.unique(ct))
    groups = {c: np.where(ct == c)[0] for c in types}
    pairs = [(a, b) for a in types for b in types if a > b]
    member = {c: np.isin(np.arange(n), g) for c, g in groups.items()}
    rows = []
    iterator = pairs
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(pairs, desc="SecAct spatial pairs")
    exp_rows = {p: E[:, gpos[p]].toarray().ravel() for p in keep}
    act_rows = {p: act[proteins.index(p)] for p in keep}
    for c1, c2 in iterator:
        sel = member[c1][i] & member[c2][j]
        src, dst = i[sel], j[sel]
        n_nb = src.size
        if (np.unique(src).size / groups[c1].size < 0.05) and (np.unique(dst).size / groups[c2].size < 0.05):
            continue
        bg1, bg2 = r_sample_replace([groups[c1], groups[c2]], n_nb * int(n_background), seed)
        for prot in keep:
            e, a_ = exp_rows[prot], act_rows[prot]
            for (s, r, s_cells, r_cells, bs, bd) in ((c1, c2, src, dst, bg1, bg2), (c2, c1, dst, src, bg2, bg1)):
                res = _pair_stats(e, a_, s_cells, r_cells, bs, bd, ratio_cutoff, n_nb)
                if res is not None:
                    rows.append((s, prot, r, groups[s].size, groups[r].size, n_nb, res[0], res[1], res[2], res[3]))
    cols = ["sender", "secretedProtein", "receiver", "sender_count", "receiver_count", "neighboringCellPairs",
            "communicatingCellPairs", "ratio", "strength", "pv"]
    cand = pd.DataFrame(rows, columns=cols)
    cand["pv.adj"] = p_adjust_bh(cand["pv"].to_numpy()) if len(cand) else []
    ccc = cand[cand["pv.adj"] < padj_cutoff].sort_values("pv.adj", kind="mergesort")
    return {"correlation": corr, "candidates": cand, "ccc": ccc.reset_index(drop=True)}
