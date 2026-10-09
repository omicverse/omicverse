"""R CellChat pipeline steps around ``computeCommunProb`` (OmicVerse port).

Each function ports the R function named in its docstring from
jinworks/CellChat 2.2.0 (presto 1.1.0 for the Wilcoxon test, igraph/sna for
centrality) and is checked numerically against R in
``tests/single/test_cellchat.py``. Communication tensors are
``(sender, receiver, LR)``, R's ``object@net$prob`` layout.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm, rankdata

from ._db import ANNOTATIONS, CellChatDB


def _complex_subunits(db: CellChatDB) -> Dict[str, List[str]]:
    return db.complex_subunits()


def sorted_interactions(db: CellChatDB) -> pd.DataFrame:
    """R ``subsetData``: order ``DB$interaction`` by the annotation factor (stable)."""
    inter = db.interaction
    if inter["annotation"].nunique() <= 1:
        return inter
    rank = {a: i for i, a in enumerate(ANNOTATIONS)}
    key = inter["annotation"].map(rank).fillna(len(ANNOTATIONS))
    return inter.loc[key.sort_values(kind="mergesort").index]


def extract_genes(db: CellChatDB) -> List[str]:
    """R ``extractGene``: every ligand/receptor subunit plus cofactor gene."""
    complexes = db.complex_subunits()
    inter = db.interaction

    def expand(values) -> List[str]:
        out: List[str] = []
        for v in pd.unique(values):
            if v:
                out += complexes.get(v, [v])
        return out

    genes = expand(inter["ligand"]) + expand(inter["receptor"])
    cof = pd.unique(pd.concat([inter[c] for c in ("agonist", "antagonist", "co_A_receptor", "co_I_receptor")
                               if c in inter.columns]))
    for name in cof:
        genes += db.cofactor_genes(name)
    return list(dict.fromkeys(genes))


def wilcoxauc(X, labels: Sequence[str], features: Sequence[str]) -> pd.DataFrame:
    """presto ``wilcoxauc``: one-vs-rest Wilcoxon rank-sum per group.

    ``X`` is cells x genes. Ranks use average ties over all cells; the
    p-value is the normal approximation with tie and continuity correction,
    exactly as ``presto:::compute_pval``.
    """
    labels = np.asarray(labels, dtype=object)
    groups = sorted(pd.unique(labels))  # factor(y): sorted levels
    codes = np.searchsorted(np.array(groups, dtype=object), labels)
    n_obs = len(labels)
    size = np.bincount(codes, minlength=len(groups)).astype(float)
    n1n2 = size * (n_obs - size)
    X = X.tocsc() if sp.issparse(X) else np.asarray(X)

    rows = []
    x1 = n_obs ** 3 - n_obs
    x2 = 1.0 / (12.0 * (n_obs ** 2 - n_obs))
    onehot = sp.csr_matrix((np.ones(n_obs), (codes, np.arange(n_obs))), shape=(len(groups), n_obs))
    for j, feat in enumerate(features):
        col = X[:, j].toarray().ravel() if sp.issparse(X) else X[:, j]
        ranks = rankdata(col, method="average")
        rank_sum = onehot @ ranks
        ustat = rank_sum - size * (size + 1) / 2.0
        _, tie_counts = np.unique(col, return_counts=True)
        rhs = (x1 - np.sum(tie_counts.astype(float) ** 3 - tie_counts)) * x2
        z = ustat - 0.5 * n1n2
        z = z - np.sign(z) * 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            z = z / np.sqrt(n1n2 * rhs)
        z[~np.isfinite(z)] = 0.0
        pval = 2.0 * norm.sf(np.abs(z))
        sums = onehot @ col
        nnz = onehot @ (col > 0).astype(float)
        mean_in = sums / size
        mean_out = (sums.sum() - sums) / (n_obs - size)
        rows.append(pd.DataFrame({
            "features": feat, "clusters": groups, "pvalues": pval,
            "logFC": mean_in - mean_out,
            "pct.1": 100.0 * nnz / size,
            "pct.2": 100.0 * (nnz.sum() - nnz) / (n_obs - size),
        }))
    return pd.concat(rows, ignore_index=True)


def identify_over_expressed_genes(X, labels, features, *, thresh_p: float = 0.05,
                                  thresh_fc: float = 0.0, thresh_pc: float = 0.0,
                                  only_pos: bool = True) -> pd.DataFrame:
    """R ``identifyOverExpressedGenes(do.fast = TRUE)`` on ``data.signaling``."""
    de = wilcoxauc(X, labels, features)
    pct_max = de[["pct.1", "pct.2"]].max(axis=1)
    keep = (de["pvalues"] < thresh_p) & (de["logFC"].abs() >= thresh_fc) & (pct_max > thresh_pc * 100)
    markers = de[keep].sort_values("pvalues", kind="mergesort")
    if only_pos and len(markers):
        markers = markers[markers["logFC"] > 0]
    return markers.reset_index(drop=True)


def identify_over_expressed_interactions(db: CellChatDB, features_sig: Iterable[str],
                                         gene_use: Iterable[str], *,
                                         variable_both: bool = True) -> pd.DataFrame:
    """R ``identifyOverExpressedInteractions`` -> ``LRsig`` (rows of DB$interaction)."""
    features_sig, gene_use = set(features_sig), set(gene_use)
    complexes = _complex_subunits(db)
    complex_sig = {n for n, sub in complexes.items() if (set(sub) & features_sig) and set(sub) <= gene_use}
    complex_use = {n for n, sub in complexes.items() if set(sub) <= gene_use}
    inter = sorted_interactions(db)
    sig_ok, use_ok = features_sig | complex_sig, gene_use | complex_use
    if variable_both:
        mask = inter["ligand"].astype(str).isin(sig_ok) & inter["receptor"].astype(str).isin(sig_ok)
    else:
        both_use = inter["ligand"].astype(str).isin(use_ok) & inter["receptor"].astype(str).isin(use_ok)
        any_sig = inter["ligand"].astype(str).isin(sig_ok) | inter["receptor"].astype(str).isin(sig_ok)
        mask = both_use & any_sig
    return inter[mask.values]


def filter_communication(prob: np.ndarray, group_sizes: Sequence[int], min_cells: int = 10) -> np.ndarray:
    """R ``filterCommunication``: zero every pair touching a group with <= min_cells cells."""
    prob = prob.copy()
    exclude = np.where(np.asarray(group_sizes) <= min_cells)[0]
    prob[exclude, :, :] = 0
    prob[:, exclude, :] = 0
    return prob


def compute_commun_prob_pathway(prob: np.ndarray, pval: np.ndarray, pathway_names: Sequence[str],
                                thresh: float = 0.05):
    """R ``computeCommunProbPathway`` -> (pathways sorted by total, (S, R, P) tensor, LR mask)."""
    prob = np.where(pval > thresh, 0.0, prob)
    names = np.asarray(pathway_names, dtype=object)
    pathways = list(pd.unique(names))
    tensor = np.stack([prob[:, :, names == pw].sum(axis=2) for pw in pathways], axis=2)
    totals = tensor.sum(axis=(0, 1))
    keep = [i for i in range(len(pathways)) if totals[i] != 0]
    order = sorted(keep, key=lambda i: -totals[i])  # R sort(decreasing=TRUE) is stable
    lr_sig = prob.sum(axis=(0, 1)) != 0
    return [pathways[i] for i in order], tensor[:, :, order], lr_sig


def aggregate_net(prob: np.ndarray, pval: np.ndarray, thresh: float = 0.05):
    """R ``aggregateNet`` -> (count, weight) sender x receiver matrices."""
    pval = np.where(prob == 0, 1.0, pval)
    prob = np.where(pval >= thresh, 0.0, prob)
    return (prob > 0).sum(axis=2).astype(float), prob.sum(axis=2)


def subset_communication(prob: np.ndarray, pval: np.ndarray, groups: Sequence[str],
                         lrsig: pd.DataFrame, thresh: float = 0.05) -> pd.DataFrame:
    """R ``subsetCommunication(slot.name = 'net')`` -> long ``df.net`` table."""
    s, r, k = np.where((prob > 0) & (pval <= thresh))
    cols = [c for c in ("interaction_name", "interaction_name_2", "pathway_name", "ligand",
                        "receptor", "annotation", "evidence") if c in lrsig.columns]
    meta = lrsig.iloc[k][cols].reset_index(drop=True)
    df = pd.DataFrame({"source": np.asarray(groups)[s], "target": np.asarray(groups)[r],
                       "prob": prob[s, r, k], "pval": pval[s, r, k]})
    df = pd.concat([df, meta], axis=1)
    return df.sort_values(["interaction_name", "source", "target"], kind="mergesort").reset_index(drop=True) \
        if "interaction_name" in df.columns else df


def _flowbet(net: np.ndarray) -> np.ndarray:
    """sna ``flowbet(cmode = 'rawflow')``: max-flow betweenness on a weighted digraph."""
    import networkx as nx

    n = net.shape[0]

    def maxflow(adj, i, j):
        g = nx.DiGraph()
        g.add_nodes_from(range(adj.shape[0]))
        for a, b in zip(*np.nonzero(adj)):
            if a != b:
                g.add_edge(int(a), int(b), capacity=float(adj[a, b]))
        return nx.maximum_flow_value(g, i, j) if i in g and j in g else 0.0

    full = {(i, j): maxflow(net, i, j) for i in range(n) for j in range(n) if i != j}
    out = np.zeros(n)
    for v in range(n):
        reduced = net.copy()
        reduced[v, :] = 0
        reduced[:, v] = 0
        for i in range(n):
            for j in range(n):
                if len({i, j, v}) == 3:
                    out[v] += full[(i, j)] - maxflow(reduced, i, j)
    return out


def _infocent(net: np.ndarray) -> np.ndarray:
    """sna ``infocent(diag = TRUE, rescale = TRUE, cmode = 'lower')``."""
    m = np.tril(net) + np.tril(net, -1).T  # symmetrize(rule = 'lower')
    iso = (m.sum(axis=1) == 0) & (m.sum(axis=0) == 0)
    keep = np.where(~iso)[0]
    ic = np.zeros(net.shape[0])
    if len(keep) < 2:
        # R: m[ix, ix] drops to a scalar, diag() then errors and CellChat's
        # tryCatch returns zeros.
        return ic
    m = m[np.ix_(keep, keep)]
    A = 1.0 - m
    A[np.diag_indices_from(A)] = 1.0 + m.sum(axis=1)
    try:
        cn = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return ic
    tr, rs = np.trace(cn), cn.sum(axis=1)
    # sna divides by the node count *before* isolates are dropped
    ic[keep] = 1.0 / (np.diag(cn) + (tr - 2.0 * rs) / net.shape[0])
    total = ic.sum()
    return ic / total if total else ic


def compute_centrality_local(net: np.ndarray) -> Dict[str, np.ndarray]:
    """R ``computeCentralityLocal`` (igraph + sna) on one weighted sender x receiver net."""
    import igraph as ig

    n = net.shape[0]
    edges = [(int(a), int(b)) for a, b in zip(*np.nonzero(net))]
    weights = [float(net[a, b]) for a, b in edges]
    g = ig.Graph(n=n, edges=edges, directed=True)
    g.es["weight"] = weights
    out = {
        "outdeg_unweighted": (net > 0).sum(axis=1).astype(float),
        "indeg_unweighted": (net > 0).sum(axis=0).astype(float),
        "outdeg": np.asarray(g.strength(mode="out", weights="weight" if edges else None), float),
        "indeg": np.asarray(g.strength(mode="in", weights="weight" if edges else None), float),
    }
    if edges:
        out["hub"] = np.asarray(g.hub_score(weights="weight"), float)
        out["authority"] = np.asarray(g.authority_score(weights="weight"), float)
        # R igraph::eigen_centrality defaults to directed = FALSE: every edge,
        # kept as is (no merging), treated as undirected.
        gu = ig.Graph(n=n, edges=edges, directed=False)
        out["eigen"] = np.asarray(gu.eigenvector_centrality(weights=weights), float)
        out["page_rank"] = np.asarray(g.pagerank(weights="weight"), float)
        g.es["weight"] = [1.0 / w for w in weights]
        out["betweenness"] = np.asarray(g.betweenness(weights="weight"), float)
    else:
        for key in ("hub", "authority", "eigen", "page_rank", "betweenness"):
            out[key] = np.zeros(n)
    try:
        out["flowbet"] = _flowbet(net)
    except Exception:  # R: tryCatch(..., error = zeros)
        out["flowbet"] = np.zeros(n)
    try:
        out["info"] = _infocent(net)
    except Exception:
        out["info"] = np.zeros(n)
    return out


def net_analysis_compute_centrality(netp_prob: np.ndarray, pathways: Sequence[str],
                                    groups: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """R ``netAnalysis_computeCentrality(slot.name = 'netP')`` -> {pathway: groups x measures}."""
    result = {}
    for i, pw in enumerate(pathways):
        centr = compute_centrality_local(np.asarray(netp_prob[:, :, i], float))
        result[pw] = pd.DataFrame(centr, index=list(groups))
    return result


def identify_communication_patterns(netp_prob: np.ndarray, pathways: Sequence[str],
                                    groups: Sequence[str], k: int,
                                    pattern: str = "outgoing", random_state: int = 0):
    """R ``identifyCommunicationPatterns``: NMF of the pathway signaling matrix.

    R uses ``NMF::nmf(method = 'lee', seed = 'nndsvd')``; this uses scikit-learn's
    multiplicative-update Frobenius NMF with NNDSVD init, which is the same model
    but not bit-identical. Returns (cell x pattern W, pattern x pathway H), with
    W row-normalised and H column-normalised to sum 1 as R's ``scaleMat``.
    """
    from sklearn.decomposition import NMF

    if pattern not in ("outgoing", "incoming"):
        raise ValueError("pattern must be 'outgoing' or 'incoming'.")
    data = netp_prob.sum(axis=1) if pattern == "outgoing" else netp_prob.sum(axis=0)  # groups x pathways
    colmax = data.max(axis=0)
    colmax[colmax == 0] = 1.0
    data = data / colmax
    keep = data.sum(axis=1) != 0
    data, kept_groups = data[keep], np.asarray(groups)[keep]
    model = NMF(n_components=k, init="nndsvd", solver="mu", beta_loss="frobenius",
                max_iter=2000, random_state=random_state)
    W = model.fit_transform(data)
    H = model.components_
    W = W / np.where(W.sum(axis=1, keepdims=True) == 0, 1, W.sum(axis=1, keepdims=True))
    H = H / np.where(H.sum(axis=0, keepdims=True) == 0, 1, H.sum(axis=0, keepdims=True))
    names = [f"Pattern {i + 1}" for i in range(k)]
    return (pd.DataFrame(W, index=kept_groups, columns=names),
            pd.DataFrame(H, index=names, columns=list(pathways)))
