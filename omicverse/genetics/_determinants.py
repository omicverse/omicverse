"""Genotype -> binary-trait determinant ranking and related-trait transfer.

Pure numpy / pandas / scikit-learn; no external backend. Written for
presence/absence (or 0/1/2 dosage) feature matrices such as bacterial
pan-genome gene tables, accessory-gene / k-mer unitig matrices or any
samples x features table, with a binary trait per sample (resistant /
susceptible, case / control).

Two failure modes of a marginal (one-feature-at-a-time) association scan
are addressed:

* **Linkage / hitch-hiking.** Features carried on the same mobile element
  or haplotype as a causal feature are associated with the trait without
  acting on it. A *conditional* (multivariable, penalised) fit over all
  features attributes the shared signal to the feature that explains it.
* **Effect modification (epistasis).** Some features act only on a
  permissive background (a second locus). A main-effects fit then credits
  the background locus with the signal and under-scores the features that
  do the work. :func:`effect_modifier_scan` finds the background locus by a
  likelihood-ratio screen over candidate modifiers, and
  :func:`trait_determinants` scores every feature by its effect in the
  stratum where it acts (``max(|b|, |b + g|)``), while the modifier itself
  is scored only by its own main effect.

:func:`transfer_trait_predict` predicts an unlabelled trait (e.g. a drug
with no phenotypes) from a genotype model pooled over labelled traits that
belong to the same group (e.g. the same drug class), and reports a
leave-one-trait-out estimate of how well such transfer works inside the
group.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import warnings

import numpy as np
import pandas as pd

from .._registry import register_function

__all__ = ["effect_modifier_scan", "trait_determinants", "transfer_trait_predict"]


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _as_matrix(X, feature_names=None):
    if isinstance(X, pd.DataFrame):
        names = [str(c) for c in X.columns]
        index = X.index
        M = X.to_numpy(dtype=float)
    else:
        M = np.asarray(X, dtype=float)
        names = (list(map(str, feature_names)) if feature_names is not None
                 else [f"f{j}" for j in range(M.shape[1])])
        index = None
    if M.ndim != 2:
        raise ValueError("X must be 2-D (samples x features)")
    if len(names) != M.shape[1]:
        raise ValueError("feature_names length does not match X")
    return M, names, index


def _labelled(M, y, index=None):
    """Align ``y`` to the rows of ``M`` and drop unlabelled (NaN) rows."""
    if isinstance(y, pd.Series) and index is not None:
        y = y.reindex(index)
    ys = pd.Series(np.asarray(y, dtype=object).ravel())
    ok = ys.notna().to_numpy()
    if len(ys) != M.shape[0]:
        raise ValueError("y length does not match the rows of X")
    return M[ok], _as_binary(ys[ok]), ok


def _as_binary(y, index=None) -> np.ndarray:
    if isinstance(y, pd.Series) and index is not None:
        y = y.reindex(index)
    y = pd.Series(np.asarray(y).ravel())
    if y.dtype == object:
        try:
            y = pd.to_numeric(y)
        except (ValueError, TypeError):
            pass
    if y.dtype == object or str(y.dtype).startswith(("string", "category")):
        u = y.astype(str).str.strip().str.upper()
        pos = u.isin(["R", "1", "TRUE", "YES", "CASE", "RESISTANT", "POS", "POSITIVE"])
        neg = u.isin(["S", "0", "FALSE", "NO", "CONTROL", "SUSCEPTIBLE", "NEG", "NEGATIVE"])
        if not (pos | neg).all():
            bad = sorted(set(u[~(pos | neg)]))[:5]
            raise ValueError(f"cannot read trait values as binary: {bad}")
        return pos.to_numpy().astype(int)
    v = y.to_numpy(dtype=float)
    if not np.isin(v, [0.0, 1.0]).all():
        raise ValueError("trait must be binary (0/1, True/False, R/S, case/control)")
    return v.astype(int)


def _logit(C, max_iter):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=max_iter)


def _loglik(model, D, y) -> float:
    p = np.clip(model.predict_proba(D)[:, 1], 1e-12, 1 - 1e-12)
    return float(y @ np.log(p) + (1 - y) @ np.log(1 - p))


def _shortlist(X, y, k, C, max_iter):
    main = _logit(C, max_iter).fit(X, y)
    coef = main.coef_[0]
    S = np.argsort(-np.abs(coef))[: min(k, X.shape[1])]
    return main, coef, S


# --------------------------------------------------------------------------- #
# effect-modifier screen                                                       #
# --------------------------------------------------------------------------- #
@register_function(
    aliases=[
        "effect_modifier_scan", "epistasis_scan", "modifier_locus_scan",
        "interaction_screen", "background_locus_scan",
        "效应修饰筛查", "上位性筛查", "背景位点筛查",
    ],
    category="genetics",
    description=(
        "Screen every feature of a samples x features genotype matrix "
        "(gene presence/absence, k-mers, SNP dosage) as a candidate EFFECT "
        "MODIFIER of a binary trait: for each candidate m (both strata "
        "populated) fit the top-k conditional features plus m plus their "
        "interactions with m, and compare to the main-effects fit with a "
        "likelihood-ratio test. The top row is the background / permissive "
        "locus on which other determinants act (epistasis). Use before "
        "ranking determinants when a main-effects fit may be crediting a "
        "background locus with other genes' effects."
    ),
    examples=[
        "scan = ov.genetics.effect_modifier_scan(genomes_df, y)",
        "scan.head()  # modifier, lr_stat, p_value, n_with, n_without",
    ],
    related=["ov.genetics.trait_determinants", "ov.genetics.gwas_association"],
)
def effect_modifier_scan(
    X: Union[pd.DataFrame, np.ndarray],
    y,
    *,
    feature_names: Optional[Sequence[str]] = None,
    shortlist: int = 30,
    min_stratum: float = 0.05,
    candidates: Optional[Sequence[str]] = None,
    C: float = 1.0,
    max_iter: int = 400,
) -> pd.DataFrame:
    """Likelihood-ratio screen for the locus that modifies other features' effects.

    Parameters
    ----------
    X
        Samples x features matrix (DataFrame or array). Binary presence /
        absence is the intended input; dosages work too.
    y
        Binary trait per sample (0/1, bool, ``'R'``/``'S'`` ...). If a Series
        and ``X`` is a DataFrame, it is aligned on the index. Samples with a
        missing label (NaN) are dropped.
    feature_names
        Column names when ``X`` is an array.
    shortlist
        Number of features (largest |coef| in a conditional main-effects fit)
        whose interactions with each candidate are tested.
    min_stratum
        A candidate is screened only if both its strata (feature present /
        absent) hold at least this fraction of samples.
    candidates
        Restrict the screen to these feature names (default: all eligible).
    C, max_iter
        Inverse L2 strength and iteration cap of the logistic fits.

    Returns
    -------
    pandas.DataFrame
        One row per screened candidate, sorted by ``lr_stat``: ``modifier``,
        ``loglik``, ``lr_stat`` (2 x gain over the shortlist main-effects fit
        with ``m`` added), ``df``, ``p_value``, ``frac_present``,
        ``lr_ratio_to_next`` (how far the row stands above the next one).
    """
    from scipy.stats import chi2

    M, names, index = _as_matrix(X, feature_names)
    M, yv, _ = _labelled(M, y, index)
    _, _, S = _shortlist(M, yv, shortlist, C, 2000)
    XS = M[:, S]
    pos = {int(j): i for i, j in enumerate(S)}
    prev = (M > 0).mean(0)
    pool = np.where((prev >= min_stratum) & (prev <= 1 - min_stratum))[0]
    if candidates is not None:
        want = set(map(str, candidates))
        pool = np.array([j for j in pool if names[j] in want], dtype=int)

    rows = []
    for m in pool:
        mv = M[:, m][:, None]
        base = XS if int(m) in pos else np.hstack([XS, mv])
        inter = XS * mv
        if int(m) in pos:
            inter = np.delete(inter, pos[int(m)], axis=1)
        D0 = base
        D1 = np.hstack([base, inter])
        f0 = _logit(C, max_iter).fit(D0, yv)
        f1 = _logit(C, max_iter).fit(D1, yv)
        ll0, ll1 = _loglik(f0, D0, yv), _loglik(f1, D1, yv)
        lr = max(0.0, 2.0 * (ll1 - ll0))
        dof = inter.shape[1]
        rows.append({
            "modifier": names[m], "loglik": ll1, "lr_stat": lr, "df": dof,
            "p_value": float(chi2.sf(lr, dof)), "frac_present": float(prev[m]),
        })
    out = pd.DataFrame(rows, columns=["modifier", "loglik", "lr_stat", "df",
                                      "p_value", "frac_present"])
    out = out.sort_values("lr_stat", ascending=False).reset_index(drop=True)
    nxt = out["lr_stat"].shift(-1)
    out["lr_ratio_to_next"] = out["lr_stat"] / nxt.where(nxt > 0)
    return out


# --------------------------------------------------------------------------- #
# stepwise conditional selection (data-driven shortlist size)                 #
# --------------------------------------------------------------------------- #
def _structure_covariates(M, n_pcs, seed=0):
    """Top principal components of the genotype matrix (lineage proxies)."""
    if not n_pcs:
        return None
    from sklearn.decomposition import PCA
    k = int(min(n_pcs, M.shape[1] - 1, M.shape[0] - 1))
    Z = PCA(k, random_state=seed).fit_transform(M - M.mean(0))
    sd = Z.std(0)
    sd[sd == 0] = 1.0
    return Z / sd


def _stepwise_select(M, y, candidates, m_idx, criterion, alpha, max_iter=2000,
                     covariates=None, gamma=1.0, n_exact=3):
    """Forward selection by likelihood ratio, each step conditional on the
    features already selected (and on the modifier, if any).

    A candidate enters as its main effect (+ its interaction with the
    modifier). ``covariates`` (e.g. genotype PCs = lineage proxies) are in
    every model, so a gene that only marks a lineage with a different trait
    rate gains nothing. Stops when the best remaining candidate fails the
    criterion: ``'ebic'`` — LR stat <= df * log(n) + 2 * gamma * log(p)
    (extended BIC, Chen & Chen 2008: the penalty grows with the number of
    genes p the candidate was chosen from); ``'bic'`` — LR stat <= df * log(n);
    ``'bonferroni'`` — chi2 p-value x number of remaining candidates >= ``alpha``. Afterwards the modifier's own
    main effect is tested the same way (drop-one), so a modifier with an
    effect of its own is still reported as a determinant.
    """
    from scipy.stats import chi2

    n = len(y)
    big_c = 1e4                               # ~unpenalised for likelihood ratios
    mv = M[:, [m_idx]] if m_idx is not None else None

    def block(j):
        x = M[:, [j]]
        return np.hstack([x, x * mv]) if mv is not None else x

    def fit_model(D):
        if D is None or D.shape[1] == 0:
            p0 = np.clip(y.mean(), 1e-12, 1 - 1e-12)
            ll0 = float(y.sum() * np.log(p0) + (n - y.sum()) * np.log(1 - p0))
            return ll0, np.full(n, p0)
        f = _logit(big_c, max_iter).fit(D, y)
        pr = np.clip(f.predict_proba(D)[:, 1], 1e-12, 1 - 1e-12)
        return float(y @ np.log(pr) + (1 - y) @ np.log(1 - pr)), pr

    def fit_ll(D):
        return fit_model(D)[0]

    def score_stats(D, pr, cands):
        """Rao score statistic for adding each candidate block to the fitted
        model (one fit per step instead of one per candidate)."""
        w = pr * (1 - pr)
        A = np.ones((n, 1)) if D is None else np.hstack([np.ones((n, 1)), D])
        AtWA_inv = np.linalg.pinv(A.T @ (A * w[:, None]))
        r = y - pr
        out = []
        for j in cands:
            Z = block(j)
            U = Z.T @ r
            ZW = Z * w[:, None]
            I = Z.T @ ZW - (ZW.T @ A) @ AtWA_inv @ (A.T @ ZW)
            try:
                out.append(float(U @ np.linalg.pinv(I) @ U))
            except np.linalg.LinAlgError:
                out.append(0.0)
        return np.asarray(out)

    p_all = M.shape[1]

    def passes(lr, dof, n_left):
        if criterion == "ebic":
            return lr > dof * np.log(n) + 2.0 * gamma * np.log(p_all)
        if criterion == "bic":
            return lr > dof * np.log(n)
        return chi2.sf(lr, dof) * max(n_left, 1) < alpha

    if covariates is not None:
        base = covariates if mv is None else np.hstack([mv, covariates])
    else:
        base = mv
    L, pr = fit_model(base)
    selected, path = [], []
    remaining = [int(j) for j in candidates if j != m_idx]
    while remaining:
        sc = score_stats(base, pr, remaining)
        shortlist = [remaining[k] for k in np.argsort(-sc)[:n_exact]]
        best = None
        for j in shortlist:                  # exact LR for the leading candidates
            D = block(j) if base is None else np.hstack([base, block(j)])
            l, pj = fit_model(D)
            if best is None or l > best[0]:
                best = (l, j, D, pj)
        l, j, D, pj = best
        dof = D.shape[1] - (0 if base is None else base.shape[1])
        lr = max(0.0, 2.0 * (l - L))
        ok = passes(lr, dof, len(remaining))
        path.append({"feature_idx": j, "lr_stat": lr, "df": dof,
                     "p_value": float(chi2.sf(lr, dof)), "n_candidates_left": len(remaining),
                     "accepted": bool(ok)})
        if not ok:
            break
        selected.append(j)
        remaining.remove(j)
        base, L, pr = D, l, pj

    mod_is_det, mod_lr = False, None
    if m_idx is not None and selected:
        rest = np.hstack([block(j) for j in selected])
        if covariates is not None:
            rest = np.hstack([covariates, rest])
        D_full = np.hstack([mv, rest])
        l_full = fit_ll(D_full)
        l_drop = fit_ll(rest)
        mod_lr = max(0.0, 2.0 * (l_full - l_drop))
        mod_is_det = passes(mod_lr, 1, 1)
        if criterion == "ebic":
            mod_is_det = mod_lr > np.log(n) + 2.0 * gamma * np.log(p_all)
    return selected, path, mod_is_det, mod_lr


# --------------------------------------------------------------------------- #
# determinant ranking                                                          #
# --------------------------------------------------------------------------- #
@register_function(
    aliases=[
        "trait_determinants", "determinant_ranking", "rank_causal_genes",
        "resistance_determinants", "bacterial_gwas_ranking", "driver_gene_ranking",
        "决定基因排序", "耐药决定基因", "因果基因排序",
    ],
    category="genetics",
    description=(
        "Rank the features (genes / loci) that DRIVE a binary trait from a "
        "samples x features presence/absence matrix, separating genuine "
        "determinants from (a) hitch-hikers that are only linked to them "
        "(e.g. other-class resistance genes on the same plasmid) and (b) a "
        "permissive background locus that only modifies other genes' effects. "
        "Conditional multivariable logistic fit + likelihood-ratio screen for "
        "the effect modifier + refit with modifier interactions; each feature "
        "is scored by its log-odds in the stratum where it acts, the modifier "
        "only by its own main effect. Typical use: antimicrobial-resistance "
        "gene ranking per drug, microbial GWAS, epistasis-aware driver lists."
    ),
    examples=[
        "res = ov.genetics.trait_determinants(genomes_df, y_resistant)",
        "res.head(8)  # feature, score, beta_main, beta_with_modifier, rank",
        "res.attrs['modifier']  # the background locus that was detected",
    ],
    related=["ov.genetics.effect_modifier_scan", "ov.genetics.gwas_association",
             "ov.genetics.transfer_trait_predict"],
)
def trait_determinants(
    X: Union[pd.DataFrame, np.ndarray],
    y,
    *,
    feature_names: Optional[Sequence[str]] = None,
    modifier: Union[str, None] = "auto",
    shortlist: int = 30,
    min_stratum: float = 0.05,
    modifier_alpha: float = 1e-3,
    C: float = 1.0,
    max_iter: int = 2000,
    direction: str = "both",
    select: Optional[str] = "ebic",
    max_candidates: int = 40,
    alpha: float = 0.05,
    n_pcs: int = 10,
    ebic_gamma: float = 1.0,
) -> pd.DataFrame:
    """Effect-modification-aware ranking of trait determinants.

    Steps
    -----
    1. Conditional main-effects logistic fit over all features (removes
       features that are only linked to a determinant).
    2. ``modifier='auto'``: :func:`effect_modifier_scan`; the top candidate is
       accepted if its LRT p-value is below ``modifier_alpha`` divided by the
       number of candidates screened (Bonferroni). ``modifier=None`` skips
       this; a feature name forces that modifier.
    3. Refit over all features plus ``feature x modifier`` interactions. A
       feature's score is ``max(|b|, |b + g|)`` — its log-odds without and with
       the modifier present, whichever stratum it acts in. The modifier is
       scored by ``|b_modifier|`` only (it is a background, not a determinant,
       unless it has an effect of its own).
    4. ``select``: how many features are determinants is decided by the data,
       not by a fixed count. Forward stepwise selection over the top
       ``max_candidates`` features by score: at each step the candidate (main
       effect + interaction with the modifier) with the largest likelihood
       gain *conditional on the features already selected* is tested, and
       selection stops at the first candidate that fails — ``'ebic'``
       (default; extended BIC: LR > df x log n + 2 x gamma x log p, which
       scales with both the cohort size and the number of genes screened),
       ``'bic'`` (LR > df x log n) or ``'bonferroni'`` (p x candidates left <
       ``alpha``). The top ``n_pcs`` principal components of the genotype
       matrix are covariates in every selection model, so genes that only
       mark a lineage (population structure) are not selected. A feature that only travels with a
       selected determinant adds nothing once that determinant is in the
       model, so it is not selected. Selected features are flagged
       ``is_determinant`` and placed first (by score); the modifier is a
       determinant only if its own main effect passes a drop-one test.
       ``select=None`` keeps the plain score order.

    Parameters
    ----------
    X, y, feature_names
        As in :func:`effect_modifier_scan`.
    modifier
        ``'auto'`` (screen), ``None`` (main effects only) or a feature name.
    shortlist, min_stratum, C
        Passed to the screen / fits.
    modifier_alpha
        Family-wise level for accepting the screened modifier.
    direction
        ``'both'`` scores |log-odds|; ``'risk'`` scores only positive
        (trait-increasing) effects, ``'protective'`` only negative ones.
    select, max_candidates, alpha, n_pcs, ebic_gamma
        Data-driven determinant set (step 4): ``'ebic'``, ``'bic'``,
        ``'bonferroni'`` or ``None``; ``n_pcs=0`` drops the lineage
        covariates.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``score``: ``feature``, ``score``, ``rank`` (1 = strongest),
        ``beta_main`` (b), ``beta_with_modifier`` (b + g), ``acts_in``
        (``'all'`` / ``'modifier_present'`` / ``'modifier_absent'``),
        ``is_modifier``, ``frac_present``, ``is_determinant``,
        ``selection_step`` (order of entry), ``selection_lr``. ``df.attrs``
        holds ``modifier``, ``n_determinants``, ``selection_path`` (every step
        incl. the first rejected one) and the screen table
        (``modifier_scan``) when one was run. The shortlist is
        ``df[df.is_determinant]`` — its size is decided by the data.
    """
    M, names, index = _as_matrix(X, feature_names)
    M, yv, _ = _labelled(M, y, index)
    prev = (M > 0).mean(0)

    scan = None
    m_idx = None
    if isinstance(modifier, str) and modifier != "auto":
        if modifier not in names:
            raise KeyError(f"modifier {modifier!r} is not a feature")
        m_idx = names.index(modifier)
    elif modifier == "auto":
        scan = effect_modifier_scan(M, yv, feature_names=names, shortlist=shortlist,
                                    min_stratum=min_stratum, C=C)
        if len(scan) and scan.loc[0, "p_value"] < modifier_alpha / max(len(scan), 1):
            m_idx = names.index(scan.loc[0, "modifier"])

    if m_idx is None:
        fit = _logit(C, max_iter).fit(M, yv)
        b = fit.coef_[0]
        g = np.zeros_like(b)
    else:
        mv = M[:, m_idx][:, None]
        full = np.hstack([M, M * mv])
        fit = _logit(C, max_iter).fit(full, yv)
        b = fit.coef_[0][: M.shape[1]]
        g = fit.coef_[0][M.shape[1]:]
        g[m_idx] = 0.0          # x_m * x_m == x_m: no separate interaction

    bg = b + g
    if direction == "risk":
        s0, s1 = np.clip(b, 0, None), np.clip(bg, 0, None)
    elif direction == "protective":
        s0, s1 = np.clip(-b, 0, None), np.clip(-bg, 0, None)
    elif direction == "both":
        s0, s1 = np.abs(b), np.abs(bg)
    else:
        raise ValueError("direction must be 'both', 'risk' or 'protective'")
    score = np.maximum(s0, s1)
    acts = np.where(np.isclose(s0, s1, rtol=0.25), "all",
                    np.where(s1 > s0, "modifier_present", "modifier_absent"))
    if m_idx is not None:
        score[m_idx] = s0[m_idx]
        acts[m_idx] = "is_modifier"
    else:
        acts[:] = "all"

    out = pd.DataFrame({
        "feature": names, "score": score, "beta_main": b,
        "beta_with_modifier": bg, "acts_in": acts,
        "is_modifier": [j == m_idx for j in range(len(names))],
        "frac_present": prev,
    })
    out["is_determinant"] = False
    out["selection_step"] = np.nan
    out["selection_lr"] = np.nan
    path_df = None
    if select is not None:
        if select not in ("ebic", "bic", "bonferroni"):
            raise ValueError("select must be 'ebic', 'bic', 'bonferroni' or None")
        cov = _structure_covariates(M, n_pcs)
        cand = [int(j) for j in np.argsort(-score, kind="mergesort") if j != m_idx]
        cand = [j for j in cand if score[j] > 0][:max_candidates]
        sel, path, mod_det, mod_lr = _stepwise_select(M, yv, cand, m_idx, select, alpha,
                                                      covariates=cov, gamma=ebic_gamma)
        if sel and direction in ("risk", "protective"):
            # keep only features whose conditional effect (in the stratum where it
            # is largest) points in the requested direction
            mvv = M[:, [m_idx]] if m_idx is not None else None
            blocks = [np.hstack([M[:, [j]], M[:, [j]] * mvv]) if mvv is not None else M[:, [j]]
                      for j in sel]
            pre = [a for a in (mvv, cov) if a is not None]
            D = np.hstack(pre + blocks)
            co = _logit(1e4, max_iter).fit(D, yv).coef_[0][sum(a.shape[1] for a in pre):]
            w = 2 if mvv is not None else 1
            keep = []
            for k, j in enumerate(sel):
                e = co[k * w: (k + 1) * w]
                strata = [e[0], e[0] + e[1]] if w == 2 else [e[0]]
                eff = max(strata, key=abs)
                if (eff > 0) == (direction == "risk"):
                    keep.append(j)
            sel = keep
        for k, j in enumerate(sel):
            out.loc[j, ["is_determinant", "selection_step"]] = [True, k + 1]
        for rec in path:
            out.loc[rec["feature_idx"], "selection_lr"] = rec["lr_stat"]
        if m_idx is not None and mod_det:
            out.loc[m_idx, "is_determinant"] = True
            out.loc[m_idx, "selection_lr"] = mod_lr
        path_df = pd.DataFrame(path)
        if len(path_df):
            path_df.insert(0, "feature", [names[j] for j in path_df.pop("feature_idx")])
    out = out.sort_values(["is_determinant", "score"], ascending=[False, False],
                          kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out = out[["feature", "score", "rank", "is_determinant", "beta_main",
               "beta_with_modifier", "acts_in", "is_modifier", "frac_present",
               "selection_step", "selection_lr"]]
    out.attrs["modifier"] = names[m_idx] if m_idx is not None else None
    out.attrs["n_determinants"] = int(out["is_determinant"].sum()) if select else None
    if path_df is not None:
        out.attrs["selection_path"] = path_df
    if scan is not None:
        out.attrs["modifier_scan"] = scan
    return out


# --------------------------------------------------------------------------- #
# related-trait transfer                                                       #
# --------------------------------------------------------------------------- #
def _stack(M, Y, traits):
    rows, ys = [], []
    for t in traits:
        col = Y[t]
        ok = col.notna().to_numpy()
        rows.append(np.where(ok)[0])
        ys.append(_as_binary(col[ok]))
    return np.concatenate(rows), np.concatenate(ys)


def _trait_attributes(groups, target=None) -> pd.DataFrame:
    """Normalise trait metadata to a DataFrame (index = trait, columns = attributes).

    A mapping trait -> group becomes one column ``group``. A ``smiles`` column
    (any case) is replaced by ``n_components`` — the number of '.'-separated
    molecules, which distinguishes a combination product (e.g. an agent plus an
    inhibitor) from a single agent without any chemistry knowledge table.
    """
    if isinstance(groups, pd.DataFrame):
        meta = groups.copy()
        if target is not None and str(target) not in meta.index.astype(str):
            # trait names held in a column rather than the index
            for c in meta.columns:
                if meta[c].astype(str).eq(str(target)).any() and meta[c].astype(str).is_unique:
                    meta = meta.set_index(c)
                    break
    elif isinstance(groups, pd.Series):
        meta = groups.rename("group").to_frame()
    else:
        meta = pd.Series(dict(groups)).rename("group").to_frame()
    meta.index = meta.index.astype(str)
    for c in list(meta.columns):
        if str(c).lower() == "smiles":
            meta["n_components"] = meta[c].astype(str).str.count(r"\.") + 1
            meta = meta.drop(columns=c)
    return meta


import re as _re

# metadata columns treated as mechanistically relevant by default: the number
# of active components (derived from SMILES) and columns that name an added
# component (inhibitor / combination partner / adjuvant / protecting agent).
_RELEVANT = _re.compile(r"(n_components|component|inhibit|combin|partner|adjuvant|"
                        r"protect|potentiat)", _re.I)


def _match_score(meta, a, b, cols):
    return sum(str(meta.at[a, c]) == str(meta.at[b, c]) for c in cols)


@register_function(
    aliases=[
        "transfer_trait_predict", "related_trait_transfer", "zero_shot_trait_transfer",
        "held_out_drug_prediction", "class_pooled_transfer", "unseen_drug_resistance",
        "相关性状迁移预测", "未见药物耐药预测", "同类药物迁移",
    ],
    category="genetics",
    description=(
        "Predict a binary trait that has NO labels (e.g. resistance to a "
        "held-out drug) from a samples x features genotype matrix, using "
        "labelled related traits of the same group (class). Mechanistically "
        "relevant metadata only (number of active components from SMILES, "
        "inhibitor / combination columns) decides equivalence; incidental "
        "columns are ignored. Equivalent sources are pooled; if the target has "
        "an added component no source shares, a sample is called positive only "
        "when every group-mate's model agrees. Returns the decision with its "
        "reasons, per-determinant agreement of the source models on carriers, "
        "leave-one-source-out metrics, per-source probabilities and the "
        "features whose effects disagree across sources."
    ),
    examples=[
        "res = ov.genetics.transfer_trait_predict(genomes_df, labels_wide, target='drugX', groups=drug_meta_df)",
        "res['proba'], res['call'], res['decision'], res['source_match']",
        "res['validation']  # leave-one-trait-out metrics within the sources",
    ],
    related=["ov.genetics.trait_determinants"],
)
def transfer_trait_predict(
    X: Union[pd.DataFrame, np.ndarray],
    Y: pd.DataFrame,
    target: str,
    *,
    sources: Optional[Sequence[str]] = None,
    groups=None,
    group_col: Optional[str] = None,
    decision: str = "auto",
    C: float = 1.0,
    max_iter: int = 3000,
    threshold: float = 0.5,
    validate: bool = True,
    n_discordant: int = 30,
    force: bool = False,
    relevant_cols: Optional[Sequence[str]] = None,
) -> dict:
    """Transfer to an unlabelled trait from labelled related traits.

    Parameters
    ----------
    X
        Samples x features; a DataFrame indexed by sample id is aligned to
        ``Y``'s index.
    Y
        Wide label table, samples x traits (NaN = unlabelled), any binary
        coding. From a long table:
        ``long.pivot(index='isolate_id', columns='drug', values='phenotype')``.
    target
        Trait to predict (need not be a column of ``Y``).
    sources
        Labelled traits to use; overrides the metadata-based choice.
    groups
        Trait metadata covering labelled AND unlabelled traits: pass the whole
        table (DataFrame; trait names in the index or a column). A mapping
        trait -> group also works but cannot show mechanistic differences. A
        ``smiles`` column becomes ``n_components``.
    relevant_cols
        Columns that define mechanistic equivalence. Default: ``n_components``
        and columns whose name mentions a component / inhibitor /
        combination / partner / adjuvant / protecting agent. Other columns are
        incidental and ignored.
    group_col
        Attribute that defines the group (default: the first column). Sources
        must share it with the target.
    decision
        Leave unset. ``'pooled'`` — one model on the stacked source labels, call at
        ``threshold``. ``'consensus'`` — one model per source; call positive
        only if every source model calls positive; ``proba`` is the minimum
        over sources. ``'auto'`` (default) — ``'pooled'`` if some source
        matches the target on every attribute (or there is a single source),
        otherwise ``'consensus'``. (Now: ``'pooled'`` over the group-mates that
        share the target's relevant attributes; ``'consensus'`` over all
        group-mates when none does.)
        A requested ``'pooled'`` is switched to ``'consensus'`` (with a
        warning) when the metadata shows the target differs from every
        source, unless ``force=True``.
    force
        Keep a requested ``decision`` against the metadata evidence (warned
        and recorded in ``warnings``).
    validate
        Leave-one-source-out AUROC / MCC (needs >= 2 sources).
    n_discordant
        Size of the table of features whose per-source effects disagree most.

    Returns
    -------
    dict
        ``proba``, ``call`` (Series indexed like ``X``), ``sources``,
        ``decision`` (the rule used), ``decision_requested``, ``warnings``
        (list of str; also emitted as UserWarning), ``source_call_disagreement``
        (fraction of samples called positive by some but not all source models), ``source_match`` (candidate sources with
        their attribute agreement), ``per_source_proba`` (DataFrame),
        ``source_agreement`` (fraction of source models calling positive),
        ``discordant_features`` (per-source coefficients, largest spread
        first), ``validation`` and ``model`` (the pooled model).
    """
    from sklearn.metrics import matthews_corrcoef, roc_auc_score

    if isinstance(X, pd.DataFrame):
        Xd = X
    else:
        Xd = pd.DataFrame(np.asarray(X, dtype=float), index=Y.index)
    Yd = Y.reindex(Xd.index)
    Yd.columns = Yd.columns.astype(str)
    M = Xd.to_numpy(dtype=float)
    target = str(target)

    match_tab = None
    reasons = []
    if sources is None:
        if groups is None:
            raise ValueError("give `sources`, or `groups` (trait metadata incl. the target)")
        meta = _trait_attributes(groups, target)
        if target not in meta.index:
            raise ValueError(
                f"target {target!r} has no group: pass `sources=[...]`, or `groups` "
                f"built from the full trait metadata (labelled AND unlabelled traits, "
                f"e.g. dict(zip(meta.drug, meta.subclass)) or meta.set_index('drug')); "
                f"groups has {sorted(meta.index)}")
        gcol = group_col if group_col is not None else meta.columns[0]
        others = [c for c in meta.columns if c != gcol]
        n_attr = len(others)
        if relevant_cols is None:
            rel = [c for c in others if _RELEVANT.search(str(c))]
        else:
            rel = [c for c in relevant_cols if c in meta.columns and c != gcol]
        labelled = [t for t in Yd.columns if t != target and t in meta.index
                    and Yd[t].notna().any()]
        cand = [t for t in labelled if str(meta.at[t, gcol]) == str(meta.at[target, gcol])]
        if not cand:
            raise ValueError(f"no labelled trait shares {gcol}={meta.at[target, gcol]!r} "
                             f"with {target!r}")
        same = {t: all(str(meta.at[t, c]) == str(meta.at[target, c]) for c in rel)
                for t in cand}
        matched = [t for t in cand if same[t]]
        if not rel:
            sources, metadata_differs = cand, False
            reasons.append(f"no mechanistically relevant attribute (component count / "
                           f"inhibitor / combination) in the metadata: all {len(cand)} "
                           f"labelled traits of {gcol}={meta.at[target, gcol]!r} are used")
        elif matched:
            sources, metadata_differs = matched, False
            reasons.append(f"sources share the target's {gcol} and its relevant attributes "
                           f"{rel} -> treated as mechanistically equivalent")
        else:
            sources, metadata_differs = cand, len(cand) >= 2
            tv = {c: meta.at[target, c] for c in rel}
            reasons.append(f"the target differs from every labelled {gcol}-mate on relevant "
                           f"attributes {tv} (e.g. an added component) -> determinants must "
                           f"act in all sources to be trusted")
        ignored = [c for c in others if c not in rel]
        if ignored:
            reasons.append(f"columns not treated as mechanistic (incidental): {ignored}")
        match_tab = pd.DataFrame({"source": cand,
                                  **{f"{c}": [meta.at[t, c] for t in cand] for c in rel},
                                  "relevant_match": [same[t] for t in cand],
                                  "used": [t in sources for t in cand]})
    else:
        sources = [str(s) for s in sources if str(s) != target]
        metadata_differs = False
        n_attr = None
        rel = None
        reasons.append("sources given explicitly")

    notes = []

    def _warn(msg):
        notes.append(msg)
        warnings.warn(f"transfer_trait_predict({target!r}): {msg}", UserWarning, stacklevel=3)

    requested = decision
    if decision not in ("auto", "pooled", "consensus"):
        raise ValueError("decision must be 'auto', 'pooled' or 'consensus'")
    if match_tab is not None and not rel:
        _warn("the metadata given has no mechanistically relevant attribute besides the "
              "group (SMILES / component count / inhibitor / combination), so whether the "
              "sources are mechanistically equivalent to the target cannot be checked. "
              "Pass the WHOLE trait metadata table (all columns, e.g. "
              "pd.read_csv(<drug table>)) as `groups` and leave `decision` unset.")
    if decision == "auto":
        decision = "consensus" if metadata_differs else "pooled"
        reasons.append(f"decision '{decision}'")
    elif decision == "pooled" and metadata_differs:
        if force:
            _warn("decision='pooled' was FORCED although the metadata shows the target "
                  "differs from every source (see source_match); determinants that act "
                  "in some sources but are neutralised in the target will be trusted.")
        else:
            _warn("decision='pooled' was requested but the metadata shows the target "
                  "differs from every source (see source_match) -> using 'consensus'. "
                  "Pass force=True only with a stated mechanistic reason.")
            decision = "consensus"

    r, yv = _stack(M, Yd, sources)
    model = _logit(C, max_iter).fit(M[r], yv)
    pooled = model.predict_proba(M)[:, 1]

    per, coefs = {}, {}
    if len(sources) >= 2:
        for s_ in sources:
            rr, yy = _stack(M, Yd, [s_])
            f = _logit(C, max_iter).fit(M[rr], yy)
            per[s_] = f.predict_proba(M)[:, 1]
            coefs[s_] = f.coef_[0]
    per_df = pd.DataFrame(per, index=Xd.index) if per else None

    if decision == "consensus" and per:
        p = per_df.min(axis=1).to_numpy()
    else:
        p = pooled
    proba = pd.Series(p, index=Xd.index, name=target)
    agreement = ((per_df >= threshold).mean(axis=1) if per_df is not None else None)
    disagreement = None
    if agreement is not None:
        any_pos = agreement > 0
        split = any_pos & (agreement < 1)
        disagreement = float(split.sum() / max(int(any_pos.sum()), 1))
        if decision == "pooled" and disagreement > 0 and (requested == "pooled" or not rel):
            _warn(f"the per-source models disagree on {disagreement:.1%} of the samples any "
                  f"source calls positive (inconsistent determinant effects, see "
                  f"discordant_features) and the pooled rule is in use.")

    disc = None
    if coefs:
        cf = pd.DataFrame(coefs, index=Xd.columns.astype(str))
        disc = cf.assign(spread=cf.max(axis=1) - cf.min(axis=1)) \
                 .sort_values("spread", ascending=False).head(n_discordant)

    det_agree = None
    if coefs:
        top = disc.index if disc is not None else []
        cf_all = pd.DataFrame(coefs, index=Xd.columns.astype(str))
        strongest = cf_all.max(axis=1).sort_values(ascending=False).head(n_discordant).index
        rows = []
        for g in strongest:
            carriers = Xd[g].to_numpy() > 0
            if carriers.sum() == 0:
                continue
            rates = {s_: float((per_df[s_].to_numpy()[carriers] >= threshold).mean())
                     for s_ in per_df.columns}
            rows.append({"feature": g, "n_carriers": int(carriers.sum()),
                         **{f"called_{k}": v for k, v in rates.items()},
                         "max_rate_gap": max(rates.values()) - min(rates.values())})
        det_agree = pd.DataFrame(rows).sort_values("max_rate_gap", ascending=False) \
            .reset_index(drop=True) if rows else None
        if det_agree is not None and len(det_agree):
            r0 = det_agree.iloc[0]
            reasons.append(f"label-free agreement: largest gap between source models on "
                           f"carriers of one determinant = {r0.max_rate_gap:.2f} "
                           f"({r0.feature}); overall source_call_disagreement = "
                           f"{disagreement:.3f}")

    val = None
    if validate and len(sources) >= 2:
        rec = []
        for s_ in sources:
            rest = [t for t in sources if t != s_]
            rr, yy = _stack(M, Yd, rest)
            mdl = _logit(C, max_iter).fit(M[rr], yy)
            ok = Yd[s_].notna().to_numpy()
            ys = _as_binary(Yd[s_][ok])
            ps = mdl.predict_proba(M[ok])[:, 1]
            rec.append({"held_out": s_, "trained_on": ",".join(rest), "n": int(ok.sum()),
                        "auroc": float(roc_auc_score(ys, ps)) if len(set(ys)) > 1 else np.nan,
                        "mcc": float(matthews_corrcoef(ys, (ps >= threshold).astype(int)))})
        val = pd.DataFrame(rec)
    return {"proba": proba, "call": (proba >= threshold).astype(int),
            "sources": list(sources), "decision": decision,
            "decision_requested": requested, "decision_reasons": reasons,
            "warnings": notes, "determinant_agreement": det_agree,
            "source_call_disagreement": disagreement, "source_match": match_tab,
            "per_source_proba": per_df, "source_agreement": agreement,
            "discordant_features": disc, "validation": val, "model": model}
