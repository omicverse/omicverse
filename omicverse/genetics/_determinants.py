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

    Returns
    -------
    pandas.DataFrame
        Sorted by ``score``: ``feature``, ``score``, ``rank`` (1 = strongest),
        ``beta_main`` (b), ``beta_with_modifier`` (b + g), ``acts_in``
        (``'all'`` / ``'modifier_present'`` / ``'modifier_absent'``),
        ``is_modifier``, ``frac_present``. ``df.attrs`` holds ``modifier`` and
        the screen table (``modifier_scan``) when one was run.
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
    }).sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out = out[["feature", "score", "rank", "beta_main", "beta_with_modifier",
               "acts_in", "is_modifier", "frac_present"]]
    out.attrs["modifier"] = names[m_idx] if m_idx is not None else None
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


@register_function(
    aliases=[
        "transfer_trait_predict", "related_trait_transfer", "zero_shot_trait_transfer",
        "held_out_drug_prediction", "class_pooled_transfer", "unseen_drug_resistance",
        "相关性状迁移预测", "未见药物耐药预测", "同类药物迁移",
    ],
    category="genetics",
    description=(
        "Predict a binary trait that has NO labels (e.g. resistance to a "
        "held-out drug) from a samples x features genotype matrix, by fitting "
        "one penalised logistic model on the stacked labels of related, "
        "labelled traits — by default those in the same group (e.g. the same "
        "drug class / subclass from a drug table). Reports leave-one-trait-out "
        "AUROC / MCC inside the group so the transfer can be validated "
        "without target labels. Pooling across groups is usually worse than "
        "staying within the group; pass sources explicitly to override."
    ),
    examples=[
        "res = ov.genetics.transfer_trait_predict(genomes_df, labels_wide, target='drugX', groups=drug_to_class)",
        "res['proba']  # Series of P(trait) per sample",
        "res['validation']  # leave-one-trait-out metrics within the group",
    ],
    related=["ov.genetics.trait_determinants"],
)
def transfer_trait_predict(
    X: Union[pd.DataFrame, np.ndarray],
    Y: pd.DataFrame,
    target: str,
    *,
    sources: Optional[Sequence[str]] = None,
    groups: Optional[Mapping[str, str]] = None,
    C: float = 1.0,
    max_iter: int = 3000,
    threshold: float = 0.5,
    validate: bool = True,
) -> dict:
    """Pooled-source logistic transfer to an unlabelled trait.

    Parameters
    ----------
    X
        Samples x features; a DataFrame indexed by sample id is aligned to
        ``Y``'s index.
    Y
        Wide label table, samples x traits (NaN = unlabelled), values in any
        binary coding (``'R'``/``'S'``, 0/1 ...). A long table can be pivoted
        first: ``long.pivot(index='isolate_id', columns='drug', values='phenotype')``.
    target
        Name of the trait to predict (need not be a column of ``Y``).
    sources
        Labelled traits to pool. Default: the columns of ``Y`` whose
        ``groups`` value equals ``groups[target]``.
    groups
        Mapping trait -> group (e.g. drug -> subclass).
    threshold
        Probability cut for the hard call.
    validate
        If two or more sources: leave each source out, fit on the rest, and
        score the left-out source (AUROC, MCC at ``threshold``).

    Returns
    -------
    dict
        ``proba`` (Series indexed like ``X``), ``call`` (Series 0/1),
        ``sources``, ``validation`` (DataFrame or None), ``model``.
    """
    from sklearn.metrics import matthews_corrcoef, roc_auc_score

    if isinstance(X, pd.DataFrame):
        Xd = X
    else:
        Xd = pd.DataFrame(np.asarray(X, dtype=float), index=Y.index)
    Yd = Y.reindex(Xd.index)
    M = Xd.to_numpy(dtype=float)

    if sources is None:
        if groups is None or target not in groups:
            raise ValueError("give `sources`, or `groups` that contains the target")
        g = groups[target]
        sources = [t for t in Yd.columns if t != target and groups.get(t) == g
                   and Yd[t].notna().any()]
        if not sources:
            raise ValueError(f"no labelled trait shares group {g!r} with {target!r}")
    sources = [s for s in sources if s != target]

    r, yv = _stack(M, Yd, sources)
    model = _logit(C, max_iter).fit(M[r], yv)
    proba = pd.Series(model.predict_proba(M)[:, 1], index=Xd.index, name=target)

    val = None
    if validate and len(sources) >= 2:
        rec = []
        for s in sources:
            rest = [t for t in sources if t != s]
            rr, yy = _stack(M, Yd, rest)
            mdl = _logit(C, max_iter).fit(M[rr], yy)
            ok = Yd[s].notna().to_numpy()
            ys = _as_binary(Yd[s][ok])
            ps = mdl.predict_proba(M[ok])[:, 1]
            rec.append({"held_out": s, "trained_on": ",".join(rest), "n": int(ok.sum()),
                        "auroc": float(roc_auc_score(ys, ps)) if len(set(ys)) > 1 else np.nan,
                        "mcc": float(matthews_corrcoef(ys, (ps >= threshold).astype(int)))})
        val = pd.DataFrame(rec)
    return {"proba": proba, "call": (proba >= threshold).astype(int),
            "sources": list(sources), "validation": val, "model": model}
