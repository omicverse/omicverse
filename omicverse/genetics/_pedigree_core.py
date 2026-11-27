"""Family / haplotype inference core (numpy + scipy only, vendorable).

This module has no omicverse import so that a deliverable which must run
without omicverse (e.g. a re-executed analysis script) can embed it
verbatim: ``ov.genetics.pedigree_core_source(path)`` writes this file.

Contents
--------
Sampling design
    ``two_phase_weights``            inverse-probability weights of a
                                     (two-)phase sample, optional trimming.
Population haplotypes
    ``haplotype_frequencies``        design-weighted (Hajek) haplotype
                                     catalog and population frequencies
                                     from phased individuals.
    ``haplotype_frequencies_em``     weighted EM (Excoffier-Slatkin) over
                                     catalog diplotypes from UNPHASED
                                     founders' genotype likelihoods, phased
                                     founders as fixed diplotypes.
    ``NoCallModel``                  NO_CALL probability depending on the
                                     true genotype, site and previous
                                     locus' call (MNAR missingness).
    ``fit_haplotype_prior``          covariate- and group-dependent
                                     population haplotype prior learned
                                     from phased families under a
                                     sampling design: latent shared-family
                                     (founder) classes whose mixing
                                     depends on a family-level group label
                                     and, within each class, a penalised
                                     multinomial-logit surface on
                                     continuous covariates; class number,
                                     basis and penalty chosen on held-out
                                     families.
    ``HaplotypePrior``               fitted model: haplotype / diplotype
                                     priors, class posteriors, conditional
                                     diplotype prior of one parent given
                                     the other, population averages.
Meiosis
    ``gamete_sequences``             transmitted sequences with at most k
                                     crossovers.
    ``gamete_distribution``          probability of each transmitted sequence
                                     (any pair of sequences, <= 1 crossover).
    ``offspring_outcome_probability`` expected offspring outcome (penetrance)
                                     of a joint parental state.
    ``fit_crossover_intervals``      EM estimate of the no-crossover mass and
                                     per-interval single-crossover masses
                                     from phased parents + ordered
                                     transmissions (ambiguous origins
                                     marginalised).
Structural certificates
    ``genotype_likelihood_arrays``   long-format assay rows -> per-sample
                                     (n_loci, 3) likelihood arrays.
    ``confident_copy_calls``         per-locus hard copy-number constraints
                                     from genotype-likelihood triplets.
    ``compatible_diplotypes``        catalog diplotypes matching constraints.
    ``structural_identifiability``   number of joint parental states
                                     compatible with all constraints
                                     (exactly one -> identifiable).

Conventions: haplotypes are equal-length strings over {'0','1'} (or 0/1
arrays); a diplotype is an unordered pair stored canonically (a <= b).
"""
from __future__ import annotations

import hashlib
import math
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "two_phase_weights",
    "haplotype_frequencies",
    "haplotype_frequencies_em",
    "NoCallModel",
    "fit_haplotype_prior",
    "HaplotypePrior",
    "gamete_sequences",
    "gamete_distribution",
    "genotype_likelihood_arrays",
    "offspring_outcome_probability",
    "fit_crossover_intervals",
    "confident_copy_calls",
    "compatible_diplotypes",
    "structural_identifiability",
    "canonical_pair",
    "simulate_family_cohort",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def canonical_pair(a, b):
    """Unordered diplotype key (homologue swap invariant)."""
    return (a, b) if a <= b else (b, a)


def _as_bits(sequences) -> np.ndarray:
    seqs = list(sequences)
    if not seqs:
        return np.zeros((0, 0), dtype=np.int8)
    if isinstance(seqs[0], str):
        return np.array([[1 if ch == "1" else 0 for ch in s] for s in seqs], dtype=np.int8)
    return np.asarray(seqs, dtype=np.int8)


def _logsumexp(a, axis=None):
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis) if axis is not None else float(out.ravel()[0])


def _stable_fold(key: str, n_folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


# --------------------------------------------------------------------------
# sampling design
# --------------------------------------------------------------------------
def two_phase_weights(first_phase_prob, second_phase_prob=None, *, trim_quantile=None,
                      normalize=True) -> np.ndarray:
    """Inverse-probability weights 1 / (pi1 * pi2) of a (two-)phase sample.

    ``first_phase_prob`` is each unit's inclusion probability into the study
    sample (ascertainment / recruitment), ``second_phase_prob`` the
    probability of being selected for the second phase (e.g. randomised
    phase validation) given the first. Weights estimate the population
    *before both phases*. ``trim_quantile`` (e.g. 0.99) caps weights at that
    quantile (variance / bias trade-off); ``normalize`` rescales to mean 1.
    """
    p = np.asarray(first_phase_prob, dtype=float)
    if second_phase_prob is not None:
        p = p * np.asarray(second_phase_prob, dtype=float)
    if np.any(~np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("inclusion probabilities must be finite and > 0")
    w = 1.0 / p
    if trim_quantile is not None:
        w = np.minimum(w, np.quantile(w, float(trim_quantile)))
    if normalize:
        w = w / w.mean()
    return w


def haplotype_frequencies(hap_a: Sequence, hap_b: Sequence, weights=None, *,
                          catalog: Optional[Sequence[str]] = None,
                          pseudocount: float = 0.0) -> Tuple[List[str], np.ndarray]:
    """Design-weighted (Hajek) population haplotype frequencies.

    Each phased individual contributes its two haplotypes with its sampling
    weight (``two_phase_weights``); frequencies are
    ``sum_i w_i (1[a_i=h] + 1[b_i=h]) / (2 sum_i w_i)``. Returns the sorted
    catalog (observed haplotypes, or ``catalog``) and the frequencies, which
    sum to one. ``pseudocount`` adds that many (weight-unit) copies to every
    catalog entry.
    """
    hap_a = list(hap_a)
    hap_b = list(hap_b)
    if len(hap_a) != len(hap_b):
        raise ValueError("hap_a and hap_b must have equal length")
    w = np.ones(len(hap_a)) if weights is None else np.asarray(weights, dtype=float)
    cat = sorted(set(hap_a) | set(hap_b)) if catalog is None else list(catalog)
    index = {h: i for i, h in enumerate(cat)}
    mass = np.full(len(cat), float(pseudocount))
    for a, b, wi in zip(hap_a, hap_b, w):
        for h in (a, b):
            if h in index:
                mass[index[h]] += wi
    total = mass.sum()
    if total <= 0:
        raise ValueError("no weight on the catalog")
    return cat, mass / total


# --------------------------------------------------------------------------
# unphased founders: genotype evidence, NO_CALL model, diplotype EM
# --------------------------------------------------------------------------
def _genotype_evidence(entry, n_loci: int, genotype_error: float = 0.0):
    """One person's genotype evidence -> (likelihood (L, 3), called (L,) bool).

    ``entry``: None (not assayed), an (L,) dosage vector (-1 / NaN = NO_CALL),
    an (L, 3) likelihood array (NaN row = NO_CALL) or ``(likelihood, called)``.
    """
    if entry is None:
        return np.full((n_loci, 3), np.nan), np.zeros(n_loci, dtype=bool)
    if isinstance(entry, tuple) and len(entry) == 2:
        lik = np.array(entry[0], dtype=float).reshape(n_loci, 3)
        called = np.asarray(entry[1], dtype=bool).reshape(n_loci) & np.all(np.isfinite(lik), axis=1)
    else:
        arr = np.asarray(entry, dtype=float)
        if arr.ndim == 2:
            lik = arr.reshape(n_loci, 3).copy()
            called = np.all(np.isfinite(lik), axis=1)
        else:
            arr = arr.reshape(n_loci)
            called = np.isfinite(arr) & (arr >= 0)
            e = float(genotype_error)
            lik = np.full((n_loci, 3), np.nan)
            lik[called] = e / 2.0
            idx = np.flatnonzero(called)
            lik[idx, arr[idx].astype(int)] = 1.0 - e
    lik = lik.copy()
    lik[~called] = np.nan
    if np.any(lik[called] < 0):
        raise ValueError("genotype likelihoods must be >= 0")
    return lik, called


class NoCallModel:
    """Genotype-dependent call-failure (NO_CALL) model.

    logit P(M_l = 1 | g_l, M_{l-1}) = b0 + s_l + b_het 1[g_l = 1]
                                       + b_alt 1[g_l = 2] + b_nb M_{l-1}

    with M_l the NO_CALL indicator at locus l (loci in catalog order), g_l
    the true copy number of allele '1', ``site`` effects s_l, ``genotype``
    effects (b_het, b_alt; heterozygote / alternative-homozygote failure,
    missing-not-at-random) and a ``neighbour`` effect of the previous
    locus' call status (clustered failures). Conditioning on M_{l-1} only
    makes prod_l P(M_l | g_l, M_{l-1}) a proper likelihood of the whole
    missingness pattern (first-order Markov chain), which multiplies the
    genotype likelihood of every candidate diplotype. Non-intercept terms
    carry a Gaussian (L2) prior. Motivation: genotype-dependent call
    failure (e.g. Anney et al. 2008, Am J Med Genet B 147B:1379; the
    allele-specific missingness model of GAW17, BMC Proc 2011 5(S9):S107)
    biases frequency estimates when NO_CALL is treated as ignorable.
    """

    FEATURES = ("site", "genotype", "neighbour")

    def __init__(self, n_loci: int, features: Sequence[str] = FEATURES, coef=None):
        bad = set(features) - set(self.FEATURES)
        if bad:
            raise ValueError(f"unknown missingness features {sorted(bad)}; use {self.FEATURES}")
        self.n_loci = int(n_loci)
        self.features = tuple(f for f in self.FEATURES if f in features)
        self.coef = np.zeros(self.n_params) if coef is None else np.asarray(coef, dtype=float)

    @property
    def n_params(self) -> int:
        return 1 + self.n_loci * ("site" in self.features) + 2 * ("genotype" in self.features) \
            + ("neighbour" in self.features)

    def _split(self, coef):
        k = 1
        b0 = coef[0]
        site = np.zeros(self.n_loci)
        gen = np.zeros(3)
        nb = 0.0
        if "site" in self.features:
            site = coef[k:k + self.n_loci]
            k += self.n_loci
        if "genotype" in self.features:
            gen = np.concatenate(([0.0], coef[k:k + 2]))
            k += 2
        if "neighbour" in self.features:
            nb = coef[k]
        return b0, site, gen, nb

    @property
    def params(self) -> Dict[str, object]:
        b0, site, gen, nb = self._split(self.coef)
        return {"intercept": float(b0), "site": site.copy(), "het": float(gen[1]),
                "hom_alt": float(gen[2]), "neighbour": float(nb)}

    @staticmethod
    def _prev(called):
        prev = np.zeros(called.shape, dtype=float)
        prev[:, 1:] = ~called[:, :-1]
        return prev

    def _eta(self, coef, prev):
        b0, site, gen, nb = self._split(coef)
        return b0 + site[None, :, None] + gen[None, None, :] + nb * prev[:, :, None]

    def no_call_prob(self, called) -> np.ndarray:
        """P(NO_CALL | g = 0, 1, 2) at every locus: (n, L, 3) given call patterns."""
        called = np.atleast_2d(np.asarray(called, dtype=bool))
        return 1.0 / (1.0 + np.exp(-self._eta(self.coef, self._prev(called))))

    def log_evidence(self, called) -> np.ndarray:
        """log P(M_l | g_l = k, M_{l-1}) for every person, locus and k: (n, L, 3)."""
        called = np.atleast_2d(np.asarray(called, dtype=bool))
        eta = self._eta(self.coef, self._prev(called))
        miss = (~called)[:, :, None]
        # log sigmoid(eta) if missing else log sigmoid(-eta)
        return -np.logaddexp(0.0, np.where(miss, -eta, eta))

    def fit(self, genotype_posterior, called, weights=None, penalty: float = 1.0, max_iter: int = 200):
        """Weighted ML (M-step) given posterior genotype probabilities (n, L, 3)."""
        from scipy.optimize import minimize

        post = np.asarray(genotype_posterior, dtype=float)
        called = np.asarray(called, dtype=bool)
        w = np.ones(len(post)) if weights is None else np.asarray(weights, dtype=float)
        W = w[:, None, None] * post
        M = (~called)[:, :, None].astype(float)
        prev = self._prev(called)
        pen_mask = np.ones(self.n_params)
        pen_mask[0] = 0.0
        has = {f: f in self.features for f in self.FEATURES}

        def objective(coef):
            eta = self._eta(coef, prev)
            ll = np.sum(W * (M * eta - np.logaddexp(0.0, eta)))
            R = W * (M - 1.0 / (1.0 + np.exp(-eta)))
            g = [R.sum()]
            if has["site"]:
                g.extend(R.sum(axis=(0, 2)))
            if has["genotype"]:
                g.extend([R[:, :, 1].sum(), R[:, :, 2].sum()])
            if has["neighbour"]:
                g.append(np.sum(R * prev[:, :, None]))
            grad = -np.asarray(g) + penalty * pen_mask * coef
            return float(-ll + 0.5 * penalty * np.sum(pen_mask * coef * coef)), grad

        res = minimize(objective, self.coef.copy(), jac=True, method="L-BFGS-B",
                       options={"maxiter": max_iter, "gtol": 1e-6})
        self.coef = res.x
        return self


def haplotype_frequencies_em(genotypes: Sequence, weights=None, *, catalog: Sequence[str],
                             phased: Optional[Sequence[Optional[Tuple[str, str]]]] = None,
                             missingness="genotype",
                             missingness_features: Sequence[str] = NoCallModel.FEATURES,
                             missingness_penalty: float = 1.0, genotype_error: float = 0.0,
                             pseudocount: float = 0.0, init=None, max_iter: int = 500,
                             tol: float = 1e-7) -> Dict[str, object]:
    """Weighted ML haplotype frequencies from unphased (and phased) founders.

    EM over catalog diplotypes (Excoffier & Slatkin 1995, Mol Biol Evol
    12:921; Long et al. 1995; the E-step enumerates every diplotype
    compatible with the genotype data as in PL-EM, Qin, Niu & Liu 2002, AJHG
    71:1242, without the partition-ligation since the catalog bounds the
    support). Under Hardy-Weinberg, P(diplotype {a, b}) = f_a f_b (x2 if
    a != b); person i contributes log sum_d P(d) P(data_i | d) with
    P(data_i | d) = prod_l [genotype likelihood at called loci] x
    P(NO_CALL pattern | genotypes of d) (``NoCallModel``; omitted when
    ``missingness`` is ``"mcar"`` / None). Each person's log-likelihood is
    multiplied by its sampling weight (pseudo-likelihood / IPW, Pfeffermann
    1993, Int Stat Rev 61:317), so the M-step is the weighted expected
    haplotype count. Phased persons (``phased[i] = (a, b)``) enter as a fixed
    diplotype; their genotypes (if given) still inform the NO_CALL model.
    Coalescent-prior (PHASE; Stephens et al. 2001, AJHG 68:978) or
    haplotype-cluster (Beagle; Browning & Browning 2007, AJHG 81:1084)
    priors are not needed here because the catalog is small and known.

    Parameters
    ----------
    genotypes
        Per person: None, (L,) dosage (-1/NaN = NO_CALL), (L, 3) likelihood
        (NaN row = NO_CALL) or ``(likelihood, called)``.
    weights
        Sampling weights (e.g. ``two_phase_weights(pi_inclusion)``).
    catalog
        Haplotype support (strings over {'0','1'}).
    phased
        Optional per-person known diplotype (or None).
    missingness
        ``"genotype"`` (fit ``NoCallModel`` in the EM), ``"mcar"`` / None
        (ignorable), or a ``NoCallModel`` kept fixed.
    pseudocount
        Added to every haplotype's expected count in the M-step.

    Returns
    -------
    dict with ``catalog``, ``frequencies``, ``diplotypes`` (iu, ju index
    arrays), ``unphased_index`` (persons resolved by EM), ``posterior``
    (n_unphased, n_diplotypes), ``evidence`` (log P(data | diplotype), same
    shape), ``missingness`` (``NoCallModel`` or None), ``loglik``,
    ``n_iter`` and ``unresolved`` (persons without any compatible
    diplotype, or without genotype data).
    """
    cat = list(catalog)
    index = {h: i for i, h in enumerate(cat)}
    H = len(cat)
    bits = _as_bits(cat).astype(np.int16)
    L = bits.shape[1]
    iu, ju = np.triu_indices(H)
    D = bits[iu] + bits[ju]                         # (n_dip, L) copy numbers
    n_dip = len(iu)
    dip_index = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iu, ju))}
    log2het = np.where(iu != ju, math.log(2.0), 0.0)
    genotypes = list(genotypes)
    n = len(genotypes)
    phased = [None] * n if phased is None else list(phased)
    if len(phased) != n:
        raise ValueError("phased must have one entry per person")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    fixed = np.full(n, -1)
    for i, p in enumerate(phased):
        if p is not None:
            a, b = index.get(p[0]), index.get(p[1])
            if a is None or b is None:
                raise ValueError(f"phased haplotypes of person {i} are not in the catalog")
            fixed[i] = dip_index[(min(a, b), max(a, b))]
    lik = np.full((n, L, 3), np.nan)
    called = np.zeros((n, L), dtype=bool)
    assayed = np.zeros(n, dtype=bool)
    for i, g in enumerate(genotypes):
        if g is not None:
            lik[i], called[i] = _genotype_evidence(g, L, genotype_error)
            assayed[i] = True
    logGL = np.where(called[:, :, None], np.log(np.maximum(np.nan_to_num(lik, nan=1.0), 1e-300)), 0.0)

    cand = np.flatnonzero((fixed < 0) & assayed)
    E0 = np.zeros((len(cand), n_dip))
    for l in range(L):
        E0 += logGL[cand, l][:, D[:, l]]
    # unresolved: no catalog diplotype is (numerically) compatible
    ok = E0.max(axis=1) > math.log(1e-300) * 0.5 if len(cand) else np.zeros(0, dtype=bool)
    unresolved = sorted(set(cand[~ok].tolist()) | set(np.flatnonzero((fixed < 0) & ~assayed).tolist()))
    cand = cand[ok]
    E0 = E0[ok]
    wc = w[cand]
    fx = np.flatnonzero(fixed >= 0)
    fixed_mass = np.zeros(n_dip)
    np.add.at(fixed_mass, fixed[fx], w[fx])
    onehot = np.stack([(D == k) for k in range(3)], axis=2).astype(float)  # (n_dip, L, 3)
    fx_assayed = fx[assayed[fx]]

    model = None
    fit_model = False
    if isinstance(missingness, NoCallModel):
        model = missingness
    elif missingness == "genotype":
        model = NoCallModel(L, missingness_features)
        fit_model = True
    elif missingness not in (None, "mcar", "none"):
        raise ValueError("missingness must be 'genotype', 'mcar', None or a NoCallModel")
    miss_rows = np.concatenate([cand, fx_assayed])

    def evidence():
        if model is None:
            return E0
        le = model.log_evidence(called[cand])
        E = E0.copy()
        for l in range(L):
            E += le[:, l][:, D[:, l]]
        return E

    if init is not None:
        f = np.asarray(init, dtype=float)
    else:
        cnt = np.full(H, 1.0)
        np.add.at(cnt, iu, fixed_mass)
        np.add.at(cnt, ju, fixed_mass)
        f = cnt / cnt.sum()
    loglik = -np.inf
    it = 0
    E = evidence()
    for it in range(1, max_iter + 1):
        lf = np.log(np.maximum(f, 1e-300))
        lp = E + (lf[iu] + lf[ju] + log2het)[None, :]
        ll_i = _logsumexp(lp, axis=1) if len(cand) else np.zeros(0)
        post = np.exp(lp - ll_i[:, None]) if len(cand) else np.zeros((0, n_dip))
        prior_fixed = lf[iu] + lf[ju] + log2het
        loglik = float(np.sum(wc * ll_i) + np.sum(w[fx] * prior_fixed[fixed[fx]]))
        dm = (wc[:, None] * post).sum(axis=0) + fixed_mass
        cnt = np.full(H, float(pseudocount))
        np.add.at(cnt, iu, dm)
        np.add.at(cnt, ju, dm)
        f_new = cnt / cnt.sum()
        delta = float(np.max(np.abs(f_new - f)))
        f = f_new
        if fit_model and len(miss_rows):
            gpost = np.concatenate([np.einsum("nd,dlk->nlk", post, onehot), onehot[fixed[fx_assayed]]])
            old = model.coef.copy()
            model.fit(gpost, called[miss_rows], w[miss_rows], penalty=missingness_penalty)
            delta = max(delta, float(np.max(np.abs(model.coef - old))) * 1e-2)
            E = evidence()
        if delta < tol:
            break
    lf = np.log(np.maximum(f, 1e-300))
    lp = E + (lf[iu] + lf[ju] + log2het)[None, :]
    post = np.exp(lp - _logsumexp(lp, axis=1)[:, None]) if len(cand) else np.zeros((0, n_dip))
    return {"catalog": cat, "frequencies": f, "diplotypes": (iu, ju), "unphased_index": cand,
            "posterior": post, "evidence": E, "missingness": model, "loglik": loglik,
            "n_iter": it, "unresolved": unresolved}


# --------------------------------------------------------------------------
# covariate basis
# --------------------------------------------------------------------------
_BASES = ("constant", "linear", "quadratic", "quadratic_rbf")


def _farthest_points(x: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(x))
    first = int(np.argmin(np.sum((x - x.mean(axis=0)) ** 2, axis=1)))
    chosen = [first]
    dist = np.sum((x - x[first]) ** 2, axis=1)
    while len(chosen) < count:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.sum((x - x[nxt]) ** 2, axis=1))
    return x[np.asarray(chosen)]


class _Basis:
    """Standardised polynomial (+ RBF) design on continuous covariates."""

    def __init__(self, kind: str, x_train: np.ndarray, n_centers: int = 10):
        if kind not in _BASES:
            raise ValueError(f"basis must be one of {_BASES}")
        self.kind = kind
        x_train = np.atleast_2d(np.asarray(x_train, dtype=float))
        self.mean = x_train.mean(axis=0)
        sd = x_train.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        self.centers = None
        self.bandwidth = None
        if kind == "quadratic_rbf" and len(x_train) > 1:
            z = (x_train - self.mean) / self.sd
            self.centers = _farthest_points(z, n_centers)
            d = np.sum((z[:, None, :] - self.centers[None, :, :]) ** 2, axis=2)
            # bandwidth from the data geometry: median squared distance to the
            # nearest center (scale-free, no tuning to a data set)
            self.bandwidth = max(float(np.median(np.min(d, axis=1))), 1e-3) * 2.0

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = len(x)
        cols = [np.ones((n, 1))]
        if self.kind == "constant":
            return cols[0]
        z = (x - self.mean) / self.sd
        cols.append(z)
        if self.kind in ("quadratic", "quadratic_rbf"):
            iu = np.triu_indices(z.shape[1])
            cols.append(z[:, iu[0]] * z[:, iu[1]])
        if self.kind == "quadratic_rbf" and self.centers is not None:
            d = np.sum((z[:, None, :] - self.centers[None, :, :]) ** 2, axis=2)
            cols.append(np.exp(-d / self.bandwidth))
        return np.hstack(cols)


def _fit_softmax(design: np.ndarray, left: np.ndarray, right: np.ndarray,
                 weights: np.ndarray, n_classes: int, penalty: float,
                 init: Optional[np.ndarray] = None, max_iter: int = 200) -> np.ndarray:
    """Weighted L2-penalised multinomial logit; each row emits two labels."""
    from scipy.optimize import minimize

    n, d = design.shape
    total = max(float(weights.sum()), 1e-12)
    rows = np.arange(n)
    if init is None:
        counts = np.full(n_classes, 0.5 / max(n_classes, 1))
        np.add.at(counts, left, weights)
        np.add.at(counts, right, weights)
        init = np.zeros((d, n_classes))
        init[0] = np.log(counts / counts.sum())
    pen = np.ones((d, n_classes))
    pen[0] = 1e-3  # intercepts nearly free; keeps unseen classes finite

    def objective(flat):
        beta = flat.reshape(d, n_classes)
        logits = design @ beta
        mx = logits.max(axis=1)
        lse = mx + np.log(np.exp(logits - mx[:, None]).sum(axis=1))
        loss = np.sum(weights * (2.0 * lse - logits[rows, left] - logits[rows, right])) / total
        loss += 0.5 * penalty * np.sum(pen * beta * beta)
        prob = np.exp(logits - lse[:, None])
        resid = 2.0 * weights[:, None] * prob
        np.subtract.at(resid, (rows, left), weights)
        np.subtract.at(resid, (rows, right), weights)
        grad = design.T @ resid / total + penalty * pen * beta
        return float(loss), grad.ravel()

    res = minimize(objective, np.asarray(init, dtype=float).ravel(), jac=True,
                   method="L-BFGS-B", options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-7})
    return res.x.reshape(d, n_classes)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------
# population haplotype prior
# --------------------------------------------------------------------------
class HaplotypePrior:
    """Fitted covariate / group-dependent population haplotype prior.

    p(h | x, g) = sum_c P(c | g) * p_c(h | x), with the latent class c shared
    by all members of a family (e.g. both parents of a couple) and
    p_c(. | x) = softmax(B(x) beta_c). Haplotypes of one person are i.i.d.
    draws given (c, x), so a diplotype {a, b} has probability
    p_c(a|x) p_c(b|x) * (2 if a != b else 1).
    """

    def __init__(self, catalog, basis, coefs, group_mixing, overall_mixing,
                 selection, population_frequencies):
        self.catalog = list(catalog)
        self.index = {h: i for i, h in enumerate(self.catalog)}
        self.basis = basis
        self.coefs = coefs  # (K, D, H)
        self.group_mixing = dict(group_mixing)
        self.overall_mixing = np.asarray(overall_mixing, dtype=float)
        self.selection = selection
        self.population_frequencies = np.asarray(population_frequencies, dtype=float)
        self.missingness = None

    @property
    def n_components(self) -> int:
        return int(self.coefs.shape[0])

    def mixing(self, group=None) -> np.ndarray:
        """P(class | group); the overall (design-weighted) mixing if unseen."""
        if group is not None and group in self.group_mixing:
            return np.asarray(self.group_mixing[group], dtype=float)
        return self.overall_mixing.copy()

    def component_probs(self, x) -> np.ndarray:
        """p_c(h | x) for every row of ``x``: array (n, K, H)."""
        design = self.basis(x)
        logits = np.einsum("nd,kdh->nkh", design, self.coefs)
        return _softmax_rows(logits)

    def haplotype_probs(self, x, group=None) -> np.ndarray:
        """Marginal prior p(h | x, g) (n, H) -- before any family evidence."""
        comp = self.component_probs(x)
        return np.einsum("k,nkh->nh", self.mixing(group), comp)

    def haplotype_prob(self, haplotype, x, group=None) -> float:
        j = self.index.get(haplotype)
        if j is None:
            return 0.0
        return float(self.haplotype_probs(np.atleast_2d(x), group)[0, j])

    def _pair_by_component(self, x, pair) -> np.ndarray:
        comp = self.component_probs(np.atleast_2d(x))[0]
        a, b = (self.index.get(h) for h in pair)
        if a is None or b is None:
            return np.zeros(self.n_components)
        val = comp[:, a] * comp[:, b]
        return val * (2.0 if a != b else 1.0)

    def diplotype_prob(self, pair, x, group=None) -> float:
        """Prior probability of the unordered diplotype ``pair``."""
        return float(self.mixing(group) @ self._pair_by_component(x, pair))

    def diplotype_matrix(self, x, group=None, component_weights=None) -> np.ndarray:
        """Upper-triangular (H, H) matrix of unordered diplotype priors."""
        comp = self.component_probs(np.atleast_2d(x))[0]
        w = self.mixing(group) if component_weights is None else np.asarray(component_weights)
        out = np.einsum("k,ki,kj->ij", w, comp, comp)
        out = np.triu(out) + np.triu(out, 1)
        return out

    def class_posterior(self, members: Sequence[Tuple[Sequence, Tuple[str, str]]], group=None) -> np.ndarray:
        """P(class | group, observed diplotypes of family members).

        ``members`` is a list of ``(x, (hap_a, hap_b))``.
        """
        logp = np.log(np.maximum(self.mixing(group), 1e-300))
        for x, pair in members:
            logp = logp + np.log(np.maximum(self._pair_by_component(x, pair), 1e-300))
        logp -= logp.max()
        p = np.exp(logp)
        return p / p.sum()

    def conditional_diplotype_prob(self, pair, x, given_pair, given_x, group=None) -> float:
        """P(diplotype ``pair`` of one member | ``given_pair`` of another member).

        Members of a family share the latent class, so the given diplotype
        updates the class weights; with one class this is the marginal prior.
        """
        num_given = self.mixing(group) * self._pair_by_component(given_x, given_pair)
        den = float(num_given.sum())
        if den <= 0:
            return 0.0
        return float(num_given @ self._pair_by_component(x, pair) / den)

    def joint_diplotype_prior(self, members_x: Sequence, candidates: Sequence[Sequence[Tuple[str, str]]],
                              group=None) -> np.ndarray:
        """Joint prior over candidate diplotypes of several family members.

        ``candidates[m]`` lists diplotypes of member m; returns an array of
        shape (len(candidates[0]), len(candidates[1]), ...) with
        sum_c P(c|g) prod_m P(pair_m | c, x_m).
        """
        mix = self.mixing(group)
        out = None
        for c in range(self.n_components):
            term = mix[c]
            factors = []
            for x, cands in zip(members_x, candidates):
                comp = self.component_probs(np.atleast_2d(x))[0, c]
                idx = [(self.index.get(a), self.index.get(b)) for a, b in cands]
                f = np.array([0.0 if (i is None or j is None) else comp[i] * comp[j] * (2.0 if i != j else 1.0)
                              for i, j in idx])
                factors.append(f)
            prod = factors[0]
            for f in factors[1:]:
                prod = np.multiply.outer(prod, f)
            out = term * prod if out is None else out + term * prod
        return out

    def population_average(self, x, groups=None, weights=None) -> np.ndarray:
        """Design-weighted average of p(h | x_i, g_i) over a sample of people.

        A model-based estimate of population haplotype frequencies when the
        covariates of a large (first-phase) sample are known.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
        if groups is None:
            groups = [None] * len(x)
        comp = self.component_probs(x)
        mix = np.array([self.mixing(g) for g in groups])
        probs = np.einsum("nk,nkh->nh", mix, comp)
        out = (w[:, None] * probs).sum(axis=0) / w.sum()
        return out / out.sum()


def _family_arrays(hap_a, hap_b, covariates, family_ids, groups, weights, catalog):
    index = {h: i for i, h in enumerate(catalog)}
    fam_order = []
    fam_pos = {}
    for f in family_ids:
        if f not in fam_pos:
            fam_pos[f] = len(fam_order)
            fam_order.append(f)
    row_family = np.array([fam_pos[f] for f in family_ids], dtype=int)
    left = np.array([index[h] for h in hap_a], dtype=int)
    right = np.array([index[h] for h in hap_b], dtype=int)
    n_fam = len(fam_order)
    fam_group = [None] * n_fam
    fam_weight = np.zeros(n_fam)
    for i, f in enumerate(family_ids):
        k = fam_pos[f]
        fam_group[k] = None if groups is None else groups[i]
        fam_weight[k] = weights[i]
    return fam_order, row_family, left, right, fam_group, fam_weight


def _class_mixing(resp, fam_group, fam_weight, alpha):
    groups = sorted({g for g in fam_group if g is not None}, key=str)
    K = resp.shape[1]
    overall = (fam_weight[:, None] * resp).sum(axis=0) + alpha
    overall /= overall.sum()
    mix = {}
    for g in groups:
        sel = np.array([fg == g for fg in fam_group])
        v = (fam_weight[sel, None] * resp[sel]).sum(axis=0) + alpha
        mix[g] = v / v.sum()
    return mix, overall


def _mixing_matrix(fam_group, mix, overall):
    return np.array([mix.get(g, overall) if g is not None else overall for g in fam_group])


def _row_loglik_components(comp_probs, left, right):
    """log p_c(a) p_c(b) * (2 if a!=b) for each row: (n, K)."""
    n = len(left)
    rows = np.arange(n)
    pa = comp_probs[rows, :, left]
    pb = comp_probs[rows, :, right]
    val = np.log(np.maximum(pa, 1e-300)) + np.log(np.maximum(pb, 1e-300))
    val += np.where(left != right, math.log(2.0), 0.0)[:, None]
    return val


def _family_loglik(row_ll, row_family, n_fam, mixing_rows):
    """Per-family log sum_c P(c|g) prod_rows p_c: returns (ll (F,), resp (F,K))."""
    K = row_ll.shape[1]
    fam_ll = np.zeros((n_fam, K))
    np.add.at(fam_ll, row_family, row_ll)
    fam_ll += np.log(np.maximum(mixing_rows, 1e-300))
    tot = _logsumexp(fam_ll, axis=1)
    resp = np.exp(fam_ll - tot[:, None])
    return tot, resp


def _count_mixture(left, right, row_family, n_fam, fam_group, fam_weight, K, H,
                   concentration, n_starts, seed, max_iter=300, mass=None):
    """Latent-class mixture of multinomials on family haplotype counts (no covariates).

    ``mass`` (per row, default 1) is the posterior weight of a soft diplotype
    row of an unphased person (expected counts, EM M-step).
    """
    counts = np.zeros((n_fam, H))
    m = np.ones(len(left)) if mass is None else np.asarray(mass, dtype=float)
    np.add.at(counts, (row_family, left), m)
    np.add.at(counts, (row_family, right), m)
    rng = np.random.default_rng(seed)
    best = None
    groups = sorted({g for g in fam_group if g is not None}, key=str)
    for start in range(n_starts if K > 1 else 1):
        if K == 1:
            resp = np.ones((n_fam, 1))
        elif start == 0 and groups:
            # initialise from the group labels (classes are unlabelled; any
            # permutation is equivalent)
            resp = np.full((n_fam, K), 0.1 / max(K - 1, 1))
            for f, g in enumerate(fam_group):
                j = (groups.index(g) if g in groups else 0) % K
                resp[f, j] = 0.9
            resp /= resp.sum(axis=1, keepdims=True)
        else:
            resp = rng.dirichlet(np.ones(K), size=n_fam)
        prev = -np.inf
        for _ in range(max_iter):
            prof = concentration + resp.T @ (counts * fam_weight[:, None])
            prof /= prof.sum(axis=1, keepdims=True)
            mix, overall = _class_mixing(resp, fam_group, fam_weight, 0.5)
            mrows = _mixing_matrix(fam_group, mix, overall)
            ll = counts @ np.log(prof).T + np.log(np.maximum(mrows, 1e-300))
            tot = _logsumexp(ll, axis=1)
            resp = np.exp(ll - tot[:, None])
            obj = float(np.sum(fam_weight * tot))
            if abs(obj - prev) < 1e-8 * max(1.0, abs(obj)):
                break
            prev = obj
        if best is None or obj > best[0]:
            best = (obj, resp, prof)
    return best[1], best[2]


def fit_haplotype_prior(hap_a: Sequence[Optional[str]], hap_b: Sequence[Optional[str]], covariates,
                        family_ids: Sequence, groups: Optional[Sequence] = None, weights=None, *,
                        catalog: Optional[Sequence[str]] = None,
                        n_components="auto", max_components: int = 5,
                        bases: Sequence[str] = ("constant", "linear", "quadratic", "quadratic_rbf"),
                        penalties: Sequence[float] = (3.0, 1.0, 0.3, 0.1, 0.03, 0.01),
                        interactions: Sequence[str] = ("shared", "per_class"),
                        concentrations: Sequence[float] = (0.5, 1.0, 2.0),
                        holdout_fraction: float = 0.2, n_rbf_centers: int = 10,
                        n_starts: int = 4, refine_iter: int = 3,
                        random_state: int = 0, verbose: bool = False,
                        genotypes: Optional[Sequence] = None,
                        missingness="genotype",
                        missingness_features: Sequence[str] = ("site", "genotype", "neighbour"),
                        missingness_penalty: float = 1.0,
                        genotype_error: float = 0.0,
                        em_max_iter: int = 500, em_tol: float = 1e-7,
                        em_rounds: int = 1, posterior_min: float = 1e-3) -> HaplotypePrior:
    """Learn a population haplotype prior that depends on covariates and a group label.

    Founders may be *phased* (both haplotypes known, e.g. phase-validated
    "anchor" families) or *unphased* (``hap_a[i] = hap_b[i] = None`` and a
    genotype entry in ``genotypes``). Unphased founders enter through their
    genotype likelihoods: the population frequencies are the weighted
    maximum-likelihood estimate of an EM over the catalog diplotypes
    compatible with each founder's genotypes (Excoffier & Slatkin 1995, Mol
    Biol Evol 12:921; the per-person diplotype enumeration is the E-step also
    used in PL-EM, Qin, Niu & Liu 2002, AJHG 71:1242), phased founders enter
    as fixed diplotypes, NO_CALL loci enter through ``missingness`` (a
    genotype- / site- / neighbour-dependent call-failure model, see
    ``NoCallModel``), and every founder is weighted by its sampling weight
    (pseudo-likelihood, Pfeffermann 1993, Int Stat Rev 61:317). The
    covariate / class model below is then fitted to the posterior-weighted
    diplotype rows (expected complete-data log-likelihood) and, for
    ``em_rounds`` more rounds, the diplotype posteriors are recomputed with
    the person-specific fitted prior p(h | x, g) and the model refitted at
    the selected configuration.

    With only phased input and ``genotypes=None`` the behaviour is exactly
    that of earlier versions.

    Parameters
    ----------
    hap_a, hap_b
        The two phased haplotypes of each person (one row per person); None
        for a person whose phase is unknown (then ``genotypes[i]`` is used).
    covariates
        (n_people, d) continuous person-level covariates (e.g. ancestry
        proportions, principal components).
    family_ids
        Family of each person. Members of a family share one latent class
        (founder / community structure), which is what makes the haplotypes
        of e.g. two parents dependent.
    groups
        Optional family-level categorical label (e.g. a noisy community
        proxy, recruitment site) on which the class mixing depends.
    weights
        Per-person sampling weights (equal within a family), e.g.
        ``two_phase_weights(pi_inclusion, pi_validation)`` when only the
        phase-validated (second-phase) families are passed. When ALL founders
        are passed (phased + unphased), use first-phase weights
        ``two_phase_weights(pi_inclusion)``: second-phase selection only
        decides which founders have phase revealed, which is ignorable for
        the likelihood when it depends on observed data only. The estimand
        is the population *before* sampling, so biased (ascertained)
        inclusion must be undone here, not modelled as a covariate.
    n_components
        Number of latent classes, or ``"auto"`` (1..max_components chosen on
        held-out families).
    bases, penalties
        Candidate covariate bases (``constant`` = class profile without
        covariates, ``linear``, ``quadratic``, ``quadratic_rbf``; covariates
        are standardised) and L2 penalties for the multinomial-logit surface.
    interactions
        ``shared``: class-specific haplotype profiles plus one covariate
        surface common to all classes; ``per_class``: an independent surface
        per class. Both are scored on the held-out families.
    holdout_fraction
        Fraction of families (hash split, never splitting a family) used to
        score candidates by design-weighted held-out family log-likelihood.
        Among candidates within one standard error of the best, the simplest
        is chosen (fewer classes, then simpler basis, then stronger penalty);
        the chosen configuration is refitted on all families.
    refine_iter
        EM iterations that update class responsibilities with the fitted
        surfaces after selection.
    genotypes
        Optional per-person genotype evidence (needed for unphased persons,
        optional for phased ones where it informs the NO_CALL model). Each
        entry is None, an (L,) dosage vector (0/1/2 copies of allele '1',
        -1 or NaN = NO_CALL), an (L, 3) genotype-likelihood array (NaN row =
        NO_CALL) or the ``(likelihood, called)`` tuple of
        ``genotype_likelihood_arrays``.
    missingness
        ``"genotype"`` (default): fit ``NoCallModel`` jointly in the EM and
        use P(NO_CALL | genotype, site, previous call) inside the genotype
        likelihood; ``"mcar"`` / None: NO_CALL loci are uninformative
        (missing completely at random); or a fitted ``NoCallModel`` (kept
        fixed).
    missingness_features, missingness_penalty
        Terms of the NO_CALL logistic model (any of ``site``, ``genotype``,
        ``neighbour``) and the Gaussian-prior (L2) penalty on its
        non-intercept coefficients (1.0 = unit-variance prior on log-odds).
    genotype_error
        Per-locus error for hard dosage input (likelihood 1-e on the
        reported genotype, e/2 on the other two). 0 = exact.
    em_max_iter, em_tol
        Limits of the frequency EM (max change of any frequency).
    em_rounds
        Rounds of person-specific E-step + refit after the selection fit.
    posterior_min
        Diplotype candidates of an unphased person with posterior below
        this are dropped (the rest renormalised) before the covariate fit.

    Returns
    -------
    HaplotypePrior
        ``.haplotype_probs(x, group)``, ``.diplotype_prob``,
        ``.conditional_diplotype_prob``, ``.joint_diplotype_prior``,
        ``.population_average``; ``.selection`` holds the held-out scores;
        ``.population_frequencies`` the (EM) design-weighted frequencies;
        ``.missingness`` the fitted ``NoCallModel`` (or None).
    """
    hap_a = list(hap_a)
    hap_b = list(hap_b)
    n = len(hap_a)
    x = np.atleast_2d(np.asarray(covariates, dtype=float))
    if x.shape[0] != n:
        x = x.T if x.shape[1] == n else x
    if x.shape[0] != n or len(hap_b) != n or len(family_ids) != n:
        raise ValueError("hap_a, hap_b, covariates and family_ids must have one row per person")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.mean()
    family_ids = list(family_ids)
    groups = None if groups is None else list(groups)
    settings = dict(n_components=n_components, max_components=max_components, bases=bases,
                    penalties=penalties, interactions=interactions, concentrations=concentrations,
                    holdout_fraction=holdout_fraction, n_rbf_centers=n_rbf_centers, n_starts=n_starts,
                    refine_iter=refine_iter, random_state=random_state, verbose=verbose)
    phased = [None if (a is None or b is None) else (a, b) for a, b in zip(hap_a, hap_b)]
    if genotypes is None and all(p is not None for p in phased):
        cat, pop_freq = haplotype_frequencies(hap_a, hap_b, w, catalog=catalog)
        model = _fit_prior_rows(hap_a, hap_b, np.ones(n), x, family_ids, groups, w, cat, pop_freq, **settings)
        model.missingness = None
        return model

    # ---- unphased founders: diplotype EM over genotype likelihoods
    if catalog is None:
        cat = sorted({h for p in phased if p is not None for h in p})
    else:
        cat = list(catalog)
    if not cat:
        raise ValueError("a haplotype catalog is needed: pass catalog= or some phased founders")
    if genotypes is None:
        genotypes = [None] * n
    if len(genotypes) != n:
        raise ValueError("genotypes must have one entry per person")
    em = haplotype_frequencies_em(genotypes, w, catalog=cat, phased=phased, missingness=missingness,
                                  missingness_features=missingness_features,
                                  missingness_penalty=missingness_penalty, genotype_error=genotype_error,
                                  max_iter=em_max_iter, tol=em_tol)
    pop_freq = em["frequencies"]
    iu, ju = em["diplotypes"]
    unph = em["unphased_index"]
    usable = np.array([p is not None for p in phased]) | np.isin(np.arange(n), unph)

    def expand(post):
        ra, rb, rm, rp = [], [], [], []
        for i in np.flatnonzero([p is not None for p in phased]):
            ra.append(phased[i][0]); rb.append(phased[i][1]); rm.append(1.0); rp.append(i)
        for r, i in enumerate(unph):
            pr = post[r]
            keep = np.flatnonzero(pr >= min(posterior_min, pr.max()))
            mass = pr[keep] / pr[keep].sum()
            for d, m in zip(keep, mass):
                ra.append(cat[iu[d]]); rb.append(cat[ju[d]]); rm.append(float(m)); rp.append(i)
        rp = np.asarray(rp, dtype=int)
        return ra, rb, np.asarray(rm), rp

    def fit(post, fixed=None):
        ra, rb, rm, rp = expand(post)
        return _fit_prior_rows(ra, rb, rm, x[rp], [family_ids[i] for i in rp],
                               None if groups is None else [groups[i] for i in rp], w[rp], cat, pop_freq,
                               fixed=fixed, **settings)

    model = fit(em["posterior"])
    post = em["posterior"]
    log2het = np.where(iu != ju, math.log(2.0), 0.0)
    for _ in range(max(0, int(em_rounds))):
        if len(unph) == 0:
            break
        gu = [None] * len(unph) if groups is None else [groups[i] for i in unph]
        probs = np.zeros((len(unph), len(cat)))
        for g in set(gu):
            sel = np.array([gg == g for gg in gu])
            probs[sel] = model.haplotype_probs(x[unph[sel]], g)
        lp = np.log(np.maximum(probs, 1e-300))
        logpost = em["evidence"] + lp[:, iu] + lp[:, ju] + log2het[None, :]
        logpost -= _logsumexp(logpost, axis=1)[:, None]
        post = np.exp(logpost)
        model = fit(post, fixed=model.selection)
    model.population_frequencies = np.asarray(pop_freq, dtype=float)
    model.missingness = em["missingness"]
    model.selection.update({
        "n_phased": int(sum(p is not None for p in phased)),
        "n_unphased": int(len(unph)),
        "n_unresolved": int(n - usable.sum()),
        "em_iterations": int(em["n_iter"]),
        "em_loglik": float(em["loglik"]),
        "em_rounds": int(em_rounds),
    })
    return model


def _fit_prior_rows(hap_a, hap_b, mass, x, family_ids, groups, w, cat, pop_freq, *,
                    n_components, max_components, bases, penalties, interactions, concentrations,
                    holdout_fraction, n_rbf_centers, n_starts, refine_iter, random_state, verbose,
                    fixed=None) -> HaplotypePrior:
    """Covariate / latent-class prior on (possibly soft) diplotype rows.

    Row r is one candidate diplotype (hap_a[r], hap_b[r]) of a person with
    posterior mass ``mass[r]`` (1 for a phased person); all likelihood terms
    are mass-weighted (expected complete-data log-likelihood, the EM
    M-step). ``fixed`` (a previous ``selection``) skips model selection.
    """
    H = len(cat)
    mass = np.asarray(mass, dtype=float)
    fam_order, row_family, left, right, fam_group, fam_weight = _family_arrays(
        hap_a, hap_b, x, family_ids, groups, w, cat)
    n_fam = len(fam_order)
    wm = w * mass

    n_folds = max(2, int(round(1.0 / max(min(holdout_fraction, 0.5), 1e-3))))
    valid_fam = np.array([_stable_fold(str(f), n_folds, random_state) == 0 for f in fam_order])
    if valid_fam.all() or (~valid_fam).all():
        valid_fam = np.arange(n_fam) % n_folds == 0
    train_rows = ~valid_fam[row_family]

    def subset(fmask):
        rows = fmask[row_family]
        remap = -np.ones(n_fam, dtype=int)
        remap[fmask] = np.arange(int(fmask.sum()))
        return rows, remap[row_family[rows]], [fam_group[i] for i in np.flatnonzero(fmask)], fam_weight[fmask]

    tr_rows, tr_rf, tr_groups, tr_fw = subset(~valid_fam)
    va_rows, va_rf, va_groups, va_fw = subset(valid_fam)

    def score(comp_va, mix, overall):
        mrows = _mixing_matrix(va_groups, mix, overall)
        rl = _row_loglik_components(comp_va, left[va_rows], right[va_rows]) * mass[va_rows, None]
        tot, _ = _family_loglik(rl, va_rf, int(valid_fam.sum()), mrows)
        return va_fw * tot / va_fw.mean()

    if fixed is None:
        # ---- stage A: number of latent classes from co-occurrence (no covariates)
        if n_components == "auto":
            k_grid = list(range(1, max(1, int(max_components)) + 1))
        else:
            k_grid = [int(n_components)]
        stage_a = []
        for K in k_grid:
            for conc in concentrations:
                resp, prof = _count_mixture(left[tr_rows], right[tr_rows], tr_rf, len(tr_fw), tr_groups,
                                            tr_fw, K, H, conc, n_starts, random_state, mass=mass[tr_rows])
                mix, overall = _class_mixing(resp, tr_groups, tr_fw, 0.5)
                comp_va = np.broadcast_to(prof[None], (int(va_rows.sum()), K, H))
                s = score(comp_va, mix, overall)
                stage_a.append({"K": K, "concentration": conc, "scores": s, "mean": float(s.mean())})
                if verbose:
                    print(f"[prior] K={K} conc={conc} heldout={s.mean():.4f}")
        chosen_a = _one_se_choice(stage_a, key=lambda c: (c["K"], -c["concentration"]))
        K = chosen_a["K"]
        # ---- stage B: covariate surface within classes
        resp_tr, _ = _count_mixture(left[tr_rows], right[tr_rows], tr_rf, len(tr_fw), tr_groups, tr_fw,
                                    K, H, chosen_a["concentration"], n_starts, random_state, mass=mass[tr_rows])
        mix_tr, overall_tr = _class_mixing(resp_tr, tr_groups, tr_fw, 0.5)
        stage_b = []
        for kind in bases:
            basis = _Basis(kind, x[train_rows], n_rbf_centers)
            d_tr = basis(x[tr_rows])
            d_va = basis(x[va_rows])
            pens = [penalties[0]] if kind == "constant" else list(penalties)
            modes = ["per_class"] if (kind == "constant" or K == 1) else list(interactions)
            for mode in modes:
                warm = None
                for pen in pens:
                    coefs = _fit_surfaces(d_tr, left[tr_rows], right[tr_rows], wm[tr_rows],
                                          resp_tr[tr_rf], H, pen, mode, warm)
                    warm = coefs
                    comp_va = _softmax_rows(np.einsum("nd,kdh->nkh", d_va, coefs))
                    s = score(comp_va, mix_tr, overall_tr)
                    stage_b.append({"basis": kind, "interaction": mode, "penalty": pen, "scores": s,
                                    "mean": float(s.mean())})
                    if verbose:
                        print(f"[prior] basis={kind} interaction={mode} penalty={pen} heldout={s.mean():.4f}")
        order = {b: i for i, b in enumerate(_BASES)}
        chosen_b = _one_se_choice(
            stage_b, key=lambda c: (order[c["basis"]], 0 if c["interaction"] == "shared" else 1, -c["penalty"]))
        chosen = {"n_components": K, "concentration": chosen_a["concentration"], "basis": chosen_b["basis"],
                  "interaction": chosen_b["interaction"], "penalty": chosen_b["penalty"]}
        stages = {"stage_components": [{k: v for k, v in c.items() if k != "scores"} for c in stage_a],
                  "stage_surface": [{k: v for k, v in c.items() if k != "scores"} for c in stage_b]}
    else:
        chosen = {k: fixed[k] for k in ("n_components", "concentration", "basis", "interaction", "penalty")}
        stages = {k: fixed.get(k, []) for k in ("stage_components", "stage_surface")}
        K = int(chosen["n_components"])

    # ---- refit on all families, EM refinement of the class responsibilities
    resp_all, _ = _count_mixture(left, right, row_family, n_fam, fam_group, fam_weight, K, H,
                                 chosen["concentration"], n_starts, random_state, mass=mass)
    basis = _Basis(chosen["basis"], x, n_rbf_centers)
    design = basis(x)
    resp = resp_all
    coef_arr = None
    for it in range(max(1, refine_iter + 1)):
        coef_arr = _fit_surfaces(design, left, right, wm, resp[row_family], H, chosen["penalty"],
                                 chosen["interaction"], coef_arr)
        mix, overall = _class_mixing(resp, fam_group, fam_weight, 0.5)
        if it == refine_iter or K == 1:
            break
        comp = _softmax_rows(np.einsum("nd,kdh->nkh", design, coef_arr))
        rl = _row_loglik_components(comp, left, right) * mass[:, None]
        _, resp = _family_loglik(rl, row_family, n_fam, _mixing_matrix(fam_group, mix, overall))
    selection = dict(chosen)
    selection.update({"n_families": n_fam, "n_holdout_families": int(valid_fam.sum())})
    selection.update(stages)
    return HaplotypePrior(cat, basis, coef_arr, mix, overall, selection, pop_freq)


def _fit_surfaces(design, left, right, w_rows, resp_rows, H, penalty, mode, init=None):
    """Within-class multinomial-logit surfaces, (K, D, H).

    ``per_class``: an independent surface per class. ``shared``: class-specific
    intercepts (class profiles) plus one covariate surface shared by all
    classes (covariate effects do not interact with the latent class).
    """
    K = resp_rows.shape[1]
    if mode == "per_class" or K == 1 or design.shape[1] == 1:
        out = []
        for c in range(K):
            out.append(_fit_softmax(design, left, right, w_rows * resp_rows[:, c], H, penalty,
                                    None if init is None else init[c]))
        return np.array(out)
    if mode != "shared":
        raise ValueError("interaction must be 'per_class' or 'shared'")
    from scipy.optimize import minimize

    n, D = design.shape
    Z = design[:, 1:]
    rows = np.arange(n)
    wr = w_rows[:, None] * resp_rows  # (n, K)
    total = max(float(wr.sum()), 1e-12)
    if init is None:
        alpha = np.zeros((K, H))
        for c in range(K):
            counts = np.full(H, 0.5 / H)
            np.add.at(counts, left, wr[:, c])
            np.add.at(counts, right, wr[:, c])
            alpha[c] = np.log(counts / counts.sum())
        beta = np.zeros((D - 1, H))
    else:
        alpha = np.asarray(init)[:, 0, :].copy()
        beta = np.asarray(init)[:, 1:, :].mean(axis=0)
    na = K * H

    def objective(flat):
        a = flat[:na].reshape(K, H)
        b = flat[na:].reshape(D - 1, H)
        logits = a[None, :, :] + (Z @ b)[:, None, :]  # (n, K, H)
        mx = logits.max(axis=2)
        lse = mx + np.log(np.exp(logits - mx[:, :, None]).sum(axis=2))
        la = logits[rows, :, left]
        lb = logits[rows, :, right]
        loss = np.sum(wr * (2.0 * lse - la - lb)) / total
        loss += 0.5 * penalty * (np.sum(b * b) + 1e-3 * np.sum(a * a))
        prob = np.exp(logits - lse[:, :, None])
        resid = 2.0 * wr[:, :, None] * prob
        for c in range(K):
            np.subtract.at(resid[:, c, :], (rows, left), wr[:, c])
            np.subtract.at(resid[:, c, :], (rows, right), wr[:, c])
        ga = resid.sum(axis=0) / total + penalty * 1e-3 * a
        gb = Z.T @ resid.sum(axis=1) / total + penalty * b
        return float(loss), np.concatenate([ga.ravel(), gb.ravel()])

    res = minimize(objective, np.concatenate([alpha.ravel(), beta.ravel()]), jac=True,
                   method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-7})
    a = res.x[:na].reshape(K, H)
    b = res.x[na:].reshape(D - 1, H)
    return np.stack([np.vstack([a[c][None, :], b]) for c in range(K)])


def _one_se_choice(candidates, key):
    """Simplest candidate within one standard error (paired) of the best."""
    best = max(candidates, key=lambda c: c["mean"])
    ok = []
    for c in candidates:
        diff = c["scores"] - best["scores"]
        se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
        if c["mean"] >= best["mean"] - se:
            ok.append(c)
    return min(ok, key=key)


# --------------------------------------------------------------------------
# meiosis
# --------------------------------------------------------------------------
def gamete_sequences(hap_1: str, hap_2: str, max_crossovers: int = 1) -> List[str]:
    """Distinct transmitted sequences from homologues with <= k crossovers."""
    L = len(hap_1)
    out = {hap_1, hap_2}
    for k in range(1, max_crossovers + 1):
        for cuts in combinations(range(1, L), k):
            for start in (0, 1):
                src = [hap_1, hap_2]
                cur = start
                pieces = []
                prev = 0
                for cut in list(cuts) + [L]:
                    pieces.append(src[cur][prev:cut])
                    cur = 1 - cur
                    prev = cut
                out.add("".join(pieces))
    return sorted(out)


def gamete_distribution(hap_1: str, hap_2: str, no_crossover: float, interval_masses) -> Tuple[List[str], np.ndarray]:
    """Probability of every transmitted sequence of one meiosis (<= 1 crossover).

    The meiosis starts on either homologue with probability 1/2, has no
    crossover with mass ``no_crossover`` or one crossover in interval j
    (between loci j and j+1) with mass ``interval_masses[j]``. Works for any
    pair of sequences (inside or outside a catalog). Returns
    ``(sequences, probabilities)`` with duplicates merged; probabilities sum
    to ``no_crossover + sum(interval_masses)`` (normally 1).
    """
    m = np.asarray(interval_masses, dtype=float)
    L = len(hap_1)
    if len(m) != L - 1:
        raise ValueError("interval_masses must have n_loci - 1 entries")
    out: Dict[str, float] = {}
    out[hap_1] = out.get(hap_1, 0.0) + 0.5 * no_crossover
    out[hap_2] = out.get(hap_2, 0.0) + 0.5 * no_crossover
    for j in range(1, L):
        for seq in (hap_1[:j] + hap_2[j:], hap_2[:j] + hap_1[j:]):
            out[seq] = out.get(seq, 0.0) + 0.5 * m[j - 1]
    seqs = sorted(out)
    return seqs, np.array([out[q] for q in seqs])


def offspring_outcome_probability(maternal_pair: Tuple[str, str], paternal_pair: Tuple[str, str],
                                  outcome, maternal_crossover: Dict[str, object],
                                  paternal_crossover: Dict[str, object]) -> float:
    """Expected per-offspring outcome probability for one joint parental state.

    ``sum_{gm, gp} P(gm | mother) P(gp | father) * outcome(gm, gp)`` over the
    transmitted sequences of both meioses (``gamete_distribution`` with the
    ``fit_crossover_intervals`` results of each role). ``outcome`` is any
    callable of the two transmitted sequences returning a probability (e.g. a
    logistic penetrance, averaged over a binary offspring covariate when that
    covariate is unobserved). Computed from the supplied sequences only, so it
    is defined for every possible state, inside or outside a catalog or a
    pruned candidate list.
    """
    gm, pm = gamete_distribution(maternal_pair[0], maternal_pair[1], maternal_crossover["no_crossover"],
                                 maternal_crossover["interval_masses"])
    gp, pp = gamete_distribution(paternal_pair[0], paternal_pair[1], paternal_crossover["no_crossover"],
                                 paternal_crossover["interval_masses"])
    total = 0.0
    for a, wa in zip(gm, pm):
        for b, wb in zip(gp, pp):
            total += wa * wb * float(outcome(a, b))
    return total / max(float(pm.sum() * pp.sum()), 1e-300)


def genotype_likelihood_arrays(sample_ids, locus_ids, status, lik0, lik1, lik2, loci_order,
                               called_value: str = "CALLED") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Long-format genotype-likelihood rows -> per-sample (n_loci, 3) arrays.

    One row per (sample, locus); ``status == called_value`` marks a called
    locus, anything else (e.g. NO_CALL) is uncalled with NaN likelihoods.
    Returns ``{sample_id: (likelihood (L, 3), called (L,) bool)}`` in
    ``loci_order``; loci without a row are uncalled. Build ONE dict over every
    assay table (parents and children) so each person is looked up by id.
    """
    pos = {l: i for i, l in enumerate(loci_order)}
    L = len(pos)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sid, loc, st, a, b, c in zip(sample_ids, locus_ids, status, lik0, lik1, lik2):
        if sid not in out:
            out[sid] = (np.full((L, 3), np.nan), np.zeros(L, dtype=bool))
        i = pos.get(loc)
        if i is None:
            continue
        if str(st) == called_value:
            vals = []
            for v in (a, b, c):
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    vals.append(np.nan)
            out[sid][0][i] = vals
            out[sid][1][i] = bool(np.all(np.isfinite(vals)))
    return out


def fit_crossover_intervals(parent_pairs: Sequence[Tuple[str, str]], transmitted: Sequence[str],
                            family_ids: Optional[Sequence] = None, *,
                            concentrations: Sequence[float] = (0.01, 0.05, 0.2, 1.0),
                            holdout_fraction: float = 0.2, random_state: int = 0,
                            max_iter: int = 2000) -> Dict[str, object]:
    """No-crossover / single-crossover interval masses from ordered transmissions.

    Model: a meiosis starts on either homologue with probability 1/2 and has
    either no crossover (mass m0) or exactly one crossover in interval j
    (mass m_j, between loci j and j+1); sum m = 1. A transmitted sequence can
    be consistent with several categories when the homologues share alleles
    around a cut; EM marginalises over them instead of counting hard calls.
    A symmetric Dirichlet concentration is chosen on held-out families.

    Returns dict with ``no_crossover`` (m0), ``interval_masses`` (L-1),
    ``concentration`` and ``n_informative``.
    """
    rows = []
    fams = []
    for i, ((h1, h2), t) in enumerate(zip(parent_pairs, transmitted)):
        L = len(t)
        e = np.zeros(L)
        e[0] = 0.5 * ((t == h1) + (t == h2))
        for cut in range(1, L):
            e[cut] = 0.5 * ((h1[:cut] + h2[cut:] == t) + (h2[:cut] + h1[cut:] == t))
        if e.max() > 0:
            rows.append(e)
            fams.append(str(family_ids[i]) if family_ids is not None else str(i))
    if not rows:
        raise ValueError("no transmission is consistent with its parent's haplotypes")
    E = np.array(rows)
    L = E.shape[1]

    def em(mask, conc):
        sel = E[mask]
        m = np.concatenate(([0.5], np.full(L - 1, 0.5 / max(L - 1, 1))))
        for _ in range(max_iter):
            r = sel * m
            r /= np.maximum(r.sum(axis=1, keepdims=True), 1e-300)
            new = (r.sum(axis=0) + conc) / (len(sel) + conc * L)
            if np.max(np.abs(new - m)) < 1e-12:
                m = new
                break
            m = new
        return m

    n_folds = max(2, int(round(1.0 / max(min(holdout_fraction, 0.5), 1e-3))))
    valid = np.array([_stable_fold(f, n_folds, random_state) == 0 for f in fams])
    if valid.all() or (~valid).all() or len(concentrations) == 1:
        conc = float(concentrations[0]) if len(concentrations) == 1 else float(np.median(concentrations))
    else:
        scored = []
        for c in concentrations:
            m = em(~valid, c)
            scored.append((float(np.mean(np.log(np.maximum(E[valid] @ m, 1e-300)))), -c, c))
        conc = max(scored)[2]
    m = em(np.ones(len(E), dtype=bool), conc)
    return {"no_crossover": float(m[0]), "interval_masses": m[1:], "concentration": conc,
            "n_informative": int(len(E))}


# --------------------------------------------------------------------------
# structural identifiability certificate
# --------------------------------------------------------------------------
def confident_copy_calls(likelihoods, called=None, *, min_probability: float, min_ratio: float) -> np.ndarray:
    """Hard copy-number constraints from per-locus genotype-likelihood triplets.

    ``likelihoods`` is (n_loci, 3) for copy numbers 0/1/2 (rows of NaN or
    ``called == False`` are unconstrained). A locus is constrained to its
    arg-max copy number when the largest value is >= ``min_probability`` and
    >= ``min_ratio`` times the second largest (both stated by the study
    design; there are no defaults). Returns an int array with -1 where
    unconstrained.
    """
    lk = np.asarray(likelihoods, dtype=float)
    out = np.full(lk.shape[0], -1, dtype=int)
    for i, row in enumerate(lk):
        if called is not None and not bool(called[i]):
            continue
        if not np.all(np.isfinite(row)):
            continue
        order = np.argsort(row)[::-1]
        top, second = row[order[0]], row[order[1]]
        if top >= min_probability and top >= min_ratio * second:
            out[i] = int(order[0])
    return out


def compatible_diplotypes(catalog: Sequence[str], copy_calls) -> List[Tuple[int, int]]:
    """Unordered catalog diplotypes (i <= j) whose copy numbers match every constraint."""
    bits = _as_bits(catalog).astype(np.int16)
    calls = np.asarray(copy_calls)
    loci = np.flatnonzero(calls >= 0)
    H = len(bits)
    iu, ju = np.triu_indices(H)
    if len(loci) == 0:
        return list(zip(iu.tolist(), ju.tolist()))
    dos = bits[iu][:, loci] + bits[ju][:, loci]
    ok = np.all(dos == calls[loci][None, :], axis=1)
    return list(zip(iu[ok].tolist(), ju[ok].tolist()))


def structural_identifiability(catalog: Sequence[str], maternal_calls, paternal_calls,
                               children_calls: Sequence = (), *, max_crossovers: int = 1,
                               stop_at: int = 2, return_states: bool = False):
    """Count joint parental states compatible with hard observations.

    A joint state is a pair of unordered catalog diplotypes (mother, father).
    It is compatible when each parent's copy numbers match that parent's
    constraints and, for every child, at least one ordered transmission (a
    maternal and a paternal gamete with <= ``max_crossovers`` crossovers)
    reproduces all of that child's constrained copy numbers. Constraints are
    int arrays with -1 = unconstrained (``confident_copy_calls``).

    This is a *structural* certificate: exactly one compatible state ->
    ``"IDENTIFIABLE"``; zero or several -> ``"AMBIGUOUS"``. It is not the
    maximum-posterior state and does not use priors or soft evidence.
    The catalog should be the population haplotype support (e.g. the
    phase-validated haplotypes), not recombinant transmitted sequences.

    Returns ``(label, n_compatible)`` with counting stopped at ``stop_at``
    (or ``(label, n, states)`` when ``return_states``; then all are counted).
    """
    catalog = list(catalog)
    L = len(catalog[0])
    mat = compatible_diplotypes(catalog, maternal_calls)
    pat = compatible_diplotypes(catalog, paternal_calls)
    kids = []
    for cc in children_calls:
        cc = np.asarray(cc)
        loci = np.flatnonzero(cc >= 0)
        if len(loci):
            kids.append((loci, cc[loci]))
    limit = None if return_states else stop_at
    states = []
    count = 0
    if not mat or not pat:
        out = ("AMBIGUOUS", 0)
        return out + ([],) if return_states else out

    gam_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def gametes(pair):
        g = gam_cache.get(pair)
        if g is None:
            g = _as_bits(gamete_sequences(catalog[pair[0]], catalog[pair[1]], max_crossovers)).astype(np.int16)
            gam_cache[pair] = g
        return g

    if not kids:
        count = len(mat) * len(pat)
        if return_states:
            states = [(m, p) for m in mat for p in pat]
        label = "IDENTIFIABLE" if count == 1 else "AMBIGUOUS"
        if return_states:
            return label, count, states
        return label, min(count, stop_at) if stop_at else count

    # For each child: code of every paternal-diplotype gamete projected on the
    # child's constrained loci (n_pat, n_gametes); a maternal gamete gm
    # requires the paternal gamete to equal (copy - gm) on those loci.
    # necessary condition per parent: every child can receive one of its
    # gametes (copy - gamete in {0, 1} on the child's constrained loci)
    def can_transmit(pair):
        g = gametes(pair)
        for loci, vals in kids:
            need = vals[None, :] - g[:, loci]
            if not np.any(np.all((need >= 0) & (need <= 1), axis=1)):
                return False
        return True

    mat = [m for m in mat if can_transmit(m)]
    pat = [p for p in pat if can_transmit(p)]
    if not mat or not pat:
        out = ("AMBIGUOUS", 0)
        return out + ([],) if return_states else out
    pat_gam = [gametes(p) for p in pat]
    n_g = max(len(g) for g in pat_gam)
    pat_codes = []
    for loci, vals in kids:
        weights = (1 << np.arange(len(loci))).astype(np.int64)
        codes = np.full((len(pat), n_g), -1, dtype=np.int64)
        for j, g in enumerate(pat_gam):
            codes[j, :len(g)] = g[:, loci].astype(np.int64) @ weights
        pat_codes.append((loci, vals, weights, codes))
    for m in mat:
        gm = gametes(m)
        feasible = np.ones(len(pat), dtype=bool)
        for loci, vals, weights, codes in pat_codes:
            need = vals[None, :] - gm[:, loci]
            valid = np.all((need >= 0) & (need <= 1), axis=1)
            if not valid.any():
                feasible[:] = False
                break
            req = np.unique(need[valid].astype(np.int64) @ weights)
            idx = np.flatnonzero(feasible)
            feasible[idx] = np.isin(codes[idx], req).any(axis=1)
            if not feasible.any():
                break
        for j in np.flatnonzero(feasible):
            count += 1
            if return_states:
                states.append((m, pat[j]))
            if limit is not None and count >= limit:
                break
        if limit is not None and count >= limit:
            break
    label = "IDENTIFIABLE" if count == 1 else "AMBIGUOUS"
    if return_states:
        return label, count, states
    return label, count


# --------------------------------------------------------------------------
# simulation (tests / tutorials)
# --------------------------------------------------------------------------
def simulate_family_cohort(n_families: int = 1500, n_loci: int = 12, n_haplotypes: int = 40,
                           n_classes: int = 3, n_groups: int = 3, group_accuracy: float = 0.7,
                           covariate_dim: int = 3, surface_strength: float = 1.5,
                           class_strength: float = 2.0, ascertainment_strength: float = 2.0,
                           second_phase_prob: float = 0.3, no_crossover_mass: float = 0.6,
                           max_children: int = 3, seed: int = 0, *,
                           validation_bias: float = 0.0, no_call_rate: float = 0.0,
                           no_call_mechanism: str = "mcar", no_call_het_logodds: float = 1.5,
                           no_call_neighbour_logodds: float = 1.0,
                           genotype_error: float = 0.02) -> Dict[str, object]:
    """Simulate nuclear families with known population haplotype structure.

    Population: each family has a latent class c (founder community) with an
    observed noisy group label (correct with ``group_accuracy``); each parent
    has continuous covariates x ~ Dirichlet (e.g. ancestry proportions) and
    two haplotypes drawn i.i.d. from
    softmax(class_strength * profile_c + surface_strength * f(x)), where f is
    a smooth non-linear function shared by all classes. Children receive one
    gamete per parent with at most one crossover (no-crossover mass
    ``no_crossover_mass``, the rest spread unevenly over intervals).

    Sampling: first-phase inclusion depends on the parents' haplotypes (an
    ascertainment on the carrier burden at locus 0, strength
    ``ascertainment_strength``); a random second phase with probability
    ``second_phase_prob`` reveals the parents' phase.

    ``validation_bias`` makes the second phase non-random: the logit of the
    validation probability is shifted by ``validation_bias * (burden > 0)``
    (recorded per family in ``second_phase_prob``).

    Every included parent is also genotyped (separate random stream, so the
    pedigrees do not depend on these options): ``dosage`` (-1 = NO_CALL),
    ``genotype_likelihood`` ((L, 3), NaN rows at NO_CALL; the true genotype
    gets 1 - ``genotype_error``) and ``called``. NO_CALL has base rate
    ``no_call_rate``; ``no_call_mechanism="mcar"`` is independent of
    everything, ``"genotype"`` adds ``no_call_het_logodds`` for
    heterozygotes and ``no_call_neighbour_logodds`` when the previous locus
    failed (missing not at random).

    Returns a dict with ``catalog``, ``people`` (list of dicts, one per
    parent of every *included* family), ``families`` (included families with
    ``first_phase_prob``, ``second_phase_prob``, ``validated``, ``group``),
    ``children`` (transmissions and copy numbers), ``crossover`` (true
    masses) and ``true_haplotype_probs(x, group)``, ``true_population_freq``.
    """
    rng = np.random.default_rng(seed)
    L, H, K = n_loci, n_haplotypes, n_classes
    cat = set()
    while len(cat) < H:
        cat.add("".join(rng.choice(["0", "1"], size=L, p=[0.7, 0.3])))
    catalog = sorted(cat)
    bits = _as_bits(catalog).astype(float)
    profile = rng.normal(0.0, 1.0, size=(K, H))
    centers = rng.dirichlet(np.ones(covariate_dim), size=4)
    amp = rng.normal(0.0, 1.0, size=(4, H))
    lin = rng.normal(0.0, 1.0, size=(covariate_dim, H))

    def surface(x):
        x = np.atleast_2d(x)
        rbf = np.exp(-6.0 * np.sum((x[:, None, :] - centers[None]) ** 2, axis=2))
        return x @ lin + rbf @ amp

    def class_probs(x):
        x = np.atleast_2d(x)
        logits = class_strength * profile[None, :, :] + surface_strength * surface(x)[:, None, :]
        return _softmax_rows(logits)  # (n, K, H)

    class_mix = rng.dirichlet(np.full(K, 3.0))
    # group label = class with prob group_accuracy, else uniform other group
    confusion = np.full((K, n_groups), (1.0 - group_accuracy) / max(n_groups - 1, 1))
    for c in range(K):
        confusion[c, c % n_groups] = group_accuracy
    joint = class_mix[:, None] * confusion
    group_mixing = {f"G{g + 1}": joint[:, g] / joint[:, g].sum() for g in range(n_groups)}

    def true_haplotype_probs(x, group=None):
        mix = group_mixing.get(group, class_mix) if group is not None else class_mix
        return np.einsum("k,nkh->nh", mix, class_probs(x))

    masses = np.concatenate(([no_crossover_mass], rng.dirichlet(np.full(L - 1, 0.7)) * (1 - no_crossover_mass)))

    def gamete(h1, h2):
        cut = int(rng.choice(L, p=masses))
        a, b = (h1, h2) if rng.random() < 0.5 else (h2, h1)
        return a if cut == 0 else a[:cut] + b[cut:]

    pop_x = []
    pop_g = []
    people = []
    families = []
    children = []
    for f in range(n_families):
        c = int(rng.choice(K, p=class_mix))
        g = f"G{int(rng.choice(n_groups, p=confusion[c])) + 1}"
        xs = rng.dirichlet(np.ones(covariate_dim), size=2)
        probs = class_probs(xs)[:, c, :]
        pairs = []
        for r in range(2):
            i, j = rng.choice(H, size=2, p=probs[r])
            pairs.append(canonical_pair(catalog[i], catalog[j]))
        pop_x.extend(xs.tolist())
        pop_g.extend([g, g])
        burden = sum(int(h[0]) for pair in pairs for h in pair)
        pi1 = float(1.0 / (1.0 + np.exp(-(-1.5 + ascertainment_strength * (burden > 0)))))
        if rng.random() >= pi1:
            continue
        fid = f"F{f:05d}"
        pi2 = float(second_phase_prob)
        if validation_bias:
            lo = math.log(pi2 / (1.0 - pi2)) + validation_bias * (burden > 0)
            pi2 = float(1.0 / (1.0 + math.exp(-lo)))
        validated = bool(rng.random() < pi2)
        families.append({"family_id": fid, "group": g, "first_phase_prob": pi1,
                         "second_phase_prob": pi2, "validated": validated,
                         "latent_class": c})
        for r, role in enumerate(("maternal", "paternal")):
            people.append({"family_id": fid, "role": role, "x": xs[r], "hap_a": pairs[r][0],
                           "hap_b": pairs[r][1], "group": g})
        for k in range(int(rng.integers(0, max_children + 1))):
            gm = gamete(*pairs[0])
            gp = gamete(*pairs[1])
            copy = np.array([int(a) + int(b) for a, b in zip(gm, gp)])
            children.append({"family_id": fid, "maternal_transmitted": gm,
                             "paternal_transmitted": gp, "copy_number": copy})
    if no_call_mechanism not in ("mcar", "genotype"):
        raise ValueError("no_call_mechanism must be 'mcar' or 'genotype'")
    arng = np.random.default_rng([int(seed), 7919])
    base = math.log(max(no_call_rate, 1e-12) / max(1.0 - no_call_rate, 1e-12))
    for p in people:
        dose = np.array([int(a) + int(b) for a, b in zip(p["hap_a"], p["hap_b"])])
        called = np.ones(L, dtype=bool)
        if no_call_rate > 0:
            prev = False
            for l in range(L):
                lo = base
                if no_call_mechanism == "genotype":
                    lo += no_call_het_logodds * (dose[l] == 1) + no_call_neighbour_logodds * prev
                prev = bool(arng.random() < 1.0 / (1.0 + math.exp(-lo)))
                called[l] = not prev
        lik = np.full((L, 3), genotype_error / 2.0)
        lik[np.arange(L), dose] = 1.0 - genotype_error
        lik[~called] = np.nan
        p["dosage"] = np.where(called, dose, -1)
        p["genotype_likelihood"] = lik
        p["called"] = called
    pop_x = np.array(pop_x)
    true_pop = np.mean([true_haplotype_probs(pop_x[i:i + 1], pop_g[i])[0] for i in range(len(pop_x))], axis=0)
    return {"catalog": catalog, "people": people, "families": families, "children": children,
            "crossover": {"no_crossover": float(masses[0]), "interval_masses": masses[1:]},
            "true_haplotype_probs": true_haplotype_probs, "true_population_freq": true_pop,
            "group_mixing": group_mixing}
