"""FastCCC pipeline: permutation-free CCC p-values (``fastccc.core`` port).

Mirrors ``statistical_analysis_method`` and
``Cauchy_combination_of_statistical_analysis_methods`` of FastCCC 1.0.1
(MIT, Hou, Ma & Zhou, Nat Commun 2025), working on an in-memory matrix and
without scanpy or result files.

Stages (per scoring branch):

I.   communication score ``CS = h(s(ligand in sender), s(receptor in receiver))``
     with ``s`` = mean / quantile, complex = min / average of subunits,
     ``h`` = arithmetic (0 unless both > 0) / geometric mean;
II.  expression-fraction filter: ligand and receptor expressed in more than
     ``min_percentile`` of the cells of their group (complex = min of subunits);
III. null law of ``s`` for a group of ``n`` cells drawn from the whole dataset
     (exact FFT convolution for ``n < 30``, CLT normal otherwise; order
     statistic law for quantiles);
IV.  ``p = P(CS_null >= CS)``; untested entries get ``p = 1``.

The Cauchy variant runs every branch and combines p-values with equal
weights; the score is the branch average.
"""
from __future__ import annotations

import io
import itertools
import warnings
from collections import Counter
from dataclasses import dataclass, field

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import _distrib as D
from ._db import CPDBTables

SINGLE_UNIT_QUANTILES = {"Median": 0.5, "Q2": 0.5, "Q3": 0.75}
DEFAULT_CAUCHY_SUMMARIES = ("Mean", "Median", "Q3", "Quantile_0.9")
DEFAULT_CAUCHY_COMPLEX = ("Minimum", "Average")
DEFAULT_CAUCHY_LR = ("Arithmetic", "Geometric")


@dataclass
class FastCCCInput:
    counts: pd.DataFrame          # cells x protein multidata ids (only used ids)
    labels: pd.Series             # cell -> group, aligned with ``counts``
    complex_table: pd.DataFrame   # complex_multidata_id, protein_multidata_id
    interactions: pd.DataFrame    # id_cp_interaction -> multidata_1_id, multidata_2_id


@dataclass
class FastCCCResult:
    strength: pd.DataFrame        # "sender|receiver" x id_cp_interaction
    pvalues: pd.DataFrame
    tested: pd.DataFrame          # expression-fraction filter passed (bool)
    interactions: pd.DataFrame
    groups: list
    group_sizes: dict
    branches: dict = field(default_factory=dict)  # branch key -> pvalues (Cauchy mode)


# --------------------------------------------------------------------------- input
def _parse_summary(single_unit_summary: str, min_percentile: float):
    """Return (kind, quantile, min_percentile) for a single-unit summary name."""
    if single_unit_summary == "Mean":
        return "Mean", None, min_percentile
    if single_unit_summary.startswith("Quantile_"):
        quantile = float(single_unit_summary.split("Quantile_")[1])
    elif single_unit_summary in SINGLE_UNIT_QUANTILES:
        quantile = SINGLE_UNIT_QUANTILES[single_unit_summary]
    else:
        raise ValueError(f"Unknown single_unit_summary '{single_unit_summary}'. "
                         "Use 'Mean', 'Median', 'Q2', 'Q3' or 'Quantile_<q>'.")
    if not 0 < quantile < 1:
        raise ValueError("The quantile should be within (0, 1).")
    if quantile < 0.5:
        warnings.warn(f"Quantile {quantile} is too low to be informative on sparse data.", UserWarning)
    return "Quantile", quantile, max(min_percentile, 1 - quantile)


def prepare_input(X, labels, var_names, db: CPDBTables, *, convert_type: str = "hgnc_symbol",
                  select_list=None, filter_: bool = False) -> FastCCCInput:
    """FastCCC ``preprocess.get_input_data`` on an in-memory matrix.

    Genes are mapped to CellPhoneDB proteins (genes of one protein averaged),
    all-zero proteins dropped, and only interactions whose every subunit is
    measured kept (CellPhoneDB's rule).
    """
    labels = pd.Series(np.asarray(labels).astype(str), name="cell_type")
    if filter_:  # scanpy filter_cells(min_counts=1) / filter_genes(min_cells=1)
        keep_cells = np.asarray(X.sum(axis=1)).ravel() >= 1
        X, labels = X[keep_cells], labels[keep_cells].reset_index(drop=True)
        keep_genes = np.asarray((X > 0).sum(axis=0)).ravel() >= 1
        X, var_names = X[:, keep_genes], np.asarray(var_names)[keep_genes]
    var_names = anndata.utils.make_index_unique(pd.Index(np.asarray(var_names).astype(str)))
    interactions = db.interactions()
    if select_list:
        interactions = interactions.loc[list(select_list)]

    tmp = db.gene_to_protein(convert_type)
    pos = pd.Series(np.arange(len(var_names)), index=var_names)
    sel = [(pid, sym) for pid, sym in zip(tmp.index, tmp[convert_type]) if sym in pos.index]
    if not sel:
        raise ValueError(f"No CellPhoneDB genes ({convert_type}) found in var_names.")
    cols = pos.loc[[s for _, s in sel]].to_numpy()
    sub = X[:, cols]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    counts = pd.DataFrame(sub, columns=[p for p, _ in sel])
    counts = counts.T.groupby(level=0).mean().T
    counts = counts[counts.columns[counts.apply(lambda col: col.sum() > 0)]]

    members = db.complex_members()
    content = members.groupby("complex_multidata_id")["protein_multidata_id"].apply(list).to_dict()

    def parts(key):
        return content.get(key, [key])

    present = set(counts.columns)
    used, keep = [], []
    for a, b in zip(interactions["multidata_1_id"], interactions["multidata_2_id"]):
        ok = all(p in present for p in parts(a)) and all(p in present for p in parts(b))
        keep.append(ok)
        if ok:
            used.extend([a, b])
    interactions = interactions[keep]
    needed = {p for item in used for p in parts(item)}
    counts = counts[[c for c in counts.columns if c in needed]]
    used = set(used)
    complex_table = members[[c in used for c in members["complex_multidata_id"]]]
    return FastCCCInput(counts=counts, labels=labels, complex_table=complex_table, interactions=interactions)


# --------------------------------------------------------------------------- stage helpers
def _with_labels(counts: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    df = counts.copy()
    df["cell_type"] = labels.to_numpy()
    return df


def _cluster_mean(counts, labels) -> pd.DataFrame:
    return _with_labels(counts, labels).groupby("cell_type", as_index=True, observed=True).mean()


def _cluster_quantile(counts, labels, qt) -> pd.DataFrame:
    values = counts.to_numpy()
    codes = labels.to_numpy()
    groups = sorted(set(codes))
    rows = [np.quantile(values[codes == g], qt, axis=0, method="lower") for g in groups]
    return pd.DataFrame(np.vstack(rows), index=pd.Index(groups, name="cell_type"), columns=counts.columns)


def _cluster_percents(counts, labels, complex_table) -> pd.DataFrame:
    frac = _with_labels(counts > 0, labels).groupby("cell_type", as_index=True, observed=True).mean()
    return _combine_complex_scores(frac, complex_table, "Minimum")


def _combine_complex_scores(scores: pd.DataFrame, complex_table, how: str) -> pd.DataFrame:
    """Append complex columns (min / mean of present subunits), as ``score.combine_complex_distribution_df``."""
    def func(x):
        x = [m for m in x if m in scores.columns]
        if len(x) == 0:
            return pd.Series(index=scores.index)
        return scores.loc[:, x].T.min() if how == "Minimum" else scores.loc[:, x].T.mean()

    if not complex_table.empty:
        complex_scores = complex_table.groupby("complex_multidata_id").apply(
            lambda x: x["protein_multidata_id"].values, include_groups=False).apply(func).T
        scores = pd.concat((scores, complex_scores), axis=1)
    return scores.dropna(axis=1)


def _mean_null(counts, labels) -> pd.DataFrame:
    """Null law of the group mean for every (group, protein)."""
    sizes = Counter(labels)
    need_small = sorted({n for n in sizes.values() if n < 30})
    out = {}
    for gene in counts.columns:
        base = D.distribution_from_samples(counts[gene].values)
        laws = {}
        if need_small:
            acc = {1: base}
            for k in range(2, max(need_small) + 1):
                acc[k] = acc[k - 1] + base
            laws = {k: acc[k] / k for k in need_small}
        out[gene] = {ct: (laws[n] if n < 30 else (base ** n) / n) for ct, n in sizes.items()}
    return pd.DataFrame(out).loc[sorted(sizes)]


def _quantile_null(counts, labels, quantile) -> pd.DataFrame:
    sizes = Counter(labels)
    out = {}
    for gene in counts.columns:
        base = D.distribution_from_samples(counts[gene].values)
        sections = D.quantile_sections(base)
        out[gene] = {ct: D.Distribution("other", pmf_array=D.quantile_pmf_for_n_iid(base, n, quantile, sections),
                                        is_align=True) for ct, n in sizes.items()}
    return pd.DataFrame(out).loc[sorted(sizes)]


def _combine_complex_nulls(nulls: pd.DataFrame, complex_table, how: str) -> pd.DataFrame:
    """Append complex null laws (``dist_complex.combine_complex_distribution_df``)."""
    law = D.minimum_distribution if how == "Minimum" else D.average_distribution

    def func(x):
        x = [m for m in x if m in nulls.columns]
        if len(x) == 0:
            return pd.Series(index=nulls.index)
        return nulls.loc[:, x].apply(lambda row: law(*row), axis=1)

    if not complex_table.empty:
        complex_nulls = complex_table.groupby("complex_multidata_id").apply(
            lambda x: x["protein_multidata_id"].values, include_groups=False).apply(func).T
        nulls = pd.concat((nulls, complex_nulls.dropna(axis=1)), axis=1)
    return nulls


def _pair_frames(table: pd.DataFrame, interactions: pd.DataFrame, sep: str = "|"):
    groups = sorted(table.index)
    prod = list(itertools.product(groups, groups))
    idx = [sep.join(p) for p in prod]
    p1 = table.loc[[p[0] for p in prod], interactions["multidata_1_id"]]
    p2 = table.loc[[p[1] for p in prod], interactions["multidata_2_id"]]
    p1.columns = p2.columns = interactions.index
    p1.index = p2.index = idx
    return p1, p2


def _interaction_strength(scores, interactions, method) -> pd.DataFrame:
    p1, p2 = _pair_frames(scores, interactions)
    if method == "Arithmetic":
        return (p1 + p2) / 2 * (p1 > 0) * (p2 > 0)
    if method == "Geometric":
        return np.sqrt(p1 * p2)
    raise ValueError("LR_combination must be 'Arithmetic' or 'Geometric'.")


def _percent_filter(percents, interactions, threshold) -> pd.DataFrame:
    p1, p2 = _pair_frames(percents, interactions)
    return (p1 > threshold) * (p2 > threshold)


def _pvalues(nulls, interactions, strength, tested, method) -> pd.DataFrame:
    p1, p2 = _pair_frames(nulls, interactions)
    where = np.where(tested)
    a, b = p1.values[where], p2.values[where]
    values = strength.values[where]
    est = []
    for i, value in enumerate(values):
        law = (a[i] & b[i]) / 2 if method == "Arithmetic" else a[i] * b[i]
        est.append(D.pvalue_from_distribution(value, law))
    pvals = np.ones_like(strength)
    pvals[where] = est
    return pd.DataFrame(pvals, index=strength.index, columns=strength.columns)


def _deg_filter(scores, nulls, interactions, pvals, threshold: float = 0.05) -> pd.DataFrame:
    """Keep a significant hit only if ligand and receptor are each above their own null."""
    p1, p2 = _pair_frames(nulls, interactions)
    c1, c2 = _pair_frames(scores, interactions)
    pvals = pvals.copy()
    where = np.where(pvals.values < threshold)
    e1 = np.array([D.pvalue_from_distribution(v, d) for v, d in zip(c1.values[where], p1.values[where])]) < threshold
    e2 = np.array([D.pvalue_from_distribution(v, d) for v, d in zip(c2.values[where], p2.values[where])]) < threshold
    mask = np.float64(np.logical_and(e1, e2))
    pvals.values[where] = mask * pvals.values[where] + (1 - mask)
    return pvals


def _tsv_roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
    """FastCCC exchanges branch results through TSV files; keep that rounding."""
    buf = io.StringIO()
    frame.to_csv(buf, sep="\t")
    buf.seek(0)
    return pd.read_csv(buf, index_col=0, sep="\t")


def cauchy_combine(pvalue_frames) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weight Cauchy combination; ``p == 1`` contributes ``tan(-pi/2)``."""
    frames = list(pvalue_frames)
    mat = np.concatenate([np.expand_dims(f.values, axis=1) for f in frames], axis=1)
    weight = np.ones(len(frames)) / len(frames)
    T = mat.copy()
    T[np.where(mat == 1)] = np.tan(-np.pi / 2)
    ne = np.where(mat != 1)
    T[ne] = np.tan(np.pi * (0.5 - T[ne]))
    T = weight @ T
    P = 0.5 - np.arctan(T) / np.pi
    ref = frames[0]
    return (pd.DataFrame(T, index=ref.index, columns=ref.columns),
            pd.DataFrame(P, index=ref.index, columns=ref.columns))


# --------------------------------------------------------------------------- drivers
def statistical_analysis(data: FastCCCInput, *, single_unit_summary: str = "Mean",
                         complex_aggregation: str = "Minimum", LR_combination: str = "Arithmetic",
                         min_percentile: float = 0.1, use_DEG: bool = False):
    """One FastCCC scoring branch (``statistical_analysis_method``)."""
    kind, quantile, min_percentile = _parse_summary(single_unit_summary, min_percentile)
    if complex_aggregation not in ("Minimum", "Average"):
        raise ValueError("complex_aggregation must be 'Minimum' or 'Average'.")
    counts, labels, ct, it = data.counts, data.labels, data.complex_table, data.interactions
    scores = _cluster_mean(counts, labels) if kind == "Mean" else _cluster_quantile(counts, labels, quantile)
    scores = _combine_complex_scores(scores, ct, complex_aggregation)
    strength = _interaction_strength(scores, it, LR_combination)
    tested = _percent_filter(_cluster_percents(counts, labels, ct), it, min_percentile)
    nulls = _mean_null(counts, labels) if kind == "Mean" else _quantile_null(counts, labels, quantile)
    nulls = _combine_complex_nulls(nulls, ct, complex_aggregation)
    pvals = _pvalues(nulls, it, strength, tested, LR_combination)
    if use_DEG:
        pvals = _deg_filter(scores, nulls, it, pvals)
    return strength, pvals, tested


def cauchy_analysis(data: FastCCCInput, *, single_unit_summary_list=DEFAULT_CAUCHY_SUMMARIES,
                    complex_aggregation_list=DEFAULT_CAUCHY_COMPLEX, LR_combination_list=DEFAULT_CAUCHY_LR,
                    min_percentile: float = 0.1, use_DEG: bool = False):
    """Every branch + Cauchy combination (``Cauchy_combination_of_statistical_analysis_methods``)."""
    counts, labels, ct, it = data.counts, data.labels, data.complex_table, data.interactions
    percents = _cluster_percents(counts, labels, ct)
    branches, strengths, tested_any = {}, [], None
    for summary in single_unit_summary_list:
        kind, quantile, cur_pct = _parse_summary(summary, min_percentile)
        tested = _percent_filter(percents, it, cur_pct)
        tested_any = tested if tested_any is None else (tested_any | tested)
        base_scores = _cluster_mean(counts, labels) if kind == "Mean" else _cluster_quantile(counts, labels, quantile)
        base_nulls = _mean_null(counts, labels) if kind == "Mean" else _quantile_null(counts, labels, quantile)
        for cagg in complex_aggregation_list:
            scores = _combine_complex_scores(base_scores, ct, cagg)
            nulls = _combine_complex_nulls(base_nulls, ct, cagg)
            for lr in LR_combination_list:
                strength = _interaction_strength(scores, it, lr)
                pvals = _pvalues(nulls, it, strength, tested, lr)
                key = f"{summary}_{cagg}_{lr}"
                branches[key] = _tsv_roundtrip(pvals)
                strengths.append(_tsv_roundtrip(strength))
    _, pvals = cauchy_combine(branches.values())
    pvals = _tsv_roundtrip(pvals)
    if use_DEG:
        scores = _combine_complex_scores(_cluster_mean(counts, labels), ct, "Minimum")
        nulls = _combine_complex_nulls(_mean_null(counts, labels), ct, "Minimum")
        pvals = _deg_filter(scores, nulls, it, pvals)
    strength = sum(strengths[1:], strengths[0].copy()) / len(strengths)
    return strength, pvals, tested_any.astype(bool), branches


def run_fastccc_engine(X, labels, var_names, db: CPDBTables, *, single_unit_summary: str = "Mean",
                       complex_aggregation: str = "Minimum", LR_combination: str = "Arithmetic",
                       min_percentile: float = 0.1, cauchy: bool = False,
                       single_unit_summary_list=DEFAULT_CAUCHY_SUMMARIES,
                       complex_aggregation_list=DEFAULT_CAUCHY_COMPLEX,
                       LR_combination_list=DEFAULT_CAUCHY_LR, use_DEG: bool = False,
                       convert_type: str = "hgnc_symbol", select_list=None,
                       filter_: bool = False) -> FastCCCResult:
    """Run FastCCC on ``X`` (cells x genes, log1p-normalised) with group ``labels``."""
    groups = sorted(set(np.asarray(labels).astype(str)))
    if any("|" in g for g in groups):
        raise ValueError("Group labels must not contain '|'.")
    data = prepare_input(X, labels, var_names, db, convert_type=convert_type, select_list=select_list,
                         filter_=filter_)
    if data.interactions.empty:
        raise RuntimeError("No CellPhoneDB interaction has all its subunits expressed.")
    branches = {}
    if cauchy:
        strength, pvals, tested, branches = cauchy_analysis(
            data, single_unit_summary_list=single_unit_summary_list,
            complex_aggregation_list=complex_aggregation_list, LR_combination_list=LR_combination_list,
            min_percentile=min_percentile, use_DEG=use_DEG)
    else:
        strength, pvals, tested = statistical_analysis(
            data, single_unit_summary=single_unit_summary, complex_aggregation=complex_aggregation,
            LR_combination=LR_combination, min_percentile=min_percentile, use_DEG=use_DEG)
    sizes = Counter(data.labels)
    return FastCCCResult(strength=strength, pvalues=pvals, tested=tested.astype(bool),
                         interactions=data.interactions, groups=sorted(sizes),
                         group_sizes={k: int(v) for k, v in sizes.items()}, branches=branches)
