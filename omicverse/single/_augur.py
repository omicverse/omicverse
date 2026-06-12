r"""Cell type prioritization via machine-learning cross-validation (Augur).

Pure-Python port of `R Augur <https://github.com/neurorestore/Augur>`_
(Skinnider et al. 2020, *Nature Communications*).  Trains a random-forest or
logistic-regression classifier per cell type to predict condition labels,
evaluates AUC in stratified cross-validation, and ranks cell types by their
predictive AUC — higher AUC means the cell type's transcriptomic profile is
more strongly shifted by the perturbation.

Output schema (written to ``adata.uns['augur']``):

* ``'AUC'`` or ``'CCC'`` — :class:`~pandas.DataFrame` ranking cell types
* ``'results'``            — per-subsample, per-fold metrics
* ``'feature_importance'`` — gene-level importance scores
* ``'parameters'``         — run configuration dict

Additionally ``adata.obs['augur_auc']`` maps each cell's AUC from its cell
type for downstream UMAP overlay plots.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import sparse
from scipy.stats import norm as _norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state
from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess_c

from .._registry import register_function
from ..pl._palette import sc_color as _sc_color

# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def _select_random(mat, feature_perc=0.5, rng=None):
    """Random feature selection (port of R ``select_random``)."""
    if feature_perc >= 1.0:
        return mat
    if rng is None:
        rng = np.random.default_rng()
    n_keep = int(mat.shape[0] * feature_perc)
    keep = rng.choice(mat.shape[0], size=n_keep, replace=False)
    keep.sort()
    return mat[keep, :]


def _select_variance(mat, var_quantile=0.5, filter_negative_residuals=False):
    """Feature selection by variance (port of R ``select_variance``).

    Computes standard deviations on sparse matrices without densifying;
    only the surviving rows are converted to dense for loess fitting.
    """
    n_genes, n_cells = mat.shape

    # Compute per-gene mean and std without full densification
    if sparse.issparse(mat):
        means = np.asarray(mat.mean(axis=1)).flatten()
        sq_means = np.asarray(mat.power(2).mean(axis=1)).flatten()
        sds = np.sqrt(np.maximum(sq_means - means ** 2, 0) * n_cells / (n_cells - 1))
    else:
        mat_dense = np.asarray(mat, dtype=np.float64)
        means = np.mean(mat_dense, axis=1)
        sds = np.std(mat_dense, axis=1, ddof=1)

    sds = np.where(np.isnan(sds), 0.0, sds)
    mask = sds > 0

    if var_quantile >= 1.0 and not filter_negative_residuals:
        if sparse.issparse(mat):
            return mat[mask, :]
        return np.asarray(mat, dtype=np.float64)[mask, :]

    # Only densify the rows that pass the non-zero-SD filter
    if sparse.issparse(mat):
        mat_sub = mat[mask, :].toarray()
    else:
        mat_sub = np.asarray(mat, dtype=np.float64)[mask, :]

    means = means[mask]
    sds_sub = sds[mask]
    cvs = means / sds_sub

    lower = np.percentile(cvs, 1)
    upper = np.percentile(cvs, 99)
    keep = (cvs >= lower) & (cvs <= upper)
    cv0 = cvs[keep]
    mean0 = means[keep]
    del cvs, means

    if np.any(mean0 < 0):
        model = _loess(mean0, cv0, span=0.75)
    else:
        fit1 = _loess(mean0, cv0, span=0.75)
        fit2 = _loess(np.log(mean0), cv0, span=0.75)
        cox = _coxtest(cv0, fit1, fit2)
        model = fit1 if cox["p1"] < cox["p2"] else fit2

    residuals = model["residuals"]
    if filter_negative_residuals:
        keep_genes = residuals > 0
    else:
        threshold = np.percentile(residuals, var_quantile * 100)
        keep_genes = residuals > threshold

    if sparse.issparse(mat):
        result_indices = np.where(mask)[0][keep][keep_genes]
        return mat[result_indices, :], result_indices
    result_indices = np.where(mask)[0][keep][keep_genes]
    return mat_sub[keep, :][keep_genes, :], result_indices


def _loess(x, y, span=0.75):
    """Local polynomial regression via statsmodels LOWESS."""
    n = len(x)
    if n == 0:
        return {"fitted": np.array([]), "residuals": np.array([])}
    result = _lowess_c(y, x, frac=span, it=2, return_sorted=True)
    order = np.argsort(x)
    fitted = np.empty(n)
    fitted[order] = result[:, 1]
    return {"fitted": fitted, "residuals": y - fitted}


def _coxtest(y, model1, model2):
    """Cox non-nested model comparison test."""
    res1, res2 = model1["residuals"], model2["residuals"]
    yhat1, yhat2 = model1["fitted"], model2["fitted"]
    z1 = res1 - (y - yhat2)
    z2 = res2 - (y - yhat1)
    sd1 = np.std(z1, ddof=1) / np.sqrt(len(z1))
    sd2 = np.std(z2, ddof=1) / np.sqrt(len(z2))
    z1_stat = np.mean(z1) / sd1 if sd1 > 0 else 0.0
    z2_stat = np.mean(z2) / sd2 if sd2 > 0 else 0.0
    p1 = 2 * (1 - _norm.cdf(abs(z1_stat)))
    p2 = 2 * (1 - _norm.cdf(abs(z2_stat)))
    return {"z1": z1_stat, "z2": z2_stat, "p1": p1, "p2": p2}


# ---------------------------------------------------------------------------
# Fast random forest
# ---------------------------------------------------------------------------


class _FastRandomForest:
    """Lightweight RF that bypasses sklearn parameter validation overhead."""

    def __init__(self, n_estimators=100, max_features=2, min_samples_split=2,
                 random_state=1, mode="classification"):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.rng = check_random_state(random_state)
        self.mode = mode
        self.trees: list = []
        self.classes_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        self.trees = []
        for _ in range(self.n_estimators):
            n = X.shape[0]
            indices = self.rng.choice(n, size=n, replace=True)
            X_boot, y_boot = X[indices], y[indices]
            if self.mode == "classification":
                tree = DecisionTreeClassifier(
                    max_features=self.max_features,
                    min_samples_split=self.min_samples_split,
                    random_state=self.rng,
                )
            else:
                tree = DecisionTreeRegressor(
                    max_features=self.max_features,
                    min_samples_split=self.min_samples_split,
                    random_state=self.rng,
                )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            importances += tree.feature_importances_
        self.feature_importances_ = importances / self.n_estimators
        return self

    def predict(self, X):
        if self.mode == "classification":
            votes = np.array([tree.predict(X) for tree in self.trees])
            cls_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
            result = np.empty(X.shape[0], dtype=self.classes_.dtype)
            for i in range(X.shape[0]):
                counts = np.bincount(
                    [cls_to_idx[v] for v in votes[:, i]]
                )
                result[i] = self.classes_[np.argmax(counts)]
            return result
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        for tree in self.trees:
            tree_proba = tree.predict_proba(X)
            tree_classes = tree.classes_
            for i, cls in enumerate(tree_classes):
                j = np.where(self.classes_ == cls)[0][0]
                proba[:, j] += tree_proba[:, i]
        proba /= self.n_estimators
        return proba


# ---------------------------------------------------------------------------
# Core AUC calculation
# ---------------------------------------------------------------------------


def _extract_input(input, meta, label_col, cell_type_col):
    """Extract expression matrix (genes x cells), cell types, labels, and gene names."""
    if hasattr(input, "obs") and hasattr(input, "X"):
        meta = input.obs
        expr = input.X.T
        if sparse.issparse(expr):
            expr = expr.tocsr()
        gene_names = np.array(input.var_names.tolist())
        return expr, meta[cell_type_col].values, meta[label_col].values, gene_names
    if isinstance(input, pd.DataFrame):
        expr = input.values
        gene_names = np.array(input.columns.tolist())
    elif sparse.issparse(input) or isinstance(input, np.ndarray):
        expr = input
        gene_names = np.array([f"gene_{i}" for i in range(expr.shape[0])])
    else:
        raise ValueError("Unsupported input type")
    if meta is None:
        raise ValueError("Must provide metadata if not supplying AnnData")
    return expr, meta[cell_type_col].values, meta[label_col].values, gene_names


def _is_numeric(arr):
    try:
        np.asarray(arr, dtype=float)
        return True
    except (ValueError, TypeError):
        return False


def _stratified_subsample(y, subsample_size, mode, rng):
    indices = np.arange(len(y))
    if mode == "classification":
        selected = []
        for lab in np.unique(y):
            lab_indices = indices[y == lab]
            n = min(subsample_size, len(lab_indices))
            selected.append(rng.choice(lab_indices, size=n, replace=False))
        return np.concatenate(selected)
    n = min(subsample_size, len(indices))
    return rng.choice(indices, size=n, replace=False)


def _compute_classification_metrics(y_true, y_pred, y_prob, classes, multiclass):
    metrics = {}
    unique_test = np.unique(y_true)
    if len(unique_test) < 2:
        metrics["roc_auc"] = np.nan
        metrics["accuracy"] = np.mean(y_true == y_pred)
        return metrics
    try:
        if multiclass:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro", labels=classes,
            )
        else:
            pos_idx = np.where(classes == unique_test[1])[0][0]
            auc_val = roc_auc_score(
                y_true, y_prob[:, pos_idx], labels=classes,
            )
            metrics["roc_auc"] = max(auc_val, 1 - auc_val)
    except ValueError:
        metrics["roc_auc"] = np.nan
    metrics["accuracy"] = np.mean(y_true == y_pred)
    return metrics


def _lin_ccc(y_true, y_pred):
    """Lin's concordance correlation coefficient.

    This is the regression/RNA-velocity metric used by the reference R Augur —
    it measures agreement with the 45° identity line (accuracy + precision),
    not merely linear correlation as Pearson's r does. Equal to Pearson's r
    only when the two series share the same mean and variance.
    """
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    if x.size < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    sxy = ((x - mx) * (y - my)).mean()
    denom = x.var() + y.var() + (mx - my) ** 2  # population variance (ddof=0)
    if denom == 0:                              # both constant and equal
        return 1.0
    return 2.0 * sxy / denom


def _calculate_auc(
    input,
    meta=None,
    label_col="label",
    cell_type_col="cell_type",
    n_subsamples=50,
    subsample_size=20,
    folds=3,
    min_cells=None,
    var_quantile=0.5,
    feature_perc=0.5,
    augur_mode="default",
    classifier="rf",
    rf_params=None,
    lr_params=None,
    seed=42,
):
    """Core AUC calculation — port of R Augur's ``calculate_auc``."""
    if rf_params is None:
        rf_params = {"trees": 100, "mtry": 2, "min_n": None, "importance": "accuracy"}
    if lr_params is None:
        lr_params = {"mixture": 1.0, "penalty": "auto"}
    if min_cells is None:
        min_cells = subsample_size

    if augur_mode == "velocity":
        feature_perc = 1.0
        var_quantile = 1.0
    elif augur_mode == "permute" and n_subsamples < 100:
        n_subsamples = 500

    expr, cell_types, labels, gene_names = _extract_input(input, meta, label_col, cell_type_col)

    if len(np.unique(labels)) < 2:
        raise ValueError(f"Need at least 2 labels, got {len(np.unique(labels))}")
    if np.any(pd.isna(labels)):
        raise ValueError("Labels contain missing values")
    if np.any(pd.isna(cell_types)):
        raise ValueError("Cell types contain missing values")

    is_numeric_labels = _is_numeric(labels)
    if is_numeric_labels:
        mode = "regression"
        multiclass = False
    else:
        mode = "classification"
        multiclass = len(np.unique(labels)) > 2
        labels = labels.astype(str)

    n_iter = max(n_subsamples, 1)
    unique_types = np.unique(cell_types)
    all_results: list[dict] = []
    all_importances: list[dict] = []

    for ct in unique_types:
        ct_mask = cell_types == ct
        y_ct = labels[ct_mask]
        X_ct = expr[:, ct_mask]
        gene_idx = np.arange(X_ct.shape[0])

        if mode == "classification":
            min_count = min(np.sum(y_ct == lab) for lab in np.unique(y_ct))
            if min_count < min_cells:
                warnings.warn(f"Skipping {ct}: min cells ({min_count}) < {min_cells}")
                continue
        elif len(y_ct) < min_cells:
            warnings.warn(f"Skipping {ct}: total cells ({len(y_ct)}) < {min_cells}")
            continue

        if X_ct.shape[0] >= 1000 and var_quantile < 1.0:
            X_ct, var_indices = _select_variance(X_ct, var_quantile, filter_negative_residuals=False)
            gene_idx = gene_idx[var_indices]

        for subsample_idx in range(1, n_iter + 1):
            rng = np.random.default_rng(seed + subsample_idx)

            if augur_mode == "permute":
                perm_rng = np.random.default_rng(subsample_idx)
                y_ct_iter = perm_rng.permutation(y_ct)
            else:
                y_ct_iter = y_ct.copy()

            if n_subsamples < 1:
                if X_ct.shape[0] >= 1000 and feature_perc < 1.0:
                    n_keep = int(X_ct.shape[0] * feature_perc)
                    keep = rng.choice(X_ct.shape[0], size=n_keep, replace=False)
                    keep.sort()
                    X_sub = X_ct[keep, :]
                    sub_gene_idx = gene_idx[keep]
                else:
                    X_sub = X_ct
                    sub_gene_idx = gene_idx
                y_sub = y_ct_iter
            else:
                subsample_idxs = _stratified_subsample(y_ct_iter, subsample_size, mode, rng)
                y_sub = y_ct_iter[subsample_idxs]
                if X_ct.shape[0] >= 1000 and feature_perc < 1.0:
                    n_keep = int(X_ct.shape[0] * feature_perc)
                    keep = rng.choice(X_ct.shape[0], size=n_keep, replace=False)
                    keep.sort()
                    X_feat = X_ct[keep, :]
                    feat_gene_idx = gene_idx[keep]
                else:
                    X_feat = X_ct
                    feat_gene_idx = gene_idx
                X_sub = X_feat[:, subsample_idxs]
                if sparse.issparse(X_sub):
                    X_sub = X_sub.toarray()
                col_vars = np.var(X_sub, axis=1, ddof=1)
                var_mask = col_vars > 0
                X_sub = X_sub[var_mask, :]
                sub_gene_idx = feat_gene_idx[var_mask]

            if sparse.issparse(X_sub):
                X_sub = X_sub.toarray()
            X_df = X_sub.T  # cells x genes

            if mode == "classification":
                skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=subsample_idx)
                fold_splits = list(skf.split(X_df, y_sub))
            else:
                kf = KFold(n_splits=folds, shuffle=True, random_state=subsample_idx)
                fold_splits = list(kf.split(X_df))

            for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
                X_train, X_test = X_df[train_idx], X_df[test_idx]
                y_train, y_test = y_sub[train_idx], y_sub[test_idx]

                if classifier == "rf":
                    model = _FastRandomForest(
                        n_estimators=rf_params["trees"],
                        max_features=rf_params["mtry"],
                        min_samples_split=rf_params.get("min_n", 2) or 2,
                        random_state=seed + subsample_idx * folds + fold_idx,
                        mode=mode,
                    )
                    model.fit(X_train, y_train)
                elif classifier == "lr":
                    penalty_val = lr_params.get("penalty", 1.0)
                    if isinstance(penalty_val, str) or penalty_val is None or penalty_val <= 0:
                        penalty_val = 1.0
                    model = LogisticRegression(
                        penalty="l1", solver="saga", C=1.0 / penalty_val,
                        max_iter=1000, random_state=seed + subsample_idx * folds + fold_idx,
                    )
                    model.fit(X_train, y_train)
                else:
                    raise ValueError(f"Invalid classifier: {classifier}")

                y_pred = model.predict(X_test)

                if mode == "classification":
                    y_prob = model.predict_proba(X_test)
                    result = _compute_classification_metrics(
                        y_test, y_pred, y_prob, model.classes_, multiclass,
                    )
                    for metric_name, estimate in result.items():
                        all_results.append({
                            "cell_type": ct, "subsample_idx": subsample_idx,
                            "fold": fold_idx + 1, "metric": metric_name,
                            "estimator": "binary" if not multiclass else "macro",
                            "estimate": estimate,
                        })
                else:
                    ccc_val = _lin_ccc(y_test, y_pred)
                    all_results.append({
                        "cell_type": ct, "subsample_idx": subsample_idx,
                        "fold": fold_idx + 1, "metric": "ccc",
                        "estimator": "standard", "estimate": ccc_val,
                    })

                if classifier == "rf":
                    importances = model.feature_importances_
                elif classifier == "lr":
                    importances = np.abs(model.coef_).mean(axis=0)
                else:
                    importances = None

                if importances is not None:
                    for g_idx in range(X_df.shape[1]):
                        all_importances.append({
                            "cell_type": ct, "subsample_idx": subsample_idx,
                            "fold": fold_idx + 1,
                            "gene": gene_names[sub_gene_idx[g_idx]],
                            "importance": importances[g_idx],
                        })

    results_df = pd.DataFrame(all_results)
    importances_df = pd.DataFrame(all_importances)

    if len(results_df) == 0:
        raise ValueError(f"No cell type had at least {min_cells} cells in all conditions")

    if mode == "classification":
        auc_rows = results_df[results_df["metric"] == "roc_auc"]
        auc_by_sub = auc_rows.groupby(["cell_type", "subsample_idx"])["estimate"].mean()
        auc_summary = auc_by_sub.groupby("cell_type").mean().reset_index()
        auc_summary.columns = ["cell_type", "auc"]
        auc_summary = auc_summary.sort_values("auc", ascending=False).reset_index(drop=True)
    else:
        auc_rows = results_df[results_df["metric"] == "ccc"]
        auc_by_sub = auc_rows.groupby(["cell_type", "subsample_idx"])["estimate"].mean()
        auc_summary = auc_by_sub.groupby("cell_type").mean().reset_index()
        auc_summary.columns = ["cell_type", "ccc"]
        auc_summary = auc_summary.sort_values("ccc", ascending=False).reset_index(drop=True)

    obj: dict[str, Any] = {
        "X": expr, "y": labels, "cell_types": cell_types,
        "parameters": {
            "n_subsamples": n_subsamples, "subsample_size": subsample_size,
            "folds": folds, "min_cells": min_cells, "var_quantile": var_quantile,
            "feature_perc": feature_perc,
            "classifier": classifier,
            "rf_params": rf_params if classifier == "rf" else None,
            "lr_params": lr_params if classifier == "lr" else None,
        },
        "results": results_df,
        "feature_importance": importances_df,
    }
    if mode == "classification":
        obj["AUC"] = auc_summary
    else:
        obj["CCC"] = auc_summary
    return obj


# ---------------------------------------------------------------------------
# Differential prioritization
# ---------------------------------------------------------------------------


def _bh_correction(pvalues):
    """Benjamini-Hochberg p-value correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([])
    order = np.argsort(pvalues)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    adjusted = pvalues * n / ranked
    adjusted_sorted = np.minimum.accumulate(adjusted[order][::-1])[::-1]
    result = np.empty(n)
    result[order] = adjusted_sorted
    return np.clip(result, 0, 1)


def _draw_mean_aucs(permuted_aucs, n_permutations, n_intervals):
    """Draw mean AUCs from permuted results."""
    results: list[dict] = []
    for perm_idx in range(1, n_permutations + 1):
        rng_inner = np.random.default_rng(perm_idx)
        perm_copy = permuted_aucs.copy()
        perm_copy["bin"] = pd.cut(
            perm_copy["subsample_idx"], bins=n_intervals, labels=False,
        ) + 1
        for ct in perm_copy["cell_type"].unique():
            ct_data = perm_copy[perm_copy["cell_type"] == ct].copy()
            bins = ct_data["bin"].values.copy()
            rng_inner.shuffle(bins)
            ct_data["bin"] = bins
            ct_bin1 = ct_data[ct_data["bin"] == 1]
            if len(ct_bin1) > 0:
                results.append({
                    "cell_type": ct, "permutation_idx": perm_idx,
                    "mean": ct_bin1["estimate"].mean(),
                    "sd": ct_bin1["estimate"].std(),
                })
    return pd.DataFrame(results)


def _calculate_differential_prioritization(
    augur1, augur2, permuted1, permuted2,
    n_subsamples=50, n_permutations=1000,
):
    """Statistical test for differential prioritization."""
    key = "AUC" if "AUC" in augur1 else "CCC"
    metric = "roc_auc" if key == "AUC" else "ccc"
    col = "auc" if key == "AUC" else "ccc"

    obs1 = augur1[key].copy()
    obs2 = augur2[key].copy()
    permuted_res1 = permuted1["results"].copy()
    permuted_res2 = permuted2["results"].copy()

    n_intervals = permuted_res1["subsample_idx"].max() // 50

    perm_auc1 = (
        permuted_res1[permuted_res1["metric"] == metric]
        .groupby(["cell_type", "subsample_idx"])["estimate"].mean().reset_index()
    )
    perm_auc2 = (
        permuted_res2[permuted_res2["metric"] == metric]
        .groupby(["cell_type", "subsample_idx"])["estimate"].mean().reset_index()
    )

    rnd1 = _draw_mean_aucs(perm_auc1, n_permutations, n_intervals)
    rnd2 = _draw_mean_aucs(perm_auc2, n_permutations, n_intervals)

    delta = obs1.merge(obs2, on="cell_type", suffixes=(".x", ".y"))
    delta["delta_auc"] = delta[f"{col}.y"] - delta[f"{col}.x"]

    rnd = rnd1.merge(rnd2, on=["cell_type", "permutation_idx"], suffixes=(".x", ".y"))
    rnd["delta_rnd"] = rnd["mean.y"] - rnd["mean.x"]

    pvals_list: list[dict] = []
    for ct in delta["cell_type"].unique():
        ct_delta = delta[delta["cell_type"] == ct]
        ct_rnd = rnd[rnd["cell_type"] == ct]
        delta_auc = ct_delta["delta_auc"].values[0]
        delta_rnd = ct_rnd["delta_rnd"].values
        b = int(np.sum(delta_rnd >= delta_auc))
        m = len(delta_rnd)
        z = (
            (delta_auc - np.mean(delta_rnd)) / np.std(delta_rnd)
            if np.std(delta_rnd) > 0 else 0.0
        )
        pval = min(2 * (b + 1) / (m + 1), 2 * (m - b + 1) / (m + 1))
        pvals_list.append({"cell_type": ct, "b": b, "m": m, "z": z, "pval": pval})

    pvals = pd.DataFrame(pvals_list)
    pvals["padj"] = _bh_correction(pvals["pval"].values)

    result = delta[["cell_type", f"{col}.x", f"{col}.y", "delta_auc"]].merge(pvals, on="cell_type")
    return result.dropna(subset=["pval"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting helpers  (omicverse style: sc_color, fontsize=14, clean spines)
# ---------------------------------------------------------------------------

def _add_arrow(ax, coords, fontsize=12, x_label="UMAP1", y_label="UMAP2",
               arrow_scale=10, arrow_width=0.01):
    """Draw axis arrows at bottom-left corner (ov ``add_arrow`` pattern)."""
    x_range = (coords[:, 0].max() - coords[:, 0].min()) / 6
    y_range = (coords[:, 1].max() - coords[:, 1].min()) / 6
    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
    ax.arrow(x=x_min - x_range / 5, y=y_min,
             dx=x_range + x_range / arrow_scale, dy=0,
             width=arrow_width, color="k",
             head_width=y_range * 2 / arrow_scale,
             head_length=x_range * 2 / arrow_scale, overhang=0.5)
    ax.arrow(x=x_min, y=y_min - y_range / 5,
             dx=0, dy=y_range + y_range / arrow_scale,
             width=arrow_width, color="k",
             head_width=x_range * 2 / arrow_scale,
             head_length=y_range * 2 / arrow_scale, overhang=0.5)
    ax.text(x=x_min, y=y_min - y_range / 2, s=x_label,
            fontsize=fontsize, multialignment='center', verticalalignment='center')
    ax.text(x=x_min - x_range / 2, y=y_min, s=y_label,
            fontsize=fontsize, rotation='vertical', multialignment='center',
            horizontalalignment='center')


def _apply_ov_style(ax, fontsize=14):
    """Apply the canonical ov clean-axes style."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.grid(False)


def _plot_lollipop(augur_results, *, top_n=None, figsize=(4, 6), fontsize=14,
                   title=None, color=None, show=None, save=None, return_fig=None,
                   ax=None, **kwargs):
    """Lollipop plot of cell type AUC values (ov style)."""
    if "AUC" in augur_results:
        auc_df = augur_results["AUC"].copy()
        metric_col = "auc"
    elif "CCC" in augur_results:
        auc_df = augur_results["CCC"].copy()
        metric_col = "ccc"
    else:
        raise ValueError("Results must contain 'AUC' or 'CCC' key")

    if top_n is not None:
        auc_df = auc_df.head(top_n)
    auc_df = auc_df.sort_values(metric_col, ascending=True).reset_index(drop=True)
    if color is None:
        color = _sc_color[0]

    created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Use per-cell-type colors from sc_color palette
    n_ct = len(auc_df)
    colors = [_sc_color[i % len(_sc_color)] for i in range(n_ct)]

    ax.hlines(y=auc_df["cell_type"], xmin=0, xmax=auc_df[metric_col],
              color=color, linewidth=1.5, alpha=0.7, **kwargs)
    for i, (_, row) in enumerate(auc_df.iterrows()):
        ax.scatter(row[metric_col], row["cell_type"], color=colors[i], s=80,
                   zorder=3, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("AUC" if metric_col == "auc" else "CCC", fontsize=fontsize)
    ax.set_ylabel("")
    if title is not None:
        ax.set_title(title, fontsize=fontsize + 1, fontweight="bold")
    for i, (_, row) in enumerate(auc_df.iterrows()):
        ax.text(row[metric_col] + 0.01, row["cell_type"], "%.3f" % row[metric_col],
                va="center", fontsize=fontsize - 4)
    xmin = min(0, auc_df[metric_col].min())
    xmax = auc_df[metric_col].max()
    x_range = max(xmax - xmin, 1e-6)
    ax.set_xlim(xmin - 0.05 * x_range, xmax + 0.15 * x_range)
    _apply_ov_style(ax, fontsize)

    if save:
        fig.savefig(save if isinstance(save, str) else "augur_lollipop.pdf",
                    dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if created or return_fig:
        return fig, ax
    return ax


def _plot_umap(input, augur_results, *, mode="default", cell_type_col="cell_type",
               label_col="label", figsize=(4, 4), point_size=None, alpha=0.8,
               palette="cividis", top_n=0, frameon="small", fontsize=14,
               title=None, show=None, save=None, return_fig=None, ax=None, **kwargs):
    """UMAP overlay colored by cell type AUC (ov style)."""
    if not hasattr(input, "obsm") or "X_umap" not in input.obsm:
        raise ValueError("Input must be AnnData with 'X_umap' in .obsm")

    if "AUC" in augur_results:
        aucs = augur_results["AUC"].copy()
    elif "CCC" in augur_results:
        aucs = augur_results["CCC"].copy()
    else:
        raise ValueError("Results must contain 'AUC' or 'CCC' key")

    metric_col = "auc" if "auc" in aucs.columns else "ccc"
    if mode == "rank":
        aucs["rank"] = aucs[metric_col].rank()
        aucs["rank_pct"] = aucs["rank"] / len(aucs)
        aucs["rank_pct"] = (
            (aucs["rank_pct"] - aucs["rank_pct"].min())
            / (aucs["rank_pct"].max() - aucs["rank_pct"].min())
        )
        aucs["fill"] = aucs["rank_pct"]
        legend_name = "Rank (%)"
    else:
        aucs["fill"] = aucs[metric_col]
        legend_name = "AUC" if metric_col == "auc" else "CCC"

    umap_coords = input.obsm["X_umap"]
    meta = input.obs.copy()
    n_cells = umap_coords.shape[0]
    auc_map = dict(zip(aucs["cell_type"], aucs["fill"]))
    cell_auc = meta[cell_type_col].map(auc_map).values
    if point_size is None:
        point_size = 120000 / n_cells

    created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cmap = matplotlib.colormaps[palette] if isinstance(palette, str) else palette
    scatter = ax.scatter(
        umap_coords[:, 0], umap_coords[:, 1], c=cell_auc, cmap=cmap,
        s=point_size, alpha=alpha, edgecolors="none", rasterized=True, **kwargs,
    )

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.ticker import MaxNLocator

    cax = inset_axes(ax, width="2%", height="30%", loc="lower right", borderpad=0)
    cb = plt.colorbar(scatter, cax=cax, orientation="vertical")
    cb.locator = MaxNLocator(nbins=3, integer=False)
    cb.update_ticks()
    cb.set_label(legend_name, fontsize=fontsize - 2)
    cb.outline.set_visible(False)

    if top_n > 0:
        labeled_types = aucs.nlargest(top_n, "fill")["cell_type"].values
        for ct in labeled_types:
            mask = meta[cell_type_col].values == ct
            if mask.any():
                median_x = np.median(umap_coords[mask, 0])
                median_y = np.median(umap_coords[mask, 1])
                ax.annotate(ct, (median_x, median_y), fontsize=fontsize - 5,
                            fontweight="bold", ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                      alpha=0.8, edgecolor="gray"))

    if frameon is False:
        ax.axis("off")
    elif frameon == "small":
        ax.axis("off")
        _add_arrow(ax, umap_coords, fontsize=fontsize - 2,
                   x_label="UMAP1", y_label="UMAP2")
    else:
        ax.set_xlabel("UMAP1", fontsize=fontsize)
        ax.set_ylabel("UMAP2", fontsize=fontsize)
        _apply_ov_style(ax, fontsize)
        ax.set_xticks([])
        ax.set_yticks([])

    if title is not None:
        ax.set_title(title, fontsize=fontsize + 1, fontweight="bold")
    ax.set_aspect("equal")

    if save:
        fig.savefig(save if isinstance(save, str) else "augur_umap.pdf",
                    dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if created or return_fig:
        return fig, ax
    return ax


def _plot_important_features(augur_results, *, cell_type=None, top_n=10,
                             figsize=(5, 4), fontsize=14, color=None, title=None,
                             show=None, save=None, return_fig=None, ax=None,
                             **kwargs):
    """Bar chart of top important features (ov style)."""
    if "feature_importance" not in augur_results:
        raise ValueError("Results must contain 'feature_importance' key")
    imp_df = augur_results["feature_importance"].copy()
    if cell_type is None:
        key = "AUC" if "AUC" in augur_results else "CCC"
        cell_type = augur_results[key]["cell_type"].iloc[0]
    if color is None:
        color = _sc_color[5]  # orange-ish

    imp_ct = imp_df[imp_df["cell_type"] == cell_type]
    imp_agg = imp_ct.groupby("gene")["importance"].mean().reset_index()
    imp_agg = imp_agg.sort_values("importance", ascending=False).head(top_n)
    imp_agg = imp_agg.sort_values("importance", ascending=True).reset_index(drop=True)

    created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y_pos = np.arange(len(imp_agg))
    ax.barh(y_pos, imp_agg["importance"], color=color, alpha=0.8, height=0.7, **kwargs)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_agg["gene"], fontsize=fontsize - 4)
    ax.set_xlabel("Importance", fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize + 1, fontweight="bold")
    _apply_ov_style(ax, fontsize)

    if save:
        fig.savefig(save if isinstance(save, str) else "augur_features.pdf",
                    dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if created or return_fig:
        return fig, ax
    return ax


def _plot_augur_combined(augur_results, *, top_n=None, figsize=(12, 5), fontsize=14,
                         title=None, show=None, save=None, return_fig=None, **kwargs):
    """Combined lollipop + feature importance figure (ov style)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _plot_lollipop(augur_results, top_n=top_n, fontsize=fontsize, ax=axes[0], **kwargs)
    _plot_important_features(augur_results, top_n=top_n or 10, fontsize=fontsize,
                             ax=axes[1], **kwargs)
    if title is not None:
        fig.suptitle(title, fontsize=fontsize + 1, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(save if isinstance(save, str) else "augur_combined.pdf",
                    dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if return_fig:
        return fig
    return fig


def _plot_scatterplot(augur1, augur2, *, top_n=None, figsize=(5, 5), point_size=20,
                      fontsize=14, show=None, save=None, return_fig=None, ax=None,
                      **kwargs):
    """Compare two Augur results as a scatterplot (ov style)."""
    key = "AUC" if "AUC" in augur1 else "CCC"
    if key not in augur2:
        raise ValueError(f"Both results must contain '{key}' key")
    col = "auc" if key == "AUC" else "ccc"
    r1, r2 = augur1[key].copy(), augur2[key].copy()
    df = r1.merge(r2, on="cell_type", suffixes=(".x", ".y"))
    df["delta"] = df[f"{col}.y"] - df[f"{col}.x"]
    df["abs_delta"] = df["delta"].abs()
    labels = (
        df if top_n is None
        else df.sort_values("abs_delta", ascending=False).head(top_n)
    )

    created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    delta_range = max(abs(df["delta"].min()), abs(df["delta"].max()))
    norm = (
        matplotlib.colors.TwoSlopeNorm(vmin=-delta_range, vcenter=0, vmax=delta_range)
        if delta_range > 0 else None
    )
    cx, cy = f"{col}.x", f"{col}.y"
    label = "AUC" if key == "AUC" else "CCC"
    scatter = ax.scatter(
        df[cx], df[cy], c=df["delta"], cmap="coolwarm", norm=norm,
        s=point_size, edgecolors="black", linewidths=0.3, **kwargs,
    )
    lim_min = min(df[cx].min(), df[cy].min()) - 0.02
    lim_max = max(df[cx].max(), df[cy].max()) + 0.02
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="dotted",
            color="gray", linewidth=0.8)
    if len(labels) > 0:
        for _, row in labels.iterrows():
            ax.annotate(row["cell_type"], (row[cx], row[cy]),
                        fontsize=fontsize - 6, ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.3))
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"$\\Delta$ {label}", fontsize=fontsize - 2)
    cbar.outline.set_visible(False)
    ax.set_xlabel(f"{label} 1", fontsize=fontsize)
    ax.set_ylabel(f"{label} 2", fontsize=fontsize)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    _apply_ov_style(ax, fontsize)

    if save:
        fig.savefig(save if isinstance(save, str) else "augur_scatter.pdf",
                    dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if created or return_fig:
        return fig, ax
    return ax


def _plot_differential_prioritization(
    results, *, top_n=0, pval_threshold=0.05,
    condition1_color=None, condition2_color=None, ns_color="#cccccc",
    figsize=(5, 5), point_size=10, fontsize=14,
    show=None, save=None, return_fig=None, ax=None, **kwargs,
):
    """Scatterplot of differential prioritization (ov style)."""
    if condition1_color is None:
        condition1_color = _sc_color[0]   # blue
    if condition2_color is None:
        condition2_color = _sc_color[5]   # orange

    # Detect AUC vs CCC columns
    if "auc.x" in results.columns:
        cx, cy = "auc.x", "auc.y"
    elif "ccc.x" in results.columns:
        cx, cy = "ccc.x", "ccc.y"
    else:
        raise ValueError("Results must contain 'auc.x'/'auc.y' or 'ccc.x'/'ccc.y' columns")

    required = ["cell_type", cx, cy, "pval", "z"]
    missing = [c for c in required if c not in results.columns]
    if missing:
        raise ValueError(f"Results missing required columns: {missing}")

    df = results.dropna(subset=[cx, cy]).copy()
    df["color_group"] = "n.s."
    df.loc[(df["pval"] < pval_threshold) & (df["z"] > 0), "color_group"] = "condition 2"
    df.loc[(df["pval"] < pval_threshold) & (df["z"] <= 0), "color_group"] = "condition 1"

    color_map = {"condition 1": condition1_color, "condition 2": condition2_color,
                 "n.s.": ns_color}
    sig_df = df[df["pval"] < pval_threshold].copy()
    sig_df["abs_z"] = sig_df["z"].abs()
    labels = sig_df.sort_values("abs_z", ascending=False).head(top_n)

    created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for group, clr in color_map.items():
        mask = df["color_group"] == group
        if mask.any():
            ax.scatter(df.loc[mask, cx], df.loc[mask, cy],
                       c=clr, s=point_size, label=group if group != "n.s." else "n.s.",
                       edgecolors="none", alpha=0.8, **kwargs)

    lim_min = min(df[cx].min(), df[cy].min()) - 0.02
    lim_max = max(df[cx].max(), df[cy].max()) + 0.02
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="dotted",
            color="gray", linewidth=0.8)
    if top_n > 0 and len(labels) > 0:
        for _, row in labels.iterrows():
            ax.annotate(row["cell_type"], (row[cx], row[cy]),
                        fontsize=fontsize - 6, ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.3))

    label = "AUC" if cx.startswith("auc") else "CCC"
    ax.set_xlabel(f"{label} 1", fontsize=fontsize)
    ax.set_ylabel(f"{label} 2", fontsize=fontsize)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    _apply_ov_style(ax, fontsize)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=condition1_color,
                    markersize=6, label="condition 1"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=condition2_color,
                    markersize=6, label="condition 2"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ns_color,
                    markersize=6, label="n.s."),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False,
              fontsize=fontsize - 4)

    if save:
        fig.savefig(save if isinstance(save, str) else "augur_dp.pdf",
                    dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if created or return_fig:
        return fig, ax
    return ax


# ---------------------------------------------------------------------------
# Public Augur class
# ---------------------------------------------------------------------------


@register_function(
    aliases=["细胞类型优先级", "augur", "cell_type_prioritization", "Augur"],
    category="single",
    description=(
        "Cell type prioritization via ML cross-validation. Trains a random-forest "
        "or logistic-regression classifier per cell type to predict condition "
        "labels, evaluates AUC, and ranks cell types by perturbation response. "
        "Results land in adata.uns['augur'] and adata.obs['augur_auc']."
    ),
    requires={"obs": ["label column", "cell_type column"]},
    produces={"uns": ["augur"], "obs": ["augur_auc"]},
    examples=[
        "# Basic cell type prioritization",
        "augur = ov.single.Augur(adata, label_col='condition', cell_type_col='cell_type')",
        "augur.run()",
        "augur.plot_lollipop()",
        "",
        "# Compare two conditions",
        "dp = augur.run_differential(adata2)",
        "augur.plot_differential_prioritization(dp)",
    ],
)
class Augur:
    """Cell type prioritization via machine-learning cross-validation.

    Wraps the py-Augur implementation of R Augur (Skinnider et al. 2020).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with expression in ``.X`` and metadata in ``.obs``.
    label_col : str
        Column in ``adata.obs`` containing condition / treatment labels.
    cell_type_col : str
        Column in ``adata.obs`` containing cell type annotations.
    classifier : str
        ``'rf'`` (random forest, default) or ``'lr'`` (logistic regression).
    seed : int
        Random seed for reproducibility.

    Examples
    --------
    >>> augur = ov.single.Augur(adata, label_col='condition', cell_type_col='cell_type')
    >>> augur.run(n_subsamples=50)
    >>> augur.plot_lollipop()
    """

    def __init__(
        self,
        adata: AnnData,
        label_col: str = "label",
        cell_type_col: str = "cell_type",
        classifier: str = "rf",
        seed: int = 42,
    ):
        self.adata = adata
        self.label_col = label_col
        self.cell_type_col = cell_type_col
        self.classifier = classifier
        self.seed = seed
        self.result: dict[str, Any] | None = None

    def run(
        self,
        n_subsamples: int = 50,
        subsample_size: int = 20,
        folds: int = 3,
        min_cells: int | None = None,
        var_quantile: float = 0.5,
        feature_perc: float = 0.5,
        augur_mode: str = "default",
        rf_params: dict | None = None,
        lr_params: dict | None = None,
    ) -> "Augur":
        """Run cell type prioritization.

        Parameters
        ----------
        n_subsamples : int
            Number of random subsamples per cell type. (50)
        subsample_size : int
            Cells per condition per subsample. (20)
        folds : int
            Number of CV folds. (3)
        min_cells : int or None
            Minimum cells per cell type. Defaults to *subsample_size*.
        var_quantile : float
            Quantile for variance-based feature selection. (0.5)
        feature_perc : float
            Proportion of features randomly selected. (0.5)
        augur_mode : str
            ``'default'``, ``'velocity'``, or ``'permute'``. (default)
        rf_params : dict, optional
            Random forest parameters: ``trees``, ``mtry``, ``min_n``, ``importance``.
        lr_params : dict, optional
            Logistic regression parameters: ``mixture``, ``penalty``.

        Returns
        -------
        self
        """
        self.result = _calculate_auc(
            self.adata,
            label_col=self.label_col,
            cell_type_col=self.cell_type_col,
            n_subsamples=n_subsamples,
            subsample_size=subsample_size,
            folds=folds,
            min_cells=min_cells,
            var_quantile=var_quantile,
            feature_perc=feature_perc,
            augur_mode=augur_mode,
            classifier=self.classifier,
            rf_params=rf_params,
            lr_params=lr_params,
            seed=self.seed,
        )

        # Store in adata.uns
        self.adata.uns["augur"] = {
            k: v for k, v in self.result.items() if k not in ("X", "y", "cell_types")
        }

        # Map AUC per cell type back to obs
        if "AUC" in self.result:
            auc_map = dict(zip(
                self.result["AUC"]["cell_type"], self.result["AUC"]["auc"],
            ))
            self.adata.obs["augur_auc"] = (
                self.adata.obs[self.cell_type_col].map(auc_map)
            )
        elif "CCC" in self.result:
            ccc_map = dict(zip(
                self.result["CCC"]["cell_type"], self.result["CCC"]["ccc"],
            ))
            self.adata.obs["augur_auc"] = (
                self.adata.obs[self.cell_type_col].map(ccc_map)
            )

        return self

    def run_differential(
        self,
        adata2: AnnData,
        n_subsamples: int = 50,
        subsample_size: int = 20,
        folds: int = 3,
        min_cells: int | None = None,
        var_quantile: float = 0.5,
        feature_perc: float = 0.5,
        n_permutations: int = 1000,
        rf_params: dict | None = None,
        lr_params: dict | None = None,
    ) -> pd.DataFrame:
        """Test differential prioritization between two conditions.

        Runs ``calculate_auc`` with ``augur_mode='permute'`` on both datasets,
        then computes a permutation-based p-value for each cell type.

        Parameters
        ----------
        adata2 : AnnData
            Second AnnData (different condition / time-point).
        n_permutations : int
            Number of permutations for null distribution. (1000)
        Other parameters same as :meth:`run`.

        Returns
        -------
        pd.DataFrame
            Columns: ``cell_type``, ``auc.x``, ``auc.y``, ``delta_auc``,
            ``b``, ``m``, ``z``, ``pval``, ``padj``.
        """
        common_kw = dict(
            n_subsamples=n_subsamples, subsample_size=subsample_size, folds=folds,
            min_cells=min_cells, var_quantile=var_quantile, feature_perc=feature_perc,
            rf_params=rf_params, lr_params=lr_params,
        )

        augur1 = _calculate_auc(
            self.adata, label_col=self.label_col, cell_type_col=self.cell_type_col,
            classifier=self.classifier, seed=self.seed, augur_mode="default", **common_kw,
        )
        augur2 = _calculate_auc(
            adata2, label_col=self.label_col, cell_type_col=self.cell_type_col,
            classifier=self.classifier, seed=self.seed, augur_mode="default", **common_kw,
        )
        perm1 = _calculate_auc(
            self.adata, label_col=self.label_col, cell_type_col=self.cell_type_col,
            classifier=self.classifier, seed=self.seed, augur_mode="permute", **common_kw,
        )
        perm2 = _calculate_auc(
            adata2, label_col=self.label_col, cell_type_col=self.cell_type_col,
            classifier=self.classifier, seed=self.seed, augur_mode="permute", **common_kw,
        )

        return _calculate_differential_prioritization(
            augur1, augur2, perm1, perm2,
            n_subsamples=n_subsamples, n_permutations=n_permutations,
        )

    # -- plotting delegates (ov style: fontsize, show, save, return_fig) -----

    def plot_lollipop(self, **kwargs):
        """Lollipop plot of cell type AUC values."""
        if self.result is None:
            raise RuntimeError("Run .run() before plotting")
        return _plot_lollipop(self.result, **kwargs)

    def plot_umap(self, **kwargs):
        """UMAP overlay colored by cell type AUC."""
        if self.result is None:
            raise RuntimeError("Run .run() before plotting")
        return _plot_umap(self.adata, self.result, **kwargs)

    def plot_important_features(self, **kwargs):
        """Bar chart of top important features for a cell type."""
        if self.result is None:
            raise RuntimeError("Run .run() before plotting")
        return _plot_important_features(self.result, **kwargs)

    def plot_augur(self, **kwargs):
        """Combined lollipop + feature importance figure."""
        if self.result is None:
            raise RuntimeError("Run .run() before plotting")
        return _plot_augur_combined(self.result, **kwargs)

    def plot_scatterplot(self, other: "Augur", **kwargs):
        """Compare two Augur results as a scatterplot."""
        if self.result is None or other.result is None:
            raise RuntimeError("Both Augur instances must have .run() results")
        return _plot_scatterplot(self.result, other.result, **kwargs)

    def plot_differential_prioritization(self, dp_results: pd.DataFrame, **kwargs):
        """Plot differential prioritization results."""
        return _plot_differential_prioritization(dp_results, **kwargs)
