# coding=utf-8

from multiprocessing import Pool
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn import mixture
from tqdm import tqdm

from .diptest import diptst


def _derive_threshold_task(args):
    regulon_name, data, seed, method = args
    assert method in {"hdt", "bic"}

    if seed is not None:
        np.random.seed(seed=seed)

    def isbimodal(data, method):
        if method == "hdt":
            # Use Hartigan's dip statistic to decide if distribution deviates from unimodality.
            _, pval, _ = diptst(data)
            return (pval is not None) and (pval <= 0.05), None
        else:
            # Compare Bayesian Information Content of two Gaussian Mixture Models.
            X = data.reshape(-1, 1)
            gmm2 = mixture.GaussianMixture(
                n_components=2, covariance_type="full", random_state=seed
            ).fit(X)
            gmm1 = mixture.GaussianMixture(
                n_components=1, covariance_type="full", random_state=seed
            ).fit(X)
            is_bimodal = gmm2.bic(X) <= gmm1.bic(X)
            return is_bimodal, gmm2 if is_bimodal else None

    is_bimodal, gmm2 = isbimodal(data, method)
    if not is_bimodal:
        # For a unimodal distribution the threshold is set as mean plus two standard deviations.
        threshold = data.mean() + 2.0 * data.std()
    else:
        if gmm2 is None:
            # Fit a two component Gaussian Mixture model on the AUC distribution using an Expectation-Maximization algorithm
            # to identify the peaks in the distribution.
            gmm2 = mixture.GaussianMixture(
                n_components=2, covariance_type="full", random_state=seed
            ).fit(data.reshape(-1, 1))
        # For a bimodal distribution the threshold is defined as the "trough" in between the two peaks.
        # This is solved as a minimization problem on the kernel smoothed density.
        threshold = minimize_scalar(
            fun=stats.gaussian_kde(data), bounds=sorted(gmm2.means_), method="bounded"
        ).x[0]
    return regulon_name, threshold


def derive_threshold(
    auc_mtx: pd.DataFrame, regulon_name: str, seed=None, method: str = "hdt"
) -> float:
    """
    Derive threshold on the AUC values of the given regulon to binarize the cells in two clusters: "on" versus "off"
    state of the regulator.

    :param auc_mtx: The dataframe with the AUC values for all cells and regulons (n_cells x n_regulons).
    :param regulon_name: the name of the regulon for which to predict the threshold.
    :param method: The method to use to decide if the distribution of AUC values for the given regulon is not unimodel.
        Can be either Hartigan's Dip Test (HDT) or Bayesian Information Content (BIC). The former method performs better
        but takes considerable more time to execute (40min for 350 regulons). The BIC compares the BIC for two Gaussian
        Mixture Models: single versus two components.
    :return: The threshold on the AUC values.
    """
    assert auc_mtx is not None and not auc_mtx.empty
    assert regulon_name in auc_mtx.columns
    assert method in {"hdt", "bic"}

    data = auc_mtx[regulon_name].values
    _, threshold = _derive_threshold_task((regulon_name, data, seed, method))
    return threshold


def binarize(
    auc_mtx: pd.DataFrame,
    threshold_overides: Optional[Mapping[str, float]] = None,
    seed=None,
    num_workers=1,
    method: str = "hdt",
    use_tqdm: bool = True,
) -> (pd.DataFrame, pd.Series):
    """
    "Binarize" the supplied AUC matrix, i.e. decide if for each cells in the matrix a regulon is active or not based
    on the bimodal distribution of the AUC values for that regulon.

    :param auc_mtx: The dataframe with the AUC values for all cells and regulons (n_cells x n_regulons).
    :param threshold_overides: A dictionary that maps name of regulons to manually set thresholds.
    :param method: The method to use to decide if each regulon distribution is not unimodal: "hdt" or "bic".
    :param use_tqdm: Whether to show a tqdm progress bar while deriving regulon thresholds.
    :return: A "binarized" dataframe and a series containing the AUC threshold used for each regulon.
    """
    assert method in {"hdt", "bic"}

    def derive_thresholds(auc_mtx, seed=seed):
        tasks = [
            (column, auc_mtx[column].values, seed, method) for column in auc_mtx.columns
        ]
        if num_workers == 1:
            results = map(_derive_threshold_task, tasks)
            results = tqdm(
                results,
                total=len(tasks),
                desc="Deriving AUC thresholds",
                disable=not use_tqdm,
            )
            thrs = list(results)
        else:
            with Pool(processes=num_workers) as p:
                results = p.imap(_derive_threshold_task, tasks)
                results = tqdm(
                    results,
                    total=len(tasks),
                    desc="Deriving AUC thresholds",
                    disable=not use_tqdm,
                )
                thrs = list(results)
        if not thrs:
            return pd.Series(index=auc_mtx.columns, dtype=float)
        names, values = zip(*thrs)
        return pd.Series(index=names, data=values).reindex(auc_mtx.columns)

    thresholds = derive_thresholds(auc_mtx)
    if threshold_overides is not None:
        thresholds[list(threshold_overides.keys())] = list(threshold_overides.values())
    return (auc_mtx > thresholds).astype(int), thresholds
