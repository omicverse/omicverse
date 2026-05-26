from functools import partial
from warnings import warn
from typing import List, Optional, Set

import pandas as pd
import numpy as np


def _check_features(df,
                    features: List[str]) -> Set[str]:
    valid_features = set(df.index) & set(features)
    if len(features) != len(valid_features):
        warn(f"Found {len(features) - len(valid_features)} invalid features (e.g. not shown in the dataframe)")
    return valid_features


def calc_auc(rank_val: pd.Series,
             max_rank: int) -> float:
    """AUC of feature ranks against ``max_rank`` (UCell scoring kernel).

    Returns 0 when every value is above ``max_rank``.
    """
    insig_part = rank_val > max_rank
    if all(insig_part):
        return 0
    else:
        rank_val[insig_part] = max_rank + 1
        rank_sum = sum(rank_val)
        n = rank_val.shape[0]
        u_val = rank_sum - (n * (n + 1)) / 2  # lower if the rank is higher
        auc = 1 - (u_val / (n * max_rank))
        return auc


def calc_U_stat_df(features: List[str],
                   df: pd.DataFrame,
                   neg_features: Optional[List[str]] = None,
                   max_rank: int = 1500,
                   w_neg: float = 1) -> np.ndarray:
    """Compute the per-cell U-statistic for a positive (and optional negative) gene set.

    Parameters
    ----------
    features
        Positive (up) gene names.
    df
        Pre-ranked gene-by-cell DataFrame.
    neg_features
        Negative (down) gene names; defaults to none.
    max_rank
        Rank cutoff above which genes are treated as not significant.
    w_neg
        Weight applied to the negative-set contribution.

    Returns
    -------
    Per-cell UCell scores as a 1-D array.
    """
    if neg_features is None:
        neg_features = []
    pos_features = list(set(features) - set(neg_features))
    if len(pos_features) > 0:
        pos = df.reindex(index=pos_features).apply(partial(calc_auc, max_rank=max_rank), axis=0).values
    else:
        pos = np.zeros(shape=(df.shape[2],))

    if len(neg_features) > 0:
        neg = df.reindex(index=neg_features).apply(partial(calc_auc, max_rank=max_rank), axis=0).values
    else:
        neg = np.zeros(shape=(df.shape[2],))
    diff = pos - w_neg * neg
    # diff[diff < 0] = 0
    return diff


def cal_Uscore(X: pd.DataFrame,
               pos_genes: List[str],
               neg_genes: List[str],
               max_rank: int = 1500,
               w_neg: float = 1,
               ties_method: str = "average") -> pd.DataFrame:
    """Compute UCell scores for every cell in ``X``.

    Parameters
    ----------
    X
        Expression DataFrame (genes x cells).
    pos_genes
        Positive (up) gene names.
    neg_genes
        Negative (down) gene names.
    max_rank
        Rank cutoff above which genes are treated as not significant.
    w_neg
        Weight applied to the negative-set contribution.
    ties_method
        Tie-breaking strategy passed to :meth:`pandas.DataFrame.rank`.

    Returns
    -------
    Single-column DataFrame of UCell scores indexed by cell.
    """
    ranked_df = X.rank(ascending=False, method=ties_method)
    pos_genes = list(_check_features(X, pos_genes))
    cell_auc = calc_U_stat_df(pos_genes, ranked_df,
                              neg_features=neg_genes,
                              max_rank=max_rank,
                              w_neg=w_neg)
    return pd.DataFrame(cell_auc, index=ranked_df.columns)
