from typing import Union

import numpy as np
import pandas as pd


def cpm_norm(X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """Counts-per-million normalize a genes-by-cells matrix.

    Parameters
    ----------
    X
        Genes-by-cells count matrix.

    Returns
    -------
    Normalized matrix with the same type and shape as ``X``.
    """
    lib_size = X.sum(axis=0)
    safe_lib_size = lib_size.replace(0, np.nan) if isinstance(lib_size, pd.Series) else np.where(lib_size == 0, np.nan, lib_size)
    normalized = X * 1e6 / safe_lib_size
    if isinstance(normalized, pd.DataFrame):
        return normalized.fillna(0)
    return np.nan_to_num(normalized)
