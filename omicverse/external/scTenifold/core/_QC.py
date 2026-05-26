import pandas as pd
from warnings import warn


def sc_QC(X: pd.DataFrame,
          min_lib_size: float = 1000,
          remove_outlier_cells: bool = True,
          min_percent: float = 0.05,
          max_mito_ratio: float = 0.1,
          min_exp_avg: float = 0,
          min_exp_sum: float = 0) -> pd.DataFrame:
    """
    main QC function in scTenifold pipelines

    Parameters
    ----------
    X: pd.DataFrame
        A single-cell RNAseq DataFrame (rows: genes, cols: cells)
    min_lib_size: int, float, default = 1000
        Minimum library size of cells
    remove_outlier_cells: bool, default = True
        Whether the QC function will remove the outlier cells
    min_percent: float, default = 0.05
        Minimum fraction of cells where the gene needs to be expressed to be included in the analysis.
    max_mito_ratio: float, default = 0.1
        Maximum mitochondrial genes ratio included in the final df
    min_exp_avg: float, default = 0
        Minimum average expression value in each gene
    min_exp_sum: float, default = 0
        Minimum sum of expression value in each gene
    Returns
    -------
    X_modified: pd.DataFrame
        The DataFrame after QC
    """
    X = X.copy()
    outlier_coef = 1.5
    X[X < 0] = 0
    lib_size = X.sum(axis=0)
    before_s = X.shape[1]
    X = X.loc[:, lib_size > min_lib_size]
    print(f"Removed {before_s - X.shape[1]} cells with lib size < {min_lib_size}")
    if remove_outlier_cells:
        lib_size = X.sum(axis=0)
        before_s = X.shape[1]
        Q1, Q3 = lib_size.quantile([0.25, 0.75])
        interquartile_range = Q3 - Q1
        X = X.loc[:, (lib_size >= Q1 - interquartile_range * outlier_coef) &
                     (lib_size <= Q3 + interquartile_range * outlier_coef)]
        print(f"Removed {before_s - X.shape[1]} outlier cells from original data")
    mt_genes = X.index.str.upper().str.match("^MT-")
    if any(mt_genes):
        print(f"Found mitochondrial genes: {X[mt_genes].index.to_list()}")
        before_s = X.shape[1]
        mt_rates = X[mt_genes].sum(axis=0) / X.sum(axis=0)
        X = X.loc[:, mt_rates < max_mito_ratio]
        print(f"Removed {before_s - X.shape[1]} samples from original data (mt genes ratio > {max_mito_ratio})")
    else:
        warn("Mitochondrial genes were not found. Be aware that apoptotic cells may be present in your sample.")
    before_g = X.shape[0]
    X = X[(X != 0).mean(axis=1) > min_percent]
    print(f"Removed {before_g - X.shape[0]} genes expressed in less than {min_percent} of data")

    before_g = X.shape[0]
    if X.shape[1] > 500:
        X = X.loc[X.mean(axis=1) >= min_exp_avg, :]
    else:
        X = X.loc[X.sum(axis=1) >= min_exp_sum, :]
    print(f"Removed {before_g - X.shape[0]} genes with expression values: average < {min_exp_avg} or sum < {min_exp_sum}")
    return X
